"""
train.py
========
Main entry point for training the vibration-conditioned separator
and the audio-only baseline.

Usage:
  python train.py --wav_dir /path/to/wavfiles --metadata_csv metadata.csv
  python train.py --wav_dir /path/to/wavfiles --model audio_only
  python train.py --wav_dir /path/to/wavfiles --model both   (trains both)

Arguments:
  --wav_dir       : Directory containing .wav bird recordings
  --metadata_csv  : Optional metadata CSV (genus, species, filename columns)
  --model         : "vibration" | "audio_only" | "both" (default: both)
  --batch_size    : Override batch size (default: 8)
  --epochs        : Override number of epochs (default: 100)
  --lr            : Override learning rate (default: 1e-3)
  --device        : "cuda" or "cpu" (default: auto-detect)
  --num_mixtures  : Total training mixtures to generate (default: 10000)
  --num_workers   : DataLoader workers (default: 4)
  --resume        : Path to checkpoint to resume from
"""

import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import torch
from pathlib import Path

from data.dataset import build_dataloaders
from models.separator import build_model, count_parameters
from utils.trainer import Trainer


# ══════════════════════════════════════════════════════════════════════════════
#  Default Configuration
# ══════════════════════════════════════════════════════════════════════════════

def build_cfg(args) -> dict:
    return {
        "dataset": {
            "wav_dir": args.wav_dir,
            "metadata_csv": getattr(args, "metadata_csv", None),
            "sample_rate": 22050,
            "clip_duration": 3.0,
            "min_clip_duration": 1.0,
            "max_species_per_mix": 3,
            "num_mixtures": args.num_mixtures,
            "train_split": 0.70,
            "val_split": 0.15,
            "test_split": 0.15,
            "seed": 42,
        },
        "vibration": {
            "lowpass_cutoff_hz": 500,
            "envelope_smoothing_ms": 20,
            "noise_std": 0.02,
            "transient_prob": 0.15,
            "transient_amplitude": 0.3,
            "per_bird": True,   # ← CRITICAL: separate vibration per bird
        },
        "model": {
            "num_speakers": 2,
            "audio_encoder_channels": 256,
            "audio_encoder_kernel_size": 16,
            "audio_encoder_stride": 8,
            "vib_encoder_channels": 128,
            "vib_encoder_layers": 4,
            "tcn_channels": 256,
            "tcn_kernel_size": 3,
            "tcn_layers": 8,
            "tcn_stacks": 3,
            "film_hidden_dim": 256,
        },
        "training": {
            "batch_size": args.batch_size,
            "num_epochs": args.epochs,
            "learning_rate": args.lr,
            "weight_decay": 1e-5,
            "lr_scheduler": "cosine",
            "warmup_epochs": 5,
            "early_stopping_patience": 15,
            "grad_clip_norm": 5.0,
            "loss_weights": {"time_domain": 0.5, "spectral": 0.5},
            "device": args.device,
            "num_workers": args.num_workers,
            "checkpoint_dir": "experiments/checkpoints",
            "log_dir": "experiments/logs",
        },
        "evaluation": {
            "significance_level": 0.05,
            "overlap_test_thresholds": [0.3, 0.6, 0.9],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train bird source separation model(s)"
    )
    parser.add_argument("--wav_dir", default="data/wavfiles/",         #required=True,
                        help="Path to wavfiles/ directory") 
    parser.add_argument("--metadata_csv", default="data/bird_songs_metadata.csv",
                        help="Path to metadata CSV (optional)")
    parser.add_argument("--model", choices=["vibration", "audio_only", "both"],
                        default="both", help="Which model(s) to train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_mixtures", type=int, default=10000,
                        help="Total mixtures (split 70/15/15)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto",
                        help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print("  Bird Source Separation — Vibration-Guided Training")
    print(f"{'='*60}")
    print(f"  wav_dir      : {args.wav_dir}")
    print(f"  model        : {args.model}")
    print(f"  device       : {args.device}")
    print(f"  epochs       : {args.epochs}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  num_mixtures : {args.num_mixtures}")
    print(f"{'='*60}\n")

    cfg = build_cfg(args)

    # Save config
    Path("experiments").mkdir(exist_ok=True)
    with open("experiments/config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Build DataLoaders
    print("[DataLoaders] Building datasets...")
    train_loader, val_loader, test_loader = build_dataloaders(
        wav_dir=cfg["dataset"]["wav_dir"],
        metadata_csv=cfg["dataset"].get("metadata_csv"),
        cfg=cfg,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val   batches: {len(val_loader)}")
    print(f"  Test  batches: {len(test_loader)}")

    # Verify per-bird vibration by inspecting one batch
    print("\n[Verify] Checking per-bird vibrations are distinct...")
    batch = next(iter(train_loader))
    vib = batch["vibrations"]   # (B, N, T)
    for b in range(min(2, vib.shape[0])):
        for n in range(vib.shape[1] - 1):
            are_same = torch.allclose(vib[b, n], vib[b, n + 1], atol=1e-4)
            print(f"  Sample {b}: bird-{n} == bird-{n+1}? {are_same} "
                  f"{'❌ BUG!' if are_same else '✓ distinct'}")

    histories = {}
    models_to_train = (
        ["vibration", "audio_only"] if args.model == "both" else [args.model]
    )

    for model_name in models_to_train:
        print(f"\n[Model] Building {model_name} model...")
        model = build_model(cfg, model_name)
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}")

        trainer = Trainer(model, model_name, cfg)

        if args.resume and os.path.exists(args.resume):
            trainer.load_checkpoint(args.resume)

        history = trainer.train(train_loader, val_loader)
        histories[model_name] = history

    print("\n✓ All training complete.")
    print("  Checkpoints in: experiments/checkpoints/")
    print("  Logs in:        experiments/logs/")
    print("\nNext step — evaluate:")
    print("  python experiments/evaluate.py \\")
    print(f"    --wav_dir {args.wav_dir} \\")
    if args.metadata_csv:
        print(f"    --metadata_csv {args.metadata_csv} \\")
    print("    --vib_checkpoint experiments/checkpoints/vibration/best.pt \\")
    print("    --baseline_checkpoint experiments/checkpoints/audio_only/best.pt")


if __name__ == "__main__":
    main()
