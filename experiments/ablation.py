"""
experiments/ablation.py
========================
Ablation studies for the thesis (RQ2: vibration signal quality).

Studies:
  1. Vibration noise sensitivity    — vary noise_std level
  2. Vibration components           — envelope-only vs lf-only vs full
  3. Per-bird vs shared vibration   — key thesis question

Usage:
  python experiments/ablation.py --wav_dir /path/to/wavfiles
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from data.dataset import XenoCantoCatalog, BirdSeparationDataset
from models.separator import build_model
from utils.metrics import evaluate_sample


BASE_CFG = {
    "dataset": {"sample_rate": 22050, "clip_duration": 3.0, "num_mixtures": 2000,
                "train_split": 0.70, "val_split": 0.15, "test_split": 0.15, "seed": 42},
    "model": {"num_speakers": 2, "audio_encoder_channels": 256,
               "audio_encoder_kernel_size": 16, "audio_encoder_stride": 8,
               "vib_encoder_channels": 128, "vib_encoder_layers": 4,
               "tcn_channels": 256, "tcn_kernel_size": 3, "tcn_layers": 8,
               "tcn_stacks": 3, "film_hidden_dim": 256},
    "vibration": {"lowpass_cutoff_hz": 500, "envelope_smoothing_ms": 20,
                  "noise_std": 0.02, "transient_prob": 0.15, "transient_amplitude": 0.3,
                  "per_bird": True},
    "training": {"batch_size": 8, "num_epochs": 50, "learning_rate": 1e-3,
                 "weight_decay": 1e-5, "warmup_epochs": 3,
                 "early_stopping_patience": 10, "grad_clip_norm": 5.0,
                 "loss_weights": {"time_domain": 0.5, "spectral": 0.5},
                 "device": "cuda", "num_workers": 4,
                 "checkpoint_dir": "experiments/ablation_checkpoints",
                 "log_dir": "experiments/ablation_logs"},
}


def quick_eval(model, dataset, device, n=200, use_vibration=True):
    """Quick evaluation on n samples, returns mean SI-SDR."""
    model.eval().to(device)
    si_sdrs = []
    indices = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    with torch.no_grad():
        for idx in indices:
            batch = dataset[int(idx)]
            mixture = batch["mixture"].unsqueeze(0).to(device)
            sources = batch["sources"].numpy()
            vib_input = batch["vibrations"].unsqueeze(0).to(device) if use_vibration else None
            estimates, _ = model(mixture, vib_input)
            est_np = estimates.squeeze(0).cpu().numpy()
            T = min(est_np.shape[1], sources.shape[1])
            from utils.metrics import compute_si_sdr
            si_sdrs.append(compute_si_sdr(est_np[:, :T], sources[:, :T]))
    return float(np.mean(si_sdrs))


def run_noise_sensitivity(wav_dir, metadata_csv, checkpoint_path, output_dir, device):
    """Study: how sensitive is performance to vibration noise level (RQ2)?"""
    print("\n=== Ablation: Vibration Noise Sensitivity ===")
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
    results = {}

    model = build_model(BASE_CFG, "vibration")
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint: {checkpoint_path}")

    catalog = XenoCantoCatalog(wav_dir, metadata_csv)

    for noise_std in noise_levels:
        vib_cfg = BASE_CFG["vibration"].copy()
        vib_cfg["noise_std"] = noise_std
        ds_cfg = BASE_CFG["dataset"]

        test_ds = BirdSeparationDataset(
            catalog=catalog,
            num_mixtures=int(ds_cfg["num_mixtures"] * ds_cfg["test_split"]),
            sample_rate=ds_cfg["sample_rate"],
            clip_duration=ds_cfg["clip_duration"],
            num_speakers=BASE_CFG["model"]["num_speakers"],
            overlap_ratio=0.6,
            vibration_cfg=vib_cfg,
            seed=ds_cfg["seed"],
            split="test",
        )
        si_sdr = quick_eval(model, test_ds, device, n=100, use_vibration=True)
        results[f"noise_{noise_std:.2f}"] = si_sdr
        print(f"  noise_std={noise_std:.2f} → SI-SDR={si_sdr:.2f} dB")

    return results


def run_per_bird_vs_shared(wav_dir, metadata_csv, checkpoint_path, output_dir, device):
    """
    Study: does separating vibrations per-bird matter vs sharing one vibration?
    This answers a core aspect of the thesis design.
    """
    print("\n=== Ablation: Per-Bird vs Shared Vibration ===")

    class SharedVibDataset(BirdSeparationDataset):
        """Overrides __getitem__ to use sum of all bird vibrations as a shared signal."""
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            # Replace per-bird vibrations with a shared (summed) vibration
            shared = item["vibrations"].sum(dim=0, keepdim=True)   # (1, T)
            # Broadcast to (N, T)
            N = item["vibrations"].shape[0]
            item["vibrations"] = shared.expand(N, -1)
            return item

    catalog = XenoCantoCatalog(wav_dir, metadata_csv)
    ds_cfg = BASE_CFG["dataset"]
    n_test = int(ds_cfg["num_mixtures"] * ds_cfg["test_split"])

    model = build_model(BASE_CFG, "vibration")
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    # Per-bird dataset
    per_bird_ds = BirdSeparationDataset(
        catalog=catalog, num_mixtures=n_test,
        sample_rate=ds_cfg["sample_rate"], clip_duration=ds_cfg["clip_duration"],
        num_speakers=BASE_CFG["model"]["num_speakers"], overlap_ratio=0.6,
        vibration_cfg=BASE_CFG["vibration"], seed=ds_cfg["seed"], split="test",
    )
    # Shared vibration dataset
    shared_ds = SharedVibDataset(
        catalog=catalog, num_mixtures=n_test,
        sample_rate=ds_cfg["sample_rate"], clip_duration=ds_cfg["clip_duration"],
        num_speakers=BASE_CFG["model"]["num_speakers"], overlap_ratio=0.6,
        vibration_cfg=BASE_CFG["vibration"], seed=ds_cfg["seed"], split="test",
    )

    si_sdr_per_bird = quick_eval(model, per_bird_ds, device, n=100)
    si_sdr_shared = quick_eval(model, shared_ds, device, n=100)

    print(f"  Per-bird vibration : SI-SDR = {si_sdr_per_bird:.2f} dB")
    print(f"  Shared vibration   : SI-SDR = {si_sdr_shared:.2f} dB")
    print(f"  Δ (per-bird gain)  : {si_sdr_per_bird - si_sdr_shared:+.2f} dB")

    return {
        "per_bird_si_sdr": si_sdr_per_bird,
        "shared_si_sdr": si_sdr_shared,
        "gain": si_sdr_per_bird - si_sdr_shared,
    }


def main():
    parser = argparse.ArgumentParser(description="Ablation studies")
    parser.add_argument("--wav_dir", required=True)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--vib_checkpoint",
                        default="experiments/checkpoints/vibration/best.pt")
    parser.add_argument("--output_dir", default="experiments/ablation_results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    noise_results = run_noise_sensitivity(
        args.wav_dir, args.metadata_csv, args.vib_checkpoint, args.output_dir, device
    )
    all_results["noise_sensitivity"] = noise_results

    per_bird_results = run_per_bird_vs_shared(
        args.wav_dir, args.metadata_csv, args.vib_checkpoint, args.output_dir, device
    )
    all_results["per_bird_vs_shared"] = per_bird_results

    out_path = Path(args.output_dir) / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Ablation results saved to: {out_path}")


if __name__ == "__main__":
    main()
