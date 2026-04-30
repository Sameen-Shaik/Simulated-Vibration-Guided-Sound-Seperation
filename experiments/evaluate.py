"""
experiments/evaluate.py
========================
Evaluation pipeline for the thesis.

Runs both models on the test set (and overlap-stratified subsets),
computes all metrics, runs statistical significance tests, and saves
a full results report.

Usage:
  python experiments/evaluate.py \
    --wav_dir /path/to/wavfiles \
    --metadata_csv /path/to/metadata.csv \
    --vib_checkpoint experiments/checkpoints/vibration/best.pt \
    --baseline_checkpoint experiments/checkpoints/audio_only/best.pt \
    --output_dir experiments/results
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional

from data.dataset import XenoCantoCatalog, BirdSeparationDataset
from models.separator import build_model, count_parameters
from utils.metrics import (
    evaluate_sample, stratified_analysis,
    run_significance_tests, print_results_table,
)
from utils.trainer import Trainer


# ══════════════════════════════════════════════════════════════════════════════
#  Default config (mirrors configs/config.yaml)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    "dataset": {
        "wav_dir": "wavfiles",
        "metadata_csv": "metadata.csv",
        "sample_rate": 22050,
        "clip_duration": 3.0,
        "num_mixtures": 10000,
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
        "per_bird": True,
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
        "batch_size": 8,
        "num_epochs": 100,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "warmup_epochs": 5,
        "early_stopping_patience": 15,
        "grad_clip_norm": 5.0,
        "loss_weights": {"time_domain": 0.5, "spectral": 0.5},
        "device": "cuda",
        "num_workers": 4,
        "checkpoint_dir": "experiments/checkpoints",
        "log_dir": "experiments/logs",
    },
    "evaluation": {
        "significance_level": 0.05,
        "overlap_test_thresholds": [0.3, 0.6, 0.9],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation Runner
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    model_name: str,
    test_dataset: BirdSeparationDataset,
    device: torch.device,
    use_vibration: bool,
    num_samples: int = 500,
) -> List[Dict]:
    """Run model on test samples and collect per-sample metrics."""
    model.eval()
    model.to(device)
    results = []

    indices = np.random.choice(len(test_dataset), min(num_samples, len(test_dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc=f"Evaluating [{model_name}]"):
            batch = test_dataset[int(idx)]
            mixture = batch["mixture"].unsqueeze(0).to(device)     # (1, 1, T)
            sources = batch["sources"].numpy()                     # (N, T)
            vibrations = batch["vibrations"].unsqueeze(0).to(device)  # (1, N, T)
            overlap_ratio = float(batch["overlap_ratio"])

            vib_input = vibrations if use_vibration else None
            estimates, _ = model(mixture, vib_input)
            estimates_np = estimates.squeeze(0).cpu().numpy()      # (N, T)

            # Trim to same length
            T = min(estimates_np.shape[1], sources.shape[1])
            estimates_np = estimates_np[:, :T]
            sources_trim = sources[:, :T]

            metrics = evaluate_sample(estimates_np, sources_trim, overlap_ratio)
            results.append(metrics)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    cfg = DEFAULT_CFG.copy()

    # Override paths from args
    cfg["dataset"]["wav_dir"] = args.wav_dir
    if args.metadata_csv:
        cfg["dataset"]["metadata_csv"] = args.metadata_csv
    cfg["training"]["device"] = args.device

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[Evaluate] Using device: {device}")

    # Build test dataset
    catalog = XenoCantoCatalog(cfg["dataset"]["wav_dir"], cfg["dataset"]["metadata_csv"])
    ds_cfg = cfg["dataset"]
    total = ds_cfg["num_mixtures"]
    n_test = total - int(total * ds_cfg["train_split"]) - int(total * ds_cfg["val_split"])

    test_ds = BirdSeparationDataset(
        catalog=catalog,
        num_mixtures=n_test,
        sample_rate=ds_cfg["sample_rate"],
        clip_duration=ds_cfg["clip_duration"],
        num_speakers=cfg["model"]["num_speakers"],
        overlap_ratio="random",
        overlap_ratio_range=(0.2, 0.95),
        vibration_cfg=cfg["vibration"],
        seed=ds_cfg["seed"],
        split="test",
    )

    # Build and load models
    vib_model = build_model(cfg, "vibration")
    baseline_model = build_model(cfg, "audio_only")

    print(f"[Models] Vibration params: {count_parameters(vib_model):,}")
    print(f"[Models] Baseline params:  {count_parameters(baseline_model):,}")

    if args.vib_checkpoint and os.path.exists(args.vib_checkpoint):
        ckpt = torch.load(args.vib_checkpoint, map_location=device)
        vib_model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded vibration checkpoint: {args.vib_checkpoint}")
    else:
        print("  WARNING: No vibration checkpoint found; using random weights")

    if args.baseline_checkpoint and os.path.exists(args.baseline_checkpoint):
        ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        baseline_model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded baseline checkpoint: {args.baseline_checkpoint}")
    else:
        print("  WARNING: No baseline checkpoint found; using random weights")

    n_eval = min(args.num_samples, len(test_ds))
    print(f"\n[Evaluate] Running on {n_eval} test samples...")

    # Run evaluation
    vib_results = evaluate_model(vib_model, "vibration", test_ds, device, use_vibration=True, num_samples=n_eval)
    bas_results = evaluate_model(baseline_model, "audio_only", test_ds, device, use_vibration=False, num_samples=n_eval)

    # ── Statistical tests ──────────────────────────────────────────────────
    alpha = cfg["evaluation"]["significance_level"]
    metrics_to_test = ["SI-SDR", "SDR", "spectral_convergence", "log_spectral_distance"]
    sig_tests = {}
    for metric in metrics_to_test:
        v = [r[metric] for r in vib_results if metric in r]
        b = [r[metric] for r in bas_results if metric in r]
        if v and b:
            n = min(len(v), len(b))
            sig_tests[metric] = run_significance_tests(v[:n], b[:n], metric, alpha)

    # ── Overlap-stratified analysis ────────────────────────────────────────
    vib_strat = stratified_analysis(vib_results)
    bas_strat = stratified_analysis(bas_results)

    # ── Print results ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS: Vibration-Conditioned vs Audio-Only Baseline")
    print("=" * 70)
    print_results_table(vib_results, bas_results, alpha=alpha)

    print("\n  Overlap-Stratified SI-SDR (dB):")
    print(f"  {'Overlap Bin':<25} {'Vibration':>12} {'Baseline':>12} {'Diff':>10}")
    print("  " + "-" * 60)
    for label in vib_strat:
        v_val = vib_strat[label].get("SI-SDR", float("nan"))
        b_val = bas_strat[label].get("SI-SDR", float("nan"))
        diff = v_val - b_val
        n = vib_strat[label].get("n_samples", 0)
        print(f"  {label:<25} {v_val:>12.3f} {b_val:>12.3f} {diff:>+10.3f}  (n={n})")

    # ── Save results ───────────────────────────────────────────────────────
    report = {
        "num_eval_samples": n_eval,
        "device": str(device),
        "significance_tests": sig_tests,
        "vibration_stratified": vib_strat,
        "baseline_stratified": bas_strat,
        "summary": {
            "vibration_mean_si_sdr": float(np.nanmean([r["SI-SDR"] for r in vib_results])),
            "baseline_mean_si_sdr": float(np.nanmean([r["SI-SDR"] for r in bas_results])),
            "vibration_mean_sdr": float(np.nanmean([r.get("SDR", float("nan")) for r in vib_results])),
            "baseline_mean_sdr": float(np.nanmean([r.get("SDR", float("nan")) for r in bas_results])),
        },
    }

    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=lambda x: "nan" if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"\n  Report saved to: {report_path}")

    # Save raw results
    import csv
    for name, res in [("vibration", vib_results), ("audio_only", bas_results)]:
        csv_path = out_dir / f"{name}_results.csv"
        if res:
            keys = list(res[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(res)
        print(f"  Per-sample results: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bird source separation models")
    parser.add_argument("--wav_dir", required=True, help="Path to wavfiles/ directory")
    parser.add_argument("--metadata_csv", default=None, help="Path to metadata CSV")
    parser.add_argument("--vib_checkpoint", default="experiments/checkpoints/vibration/best.pt")
    parser.add_argument("--baseline_checkpoint", default="experiments/checkpoints/audio_only/best.pt")
    parser.add_argument("--output_dir", default="experiments/results")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    main(args)
