"""
wilcoxon_test.py
----------------
Run this ONCE per checkpoint pair to get the directional Wilcoxon p-value,
median pairwise improvement, and 95% bootstrap CI needed for the thesis.

Requires the same eval infrastructure as eval.py.
Saves per-sample SI-SDRi arrays so you never need to re-run.

Usage:
    python analysis/statistical_test.py \
        --ckpt_a  ./checkpoints/baseline/best.pt \
        --ckpt_b  ./checkpoints/multimodal/best.pt \
        --data_root /path/to/data \
        --metadata  /path/to/metadata.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import BirdMixDataset, recording_level_split
from loss import si_sdr, pit_si_sdr_loss
from model import build_model


# ── helpers ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_sisdri(model, loader, device, model_type: str) -> np.ndarray:
    """Return per-source SI-SDRi values as a 1-D numpy array."""
    model.eval()
    records = []

    for batch in loader:
        Ymix = batch["Ymix"].to(device)
        Vmix = batch["Vmix"].to(device)
        Y1   = batch["Y1"].to(device)
        Y2   = batch["Y2"].to(device)
        targets = torch.stack([Y1, Y2], dim=1)   # [B, 2, T]

        if model_type == "baseline":
            estimates = model(Ymix)
        else:
            estimates, _ = model(Ymix, Vmix)

        _, best_perms = pit_si_sdr_loss(estimates, targets)

        est_aligned = torch.zeros_like(estimates)
        for b in range(Ymix.shape[0]):
            for s in range(2):
                est_aligned[b, s] = estimates[b, best_perms[b, s]]

        for s in range(2):
            est_s      = est_aligned[:, s, :]
            ref_s      = targets[:, s, :]
            sisdr_out  = si_sdr(est_s, ref_s)          # [B]
            sisdr_mix  = si_sdr(Ymix,  ref_s)          # [B]
            sisdri     = (sisdr_out - sisdr_mix).cpu().numpy()
            records.append(sisdri)

    return np.concatenate(records)   # shape: [n_samples]


def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt["cfg"]
    model_kwargs = {k: cfg[k] for k in
                    ["n_filters","filter_len","bottleneck","hidden",
                     "kernel_size","n_blocks","n_repeats"]}
    model = build_model(cfg["model_type"], **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, cfg


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000,
                 alpha: float = 0.05) -> tuple:
    medians = [
        np.median(np.random.choice(diffs, len(diffs), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(medians, 100 * alpha / 2)
    hi = np.percentile(medians, 100 * (1 - alpha / 2))
    return lo, hi


# ── main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stat] Device: {device}")

    # Use the SAME seed/SNR range as your standard test set in eval.py
    model_a, cfg_a = load_model(args.ckpt_a, device)
    model_b, cfg_b = load_model(args.ckpt_b, device)

    meta = pd.read_csv(args.metadata)
    _, _, test_meta = recording_level_split(
        meta, val_fraction=0.15, test_fraction=0.15, seed=42
    )

    # CRITICAL: identical dataset for both models (same seed → same mixtures)
    def make_loader(model_cfg):
        ds = BirdMixDataset(
            meta            = test_meta,
            data_root       = args.data_root,
            num_mixtures    = args.num_mixtures,
            sample_rate     = model_cfg["sample_rate"],
            clip_duration_s = model_cfg["clip_duration_s"],
            snr_range_db    = (-1.0, 1.0),
            add_bg_noise    = False,
            seed            = 999,         # same seed as your eval.py standard test
        )
        return DataLoader(ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)

    loader_a = make_loader(cfg_a)
    loader_b = make_loader(cfg_b)

    # ── Collect per-sample SI-SDRi ────────────────────────────────────────
    cache_a = Path(args.cache_dir) / "sisdri_model_a.npy"
    cache_b = Path(args.cache_dir) / "sisdri_model_b.npy"
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    if cache_a.exists() and not args.force:
        print(f"[stat] Loading cached Model A scores from {cache_a}")
        scores_a = np.load(cache_a)
    else:
        print("[stat] Collecting Model A per-sample SI-SDRi ...")
        scores_a = collect_sisdri(model_a, loader_a, device, cfg_a["model_type"])
        np.save(cache_a, scores_a)
        print(f"[stat] Saved to {cache_a}")

    if cache_b.exists() and not args.force:
        print(f"[stat] Loading cached Model B scores from {cache_b}")
        scores_b = np.load(cache_b)
    else:
        print("[stat] Collecting Model B per-sample SI-SDRi ...")
        scores_b = collect_sisdri(model_b, loader_b, device, cfg_b["model_type"])
        np.save(cache_b, scores_b)
        print(f"[stat] Saved to {cache_b}")

    assert len(scores_a) == len(scores_b), (
        f"Sample count mismatch: {len(scores_a)} vs {len(scores_b)}. "
        "Both models must evaluate the same mixtures in the same order."
    )

    # ── Wilcoxon signed-rank test ─────────────────────────────────────────
    diffs = scores_b - scores_a
    stat, p = wilcoxon(diffs, alternative="greater")   # H1: B > A
    median_diff = np.median(diffs)
    ci_lo, ci_hi = bootstrap_ci(diffs)

    # ── Print ─────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Wilcoxon Signed-Rank Test  (Model B > Model A)")
    print("="*55)
    print(f"  n samples              : {len(diffs)}")
    print(f"  Wilcoxon statistic     : {stat:.2f}")
    print(f"  p-value                : {p:.4e}")
    print(f"  Median improvement     : {median_diff:+.4f} dB")
    print(f"  95% bootstrap CI       : [{ci_lo:+.4f}, {ci_hi:+.4f}] dB")
    print(f"  H0 rejected (α=0.05)   : {'YES' if p < 0.05 else 'NO'}")
    print("="*55)

    # ── LaTeX snippet ready to paste ──────────────────────────────────────
    reject = "rejected" if p < 0.05 else "not rejected"
    print("\n--- Copy this into your thesis ---")
    print(
        f"A directional paired Wilcoxon signed-rank test (Model B > Model A) on the per-sample "
        f"SI-SDRi values yields $p = {p:.2e}$, with a median pairwise "
        f"improvement of ${median_diff:+.3f}$~dB "
        f"(95\\% bootstrap CI: $[{ci_lo:+.3f},\\,{ci_hi:+.3f}]$~dB). "
        f"The null hypothesis $H_0$ is {reject} at $\\alpha = 0.05$."
    )

    # ── Save JSON ─────────────────────────────────────────────────────────
    out = {
        "n_samples":        int(len(diffs)),
        "wilcoxon_stat":    float(stat),
        "p_value":          float(p),
        "median_diff_dB":   float(median_diff),
        "ci_95_lo":         float(ci_lo),
        "ci_95_hi":         float(ci_hi),
        "h0_rejected":      bool(p < 0.05),
    }
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[stat] Full results saved to {args.output_json}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_a",       required=True, help="Model A checkpoint (baseline)")
    p.add_argument("--ckpt_b",       required=True, help="Model B checkpoint (multimodal)")
    p.add_argument("--data_root",    required=True, help="Directory containing the WAV clips")
    p.add_argument("--metadata",     required=True, help="Metadata CSV with id, name, and filename columns")
    p.add_argument("--num_mixtures", type=int, default=500)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--cache_dir",    default="./stat_cache/run_2",
                   help="Directory to cache per-sample score arrays")
    p.add_argument("--output_json",  default="./results/statistics/wilcoxon_results.json")
    p.add_argument("--force",        action="store_true",
                   help="Re-collect scores even if cache exists")
    main(p.parse_args())
