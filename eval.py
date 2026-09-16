"""
eval.py
-------
Evaluation script for the trained separation models.

Computes:
  - SI-SDR improvement (SI-SDRi = SI-SDR_output - SI-SDR_input)
  - SDR (via fast_bss_eval or manual implementation)
  - Spectral convergence (optional)
  - STFT L1 loss (optional)

Evaluates on:
  - Full test set
  - High-overlap subset (SNR near 0 dB → equal-energy overlap)

Outputs:
  - Console summary table
  - JSON results file for further statistical analysis
  - (Optional) saved separated audio samples

Usage:
    python eval.py --model baseline  --ckpt ./checkpoints/baseline/best.pt  ...
    python eval.py --model multimodal --ckpt ./checkpoints/multimodal/best.pt ...
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchaudio

from dataset import BirdMixDataset, recording_level_split
from loss import si_sdr, pit_si_sdr_loss
from model import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def sdr(estimate: torch.Tensor, reference: torch.Tensor,
        eps: float = 1e-8) -> torch.Tensor:
    """
    Classic SDR (Signal-to-Distortion Ratio).
    estimate, reference: [B, T]
    returns: [B]
    """
    dot      = (estimate * reference).sum(-1, keepdim=True)
    ref_pow  = (reference ** 2).sum(-1, keepdim=True) + eps
    proj     = dot / ref_pow * reference
    noise    = estimate - proj
    ratio    = (proj ** 2).sum(-1) / ((noise ** 2).sum(-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def stft_loss(estimate: torch.Tensor, reference: torch.Tensor,
              n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
    """
    Spectral L1 loss (log-magnitude STFT difference).
    estimate, reference: [B, T]
    returns: scalar
    """
    window = torch.hann_window(n_fft, device=estimate.device)
    def _stft(x):
        return torch.stft(x, n_fft=n_fft, hop_length=hop,
                          win_length=n_fft, window=window,
                          return_complex=True)

    est_mag = _stft(estimate).abs().clamp(min=1e-8).log()
    ref_mag = _stft(reference).abs().clamp(min=1e-8).log()
    return F.l1_loss(est_mag, ref_mag)


def spectral_convergence(estimate: torch.Tensor,
                         reference: torch.Tensor,
                         n_fft: int = 1024, hop: int = 256) -> torch.Tensor:
    """
    Spectral convergence = ‖|STFT(ref)| − |STFT(est)|‖_F / ‖|STFT(ref)|‖_F
    Lower is better.
    """
    window = torch.hann_window(n_fft, device=estimate.device)
    def _mag(x):
        return torch.stft(x, n_fft=n_fft, hop_length=hop,
                          win_length=n_fft, window=window,
                          return_complex=True).abs()

    est_mag = _mag(estimate)
    ref_mag = _mag(reference)
    num  = (ref_mag - est_mag).norm(p="fro", dim=(-2, -1))
    den  = ref_mag.norm(p="fro", dim=(-2, -1)).clamp(min=1e-8)
    return (num / den).mean()


def mixture_si_sdr(mix_audio: torch.Tensor,
                   y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
    """
    SI-SDR of the raw mixture against each source.
    Used to compute SI-SDRi = output SI-SDR - mixture SI-SDR.
    """
    s1 = si_sdr(mix_audio, y1).mean()
    s2 = si_sdr(mix_audio, y2).mean()
    return (s1 + s2) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, model_type: str, sr: int = 22050,
             save_audio: bool = False, audio_dir: str = "./eval_audio") -> dict:
    """
    Run evaluation over a DataLoader and return aggregated metrics.
    """
    model.eval()
    all_sisdr, all_sisdri, all_sdr, all_sc, all_stft = [], [], [], [], []

    audio_dir = Path(audio_dir)
    if save_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        Ymix = batch["Ymix"].to(device)   # [B, T]
        Vmix = batch["Vmix"].to(device)   # [B, T]
        Y1   = batch["Y1"].to(device)     # [B, T]
        Y2   = batch["Y2"].to(device)     # [B, T]
        B, T = Ymix.shape
        targets = torch.stack([Y1, Y2], dim=1)   # [B, 2, T]

        # ── Forward ──────────────────────────────────────────────────
        if model_type == "baseline":
            estimates = model(Ymix)
        else:
            estimates, _ = model(Ymix, Vmix)   # unpack (y_hat, v_hat); v_hat ignored at eval

        # ── Find best permutation (PIT alignment for metrics) ─────────
        _, best_perms = pit_si_sdr_loss(estimates, targets)

        # Reorder estimates to match best permutation
        est_aligned = torch.zeros_like(estimates)
        for b in range(B):
            for s in range(2):
                est_aligned[b, s] = estimates[b, best_perms[b, s]]

        # ── Per-source metrics ────────────────────────────────────────
        for s in range(2):
            est_s = est_aligned[:, s, :]   # [B, T]
            ref_s = targets[:, s, :]       # [B, T]

            sisdr_s = si_sdr(est_s, ref_s)                     # [B]
            sdr_s   = sdr(est_s, ref_s)                         # [B]
            sc_s    = spectral_convergence(est_s, ref_s)        # scalar
            stft_s  = stft_loss(est_s, ref_s)                   # scalar

            # SI-SDR improvement: output SI-SDR minus mixture SI-SDR for same ref
            mix_sisdr_s = si_sdr(Ymix, ref_s)              # [B]
            sisdri_s    = (sisdr_s - mix_sisdr_s)               # [B]

            all_sisdr.append(sisdr_s.cpu())
            all_sisdri.append(sisdri_s.cpu())
            all_sdr.append(sdr_s.cpu())
            all_sc.append(sc_s.item())
            all_stft.append(stft_s.item())

        # ── Optionally save first few separated clips ─────────────────
        if save_audio and batch_idx < 3:
            for b in range(min(2, B)):
                torchaudio.save(
                    audio_dir / f"batch{batch_idx}_sample{b}_mix.wav",
                    Ymix[b:b+1].cpu(), sample_rate=sr
                )
                for s in range(2):
                    torchaudio.save(
                        audio_dir / f"batch{batch_idx}_sample{b}_est{s+1}.wav",
                        est_aligned[b:b+1, s, :].cpu(), sample_rate=sr
                    )
                    torchaudio.save(
                        audio_dir / f"batch{batch_idx}_sample{b}_ref{s+1}.wav",
                        targets[b:b+1, s, :].cpu(), sample_rate=sr
                    )

    all_sisdr  = torch.cat(all_sisdr)
    all_sisdri = torch.cat(all_sisdri)
    all_sdr    = torch.cat(all_sdr)

    return {
        "SI-SDR_mean":  all_sisdr.mean().item(),
        "SI-SDR_std":   all_sisdr.std().item(),
        "SI-SDRi_mean": all_sisdri.mean().item(),
        "SI-SDRi_std":  all_sisdri.std().item(),
        "SDR_mean":     all_sdr.mean().item(),
        "SDR_std":      all_sdr.std().item(),
        "SpectralConv": np.mean(all_sc),
        "STFT_L1":      np.mean(all_stft),
        "n_samples":    len(all_sisdr),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_results(name: str, metrics: dict):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20}: {v:>8.4f}")
        else:
            print(f"  {k:<20}: {v}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Device: {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = ckpt["cfg"]
    print(f"[eval] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── Build model ─────────────────────────────────────────────────────────
    model_kwargs = {k: cfg[k] for k in
                    ["n_filters","filter_len","bottleneck","hidden",
                     "kernel_size","n_blocks","n_repeats"]}
    model = build_model(cfg["model_type"], **model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sr          = cfg["sample_rate"]
    clip_dur    = cfg["clip_duration_s"]

    # ── Load metadata and get test split ────────────────────────────────────
    meta = pd.read_csv(args.metadata)
    _, _, test_meta = recording_level_split(
        meta, val_fraction=0.15, test_fraction=0.15, seed=42
    )

    # ── Build test datasets ─────────────────────────────────────────────────
    # Standard test set
    test_ds = BirdMixDataset(
        meta            = test_meta,
        data_root       = args.data_root,
        num_mixtures    = args.num_mixtures,
        sample_rate     = sr,
        clip_duration_s = clip_dur,
        snr_range_db    = (-1.0, 1.0),   # balanced test
        add_bg_noise    = False,
        seed            = 999,
    )

    # High-overlap test: SNR strictly 0 dB (equal energy → maximum masking)
    hi_overlap_ds = BirdMixDataset(
        meta            = test_meta,
        data_root       = args.data_root,
        num_mixtures    = args.num_mixtures,
        sample_rate     = sr,
        clip_duration_s = clip_dur,
        snr_range_db    = (-0.5, 0.5),   # near-zero SNR → highest overlap
        add_bg_noise    = False,
        seed            = 1000,
    )

    loader_kw = dict(batch_size=args.batch_size, num_workers=4,
                     pin_memory=True, shuffle=False)
    test_loader     = DataLoader(test_ds, **loader_kw)
    hi_ov_loader    = DataLoader(hi_overlap_ds, **loader_kw)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print(f"\n[eval] Evaluating: {cfg['model_type']} on standard test set...")
    std_metrics = evaluate(model, test_loader, device, cfg["model_type"],
                           sr=sr, save_audio=args.save_audio,
                           audio_dir=args.audio_dir)

    print(f"[eval] Evaluating: {cfg['model_type']} on high-overlap subset...")
    hi_metrics = evaluate(model, hi_ov_loader, device, cfg["model_type"],
                          sr=sr, save_audio=False)

    # ── Display ───────────────────────────────────────────────────────────────
    label = cfg["model_type"].upper()
    print_results(f"{label} — Standard Test Set", std_metrics)
    print_results(f"{label} — High-Overlap Subset", hi_metrics)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = Path(args.output_json)
    results = {
        "model":        cfg["model_type"],
        "ckpt":         args.ckpt,
        "standard":     std_metrics,
        "high_overlap": hi_metrics,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate bird separation model")
    p.add_argument("--model",       type=str, required=True,
                   choices=["baseline", "multimodal"])
    p.add_argument("--ckpt",        type=str, required=True,
                   help="Path to best.pt checkpoint")
    p.add_argument("--data_root",   type=str, required=True)
    p.add_argument("--metadata",    type=str, required=True)
    p.add_argument("--num_mixtures", type=int, default=500,
                   help="Number of test mixtures to evaluate")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--output_json", type=str, default="./results.json")
    p.add_argument("--save_audio",  action="store_true",
                   help="Save a few separated audio samples")
    p.add_argument("--audio_dir",   type=str, default="./eval_audio")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
