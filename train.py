"""
train.py
--------
Training pipeline for both experiments.

    Model A (baseline)   : audio-only  — ConvTasNet
    Model B (multimodal) : vibration-guided — MultimodalSep + VibrationHead

Both models MUST use the SAME DataLoaders (identical mixture pool)
for a fair comparison (RQ1 of the thesis).

Variable names follow the spec document:
    Ymix  — mixture waveform            (model input, both)
    Vmix  — mixture vibration           (Model B input only)
    Y1,Y2 — clean sources               (separation targets)
    V1,V2 — per-source vibrations       (Model B aux targets, NEVER model input)
    Y1_hat, Y2_hat — audio estimates    (model audio outputs)
    V1_hat, V2_hat — vib estimates      (Model B VibrationHead outputs)

Loss:
    Model A: L_total = L_audio
    Model B: L_total = L_audio + lambda * L_vib

Usage:
    # Model A (baseline)
    python train.py --model baseline \\
        --data_root ./data --metadata ./metadata.csv \\
        --num_mixtures 3800

    # Model B (multimodal), batch=2 + grad accum → effective batch=16
    python train.py --model multimodal \\
        --data_root ./data --metadata ./metadata.csv \\
        --num_mixtures 3800 --batch_size 2 --accum_steps 8 --norm_type gN
"""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import build_dataloaders
from loss import total_loss, pit_si_sdr_loss
from model import build_model


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch — with gradient accumulation
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, cfg,
                    accum_steps: int = 1) -> dict:
    """
    One training epoch.

    Gradient accumulation:
        effective_batch = batch_size * accum_steps
        Loss is divided by accum_steps before .backward() so accumulated
        gradients equal a single large-batch forward pass.
        optimizer.step() and zero_grad() fire only at accum boundaries.
        Gradient clipping happens AFTER full accumulation, never mid-step.
    """
    model.train()
    is_multimodal = cfg["model_type"] == "multimodal"
    metrics = {"total": 0., "audio": 0., "vib": 0., "n": 0}

    optimizer.zero_grad()

    for step_idx, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        # ── Unpack batch (spec variable names) ───────────────────────
        Ymix = batch["Ymix"].to(device)   # [B, T]  primary input
        Vmix = batch["Vmix"].to(device)   # [B, T]  Model B conditioning
        Y1   = batch["Y1"].to(device)     # [B, T]  separation target
        Y2   = batch["Y2"].to(device)     # [B, T]  separation target
        V1   = batch["V1"].to(device)     # [B, T]  vib aux target (Model B)
        V2   = batch["V2"].to(device)     # [B, T]  vib aux target (Model B)

        targets = torch.stack([Y1, Y2], dim=1)   # [B, 2, T]

        # ── Forward pass ─────────────────────────────────────────────
        # Model A: Ymix → (Y1_hat, Y2_hat)
        # Model B: Ymix + Vmix → (Y1_hat, Y2_hat), (V1_hat, V2_hat)
        # CRITICAL: V1, V2 are NEVER passed to the model.
        if is_multimodal:
            y_hat, v_hat = model(Ymix, Vmix)    # [B,2,T], [B,2,T]
        else:
            y_hat = model(Ymix)                 # [B, 2, T]
            v_hat = None

        # ── Loss ──────────────────────────────────────────────────────
        losses = total_loss(
            y_hat        = y_hat,
            targets      = targets,
            V1           = V1,
            V2           = V2,
            v_hat        = v_hat,
            lambda_vib   = cfg["lambda_vib"],
            use_vib_loss = is_multimodal,
        )

        # Scale loss before backward — keeps gradient magnitude correct
        # regardless of accumulation depth.
        (losses["total"] / accum_steps).backward()

        # Accumulate metrics (un-scaled, for readable logging)
        B = Ymix.shape[0]
        metrics["total"] += losses["total"].item() * B
        metrics["audio"] += losses["audio"].item() * B
        metrics["vib"]   += losses["vib"].item()   * B
        metrics["n"]     += B

        # ── Optimizer step at accumulation boundary ───────────────────
        is_boundary    = (step_idx + 1) % accum_steps == 0
        is_last        = (step_idx + 1) == len(loader)
        if is_boundary or is_last:
            # Clip after full accumulation — not mid-accumulation
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

    n = metrics.pop("n")
    return {k: v / n for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, cfg) -> dict:
    model.eval()
    is_multimodal = cfg["model_type"] == "multimodal"
    metrics = {"total": 0., "audio": 0., "vib": 0., "si_sdr": 0., "n": 0}

    for batch in tqdm(loader, desc="Validation", leave=False):
        Ymix = batch["Ymix"].to(device)
        Vmix = batch["Vmix"].to(device)
        Y1   = batch["Y1"].to(device)
        Y2   = batch["Y2"].to(device)
        V1   = batch["V1"].to(device)
        V2   = batch["V2"].to(device)
        targets = torch.stack([Y1, Y2], dim=1)

        if is_multimodal:
            y_hat, v_hat = model(Ymix, Vmix)
        else:
            y_hat = model(Ymix)
            v_hat = None

        losses = total_loss(
            y_hat        = y_hat,
            targets      = targets,
            V1           = V1,
            V2           = V2,
            v_hat        = v_hat,
            lambda_vib   = cfg["lambda_vib"],
            use_vib_loss = is_multimodal,
        )

        # Track raw SI-SDR (positive, for monitoring)
        l_audio, _ = pit_si_sdr_loss(y_hat, targets)
        si_sdr_val = -l_audio.item()

        B = Ymix.shape[0]
        metrics["total"]  += losses["total"].item() * B
        metrics["audio"]  += losses["audio"].item() * B
        metrics["vib"]    += losses["vib"].item()   * B
        metrics["si_sdr"] += si_sdr_val             * B
        metrics["n"]      += B

    n = metrics.pop("n")
    return {k: v / n for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    accum_steps     = max(1, cfg["accum_steps"])
    effective_batch = cfg["batch_size"] * accum_steps

    print(f"[train] Device          : {device}")
    print(f"[train] Model           : {cfg['model_type']}")
    print(f"[train] Physical batch  : {cfg['batch_size']}")
    print(f"[train] Accum steps     : {accum_steps}")
    print(f"[train] Effective batch : {effective_batch}")
    print(f"[train] Norm type       : {cfg['norm_type']}")
    print(f"[train] Train mixtures  : {cfg['num_mixtures_train']}")

    # ── Data — shared between Model A and Model B ─────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        data_root           = cfg["data_root"],
        metadata_csv        = cfg["metadata_csv"],
        num_mixtures_train  = cfg["num_mixtures_train"],
        num_mixtures_val    = cfg["num_mixtures_val"],
        num_mixtures_test   = cfg["num_mixtures_test"],
        batch_size          = cfg["batch_size"],
        num_workers         = cfg["num_workers"],
        sample_rate         = cfg["sample_rate"],
        clip_duration_s     = cfg["clip_duration_s"],
        seed                = cfg["seed"],
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model_kwargs = dict(
        n_filters   = cfg["n_filters"],
        filter_len  = cfg["filter_len"],
        bottleneck  = cfg["bottleneck"],
        hidden      = cfg["hidden"],
        kernel_size = cfg["kernel_size"],
        n_blocks    = cfg["n_blocks"],
        n_repeats   = cfg["n_repeats"],
        norm_type   = cfg["norm_type"],
        num_groups  = cfg["num_groups"],
    )
    model = build_model(cfg["model_type"], **model_kwargs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Parameters      : {n_params:,}")

    # ── Optimiser & scheduler ──────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                  patience=5)

    # ── Logs dir (run-wise) ─────────────────────────────────────────────────
    now = datetime.now()
    time_str = now.strftime("%H-%M")
    date_str = now.strftime("%m-%d")

    # Find next run number
    logs_base = Path("./logs") / cfg["model_type"]
    existing_runs = [d for d in logs_base.glob("run_*") if d.is_dir()]
    run_num = len(existing_runs) + 1

    run_name = f"run_{run_num} -- {time_str} -- {date_str}"
    log_dir = logs_base / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Checkpoint dir (run-wise) ────────────────────────────────────────────
    ckpt_dir = Path(cfg["ckpt_dir"]) / cfg["model_type"] / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config metadata with comment
    config_log = {
        "seed": cfg["seed"],
        "batch_size": cfg["batch_size"],
        "accum_steps": cfg["accum_steps"],
        "effective_batch": cfg["batch_size"] * max(1, cfg["accum_steps"]),
        "train_mixtures": cfg["num_mixtures_train"],
        "val_mixtures": cfg["num_mixtures_val"],
        "model_type": cfg["model_type"],
        "n_filters": cfg["n_filters"],
        "hidden": cfg["hidden"],
        "bottleneck": cfg["bottleneck"],
        "n_blocks": cfg["n_blocks"],
        "n_repeats": cfg["n_repeats"],
        "norm_type": cfg["norm_type"],
        "lr": cfg["lr"],
        "lambda_vib": cfg["lambda_vib"],
        "epochs": cfg["epochs"],
        "patience": cfg["patience"],
        "timestamp": now.strftime("%Y-%m-%d_%H-%M-%S"),
    }

    summary = f"// Run {run_num}: {cfg['model_type']} model with seed={cfg['seed']}, batch={cfg['batch_size']}, mixtures={cfg['num_mixtures_train']}\n"
    with open(log_dir / "config.json", "w") as f:
        f.write(summary)
        json.dump(config_log, f, indent=2)

    print(f"[train] Logging to: {log_dir}")

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_sisdr   = -float("inf")
    patience_counter = 0
    history          = []

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, device, cfg,
                                  accum_steps=accum_steps)
        val_m   = validate(model, val_loader, device, cfg)
        elapsed = time.time() - t0
        scheduler.step(val_m["si_sdr"])

        row = {
            "epoch":       epoch,
            "train_total": train_m["total"],
            "train_audio": train_m["audio"],
            "train_vib":   train_m["vib"],
            "val_total":   val_m["total"],
            "val_audio":   val_m["audio"],
            "val_vib":     val_m["vib"],
            "val_si_sdr":  val_m["si_sdr"],
            "lr":          optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        # ── Save metrics to CSV ───────────────────────────────────────
        csv_path = log_dir / "metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(history)

        print(
            f"Epoch {epoch:03d}/{cfg['epochs']} | "
            f"Train: {train_m['total']:.4f} "
            f"(L_audio={train_m['audio']:.4f}, L_vib={train_m['vib']:.4f}) | "
            f"Val SI-SDR: {val_m['si_sdr']:.2f} dB | "
            f"LR: {row['lr']:.2e} | {elapsed:.1f}s"
        )

        # ── Save latest ───────────────────────────────────────────────
        torch.save({
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg":             cfg,
            "history":         history,
        }, ckpt_dir / "latest.pt")

        # ── Save best ─────────────────────────────────────────────────
        if val_m["si_sdr"] > best_val_sisdr:
            best_val_sisdr   = val_m["si_sdr"]
            patience_counter = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "cfg":             cfg,
                "best_val_si_sdr": best_val_sisdr,
            }, ckpt_dir / "best.pt")
            print(f"  -> New best (val SI-SDR = {best_val_sisdr:.2f} dB) saved.")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"[train] Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg['patience']} epochs).")
                break

    print(f"[train] Done. Best val SI-SDR = {best_val_sisdr:.2f} dB")
    print(f"[train] Checkpoints: {ckpt_dir}")
    print(f"[train] Logs: {log_dir}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train bird source separation (Model A or Model B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Experiment ────────────────────────────────────────────────────
    p.add_argument("--model", type=str, default="baseline",
                   choices=["baseline", "multimodal"],
                   help="baseline=Model A (audio-only); multimodal=Model B (vibration-guided)")
    p.add_argument("--data_root", type=str, required=True,
                   help="Directory containing .wav clip files")
    p.add_argument("--metadata",  type=str, required=True,
                   help="Path to bird_songs_metadata.csv")

    # ── Mixture pool ──────────────────────────────────────────────────
    p.add_argument("--num_mixtures", type=int, default=3800,
                   help=(
                       "Fixed mixture pool size for training. "
                       "Both Model A and Model B use this SAME pool for fair comparison. "
                       "Recommended: ~3795 (= num train clips). "
                       "Use 2-4x for more training signal per epoch."
                   ))
    p.add_argument("--num_mixtures_val",  type=int, default=500,
                   help="Val pool size (smaller = faster epoch end).")
    p.add_argument("--num_mixtures_test", type=int, default=500,
                   help="Test pool size.")

    # ── Training schedule ─────────────────────────────────────────────
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--lambda_vib", type=float, default=0.1,
                   help="Weight of L_vib (Model B only; ignored for Model A).")

    # ── Batch / gradient accumulation ────────────────────────────────
    p.add_argument("--batch_size",  type=int, default=8,
                   help="Physical batch size per forward pass.")
    p.add_argument("--accum_steps", type=int, default=1,
                   help="Gradient accumulation steps. "
                        "Effective batch = batch_size * accum_steps.")

    # ── Normalisation ─────────────────────────────────────────────────
    p.add_argument("--norm_type",  type=str, default="gLN",
                   choices=["gLN", "gN", "iN", "lN"],
                   help="Per-sample norm (all are batch-size safe). "
                        "Use gN for batch_size <= 2.")
    p.add_argument("--num_groups", type=int, default=8,
                   help="Groups for GroupNorm (--norm_type gN only).")

    # ── Audio settings ────────────────────────────────────────────────
    p.add_argument("--sample_rate",  type=int,   default=22050)
    p.add_argument("--clip_dur",     type=float, default=3.0)
    p.add_argument("--num_workers",  type=int,   default=4)

    # ── I/O ───────────────────────────────────────────────────────────
    p.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    p.add_argument("--seed",     type=int, default=42)

    # ── Model architecture ────────────────────────────────────────────
    # ORIGINAL defaults (commented out for smaller model to reduce VRAM usage):
    # p.add_argument("--n_filters",   type=int, default=512)
    # p.add_argument("--filter_len",  type=int, default=16)
    # p.add_argument("--bottleneck",  type=int, default=128)
    # p.add_argument("--hidden",      type=int, default=512)
    # p.add_argument("--kernel_size", type=int, default=3)
    # p.add_argument("--n_blocks",    type=int, default=8)
    # p.add_argument("--n_repeats",   type=int, default=3)

    # SMALLER model defaults (reduced VRAM footprint):
    p.add_argument("--n_filters",   type=int, default=256)
    p.add_argument("--filter_len",  type=int, default=16)
    p.add_argument("--bottleneck",  type=int, default=64)
    p.add_argument("--hidden",      type=int, default=256)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--n_blocks",    type=int, default=6)
    p.add_argument("--n_repeats",   type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = dict(
        model_type          = args.model,
        data_root           = args.data_root,
        metadata_csv        = args.metadata,
        num_mixtures_train  = args.num_mixtures,
        num_mixtures_val    = args.num_mixtures_val,
        num_mixtures_test   = args.num_mixtures_test,
        epochs              = args.epochs,
        lr                  = args.lr,
        patience            = args.patience,
        lambda_vib          = args.lambda_vib,
        batch_size          = args.batch_size,
        accum_steps         = args.accum_steps,
        norm_type           = args.norm_type,
        num_groups          = args.num_groups,
        sample_rate         = args.sample_rate,
        clip_duration_s     = args.clip_dur,
        num_workers         = args.num_workers,
        ckpt_dir            = args.ckpt_dir,
        seed                = args.seed,
        n_filters           = args.n_filters,
        filter_len          = args.filter_len,
        bottleneck          = args.bottleneck,
        hidden              = args.hidden,
        kernel_size         = args.kernel_size,
        n_blocks            = args.n_blocks,
        n_repeats           = args.n_repeats,
    )
    train(cfg)
