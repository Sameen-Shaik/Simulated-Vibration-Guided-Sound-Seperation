"""
utils/trainer.py
================
Training loop for the vibration-conditioned separator.

Features:
  - Mixed-precision optional
  - LR scheduling (cosine with warmup)
  - Early stopping
  - Checkpoint save/load
  - Tensorboard + CSV logging
  - Supports both vibration-conditioned and audio-only modes
"""

import sys
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import os
import json
import time
import math
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.losses import SeparationLoss


# ══════════════════════════════════════════════════════════════════════════════
#  Cosine LR with Linear Warmup
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.min_lr = min_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = max(scale, self.min_lr / self.base_lrs[0])

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale


# ══════════════════════════════════════════════════════════════════════════════
#  CSV Logger
# ══════════════════════════════════════════════════════════════════════════════

class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        self._header_written = False

    def log(self, row: dict):
        write_header = not os.path.exists(self.path) or not self._header_written
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
#  Trainer
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Manages the full training lifecycle for one model (vibration or baseline).
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,                  # "vibration" | "audio_only"
        cfg: dict,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.model_name = model_name
        self.cfg = cfg
        tr = cfg["training"]

        # Device
        if device is None:
            device = torch.device(
                tr.get("device", "cpu")
                if torch.cuda.is_available()
                else "cpu"
            )
        self.device = device
        self.model.to(self.device)

        # Optimiser
        self.optimizer = Adam(
            model.parameters(),
            lr=tr["learning_rate"],
            weight_decay=tr["weight_decay"],
        )

        # LR Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=tr.get("warmup_epochs", 5),
            total_epochs=tr["num_epochs"],
        )

        # Loss
        lw = tr["loss_weights"]
        self.loss_fn = SeparationLoss(
            time_weight=lw["time_domain"],
            spectral_weight=lw["spectral"],
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=self.device.type == "cuda")

        # Directories
        self.ckpt_dir = Path(tr["checkpoint_dir"]) / model_name
        self.log_dir = Path(tr.get("log_dir", "experiments/logs"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = CSVLogger(str(self.log_dir / f"{model_name}_training.csv"))

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.patience = tr.get("early_stopping_patience", 15)
        self.grad_clip = tr.get("grad_clip_norm", 5.0)
        self.num_epochs = tr["num_epochs"]

        # Whether to use vibration signals (audio-only baseline skips them)
        self.use_vibration = (model_name != "audio_only")

        print(f"[Trainer:{model_name}] device={self.device}, "
              f"use_vibration={self.use_vibration}")

    # ─────────────────────────────────────────────────────────────────────────

    def _run_epoch(self, loader, train: bool = True) -> Dict[str, float]:
        self.model.train(train)
        context = torch.enable_grad if train else torch.no_grad

        total_loss, total_sisdr, n_batches = 0.0, 0.0, 0

        desc = ("Train" if train else "Val") + f" [{self.model_name}]"
        with context():
            for batch in tqdm(loader, desc=desc, leave=False):
                mixture = batch["mixture"].to(self.device)         # (B, 1, T)
                sources = batch["sources"].to(self.device)         # (B, N, T)
                vibrations = batch["vibrations"].to(self.device)   # (B, N, T)

                vib_input = vibrations if self.use_vibration else None

                if train:
                    self.optimizer.zero_grad()

                with autocast(enabled=self.device.type == "cuda"):
                    estimates, _ = self.model(mixture, vib_input)
                    loss, metrics = self.loss_fn(estimates, sources)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += metrics["loss"]
                total_sisdr += metrics["si_sdr"]
                n_batches += 1

        n = max(n_batches, 1)
        return {"loss": total_loss / n, "si_sdr": total_sisdr / n}

    # ─────────────────────────────────────────────────────────────────────────

    def train(self, train_loader, val_loader) -> Dict:
        """Full training loop. Returns training history dict."""
        history = {"train_loss": [], "val_loss": [], "train_sisdr": [], "val_sisdr": []}
        print(f"\n{'='*60}")
        print(f"  Training: {self.model_name.upper()}")
        print(f"{'='*60}")

        for epoch in range(1, self.num_epochs + 1):
            self.scheduler.step(epoch - 1)
            t0 = time.time()

            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"  Epoch {epoch:03d}/{self.num_epochs} | "
                f"Train loss={train_metrics['loss']:.4f} SI-SDR={train_metrics['si_sdr']:.2f}dB | "
                f"Val loss={val_metrics['loss']:.4f} SI-SDR={val_metrics['si_sdr']:.2f}dB | "
                f"LR={lr:.6f} | {elapsed:.1f}s"
            )

            # Log
            row = {
                "epoch": epoch,
                "model": self.model_name,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_si_sdr": train_metrics["si_sdr"],
                "val_si_sdr": val_metrics["si_sdr"],
                "lr": lr,
                "elapsed_s": elapsed,
            }
            self.logger.log(row)
            for k in ["train_loss", "val_loss", "train_sisdr", "val_sisdr"]:
                history.setdefault(k, []).append(
                    train_metrics["loss"] if "train" in k else val_metrics["loss"]
                )

            # Checkpoint
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics["loss"], best=True)
                print(f"    ↳ New best val loss: {self.best_val_loss:.4f} — checkpoint saved")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch} (patience={self.patience})")
                    break

            # Periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_metrics["loss"], best=False)

        self._load_best_checkpoint()
        print(f"\n  Training complete. Best val loss: {self.best_val_loss:.4f}")
        return history

    # ─────────────────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, best: bool = False):
        fname = "best.pt" if best else f"epoch_{epoch:03d}.pt"
        path = self.ckpt_dir / fname
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "model_name": self.model_name,
            },
            path,
        )

    def _load_best_checkpoint(self):
        best_path = self.ckpt_dir / "best.pt"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded best checkpoint (epoch {ckpt['epoch']}, "
                  f"val_loss={ckpt['val_loss']:.4f})")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded checkpoint from {path}")

    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_batch(
        self, mixture: torch.Tensor, vibrations: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run inference on a single batch. Returns separated waveforms (B, N, T)."""
        self.model.eval()
        mixture = mixture.to(self.device)
        vib_input = vibrations.to(self.device) if (vibrations is not None and self.use_vibration) else None
        separated, _ = self.model(mixture, vib_input)
        return separated.cpu()
