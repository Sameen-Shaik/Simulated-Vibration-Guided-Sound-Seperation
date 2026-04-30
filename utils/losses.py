"""
utils/losses.py
===============
Loss functions for source separation training.

  1. SI-SDR loss           — Scale-Invariant Signal-to-Distortion Ratio
  2. Multi-scale spectral  — Spectral convergence + log spectral distance
  3. PIT wrapper           — Permutation-Invariant Training (find optimal assignment)
  4. Combined loss         — Weighted sum (time + spectral domains)
"""

import sys
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from typing import Tuple, List


# ══════════════════════════════════════════════════════════════════════════════
#  SI-SDR
# ══════════════════════════════════════════════════════════════════════════════

def si_sdr(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio (higher is better).

    Parameters
    ----------
    estimate : (B, T) or (B, N, T)
    target   : same shape as estimate

    Returns
    -------
    si_sdr_value : scalar mean over batch (and speakers if applicable)
    """
    # Flatten speaker dim into batch if needed
    if estimate.dim() == 3:
        B, N, T = estimate.shape
        estimate = estimate.view(B * N, T)
        target = target.view(B * N, T)

    # Remove mean
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)

    # Optimal scaling
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = dot / target_energy * target

    e_noise = estimate - s_target
    si_sdr_val = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    return si_sdr_val.mean()


def si_sdr_loss(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative SI-SDR (to minimise)."""
    return -si_sdr(estimate, target)


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-Scale Spectral Loss
# ══════════════════════════════════════════════════════════════════════════════

class MultiScaleSpectralLoss(nn.Module):
    """
    Combines spectral convergence and log spectral distance across
    multiple STFT window sizes.

    Reference: Defossez et al. (2019) — Demucs
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [128, 256, 512],
        win_sizes: List[int] = [512, 1024, 2048],
        eps: float = 1e-8,
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.stft_params = list(zip(fft_sizes, hop_sizes, win_sizes))
        self.eps = eps

    def forward(
        self, estimate: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        estimate, target : (B, T) or (B, N, T)
        """
        if estimate.dim() == 3:
            B, N, T = estimate.shape
            estimate = estimate.reshape(B * N, T)
            target = target.reshape(B * N, T)

        total_loss = torch.tensor(0.0, device=estimate.device)
        for n_fft, hop, win in self.stft_params:
            window = torch.hann_window(win, device=estimate.device)

            est_stft = torch.stft(
                estimate, n_fft=n_fft, hop_length=hop, win_length=win,
                window=window, return_complex=True
            )
            tgt_stft = torch.stft(
                target, n_fft=n_fft, hop_length=hop, win_length=win,
                window=window, return_complex=True
            )

            est_mag = est_stft.abs() + self.eps
            tgt_mag = tgt_stft.abs() + self.eps

            # Spectral convergence: ||tgt_mag - est_mag|| / ||tgt_mag||
            sc_loss = torch.norm(tgt_mag - est_mag, p="fro") / (torch.norm(tgt_mag, p="fro") + self.eps)

            # Log spectral distance
            lsd_loss = (torch.log(tgt_mag) - torch.log(est_mag)).abs().mean()

            total_loss = total_loss + sc_loss + lsd_loss

        return total_loss / len(self.stft_params)


# ══════════════════════════════════════════════════════════════════════════════
#  Permutation-Invariant Training (PIT)
# ══════════════════════════════════════════════════════════════════════════════

def pit_loss(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    loss_fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the best permutation of estimates matching targets and return
    the minimum-loss assignment.

    Parameters
    ----------
    estimates : (B, N, T) — model output for N speakers
    targets   : (B, N, T) — clean reference signals
    loss_fn   : callable(estimate, target) → scalar

    Returns
    -------
    min_loss : scalar
    best_perm : (B, N) best permutation indices
    """
    B, N, T = estimates.shape
    perms = list(permutations(range(N)))

    losses = torch.stack([
        torch.stack([
            loss_fn(estimates[:, perm_idx, :], targets)
            for perm_idx in [list(p) for p in perms]
        ], dim=0)
    ], dim=0).squeeze(0)   # (num_perms, B) — approximate via mean

    # For small N (2), exhaustive is fast; compute per-sample
    perm_losses = []
    for perm in perms:
        perm_t = torch.tensor(perm, device=estimates.device)
        est_perm = estimates[:, perm_t, :]    # (B, N, T)
        loss_per_sample = torch.stack([
            loss_fn(est_perm[b].unsqueeze(0), targets[b].unsqueeze(0))
            for b in range(B)
        ])
        perm_losses.append(loss_per_sample)

    perm_losses_t = torch.stack(perm_losses, dim=1)   # (B, num_perms)
    best_perm_idx = perm_losses_t.argmin(dim=1)        # (B,)
    min_loss = perm_losses_t.gather(1, best_perm_idx.unsqueeze(1)).mean()

    best_perms = torch.tensor(
        [perms[idx] for idx in best_perm_idx.tolist()],
        device=estimates.device,
    )
    return min_loss, best_perms


# ══════════════════════════════════════════════════════════════════════════════
#  Combined Loss
# ══════════════════════════════════════════════════════════════════════════════

class SeparationLoss(nn.Module):
    """
    Combined time-domain (SI-SDR) + spectral loss with PIT.
    """

    def __init__(
        self,
        time_weight: float = 0.5,
        spectral_weight: float = 0.5,
        fft_sizes: List[int] = [512, 1024, 2048],
    ):
        super().__init__()
        assert abs(time_weight + spectral_weight - 1.0) < 1e-6, \
            "Weights must sum to 1"
        self.time_weight = time_weight
        self.spectral_weight = spectral_weight
        self.spectral_loss = MultiScaleSpectralLoss(
            fft_sizes=fft_sizes,
            hop_sizes=[f // 4 for f in fft_sizes],
            win_sizes=fft_sizes,
        )

    def _combined(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss for one permutation (B, N, T) pairs."""
        t_loss = si_sdr_loss(estimate, target)
        s_loss = self.spectral_loss(estimate, target)
        return self.time_weight * t_loss + self.spectral_weight * s_loss

    def forward(
        self,
        estimates: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        estimates : (B, N, T)
        targets   : (B, N, T)

        Returns
        -------
        total_loss : scalar
        metrics    : dict with component losses
        """
        # PIT: find best permutation
        pit, best_perm = pit_loss(estimates, targets, self._combined)

        # Reorder estimates by best permutation for metric logging
        B = estimates.shape[0]
        reordered = torch.stack(
            [estimates[b, best_perm[b]] for b in range(B)], dim=0
        )

        # Compute component losses for logging
        with torch.no_grad():
            t_loss = si_sdr_loss(reordered, targets).item()
            s_loss = self.spectral_loss(reordered, targets).item()
            sdr_val = si_sdr(reordered, targets).item()

        metrics = {
            "loss": pit.item(),
            "si_sdr": sdr_val,
            "time_loss": t_loss,
            "spectral_loss": s_loss,
        }
        return pit, metrics


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Loss Function Smoke Test ===")
    B, N, T = 2, 2, 22050 * 3

    estimates = torch.randn(B, N, T)
    targets = torch.randn(B, N, T)

    loss_fn = SeparationLoss(time_weight=0.5, spectral_weight=0.5)
    loss, metrics = loss_fn(estimates, targets)
    print(f"  Loss : {loss.item():.4f}")
    print(f"  SI-SDR: {metrics['si_sdr']:.2f} dB")
    print("  ✓ Loss functions OK")
