"""
loss.py
-------
Loss functions for the bird source separation system.

Variable names match the spec document exactly:

    Y1, Y2       clean source waveforms            (separation targets)
    Y1_hat, Y2_hat  model audio estimates          (model outputs)
    V1, V2       simulate_vibration(Y1/Y2)         (weak supervision targets)
    V1_hat, V2_hat  VibrationHead(Y1_hat/Y2_hat)  (model vib outputs, Model B)

Loss terms:
    L_audio  = PIT SI-SDR(Y_hat, Y)          — both models
    L_vib    = MSE(V_hat, V)                 — Model B only (weak supervision)
    L_total  = L_audio + lambda * L_vib

PIT (Permutation Invariant Training):
    We do not know a priori which output slot maps to which source.
    PIT tries both orderings [0→0,1→1] and [0→1,1→0] and picks the
    one that maximises SI-SDR.  The best permutation is then used to
    align V1/V2 targets for the vibration loss too.
"""

from itertools import permutations

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SI-SDR
# ─────────────────────────────────────────────────────────────────────────────

def si_sdr(estimate: torch.Tensor, reference: torch.Tensor,
           eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-Invariant SDR for a batch of single-channel signals.

    Args:
        estimate  : [B, T]
        reference : [B, T]
    Returns:
        si_sdr_val: [B]   (higher = better)
    """
    estimate  = estimate  - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)

    dot        = (estimate * reference).sum(dim=-1, keepdim=True)
    ref_energy = (reference ** 2).sum(dim=-1, keepdim=True) + eps
    proj       = (dot / ref_energy) * reference
    noise      = estimate - proj

    ratio = (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)          # [B]


# ─────────────────────────────────────────────────────────────────────────────
# PIT SI-SDR loss
# ─────────────────────────────────────────────────────────────────────────────

def pit_si_sdr_loss(
    estimates: torch.Tensor,   # [B, 2, T]  Y1_hat, Y2_hat stacked
    targets:   torch.Tensor,   # [B, 2, T]  Y1,     Y2     stacked
    eps: float = 1e-8,
) -> tuple:
    """
    Permutation Invariant SI-SDR loss.

    Returns:
        loss      : scalar  (mean negated SI-SDR, for minimisation)
        best_perms: [B, 2]  best permutation indices per sample
                            used to align V targets in vibration loss
    """
    B, n_src, T = estimates.shape
    assert targets.shape == estimates.shape, \
        f"Shape mismatch: estimates {estimates.shape} vs targets {targets.shape}"

    perms = list(permutations(range(n_src)))   # [(0,1), (1,0)]

    perm_sisdrs = []
    for perm in perms:
        perm_t = torch.stack([targets[:, i, :] for i in perm], dim=1)
        sisdrs = torch.stack(
            [si_sdr(estimates[:, s, :], perm_t[:, s, :], eps)
             for s in range(n_src)], dim=1           # [B, 2]
        ).mean(dim=1)                                # [B]
        perm_sisdrs.append(sisdrs)

    perm_sisdrs    = torch.stack(perm_sisdrs, dim=1) # [B, n_perms]
    best_sisdr, best_perm_idx = perm_sisdrs.max(dim=1)

    best_perms = torch.tensor(
        [perms[i] for i in best_perm_idx.tolist()],
        dtype=torch.long, device=estimates.device,
    )                                                # [B, 2]

    loss = -best_sisdr.mean()                        # scalar
    return loss, best_perms


# ─────────────────────────────────────────────────────────────────────────────
# Vibration auxiliary loss  (Model B — weak supervision)
# ─────────────────────────────────────────────────────────────────────────────

def vibration_auxiliary_loss(
    v_hat:      torch.Tensor,   # [B, 2, T]  V1_hat, V2_hat from VibrationHead
    V1:         torch.Tensor,   # [B, T]     simulate_vibration(Y1)
    V2:         torch.Tensor,   # [B, T]     simulate_vibration(Y2)
    best_perms: torch.Tensor,   # [B, 2]     from PIT — aligns hat with target
) -> torch.Tensor:
    """
    Weak supervision loss: compare model's predicted vibration outputs
    (V1_hat, V2_hat from VibrationHead) against the true per-source
    vibration targets (V1, V2 computed from clean sources).

    Steps:
        1. Stack V1, V2 → true_v [B, 2, T]
        2. Reorder rows using best_perms so hat slot s aligns with
           the correct true source (same permutation PIT found for audio)
        3. MSE(v_hat, true_v_aligned)

    Why v_hat comes from VibrationHead (not re-simulated):
        simulate_vibration uses Bernoulli sampling which has no gradient.
        The learned VibrationHead is fully differentiable, so the
        vibration loss propagates gradients back through the separator,
        forcing it to produce sources with correct energy profiles.

    Args:
        v_hat      : [B, 2, T]  VibrationHead(Y1_hat), VibrationHead(Y2_hat)
        V1         : [B, T]     simulate_vibration(Y1)   — ground truth
        V2         : [B, T]     simulate_vibration(Y2)   — ground truth
        best_perms : [B, 2]     permutation from PIT
    Returns:
        scalar MSE loss
    """
    B, n_src, T = v_hat.shape

    # Stack true vibration targets: [B, 2, T]
    true_v = torch.stack([V1, V2], dim=1)

    # Align: best_perms[b] = [i, j] means hat[b,0]→target[b,i], hat[b,1]→target[b,j]
    true_v_aligned = torch.zeros_like(true_v)
    for b in range(B):
        for s in range(n_src):
            true_v_aligned[b, s] = true_v[b, best_perms[b, s]]

    return F.mse_loss(v_hat, true_v_aligned)


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def total_loss(
    y_hat:       torch.Tensor,          # [B, 2, T]  audio estimates
    targets:     torch.Tensor,          # [B, 2, T]  Y1, Y2 stacked
    V1:          torch.Tensor,          # [B, T]     vibration target source 1
    V2:          torch.Tensor,          # [B, T]     vibration target source 2
    v_hat:       torch.Tensor | None,   # [B, 2, T]  VibrationHead outputs (Model B)
    lambda_vib:  float = 0.1,
    use_vib_loss: bool = False,         # True for Model B, False for Model A
) -> dict:
    """
    L_total = L_audio + lambda * L_vib

    Args:
        y_hat        : separated audio estimates from model
        targets      : [Y1, Y2] stacked along dim=1
        V1, V2       : per-source vibration targets (from dataset, NOT model input)
        v_hat        : [V1_hat, V2_hat] from VibrationHead (None for Model A)
        lambda_vib   : weight of the auxiliary vibration loss
        use_vib_loss : True only for Model B (multimodal)

    Returns dict with keys: total, audio, vib
    """
    # ── Audio separation loss (both models) ──────────────────────────
    l_audio, best_perms = pit_si_sdr_loss(y_hat, targets)

    # ── Vibration auxiliary loss (Model B only) ───────────────────────
    if use_vib_loss and lambda_vib > 0 and v_hat is not None:
        l_vib = vibration_auxiliary_loss(v_hat, V1, V2, best_perms)
    else:
        l_vib = torch.zeros(1, device=y_hat.device).squeeze()

    l_total = l_audio + lambda_vib * l_vib

    return {
        "total": l_total,
        "audio": l_audio,   # renamed from "sep" to match spec: L_audio
        "vib":   l_vib,
    }
