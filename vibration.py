"""
vibration.py
------------
Simulates on-body accelerometer-style vibration signals from clean audio.

Design rationale (from thesis proposal):
  Real accelerometers capture low-frequency mechanical coupling of vocalizations
  through the bird's body. We approximate this by:
    1. Extracting a temporal energy envelope (short-time RMS)
    2. Low-pass filtering to keep only sub-300 Hz mechanical frequencies
    3. Adding mild Gaussian noise + transient artifacts to mimic sensor imperfections

This signal is used in TWO ways:
  - Per-source vibrations v1, v2  → auxiliary weak-supervision loss ONLY (training)
  - Mixture vibration mix_vib     → model conditioning input (train + inference)
"""

import torch
import torch.nn.functional as F


def simulate_vibration(
    audio: torch.Tensor,
    sr: int = 22050,
    frame_size: int = 512,
    hop_size: int = 128,
    lp_cutoff_hz: float = 300.0,
    noise_std: float = 0.005,
    transient_prob: float = 0.02,
) -> torch.Tensor:
    """
    Simulate a vibration signal from an audio waveform.

    Args:
        audio       : Tensor of shape [T] or [B, T] or [B, 1, T]
        sr          : Sample rate in Hz (default 22050)
        frame_size  : STFT-style frame for RMS envelope
        hop_size    : Hop between frames
        lp_cutoff_hz: Low-pass cutoff (mechanical frequency ceiling)
        noise_std   : Gaussian sensor noise standard deviation
        transient_prob: Per-sample probability of adding a transient spike

    Returns:
        vibration   : Tensor of the same shape as `audio`
    """
    # ── normalise shape to [B, T] ──────────────────────────────────────────
    original_shape = audio.shape
    squeeze_batch = False
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)          # [1, T]
        squeeze_batch = True
    elif audio.dim() == 3:
        audio = audio.squeeze(1)            # [B, 1, T] → [B, T]

    B, T = audio.shape

    # ── 1. Short-time RMS envelope ─────────────────────────────────────────
    # Unfold into overlapping frames: [B, num_frames, frame_size]
    audio_padded = F.pad(audio, (frame_size // 2, frame_size // 2))
    frames = audio_padded.unfold(-1, frame_size, hop_size)          # [B, F, frame_size]
    rms = frames.pow(2).mean(dim=-1).sqrt()                         # [B, F]

    # Upsample RMS back to original length
    rms = rms.unsqueeze(1)                                          # [B, 1, F]
    vib = F.interpolate(rms, size=T, mode='linear', align_corners=False)  # [B, 1, T]
    vib = vib.squeeze(1)                                            # [B, T]

    # ── 2. Low-pass filter (approximate with large average pool) ──────────
    # Cutoff ≈ sr / (2 * kernel_half_size); kernel chosen to match lp_cutoff_hz
    kernel_size = max(3, int(sr / (2 * lp_cutoff_hz)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    vib = vib.unsqueeze(1)                                          # [B, 1, T]
    vib = F.avg_pool1d(vib, kernel_size=kernel_size,
                       stride=1, padding=kernel_size // 2)
    vib = vib.squeeze(1)                                            # [B, T]

    # ── 3. Sensor noise ───────────────────────────────────────────────────
    noise = torch.randn_like(vib) * noise_std
    vib = vib + noise

    # ── 4. Sparse transient artifacts ────────────────────────────────────
    mask = torch.bernoulli(torch.full_like(vib, transient_prob))
    spikes = torch.randn_like(vib) * 0.1 * mask
    vib = vib + spikes

    # ── 5. Normalise to [-1, 1] per sample ───────────────────────────────
    peak = vib.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    vib = vib / peak

    # ── restore original shape ────────────────────────────────────────────
    if squeeze_batch:
        vib = vib.squeeze(0)
    elif len(original_shape) == 3:
        vib = vib.unsqueeze(1)

    return vib
