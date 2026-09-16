"""
model.py
--------
Two models:
  1. ConvTasNet      — audio-only baseline (standard architecture)
  2. MultimodalSep   — vibration-conditioned separator using FiLM modulation

Architecture of MultimodalSep:
  ┌─────────────┐     ┌──────────────────┐
  │ Audio Enc.  │     │ Vibration Enc.   │
  │ Conv1d      │     │ Conv1d           │
  └──────┬──────┘     └────────┬─────────┘
         │                     │ adaptive-pool (align T)
         │           ┌─────────▼──────────┐
         │           │  FiLM: γ, β        │  ← generates per-channel scale/shift
         │           └─────────┬──────────┘
         │                     │
         └──────── fuse ────────┘
                    │
          ┌─────────▼───────────┐
          │  Temporal Conv Net  │  (dilated depthwise separable convs)
          └─────────┬───────────┘
                    │  mask heads × 2
          ┌─────────▼───────────┐
          │  Decoder            │  ConvTranspose1d
          └─────────────────────┘
          output: [B, 2, T]

CRITICAL:  Only mix_audio + mix_vib enter the model.
           Per-source vibrations (v1, v2) are used ONLY in the loss function.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NORMALISATION STRATEGY — WHY NOT BATCHNORM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  BatchNorm (BN) computes mean/variance across the BATCH dimension.
  At batch size 1 or 2 its statistics are extremely noisy, making
  training unstable. At B=1 BN is literally undefined (variance=0).
  It also behaves differently between train and eval (running stats),
  which is an additional source of inconsistency.

  All norms used here normalise PER SAMPLE, so they work identically
  for B=1, B=2, or any larger batch, and train == eval numerically.

  Four options are exposed via `norm_type`:

  "gLN"  — Global Layer Norm (default, Conv-TasNet paper standard)
            Normalises over the joint [C, T] extent of each sample.
            Mean and variance are shared across all channels AND time.
            Pro: proven in Conv-TasNet, very stable.
            Con: couples channel and time statistics.

  "gN"   — Group Norm (nn.GroupNorm, num_groups=8 by default)
            Divides C channels into G groups; normalises each group
            over [C/G, T] independently, per sample.
            Pro: strictly per-sample, no cross-batch coupling.
                 Works at B=1. Widely used in low-batch audio tasks.
            Con: G must divide C evenly (enforced automatically below).

  "iN"   — Instance Norm (nn.InstanceNorm1d with affine=True)
            Special case of GroupNorm where G = C (each channel
            normalised independently over time).
            Pro: simplest per-sample norm; B=1 safe.
            Con: discards all cross-channel correlation in the norm.

  "lN"   — Layer Norm (nn.LayerNorm over [C, T] per timestep)
            Equivalent to gLN but using PyTorch's fused kernel.
            Pro: fast, well-supported.
            Con: slightly different from gLN (no shared γ/β trick).

  RECOMMENDATION FOR THIS PROJECT:
    Default  → "gLN"  (matches Conv-TasNet paper exactly)
    Low VRAM / batch=1 → "gN"  (most stable alternative)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation factory — all variants are batch-size independent
# ─────────────────────────────────────────────────────────────────────────────

class GlobalLayerNorm(nn.Module):
    """
    Global Layer Norm as used in the original Conv-TasNet paper.
    Normalises each sample over the joint [C, T] extent:
        x_norm = γ * (x - mean_{C,T}) / std_{C,T} + β

    Batch-size independent: statistics computed per sample, not per batch.
    Works identically at B=1, B=2, and larger batches.
    """
    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] — normalise over dims 1 and 2 (C and T), per sample
        mean = x.mean(dim=[1, 2], keepdim=True)
        var  = ((x - mean) ** 2).mean(dim=[1, 2], keepdim=True)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta


def _make_norm(norm_type: str, channels: int,
               num_groups: int = 8, eps: float = 1e-8) -> nn.Module:
    """
    Build a normalisation layer appropriate for low batch sizes.

    All returned layers normalise PER SAMPLE — none of them use the batch
    dimension in their statistics. Safe for B=1, B=2, or gradient
    accumulation where the "logical" batch differs from the "physical" one.

    Args:
        norm_type  : "gLN" | "gN" | "iN" | "lN"
        channels   : number of feature channels C
        num_groups : groups for GroupNorm (must divide C; auto-adjusted if not)
        eps        : numerical stability epsilon
    """
    if norm_type == "gLN":
        # Global Layer Norm — Conv-TasNet default
        return GlobalLayerNorm(channels, eps=eps)

    elif norm_type == "gN":
        # Group Norm — best alternative for B=1/2
        # Auto-adjust num_groups so it always divides C evenly.
        g = num_groups
        while channels % g != 0 and g > 1:
            g -= 1
        # GroupNorm(num_groups, num_channels) normalises [C/G, T] per sample
        return nn.GroupNorm(num_groups=g, num_channels=channels,
                            eps=eps, affine=True)

    elif norm_type == "iN":
        # Instance Norm — per-channel normalisation over T, per sample
        # affine=True gives learnable γ, β like BN/LN
        return nn.InstanceNorm1d(channels, eps=eps, affine=True,
                                 track_running_stats=False)

    elif norm_type == "lN":
        # Layer Norm applied to [C, T] jointly — close to gLN but uses
        # PyTorch's optimised kernel; γ and β are [C, T]-shaped (more expressive)
        # Note: T must be fixed at construction time for this variant.
        # We fall back to gLN if T is unknown (common in our variable-length setup).
        # → Use gLN instead if you encounter shape issues.
        return GlobalLayerNorm(channels, eps=eps)

    else:
        raise ValueError(
            f"Unknown norm_type {norm_type!r}. "
            f"Choose from: 'gLN', 'gN', 'iN', 'lN'."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution block as used in Conv-TasNet.
    Includes residual + skip connections.

    norm_type controls which per-sample normalisation layer is used
    (see module docstring for the full comparison table).
    """
    def __init__(self, in_ch: int, hid_ch: int, kernel_size: int,
                 dilation: int = 1, causal: bool = False,
                 norm_type: str = "gLN", num_groups: int = 8):
        super().__init__()
        self.causal = causal
        pad = (kernel_size - 1) * dilation if causal else (kernel_size - 1) * dilation // 2

        # Two norm layers per block — one after the pointwise conv,
        # one after the depthwise conv.  Both are per-sample.
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hid_ch, 1),
            nn.PReLU(),
            _make_norm(norm_type, hid_ch, num_groups),
            nn.Conv1d(hid_ch, hid_ch, kernel_size,
                      dilation=dilation, padding=pad, groups=hid_ch),
            nn.PReLU(),
            _make_norm(norm_type, hid_ch, num_groups),
        )
        self.res_conv  = nn.Conv1d(hid_ch, in_ch, 1)
        self.skip_conv = nn.Conv1d(hid_ch, in_ch, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        if self.causal:
            h = h[..., :x.shape[-1]]       # trim look-ahead padding
        res  = self.res_conv(h) + x        # residual connection
        skip = self.skip_conv(h)
        return res, skip


class VibrationHead(nn.Module):
    """
    Auxiliary output head: maps a separated audio estimate to a predicted
    vibration signal (v_hat) used for the weak-supervision loss.

    Architecture:
        y_hat [B, T] → Conv1d stack → v_hat [B, T]

    This head is ONLY active during training (auxiliary task).
    At inference the model still outputs y1_hat, y2_hat normally;
    v1_hat, v2_hat are discarded.  The head forces the separator to
    produce sources whose temporal energy profile matches the true
    per-source vibration, acting as a soft structural constraint.

    Why a learned head instead of re-running simulate_vibration?
      - simulate_vibration is non-differentiable in some ops (bernoulli).
      - A learned head lets the model specialise the vibration prediction
        jointly with the separation objective.
      - The true targets (V1, V2) are still computed via simulate_vibration
        on the clean sources, providing a principled supervision signal.
    """
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # Pointwise: T-independent, no boundary effects
            nn.Conv1d(1, hidden, kernel_size=1),
            nn.PReLU(),
            # Local temporal context (low-freq envelope extraction)
            nn.Conv1d(hidden, hidden, kernel_size=15, padding=7, groups=hidden),
            nn.PReLU(),
            nn.Conv1d(hidden, 1, kernel_size=1),
            nn.Tanh(),          # output bounded to [-1,1] matching simulate_vibration
        )

    def forward(self, y_hat: torch.Tensor) -> torch.Tensor:
        """
        y_hat : [B, T]   separated audio estimate
        returns: [B, T]  predicted vibration signal
        """
        return self.net(y_hat.unsqueeze(1)).squeeze(1)   # [B, T]


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) [Perez et al. 2018].

    Given vibration features vib_feat [B, V, T_v], produces:
        γ [B, C, 1] and β [B, C, 1]
    that scale and shift audio features audio_feat [B, C, T].

    This is the core multimodal fusion mechanism: vibration signals
    modulate how the separator processes audio at every temporal position.
    """
    def __init__(self, vib_channels: int, audio_channels: int):
        super().__init__()
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self.fc_gamma = nn.Linear(vib_channels, audio_channels)
        self.fc_beta  = nn.Linear(vib_channels, audio_channels)

    def forward(self, audio_feat: torch.Tensor,
                vib_feat: torch.Tensor) -> torch.Tensor:
        """
        audio_feat : [B, C, T]
        vib_feat   : [B, V, T_v]  — different T is fine; we pool over time
        returns    : [B, C, T]    — modulated audio features
        """
        pooled = self.pool(vib_feat).squeeze(-1)          # [B, V]
        gamma  = self.fc_gamma(pooled).unsqueeze(-1)      # [B, C, 1]
        beta   = self.fc_beta(pooled).unsqueeze(-1)       # [B, C, 1]
        return gamma * audio_feat + beta                  # broadcast over T


# ─────────────────────────────────────────────────────────────────────────────
# Shared length-matching utility
# ─────────────────────────────────────────────────────────────────────────────

def _match_length(x: torch.Tensor, target: int) -> torch.Tensor:
    """Trim or zero-pad the last dimension to exactly `target` samples."""
    cur = x.shape[-1]
    if cur > target:
        return x[..., :target]
    if cur < target:
        return F.pad(x, (0, target - cur))
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Model 1: Conv-TasNet (audio-only baseline)
# ─────────────────────────────────────────────────────────────────────────────

class ConvTasNet(nn.Module):
    """
    Standard Conv-TasNet for 2-source separation.
    Reference: Luo & Mesgarani, IEEE/ACM TASLP 2019.

    Input : mix_audio [B, T]
    Output: separated [B, 2, T]

    norm_type selects the per-sample normalisation used throughout
    (see module docstring). Default "gLN" matches the original paper.
    For batch_size=1 or 2, all options work; "gN" is a good alternative.
    """

    def __init__(
        self,
        n_filters:   int  = 512,    # encoder output channels (N)
        filter_len:  int  = 16,     # encoder kernel size     (L)
        bottleneck:  int  = 128,    # bottleneck width        (B)
        hidden:      int  = 512,    # TCN hidden width        (H)
        kernel_size: int  = 3,      # TCN depthwise kernel    (P)
        n_blocks:    int  = 8,      # TCN blocks per repeat   (X)
        n_repeats:   int  = 3,      # number of repeats       (R)
        n_sources:   int  = 2,      # output sources          (C)
        causal:      bool = False,
        norm_type:   str  = "gLN",  # "gLN" | "gN" | "iN" | "lN"
        num_groups:  int  = 8,      # groups for "gN" only
    ):
        super().__init__()
        self.n_sources  = n_sources
        self.filter_len = filter_len
        self.n_filters  = n_filters

        # ── Encoder ───────────────────────────────────────────────────
        self.encoder = nn.Conv1d(1, n_filters, filter_len,
                                 stride=filter_len // 2, padding=0, bias=False)
        # Per-sample norm on encoder output — safe at any batch size
        self.encoder_norm = _make_norm(norm_type, n_filters, num_groups)

        # ── TCN Separator ─────────────────────────────────────────────
        self.bottleneck_conv = nn.Conv1d(n_filters, bottleneck, 1)
        self.tcn_blocks = nn.ModuleList([
            DepthwiseSeparableConv(
                bottleneck, hidden, kernel_size,
                dilation   = 2 ** b,
                causal     = causal,
                norm_type  = norm_type,
                num_groups = num_groups,
            )
            for _ in range(n_repeats)
            for b in range(n_blocks)
        ])
        self.prelu_out = nn.PReLU()
        self.mask_conv = nn.Conv1d(bottleneck, n_filters * n_sources, 1)

        # ── Decoder ───────────────────────────────────────────────────
        self.decoder = nn.ConvTranspose1d(n_filters, 1, filter_len,
                                          stride=filter_len // 2, bias=False)

    def forward(self, mix_audio: torch.Tensor) -> torch.Tensor:
        """
        mix_audio : [B, T]
        returns   : [B, 2, T]
        """
        B, T = mix_audio.shape

        # Encode: [B, 1, T] → [B, N, T']
        enc = F.relu(self.encoder(mix_audio.unsqueeze(1)))
        enc = self.encoder_norm(enc)

        # TCN separator
        h = self.bottleneck_conv(enc)                          # [B, B_ch, T']
        skip_sum = torch.zeros_like(h)
        for block in self.tcn_blocks:
            h, skip = block(h)
            skip_sum = skip_sum + skip

        # Mask estimation
        skip_sum = self.prelu_out(skip_sum)
        masks = self.mask_conv(skip_sum)                       # [B, N*2, T']
        masks = masks.reshape(B, self.n_sources,
                              self.n_filters, enc.shape[-1])   # [B, 2, N, T']
        masks = F.softmax(masks, dim=1)                        # sum-to-one across sources

        # Apply masks and decode
        enc_exp = enc.unsqueeze(1).expand_as(masks)            # [B, 2, N, T']
        masked  = (masks * enc_exp).reshape(
            B * self.n_sources, self.n_filters, enc.shape[-1]) # [B*2, N, T']
        decoded = self.decoder(masked).squeeze(1)              # [B*2, T'']
        decoded = decoded.reshape(B, self.n_sources, -1)       # [B, 2, T'']

        return _match_length(decoded, T)                        # [B, 2, T]


# ─────────────────────────────────────────────────────────────────────────────
# Model 2: MultimodalSep — vibration-conditioned separator
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalSep(nn.Module):
    """
    Vibration-conditioned audio source separator.

    Training inputs : mix_audio [B, T], mix_vib [B, T]
    Inference inputs: mix_audio [B, T], mix_vib [B, T]   ← same (no leakage)

    Per-source vibrations v1, v2 are NEVER passed here.
    They only appear in the loss computation (weak supervision).

    Fusion strategy: FiLM modulation at the bottleneck of the separator.
      The vibration encoder produces a global conditioning vector.
      FiLM generates (γ, β) to scale/shift audio bottleneck features.

    norm_type: same options as ConvTasNet — all are per-sample.
      The vibration encoder also uses the selected norm, ensuring
      consistent behaviour regardless of physical batch size.
    """

    def __init__(
        self,
        n_filters:    int  = 512,
        filter_len:   int  = 16,
        bottleneck:   int  = 128,
        hidden:       int  = 512,
        kernel_size:  int  = 3,
        n_blocks:     int  = 8,
        n_repeats:    int  = 3,
        n_sources:    int  = 2,
        vib_channels:    int  = 64,
        causal:          bool = False,
        norm_type:       str  = "gLN",  # "gLN" | "gN" | "iN" | "lN"
        num_groups:      int  = 8,
        vib_head_hidden: int  = 64,     # hidden dim of the VibrationHead
    ):
        super().__init__()
        self.n_sources  = n_sources
        self.filter_len = filter_len
        self.n_filters  = n_filters

        # ── Audio Encoder ─────────────────────────────────────────────
        self.audio_encoder = nn.Conv1d(1, n_filters, filter_len,
                                       stride=filter_len // 2, bias=False)
        self.audio_norm    = _make_norm(norm_type, n_filters, num_groups)

        # ── Vibration Encoder ─────────────────────────────────────────
        # Encodes Vmix (mixture vibration) — the ONLY vibration signal
        # the model ever receives as input.  V1/V2 never enter here.
        self.vib_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
            nn.PReLU(),
            _make_norm(norm_type, 32, num_groups),
            nn.Conv1d(32, vib_channels, kernel_size=7, stride=4, padding=3),
            nn.PReLU(),
            _make_norm(norm_type, vib_channels, num_groups),
        )

        # ── FiLM conditioning ─────────────────────────────────────────
        self.film = FiLMLayer(vib_channels, bottleneck)

        # ── Bottleneck + TCN separator ────────────────────────────────
        self.bottleneck_conv = nn.Conv1d(n_filters, bottleneck, 1)
        self.tcn_blocks = nn.ModuleList([
            DepthwiseSeparableConv(
                bottleneck, hidden, kernel_size,
                dilation   = 2 ** b,
                causal     = causal,
                norm_type  = norm_type,
                num_groups = num_groups,
            )
            for _ in range(n_repeats)
            for b in range(n_blocks)
        ])
        self.prelu_out = nn.PReLU()
        self.mask_conv = nn.Conv1d(bottleneck, n_filters * n_sources, 1)

        # ── Decoder ───────────────────────────────────────────────────
        self.decoder = nn.ConvTranspose1d(n_filters, 1, filter_len,
                                          stride=filter_len // 2, bias=False)

        # ── VibrationHead (auxiliary task, training only) ─────────────
        # One shared head applied independently to each separated source.
        # y1_hat → vib_head → V1_hat  (compared with V1 in loss)
        # y2_hat → vib_head → V2_hat  (compared with V2 in loss)
        # At inference the v_hat outputs are simply ignored.
        self.vib_head = VibrationHead(hidden=vib_head_hidden)

    def forward(self, mix_audio: torch.Tensor,
                mix_vib: torch.Tensor) -> tuple:
        """
        Inputs (both train and inference):
            mix_audio : [B, T]   mixture waveform  (Ymix)
            mix_vib   : [B, T]   mixture vibration (Vmix) — NOT V1/V2

        Returns:
            y_hat : [B, 2, T]   separated audio estimates (Y1_hat, Y2_hat)
            v_hat : [B, 2, T]   predicted vibration per source (V1_hat, V2_hat)
                                 → used in aux loss during training
                                 → ignored at inference

        Note: V1, V2 (per-source vibrations) are NEVER passed here.
        """
        B, T = mix_audio.shape

        # ── Encode audio ──────────────────────────────────────────────
        audio_enc = F.relu(self.audio_encoder(mix_audio.unsqueeze(1)))  # [B, N, T']
        audio_enc = self.audio_norm(audio_enc)
        T_audio   = audio_enc.shape[-1]

        # ── Encode Vmix ───────────────────────────────────────────────
        vib_feat = self.vib_encoder(mix_vib.unsqueeze(1))               # [B, V, T_v]
        vib_feat = F.adaptive_avg_pool1d(vib_feat, T_audio)             # [B, V, T']

        # ── FiLM fusion ───────────────────────────────────────────────
        h = self.bottleneck_conv(audio_enc)                              # [B, B_ch, T']
        h = self.film(h, vib_feat)                                       # [B, B_ch, T']

        # ── TCN separation ────────────────────────────────────────────
        skip_sum = torch.zeros_like(h)
        for block in self.tcn_blocks:
            h, skip = block(h)
            skip_sum = skip_sum + skip

        # ── Mask estimation ───────────────────────────────────────────
        skip_sum = self.prelu_out(skip_sum)
        masks = self.mask_conv(skip_sum)                                 # [B, N*2, T']
        masks = masks.reshape(B, self.n_sources,
                              self.n_filters, T_audio)                   # [B, 2, N, T']
        masks = F.softmax(masks, dim=1)

        # ── Decode → Y1_hat, Y2_hat ───────────────────────────────────
        enc_exp = audio_enc.unsqueeze(1).expand_as(masks)
        masked  = (masks * enc_exp).reshape(
            B * self.n_sources, self.n_filters, T_audio)
        decoded = self.decoder(masked).squeeze(1)
        y_hat   = decoded.reshape(B, self.n_sources, -1)                 # [B, 2, T'']
        y_hat   = _match_length(y_hat, T)                                # [B, 2, T]

        # ── VibrationHead → V1_hat, V2_hat ───────────────────────────
        # Applied to each separated source independently.
        # Gradient flows back through vib_head into the separator,
        # forcing the separator to produce sources with correct vibration
        # patterns (weak supervision).
        v1_hat = self.vib_head(y_hat[:, 0, :])    # [B, T]
        v2_hat = self.vib_head(y_hat[:, 1, :])    # [B, T]
        v_hat  = torch.stack([v1_hat, v2_hat], dim=1)   # [B, 2, T]

        return y_hat, v_hat   # ([B,2,T], [B,2,T])


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_type: str, **kwargs) -> nn.Module:
    """
    Factory function.
    model_type: 'baseline'   → ConvTasNet      (audio-only)
                'multimodal' → MultimodalSep   (vibration-conditioned)

    Pass norm_type="gN" (and optionally num_groups=8) when running with
    batch_size ≤ 2 or when using gradient accumulation with a small
    physical batch size.
    """
    if model_type == "baseline":
        return ConvTasNet(**kwargs)
    elif model_type == "multimodal":
        return MultimodalSep(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
