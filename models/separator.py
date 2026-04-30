"""
models/separator.py
===================
Multimodal neural audio source separator with FiLM conditioning.

Architecture (three main components per thesis proposal):

  ┌─────────────────────────────────────────────────────────────┐
  │  AudioEncoder                                               │
  │    Conv1d encoder → latent audio representation            │
  │                                                             │
  │  VibrationEncoder  (per bird, separate weights)            │
  │    1-D conv stack → temporal activity + identity features  │
  │                                                             │
  │  SeparatorDecoder with FiLM conditioning                   │
  │    Stacked TCN blocks + FiLM modulation from vibration     │
  │    → masked latent codes → decoder → N waveforms           │
  └─────────────────────────────────────────────────────────────┘

Also provides:
  AudioOnlyBaseline  – identical architecture without vibration path
"""

import sys
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  Utility Blocks
# ══════════════════════════════════════════════════════════════════════════════

class GlobalLayerNorm(nn.Module):
    """Global Layer Normalisation (gLN) as used in Conv-TasNet."""

    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=[1, 2], keepdim=True)
        var = ((x - mean) ** 2).mean(dim=[1, 2], keepdim=True)
        return self.gamma * (x - mean) / (var + self.eps).sqrt() + self.beta


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable causal convolution block (TCN building block)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.PReLU(),
            GlobalLayerNorm(hidden_channels),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
                groups=hidden_channels,
            ),
            nn.PReLU(),
            GlobalLayerNorm(hidden_channels),
            nn.Conv1d(hidden_channels, in_channels, 1),
        )
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trim non-causal padding and add residual
        out = self.net(x)
        out = out[..., : x.shape[-1]]
        return out + x


# ══════════════════════════════════════════════════════════════════════════════
#  FiLM: Feature-wise Linear Modulation
# ══════════════════════════════════════════════════════════════════════════════

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Conditions audio features on vibration features via affine transform:
        output = gamma(vib) * audio + beta(vib)

    Reference: Perez et al. (2018) AAAI
    """

    def __init__(self, audio_channels: int, vib_feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(vib_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * audio_channels),  # gamma + beta
        )
        self.audio_channels = audio_channels

    def forward(
        self, audio_feat: torch.Tensor, vib_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        audio_feat : (B, C_audio, T)
        vib_feat   : (B, C_vib)  — global vibration representation

        Returns
        -------
        modulated  : (B, C_audio, T)
        """
        params = self.generator(vib_feat)           # (B, 2*C_audio)
        gamma, beta = params.chunk(2, dim=-1)       # each (B, C_audio)
        gamma = gamma.unsqueeze(-1)                 # (B, C_audio, 1)
        beta = beta.unsqueeze(-1)
        return (1.0 + gamma) * audio_feat + beta    # affine modulation


# ══════════════════════════════════════════════════════════════════════════════
#  Component 1: Audio Encoder
# ══════════════════════════════════════════════════════════════════════════════

class AudioEncoder(nn.Module):
    """
    1-D convolutional encoder: waveform → latent representation.
    Equivalent to the analysis filterbank in Conv-TasNet.
    """

    def __init__(
        self,
        out_channels: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(
            1, out_channels, kernel_size, stride=stride, bias=False
        )
        self.norm = GlobalLayerNorm(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T_audio)
        w = self.relu(self.conv(x))     # (B, C, T_frames)
        return self.norm(w)


# ══════════════════════════════════════════════════════════════════════════════
#  Component 2: Vibration Encoder (per bird)
# ══════════════════════════════════════════════════════════════════════════════

class VibrationEncoder(nn.Module):
    """
    Encodes a single bird's vibration signal into a fixed-size global feature
    vector that captures temporal activity and bird identity.

    Input:  (B, T_audio) — raw vibration waveform
    Output: (B, C_vib)   — global vibration representation
    """

    def __init__(
        self,
        out_channels: int = 128,
        num_layers: int = 4,
        kernel_size: int = 8,
        stride: int = 4,
    ):
        super().__init__()
        layers = []
        in_ch = 1
        for i in range(num_layers):
            out_ch = out_channels if i == num_layers - 1 else out_channels // 2
            out_ch = max(out_ch, 32)
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(in_ch, out_channels)

    def forward(self, vib: torch.Tensor) -> torch.Tensor:
        # vib: (B, T)
        x = vib.unsqueeze(1)                            # (B, 1, T)
        x = self.conv_stack(x)                          # (B, C, T')
        x = self.global_pool(x).squeeze(-1)             # (B, C)
        return self.project(x)                          # (B, C_vib)


# ══════════════════════════════════════════════════════════════════════════════
#  Component 3: TCN Separator with FiLM Conditioning
# ══════════════════════════════════════════════════════════════════════════════

class FiLMConditionedTCN(nn.Module):
    """
    Temporal Convolutional Network with FiLM conditioning blocks.
    Takes audio latent + aggregated vibration feature → N speaker masks.
    """

    def __init__(
        self,
        audio_channels: int = 256,
        tcn_channels: int = 256,
        kernel_size: int = 3,
        num_layers: int = 8,
        num_stacks: int = 3,
        num_speakers: int = 2,
        vib_feature_dim: int = 128,
        film_hidden: int = 256,
    ):
        super().__init__()
        self.num_speakers = num_speakers

        # Layer norm before TCN
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(audio_channels),
            nn.Linear(audio_channels, tcn_channels),
        )

        # Build stacked TCN blocks
        self.tcn_blocks = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        total_blocks = num_stacks * num_layers
        for i in range(total_blocks):
            dilation = 2 ** (i % num_layers)
            self.tcn_blocks.append(
                DepthwiseSeparableConv(tcn_channels, tcn_channels, kernel_size, dilation)
            )
            # FiLM after every block
            self.film_layers.append(
                FiLMLayer(tcn_channels, vib_feature_dim, film_hidden)
            )

        # Output: predict N speaker masks
        self.mask_head = nn.Sequential(
            nn.Conv1d(tcn_channels, num_speakers * audio_channels, 1),
            nn.Sigmoid(),
        )
        self.tcn_channels = tcn_channels
        self.audio_channels = audio_channels

    def forward(
        self,
        audio_latent: torch.Tensor,
        vib_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        audio_latent : (B, C_audio, T_frames)
        vib_feat     : (B, C_vib) aggregated vibration feature; None for audio-only

        Returns
        -------
        masks : (B, N, C_audio, T_frames)
        """
        B, C, T = audio_latent.shape

        # Project to TCN dimension: transpose for LayerNorm then back
        x = audio_latent.permute(0, 2, 1)    # (B, T, C)
        x = self.bottleneck(x)               # (B, T, tcn_ch)
        x = x.permute(0, 2, 1)              # (B, tcn_ch, T)

        for tcn_block, film_layer in zip(self.tcn_blocks, self.film_layers):
            x = tcn_block(x)
            if vib_feat is not None:
                x = film_layer(x, vib_feat)

        # Predict N masks
        masks_flat = self.mask_head(x)              # (B, N*C_audio, T)
        masks = masks_flat.view(B, self.num_speakers, self.audio_channels, T)
        return masks


# ══════════════════════════════════════════════════════════════════════════════
#  Audio Decoder
# ══════════════════════════════════════════════════════════════════════════════

class AudioDecoder(nn.Module):
    """Synthesis filterbank: latent → waveform (inverse of AudioEncoder)."""

    def __init__(self, in_channels: int = 256, kernel_size: int = 16, stride: int = 8):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels, 1, kernel_size, stride=stride, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T_frames)
        return self.deconv(x).squeeze(1)     # (B, T_audio)


# ══════════════════════════════════════════════════════════════════════════════
#  Full Model: Vibration-Conditioned Separator
# ══════════════════════════════════════════════════════════════════════════════

class VibrationConditionedSeparator(nn.Module):
    """
    Complete multimodal source separation system.

    Forward pass:
      1. Encode mixture waveform → audio latent
      2. Encode each bird's vibration → per-bird vib feature
      3. Aggregate per-bird vib features → single conditioning vector
      4. Apply FiLM-conditioned TCN → N speaker masks
      5. Apply masks to audio latent → N masked latents
      6. Decode each masked latent → N separated waveforms

    This ensures per-bird vibration signals individually inform the separator
    about the timing and identity of each sound source.
    """

    def __init__(
        self,
        num_speakers: int = 2,
        audio_enc_channels: int = 256,
        audio_enc_kernel: int = 16,
        audio_enc_stride: int = 8,
        vib_enc_channels: int = 128,
        vib_enc_layers: int = 4,
        tcn_channels: int = 256,
        tcn_kernel: int = 3,
        tcn_layers: int = 8,
        tcn_stacks: int = 3,
        film_hidden: int = 256,
    ):
        super().__init__()
        self.num_speakers = num_speakers
        self.stride = audio_enc_stride

        # Component 1: Audio Encoder
        self.audio_encoder = AudioEncoder(
            audio_enc_channels, audio_enc_kernel, audio_enc_stride
        )

        # Component 2: Vibration Encoder (shared weights across birds)
        self.vib_encoder = VibrationEncoder(
            vib_enc_channels, vib_enc_layers
        )

        # Aggregation: sum per-bird features → single conditioning vector
        # Then project to a unified conditioning space
        self.vib_aggregator = nn.Sequential(
            nn.Linear(vib_enc_channels, film_hidden),
            nn.ReLU(),
            nn.Linear(film_hidden, vib_enc_channels),
        )

        # Component 3: FiLM-conditioned TCN Separator
        self.separator = FiLMConditionedTCN(
            audio_channels=audio_enc_channels,
            tcn_channels=tcn_channels,
            kernel_size=tcn_kernel,
            num_layers=tcn_layers,
            num_stacks=tcn_stacks,
            num_speakers=num_speakers,
            vib_feature_dim=vib_enc_channels,
            film_hidden=film_hidden,
        )

        # Audio Decoder
        self.audio_decoder = AudioDecoder(audio_enc_channels, audio_enc_kernel, audio_enc_stride)

    def encode_vibrations(self, vibrations: torch.Tensor) -> torch.Tensor:
        """
        Encode per-bird vibration signals and aggregate into a single
        conditioning vector.

        Parameters
        ----------
        vibrations : (B, N, T) — N per-bird vibration signals

        Returns
        -------
        agg_feat : (B, C_vib) — aggregated vibration conditioning feature
        """
        B, N, T = vibrations.shape
        # Encode each bird's vibration separately
        vib_flat = vibrations.view(B * N, T)          # (B*N, T)
        per_bird_feats = self.vib_encoder(vib_flat)   # (B*N, C_vib)
        per_bird_feats = per_bird_feats.view(B, N, -1) # (B, N, C_vib)

        # Sum-pool over birds (permutation-invariant aggregation)
        agg = per_bird_feats.sum(dim=1)               # (B, C_vib)
        agg_feat = self.vib_aggregator(agg)           # (B, C_vib)
        return agg_feat

    def forward(
        self,
        mixture: torch.Tensor,
        vibrations: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        mixture    : (B, 1, T) — mixed microphone recording
        vibrations : (B, N, T) — per-bird vibration signals (N = num_speakers)
                     None for audio-only baseline mode

        Returns
        -------
        separated  : (B, N, T_out) — N separated waveforms
        masks      : (B, N, C, T_frames) — intermediate masks (for analysis)
        """
        # ── Audio encoding ────────────────────────────────────────────────────
        audio_latent = self.audio_encoder(mixture)    # (B, C, T_frames)

        # ── Vibration conditioning ────────────────────────────────────────────
        vib_feat = None
        if vibrations is not None:
            vib_feat = self.encode_vibrations(vibrations)   # (B, C_vib)

        # ── Separation with FiLM conditioning ────────────────────────────────
        masks = self.separator(audio_latent, vib_feat)      # (B, N, C, T_frames)

        # ── Apply masks and decode ────────────────────────────────────────────
        T_audio = mixture.shape[-1]
        separated_waves = []
        for n in range(self.num_speakers):
            masked = masks[:, n, :, :] * audio_latent       # (B, C, T_frames)
            wave = self.audio_decoder(masked)               # (B, T_out)
            # Trim/pad to match input length
            if wave.shape[-1] > T_audio:
                wave = wave[..., :T_audio]
            elif wave.shape[-1] < T_audio:
                wave = F.pad(wave, (0, T_audio - wave.shape[-1]))
            separated_waves.append(wave)

        separated = torch.stack(separated_waves, dim=1)     # (B, N, T)
        return separated, masks


# ══════════════════════════════════════════════════════════════════════════════
#  Audio-Only Baseline (identical architecture, vibration path disabled)
# ══════════════════════════════════════════════════════════════════════════════

class AudioOnlyBaseline(VibrationConditionedSeparator):
    """
    Baseline separator with identical architecture but vibration path removed.
    Used for fair comparison per the thesis experimental design.
    """

    def forward(
        self,
        mixture: torch.Tensor,
        vibrations: Optional[torch.Tensor] = None,  # always ignored
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pass vibrations=None to disable FiLM conditioning
        return super().forward(mixture, vibrations=None)


# ══════════════════════════════════════════════════════════════════════════════
#  Model Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(cfg: dict, model_type: str = "vibration") -> nn.Module:
    """
    Build a model from config dict.

    Parameters
    ----------
    cfg        : full config dict (cfg["model"] used)
    model_type : "vibration" | "audio_only"
    """
    m = cfg["model"]
    kwargs = dict(
        num_speakers=m["num_speakers"],
        audio_enc_channels=m["audio_encoder_channels"],
        audio_enc_kernel=m["audio_encoder_kernel_size"],
        audio_enc_stride=m["audio_encoder_stride"],
        vib_enc_channels=m["vib_encoder_channels"],
        vib_enc_layers=m["vib_encoder_layers"],
        tcn_channels=m["tcn_channels"],
        tcn_kernel=m["tcn_kernel_size"],
        tcn_layers=m["tcn_layers"],
        tcn_stacks=m["tcn_stacks"],
        film_hidden=m["film_hidden_dim"],
    )
    if model_type == "audio_only":
        return AudioOnlyBaseline(**kwargs)
    return VibrationConditionedSeparator(**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Model Architecture Smoke Test ===")
    B, N, T = 2, 2, 22050 * 3
    mixture = torch.randn(B, 1, T)
    vibrations = torch.randn(B, N, T)

    model = VibrationConditionedSeparator()
    baseline = AudioOnlyBaseline()

    with torch.no_grad():
        sep, masks = model(mixture, vibrations)
        sep_b, _ = baseline(mixture, vibrations)

    print(f"  Vibration model   — separated: {sep.shape}, masks: {masks.shape}")
    print(f"  Audio-only model  — separated: {sep_b.shape}")
    print(f"  Vibration params  : {count_parameters(model):,}")
    print(f"  Audio-only params : {count_parameters(baseline):,}")
    print("  ✓ Model forward pass OK")
