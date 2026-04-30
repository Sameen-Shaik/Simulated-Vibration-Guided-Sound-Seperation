"""
data/dataset.py
===============
Dataset preparation, mixture generation, and DataLoader utilities.

Key design:
  - Loads clean bird vocalizations from Xeno-Canto WAV files
  - Generates controlled mixtures with specified overlap ratios
  - Produces PER-BIRD simulated vibration signals (separate for each bird)
  - Returns (mixture, [per_bird_vibration], [clean_sources]) tuples
"""

import os
import sys
import random
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf

# ─── graceful torch import ────────────────────────────────────────────────────
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  Vibration Signal Simulator
# ══════════════════════════════════════════════════════════════════════════════

class VibrationSimulator:
    """
    Simulates per-bird on-body vibration signals inspired by accelerometer
    or contact-microphone measurements.

    Pipeline for each clean vocalization:
      1. Extract temporal energy envelope (captures activity timing)
      2. Apply low-pass filter to retain mechanical low-frequency content
      3. Add calibrated Gaussian noise (sensor noise model)
      4. Optionally inject transient artifacts (beak/body movements)

    Each bird gets its OWN independent vibration signal, so the separator
    can use per-bird identity and timing cues during training.
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        lowpass_cutoff_hz: float = 500.0,
        envelope_smoothing_ms: float = 20.0,
        noise_std: float = 0.02,
        transient_prob: float = 0.15,
        transient_amplitude: float = 0.3,
        seed: Optional[int] = None,
    ):
        self.sr = sample_rate
        self.lowpass_cutoff = lowpass_cutoff_hz
        self.env_window = max(1, int(envelope_smoothing_ms * 1e-3 * sample_rate))
        self.noise_std = noise_std
        self.transient_prob = transient_prob
        self.transient_amplitude = transient_amplitude
        self.rng = np.random.default_rng(seed)

        # Design Butterworth low-pass filter
        nyq = sample_rate / 2.0
        cutoff_norm = min(lowpass_cutoff_hz / nyq, 0.99)
        self.b_lp, self.a_lp = signal.butter(4, cutoff_norm, btype="low")

    def simulate(self, clean_audio: np.ndarray, bird_id: int = 0) -> np.ndarray:
        """
        Generate a simulated vibration signal from a clean bird vocalization.

        Parameters
        ----------
        clean_audio : np.ndarray, shape (T,)
            Clean single-bird audio waveform, normalised to [-1, 1].
        bird_id : int
            Bird index within the mixture — used to introduce slight per-bird
            phase/timing differences (identity cue).

        Returns
        -------
        vibration : np.ndarray, shape (T,)
            Simulated vibration signal, same length as input.
        """
        audio = clean_audio.copy().astype(np.float64)

        # ── Step 1: Temporal energy envelope ──────────────────────────────────
        # Sliding RMS over envelope window
        audio_sq = audio ** 2
        kernel = np.ones(self.env_window) / self.env_window
        envelope = np.sqrt(np.convolve(audio_sq, kernel, mode="same") + 1e-8)

        # ── Step 2: Low-frequency content extraction ──────────────────────────
        # Keep low-frequency mechanical content from original signal
        try:
            lf_content = signal.filtfilt(self.b_lp, self.a_lp, audio)
        except Exception:
            lf_content = audio.copy()

        # Combine: envelope (timing) + low-frequency (content)
        vibration = 0.6 * envelope + 0.4 * np.abs(lf_content)

        # ── Step 3: Per-bird identity perturbation ────────────────────────────
        # Slight phase shift based on bird_id (simulates different body positions)
        shift_samples = int(bird_id * 3)  # 0, 3, 6 … samples
        if shift_samples > 0 and len(vibration) > shift_samples:
            vibration = np.roll(vibration, shift_samples)

        # ── Step 4: Sensor noise ──────────────────────────────────────────────
        noise = self.rng.normal(0, self.noise_std, size=len(vibration))
        vibration = vibration + noise

        # ── Step 5: Transient artifacts ───────────────────────────────────────
        if self.rng.random() < self.transient_prob:
            n_transients = self.rng.integers(1, 4)
            positions = self.rng.integers(0, len(vibration), size=n_transients)
            for pos in positions:
                sign = self.rng.choice([-1, 1])
                width = self.rng.integers(3, 15)
                end = min(len(vibration), pos + width)
                vibration[pos:end] += sign * self.transient_amplitude * self.rng.random(end - pos)

        # ── Step 6: Normalise ─────────────────────────────────────────────────
        max_val = np.max(np.abs(vibration))
        if max_val > 1e-6:
            vibration /= max_val
        vibration = vibration.astype(np.float32)

        return vibration


# ══════════════════════════════════════════════════════════════════════════════
#  Audio Loading Utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(path: str, target_sr: int, duration: float) -> Optional[np.ndarray]:
    """Load a WAV file, resample if needed, extract a random clip."""
    try:
        import librosa
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
    except Exception:
        try:
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except Exception:
            return None

    target_len = int(duration * target_sr)
    if len(audio) < target_len:
        # Pad with zeros if too short
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        # Random crop
        max_start = len(audio) - target_len
        start = random.randint(0, max_start)
        audio = audio[start : start + target_len]

    # Normalise
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        audio = audio / max_val
    return audio.astype(np.float32)


def mix_signals(
    sources: List[np.ndarray],
    overlap_ratio: float,
    target_len: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Mix clean sources into an overlapping mixture.

    Parameters
    ----------
    sources : list of np.ndarray, shape (T,)
        Clean bird waveforms, each of exactly `target_len` samples.
    overlap_ratio : float in [0, 1]
        Fraction of each source that overlaps with another source.
    target_len : int
        Total output length in samples.
    rng : np.random.Generator

    Returns
    -------
    mixture : np.ndarray, shape (target_len,)
    placed_sources : list of np.ndarray, shape (target_len,) each
        Zero-padded clean sources at their actual time positions.
    """
    placed = [np.zeros(target_len, dtype=np.float32) for _ in sources]
    n = len(sources)

    if n == 1:
        placed[0][:len(sources[0])] = sources[0]
    else:
        src_len = len(sources[0])
        # Compute offsets so that `overlap_ratio` fraction of each clip overlaps
        max_offset = int((1.0 - overlap_ratio) * src_len)
        max_offset = max(0, min(max_offset, target_len - src_len))

        offsets = [0]
        for i in range(1, n):
            prev_end = offsets[-1] + src_len
            ideal_start = prev_end - int(overlap_ratio * src_len)
            jitter = rng.integers(-int(0.05 * src_len), int(0.05 * src_len) + 1)
            start = max(0, min(ideal_start + jitter, target_len - src_len))
            offsets.append(start)

        for i, (src, offset) in enumerate(zip(sources, offsets)):
            end = min(target_len, offset + len(src))
            length = end - offset
            placed[i][offset:end] = src[:length]

    mixture = sum(placed)
    # Normalise mixture
    max_val = np.max(np.abs(mixture))
    if max_val > 1e-6:
        mixture = mixture / max_val
    return mixture, placed


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset Scanner
# ══════════════════════════════════════════════════════════════════════════════

class XenoCantoCatalog:
    """Scans wav directory + metadata CSV and groups files by species."""

    def __init__(self, wav_dir: str, metadata_csv: Optional[str] = None):
        self.wav_dir = Path(wav_dir)
        self.species_files: Dict[str, List[str]] = {}

        if metadata_csv and os.path.exists(metadata_csv):
            self._load_from_csv(metadata_csv)
        else:
            self._load_from_dir()

        # Filter species with at least 2 files
        self.species_files = {
            sp: files
            for sp, files in self.species_files.items()
            if len(files) >= 2
        }
        self.species_list = sorted(self.species_files.keys())
        print(f"[Catalog] Loaded {len(self.species_list)} species, "
              f"{sum(len(v) for v in self.species_files.values())} files")

    def _load_from_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        # Normalise column names
        df.columns = [c.lower().strip() for c in df.columns]
        # Determine species column
        if "species" in df.columns and "genus" in df.columns:
            df["species_key"] = df["genus"] + "_" + df["species"]
        elif "name" in df.columns:
            df["species_key"] = df["name"]
        else:
            df["species_key"] = "unknown"

        filename_col = next(
            (c for c in df.columns if "file" in c or "filename" in c), None
        )
        if filename_col is None:
            self._load_from_dir()
            return

        for _, row in df.iterrows():
            fname = str(row[filename_col]).strip()
            sp = str(row["species_key"]).strip()
            path = self.wav_dir / fname
            if not path.exists():
                path = self.wav_dir / (fname + ".wav")
            if path.exists():
                self.species_files.setdefault(sp, []).append(str(path))

        if not self.species_files:
            self._load_from_dir()

    def _load_from_dir(self):
        """Fallback: group WAVs by filename prefix (assumes 'NNNNN-0.wav' pattern)."""
        for wav_file in self.wav_dir.glob("**/*.wav"):
            # Use parent folder name as species if nested, else first token of filename
            if wav_file.parent != self.wav_dir:
                sp = wav_file.parent.name
            else:
                parts = wav_file.stem.split("-")
                sp = parts[0] if len(parts) > 1 else "unknown"
            self.species_files.setdefault(sp, []).append(str(wav_file))

    def sample_files(self, n: int, rng: random.Random) -> List[Tuple[str, str]]:
        """Sample n (species, filepath) pairs from distinct species."""
        species = rng.sample(self.species_list, min(n, len(self.species_list)))
        result = []
        for sp in species:
            fp = rng.choice(self.species_files[sp])
            result.append((sp, fp))
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class BirdSeparationDataset(Dataset):
    """
    Generates on-the-fly bird vocalization mixtures with per-bird vibration signals.

    Each sample:
      mixture       : Tensor (1, T) — overlapping microphone mix
      vibrations    : Tensor (N, T) — per-bird vibration signals (N = num_speakers)
      sources       : Tensor (N, T) — clean reference signals for loss computation
      species_ids   : Tensor (N,)   — integer species labels
      overlap_ratio : float         — actual overlap fraction used
    """

    def __init__(
        self,
        catalog: XenoCantoCatalog,
        num_mixtures: int,
        sample_rate: int = 22050,
        clip_duration: float = 3.0,
        num_speakers: int = 2,
        overlap_ratio: float = 0.6,         # fixed or "random"
        overlap_ratio_range: Tuple = (0.2, 0.95),
        vibration_cfg: Optional[dict] = None,
        seed: int = 42,
        split: str = "train",
    ):
        self.catalog = catalog
        self.num_mixtures = num_mixtures
        self.sr = sample_rate
        self.duration = clip_duration
        self.target_len = int(clip_duration * sample_rate)
        self.num_speakers = num_speakers
        self.overlap_ratio = overlap_ratio
        self.overlap_ratio_range = overlap_ratio_range
        self.split = split

        # Seed per split for reproducibility
        split_seeds = {"train": seed, "val": seed + 1, "test": seed + 2}
        self.rng_np = np.random.default_rng(split_seeds[split])
        self.rng_py = random.Random(split_seeds[split])

        # Vibration simulator
        vib_cfg = vibration_cfg or {}
        self.vibsim = VibrationSimulator(
            sample_rate=sample_rate,
            lowpass_cutoff_hz=vib_cfg.get("lowpass_cutoff_hz", 500.0),
            envelope_smoothing_ms=vib_cfg.get("envelope_smoothing_ms", 20.0),
            noise_std=vib_cfg.get("noise_std", 0.02),
            transient_prob=vib_cfg.get("transient_prob", 0.15),
            transient_amplitude=vib_cfg.get("transient_amplitude", 0.3),
            seed=split_seeds[split],
        )

        # Species → integer mapping
        self.species2id = {sp: i for i, sp in enumerate(catalog.species_list)}

    def __len__(self):
        return self.num_mixtures

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Determine overlap ratio for this sample
        if self.overlap_ratio == "random":
            lo, hi = self.overlap_ratio_range
            ov = float(self.rng_np.uniform(lo, hi))
        else:
            ov = float(self.overlap_ratio)

        # Sample species + files
        pairs = self.catalog.sample_files(self.num_speakers, self.rng_py)

        # Load clean audio clips
        clean_clips: List[np.ndarray] = []
        species_ids: List[int] = []
        for sp, fp in pairs:
            audio = load_audio(fp, self.sr, self.duration)
            if audio is None:
                audio = np.zeros(self.target_len, dtype=np.float32)
            clean_clips.append(audio)
            species_ids.append(self.species2id.get(sp, 0))

        # Generate per-bird vibration BEFORE mixing (clean source → clean vib)
        # This is the key: each bird gets its own vibration from ITS OWN clean audio
        per_bird_vibrations: List[np.ndarray] = []
        for bird_idx, clean in enumerate(clean_clips):
            vib = self.vibsim.simulate(clean, bird_id=bird_idx)
            per_bird_vibrations.append(vib)

        # Mix audio signals with controlled overlap
        mixture, placed_sources = mix_signals(
            clean_clips, ov, self.target_len, self.rng_np
        )

        # ── Stack into tensors ────────────────────────────────────────────────
        mixture_t = torch.from_numpy(mixture).unsqueeze(0)          # (1, T)
        sources_t = torch.stack(
            [torch.from_numpy(s) for s in placed_sources], dim=0    # (N, T)
        )
        vibrations_t = torch.stack(
            [torch.from_numpy(v) for v in per_bird_vibrations], dim=0  # (N, T)
        )
        species_t = torch.tensor(species_ids, dtype=torch.long)     # (N,)

        return {
            "mixture": mixture_t,               # (1, T)
            "vibrations": vibrations_t,         # (N, T) — PER-BIRD!
            "sources": sources_t,               # (N, T)
            "species_ids": species_t,           # (N,)
            "overlap_ratio": torch.tensor(ov),  # scalar
        }


# ══════════════════════════════════════════════════════════════════════════════
#  DataLoader Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    wav_dir: str,
    metadata_csv: Optional[str],
    cfg: dict,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from config dict."""

    catalog = XenoCantoCatalog(wav_dir, metadata_csv)

    ds_cfg = cfg["dataset"]
    vib_cfg = cfg["vibration"]
    model_cfg = cfg["model"]
    total = ds_cfg["num_mixtures"]
    seed = ds_cfg.get("seed", 42)

    n_train = int(total * ds_cfg["train_split"])
    n_val = int(total * ds_cfg["val_split"])
    n_test = total - n_train - n_val

    common = dict(
        catalog=catalog,
        sample_rate=ds_cfg["sample_rate"],
        clip_duration=ds_cfg["clip_duration"],
        num_speakers=model_cfg["num_speakers"],
        vibration_cfg=vib_cfg,
        seed=seed,
    )

    train_ds = BirdSeparationDataset(
        **common,
        num_mixtures=n_train,
        overlap_ratio="random",
        overlap_ratio_range=(0.2, 0.95),
        split="train",
    )
    val_ds = BirdSeparationDataset(
        **common,
        num_mixtures=n_val,
        overlap_ratio=0.6,
        split="val",
    )
    test_ds = BirdSeparationDataset(
        **common,
        num_mixtures=n_test,
        overlap_ratio=0.6,
        split="test",
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke test with synthetic data
    print("=== Smoke-testing VibrationSimulator ===")
    sim = VibrationSimulator(sample_rate=22050, noise_std=0.02)
    dummy = np.random.randn(22050 * 3).astype(np.float32)
    vib0 = sim.simulate(dummy, bird_id=0)
    vib1 = sim.simulate(dummy, bird_id=1)
    print(f"  Bird-0 vib: shape={vib0.shape}, range=[{vib0.min():.3f}, {vib0.max():.3f}]")
    print(f"  Bird-1 vib: shape={vib1.shape}, range=[{vib1.min():.3f}, {vib1.max():.3f}]")
    assert not np.allclose(vib0, vib1), "Per-bird vibrations must differ!"
    print("  ✓ Per-bird vibrations are distinct")
