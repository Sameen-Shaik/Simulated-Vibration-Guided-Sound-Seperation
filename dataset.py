"""
dataset.py — BirdMixDataset with fixed mixture pool and recording-level split.

REAL DATASET STRUCTURE (bird_songs_metadata.csv)
  477 original recordings  — column "id"   e.g. 557838
  5422 augmented clips     — column "filename"  e.g. 557838-0.wav
  5 species (column "name"):
      Song Sparrow         1256 clips / 137 ids
      Northern Mockingbird 1182 clips /  88 ids
      Northern Cardinal    1074 clips /  92 ids
      American Robin       1017 clips /  81 ids
      Bewick's Wren         893 clips /  79 ids

SPLIT STRATEGY (no leakage)
  Split at RECORDING level (477 ids), stratified by "name".
      Train : 70% ids → ~3795 clips
      Val   : 15% ids →  ~814 clips
      Test  : 15% ids →  ~813 clips
  All clips of one recording land in exactly one split.

MIXTURE POOL (fair comparison)
  Mixture pairs (idx1, idx2) are generated ONCE and frozen.
  Both Model A and Model B iterate this identical pool every epoch.
  num_mixtures (CLI arg) controls the pool size.
  Recommended: ~3795 (= number of train clips) or 2-4x for more signal.

VARIABLE MAP (spec document)
  Y1, Y2   — clean source waveforms           (separation targets)
  Ymix     — Y1 + Y2 normalised               (model input, both models)
  V1       — simulate_vibration(Y1)           (Model B aux target ONLY)
  V2       — simulate_vibration(Y2)           (Model B aux target ONLY)
  Vmix     — simulate_vibration(Ymix)         (Model B conditioning input)

  V1/V2 are NEVER fed to any model.
  Vmix is NOT V1+V2 — derived independently from Ymix.
"""

import random
from pathlib import Path

import pandas as pd
import torch
# import torchaudio  # Commented out: RuntimeError with torchcodec - FFmpeg not properly installed on Windows
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from vibration import simulate_vibration


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

# def _load_audio_torchaudio(path: str, target_sr: int, target_len: int) -> torch.Tensor:
#     """Load mono audio, resample if needed, pad/trim to target_len samples."""
#     # DISABLED: torchaudio.load causes RuntimeError with torchcodec - FFmpeg not properly installed on Windows
#     # Error: "Could not load libtorchcodec. Likely causes: FFmpeg is not properly installed in your environment."
#     wav, sr = torchaudio.load(path)
#     if wav.shape[0] > 1:
#         wav = wav.mean(dim=0, keepdim=True)
#     if sr != target_sr:
#         wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
#     if wav.shape[1] > target_len:
#         wav = wav[:, :target_len]
#     elif wav.shape[1] < target_len:
#         pad = target_len - wav.shape[1]
#         wav = torch.nn.functional.pad(wav, (0, pad))
#     return wav.squeeze(0)  # [T]


def _load_audio(path: str, target_sr: int, target_len: int) -> torch.Tensor:
    """Load mono audio, resample if needed, pad/trim to target_len samples."""
    wav, sr = sf.read(path, dtype='float32')
    wav = torch.from_numpy(wav)
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    if sr != target_sr:
        # Simple resampling using linear interpolation
        old_len = wav.shape[0]
        new_len = int(old_len * target_sr / sr)
        wav = torch.nn.functional.interpolate(
            wav.unsqueeze(0).unsqueeze(0), size=new_len, mode='linear', align_corners=False
        ).squeeze()
    if wav.shape[0] > target_len:
        wav = wav[:target_len]
    elif wav.shape[0] < target_len:
        pad = target_len - wav.shape[0]
        wav = torch.nn.functional.pad(wav, (0, pad))
    return wav


def _rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x.pow(2).mean().sqrt().clamp(min=eps)


# ─────────────────────────────────────────────────────────────────────────────
# Recording-level stratified split
# ─────────────────────────────────────────────────────────────────────────────

def recording_level_split(
    meta: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Split at the ORIGINAL RECORDING level (by 'id'), stratified by 'name'.
    All clips from one recording always go into exactly one split.
    Returns (train_meta, val_meta, test_meta) DataFrames.
    """
    required = {"id", "name", "filename"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # One row per recording
    id_df = (meta[["id", "name"]]
             .drop_duplicates(subset="id")
             .reset_index(drop=True))

    # Merge rare species so sklearn stratify does not crash
    counts = id_df["name"].value_counts()
    rare = counts[counts < 2].index
    if len(rare):
        id_df = id_df.copy()
        id_df.loc[id_df["name"].isin(rare), "name"] = "__rare__"

    trainval_ids, test_ids = train_test_split(
        id_df["id"].tolist(),
        test_size=test_fraction,
        stratify=id_df["name"].tolist(),
        random_state=seed,
    )
    tv_df = id_df[id_df["id"].isin(trainval_ids)].reset_index(drop=True)
    rel_val = val_fraction / (1.0 - test_fraction)
    train_ids, val_ids = train_test_split(
        tv_df["id"].tolist(),
        test_size=rel_val,
        stratify=tv_df["name"].tolist(),
        random_state=seed,
    )

    train_meta = meta[meta["id"].isin(train_ids)].reset_index(drop=True)
    val_meta   = meta[meta["id"].isin(val_ids)  ].reset_index(drop=True)
    test_meta  = meta[meta["id"].isin(test_ids) ].reset_index(drop=True)
    return train_meta, val_meta, test_meta


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_splits(train_meta, val_meta, test_meta):
    """Assert zero id overlap across splits and print distribution table."""
    train_ids = set(train_meta["id"])
    val_ids   = set(val_meta["id"])
    test_ids  = set(test_meta["id"])

    assert len(train_ids & val_ids)  == 0, "LEAKAGE: train ∩ val"
    assert len(train_ids & test_ids) == 0, "LEAKAGE: train ∩ test"
    assert len(val_ids   & test_ids) == 0, "LEAKAGE: val  ∩ test"

    total = len(train_meta) + len(val_meta) + len(test_meta)
    print("\n" + "=" * 62)
    print("  RECORDING-LEVEL SPLIT — VERIFICATION")
    print("=" * 62)
    print(f"  {'Split':<8}  {'IDs':>5}  {'Clips':>6}  {'Clips%':>7}")
    print(f"  {'-'*38}")
    for tag, ids, df in [("Train", train_ids, train_meta),
                          ("Val",   val_ids,   val_meta),
                          ("Test",  test_ids,  test_meta)]:
        pct = 100 * len(df) / total if total else 0
        print(f"  {tag:<8}  {len(ids):>5}  {len(df):>6}  {pct:>6.1f}%")
    print(f"  {'TOTAL':<8}  "
          f"{len(train_ids)+len(val_ids)+len(test_ids):>5}  {total:>6}")
    print(f"  {'-'*38}")
    print(f"  Train∩Val={len(train_ids & val_ids)}  "
          f"Train∩Test={len(train_ids & test_ids)}  "
          f"Val∩Test={len(val_ids & test_ids)}  OK no leakage")
    print("\n  SPECIES DISTRIBUTION (clips per split):")
    species = sorted(set(train_meta["name"]) | set(val_meta["name"]) | set(test_meta["name"]))
    print(f"  {'Species':<24}  {'Train':>6}  {'Val':>6}  {'Test':>6}")
    print(f"  {'-'*46}")
    for sp in species:
        tr = (train_meta["name"] == sp).sum()
        va = (val_meta["name"]   == sp).sum()
        te = (test_meta["name"]  == sp).sum()
        print(f"  {sp:<24}  {tr:>6}  {va:>6}  {te:>6}")
    print("=" * 62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mixture pool builder
# ─────────────────────────────────────────────────────────────────────────────

def build_mixture_pool(n_clips: int, num_mixtures: int, seed: int) -> list:
    """
    Generate a FIXED list of (idx1, idx2) unique clip-index pairs.

    This pool is built once before training starts and shared by both
    Model A and Model B, guaranteeing they train on identical mixtures.
    The pool is frozen; DataLoader shuffle=True handles epoch ordering.

    Args:
        n_clips      : clips available in this split
        num_mixtures : desired pool size
        seed         : reproducibility seed
    """
    if n_clips < 2:
        raise ValueError(f"Need >= 2 clips, got {n_clips}.")
    max_unique = n_clips * (n_clips - 1)
    if num_mixtures > max_unique:
        raise ValueError(
            f"Requested {num_mixtures} mixtures but max unique pairs "
            f"from {n_clips} clips is {max_unique}. Reduce --num_mixtures."
        )

    rng  = random.Random(seed)
    pool = []
    seen = set()
    max_attempts = num_mixtures * 20
    attempts = 0

    while len(pool) < num_mixtures and attempts < max_attempts:
        i = rng.randint(0, n_clips - 1)
        j = rng.randint(0, n_clips - 1)
        if i != j and (i, j) not in seen:
            pool.append((i, j))
            seen.add((i, j))
        attempts += 1

    if len(pool) < num_mixtures:
        raise RuntimeError(
            f"Generated only {len(pool)}/{num_mixtures} pairs from {n_clips} clips. "
            f"Reduce --num_mixtures."
        )
    return pool


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BirdMixDataset(Dataset):
    """
    Fixed-mixture dataset. The pair list and per-mixture augmentation
    values are frozen at construction so both models train identically.

    Returns per sample:
        Ymix  [T]  mixture waveform             -> model input (both models)
        Vmix  [T]  simulate_vibration(Ymix)     -> Model B conditioning only
        Y1    [T]  clean source 1               -> separation target
        Y2    [T]  clean source 2               -> separation target
        V1    [T]  simulate_vibration(Y1)       -> Model B aux loss ONLY
        V2    [T]  simulate_vibration(Y2)       -> Model B aux loss ONLY
        snr_db scalar                            -> logging
    """

    def __init__(
        self,
        meta,
        data_root: str,
        num_mixtures: int,
        sample_rate: int = 22050,
        clip_duration_s: float = 3.0,
        snr_range_db: tuple = (-5.0, 5.0),
        gain_range: tuple = (0.7, 1.0),
        add_bg_noise: bool = True,
        noise_std: float = 0.005,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root  = Path(data_root)
        self.sr         = sample_rate
        self.target_len = int(clip_duration_s * sample_rate)
        self.add_bg_noise = add_bg_noise
        self.noise_std    = noise_std

        self.meta    = meta.reset_index(drop=True)
        self.files   = [str(self.data_root / fn) for fn in self.meta["filename"]]
        self.n_clips = len(self.files)

        # Fixed mixture pool — frozen before any training
        self.mixture_pool = build_mixture_pool(self.n_clips, num_mixtures, seed)

        # Pre-drawn, fixed augmentation values per mixture
        rng = random.Random(seed + 1)
        self._snr = [rng.uniform(*snr_range_db) for _ in range(num_mixtures)]
        self._g1  = [rng.uniform(*gain_range)   for _ in range(num_mixtures)]
        self._g2  = [rng.uniform(*gain_range)   for _ in range(num_mixtures)]

    def __len__(self):
        return len(self.mixture_pool)

    def __getitem__(self, mix_idx: int) -> dict:
        idx1, idx2 = self.mixture_pool[mix_idx]
        snr_db     = self._snr[mix_idx]
        g1, g2     = self._g1[mix_idx], self._g2[mix_idx]

        # Load Y1, Y2
        Y1 = _load_audio(self.files[idx1], self.sr, self.target_len)
        Y2 = _load_audio(self.files[idx2], self.sr, self.target_len)

        # Gain augmentation
        Y1 = Y1 * g1
        Y2 = Y2 * g2

        # SNR scaling: scale Y2 so that SNR(Y1, Y2) = snr_db
        scale = (_rms(Y1) / _rms(Y2)) * (10.0 ** (-snr_db / 20.0))
        Y2    = Y2 * scale

        # Ymix = Y1 + Y2, normalised
        Ymix = Y1 + Y2
        peak = Ymix.abs().max().clamp(min=1e-8)
        Ymix = Ymix / peak
        Y1   = Y1   / peak      # keep sources on same scale as mixture
        Y2   = Y2   / peak

        if self.add_bg_noise:
            Ymix = Ymix + torch.randn_like(Ymix) * self.noise_std

        # Vibration signals
        #   V1 = simulate_vibration(Y1)    — aux supervision target, Model B
        #   V2 = simulate_vibration(Y2)    — aux supervision target, Model B
        #   Vmix = simulate_vibration(Ymix)— conditioning input,     Model B
        #
        # Vmix is derived from Ymix, NOT from V1+V2.
        # V1 and V2 never enter any model's forward().
        V1   = simulate_vibration(Y1,   sr=self.sr)
        V2   = simulate_vibration(Y2,   sr=self.sr)
        Vmix = simulate_vibration(Ymix, sr=self.sr)

        return {
            "Ymix":   Ymix,
            "Vmix":   Vmix,
            "Y1":     Y1,
            "Y2":     Y2,
            "V1":     V1,
            "V2":     V2,
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public factory
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_root: str,
    metadata_csv: str,
    num_mixtures_train: int,
    num_mixtures_val:   int   = 500,
    num_mixtures_test:  int   = 500,
    batch_size:         int   = 8,
    num_workers:        int   = 4,
    val_fraction:       float = 0.15,
    test_fraction:      float = 0.15,
    sample_rate:        int   = 22050,
    clip_duration_s:    float = 3.0,
    seed:               int   = 42,
):
    """
    Build train/val/test DataLoaders backed by fixed mixture pools.

    Both Model A and Model B MUST use the loaders returned here
    to guarantee fair comparison on identical mixture sequences.

    Args:
        num_mixtures_train : pool size for training.
            Recommended starting point: len(train_clips) ~ 3795.
            Use 2-4x for more training signal per epoch.
        num_mixtures_val   : val pool size   (default 500).
        num_mixtures_test  : test pool size  (default 500).
    """
    meta = pd.read_csv(metadata_csv)

    train_meta, val_meta, test_meta = recording_level_split(
        meta, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed,
    )
    verify_splits(train_meta, val_meta, test_meta)

    common = dict(data_root=data_root, sample_rate=sample_rate,
                  clip_duration_s=clip_duration_s)

    train_ds = BirdMixDataset(
        train_meta, num_mixtures=num_mixtures_train,
        snr_range_db=(-5.0, 5.0), add_bg_noise=True,  seed=seed,     **common)
    val_ds   = BirdMixDataset(
        val_meta,   num_mixtures=num_mixtures_val,
        snr_range_db=(-2.0, 2.0), add_bg_noise=False, seed=seed + 1, **common)
    test_ds  = BirdMixDataset(
        test_meta,  num_mixtures=num_mixtures_test,
        snr_range_db=(-0.5, 0.5), add_bg_noise=False, seed=seed + 2, **common)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=False)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kw)

    print(f"[dataset] Train  {len(train_ds):>5} mixtures  "
          f"({len(train_meta):>4} clips / {train_meta['id'].nunique()} recordings)")
    print(f"[dataset] Val    {len(val_ds):>5} mixtures  "
          f"({len(val_meta):>4} clips / {val_meta['id'].nunique()} recordings)")
    print(f"[dataset] Test   {len(test_ds):>5} mixtures  "
          f"({len(test_meta):>4} clips / {test_meta['id'].nunique()} recordings)")

    return train_loader, val_loader, test_loader
