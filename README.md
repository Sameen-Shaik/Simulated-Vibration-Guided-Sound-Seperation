# Audio Source Separation Guided by Simulated On-Body Vibration Signals
### for Overlapping Bird Vocalizations
**DV1478 VT26 — Bachelor's Thesis in Computer Science, BTH**
*Mahammed Sameen Shaik — Supervisor: Ilir Jusufi*

---

## Overview

This repository implements the complete system described in the thesis proposal:
a multimodal neural audio source separator that uses **per-bird simulated vibration
signals** as weak supervisory guidance to disentangle overlapping bird vocalizations.

```
bird_separation/
├── data/
│   └── dataset.py          ← Dataset, VibrationSimulator, DataLoader factory
├── models/
│   └── separator.py        ← AudioEncoder, VibrationEncoder, FiLM TCN, Decoder
├── utils/
│   ├── losses.py           ← SI-SDR, multi-scale spectral, PIT, combined loss
│   ├── metrics.py          ← SDR/SIR/SAR/SI-SDR, spectral, statistical tests
│   └── trainer.py          ← Training loop, LR scheduling, checkpointing
├── experiments/
│   ├── evaluate.py         ← Full evaluation + overlap-stratified analysis
│   └── ablation.py         ← Noise sensitivity & per-bird vs shared vibration
├── configs/
│   └── config.yaml         ← All hyperparameters
├── train.py                ← Main training entry point
└── infer.py                ← Separate a new mixed WAV file
```

---

## Dataset Setup

Your dataset layout (Xeno-Canto, 9,107 WAV files, 100 species):

```
your_dataset/
├── wavfiles/
│   ├── 544036-0.wav
│   ├── 544037-0.wav
│   └── ...
└── metadata.csv        ← columns: genus, species, filename (+ others)
```

The system reads `metadata.csv` to group recordings by species (using
`genus + "_" + species` as the key). Each species needs ≥ 2 files.
If no CSV is provided, files are grouped by filename prefix automatically.

---

## Installation

```bash
pip install torch torchaudio librosa soundfile scikit-learn mir_eval scipy pandas tqdm
```

---

## Training

### Train both models (vibration-conditioned + audio-only baseline):
```bash
python train.py \
  --wav_dir /path/to/wavfiles \
  --metadata_csv /path/to/metadata.csv \
  --model both \
  --epochs 100 \
  --batch_size 8 \
  --num_mixtures 10000
```

### Train only the vibration model:
```bash
python train.py --wav_dir /path/to/wavfiles --model vibration
```

### Train only the baseline:
```bash
python train.py --wav_dir /path/to/wavfiles --model audio_only
```

Output:
- Checkpoints: `experiments/checkpoints/{model_name}/best.pt`
- Training logs: `experiments/logs/{model_name}_training.csv`

---

## Architecture

### 1. Audio Encoder
1-D convolutional analysis filterbank (Conv-TasNet style):
- `Conv1d(1, 256, kernel=16, stride=8)` → latent frames

### 2. Vibration Encoder (per bird, shared weights)
Processes each bird's vibration signal **separately**:
- 4-layer 1-D conv stack with BatchNorm + ReLU
- Global average pooling → 128-D feature vector
- Sum-pooling across birds → single conditioning vector

### 3. FiLM-Conditioned TCN Separator
- 3 stacks × 8 dilated depthwise-separable Conv blocks
- After **every** block: FiLM layer modulates audio features with vibration:
  ```
  output = (1 + γ(vib)) × audio_feat + β(vib)
  ```
- Output: N speaker masks → N masked latents

### 4. Audio Decoder
ConvTranspose1d synthesis filterbank → N separated waveforms

**Parameter count:** ~7.6M per model (identical for fair comparison)

---

## Per-Bird Vibration Simulation

Key design principle from the thesis: **each bird gets its own separate
vibration signal**, derived from its own clean audio during training.

```python
VibrationSimulator.simulate(clean_audio, bird_id=n)
```

Pipeline per bird:
1. **Temporal energy envelope** — captures activity timing (RMS sliding window)
2. **Low-pass filter** (< 500 Hz) — mechanical low-frequency content
3. **Per-bird identity shift** — slight temporal offset per `bird_id`
4. **Gaussian sensor noise** — noise_std=0.02
5. **Transient artifacts** — random spikes (prob=0.15)
6. **Normalise** to [-1, 1]

At **inference time** (no clean sources available), vibrations are derived
from the mixture itself using different `bird_id` offsets.

---

## Training Loop

- **Loss**: 0.5 × SI-SDR + 0.5 × Multi-Scale Spectral Loss
- **PIT**: Permutation-Invariant Training finds optimal speaker assignment
- **Optimiser**: Adam (lr=1e-3, weight_decay=1e-5)
- **LR Schedule**: Linear warmup (5 epochs) + Cosine annealing
- **Early Stopping**: patience=15 epochs
- **Mixed Precision**: via torch.cuda.amp (auto-enabled on CUDA)

---

## Evaluation

```bash
python experiments/evaluate.py \
  --wav_dir /path/to/wavfiles \
  --metadata_csv /path/to/metadata.csv \
  --vib_checkpoint experiments/checkpoints/vibration/best.pt \
  --baseline_checkpoint experiments/checkpoints/audio_only/best.pt \
  --output_dir experiments/results \
  --num_samples 500
```

**Metrics computed** (per RQ1, RQ2, RQ3):
| Metric | Answers |
|--------|---------|
| SDR (Signal-to-Distortion Ratio) | RQ1 |
| SI-SDR (Scale-Invariant SDR) | RQ1, RQ2 |
| SIR (Signal-to-Interference Ratio) | RQ1 |
| SAR (Signal-to-Artifacts Ratio) | RQ1 |
| Spectral Convergence | RQ3 |
| Log Spectral Distance | RQ3 |

**Statistical tests** (H0 vs H1, α=0.05):
- Paired t-test
- Wilcoxon signed-rank test

**Overlap-stratified analysis** (answers RQ1 specifically):
- Low overlap: 0–0.4
- Medium: 0.4–0.7
- High: 0.7–1.0 ← focus area per thesis

Outputs:
- `experiments/results/evaluation_report.json`
- `experiments/results/vibration_results.csv`
- `experiments/results/audio_only_results.csv`

---

## Ablation Studies

```bash
python experiments/ablation.py \
  --wav_dir /path/to/wavfiles \
  --vib_checkpoint experiments/checkpoints/vibration/best.pt
```

Studies run:
1. **Noise sensitivity** (RQ2): performance vs vibration noise_std ∈ {0, 0.01, 0.02, 0.05, 0.10, 0.20}
2. **Per-bird vs shared vibration**: quantifies the benefit of separate per-bird signals

---

## Inference on New Files

```bash
python infer.py \
  --input mixed_birds.wav \
  --checkpoint experiments/checkpoints/vibration/best.pt \
  --output_dir separated/ \
  --num_speakers 2
```

Saves `separated/mixed_birds_separated_bird1.wav` and `_bird2.wav`.

---

## Research Questions Mapping

| RQ | Component |
|----|-----------|
| RQ1: Separation performance vs overlap | `evaluate.py` → stratified_analysis |
| RQ2: Vibration signal effectiveness | `ablation.py` → noise sensitivity, per-bird study |
| RQ3: Preservation of acoustic features | `evaluate.py` → spectral_convergence, log_spectral_distance |

---

## Hypotheses

- **H0**: No performance difference between audio-only and vibration-conditioned
- **H1**: Vibration-conditioned improves SDR and SI-SDR, especially in overlap-heavy segments

Results are reported with p-values from both paired t-test and Wilcoxon signed-rank test
at significance level α = 0.05.

---

## References

1. Défossez et al. (2019). Music source separation in the waveform domain. arXiv:1911.13254
2. Hershey et al. (2016). Deep clustering. ICASSP
3. Kahl et al. (2022). BirdCLEF 2022. CLEF Working Notes
4. Kahl et al. (2021). BirdNET. Ecological Informatics
5. Luo & Mesgarani (2019). Conv-TasNet. IEEE/ACM TASLP
6. Perez et al. (2018). FiLM. AAAI
7. Stowell et al. (2019). Bird Audio Detection. Methods in Ecology and Evolution
8. Wang & Chen (2018). Supervised speech separation. IEEE/ACM TASLP
