# Audio Source Separation Guided by Simulated Vibration Signals

Bachelor's thesis implementation, **DV1478 VT26 — Bachelor's Thesis in Computer Science, BTH**.

The project studies whether a simulated body-vibration signal can help a neural
separator disentangle two overlapping bird vocalisations. The vibration is a
computational proxy; this repository does not contain real accelerometer or
contact-microphone measurements.

The authoritative academic artifact is the [final submitted thesis](docs/thesis/Thesis-Final-Submission.pdf).

## Method

The published executable pipeline is a two-source waveform separator with two
operating modes:

- `baseline`: audio-only Conv-TasNet-style separation;
- `multimodal`: the same separation backbone conditioned on a simulated mixture
  vibration through a vibration encoder and FiLM modulation.

Each dataset item contains the mixture waveform `Ymix`, clean targets `Y1` and
`Y2`, mixture vibration `Vmix`, and per-source vibration targets `V1` and `V2`.
Only `Ymix` reaches both models. `Vmix` is the multimodal conditioning input;
`V1` and `V2` are used only by the auxiliary training loss and are never passed
to the model at evaluation time.

The implementation uses:

- 22,050 Hz mono audio and 3-second clips;
- a recording-level, stratified split by the metadata `id` column;
- deterministic fixed mixture pools with seeded SNR and gain values;
- a 256-filter, length-16 convolutional encoder, 64-channel bottleneck, and
  six depthwise-separable convolution blocks repeated twice;
- per-sample global layer normalisation (`gLN`) by default;
- a two-stage vibration encoder, global pooling, and bottleneck FiLM scale/shift
  for the multimodal model;
- softmax source masks followed by a transposed-convolution decoder;
- PIT SI-SDR as the audio objective, with an optional auxiliary vibration MSE
  weighted by `lambda_vib` for the multimodal model.

### Simulated vibration

`vibration.py` turns a waveform into a same-length simulated sensor signal:

1. short-time RMS envelope (`frame_size=512`, `hop_size=128`);
2. linear interpolation back to waveform length;
3. moving-average low-pass smoothing at approximately 300 Hz;
4. Gaussian sensor noise and sparse transient artefacts;
5. per-sample peak normalisation to `[-1, 1]`.

At inference, the model receives vibration derived from the observed mixture
waveform. It does not receive clean source audio or oracle per-source vibration.

## Dataset setup

The repository includes the small `data/cleaned_metadata.csv` manifest used by
the final workspace. It contains the loader-required `id`, `name`, and
`filename` fields for the 5,422 clip entries associated with 477 recording IDs
and five species. The audio files themselves are not redistributed here.

To run the code, obtain the source audio separately and place the WAV files in a
local directory whose filenames match the manifest. The code samples ordered
clip pairs from each split; the current implementation does not enforce a
different-species or different-recording constraint when building a pair.

This is intentional publication documentation of the archived code. The thesis
PDF describes a different dataset summary and mixture description in places;
see [Implementation and thesis notes](#implementation-and-thesis-notes).

## Installation

Use a fresh virtual environment and install the published dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project uses PyTorch, torchaudio, NumPy, pandas, SciPy, scikit-learn,
SoundFile, tqdm, and matplotlib. CPU execution is supported for lightweight
checks; training and full evaluation are considerably more practical with a
GPU.

## Training

Train the audio-only baseline:

```bash
python train.py \
  --model baseline \
  --data_root /path/to/wavfiles \
  --metadata data/cleaned_metadata.csv \
  --num_mixtures 3800 \
  --num_mixtures_val 500 \
  --num_mixtures_test 500 \
  --epochs 100 \
  --ckpt_dir checkpoints
```

Train the vibration-conditioned model with the same data-pool settings:

```bash
python train.py \
  --model multimodal \
  --data_root /path/to/wavfiles \
  --metadata data/cleaned_metadata.csv \
  --num_mixtures 3800 \
  --num_mixtures_val 500 \
  --num_mixtures_test 500 \
  --lambda_vib 0.1 \
  --epochs 100 \
  --ckpt_dir checkpoints
```

The CLI trains one model per invocation. Useful options include `--batch_size`,
`--accum_steps`, `--norm_type`, `--num_groups`, `--sample_rate`, `--clip_dur`,
`--num_workers`, and `--seed`. The default training pool is 3,800 mixtures;
validation and test pools default to 500 each. Training writes run-specific
logs below `logs/` and checkpoints below the requested checkpoint directory.
Those generated directories and weight files are ignored by Git.

## Evaluation

Evaluate a baseline checkpoint:

```bash
python eval.py \
  --model baseline \
  --ckpt /path/to/baseline/best.pt \
  --data_root /path/to/wavfiles \
  --metadata data/cleaned_metadata.csv \
  --num_mixtures 500 \
  --output_json results/primary/baseline_eval.json
```

Evaluate the multimodal checkpoint with the same command, changing
`--model` and `--ckpt`:

```bash
python eval.py \
  --model multimodal \
  --ckpt /path/to/multimodal/best.pt \
  --data_root /path/to/wavfiles \
  --metadata data/cleaned_metadata.csv \
  --num_mixtures 500 \
  --output_json results/primary/multimodal_eval.json
```

Evaluation reports mean and standard deviation for SI-SDR, SI-SDR improvement,
SDR, spectral convergence, and STFT L1. It evaluates a standard test pool with
SNR uniformly sampled from `[-1, 1]` dB and a near-zero-SNR high-overlap pool
with SNR sampled from `[-0.5, 0.5]` dB. Use `--save_audio --audio_dir ...` only
when local WAV examples are wanted; generated audio is ignored by Git.

Run analysis helpers from the repository root:

```bash
python analysis/aggregate_runs.py --base_dir logs --metric val_si_sdr
python analysis/statistical_test.py \
  --ckpt_a /path/to/baseline/best.pt \
  --ckpt_b /path/to/multimodal/best.pt \
  --data_root /path/to/wavfiles \
  --metadata data/cleaned_metadata.csv \
  --num_mixtures 500 \
  --output_json results/statistics/wilcoxon_results.json
```

For the pooled analysis, provide two directories containing the cached
`sisdri_model_a.npy` and `sisdri_model_b.npy` arrays:

```bash
python analysis/pooled_wilcoxon.py \
  --run_1_dir /path/to/stat_cache/run_1 \
  --run_2_dir /path/to/stat_cache/run_2 \
  --output_json results/statistics/pooled_wilcoxon_results.json
```

These scripts do not launch training automatically and do not download data.

## Thesis results

The following are the headline values reported in the submitted thesis. They
are two-run averages; the small per-run inputs are preserved in
`results/primary/` and `results/ablation/`.

| Test condition | Audio-only SI-SDRi | Multimodal SI-SDRi | Absolute improvement |
| --- | ---: | ---: | ---: |
| Standard (`[-1, 1]` dB SNR) | 3.652 dB | 4.618 dB | +0.966 dB |
| High-overlap (`[-0.5, 0.5]` dB SNR) | 3.541 dB | 4.618 dB | +1.077 dB |

The pooled statistical artifact contains 2,000 source estimates, median paired
improvement `0.491 dB`, 95% bootstrap CI `[0.426, 0.568] dB`, and
`p = 2.2420775429197073e-44`. The ablation summaries retain the raw per-seed
JSON values for `lambda_vib` in `{0.0, 0.05, 0.1, 0.20, 0.5}`.

## Reproducibility and limitations

The fixed mixture pools, recording-level split, stored seeds, and preserved
result JSON files support inspection and rerunning with the same local inputs.
The repository is not a fully self-contained reproduction because the source
audio dataset and trained checkpoints are not redistributed. Dataset licences,
download terms, and the exact local audio files must be checked before running
new experiments.

Other limitations include simulated rather than measured vibration, a fixed
two-source separation problem, stochastic sensor-noise/transient simulation,
and the pairing behavior described in the dataset section. Checkpoints should
remain outside normal Git history; if sharing weights later is useful, a review
can consider a GitHub Release rather than committing model dumps.

## Repository layout

```text
.
├── dataset.py                 # manifest, split, mixture pools, data loading
├── vibration.py               # simulated vibration signal
├── model.py                   # baseline and multimodal separators
├── loss.py                    # PIT SI-SDR and auxiliary vibration loss
├── train.py                   # training CLI
├── eval.py                    # evaluation CLI
├── analysis/
│   ├── aggregate_runs.py
│   ├── statistical_test.py
│   └── pooled_wilcoxon.py
├── data/cleaned_metadata.csv  # small audio manifest; no WAV data
├── results/                   # curated JSON summaries
├── assets/                    # selected thesis-era figures
├── notebooks/                 # vibration simulation walkthrough
└── docs/thesis/
    └── Thesis-Final-Submission.pdf
```

## Implementation and thesis notes

The PDF is the final submitted academic artifact and is preserved unchanged.
The executable source is documented according to its actual behavior. The
submitted thesis and the archived implementation contain several historical
description differences, including the dataset-size summary, the default
normalisation (`gLN` in code versus group normalisation in the thesis text),
mask wording (softmax in code versus sigmoid in the thesis figure/text), and
some vibration-encoder and mixture-pool details. The code and numerical result
files were not changed to make these descriptions superficially agree.

The statistical script uses SciPy's directional `alternative="greater"`
Wilcoxon calculation, while the submitted thesis calls the reported test
two-sided. The archived JSON values are preserved exactly; this distinction
should be resolved or explicitly discussed before claiming an independently
recomputed statistical result.

Development was carried out during the bachelor's thesis project in spring
2026. The repository initially contained an earlier prototype and was
synchronized with the final local implementation in September 2026. The
publication commits in this repository are current synchronization/publication
work; they do not backdate the research or rewrite the earlier Git history.

## License

The code is released under the MIT License. The external bird recordings are
not part of this repository and remain subject to their original licences and
terms.
