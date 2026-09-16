# Evaluation and analysis

These scripts operate on the final training/evaluation pipeline in the repository.
They do not download data or checkpoints.

## Training-run summaries

Aggregate the best validation metric from local run logs with:

```bash
python analysis/aggregate_runs.py --base_dir logs --metric val_si_sdr
```

## Evaluation

`eval.py` evaluates one checkpoint on the fixed recording-level test split and
writes standard and near-zero-SNR high-overlap summaries. See the root README for
a complete command using `data_root`, `metadata`, and `output_json`.

## Paired statistics

`statistical_test.py` collects per-source SI-SDRi values for two checkpoints on
the same mixture pool, caches the arrays, and writes a Wilcoxon summary. The
implementation uses SciPy's directional alternative `Model B > Model A`; this
matches the archived numerical JSON artifacts. The submitted thesis describes
the reported test as two-sided, so interpret that distinction when reproducing
or extending the statistical analysis.

`pooled_wilcoxon.py` combines two cached runs without changing their arrays:

```bash
python analysis/pooled_wilcoxon.py \
  --run_1_dir /path/to/stat_cache/run_1 \
  --run_2_dir /path/to/stat_cache/run_2 \
  --output_json results/statistics/pooled_wilcoxon_results.json
```

The source dataset audio and trained checkpoints are intentionally not included
in this repository. Consequently, the scripts are executable with a compatible
local dataset/checkpoint set, but the archived result files remain the primary
record of the submitted experiment in this clone.
