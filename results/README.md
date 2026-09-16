# Curated evaluation artifacts

The JSON files in this directory are copied from the final thesis workspace
without changing their numerical contents. Checkpoint paths inside the JSON
files are provenance fields; the corresponding model weights are intentionally
not published.

The primary headline values in the submitted thesis are two-run averages. The
two baseline files and the two `lambda=0.1` multimodal files provide those
inputs:

- standard test SI-SDRi: 3.652 dB (baseline) and 4.618 dB (multimodal),
  an absolute gain of 0.966 dB;
- near-zero-SNR high-overlap SI-SDRi: 3.541 dB and 4.618 dB, a gain of
  1.077 dB.

`statistics/pooled_wilcoxon_results.json` records the pooled result reported by
the thesis: median pairwise improvement 0.491 dB, 95% CI [0.426, 0.568] dB,
and p approximately 2.24e-44.

The ablation files retain the per-seed summaries for the tested vibration-loss
weights. `summary/results_aggregated.json` is the workspace's generated
aggregation of these small JSON files and is retained as a convenience copy.
