import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

# ── Helpers ──────────────────────────────────────────────────────────────────

def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05) -> tuple:
    """Calculates the 95% bootstrap confidence interval for the median."""
    medians = [
        np.median(np.random.choice(diffs, len(diffs), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(medians, 100 * alpha / 2)
    hi = np.percentile(medians, 100 * (1 - alpha / 2))
    return lo, hi


# ── Main Pooling Logic ───────────────────────────────────────────────────────

def main(args):
    run_1 = Path(args.run_1_dir)
    run_2 = Path(args.run_2_dir)

    path_a1 = run_1 / "sisdri_model_a.npy"
    path_b1 = run_1 / "sisdri_model_b.npy"
    path_a2 = run_2 / "sisdri_model_a.npy"
    path_b2 = run_2 / "sisdri_model_b.npy"

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    print("[stat] Loading cached run scores...")
    a1 = np.load(path_a1)
    b1 = np.load(path_b1)
    a2 = np.load(path_a2)
    b2 = np.load(path_b2)

    # Verify run alignments before pooling
    assert len(a1) == len(b1), f"Run 1 shape mismatch: Model A ({len(a1)}) vs Model B ({len(b1)})"
    assert len(a2) == len(b2), f"Run 2 shape mismatch: Model A ({len(a2)}) vs Model B ({len(b2)})"

    # Pool pairwise differences across both seed/run splits
    diffs = np.concatenate([b1 - a1, b2 - a2])

    print(f"[stat] Running statistical analysis on {len(diffs)} total pooled samples...")

    # Wilcoxon signed-rank test (H1: Model B performs better than Model A)
    stat, p = wilcoxon(diffs, alternative="greater")
    median_diff = np.median(diffs)

    # Compute 95% Bootstrap Confidence Intervals
    ci_lo, ci_hi = bootstrap_ci(diffs, n_boot=10_000, alpha=0.05)

    # ── Terminal Output ───────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  Pooled Wilcoxon Signed-Rank Test (Model B > Model A)")
    print("="*55)
    print(f"  Total n samples        : {len(diffs)} ({len(a1)} from Run 1 + {len(a2)} from Run 2)")
    print(f"  Wilcoxon statistic     : {stat:.2f}")
    print(f"  p-value                : {p:.4e}")
    print(f"  Median improvement     : {median_diff:+.4f} dB")
    print(f"  95% bootstrap CI       : [{ci_lo:+.4f}, {ci_hi:+.4f}] dB")
    print(f"  H0 rejected (α=0.05)   : {'YES' if p < 0.05 else 'NO'}")
    print("="*55)

    # ── LaTeX Snippet for Thesis ──────────────────────────────────────────────
    reject = "rejected" if p < 0.05 else "not rejected"
    print("\n--- Copy this into your thesis ---")
    print(
        f"A pooled paired Wilcoxon signed-rank test on the compiled per-sample "
        f"SI-SDRi values ($N = {len(diffs)}$) yields $p = {p:.2e}$, with a median pairwise "
        f"improvement of ${median_diff:+.3f}$~dB "
        f"(95\\% bootstrap CI: $[{ci_lo:+.3f},\\,{ci_hi:+.3f}]$~dB). "
        f"The null hypothesis $H_0$ is {reject} at $\\alpha = 0.05$."
    )

    # ── Save JSON Metadata ────────────────────────────────────────────────────
    out = {
        "n_samples":        int(len(diffs)),
        "run_1_samples":    int(len(a1)),
        "run_2_samples":    int(len(a2)),
        "wilcoxon_stat":    float(stat),
        "p_value":          float(p),
        "median_diff_dB":   float(median_diff),
        "ci_95_lo":         float(ci_lo),
        "ci_95_hi":         float(ci_hi),
        "h0_rejected":      bool(p < 0.05),
    }

    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[stat] Consolidated summary successfully saved to: {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pool cached per-source SI-SDRi arrays from two evaluation runs."
    )
    parser.add_argument("--run_1_dir", required=True,
                        help="Directory containing run 1 sisdri_model_{a,b}.npy files")
    parser.add_argument("--run_2_dir", required=True,
                        help="Directory containing run 2 sisdri_model_{a,b}.npy files")
    parser.add_argument("--output_json",
                        default="./results/statistics/pooled_wilcoxon_results.json")
    main(parser.parse_args())
