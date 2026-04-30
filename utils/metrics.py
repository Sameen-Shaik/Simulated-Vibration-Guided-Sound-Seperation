"""
utils/metrics.py
================
Evaluation metrics for source separation.

Implements:
  - SDR, SI-SDR, SIR, SAR (via mir_eval)
  - Spectral Convergence
  - Log Spectral Distance
  - Statistical significance tests (paired t-test, Wilcoxon signed-rank)
  - Per-overlap-ratio breakdown for RQ1 analysis
"""

import sys
sys.path.insert(0, "/usr/local/lib/python3.12/dist-packages")

import warnings
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from scipy import stats

try:
    import mir_eval
    HAS_MIR_EVAL = True
except ImportError:
    HAS_MIR_EVAL = False
    warnings.warn("mir_eval not found; SDR/SIR/SAR unavailable. pip install mir_eval")


# ══════════════════════════════════════════════════════════════════════════════
#  BSS Eval (SDR, SIR, SAR)
# ══════════════════════════════════════════════════════════════════════════════

def compute_bss_eval(
    estimates: np.ndarray,   # (N, T)
    references: np.ndarray,  # (N, T)
) -> Dict[str, float]:
    """
    Compute BSS-Eval metrics: SDR, SIR, SAR using mir_eval.
    Returns mean over speakers.
    """
    if not HAS_MIR_EVAL:
        return {"SDR": float("nan"), "SIR": float("nan"), "SAR": float("nan")}

    # mir_eval expects (n_sources, n_samples)
    try:
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            references.astype(np.float64),
            estimates.astype(np.float64),
            compute_permutation=True,
        )
        return {
            "SDR": float(np.mean(sdr)),
            "SIR": float(np.mean(sir)),
            "SAR": float(np.mean(sar)),
        }
    except Exception as e:
        return {"SDR": float("nan"), "SIR": float("nan"), "SAR": float("nan")}


# ══════════════════════════════════════════════════════════════════════════════
#  SI-SDR (numpy)
# ══════════════════════════════════════════════════════════════════════════════

def si_sdr_np(estimate: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Compute SI-SDR for a single source pair (1-D arrays)."""
    target = target - target.mean()
    estimate = estimate - estimate.mean()
    dot = np.dot(estimate, target)
    target_energy = np.dot(target, target) + eps
    s_target = dot / target_energy * target
    e_noise = estimate - s_target
    ratio = np.dot(s_target, s_target) / (np.dot(e_noise, e_noise) + eps)
    return float(10 * np.log10(ratio + eps))


def compute_si_sdr(
    estimates: np.ndarray,   # (N, T)
    references: np.ndarray,  # (N, T)
) -> float:
    """Mean SI-SDR over N speakers (uses best permutation)."""
    from itertools import permutations
    N = estimates.shape[0]
    best = -float("inf")
    for perm in permutations(range(N)):
        vals = [si_sdr_np(estimates[p], references[i]) for i, p in enumerate(perm)]
        mean_val = float(np.mean(vals))
        if mean_val > best:
            best = mean_val
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  Spectral Metrics
# ══════════════════════════════════════════════════════════════════════════════

def spectral_convergence(
    estimate: np.ndarray,
    target: np.ndarray,
    n_fft: int = 1024,
    eps: float = 1e-8,
) -> float:
    """
    Spectral Convergence = ||mag_ref - mag_est||_F / ||mag_ref||_F
    (lower is better, 0 = perfect)
    """
    window = np.hanning(n_fft)
    hop = n_fft // 4

    def _mag(x):
        frames = [
            np.abs(np.fft.rfft(x[i : i + n_fft] * window))
            for i in range(0, len(x) - n_fft, hop)
        ]
        return np.array(frames).T if frames else np.zeros((n_fft // 2 + 1, 1))

    mag_ref = _mag(target)
    mag_est = _mag(estimate)
    min_len = min(mag_ref.shape[1], mag_est.shape[1])
    mag_ref = mag_ref[:, :min_len]
    mag_est = mag_est[:, :min_len]

    num = np.linalg.norm(mag_ref - mag_est, "fro")
    den = np.linalg.norm(mag_ref, "fro") + eps
    return float(num / den)


def log_spectral_distance(
    estimate: np.ndarray,
    target: np.ndarray,
    n_fft: int = 1024,
    eps: float = 1e-8,
) -> float:
    """
    Mean Log Spectral Distance (dB), averaged over time frames.
    (lower is better)
    """
    window = np.hanning(n_fft)
    hop = n_fft // 4

    def _log_mag(x):
        frames = [
            np.log(np.abs(np.fft.rfft(x[i : i + n_fft] * window)) + eps)
            for i in range(0, len(x) - n_fft, hop)
        ]
        return np.array(frames).T if frames else np.zeros((n_fft // 2 + 1, 1))

    log_ref = _log_mag(target)
    log_est = _log_mag(estimate)
    min_len = min(log_ref.shape[1], log_est.shape[1])
    diff = log_ref[:, :min_len] - log_est[:, :min_len]
    return float(np.sqrt(np.mean(diff ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
#  Full Per-Sample Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_sample(
    estimates: np.ndarray,   # (N, T)
    references: np.ndarray,  # (N, T)
    overlap_ratio: float = 0.0,
) -> Dict[str, float]:
    """Compute all metrics for one mixture sample."""
    results = {}
    results["overlap_ratio"] = overlap_ratio

    # SI-SDR
    results["SI-SDR"] = compute_si_sdr(estimates, references)

    # BSS-Eval
    bss = compute_bss_eval(estimates, references)
    results.update(bss)

    # Spectral metrics (mean over speakers, best permutation)
    from itertools import permutations
    N = estimates.shape[0]
    best_sc, best_lsd = float("inf"), float("inf")
    for perm in permutations(range(N)):
        scs = [spectral_convergence(estimates[p], references[i]) for i, p in enumerate(perm)]
        lsds = [log_spectral_distance(estimates[p], references[i]) for i, p in enumerate(perm)]
        sc_m = float(np.mean(scs))
        lsd_m = float(np.mean(lsds))
        if sc_m < best_sc:
            best_sc = sc_m
            best_lsd = lsd_m

    results["spectral_convergence"] = best_sc
    results["log_spectral_distance"] = best_lsd

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Statistical Significance Tests (RQ1, RQ2)
# ══════════════════════════════════════════════════════════════════════════════

def run_significance_tests(
    model_a_scores: List[float],
    model_b_scores: List[float],
    metric_name: str = "SI-SDR",
    alpha: float = 0.05,
) -> Dict:
    """
    Run paired t-test AND Wilcoxon signed-rank test between two models.
    Per thesis: significance level α = 0.05

    Returns dict with test statistics and interpretation.
    """
    a = np.array(model_a_scores)
    b = np.array(model_b_scores)
    diff = a - b

    # Paired t-test
    t_stat, t_pval = stats.ttest_rel(a, b)

    # Wilcoxon signed-rank test (non-parametric alternative)
    try:
        w_stat, w_pval = stats.wilcoxon(a, b)
    except Exception:
        w_stat, w_pval = float("nan"), float("nan")

    mean_diff = float(np.mean(diff))
    reject_h0 = (t_pval < alpha) or (w_pval < alpha)

    return {
        "metric": metric_name,
        "model_a_mean": float(np.mean(a)),
        "model_b_mean": float(np.mean(b)),
        "mean_difference": mean_diff,
        "t_statistic": float(t_stat),
        "t_pvalue": float(t_pval),
        "wilcoxon_statistic": float(w_stat),
        "wilcoxon_pvalue": float(w_pval),
        "alpha": alpha,
        "reject_h0": reject_h0,
        "conclusion": (
            f"Model A significantly {'better' if mean_diff > 0 else 'worse'} "
            f"than Model B (p={t_pval:.4f})"
            if reject_h0
            else f"No significant difference found (p={t_pval:.4f})"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Overlap-Stratified Analysis (RQ1)
# ══════════════════════════════════════════════════════════════════════════════

def stratified_analysis(
    results: List[Dict],
    overlap_bins: List[float] = [0.0, 0.4, 0.7, 1.0],
    bin_labels: List[str] = ["low (0–0.4)", "medium (0.4–0.7)", "high (0.7–1.0)"],
) -> Dict[str, Dict[str, float]]:
    """
    Group evaluation results by overlap ratio and compute per-group means.
    Used to answer RQ1: how does vibration guidance help at different overlap levels?
    """
    groups: Dict[str, List[Dict]] = {label: [] for label in bin_labels}

    for r in results:
        ov = r.get("overlap_ratio", 0.0)
        for i, (lo, hi) in enumerate(zip(overlap_bins[:-1], overlap_bins[1:])):
            if lo <= ov < hi:
                groups[bin_labels[i]].append(r)
                break

    summary = {}
    for label, group_results in groups.items():
        if not group_results:
            summary[label] = {}
            continue
        metrics = ["SI-SDR", "SDR", "SIR", "SAR", "spectral_convergence", "log_spectral_distance"]
        summary[label] = {
            m: float(np.mean([r[m] for r in group_results if m in r and not np.isnan(r[m])]))
            for m in metrics
        }
        summary[label]["n_samples"] = len(group_results)

    return summary


def print_results_table(
    vib_results: List[Dict],
    baseline_results: List[Dict],
    alpha: float = 0.05,
):
    """Print a formatted comparison table to stdout."""
    metrics = ["SI-SDR", "SDR", "spectral_convergence", "log_spectral_distance"]
    header = f"{'Metric':<25} {'Vibration':>12} {'Baseline':>12} {'Diff':>10} {'p-value':>10} {'Sig?':>6}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for metric in metrics:
        vib_vals = [r[metric] for r in vib_results if metric in r and not np.isnan(r.get(metric, float("nan")))]
        bas_vals = [r[metric] for r in baseline_results if metric in r and not np.isnan(r.get(metric, float("nan")))]
        if not vib_vals or not bas_vals:
            continue
        n = min(len(vib_vals), len(bas_vals))
        test = run_significance_tests(vib_vals[:n], bas_vals[:n], metric, alpha)
        sig = "✓" if test["reject_h0"] else "✗"
        diff = test["mean_difference"]
        print(
            f"{metric:<25} {test['model_a_mean']:>12.3f} {test['model_b_mean']:>12.3f} "
            f"{diff:>+10.3f} {test['t_pvalue']:>10.4f} {sig:>6}"
        )
    print("=" * len(header))


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Metrics Smoke Test ===")
    N, T = 2, 22050 * 3
    estimates = np.random.randn(N, T).astype(np.float32)
    references = np.random.randn(N, T).astype(np.float32)

    result = evaluate_sample(estimates, references, overlap_ratio=0.7)
    for k, v in result.items():
        print(f"  {k}: {v:.4f}")
    print("  ✓ Metrics OK")
