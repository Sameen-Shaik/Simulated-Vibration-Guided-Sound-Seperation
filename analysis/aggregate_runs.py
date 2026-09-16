import argparse
import os
import pandas as pd
import numpy as np


def get_best_metric(csv_path, metric="val_si_sdr"):
    """
    Extract best (max) value of a metric from a run CSV.
    """
    df = pd.read_csv(csv_path)

    if metric not in df.columns:
        raise ValueError(f"{metric} not found in {csv_path}")

    return df[metric].max()


def collect_results(base_dir, metric="val_si_sdr"):
    """
    Folder structure:

    logs/
        baseline/
            run_1/metrics.csv
            run_2/metrics.csv
        multimodal/
            run_1/metrics.csv
            run_2/metrics.csv
    """

    results = {}

    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)

        if not os.path.isdir(model_path):
            continue

        scores = []

        for run_folder in os.listdir(model_path):
            run_path = os.path.join(model_path, run_folder)

            csv_path = os.path.join(run_path, "metrics.csv")

            if not os.path.exists(csv_path):
                print(f"[WARN] Missing: {csv_path}")
                continue

            try:
                score = get_best_metric(csv_path, metric)
                scores.append(score)
            except Exception as e:
                print(f"[WARN] Error in {csv_path}: {e}")

        if scores:
            results[model_name] = {
                "scores": scores,
                "mean": np.mean(scores),
                "std": np.std(scores)
            }

    return results


def print_table(results):
    print("\n===== MODEL COMPARISON =====\n")

    # print(f"{'Model':<15}{'Runs':<20}{'Mean':<10}{'Std':<10}")
    print(f"{'Model':<15}{'Mean':<10}{'Std':<10}")
    print("-" * 60)

    for model, stats in results.items():
        print(f"{model:<15}"
            #   f"{len(stats['scores']):<20}"
              f"{stats['mean']:<10.3f}"
              f"{stats['std']:<10.3f}")

        # print(f"  scores: {np.round(stats['scores'], 3)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate best validation metrics across training runs.")
    parser.add_argument("--base_dir", default="./logs",
                        help="Directory containing one subdirectory per model type.")
    parser.add_argument("--metric", default="val_si_sdr",
                        help="CSV column to aggregate.")
    args = parser.parse_args()

    results = collect_results(
        args.base_dir,
        metric=args.metric,
    )

    print_table(results)
