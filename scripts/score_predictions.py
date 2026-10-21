"""Scores the predictions written by write_predictions.py against the backbone's grid.

Usage:
    VLM_BACKBONE=rn50 python scripts/score_predictions.py
"""

import argparse, sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]

from backbones import BACKBONE, GRID
from score_forecast import KIND, LABEL, build_cells, forecast_better, miss_type, score_mixed

METHODS = {m: k for m, k in KIND.items() if m != "tpt"}
KEY = ["method", "dataset", "seed"]


def score(predictions, grid):
    cells = build_cells(grid, METHODS)
    merged = predictions.merge(cells, on=KEY, how="outer", suffixes=("_pred", ""), indicator=True)
    unmatched = merged[merged._merge != "both"]
    assert unmatched.empty, unmatched[KEY + ["_merge"]]
    assert ((merged.g_shots_pred - merged.g_shots).abs() < 1e-3).all(), "g_shots differ from the grid"
    assert (merged.n_shot_errors_pred == merged.n_shot_errors).all(), "shot errors differ from the grid"
    assert (merged.kind_pred == merged.kind).all()
    rule = [forecast_better(k, g) for k, g in zip(merged.kind, merged.g_shots_pred)]
    assert ((merged.predicted_ece == "better") == rule).all(), "a prediction does not follow the rule"

    merged["improved"] = merged.d_ece < 0
    merged["hit"] = (merged.predicted_ece == "better") == merged.improved
    merged["hit_test_gap"] = [
        forecast_better(k, g) == i for k, g, i in zip(merged.kind, merged.g_test, merged.improved)
    ]
    merged["miss_type"] = [None if h else miss_type(c) for h, (_, c) in zip(merged.hit, merged.iterrows())]
    cols = KEY + ["kind", "g_shots", "g_test", "n_shot_errors", "predicted_ece", "delta", "d_ece",
                  "improved", "hit", "hit_test_gap", "miss_type"]
    return merged[cols].sort_values(KEY).reset_index(drop=True)


def report(scored):
    print("Predictions scored (g_shots / test gap):")
    for method in METHODS:
        m = scored[scored.method == method]
        print(f"  {LABEL[method]:<20} {int(m.hit.sum())}/{len(m)}   {int(m.hit_test_gap.sum())}/{len(m)}")
    with_errors = scored[scored.n_shot_errors > 0]
    print(f"  total {int(scored.hit.sum())}/{len(scored)}, "
          f"excluding no-error cells {int(with_errors.hit.sum())}/{len(with_errors)}")
    over = scored[scored.predicted_ece == "worse"]
    print(f"  predicted worse {int(over.hit.sum())}/{len(over)}, "
          f"predicted better {int(scored.hit.sum() - over.hit.sum())}/{len(scored) - len(over)}\n")
    print("Misses:")
    misses = scored[~scored.hit]
    print(misses[KEY + ["g_shots", "g_test", "delta", "d_ece", "miss_type"]].round(2).to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=REPO_ROOT / "results" / f"predictions_{BACKBONE}.csv")
    parser.add_argument("--grid", type=Path, default=GRID)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / f"predictions_{BACKBONE}_scored.csv")
    args = parser.parse_args()

    grid = pd.read_csv(args.grid)
    predictions = pd.read_csv(args.predictions)
    models = set(grid.model_name) | set(predictions.model_name)
    assert len(models) == 1, f"grid and predictions come from different models: {sorted(models)}"
    print(f"{models.pop()}  predictions from {', '.join(sorted(set(predictions.git_commit)))}, "
          f"grid from {', '.join(sorted(set(grid.git_commit)))}\n")

    scored = score(predictions, grid)
    report(scored)
    score_mixed(grid)
    scored.round(4).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
