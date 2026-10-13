"""Compare temperature scaling and CoOp against zero-shot on every dataset.

Usage:
    python scripts/compare_ts_coop.py
"""

import argparse
from pathlib import Path

import pandas as pd

KEY = ["dataset", "seed"]


def per_seed(grid):
    zs = grid[grid.method == "zero_shot"].set_index(KEY)
    ts = grid[grid.method == "zero_shot_ts"].set_index(KEY)
    coop = grid[grid.method == "coop"].set_index(KEY)
    table = pd.DataFrame({
        "start_gap": zs.signed_gap,
        "n_shot_errors": zs.n_shot_errors,
        "ece_zs": zs.ece,
        "ece_ts": ts.ece,
        "ece_coop": coop.ece,
        "temperature": ts.temperature,
        "acc_zs": zs.accuracy,
        "acc_coop": coop.accuracy,
    })
    table["acc_gain_coop"] = table.acc_coop - table.acc_zs
    return table.reset_index()


def per_dataset(seeds):
    table = seeds.groupby("dataset").agg(
        start_gap=("start_gap", "mean"),
        n_shot_errors=("n_shot_errors", "mean"),
        n_seeds=("seed", "count"),
        n_seeds_ts=("ece_ts", "count"),
        ece_zs=("ece_zs", "mean"),
        ece_ts=("ece_ts", "mean"),
        ece_coop=("ece_coop", "mean"),
        temperature=("temperature", "mean"),
        acc_gain_coop=("acc_gain_coop", "mean"),
    )
    return table.sort_values("start_gap").reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="results/grid.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    seeds = per_seed(pd.read_csv(args.grid))
    datasets = per_dataset(seeds)
    assert len(seeds) == 31, len(seeds)
    assert datasets.n_seeds_ts.sum() == 26, datasets.n_seeds_ts.sum()

    print(datasets.round(2).to_string(index=False))
    seeds.round(4).to_csv(out_dir / "ts_vs_coop_seeds.csv", index=False)
    datasets.round(4).to_csv(out_dir / "ts_vs_coop.csv", index=False)
    print(f"\nwrote {out_dir / 'ts_vs_coop.csv'} and {out_dir / 'ts_vs_coop_seeds.csv'}")


if __name__ == "__main__":
    main()
