"""Seed-to-seed spread of the gap and ECE changes, per method and dataset.

Usage:
    python scripts/seed_intervals.py
    python scripts/seed_intervals.py --grid results/grid.csv --out-dir results
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["tip_adapter", "tip_adapter_f", "coop", "zero_shot_ts", "clap"]
EXPECTED_PAIRS = {
    "tip_adapter": 31,
    "tip_adapter_f": 31,
    "coop": 31,
    "zero_shot_ts": 26,
    "clap": 31,
}
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
METRICS = {"gap": "signed_gap", "ece": "ece"}


def paired(grid):
    zs = grid[grid.method == "zero_shot"].set_index(["dataset", "seed"])
    out = []
    for method in METHODS:
        rows = grid[grid.method == method]
        if method == "tip_adapter":
            rows = rows[rows.beta == 5.0]
        rows = rows.set_index(["dataset", "seed"])
        d = pd.DataFrame({"start_gap": zs.signed_gap.loc[rows.index]})
        for name, col in METRICS.items():
            d[f"d_{name}"] = rows[col] - zs[col].loc[rows.index]
        d = d.dropna()
        assert len(d) == EXPECTED_PAIRS[method], (method, len(d))
        d["method"] = method
        d["spread_over"] = "shot seeds"
        out.append(d.reset_index().rename(columns={"seed": "draw"}))
    return pd.concat(out)


def paired_tpt(grid):
    big = grid[grid.n_test != 500]
    base = big[big.method == "zero_shot_single"].set_index("dataset")
    tpt = big[big.method == "tpt"]
    assert len(base) == 9 and len(tpt) == 15, (len(base), len(tpt))
    d = pd.DataFrame({
        "dataset": tpt.dataset.values,
        "draw": tpt["run"].astype(int).values,
        "start_gap": base.signed_gap.loc[tpt.dataset].values,
    })
    for name, col in METRICS.items():
        d[f"d_{name}"] = tpt[col].values - base[col].loc[tpt.dataset].values
    d["method"] = "tpt"
    d["spread_over"] = "augmentation runs"
    return d


def summarise(d):
    rows = []
    for (method, dataset), g in d.groupby(["method", "dataset"], sort=False):
        row = {
            "method": method,
            "dataset": dataset,
            "spread_over": g.spread_over.iloc[0],
            "n": len(g),
            "start_gap": g.start_gap.mean(),
        }
        for name in METRICS:
            x = g[f"d_{name}"].to_numpy()
            mean = x.mean()
            if len(x) > 1:
                half = T95[len(x) - 1] * x.std(ddof=1) / np.sqrt(len(x))
                lo, hi = mean - half, mean + half
            else:
                lo = hi = np.nan
            row[f"d_{name}"] = mean
            row[f"d_{name}_lo"] = lo
            row[f"d_{name}_hi"] = hi
            row[f"d_{name}_sign"] = f"{int((np.sign(x) == np.sign(mean)).sum())}/{len(x)}"
        rows.append(row)
    table = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS + ["tpt"])}
    return table.sort_values(
        ["method", "start_gap"], key=lambda c: c.map(order) if c.name == "method" else c
    ).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", default="results/grid.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    grid = pd.read_csv(args.grid)
    draws = pd.concat([paired(grid), paired_tpt(grid)])
    table = summarise(draws)
    assert len(table) == 53, len(table)
    assert (table.n == 1).sum() == 7, (table.n == 1).sum()

    print(table.round(2).to_string(index=False))
    path = out_dir / "seed_intervals.csv"
    table.round(4).to_csv(path, index=False)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
