"""Score the calibration forecast on the frozen grid, then plot it.

Rule: a booster improves ECE only from an underconfident start (g < 0);
a corrector improves ECE from either side.

Usage:
    python scripts/score_forecast.py
    python scripts/score_forecast.py --grid results/grid.csv --out-dir results
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

KIND = {
    "tip_adapter": "booster",
    "tip_adapter_f": "booster",
    "tpt": "booster",
    "coop": "corrector",
    "zero_shot_ts": "corrector",
}
LABEL = {
    "tip_adapter": "Tip-Adapter",
    "tip_adapter_f": "Tip-Adapter-F",
    "tpt": "TPT",
    "coop": "CoOp",
    "zero_shot_ts": "Temperature scaling",
}
MARKER = {
    "tip_adapter": "o",
    "tip_adapter_f": "s",
    "tpt": "^",
    "coop": "o",
    "zero_shot_ts": "s",
}


MIXED = {"clap": "CLAP"}

EXPECTED_GSHOTS = {
    "tip_adapter": (27, 31),
    "tip_adapter_f": (31, 31),
    "coop": (30, 31),
    "zero_shot_ts": (24, 26),
    "tpt": (7, 9),
}
EXPECTED_TEST = {
    "tip_adapter": (28, 31),
    "tip_adapter_f": (30, 31),
    "coop": (30, 31),
    "zero_shot_ts": (24, 26),
    "tpt": (7, 9),
}
EXPECTED_TOTAL = (119, 128)
EXPECTED_WITH_ERRORS = (102, 111)

NEAR_ZERO = 0.5

MISS_COLOR = {
    "bad shot estimate": "#eb6834",
    "overshoot": "#2a78d6",
    "near-zero delta": "#1baf7a",
    "other": "#e87ba4",
}
HIT_COLOR = "#b9b8b1"


def build_cells(grid, kinds=KIND):
    """One row per (method, dataset, seed) cell, paired with its zero-shot start."""
    grid = grid[~((grid.method == "tip_adapter") & (grid.beta != 5))]
    zs = grid[grid.method == "zero_shot"].set_index(["dataset", "seed"])
    zs_single = grid[(grid.method == "zero_shot_single") & (grid.n_test != 500)]
    zs_single = zs_single.set_index("dataset")
    assert (
        zs_single.index.is_unique
    ), "expected one n=2,000 single-template row per dataset"

    rows = []
    for method in kinds:
        adapted = grid[grid.method == method]
        if method == "tpt":
            adapted = adapted[adapted.n_test != 500]
        for _, a in adapted.iterrows():
            if pd.isna(a.ece):
                continue
            z = (
                zs_single.loc[a.dataset]
                if method == "tpt"
                else zs.loc[(a.dataset, a.seed)]
            )
            rows.append(
                {
                    "method": method,
                    "kind": kinds[method],
                    "dataset": a.dataset,
                    "seed": a.seed,
                    "g_shots": a.g_shots,
                    "g_test": z.signed_gap,
                    "delta": a.signed_gap - z.signed_gap,
                    "d_ece": a.ece - z.ece,
                    "n_shot_errors": a.n_shot_errors,
                }
            )
    return pd.DataFrame(rows)


def forecast_better(kind, g):
    return (g < 0) if kind == "booster" else True


def miss_type(c):
    if (c.g_shots > 0) != (c.g_test > 0):
        return "bad shot estimate"
    if c.kind == "booster" and c.g_test < 0 and c.delta > 2 * abs(c.g_test):
        return "overshoot"
    if abs(c.delta) < NEAR_ZERO:
        return "near-zero delta"
    return "other"


def loo_kind(cells, method, held_out):
    """Booster if the median delta is positive on the other datasets' overconfident starts."""
    train = cells[
        (cells.method == method) & (cells.dataset != held_out) & (cells.g_test > 0)
    ]
    if train.empty:
        return None
    return "booster" if train.delta.median() > 0 else "corrector"


def score(cells):
    cells = cells.copy()
    cells["improved"] = cells.d_ece < 0
    cells["hit"] = [
        forecast_better(k, g) == i
        for k, g, i in zip(cells.kind, cells.g_shots, cells.improved)
    ]
    cells["hit_test_gap"] = [
        forecast_better(k, g) == i
        for k, g, i in zip(cells.kind, cells.g_test, cells.improved)
    ]
    cells["miss_type"] = [
        None if h else miss_type(c) for h, (_, c) in zip(cells.hit, cells.iterrows())
    ]
    loo = []
    for _, c in cells.iterrows():
        k = loo_kind(cells, c.method, c.dataset)
        loo.append(None if k is None else forecast_better(k, c.g_shots) == c.improved)
    cells["loo_kind_defined"] = [h is not None for h in loo]
    cells["hit_loo"] = loo
    return cells


def check_counts(cells):
    print("Forecast hits per method (g_shots / test gap):")
    for method in KIND:
        m = cells[cells.method == method]
        got = (int(m.hit.sum()), len(m))
        got_test = (int(m.hit_test_gap.sum()), len(m))
        print(f"  {LABEL[method]:<20} {got[0]}/{got[1]}   {got_test[0]}/{got_test[1]}")
        assert (
            got == EXPECTED_GSHOTS[method]
        ), f"{method}: {got} != {EXPECTED_GSHOTS[method]}"
        assert got_test == EXPECTED_TEST[method], f"{method} (test gap): {got_test}"
    total = (int(cells.hit.sum()), len(cells))
    with_errors = cells[cells.n_shot_errors > 0]
    with_errors = (int(with_errors.hit.sum()), len(with_errors))
    print(
        f"  total {total[0]}/{total[1]}, excluding no-error cells {with_errors[0]}/{with_errors[1]}"
    )
    assert total == EXPECTED_TOTAL, total
    assert with_errors == EXPECTED_WITH_ERRORS, with_errors
    print("Counts reproduce the Phase 2 freeze.\n")


def report(cells):
    print("Misses:")
    misses = cells[~cells.hit].sort_values(["method", "dataset", "seed"])
    print(
        misses[
            [
                "method",
                "dataset",
                "seed",
                "g_shots",
                "g_test",
                "delta",
                "d_ece",
                "miss_type",
            ]
        ]
        .round(2)
        .to_string(index=False)
    )
    print()
    defined = cells[cells.loo_kind_defined]
    print(
        f"Leave one dataset out: {int(defined.hit_loo.sum())}/{len(defined)} "
        f"({len(cells) - len(defined)} cells with no overconfident start left to measure kind)"
    )
    for method in KIND:
        m = defined[defined.method == method]
        print(f"  {LABEL[method]:<20} {int(m.hit_loo.sum())}/{len(m)}")


def score_mixed(grid):
    cells = build_cells(grid, {m: "mixed" for m in MIXED})
    cells["improved"] = cells.d_ece < 0
    cells["hit_as_booster"] = (cells.g_shots < 0) == cells.improved
    cells["hit_as_corrector"] = cells.improved
    cells["loo_kind"] = [loo_kind(cells, c.method, c.dataset) for _, c in cells.iterrows()]
    cells["hit_loo"] = [
        forecast_better(k, c.g_shots) == c.improved
        for k, (_, c) in zip(cells.loo_kind, cells.iterrows())
    ]
    for method, label in MIXED.items():
        m = cells[cells.method == method]
        over, under = m[m.g_test > 0], m[m.g_test < 0]
        print(f"{label} (measured kind: mixed), {len(m)} cells")
        print(f"  gap lowered from overconfident starts {int((over.delta < 0).sum())}/{len(over)}, "
              f"from underconfident starts {int((under.delta < 0).sum())}/{len(under)}")
        print(f"  forecast if called a booster   {int(m.hit_as_booster.sum())}/{len(m)}")
        print(f"  forecast if called a corrector {int(m.hit_as_corrector.sum())}/{len(m)}")
        print(f"  leave one dataset out          {int(m.hit_loo.sum())}/{len(m)} "
              f"(kind measured as {', '.join(sorted(set(m.loo_kind)))})")
        table = m.groupby("dataset").agg(
            start=("g_test", "mean"), delta=("delta", "mean"), d_ece=("d_ece", "mean"),
            improved=("improved", "sum"), n=("seed", "count"),
        ).sort_values("start")
        print(table.round(2).to_string())
    return cells


def plot(cells, x_col, x_label, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    panels = [
        ("booster", "Boosters: help only from g < 0"),
        ("corrector", "Correctors: help from either side"),
    ]
    for ax, (kind, title) in zip(axes, panels):
        ax.axhline(0, color="#7a7974", lw=0.8, zorder=0)
        ax.axvline(0, color="#7a7974", lw=0.8, zorder=0)
        ax.grid(color="#e6e5e0", lw=0.6, zorder=0)
        for method in [m for m, k in KIND.items() if k == kind]:
            m = cells[cells.method == method]
            for _, c in m.iterrows():
                color = HIT_COLOR if c.hit else MISS_COLOR[c.miss_type]
                hollow = c.n_shot_errors == 0
                ax.scatter(
                    c[x_col],
                    c.d_ece,
                    marker=MARKER[method],
                    s=42,
                    facecolors="none" if hollow else color,
                    edgecolors=color,
                    linewidths=1.6 if hollow else 0.6,
                    zorder=3 if c.hit else 4,
                )
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel(x_label)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(
        "Change in ECE (adapted − zero-shot, points)\nbelow 0 = calibration improved"
    )

    handles = []
    for method in KIND:
        handles.append(
            plt.Line2D(
                [],
                [],
                marker=MARKER[method],
                ls="",
                color="#4a4945",
                label=f"{LABEL[method]} ({KIND[method]})",
            )
        )
    handles.append(
        plt.Line2D([], [], marker="o", ls="", color=HIT_COLOR, label="forecast hit")
    )
    for t, col in MISS_COLOR.items():
        if (cells.miss_type == t).any():
            handles.append(
                plt.Line2D([], [], marker="o", ls="", color=col, label=f"miss: {t}")
            )
    handles.append(
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor="#4a4945",
            label="no shot errors (g ≤ 0 forced)",
        )
    )
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grid", default="results/grid.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    cells = score(build_cells(pd.read_csv(args.grid)))
    check_counts(cells)
    report(cells)
    print()
    mixed = score_mixed(pd.read_csv(args.grid))
    print()
    mixed.round(4).to_csv(out_dir / "forecast_mixed.csv", index=False)
    print(f"wrote {out_dir / 'forecast_mixed.csv'}")

    cells.round(4).to_csv(out_dir / "forecast_scores.csv", index=False)
    print(f"wrote {out_dir / 'forecast_scores.csv'}")
    plot(
        cells,
        "g_shots",
        "Starting gap estimated on the shots, g_shots (points)",
        out_dir / "forecast_gshots.png",
    )
    plot(
        cells,
        "g_test",
        "Starting gap on the test set (points), diagnostic",
        out_dir / "forecast_testgap.png",
    )


if __name__ == "__main__":
    main()
