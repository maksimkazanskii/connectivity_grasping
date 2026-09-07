#!/usr/bin/env python3
"""
Overlay E2 and E3 exploration-speedup distributions on ONE plot.

Object-level speedup at Random@target:
    speedup = target / connectivity_cost

Unlike the earlier version, this script DOES NOT normalize speedup by the
experiment median. Raw speedup is the scientifically meaningful quantity.

Defaults:
  E2: results/online_connectivity/online_connectivity_matches.csv
  E3: results/active_connectivity_e3/online_connectivity_matches.csv
  target: Random@100
  bin width: 0.25x

Outputs:
  exploration_speedup_histogram_e2_e3.png
  exploration_speedup_histogram_e2_e3.pdf
  exploration_speedups_e2_e3.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--e2",
        default="results/online_connectivity/online_connectivity_matches.csv",
    )
    p.add_argument(
        "--e3",
        default="results/active_connectivity_e3/online_connectivity_matches.csv",
    )
    p.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[100, 200],
        help="Random targets to plot (default: 100 200)",
    )
    p.add_argument(
        "--bin-width",
        type=float,
        default=0.25,
        help="Histogram bin width in raw speedup units (default: 0.25)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.50,
        help="Histogram transparency (default: 0.50)",
    )
    p.add_argument(
        "--outdir",
        default="results/e2_e3_comparison",
    )
    p.add_argument(
        "--max-x",
        type=float,
        default=None,
        help="Optional x-axis maximum. Default uses all observed values.",
    )
    return p.parse_args()


def find_column(df, candidates, required=True):
    lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {candidates}.\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def load_speedups(path, target, experiment):
    df = pd.read_csv(path)

    target_col = find_column(
        df,
        ["target", "random_target", "target_points", "target_queries"],
    )
    cost_col = find_column(
        df,
        [
            "connectivity_points",
            "connectivity_queries",
            "connectivity_cost",
            "connectivity_match",
            "connectivity_first_match",
        ],
    )
    object_col = find_column(
        df,
        ["object", "object_name", "object_id", "file", "filename", "path"],
        required=False,
    )
    reached_col = find_column(
        df,
        ["connectivity_reached", "reached_connectivity", "connectivity_success"],
        required=False,
    )

    sub = df[df[target_col] == target].copy()

    if sub.empty:
        raise ValueError(
            f"{experiment}: target={target} not found. "
            f"Available targets: {sorted(pd.unique(df[target_col]))}"
        )

    if reached_col is not None:
        r = sub[reached_col]
        if r.dtype == bool:
            sub = sub[r]
        else:
            sub = sub[
                r.astype(str).str.lower().isin(
                    ["1", "true", "yes", "reached"]
                )
            ]

    sub[cost_col] = pd.to_numeric(sub[cost_col], errors="coerce")
    sub = sub[np.isfinite(sub[cost_col]) & (sub[cost_col] > 0)].copy()

    # Ensure one observation per object at the selected target.
    if object_col is not None:
        sub = sub.drop_duplicates(object_col)
        objects = sub[object_col].astype(str)
    else:
        objects = pd.Series(np.arange(len(sub)), index=sub.index).astype(str)

    sub["speedup"] = float(target) / sub[cost_col]

    return pd.DataFrame(
        {
            "experiment": experiment,
            "object": objects,
            "connectivity_cost": sub[cost_col].to_numpy(),
            "speedup": sub["speedup"].to_numpy(),
        }
    )


def plot_target(args, target, outdir):
    e2 = load_speedups(args.e2, target, "E2")
    e3 = load_speedups(args.e3, target, "E3")
    pd.concat([e2, e3], ignore_index=True).to_csv(
        outdir / f"exploration_speedups_e2_e3_random{target}.csv", index=False)

    x2, x3 = e2["speedup"].to_numpy(), e3["speedup"].to_numpy()
    m2, m3 = float(np.median(x2)), float(np.median(x3))
    q2, q3 = np.quantile(x2, [.25,.5,.75]), np.quantile(x3, [.25,.5,.75])
    xmax = args.max_x if args.max_x is not None else max(x2.max(), x3.max())
    bins = np.arange(0, xmax + args.bin_width, args.bin_width)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    c2, c3 = "tab:blue", "tab:orange"

    ax.hist(x2, bins=bins, density=True, alpha=args.alpha, color=c2,
            label=f"E2 (n={len(x2)})", edgecolor="black", linewidth=.5)
    ax.hist(x3, bins=bins, density=True, alpha=args.alpha, color=c3,
            label=f"E3 (n={len(x3)})", edgecolor="black", linewidth=.5)

    # Median colors match their corresponding histograms.
    ax.axvline(m2, color=c2, ls="--", lw=2.2,
               label=f"E2 median = {m2:.2f}×")
    ax.axvline(m3, color=c3, ls="--", lw=2.2,
               label=f"E3 median = {m3:.2f}×")

    ax.set_xlabel("Exploration speedup (×)")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of exploration speedup — Random@{target}")
    ax.set_xlim(0, xmax)
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    stem = outdir / f"exploration_speedup_histogram_e2_e3_random{target}"
    fig.savefig(str(stem)+".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(stem)+".pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nRandom@{target}")
    print("="*60)
    print(f"E2: n={len(x2)} | median={m2:.3f}x | Q25={q2[0]:.3f}x | Q75={q2[2]:.3f}x")
    print(f"E3: n={len(x3)} | median={m3:.3f}x | Q25={q3[0]:.3f}x | Q75={q3[2]:.3f}x")
    print(f"Median ratio E3/E2: {m3/m2:.3f}x")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        plot_target(args, target, outdir)
    print(f"\nSaved plots to {outdir}")


if __name__ == "__main__":
    main()
