#!/usr/bin/env python3
"""Plot E3 N=1600 CA acquisition-component ablation."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REGIMES = ["full", "bridge", "frontier", "boundary", "novelty"]
LABELS = {
    "full": "Full CA",
    "bridge": "Bridge only",
    "frontier": "Frontier only",
    "boundary": "Boundary only",
    "novelty": "Novelty only",
}


def find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"None of {names} found. Columns: {list(df.columns)}")


def ca_method(df):
    mcol = find_col(df, ["method", "Method"])
    methods = df[mcol].astype(str).str.lower().unique()

    for name in ["connectivity-aware", "connectivity_aware", "ca"]:
        if name in methods:
            return name

    other = [m for m in methods if m not in {"random", "fps"}]
    if len(other) == 1:
        return other[0]

    raise ValueError(f"Cannot identify CA method: {methods}")


def paired_gain(df):
    mcol = find_col(df, ["method", "Method"])
    acol = find_col(
        df,
        ["phase2_auc", "phase_2_auc", "auc", "AUC", "exploitation_auc"],
    )
    ocol = find_col(
        df,
        ["object", "object_name", "object_file", "path", "filename"],
    )
    scol = find_col(df, ["seed", "seed_index", "seed_id"])

    d = df.copy()
    d["_method"] = d[mcol].astype(str).str.lower()

    method = ca_method(d)

    rnd = (
        d[d["_method"] == "random"][[ocol, scol, acol]]
        .rename(columns={acol: "random"})
    )
    ca = (
        d[d["_method"] == method][[ocol, scol, acol]]
        .rename(columns={acol: "ca"})
    )

    p = ca.merge(rnd, on=[ocol, scol], validate="one_to_one")

    if p.empty:
        raise ValueError("No paired CA/Random rows found.")

    return (100.0 * (p["ca"] - p["random"]) / p["random"]).to_numpy()


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    ci = 0.0 if len(x) <= 1 else 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return mean, ci


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="results/e3_N1600_component_ablation",
    )
    parser.add_argument(
        "--output",
        default="results/e3_N1600_component_ablation/"
                "e3_N1600_component_ablation.png",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    names, means, errors = [], [], []

    print("\n" + "=" * 70)
    print("E3 N=1600 — CA ACQUISITION COMPONENT ABLATION")
    print("=" * 70)

    for regime in REGIMES:
        path = results_dir / regime / "phase_results_per_seed.csv"

        if not path.exists():
            print(f"Skipping missing: {path}")
            continue

        values = paired_gain(pd.read_csv(path))
        mean, ci = mean_ci(values)

        names.append(regime)
        means.append(mean)
        errors.append(ci)

        print(f"{LABELS[regime]:15s} {mean:+.3f}% ± {ci:.3f}% (95% CI)")

    print("=" * 70)

    if not names:
        raise RuntimeError(f"No result files found under {results_dir}")

    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0, linestyle="--", linewidth=1.5, alpha=0.7)

    ax.errorbar(
        x,
        means,
        yerr=errors,
        marker="o",
        linewidth=2,
        markersize=7,
        capsize=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names])
    ax.set_ylabel("Phase-2 AUC gain over Random (%)")
    ax.set_xlabel("CA acquisition function, $N=1600$")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()