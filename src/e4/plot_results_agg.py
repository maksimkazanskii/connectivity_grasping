#!/usr/bin/env python3
"""
Aggregate E4 results across attempt budgets.

Produces exactly two figures for attempts 100, 200, 400 at fixed memory M=200:

1) Heatmap:
   rows    = attempt budget
   columns = SA+Memory for each alpha, then SA, Random, FPS
   values  = mean normalized AUC ± 95% CI

2) Ranked per-object gain:
   one ordered curve per attempt budget
   delta = (SA + Memory) - SA
   fixed M=200 and fixed alpha (default 0.50)

Expected directory layout:
  RESULTS_DIR/
    attempts_100/independent_targets/{results_per_seed.csv OR seed_*/results_per_seed.csv}
    attempts_200/independent_targets/{...}
    attempts_400/independent_targets/{...}
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


ATTEMPTS = [100, 200, 400]
MEMORY = 200
ALPHAS = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00]
CMAP = LinearSegmentedColormap.from_list(
    "gold", ["#fff1d6", "#fdae6b", "#d94801", "#7f2704"]
)


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_objects(root):
    """Load all seeds and average to one row per target/configuration."""
    seed_files = list(root.glob("seed_*/results_per_seed.csv"))
    seed_files.sort(key=lambda p: int(p.parent.name.split("_", 1)[1]))
    direct_file = root / "results_per_seed.csv"

    if seed_files:
        result_files = seed_files
    elif direct_file.exists():
        result_files = [direct_file]
    else:
        raise FileNotFoundError(
            f"No results_per_seed.csv under {root} or {root}/seed_*/"
        )

    print(f"\nLoading {root}")
    for f in result_files:
        print(f"  {f}")

    raw = pd.concat([pd.read_csv(f) for f in result_files], ignore_index=True)

    req = {"target", "seed", "method", "library_size", "alpha", "normalized_auc"}
    missing = req - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns in {root}: {sorted(missing)}")

    obj = (
        raw.groupby(
            ["target", "method", "library_size", "alpha"],
            dropna=False,
            as_index=False,
        )
        .agg(
            normalized_auc=("normalized_auc", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )

    # Baselines must be one row per target after seed averaging.
    for method in ["random", "fps", "ca"]:
        x = obj.loc[obj.method.eq(method)]
        if x.empty:
            raise ValueError(f"No {method!r} baseline in {root}")
        if not x.target.is_unique:
            raise ValueError(
                f"{method!r} baseline has duplicate target rows in {root}"
            )

    return obj


def baseline(obj, method):
    x = obj.loc[obj.method.eq(method), ["target", "normalized_auc"]].copy()
    if not x.target.is_unique:
        raise ValueError(f"{method} baseline has duplicate targets")
    return x


def transfer_at(obj, memory, alpha):
    x = obj.loc[
        obj.library_size.eq(memory)
        & np.isclose(obj.alpha.astype(float), alpha),
        ["target", "normalized_auc"],
    ].copy()

    # Explicitly exclude baseline rows if they carry library_size/alpha values.
    if "method" in obj.columns:
        idx = obj.library_size.eq(memory) & np.isclose(obj.alpha.astype(float), alpha)
        candidates = obj.loc[idx].copy()

        # In the current E4 format transfer is the non-baseline method.
        candidates = candidates.loc[
            ~candidates.method.isin(["random", "fps", "ca"])
        ]
        x = candidates[["target", "normalized_auc"]].copy()

    if x.empty:
        raise ValueError(f"No SA+Memory rows for M={memory}, alpha={alpha:g}")
    if not x.target.is_unique:
        raise ValueError(
            f"SA+Memory has duplicate targets for M={memory}, alpha={alpha:g}"
        )
    return x


def mean_ci(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    mean = v.mean()
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return mean, ci


def load_all(base, attempts):
    data = {}
    for a in attempts:
        root = base / f"attempts_{a}" / "independent_targets"
        data[a] = load_objects(root)
    return data


def aggregated_heatmap(data, out, memory):
    """
    Rows: attempt budgets.
    Columns: SA+Memory alpha sweep followed by SA, Random, FPS.

    This keeps M fixed and directly shows how alpha and attempt budget interact,
    while retaining the three attempt-matched baselines.
    """
    columns = [f"α={a:g}" for a in ALPHAS] + ["SA", "Random", "FPS"]
    z = np.full((len(data), len(columns)), np.nan)
    ci_mat = np.full_like(z, np.nan)
    summary_rows = []

    for i, attempts in enumerate(sorted(data)):
        obj = data[attempts]

        for j, alpha in enumerate(ALPHAS):
            tr = transfer_at(obj, memory, alpha)
            mean, ci = mean_ci(tr.normalized_auc)
            z[i, j] = mean
            ci_mat[i, j] = ci
            summary_rows.append({
                "attempts": attempts,
                "method": "SA+Memory",
                "memory": memory,
                "alpha": alpha,
                "mean_nAUC": mean,
                "ci95": ci,
                "N": len(tr),
            })

        for k, (method, label) in enumerate(
                [("ca", "SA"), ("random", "Random"), ("fps", "FPS")]
        ):
            x = baseline(obj, method)
            mean, ci = mean_ci(x.normalized_auc)
            j = len(ALPHAS) + k
            z[i, j] = mean
            ci_mat[i, j] = ci
            summary_rows.append({
                "attempts": attempts,
                "method": label,
                "memory": np.nan,
                "alpha": np.nan,
                "mean_nAUC": mean,
                "ci95": ci,
                "N": len(x),
            })

    pd.DataFrame(summary_rows).to_csv(
        out / f"attempts_heatmap_summary_M{memory}.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    im = ax.imshow(z, aspect="auto", cmap=CMAP)

    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(
        range(len(data)),
        [str(a) for a in sorted(data)],
    )
    ax.set_xlabel(r"SA + Memory transfer weight $\alpha$ / baseline")
    ax.set_ylabel("Attempts")
    ax.set_title(
        rf"Performance across attempt budgets at fixed memory $M={memory}$"
    )

    # Separate alpha sweep from baselines.
    ax.axvline(len(ALPHAS) - 0.5, color="black", lw=1.4)

    finite = z[np.isfinite(z)]
    mid = finite.min() + 0.55 * (finite.max() - finite.min())

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isfinite(z[i, j]):
                ax.text(
                    j, i,
                    f"{z[i, j]:.3f}\n±{ci_mat[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if z[i, j] > mid else "black",
                )

    fig.colorbar(im, ax=ax, label="Mean normalized AUC")
    save(fig, out, f"heatmap_alpha_vs_attempts_M{memory}")


def ranked_gain_by_attempts(data, out, memory, alpha):
    """
    Ranked per-object gain for each attempt budget:
        delta = SA+Memory - SA

    Each attempt budget is sorted independently, exactly like the existing
    ranked-per-object plot, but attempts replace memory size as the series.
    """
    series = []
    csv_rows = []

    for attempts in sorted(data):
        obj = data[attempts]
        sa = baseline(obj, "ca").rename(columns={"normalized_auc": "sa"})
        tr = transfer_at(obj, memory, alpha).rename(
            columns={"normalized_auc": "sa_memory"}
        )

        p = sa.merge(tr, on="target", how="inner", validate="one_to_one")
        p["delta"] = p.sa_memory - p.sa
        p = p.sort_values("delta").reset_index(drop=True)

        if p.empty:
            continue

        p["rank_percentile"] = (
            50.0 if len(p) == 1
            else np.arange(len(p)) / (len(p) - 1) * 100
        )
        p["attempts"] = attempts
        p["memory"] = memory
        p["alpha"] = alpha
        csv_rows.append(p)

        d = p.delta.to_numpy(float)
        rank = p.rank_percentile.to_numpy(float)
        series.append((
            attempts,
            rank,
            d,
            d.mean(),
            np.median(d),
            (d > 0).mean(),
        ))

    if not series:
        raise ValueError("No ranked-gain series could be constructed.")

    pd.concat(csv_rows, ignore_index=True).to_csv(
        out / f"ranked_per_object_gain_M{memory}_alpha{alpha:g}.csv",
        index=False,
        )

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for (attempts, x, y, mu, med, win), q in zip(
            series, np.linspace(0.18, 0.88, len(series))
    ):
        ax.plot(
            x, y,
            marker="o",
            markersize=2.8,
            lw=1.5,
            color=CMAP(q),
            label=(
                f"{attempts} attempts "
                f"(mean={mu:+.3f}, median={med:+.3f}, wins={100*win:.1f}%)"
            ),
        )

    ax.axhline(0, color="black", ls="--", lw=1.2)
    ax.set(
        xlim=(0, 100),
        xlabel="Target-object percentile (sorted separately for each attempt budget)",
        ylabel=r"$\Delta$ nAUC (SA + Memory $-$ SA)",
        title=(
            rf"Ranked per-object memory gain at $M={memory}$, "
            rf"$\alpha={alpha:g}$"
        ),
    )
    ax.grid(axis="y", alpha=0.20, lw=0.7)
    ax.legend(loc="best", fontsize=8, frameon=True)
    save(fig, out, f"ranked_gain_by_attempts_M{memory}_alpha{alpha:g}")


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate E4 plots across attempts 100/200/400"
    )
    ap.add_argument(
        "--results_dir",
        default="results/e4_independent_objects_memory_scaling_margin",
        help="Directory containing attempts_100, attempts_200, attempts_400.",
    )
    ap.add_argument(
        "--attempts",
        nargs="+",
        type=int,
        default=ATTEMPTS,
        help="Attempt budgets to aggregate (default: 100 200 400).",
    )
    ap.add_argument(
        "--memory",
        type=int,
        default=MEMORY,
        help="Fixed memory/library size (default: 200).",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.50,
        help="Fixed alpha for ranked per-object plot (default: 0.50).",
    )
    args = ap.parse_args()

    base = Path(args.results_dir)
    out = base / "aggregated_plots"
    out.mkdir(parents=True, exist_ok=True)

    data = load_all(base, args.attempts)

    aggregated_heatmap(data, out, args.memory)
    ranked_gain_by_attempts(data, out, args.memory, args.alpha)

    print("\nDone.")
    print(f"Attempts: {sorted(data)}")
    print(f"Fixed memory: M={args.memory}")
    print(f"Ranked-plot alpha: {args.alpha:g}")
    print(f"Saved to: {out}")


if __name__ == "__main__":
    main()