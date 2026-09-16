#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MS = [200, 400, 600, 800]
ALPHAS = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00]
CMAP = LinearSegmentedColormap.from_list("gold", ["#fff1d6", "#fdae6b", "#d94801", "#7f2704"])

def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def load_objects(root):
    """Load seed-level results and reduce to one row per target/configuration.

    Baselines MUST reduce to exactly one row per target.  We validate this
    explicitly because all paired CA-vs-Random/FPS and transfer-vs-CA
    comparisons assume one-to-one target matching.
    """
    # Support both layouts:
    #   legacy/sequential: root/results_per_seed.csv
    #   seed-parallel:     root/seed_1/results_per_seed.csv, ..., root/seed_N/results_per_seed.csv
    # Prefer the new seed-parallel layout whenever seed directories exist.
    # This avoids accidentally reading a stale legacy root/results_per_seed.csv.
    seed_files = list(root.glob("seed_*/results_per_seed.csv"))
    seed_files.sort(
        key=lambda p: int(p.parent.name.split("_", 1)[1])
    )

    direct_file = root / "results_per_seed.csv"

    if seed_files:
        result_files = seed_files
    elif direct_file.exists():
        result_files = [direct_file]
    else:
        raise FileNotFoundError(
            f"No results_per_seed.csv found directly under {root} "
            f"or in {root}/seed_*/"
        )

    print("\nLoading seed result files:")
    for f in result_files:
        print(f"  {f}")

    raw = pd.concat(
        [pd.read_csv(f) for f in result_files],
        ignore_index=True,
    )

    print(
        f"Loaded {len(result_files)} result file(s), "
        f"{len(raw)} rows, "
        f"{raw['target'].nunique() if 'target' in raw.columns else 0} unique targets, "
        f"seeds={sorted(raw['seed'].dropna().astype(int).unique().tolist()) if 'seed' in raw.columns else []}"
    )

    req = {"target", "seed", "method", "library_size", "alpha", "normalized_auc"}
    if miss := req - set(raw.columns):
        raise ValueError(f"Missing columns: {sorted(miss)}")

    obj = (
        raw.groupby(
            ["target", "method", "library_size", "alpha"],
            dropna=False,
            as_index=False,
        )
        .agg(
            normalized_auc=("normalized_auc", "mean"),
            n_seeds=("seed", "nunique"),
            n_rows=("seed", "size"),
        )
    )

    # ---- Baseline integrity checks -----------------------------------------
    print("\nBaseline integrity after seed averaging:")
    expected_targets = obj["target"].nunique()
    for method in ["random", "fps", "ca"]:
        x = obj.loc[obj.method.eq(method)].copy()
        n_rows = len(x)
        n_targets = x.target.nunique()
        n_dups = n_rows - n_targets
        print(
            f"  {method:6s}: rows={n_rows:4d}, targets={n_targets:4d}, "
            f"duplicate target rows={n_dups:4d}"
        )
        if n_rows == 0:
            raise ValueError(f"No baseline rows found for method={method!r}.")
        if not x.target.is_unique:
            dup = x.loc[x.target.duplicated(False)].sort_values("target")
            cols = ["target", "method", "library_size", "alpha",
                    "normalized_auc", "n_seeds", "n_rows"]
            raise ValueError(
                f"Baseline {method!r} does not have exactly one configuration "
                f"per target after seed averaging.\n"
                f"Duplicate rows:\n{dup[cols].to_string(index=False)}"
            )
        if n_targets != expected_targets:
            missing = sorted(set(obj.target.astype(str)) - set(x.target.astype(str)))
            raise ValueError(
                f"Baseline {method!r} covers {n_targets}/{expected_targets} targets. "
                f"Missing examples: {missing[:10]}"
            )

    return obj

def transfer(obj):
    return obj[obj.library_size.isin(MS) &
               np.logical_or.reduce([np.isclose(obj.alpha.astype(float), a) for a in ALPHAS])].copy()

def grid(obj):
    ca = unique_ca(obj)
    rows = []
    for (m, a), g in transfer(obj).groupby(["library_size", "alpha"]):
        p = one_to_one_transfer_pair(ca, g, context=f"M={m}, alpha={a}")
        d = p.normalized_auc - p.ca
        rows.append([
            int(m), float(a), len(p), p.normalized_auc.mean(),
            d.mean(), d.median(), (d > 0).mean()
        ])
    return pd.DataFrame(
        rows,
        columns=["M", "alpha", "N", "mean_nAUC",
                 "mean_delta", "median_delta", "win_rate"],
    )

def baseline_frame(obj):
    """Return exactly one target-level value for each E3-style baseline."""
    frames = []
    for method in ["random", "fps", "ca"]:
        x = obj.loc[obj.method.eq(method), ["target", "normalized_auc"]].copy()
        if not x.target.is_unique:
            raise ValueError(f"{method} baseline has duplicate target rows.")
        frames.append(x.rename(columns={"normalized_auc": method}))

    p = frames[0]
    p = p.merge(frames[1], on="target", how="inner", validate="one_to_one")
    p = p.merge(frames[2], on="target", how="inner", validate="one_to_one")
    return p


def baseline_diagnostics(obj, out):
    """Print and save paired target-level baseline diagnostics."""
    p = baseline_frame(obj)
    p["ca_minus_random"] = p["ca"] - p["random"]
    p["fps_minus_random"] = p["fps"] - p["random"]
    p["ca_minus_fps"] = p["ca"] - p["fps"]

    print("\\nPaired baseline diagnostics (one value per target):")
    print(f"  N              = {len(p)}")
    print(f"  Random mean    = {p.random.mean():.6f}")
    print(f"  FPS mean       = {p.fps.mean():.6f}")
    print(f"  CA mean        = {p.ca.mean():.6f}")
    print(f"  CA - Random    = {p.ca_minus_random.mean():+.6f}")
    print(f"  median delta   = {p.ca_minus_random.median():+.6f}")
    print(f"  CA win rate    = {(p.ca_minus_random > 0).mean():.1%}")
    if len(p) > 1:
        sd = p.ca_minus_random.std(ddof=1)
        se = sd / np.sqrt(len(p))
        ci = 1.96 * se
        print(f"  SD(delta)      = {sd:.6f}")
        print(f"  SE(delta)      = {se:.6f}")
        print(f"  95% CI delta   = [{p.ca_minus_random.mean()-ci:+.6f}, "
              f"{p.ca_minus_random.mean()+ci:+.6f}]")

    p.to_csv(out / "paired_baseline_diagnostics.csv", index=False)
    return p


def unique_ca(obj):
    ca = obj.loc[obj.method.eq("ca"), ["target", "normalized_auc"]].copy()
    if not ca.target.is_unique:
        raise ValueError("CA baseline contains duplicate target rows.")
    return ca.rename(columns={"normalized_auc": "ca"})


def one_to_one_transfer_pair(ca, g, context="transfer"):
    tr = g[["target", "normalized_auc"]].copy()
    if not tr.target.is_unique:
        dup = tr.loc[tr.target.duplicated(False)].sort_values("target")
        raise ValueError(
            f"{context} contains duplicate target rows:\\n"
            f"{dup.to_string(index=False)}"
        )
    return ca.merge(tr, on="target", validate="one_to_one")

def heatmap(obj, out):
    tr = transfer(obj)
    stats = tr.groupby(["library_size", "alpha"]).normalized_auc.agg(["mean", "std", "count"]).reset_index()
    stats["ci"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
    means = stats.pivot(index="library_size", columns="alpha", values="mean").reindex(index=MS, columns=ALPHAS)
    cis = stats.pivot(index="library_size", columns="alpha", values="ci").reindex(index=MS, columns=ALPHAS)

    bases = []
    bp = baseline_frame(obj)
    for key, name in [("random", "Random"), ("fps", "FPS"), ("ca", "CA")]:
        v = bp[key].dropna().to_numpy(float)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        bases.append((name, v.mean(), ci))

    z = np.vstack([means.to_numpy(), *[np.full(len(ALPHAS), m) for _, m, _ in bases]])
    labels = [f"M={m}" for m in MS] + [x[0] for x in bases]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    im = ax.imshow(z, aspect="auto", cmap=CMAP)
    ax.set(xticks=range(len(ALPHAS)), yticks=range(len(labels)),
           xticklabels=[f"{x:g}" for x in ALPHAS], yticklabels=labels,
           xlabel=r"Transfer weight $\alpha$", ylabel="Method / library size",
           title="E4 cross-object viability-history transfer")
    ax.axhline(len(MS) - .5, color="black", lw=1.4)
    mid = np.nanmin(z) + .55 * (np.nanmax(z) - np.nanmin(z))

    for i in range(len(MS)):
        for j in range(len(ALPHAS)):
            m, ci = means.iloc[i, j], cis.iloc[i, j]
            if np.isfinite(m):
                ax.text(j, i, f"{m:.3f}\n±{ci:.3f}", ha="center", va="center",
                        fontsize=8, color="white" if m > mid else "black")
    for k, (_, m, ci) in enumerate(bases):
        ax.text((len(ALPHAS) - 1) / 2, len(MS) + k, f"{m:.3f} ± {ci:.3f}",
                ha="center", va="center", color="white" if m > mid else "black")

    fig.colorbar(im, ax=ax, label="Mean normalized AUC")
    save(fig, out, "heatmap_transfer_nauc")

def gain_plot(obj, summary, out):
    best = summary.sort_values("mean_delta", ascending=False).iloc[0]
    m, a = int(best.M), float(best.alpha)
    ca = unique_ca(obj)
    tr = obj[
        obj.library_size.eq(m) & np.isclose(obj.alpha.astype(float), a)
        ][["target", "normalized_auc"]].copy()
    p = one_to_one_transfer_pair(ca, tr, context=f"best M={m}, alpha={a}")
    p = p.rename(columns={"normalized_auc": "e4"})
    p["delta"] = p.e4 - p.ca
    p = p.sort_values("delta").reset_index(drop=True)

    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(7, min(10, max(4.5, .055 * len(p) + 2.5))))
    for i, r in p.iterrows():
        ax.plot([r.ca, r.e4], [i, i], lw=1, alpha=.45)
    ax.scatter(p.ca, y, s=22, label="CA", zorder=3)
    ax.scatter(p.e4, y, s=22, marker="s", label="CA + Memory", zorder=3)
    ax.set(xlabel="Normalized AUC", ylabel="Targets sorted by CA + Memory − CA",
           title=rf"CA vs CA + Memory ($M={m}$, $\alpha={a:g}$)")
    ax.set_yticks(y, p.target, fontsize=7) if len(p) <= 30 else ax.set_yticks([])
    ax.legend()
    ax.text(.02, .02, f"N={len(p)}\nMean Δ={p.delta.mean():+.3f}\n"
                      f"Median Δ={p.delta.median():+.3f}\nWins={(p.delta > 0).mean():.1%}",
            transform=ax.transAxes, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=.85))
    p.to_csv(out / "per_object_transfer_gain.csv", index=False)
    save(fig, out, "transfer_vs_ca_per_object")
    return m, a

def ranked_gain_lines(obj, out):
    alpha = .50
    ca = unique_ca(obj)
    series = []
    for m in MS:
        tr = obj[
            obj.library_size.eq(m) & np.isclose(obj.alpha.astype(float), alpha)
            ][["target", "normalized_auc"]].copy()
        p = one_to_one_transfer_pair(ca, tr, context=f"M={m}, alpha={alpha}")
        d = np.sort((p.normalized_auc - p.ca).to_numpy(float))
        if not len(d): continue
        rank = np.array([50.]) if len(d) == 1 else np.arange(len(d)) / (len(d) - 1) * 100
        series.append((m, rank, d, d.mean(), d.mean() if len(d) == 1 else np.median(d), (d > 0).mean()))

    if not series: return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for (m, x, y, mu, med, w), q in zip(series, np.linspace(.18, .88, len(series))):
        ax.plot(x, y, marker="o", markersize=2.8, lw=1.5, color=CMAP(q),
                label=f"M={m} (mean={mu:+.3f}, median={med:+.3f}, wins={100*w:.1f}%)")
    ax.axhline(0, color="black", ls="--", lw=1.2)
    ax.set(xlim=(0, 100), xlabel="Target-object percentile (sorted separately for each memory size)",
           ylabel=r"$\Delta$ nAUC (CA + Memory $-$ CA)",
           title=rf"Ranked per-object transfer gain at fixed $\alpha={alpha:g}$")
    ax.grid(axis="y", alpha=.20, lw=.7)
    ax.legend(loc="best", fontsize=8, frameon=True)
    save(fig, out, "alpha05_ranked_gain_by_memory")

def violin_plots(obj, out):
    blue, gold = "steelblue", "#E87511"
    rng = np.random.default_rng(12345)
    groups, labels, kinds = [], [], []

    bp = baseline_frame(obj)
    for key, label in [("random", "Random"), ("fps", "FPS"), ("ca", "CA")]:
        v = bp[key].dropna().to_numpy(float)
        if len(v):
            groups.append(v); labels.append(label); kinds.append("base")

    tr = transfer(obj)
    best = (tr.groupby(["library_size", "alpha"]).normalized_auc.mean().reset_index(name="mean_nAUC")
            .sort_values(["library_size", "mean_nAUC"], ascending=[True, False])
            .groupby("library_size", as_index=False).first())
    best.to_csv(out / "best_alpha_per_memory.csv", index=False)
    print("\nBest alpha independently for each memory size:")
    print(best.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    for m in MS:
        row = best[best.library_size.eq(m)]
        if row.empty: continue
        a = float(row.iloc[0].alpha)
        v = obj.loc[obj.library_size.eq(m) & np.isclose(obj.alpha.astype(float), a),
        "normalized_auc"].dropna().to_numpy(float)
        if len(v):
            groups.append(v); labels.append(f"E4\nM={m}\n" + rf"$\alpha$={a:g}"); kinds.append("memory")

    fig, ax = plt.subplots(figsize=(max(8.4, 1.05 * len(groups)), 4.9))
    pos = np.arange(1, len(groups) + 1)
    vp = ax.violinplot(groups, positions=pos, showmeans=False, showmedians=False,
                       showextrema=False, widths=.78)

    for body, kind in zip(vp["bodies"], kinds):
        c = blue if kind == "base" else gold
        body.set_facecolor(c); body.set_alpha(.10)
        body.set_edgecolor(c); body.set_linewidth(1.8)
    for x, v, kind in zip(pos, groups, kinds):
        c = blue if kind == "base" else gold
        ax.scatter(x + rng.uniform(-.12, .12, len(v)), v, s=14, color=c,
                   alpha=.14, edgecolors="none", zorder=2)
        mu, sd = v.mean(), v.std(ddof=1) if len(v) > 1 else 0
        ax.vlines(x, mu - sd, mu + sd, color=c, lw=3, zorder=3)
        ax.scatter(x, mu, s=70, facecolor="white", edgecolor=c, lw=1.2, zorder=4)

    ax.set_xticks(pos, labels)
    ax.set(ylabel="Normalized AUC", xlabel="Method",
           title=r"Object-level performance at best $\alpha$ for each memory size")
    ax.grid(axis="y", ls="--", alpha=.28, lw=.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    save(fig, out, "best_alpha_per_memory_violin")

def main():
    ap = argparse.ArgumentParser(description="Compact E4 plotting")
    ap.add_argument("--results_dir",
                    default="results/e4_independent_objects_memory_scaling_margin")
    ap.add_argument("--attempts", type=int, default=400,
                    help="Attempt-count folder to plot, e.g. 50 -> attempts_50.")
    args = ap.parse_args()

    base = Path(args.results_dir)
    if base.name == "independent_targets":
        root = base
    elif base.name.startswith("attempts_"):
        root = base / "independent_targets"
    else:
        root = base / f"attempts_{args.attempts}" / "independent_targets"

    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)

    obj = load_objects(root)
    baseline_diagnostics(obj, out)
    s = grid(obj)
    s.to_csv(out / "transfer_grid_summary.csv", index=False)
    heatmap(obj, out)
    m, a = gain_plot(obj, s, out)
    ranked_gain_lines(obj, out)
    violin_plots(obj, out)

    print(f"\nTargets: {obj.target.nunique()} | best fixed config: M={m}, alpha={a:g}")
    print(s.sort_values("mean_delta", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()