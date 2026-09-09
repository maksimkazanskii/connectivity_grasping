#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MS = [100, 200, 400, 800]
ALPHAS = [0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 1.00]
CMAP = LinearSegmentedColormap.from_list("gold", ["#fff1d6", "#fdae6b", "#d94801", "#7f2704"])

def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def load_objects(root):
    df = pd.read_csv(root / "results_per_seed.csv")
    req = {"target", "seed", "method", "library_size", "alpha", "normalized_auc"}
    if miss := req - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(miss)}")
    return (df.groupby(["target", "method", "library_size", "alpha"], dropna=False, as_index=False)
            .agg(normalized_auc=("normalized_auc", "mean"), n_seeds=("seed", "nunique")))

def transfer(obj):
    return obj[obj.library_size.isin(MS) & obj.alpha.astype(float).isin(ALPHAS)].copy()

def grid(obj):
    ca = obj[obj.method.eq("ca")][["target", "normalized_auc"]].rename(columns={"normalized_auc": "ca"})
    rows = []
    for (m, a), g in transfer(obj).groupby(["library_size", "alpha"]):
        p = ca.merge(g[["target", "normalized_auc"]], on="target")
        d = p.normalized_auc - p.ca
        rows.append([int(m), float(a), len(p), p.normalized_auc.mean(), d.mean(), d.median(), (d > 0).mean()])
    return pd.DataFrame(rows, columns=["M", "alpha", "N", "mean_nAUC", "mean_delta", "median_delta", "win_rate"])

def heatmap(obj, out):
    tr = transfer(obj)
    stats = tr.groupby(["library_size", "alpha"]).normalized_auc.agg(["mean", "std", "count"]).reset_index()
    stats["ci"] = 1.96 * stats["std"] / np.sqrt(stats["count"])
    means = stats.pivot(index="library_size", columns="alpha", values="mean").reindex(index=MS, columns=ALPHAS)
    cis = stats.pivot(index="library_size", columns="alpha", values="ci").reindex(index=MS, columns=ALPHAS)

    bases = []
    for key, name in [("random", "Random"), ("fps", "FPS"), ("ca", "CA")]:
        v = obj.loc[obj.method.eq(key), "normalized_auc"].dropna().to_numpy(float)
        ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        bases.append((name, v.mean(), ci))

    z = np.vstack([means.to_numpy(), *[np.full(len(ALPHAS), m) for _, m, _ in bases]])
    labels = [f"M={m}" for m in MS] + [x[0] for x in bases]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    im = ax.imshow(z, aspect="auto", cmap=CMAP)
    ax.set(xticks=range(len(ALPHAS)), yticks=range(len(labels)),
           xticklabels=[f"{x:g}" for x in ALPHAS], yticklabels=labels,
           xlabel=r"Transfer weight $\alpha$", ylabel="Method / library size",
           title="E4 connectivity-profile transfer")
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
    ca = obj[obj.method.eq("ca")][["target", "normalized_auc"]].rename(columns={"normalized_auc": "ca"})
    tr = obj[obj.library_size.eq(m) & np.isclose(obj.alpha.astype(float), a)][["target", "normalized_auc"]]
    p = ca.merge(tr.rename(columns={"normalized_auc": "e4"}), on="target")
    p["delta"] = p.e4 - p.ca
    p = p.sort_values("delta").reset_index(drop=True)

    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(7, min(10, max(4.5, .055 * len(p) + 2.5))))
    for i, r in p.iterrows():
        ax.plot([r.ca, r.e4], [i, i], lw=1, alpha=.45)
    ax.scatter(p.ca, y, s=22, label="CA", zorder=3)
    ax.scatter(p.e4, y, s=22, marker="s", label="E4", zorder=3)
    ax.set(xlabel="Normalized AUC", ylabel="Targets sorted by E4 − CA",
           title=rf"CA vs E4 ($M={m}$, $\alpha={a:g}$)")
    ax.set_yticks(y, p.target, fontsize=7) if len(p) <= 30 else ax.set_yticks([])
    ax.legend()
    ax.text(.02, .02, f"N={len(p)}\nMean Δ={p.delta.mean():+.3f}\nMedian Δ={p.delta.median():+.3f}\nWins={(p.delta > 0).mean():.1%}",
            transform=ax.transAxes, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=.85))
    p.to_csv(out / "per_object_transfer_gain.csv", index=False)
    save(fig, out, "transfer_vs_ca_per_object")
    return m, a

def ranked_gain_lines(obj, out):
    alpha = .10
    ca = obj[obj.method.eq("ca")][["target", "normalized_auc"]].rename(columns={"normalized_auc": "ca"})
    series = []
    for m in MS:
        tr = obj[obj.library_size.eq(m) & np.isclose(obj.alpha.astype(float), alpha)][["target", "normalized_auc"]]
        p = ca.merge(tr, on="target")
        d = np.sort((p.normalized_auc - p.ca).to_numpy(float))
        if not len(d): continue
        rank = np.array([50.]) if len(d) == 1 else np.arange(len(d)) / (len(d) - 1) * 100
        series.append((m, rank, d, d.mean(), (d > 0).mean()))

    if not series: return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for (m, x, y, mu, w), q in zip(series, np.linspace(.18, .88, len(series))):
        ax.plot(x, y, marker="o", markersize=2.8, lw=1.5, color=CMAP(q),
                label=f"M={m}  (mean Δ={mu:+.3f}, wins={100*w:.1f}%)")
    ax.axhline(0, color="black", ls="--", lw=1.2)
    ax.set(xlim=(0, 100), xlabel="Target-object percentile (sorted separately for each memory size)",
           ylabel=r"$\Delta$ nAUC (E4 $-$ CA)",
           title=rf"Ranked per-object transfer gain at $\alpha={alpha:g}$")
    ax.grid(axis="y", alpha=.20, lw=.7)
    ax.legend(loc="best", fontsize=8, frameon=True)
    save(fig, out, "alpha01_ranked_gain_by_memory")

def violin_plots(obj, out):
    blue, gold = "steelblue", "#E87511"
    rng = np.random.default_rng(12345)
    groups, labels, kinds = [], [], []

    for key, label in [("random", "Random"), ("fps", "FPS"), ("ca", "CA")]:
        v = obj.loc[obj.method.eq(key), "normalized_auc"].dropna().to_numpy(float)
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
    vp = ax.violinplot(groups, positions=pos, showmeans=False, showmedians=False, showextrema=False, widths=.78)

    for body, kind in zip(vp["bodies"], kinds):
        c = blue if kind == "base" else gold
        body.set_facecolor(c); body.set_alpha(.10)
        body.set_edgecolor(c); body.set_linewidth(1.8)
    for x, v, kind in zip(pos, groups, kinds):
        c = blue if kind == "base" else gold
        ax.scatter(x + rng.uniform(-.12, .12, len(v)), v, s=14, color=c, alpha=.14, edgecolors="none", zorder=2)
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
    ap.add_argument("--results_dir", default="results/e4_transfer_results")
    args = ap.parse_args()
    root, out = Path(args.results_dir), Path(args.results_dir) / "plots"
    out.mkdir(parents=True, exist_ok=True)

    obj = load_objects(root)
    s = grid(obj)
    s.to_csv(out / "transfer_grid_summary.csv", index=False)
    heatmap(obj, out)
    best_fixed = s.sort_values("mean_delta", ascending=False).iloc[0]
    m, a = int(best_fixed.M), float(best_fixed.alpha)
    ranked_gain_lines(obj, out)
    violin_plots(obj, out)

    print(f"\nTargets: {obj.target.nunique()} | best fixed config: M={m}, alpha={a:g}")
    print(s.sort_values("mean_delta", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()