from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DIR = Path("results/e1/convergence")
INPUT = DIR / "e1_within_between.csv"
CONVERGENCE = DIR / "e1_convergence.csv"
BOOTSTRAPS = 10_000

SIZES = [50, 100, 300]
SEED = 42

ORANGE = "#E67E22"



def bootstrap_ci(x):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x)
    values = np.empty(BOOTSTRAPS)

    for i in range(BOOTSTRAPS):
        sample = rng.choice(x, size=len(x), replace=True)
        values[i] = np.mean(sample)

    return np.percentile(values, [2.5, 97.5])


def plot_convergence():
    df = pd.read_csv(CONVERGENCE)
    rows = []

    for n, group in df.groupby("N"):
        x = group["mean_distance"].to_numpy()
        lo, hi = bootstrap_ci(x)
        rows.append({
            "N": n,
            "mean": x.mean(),
            "median": np.median(x),
            "ci_low": lo,
            "ci_high": hi,
        })

    stats = pd.DataFrame(rows).sort_values("N")
    stats.to_csv(DIR / "e1_convergence_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    ax.plot(
        stats["N"], stats["mean"],
        marker="o", linewidth=1.4, markersize=5,
        color=ORANGE,
    )
    ax.fill_between(
        stats["N"], stats["ci_low"], stats["ci_high"],
        color=ORANGE, alpha=0.15, linewidth=0,
    )

    ax.set_xlabel("Number of successful grasps $N$")
    ax.set_ylabel("Within-object structural distance $E(N)$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(DIR / "e1_convergence.pdf", bbox_inches="tight")
    fig.savefig(DIR / "e1_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {DIR / 'e1_convergence.pdf'}")
    print(f"Saved: {DIR / 'e1_convergence.png'}")



def plot_within_between():
    df = pd.read_csv(INPUT)

    if "N" not in df.columns:
        raise ValueError(
            "e1_within_between.csv must contain an 'N' column. "
            "Re-run convergence.py after adding multiple-N "
            "within/between comparisons."
        )

    rng = np.random.default_rng(SEED)

    # Two positions per N: within, between.
    positions = {
        50: (1.0, 1.7),
        100: (3.0, 3.7),
        300: (5.0, 5.7),
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    for n in SIZES:
        sub = df[df["N"] == n]

        within = sub.loc[
            sub["type"] == "Within object",
            "distance",
        ].to_numpy()

        between = sub.loc[
            sub["type"] == "Between objects",
            "distance",
        ].to_numpy()

        if len(within) == 0 or len(between) == 0:
            raise ValueError(
                f"Missing within/between data for N={n}"
            )

        pos_within, pos_between = positions[n]

        # -----------------------------------------------------
        # Violin outlines
        # -----------------------------------------------------
        vp = ax.violinplot(
            [within, between],
            positions=[pos_within, pos_between],
            widths=0.58,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )

        for body in vp["bodies"]:
            body.set_facecolor("none")
            body.set_edgecolor(ORANGE)
            body.set_linewidth(1.4)
            body.set_alpha(1.0)
            body.set_zorder(1)

        # -----------------------------------------------------
        # Raw observations
        # -----------------------------------------------------

        # All within-object values.
        jitter = rng.uniform(
            -0.12,
            0.12,
            size=len(within),
        )

        ax.scatter(
            pos_within + jitter,
            within,
            s=5,
            color=ORANGE,
            alpha=0.10,
            edgecolors="none",
            rasterized=True,
            zorder=2,
            )

        # Display only a random subset of between-object
        # observations to avoid an opaque cloud.
        max_between_points = 1500

        if len(between) > max_between_points:
            idx = rng.choice(
                len(between),
                size=max_between_points,
                replace=False,
            )
            between_display = between[idx]
        else:
            between_display = between

        jitter = rng.uniform(
            -0.12,
            0.12,
            size=len(between_display),
        )

        ax.scatter(
            pos_between + jitter,
            between_display,
            s=5,
            color=ORANGE,
            alpha=0.06,
            edgecolors="none",
            rasterized=True,
            zorder=2,
            )

        # -----------------------------------------------------
        # IQR + median
        # -----------------------------------------------------
        for pos, values in [
            (pos_within, within),
            (pos_between, between),
        ]:
            q1, median, q3 = np.percentile(
                values,
                [25, 50, 75],
            )

            # IQR
            ax.vlines(
                pos,
                q1,
                q3,
                color=ORANGE,
                linewidth=2.0,
                zorder=4,
            )

            # Median
            ax.scatter(
                pos,
                median,
                s=28,
                color=ORANGE,
                edgecolors="white",
                linewidth=0.7,
                zorder=5,
            )

        print(
            f"N={n:3d} | "
            f"within median={np.median(within):.4f} | "
            f"between median={np.median(between):.4f} | "
            f"ratio={np.median(between) / np.median(within):.2f}x"
        )

    # ---------------------------------------------------------
    # Axis labels
    # ---------------------------------------------------------

    xticks = []
    xlabels = []

    for n in SIZES:
        w, b = positions[n]

        xticks.extend([w, b])
        xlabels.extend(["Within", "Between"])

        # N label centered underneath each pair.
        center = (w + b) / 2

        ax.text(
            center,
            -0.16,
            f"$N={n}$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
        )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    ax.set_ylabel("Structural distance $D$")

    ax.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Extra bottom margin for N labels.
    fig.subplots_adjust(bottom=0.23)

    output_pdf = DIR / "e1_within_between_by_N.pdf"
    output_png = DIR / "e1_within_between_by_N.png"

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\nSaved: {output_pdf}")
    print(f"Saved: {output_png}")

def main():
    plot_convergence()
    plot_within_between()


if __name__ == "__main__":
    main()