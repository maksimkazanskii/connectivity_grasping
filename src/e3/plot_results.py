from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


POOL_CONFIG = {
    200: 50,
    400: 100,
    800: 200,
    1600: 400,
}

METHODS = {
    "fps": {
        "label": "FPS",
        "auc": "fps_exploitation_auc",
    },
    "connectivity": {
        "label": "Connectivity-aware",
        "auc": "connectivity_exploitation_auc",
    },
}

RANDOM_AUC = "random_exploitation_auc"


# ============================================================
# Statistics
# ============================================================

def object_gain(method_auc, random_auc):
    """
    Object-level percentage AUC gain relative to Random.
    """
    method_auc = np.asarray(method_auc, dtype=float)
    random_auc = np.asarray(random_auc, dtype=float)

    gain = np.full(len(method_auc), np.nan)

    valid = (
            np.isfinite(method_auc)
            & np.isfinite(random_auc)
            & (random_auc != 0)
    )

    gain[valid] = (
            100.0
            * (method_auc[valid] - random_auc[valid])
            / random_auc[valid]
    )

    return gain


def plot_policy_correlation(
        results,
        output_dir,
        n=1600,
):
    """
    Object-level relationship between FPS and CA gains over Random.

    Each point is one object after seed averaging.
    """

    from scipy.stats import pearsonr, spearmanr

    df = results[n]

    random_auc = np.asarray(
        df[RANDOM_AUC],
        dtype=float,
    )

    fps_gain = object_gain(
        df[METHODS["fps"]["auc"]],
        random_auc,
    )

    ca_gain = object_gain(
        df[METHODS["connectivity"]["auc"]],
        random_auc,
    )

    valid = (
            np.isfinite(fps_gain)
            & np.isfinite(ca_gain)
    )

    x = fps_gain[valid]
    y = ca_gain[valid]

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)

    # --------------------------------------------------------
    # Quadrant counts
    # --------------------------------------------------------

    both_win = np.sum((x > 0) & (y > 0))
    fps_only = np.sum((x > 0) & (y <= 0))
    ca_only = np.sum((x <= 0) & (y > 0))
    neither = np.sum((x <= 0) & (y <= 0))

    total = len(x)

    print()
    print("=" * 72)
    print(f"POLICY CORRELATION: N={n}")
    print("=" * 72)

    print(
        f"Pearson r   = {pearson_r:+.3f}, "
        f"p = {pearson_p:.3g}"
    )
    print(
        f"Spearman rho = {spearman_rho:+.3f}, "
        f"p = {spearman_p:.3g}"
    )

    print()
    print(
        f"Both win : {both_win:3d}/{total} "
        f"({100 * both_win / total:.1f}%)"
    )
    print(
        f"FPS only : {fps_only:3d}/{total} "
        f"({100 * fps_only / total:.1f}%)"
    )
    print(
        f"CA only  : {ca_only:3d}/{total} "
        f"({100 * ca_only / total:.1f}%)"
    )
    print(
        f"Neither  : {neither:3d}/{total} "
        f"({100 * neither / total:.1f}%)"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(5.8, 5.2)
    )

    ax.scatter(
        x,
        y,
        s=24,
        alpha=0.55,
        edgecolors="none",
        zorder=3,
    )

    # Random boundaries.
    ax.axvline(
        0,
        color="0.4",
        linewidth=1.1,
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )

    ax.axhline(
        0,
        color="0.4",
        linewidth=1.1,
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )

    # --------------------------------------------------------
    # Correlation annotation
    # --------------------------------------------------------

    annotation = (
        rf"$r={pearson_r:.2f}$"
        "\n"
        rf"$\rho={spearman_rho:.2f}$"
    )

    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    ax.set_xlabel(
        "FPS Phase-2 AUC gain over Random (%)"
    )

    ax.set_ylabel(
        "CA Phase-2 AUC gain over Random (%)"
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
        zorder=0,
    )

    fig.tight_layout()

    pdf_path = (
            output_dir
            / f"e3_policy_correlation_N{n}.pdf"
    )

    png_path = (
            output_dir
            / f"e3_policy_correlation_N{n}.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
def bootstrap_mean_ci(
        x,
        n_boot=10000,
        seed=12345,
):
    """
    Percentile bootstrap 95% CI for the mean across objects.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(
            x,
            size=len(x),
            replace=True,
        )
        bootstrap_means[i] = sample.mean()

    mean = x.mean()

    low, high = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    return mean, low, high


# ============================================================
# Load results
# ============================================================

def load_results(base_dir):
    """
    Load object-level summaries for all candidate-set sizes.
    """
    results = {}

    for n in POOL_CONFIG:

        path = (
                base_dir
                / f"pool_{n}"
                / "object_summary.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Could not find:\n{path}"
            )

        df = pd.read_csv(path)

        required = [RANDOM_AUC]

        for config in METHODS.values():
            required.append(config["auc"])

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns in {path}:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in missing
                )
            )

        results[n] = df

        print(
            f"N={n}: {len(df)} objects"
        )

    return results


# ============================================================
# Figure 1
# Scaling of Phase-2 AUC gain
# ============================================================

def plot_scaling(
        results,
        output_dir,
):

    fig, ax = plt.subplots(
        figsize=(6.5, 4.3)
    )

    pool_sizes = list(
        POOL_CONFIG.keys()
    )

    # Random reference line.
    ax.axhline(
        0,
        linewidth=1.3,
        linestyle="--",
        color="0.4",
        alpha=0.7,
        label="Random",
        zorder=2,
    )

    print()
    print("=" * 72)
    print("SCALING PLOT")
    print("=" * 72)

    for method_index, (method, config) in enumerate(
            METHODS.items()
    ):

        means = []
        lows = []
        highs = []

        for n in pool_sizes:

            df = results[n]

            gain = object_gain(
                df[config["auc"]],
                df[RANDOM_AUC],
            )

            mean, low, high = bootstrap_mean_ci(
                gain,
                n_boot=10000,
                seed=1000 + n + method_index,
            )

            means.append(mean)
            lows.append(low)
            highs.append(high)

            print(
                f"{config['label']:20s} | "
                f"N={n:4d} | "
                f"{mean:+.2f}% "
                f"[{low:+.2f}, {high:+.2f}]"
            )

        means = np.asarray(means)
        lows = np.asarray(lows)
        highs = np.asarray(highs)

        yerr = np.vstack(
            [
                means - lows,
                highs - means,
                ]
        )

        ax.errorbar(
            pool_sizes,
            means,
            yerr=yerr,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            capsize=3,
            alpha=0.6,
            label=config["label"],
            zorder=3,
        )

    # Candidate-set size doubles at each step.
    ax.set_xscale(
        "log",
        base=2,
    )

    ax.set_xticks(pool_sizes)

    ax.set_xticklabels(
        [str(n) for n in pool_sizes]
    )

    ax.set_xlabel(
        "Total candidate-set size $N$"
    )

    ax.set_ylabel(
        "Phase-2 AUC gain over Random (%)"
    )

    # Dashed grid.
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
        zorder=0,
    )

    ax.legend(
        frameon=False
    )

    fig.tight_layout()

    pdf_path = (
            output_dir
            / "e3_gain_scaling.pdf"
    )

    png_path = (
            output_dir
            / "e3_gain_scaling.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# ============================================================
# Figure 2
# Distribution of object-level Phase-2 AUC gains
# ============================================================

def plot_gain_distribution(
        results,
        output_dir,
        n=1600,
):

    df = results[n]

    random_auc = np.asarray(
        df[RANDOM_AUC],
        dtype=float,
    )

    gains = []
    labels = []

    for method, config in METHODS.items():

        gain = object_gain(
            df[config["auc"]],
            random_auc,
        )

        gain = gain[
            np.isfinite(gain)
        ]

        gains.append(gain)
        labels.append(
            config["label"]
        )

    fig, ax = plt.subplots(
        figsize=(6.2, 4.3)
    )

    positions = np.arange(
        1,
        len(gains) + 1,
        )

    # Default Matplotlib colors.
    colors = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    method_colors = colors[
        :len(gains)
    ]

    # --------------------------------------------------------
    # Violin distributions
    # --------------------------------------------------------

    parts = ax.violinplot(
        gains,
        positions=positions,
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, color in zip(
            parts["bodies"],
            method_colors,
    ):

        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(1.0)

    # Reproducible horizontal jitter.
    rng = np.random.default_rng(
        12345
    )

    print()
    print("=" * 72)
    print(
        f"GAIN DISTRIBUTION: N={n}"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Individual objects + IQR + median
    # --------------------------------------------------------

    for i, (
            gain,
            label,
            color,
    ) in enumerate(
        zip(
            gains,
            labels,
            method_colors,
        ),
        start=1,
    ):

        q25, median, q75 = np.percentile(
            gain,
            [25, 50, 75],
        )

        mean = gain.mean()

        # Individual object-level observations.
        jitter = rng.normal(
            loc=0.0,
            scale=0.035,
            size=len(gain),
        )

        ax.scatter(
            i + jitter,
            gain,
            s=9,
            alpha=0.3,
            color=color,
            edgecolors="none",
            zorder=2,
            )

        # Thick colored IQR line.
        ax.vlines(
            i,
            q25,
            q75,
            color=color,
            linewidth=3,
            zorder=3,
        )

        # Colored median point.
        ax.scatter(
            i,
            median,
            s=45,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )

        print(
            f"{label:20s} | "
            f"mean={mean:+.2f}% | "
            f"median={median:+.2f}% | "
            f"IQR=[{q25:+.2f}, {q75:+.2f}]"
        )

    # Random = zero gain.
    ax.axhline(
        0,
        color="0.4",
        linewidth=1.2,
        linestyle="--",
        alpha=0.7,
        zorder=1,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Phase-2 AUC gain over Random (%)"
    )

    ax.set_xlabel(
        "Exploration policy"
    )

    # Dashed grid.
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.35,
        zorder=0,
    )

    fig.tight_layout()

    pdf_path = (
            output_dir
            / f"e3_gain_distribution_N{n}.pdf"
    )

    png_path = (
            output_dir
            / f"e3_gain_distribution_N{n}.png"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print()
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "base_results_dir",
        nargs="?",
        default=(
            "results/"
            "e3_disjoint_pools_results"
        ),
        help=(
            "Directory containing pool_200, "
            "pool_400, pool_800, and pool_1600"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for generated figures. "
            "Default: <base_results_dir>/figures"
        ),
    )

    parser.add_argument(
        "--distribution_pool",
        type=int,
        default=1600,
        choices=list(
            POOL_CONFIG.keys()
        ),
        help=(
            "Candidate-set size used for "
            "the gain-distribution plot."
        ),
    )

    args = parser.parse_args()

    base_dir = Path(
        args.base_results_dir
    )

    if args.output_dir is None:
        output_dir = (
                base_dir
                / "figures"
        )
    else:
        output_dir = Path(
            args.output_dir
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = load_results(
        base_dir
    )

    # Figure 1
    plot_scaling(
        results,
        output_dir,
    )

    # Figure 2
    plot_gain_distribution(
        results,
        output_dir,
        n=args.distribution_pool,
    )
    plot_policy_correlation(
        results,
        output_dir,
        n=args.distribution_pool,
    )

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)

    print(
        f"Figures saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()