# e2/plot_reconstruction_efficiency.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = Path("results") / "e2_results"

MATCHES_PATH = (
        RESULTS_DIR
        / "online_connectivity_matches.csv"
)

OUTPUT_PDF = (
        RESULTS_DIR
        / "e2_reconstruction_efficiency.pdf"
)

OUTPUT_PNG = (
        RESULTS_DIR
        / "e2_reconstruction_efficiency.png"
)


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(MATCHES_PATH)

targets = np.array(
    sorted(df["random_target"].unique())
)


# ============================================================
# SUMMARIZE OBJECT-LEVEL RESULTS
# ============================================================

rows = []

for target in targets:

    x = df[
        df["random_target"] == target
        ]

    reached = x[
        "connectivity_points"
    ].notna()

    points = x.loc[
        reached,
        "connectivity_points",
    ]

    # IMPORTANT:
    # Compute speedup object-by-object first,
    # then take the median.
    speedup = (
            target / points
    )

    rows.append(
        {
            "target": target,
            "n_objects": len(x),
            "n_reached": reached.sum(),
            "reach_percent": 100 * reached.mean(),
            "median_points": points.median(),
            "median_speedup": speedup.median(),
        }
    )

summary = pd.DataFrame(rows)

print(summary.to_string(index=False))


# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.2, 3.0),
)

x = np.arange(len(summary))


# ------------------------------------------------------------
# A: observations required
# ------------------------------------------------------------

ax = axes[0]

width = 0.34

ax.bar(
    x - width / 2,
    summary["target"],
    width=width,
    label="Random reference",
    )

ax.bar(
    x + width / 2,
    summary["median_points"],
    width=width,
    label="Connectivity-aware",
    )

ax.set_xticks(x)

ax.set_xticklabels(
    [
        f"Random@{int(t)}"
        for t in summary["target"]
    ]
)

ax.set_ylabel(
    "Observations"
)

ax.set_title(
    "(a) Reconstruction sample efficiency"
)

ax.legend(
    frameon=False,
    fontsize=8,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    alpha=0.2,
)

# Annotate CA medians
for i, row in summary.iterrows():

    ax.text(
        i + width / 2,
        row["median_points"] + 4,
        f'{row["median_points"]:.0f}',
        ha="center",
        va="bottom",
        fontsize=8,
        )


# ------------------------------------------------------------
# B: object-level speedup
# ------------------------------------------------------------

ax = axes[1]

bars = ax.bar(
    x,
    summary["median_speedup"],
    width=0.55,
)

ax.axhline(
    1.0,
    linewidth=1,
    linestyle="--",
)

ax.set_xticks(x)

ax.set_xticklabels(
    [
        f"Random@{int(t)}"
        for t in summary["target"]
    ]
)

ax.set_ylabel(
    "Median object-level speedup"
)

ax.set_title(
    "(b) Connectivity-aware speedup"
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    alpha=0.2,
)

# Speedup + reach-rate annotations
for i, row in summary.iterrows():

    ax.text(
        i,
        row["median_speedup"] + 0.10,
        f'{row["median_speedup"]:.2f}$\\times$',
        ha="center",
        va="bottom",
        fontsize=9,
        )

    ax.text(
        i,
        0.12,
        (
            f'{int(row["n_reached"])}/'
            f'{int(row["n_objects"])} reached'
        ),
        ha="center",
        va="bottom",
        fontsize=7,
        rotation=90,
    )


# ============================================================
# FINISH
# ============================================================

fig.tight_layout()

fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)

print()
print("Saved:")
print(f"  {OUTPUT_PDF}")
print(f"  {OUTPUT_PNG}")

plt.show()