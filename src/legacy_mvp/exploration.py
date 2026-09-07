"""
05_structure_sample_complexity.py

Question
--------
How many successful grasp observations are needed to recover the
multiscale structure of an object's viable grasp space?

For each object:
    1. Load all successful grasps.
    2. Draw a fixed reference sample of N_REF successful grasps.
    3. Compute reference connectivity curves.
    4. For n = 10, 20, 50, 100, 200, 300:
         - sample n grasps from the SAME reference population
         - compute connectivity curves
         - compare against a matched-size sample drawn from the
           reference population.

Important
---------
We do NOT directly compare raw beta_0 curves for different point counts.
For every n, the reference comparison also contains n points.

The question is therefore:

    "Does a small random sample give a connectivity signature similar
     to other samples from the same object's full viable population?"

Outputs
-------
results/structure_sample_complexity.csv
results/structure_sample_complexity_summary.csv
results/structure_sample_complexity.png

Run
---
python src/05_structure_sample_complexity.py grasps \
    --n_objects 20 \
    --n_repeats 20
"""

from pathlib import Path
import argparse

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------

def load_successful_grasps(path):
    with h5py.File(path, "r") as f:
        T = np.asarray(f["grasps/transforms"])
        y = np.asarray(
            f["grasps/qualities/flex/object_in_gripper"]
        ).astype(bool)

    T = T[y]

    p = T[:, :3, 3]
    R = T[:, :3, :3]

    return p, R


# ------------------------------------------------------------
# Pairwise SE(3) distances
# ------------------------------------------------------------

def pairwise_parts(p, R):
    """
    Return translation and SO(3) geodesic distance matrices.
    """

    dp = cdist(p, p)

    relative = np.einsum(
        "aik,bil->abkl",
        R,
        R,
    )

    tr = np.trace(
        relative,
        axis1=2,
        axis2=3,
    )

    cos_angle = np.clip(
        (tr - 1.0) / 2.0,
        -1.0,
        1.0,
        )

    dr = np.arccos(cos_angle)

    return dp, dr


def normalized_distance(dp, dr):
    """
    Normalize translation and rotation by median nonzero
    pairwise distances.
    """

    n = len(dp)

    mask = ~np.eye(n, dtype=bool)

    sp = np.median(dp[mask])
    sr = np.median(dr[mask])

    sp = max(sp, 1e-12)
    sr = max(sr, 1e-12)

    return np.sqrt(
        (dp / sp) ** 2 +
        (dr / sr) ** 2
    )


# ------------------------------------------------------------
# Connectivity signature
# ------------------------------------------------------------

def connectivity_signature(D, eps_values):
    """
    For every epsilon calculate:

        beta0 / N
        largest component / N
    """

    n = len(D)

    beta = []
    largest = []

    for eps in eps_values:

        A = D <= eps

        # no self edges needed
        np.fill_diagonal(A, False)

        graph = csr_matrix(A)

        n_components, labels = connected_components(
            graph,
            directed=False,
        )

        counts = np.bincount(labels)

        beta.append(
            n_components / n
        )

        largest.append(
            counts.max() / n
        )

    return np.asarray(beta), np.asarray(largest)


def signature_distance(sig1, sig2):
    """
    Equal-weight mean absolute distance between beta0 and L curves.
    """

    beta1, largest1 = sig1
    beta2, largest2 = sig2

    d_beta = np.mean(
        np.abs(beta1 - beta2)
    )

    d_largest = np.mean(
        np.abs(largest1 - largest2)
    )

    combined = 0.5 * (
            d_beta + d_largest
    )

    return combined, d_beta, d_largest


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "grasp_dir",
    )

    parser.add_argument(
        "--n_objects",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--n_repeats",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--n_reference",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    rng = np.random.default_rng(
        args.seed
    )

    sample_sizes = [
        10,
        20,
        50,
        100,
        200,
        300,
    ]

    eps_values = np.linspace(
        0.02,
        1.5,
        100,
    )

    # --------------------------------------------------------
    # Find eligible objects
    # --------------------------------------------------------

    files = list(
        Path(args.grasp_dir).rglob("*.h5")
    )

    if len(files) == 0:
        files = list(
            Path(args.grasp_dir).rglob("*.hdf5")
        )

    rng.shuffle(files)

    eligible = []

    for path in files:

        with h5py.File(path, "r") as f:
            y = np.asarray(
                f[
                    "grasps/qualities/flex/"
                    "object_in_gripper"
                ]
            )

        if np.sum(y) >= args.n_reference:
            eligible.append(path)

        if len(eligible) >= args.n_objects:
            break

    print(
        f"Eligible objects selected: "
        f"{len(eligible)}"
    )

    print(
        f"Reference successes/object: "
        f"{args.n_reference}"
    )

    print(
        f"Repeats/sample size: "
        f"{args.n_repeats}"
    )

    print()

    rows = []

    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    for oi, path in enumerate(eligible):

        p_all, R_all = load_successful_grasps(
            path
        )

        print(
            f"[{oi + 1:02d}/{len(eligible)}] "
            f"{path.name}"
        )

        print(
            f"    successful grasps = "
            f"{len(p_all)}"
        )

        # Fixed reference population.
        reference_idx = rng.choice(
            len(p_all),
            size=args.n_reference,
            replace=False,
        )

        p = p_all[reference_idx]
        R = R_all[reference_idx]

        # Compute full distance matrices ONCE.
        dp, dr = pairwise_parts(
            p,
            R,
        )

        # ----------------------------------------------------
        # Each sample size
        # ----------------------------------------------------

        for n in sample_sizes:

            if n > args.n_reference:
                continue

            distances = []

            for repeat in range(
                    args.n_repeats
            ):

                # Two independent samples from the same
                # object's reference population.
                #
                # This avoids directly comparing a 20-point
                # beta0 curve with a 500-point beta0 curve.

                idx_a = rng.choice(
                    args.n_reference,
                    size=n,
                    replace=False,
                )

                idx_b = rng.choice(
                    args.n_reference,
                    size=n,
                    replace=False,
                )

                # Extract distance submatrices.
                dp_a = dp[
                    np.ix_(idx_a, idx_a)
                ]

                dr_a = dr[
                    np.ix_(idx_a, idx_a)
                ]

                dp_b = dp[
                    np.ix_(idx_b, idx_b)
                ]

                dr_b = dr[
                    np.ix_(idx_b, idx_b)
                ]

                D_a = normalized_distance(
                    dp_a,
                    dr_a,
                )

                D_b = normalized_distance(
                    dp_b,
                    dr_b,
                )

                sig_a = connectivity_signature(
                    D_a,
                    eps_values,
                )

                sig_b = connectivity_signature(
                    D_b,
                    eps_values,
                )

                (
                    d,
                    d_beta,
                    d_largest,
                ) = signature_distance(
                    sig_a,
                    sig_b,
                )

                distances.append(d)

                rows.append({
                    "object": path.name,
                    "sample_size": n,
                    "repeat": repeat,
                    "distance": d,
                    "beta_distance": d_beta,
                    "largest_distance": d_largest,
                })

            print(
                f"    n={n:3d} "
                f"distance="
                f"{np.mean(distances):.4f}"
            )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    Path("results").mkdir(
        exist_ok=True
    )

    df = pd.DataFrame(rows)

    df.to_csv(
        "results/"
        "structure_sample_complexity.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = (
        df
        .groupby("sample_size")
        .agg(
            mean_distance=(
                "distance",
                "mean",
            ),
            std_distance=(
                "distance",
                "std",
            ),
            median_distance=(
                "distance",
                "median",
            ),
            mean_beta_distance=(
                "beta_distance",
                "mean",
            ),
            mean_largest_distance=(
                "largest_distance",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        "results/"
        "structure_sample_complexity_summary.csv",
        index=False,
    )

    print()
    print("SUMMARY")
    print()
    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 5)
    )

    plt.errorbar(
        summary["sample_size"],
        summary["mean_distance"],
        yerr=summary["std_distance"],
        marker="o",
        capsize=4,
    )

    plt.xlabel(
        "Number of observed successful grasps"
    )

    plt.ylabel(
        "Connectivity signature distance"
    )

    plt.xscale("log")

    plt.tight_layout()

    plt.savefig(
        "results/"
        "structure_sample_complexity.png",
        dpi=200,
    )

    plt.close()

    print()
    print("Saved:")
    print(
        "  results/"
        "structure_sample_complexity.csv"
    )
    print(
        "  results/"
        "structure_sample_complexity_summary.csv"
    )
    print(
        "  results/"
        "structure_sample_complexity.png"
    )


if __name__ == "__main__":
    main()