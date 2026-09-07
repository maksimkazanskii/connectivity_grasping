"""
Experiment 1: Connectivity of successful grasp space.

Pipeline
--------
All ACRONYM objects
    -> objects with >= 500 successful grasps
    -> randomly select 100 objects
    -> sample exactly 500 successful grasps/object
    -> pairwise translation distances
    -> pairwise SO(3) rotation distances
    -> normalize each by median pairwise distance
    -> combined SE(3) distance
    -> epsilon-neighborhood graph
    -> connected components

Primary outputs
---------------
1. beta_0(epsilon): number of connected components
2. largest-component fraction L(epsilon)

Usage
-----
From project root:

    python src/02_grasp_connectivity.py grasps

Dependencies
------------
    pip install numpy pandas h5py scipy matplotlib
"""

from pathlib import Path
import argparse
import re

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


DEFAULT_N_OBJECTS = 100
DEFAULT_N_GRASPS = 500
DEFAULT_SEED = 42
DEFAULT_N_EPS = 100
DEFAULT_EPS_MIN = 0.02
DEFAULT_EPS_MAX = 1.50


# ============================================================
# File utilities
# ============================================================

def find_h5_files(root: Path):
    files = list(root.rglob("*.h5"))
    files += list(root.rglob("*.hdf5"))
    return sorted(set(files))


def object_name_from_file(path: Path):
    """
    Extract a short readable object/category name from filename.

    Example:
        Bottle_abc123_0.0012.h5 -> Bottle
    """
    stem = path.stem
    return stem.split("_")[0]


def safe_filename(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


# ============================================================
# ACRONYM loading
# ============================================================

def load_successful_transforms(path: Path):
    """
    Load successful grasp transforms only.

    Returns
    -------
    transforms : ndarray, shape (N_success, 4, 4)
    """

    with h5py.File(path, "r") as f:
        transforms = np.asarray(f["grasps/transforms"])
        success = np.asarray(
            f["grasps/qualities/flex/object_in_gripper"]
        ).reshape(-1)

    mask = success > 0

    return transforms[mask]


def count_successes(path: Path):
    """Count successful grasps without loading transforms."""

    with h5py.File(path, "r") as f:
        success = np.asarray(
            f["grasps/qualities/flex/object_in_gripper"]
        )

    return int(np.sum(success > 0))


# ============================================================
# Distance calculations
# ============================================================

def translation_distance_matrix(transforms):
    """
    Pairwise Euclidean translation distances.

    p_i = transforms[i, :3, 3]
    """

    positions = transforms[:, :3, 3]

    return squareform(
        pdist(positions, metric="euclidean")
    )


def rotation_distance_matrix(transforms):
    """
    Pairwise geodesic SO(3) distance:

        theta = acos((trace(R_i^T R_j) - 1) / 2)

    Returns radians.
    """

    rotations = transforms[:, :3, :3]

    n = len(rotations)

    # Relative rotations:
    #
    # relative[i,j] = R_i^T R_j

    relative = np.einsum(
        "aik,bil->abkl",
        rotations,
        rotations,
    )

    traces = np.trace(
        relative,
        axis1=2,
        axis2=3,
    )

    cos_theta = (traces - 1.0) / 2.0

    # Numerical safety
    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    return np.arccos(cos_theta)


def median_upper_triangle(matrix):
    """
    Median over unique pairwise distances,
    excluding diagonal.
    """

    values = matrix[
        np.triu_indices_from(matrix, k=1)
    ]

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return float(np.median(values))


def combined_distance_matrix(transforms):
    """
    Compute dimensionless combined SE(3) distance:

        d = sqrt(
            (d_translation / median_translation)^2
            +
            (d_rotation / median_rotation)^2
        )
    """

    d_position = translation_distance_matrix(
        transforms
    )

    d_rotation = rotation_distance_matrix(
        transforms
    )

    position_scale = median_upper_triangle(
        d_position
    )

    rotation_scale = median_upper_triangle(
        d_rotation
    )

    if not np.isfinite(position_scale):
        raise ValueError(
            "Invalid translation distance scale."
        )

    if not np.isfinite(rotation_scale):
        raise ValueError(
            "Invalid rotation distance scale."
        )

    if position_scale <= 1e-12:
        raise ValueError(
            "Translation scale is approximately zero."
        )

    if rotation_scale <= 1e-12:
        raise ValueError(
            "Rotation scale is approximately zero."
        )

    d_position_normalized = (
            d_position / position_scale
    )

    d_rotation_normalized = (
            d_rotation / rotation_scale
    )

    combined = np.sqrt(
        d_position_normalized ** 2
        +
        d_rotation_normalized ** 2
    )

    np.fill_diagonal(combined, 0.0)

    return (
        combined,
        position_scale,
        rotation_scale,
    )


# ============================================================
# Connectivity
# ============================================================

def connectivity_at_epsilon(distance_matrix, epsilon):
    """
    Construct epsilon-neighborhood graph and return:

        beta_0 = number of connected components

        largest_fraction =
            size(largest component) / N
    """

    adjacency = (
            distance_matrix <= epsilon
    )

    # Remove self-edges.
    np.fill_diagonal(adjacency, False)

    graph = csr_matrix(adjacency)

    n_components, labels = connected_components(
        graph,
        directed=False,
        return_labels=True,
    )

    sizes = np.bincount(labels)

    largest_fraction = (
            sizes.max() / len(labels)
    )

    return (
        int(n_components),
        float(largest_fraction),
    )


def connectivity_curve(
        distance_matrix,
        epsilons,
):
    beta0 = []
    largest = []

    for epsilon in epsilons:

        n_components, largest_fraction = (
            connectivity_at_epsilon(
                distance_matrix,
                epsilon,
            )
        )

        beta0.append(n_components)
        largest.append(largest_fraction)

    return (
        np.asarray(beta0),
        np.asarray(largest),
    )


# ============================================================
# Plotting
# ============================================================

def plot_all_beta0(
        results,
        epsilons,
        n_grasps,
        output_path,
):
    plt.figure(figsize=(9, 6))

    for result in results:
        plt.plot(
            epsilons,
            result["beta0"] / n_grasps,
            alpha=0.30,
            linewidth=1.0,
            )

    plt.xlabel(r"$\epsilon$")
    plt.ylabel(r"$\beta_0(\epsilon) / N$")
    plt.title(
        "Connectivity of Successful Grasp Space"
    )

    plt.ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def plot_all_largest(
        results,
        epsilons,
        output_path,
):
    plt.figure(figsize=(9, 6))

    for result in results:
        plt.plot(
            epsilons,
            result["largest_fraction"],
            alpha=0.30,
            linewidth=1.0,
        )

    plt.xlabel(r"$\epsilon$")
    plt.ylabel(
        "Largest connected component fraction"
    )

    plt.title(
        "Growth of Largest Viable Grasp Component"
    )

    plt.ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def plot_representative_curves(
        results,
        epsilons,
        output_path,
):
    """
    Pick representative objects according to the epsilon
    at which the largest component first exceeds 90%.
    """

    def epsilon_90(result):
        values = result["largest_fraction"]

        indices = np.where(values >= 0.90)[0]

        if len(indices) == 0:
            return np.inf

        return epsilons[indices[0]]

    ordered = sorted(
        results,
        key=epsilon_90,
    )

    if len(ordered) <= 6:
        chosen = ordered
    else:
        indices = np.linspace(
            0,
            len(ordered) - 1,
            6,
            dtype=int,
            )

        chosen = [
            ordered[i]
            for i in indices
        ]

    plt.figure(figsize=(9, 6))

    for result in chosen:

        label = (
            f"{result['category']} "
            f"(N+={result['n_success_total']})"
        )

        plt.plot(
            epsilons,
            result["largest_fraction"],
            linewidth=2,
            label=label,
        )

    plt.xlabel(r"$\epsilon$")
    plt.ylabel(
        "Largest connected component fraction"
    )

    plt.title(
        "Representative Grasp-Space Connectivity"
    )

    plt.ylim(-0.02, 1.02)

    plt.legend(
        fontsize=8,
        frameon=False,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir",
        nargs="?",
        default="grasps",
        help="ACRONYM grasp directory.",
    )

    parser.add_argument(
        "--n_objects",
        type=int,
        default=DEFAULT_N_OBJECTS,
    )

    parser.add_argument(
        "--n_grasps",
        type=int,
        default=DEFAULT_N_GRASPS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--n_eps",
        type=int,
        default=DEFAULT_N_EPS,
    )

    parser.add_argument(
        "--eps_min",
        type=float,
        default=DEFAULT_EPS_MIN,
    )

    parser.add_argument(
        "--eps_max",
        type=float,
        default=DEFAULT_EPS_MAX,
    )

    args = parser.parse_args()

    root = Path(
        args.data_dir
    ).expanduser().resolve()

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(
        args.seed
    )

    # --------------------------------------------------------
    # Find all ACRONYM objects
    # --------------------------------------------------------

    files = find_h5_files(root)

    print()
    print("=" * 70)
    print("EXPERIMENT 1: GRASP-SPACE CONNECTIVITY")
    print("=" * 70)

    print(f"Data directory : {root}")
    print(f"Objects found  : {len(files)}")
    print(f"Required N+    : {args.n_grasps}")

    if not files:
        raise RuntimeError(
            "No HDF5 files found."
        )

    # --------------------------------------------------------
    # Find eligible objects
    # --------------------------------------------------------

    print()
    print("Scanning successful grasp counts...")

    eligible = []

    for i, path in enumerate(files, start=1):

        try:
            n_success = count_successes(path)

            if n_success >= args.n_grasps:
                eligible.append(
                    (path, n_success)
                )

        except Exception as exc:
            print(
                f"Warning: could not read {path.name}: "
                f"{exc}"
            )

        if i % 500 == 0:
            print(
                f"  scanned {i:5d}/{len(files)} | "
                f"eligible={len(eligible):5d}"
            )

    print()
    print(
        f"Eligible objects: "
        f"{len(eligible)}/{len(files)}"
    )

    if len(eligible) < args.n_objects:

        raise RuntimeError(
            f"Only {len(eligible)} objects have "
            f">= {args.n_grasps} successful grasps."
        )

    # --------------------------------------------------------
    # Randomly select objects
    # --------------------------------------------------------

    selected_indices = rng.choice(
        len(eligible),
        size=args.n_objects,
        replace=False,
    )

    selected = [
        eligible[i]
        for i in selected_indices
    ]

    epsilons = np.linspace(
        args.eps_min,
        args.eps_max,
        args.n_eps,
    )

    results = []
    summary_rows = []

    # --------------------------------------------------------
    # Process each object
    # --------------------------------------------------------

    print()
    print(
        f"Analyzing {args.n_objects} objects "
        f"with {args.n_grasps} successful grasps each..."
    )
    print()

    for object_index, (
            path,
            n_success_total,
    ) in enumerate(selected, start=1):

        successful = load_successful_transforms(
            path
        )

        # Equalize sampling density.
        sample_indices = rng.choice(
            len(successful),
            size=args.n_grasps,
            replace=False,
        )

        transforms = successful[
            sample_indices
        ]

        (
            distance_matrix,
            position_scale,
            rotation_scale,
        ) = combined_distance_matrix(
            transforms
        )

        beta0, largest = connectivity_curve(
            distance_matrix,
            epsilons,
        )

        category = object_name_from_file(
            path
        )

        # epsilon at which >= 90% belongs
        # to the largest component.
        idx90 = np.where(
            largest >= 0.90
        )[0]

        if len(idx90):
            epsilon_90 = float(
                epsilons[idx90[0]]
            )
        else:
            epsilon_90 = np.nan

        # epsilon at which graph becomes fully connected.
        idx_full = np.where(
            beta0 == 1
        )[0]

        if len(idx_full):
            epsilon_connected = float(
                epsilons[idx_full[0]]
            )
        else:
            epsilon_connected = np.nan

        # Area under normalized beta0 curve.
        beta0_normalized = (
                beta0 / args.n_grasps
        )

        beta0_auc = float(
            np.trapezoid(
                beta0_normalized,
                epsilons,
            )
        )

        result = {
            "path": str(path),
            "file": path.name,
            "category": category,
            "n_success_total": n_success_total,
            "position_scale": position_scale,
            "rotation_scale": rotation_scale,
            "beta0": beta0,
            "largest_fraction": largest,
            "epsilon_90": epsilon_90,
            "epsilon_connected": epsilon_connected,
            "beta0_auc": beta0_auc,
        }

        results.append(result)

        summary_rows.append(
            {
                "file": path.name,
                "category": category,
                "n_success_total": n_success_total,
                "n_sampled": args.n_grasps,
                "position_scale": position_scale,
                "rotation_scale_rad": rotation_scale,
                "rotation_scale_deg": np.degrees(
                    rotation_scale
                ),
                "epsilon_90": epsilon_90,
                "epsilon_connected": (
                    epsilon_connected
                ),
                "beta0_auc": beta0_auc,
            }
        )

        print(
            f"{object_index:3d}/{args.n_objects} | "
            f"{category:18s} | "
            f"N+={n_success_total:4d} | "
            f"pos_scale={position_scale:.4g} | "
            f"rot_scale={np.degrees(rotation_scale):6.1f}° | "
            f"eps90={epsilon_90:.3f}"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        summary_rows
    )

    summary_path = (
            output_dir /
            "grasp_connectivity_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save full curves
    # --------------------------------------------------------

    curve_rows = []

    for result in results:

        for epsilon, beta, largest in zip(
                epsilons,
                result["beta0"],
                result["largest_fraction"],
        ):

            curve_rows.append(
                {
                    "file": result["file"],
                    "category": result["category"],
                    "epsilon": epsilon,
                    "beta0": beta,
                    "beta0_fraction": (
                            beta / args.n_grasps
                    ),
                    "largest_fraction": largest,
                }
            )

    curves = pd.DataFrame(
        curve_rows
    )

    curves_path = (
            output_dir /
            "grasp_connectivity_curves.csv"
    )

    curves.to_csv(
        curves_path,
        index=False,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    beta0_plot = (
            output_dir /
            "grasp_beta0_curves.png"
    )

    largest_plot = (
            output_dir /
            "grasp_largest_component_curves.png"
    )

    representative_plot = (
            output_dir /
            "grasp_representative_connectivity.png"
    )

    plot_all_beta0(
        results,
        epsilons,
        args.n_grasps,
        beta0_plot,
    )

    plot_all_largest(
        results,
        epsilons,
        largest_plot,
    )

    plot_representative_curves(
        results,
        epsilons,
        representative_plot,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print()
    print("epsilon_90:")
    print(
        summary["epsilon_90"].describe()
    )

    print()
    print("epsilon_connected:")
    print(
        summary[
            "epsilon_connected"
        ].describe()
    )

    print()
    print("beta0 AUC:")
    print(
        summary["beta0_auc"].describe()
    )

    print()
    print("Files saved:")
    print(f"  {summary_path}")
    print(f"  {curves_path}")
    print(f"  {beta0_plot}")
    print(f"  {largest_plot}")
    print(f"  {representative_plot}")

    print()
    print(
        "Objects with LOWEST beta0 AUC "
        "(fastest connectivity):"
    )

    print(
        summary.nsmallest(
            5,
            "beta0_auc",
        )[
            [
                "category",
                "n_success_total",
                "epsilon_90",
                "epsilon_connected",
                "beta0_auc",
            ]
        ].to_string(index=False)
    )

    print()
    print(
        "Objects with HIGHEST beta0 AUC "
        "(slowest connectivity):"
    )

    print(
        summary.nlargest(
            5,
            "beta0_auc",
        )[
            [
                "category",
                "n_success_total",
                "epsilon_90",
                "epsilon_connected",
                "beta0_auc",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()