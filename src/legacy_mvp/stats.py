"""
Test whether viable grasp-space structure is reproducible within objects.

For each object:
    1. Sample N successful grasps independently R times.
    2. Compute normalized SE(3) distances.
    3. Compute multiscale connectivity curves:
           B(eps) = beta_0(eps) / N
           L(eps) = largest_component_size / N
    4. Compare curve distances:
           within-object vs between-object
    5. Test whether connectivity signatures identify the object.

Usage
-----
python src/04_grasp_structure_reproducibility.py grasps

Quick test:
python src/04_grasp_structure_reproducibility.py grasps \
    --n_objects 20 \
    --n_repeats 5 \
    --n_success 300

Full:
python src/04_grasp_structure_reproducibility.py grasps \
    --n_objects 100 \
    --n_repeats 10 \
    --n_success 500

Outputs
-------
results/reproducibility_curves.csv
results/reproducibility_summary.csv
results/reproducibility_distances.csv
results/reproducibility_within_between.png
results/reproducibility_example_curves.png
"""

from pathlib import Path
import argparse

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu


# ============================================================
# Data
# ============================================================

def find_h5_files(root):
    files = list(root.rglob("*.h5"))
    files += list(root.rglob("*.hdf5"))
    return sorted(set(files))


def category_from_path(path):
    return path.stem.split("_")[0]


def load_successful_grasps(path):
    with h5py.File(path, "r") as f:
        transforms = np.asarray(f["grasps/transforms"])

        labels = np.asarray(
            f["grasps/qualities/flex/object_in_gripper"]
        ).reshape(-1)

    return transforms[labels > 0]


def count_successes(path):
    try:
        with h5py.File(path, "r") as f:
            labels = np.asarray(
                f["grasps/qualities/flex/object_in_gripper"]
            ).reshape(-1)

        return int(np.sum(labels > 0))

    except Exception:
        return 0


# ============================================================
# SE(3) distances
# ============================================================

def translation_distance_matrix(transforms):
    positions = transforms[:, :3, 3]

    return squareform(
        pdist(positions, metric="euclidean")
    )


def rotation_distance_matrix(transforms):
    """
    SO(3) geodesic distance:
        acos((trace(R_i^T R_j) - 1) / 2)
    """

    rotations = transforms[:, :3, :3]

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

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    return np.arccos(cos_theta)


def upper_median(matrix):
    values = matrix[
        np.triu_indices_from(matrix, k=1)
    ]

    values = values[
        np.isfinite(values) & (values > 1e-12)
        ]

    if len(values) == 0:
        return 1.0

    return float(np.median(values))


def normalized_se3_distance(transforms):
    """
    Same per-sample normalization as previous experiment:

        d = sqrt(
            (dp / median(dp))^2
            +
            (dR / median(dR))^2
        )
    """

    dp = translation_distance_matrix(transforms)
    dr = rotation_distance_matrix(transforms)

    position_scale = upper_median(dp)
    rotation_scale = upper_median(dr)

    distance = np.sqrt(
        (dp / position_scale) ** 2
        +
        (dr / rotation_scale) ** 2
    )

    np.fill_diagonal(distance, 0.0)

    return distance


# ============================================================
# Connectivity curves
# ============================================================

def connectivity_curves(distance_matrix, epsilons):
    """
    Returns:
        beta_curve:
            beta_0(eps) / N

        largest_curve:
            largest connected component / N
    """

    n = len(distance_matrix)

    beta_curve = np.zeros(
        len(epsilons),
        dtype=float,
    )

    largest_curve = np.zeros(
        len(epsilons),
        dtype=float,
    )

    for i, epsilon in enumerate(epsilons):

        adjacency = (
                distance_matrix <= epsilon
        )

        np.fill_diagonal(
            adjacency,
            False,
        )

        graph = csr_matrix(adjacency)

        n_components, labels = (
            connected_components(
                graph,
                directed=False,
                return_labels=True,
            )
        )

        sizes = np.bincount(labels)

        beta_curve[i] = (
                n_components / n
        )

        largest_curve[i] = (
                sizes.max() / n
        )

    return beta_curve, largest_curve


# ============================================================
# Curve distances
# ============================================================

def curve_distance(a, b):
    """
    Mean absolute difference between two curves.
    """

    return float(
        np.mean(
            np.abs(a - b)
        )
    )


def combined_curve_distance(sample_a, sample_b):
    """
    Equal weighting of beta_0 and largest-component curves.
    """

    db = curve_distance(
        sample_a["beta"],
        sample_b["beta"],
    )

    dl = curve_distance(
        sample_a["largest"],
        sample_b["largest"],
    )

    return 0.5 * (db + dl)


# ============================================================
# Effect size
# ============================================================

def cliffs_delta(x, y):
    """
    Cliff's delta.

    Negative here means within-object distances tend to be
    smaller than between-object distances.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    greater = 0
    smaller = 0

    for value in x:
        greater += np.sum(value > y)
        smaller += np.sum(value < y)

    return (
            greater - smaller
    ) / (len(x) * len(y))


# ============================================================
# Permutation test
# ============================================================

def permutation_test(
        samples,
        observed_difference,
        rng,
        n_permutations=1000,
):
    """
    Permute object labels while preserving the number of
    replicates assigned to each object.

    Statistic:
        mean(between) - mean(within)

    Large positive values support reproducibility.
    """

    n = len(samples)

    original_labels = np.array(
        [s["object_id"] for s in samples]
    )

    # Precompute all pair distances once.
    distance_matrix = np.zeros(
        (n, n),
        dtype=float,
    )

    for i in range(n):
        for j in range(i + 1, n):

            d = combined_curve_distance(
                samples[i],
                samples[j],
            )

            distance_matrix[i, j] = d
            distance_matrix[j, i] = d

    permutation_statistics = []

    for _ in range(n_permutations):

        labels = rng.permutation(
            original_labels
        )

        within = []
        between = []

        for i in range(n):
            for j in range(i + 1, n):

                d = distance_matrix[i, j]

                if labels[i] == labels[j]:
                    within.append(d)
                else:
                    between.append(d)

        statistic = (
                np.mean(between)
                - np.mean(within)
        )

        permutation_statistics.append(
            statistic
        )

    permutation_statistics = np.asarray(
        permutation_statistics
    )

    p_value = (
                      1
                      + np.sum(
                  permutation_statistics
                  >= observed_difference
              )
              ) / (
                      n_permutations + 1
              )

    return float(p_value)


# ============================================================
# Nearest-neighbor identification
# ============================================================

def nearest_neighbor_accuracy(samples):
    """
    For each replicate, find the closest other replicate.

    Correct if it belongs to the same object.
    """

    correct = 0

    for i, sample in enumerate(samples):

        best_distance = np.inf
        best_object = None

        for j, candidate in enumerate(samples):

            if i == j:
                continue

            d = combined_curve_distance(
                sample,
                candidate,
            )

            if d < best_distance:
                best_distance = d
                best_object = candidate["object_id"]

        if best_object == sample["object_id"]:
            correct += 1

    return correct / len(samples)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir",
        nargs="?",
        default="grasps",
    )

    parser.add_argument(
        "--n_objects",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--n_repeats",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--n_success",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--n_eps",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--eps_min",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--eps_max",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--n_permutations",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    epsilons = np.linspace(
        args.eps_min,
        args.eps_max,
        args.n_eps,
    )

    print()
    print("=" * 72)
    print("GRASP-SPACE STRUCTURAL REPRODUCIBILITY")
    print("=" * 72)

    files = find_h5_files(root)

    print(f"HDF5 files found: {len(files)}")
    print("Scanning for eligible objects...")

    eligible = []

    for i, path in enumerate(files, start=1):

        n_success = count_successes(path)

        if n_success >= args.n_success:
            eligible.append(
                (path, n_success)
            )

        if i % 1000 == 0:
            print(
                f"  {i:5d}/{len(files)} "
                f"eligible={len(eligible)}"
            )

    print()
    print(f"Eligible objects: {len(eligible)}")

    if len(eligible) < args.n_objects:
        raise RuntimeError(
            f"Requested {args.n_objects} objects, "
            f"but only {len(eligible)} are eligible."
        )

    selected_indices = rng.choice(
        len(eligible),
        size=args.n_objects,
        replace=False,
    )

    selected = [
        eligible[i]
        for i in selected_indices
    ]

    # ========================================================
    # Generate replicate signatures
    # ========================================================

    samples = []
    curve_rows = []
    summary_rows = []

    for object_index, (
            path,
            total_success,
    ) in enumerate(
        selected,
        start=1,
    ):

        successful = load_successful_grasps(
            path
        )

        object_id = path.stem
        category = category_from_path(path)

        print()
        print(
            f"[{object_index:3d}/{args.n_objects}] "
            f"{category} | "
            f"successes={len(successful)}"
        )

        for repeat in range(
                args.n_repeats
        ):

            indices = rng.choice(
                len(successful),
                size=args.n_success,
                replace=False,
            )

            sampled = successful[
                indices
            ]

            distance_matrix = (
                normalized_se3_distance(
                    sampled
                )
            )

            beta_curve, largest_curve = (
                connectivity_curves(
                    distance_matrix,
                    epsilons,
                )
            )

            sample = {
                "object_id": object_id,
                "category": category,
                "repeat": repeat,
                "beta": beta_curve,
                "largest": largest_curve,
            }

            samples.append(sample)

            # scalar summaries
            epsilon_90 = np.nan

            indices_90 = np.where(
                largest_curve >= 0.90
            )[0]

            if len(indices_90) > 0:
                epsilon_90 = epsilons[
                    indices_90[0]
                ]

            beta_auc = np.trapezoid(
                beta_curve,
                epsilons,
            )

            largest_auc = np.trapezoid(
                largest_curve,
                epsilons,
            )

            summary_rows.append({
                "object_id": object_id,
                "category": category,
                "repeat": repeat,
                "total_success_available": total_success,
                "epsilon_90": epsilon_90,
                "beta0_auc": beta_auc,
                "largest_component_auc": largest_auc,
            })

            for epsilon, beta, largest in zip(
                    epsilons,
                    beta_curve,
                    largest_curve,
            ):

                curve_rows.append({
                    "object_id": object_id,
                    "category": category,
                    "repeat": repeat,
                    "epsilon": epsilon,
                    "beta0_fraction": beta,
                    "largest_component_fraction": largest,
                })

            print(
                f"    repeat "
                f"{repeat + 1:2d}/{args.n_repeats}",
                end="\r",
                flush=True,
            )

        print()

    # ========================================================
    # Pairwise within/between distances
    # ========================================================

    print()
    print("Computing pairwise curve distances...")

    within = []
    between = []
    distance_rows = []

    for i in range(len(samples)):

        for j in range(i + 1, len(samples)):

            db = curve_distance(
                samples[i]["beta"],
                samples[j]["beta"],
            )

            dl = curve_distance(
                samples[i]["largest"],
                samples[j]["largest"],
            )

            combined = 0.5 * (
                    db + dl
            )

            same_object = (
                    samples[i]["object_id"]
                    ==
                    samples[j]["object_id"]
            )

            if same_object:
                within.append(combined)
                pair_type = "within"
            else:
                between.append(combined)
                pair_type = "between"

            distance_rows.append({
                "object_a": samples[i]["object_id"],
                "repeat_a": samples[i]["repeat"],
                "object_b": samples[j]["object_id"],
                "repeat_b": samples[j]["repeat"],
                "pair_type": pair_type,
                "beta_distance": db,
                "largest_distance": dl,
                "combined_distance": combined,
            })

    within = np.asarray(within)
    between = np.asarray(between)

    # ========================================================
    # Statistics
    # ========================================================

    mean_within = float(
        np.mean(within)
    )

    mean_between = float(
        np.mean(between)
    )

    median_within = float(
        np.median(within)
    )

    median_between = float(
        np.median(between)
    )

    ratio = (
            mean_within
            / mean_between
    )

    reliability = (
            1.0 - ratio
    )

    observed_difference = (
            mean_between
            - mean_within
    )

    u_stat, mw_p = mannwhitneyu(
        within,
        between,
        alternative="less",
    )

    delta = cliffs_delta(
        within,
        between,
    )

    print()
    print("Running permutation test...")

    permutation_p = permutation_test(
        samples=samples,
        observed_difference=observed_difference,
        rng=rng,
        n_permutations=args.n_permutations,
    )

    print()
    print("Running nearest-neighbor identification...")

    nn_accuracy = (
        nearest_neighbor_accuracy(
            samples
        )
    )

    chance_accuracy = (
            1.0 / args.n_objects
    )

    # ========================================================
    # Print results
    # ========================================================

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(
        f"Objects:                  "
        f"{args.n_objects}"
    )

    print(
        f"Replicates/object:        "
        f"{args.n_repeats}"
    )

    print(
        f"Successful grasps/sample: "
        f"{args.n_success}"
    )

    print()
    print(
        f"Mean within distance:     "
        f"{mean_within:.5f}"
    )

    print(
        f"Mean between distance:    "
        f"{mean_between:.5f}"
    )

    print(
        f"Median within distance:   "
        f"{median_within:.5f}"
    )

    print(
        f"Median between distance:  "
        f"{median_between:.5f}"
    )

    print()
    print(
        f"Within / between ratio:   "
        f"{ratio:.3f}"
    )

    print(
        f"Reliability score:        "
        f"{reliability:.3f}"
    )

    print(
        f"Cliff's delta:            "
        f"{delta:.3f}"
    )

    print(
        f"Mann-Whitney p:           "
        f"{mw_p:.3e}"
    )

    print(
        f"Permutation p:            "
        f"{permutation_p:.4g}"
    )

    print()
    print(
        f"1-NN object accuracy:     "
        f"{100 * nn_accuracy:.2f}%"
    )

    print(
        f"Chance accuracy:          "
        f"{100 * chance_accuracy:.2f}%"
    )

    # ========================================================
    # Save tables
    # ========================================================

    curves_df = pd.DataFrame(
        curve_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    distances_df = pd.DataFrame(
        distance_rows
    )

    curves_path = (
            output_dir
            / "reproducibility_curves.csv"
    )

    summary_path = (
            output_dir
            / "reproducibility_summary.csv"
    )

    distances_path = (
            output_dir
            / "reproducibility_distances.csv"
    )

    curves_df.to_csv(
        curves_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    distances_df.to_csv(
        distances_path,
        index=False,
    )

    # ========================================================
    # Figure 1: within vs between
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.boxplot(
        [within, between],
        tick_labels=[
            "Same object",
            "Different objects",
        ],
        showfliers=False,
    )

    # Light jittered subset for readability.
    max_points = 2000

    for x_position, values in enumerate(
            [within, between],
            start=1,
    ):

        if len(values) > max_points:
            plot_values = rng.choice(
                values,
                size=max_points,
                replace=False,
            )
        else:
            plot_values = values

        jitter = rng.normal(
            loc=0.0,
            scale=0.035,
            size=len(plot_values),
        )

        ax.scatter(
            x_position + jitter,
            plot_values,
            s=7,
            alpha=0.15,
            )

    ax.set_ylabel(
        "Connectivity-signature distance"
    )

    ax.set_title(
        "Within-object vs Between-object "
        "Grasp-Space Structure"
    )

    ax.text(
        0.02,
        0.98,
        (
            f"mean within = {mean_within:.4f}\n"
            f"mean between = {mean_between:.4f}\n"
            f"ratio = {ratio:.3f}\n"
            f"Cliff's δ = {delta:.3f}\n"
            f"permutation p = {permutation_p:.3g}\n"
            f"1-NN = {100 * nn_accuracy:.1f}% "
            f"(chance {100 * chance_accuracy:.1f}%)"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    plt.tight_layout()

    figure_path = (
            output_dir
            / "reproducibility_within_between.png"
    )

    plt.savefig(
        figure_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Figure 2: example replicate curves
    # ========================================================

    n_examples = min(
        6,
        args.n_objects,
    )

    example_objects = [
        selected[i][0].stem
        for i in range(n_examples)
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
    )

    axes = axes.ravel()

    for ax, object_id in zip(
            axes,
            example_objects,
    ):

        object_samples = [
            s
            for s in samples
            if s["object_id"] == object_id
        ]

        for sample in object_samples:

            ax.plot(
                epsilons,
                sample["largest"],
                alpha=0.55,
                linewidth=1.2,
            )

        category = (
            object_samples[0]["category"]
        )

        ax.set_title(category)

        ax.set_xlabel(
            r"$\epsilon$"
        )

        ax.set_ylabel(
            "Largest component fraction"
        )

        ax.set_ylim(
            -0.02,
            1.02,
        )

    for ax in axes[
        len(example_objects):
    ]:
        ax.axis("off")

    fig.suptitle(
        "Independent Samples from the Same Object",
        fontsize=15,
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    example_path = (
            output_dir
            / "reproducibility_example_curves.png"
    )

    plt.savefig(
        example_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(curves_path)
    print(summary_path)
    print(distances_path)
    print(figure_path)
    print(example_path)


if __name__ == "__main__":
    main()