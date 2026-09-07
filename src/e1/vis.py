"""
Visualize ACRONYM grasp solution spaces in 2D.

Successful and failed 6-DoF grasps are projected from SE(3) to 2D
using metric multidimensional scaling (MDS).

IMPORTANT:
    Connected components are computed in the ORIGINAL normalized
    SE(3) distance space. The 2D projection is visualization only.

Usage
-----
    python src/03_visualize_grasp_space.py grasps

Optional:
    python src/03_visualize_grasp_space.py grasps \
        --n_objects 6 \
        --n_success 500 \
        --n_failed 500 \
        --seed 42

Dependencies
------------
    pip install numpy h5py scipy scikit-learn matplotlib
"""

from pathlib import Path
import argparse
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.manifold import MDS


# ============================================================
# Data
# ============================================================

def find_h5_files(root):
    files = list(root.rglob("*.h5"))
    files += list(root.rglob("*.hdf5"))
    return sorted(set(files))


def category_from_path(path):
    return path.stem.split("_")[0]


def load_grasps(path):
    with h5py.File(path, "r") as f:
        transforms = np.asarray(
            f["grasps/transforms"]
        )

        success = np.asarray(
            f["grasps/qualities/flex/object_in_gripper"]
        ).reshape(-1)

    return transforms, success > 0


# ============================================================
# Distances
# ============================================================

def translation_distance_matrix(transforms):
    positions = transforms[:, :3, 3]
    return squareform(pdist( positions, metric="euclidean"))


def rotation_distance_matrix(transforms):
    """
    SO(3) geodesic distance:

        d(R_i,R_j) =
            acos((trace(R_i^T R_j)-1)/2)
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

    cos_theta = (
                        traces - 1.0
                ) / 2.0

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    return np.arccos(
        cos_theta
    )


def upper_median(matrix):

    values = matrix[
        np.triu_indices_from(
            matrix,
            k=1,
        )
    ]

    values = values[
        np.isfinite(values)
    ]

    return float(
        np.median(values)
    )


def normalized_se3_distance(
        transforms,
        position_scale=None,
        rotation_scale=None,
):
    """
    Combined dimensionless distance:

        sqrt(
            (dp / median(dp))^2
            +
            (dR / median(dR))^2
        )

    If scales are provided, those scales are used.
    """

    dp = translation_distance_matrix(
        transforms
    )

    dr = rotation_distance_matrix(
        transforms
    )

    if position_scale is None:
        position_scale = upper_median(dp)

    if rotation_scale is None:
        rotation_scale = upper_median(dr)

    if position_scale <= 1e-12:
        raise ValueError(
            "Position scale approximately zero."
        )

    if rotation_scale <= 1e-12:
        raise ValueError(
            "Rotation scale approximately zero."
        )

    distance = np.sqrt(
        (dp / position_scale) ** 2
        +
        (dr / rotation_scale) ** 2
    )

    np.fill_diagonal(
        distance,
        0.0,
    )

    return (
        distance,
        position_scale,
        rotation_scale,
    )


# ============================================================
# Connectivity
# ============================================================

def component_labels(
        distance_matrix,
        epsilon,
):

    adjacency = (
            distance_matrix <= epsilon
    )

    np.fill_diagonal(
        adjacency,
        False,
    )

    graph = csr_matrix(
        adjacency
    )

    n_components, labels = (
        connected_components(
            graph,
            directed=False,
            return_labels=True,
        )
    )

    return n_components, labels


def largest_component_fraction(labels):

    sizes = np.bincount(
        labels
    )

    return (
            sizes.max()
            / len(labels)
    )


def choose_epsilon(
        distance_matrix,
        target_largest=0.50,
):
    """
    Find an epsilon where the largest connected component
    is approximately target_largest.

    We search only the successful grasp cloud.
    """

    epsilons = np.linspace(
        0.05,
        1.0,
        150,
    )

    best_epsilon = None
    best_error = np.inf
    best_fraction = None

    for epsilon in epsilons:

        _, labels = component_labels(
            distance_matrix,
            epsilon,
        )

        fraction = (
            largest_component_fraction(
                labels
            )
        )

        error = abs(
            fraction - target_largest
        )

        if error < best_error:

            best_error = error
            best_epsilon = epsilon
            best_fraction = fraction

    return (
        float(best_epsilon),
        float(best_fraction),
    )


# ============================================================
# Component processing
# ============================================================

def relabel_components_by_size(labels):
    """
    Relabel components so:
        0 = largest
        1 = second largest
        ...
    """

    counts = np.bincount(
        labels
    )

    order = np.argsort(
        counts
    )[::-1]

    mapping = {
        old: new
        for new, old
        in enumerate(order)
    }

    new_labels = np.array(
        [
            mapping[x]
            for x in labels
        ]
    )

    new_counts = counts[
        order
    ]

    return (
        new_labels,
        new_counts,
    )


# ============================================================
# MDS
# ============================================================

def run_mds(
        distance_matrix,
        seed,
):
    """
    Metric MDS from precomputed distances.
    """

    print(
        f"      running MDS on "
        f"{len(distance_matrix)} grasps..."
    )

    mds = MDS(
        n_components=2,
        metric=True,
        dissimilarity="precomputed",
        random_state=seed,
        n_init=2,
        max_iter=300,
        eps=1e-6,
        normalized_stress="auto",
        n_jobs=-1,
    )

    coordinates = mds.fit_transform(
        distance_matrix
    )

    return (
        coordinates,
        float(mds.stress_),
    )


# ============================================================
# Object selection
# ============================================================

def find_eligible_objects(
        files,
        min_success,
        min_failed,
):

    eligible = []

    print(
        "Scanning objects..."
    )

    for i, path in enumerate(
            files,
            start=1,
    ):

        try:

            with h5py.File(
                    path,
                    "r",
            ) as f:

                labels = np.asarray(
                    f[
                        "grasps/qualities/"
                        "flex/object_in_gripper"
                    ]
                ).reshape(-1)

            n_success = int(
                np.sum(labels > 0)
            )

            n_failed = (
                    len(labels)
                    - n_success
            )

            if (
                    n_success >= min_success
                    and
                    n_failed >= min_failed
            ):
                eligible.append(
                    (
                        path,
                        n_success,
                        n_failed,
                    )
                )

        except Exception:
            pass

        if i % 1000 == 0:

            print(
                f"  {i:5d}/{len(files)} "
                f"eligible={len(eligible)}"
            )

    return eligible


# ============================================================
# Visualization
# ============================================================

def visualize_object(
        path,
        seed,
):
    transforms, success_mask = (
        load_grasps(path)
    )

    success_indices = np.where(
        success_mask
    )[0]

    failed_indices = np.where(
        ~success_mask
    )[0]

    success_indices = np.where(success_mask)[0]
    failed_indices = np.where(~success_mask)[0]

    success_transforms = transforms[
        success_indices
    ]

    failed_transforms = transforms[
        failed_indices
    ]

    # --------------------------------------------------------
    # Distance scales come from SUCCESSFUL grasp space.
    # This matches Experiment 1.
    # --------------------------------------------------------

    (
        success_distance,
        position_scale,
        rotation_scale,
    ) = normalized_se3_distance(
        success_transforms
    )

    # --------------------------------------------------------
    # Choose informative epsilon
    # --------------------------------------------------------

    epsilon, largest_fraction = (
        choose_epsilon(
            success_distance,
            target_largest=0.50,
        )
    )

    n_components, labels = (
        component_labels(
            success_distance,
            epsilon,
        )
    )

    labels, component_sizes = (
        relabel_components_by_size(
            labels
        )
    )

    # --------------------------------------------------------
    # MDS uses BOTH successful and failed grasps.
    #
    # Important: use the SAME scales determined from the
    # successful grasp cloud.
    # --------------------------------------------------------

    all_transforms = np.concatenate(
        [
            success_transforms,
            failed_transforms,
        ],
        axis=0,
    )

    (
        all_distance,
        _,
        _,
    ) = normalized_se3_distance(
        all_transforms,
        position_scale=position_scale,
        rotation_scale=rotation_scale,
    )

    coordinates, stress = run_mds(
        all_distance,
        seed,
    )

    actual_n_success = len(success_indices)

    success_xy = coordinates[:actual_n_success]
    failed_xy = coordinates[actual_n_success:]

    return {
        "category": category_from_path(
            path
        ),
        "file": path.name,
        "success_xy": success_xy,
        "failed_xy": failed_xy,
        "labels": labels,
        "component_sizes": component_sizes,
        "epsilon": epsilon,
        "largest_fraction": largest_fraction,
        "n_components": n_components,
        "stress": stress,
        "position_scale": position_scale,
        "rotation_scale": rotation_scale,
    }


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
        default=6,
    )

    parser.add_argument(
        "--n_success",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--n_failed",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    # Fixed seed for reproducible paper figures
    FIXED_SEED = 42
    args.seed = FIXED_SEED

    print(f"Fixed random seed: {FIXED_SEED}")
    root = Path(
        args.data_dir
    ).expanduser().resolve()

    output_dir = (
            Path("results")
            / "plots_examples"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        args.seed
    )

    files = find_h5_files(
        root
    )

    print()
    print("=" * 70)
    print("2D VISUALIZATION OF VIABLE GRASP SPACE")
    print("=" * 70)
    print(
        f"Objects found: {len(files)}"
    )

    eligible = find_eligible_objects(
        files,
        args.n_success,
        args.n_failed,
    )

    print()
    print(
        f"Eligible objects: "
        f"{len(eligible)}"
    )

    if len(eligible) < args.n_objects:
        raise RuntimeError(
            "Not enough eligible objects."
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

    results = []

    for i, (
            path,
            total_success,
            total_failed,
    ) in enumerate(
        selected,
        start=1,
    ):

        print()
        print(
            f"[{i}/{args.n_objects}] "
            f"{category_from_path(path)}"
        )

        print(
            f"      total successful = "
            f"{total_success}"
        )

        print(
            f"      total failed     = "
            f"{total_failed}"
        )

        result = visualize_object(
            path=path,
            seed=args.seed + i,
        )

        results.append(
            result
        )

        print(
            f"      epsilon          = "
            f"{result['epsilon']:.3f}"
        )

        print(
            f"      components       = "
            f"{result['n_components']}"
        )

        print(
            f"      largest fraction = "
            f"{result['largest_fraction']:.3f}"
        )

        print(
            f"      MDS stress       = "
            f"{result['stress']:.3f}"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    # --------------------------------------------------------
    # Plot each object separately
    # --------------------------------------------------------

    for i, result in enumerate(results, start=1):

        fig, ax = plt.subplots(
            figsize=(6.0, 5.5)
        )

        # ----------------------------------------------------
        # Failed grasps
        # ----------------------------------------------------

        failed_xy = result["failed_xy"]

        # Failed grasps
        ax.scatter(
            failed_xy[:, 0],
            failed_xy[:, 1],
            s=11,                 # larger crosses
            alpha=0.7,
            c="gray",
            marker="x",
            linewidths=0.8,
            label="Failed",
        )

        # ----------------------------------------------------
        # Successful grasps by ORIGINAL-SE(3) component
        # ----------------------------------------------------

        success_xy = result["success_xy"]
        labels = result["labels"]
        component_sizes = result["component_sizes"]

        max_components_to_show = 8

        for component in range(
                min(
                    len(component_sizes),
                    max_components_to_show,
                )
        ):

            mask = labels == component
            size = int(np.sum(mask))

            ax.scatter(
                success_xy[mask, 0],
                success_xy[mask, 1],
                s=11,                 # larger circles
                alpha=0.7,
                edgecolors="none",   # no border
                linewidths=0,
                label=f"C{component + 1} (n={size})",
            )

        # Small remaining components
        remaining = labels >= max_components_to_show

        if np.any(remaining):
            ax.scatter(
                success_xy[remaining, 0],
                success_xy[remaining, 1],
                s=11,
                alpha=0.7
                ,
                c="black",
                label="Other",
            )

        title = (
            f"{result['category']}\n"
            f"$\\epsilon$={result['epsilon']:.2f}, "
            f"$\\beta_0$={result['n_components']}, "
            f"$L$={result['largest_fraction']:.2f}"
        )

        ax.set_title(
            title,
            fontsize=12,
        )



        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(
            fontsize=11,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.12),
            ncol=3,
        )

        plt.tight_layout()

        output_path = (
                output_dir
                / f"grasp_space_mds_{i:02d}.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()