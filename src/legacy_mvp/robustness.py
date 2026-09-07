"""
Does viable-action geometry identify robust grasps?

For each ACRONYM object:

1. Split candidate grasps into:
       CONSTRUCTION set
       EVALUATION set

2. Use ONLY the construction set to characterize successful-grasp
   geometry.

3. Select one successful grasp using several strategies:

       Random
       Local density
       Largest component
       Component centrality

4. Evaluate the selected grasp using held-out FAILED grasps:

       failure margin =
           distance to nearest held-out failed grasp

A larger failure margin means that the selected successful grasp is
farther from known failures in SE(3), and is therefore an empirical
proxy for robustness.

Important:
    Evaluation failures are NEVER used for selection.

Usage
-----
Quick:
    python src/robustness.py grasps \
        --n_objects 100 \
        --n_repeats 20

Full:
    python src/robustness.py grasps \
        --n_objects 500 \
        --n_repeats 50

Dependencies
------------
pip install numpy pandas h5py scipy matplotlib
"""

from pathlib import Path
import argparse

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import wilcoxon


# ============================================================
# Data
# ============================================================

def find_h5_files(root):
    files = list(root.rglob("*.h5"))
    files += list(root.rglob("*.hdf5"))
    return sorted(set(files))


def category_from_path(path):
    return path.stem.split("_")[0]


def load_object(path):

    with h5py.File(path, "r") as f:

        transforms = np.asarray(
            f["grasps/transforms"]
        )

        labels = np.asarray(
            f[
                "grasps/qualities/"
                "flex/object_in_gripper"
            ]
        ).reshape(-1)

    return transforms, labels > 0


# ============================================================
# SE(3) components
# ============================================================

def positions(transforms):
    return transforms[:, :3, 3]


def rotations(transforms):
    return transforms[:, :3, :3]


def rotation_pairwise(rotations_array):
    """
    Full pairwise SO(3) geodesic distance matrix.
    """

    relative = np.einsum(
        "aik,bil->abkl",
        rotations_array,
        rotations_array,
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


def rotation_cross(rot_a, rot_b):
    """
    SO(3) geodesic distances between two different sets.
    Shape:
        len(A) x len(B)
    """

    relative = np.einsum(
        "aik,bil->abkl",
        rot_a,
        rot_b,
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


def median_nonzero(values):

    values = np.asarray(values)

    values = values[
        np.isfinite(values)
        &
        (values > 1e-12)
        ]

    if len(values) == 0:
        return 1.0

    return float(
        np.median(values)
    )


# ============================================================
# Construction-space metric
# ============================================================

def construction_distance(transforms):
    """
    Calculate normalized SE(3) distances and return the
    normalization constants.

    IMPORTANT:
        scales are estimated ONLY from construction grasps.
    """

    pos = positions(transforms)
    rot = rotations(transforms)

    dp = squareform(
        pdist(
            pos,
            metric="euclidean",
        )
    )

    dr = rotation_pairwise(rot)

    position_scale = median_nonzero(
        dp[
            np.triu_indices_from(
                dp,
                k=1,
            )
        ]
    )

    rotation_scale = median_nonzero(
        dr[
            np.triu_indices_from(
                dr,
                k=1,
            )
        ]
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


def cross_distance(
        transforms_a,
        transforms_b,
        position_scale,
        rotation_scale,
):
    """
    Normalized SE(3) distance between two sets using FIXED
    construction-set normalization.
    """

    dp = cdist(
        positions(transforms_a),
        positions(transforms_b),
        metric="euclidean",
    )

    dr = rotation_cross(
        rotations(transforms_a),
        rotations(transforms_b),
    )

    return np.sqrt(
        (dp / position_scale) ** 2
        +
        (dr / rotation_scale) ** 2
    )


# ============================================================
# Geometry-based selectors
# ============================================================

def choose_connectivity_epsilon(
        success_distance,
        target_largest=0.5,
):
    """
    Pick epsilon where largest component is approximately
    target_largest.

    This is computed using successful CONSTRUCTION grasps only.
    """

    epsilons = np.linspace(
        0.05,
        1.0,
        100,
    )

    best_epsilon = epsilons[0]
    best_error = np.inf

    for epsilon in epsilons:

        adjacency = (
                success_distance <= epsilon
        )

        np.fill_diagonal(
            adjacency,
            False,
        )

        n_components, labels = (
            connected_components(
                csr_matrix(adjacency),
                directed=False,
                return_labels=True,
            )
        )

        sizes = np.bincount(labels)

        largest_fraction = (
                sizes.max()
                /
                len(labels)
        )

        error = abs(
            largest_fraction
            - target_largest
        )

        if error < best_error:

            best_error = error
            best_epsilon = epsilon

    return float(best_epsilon)


def largest_component_indices(
        success_distance,
        epsilon,
):

    adjacency = (
            success_distance <= epsilon
    )

    np.fill_diagonal(
        adjacency,
        False,
    )

    _, labels = connected_components(
        csr_matrix(adjacency),
        directed=False,
        return_labels=True,
    )

    sizes = np.bincount(labels)

    largest_label = int(
        np.argmax(sizes)
    )

    return np.where(
        labels == largest_label
    )[0]


def local_density_scores(
        success_distance,
        k=20,
):
    """
    Density among successful construction grasps.

    Smaller mean k-NN distance = denser region.
    We negate it so larger score = denser.
    """

    n = len(success_distance)

    k = min(
        k,
        n - 1,
        )

    sorted_distances = np.sort(
        success_distance,
        axis=1,
    )

    mean_knn_distance = np.mean(
        sorted_distances[
            :,
            1:k + 1,
        ],
        axis=1,
    )

    return -mean_knn_distance


def component_centrality_selector(
        success_distance,
        component_indices,
):
    """
    Select the medoid of the largest connected component.

    This is a very simple topology/geometry-aware strategy.
    """

    component_distance = (
        success_distance[
            np.ix_(
                component_indices,
                component_indices,
            )
        ]
    )

    mean_distance = np.mean(
        component_distance,
        axis=1,
    )

    local_index = int(
        np.argmin(mean_distance)
    )

    return int(
        component_indices[
            local_index
        ]
    )


# ============================================================
# Evaluation
# ============================================================

def failure_margin(
        selected_transform,
        evaluation_failures,
        position_scale,
        rotation_scale,
):
    """
    Distance from selected successful grasp to nearest
    HELD-OUT failed grasp.
    """

    distances = cross_distance(
        selected_transform[None, ...],
        evaluation_failures,
        position_scale,
        rotation_scale,
    )[0]

    return float(
        np.min(distances)
    )


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
        default=20,
    )

    parser.add_argument(
        "--construction_fraction",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--k_density",
        type=int,
        default=20,
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

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    rng = np.random.default_rng(
        args.seed
    )

    files = find_h5_files(
        root
    )

    print()
    print("=" * 72)
    print("VIABLE-ACTION GEOMETRY -> ROBUST GRASP SELECTION")
    print("=" * 72)

    print(
        f"Objects found: {len(files)}"
    )

    # ========================================================
    # Find eligible objects
    # ========================================================

    eligible = []

    print(
        "Scanning objects..."
    )

    for i, path in enumerate(
            files,
            start=1,
    ):

        try:

            _, labels = load_object(
                path
            )

            n_success = int(
                np.sum(labels)
            )

            n_failed = int(
                np.sum(~labels)
            )

            # Enough of both classes for a meaningful split.
            if (
                    n_success >= 200
                    and
                    n_failed >= 200
            ):
                eligible.append(path)

        except Exception:
            pass

        if i % 1000 == 0:

            print(
                f"  {i:5d}/{len(files)} "
                f"eligible={len(eligible)}"
            )

    print()
    print(
        f"Eligible objects: "
        f"{len(eligible)}"
    )

    if len(eligible) < args.n_objects:

        raise RuntimeError(
            f"Requested {args.n_objects} objects "
            f"but only {len(eligible)} eligible."
        )

    selected_objects = rng.choice(
        eligible,
        size=args.n_objects,
        replace=False,
    )

    rows = []

    # ========================================================
    # Experiment
    # ========================================================

    for object_number, path in enumerate(
            selected_objects,
            start=1,
    ):

        path = Path(path)

        transforms, labels = load_object(
            path
        )

        success_indices_all = np.where(
            labels
        )[0]

        failure_indices_all = np.where(
            ~labels
        )[0]

        print()
        print(
            f"[{object_number:3d}/{args.n_objects}] "
            f"{category_from_path(path)} "
            f"| success={len(success_indices_all)} "
            f"failure={len(failure_indices_all)}"
        )

        for repeat in range(
                args.n_repeats
        ):

            # ------------------------------------------------
            # Split SUCCESS grasps
            # ------------------------------------------------

            success_perm = rng.permutation(
                success_indices_all
            )

            n_success_construction = int(
                len(success_perm)
                *
                args.construction_fraction
            )

            success_construction_idx = (
                success_perm[
                    :n_success_construction
                ]
            )

            # Remaining successes are deliberately unused.
            # We only need held-out failures for the margin.

            # ------------------------------------------------
            # Split FAILURE grasps
            # ------------------------------------------------

            failure_perm = rng.permutation(
                failure_indices_all
            )

            n_failure_construction = int(
                len(failure_perm)
                *
                args.construction_fraction
            )

            failure_construction_idx = (
                failure_perm[
                    :n_failure_construction
                ]
            )

            failure_evaluation_idx = (
                failure_perm[
                    n_failure_construction:
                ]
            )

            # ------------------------------------------------
            # Construction metric
            #
            # Scale is estimated using the entire construction
            # candidate set, WITHOUT evaluation failures.
            # ------------------------------------------------

            construction_indices = (
                np.concatenate(
                    [
                        success_construction_idx,
                        failure_construction_idx,
                    ]
                )
            )

            construction_transforms = (
                transforms[
                    construction_indices
                ]
            )

            (
                _,
                position_scale,
                rotation_scale,
            ) = construction_distance(
                construction_transforms
            )

            success_transforms = (
                transforms[
                    success_construction_idx
                ]
            )

            evaluation_failures = (
                transforms[
                    failure_evaluation_idx
                ]
            )

            # ------------------------------------------------
            # Successful grasp geometry
            # ------------------------------------------------

            success_distance = cross_distance(
                success_transforms,
                success_transforms,
                position_scale,
                rotation_scale,
            )

            np.fill_diagonal(
                success_distance,
                0.0,
            )

            # ------------------------------------------------
            # Selectors
            # ------------------------------------------------

            # 1. Random successful grasp
            random_selected = int(
                rng.integers(
                    len(success_transforms)
                )
            )

            # 2. Densest successful region
            density_scores = (
                local_density_scores(
                    success_distance,
                    k=args.k_density,
                )
            )

            density_selected = int(
                np.argmax(
                    density_scores
                )
            )

            # 3. Any point in largest component:
            #    choose its medoid for deterministic selection.
            epsilon = (
                choose_connectivity_epsilon(
                    success_distance,
                    target_largest=0.5,
                )
            )

            largest_indices = (
                largest_component_indices(
                    success_distance,
                    epsilon,
                )
            )

            # Random member of largest component.
            largest_component_selected = int(
                rng.choice(
                    largest_indices
                )
            )

            # 4. Central point of largest component.
            central_selected = (
                component_centrality_selector(
                    success_distance,
                    largest_indices,
                )
            )

            selections = {
                "random":
                    random_selected,

                "density":
                    density_selected,

                "largest_component":
                    largest_component_selected,

                "component_central":
                    central_selected,
            }

            # ------------------------------------------------
            # Held-out evaluation
            # ------------------------------------------------

            margins = {}

            for method, selected_index in (
                    selections.items()
            ):

                margin = failure_margin(
                    success_transforms[
                        selected_index
                    ],
                    evaluation_failures,
                    position_scale,
                    rotation_scale,
                )

                margins[method] = margin

            rows.append({
                "object_id":
                    path.stem,

                "category":
                    category_from_path(path),

                "repeat":
                    repeat,

                "success_fraction":
                    float(np.mean(labels)),

                "epsilon":
                    epsilon,

                "largest_component_fraction":
                    len(largest_indices)
                    /
                    len(success_transforms),

                "random_margin":
                    margins["random"],

                "density_margin":
                    margins["density"],

                "largest_component_margin":
                    margins[
                        "largest_component"
                    ],

                "component_central_margin":
                    margins[
                        "component_central"
                    ],
            })

            print(
                f"    repeat "
                f"{repeat + 1:2d}/{args.n_repeats}",
                end="\r",
                flush=True,
            )

        print()

    # ========================================================
    # Save raw results
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    csv_path = (
            output_dir
            / "robust_grasp_selection.csv"
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    # ========================================================
    # Aggregate per object
    #
    # IMPORTANT:
    # Statistical unit = object, not individual repeat.
    # ========================================================

    margin_columns = [
        "random_margin",
        "density_margin",
        "largest_component_margin",
        "component_central_margin",
    ]

    object_df = (
        df.groupby(
            "object_id"
        )[margin_columns]
        .mean()
        .reset_index()
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 72)
    print("HELD-OUT FAILURE MARGIN")
    print("=" * 72)

    labels_pretty = {
        "random_margin":
            "Random",

        "density_margin":
            "Local density",

        "largest_component_margin":
            "Largest component",

        "component_central_margin":
            "Component centrality",
    }

    random_mean = float(
        object_df[
            "random_margin"
        ].mean()
    )

    for column in margin_columns:

        values = object_df[
            column
        ].values

        mean = float(
            np.mean(values)
        )

        median = float(
            np.median(values)
        )

        improvement = (
                              mean / random_mean - 1.0
                      ) * 100.0

        print()
        print(
            f"{labels_pretty[column]:22s}"
        )

        print(
            f"    mean margin:      "
            f"{mean:.4f}"
        )

        print(
            f"    median margin:    "
            f"{median:.4f}"
        )

        if column != "random_margin":

            print(
                f"    vs random:       "
                f"{improvement:+.1f}%"
            )

            differences = (
                    object_df[column].values
                    -
                    object_df[
                        "random_margin"
                    ].values
            )

            nonzero = (
                    np.abs(differences)
                    > 1e-12
            )

            if np.sum(nonzero) >= 5:

                statistic, p_value = (
                    wilcoxon(
                        object_df[column],
                        object_df[
                            "random_margin"
                        ],
                        alternative="greater",
                    )
                )

                print(
                    f"    Wilcoxon p:       "
                    f"{p_value:.3g}"
                )

            wins = np.mean(
                object_df[column]
                >
                object_df[
                    "random_margin"
                ]
            )

            print(
                f"    objects improved: "
                f"{100 * wins:.1f}%"
            )

    # ========================================================
    # Figure
    # ========================================================

    means = [
        object_df[column].mean()
        for column in margin_columns
    ]

    errors = [
        object_df[column].std(ddof=1)
        /
        np.sqrt(len(object_df))
        for column in margin_columns
    ]

    names = [
        "Random",
        "Local\ndensity",
        "Largest\ncomponent",
        "Component\ncentrality",
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        names,
        means,
        yerr=errors,
        capsize=5,
    )

    ax.set_ylabel(
        "Distance to nearest held-out failure"
    )

    ax.set_title(
        "Can Viable-Action Geometry Select More Robust Grasps?"
    )

    plt.tight_layout()

    figure_path = (
            output_dir
            / "robust_grasp_selection.png"
    )

    plt.savefig(
        figure_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Paired scatter: strongest geometry method vs random
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    x = object_df[
        "random_margin"
    ].values

    y = object_df[
        "component_central_margin"
    ].values

    ax.scatter(
        x,
        y,
        alpha=0.65,
    )

    low = min(
        np.min(x),
        np.min(y),
    )

    high = max(
        np.max(x),
        np.max(y),
    )

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "Random successful grasp"
    )

    ax.set_ylabel(
        "Component-central grasp"
    )

    ax.set_title(
        "Held-out Failure Margin by Object"
    )

    plt.tight_layout()

    scatter_path = (
            output_dir
            / "robust_grasp_paired.png"
    )

    plt.savefig(
        scatter_path,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Save object-level table
    # ========================================================

    object_csv_path = (
            output_dir
            / "robust_grasp_selection_by_object.csv"
    )

    object_df.to_csv(
        object_csv_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(csv_path)
    print(object_csv_path)
    print(figure_path)
    print(scatter_path)


if __name__ == "__main__":
    main()