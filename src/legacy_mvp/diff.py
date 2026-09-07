"""
Experiment: Does viable-action structure predict grasp-search difficulty?

Scientific question
-------------------
Does the organization of successful actions predict how difficult it is
to find a ROBUST successful grasp, beyond simply knowing the fraction
of successful grasps?

For each ACRONYM object:

1. Load all candidate grasps and binary success labels.
2. Compute a normalized SE(3) distance matrix.
3. Characterize the successful-grasp connectivity structure.
4. Define robust successes using local success density.
5. Run many local-search trials from random starting grasps.
6. Measure search difficulty:
       - evaluations required
       - probability of success within budget
7. Test whether structural descriptors explain difficulty beyond
   raw success fraction.

The experiment compares:

    Model A: difficulty ~ success_fraction

    Model B: difficulty ~ structural_features

    Model C: difficulty ~ success_fraction + structural_features

If Model C substantially improves over Model A under object-level
cross-validation, structure contains information about search difficulty
beyond merely "how many good actions exist."

Usage
-----
Quick:
    python src/diff.py grasps \
        --n_objects 100 \
        --n_trials 100

Larger:
    python src/diff.py grasps \
        --n_objects 500 \
        --n_trials 200

Dependencies
------------
    pip install numpy pandas h5py scipy scikit-learn matplotlib
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
from scipy.stats import spearmanr

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


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
            f["grasps/qualities/flex/object_in_gripper"]
        ).reshape(-1)

    return transforms, (labels > 0).astype(int)


# ============================================================
# SE(3) metric
# ============================================================

def translation_distance_matrix(transforms):

    positions = transforms[:, :3, 3]

    return squareform(
        pdist(
            positions,
            metric="euclidean",
        )
    )


def rotation_distance_matrix(transforms):

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
        &
        (values > 1e-12)
        ]

    if len(values) == 0:
        return 1.0

    return float(
        np.median(values)
    )


def normalized_se3_distance(transforms):

    dp = translation_distance_matrix(
        transforms
    )

    dr = rotation_distance_matrix(
        transforms
    )

    position_scale = upper_median(dp)
    rotation_scale = upper_median(dr)

    distance = np.sqrt(
        (dp / position_scale) ** 2
        +
        (dr / rotation_scale) ** 2
    )

    np.fill_diagonal(
        distance,
        0.0,
    )

    return distance


# ============================================================
# Structural descriptors
# ============================================================

def connectivity_features(
        success_distance,
        epsilons,
):

    n = len(success_distance)

    beta_curve = []
    largest_curve = []

    for epsilon in epsilons:

        adjacency = (
                success_distance <= epsilon
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

        sizes = np.bincount(
            labels
        )

        beta_curve.append(
            n_components / n
        )

        largest_curve.append(
            sizes.max() / n
        )

    beta_curve = np.asarray(
        beta_curve
    )

    largest_curve = np.asarray(
        largest_curve
    )

    beta_auc = np.trapezoid(
        beta_curve,
        epsilons,
    )

    largest_auc = np.trapezoid(
        largest_curve,
        epsilons,
    )

    idx90 = np.where(
        largest_curve >= 0.90
    )[0]

    if len(idx90):
        epsilon_90 = float(
            epsilons[idx90[0]]
        )
    else:
        epsilon_90 = float(
            epsilons[-1]
        )

    idx50 = np.where(
        largest_curve >= 0.50
    )[0]

    if len(idx50):
        epsilon_50 = float(
            epsilons[idx50[0]]
        )
    else:
        epsilon_50 = float(
            epsilons[-1]
        )

    return {
        "beta_auc": float(beta_auc),
        "largest_auc": float(largest_auc),
        "epsilon_50": epsilon_50,
        "epsilon_90": epsilon_90,
    }


# ============================================================
# Local neighborhoods / robustness
# ============================================================

def knn_indices(
        distance_matrix,
        k,
):
    """
    k nearest OTHER candidate grasps for every grasp.
    """

    order = np.argsort(
        distance_matrix,
        axis=1,
    )

    return order[
        :,
        1:k + 1,
    ]


def local_success_density(
        neighbor_indices,
        labels,
):
    """
    Fraction of successful grasps among each grasp's k nearest
    neighbors.
    """

    return np.mean(
        labels[
            neighbor_indices
        ],
        axis=1,
    )


def define_robust_successes(
        labels,
        density,
        threshold,
):
    """
    A robust solution must:
        1. itself succeed
        2. lie in a locally success-rich neighborhood
    """

    return (
            (labels == 1)
            &
            (density >= threshold)
    )


# ============================================================
# Search
# ============================================================

def run_local_search(
        distance_matrix,
        labels,
        density,
        robust_mask,
        neighbor_indices,
        rng,
        budget,
        proposal_size,
):
    """
    Simple local search.

    Start from a random grasp.

    At each step:
        - evaluate current grasp
        - if robust success: stop
        - consider nearby unvisited candidates
        - evaluate up to proposal_size candidates
        - move to candidate with highest estimated local
          success density

    IMPORTANT:
    The search algorithm is intentionally simple. The purpose
    is not to beat grasp planners. It is to provide a fixed
    search process whose difficulty may depend on action-space
    organization.

    We count each inspected candidate as one evaluation.

    The precomputed density is used as a local search heuristic.
    """

    n = len(labels)

    start = int(
        rng.integers(n)
    )

    visited = set()
    current = start

    evaluations = 0

    while evaluations < budget:

        if current not in visited:

            visited.add(current)
            evaluations += 1

            if robust_mask[current]:

                return (
                    True,
                    evaluations,
                )

        candidates = [
            int(x)
            for x in neighbor_indices[current]
            if int(x) not in visited
        ]

        if not candidates:

            # Restart somewhere unexplored.
            remaining = np.array(
                [
                    i
                    for i in range(n)
                    if i not in visited
                ],
                dtype=int,
            )

            if len(remaining) == 0:
                break

            current = int(
                rng.choice(remaining)
            )

            continue

        if len(candidates) > proposal_size:

            candidates = list(
                rng.choice(
                    candidates,
                    size=proposal_size,
                    replace=False,
                )
            )

        # These candidate evaluations count.
        best_candidate = None
        best_density = -np.inf

        for candidate in candidates:

            if evaluations >= budget:
                break

            candidate = int(candidate)

            if candidate not in visited:

                visited.add(candidate)
                evaluations += 1

                if robust_mask[candidate]:

                    return (
                        True,
                        evaluations,
                    )

            if density[candidate] > best_density:

                best_density = density[candidate]
                best_candidate = candidate

        if best_candidate is None:
            break

        current = best_candidate

    return (
        False,
        evaluations,
    )


def run_random_search(
        robust_mask,
        rng,
        budget,
):
    """
    Random-search baseline.
    """

    n = len(robust_mask)

    order = rng.permutation(n)

    for evaluation, idx in enumerate(
            order[:budget],
            start=1,
    ):

        if robust_mask[idx]:

            return (
                True,
                evaluation,
            )

    return (
        False,
        min(budget, n),
    )


# ============================================================
# Cross-validated prediction
# ============================================================

def cv_r2(
        dataframe,
        feature_names,
        target,
        seed,
):
    X = dataframe[
        feature_names
    ].values

    y = dataframe[
        target
    ].values

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    model = make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2",
    )

    return (
        float(np.mean(scores)),
        float(np.std(scores)),
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
        "--n_trials",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--proposal_size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--robust_threshold",
        type=float,
        default=0.8,
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
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    rng = np.random.default_rng(
        args.seed
    )

    root = Path(
        args.data_dir
    ).expanduser().resolve()

    output_dir = Path(
        "results"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    epsilons = np.linspace(
        args.eps_min,
        args.eps_max,
        args.n_eps,
    )

    files = find_h5_files(
        root
    )

    print()
    print("=" * 72)
    print("VIABLE-ACTION STRUCTURE -> SEARCH DIFFICULTY")
    print("=" * 72)

    print(
        f"Objects found: {len(files)}"
    )

    # --------------------------------------------------------
    # Eligible objects
    # --------------------------------------------------------

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
                labels.sum()
            )

            n_failed = int(
                len(labels) - n_success
            )

            # Need both classes.
            if (
                    n_success >= 100
                    and
                    n_failed >= 100
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
            f"Only {len(eligible)} eligible objects."
        )

    selected = rng.choice(
        eligible,
        size=args.n_objects,
        replace=False,
    )

    rows = []

    # ========================================================
    # Objects
    # ========================================================

    for object_number, path in enumerate(
            selected,
            start=1,
    ):

        path = Path(path)

        transforms, labels = load_object(
            path
        )

        print()
        print(
            f"[{object_number:3d}/{args.n_objects}] "
            f"{category_from_path(path)}"
        )

        # ----------------------------------------------------
        # Full candidate-space metric
        # ----------------------------------------------------

        distance_matrix = (
            normalized_se3_distance(
                transforms
            )
        )

        success_fraction = float(
            np.mean(labels)
        )

        success_indices = np.where(
            labels == 1
        )[0]

        # ----------------------------------------------------
        # Structural features from successful grasps only
        # ----------------------------------------------------

        success_distance = (
            distance_matrix[
                np.ix_(
                    success_indices,
                    success_indices,
                )
            ]
        )

        structure = (
            connectivity_features(
                success_distance,
                epsilons,
            )
        )

        # ----------------------------------------------------
        # Local density
        # ----------------------------------------------------

        neighbors = knn_indices(
            distance_matrix,
            args.k_neighbors,
        )

        density = (
            local_success_density(
                neighbors,
                labels,
            )
        )

        robust_mask = (
            define_robust_successes(
                labels,
                density,
                args.robust_threshold,
            )
        )

        robust_fraction = float(
            np.mean(robust_mask)
        )

        robust_given_success = float(
            np.sum(robust_mask)
            /
            max(
                np.sum(labels),
                1,
            )
        )

        # ----------------------------------------------------
        # Search trials
        # ----------------------------------------------------

        local_evaluations = []
        local_successes = []

        random_evaluations = []
        random_successes = []

        for _ in range(
                args.n_trials
        ):

            success, evaluations = (
                run_local_search(
                    distance_matrix=distance_matrix,
                    labels=labels,
                    density=density,
                    robust_mask=robust_mask,
                    neighbor_indices=neighbors,
                    rng=rng,
                    budget=args.budget,
                    proposal_size=args.proposal_size,
                )
            )

            local_successes.append(
                int(success)
            )

            local_evaluations.append(
                evaluations
            )

            success, evaluations = (
                run_random_search(
                    robust_mask=robust_mask,
                    rng=rng,
                    budget=args.budget,
                )
            )

            random_successes.append(
                int(success)
            )

            random_evaluations.append(
                evaluations
            )

        local_search_success = float(
            np.mean(local_successes)
        )

        random_search_success = float(
            np.mean(random_successes)
        )

        mean_local_evaluations = float(
            np.mean(local_evaluations)
        )

        mean_random_evaluations = float(
            np.mean(random_evaluations)
        )

        print(
            f"    success fraction       "
            f"{success_fraction:.3f}"
        )

        print(
            f"    robust fraction        "
            f"{robust_fraction:.3f}"
        )

        print(
            f"    local search success   "
            f"{local_search_success:.3f}"
        )

        print(
            f"    random search success  "
            f"{random_search_success:.3f}"
        )

        print(
            f"    mean local evals       "
            f"{mean_local_evaluations:.1f}"
        )

        rows.append({
            "object_id": path.stem,
            "category": category_from_path(path),

            "success_fraction":
                success_fraction,

            "robust_fraction":
                robust_fraction,

            "robust_given_success":
                robust_given_success,

            "beta_auc":
                structure["beta_auc"],

            "largest_auc":
                structure["largest_auc"],

            "epsilon_50":
                structure["epsilon_50"],

            "epsilon_90":
                structure["epsilon_90"],

            "local_search_success":
                local_search_success,

            "random_search_success":
                random_search_success,

            "mean_local_evaluations":
                mean_local_evaluations,

            "mean_random_evaluations":
                mean_random_evaluations,
        })

    # ========================================================
    # Analysis
    # ========================================================

    df = pd.DataFrame(rows)

    output_csv = (
            output_dir
            / "search_difficulty_results.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    print()
    print("=" * 72)
    print("PREDICTING LOCAL SEARCH DIFFICULTY")
    print("=" * 72)

    # Difficulty = evaluations.
    target = "mean_local_evaluations"

    baseline_features = [
        "success_fraction",
    ]

    structure_features = [
        "beta_auc",
        "largest_auc",
        "epsilon_50",
        "epsilon_90",
    ]

    combined_features = (
            baseline_features
            +
            structure_features
    )

    baseline_r2 = cv_r2(
        df,
        baseline_features,
        target,
        args.seed,
    )

    structure_r2 = cv_r2(
        df,
        structure_features,
        target,
        args.seed,
    )

    combined_r2 = cv_r2(
        df,
        combined_features,
        target,
        args.seed,
    )

    print()
    print(
        "5-fold object-level CV R²"
    )

    print(
        f"Success fraction only:   "
        f"{baseline_r2[0]: .3f} "
        f"± {baseline_r2[1]:.3f}"
    )

    print(
        f"Structure only:          "
        f"{structure_r2[0]: .3f} "
        f"± {structure_r2[1]:.3f}"
    )

    print(
        f"Success + structure:     "
        f"{combined_r2[0]: .3f} "
        f"± {combined_r2[1]:.3f}"
    )

    print()
    print(
        "Incremental structural value:"
    )

    print(
        f"ΔR² = "
        f"{combined_r2[0] - baseline_r2[0]:+.3f}"
    )

    # ========================================================
    # Simple correlations
    # ========================================================

    print()
    print("=" * 72)
    print("SPEARMAN CORRELATIONS WITH SEARCH DIFFICULTY")
    print("=" * 72)

    correlation_features = [
        "success_fraction",
        "robust_fraction",
        "beta_auc",
        "largest_auc",
        "epsilon_50",
        "epsilon_90",
    ]

    for feature in correlation_features:

        rho, p = spearmanr(
            df[feature],
            df[target],
        )

        print(
            f"{feature:22s} "
            f"rho={rho:+.3f} "
            f"p={p:.3g}"
        )

    # ========================================================
    # Figure 1
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.scatter(
        df["success_fraction"],
        df[target],
        alpha=0.7,
    )

    ax.set_xlabel(
        "Fraction of successful grasps"
    )

    ax.set_ylabel(
        "Mean evaluations to robust success"
    )

    ax.set_title(
        "Does Success Fraction Explain Search Difficulty?"
    )

    plt.tight_layout()

    path1 = (
            output_dir
            / "difficulty_vs_success_fraction.png"
    )

    plt.savefig(
        path1,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Figure 2
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.scatter(
        df["beta_auc"],
        df[target],
        alpha=0.7,
    )

    ax.set_xlabel(
        r"Connectivity fragmentation "
        r"($\beta_0$ AUC)"
    )

    ax.set_ylabel(
        "Mean evaluations to robust success"
    )

    ax.set_title(
        "Viable-Action Structure vs Search Difficulty"
    )

    plt.tight_layout()

    path2 = (
            output_dir
            / "difficulty_vs_structure.png"
    )

    plt.savefig(
        path2,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    # ========================================================
    # Figure 3: model comparison
    # ========================================================

    names = [
        "Success\nfraction",
        "Structure",
        "Success +\nstructure",
    ]

    values = [
        baseline_r2[0],
        structure_r2[0],
        combined_r2[0],
    ]

    errors = [
        baseline_r2[1],
        structure_r2[1],
        combined_r2[1],
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        names,
        values,
        yerr=errors,
        capsize=5,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_ylabel(
        "5-fold cross-validated R²"
    )

    ax.set_title(
        "Does Action-Space Structure Add Predictive Information?"
    )

    plt.tight_layout()

    path3 = (
            output_dir
            / "search_difficulty_model_comparison.png"
    )

    plt.savefig(
        path3,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close()

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(output_csv)
    print(path1)
    print(path2)
    print(path3)


if __name__ == "__main__":
    main()