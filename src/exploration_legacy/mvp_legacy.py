import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ============================================================
# DATA
# ============================================================

def find_objects(root):
    root = Path(root)
    return sorted(
        list(root.rglob("*.h5")) +
        list(root.rglob("*.hdf5"))
    )


def load_object(path):
    with h5py.File(path, "r") as f:
        transforms = f["grasps/transforms"][:]
        labels = (
            f["grasps/qualities/flex/object_in_gripper"][:]
            .astype(int)
        )

    return transforms, labels


# ============================================================
# SE(3) DISTANCE
# ============================================================

def grasp_distance_matrix(transforms):
    """
    Normalized SE(3) distance.

    IMPORTANT:
    Scale is computed ONCE using all candidate poses.
    Labels are not involved, so this does not leak outcomes.
    """

    p = transforms[:, :3, 3]
    R = transforms[:, :3, :3]

    # Translation
    dp = np.linalg.norm(
        p[:, None, :] - p[None, :, :],
        axis=-1,
        )

    # Relative rotation R_i^T R_j
    rel = np.einsum(
        "aik,bil->abkl",
        R,
        R,
    )

    traces = np.trace(
        rel,
        axis1=-2,
        axis2=-1,
    )

    cos_angle = np.clip(
        (traces - 1.0) / 2.0,
        -1.0,
        1.0,
        )

    dr = np.arccos(cos_angle)

    dp_nz = dp[dp > 0]
    dr_nz = dr[dr > 0]

    scale_p = (
        np.median(dp_nz)
        if len(dp_nz)
        else 1.0
    )

    scale_r = (
        np.median(dr_nz)
        if len(dr_nz)
        else 1.0
    )

    return np.sqrt(
        (dp / scale_p) ** 2 +
        (dr / scale_r) ** 2
    ).astype(np.float32)


# ============================================================
# CONNECTIVITY SIGNATURE
# ============================================================

def connectivity_signature(indices, D, eps_values):
    """
    Connectivity signature of a set of successful grasps.

    B(eps) = beta_0(eps) / N
    L(eps) = largest component / N
    """

    indices = np.asarray(indices, dtype=int)
    n = len(indices)

    if n == 0:
        return (
            np.ones(len(eps_values)),
            np.zeros(len(eps_values)),
        )

    if n == 1:
        return (
            np.ones(len(eps_values)),
            np.ones(len(eps_values)),
        )

    sub = D[np.ix_(indices, indices)]

    beta = []
    largest = []

    for eps in eps_values:

        adjacency = sub <= eps

        graph = csr_matrix(adjacency)

        n_components, component_labels = (
            connected_components(
                graph,
                directed=False,
            )
        )

        sizes = np.bincount(component_labels)

        beta.append(
            n_components / n
        )

        largest.append(
            sizes.max() / n
        )

    return (
        np.asarray(beta),
        np.asarray(largest),
    )


def signature_distance(
        beta_a,
        largest_a,
        beta_b,
        largest_b,
):
    """
    Equal-weight distance between two multiscale
    connectivity signatures.
    """

    d_beta = np.mean(
        np.abs(beta_a - beta_b)
    )

    d_largest = np.mean(
        np.abs(largest_a - largest_b)
    )

    total = 0.5 * (
            d_beta + d_largest
    )

    return (
        float(total),
        float(d_beta),
        float(d_largest),
    )


# ============================================================
# LOCAL SUCCESS MODEL
# ============================================================

def local_success_probability(
        candidate_idx,
        queried_idx,
        labels,
        D,
        k=10,
):
    """
    kNN estimate using ONLY queried labels.
    """

    candidate_idx = np.asarray(
        candidate_idx,
        dtype=int,
    )

    queried_idx = np.asarray(
        queried_idx,
        dtype=int,
    )

    if len(queried_idx) == 0:
        return np.full(
            len(candidate_idx),
            0.5,
        )

    k = min(k, len(queried_idx))

    distances = D[
        np.ix_(
            candidate_idx,
            queried_idx,
        )
    ]

    nearest = np.argpartition(
        distances,
        kth=k - 1,
        axis=1,
    )[:, :k]

    queried_labels = labels[
        queried_idx
    ]

    # Distance weighting
    d_near = np.take_along_axis(
        distances,
        nearest,
        axis=1,
    )

    weights = 1.0 / (
            d_near + 1e-6
    )

    y_near = queried_labels[
        nearest
    ]

    return (
            (weights * y_near).sum(axis=1)
            /
            weights.sum(axis=1)
    )


# ============================================================
# BASIC STRATEGIES
# ============================================================

def choose_random(
        candidate_idx,
        rng,
):
    return int(
        rng.choice(candidate_idx)
    )


def choose_fps(
        candidate_idx,
        queried_idx,
        D,
        rng,
):
    """
    Pure geometric farthest-point sampling.
    """

    if len(queried_idx) == 0:
        return choose_random(
            candidate_idx,
            rng,
        )

    distances = D[
        np.ix_(
            candidate_idx,
            queried_idx,
        )
    ]

    nearest = distances.min(axis=1)

    return int(
        candidate_idx[
            np.argmax(nearest)
        ]
    )


def choose_local(
        candidate_idx,
        queried_idx,
        labels,
        D,
        rng,
):
    """
    Strong simple local baseline:
    choose the candidate with highest predicted
    probability of success.

    Tiny geometric term breaks ties / reduces
    repeated sampling of identical neighborhoods.
    """

    if len(queried_idx) < 2:
        return choose_fps(
            candidate_idx,
            queried_idx,
            D,
            rng,
        )

    p = local_success_probability(
        candidate_idx,
        queried_idx,
        labels,
        D,
    )

    novelty = D[
        np.ix_(
            candidate_idx,
            queried_idx,
        )
    ].min(axis=1)

    if novelty.max() > 0:
        novelty = (
                novelty / novelty.max()
        )

    score = (
            p +
            0.05 * novelty
    )

    return int(
        candidate_idx[
            np.argmax(score)
        ]
    )


# ============================================================
# CONNECTIVITY-AWARE ACQUISITION
# ============================================================

def components_at_epsilon(
        success_idx,
        D,
        eps,
):
    """
    Return component label for every observed
    successful grasp.
    """

    success_idx = np.asarray(
        success_idx,
        dtype=int,
    )

    if len(success_idx) == 0:
        return np.array(
            [],
            dtype=int,
        )

    if len(success_idx) == 1:
        return np.array(
            [0],
            dtype=int,
        )

    sub = D[
        np.ix_(
            success_idx,
            success_idx,
        )
    ]

    graph = csr_matrix(
        sub <= eps
    )

    _, labels = connected_components(
        graph,
        directed=False,
    )

    return labels


def choose_connectivity_ms(
        candidate_idx,
        queried_idx,
        labels,
        D,
        eps_values,
        rng,
):
    """
    Connectivity-aware exploration.

    Main idea:
        Query points whose outcomes are expected
        to be informative about the CURRENT
        multiscale component structure.

    Candidate receives information value if,
    were it successful, it would:

      1. extend a known component near its boundary;
      2. bridge two or more currently separate
         components;
      3. potentially seed a new component.

    Crucially:
      - validation labels are never used;
      - candidate's true label is never used;
      - score uses only geometry + queried labels.
    """

    queried_idx = np.asarray(
        queried_idx,
        dtype=int,
    )

    candidate_idx = np.asarray(
        candidate_idx,
        dtype=int,
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    if len(queried_idx) == 0:
        return choose_random(
            candidate_idx,
            rng,
        )

    success_idx = queried_idx[
        labels[queried_idx] == 1
        ]

    # Need successful points before connectivity exists.
    if len(success_idx) < 2:
        return choose_fps(
            candidate_idx,
            queried_idx,
            D,
            rng,
        )

    # --------------------------------------------------------
    # Probability of candidate being successful
    # --------------------------------------------------------

    p_success = local_success_probability(
        candidate_idx,
        queried_idx,
        labels,
        D,
    )

    candidate_to_success = D[
        np.ix_(
            candidate_idx,
            success_idx,
        )
    ]

    structural = np.zeros(
        len(candidate_idx),
        dtype=float,
    )

    # --------------------------------------------------------
    # Multiscale structural information
    # --------------------------------------------------------

    for eps in eps_values:

        comp_labels = (
            components_at_epsilon(
                success_idx,
                D,
                eps,
            )
        )

        within = (
                candidate_to_success
                <= eps
        )

        for i in range(
                len(candidate_idx)
        ):

            neighbors = np.flatnonzero(
                within[i]
            )

            # ----------------------------------------------
            # CASE 1:
            # Candidate is outside all known components.
            #
            # Do NOT automatically give maximum reward.
            # Reward is largest near the frontier and
            # decays when extremely far away.
            # ----------------------------------------------

            if len(neighbors) == 0:

                dmin = (
                    candidate_to_success[
                        i
                    ].min()
                )

                #
                # Frontier score:
                # maximal around ~1.5 epsilon.
                #
                target = 1.5 * eps

                frontier = np.exp(
                    -(
                             (dmin - target)
                             / (eps + 1e-8)
                     ) ** 2
                )

                structural[i] += (
                        0.5 * frontier
                )

                continue

            # ----------------------------------------------
            # Candidate touches known successful components
            # ----------------------------------------------

            touched = np.unique(
                comp_labels[
                    neighbors
                ]
            )

            # ----------------------------------------------
            # CASE 2:
            # Bridge.
            #
            # This is the strongest connectivity event.
            # ----------------------------------------------

            if len(touched) >= 2:

                structural[i] += (
                        len(touched) - 1
                )

                continue

            # ----------------------------------------------
            # CASE 3:
            # Candidate extends exactly one component.
            #
            # Prefer boundary points rather than points
            # deep inside a known component.
            # ----------------------------------------------

            dmin = (
                candidate_to_success[
                    i
                ].min()
            )

            boundary = (
                    dmin / (eps + 1e-8)
            )

            boundary = np.clip(
                boundary,
                0.0,
                1.0,
            )

            structural[i] += (
                    0.25 * boundary
            )

    structural /= len(
        eps_values
    )

    # --------------------------------------------------------
    # Normalize structural information
    # --------------------------------------------------------

    s_min = structural.min()
    s_max = structural.max()

    if s_max > s_min:
        structural = (
                (structural - s_min)
                /
                (s_max - s_min)
        )
    else:
        structural[:] = 0.0

    # --------------------------------------------------------
    # Uncertainty of local success model
    #
    # Maximum at p = 0.5.
    # Querying p ~ 0 or p ~ 1 is less informative.
    # --------------------------------------------------------

    uncertainty = (
            1.0 -
            2.0 * np.abs(
        p_success - 0.5
    )
    )

    uncertainty = np.clip(
        uncertainty,
        0.0,
        1.0,
    )

    # --------------------------------------------------------
    # FINAL SCORE
    #
    # Structural information is primary.
    #
    # Uncertainty is secondary.
    #
    # We intentionally DO NOT maximize success probability.
    # The objective is reconstruction, not reward.
    # --------------------------------------------------------

    score = (
            structural
            +
            0.25 * uncertainty
    )

    return int(
        candidate_idx[
            np.argmax(score)
        ]
    )


# ============================================================
# SECONDARY COVERAGE METRIC
# ============================================================

def validation_coverage(
        D,
        discovered_successes,
        validation_idx,
        labels,
        eps_values,
):
    """
    Secondary metric:
    multiscale coverage of successful validation grasps.
    """

    validation_success = (
        validation_idx[
            labels[
                validation_idx
            ] == 1
            ]
    )

    if len(validation_success) == 0:
        return np.nan

    if len(discovered_successes) == 0:
        return 0.0

    nearest = D[
        np.ix_(
            validation_success,
            discovered_successes,
        )
    ].min(axis=1)

    coverage = [
        np.mean(
            nearest <= eps
        )
        for eps in eps_values
    ]

    return float(
        np.mean(coverage)
    )


# ============================================================
# RUN ONE METHOD
# ============================================================

def run_strategy(
        method,
        D,
        labels,
        exploration_idx,
        validation_idx,
        max_steps,
        eps_values,
        rng,
):
    """
    Run one exploration strategy.

    IMPORTANT:
    Validation labels are used ONLY below for evaluation.
    They never enter choose_*().
    """

    queried = []

    # --------------------------------------------------------
    # Validation "answer key"
    # --------------------------------------------------------

    validation_success = (
        validation_idx[
            labels[
                validation_idx
            ] == 1
            ]
    )

    beta_true, largest_true = (
        connectivity_signature(
            validation_success,
            D,
            eps_values,
        )
    )

    rows = []

    available = np.ones(
        len(labels),
        dtype=bool,
    )

    available[:] = False
    available[exploration_idx] = True

    for step in range(
            1,
            max_steps + 1,
    ):

        candidate_idx = np.flatnonzero(
            available
        )

        queried_arr = np.asarray(
            queried,
            dtype=int,
        )

        # ----------------------------------------------------
        # Choose next point
        # ----------------------------------------------------

        if method == "random":

            chosen = choose_random(
                candidate_idx,
                rng,
            )

        elif method == "fps":

            chosen = choose_fps(
                candidate_idx,
                queried_arr,
                D,
                rng,
            )

        elif method == "local":

            chosen = choose_local(
                candidate_idx,
                queried_arr,
                labels,
                D,
                rng,
            )

        elif method == "connectivity_ms":

            chosen = choose_connectivity_ms(
                candidate_idx,
                queried_arr,
                labels,
                D,
                eps_values,
                rng,
            )

        else:
            raise ValueError(
                f"Unknown method: {method}"
            )

        queried.append(chosen)
        available[chosen] = False

        queried_arr = np.asarray(
            queried,
            dtype=int,
        )

        discovered_successes = (
            queried_arr[
                labels[
                    queried_arr
                ] == 1
                ]
        )

        # ----------------------------------------------------
        # Estimated connectivity from discovered successes
        # ----------------------------------------------------

        beta_hat, largest_hat = (
            connectivity_signature(
                discovered_successes,
                D,
                eps_values,
            )
        )

        (
            structure_error,
            beta_error,
            largest_error,
        ) = signature_distance(
            beta_hat,
            largest_hat,
            beta_true,
            largest_true,
        )

        # ----------------------------------------------------
        # Secondary coverage
        # ----------------------------------------------------

        coverage = (
            validation_coverage(
                D,
                discovered_successes,
                validation_idx,
                labels,
                eps_values,
            )
        )

        rows.append({
            "step": step,
            "method": method,
            "chosen": chosen,
            "chosen_success": int(
                labels[chosen]
            ),
            "n_success": len(
                discovered_successes
            ),

            # PRIMARY
            "structure_error":
                structure_error,

            "beta_error":
                beta_error,

            "largest_error":
                largest_error,

            # SECONDARY
            "validation_coverage":
                coverage,
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "grasp_dir",
        type=str,
    )

    parser.add_argument(
        "--n_objects",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "independent_exploration_v2.csv"
        ),
    )

    args = parser.parse_args()

    master_rng = (
        np.random.default_rng(
            args.seed
        )
    )

    paths = find_objects(
        args.grasp_dir
    )

    if len(paths) == 0:
        raise RuntimeError(
            "No HDF5 files found."
        )

    n_objects = min(
        args.n_objects,
        len(paths),
    )

    chosen_indices = (
        master_rng.choice(
            len(paths),
            size=n_objects,
            replace=False,
        )
    )

    chosen_paths = [
        paths[i]
        for i in chosen_indices
    ]

    # Same multiscale range for all methods.
    eps_values = np.linspace(
        0.10,
        0.80,
        15,
    )

    methods = [
        "random",
        "fps",
        "local",
        "connectivity_ms",
    ]

    all_rows = []

    for object_number, path in enumerate(
            chosen_paths,
            start=1,
    ):

        print()
        print("=" * 72)
        print(
            f"Object "
            f"{object_number}/{n_objects}: "
            f"{Path(path).name}"
        )

        transforms, labels = (
            load_object(path)
        )

        n = len(labels)

        # ----------------------------------------------------
        # Fixed 70 / 30 split for this object
        # ----------------------------------------------------

        split_rng = (
            np.random.default_rng(
                args.seed
                + object_number
                * 100_000
            )
        )

        indices = (
            split_rng.permutation(n)
        )

        split_point = int(
            0.70 * n
        )

        exploration_idx = (
            indices[:split_point]
        )

        validation_idx = (
            indices[split_point:]
        )

        print(
            f"Candidates:   {n}"
        )

        print(
            f"Exploration:  "
            f"{len(exploration_idx)}"
        )

        print(
            f"Validation:   "
            f"{len(validation_idx)}"
        )

        print(
            f"Success rate: "
            f"{labels.mean():.3f}"
        )

        print(
            "Computing SE(3) distances..."
        )

        D = grasp_distance_matrix(
            transforms
        )

        # ----------------------------------------------------
        # Run every strategy on exactly same split
        # ----------------------------------------------------

        for method_number, method in enumerate(
                methods
        ):

            print(
                f"  Running {method}..."
            )

            method_rng = (
                np.random.default_rng(
                    args.seed
                    + object_number
                    * 10_000
                    + method_number
                    * 100
                )
            )

            rows = run_strategy(
                method=method,
                D=D,
                labels=labels,
                exploration_idx=(
                    exploration_idx
                ),
                validation_idx=(
                    validation_idx
                ),
                max_steps=min(
                    args.max_steps,
                    len(
                        exploration_idx
                    ),
                ),
                eps_values=eps_values,
                rng=method_rng,
            )

            for row in rows:

                row["object"] = (
                    Path(path).name
                )

                row[
                    "object_number"
                ] = object_number

                row[
                    "success_fraction"
                ] = float(
                    labels.mean()
                )

                all_rows.append(row)

            final = rows[-1]

            print(
                f"    successes="
                f"{final['n_success']:4d}"
                f" | structure error="
                f"{final['structure_error']:.4f}"
                f" | beta="
                f"{final['beta_error']:.4f}"
                f" | largest="
                f"{final['largest_error']:.4f}"
                f" | coverage="
                f"{final['validation_coverage']:.3f}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    df = pd.DataFrame(
        all_rows
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    checkpoints = [
        10,
        20,
        50,
        100,
        200,
    ]

    checkpoints = [
        x
        for x in checkpoints
        if x <= args.max_steps
    ]

    check_df = df[
        df["step"].isin(
            checkpoints
        )
    ]

    summary = (
        check_df
        .groupby(
            ["method", "step"]
        )
        .agg(
            structure_error=(
                "structure_error",
                "mean",
            ),
            beta_error=(
                "beta_error",
                "mean",
            ),
            largest_error=(
                "largest_error",
                "mean",
            ),
            coverage=(
                "validation_coverage",
                "mean",
            ),
            successes=(
                "n_success",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_path = (
        output.with_name(
            output.stem
            + "_summary.csv"
        )
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 72)
    print(
        "PRIMARY METRIC: "
        "STRUCTURE ERROR "
        "(LOWER IS BETTER)"
    )
    print()

    structure_table = (
        summary.pivot(
            index="step",
            columns="method",
            values="structure_error",
        )
        .round(4)
    )

    print(
        structure_table
    )

    print()
    print("=" * 72)
    print(
        "SECONDARY METRIC: "
        "VALIDATION COVERAGE "
        "(HIGHER IS BETTER)"
    )
    print()

    coverage_table = (
        summary.pivot(
            index="step",
            columns="method",
            values="coverage",
        )
        .round(3)
    )

    print(
        coverage_table
    )

    print()
    print("=" * 72)
    print(
        "MEAN DISCOVERED SUCCESSES"
    )
    print()

    success_table = (
        summary.pivot(
            index="step",
            columns="method",
            values="successes",
        )
        .round(1)
    )

    print(
        success_table
    )

    print()
    print(
        f"Saved raw results: "
        f"{output}"
    )

    print(
        f"Saved summary:     "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()