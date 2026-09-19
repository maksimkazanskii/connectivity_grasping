# src/e3/experiment.py

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.utils.grasp_io import find_objects
from src.utils.samplers import (
    random_order,
    fps_order,
    choose_connectivity_point,
)


def distance_matrix_with_reference_scaling(
        transforms,
        reference_indices,
):
    """
    Compute normalized SE(3) distances for all transforms, while
    estimating the translation and rotation normalization constants
    ONLY from a reference subset (Pool A).

    The resulting Pool-A scales are then frozen and used for:
        - A <-> A distances
        - A <-> B distances
        - B <-> B distances

    Thus Pool B cannot influence the metric used during Phase 1,
    while all Phase-2 cross-pool distances remain in the same metric.
    """
    transforms = np.asarray(transforms)
    reference_indices = np.asarray(reference_indices, dtype=int)

    p = transforms[:, :3, 3]
    R = transforms[:, :3, :3]

    # Raw translation distances for all A+B candidates.
    dp = np.linalg.norm(
        p[:, None, :] - p[None, :, :],
        axis=-1,
        )

    # Raw SO(3) geodesic distances for all A+B candidates.
    relative = np.einsum(
        "aik,bil->abkl",
        R,
        R,
    )

    trace = np.trace(
        relative,
        axis1=-2,
        axis2=-1,
    )

    cosine = np.clip(
        (trace - 1.0) / 2.0,
        -1.0,
        1.0,
        )

    dr = np.arccos(cosine)

    # Estimate BOTH normalization constants from Pool A only.
    dp_ref = dp[
        np.ix_(
            reference_indices,
            reference_indices,
        )
    ]

    dr_ref = dr[
        np.ix_(
            reference_indices,
            reference_indices,
        )
    ]

    positive_dp = dp_ref[dp_ref > 0]
    positive_dr = dr_ref[dr_ref > 0]

    if len(positive_dp) == 0:
        raise ValueError(
            "Pool A has no nonzero translation distances; "
            "cannot estimate translation scale."
        )

    if len(positive_dr) == 0:
        raise ValueError(
            "Pool A has no nonzero rotation distances; "
            "cannot estimate rotation scale."
        )

    scale_p = float(np.median(positive_dp))
    scale_r = float(np.median(positive_dr))

    D = np.sqrt(
        (dp / scale_p) ** 2
        +
        (dr / scale_r) ** 2
    )

    return (
        D.astype(np.float32),
        scale_p,
        scale_r,
    )


# ============================================================
# E3: CONNECTIVITY-AWARE EXPLORE -> COMPARE EXPLOITATION RULES
# ============================================================
#
# Per object, construct two DISJOINT balanced hidden-label pools.
#
# Phase 1:
#   A single connectivity-aware exploration trajectory is generated on Pool A.
#   The SAME frozen Phase-1 observations are supplied to every
#   Phase-2 exploitation rule.
#
# Phase 2 compares four fixed rankings on untouched Pool B:
#   1) Random: random Pool-B ordering (no Phase-1 information).
#   2) Nearest-success: ascending distance to a Phase-1 success.
#   3) Local-viability: nearest-success vs nearest-failure margin.
#   4) Connectivity: multiscale connectivity-aware success score.
#
# Pool-B labels are used only for evaluation and never update rankings.
# Primary evaluation uses PHASE 2 ONLY.
# ============================================================


def load_labels(path):
    """Load ACRONYM binary grasp-success labels."""
    with h5py.File(path, "r") as f:
        labels = f[
            "grasps/qualities/flex/object_in_gripper"
        ][:].astype(int)

    return labels


def load_two_balanced_pools(
        path,
        pool_success,
        pool_failure,
        rng,
):
    """
    Construct two disjoint balanced hidden-label pools.

    Pool A:
        pool_success successful + pool_failure failed grasps.

    Pool B:
        pool_success successful + pool_failure failed grasps.

    The two pools contain no overlapping grasps.

    Returns
    -------
    transforms : ndarray
        Combined transforms for Pool A followed by Pool B.

    labels : ndarray
        Hidden binary labels in the same combined order.

    pool_a : ndarray
        Global indices of Pool-A candidates.

    pool_b : ndarray
        Global indices of Pool-B candidates.
    """
    with h5py.File(path, "r") as f:
        labels_all = f[
            "grasps/qualities/flex/object_in_gripper"
        ][:].astype(int)

        success_idx = np.flatnonzero(
            labels_all == 1
        )
        failure_idx = np.flatnonzero(
            labels_all == 0
        )

        needed_success = 2 * pool_success
        needed_failure = 2 * pool_failure

        if len(success_idx) < needed_success:
            raise ValueError(
                f"{Path(path).name}: only "
                f"{len(success_idx)} successes; "
                f"need {needed_success}."
            )

        if len(failure_idx) < needed_failure:
            raise ValueError(
                f"{Path(path).name}: only "
                f"{len(failure_idx)} failures; "
                f"need {needed_failure}."
            )

        # Sample once without replacement, then split.
        chosen_success = rng.choice(
            success_idx,
            size=needed_success,
            replace=False,
        )

        chosen_failure = rng.choice(
            failure_idx,
            size=needed_failure,
            replace=False,
        )

        pool_a_original = np.concatenate([
            chosen_success[:pool_success],
            chosen_failure[:pool_failure],
        ])

        pool_b_original = np.concatenate([
            chosen_success[pool_success:],
            chosen_failure[pool_failure:],
        ])

        # Shuffle each pool independently so array order contains
        # no success/failure signal.
        rng.shuffle(pool_a_original)
        rng.shuffle(pool_b_original)

        transforms_all = f[
            "grasps/transforms"
        ][:]

    combined_original = np.concatenate([
        pool_a_original,
        pool_b_original,
    ])

    transforms = transforms_all[
        combined_original
    ]

    labels = labels_all[
        combined_original
    ]

    per_pool_size = (
            pool_success
            + pool_failure
    )

    pool_a = np.arange(
        0,
        per_pool_size,
        dtype=int,
    )

    pool_b = np.arange(
        per_pool_size,
        2 * per_pool_size,
        dtype=int,
        )

    return (
        transforms,
        labels,
        pool_a,
        pool_b,
    )


# ============================================================
# PHASE 1: CONNECTIVITY-AWARE EXPLORATION ON POOL A
# ============================================================

def hidden_connectivity_exploration(
        labels,
        D,
        eps_values,
        candidate_indices,
        n_attempts,
        rng,
        bootstrap=5,
):
    """
    Connectivity-aware exploration restricted to Pool A.

    Only queried successful grasps enter the observed viable
    connectivity map.

    Failed grasps consume an attempt but do not enter the graph.
    """
    candidate_indices = np.asarray(
        candidate_indices,
        dtype=int,
    )

    if n_attempts > len(candidate_indices):
        raise ValueError(
            f"n_attempts={n_attempts} exceeds "
            f"exploration-pool size "
            f"{len(candidate_indices)}."
        )

    available = np.zeros(
        len(labels),
        dtype=bool,
    )

    available[
        candidate_indices
    ] = True

    queried = []
    successful = []

    bootstrap_size = min(
        bootstrap,
        n_attempts,
    )

    # --------------------------------------------------------
    # Random bootstrap
    # --------------------------------------------------------
    for _ in range(
            bootstrap_size
    ):
        candidates = np.flatnonzero(
            available
        )

        chosen = int(
            rng.choice(
                candidates
            )
        )

        queried.append(
            chosen
        )

        available[
            chosen
        ] = False

        # Reveal hidden outcome only after query.
        if labels[chosen] == 1:
            successful.append(
                chosen
            )

    # --------------------------------------------------------
    # Connectivity-aware exploration
    # --------------------------------------------------------
    while len(queried) < n_attempts:

        # Until one success exists, there is no viable
        # connectivity map to use.
        if len(successful) == 0:
            candidates = np.flatnonzero(
                available
            )

            chosen = int(
                rng.choice(
                    candidates
                )
            )

        else:
            chosen = (
                choose_connectivity_point(
                    selected=successful,
                    available=available,
                    D=D,
                    eps_values=eps_values,
                )
            )

        queried.append(
            chosen
        )

        available[
            chosen
        ] = False

        # Reveal hidden outcome only after query.
        if labels[chosen] == 1:
            successful.append(
                chosen
            )

    return np.asarray(
        queried,
        dtype=int,
    )


# ============================================================
# PHASE 1: UNCERTAINTY SAMPLING ON POOL A
# ============================================================

def hidden_uncertainty_exploration(
        labels,
        D,
        candidate_indices,
        n_attempts,
        rng,
        bootstrap=5,
):
    """
    Outcome-aware uncertainty sampling restricted to Pool A.

    After the same random-bootstrap size used by connectivity-aware
    exploration, select candidates closest to the current empirical
    success/failure boundary.

    For each unqueried candidate x:

        d_success(x) = min distance to an observed success
        d_failure(x) = min distance to an observed failure

        U(x) = - |d_success(x) - d_failure(x)|
               / (d_success(x) + d_failure(x) + tiny)

    The candidate with maximum U(x) is queried next.

    If only one outcome class has been observed, choose the candidate
    farthest from the observed set until both classes are available.

    Hidden labels are revealed only after querying.
    """
    candidate_indices = np.asarray(
        candidate_indices,
        dtype=int,
    )

    if n_attempts > len(candidate_indices):
        raise ValueError(
            f"n_attempts={n_attempts} exceeds "
            f"exploration-pool size "
            f"{len(candidate_indices)}."
        )

    available = np.zeros(
        len(labels),
        dtype=bool,
    )

    available[
        candidate_indices
    ] = True

    queried = []
    successful = []
    failed = []

    bootstrap_size = min(
        bootstrap,
        n_attempts,
    )

    # Random bootstrap.
    for _ in range(
            bootstrap_size
    ):
        candidates = np.flatnonzero(
            available
        )

        chosen = int(
            rng.choice(
                candidates
            )
        )

        queried.append(
            chosen
        )

        available[
            chosen
        ] = False

        if labels[chosen] == 1:
            successful.append(
                chosen
            )
        else:
            failed.append(
                chosen
            )

    tiny = 1e-12

    while len(queried) < n_attempts:

        candidates = np.flatnonzero(
            available
        )

        if (
                len(successful) > 0
                and
                len(failed) > 0
        ):
            d_success = D[
                np.ix_(
                    candidates,
                    np.asarray(
                        successful,
                        dtype=int,
                    ),
                )
            ].min(
                axis=1
            )

            d_failure = D[
                np.ix_(
                    candidates,
                    np.asarray(
                        failed,
                        dtype=int,
                    ),
                )
            ].min(
                axis=1
            )

            uncertainty = (
                    -np.abs(
                        d_success
                        - d_failure
                    )
                    /
                    (
                            d_success
                            + d_failure
                            + tiny
                    )
            )

            chosen = int(
                candidates[
                    np.argmax(
                        uncertainty
                    )
                ]
            )

        else:
            observed = np.asarray(
                queried,
                dtype=int,
            )

            if len(observed) == 0:
                chosen = int(
                    rng.choice(
                        candidates
                    )
                )
            else:
                nearest_observed = D[
                    np.ix_(
                        candidates,
                        observed,
                    )
                ].min(
                    axis=1
                )

                chosen = int(
                    candidates[
                        np.argmax(
                            nearest_observed
                        )
                    ]
                )

        queried.append(
            chosen
        )

        available[
            chosen
        ] = False

        if labels[chosen] == 1:
            successful.append(
                chosen
            )
        else:
            failed.append(
                chosen
            )

    return np.asarray(
        queried,
        dtype=int,
    )


# ============================================================
# PHASE 2: COMMON EXPLOITATION ON UNTOUCHED POOL B
# ============================================================

def nearest_success_exploitation_order(
        exploration_order,
        exploitation_candidates,
        labels,
        D,
        n_attempts,
        rng,
):
    """
    Common exploitation policy used after every exploration method.

    Pool B is untouched before this phase.

    IMPORTANT:
        The exploitation anchor set is FROZEN at the successful
        grasps discovered during Phase 1.

        Pool-B successes count toward the Phase-2 metric, but they
        do NOT become new anchors and therefore do not influence
        later Phase-2 choices.

    This preserves the causal chain:

        Phase-1 exploration quality
            -> fixed successful anchor set
            -> Phase-2 performance

    At each exploitation step:
        query the remaining Pool-B candidate with the smallest
        distance to ANY successful grasp discovered in Phase 1.

    If Phase 1 found no successful grasp, Phase 2 falls back to
    a fixed random ordering of Pool B.
    """
    exploration_order = np.asarray(
        exploration_order,
        dtype=int,
    )

    exploitation_candidates = np.asarray(
        exploitation_candidates,
        dtype=int,
    )

    if n_attempts > len(
            exploitation_candidates
    ):
        raise ValueError(
            f"n_attempts={n_attempts} exceeds "
            f"exploitation-pool size "
            f"{len(exploitation_candidates)}."
        )

    available = np.zeros(
        len(labels),
        dtype=bool,
    )

    available[
        exploitation_candidates
    ] = True

    # --------------------------------------------------------
    # FROZEN Phase-1 successful anchor set
    # --------------------------------------------------------
    phase1_successful = np.asarray(
        [
            int(idx)
            for idx in exploration_order
            if labels[idx] == 1
        ],
        dtype=int,
    )

    queried = []

    # --------------------------------------------------------
    # Rare fallback: no Phase-1 success
    # --------------------------------------------------------
    if len(phase1_successful) == 0:
        fallback_order = rng.permutation(
            exploitation_candidates
        )

        return np.asarray(
            fallback_order[:n_attempts],
            dtype=int,
        )

    # --------------------------------------------------------
    # Fixed-anchor exploitation
    # --------------------------------------------------------
    # Distances from every Pool-B candidate to the frozen
    # Phase-1 successful anchor set.
    candidate_distances = D[
        np.ix_(
            exploitation_candidates,
            phase1_successful,
        )
    ]

    nearest_phase1_success = (
        candidate_distances.min(
            axis=1
        )
    )

    # Sort Pool-B candidates once by proximity to Phase-1
    # successful anchors. No Phase-2 outcome can change this order.
    local_order = np.argsort(
        nearest_phase1_success,
        kind="stable",
    )

    exploit_order = exploitation_candidates[
        local_order[:n_attempts]
    ]

    return np.asarray(
        exploit_order,
        dtype=int,
    )


def random_exploitation_order(
        exploitation_candidates,
        n_attempts,
        rng,
):
    """Random fixed ordering of untouched Pool B."""
    exploitation_candidates = np.asarray(exploitation_candidates, dtype=int)
    if n_attempts > len(exploitation_candidates):
        raise ValueError("n_attempts exceeds exploitation-pool size.")
    return np.asarray(
        rng.permutation(exploitation_candidates)[:n_attempts],
        dtype=int,
    )


def local_viability_exploitation_order(
        exploration_order,
        exploitation_candidates,
        labels,
        D,
        n_attempts,
        rng,
):
    """
    Fixed local success/failure exploitation baseline.

    Uses exactly the same frozen connectivity-aware Phase-1 observations as the
    connectivity method, but does not construct connected components or
    average over scales. Candidates are ranked by the normalized nearest-
    class margin

        (d_failure - d_success) / (d_failure + d_success + tiny),

    so larger values indicate greater local support for success.
    """
    exploration_order = np.asarray(exploration_order, dtype=int)
    exploitation_candidates = np.asarray(exploitation_candidates, dtype=int)
    if n_attempts > len(exploitation_candidates):
        raise ValueError("n_attempts exceeds exploitation-pool size.")

    phase1_labels = labels[exploration_order]
    successful = exploration_order[phase1_labels == 1]
    failed = exploration_order[phase1_labels == 0]

    if len(successful) == 0:
        return np.asarray(
            rng.permutation(exploitation_candidates)[:n_attempts],
            dtype=int,
        )

    d_success = D[np.ix_(exploitation_candidates, successful)].min(axis=1)

    if len(failed) == 0:
        scores = -d_success
    else:
        d_failure = D[np.ix_(exploitation_candidates, failed)].min(axis=1)
        scores = (d_failure - d_success) / (d_failure + d_success + 1e-12)

    local_order = np.argsort(-scores, kind="stable")
    return np.asarray(
        exploitation_candidates[local_order[:n_attempts]],
        dtype=int,
    )


def connectivity_exploitation_order(
        exploration_order,
        exploitation_candidates,
        labels,
        D,
        eps_values,
        n_attempts,
        rng,
):
    """
    Common multiscale Phase-2 exploitation used after every Phase-1 policy.

    Phase-1 observations are frozen.

    Successful Phase-1 grasps form multiscale connected components.
    Failed Phase-1 grasps provide negative evidence.

    For candidate x and scale eps:

        positive_density(x) =
            max_C [
                sum_{s in C} K(x, s)
                / N_success
            ]

        negative_density(x) =
            sum_{f in F} K(x, f)
            / N_failure

        p_success(x) =
            positive_density
            /
            (positive_density + negative_density + tiny)

    The final score is the mean p_success(x) across eps values.

    Notes
    -----
    - Pool-B labels are NEVER used to compute the ranking.
    - Pool-B outcomes NEVER update the ranking.
    - Large successful components retain more evidence than tiny
      components, but the score is normalized for the number of
      observed successes/failures.
    """

    exploration_order = np.asarray(
        exploration_order,
        dtype=int,
    )

    exploitation_candidates = np.asarray(
        exploitation_candidates,
        dtype=int,
    )

    if n_attempts > len(exploitation_candidates):
        raise ValueError(
            f"n_attempts={n_attempts} exceeds "
            f"exploitation-pool size "
            f"{len(exploitation_candidates)}."
        )

    # --------------------------------------------------------
    # Frozen Phase-1 observations
    # --------------------------------------------------------

    phase1_labels = labels[
        exploration_order
    ]

    phase1_successful = exploration_order[
        phase1_labels == 1
        ]

    phase1_failed = exploration_order[
        phase1_labels == 0
        ]

    # If no viable grasp was discovered, there is no successful
    # structure to exploit.
    if len(phase1_successful) == 0:

        fallback_order = rng.permutation(
            exploitation_candidates
        )

        return np.asarray(
            fallback_order[:n_attempts],
            dtype=int,
        )

    n_success = len(
        phase1_successful
    )

    n_failure = len(
        phase1_failed
    )

    # --------------------------------------------------------
    # Required distances
    # --------------------------------------------------------

    d_candidate_success = D[
        np.ix_(
            exploitation_candidates,
            phase1_successful,
        )
    ]

    if n_failure > 0:

        d_candidate_failure = D[
            np.ix_(
                exploitation_candidates,
                phase1_failed,
            )
        ]

    else:

        d_candidate_failure = None

    d_success_success = D[
        np.ix_(
            phase1_successful,
            phase1_successful,
        )
    ]

    candidate_scores = np.zeros(
        len(exploitation_candidates),
        dtype=float,
    )

    tiny = 1e-12

    # ========================================================
    # MULTISCALE CONNECTIVITY-AWARE SUCCESS PROBABILITY
    # ========================================================

    for eps in eps_values:

        # ----------------------------------------------------
        # Successful connectivity graph
        # ----------------------------------------------------

        adjacency = (
                d_success_success <= eps
        )

        component_ids = np.full(
            n_success,
            -1,
            dtype=int,
        )

        n_components = 0

        for root in range(n_success):

            if component_ids[root] != -1:
                continue

            stack = [root]

            component_ids[root] = (
                n_components
            )

            while stack:

                node = stack.pop()

                neighbors = np.flatnonzero(
                    adjacency[node]
                    &
                    (component_ids == -1)
                )

                component_ids[
                    neighbors
                ] = n_components

                stack.extend(
                    neighbors.tolist()
                )

            n_components += 1

        # ----------------------------------------------------
        # Kernel similarity to successful observations
        # ----------------------------------------------------

        success_kernel = np.exp(
            -(
                    d_candidate_success ** 2
            )
            /
            (
                    2.0 * eps ** 2
            )
        )

        # ----------------------------------------------------
        # Positive density
        # ----------------------------------------------------
        #
        # For each connected viable component:
        #
        #     sum K(x,s) / N_success
        #
        # This is equivalent to:
        #
        #   component prior
        #       *
        #   component mean kernel density
        #
        # because:
        #
        #   |C|/N_success * (sum K / |C|)
        #       = sum K / N_success
        #
        # Therefore large, coherent successful components carry
        # appropriately more probability mass without allowing
        # raw success count to arbitrarily scale the score.

        positive_density = np.zeros(
            len(exploitation_candidates),
            dtype=float,
        )

        for component_id in range(
                n_components
        ):

            members = (
                    component_ids
                    == component_id
            )

            component_density = (
                    success_kernel[
                        :,
                        members
                    ].sum(
                        axis=1
                    )
                    / float(n_success)
            )

            positive_density = np.maximum(
                positive_density,
                component_density,
            )

        # ----------------------------------------------------
        # Negative density
        # ----------------------------------------------------

        if d_candidate_failure is None:

            negative_density = np.zeros(
                len(exploitation_candidates),
                dtype=float,
            )

        else:

            failure_kernel = np.exp(
                -(
                        d_candidate_failure ** 2
                )
                /
                (
                        2.0 * eps ** 2
                )
            )

            negative_density = (
                    failure_kernel.sum(
                        axis=1
                    )
                    / float(n_failure)
            )

        # ----------------------------------------------------
        # Probability-like success score
        # ----------------------------------------------------

        local_probability = (
                positive_density
                /
                (
                        positive_density
                        + negative_density
                        + tiny
                )
        )

        candidate_scores += (
            local_probability
        )

    # Average over connectivity scales.
    candidate_scores /= float(
        len(eps_values)
    )

    # --------------------------------------------------------
    # Fixed Phase-2 ranking
    # --------------------------------------------------------

    local_order = np.argsort(
        -candidate_scores,
        kind="stable",
    )

    exploit_order = (
        exploitation_candidates[
            local_order[
                :n_attempts
            ]
        ]
    )

    return np.asarray(
        exploit_order,
        dtype=int,
    )

# ============================================================
# METRICS: PHASE 2 ONLY
# ============================================================

def cumulative_success_curve(
        order,
        labels,
):
    """
    Cumulative successes within the supplied phase.

    For Phase 2 this starts from zero. Phase-1 successes provide
    knowledge but do not directly contribute to the metric.
    """
    queried_labels = labels[
        np.asarray(
            order,
            dtype=int,
        )
    ]

    return np.cumsum(
        queried_labels == 1
    ).astype(float)


def discovery_auc(curve):
    """
    Area under cumulative-success-vs-attempt curve.

    Includes the origin:
        0 attempts -> 0 successes.
    """
    y = np.concatenate([
        np.array([0.0]),
        np.asarray(
            curve,
            dtype=float,
        ),
    ])

    return float(
        np.sum(
            0.5 * (
                    y[:-1]
                    + y[1:]
            )
        )
    )


def maximum_discovery_auc(
        n_attempts,
        available_successes,
):
    """
    Maximum possible AUC with a finite number of successes.

    If available_successes >= n_attempts:
        C(k) = k.

    Otherwise the optimal curve reaches all available successes
    first and then plateaus.
    """
    k = int(
        n_attempts
    )

    s = min(
        int(
            available_successes
        ),
        k,
    )

    return (
            (s ** 2) / 2.0
            + s * (
                    k - s
            )
    )


def normalized_discovery_auc(
        curve,
        available_successes,
):
    """
    Phase-2 AUC divided by its theoretical maximum.
    """
    k = len(
        curve
    )

    if k == 0:
        return np.nan

    max_auc = maximum_discovery_auc(
        n_attempts=k,
        available_successes=available_successes,
    )

    if max_auc == 0:
        return np.nan

    return (
            discovery_auc(
                curve
            )
            / max_auc
    )


def summarize_curves(
        curves,
        available_successes,
):
    """
    Summarize repeated Phase-2 curves across seeds.
    """
    curves = np.asarray(
        curves,
        dtype=float,
    )

    aucs = np.asarray([
        discovery_auc(
            curve
        )
        for curve in curves
    ])

    normalized_aucs = np.asarray([
        normalized_discovery_auc(
            curve,
            available_successes=available_successes,
        )
        for curve in curves
    ])

    final_successes = (
        curves[:, -1]
    )

    return {
        "mean_curve":
            curves.mean(
                axis=0
            ),

        "std_curve":
            curves.std(
                axis=0
            ),

        "aucs":
            aucs,

        "normalized_aucs":
            normalized_aucs,

        "final_successes":
            final_successes,
    }


# ============================================================
# CHECKPOINTS
# ============================================================

def save_checkpoints(
        results_dir,
        curve_rows,
        seed_rows,
        object_rows,
        selection_rows,
):
    """Save progress after every completed object."""

    pd.DataFrame(
        curve_rows
    ).to_csv(
        results_dir
        / "exploitation_curves.csv",
        index=False,
        )

    pd.DataFrame(
        seed_rows
    ).to_csv(
        results_dir
        / "phase_results_per_seed.csv",
        index=False,
        )

    pd.DataFrame(
        object_rows
    ).to_csv(
        results_dir
        / "object_summary.csv",
        index=False,
        )

    pd.DataFrame(
        selection_rows
    ).to_csv(
        results_dir
        / "selected_objects.csv",
        index=False,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "grasp_dir",
        nargs="?",
        default="grasps",
        help=(
            "ACRONYM grasp directory "
            "(default: ./grasps)"
        ),
    )

    parser.add_argument(
        "--n_objects",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--pool_success",
        type=int,
        default=100,
        help=(
            "Successful candidates in EACH pool "
            "(default: 100)"
        ),
    )

    parser.add_argument(
        "--pool_failure",
        type=int,
        default=100,
        help=(
            "Failed candidates in EACH pool "
            "(default: 100)"
        ),
    )

    parser.add_argument(
        "--exploration_attempts",
        type=int,
        default=100,
        help=(
            "Phase-1 attempts on Pool A "
            "(default: 100)"
        ),
    )

    parser.add_argument(
        "--exploitation_attempts",
        type=int,
        default=100,
        help=(
            "Phase-2 attempts on Pool B "
            "(default: 100)"
        ),
    )

    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    per_pool_size = (
            args.pool_success
            + args.pool_failure
    )

    combined_pool_size = (
            2
            * per_pool_size
    )

    if (
            args.exploration_attempts
            > per_pool_size
    ):
        raise ValueError(
            "exploration_attempts cannot "
            "exceed Pool-A size."
        )

    if (
            args.exploitation_attempts
            > per_pool_size
    ):
        raise ValueError(
            "exploitation_attempts cannot "
            "exceed Pool-B size."
        )

    # Configuration-specific output directory so pool-size
    # ablation runs cannot overwrite one another.
    #
    # Examples:
    #   total A+B = 200 -> results/e3_disjoint_pools_results/pool_200
    #   total A+B = 400 -> results/e3_disjoint_pools_results/pool_400
    #   total A+B = 800 -> results/e3_disjoint_pools_results/pool_800
    results_dir = (
            Path("results")
            / "e3_disjoint_pools_results"
            / f"pool_{combined_pool_size}"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Same multiscale radii as current connectivity sampler.
    eps_values = np.linspace(
        0.10,
        0.80,
        15,
    )

    master_rng = np.random.default_rng(
        args.seed
    )

    # ========================================================
    # FIND ELIGIBLE OBJECTS
    # ========================================================

    paths = find_objects(
        args.grasp_dir
    )

    if len(paths) == 0:
        raise RuntimeError(
            f"No .h5 files found under "
            f"{args.grasp_dir!r}."
        )

    # ========================================================
    # FIXED OBJECT ELIGIBILITY FOR POOL-SIZE ABLATION
    # ========================================================
    # Use the largest planned condition (A+B = 800) to define
    # eligibility for EVERY run:
    #
    #   Pool A = 200 success + 200 failure
    #   Pool B = 200 success + 200 failure
    #
    # Therefore every selected object must contain at least
    # 400 successes and 400 failures in the full ACRONYM file.
    #
    # Because object selection also uses the same master seed,
    # runs with total pools 200, 400, and 800 select the exact
    # same objects (assuming the same --n_objects and --seed).
    largest_pool_success_each = 400
    largest_pool_failure_each = 400

    needed_success = 2 * largest_pool_success_each
    needed_failure = 2 * largest_pool_failure_each

    print(
        f"Found {len(paths)} object files."
    )

    print(
        f"Fixed eligibility requirement for pool-size ablation: "
        f">= {needed_success} successes and "
        f">= {needed_failure} failures "
        f"(largest A+B=800 condition)."
    )

    eligible = []
    eligibility_counts = {}

    for k, path in enumerate(
            paths,
            start=1,
    ):

        labels = load_labels(
            path
        )

        n_success = int(
            np.sum(
                labels == 1
            )
        )

        n_failure = int(
            np.sum(
                labels == 0
            )
        )

        if (
                n_success
                >= needed_success
                and
                n_failure
                >= needed_failure
        ):
            eligible.append(
                path
            )

            eligibility_counts[
                str(path)
            ] = (
                n_success,
                n_failure,
            )

        if (
                k % 500 == 0
                or
                k == len(paths)
        ):
            print(
                f"  scanned {k}/{len(paths)} | "
                f"eligible so far: "
                f"{len(eligible)}"
            )

    if len(eligible) < args.n_objects:
        raise RuntimeError(
            f"Only {len(eligible)} eligible objects, "
            f"but --n_objects={args.n_objects}."
        )

    selected_paths = master_rng.choice(
        np.asarray(
            eligible,
            dtype=object,
        ),
        size=args.n_objects,
        replace=False,
    )

    print()

    print(
        f"Randomly selected "
        f"{args.n_objects} objects from "
        f"{len(eligible)} eligible objects."
    )

    print(
        f"Pool A (exploration): "
        f"{args.pool_success} success + "
        f"{args.pool_failure} failure = "
        f"{per_pool_size}."
    )

    print(
        f"Pool B (exploitation): "
        f"{args.pool_success} success + "
        f"{args.pool_failure} failure = "
        f"{per_pool_size}."
    )

    print(
        f"Pools are disjoint; total sampled "
        f"candidates/object = "
        f"{combined_pool_size}."
    )

    print(
        f"Phase 1 attempts = "
        f"{args.exploration_attempts} "
        f"(Pool A only)."
    )

    print(
        f"Phase 2 attempts = "
        f"{args.exploitation_attempts} "
        f"(untouched Pool B only)."
    )

    print(
        "Phase 1 uses one shared connectivity-aware trajectory; Phase 2 compares "
        "Random, nearest-success, local-viability, and connectivity rankings."
    )

    print(
        "Primary metric: Phase-2 AUC only."
    )

    print(
        f"Seeds per method = "
        f"{args.n_seeds}."
    )

    print(
        f"Results directory: {results_dir}"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    curve_rows = []
    seed_rows = []
    object_rows = []
    selection_rows = []

    # ========================================================
    # OBJECT LOOP
    # ========================================================

    for object_number, path in enumerate(
            selected_paths,
            start=1,
    ):

        path = Path(
            path
        )

        print()
        print(
            "=" * 78
        )

        print(
            f"Object "
            f"{object_number}/"
            f"{args.n_objects}: "
            f"{path.name}"
        )

        (
            total_success,
            total_failure,
        ) = eligibility_counts[
            str(path)
        ]

        # Fixed pool assignment for this object,
        # shared by all methods and seeds.
        pool_rng = np.random.default_rng(
            args.seed
            + object_number * 10000
        )

        (
            transforms,
            labels,
            pool_a,
            pool_b,
        ) = load_two_balanced_pools(
            path=path,
            pool_success=args.pool_success,
            pool_failure=args.pool_failure,
            rng=pool_rng,
        )

        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        assert int(
            np.sum(
                labels[
                    pool_a
                ] == 1
            )
        ) == args.pool_success

        assert int(
            np.sum(
                labels[
                    pool_a
                ] == 0
            )
        ) == args.pool_failure

        assert int(
            np.sum(
                labels[
                    pool_b
                ] == 1
            )
        ) == args.pool_success

        assert int(
            np.sum(
                labels[
                    pool_b
                ] == 0
            )
        ) == args.pool_failure

        assert len(
            np.intersect1d(
                pool_a,
                pool_b,
            )
        ) == 0

        selection_rows.append({
            "object":
                path.name,

            "available_successes":
                total_success,

            "available_failures":
                total_failure,

            "pool_a_successes":
                args.pool_success,

            "pool_a_failures":
                args.pool_failure,

            "pool_b_successes":
                args.pool_success,

            "pool_b_failures":
                args.pool_failure,

            "exploration_attempts":
                args.exploration_attempts,

            "exploitation_attempts":
                args.exploitation_attempts,
        })

        print(
            f"  Full object: "
            f"success={total_success}, "
            f"failure={total_failure}"
        )

        print(
            f"  Pool A: "
            f"success={args.pool_success}, "
            f"failure={args.pool_failure}"
        )

        print(
            f"  Pool B: "
            f"success={args.pool_success}, "
            f"failure={args.pool_failure}"
        )

        print(
            f"  Computing "
            f"{combined_pool_size}-candidate "
            f"SE(3) distance matrix..."
        )

        # Estimate SE(3) normalization constants from Pool A ONLY,
        # then freeze and reuse them for all A+A, A+B, and B+B
        # distances. Pool B therefore cannot influence Phase-1
        # metric scaling.
        D, scale_p, scale_r = (
            distance_matrix_with_reference_scaling(
                transforms=transforms,
                reference_indices=pool_a,
            )
        )

        print(
            f"  Pool-A metric scales: "
            f"translation={scale_p:.6g}, "
            f"rotation={scale_r:.6g}"
        )

        # Local Pool-A distance matrix for FPS.
        D_a = D[
            np.ix_(
                pool_a,
                pool_a,
            )
        ]

        methods = [
            "random",
            "nearest_success",
            "local_viability",
            "connectivity",
        ]

        exploitation_curves = {method: [] for method in methods}
        exploration_success_counts = {method: [] for method in methods}

        # ====================================================
        # SEEDS
        # ====================================================
        for seed_index in range(args.n_seeds):
            base_seed = args.seed + object_number * 100000 + seed_index

            # ------------------------------------------------
            # PHASE 1: ONE CONNECTIVITY-AWARE EXPLORATION TRAJECTORY
            # ------------------------------------------------
            explore_rng = np.random.default_rng(base_seed + 1000)
            ca_explore = hidden_connectivity_exploration(
                labels=labels,
                D=D,
                eps_values=eps_values,
                candidate_indices=pool_a,
                n_attempts=args.exploration_attempts,
                rng=explore_rng,
                bootstrap=args.bootstrap,
            )

            n_explore_success = int(
                np.sum(labels[ca_explore] == 1)
            )

            # Every exploitation rule receives this exact same frozen
            # connectivity-aware Phase-1 trajectory. Thus Phase-2 differences
            # isolate the exploitation/inference rule while holding exploration fixed.
            exploit_orders = {}

            exploit_orders["random"] = random_exploitation_order(
                exploitation_candidates=pool_b,
                n_attempts=args.exploitation_attempts,
                rng=np.random.default_rng(base_seed + 10000),
            )

            exploit_orders["nearest_success"] = (
                nearest_success_exploitation_order(
                    exploration_order=ca_explore,
                    exploitation_candidates=pool_b,
                    labels=labels,
                    D=D,
                    n_attempts=args.exploitation_attempts,
                    rng=np.random.default_rng(base_seed + 11000),
                )
            )

            exploit_orders["local_viability"] = (
                local_viability_exploitation_order(
                    exploration_order=ca_explore,
                    exploitation_candidates=pool_b,
                    labels=labels,
                    D=D,
                    n_attempts=args.exploitation_attempts,
                    rng=np.random.default_rng(base_seed + 12000),
                )
            )

            exploit_orders["connectivity"] = (
                connectivity_exploitation_order(
                    exploration_order=ca_explore,
                    exploitation_candidates=pool_b,
                    labels=labels,
                    D=D,
                    eps_values=eps_values,
                    n_attempts=args.exploitation_attempts,
                    rng=np.random.default_rng(base_seed + 13000),
                )
            )

            seed_exploit_success = {}

            for method in methods:
                exploit_order = exploit_orders[method]
                exploit_curve = cumulative_success_curve(exploit_order, labels)
                exploitation_curves[method].append(exploit_curve)
                exploration_success_counts[method].append(n_explore_success)

                n_exploit_success = int(exploit_curve[-1])
                seed_exploit_success[method] = n_exploit_success

                seed_rows.append({
                    "object": path.name,
                    "seed": seed_index + 1,
                    "method": method,
                    "exploration_successes": n_explore_success,
                    "exploitation_successes": n_exploit_success,
                    "exploitation_auc": discovery_auc(exploit_curve),
                    "exploitation_normalized_auc": normalized_discovery_auc(
                        exploit_curve,
                        available_successes=args.pool_success,
                    ),
                })

            print(
                f"  seed {seed_index + 1}/{args.n_seeds}: "
                f"shared CA Phase-1 successes={n_explore_success}"
            )
            print(
                "               exploit successes | "
                f"Random={seed_exploit_success['random']}, "
                f"Nearest={seed_exploit_success['nearest_success']}, "
                f"Local={seed_exploit_success['local_viability']}, "
                f"CA={seed_exploit_success['connectivity']}"
            )

        # ====================================================
        # PHASE-2 SUMMARIES
        # ====================================================

        summaries = {
            method:
                summarize_curves(
                    exploitation_curves[
                        method
                    ],
                    available_successes=(
                        args.pool_success
                    ),
                )
            for method in methods
        }

        # Save mean exploitation curves.
        for method, summary in (
                summaries.items()
        ):

            for step in range(
                    1,
                    args.exploitation_attempts + 1,
            ):

                curve_rows.append({
                    "object":
                        path.name,

                    "method":
                        method,

                    "exploitation_attempt":
                        step,

                    "successes_mean":
                        summary[
                            "mean_curve"
                        ][step - 1],

                    "successes_std":
                        summary[
                            "std_curve"
                        ][step - 1],
                })

        # ----------------------------------------------------
        # Mean Phase-2 AUC across seeds
        # ----------------------------------------------------

        random_auc = float(
            summaries[
                "random"
            ][
                "aucs"
            ].mean()
        )

        nearest_auc = float(
            summaries[
                "nearest_success"
            ][
                "aucs"
            ].mean()
        )

        local_auc = float(
            summaries[
                "local_viability"
            ][
                "aucs"
            ].mean()
        )

        connectivity_auc = float(
            summaries[
                "connectivity"
            ][
                "aucs"
            ].mean()
        )

        random_auc_norm = float(
            summaries[
                "random"
            ][
                "normalized_aucs"
            ].mean()
        )

        nearest_auc_norm = float(
            summaries[
                "nearest_success"
            ][
                "normalized_aucs"
            ].mean()
        )

        local_auc_norm = float(
            summaries[
                "local_viability"
            ][
                "normalized_aucs"
            ].mean()
        )

        connectivity_auc_norm = float(
            summaries[
                "connectivity"
            ][
                "normalized_aucs"
            ].mean()
        )

        # ====================================================
        # HEADLINE EFFECT SIZES
        # ====================================================

        nearest_gain_random = (
                (
                        nearest_auc
                        - random_auc
                )
                / random_auc
                * 100.0
        )

        local_gain_random = (
                (
                        local_auc
                        - random_auc
                )
                / random_auc
                * 100.0
        )

        ca_gain_random = (
                (
                        connectivity_auc
                        - random_auc
                )
                / random_auc
                * 100.0
        )

        # ----------------------------------------------------
        # Mean phase totals across seeds
        # ----------------------------------------------------

        random_explore_mean = float(
            np.mean(
                exploration_success_counts[
                    "random"
                ]
            )
        )

        nearest_explore_mean = float(
            np.mean(
                exploration_success_counts[
                    "nearest_success"
                ]
            )
        )

        local_explore_mean = float(
            np.mean(
                exploration_success_counts[
                    "local_viability"
                ]
            )
        )

        connectivity_explore_mean = float(
            np.mean(
                exploration_success_counts[
                    "connectivity"
                ]
            )
        )

        random_exploit_mean = float(
            summaries[
                "random"
            ][
                "final_successes"
            ].mean()
        )

        nearest_exploit_mean = float(
            summaries[
                "nearest_success"
            ][
                "final_successes"
            ].mean()
        )

        local_exploit_mean = float(
            summaries[
                "local_viability"
            ][
                "final_successes"
            ].mean()
        )

        connectivity_exploit_mean = float(
            summaries[
                "connectivity"
            ][
                "final_successes"
            ].mean()
        )

        object_rows.append({
            "object":
                path.name,

            "pool_success_each":
                args.pool_success,

            "pool_failure_each":
                args.pool_failure,

            "exploration_attempts":
                args.exploration_attempts,

            "exploitation_attempts":
                args.exploitation_attempts,

            "shared_ca_exploration_successes":
                random_explore_mean,

            "nearest_success_exploration_successes":
                nearest_explore_mean,

            "local_viability_exploration_successes":
                local_explore_mean,

            "connectivity_exploration_successes":
                connectivity_explore_mean,

            "random_exploitation_successes":
                random_exploit_mean,

            "nearest_exploitation_successes":
                nearest_exploit_mean,

            "local_exploitation_successes":
                local_exploit_mean,

            "connectivity_exploitation_successes":
                connectivity_exploit_mean,

            "random_exploitation_auc":
                random_auc,

            "nearest_exploitation_auc":
                nearest_auc,

            "local_exploitation_auc":
                local_auc,

            "connectivity_exploitation_auc":
                connectivity_auc,

            "random_exploitation_normalized_auc":
                random_auc_norm,

            "nearest_exploitation_normalized_auc":
                nearest_auc_norm,

            "local_exploitation_normalized_auc":
                local_auc_norm,

            "connectivity_exploitation_normalized_auc":
                connectivity_auc_norm,

            "nearest_gain_vs_random_percent":
                nearest_gain_random,

            "local_gain_vs_random_percent":
                local_gain_random,

            "ca_gain_vs_random_percent":
                ca_gain_random,
        })

        # ====================================================
        # OBJECT REPORT
        # ====================================================

        print(
            f"  Phase-2 AUC: "
            f"Random={random_auc:.1f}, "
            f"Nearest={nearest_auc:.1f}, "
            f"Local={local_auc:.1f}, "
            f"CA={connectivity_auc:.1f}"
        )

        print(
            f"  Phase-2 normalized AUC: "
            f"Random={random_auc_norm:.4f}, "
            f"Nearest={nearest_auc_norm:.4f}, "
            f"Local={local_auc_norm:.4f}, "
            f"CA={connectivity_auc_norm:.4f}"
        )

        print(
            f"  Nearest phase-2 gain vs Random: "
            f"{nearest_gain_random:+.2f}%"
        )

        print(
            f"  Local phase-2 gain vs Random:  "
            f"{local_gain_random:+.2f}%"
        )

        print(
            f"  CA phase-2 gain vs Random:  "
            f"{ca_gain_random:+.2f}%"
        )

        # Checkpoint after every object.
        save_checkpoints(
            results_dir=results_dir,
            curve_rows=curve_rows,
            seed_rows=seed_rows,
            object_rows=object_rows,
            selection_rows=selection_rows,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    object_df = pd.DataFrame(
        object_rows
    )

    seed_df = pd.DataFrame(
        seed_rows
    )

    print()
    print(
        "=" * 78
    )
    print(
        "FINAL E3 CA-EXPLORE -> MULTI-EXPLOIT SUMMARY"
    )
    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # Method-level summary
    # --------------------------------------------------------

    method_summary_rows = []

    for method in [
        "random",
        "nearest_success",
        "local_viability",
        "connectivity",
    ]:

        x = seed_df[
            seed_df[
                "method"
            ] == method
            ]

        method_summary_rows.append({
            "method":
                method,

            "mean_exploration_successes":
                x[
                    "exploration_successes"
                ].mean(),

            "median_exploration_successes":
                x[
                    "exploration_successes"
                ].median(),

            "mean_exploitation_successes":
                x[
                    "exploitation_successes"
                ].mean(),

            "median_exploitation_successes":
                x[
                    "exploitation_successes"
                ].median(),

            "mean_exploitation_normalized_auc":
                x[
                    "exploitation_normalized_auc"
                ].mean(),

            "median_exploitation_normalized_auc":
                x[
                    "exploitation_normalized_auc"
                ].median(),
        })

    method_summary_df = pd.DataFrame(
        method_summary_rows
    )

    # --------------------------------------------------------
    # Gain summary: Nearest vs Random and CA vs Random
    # --------------------------------------------------------

    gain_summary_df = pd.DataFrame([{
        "objects":
            len(
                object_df
            ),

        "exploration_attempts":
            args.exploration_attempts,

        "exploitation_attempts":
            args.exploitation_attempts,

        "median_nearest_gain_vs_random_percent":
            object_df[
                "nearest_gain_vs_random_percent"
            ].median(),

        "q25_nearest_gain_vs_random_percent":
            object_df[
                "nearest_gain_vs_random_percent"
            ].quantile(
                0.25
            ),

        "q75_nearest_gain_vs_random_percent":
            object_df[
                "nearest_gain_vs_random_percent"
            ].quantile(
                0.75
            ),

        "mean_nearest_gain_vs_random_percent":
            object_df[
                "nearest_gain_vs_random_percent"
            ].mean(),

        "fraction_nearest_beats_random":
            (
                    object_df[
                        "nearest_gain_vs_random_percent"
                    ] > 0
            ).mean(),

        "median_local_gain_vs_random_percent":
            object_df[
                "local_gain_vs_random_percent"
            ].median(),

        "q25_local_gain_vs_random_percent":
            object_df[
                "local_gain_vs_random_percent"
            ].quantile(
                0.25
            ),

        "q75_local_gain_vs_random_percent":
            object_df[
                "local_gain_vs_random_percent"
            ].quantile(
                0.75
            ),

        "mean_local_gain_vs_random_percent":
            object_df[
                "local_gain_vs_random_percent"
            ].mean(),

        "fraction_local_beats_random":
            (
                    object_df[
                        "local_gain_vs_random_percent"
                    ] > 0
            ).mean(),

        "median_ca_gain_vs_random_percent":
            object_df[
                "ca_gain_vs_random_percent"
            ].median(),

        "q25_ca_gain_vs_random_percent":
            object_df[
                "ca_gain_vs_random_percent"
            ].quantile(
                0.25
            ),

        "q75_ca_gain_vs_random_percent":
            object_df[
                "ca_gain_vs_random_percent"
            ].quantile(
                0.75
            ),

        "mean_ca_gain_vs_random_percent":
            object_df[
                "ca_gain_vs_random_percent"
            ].mean(),

        "fraction_ca_beats_random":
            (
                    object_df[
                        "ca_gain_vs_random_percent"
                    ] > 0
            ).mean(),
    }])

    print()
    print(
        "Method performance:"
    )

    print(
        method_summary_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    print()
    print(
        "Phase-2 gains relative to Random:"
    )

    print(
        gain_summary_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE FINAL SUMMARIES
    # ========================================================

    method_summary_path = (
            results_dir
            / "method_summary.csv"
    )

    gain_summary_path = (
            results_dir
            / "gain_summary.csv"
    )

    method_summary_df.to_csv(
        method_summary_path,
        index=False,
    )

    gain_summary_df.to_csv(
        gain_summary_path,
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        f"  "
        f"{results_dir / 'selected_objects.csv'}"
    )

    print(
        f"  "
        f"{results_dir / 'exploitation_curves.csv'}"
    )

    print(
        f"  "
        f"{results_dir / 'phase_results_per_seed.csv'}"
    )

    print(
        f"  "
        f"{results_dir / 'object_summary.csv'}"
    )

    print(
        f"  "
        f"{method_summary_path}"
    )

    print(
        f"  "
        f"{gain_summary_path}"
    )


if __name__ == "__main__":
    main()