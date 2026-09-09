# src/e4/build_reference_library.py

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.utils.grasp_io import find_objects
from src.utils.samplers import choose_connectivity_point


# ============================================================
# E4: BUILD REFERENCE EXPERIENCE LIBRARY
# ============================================================
#
# For every reference object:
#
#   hidden candidate pool:
#       800 successful + 800 failed = 1600 candidates
#
#   CA exploration:
#       800 queries
#
#   stored reference memory:
#       all OBSERVED successful queries
#       all OBSERVED failed queries
#
# The stored memory is NOT forced to be balanced.
#
# Reference and target objects are separated at the OBJECT level.
#
# Nested libraries:
#
#       L_100 subset L_200 subset L_400 subset L_800
#
# are defined by reference_rank.
# ============================================================


# ============================================================
# BASIC DATA ACCESS
# ============================================================

def load_labels(path):
    """Load ACRONYM binary grasp-success labels."""

    with h5py.File(path, "r") as f:
        labels = f[
            "grasps/qualities/flex/object_in_gripper"
        ][:].astype(np.int8)

    return labels


def count_labels(path):
    """Return number of successful and failed grasps."""

    labels = load_labels(path)

    return (
        int(np.sum(labels == 1)),
        int(np.sum(labels == 0)),
    )


def stable_object_id(path, grasp_dir):
    """Stable object ID based on path relative to grasp root."""

    path = Path(path).resolve()
    grasp_dir = Path(grasp_dir).resolve()

    try:
        return path.relative_to(grasp_dir).as_posix()
    except ValueError:
        return path.as_posix()


# ============================================================
# BALANCED HIDDEN CANDIDATE POOL
# ============================================================

def load_balanced_hidden_pool(
        path,
        n_success,
        n_failure,
        rng,
):
    """
    Construct a balanced hidden-label candidate pool.

    The exploration policy receives candidate transforms/distances,
    but labels are revealed only after querying.
    """

    with h5py.File(path, "r") as f:

        labels_all = f[
            "grasps/qualities/flex/object_in_gripper"
        ][:].astype(np.int8)

        success_indices = np.flatnonzero(
            labels_all == 1
        )

        failure_indices = np.flatnonzero(
            labels_all == 0
        )

        if len(success_indices) < n_success:
            raise ValueError(
                f"{Path(path).name}: "
                f"{len(success_indices)} successes available, "
                f"need {n_success}."
            )

        if len(failure_indices) < n_failure:
            raise ValueError(
                f"{Path(path).name}: "
                f"{len(failure_indices)} failures available, "
                f"need {n_failure}."
            )

        chosen_success = rng.choice(
            success_indices,
            size=n_success,
            replace=False,
        )

        chosen_failure = rng.choice(
            failure_indices,
            size=n_failure,
            replace=False,
        )

        original_indices = np.concatenate([
            chosen_success,
            chosen_failure,
        ]).astype(np.int32)

        # Hide the success/failure ordering.
        rng.shuffle(original_indices)

        transforms_all = f[
            "grasps/transforms"
        ][:]

        transforms = transforms_all[
            original_indices
        ]

        labels = labels_all[
            original_indices
        ]

    assert int(np.sum(labels == 1)) == n_success
    assert int(np.sum(labels == 0)) == n_failure

    return (
        transforms,
        labels,
        original_indices,
    )


# ============================================================
# NORMALIZED SE(3) DISTANCE
# ============================================================

def distance_matrix(transforms):
    """
    Compute normalized SE(3) distance matrix.

    Translation and rotation scales are estimated from the entire
    candidate pool. Viability labels are not used.
    """

    transforms = np.asarray(transforms)

    positions = transforms[
        :,
        :3,
        3
    ]

    rotations = transforms[
        :,
        :3,
        :3
    ]

    # Translation distances.
    dp = np.linalg.norm(
        positions[:, None, :]
        -
        positions[None, :, :],
        axis=-1,
        )

    # SO(3) geodesic distances.
    relative = np.einsum(
        "aik,bil->abkl",
        rotations,
        rotations,
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

    positive_dp = dp[dp > 0]
    positive_dr = dr[dr > 0]

    if len(positive_dp) == 0:
        raise ValueError(
            "Cannot estimate translation scale: "
            "all translation distances are zero."
        )

    if len(positive_dr) == 0:
        raise ValueError(
            "Cannot estimate rotation scale: "
            "all rotational distances are zero."
        )

    scale_p = float(
        np.median(positive_dp)
    )

    scale_r = float(
        np.median(positive_dr)
    )

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
# CONNECTIVITY-AWARE EXPLORATION
# ============================================================

def hidden_connectivity_exploration(
        labels,
        D,
        eps_values,
        n_attempts,
        rng,
        bootstrap=5,
):
    """
    Hidden-viability connectivity-aware exploration.

    Only successful queried candidates enter the connectivity graph.
    Failures consume attempts but do not enter the graph.
    """

    n_candidates = len(labels)

    if n_attempts > n_candidates:
        raise ValueError(
            f"n_attempts={n_attempts} exceeds "
            f"candidate pool size={n_candidates}."
        )

    available = np.ones(
        n_candidates,
        dtype=bool,
    )

    queried = []
    successful = []

    # --------------------------------------------------------
    # Random bootstrap
    # --------------------------------------------------------

    bootstrap_size = min(
        bootstrap,
        n_attempts,
    )

    for _ in range(bootstrap_size):

        candidates = np.flatnonzero(
            available
        )

        chosen = int(
            rng.choice(candidates)
        )

        queried.append(chosen)

        available[chosen] = False

        # Reveal label after query.
        if labels[chosen] == 1:
            successful.append(chosen)

    # --------------------------------------------------------
    # Connectivity-aware exploration
    # --------------------------------------------------------

    while len(queried) < n_attempts:

        if len(successful) == 0:

            candidates = np.flatnonzero(
                available
            )

            chosen = int(
                rng.choice(candidates)
            )

        else:

            chosen = int(
                choose_connectivity_point(
                    selected=successful,
                    available=available,
                    D=D,
                    eps_values=eps_values,
                )
            )

        queried.append(chosen)

        available[chosen] = False

        # Reveal label after query.
        if labels[chosen] == 1:
            successful.append(chosen)

    return np.asarray(
        queried,
        dtype=np.int32,
    )


# ============================================================
# BUILD ONE REFERENCE MEMORY
# ============================================================

def build_reference_memory(
        path,
        pool_success,
        pool_failure,
        n_queries,
        eps_values,
        pool_rng,
        exploration_rng,
        bootstrap,
):
    """
    Build one reference-object experience.

    Timing is measured separately for:

        1. candidate-pool construction/loading
        2. SE(3) distance computation
        3. CA exploration
    """

    # --------------------------------------------------------
    # 1. Candidate pool
    # --------------------------------------------------------

    t0 = time.perf_counter()

    (
        transforms,
        labels,
        original_indices,
    ) = load_balanced_hidden_pool(
        path=path,
        n_success=pool_success,
        n_failure=pool_failure,
        rng=pool_rng,
    )

    t1 = time.perf_counter()

    # --------------------------------------------------------
    # 2. SE(3) metric
    # --------------------------------------------------------

    (
        D,
        scale_p,
        scale_r,
    ) = distance_matrix(
        transforms
    )

    t2 = time.perf_counter()

    # --------------------------------------------------------
    # 3. CA exploration
    # --------------------------------------------------------

    queried = hidden_connectivity_exploration(
        labels=labels,
        D=D,
        eps_values=eps_values,
        n_attempts=n_queries,
        rng=exploration_rng,
        bootstrap=bootstrap,
    )

    t3 = time.perf_counter()

    # --------------------------------------------------------
    # Construct observed memory
    # --------------------------------------------------------

    queried_labels = labels[
        queried
    ]

    success_mask = (
            queried_labels == 1
    )

    failure_mask = (
            queried_labels == 0
    )

    success_pool_indices = queried[
        success_mask
    ]

    failure_pool_indices = queried[
        failure_mask
    ]

    return {
        # Complete fixed candidate-pool IDs.
        "pool_original_indices":
            original_indices,

        # Query history.
        "queried_pool_indices":
            queried,

        "queried_original_indices":
            original_indices[queried],

        "queried_labels":
            queried_labels,

        # Observed successes.
        "success_pool_indices":
            success_pool_indices,

        "success_original_indices":
            original_indices[
                success_pool_indices
            ],

        "success_transforms":
            transforms[
                success_pool_indices
            ],

        # Observed failures.
        "failure_pool_indices":
            failure_pool_indices,

        "failure_original_indices":
            original_indices[
                failure_pool_indices
            ],

        "failure_transforms":
            transforms[
                failure_pool_indices
            ],

        # Metric.
        "scale_p":
            scale_p,

        "scale_r":
            scale_r,

        # Timing.
        "pool_seconds":
            t1 - t0,

        "distance_seconds":
            t2 - t1,

        "exploration_seconds":
            t3 - t2,

        "build_seconds":
            t3 - t0,
    }


# ============================================================
# SAVE REFERENCE MEMORY
# ============================================================

def save_reference_memory(
        output_path,
        memory,
):
    """Save one explored reference-object memory."""

    np.savez_compressed(
        output_path,

        pool_original_indices=(
            memory["pool_original_indices"]
        ),

        queried_pool_indices=(
            memory["queried_pool_indices"]
        ),

        queried_original_indices=(
            memory["queried_original_indices"]
        ),

        queried_labels=(
            memory["queried_labels"]
        ),

        success_pool_indices=(
            memory["success_pool_indices"]
        ),

        success_original_indices=(
            memory["success_original_indices"]
        ),

        success_transforms=(
            memory["success_transforms"]
        ),

        failure_pool_indices=(
            memory["failure_pool_indices"]
        ),

        failure_original_indices=(
            memory["failure_original_indices"]
        ),

        failure_transforms=(
            memory["failure_transforms"]
        ),

        scale_p=np.asarray(
            memory["scale_p"],
            dtype=np.float64,
        ),

        scale_r=np.asarray(
            memory["scale_r"],
            dtype=np.float64,
        ),
    )


# ============================================================
# TIME FORMATTING
# ============================================================

def format_time(seconds):
    """Human-readable duration."""

    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.1f} s"

    if seconds < 3600:
        return f"{seconds / 60:.1f} min"

    return f"{seconds / 3600:.2f} h"


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build the fixed E4 reference-object experience library."
        )
    )

    parser.add_argument(
        "grasp_dir",
        nargs="?",
        default="grasps",
    )

    parser.add_argument(
        "--n_reference",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--pool_success",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--pool_failure",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--n_queries",
        type=int,
        default=800,
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

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/e4_reference_library",
    )

    args = parser.parse_args()

    pool_size = (
            args.pool_success
            +
            args.pool_failure
    )

    if args.n_queries > pool_size:
        raise ValueError(
            "--n_queries cannot exceed candidate pool size."
        )

    grasp_dir = Path(
        args.grasp_dir
    )

    output_dir = Path(
        args.output_dir
    )

    memory_dir = (
            output_dir
            / "memories"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    memory_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eps_values = np.linspace(
        0.10,
        0.80,
        15,
    )

    # ========================================================
    # DISCOVER OBJECTS
    # ========================================================

    paths = find_objects(
        grasp_dir
    )

    if len(paths) == 0:
        raise RuntimeError(
            f"No object files found under {grasp_dir}."
        )

    paths = sorted(
        [Path(path) for path in paths],
        key=lambda x: str(x),
    )

    print()
    print("=" * 78)
    print("E4 REFERENCE LIBRARY")
    print("=" * 78)

    print(
        f"Found object files:       {len(paths)}"
    )

    print(
        f"Hidden pool/reference:    "
        f"{args.pool_success} S + "
        f"{args.pool_failure} F = "
        f"{pool_size}"
    )

    print(
        f"CA queries/reference:     "
        f"{args.n_queries}"
    )

    print(
        f"Requested references:     "
        f"{args.n_reference}"
    )

    # ========================================================
    # ELIGIBILITY SCAN
    # ========================================================

    print()
    print("Scanning eligibility...")

    eligibility_start = time.perf_counter()

    eligible_rows = []

    for i, path in enumerate(
            paths,
            start=1,
    ):

        (
            n_success,
            n_failure,
        ) = count_labels(
            path
        )

        if (
                n_success >= args.pool_success
                and
                n_failure >= args.pool_failure
        ):

            eligible_rows.append({
                "object_id":
                    stable_object_id(
                        path,
                        grasp_dir,
                    ),

                "path":
                    str(path.resolve()),

                "available_successes":
                    n_success,

                "available_failures":
                    n_failure,
            })

        if (
                i % 500 == 0
                or
                i == len(paths)
        ):

            elapsed = (
                    time.perf_counter()
                    -
                    eligibility_start
            )

            print(
                f"  scanned {i}/{len(paths)} | "
                f"eligible={len(eligible_rows)} | "
                f"time={format_time(elapsed)}"
            )

    eligibility_seconds = (
            time.perf_counter()
            -
            eligibility_start
    )

    eligible_df = pd.DataFrame(
        eligible_rows
    )

    if len(eligible_df) < args.n_reference:
        raise RuntimeError(
            f"Only {len(eligible_df)} eligible objects, "
            f"but {args.n_reference} requested."
        )

    if eligible_df[
        "object_id"
    ].duplicated().any():

        raise RuntimeError(
            "Object IDs are not unique."
        )

    print(
        f"Eligibility scan complete: "
        f"{len(eligible_df)} eligible | "
        f"{format_time(eligibility_seconds)}"
    )

    # ========================================================
    # FIXED REFERENCE/TARGET SPLIT
    # ========================================================

    split_rng = np.random.default_rng(
        args.seed
    )

    reference_positions = split_rng.choice(
        len(eligible_df),
        size=args.n_reference,
        replace=False,
    )

    reference_df = (
        eligible_df
        .iloc[reference_positions]
        .copy()
        .reset_index(drop=True)
    )

    reference_df.insert(
        0,
        "reference_rank",
        np.arange(
            1,
            args.n_reference + 1,
            dtype=int,
            ),
    )

    reference_ids = set(
        reference_df["object_id"]
    )

    target_df = (
        eligible_df[
            ~eligible_df[
                "object_id"
            ].isin(reference_ids)
        ]
        .copy()
        .reset_index(drop=True)
    )

    target_ids = set(
        target_df["object_id"]
    )

    overlap = (
            reference_ids
            &
            target_ids
    )

    if overlap:
        raise RuntimeError(
            "REFERENCE/TARGET OBJECT LEAKAGE: "
            f"{sorted(overlap)[:10]}"
        )

    assert (
            len(reference_df)
            +
            len(target_df)
            ==
            len(eligible_df)
    )

    # ========================================================
    # SAVE SPLIT
    # ========================================================

    eligible_df.to_csv(
        output_dir
        / "eligible_objects.csv",
        index=False,
        )

    reference_df.to_csv(
        output_dir
        / "reference_objects.csv",
        index=False,
        )

    target_df.to_csv(
        output_dir
        / "target_candidate_objects.csv",
        index=False,
        )

    with open(
            output_dir
            / "reference_object_ids.txt",
            "w",
    ) as f:

        for object_id in reference_df[
            "object_id"
        ]:
            f.write(
                f"{object_id}\n"
            )

    with open(
            output_dir
            / "target_candidate_object_ids.txt",
            "w",
    ) as f:

        for object_id in target_df[
            "object_id"
        ]:
            f.write(
                f"{object_id}\n"
            )

    manifest = {
        "experiment":
            "E4",

        "master_seed":
            int(args.seed),

        "n_reference":
            int(args.n_reference),

        "n_target_candidates":
            int(len(target_df)),

        "hidden_pool_success":
            int(args.pool_success),

        "hidden_pool_failure":
            int(args.pool_failure),

        "hidden_pool_size":
            int(pool_size),

        "queries_per_reference":
            int(args.n_queries),

        "bootstrap":
            int(args.bootstrap),

        "eps_values":
            eps_values.tolist(),

        "nested_library_sizes":
            [
                m
                for m in [
                100,
                200,
                400,
                800,
            ]
                if m <= args.n_reference
            ],

        "reference_target_overlap":
            0,

        "eligibility_scan_seconds":
            float(eligibility_seconds),
    }

    with open(
            output_dir
            / "manifest.json",
            "w",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("E4 OBJECT SPLIT")
    print("=" * 78)

    print(
        f"Eligible:          {len(eligible_df)}"
    )

    print(
        f"References:        {len(reference_df)}"
    )

    print(
        f"Target candidates: {len(target_df)}"
    )

    print(
        "Object overlap:    0"
    )

    # ========================================================
    # BUILD REFERENCE EXPERIENCES
    # ========================================================

    print()
    print("=" * 78)
    print("BUILDING REFERENCE EXPERIENCES")
    print("=" * 78)

    experiment_start = time.perf_counter()

    memory_rows = []

    for _, row in reference_df.iterrows():

        reference_rank = int(
            row["reference_rank"]
        )

        object_id = str(
            row["object_id"]
        )

        path = Path(
            row["path"]
        )

        object_start = time.perf_counter()

        print()
        print(
            f"[{reference_rank:03d}/{args.n_reference}] "
            f"{object_id}"
        )

        # Independent deterministic RNG streams.
        pool_seed = (
                args.seed
                +
                reference_rank * 100000
                +
                1000
        )

        exploration_seed = (
                args.seed
                +
                reference_rank * 100000
                +
                2000
        )

        pool_rng = np.random.default_rng(
            pool_seed
        )

        exploration_rng = np.random.default_rng(
            exploration_seed
        )

        # ----------------------------------------------------
        # Build reference memory
        # ----------------------------------------------------

        memory = build_reference_memory(
            path=path,
            pool_success=args.pool_success,
            pool_failure=args.pool_failure,
            n_queries=args.n_queries,
            eps_values=eps_values,
            pool_rng=pool_rng,
            exploration_rng=exploration_rng,
            bootstrap=args.bootstrap,
        )

        n_observed_success = len(
            memory[
                "success_original_indices"
            ]
        )

        n_observed_failure = len(
            memory[
                "failure_original_indices"
            ]
        )

        assert (
                n_observed_success
                +
                n_observed_failure
                ==
                args.n_queries
        )

        # ----------------------------------------------------
        # Save memory
        # ----------------------------------------------------

        memory_filename = (
            f"reference_{reference_rank:04d}.npz"
        )

        memory_path = (
                memory_dir
                /
                memory_filename
        )

        save_reference_memory(
            output_path=memory_path,
            memory=memory,
        )

        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        object_seconds = (
                time.perf_counter()
                -
                object_start
        )

        elapsed_seconds = (
                time.perf_counter()
                -
                experiment_start
        )

        objects_done = reference_rank

        objects_left = (
                args.n_reference
                -
                objects_done
        )

        mean_seconds_per_object = (
                elapsed_seconds
                /
                objects_done
        )

        eta_seconds = (
                mean_seconds_per_object
                *
                objects_left
        )

        estimated_total_seconds = (
                elapsed_seconds
                +
                eta_seconds
        )

        success_rate = (
                n_observed_success
                /
                args.n_queries
        )

        # ----------------------------------------------------
        # Index row
        # ----------------------------------------------------

        memory_rows.append({
            "reference_rank":
                reference_rank,

            "object_id":
                object_id,

            "source_path":
                str(path),

            "memory_path":
                str(memory_path.resolve()),

            "pool_seed":
                pool_seed,

            "exploration_seed":
                exploration_seed,

            "pool_success":
                args.pool_success,

            "pool_failure":
                args.pool_failure,

            "pool_size":
                pool_size,

            "n_queries":
                args.n_queries,

            "observed_successes":
                n_observed_success,

            "observed_failures":
                n_observed_failure,

            "observed_success_rate":
                success_rate,

            "scale_p":
                memory["scale_p"],

            "scale_r":
                memory["scale_r"],

            # Stage timing.
            "pool_seconds":
                memory["pool_seconds"],

            "distance_seconds":
                memory["distance_seconds"],

            "exploration_seconds":
                memory["exploration_seconds"],

            "build_seconds":
                memory["build_seconds"],

            # Full object timing includes saving.
            "runtime_seconds":
                object_seconds,

            "elapsed_seconds":
                elapsed_seconds,

            "eta_seconds":
                eta_seconds,

            "estimated_total_seconds":
                estimated_total_seconds,
        })

        # Checkpoint after every object.
        pd.DataFrame(
            memory_rows
        ).to_csv(
            output_dir
            / "reference_memory_index.csv",
            index=False,
            )

        # ----------------------------------------------------
        # Progress report
        # ----------------------------------------------------

        print(
            f"  observed:     "
            f"{n_observed_success} S + "
            f"{n_observed_failure} F "
            f"(success={success_rate:.3f})"
        )

        print(
            f"  pool:         "
            f"{format_time(memory['pool_seconds'])}"
        )

        print(
            f"  distances:    "
            f"{format_time(memory['distance_seconds'])}"
        )

        print(
            f"  CA explore:   "
            f"{format_time(memory['exploration_seconds'])}"
        )

        print(
            f"  object total: "
            f"{format_time(object_seconds)}"
        )

        print(
            f"  elapsed:      "
            f"{format_time(elapsed_seconds)}"
        )

        print(
            f"  mean/object:  "
            f"{format_time(mean_seconds_per_object)}"
        )

        print(
            f"  ETA:          "
            f"{format_time(eta_seconds)}"
        )

        print(
            f"  est. total:   "
            f"{format_time(estimated_total_seconds)}"
        )

    # ========================================================
    # FINAL TIMING
    # ========================================================

    total_seconds = (
            time.perf_counter()
            -
            experiment_start
    )

    memory_df = pd.DataFrame(
        memory_rows
    )

    # Add final timing to manifest.
    manifest[
        "reference_build_seconds"
    ] = float(total_seconds)

    manifest[
        "mean_seconds_per_reference"
    ] = float(
        total_seconds
        /
        args.n_reference
    )

    manifest[
        "total_script_seconds"
    ] = float(
        eligibility_seconds
        +
        total_seconds
    )

    with open(
            output_dir
            / "manifest.json",
            "w",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 78)
    print("E4 REFERENCE LIBRARY COMPLETE")
    print("=" * 78)

    print(
        f"Reference objects:             "
        f"{len(memory_df)}"
    )

    print(
        f"Mean observed successes:       "
        f"{memory_df['observed_successes'].mean():.2f}"
    )

    print(
        f"Mean observed failures:        "
        f"{memory_df['observed_failures'].mean():.2f}"
    )

    print(
        f"Mean CA success rate:          "
        f"{memory_df['observed_success_rate'].mean():.4f}"
    )

    print(
        f"Median CA success rate:        "
        f"{memory_df['observed_success_rate'].median():.4f}"
    )

    print()
    print("TIMING")

    print(
        f"Eligibility scan:              "
        f"{format_time(eligibility_seconds)}"
    )

    print(
        f"Reference construction:        "
        f"{format_time(total_seconds)}"
    )

    print(
        f"Mean/reference:                "
        f"{format_time(total_seconds / args.n_reference)}"
    )

    print(
        f"Mean pool construction:        "
        f"{format_time(memory_df['pool_seconds'].mean())}"
    )

    print(
        f"Mean distance construction:    "
        f"{format_time(memory_df['distance_seconds'].mean())}"
    )

    print(
        f"Mean CA exploration:           "
        f"{format_time(memory_df['exploration_seconds'].mean())}"
    )

    print(
        f"Total script time:             "
        f"{format_time(eligibility_seconds + total_seconds)}"
    )

    print()
    print("Saved:")

    print(
        f"  {output_dir / 'reference_objects.csv'}"
    )

    print(
        f"  {output_dir / 'reference_object_ids.txt'}"
    )

    print(
        f"  {output_dir / 'target_candidate_objects.csv'}"
    )

    print(
        f"  {output_dir / 'target_candidate_object_ids.txt'}"
    )

    print(
        f"  {output_dir / 'reference_memory_index.csv'}"
    )

    print(
        f"  {output_dir / 'manifest.json'}"
    )

    print(
        f"  {memory_dir}/"
    )

    print()
    print(
        "Verified: reference objects cannot be used as target objects."
    )


if __name__ == "__main__":
    main()