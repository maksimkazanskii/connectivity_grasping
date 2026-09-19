#!/usr/bin/env python3
"""
E4 computational timing benchmark.

Uses the SAME core functions as:
    src/e4/transfer_experiment_sanity_independ_objects.py

Measures, for Q in {100,200,400} by default:
  * Random: Pool-A exploration, Pool-B ranking, total
  * FPS:    Pool-A exploration, Pool-B ranking, total
  * CA:     Pool-A exploration, Pool-B ranking, total
  * E4:     CA Pool-A exploration, one final reference retrieval,
            memory-augmented Pool-B scoring/ranking, total,
            for M in {200,400,600,800}

Important:
  * Data loading and the common target distance-matrix construction are NOT
    included in method runtime, because all methods use them.
  * For E4, retrieval is performed exactly once after all Q Pool-A
    observations have been collected. Retrieval timing includes preparation
    of the nearest-success / nearest-failure reference geometry.
  * Pool-B E4 timing includes loading the selected reference history,
    constructing the augmented distance matrix, computing memory scores,
    blending with CA scores, and sorting.
  * Reference-library construction is an offline cost and is not timed here.
  * alpha is fixed because retrieval and memory-score computation do not
    depend on alpha; alpha only changes the final inexpensive blend/sort.

Example:
    python src/e4/timing_experiment.py \
        --attempts 100 200 400 \
        --library_sizes 200 400 600 800 \
        --n_targets 5 \
        --n_seeds 5 \
        --alpha 0.10

Output:
    results/e4_timing/timing_raw.csv
    results/e4_timing/timing_summary.csv
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Import the exact implementation used by E4.
try:
    from src.e4 import transfer_experiment_sanity_independ_objects as e4
except ImportError as exc:
    raise ImportError(
        "Could not import src.e4.transfer_experiment_sanity_independ_objects. "
        "Run this script from the repository root and keep it under src/e4/."
    ) from exc


POOL_SUCCESS = 400
POOL_FAILURE = 400
EPS_VALUES = np.linspace(0.1, 0.8, 15)


def now():
    return time.perf_counter()


def ms(seconds):
    return 1000.0 * float(seconds)


def load_fixed_targets(args, reference_df):
    target_csv = Path(args.library_dir) / "target_candidate_objects.csv"
    if not target_csv.exists():
        raise FileNotFoundError(target_csv)

    target_df = pd.read_csv(target_csv)
    required = {"object_id", "path"}
    missing = required - set(target_df.columns)
    if missing:
        raise ValueError(f"Target CSV missing columns: {sorted(missing)}")

    # Match E4: target objects cannot occur in any memory bank.
    max_m = max(args.library_sizes)
    memory_order = (
        reference_df.sort_values("reference_rank")
        .head(max_m)
        .copy()
        .reset_index(drop=True)
    )
    if len(memory_order) < max_m:
        raise RuntimeError(
            f"Need {max_m} reference memories, found {len(memory_order)}."
        )

    memory_ids = set(memory_order["object_id"].astype(str))
    target_df = target_df[
        ~target_df["object_id"].astype(str).isin(memory_ids)
    ].copy().reset_index(drop=True)

    if len(target_df) < args.n_targets:
        raise RuntimeError(
            f"Only {len(target_df)} eligible targets; requested {args.n_targets}."
        )

    rng = np.random.default_rng(args.seed)
    positions = rng.choice(len(target_df), size=args.n_targets, replace=False)
    targets = target_df.iloc[positions].copy().reset_index(drop=True)

    memory_banks = {
        M: memory_order.iloc[:M].copy().reset_index(drop=True)
        for M in args.library_sizes
    }
    return targets, memory_banks


def time_baselines(
        *,
        labels,
        D,
        D_a,
        pool_a,
        pool_b,
        Q,
        bootstrap,
        base_seed,
):
    rows = []

    # ---------------- Random ----------------
    rng = np.random.default_rng(base_seed + 1000)
    t0 = now()
    local_order = e4.random_order(len(pool_a), rng)
    random_explore = pool_a[local_order[:Q]]
    t_a = now() - t0

    t0 = now()
    _ = e4.nearest_success_exploitation_order(
        exploration_order=random_explore,
        exploitation_candidates=pool_b,
        labels=labels,
        D=D,
        n_attempts=Q,
        rng=np.random.default_rng(base_seed + 10000),
    )
    t_b = now() - t0

    rows.append({
        "method": "Random",
        "library_size": 0,
        "pool_a_total_s": t_a,
        "pool_a_per_query_ms": ms(t_a) / Q,
        "retrieval_total_s": np.nan,
        "retrieval_updates": 0,
        "retrieval_ms": np.nan,
        "pool_b_ranking_ms": ms(t_b),
        "total_s": t_a + t_b,
    })

    # ---------------- FPS ----------------
    rng = np.random.default_rng(base_seed + 2000)
    t0 = now()
    fps_local_order = e4.fps_order(
        D=D_a,
        max_points=Q,
        rng=rng,
    )
    fps_explore = pool_a[fps_local_order]
    t_a = now() - t0

    t0 = now()
    _ = e4.nearest_success_exploitation_order(
        exploration_order=fps_explore,
        exploitation_candidates=pool_b,
        labels=labels,
        D=D,
        n_attempts=Q,
        rng=np.random.default_rng(base_seed + 10000),
    )
    t_b = now() - t0

    rows.append({
        "method": "FPS",
        "library_size": 0,
        "pool_a_total_s": t_a,
        "pool_a_per_query_ms": ms(t_a) / Q,
        "retrieval_total_s": np.nan,
        "retrieval_updates": 0,
        "retrieval_ms": np.nan,
        "pool_b_ranking_ms": ms(t_b),
        "total_s": t_a + t_b,
    })

    # ---------------- CA ----------------
    rng = np.random.default_rng(base_seed + 3000)
    t0 = now()
    ca_explore = e4.hidden_connectivity_exploration_on_pool(
        labels=labels,
        D=D,
        eps_values=EPS_VALUES,
        candidate_indices=pool_a,
        n_attempts=Q,
        rng=rng,
        bootstrap=bootstrap,
    )
    t_a = now() - t0

    t0 = now()
    _ = e4.connectivity_exploitation_order(
        exploration_order=ca_explore,
        exploitation_candidates=pool_b,
        labels=labels,
        D=D,
        eps_values=EPS_VALUES,
        n_attempts=Q,
        rng=np.random.default_rng(base_seed + 10000),
    )
    t_b = now() - t0

    rows.append({
        "method": "CA",
        "library_size": 0,
        "pool_a_total_s": t_a,
        "pool_a_per_query_ms": ms(t_a) / Q,
        "retrieval_total_s": np.nan,
        "retrieval_updates": 0,
        "retrieval_ms": np.nan,
        "pool_b_ranking_ms": ms(t_b),
        "total_s": t_a + t_b,
    })

    return rows


def time_e4(
        *,
        transforms,
        labels,
        D,
        pool_a,
        pool_b,
        scale_p,
        scale_r,
        memory_bank,
        Q,
        M,
        alpha,
        bootstrap,
        base_seed,
):
    # Pool A is ordinary CA exactly as in the E4 experiment.
    transfer_rng = np.random.default_rng(base_seed + 3000)

    t0 = now()
    transfer_explore = e4.hidden_connectivity_exploration_on_pool(
        labels=labels,
        D=D,
        eps_values=EPS_VALUES,
        candidate_indices=pool_a,
        n_attempts=Q,
        rng=transfer_rng,
        bootstrap=bootstrap,
    )
    t_a = now() - t0

    # Target-only CA Pool-B scores are part of the E4 scoring path.
    # Time them with Pool-B because they are needed to form S_{M,alpha}.
    t0_b = now()
    ca_scores = e4.connectivity_exploitation_order(
        exploration_order=transfer_explore,
        exploitation_candidates=pool_b,
        labels=labels,
        D=D,
        eps_values=EPS_VALUES,
        n_attempts=Q,
        rng=transfer_rng,
        return_scores=True,
    )
    ca_score_time = now() - t0_b

    # Retrieval is performed exactly once, after the complete Pool-A
    # trajectory has been generated. Preparation of the reference geometry
    # is included in the retrieval runtime.
    e4.clear_memory_caches(keep_reference_files=False)
    t0 = now()
    prepared_geometry = e4.prepare_margin_retrieval_geometry(
        target_transforms=transforms,
        candidate_df=memory_bank,
        scale_p=scale_p,
        scale_r=scale_r,
    )

    observed_pool_a = np.asarray(transfer_explore, dtype=int)
    reference, _ = e4.retrieve_labeled_memory_online(
        observed_indices=observed_pool_a,
        labels=labels,
        candidate_df=memory_bank,
        prepared_geometry=prepared_geometry,
        query_number=Q,
    )
    t_retrieval = now() - t0
    n_updates = 1

    # Pool B: selected-memory loading + augmented geometry + memory score
    # + alpha blend + fixed ranking.
    t0 = now()

    selected_row = memory_bank[
        memory_bank["object_id"].astype(str)
        == str(reference["reference_object"])
        ].iloc[0]

    sel_successes, _, _ = e4.load_reference_successes(selected_row)
    sel_failures = e4.load_reference_failures_from_path(
        selected_row["memory_path"]
    )

    memory_transforms = np.concatenate(
        [sel_successes, sel_failures], axis=0
    )
    all_aug = np.concatenate([transforms, memory_transforms], axis=0)

    D_aug = e4.distance_matrix_fixed_scales(
        all_aug, scale_p=scale_p, scale_r=scale_r
    ).astype(np.float32, copy=False)

    n_target = len(transforms)
    mem_success_idx = np.arange(
        n_target, n_target + len(sel_successes), dtype=int
    )
    mem_failure_idx = np.arange(
        n_target + len(sel_successes), len(all_aug), dtype=int
    )

    labels_aug = np.concatenate([
        labels,
        np.ones(len(sel_successes), dtype=int),
        np.zeros(len(sel_failures), dtype=int),
    ])

    observed_structure = np.concatenate([
        mem_success_idx,
        mem_failure_idx,
        transfer_explore,
    ]).astype(int)

    memory_scores = e4.connectivity_exploitation_order(
        exploration_order=observed_structure,
        exploitation_candidates=pool_b,
        labels=labels_aug,
        D=D_aug,
        eps_values=EPS_VALUES,
        n_attempts=Q,
        rng=transfer_rng,
        return_scores=True,
    )

    blended_scores = (
            (1.0 - alpha) * ca_scores + alpha * memory_scores
    )
    local_order = np.argsort(-blended_scores, kind="stable")
    _ = np.asarray(pool_b[local_order[:Q]], dtype=int)

    memory_b_time = now() - t0
    t_b = ca_score_time + memory_b_time

    return {
        "method": "E4",
        "library_size": M,
        "pool_a_total_s": t_a,
        "pool_a_per_query_ms": ms(t_a) / Q,
        "retrieval_total_s": t_retrieval,
        "retrieval_updates": 1,
        "retrieval_ms": ms(t_retrieval),
        "pool_b_ranking_ms": ms(t_b),
        "total_s": t_a + t_retrieval + t_b,
    }


def summarize(raw):
    keys = ["Q", "library_size", "method"]
    metrics = [
        "pool_a_per_query_ms",
        "retrieval_ms",
        "pool_b_ranking_ms",
        "total_s",
    ]

    out = (
        raw.groupby(keys, dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out.columns = [
        "_".join(str(x) for x in col if str(x))
        if isinstance(col, tuple) else col
        for col in out.columns
    ]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Timing benchmark for E4 vs Random/FPS/CA."
    )
    parser.add_argument(
        "--library_dir",
        type=str,
        default="results/e4_reference_library",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        nargs="+",
        default=[100, 200, 400],
        help="Per-phase query budgets Q.",
    )
    parser.add_argument(
        "--library_sizes",
        type=int,
        nargs="+",
        default=[200, 400, 600, 800],
    )
    parser.add_argument("--n_targets", type=int, default=5)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/e4_timing",
    )
    args = parser.parse_args()

    if max(args.attempts) > POOL_SUCCESS + POOL_FAILURE:
        raise ValueError("Q exceeds Pool-A/Pool-B size.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_df = e4.load_reference_index(Path(args.library_dir))
    targets, memory_banks = load_fixed_targets(args, reference_df)

    targets.to_csv(out_dir / "timing_target_objects.csv", index=False)

    raw_rows = []

    print("=" * 78)
    print("E4 TIMING BENCHMARK")
    print("=" * 78)
    print(f"Q: {args.attempts}")
    print(f"M: {args.library_sizes}")
    print(f"targets: {args.n_targets}")
    print(f"seeds: {args.n_seeds}")
    print(f"alpha: {args.alpha}")
    print("Common data loading / target D construction excluded from method timing.")
    print()

    for target_number, target_row in targets.iterrows():
        object_number = target_number + 1
        object_id = str(target_row["object_id"])
        path = Path(target_row["path"])

        pool_rng = np.random.default_rng(
            args.seed + object_number * 10000
        )
        (
            transforms,
            labels,
            pool_a,
            pool_b,
            _original_indices,
        ) = e4.load_two_balanced_pools(
            path=path,
            pool_success=POOL_SUCCESS,
            pool_failure=POOL_FAILURE,
            rng=pool_rng,
            excluded_transforms=None,
        )

        # Common preprocessing, excluded from method runtime.
        D, scale_p, scale_r = e4.distance_matrix_with_reference_scaling(
            transforms=transforms,
            reference_indices=pool_a,
        )
        D_a = D[np.ix_(pool_a, pool_a)]

        for seed_index in range(args.n_seeds):
            base_seed = (
                    args.seed
                    + object_number * 100000
                    + seed_index
            )

            for Q in args.attempts:
                print(
                    f"target {object_number}/{args.n_targets} "
                    f"seed {seed_index + 1}/{args.n_seeds} Q={Q}"
                )

                baseline_rows = time_baselines(
                    labels=labels,
                    D=D,
                    D_a=D_a,
                    pool_a=pool_a,
                    pool_b=pool_b,
                    Q=Q,
                    bootstrap=args.bootstrap,
                    base_seed=base_seed,
                )
                for row in baseline_rows:
                    row.update({
                        "target": object_id,
                        "seed": seed_index + 1,
                        "Q": Q,
                        "alpha": np.nan,
                    })
                    raw_rows.append(row)

                for M in args.library_sizes:
                    row = time_e4(
                        transforms=transforms,
                        labels=labels,
                        D=D,
                        pool_a=pool_a,
                        pool_b=pool_b,
                        scale_p=scale_p,
                        scale_r=scale_r,
                        memory_bank=memory_banks[M],
                        Q=Q,
                        M=M,
                        alpha=args.alpha,
                        bootstrap=args.bootstrap,
                        base_seed=base_seed,
                    )
                    row.update({
                        "target": object_id,
                        "seed": seed_index + 1,
                        "Q": Q,
                        "alpha": args.alpha,
                    })
                    raw_rows.append(row)

                # Save incrementally so a long benchmark is recoverable.
                raw = pd.DataFrame(raw_rows)
                raw.to_csv(out_dir / "timing_raw.csv", index=False)
                summarize(raw).to_csv(
                    out_dir / "timing_summary.csv", index=False
                )

        del D, D_a, transforms, labels, pool_a, pool_b
        e4.clear_memory_caches(keep_reference_files=False)
        gc.collect()

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)

    raw.to_csv(out_dir / "timing_raw.csv", index=False)
    summary.to_csv(out_dir / "timing_summary.csv", index=False)

    print()
    print("=" * 78)
    print("TIMING COMPLETE")
    print("=" * 78)
    print(f"Raw:     {out_dir / 'timing_raw.csv'}")
    print(f"Summary: {out_dir / 'timing_summary.csv'}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
