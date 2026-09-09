import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from src.utils.samplers import random_order, fps_order, choose_connectivity_point

MATCHED_SUBSETS_K = 10
LIBRARY_SIZES = [100, 200, 400, 800]
ALPHAS = [0.0, 0.01, 0.025, 0.05, 0.10, 0.25, 0.5, 1.0]
MEMORY_WARMUP_ATTEMPTS = 10


def raw_se3_distances(transforms_a, transforms_b):
    transforms_a = np.asarray(transforms_a)
    transforms_b = np.asarray(transforms_b)
    p_a = transforms_a[:, :3, 3]
    p_b = transforms_b[:, :3, 3]
    R_a = transforms_a[:, :3, :3]
    R_b = transforms_b[:, :3, :3]
    dp = np.linalg.norm(p_a[:, None, :] - p_b[None, :, :], axis=-1)
    relative = np.einsum('aik,bil->abkl', R_a, R_b)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    dr = np.arccos(cosine)
    return dp, dr


def distance_matrix_fixed_scales(transforms, scale_p, scale_r):
    dp, dr = raw_se3_distances(transforms, transforms)
    D = np.sqrt((dp / float(scale_p)) ** 2 + (dr / float(scale_r)) ** 2)
    return D.astype(np.float32)


def connectivity_signature_from_D(D, eps_values):
    n = len(D)
    if n == 0:
        raise ValueError('Cannot construct topology from zero points.')
    B, L = [], []
    for eps in eps_values:
        adjacency = (D <= eps).astype(np.int8)
        n_components, labels = connected_components(adjacency, directed=False, return_labels=True)
        component_sizes = np.bincount(labels)
        B.append(n_components / float(n))
        L.append(component_sizes.max() / float(n))
    return np.asarray(B, dtype=float), np.asarray(L, dtype=float)


def signature_distance(signature_a, signature_b):
    B_a, L_a = signature_a
    B_b, L_b = signature_b
    return 0.5 * (np.mean(np.abs(B_a - B_b)) + np.mean(np.abs(L_a - L_b)))


def connectivity_acquisition_scores(selected, candidate_indices, D, eps_values):
    selected = np.asarray(selected, dtype=int)
    candidates = np.asarray(candidate_indices, dtype=int)
    if len(candidates) == 0:
        return np.empty(0, dtype=float)
    if len(selected) == 0:
        return np.zeros(len(candidates), dtype=float)

    candidate_to_selected = D[np.ix_(candidates, selected)]
    nearest_selected = candidate_to_selected.min(axis=1)
    novelty = nearest_selected.copy()
    if novelty.max() > 0:
        novelty = novelty / novelty.max()

    structural_score = np.zeros(len(candidates), dtype=float)
    D_selected = D[np.ix_(selected, selected)]

    for eps in eps_values:
        adjacency = (D_selected <= eps).astype(np.int8)
        n_components, component_labels = connected_components(
            adjacency, directed=False, return_labels=True
        )
        touched = np.zeros(len(candidates), dtype=int)
        minimum_component_distance = np.full(len(candidates), np.inf, dtype=float)

        for component_id in range(n_components):
            component = selected[component_labels == component_id]
            distances = D[np.ix_(candidates, component)]
            distance_to_component = distances.min(axis=1)
            touched += distance_to_component <= eps
            minimum_component_distance = np.minimum(
                minimum_component_distance, distance_to_component
            )

        bridge = np.maximum(touched - 1, 0).astype(float)
        frontier = np.exp(
            -((minimum_component_distance - 1.25 * eps) / (0.5 * eps + 1e-08)) ** 2
        )
        frontier[touched > 0] *= 0.25

        boundary = np.zeros(len(candidates), dtype=float)
        one_component = touched == 1
        boundary[one_component] = np.clip(
            minimum_component_distance[one_component] / (eps + 1e-08), 0.0, 1.0
        )

        structural_score += 2.0 * bridge + 1.0 * frontier + 0.5 * boundary

    structural_score /= float(len(eps_values))
    if structural_score.max() > 0:
        structural_score = structural_score / structural_score.max()

    return structural_score + 0.2 * novelty


def minmax_normalize(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def load_two_balanced_pools(path, pool_success, pool_failure, rng):
    with h5py.File(path, 'r') as f:
        labels_all = f['grasps/qualities/flex/object_in_gripper'][:].astype(int)
        transforms_all = f['grasps/transforms'][:]

    success_idx = np.flatnonzero(labels_all == 1)
    failure_idx = np.flatnonzero(labels_all == 0)

    needed_success = 2 * int(pool_success)
    needed_failure = 2 * int(pool_failure)

    if len(success_idx) < needed_success:
        raise ValueError(
            f'{Path(path).name}: only {len(success_idx)} successes; need {needed_success}.'
        )
    if len(failure_idx) < needed_failure:
        raise ValueError(
            f'{Path(path).name}: only {len(failure_idx)} failures; need {needed_failure}.'
        )

    chosen_success = rng.choice(success_idx, size=needed_success, replace=False)
    chosen_failure = rng.choice(failure_idx, size=needed_failure, replace=False)

    pool_a_original = np.concatenate(
        [chosen_success[:pool_success], chosen_failure[:pool_failure]]
    )
    pool_b_original = np.concatenate(
        [chosen_success[pool_success:], chosen_failure[pool_failure:]]
    )

    rng.shuffle(pool_a_original)
    rng.shuffle(pool_b_original)

    combined_original = np.concatenate([pool_a_original, pool_b_original])
    transforms = transforms_all[combined_original]
    labels = labels_all[combined_original]

    per_pool_size = pool_success + pool_failure
    pool_a = np.arange(0, per_pool_size, dtype=int)
    pool_b = np.arange(per_pool_size, 2 * per_pool_size, dtype=int)

    return transforms, labels, pool_a, pool_b, combined_original


def distance_matrix_with_reference_scaling(transforms, reference_indices):
    transforms = np.asarray(transforms)
    reference_indices = np.asarray(reference_indices, dtype=int)

    dp, dr = raw_se3_distances(transforms, transforms)
    dp_ref = dp[np.ix_(reference_indices, reference_indices)]
    dr_ref = dr[np.ix_(reference_indices, reference_indices)]

    positive_dp = dp_ref[dp_ref > 0]
    positive_dr = dr_ref[dr_ref > 0]

    if len(positive_dp) == 0:
        raise ValueError('Pool A has no nonzero translation distances.')
    if len(positive_dr) == 0:
        raise ValueError('Pool A has no nonzero rotation distances.')

    scale_p = float(np.median(positive_dp))
    scale_r = float(np.median(positive_dr))
    D = np.sqrt((dp / scale_p) ** 2 + (dr / scale_r) ** 2)

    return D.astype(np.float32), scale_p, scale_r


def hidden_connectivity_exploration_on_pool(
        labels, D, eps_values, candidate_indices, n_attempts, rng, bootstrap=5
):
    candidate_indices = np.asarray(candidate_indices, dtype=int)

    if n_attempts > len(candidate_indices):
        raise ValueError('n_attempts exceeds Pool-A size.')

    available = np.zeros(len(labels), dtype=bool)
    available[candidate_indices] = True

    queried = []
    successful = []

    bootstrap_size = min(bootstrap, n_attempts)

    for _ in range(bootstrap_size):
        candidates = np.flatnonzero(available)
        chosen = int(rng.choice(candidates))
        queried.append(chosen)
        available[chosen] = False
        if labels[chosen] == 1:
            successful.append(chosen)

    while len(queried) < n_attempts:
        if len(successful) == 0:
            candidates = np.flatnonzero(available)
            chosen = int(rng.choice(candidates))
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

        if labels[chosen] == 1:
            successful.append(chosen)

    return np.asarray(queried, dtype=int)


def nearest_success_exploitation_order(
        exploration_order,
        exploitation_candidates,
        labels,
        D,
        n_attempts,
        rng,
):
    """
    E3 Phase-2 exploitation used after Random and FPS exploration.

    The successful anchor set is FROZEN at the successful grasps discovered
    during Phase 1. Pool-B outcomes never update the ranking.
    """
    exploration_order = np.asarray(exploration_order, dtype=int)
    exploitation_candidates = np.asarray(exploitation_candidates, dtype=int)

    if n_attempts > len(exploitation_candidates):
        raise ValueError(
            f"n_attempts={n_attempts} exceeds exploitation-pool size "
            f"{len(exploitation_candidates)}."
        )

    phase1_successful = np.asarray(
        [int(idx) for idx in exploration_order if labels[idx] == 1],
        dtype=int,
    )

    if len(phase1_successful) == 0:
        fallback_order = rng.permutation(exploitation_candidates)
        return np.asarray(fallback_order[:n_attempts], dtype=int)

    candidate_distances = D[
        np.ix_(exploitation_candidates, phase1_successful)
    ]
    nearest_phase1_success = candidate_distances.min(axis=1)

    local_order = np.argsort(
        nearest_phase1_success,
        kind="stable",
    )

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
    Exact E3 connectivity-aware Phase-2 exploitation for CA.

    Phase-1 observations are frozen. Successful Phase-1 grasps form
    multiscale connected components; failed Phase-1 grasps provide negative
    evidence. Pool-B labels/outcomes are never used to update the ranking.
    """
    exploration_order = np.asarray(exploration_order, dtype=int)
    exploitation_candidates = np.asarray(exploitation_candidates, dtype=int)

    if n_attempts > len(exploitation_candidates):
        raise ValueError(
            f"n_attempts={n_attempts} exceeds exploitation-pool size "
            f"{len(exploitation_candidates)}."
        )

    phase1_labels = labels[exploration_order]
    phase1_successful = exploration_order[phase1_labels == 1]
    phase1_failed = exploration_order[phase1_labels == 0]

    if len(phase1_successful) == 0:
        fallback_order = rng.permutation(exploitation_candidates)
        return np.asarray(fallback_order[:n_attempts], dtype=int)

    n_success = len(phase1_successful)
    n_failure = len(phase1_failed)

    d_candidate_success = D[
        np.ix_(exploitation_candidates, phase1_successful)
    ]

    if n_failure > 0:
        d_candidate_failure = D[
            np.ix_(exploitation_candidates, phase1_failed)
        ]
    else:
        d_candidate_failure = None

    d_success_success = D[
        np.ix_(phase1_successful, phase1_successful)
    ]

    candidate_scores = np.zeros(
        len(exploitation_candidates),
        dtype=float,
    )
    tiny = 1e-12

    for eps in eps_values:
        adjacency = d_success_success <= eps

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
            component_ids[root] = n_components

            while stack:
                node = stack.pop()
                neighbors = np.flatnonzero(
                    adjacency[node] & (component_ids == -1)
                )
                component_ids[neighbors] = n_components
                stack.extend(neighbors.tolist())

            n_components += 1

        success_kernel = np.exp(
            -(d_candidate_success ** 2) / (2.0 * eps ** 2)
        )

        positive_density = np.zeros(
            len(exploitation_candidates),
            dtype=float,
        )

        for component_id in range(n_components):
            members = component_ids == component_id
            component_density = (
                    success_kernel[:, members].sum(axis=1)
                    / float(n_success)
            )
            positive_density = np.maximum(
                positive_density,
                component_density,
            )

        if d_candidate_failure is None:
            negative_density = np.zeros(
                len(exploitation_candidates),
                dtype=float,
            )
        else:
            failure_kernel = np.exp(
                -(d_candidate_failure ** 2) / (2.0 * eps ** 2)
            )
            negative_density = (
                    failure_kernel.sum(axis=1)
                    / float(n_failure)
            )

        local_probability = (
                positive_density
                / (positive_density + negative_density + tiny)
        )
        candidate_scores += local_probability

    candidate_scores /= float(len(eps_values))

    local_order = np.argsort(
        -candidate_scores,
        kind="stable",
    )

    return np.asarray(
        exploitation_candidates[local_order[:n_attempts]],
        dtype=int,
    )

def load_reference_index(library_dir):
    path = Path(library_dir) / 'reference_memory_index.csv'

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {'reference_rank', 'object_id', 'memory_path'}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f'Reference index missing columns: {sorted(missing)}'
        )

    return df.sort_values('reference_rank').reset_index(drop=True)


def load_reference_successes(row):
    memory_path = Path(row['memory_path'])

    if not memory_path.exists():
        raise FileNotFoundError(memory_path)

    with np.load(memory_path, allow_pickle=False) as memory:
        successes = memory['success_transforms']
        scale_p = float(memory['scale_p'])
        scale_r = float(memory['scale_r'])

    return successes, scale_p, scale_r


def load_reference_failures_from_path(memory_path):
    with np.load(Path(memory_path), allow_pickle=False) as memory:
        return np.asarray(memory['failure_transforms'])


def sample_reference_signatures(
        success_transforms,
        subset_size,
        eps_values,
        scale_p,
        scale_r,
        n_subsets,
        rng,
):
    n_success = len(success_transforms)

    if subset_size > n_success:
        return None

    signatures = []

    for _ in range(n_subsets):
        subset_idx = rng.choice(
            n_success, size=subset_size, replace=False
        )
        subset = success_transforms[subset_idx]

        D_subset = distance_matrix_fixed_scales(
            subset,
            scale_p=scale_p,
            scale_r=scale_r,
        )

        signatures.append(
            connectivity_signature_from_D(D_subset, eps_values)
        )

    return signatures


def retrieve_reference(
        target_signature,
        subset_size,
        reference_df,
        library_size,
        eps_values,
        seed,
):
    library = reference_df[
        reference_df['reference_rank'] <= library_size
        ].copy().reset_index(drop=True)

    if len(library) != library_size:
        raise ValueError(
            f'Requested L{library_size}, but found '
            f'{len(library)} references.'
        )

    best = None
    retrieval_rows = []

    for _, row in library.iterrows():
        reference_rank = int(row['reference_rank'])
        successes, scale_p, scale_r = load_reference_successes(row)

        if len(successes) < subset_size:
            continue

        reference_rng = np.random.default_rng(
            seed
            + reference_rank * 100003
            + subset_size * 1009
        )

        signatures = sample_reference_signatures(
            success_transforms=successes,
            subset_size=subset_size,
            eps_values=eps_values,
            scale_p=scale_p,
            scale_r=scale_r,
            n_subsets=MATCHED_SUBSETS_K,
            rng=reference_rng,
        )

        if signatures is None:
            continue

        distances = np.asarray(
            [
                signature_distance(target_signature, signature)
                for signature in signatures
            ]
        )

        mean_distance = float(distances.mean())

        retrieval_rows.append(
            {
                'reference_rank': reference_rank,
                'reference_object': row['object_id'],
                'matched_size': subset_size,
                'K': MATCHED_SUBSETS_K,
                'mean_distance': mean_distance,
                'distance_std': float(distances.std()),
            }
        )

        if best is None or mean_distance < best['mean_distance']:
            best = {
                'reference_rank': reference_rank,
                'reference_object': row['object_id'],
                'memory_path': row['memory_path'],
                'mean_distance': mean_distance,
                'signatures_k': signatures,
                'success_transforms': successes,
                'scale_p': scale_p,
                'scale_r': scale_r,
            }

    if best is None:
        raise RuntimeError(
            'No reference object has enough observed successes '
            f'for matched size k={subset_size}.'
        )

    best['failure_transforms'] = load_reference_failures_from_path(
        best['memory_path']
    )

    return best, retrieval_rows


def reference_success_connectivity_profile(reference, eps_values):
    successes = np.asarray(reference['success_transforms'])
    n_ref = len(successes)

    if n_ref < 2:
        return np.zeros(len(eps_values), dtype=float)

    D_ref = distance_matrix_fixed_scales(
        successes,
        scale_p=reference['scale_p'],
        scale_r=reference['scale_r'],
    )

    profile = np.empty(len(eps_values), dtype=float)
    denom = float(n_ref - 1)

    for j, eps in enumerate(eps_values):
        degrees = (
                np.sum(D_ref <= eps, axis=1).astype(float) - 1.0
        )
        profile[j] = np.mean(degrees / denom)

    return profile


def target_candidate_connectivity_profiles(
        successful_indices,
        candidate_indices,
        D_target,
        eps_values,
):
    successful = np.asarray(successful_indices, dtype=int)
    candidates = np.asarray(candidate_indices, dtype=int)

    k = len(successful)

    profiles = np.zeros(
        (len(candidates), len(eps_values)),
        dtype=float,
    )

    if k == 0 or len(candidates) == 0:
        return profiles

    D_cs = D_target[np.ix_(candidates, successful)]

    for j, eps in enumerate(eps_values):
        profiles[:, j] = (
                np.sum(D_cs <= eps, axis=1) / float(k)
        )

    return profiles


def memory_assisted_connectivity_exploration_on_pool(
        labels,
        D,
        eps_values,
        candidate_indices,
        n_attempts,
        rng,
        bootstrap,
        reference_df,
        library_size,
        alpha,
        retrieval_seed,
        memory_warmup=10,
):
    """
    CA exploration with structural memory after a fixed warm-up.

    Queries 1..memory_warmup:
        ordinary CA only.

    Immediately after the warm-up:
        retrieve one homologous reference from the successful target
        topology observed so far.

    Remaining exploration:
        score(x) =
            (1-alpha) * normalized_CA(x)
            + alpha * normalized_memory(x)

    The retrieved reference is frozen, but the target-side memory
    profile is recomputed after each newly observed success.

    Memory is used ONLY during exploration.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=int)

    if n_attempts > len(candidate_indices):
        raise ValueError('n_attempts exceeds Pool-A size.')

    if memory_warmup < 1:
        raise ValueError('memory_warmup must be >= 1.')

    if memory_warmup > n_attempts:
        raise ValueError(
            'memory_warmup cannot exceed n_attempts.'
        )

    available = np.zeros(len(labels), dtype=bool)
    available[candidate_indices] = True

    queried = []
    successful = []

    reference = None
    retrieval_rows = []
    reference_profile = None

    # Same random bootstrap as ordinary CA.
    bootstrap_size = min(bootstrap, n_attempts)

    for _ in range(bootstrap_size):
        candidates = np.flatnonzero(available)
        chosen = int(rng.choice(candidates))

        queried.append(chosen)
        available[chosen] = False

        if labels[chosen] == 1:
            successful.append(chosen)

    # Ordinary CA until the 10-query warm-up is complete.
    while len(queried) < min(memory_warmup, n_attempts):
        candidates = np.flatnonzero(available)

        if len(successful) == 0:
            chosen = int(rng.choice(candidates))
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

        if labels[chosen] == 1:
            successful.append(chosen)

    # Homology comparison begins after exactly memory_warmup queries.
    if len(queried) < n_attempts and len(successful) > 0:
        successful_arr = np.asarray(successful, dtype=int)

        D_success = D[
            np.ix_(successful_arr, successful_arr)
        ]

        target_signature = connectivity_signature_from_D(
            D_success,
            eps_values,
        )

        reference, retrieval_rows = retrieve_reference(
            target_signature=target_signature,
            subset_size=len(successful_arr),
            reference_df=reference_df,
            library_size=library_size,
            eps_values=eps_values,
            seed=retrieval_seed,
        )

        reference_profile = (
            reference_success_connectivity_profile(
                reference,
                eps_values,
            )
        )

    # Remaining exploration uses CA + memory.
    while len(queried) < n_attempts:
        candidates = np.flatnonzero(available)

        if len(successful) == 0:
            chosen = int(rng.choice(candidates))
        else:
            successful_arr = np.asarray(
                successful,
                dtype=int,
            )

            ca_raw = connectivity_acquisition_scores(
                selected=successful_arr,
                candidate_indices=candidates,
                D=D,
                eps_values=eps_values,
            )
            ca_norm = minmax_normalize(ca_raw)

            if reference is None or alpha == 0.0:
                final_score = ca_norm
            else:
                candidate_profiles = (
                    target_candidate_connectivity_profiles(
                        successful_indices=successful_arr,
                        candidate_indices=candidates,
                        D_target=D,
                        eps_values=eps_values,
                    )
                )

                memory_raw = -np.mean(
                    np.abs(
                        candidate_profiles
                        - reference_profile[None, :]
                    ),
                    axis=1,
                )

                memory_norm = minmax_normalize(memory_raw)

                final_score = (
                        (1.0 - alpha) * ca_norm
                        + alpha * memory_norm
                )

            chosen = int(
                candidates[np.argmax(final_score)]
            )

        queried.append(chosen)
        available[chosen] = False

        if labels[chosen] == 1:
            successful.append(chosen)

    return (
        np.asarray(queried, dtype=int),
        reference,
        retrieval_rows,
    )


def cumulative_success_curve(order, labels):
    queried_labels = labels[np.asarray(order, dtype=int)]
    return np.cumsum(queried_labels == 1).astype(float)


def discovery_auc(curve):
    y = np.concatenate(
        [np.asarray([0.0]), np.asarray(curve, dtype=float)]
    )
    return float(np.sum(0.5 * (y[:-1] + y[1:])))


def maximum_discovery_auc(n_attempts, available_successes):
    k = int(n_attempts)
    s = min(int(available_successes), k)
    return s ** 2 / 2.0 + s * (k - s)


def normalized_discovery_auc(curve, available_successes):
    max_auc = maximum_discovery_auc(
        n_attempts=len(curve),
        available_successes=available_successes,
    )
    if max_auc == 0:
        return np.nan
    return discovery_auc(curve) / max_auc


def save_results(
        results_dir,
        curve_rows,
        seed_rows,
        retrieval_rows,
        target_rows,
):
    pd.DataFrame(curve_rows).to_csv(
        results_dir / 'exploitation_curves.csv',
        index=False,
        )
    pd.DataFrame(seed_rows).to_csv(
        results_dir / 'results_per_seed.csv',
        index=False,
        )
    pd.DataFrame(retrieval_rows).to_csv(
        results_dir / 'retrieval_results.csv',
        index=False,
        )
    pd.DataFrame(target_rows).to_csv(
        results_dir / 'selected_targets.csv',
        index=False,
        )


def print_target_summary(
        seed_rows,
        object_id,
        object_number,
        n_targets,
):
    rows = [
        row for row in seed_rows
        if row['target'] == object_id
    ]

    if not rows:
        return

    df = pd.DataFrame(rows)

    print()
    print('-' * 78)
    print(
        f'RESULTS — Target {object_number}/{n_targets}: '
        f'{object_id}'
    )
    print('-' * 78)

    print('Baselines:')

    for method in ['random', 'fps', 'ca']:
        values = df.loc[
            df['method'] == method,
            'normalized_auc',
        ].to_numpy(dtype=float)

        if len(values) == 0:
            continue

        print(
            f'  {method.upper():<10} '
            f'{np.mean(values):.4f} ± '
            f'{np.std(values):.4f}'
        )

    print()
    print('Transfer:')

    best = None

    for library_size in LIBRARY_SIZES:
        print(f'  M={library_size}')

        for alpha in ALPHAS[1:]:
            mask = (
                    (df['library_size'] == library_size)
                    & np.isclose(
                df['alpha'].astype(float),
                alpha,
            )
            )

            values = df.loc[
                mask,
                'normalized_auc',
            ].to_numpy(dtype=float)

            if len(values) == 0:
                continue

            mean_value = float(np.mean(values))
            std_value = float(np.std(values))

            print(
                f'    alpha={alpha:<4.2f}  '
                f'{mean_value:.4f} ± {std_value:.4f}'
            )

            if (
                    best is None
                    or mean_value > best['mean']
            ):
                best = {
                    'M': library_size,
                    'alpha': alpha,
                    'mean': mean_value,
                    'std': std_value,
                }

    ca_values = df.loc[
        df['method'] == 'ca',
        'normalized_auc',
    ].to_numpy(dtype=float)

    if best is not None and len(ca_values) > 0:
        ca_mean = float(np.mean(ca_values))

        print()
        print(
            f"Best transfer: M={best['M']}, "
            f"alpha={best['alpha']:.2f}, "
            f"nAUC={best['mean']:.4f}"
        )

        if ca_mean != 0:
            relative_gain = (
                    100.0
                    * (best['mean'] - ca_mean)
                    / ca_mean
            )
            print(
                f'Gain vs CA: {relative_gain:+.2f}%'
            )

        print(
            'Absolute Δ vs CA: '
            f"{best['mean'] - ca_mean:+.4f}"
        )

    print('-' * 78)


def main():
    parser = argparse.ArgumentParser(
        description=(
            'E4: 125 exploration + 125 exploitation. '
            'Memory affects CA exploration only after a '
            '10-query warm-up.'
        )
    )

    parser.add_argument(
        '--library_dir',
        type=str,
        default='results/e4_reference_library',
    )
    parser.add_argument(
        '--n_targets',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--n_seeds',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--bootstrap',
        type=int,
        default=5,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
    )

    args = parser.parse_args()

    # Kept exactly as in the supplied E4 benchmark except for the
    # requested attempt-budget change.
    POOL_SUCCESS = 400
    POOL_FAILURE = 400

    # Requested total budget:
    # 125 exploration + 125 exploitation = 250 attempts.
    EXPLORATION_ATTEMPTS = 125
    EXPLOITATION_ATTEMPTS = 125

    per_pool_size = POOL_SUCCESS + POOL_FAILURE

    if EXPLORATION_ATTEMPTS > per_pool_size:
        raise ValueError(
            'Exploration attempts cannot exceed Pool-A size.'
        )

    if EXPLOITATION_ATTEMPTS > per_pool_size:
        raise ValueError(
            'Exploitation attempts cannot exceed Pool-B size.'
        )

    if MEMORY_WARMUP_ATTEMPTS > EXPLORATION_ATTEMPTS:
        raise ValueError(
            'Memory warm-up cannot exceed exploration budget.'
        )

    library_dir = Path(args.library_dir)
    results_dir = (
            Path('results') / 'e4_transfer_results'
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eps_values = np.linspace(0.1, 0.8, 15)

    print('=' * 78)
    print('E4: MEMORY-ASSISTED EXPLORATION')
    print('=' * 78)
    print(
        f'Pool A: {POOL_SUCCESS} S + '
        f'{POOL_FAILURE} F = {per_pool_size}'
    )
    print(
        f'Pool B: {POOL_SUCCESS} S + '
        f'{POOL_FAILURE} F = {per_pool_size}'
    )
    print(
        f'Exploration attempts: '
        f'{EXPLORATION_ATTEMPTS}'
    )
    print(
        f'  ordinary CA warm-up: '
        f'{MEMORY_WARMUP_ATTEMPTS}'
    )
    print(
        f'  memory-assisted exploration: '
        f'{EXPLORATION_ATTEMPTS - MEMORY_WARMUP_ATTEMPTS}'
    )
    print(
        f'Exploitation attempts: '
        f'{EXPLOITATION_ATTEMPTS}'
    )
    print('Memory during exploitation: NO')
    print(
        f'Total attempt budget: '
        f'{EXPLORATION_ATTEMPTS + EXPLOITATION_ATTEMPTS}'
    )
    print(
        f'Matched topology subsets K: '
        f'{MATCHED_SUBSETS_K}'
    )
    print(f'Library sizes: {LIBRARY_SIZES}')
    print(f'Alpha values: {ALPHAS}')

    reference_df = load_reference_index(
        library_dir
    )

    if len(reference_df) < max(LIBRARY_SIZES):
        raise RuntimeError(
            f'Reference library contains '
            f'{len(reference_df)} objects; need at least '
            f'{max(LIBRARY_SIZES)}.'
        )

    target_csv = (
            library_dir / 'target_candidate_objects.csv'
    )

    if not target_csv.exists():
        raise FileNotFoundError(target_csv)

    target_df = pd.read_csv(target_csv)

    required_target_columns = {
        'object_id',
        'path',
    }

    missing = (
            required_target_columns
            - set(target_df.columns)
    )

    if missing:
        raise ValueError(
            'Target CSV missing columns: '
            f'{sorted(missing)}'
        )

    if len(target_df) < args.n_targets:
        raise RuntimeError(
            f'Only {len(target_df)} target candidates; '
            f'requested {args.n_targets}.'
        )

    master_rng = np.random.default_rng(
        args.seed
    )

    selected_positions = master_rng.choice(
        len(target_df),
        size=args.n_targets,
        replace=False,
    )

    selected_targets = (
        target_df
        .iloc[selected_positions]
        .copy()
        .reset_index(drop=True)
    )

    curve_rows = []
    seed_rows = []
    retrieval_output_rows = []
    target_rows = []

    for target_number, target_row in (
            selected_targets.iterrows()
    ):
        object_number = target_number + 1
        object_id = str(target_row['object_id'])
        path = Path(target_row['path'])

        print()
        print('=' * 78)
        print(
            f'Target {object_number}/{args.n_targets}: '
            f'{object_id}'
        )

        pool_rng = np.random.default_rng(
            args.seed + object_number * 10000
        )

        (
            transforms,
            labels,
            pool_a,
            pool_b,
            original_indices,
        ) = load_two_balanced_pools(
            path=path,
            pool_success=POOL_SUCCESS,
            pool_failure=POOL_FAILURE,
            rng=pool_rng,
        )

        assert (
                int(np.sum(labels[pool_a] == 1))
                == POOL_SUCCESS
        )
        assert (
                int(np.sum(labels[pool_a] == 0))
                == POOL_FAILURE
        )
        assert (
                int(np.sum(labels[pool_b] == 1))
                == POOL_SUCCESS
        )
        assert (
                int(np.sum(labels[pool_b] == 0))
                == POOL_FAILURE
        )
        assert (
                len(np.intersect1d(pool_a, pool_b))
                == 0
        )

        D, scale_p, scale_r = (
            distance_matrix_with_reference_scaling(
                transforms=transforms,
                reference_indices=pool_a,
            )
        )

        D_a = D[np.ix_(pool_a, pool_a)]

        target_rows.append(
            {
                'target_number': object_number,
                'object_id': object_id,
                'path': str(path),
                'pool_a_successes': POOL_SUCCESS,
                'pool_a_failures': POOL_FAILURE,
                'pool_b_successes': POOL_SUCCESS,
                'pool_b_failures': POOL_FAILURE,
                'exploration_attempts':
                    EXPLORATION_ATTEMPTS,
                'memory_warmup_attempts':
                    MEMORY_WARMUP_ATTEMPTS,
                'exploitation_attempts':
                    EXPLOITATION_ATTEMPTS,
                'total_attempt_budget':
                    EXPLORATION_ATTEMPTS
                    + EXPLOITATION_ATTEMPTS,
                'scale_p': scale_p,
                'scale_r': scale_r,
            }
        )

        print(
            f'  Pool A: {POOL_SUCCESS} S + '
            f'{POOL_FAILURE} F'
        )
        print(
            f'  Pool B: {POOL_SUCCESS} S + '
            f'{POOL_FAILURE} F'
        )
        print(
            '  metric scales from Pool A only: '
            f'translation={scale_p:.6g}, '
            f'rotation={scale_r:.6g}'
        )

        for seed_index in range(args.n_seeds):
            base_seed = (
                    args.seed
                    + object_number * 100000
                    + seed_index
            )

            # --------------------------------------------------
            # BASELINES
            # --------------------------------------------------

            random_rng = np.random.default_rng(
                base_seed + 1000
            )
            random_local_order = random_order(
                len(pool_a),
                random_rng,
            )
            random_explore = pool_a[
                random_local_order[
                    :EXPLORATION_ATTEMPTS
                ]
            ]

            fps_rng = np.random.default_rng(
                base_seed + 2000
            )
            fps_local_order = fps_order(
                D=D_a,
                max_points=EXPLORATION_ATTEMPTS,
                rng=fps_rng,
            )
            fps_explore = pool_a[fps_local_order]

            ca_rng = np.random.default_rng(
                base_seed + 3000
            )
            ca_explore = (
                hidden_connectivity_exploration_on_pool(
                    labels=labels,
                    D=D,
                    eps_values=eps_values,
                    candidate_indices=pool_a,
                    n_attempts=EXPLORATION_ATTEMPTS,
                    rng=ca_rng,
                    bootstrap=args.bootstrap,
                )
            )

            n_random_obs_success = int(
                np.sum(
                    labels[random_explore] == 1
                )
            )
            n_fps_obs_success = int(
                np.sum(
                    labels[fps_explore] == 1
                )
            )
            n_ca_obs_success = int(
                np.sum(
                    labels[ca_explore] == 1
                )
            )

            print(
                f'  seed {seed_index + 1}/'
                f'{args.n_seeds}: '
                'Pool-A successes | '
                f'Random={n_random_obs_success}, '
                f'FPS={n_fps_obs_success}, '
                f'CA={n_ca_obs_success}'
            )

            # Exact E3 Phase-2 exploitation.
            # Random/FPS: frozen nearest-success ranking.
            # CA: frozen connectivity-aware ranking using Phase-1 successes
            # and failures. Pool-B outcomes never update any ranking.
            random_exploit = nearest_success_exploitation_order(
                exploration_order=random_explore,
                exploitation_candidates=pool_b,
                labels=labels,
                D=D,
                n_attempts=EXPLOITATION_ATTEMPTS,
                rng=np.random.default_rng(base_seed + 10000),
            )

            fps_exploit = nearest_success_exploitation_order(
                exploration_order=fps_explore,
                exploitation_candidates=pool_b,
                labels=labels,
                D=D,
                n_attempts=EXPLOITATION_ATTEMPTS,
                rng=np.random.default_rng(base_seed + 10000),
            )

            ca_exploit = connectivity_exploitation_order(
                exploration_order=ca_explore,
                exploitation_candidates=pool_b,
                labels=labels,
                D=D,
                eps_values=eps_values,
                n_attempts=EXPLOITATION_ATTEMPTS,
                rng=np.random.default_rng(base_seed + 10000),
            )

            baseline_specs = [
                (
                    'random',
                    random_exploit,
                    n_random_obs_success,
                ),
                (
                    'fps',
                    fps_exploit,
                    n_fps_obs_success,
                ),
                (
                    'ca',
                    ca_exploit,
                    n_ca_obs_success,
                ),
            ]

            for (
                    method,
                    exploit_order,
                    n_observed_success,
            ) in baseline_specs:
                curve = cumulative_success_curve(
                    exploit_order,
                    labels,
                )

                seed_rows.append(
                    {
                        'target': object_id,
                        'seed': seed_index + 1,
                        'method': method,
                        'library_size': 0,
                        'alpha':
                            0.0
                            if method == 'ca'
                            else np.nan,
                        'K': MATCHED_SUBSETS_K,
                        'memory_warmup':
                            MEMORY_WARMUP_ATTEMPTS,
                        'exploration_attempts':
                            EXPLORATION_ATTEMPTS,
                        'observed_successes':
                            n_observed_success,
                        'observed_failures':
                            int(
                                EXPLORATION_ATTEMPTS
                                - n_observed_success
                            ),
                        'exploitation_successes':
                            int(curve[-1]),
                        'exploitation_auc':
                            discovery_auc(curve),
                        'normalized_auc':
                            normalized_discovery_auc(
                                curve,
                                available_successes=
                                POOL_SUCCESS,
                            ),
                    }
                )

                for step, successes in enumerate(
                        curve,
                        start=1,
                ):
                    curve_rows.append(
                        {
                            'target': object_id,
                            'seed': seed_index + 1,
                            'method': method,
                            'library_size': 0,
                            'alpha':
                                0.0
                                if method == 'ca'
                                else np.nan,
                            'memory_warmup':
                                MEMORY_WARMUP_ATTEMPTS,
                            'attempt': step,
                            'cumulative_successes':
                                successes,
                        }
                    )

            # --------------------------------------------------
            # MEMORY-ASSISTED EXPLORATION
            # --------------------------------------------------

            for library_size in LIBRARY_SIZES:
                transfer_seed = (
                        base_seed
                        + 40000
                        + library_size * 100
                )

                for alpha in ALPHAS[1:]:
                    # Same RNG as baseline CA so queries 1..10
                    # are exactly the same warm-up trajectory.
                    transfer_rng = np.random.default_rng(
                        base_seed + 3000
                    )

                    (
                        transfer_explore,
                        reference,
                        retrieval_rows,
                    ) = (
                        memory_assisted_connectivity_exploration_on_pool(
                            labels=labels,
                            D=D,
                            eps_values=eps_values,
                            candidate_indices=pool_a,
                            n_attempts=
                            EXPLORATION_ATTEMPTS,
                            rng=transfer_rng,
                            bootstrap=args.bootstrap,
                            reference_df=reference_df,
                            library_size=library_size,
                            alpha=alpha,
                            retrieval_seed=transfer_seed,
                            memory_warmup=
                            MEMORY_WARMUP_ATTEMPTS,
                        )
                    )

                    # Critical warm-up invariant.
                    assert np.array_equal(
                        transfer_explore[
                            :MEMORY_WARMUP_ATTEMPTS
                        ],
                        ca_explore[
                            :MEMORY_WARMUP_ATTEMPTS
                        ],
                    )

                    for retrieval_row in retrieval_rows:
                        retrieval_output_rows.append(
                            {
                                'target': object_id,
                                'seed':
                                    seed_index + 1,
                                'library_size':
                                    library_size,
                                'alpha': alpha,
                                'memory_warmup':
                                    MEMORY_WARMUP_ATTEMPTS,
                                **retrieval_row,
                            }
                        )

                    n_transfer_obs_success = int(
                        np.sum(
                            labels[
                                transfer_explore
                            ] == 1
                        )
                    )

                    # Memory is OFF during exploitation.
                    # E4 uses EXACTLY the same E3 connectivity-aware Phase-2
                    # exploitation as ordinary CA. Thus the only CA-vs-E4
                    # intervention is memory during Pool-A exploration.
                    transfer_exploit = connectivity_exploitation_order(
                        exploration_order=transfer_explore,
                        exploitation_candidates=pool_b,
                        labels=labels,
                        D=D,
                        eps_values=eps_values,
                        n_attempts=EXPLOITATION_ATTEMPTS,
                        rng=np.random.default_rng(base_seed + 10000),
                    )

                    curve = cumulative_success_curve(
                        transfer_exploit,
                        labels,
                    )

                    method_name = (
                        f'ca_mem_L{library_size}'
                        f'_a{alpha:.2f}'
                    )

                    seed_rows.append(
                        {
                            'target': object_id,
                            'seed': seed_index + 1,
                            'method': method_name,
                            'library_size':
                                library_size,
                            'alpha': alpha,
                            'K': MATCHED_SUBSETS_K,
                            'memory_warmup':
                                MEMORY_WARMUP_ATTEMPTS,
                            'exploration_attempts':
                                EXPLORATION_ATTEMPTS,
                            'observed_successes':
                                n_transfer_obs_success,
                            'observed_failures':
                                int(
                                    EXPLORATION_ATTEMPTS
                                    - n_transfer_obs_success
                                ),
                            'retrieved_reference_rank':
                                reference[
                                    'reference_rank'
                                ]
                                if reference is not None
                                else np.nan,
                            'retrieved_reference_object':
                                reference[
                                    'reference_object'
                                ]
                                if reference is not None
                                else None,
                            'retrieval_distance':
                                reference[
                                    'mean_distance'
                                ]
                                if reference is not None
                                else np.nan,
                            'exploitation_successes':
                                int(curve[-1]),
                            'exploitation_auc':
                                discovery_auc(curve),
                            'normalized_auc':
                                normalized_discovery_auc(
                                    curve,
                                    available_successes=
                                    POOL_SUCCESS,
                                ),
                        }
                    )

                    for step, successes in enumerate(
                            curve,
                            start=1,
                    ):
                        curve_rows.append(
                            {
                                'target': object_id,
                                'seed':
                                    seed_index + 1,
                                'method':
                                    method_name,
                                'library_size':
                                    library_size,
                                'alpha': alpha,
                                'memory_warmup':
                                    MEMORY_WARMUP_ATTEMPTS,
                                'attempt': step,
                                'cumulative_successes':
                                    successes,
                            }
                        )

            save_results(
                results_dir=results_dir,
                curve_rows=curve_rows,
                seed_rows=seed_rows,
                retrieval_rows=
                retrieval_output_rows,
                target_rows=target_rows,
            )

            if (
                    seed_index + 1
                    == args.n_seeds
            ):
                print_target_summary(
                    seed_rows=seed_rows,
                    object_id=object_id,
                    object_number=object_number,
                    n_targets=args.n_targets,
                )

    manifest = {
        'experiment':
            'E4_memory_assisted_exploration',
        'protocol':
            'disjoint_balanced_pool_A_pool_B',
        'total_attempt_budget':
            EXPLORATION_ATTEMPTS
            + EXPLOITATION_ATTEMPTS,
        'exploration_attempts':
            EXPLORATION_ATTEMPTS,
        'memory_warmup_attempts':
            MEMORY_WARMUP_ATTEMPTS,
        'memory_assisted_exploration_attempts':
            EXPLORATION_ATTEMPTS
            - MEMORY_WARMUP_ATTEMPTS,
        'exploitation_attempts':
            EXPLOITATION_ATTEMPTS,
        'memory_during_exploitation': False,
        'baseline_random':
            'random_pool_A_exploration_then_'
            'frozen_E3_nearest_success_pool_B_exploitation',
        'baseline_fps':
            'fps_pool_A_exploration_then_'
            'frozen_E3_nearest_success_pool_B_exploitation',
        'baseline_ca':
            'ordinary_CA_pool_A_exploration_then_'
            'E3_connectivity_aware_pool_B_exploitation',
        'transfer':
            '10_CA_warmup_then_CA_plus_retrieved_'
            'success_connectivity_profile_memory_'
            'during_remaining_exploration_only',
        'transfer_exploitation':
            'E3_connectivity_aware_pool_B_exploitation_memory_off',
        'retrieval':
            'single_reference_retrieval_after_'
            'memory_warmup_then_reference_frozen',
        'memory_score':
            'normalized_negative_L1_distance_to_'
            'retrieved_reference_mean_success_'
            'connectivity_profile',
        'matched_subsets_K':
            MATCHED_SUBSETS_K,
        'library_sizes':
            LIBRARY_SIZES,
        'alphas':
            ALPHAS,
        'n_targets':
            args.n_targets,
        'n_seeds':
            args.n_seeds,
        'pool_a_success':
            POOL_SUCCESS,
        'pool_a_failure':
            POOL_FAILURE,
        'pool_b_success':
            POOL_SUCCESS,
        'pool_b_failure':
            POOL_FAILURE,
        'bootstrap':
            args.bootstrap,
        'seed':
            args.seed,
        'eps_values':
            eps_values.tolist(),
    }

    with open(
            results_dir / 'manifest.json',
            'w',
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print()
    print('=' * 78)
    print('E4 COMPLETE')
    print('=' * 78)
    print(f'Saved to: {results_dir}')


if __name__ == '__main__':
    main()