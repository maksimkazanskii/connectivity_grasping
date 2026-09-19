import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from src.utils.samplers import random_order, fps_order, choose_connectivity_point
import gc

def clear_memory_caches(keep_reference_files=False):
    global _REFERENCE_FILE_CACHE
    global _REFERENCE_DISTANCE_CACHE
    global _REFERENCE_SUBSET_CACHE
    global _LABELED_RETRIEVAL_CACHE
    global _LABELED_DISTANCE_CACHE

    # Small/useful cache: loaded transforms from NPZ files.
    # Keeping this avoids repeated disk I/O.
    if not keep_reference_files:
        _REFERENCE_FILE_CACHE.clear()

    # Potentially large caches.
    _REFERENCE_DISTANCE_CACHE.clear()
    _REFERENCE_SUBSET_CACHE.clear()
    _LABELED_RETRIEVAL_CACHE.clear()
    _LABELED_DISTANCE_CACHE.clear()

    gc.collect()
MATCHED_SUBSETS_K = 10
BEST_MATCH_COUNT = 3
TOP_RETRIEVED_REFERENCES = 1
RETRIEVAL_CANDIDATES = 800
TOP_REFERENCE_ACTIONS = 1
LIBRARY_SIZES = [200]  # fixed memory library for the retrieval-control experiment
ALPHAS = [0.0, 0.01, 0.03, 0.05, 0.10, 0.25, 0.50, 1.00]
MEMORY_WARMUP_ATTEMPTS = 10  # ordinary target CA before 1-of-10 memory retrieval


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
    """
    Exact B/L connectivity signature, accelerated for monotone epsilon grids.

    For each epsilon this is mathematically identical to connected_components
    on adjacency (D <= epsilon).  Instead of rebuilding the graph at every
    epsilon, edges are sorted once and inserted incrementally with union-find.
    """
    D = np.asarray(D)
    eps_values = np.asarray(eps_values, dtype=float)
    n = len(D)
    if n == 0:
        raise ValueError('Cannot construct topology from zero points.')

    # Preserve support for arbitrary epsilon ordering without changing output.
    eps_order = np.argsort(eps_values, kind='stable')
    eps_sorted = eps_values[eps_order]

    # Self-edges do not affect connectivity. For an undirected symmetric
    # distance matrix, one copy of each i<j edge is sufficient.
    ii, jj = np.triu_indices(n, k=1)
    weights = D[ii, jj]
    edge_order = np.argsort(weights, kind='stable')
    weights = weights[edge_order]
    ii = ii[edge_order]
    jj = jj[edge_order]

    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)
    n_components = n
    largest = 1
    edge_pos = 0

    B_sorted = np.empty(len(eps_sorted), dtype=float)
    L_sorted = np.empty(len(eps_sorted), dtype=float)

    def find_root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for epos, eps in enumerate(eps_sorted):
        while edge_pos < len(weights) and weights[edge_pos] <= eps:
            ra = find_root(int(ii[edge_pos]))
            rb = find_root(int(jj[edge_pos]))
            if ra != rb:
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]
                n_components -= 1
                if size[ra] > largest:
                    largest = int(size[ra])
            edge_pos += 1

        B_sorted[epos] = n_components / float(n)
        L_sorted[epos] = largest / float(n)

    B = np.empty_like(B_sorted)
    L = np.empty_like(L_sorted)
    B[eps_order] = B_sorted
    L[eps_order] = L_sorted
    return B, L


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


def _transform_keys(transforms, decimals=10):
    """Stable keys for identifying the same grasp transform across H5/NPZ files."""
    arr = np.asarray(transforms, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f'Expected a batch of transforms, got shape {arr.shape}.')
    flat = np.round(arr.reshape(len(arr), -1), decimals=decimals)
    return [row.tobytes() for row in np.ascontiguousarray(flat)]


def load_two_balanced_pools(
        path, pool_success, pool_failure, rng, excluded_transforms=None
):
    """Sample disjoint A/B pools, optionally excluding every grasp present in R."""
    with h5py.File(path, 'r') as f:
        labels_all = f['grasps/qualities/flex/object_in_gripper'][:].astype(int)
        transforms_all = f['grasps/transforms'][:]

    allowed = np.ones(len(labels_all), dtype=bool)
    if excluded_transforms is not None and len(excluded_transforms) > 0:
        excluded_keys = set(_transform_keys(excluded_transforms))
        all_keys = _transform_keys(transforms_all)
        allowed = np.fromiter(
            (key not in excluded_keys for key in all_keys),
            dtype=bool, count=len(all_keys),
        )

    success_idx = np.flatnonzero((labels_all == 1) & allowed)
    failure_idx = np.flatnonzero((labels_all == 0) & allowed)

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
        return_scores=False,
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
        if return_scores:
            return np.zeros(len(exploitation_candidates), dtype=float)
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

    if return_scores:
        return candidate_scores

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


_REFERENCE_FILE_CACHE = {}

def _load_reference_memory_cached(memory_path):
    key = str(Path(memory_path))
    cached = _REFERENCE_FILE_CACHE.get(key)
    if cached is None:
        path = Path(memory_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as memory:
            cached = {
                'success_transforms': np.asarray(memory['success_transforms']),
                'failure_transforms': np.asarray(memory['failure_transforms']),
                'scale_p': float(memory['scale_p']),
                'scale_r': float(memory['scale_r']),
            }
        _REFERENCE_FILE_CACHE[key] = cached
    return cached


def load_reference_successes(row):
    cached = _load_reference_memory_cached(row['memory_path'])
    return (
        cached['success_transforms'],
        cached['scale_p'],
        cached['scale_r'],
    )


def load_reference_failures_from_path(memory_path):
    return _load_reference_memory_cached(memory_path)['failure_transforms']


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


def sample_reference_signature_records(
        success_transforms,
        subset_size,
        eps_values,
        scale_p,
        scale_r,
        n_subsets,
        rng,
):
    """Sample matched-cardinality subsets while preserving subset identity."""
    n_success = len(success_transforms)
    if subset_size > n_success:
        return None

    records = []
    for _ in range(n_subsets):
        subset_idx = rng.choice(n_success, size=subset_size, replace=False)
        subset = success_transforms[subset_idx]
        D_subset = distance_matrix_fixed_scales(
            subset, scale_p=scale_p, scale_r=scale_r
        )
        records.append({
            'indices': np.asarray(subset_idx, dtype=int),
            'signature': connectivity_signature_from_D(D_subset, eps_values),
        })
    return records


def cross_distance_fixed_scales(transforms_a, transforms_b, scale_p, scale_r):
    """Normalized SE(3) distances between two pose sets using fixed scales."""
    dp, dr = raw_se3_distances(transforms_a, transforms_b)
    return np.sqrt(
        (dp / float(scale_p)) ** 2 + (dr / float(scale_r)) ** 2
    ).astype(np.float32)


def retrieve_reference(
        target_signature,
        subset_size,
        reference_df,
        library_size,
        eps_values,
        seed,
):
    """
    Retrieve by the mean of the three closest matched-cardinality topology realizations.

    Crucially, preserve the identity of the winning reference subset.  That
    subset is the part of the reference that best explains the warm-up target
    successes; the remaining reference successes are left intact and become
    the discovery memory used after retrieval.
    """
    library = reference_df[
        reference_df['reference_rank'] <= library_size
        ].copy().reset_index(drop=True)

    if len(library) != library_size:
        raise ValueError(
            f'Requested L{library_size}, but found {len(library)} references.'
        )

    best = None
    retrieval_rows = []

    for _, row in library.iterrows():
        reference_rank = int(row['reference_rank'])
        successes, scale_p, scale_r = load_reference_successes(row)

        if len(successes) < subset_size:
            continue

        reference_rng = np.random.default_rng(
            seed + reference_rank * 100003 + subset_size * 1009
        )

        records = sample_reference_signature_records(
            success_transforms=successes,
            subset_size=subset_size,
            eps_values=eps_values,
            scale_p=scale_p,
            scale_r=scale_r,
            n_subsets=MATCHED_SUBSETS_K,
            rng=reference_rng,
        )
        if records is None:
            continue

        distances = np.asarray([
            signature_distance(target_signature, rec['signature'])
            for rec in records
        ], dtype=float)
        order = np.argsort(distances)
        best_locals = order[:min(BEST_MATCH_COUNT, len(order))]
        winner_local = int(best_locals[0])
        best_match_distance = float(distances[winner_local])
        best_few_distance = float(np.mean(distances[best_locals]))

        retrieval_rows.append({
            'reference_rank': reference_rank,
            'reference_object': row['object_id'],
            'matched_size': subset_size,
            'K': MATCHED_SUBSETS_K,
            'best_match_count': int(len(best_locals)),
            'best_few_distance': best_few_distance,
            'best_match_distance': best_match_distance,
            'mean_distance': float(distances.mean()),
            'distance_std': float(distances.std()),
            'winning_subset_draw': winner_local,
        })

        if best is None or best_few_distance < best['best_few_distance']:
            # The reference is selected by the mean distance of its three best
            # matched subsets.  Its single closest subset is retained only as
            # the geometric anchor for discovery memory.
            winning = records[winner_local]
            best = {
                'reference_rank': reference_rank,
                'reference_object': row['object_id'],
                'memory_path': row['memory_path'],
                'best_few_distance': best_few_distance,
                'best_match_distance': best_match_distance,
                'mean_distance': float(distances.mean()),
                'winning_subset_draw': winner_local,
                'winning_subset_indices': winning['indices'].copy(),
                'winning_subset_signature': winning['signature'],
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

    all_idx = np.arange(len(best['success_transforms']), dtype=int)
    keep = np.ones(len(all_idx), dtype=bool)
    keep[best['winning_subset_indices']] = False
    best['unmatched_success_indices'] = all_idx[keep]

    return best, retrieval_rows



_REFERENCE_DISTANCE_CACHE = {}
_REFERENCE_SUBSET_CACHE = {}

def _reference_cache_key(reference):
    return (
        str(reference.get('reference_object', reference.get('object', ''))),
        int(len(reference['success_transforms'])),
        float(reference['scale_p']),
        float(reference['scale_r']),
    )

def _get_reference_distance_matrix(reference):
    key = _reference_cache_key(reference)
    D_ref = _REFERENCE_DISTANCE_CACHE.get(key)
    if D_ref is None:
        successes = np.asarray(reference['success_transforms'])
        D_ref = distance_matrix_fixed_scales(
            successes, scale_p=reference['scale_p'], scale_r=reference['scale_r']
        ).astype(np.float32, copy=False)
        _REFERENCE_DISTANCE_CACHE[key] = D_ref
    return D_ref

def prepare_online_action_transfer(reference):
    """
    Prepare one retrieved reference for top-q online action transfer.

    The winning matched-cardinality subset U* is treated as the analogue of
    the target successes observed at the current time.  Rather than retaining
    only CA's single highest-scoring unmatched successful reference grasp, we
    retain the TOP_REFERENCE_ACTIONS highest-scoring reference grasps.

    Each action is represented by its permutation-invariant sorted vector of
    normalized SE(3) distances to U*.  No target/reference point
    correspondence is assumed.
    """
    successes = np.asarray(reference['success_transforms'])
    matched_idx = np.asarray(reference['winning_subset_indices'], dtype=int)

    if len(matched_idx) == 0:
        reference['reference_action_indices'] = np.empty(0, dtype=int)
        reference['reference_action_profiles'] = np.empty((0, 0), dtype=np.float32)
        reference['reference_action_scores'] = np.empty(0, dtype=float)
        return reference

    all_idx = np.arange(len(successes), dtype=int)
    unmatched_idx = all_idx[~np.isin(all_idx, matched_idx)]

    if len(unmatched_idx) == 0:
        reference['reference_action_indices'] = np.empty(0, dtype=int)
        reference['reference_action_profiles'] = np.empty((0, len(matched_idx)), dtype=np.float32)
        reference['reference_action_scores'] = np.empty(0, dtype=float)
        return reference

    D_ref = _get_reference_distance_matrix(reference)
    action_scores = connectivity_acquisition_scores(
        selected=matched_idx,
        candidate_indices=unmatched_idx,
        D=D_ref,
        eps_values=reference['eps_values'],
    )

    q = min(TOP_REFERENCE_ACTIONS, len(unmatched_idx))
    # Stable descending ordering keeps the former argmax choice first when q=1.
    order = np.argsort(-np.asarray(action_scores), kind='stable')[:q]
    action_indices = unmatched_idx[order].astype(int)
    profiles = np.sort(
        D_ref[np.ix_(action_indices, matched_idx)], axis=1
    ).astype(np.float32)

    reference['reference_action_indices'] = action_indices
    reference['reference_action_profiles'] = profiles
    reference['reference_action_scores'] = np.asarray(action_scores, dtype=float)[order]
    reference['matched_success_transforms'] = successes[matched_idx]
    return reference


def online_action_transfer_scores(
        current_successful_indices,
        candidate_indices,
        D_target,
        references,
):
    """
    Transfer the top-q actions from the top-q retrieved references.

    For every retained reference action r, compare its sorted distance profile
    to the current target-success anchors with every unqueried target
    candidate x.  The memory score of x is its best match to ANY retained
    action across the retrieved references:

        S_M(x) = max_r [-mean_i |v_T(x)_i - v_R(r)_i|].

    Thus TOP_RETRIEVED_REFERENCES=5 and TOP_REFERENCE_ACTIONS=5 expose up to
    25 plausible next actions without adding another weighting hyperparameter.
    """
    anchors = np.asarray(current_successful_indices, dtype=int)
    candidates = np.asarray(candidate_indices, dtype=int)

    if len(candidates) == 0:
        return np.empty(0, dtype=float)
    if references is None or len(anchors) == 0:
        return np.zeros(len(candidates), dtype=float)
    if isinstance(references, dict):
        references = [references]
    if len(references) == 0:
        return np.zeros(len(candidates), dtype=float)

    target_profiles = np.sort(
        D_target[np.ix_(candidates, anchors)], axis=1
    ).astype(float)

    best_scores = np.full(len(candidates), -np.inf, dtype=float)
    found_action = False
    for reference in references:
        action_profiles = reference.get('reference_action_profiles')
        if action_profiles is None or len(action_profiles) == 0:
            continue
        action_profiles = np.asarray(action_profiles, dtype=float)
        if action_profiles.ndim == 1:
            action_profiles = action_profiles[None, :]
        if action_profiles.shape[1] != len(anchors):
            raise ValueError(
                'Reference action profile cardinality does not match the current '
                'number of successful target observations.'
            )

        # candidates x actions x anchors -> candidates x actions
        scores = -np.mean(
            np.abs(target_profiles[:, None, :] - action_profiles[None, :, :]),
            axis=2,
        )
        best_scores = np.maximum(best_scores, np.max(scores, axis=1))
        found_action = True

    if not found_action:
        return np.zeros(len(candidates), dtype=float)
    return best_scores


def retrieve_online_reference(
        successful, D_target, eps_values, reference_df, library_size,
        retrieval_seed, query_number, target_object_id):
    """
    Online retrieval with the original per-reference topology rule, retaining
    the TOP_RETRIEVED_REFERENCES best reference objects.

    Everything inside each reference is unchanged: MATCHED_SUBSETS_K subsets
    are sampled, a reference is scored by its BEST_MATCH_COUNT closest subset
    signatures, and its single closest subset is the geometric anchor.
    """
    successful_arr = np.asarray(successful, dtype=int)
    k = len(successful_arr)
    if k == 0:
        return [], []

    target_signature = connectivity_signature_from_D(
        D_target[np.ix_(successful_arr, successful_arr)], eps_values,
    )
    # 1-of-N retrieval sanity check: reference_df is already the candidate set
    # containing the true same-object memory plus distractors.  Object identity
    # is NOT used here.
    library = reference_df.copy().reset_index(drop=True)
    if len(library) == 0:
        raise RuntimeError('Empty retrieval candidate set.')

    eps_key = tuple(np.asarray(eps_values, dtype=float).round(12).tolist())
    retrieval_rows = []
    candidates_refs = []

    for _, row in library.iterrows():
        reference_rank = int(row['reference_rank'])
        successes, scale_p, scale_r = load_reference_successes(row)
        n_success = len(successes)
        if n_success < k:
            continue

        reference_stub = {
            'reference_rank': reference_rank,
            'reference_object': row['object_id'],
            'memory_path': row['memory_path'],
            'success_transforms': successes,
            'scale_p': scale_p,
            'scale_r': scale_r,
        }
        subset_seed = int(retrieval_seed) + reference_rank * 100003 + k * 1009
        cache_key = (
            _reference_cache_key(reference_stub), int(k), eps_key,
            int(subset_seed), int(MATCHED_SUBSETS_K),
        )
        cached = _REFERENCE_SUBSET_CACHE.get(cache_key)
        if cached is None:
            D_ref = _get_reference_distance_matrix(reference_stub)
            reference_rng = np.random.default_rng(subset_seed)
            subset_indices_list, subset_signatures = [], []
            for _draw in range(MATCHED_SUBSETS_K):
                subset_idx = reference_rng.choice(
                    n_success, size=k, replace=False
                ).astype(int)
                signature = connectivity_signature_from_D(
                    D_ref[np.ix_(subset_idx, subset_idx)], eps_values,
                )
                subset_indices_list.append(subset_idx)
                subset_signatures.append(signature)
            cached = (subset_indices_list, subset_signatures)
            _REFERENCE_SUBSET_CACHE[cache_key] = cached

        subset_indices_list, subset_signatures = cached
        target_B, target_L = target_signature
        sig_B = np.asarray([sig[0] for sig in subset_signatures], dtype=float)
        sig_L = np.asarray([sig[1] for sig in subset_signatures], dtype=float)
        distances = 0.5 * (
                np.mean(np.abs(sig_B - target_B[None, :]), axis=1)
                + np.mean(np.abs(sig_L - target_L[None, :]), axis=1)
        )
        order = np.argsort(distances)
        best_locals = order[:min(BEST_MATCH_COUNT, len(order))]
        winner_local = int(best_locals[0])
        best_match_distance = float(distances[winner_local])
        best_few_distance = float(np.mean(distances[best_locals]))

        retrieval_rows.append({
            'reference_rank': reference_rank,
            'reference_object': row['object_id'],
            'matched_size': int(k),
            'K': MATCHED_SUBSETS_K,
            'best_match_count': int(len(best_locals)),
            'best_few_distance': best_few_distance,
            'best_match_distance': best_match_distance,
            'mean_distance': float(distances.mean()),
            'distance_std': float(distances.std()),
            'winning_subset_draw': winner_local,
            'query_number': int(query_number),
            'n_target_successes': int(k),
        })
        candidates_refs.append({
            'reference_rank': reference_rank,
            'reference_object': row['object_id'],
            'memory_path': row['memory_path'],
            'best_few_distance': best_few_distance,
            'best_match_distance': best_match_distance,
            'mean_distance': float(distances.mean()),
            'winning_subset_draw': winner_local,
            'winning_subset_indices':
                np.asarray(subset_indices_list[winner_local], dtype=int).copy(),
            'winning_subset_signature': subset_signatures[winner_local],
            'success_transforms': successes,
            'scale_p': scale_p,
            'scale_r': scale_r,
        })

    if not candidates_refs:
        raise RuntimeError(
            'No reference object has enough observed successes '
            f'for matched size k={k}.'
        )

    # Keep the five references with the smallest original best-few score.
    candidates_refs.sort(key=lambda r: (r['best_few_distance'], r['reference_rank']))
    selected_refs = candidates_refs[:min(TOP_RETRIEVED_REFERENCES, len(candidates_refs))]

    for ref in selected_refs:
        ref['failure_transforms'] = load_reference_failures_from_path(ref['memory_path'])
        all_idx = np.arange(len(ref['success_transforms']), dtype=int)
        keep = np.ones(len(all_idx), dtype=bool)
        keep[ref['winning_subset_indices']] = False
        ref['unmatched_success_indices'] = all_idx[keep]
        ref['eps_values'] = np.asarray(eps_values, dtype=float)
        prepare_online_action_transfer(ref)

    return selected_refs, retrieval_rows




_LABELED_RETRIEVAL_CACHE = {}
_LABELED_DISTANCE_CACHE = {}

def _reference_label_signature_samples(row, label_name, subset_size, eps_values, retrieval_seed):
    """Deterministic matched-cardinality signature samples for one S/F class."""
    if subset_size <= 0:
        return []
    cached_mem = _load_reference_memory_cached(row['memory_path'])
    transforms = cached_mem['success_transforms'] if label_name == 'success' else cached_mem['failure_transforms']
    if len(transforms) < subset_size:
        return None
    rank = int(row['reference_rank'])
    eps_key = tuple(np.asarray(eps_values, dtype=float).round(12).tolist())
    key = (str(row['memory_path']), label_name, int(subset_size), eps_key, int(retrieval_seed), int(MATCHED_SUBSETS_K))
    if key in _LABELED_RETRIEVAL_CACHE:
        return _LABELED_RETRIEVAL_CACHE[key]
    dkey = (str(row['memory_path']), label_name)
    D_ref = _LABELED_DISTANCE_CACHE.get(dkey)
    if D_ref is None:
        D_ref = distance_matrix_fixed_scales(
            transforms, cached_mem['scale_p'], cached_mem['scale_r']
        ).astype(np.float32, copy=False)
        _LABELED_DISTANCE_CACHE[dkey] = D_ref
    rng = np.random.default_rng(int(retrieval_seed) + rank * 100003 + subset_size * 1009 + (0 if label_name == 'success' else 7919))
    records = []
    for _ in range(MATCHED_SUBSETS_K):
        idx = rng.choice(len(transforms), size=subset_size, replace=False).astype(int)
        sig = connectivity_signature_from_D(D_ref[np.ix_(idx, idx)], eps_values)
        records.append((idx, sig))
    _LABELED_RETRIEVAL_CACHE[key] = records
    return records

def prepare_margin_retrieval_geometry(target_transforms, candidate_df, scale_p, scale_r):
    """Precompute nearest-success / nearest-failure distances for each memory.

    Distances are measured in the target Pool-A metric.  For each target grasp x
    and candidate memory R_j, store

        d_plus(x)  = min_{r in R_j^+} D(x, r)
        d_minus(x) = min_{r in R_j^-} D(x, r)

    so online retrieval after each new observation is cheap and deterministic.
    """
    target_transforms = np.asarray(target_transforms)
    prepared = {}
    for _, row in candidate_df.iterrows():
        cached = _load_reference_memory_cached(row['memory_path'])
        success_transforms = np.asarray(cached['success_transforms'])
        failure_transforms = np.asarray(cached['failure_transforms'])
        if len(success_transforms) == 0 or len(failure_transforms) == 0:
            continue

        D_plus = cross_distance_fixed_scales(
            target_transforms, success_transforms,
            scale_p=scale_p, scale_r=scale_r,
        )
        D_minus = cross_distance_fixed_scales(
            target_transforms, failure_transforms,
            scale_p=scale_p, scale_r=scale_r,
        )
        prepared[str(row['object_id'])] = {
            'd_plus': np.min(D_plus, axis=1).astype(np.float32, copy=False),
            'd_minus': np.min(D_minus, axis=1).astype(np.float32, copy=False),
        }
    return prepared


def retrieve_labeled_memory_online(
        observed_indices, labels, candidate_df, prepared_geometry, query_number):
    """Rank memories by how well their local S/F geometry predicts observations.

    For each observed target grasp x_i under candidate memory j:

        m_ij = d_minus_ij - d_plus_ij

    A success wants m_ij > 0; a failure wants m_ij < 0.  We therefore use the
    signed margin

        z_ij = (2 y_i - 1) * m_ij,

    and select the memory with the largest class-balanced mean signed margin.
    The returned ``score`` is its negative so, as elsewhere in this script,
    smaller retrieval score is better.
    """
    observed = np.asarray(observed_indices, dtype=int)
    y = np.asarray(labels[observed], dtype=int)
    s_mask = (y == 1)
    f_mask = (y == 0)

    ranked = []
    diagnostics = []
    for _, row in candidate_df.iterrows():
        object_id = str(row['object_id'])
        geom = prepared_geometry.get(object_id)
        if geom is None:
            continue

        d_plus = geom['d_plus'][observed].astype(float, copy=False)
        d_minus = geom['d_minus'][observed].astype(float, copy=False)
        margin = d_minus - d_plus
        signed_margin = np.where(s_mask, margin, -margin)

        success_margin = (
            float(np.mean(signed_margin[s_mask])) if np.any(s_mask) else np.nan
        )
        failure_margin = (
            float(np.mean(signed_margin[f_mask])) if np.any(f_mask) else np.nan
        )
        class_parts = [v for v in (success_margin, failure_margin) if np.isfinite(v)]
        if not class_parts:
            continue
        balanced_margin = float(np.mean(class_parts))

        prediction = (margin > 0).astype(int)
        accuracy = float(np.mean(prediction == y))
        success_accuracy = (
            float(np.mean(prediction[s_mask] == 1)) if np.any(s_mask) else np.nan
        )
        failure_accuracy = (
            float(np.mean(prediction[f_mask] == 0)) if np.any(f_mask) else np.nan
        )
        acc_parts = [v for v in (success_accuracy, failure_accuracy) if np.isfinite(v)]
        balanced_accuracy = float(np.mean(acc_parts))

        score = -balanced_margin
        item = {
            'reference_rank': int(row['reference_rank']),
            'reference_object': row['object_id'],
            'memory_path': row['memory_path'],
            'score': score,
            'balanced_signed_margin': balanced_margin,
            'mean_signed_margin': float(np.mean(signed_margin)),
            'success_signed_margin': success_margin,
            'failure_signed_margin': failure_margin,
            'classification_accuracy': accuracy,
            'balanced_classification_accuracy': balanced_accuracy,
        }
        ranked.append(item)
        diagnostics.append({
            **item,
            'query_number': int(query_number),
            'n_target_successes': int(np.sum(s_mask)),
            'n_target_failures': int(np.sum(f_mask)),
        })

    if not ranked:
        raise RuntimeError(
            f'No candidate memory has usable success/failure geometry at query {query_number}.'
        )
    ranked.sort(key=lambda z: (z['score'], z['reference_rank']))
    return ranked[0], diagnostics


def known_memory_connectivity_exploration_on_pool(
        labels, D_aug, eps_values, candidate_indices, n_attempts,
        memory_success_indices, rng, initial_success_indices=None,
):
    """Explore new Pool-A points with CA initialized by SAME-object success memory.

    Historical successful memory points are fixed CA anchors from query 1.
    Every newly discovered Pool-A success is added to the anchor set.
    Failed Pool-A queries are observed but do not enter the CA exploration score,
    matching the original E3 exploration rule.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    memory_success_indices = np.asarray(memory_success_indices, dtype=int)
    available = np.zeros(len(labels), dtype=bool)
    available[candidate_indices] = True
    queried = []
    successful_target = ([] if initial_success_indices is None else
                         list(np.asarray(initial_success_indices, dtype=int)))

    for _ in range(n_attempts):
        candidates = np.flatnonzero(available)
        selected = np.concatenate([
            memory_success_indices,
            np.asarray(successful_target, dtype=int),
        ])
        if len(selected) == 0:
            chosen = int(rng.choice(candidates))
        else:
            scores = connectivity_acquisition_scores(
                selected=selected, candidate_indices=candidates,
                D=D_aug, eps_values=eps_values,
            )
            chosen = int(candidates[int(np.argmax(scores))])
        queried.append(chosen)
        available[chosen] = False
        if labels[chosen] == 1:
            successful_target.append(chosen)

    return np.asarray(queried, dtype=int)

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


def success_checkpoints(curve):
    """Return cumulative exploitation successes at attempts 25, 50, and 100."""
    curve = np.asarray(curve)
    if len(curve) < 100:
        raise ValueError(f'Expected at least 100 exploitation attempts, got {len(curve)}.')
    return {
        'success_at_25': int(curve[24]),
        'success_at_50': int(curve[49]),
        'success_at_100': int(curve[99]),
    }


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

        method_df = df.loc[df['method'] == method]
        s25 = method_df['success_at_25'].to_numpy(dtype=float)
        s50 = method_df['success_at_50'].to_numpy(dtype=float)
        s100 = method_df['success_at_100'].to_numpy(dtype=float)

        print(
            f'  {method.upper():<10} '
            f'nAUC={np.mean(values):.4f} ± {np.std(values):.4f} | '
            f'S@25={np.mean(s25):.1f} ± {np.std(s25):.1f} | '
            f'S@50={np.mean(s50):.1f} ± {np.std(s50):.1f} | '
            f'S@100={np.mean(s100):.1f} ± {np.std(s100):.1f}'
        )

    print()
    print('Transfer:')

    best = None

    for library_size in LIBRARY_SIZES:
        print(f'  candidate memories={library_size}')

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

            config_df = df.loc[mask].sort_values('seed')
            s25 = config_df['success_at_25'].to_numpy(dtype=float)
            s50 = config_df['success_at_50'].to_numpy(dtype=float)
            s100 = config_df['success_at_100'].to_numpy(dtype=float)

            ca_by_seed = (
                df.loc[df['method'] == 'ca', ['seed', 'normalized_auc']]
                .drop_duplicates('seed')
                .set_index('seed')['normalized_auc']
            )
            paired = config_df[['seed', 'normalized_auc']].copy()
            paired['ca_auc'] = paired['seed'].map(ca_by_seed)
            paired = paired.dropna(subset=['ca_auc'])
            deltas = (paired['normalized_auc'] - paired['ca_auc']).to_numpy(dtype=float)
            delta_text = ''
            if len(deltas):
                delta_text = (
                    f' | paired Δ={np.mean(deltas):+.4f} ± {np.std(deltas):.4f}'
                )

            print(
                f'    alpha={alpha:<4.2f}  '
                f'nAUC={mean_value:.4f} ± {std_value:.4f} | '
                f'S@25={np.mean(s25):.1f} ± {np.std(s25):.1f} | '
                f'S@50={np.mean(s50):.1f} ± {np.std(s50):.1f} | '
                f'S@100={np.mean(s100):.1f} ± {np.std(s100):.1f}'
                f'{delta_text}'
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


def run_group(args, target_group):
    args.target_group = target_group


    # Match E3 N=1600 exactly on the target-side pool composition and budgets.
    # Pool A: 400 success + 400 failure; Pool B: 400 success + 400 failure.
    # Phase 1: 400 exploration attempts; Phase 2: 400 exploitation attempts.
    POOL_SUCCESS = 400
    POOL_FAILURE = 400


    EXPLORATION_ATTEMPTS = args.attempts
    EXPLOITATION_ATTEMPTS = args.attempts


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
            Path('results')
            / 'e4_structural_memory_retrieval_control'
            / f'attempts_{args.attempts}'
            / args.target_group
    )

    if args.seed_index is not None:
        results_dir = (
                results_dir
                / f'seed_{args.seed_index + 1}'
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eps_values = np.linspace(0.1, 0.8, 15)

    print('=' * 78)
    print('E4 RETRIEVAL CONTROL: CA vs RANDOM MEMORY vs RETRIEVED MEMORY')
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
    print('Memory during exploitation: YES — historical S/F memory + observed Pool-A outcomes')
    print(
        f'Total attempt budget: '
        f'{EXPLORATION_ATTEMPTS + EXPLOITATION_ATTEMPTS}'
    )
    print(
        f'Matched topology subsets K (retrieval only): '
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

    heldout_target_df = pd.read_csv(target_csv)
    required_target_columns = {'object_id', 'path'}
    missing = required_target_columns - set(heldout_target_df.columns)
    if missing:
        raise ValueError(
            'Target CSV missing columns: '
            f'{sorted(missing)}'
        )

    master_rng = np.random.default_rng(args.seed)

    # --------------------------------------------------
    # FIX THE MEMORY BANKS ONCE, BEFORE ANY TARGET IS RUN.
    # The banks are nested: M200 ⊂ M400 ⊂ M600 ⊂ M800.
    # They never depend on the target object or seed.
    # --------------------------------------------------
    memory_order = reference_df.sort_values('reference_rank').head(max(LIBRARY_SIZES)).copy().reset_index(drop=True)
    if len(memory_order) != max(LIBRARY_SIZES):
        raise RuntimeError(f'Need exactly {max(LIBRARY_SIZES)} reference memories, found {len(memory_order)}.')
    memory_banks = {}
    for M in LIBRARY_SIZES:
        bank = memory_order.iloc[:M].copy().reset_index(drop=True)
        memory_banks[M] = bank
        bank.to_csv(results_dir / f'memory_bank_M{M}.csv', index=False)
    print('Fixed M=200 memory bank stored before target experiments.')

    # Targets are drawn ONLY from the held-out target list. They are not members
    # of any memory bank. This gives object-level independence between memory R
    # and target objects Q.
    memory_ids = set(memory_order['object_id'].astype(str))
    target_df = heldout_target_df[
        ~heldout_target_df['object_id'].astype(str).isin(memory_ids)
    ].copy().reset_index(drop=True)
    if len(target_df) < args.n_targets:
        raise RuntimeError(
            f'Only {len(target_df)} target objects remain after excluding all memory objects; '
            f'requested {args.n_targets}.'
        )
    selected_positions = master_rng.choice(
        len(target_df), size=args.n_targets, replace=False
    )
    selected_targets = target_df.iloc[selected_positions].copy().reset_index(drop=True)
    selected_targets.to_csv(results_dir / 'fixed_target_objects.csv', index=False)

    # Hard object-level independence check for every memory size.
    target_ids = set(selected_targets['object_id'].astype(str))
    for M, bank in memory_banks.items():
        overlap = target_ids.intersection(set(bank['object_id'].astype(str)))
        if overlap:
            raise RuntimeError(f'Object leakage between targets and M={M} memory bank: {sorted(overlap)[:5]}')

    print(f'Target group: {args.target_group}')
    print(f'Results directory: {results_dir}')
    print('Retrieval metric: class-balanced signed nearest-S/F SE(3) margin')
    print('Target objects are independent of ALL memory banks: YES')

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

        # Target objects are absent from every memory bank by construction.
        # Pool A and Pool B are sampled independently from the target H5 and are
        # mutually disjoint. No target historical memory is loaded or used.
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
            excluded_transforms=None,
        )

        assert int(np.sum(labels[pool_a] == 1)) == POOL_SUCCESS
        assert int(np.sum(labels[pool_a] == 0)) == POOL_FAILURE
        assert int(np.sum(labels[pool_b] == 1)) == POOL_SUCCESS
        assert int(np.sum(labels[pool_b] == 0)) == POOL_FAILURE
        assert len(np.intersect1d(pool_a, pool_b)) == 0
        a_keys = set(_transform_keys(transforms[pool_a]))
        b_keys = set(_transform_keys(transforms[pool_b]))
        assert a_keys.isdisjoint(b_keys), 'Protocol violation: A and B overlap.'
        print('  independence verified: target not in memory banks; A∩B=0')

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
                'target_group': args.target_group,
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

        if args.seed_index is None:
            seed_indices = range(args.n_seeds)
        else:
            if not 0 <= args.seed_index < args.n_seeds:
                raise ValueError(
                    f'--seed_index must be between 0 and '
                    f'{args.n_seeds - 1}.'
                )
            seed_indices = [args.seed_index]

        for seed_index in seed_indices:
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
                        **success_checkpoints(curve),
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
            # ONLINE RETRIEVAL AGAINST FIXED, TARGET-INDEPENDENT MEMORY BANKS
            # --------------------------------------------------
            # Precompute geometry for the largest fixed bank once. Smaller banks
            # are nested prefixes and reuse the same prepared distances.
            full_memory_df = memory_banks[max(LIBRARY_SIZES)].copy().reset_index(drop=True)
            if str(object_id) in set(full_memory_df['object_id'].astype(str)):
                raise RuntimeError('Target object leaked into the fixed memory bank.')
            prepared_margin_geometry = prepare_margin_retrieval_geometry(
                target_transforms=transforms,
                candidate_df=full_memory_df,
                scale_p=scale_p, scale_r=scale_r,
            )

            # Pool-A CA trajectory is independent of memory size and alpha, so compute it once
            # per target/seed and reuse it for every memory bank.
            transfer_rng = np.random.default_rng(base_seed + 3000)
            transfer_explore = hidden_connectivity_exploration_on_pool(
                labels=labels, D=D, eps_values=eps_values,
                candidate_indices=pool_a,
                n_attempts=EXPLORATION_ATTEMPTS,
                rng=transfer_rng, bootstrap=args.bootstrap,
            )
            n_transfer_obs_success = int(np.sum(labels[transfer_explore] == 1))

            # Target-only CA Pool-B scores are also independent of M and alpha.
            ca_scores = connectivity_exploitation_order(
                exploration_order=transfer_explore,
                exploitation_candidates=pool_b, labels=labels, D=D,
                eps_values=eps_values, n_attempts=EXPLOITATION_ATTEMPTS,
                rng=transfer_rng, return_scores=True,
            )

            for library_size in LIBRARY_SIZES:
                retrieval_candidates_df = memory_banks[library_size].copy().reset_index(drop=True)

                # Retrieve ONCE for this (target, seed, M). Retrieval does not depend on alpha.
                reference = None
                for qnum in range(MEMORY_WARMUP_ATTEMPTS, EXPLORATION_ATTEMPTS + 1):
                    observed_prefix = np.asarray(transfer_explore[:qnum], dtype=int)
                    reference, retrieval_rows = retrieve_labeled_memory_online(
                        observed_indices=observed_prefix, labels=labels,
                        candidate_df=retrieval_candidates_df,
                        prepared_geometry=prepared_margin_geometry,
                        query_number=qnum,
                    )
                    selected_rr = next(
                        rr for rr in retrieval_rows
                        if str(rr['reference_object']) ==
                        str(reference['reference_object'])
                    )

                    selected_rr = selected_rr.copy()
                    selected_rr.update({
                        'target': object_id,
                        'seed': seed_index + 1,
                        'candidate_set_size': len(retrieval_candidates_df),
                        'selected': True,
                    })

                    retrieval_output_rows.append(selected_rr)

                # The q=100 retrieval is the FINAL retrieval used for Pool B.
                selected_row = retrieval_candidates_df[
                    retrieval_candidates_df['object_id'].astype(str) ==
                    str(reference['reference_object'])
                    ].iloc[0]
                sel_successes, _, _ = load_reference_successes(selected_row)
                sel_failures = load_reference_failures_from_path(selected_row['memory_path'])
                memory_transforms = np.concatenate([sel_successes, sel_failures], axis=0)
                all_aug = np.concatenate([transforms, memory_transforms], axis=0)
                D_aug = distance_matrix_fixed_scales(
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
                    labels, np.ones(len(sel_successes), dtype=int),
                    np.zeros(len(sel_failures), dtype=int),
                ])
                observed_structure = np.concatenate([
                    mem_success_idx, mem_failure_idx, transfer_explore
                ]).astype(int)

                # Compute the memory-augmented Pool-B score ONCE for this M.
                memory_scores = connectivity_exploitation_order(
                    exploration_order=observed_structure,
                    exploitation_candidates=pool_b, labels=labels_aug, D=D_aug,
                    eps_values=eps_values, n_attempts=EXPLOITATION_ATTEMPTS,
                    rng=transfer_rng, return_scores=True,
                )

                # RANDOM-MEMORY CONTROL: choose one reference uniformly from the SAME
                # M=200 bank. It uses the identical target trajectory, metric scales,
                # exploitation policy, and alpha values. Only memory selection differs.
                random_memory_rng = np.random.default_rng(base_seed + 700000 + library_size)
                random_row = retrieval_candidates_df.iloc[
                    int(random_memory_rng.integers(len(retrieval_candidates_df)))
                ]
                rand_successes, _, _ = load_reference_successes(random_row)
                rand_failures = load_reference_failures_from_path(random_row['memory_path'])
                rand_memory_transforms = np.concatenate([rand_successes, rand_failures], axis=0)
                rand_all_aug = np.concatenate([transforms, rand_memory_transforms], axis=0)
                D_rand_aug = distance_matrix_fixed_scales(
                    rand_all_aug, scale_p=scale_p, scale_r=scale_r
                ).astype(np.float32, copy=False)
                rand_n_target = len(transforms)
                rand_mem_success_idx = np.arange(
                    rand_n_target, rand_n_target + len(rand_successes), dtype=int
                )
                rand_mem_failure_idx = np.arange(
                    rand_n_target + len(rand_successes), len(rand_all_aug), dtype=int
                )
                rand_labels_aug = np.concatenate([
                    labels, np.ones(len(rand_successes), dtype=int),
                    np.zeros(len(rand_failures), dtype=int),
                ])
                rand_observed_structure = np.concatenate([
                    rand_mem_success_idx, rand_mem_failure_idx, transfer_explore
                ]).astype(int)
                random_memory_scores = connectivity_exploitation_order(
                    exploration_order=rand_observed_structure,
                    exploitation_candidates=pool_b, labels=rand_labels_aug, D=D_rand_aug,
                    eps_values=eps_values, n_attempts=EXPLOITATION_ATTEMPTS,
                    rng=transfer_rng, return_scores=True,
                )

                print(
                    f'    final retrieval 1/{len(retrieval_candidates_df)}: '
                    f"{reference['reference_object']} | "
                    f'A successes={n_transfer_obs_success}'
                )

                # Alpha changes only the Pool-B blend; no retrieval or Pool-A work is repeated.
                for alpha in ALPHAS[1:]:
                    blended_scores = (1.0 - alpha) * ca_scores + alpha * memory_scores
                    local_order = np.argsort(-blended_scores, kind='stable')
                    transfer_exploit = np.asarray(
                        pool_b[local_order[:EXPLOITATION_ATTEMPTS]], dtype=int
                    )
                    curve = cumulative_success_curve(transfer_exploit, labels)
                    method_name = 'retrieval_margin_geometry_mem'
                    seed_rows.append({
                        'target': object_id, 'seed': seed_index + 1,
                        'method': method_name, 'library_size': len(retrieval_candidates_df),
                        'alpha': float(alpha), 'K': MATCHED_SUBSETS_K,
                        'memory_warmup': MEMORY_WARMUP_ATTEMPTS,
                        'exploration_attempts': EXPLORATION_ATTEMPTS,
                        'observed_successes': n_transfer_obs_success,
                        'observed_failures': EXPLORATION_ATTEMPTS - n_transfer_obs_success,
                        'retrieved_reference_rank': reference['reference_rank'],
                        'retrieved_reference_object': reference['reference_object'],
                        'retrieval_distance': reference['score'],
                        'candidate_set_size': len(retrieval_candidates_df),
                        'exploitation_successes': int(curve[-1]),
                        **success_checkpoints(curve),
                        'exploitation_auc': discovery_auc(curve),
                        'normalized_auc': normalized_discovery_auc(
                            curve, available_successes=POOL_SUCCESS),
                    })
                    for step, successes in enumerate(curve, start=1):
                        curve_rows.append({
                            'target': object_id, 'seed': seed_index + 1,
                            'method': method_name, 'library_size': len(retrieval_candidates_df),
                            'alpha': float(alpha), 'memory_warmup': MEMORY_WARMUP_ATTEMPTS,
                            'attempt': step, 'cumulative_successes': successes,
                        })

                    # Same alpha, but with a uniformly sampled memory object.
                    random_blended_scores = (1.0 - alpha) * ca_scores + alpha * random_memory_scores
                    random_local_order = np.argsort(-random_blended_scores, kind='stable')
                    random_memory_exploit = np.asarray(
                        pool_b[random_local_order[:EXPLOITATION_ATTEMPTS]], dtype=int
                    )
                    random_curve = cumulative_success_curve(random_memory_exploit, labels)
                    random_method_name = 'random_memory_control'
                    seed_rows.append({
                        'target': object_id, 'seed': seed_index + 1,
                        'method': random_method_name, 'library_size': len(retrieval_candidates_df),
                        'alpha': float(alpha), 'K': MATCHED_SUBSETS_K,
                        'memory_warmup': MEMORY_WARMUP_ATTEMPTS,
                        'exploration_attempts': EXPLORATION_ATTEMPTS,
                        'observed_successes': n_transfer_obs_success,
                        'observed_failures': EXPLORATION_ATTEMPTS - n_transfer_obs_success,
                        'retrieved_reference_rank': int(random_row['reference_rank']),
                        'retrieved_reference_object': str(random_row['object_id']),
                        'retrieval_distance': np.nan,
                        'candidate_set_size': len(retrieval_candidates_df),
                        'exploitation_successes': int(random_curve[-1]),
                        **success_checkpoints(random_curve),
                        'exploitation_auc': discovery_auc(random_curve),
                        'normalized_auc': normalized_discovery_auc(
                            random_curve, available_successes=POOL_SUCCESS),
                    })
                    for step, successes in enumerate(random_curve, start=1):
                        curve_rows.append({
                            'target': object_id, 'seed': seed_index + 1,
                            'method': random_method_name,
                            'library_size': len(retrieval_candidates_df),
                            'alpha': float(alpha),
                            'memory_warmup': MEMORY_WARMUP_ATTEMPTS,
                            'attempt': step, 'cumulative_successes': successes,
                        })

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

        # ---------------------------------------------
        # TARGET FINISHED — release target-level memory
        # ---------------------------------------------
        del D
        del D_a
        del transforms
        del labels
        del pool_a
        del pool_b
        clear_memory_caches(keep_reference_files=False)
    manifest = {
        'experiment':
            'E4_1of10_margin_geometry_retrieval',
        'target_group':
            args.target_group,
        'oracle_self_reference': False,
        'target_excluded_from_reference': True,
        'top_retrieved_references': 1,
        'retrieval_candidate_memories': RETRIEVAL_CANDIDATES,
        'top_reference_actions': 1,
        'protocol':
            'target_excluded_from_reference; target_memory_used_only_for_A_B_overlap_exclusion; '
            'Pool-A ordinary CA only; retrieval among all 799 non-target memories after every '
            'observation 10..100 using full labeled nearest-success/nearest-failure SE(3) margin '
            'geometry; retrieval cannot affect Phase-A sampling',
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
        'seed_index':
            args.seed_index,
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            'E4 structural-memory retrieval control: compare target-only CA, '
            'uniformly random memory, and structurally retrieved memory at M=200.'
        )
    )
    parser.add_argument(
        '--library_dir', type=str, default='results/e4_reference_library'
    )
    parser.add_argument('--n_targets', type=int, default=200)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--bootstrap', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--seed_index',
        type=int,
        default=None,
        help=(
            'Run only this zero-based replicate seed. '
            'Example: --seed_index 0. '
            'If omitted, run all --n_seeds as usual.'
        ),
    )
    parser.add_argument(
        '--objects_dir',
        type=str,
        default=None,
        help=(
            'Root containing original .h5 files for in-memory targets, '
            'unless reference_memory_index.csv already contains H5 paths.'
        ),
    )
    parser.add_argument(
        '--attempts',
        type=int,
        default=400,
        help=(
            'Number of attempts per phase. '
            'Used for both exploration and exploitation. '
            'Default: 400.'
        ),
    )
    args = parser.parse_args()

    print()
    print('#' * 78)
    print('E4 STRUCTURAL-MEMORY RETRIEVAL CONTROL: M=200, attempts<=400')
    print('#' * 78)
    run_group(args, 'independent_targets')


if __name__ == '__main__':
    main()
