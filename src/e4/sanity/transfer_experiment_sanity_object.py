import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from src.utils.samplers import random_order, fps_order, choose_connectivity_point

MATCHED_SUBSETS_K = 10
BEST_MATCH_COUNT = 3
TOP_RETRIEVED_REFERENCES = 1
TOP_REFERENCE_ACTIONS = 1
LIBRARY_SIZES = [800]  # controlled in-memory/out-of-memory comparison
ALPHAS = [0.0, 1.0]  # one known-memory method; no blend hyperparameter
MEMORY_WARMUP_ATTEMPTS = 0  # historical memory initializes exploration from query 1


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
    # ORACLE-SELF SANITY CHECK: the target object's identity is GIVEN.
    # Restrict retrieval to the exact same object's reference-memory entry.
    # We still match the current target-success topology to a same-cardinality
    # subset inside that object's stored successful grasps.
    library = reference_df[
        reference_df['object_id'].astype(str) == str(target_object_id)
        ].copy().reset_index(drop=True)
    if len(library) != 1:
        raise RuntimeError(
            f'Expected exactly one reference-memory entry for target '
            f'{target_object_id!r}, found {len(library)}.'
        )

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


def known_memory_connectivity_exploration_on_pool(
        labels, D_aug, eps_values, candidate_indices, n_attempts,
        memory_success_indices, rng,
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
    successful_target = []

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


def run_group(args, target_group):
    args.target_group = target_group


    # Match E3 exactly on the target-side pool composition and budgets.
    # Pool A: 100 success + 100 failure; Pool B: 100 success + 100 failure.
    # Phase 1: 100 exploration attempts; Phase 2: 100 exploitation attempts.
    POOL_SUCCESS = 100
    POOL_FAILURE = 100

    EXPLORATION_ATTEMPTS = 100
    EXPLOITATION_ATTEMPTS = 100

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
            Path('results') / 'e4_known_object_memory_greedy' / args.target_group
    )
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    eps_values = np.linspace(0.1, 0.8, 15)

    print('=' * 78)
    print('E4: KNOWN OBJECT MEMORY: CA-EXPLORE NEW A, GREEDILY EXPLOIT ENRICHED MEMORY ON B')
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

    if args.target_group == 'out_of_memory':
        target_df = heldout_target_df.copy()
        if len(target_df) < args.n_targets:
            raise RuntimeError(
                f'Only {len(target_df)} out-of-memory target candidates; '
                f'requested {args.n_targets}.'
            )
        selected_positions = master_rng.choice(
            len(target_df), size=args.n_targets, replace=False
        )
        selected_targets = (
            target_df.iloc[selected_positions].copy().reset_index(drop=True)
        )
    else:
        # Controlled same-object condition: every selected target is one of the
        # 800 objects available to retrieval.  We still load the ORIGINAL H5
        # object and resample fresh disjoint Pool A/B target sets; we do not use
        # the reference NPZ as the target pool.
        in_memory_df = reference_df[
            reference_df['reference_rank'] <= max(LIBRARY_SIZES)
            ].copy().reset_index(drop=True)

        if len(in_memory_df) < args.n_targets:
            raise RuntimeError(
                f'Only {len(in_memory_df)} in-memory reference objects; '
                f'requested {args.n_targets}.'
            )

        # Resolve original H5 paths. Prefer a path/source_path column if the
        # reference index already contains one. Otherwise scan --objects_dir.
        path_col = None
        for candidate_col in ('path', 'source_path', 'h5_path', 'object_path'):
            if candidate_col in in_memory_df.columns:
                path_col = candidate_col
                break

        if path_col is not None:
            in_memory_df['path'] = in_memory_df[path_col].astype(str)
        else:
            if args.objects_dir is None:
                raise RuntimeError(
                    'In-memory targets require original H5 files. '
                    'reference_memory_index.csv has no path column, so pass '
                    '--objects_dir /path/to/the/object-h5-root.'
                )
            objects_root = Path(args.objects_dir)
            if not objects_root.exists():
                raise FileNotFoundError(objects_root)

            wanted = set(in_memory_df['object_id'].astype(str))
            resolved = {}
            for p in objects_root.rglob('*.h5'):
                name = p.name
                if name in wanted:
                    resolved[name] = str(p)
                # Also support object_id stored without the .h5 suffix.
                if p.stem in wanted:
                    resolved[p.stem] = str(p)

            in_memory_df['path'] = [
                resolved.get(str(object_id))
                for object_id in in_memory_df['object_id']
            ]
            missing_objects = in_memory_df.loc[
                in_memory_df['path'].isna(), 'object_id'
            ].astype(str).tolist()
            if missing_objects:
                preview = ', '.join(missing_objects[:5])
                raise RuntimeError(
                    f'Could not resolve {len(missing_objects)} in-memory H5 '
                    f'files under {objects_root}. Examples: {preview}'
                )

        selected_positions = master_rng.choice(
            len(in_memory_df), size=args.n_targets, replace=False
        )
        selected_targets = (
            in_memory_df.iloc[selected_positions][['object_id', 'path']]
            .copy()
            .reset_index(drop=True)
        )

    print(f'Target group: {args.target_group}')
    print(f'Results directory: {results_dir}')

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

        # Exact same-object historical memory R is GIVEN, but its grasp points
        # must never be reused as Pool-A or Pool-B candidates.
        exact_rows = reference_df[
            reference_df['object_id'].astype(str) == str(object_id)
            ]
        if len(exact_rows) == 0:
            raise RuntimeError(f"No exact-object memory found for {object_id}.")
        memory_row = exact_rows.iloc[0]
        ref_successes, ref_scale_p, ref_scale_r = load_reference_successes(memory_row)
        ref_failures = load_reference_failures_from_path(memory_row['memory_path'])
        ref_all_transforms = np.concatenate([ref_successes, ref_failures], axis=0)

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
            excluded_transforms=ref_all_transforms,
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

        # Hard protocol checks: R, A and B must contain different grasp points.
        a_keys = set(_transform_keys(transforms[pool_a]))
        b_keys = set(_transform_keys(transforms[pool_b]))
        r_keys = set(_transform_keys(ref_all_transforms))
        assert a_keys.isdisjoint(b_keys), 'Protocol violation: A and B overlap.'
        assert r_keys.isdisjoint(a_keys), 'Protocol violation: R and A overlap.'
        assert r_keys.isdisjoint(b_keys), 'Protocol violation: R and B overlap.'

        print(
            f'  disjointness verified: R∩A=0, R∩B=0, A∩B=0; '
            f'|R|={len(ref_all_transforms)}'
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

                    # Exact same-object historical memory is GIVEN.
                    # It was loaded before pool construction so A/B could exclude R.
                    row = memory_row

                    # Augmented metric: target candidates first, then historical memory.
                    memory_transforms = np.concatenate([ref_successes, ref_failures], axis=0)
                    all_aug = np.concatenate([transforms, memory_transforms], axis=0)
                    D_aug = distance_matrix_fixed_scales(
                        all_aug, scale_p=scale_p, scale_r=scale_r
                    ).astype(np.float32, copy=False)
                    n_target = len(transforms)
                    mem_success_idx = np.arange(
                        n_target, n_target + len(ref_successes), dtype=int
                    )
                    mem_failure_idx = np.arange(
                        n_target + len(ref_successes), len(all_aug), dtype=int
                    )
                    labels_aug = np.concatenate([
                        labels,
                        np.ones(len(ref_successes), dtype=int),
                        np.zeros(len(ref_failures), dtype=int),
                    ])

                    # Phase A: genuine CA exploration initialized by historical memory.
                    transfer_explore = known_memory_connectivity_exploration_on_pool(
                        labels=labels_aug, D_aug=D_aug, eps_values=eps_values,
                        candidate_indices=pool_a, n_attempts=EXPLORATION_ATTEMPTS,
                        memory_success_indices=mem_success_idx, rng=transfer_rng,
                    )
                    reference = {
                        'reference_rank': int(row['reference_rank']),
                        'reference_object': str(row['object_id']),
                        'mean_distance': 0.0,
                    }
                    retrieval_rows = []

                    n_transfer_obs_success = int(
                        np.sum(
                            labels[
                                transfer_explore
                            ] == 1
                        )
                    )

                    # Phase B: greedy exploitation of the ENRICHED same-object
                    # viability structure: historical S/F memory + all observed A outcomes.
                    observed_structure = np.concatenate([
                        mem_success_idx, mem_failure_idx, transfer_explore
                    ]).astype(int)
                    transfer_exploit = connectivity_exploitation_order(
                        exploration_order=observed_structure,
                        exploitation_candidates=pool_b,
                        labels=labels_aug, D=D_aug, eps_values=eps_values,
                        n_attempts=EXPLOITATION_ATTEMPTS, rng=transfer_rng,
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
            'E4_discovery_memory_exploration',
        'target_group':
            args.target_group,
        'oracle_self_reference': True,
        'top_retrieved_references': 1,
        'top_reference_actions': 1,
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            'E4 oracle-self EXPLORATION sanity check: the target object identity is given, '
            'and one top reference action is transferred.'
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
        '--objects_dir',
        type=str,
        default=None,
        help=(
            'Root containing original .h5 files for in-memory targets, '
            'unless reference_memory_index.csv already contains H5 paths.'
        ),
    )
    args = parser.parse_args()

    print()
    print('#' * 78)
    print('ORACLE-SELF SANITY CHECK: SAME OBJECT GIVEN, TOP-1 ACTION')
    print('#' * 78)
    run_group(args, 'in_memory')


if __name__ == '__main__':
    main()
