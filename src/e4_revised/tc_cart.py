"""Target-conditioned CART (TC-CART) for ACRONYM HDF5 grasp annotations.

Run: python -m src.e4_revised.cart_v2 --grasps-dir grasps --references 20 --targets 10 --seeds 0 1 2 --budget 100
Dependencies: numpy scipy h5py

Target labels are accessed only after a candidate is selected, for offline evaluation.
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import h5py
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

@dataclass
class ObjectData:
    path: Path
    pose: np.ndarray  # normalized xyz + rotation vector, N x 6
    labels: np.ndarray


def load_object(path: Path, max_grasps: int, seed: int) -> ObjectData:
    with h5py.File(path, 'r') as f:
        transforms = np.asarray(f['grasps/transforms'], dtype=np.float64)
        labels = np.asarray(f['grasps/qualities/flex/object_in_gripper']).reshape(-1) > 0
    if len(transforms) != len(labels):
        raise ValueError(f'Pose/label mismatch: {path}')
    valid = np.isfinite(transforms).all(axis=(1, 2))
    idx = np.flatnonzero(valid)
    if len(idx) > max_grasps:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_grasps, replace=False))
    transforms, labels = transforms[idx], labels[idx]
    xyz = transforms[:, :3, 3]
    # Center and normalize using the entire unlabeled candidate set, never target labels.
    xyz = xyz - np.median(xyz, axis=0)
    scale = np.median(np.linalg.norm(xyz, axis=1))
    xyz /= max(float(scale), 1e-8)
    rot = Rotation.from_matrix(transforms[:, :3, :3]).as_rotvec()
    return ObjectData(path, np.concatenate([xyz, rot], axis=1), labels)


def pose_dist(a: np.ndarray, b: np.ndarray, rotation_weight: float = 0.5) -> np.ndarray:
    """Approximate pose metric. Rotation-vector Euclidean distance is NOT SO(3) geodesic."""
    return np.sqrt(cdist(a[:, :3], b[:, :3], 'sqeuclidean') +
                   rotation_weight ** 2 * cdist(a[:, 3:], b[:, 3:], 'sqeuclidean'))


@dataclass
class RegionMemory:
    prototypes: np.ndarray
    weights: np.ndarray
    bandwidths: np.ndarray
    component_ids: np.ndarray


def reference_memory(references: list[ObjectData], eps: float, min_size: int,
                     medoids: int, rotation_weight: float, seed: int,
                     memory_budget: int = 0, size_power: float = 0.5,
                     density_power: float = 0.5, spread_factor: float = 1.0,
                     min_bandwidth: float = 0.2, max_bandwidth: float = 1.0,
                     prototypes_per_object: int | None = None,
                     _normalize_weights: bool = True
                     ) -> tuple[RegionMemory, np.ndarray]:
    """Equal prototype count, but CART v2 retains component identity and geometry.

    Density = inverse median nearest-neighbor distance within a component.
    Component weight ~ size**size_power * density**density_power.
    Width = clipped component median medoid-distance * spread_factor.
    Weights normalized over components, not over prototypes.
    """
    # Optional per-reference-object cap. Preserve the existing global-budget
    # behavior when prototypes_per_object is not supplied.
    if prototypes_per_object is not None:
        if prototypes_per_object < 1:
            raise ValueError('prototypes_per_object must be positive')
        pieces, point_sets = [], []
        for oi, obj in enumerate(references):
            if not np.any(obj.labels):
                continue
            try:
                piece, points = reference_memory(
                    [obj], eps, min_size, medoids, rotation_weight, seed + oi,
                    memory_budget=prototypes_per_object,
                    size_power=size_power, density_power=density_power,
                    spread_factor=spread_factor,
                    min_bandwidth=min_bandwidth, max_bandwidth=max_bandwidth,
                    prototypes_per_object=None,
                    _normalize_weights=False)
            except ValueError as exc:
                if 'No viable reference regions' in str(exc):
                    continue
                raise
            pieces.append(piece)
            point_sets.append(points)
        if not pieces:
            raise ValueError('No viable reference regions')
        offsets = np.cumsum([0] + [len(m.weights) for m in pieces[:-1]])
        component_ids = np.concatenate([m.component_ids + off for m, off in zip(pieces, offsets)])
        # Normalize raw component weights globally across all reference objects.
        weights = np.concatenate([m.weights for m in pieces])
        weights /= weights.sum()
        return RegionMemory(
            prototypes=np.concatenate([m.prototypes for m in pieces]),
            weights=weights,
            bandwidths=np.concatenate([m.bandwidths for m in pieces]),
            component_ids=component_ids), np.concatenate(point_sets)

    components = []
    all_points = []
    for obj in references:
        good = obj.pose[obj.labels]
        if not len(good):
            continue
        all_points.append(good)
        d = pose_dist(good, good, rotation_weight)
        ncomp, comp = connected_components(csr_matrix(d <= eps), directed=False)
        for c in range(ncomp):
            ids = np.flatnonzero(comp == c)
            if len(ids) < min_size:
                continue
            local = d[np.ix_(ids, ids)]
            chosen = [int(np.argmin(local.sum(axis=1)))]
            mind = local[chosen[0]].copy()
            for _ in range(1, min(medoids, len(ids))):
                j = int(np.argmax(mind))
                if j in chosen:
                    break
                chosen.append(j)
                mind = np.minimum(mind, local[j])
            nearest = local.copy()
            np.fill_diagonal(nearest, np.inf)
            spacing = float(np.median(nearest.min(axis=1)))
            density = 1.0 / max(spacing, 1e-4)
            spread = float(np.median(mind))
            width = float(np.clip(spread_factor * max(spread, min_bandwidth),
                                  min_bandwidth, max_bandwidth))
            weight = float(len(ids)**size_power * density**density_power)
            components.append((good[ids[chosen]], weight, width))
    if not all_points or not components:
        raise ValueError('No viable reference regions; adjust --eps/--min-region-size')
    all_points = np.concatenate(all_points)
    rng = np.random.default_rng(seed + 104729)
    # Retain whole component identities. If the cap cuts a component, retain
    # its first (medoid-first) prototypes; distribute slots round-robin.
    total = sum(len(c[0]) for c in components)
    k = min(total, len(all_points), memory_budget if memory_budget > 0 else total)
    selected = [[] for _ in components]
    remaining = k
    for level in range(max(len(c[0]) for c in components)):
        for ci, (poses, _, _) in enumerate(components):
            if remaining == 0:
                break
            if level < len(poses):
                selected[ci].append(level)
                remaining -= 1
        if remaining == 0:
            break
    protos, weights, widths, ids = [], [], [], []
    for ci, (poses, weight, width) in enumerate(components):
        if not selected[ci]:
            continue
        subset = poses[selected[ci]]
        protos.append(subset)
        weights.append(weight)
        widths.append(width)
        ids.extend([len(weights) - 1] * len(subset))
    prototypes = np.concatenate(protos)
    weights = np.asarray(weights, dtype=float)
    if _normalize_weights:
        weights /= weights.sum()
    points = all_points[rng.choice(len(all_points), len(prototypes), replace=False)]
    assert len(points) == len(prototypes) == k
    return RegionMemory(prototypes, weights, np.asarray(widths), np.asarray(ids)), points


def region_prior_scores(candidates: np.ndarray, memory: RegionMemory,
                        rotation_weight: float, chunk: int = 256) -> np.ndarray:
    """Weighted mixture of component kernels; each component contributes once.

    Region support is exp(-0.5*(distance to closest component medoid / width)^2).
    Scores lie in [0,1]. Complexity O(num_candidates * num_prototypes).
    """
    result = np.zeros(len(candidates), dtype=float)
    ncomp = len(memory.weights)
    for start in range(0, len(candidates), chunk):
        d = pose_dist(candidates[start:start+chunk], memory.prototypes, rotation_weight)
        local = np.zeros(len(d), dtype=float)
        for ci in range(ncomp):
            md = d[:, memory.component_ids == ci].min(axis=1)
            local += memory.weights[ci] * np.exp(-0.5 * (md / memory.bandwidths[ci])**2)
        result[start:start+chunk] = local
    return result


def prior_scores(candidates: np.ndarray, memory: np.ndarray, bandwidth: float,
                 rotation_weight: float, chunk: int = 512) -> np.ndarray:
    """Nearest prototype Gaussian score; bounded [0,1]."""
    result = np.empty(len(candidates))
    for start in range(0, len(candidates), chunk):
        d = pose_dist(candidates[start:start + chunk], memory, rotation_weight)
        result[start:start + chunk] = np.exp(-0.5 * (d.min(axis=1) / bandwidth) ** 2)
    return result


def explore_legacy(obj: ObjectData, prior: np.ndarray, method: str, budget: int, seed: int,
                   bandwidth: float, rotation_weight: float, alpha: float) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = len(obj.labels)
    seen = np.zeros(n, dtype=bool)
    observed_good, observed_bad = [], []
    records = []
    successes = 0
    for t in range(min(budget, n)):
        if method == 'random':
            candidate = int(rng.choice(np.flatnonzero(~seen)))
        else:
            score = np.zeros(n) if method == 'target_ca' else prior.copy()
            if observed_good:
                d = pose_dist(obj.pose, obj.pose[observed_good], rotation_weight).min(axis=1)
                local = np.exp(-0.5 * (d / bandwidth) ** 2)
                score = (1 - alpha) * score + alpha * local if method != 'target_ca' else local
            if observed_bad:
                d = pose_dist(obj.pose, obj.pose[observed_bad], rotation_weight).min(axis=1)
                score -= 0.5 * np.exp(-0.5 * (d / bandwidth) ** 2)
            # Prior alone is deterministic; random tie-breaking is seeded.
            score += rng.uniform(0, 1e-10, n)
            score[seen] = -np.inf
            candidate = int(np.argmax(score))
        seen[candidate] = True
        outcome = bool(obj.labels[candidate])  # Only reveal this label AFTER acquisition.
        (observed_good if outcome else observed_bad).append(candidate)
        successes += int(outcome)
        records.append({'step': t + 1, 'success': int(outcome), 'cumulative_success': successes})
    return records


def region_support_matrix(candidates: np.ndarray, memory: RegionMemory,
                          rotation_weight: float, chunk: int = 256) -> np.ndarray:
    """Candidate x component kernel supports. No target labels are used."""
    out = np.empty((len(candidates), len(memory.weights)), dtype=np.float32)
    component_indices = [np.flatnonzero(memory.component_ids == c)
                         for c in range(len(memory.weights))]
    for start in range(0, len(candidates), chunk):
        d = pose_dist(candidates[start:start + chunk], memory.prototypes, rotation_weight)
        for c, ids in enumerate(component_indices):
            out[start:start + chunk, c] = np.exp(
                -0.5 * (d[:, ids].min(axis=1) / memory.bandwidths[c]) ** 2)
    return out


def target_ca_score(poses: np.ndarray, good: list[int], bad: list[int],
                    bandwidth: float, rotation_weight: float) -> np.ndarray:
    score = np.zeros(len(poses), dtype=float)
    if good:
        d = pose_dist(poses, poses[good], rotation_weight).min(axis=1)
        score += np.exp(-0.5 * (d / bandwidth) ** 2)
    if bad:
        d = pose_dist(poses, poses[bad], rotation_weight).min(axis=1)
        score -= 0.5 * np.exp(-0.5 * (d / bandwidth) ** 2)
    return score


def explore_tc_cart(obj: ObjectData, supports: np.ndarray, memory: RegionMemory,
                    budget: int, seed: int, bandwidth: float, rotation_weight: float,
                    warmup: int, top_regions: int, max_transfer: float,
                    min_transfer: float, agreement_temperature: float,
                    refresh_every: int) -> list[dict]:
    """Target-conditioned region retrieval with confidence-gated transfer.

    Warm-up follows exactly the target_ca acquisition policy. Subsequently,
    rank regions using ONLY observed target outcomes. A centered kernel
    agreement penalizes regions that support observed failures. Adaptive
    transfer is zero if evidence does not exceed a zero-information baseline.
    """
    rng = np.random.default_rng(seed)
    n = len(obj.labels)
    seen = np.zeros(n, bool)
    good, bad, records = [], [], []
    successes = 0
    region_prior = np.zeros(n, dtype=float)
    gate = 0.0
    selected_count = 0

    for t in range(min(budget, n)):
        ca = target_ca_score(obj.pose, good, bad, bandwidth, rotation_weight)
        if t < warmup:
            score = ca
        else:
            if t == warmup or (t - warmup) % refresh_every == 0:
                observed = np.asarray(good + bad, dtype=int)
                y = np.r_[np.ones(len(good)), np.zeros(len(bad))]
                # Correlation with observed labels after centering the labels;
                # if all outcomes are identical, avoid false confidence.
                if len(observed) >= 4 and 0 < y.mean() < 1:
                    obs_support = supports[observed].astype(float)
                    agreement = ((y - y.mean())[:, None] * obs_support).mean(axis=0)
                    # Normalize for number of observations, retain region weights.
                    rank = agreement + agreement_temperature * np.log(
                        np.maximum(memory.weights, 1e-12))
                    chosen = np.argsort(rank)[-min(top_regions, len(rank)):]
                    chosen = chosen[agreement[chosen] > 0]
                    selected_count = len(chosen)
                    if selected_count:
                        w = memory.weights[chosen] * np.maximum(agreement[chosen], 0)
                        w = w / max(w.sum(), 1e-12)
                        region_prior = supports[:, chosen] @ w
                        # Observed predictive separation; no use of unseen labels.
                        positive = region_prior[good].mean() if good else 0.0
                        negative = region_prior[bad].mean() if bad else 0.0
                        separation = max(0.0, positive - negative)
                        gate = float(np.clip(separation, 0, 1)) * max_transfer
                    else:
                        gate = 0.0
                        region_prior.fill(0.0)
                else:
                    gate = 0.0
                    selected_count = 0
                    region_prior.fill(0.0)
            transfer = max(min_transfer, gate) if selected_count else 0.0
            # Keep failure penalty from CA; transfer only positive prior.
            score = (1 - transfer) * ca + transfer * region_prior
        score = score.copy()
        score += rng.uniform(0, 1e-10, n)
        score[seen] = -np.inf
        candidate = int(np.argmax(score))
        seen[candidate] = True
        outcome = bool(obj.labels[candidate])  # Label revealed AFTER selection.
        (good if outcome else bad).append(candidate)
        successes += int(outcome)
        records.append({'step': t + 1, 'success': int(outcome),
                        'cumulative_success': successes})
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--grasps-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, default=Path('cart_results.csv'))
    p.add_argument('--references', type=int, default=20)
    p.add_argument('--targets', type=int, default=10)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--budget', type=int, default=100)
    p.add_argument('--max-grasps', type=int, default=1000)
    p.add_argument('--eps', type=float, default=0.35)
    p.add_argument('--min-region-size', type=int, default=3)
    p.add_argument('--medoids', type=int, default=3)
    p.add_argument('--memory-budget', type=int, default=0, help='Maximum prototypes PER method; 0 uses all CART prototypes')
    p.add_argument('--bandwidth', type=float, default=0.5)
    p.add_argument('--rotation-weight', type=float, default=0.5)
    p.add_argument('--alpha', type=float, default=0.5)
    p.add_argument('--size-power', type=float, default=0.5)
    p.add_argument('--density-power', type=float, default=0.5)
    p.add_argument('--spread-factor', type=float, default=1.0)
    p.add_argument('--min-bandwidth', type=float, default=0.2)
    p.add_argument('--max-bandwidth', type=float, default=1.0)
    p.add_argument('--warmup', type=int, default=15)
    p.add_argument('--top-regions', type=int, default=20)
    p.add_argument('--max-transfer', type=float, default=0.5)
    p.add_argument('--min-transfer', type=float, default=0.0)
    p.add_argument('--agreement-temperature', type=float, default=0.01)
    p.add_argument('--refresh-every', type=int, default=5)
    args = p.parse_args()
    if not (0 <= args.min_transfer <= args.max_transfer <= 1):
        p.error('Require 0 <= min-transfer <= max-transfer <= 1')
    if args.top_regions < 1 or args.refresh_every < 1 or args.warmup < 0:
        p.error('top-regions and refresh-every must be positive; warmup nonnegative')
    paths = sorted(args.grasps_dir.rglob('*.h5'))
    if len(paths) < args.references + args.targets:
        raise ValueError('Not enough HDF5 objects')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['seed', 'target', 'method', 'step', 'success', 'cumulative_success'])
        writer.writeheader()
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            selected = rng.choice(len(paths), args.references + args.targets, replace=False)
            ref_paths = [paths[i] for i in selected[:args.references]]
            target_paths = [paths[i] for i in selected[args.references:]]
            references = [load_object(path, args.max_grasps, seed) for path in ref_paths]
            regions, points = reference_memory(references, args.eps, args.min_region_size,
                                               args.medoids, args.rotation_weight, seed, args.memory_budget,
                                               args.size_power, args.density_power, args.spread_factor,
                                               args.min_bandwidth, args.max_bandwidth)
            print(f'seed={seed} equal_memory={len(regions.prototypes)} region_prototypes={len(regions.prototypes)} point_prototypes={len(points)}', flush=True)
            for target_path in target_paths:
                obj = load_object(target_path, args.max_grasps, seed)
                cart_prior = region_prior_scores(obj.pose, regions, args.rotation_weight)
                point_prior = prior_scores(obj.pose, points, args.bandwidth, args.rotation_weight)
                supports = region_support_matrix(obj.pose, regions, args.rotation_weight)
                for method in ('random', 'target_ca', 'pointwise', 'cart_v2', 'tc_cart'):
                    prior = cart_prior if method == 'cart_v2' else point_prior if method == 'pointwise' else np.zeros(len(obj.labels))
                    if method == 'tc_cart':
                        rows = explore_tc_cart(
                            obj, supports, regions, args.budget, seed,
                            args.bandwidth, args.rotation_weight,
                            args.warmup, args.top_regions, args.max_transfer,
                            args.min_transfer, args.agreement_temperature,
                            args.refresh_every)
                    else:
                        rows = explore_legacy(
                            obj, prior, method, args.budget, seed,
                            args.bandwidth, args.rotation_weight, args.alpha)
                    for row in rows:
                        writer.writerow({'seed': seed, 'target': target_path.name, 'method': method, **row})
                    print(f'{target_path.stem[:25]:25s} {method:10s} {rows[-1]["cumulative_success"]:3d}/{len(rows)}', flush=True)
                file.flush()
    print(f'Saved {args.out}', flush=True)

if __name__ == '__main__':
    main()
