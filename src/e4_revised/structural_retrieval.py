"""Label-safe connectivity-signature retrieval (no cross-object pose matching).

Reference and target signatures both use ALL unlabeled candidate poses.
Reference prototype signatures are sampled at the prototype poses from their
source object candidate graphs. The DynamicStructuralIndex adds online outcome-conditioned reranking.
It is not a learned topology embedding or explicit graph-isomorphism matching.
"""
import numpy as np
from scipy.spatial import cKDTree


def signatures(poses, rotation_weight=0.5, k=8):
    """Eight local graph statistics, invariant to translation/rotation of pose coordinates.

    Graph is a kNN graph in *within-object* pose space. Pose is used only to
    construct within-object adjacency; it is never used for cross-object retrieval.
    """
    n = len(poses)
    if n == 0:
        return np.empty((0, 8), np.float32)
    if n == 1:
        return np.zeros((1, 8), np.float32)
    x = np.asarray(poses, dtype=np.float64).copy()
    x[:, 3:] *= rotation_weight
    kk = min(k, n - 1)
    dist, ids = cKDTree(x).query(x, k=kk + 1, workers=1)
    dist, ids = dist[:, 1:], ids[:, 1:]
    # Dimensionless local distances, normalized by the object's global kNN scale.
    scale = max(float(np.median(dist[:, -1])), 1e-6)
    dn = dist / scale
    # Directed kNN graph statistics, using neighbors' own local graph geometry.
    degree_radius = np.mean(dn <= 1.0, axis=1)
    neighbor_scale = np.mean(dist[ids, -1], axis=1) / scale
    # Mutual edges, local neighborhood closure, and variation in local spacing.
    mutual = np.array([np.mean([i in ids[j] for j in ids[i]]) for i in range(n)])
    closure = np.empty(n)
    for i in range(n):
        nei = ids[i]
        closure[i] = np.mean([np.mean(np.isin(ids[j], nei)) for j in nei])
    dmean = dn.mean(axis=1)
    dstd = dn.std(axis=1)
    dmax = dn[:, -1]
    # Mean second-hop normalized local scale.
    twohop = (dist[ids, -1] / scale).mean(axis=1)
    return np.stack([dmean, dstd, dmax, degree_radius, neighbor_scale,
                     mutual, closure, twohop], axis=1).astype(np.float32)


def attach_reference_signatures(memory, objects, owners, rotation_weight):
    """Assign each successful reference prototype a signature from its source
    object's full unlabeled candidate graph (same population as test queries)."""
    sig = np.zeros((len(memory.prototypes), 8), np.float32)
    for owner, obj in enumerate(objects):
        idx = np.flatnonzero(owners == owner)
        if not len(idx):
            continue
        candidates = obj.pose
        desc = signatures(candidates, rotation_weight)
        xyz = candidates.copy(); xyz[:, 3:] *= rotation_weight
        proto = memory.prototypes[idx].copy(); proto[:, 3:] *= rotation_weight
        nearest = cKDTree(xyz).query(proto, workers=1)[1]
        sig[idx] = desc[nearest]
    memory.structural_signatures = sig
    memory.signature_scale = np.maximum(sig.std(axis=0), 0.05).astype(np.float32)
    return memory


def structural_memory_features(poses, memory, k, rotation_weight, mask=None):
    """Nearest signatures only; 8 descriptor differences + 1 descriptor distance."""
    q = signatures(poses, rotation_weight)
    ref = memory.structural_signatures
    if mask is not None:
        ref = ref[mask]
    if len(ref) == 0:
        raise ValueError('No reference signatures available')
    scale = memory.signature_scale
    tree = cKDTree(ref / scale)
    distance, ids = tree.query(q / scale, k=min(k, len(ref)), workers=1)
    if np.ndim(ids) == 1:
        ids = ids[:, None]; distance = distance[:, None]
    delta = (ref[ids] - q[:, None, :]) / scale
    features = np.concatenate([delta, np.asarray(distance)[..., None]], axis=-1)
    # Keep the same memory encoder input shape if fewer than k references exist.
    if features.shape[1] < k:
        features = np.concatenate([features, np.repeat(features[:, -1:, :], k - features.shape[1], axis=1)], axis=1)
    return features.astype(np.float32)

class DynamicStructuralIndex:
    """Static candidate geometry, dynamically reranked by observed target outcomes.

    All signatures use unlabeled candidate graphs. The success/failure evidence
    changes only after an observed target outcome. No future labels are accessed.
    """
    def __init__(self, poses, memory, k, rotation_weight, mask=None,
                 candidates=64, strength=0.75, bandwidth=1.0):
        self.query = signatures(poses, rotation_weight)
        self.ref = memory.structural_signatures if mask is None else memory.structural_signatures[mask]
        if not len(self.ref):
            raise ValueError('Empty reference set')
        self.scale = memory.signature_scale
        self.k = int(k)
        self.strength = float(strength)
        self.bandwidth = float(bandwidth)
        if self.bandwidth <= 0:
            raise ValueError('bandwidth must be positive')
        self.ref_scaled = self.ref / self.scale
        self.query_scaled = self.query / self.scale
        self.tree = cKDTree(self.ref_scaled)
        self.d, self.ids = self.tree.query(self.query_scaled,
            k=min(len(self.ref), max(self.k, int(candidates))), workers=1)
        self.d = np.asarray(self.d).reshape(len(poses), -1)
        self.ids = np.asarray(self.ids).reshape(len(poses), -1)

    def features(self, good=(), bad=(), query_ids=None):
        """Return (N,k,9), reselecting reference prototypes from observed outcomes."""
        support = np.zeros(len(self.ref), dtype=np.float32)
        for observed, sign in ((good, 1.0), (bad, -1.0)):
            observed = np.asarray(observed, dtype=int)
            if not len(observed):
                continue
            # Evidence about reference signatures from already-observed target
            # candidates; compare each observation against all reference prototypes.
            d, ids = self.d[observed], self.ids[observed]
            weight = np.exp(-0.5 * (d / self.bandwidth) ** 2)
            np.add.at(support, ids.ravel(), (sign * weight / len(observed)).ravel())
        # Avoid evidence magnitude scaling with the number of prototypes.
        if np.any(support):
            support /= max(float(np.max(np.abs(support))), 1e-8)
        qi = np.arange(len(self.query)) if query_ids is None else np.asarray(query_ids, dtype=int)
        ids = self.ids[qi]
        d = self.d[qi]
        adjusted = d - self.strength * support[ids]
        chosen = np.argsort(adjusted, axis=1, kind='stable')[:, :min(self.k, ids.shape[1])]
        selected = np.take_along_axis(ids, chosen, axis=1)
        original_d = np.take_along_axis(d, chosen, axis=1)
        delta = (self.ref[selected] - self.query[qi, None, :]) / self.scale
        out = np.concatenate([delta, original_d[..., None]], axis=-1)
        if out.shape[1] < self.k:
            out = np.concatenate([out, np.repeat(out[:, -1:, :], self.k-out.shape[1], axis=1)], axis=1)
        return out.astype(np.float32)
