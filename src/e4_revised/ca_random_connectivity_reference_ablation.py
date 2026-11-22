"""Four-way comparison: random, target-only CA, geometry encoder, connectivity encoder.

Place alongside tc_cart.py in src/e4_revised/. Run as python -m src.e4_revised.ca_random_connectivity.
Requires numpy, scipy, h5py, torch, and tc_cart.py.

Adaptive mode: reference objects are supervised training objects; validation and evaluation are disjoint.
No evaluation-target labels are revealed before selection. Target context labels
are observed only after selecting each grasp. All target candidate poses are
available as unlabeled features (same assumption as existing CART pilots).
"""
from __future__ import annotations
import argparse, csv, random, time, copy, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.spatial import cKDTree
from .tc_cart import load_object, reference_memory, explore_legacy
from .structural_retrieval import attach_reference_signatures, structural_memory_features, DynamicStructuralIndex


def pose_features(x):
    # continuous representation avoids rotation-vector seam in learned encoder
    return np.concatenate([x[:, :3], np.sin(x[:, 3:]), np.cos(x[:, 3:])], axis=1).astype('float32')


def build_memory_features(poses, prototypes, comp_ids, weights, widths, k, rotation_weight, connectivity):
    """Local memory set for each query, with optional component descriptors.
    All methods use IDENTICAL neighbors and prototypes; connectivity only
    changes the two component-metadata channels.
    """
    scale = np.array([1, 1, 1, rotation_weight, rotation_weight, rotation_weight], dtype=np.float32)
    tree = cKDTree(prototypes * scale)
    dist, ids = tree.query(poses * scale, k=min(k, len(prototypes)), workers=1)
    if np.ndim(ids) == 1:
        ids = ids[:, None]; dist = dist[:, None]
    rel = (prototypes[ids] - poses[:, None, :]).astype('float32')
    rel[:, :, 3:] *= rotation_weight
    dist = np.asarray(dist, dtype='float32')
    # memory rows: relative pose, distance, component weight, width
    if connectivity:
        meta = np.stack([np.log1p(weights[comp_ids[ids]] * len(weights)), widths[comp_ids[ids]]], axis=-1)
    else:
        meta = np.zeros((*ids.shape, 2), dtype=np.float32)
    feats = np.concatenate([rel, dist[..., None], meta.astype('float32')], axis=-1)
    return feats.astype('float32')


def target_conditioned_memory_features(poses, memory, k, rot_weight, good, bad,
                                       connectivity, retrieval_strength=0.5,
                                       retrieval_candidates=64, retrieval_bandwidth=0.5):
    """Re-rank reference kNN using observed target outcomes (no test-label leakage).

    For each reference component, estimate positive/negative support from already
    observed target grasps. Favor components agreeing with successful observations
    and penalize those agreeing with failures. Keep actual geometric distance in
    model features; use adjusted distance only to select neighbors.
    """
    proto = memory.prototypes
    scale = np.array([1, 1, 1, rot_weight, rot_weight, rot_weight], dtype=np.float32)
    tree = cKDTree(proto * scale)
    n_search = min(len(proto), max(k, retrieval_candidates))
    dist, ids = tree.query(poses * scale, k=n_search, workers=1)
    dist = np.asarray(dist).reshape(len(poses), -1)
    ids = np.asarray(ids).reshape(len(poses), -1)
    if not good and not bad:
        return build_memory_features(poses, proto, memory.component_ids,
                                     memory.weights, memory.bandwidths,
                                     k, rot_weight, connectivity)
    # Evidence from already attempted target grasps only.
    n_components = len(memory.weights)
    support = np.zeros(n_components, dtype=np.float64)
    for indices, sign in ((good, 1.0), (bad, -1.0)):
        if not indices:
            continue
        d_obs, id_obs = tree.query(poses[np.asarray(indices)] * scale,
                                   k=n_search, workers=1)
        d_obs = np.asarray(d_obs).reshape(len(indices), -1)
        id_obs = np.asarray(id_obs).reshape(len(indices), -1)
        kernel = np.exp(-0.5 * (d_obs / retrieval_bandwidth)**2)
        # Maximum local support for each component per observation.
        for drow, irow, kvals in zip(d_obs, id_obs, kernel):
            local = np.zeros(n_components)
            np.maximum.at(local, memory.component_ids[irow], kvals)
            support += sign * local / len(indices)
    # Zero-centered relative evidence: only relative component agreement matters.
    support -= np.mean(support)
    maxabs = np.max(np.abs(support))
    if maxabs > 1e-12:
        support /= maxabs
    # Candidate-specific geometric neighbors are re-ranked by region agreement.
    adjusted = dist - retrieval_strength * support[memory.component_ids[ids]]
    chosen = np.argsort(adjusted, axis=1, kind="stable")[:, :min(k, n_search)]
    selected_ids = np.take_along_axis(ids, chosen, axis=1)
    selected_dist = np.take_along_axis(dist, chosen, axis=1)
    rel = (proto[selected_ids] - poses[:, None, :]).astype(np.float32)
    rel[:, :, 3:] *= rot_weight
    if connectivity:
        meta = np.stack([
            np.log1p(memory.weights[memory.component_ids[selected_ids]] * len(memory.weights)),
            memory.bandwidths[memory.component_ids[selected_ids]]
        ], axis=-1).astype(np.float32)
    else:
        meta = np.zeros((*selected_ids.shape, 2), dtype=np.float32)
    return np.concatenate([rel, selected_dist[..., None].astype(np.float32), meta], axis=-1)



class DynamicGeometryIndex:
    """Outcome-conditioned prototype retrieval in geometric pose space.

    Reference prototypes and model weights stay fixed. Only attempted target
    outcomes can change the top-k retrieved neighbors. Uses the same
    shortlist/reranking scheme as DynamicStructuralIndex.
    """
    def __init__(self, poses, memory, k, rotation_weight, mask=None,
                 candidates=64, strength=0.75, bandwidth=1.0):
        self.poses = np.asarray(poses, dtype=np.float32)
        self.ref = memory.prototypes if mask is None else memory.prototypes[mask]
        if len(self.ref) == 0:
            raise ValueError('Empty geometric reference set')
        self.k = int(k)
        self.strength = float(strength)
        self.bandwidth = float(bandwidth)
        if self.bandwidth <= 0:
            raise ValueError('bandwidth must be positive')
        self.rotation_weight = rotation_weight
        scale = np.array([1, 1, 1, rotation_weight, rotation_weight, rotation_weight], dtype=np.float32)
        tree = cKDTree(self.ref * scale)
        self.d, self.ids = tree.query(self.poses * scale,
            k=min(len(self.ref), max(self.k, int(candidates))), workers=1)
        self.d = np.asarray(self.d).reshape(len(poses), -1)
        self.ids = np.asarray(self.ids).reshape(len(poses), -1)

    def features(self, good=(), bad=(), query_ids=None):
        support = np.zeros(len(self.ref), dtype=np.float32)
        for observed, sign in ((good, 1.0), (bad, -1.0)):
            observed = np.asarray(observed, dtype=int)
            if not len(observed):
                continue
            d, ids = self.d[observed], self.ids[observed]
            weight = np.exp(-0.5 * (d / self.bandwidth) ** 2)
            np.add.at(support, ids.ravel(), (sign * weight / len(observed)).ravel())
        if np.any(support):
            support /= max(float(np.max(np.abs(support))), 1e-8)
        qi = np.arange(len(self.poses)) if query_ids is None else np.asarray(query_ids, dtype=int)
        ids, d = self.ids[qi], self.d[qi]
        adjusted = d - self.strength * support[ids]
        chosen = np.argsort(adjusted, axis=1, kind='stable')[:, :min(self.k, ids.shape[1])]
        selected = np.take_along_axis(ids, chosen, axis=1)
        original_d = np.take_along_axis(d, chosen, axis=1)
        rel = (self.ref[selected] - self.poses[qi, None, :]).astype(np.float32)
        rel[:, :, 3:] *= self.rotation_weight
        # Match the original geometric encoder's 9 channels: 6 pose offsets,
        # geometric distance, and 2 zero metadata channels.
        out = np.concatenate([rel, original_d[..., None].astype(np.float32),
                              np.zeros((*selected.shape, 2), dtype=np.float32)], axis=-1)
        if out.shape[1] < self.k:
            out = np.concatenate([out, np.repeat(out[:, -1:, :], self.k - out.shape[1], axis=1)], axis=1)
        return out.astype(np.float32)

def context_features(poses, good, bad, rot_weight):
    """Target context, independent of reference memory; no hidden labels."""
    n = len(poses)
    scale = np.array([1,1,1,rot_weight,rot_weight,rot_weight],dtype=np.float32)
    z = poses * scale
    vals = []
    for indices in (good, bad):
        if len(indices):
            d = cKDTree(z[np.asarray(indices)]).query(z, workers=1)[0]
            vals.append(np.exp(-0.5*(d / 0.5)**2).astype('float32'))
        else:
            vals.append(np.zeros(n,dtype='float32'))
    vals += [np.full(n,len(good)/(len(good)+len(bad)+1),dtype='float32'),
             np.full(n,min((len(good)+len(bad))/20,1),dtype='float32')]
    return np.stack(vals,axis=1)


class MemoryPredictor(nn.Module):
    def __init__(self, hidden=64, use_memory=True):
        super().__init__()
        self.use_memory=use_memory
        self.memory_encoder=nn.Sequential(nn.Linear(9,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(9+4+(hidden if use_memory else 0),hidden),nn.ReLU(),
                                nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
    def forward(self, pose, ctx, mem):
        xs=[pose,ctx]
        if self.use_memory:
            # DeepSets-style permutation-invariant local memory encoder
            xs.append(self.memory_encoder(mem).mean(dim=1))
        return self.head(torch.cat(xs,dim=-1)).squeeze(-1)


def sample_contexts(obj, query_ids, rng, max_context, rot_weight, variable=True):
    """Draw observed grasps independently of each query label."""
    n = len(obj.labels)
    out = []
    for qi in query_ids:
        avail = np.arange(n)
        avail = avail[avail != qi]
        cap = min(max_context, len(avail))
        count = int(rng.integers(0, cap + 1)) if variable else cap
        chosen = rng.choice(avail, count, replace=False)
        if count == 0:
            out.append(context_features(obj.pose[qi:qi + 1], [], [], rot_weight)[0])
        else:
            joined = np.vstack((obj.pose[qi:qi + 1], obj.pose[chosen]))
            good = (np.flatnonzero(obj.labels[chosen]) + 1).tolist()
            bad = (np.flatnonzero(~obj.labels[chosen]) + 1).tolist()
            out.append(context_features(joined, good, bad, rot_weight)[0])
    return np.asarray(out, dtype=np.float32)


def dynamic_training_context(obj, query_ids, rng, max_context, rot_weight, variable=True):
    """One causal-like partial observation history per training object/epoch.

    The held-out query labels never enter the observed history. This allows
    consistent dynamic reference retrieval and context for every query.
    """
    available = np.setdiff1d(np.arange(len(obj.labels)), query_ids)
    cap = min(max_context, len(available))
    count = int(rng.integers(0, cap + 1)) if variable else cap
    observed = rng.choice(available, count, replace=False)
    good = observed[obj.labels[observed]].tolist()
    bad = observed[~obj.labels[observed]].tolist()
    return good, bad, context_features(obj.pose, good, bad, rot_weight)[query_ids]


class FeatureScaler:
    """Channelwise train-only standardization for pose and local memory."""
    def fit(self, samples):
        poses = np.concatenate([x[0] for x in samples], axis=0)
        memories = np.concatenate([x[1].reshape(-1, 9) for x in samples], axis=0)
        self.p_mean = poses.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.p_std = np.maximum(poses.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32)
        self.m_mean = memories.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.m_std = np.maximum(memories.std(axis=0, dtype=np.float64), 1e-4).astype(np.float32)
        return self

    def pose(self, p):
        return np.asarray((p - self.p_mean) / self.p_std, dtype=np.float32)

    def memory(self, m):
        return np.asarray((m - self.m_mean) / self.m_std, dtype=np.float32)


def fit_model(objects, memory, k, rot_weight, connectivity, target_only, args, seed,
              val_objects=None, prototype_owners=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(args.device)
    model = MemoryPredictor(args.hidden, not target_only).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Adaptive protocol: explicit validation objects, all references used for training.
    # Legacy protocol remains available when val_objects is None.
    if val_objects is None:
        order = rng.permutation(len(objects))
        n_val = max(1, int(round(len(objects) * args.val_fraction)))
        if n_val >= len(objects):
            raise ValueError('Need at least one training and one validation object')
        val_ids = set(order[:n_val].tolist())
        training_pairs = [(i, obj) for i, obj in enumerate(objects) if i not in val_ids]
        validation_pairs = [(i, obj) for i, obj in enumerate(objects) if i in val_ids]
    else:
        training_pairs = list(enumerate(objects))
        validation_pairs = [(None, obj) for obj in val_objects]
        if not validation_pairs:
            raise ValueError('Explicit validation set is empty')

    def make_item(obj, owner):
        query_ids = rng.choice(len(obj.labels), min(args.train_queries, len(obj.labels)), replace=False)
        pose = pose_features(obj.pose)[query_ids]
        if owner is not None and prototype_owners is not None:
            # Leave-one-object-out retrieval. Do not expose own positive labels via memory.
            keep = prototype_owners != owner
            if not np.any(keep):
                raise ValueError(f'No other-object prototypes for training object {owner}')
            proto = memory.prototypes[keep]
            component_ids = memory.component_ids[keep]
            weights = memory.weights
            if connectivity:
                # Re-normalize the retained component weights, preserving the global scheme.
                active = np.unique(component_ids)
                weights = weights.copy()
                weights[active] /= weights[active].sum()
        else:
            proto, component_ids, weights = memory.prototypes, memory.component_ids, memory.weights
        index = None
        if connectivity or args.dynamic_geometry:
            # Structural signatures use the full UNLABELED candidate graph;
            # mask the query object's own reference prototypes during training.
            index_class = DynamicStructuralIndex if connectivity else DynamicGeometryIndex
            index = index_class(
                obj.pose, memory, k, rot_weight,
                keep if owner is not None and prototype_owners is not None else None,
                candidates=args.dynamic_candidates, strength=args.dynamic_strength,
                bandwidth=args.dynamic_bandwidth)
            mem = index.features(query_ids=query_ids)
        else:
            mem = build_memory_features(obj.pose[query_ids], proto,
                                        component_ids, weights, memory.bandwidths,
                                        k, rot_weight, False)
        return (pose, mem, obj.labels[query_ids].astype(np.float32), obj, query_ids, index)

    train_cache = [make_item(obj, owner) for owner, obj in training_pairs]
    val_cache = [make_item(obj, None) for _, obj in validation_pairs]
    scaler = FeatureScaler().fit([(p, m) for p, m, _, _, _, _ in train_cache])
    def scaled_cache(cache):
        return [(scaler.pose(p), scaler.memory(m), y, obj, q, index)
                for p, m, y, obj, q, index in cache]
    train_cache = scaled_cache(train_cache)
    val_cache = scaled_cache(val_cache)
    print(f'training objects={len(train_cache)} validation objects={len(val_cache)} '
          f'variable_context={args.variable_context} max_context={args.train_context}', flush=True)

    # Fixed validation contexts make epoch-to-epoch validation comparable.
    val_rng = np.random.default_rng(seed + 1000003)
    val_contexts = [
        dynamic_training_context(obj, q, val_rng, args.train_context, rot_weight,
                                 args.variable_context)
        if connectivity or args.dynamic_geometry else sample_contexts(obj, q, val_rng, args.train_context,
                                             rot_weight, args.variable_context)
        for _, _, _, obj, q, _ in val_cache]

    def evaluate():
        model.eval()
        total_loss = 0.0
        total_n = 0
        with torch.no_grad():
            for (p, m, y, _, q, index), ctx in zip(val_cache, val_contexts):
                if connectivity or args.dynamic_geometry:
                    good, bad, c = ctx
                    m = scaler.memory(index.features(good, bad, query_ids=q))
                else:
                    c = ctx
                for start in range(0, len(y), args.batch_size):
                    end = start + args.batch_size
                    logits = model(torch.from_numpy(p[start:end]).to(device),
                                   torch.from_numpy(c[start:end]).to(device),
                                   torch.from_numpy(m[start:end]).to(device))
                    yt = torch.from_numpy(y[start:end]).to(device)
                    total_loss += F.binary_cross_entropy_with_logits(logits, yt, reduction='sum').item()
                    total_n += len(yt)
        return total_loss / max(total_n, 1)

    best_val, best_epoch, best_state, stale = float('inf'), 0, None, 0
    for epoch in range(args.epochs):
        model.train()
        train_sum = train_n = 0
        for j in rng.permutation(len(train_cache)):
            p, m, y, obj, query_ids, index = train_cache[j]
            # Match training and inference: observed labels update structural kNN.
            if connectivity or args.dynamic_geometry:
                good, bad, c = dynamic_training_context(
                    obj, query_ids, rng, args.train_context, rot_weight,
                    args.variable_context)
                m = scaler.memory(index.features(good, bad, query_ids=query_ids))
            else:
                c = sample_contexts(obj, query_ids, rng, args.train_context,
                                    rot_weight, args.variable_context)
            ids_all = rng.permutation(len(y))
            for start in range(0, len(y), args.batch_size):
                ids = ids_all[start:start + args.batch_size]
                pt = torch.from_numpy(p[ids]).to(device)
                ct = torch.from_numpy(c[ids]).to(device)
                mt = torch.from_numpy(m[ids]).to(device)
                yt = torch.from_numpy(y[ids]).to(device)
                logits = model(pt, ct, mt)
                loss = F.binary_cross_entropy_with_logits(logits, yt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_sum += loss.item() * len(ids)
                train_n += len(ids)
        val_loss = evaluate()
        improved = val_loss < best_val - args.min_delta
        if improved:
            best_val, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        print(f'train model={"connectivity" if connectivity else "geometry"} epoch={epoch + 1}/{args.epochs} '
              f'loss={train_sum / max(train_n, 1):.4f} val_loss={val_loss:.4f} '
              f'best_epoch={best_epoch}' + (' *' if improved else ''), flush=True)
        if args.patience > 0 and stale >= args.patience:
            print(f'early stopping after {epoch + 1} epochs', flush=True)
            break
    model.load_state_dict(best_state)
    model.eval()
    print(f'restored best validation model epoch={best_epoch} val_loss={best_val:.4f}', flush=True)
    return model, scaler


@torch.no_grad()
def predict(model, obj, memory, k, rot_weight, good, bad, batch_size, device, connectivity):
    p=pose_features(obj.pose)
    c=context_features(obj.pose,good,bad,rot_weight)
    m=(structural_memory_features(obj.pose, memory, k, rot_weight)
       if connectivity else build_memory_features(obj.pose,memory.prototypes,memory.component_ids,
                                                  memory.weights,memory.bandwidths,k,rot_weight,False))
    scores=[]
    for start in range(0,len(p),batch_size):
        end=start+batch_size
        logits=model(torch.from_numpy(p[start:end]).to(device),
                     torch.from_numpy(c[start:end]).to(device),
                     torch.from_numpy(m[start:end]).to(device))
        scores.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(scores)


def explore_model(obj, model, memory, args, seed, connectivity, scaler):
    rng=np.random.default_rng(seed)
    n=len(obj.labels); seen=np.zeros(n,dtype=bool); good=[];bad=[];records=[]; successes=0
    # Cache unlabeled signatures; structural reference matches update online
    p=scaler.pose(pose_features(obj.pose))
    if connectivity or args.dynamic_geometry:
        index_class = DynamicStructuralIndex if connectivity else DynamicGeometryIndex
        dynamic_index = index_class(
            obj.pose, memory, args.neighbors, args.rotation_weight,
            candidates=args.dynamic_candidates, strength=args.dynamic_strength,
            bandwidth=args.dynamic_bandwidth)
        mem = dynamic_index.features()
    else:
        mem = build_memory_features(obj.pose,memory.prototypes,memory.component_ids,
                                    memory.weights,memory.bandwidths,args.neighbors,
                                    args.rotation_weight,False)
    mem=scaler.memory(mem)
    device=torch.device(args.device)
    with torch.no_grad():
        pt=torch.from_numpy(p).to(device); mt=torch.from_numpy(mem).to(device)
        for t in range(min(args.budget,n)):
            if (connectivity or args.dynamic_geometry) and (good or bad):
                mt = torch.from_numpy(scaler.memory(
                    dynamic_index.features(good, bad))).to(device)
            if args.target_conditioned_retrieval and not connectivity and not args.dynamic_geometry and (good or bad):
                dynamic_mem = target_conditioned_memory_features(
                    obj.pose, memory, args.neighbors, args.rotation_weight,
                    good, bad, connectivity,
                    args.retrieval_strength, args.retrieval_candidates,
                    args.retrieval_bandwidth)
                mt = torch.from_numpy(scaler.memory(dynamic_mem)).to(device)
            ctx=context_features(obj.pose,good,bad,args.rotation_weight)
            ct=torch.from_numpy(ctx).to(device)
            pieces=[]
            for start in range(0,n,args.batch_size):
                end=start+args.batch_size
                pieces.append(torch.sigmoid(model(pt[start:end],ct[start:end],mt[start:end])).cpu().numpy())
            score=np.concatenate(pieces).astype('float64')
            score+=rng.uniform(0,1e-10,n);score[seen]=-np.inf
            chosen=int(np.argmax(score));seen[chosen]=True
            y=bool(obj.labels[chosen]);(good if y else bad).append(chosen)
            successes+=int(y)
            records.append({'step':t+1,'success':int(y),'cumulative_success':successes})
    return records


def sync_device(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()



def geometry_control_memories(reference_objects, prototype_count, rotation_weight, seed, per_object_counts=None):
    """Unlabeled-metadata geometric controls drawn from identical successful reference poses.

    Random and FPS have exactly prototype_count prototypes, matching the region memory.
    FPS uses the existing scaled Euclidean pose metric (not an SO(3) geodesic).
    """
    from .tc_cart import RegionMemory
    if per_object_counts is not None:
        # Match the connectivity memory's per-object counts, not only its total.
        random_parts, fps_parts = [], []
        for i, (obj, count) in enumerate(zip(reference_objects, per_object_counts)):
            if count == 0:
                continue
            r, f = geometry_control_memories([obj], count, rotation_weight, seed + i)
            random_parts.append(r.prototypes)
            fps_parts.append(f.prototypes)
        def wrap(protos):
            return RegionMemory(prototypes=protos, weights=np.ones(1),
                                bandwidths=np.ones(1), component_ids=np.zeros(len(protos), dtype=int))
        return wrap(np.concatenate(random_parts)), wrap(np.concatenate(fps_parts))
    good = [obj.pose[obj.labels] for obj in reference_objects if np.any(obj.labels)]
    if not good:
        raise ValueError('No successful reference grasps for geometry controls')
    pool = np.concatenate(good, axis=0)
    k = int(prototype_count)
    if len(pool) < k:
        raise ValueError(f'Only {len(pool)} positive poses available for {k} prototypes')
    rng = np.random.default_rng(seed + 104729)
    # A deterministic common candidate pool keeps FPS tractable at larger scales.
    cap = min(len(pool), max(4*k, 20000))
    candidates = pool[rng.choice(len(pool), cap, replace=False)]
    random_prototypes = candidates[rng.choice(cap, k, replace=False)].copy()
    scaled = candidates.astype(np.float64).copy()
    scaled[:, 3:] *= rotation_weight
    chosen = np.empty(k, dtype=np.int64)
    chosen[0] = int(rng.integers(cap))
    mind2 = np.full(cap, np.inf)
    for i in range(k):
        p = scaled[chosen[i]]
        dist2 = np.einsum('ij,ij->i', scaled-p, scaled-p)
        np.minimum(mind2, dist2, out=mind2)
        mind2[chosen[:i+1]] = -np.inf
        if i+1 < k:
            chosen[i+1] = int(np.argmax(mind2))
    fps_prototypes = candidates[chosen].copy()
    def wrap(protos):
        return RegionMemory(prototypes=protos, weights=np.ones(1, dtype=float),
                            bandwidths=np.ones(1, dtype=float),
                            component_ids=np.zeros(k, dtype=int))
    return wrap(random_prototypes), wrap(fps_prototypes)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--grasps-dir',default='grasps');ap.add_argument('--out',default='results/ca_random_connectivity.csv')
    ap.add_argument('--references',type=int,default=200);ap.add_argument('--reference-pool',type=int,default=None,help='Fixed reference pool capacity across scales; seed selects nested subsets');ap.add_argument('--train-objects',type=int,default=300)
    ap.add_argument('--targets',type=int,default=10,help='Target objects per seed, paired across methods; may overlap between seeds')
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    ap.add_argument('--budget',type=int,default=100);ap.add_argument('--max-grasps',type=int,default=1000)
    ap.add_argument('--memory-budget',type=int,default=3000);ap.add_argument('--prototypes-per-object',type=int,default=None,help='Maximum prototypes per reference object; overrides global memory budget');ap.add_argument('--medoids',type=int,default=8)
    ap.add_argument('--eps',type=float,default=0.5);ap.add_argument('--min-region-size',type=int,default=2)
    ap.add_argument('--rotation-weight',type=float,default=0.5);ap.add_argument('--bandwidth',type=float,default=0.5)
    ap.add_argument('--alpha',type=float,default=0.5)
    ap.add_argument('--neighbors',type=int,default=16);ap.add_argument('--hidden',type=int,default=64)
    ap.add_argument('--target-conditioned-retrieval',action='store_true',
                    help='Use observed target successes/failures to re-rank reference-region neighbors at test time')
    ap.add_argument('--retrieval-strength',type=float,default=0.5,
                    help='Strength of target-region evidence in distance units')
    ap.add_argument('--retrieval-candidates',type=int,default=64,
                    help='Geometric shortlist size before contextual re-ranking')
    ap.add_argument('--retrieval-bandwidth',type=float,default=0.5)
    ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--train-queries',type=int,default=128)
    ap.add_argument('--train-context',type=int,default=99);ap.add_argument('--batch-size',type=int,default=256)
    ap.add_argument('--lr',type=float,default=1e-3);ap.add_argument('--device',default='cpu')
    ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--variable-context',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--val-fraction',type=float,default=0.2,help='Adaptive: extra validation objects as fraction of reference count; legacy: training holdout')
    ap.add_argument('--training-mode',choices=['separate','reference'],default='separate',help='reference: train on all references, validate on extra disjoint objects')
    ap.add_argument('--patience',type=int,default=8,help='Early stopping epochs without validation improvement; 0 disables')
    ap.add_argument('--min-delta',type=float,default=0.0,help='Minimum validation improvement')
    ap.add_argument('--split-seed',type=int,default=4242)
    ap.add_argument('--dynamic-geometry', action='store_true', help='Train and evaluate all three geometric encoders with outcome-conditioned prototype reranking')
    ap.add_argument('--dynamic-strength', type=float, default=0.75)
    ap.add_argument('--dynamic-bandwidth', type=float, default=1.0)
    ap.add_argument('--dynamic-candidates', type=int, default=64)
    args=ap.parse_args()
    if args.references<1 or args.train_objects<1 or args.targets<1 or args.neighbors<1:
        ap.error('references, train-objects, targets and neighbors must be positive')
    if not 0 < args.val_fraction < 1:ap.error('val-fraction must be between 0 and 1')
    if args.train_context < 0:ap.error('train-context must be nonnegative')
    if args.training_mode == 'separate' and args.train_objects < 2:ap.error('Need at least 2 training objects for validation split')
    if args.device=='mps' and not torch.backends.mps.is_available():ap.error('MPS unavailable')
    paths=sorted(Path(args.grasps_dir).rglob('*.hdf5'))
    if not paths: paths=sorted(Path(args.grasps_dir).rglob('*.h5'))
    pool=args.reference_pool if args.reference_pool is not None else args.references
    if pool < args.references: ap.error('reference-pool must be >= references')
    if len(args.seeds) != 1:
        ap.error('Reference-scaling independent-split protocol requires exactly one --seeds value per invocation')
    seed = args.seeds[0]
    # Keep evaluation pool fixed across reference scales and seeds.
    # In adaptive mode, reserve a validation pool large enough for the largest R.
    n_val = int(np.ceil(args.val_fraction * args.references))
    validation_capacity = int(np.ceil(args.val_fraction * pool))
    needed = pool + (validation_capacity if args.training_mode == 'reference' else args.train_objects) + args.targets
    if len(paths) < needed:
        ap.error(f'Need >= {needed} objects for disjoint pools, found {len(paths)}')
    base_rng = np.random.default_rng(args.split_seed)
    paths = [paths[i] for i in base_rng.permutation(len(paths))]
    reference_pool = paths[:pool]
    if args.training_mode == 'reference':
        validation_pool = paths[pool:pool + validation_capacity]
        evaluation_pool = paths[pool + validation_capacity:]
        trains = None
    else:
        trains = paths[pool:pool + args.train_objects]
        validation_pool = None
        evaluation_pool = paths[pool + args.train_objects:]
    ref_rng = np.random.default_rng(np.random.SeedSequence([args.split_seed, seed, 101]))
    ref_order = ref_rng.permutation(len(reference_pool))
    refs = [reference_pool[i] for i in ref_order[:args.references]]
    if args.training_mode == 'reference':
        val_rng = np.random.default_rng(np.random.SeedSequence([args.split_seed, seed, 303]))
        val_order = val_rng.permutation(len(validation_pool))
        vals = [validation_pool[i] for i in val_order[:n_val]]
        trains = refs
    else:
        vals = []
    test_rng = np.random.default_rng(np.random.SeedSequence([args.split_seed, seed, 202]))
    test_ids = test_rng.choice(len(evaluation_pool), args.targets, replace=False)
    tests = [evaluation_pool[i] for i in test_ids]
    assert len(set(refs) & set(vals)) == 0
    assert len(set(vals) & set(tests)) == 0
    assert len(set(refs) & set(tests)) == 0
    assert len(set(trains) & set(tests)) == 0
    if args.training_mode == 'separate':
        assert len(set(refs) & set(trains)) == 0
    print(f'mode={args.training_mode} reference_pool={pool} train={len(trains)} '
          f'validation={len(vals)} evaluation_pool={len(evaluation_pool)} '
          f'seed={seed} references={len(refs)} targets={len(tests)}', flush=True)

    def grasp_seed(path, namespace):
        # Stable per-object subsampling across reference scales and methods.
        digest = hashlib.blake2b(str(path).encode('utf-8'), digest_size=8).digest()
        return int.from_bytes(digest, 'little') % (2**32) ^ namespace

    preparation_start=time.perf_counter()
    ref_objs=[load_object(p,args.max_grasps,grasp_seed(p, 4242)) for p in refs]
    memory_start=time.perf_counter()
    memory,points=reference_memory(ref_objs,args.eps,args.min_region_size,args.medoids,
                                   args.rotation_weight,seed,args.memory_budget,
                                   prototypes_per_object=args.prototypes_per_object)
    memory_build_sec=time.perf_counter()-memory_start
    print(f'memory prototypes={len(memory.prototypes)} regions={len(memory.weights)}',flush=True)
    # Each reference contributes at most the requested number of prototypes.
    per_object_counts = None
    if args.prototypes_per_object is not None:
        per_object_counts = []
        for oi, obj in enumerate(ref_objs):
            if not np.any(obj.labels):
                per_object_counts.append(0)
                continue
            try:
                local, _ = reference_memory([obj], args.eps, args.min_region_size,
                                            args.medoids, args.rotation_weight,
                                            seed + oi, args.prototypes_per_object)
                per_object_counts.append(len(local.prototypes))
            except ValueError as exc:
                if 'No viable reference regions' not in str(exc):
                    raise
                per_object_counts.append(0)
        assert sum(per_object_counts) == len(memory.prototypes)
        print(f'prototypes_per_object={args.prototypes_per_object} '
              f'actual_prototypes={len(memory.prototypes)} '
              f'objects_with_prototypes={sum(c > 0 for c in per_object_counts)}', flush=True)
    random_geometry_memory, fps_geometry_memory = geometry_control_memories(
        ref_objs, len(memory.prototypes), args.rotation_weight, seed,
        per_object_counts=per_object_counts)
    print(f'geometry controls prototypes={len(random_geometry_memory.prototypes)} '
          f'fps_prototypes={len(fps_geometry_memory.prototypes)}', flush=True)
    # Reference signatures are attached only to the connectivity memory;
    # geometric control memories and algorithms are untouched.
    if args.training_mode == 'reference':
        signature_owners = np.repeat(np.arange(len(ref_objs)), per_object_counts)
    else:
        # Derive exact prototype provenance from the reference memory build.
        # With no per-object cap, reconstruct by repeating the same deterministic
        # per-object region builder only if needed; see requirement below.
        signature_owners = None
    if signature_owners is None:
        raise ValueError('Structural retrieval currently requires --training-mode reference '
                         'and --prototypes-per-object for exact prototype provenance')
    attach_reference_signatures(memory, ref_objs, signature_owners, args.rotation_weight)
    if args.target_conditioned_retrieval and not args.dynamic_geometry:
        raise ValueError('--target-conditioned-retrieval is test-only; use --dynamic-geometry for train/test-matched geometric baselines')
    if args.target_conditioned_retrieval:
        print('NOTE: target-conditioned geometric reranking is not applied to the '
              'structural connectivity encoder; geometry controls retain their behavior.', flush=True)
    if args.training_mode == 'reference':
        train_objs = ref_objs  # identical grasp subsampling, not reloaded
        val_objs = [load_object(p,args.max_grasps,grasp_seed(p, 14242)) for p in vals]
        if args.prototypes_per_object is None:
            raise ValueError('Adaptive training requires --prototypes-per-object for exact object provenance')
        prototype_owners = np.repeat(np.arange(len(ref_objs)), per_object_counts)
        assert len(prototype_owners) == len(memory.prototypes)
    else:
        train_objs=[load_object(p,args.max_grasps,grasp_seed(p, 14242)) for p in trains]
        val_objs = None
        prototype_owners = None
    preparation_sec=time.perf_counter()-preparation_start
    # All four learned encoders vary their training initialization by seed.
    training_seed = seed
    print(f'INDEPENDENT TRAINING seed={training_seed}', flush=True)
    print('=== Training connectivity encoder ===', flush=True)
    sync_device(args.device)
    train_start=time.perf_counter()
    model, scaler = fit_model(train_objs,memory,args.neighbors,args.rotation_weight,
                              True,False,args,training_seed,
                              val_objects=val_objs, prototype_owners=prototype_owners)
    sync_device(args.device)
    connectivity_train_sec=time.perf_counter()-train_start
    print('=== Training geometry-only encoder ===', flush=True)
    train_start=time.perf_counter()
    geometry_model, geometry_scaler = fit_model(train_objs,memory,args.neighbors,args.rotation_weight,
                                                False,False,args,training_seed,
                                                val_objects=val_objs, prototype_owners=prototype_owners)
    sync_device(args.device)
    geometry_train_sec=time.perf_counter()-train_start
    print('=== Training random-prototype geometry encoder ===', flush=True)
    sync_device(args.device)
    train_start=time.perf_counter()
    random_geometry_model, random_geometry_scaler = fit_model(
        train_objs, random_geometry_memory, args.neighbors, args.rotation_weight,
        False, False, args, training_seed,
        val_objects=val_objs, prototype_owners=prototype_owners)
    sync_device(args.device)
    random_geometry_train_sec=time.perf_counter()-train_start
    print('=== Training FPS-prototype geometry encoder ===', flush=True)
    sync_device(args.device)
    train_start=time.perf_counter()
    fps_geometry_model, fps_geometry_scaler = fit_model(
        train_objs, fps_geometry_memory, args.neighbors, args.rotation_weight,
        False, False, args, training_seed,
        val_objects=val_objs, prototype_owners=prototype_owners)
    sync_device(args.device)
    fps_geometry_train_sec=time.perf_counter()-train_start
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    timing_rows=[]
    timing_path=Path(args.out).with_name(Path(args.out).stem+"_timing.csv")
    print(f"TIMING memory_build_sec={memory_build_sec:.6f} reference_and_training_preparation_sec={preparation_sec:.6f} connectivity_train_sec={connectivity_train_sec:.6f} geometry_train_sec={geometry_train_sec:.6f}",flush=True)
    with open(args.out,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','target','method','step','success','cumulative_success','runtime_sec'])
        writer.writeheader()
        for si,seed in enumerate(args.seeds):
            for ti in range(args.targets):
                idx=si*args.targets+ti;obj=load_object(tests[idx],args.max_grasps,grasp_seed(tests[idx], 24242))
                for method in ('random','target_ca','random_geometry_encoder','fps_geometry_encoder','geometry_encoder','connectivity_encoder'):
                    sync_device(args.device)
                    start=time.perf_counter()
                    if method=='connectivity_encoder':
                        rows=explore_model(obj,model,memory,args,seed,True,scaler)
                    elif method=='random_geometry_encoder':
                        rows=explore_model(obj,random_geometry_model,random_geometry_memory,args,seed,False,random_geometry_scaler)
                    elif method=='fps_geometry_encoder':
                        rows=explore_model(obj,fps_geometry_model,fps_geometry_memory,args,seed,False,fps_geometry_scaler)
                    elif method=='geometry_encoder':
                        rows=explore_model(obj,geometry_model,memory,args,seed,False,geometry_scaler)
                    else:
                        prior=np.zeros(len(obj.labels),dtype=np.float32)
                        rows=explore_legacy(obj,prior,method,args.budget,seed,
                                            args.bandwidth,args.rotation_weight,args.alpha)
                    sync_device(args.device)
                    elapsed=time.perf_counter()-start
                    steps=len(rows)
                    training_sec={
                        'connectivity_encoder':connectivity_train_sec,
                        'geometry_encoder':geometry_train_sec,
                        'random_geometry_encoder':random_geometry_train_sec,
                        'fps_geometry_encoder':fps_geometry_train_sec,
                    }.get(method,0.0)
                    output_method = ('dynamic_' + method if args.dynamic_geometry and method in ('random_geometry_encoder','fps_geometry_encoder','geometry_encoder') else method)
                    timing_rows.append(dict(seed=seed,target=str(tests[idx]),method=output_method,reference_objects=args.references,training_objects=len(train_objs),validation_objects=(len(val_objs) if val_objs is not None else max(1,round(len(train_objs)*args.val_fraction))),training_mode=args.training_mode,memory_budget=(args.references * args.prototypes_per_object if args.prototypes_per_object is not None else args.memory_budget),prototypes_per_object=args.prototypes_per_object,actual_prototypes=len(memory.prototypes),connected_regions=len(memory.weights),memory_build_sec=memory_build_sec,preparation_sec=preparation_sec,training_sec=training_sec,inference_episode_sec=elapsed,inference_step_ms=1000*elapsed/max(steps,1),steps=steps))
                    for row in rows:writer.writerow({'seed':seed,'target':str(tests[idx]),'method':output_method,**row,'runtime_sec':elapsed})
                    f.flush()
                    print(f'seed={seed} target={ti+1}/{args.targets} method={output_method} success={rows[-1]["cumulative_success"]} sec={elapsed:.2f}',flush=True)
    with timing_path.open('w',newline='') as tf:
        fields=list(timing_rows[0])
        tw=csv.DictWriter(tf,fieldnames=fields)
        tw.writeheader();tw.writerows(timing_rows)
    print('Saved:',args.out,'timing:',timing_path)

if __name__=='__main__':main()
