from pathlib import Path
from itertools import combinations

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


GRASP_DIR = Path("grasps")
OUT = Path("results/e1/convergence")

N_OBJECTS = 100
SIZES = [10, 20, 50, 100, 200, 300]

# Sample sizes used for within-vs-between comparison.
COMPARE_SIZES = [50, 100, 300]

REPEATS = 5
EPS = np.linspace(0.02, 1.5, 100)
SEED = 42


def load_successes(path):
    with h5py.File(path, "r") as f:
        transforms = f["grasps/transforms"][:]
        labels = f[
            "grasps/qualities/flex/object_in_gripper"
        ][:].astype(int)

    return transforms[labels == 1]


def distance_matrix(transforms):
    p = transforms[:, :3, 3]
    R = transforms[:, :3, :3]

    dp = np.linalg.norm(
        p[:, None, :] - p[None, :, :],
        axis=-1,
        )

    # Keep identical to existing implementation.
    relative = np.einsum("aik,bil->abkl", R, R)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    dr = np.arccos(cosine)

    scale_p = np.median(dp[dp > 0])
    scale_r = np.median(dr[dr > 0])

    return np.sqrt(
        (dp / scale_p) ** 2
        + (dr / scale_r) ** 2
    ).astype(np.float32)


def signature(indices, D):
    sub = D[np.ix_(indices, indices)]
    n = len(indices)

    B = []
    L = []

    for eps in EPS:
        n_components, labels = connected_components(
            csr_matrix(sub <= eps),
            directed=False,
        )

        sizes = np.bincount(labels)

        B.append(n_components / n)
        L.append(sizes.max() / n)

    return np.asarray(B), np.asarray(L)


def signature_distance(a, b):
    B1, L1 = a
    B2, L2 = b

    return 0.5 * (
            np.mean(np.abs(B1 - B2))
            + np.mean(np.abs(L1 - L2))
    )


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Find eligible objects
    # ---------------------------------------------------------
    eligible = []

    for path in sorted(GRASP_DIR.rglob("*.h5")):
        transforms = load_successes(path)

        if len(transforms) >= max(SIZES):
            eligible.append(path)

    if len(eligible) < N_OBJECTS:
        raise RuntimeError(
            f"Only {len(eligible)} eligible objects found."
        )

    # Fixed random set of objects.
    chosen = rng.choice(
        len(eligible),
        size=N_OBJECTS,
        replace=False,
    )

    convergence_rows = []

    # signatures_by_n[N][object] -> list of 5 signatures
    signatures_by_n = {
        n: {} for n in COMPARE_SIZES
    }

    # ---------------------------------------------------------
    # Convergence
    # ---------------------------------------------------------
    for j, object_idx in enumerate(chosen, 1):
        path = eligible[object_idx]
        transforms = load_successes(path)

        # Compute normalization and pairwise distances once
        # for this object.
        D = distance_matrix(transforms)

        for n in SIZES:
            signatures = []

            for _ in range(REPEATS):
                indices = rng.choice(
                    len(transforms),
                    size=n,
                    replace=False,
                )

                signatures.append(
                    signature(indices, D)
                )

            # 5 replicates -> 10 pairwise comparisons.
            distances = [
                signature_distance(
                    signatures[a],
                    signatures[b],
                )
                for a, b in combinations(
                    range(REPEATS), 2
                )
            ]

            convergence_rows.append({
                "object": path.stem,
                "N": n,
                "mean_distance": np.mean(distances),
            })

            # Retain signatures needed for object-specificity
            # analysis.
            if n in COMPARE_SIZES:
                signatures_by_n[n][path.stem] = signatures

        print(
            f"[{j:3d}/{N_OBJECTS}] "
            f"{path.stem}"
        )

    # ---------------------------------------------------------
    # Within-object vs between-object comparison
    # ---------------------------------------------------------
    wb_rows = []

    for n in COMPARE_SIZES:

        signatures_n = signatures_by_n[n]
        objects = list(signatures_n)

        # -----------------------------------------------------
        # Within-object
        #
        # 100 objects x C(5,2) = 1000 distances per N
        # -----------------------------------------------------
        for obj in objects:
            sigs = signatures_n[obj]

            for r, s in combinations(
                    range(REPEATS), 2
            ):
                wb_rows.append({
                    "N": n,
                    "type": "Within object",
                    "object_a": obj,
                    "object_b": obj,
                    "replicate_a": r,
                    "replicate_b": s,
                    "distance": signature_distance(
                        sigs[r],
                        sigs[s],
                    ),
                })

        # -----------------------------------------------------
        # Between-object
        #
        # Compare corresponding replicates across object pairs.
        #
        # C(100,2) x 5 = 24,750 distances per N
        # -----------------------------------------------------
        for i, j in combinations(
                range(len(objects)), 2
        ):
            obj_i = objects[i]
            obj_j = objects[j]

            for r in range(REPEATS):
                wb_rows.append({
                    "N": n,
                    "type": "Between objects",
                    "object_a": obj_i,
                    "object_b": obj_j,
                    "replicate_a": r,
                    "replicate_b": r,
                    "distance": signature_distance(
                        signatures_n[obj_i][r],
                        signatures_n[obj_j][r],
                    ),
                })

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    convergence_df = pd.DataFrame(
        convergence_rows
    )

    wb_df = pd.DataFrame(
        wb_rows
    )

    convergence_df.to_csv(
        OUT / "e1_convergence.csv",
        index=False,
        )

    wb_df.to_csv(
        OUT / "e1_within_between.csv",
        index=False,
        )

    print(
        f"\nSaved: "
        f"{OUT / 'e1_convergence.csv'}"
    )

    print(
        f"Saved: "
        f"{OUT / 'e1_within_between.csv'}"
    )

    print("\nWithin/between rows:")
    print(
        wb_df.groupby(
            ["N", "type"]
        ).size()
    )


if __name__ == "__main__":
    main()