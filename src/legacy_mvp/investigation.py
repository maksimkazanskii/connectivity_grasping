"""
Experiment 0: Inspect ACRONYM grasp data.

What this script does
---------------------
1. Finds ACRONYM .h5/.hdf5 files recursively.
2. Randomly selects at most MAX_OBJECTS objects.
3. Loads grasp transforms and success labels.
4. Reports:
       - total candidate grasps
       - successful grasps
       - failed grasps
       - success fraction
       - translation range
5. Saves a CSV summary.

No topology is computed yet.

Usage
-----
python src/01_inspect_acronym.py /path/to/acronym/data

Example
-------
python src/01_inspect_acronym.py ~/grasp_topology/acronym_data
"""

from pathlib import Path
import argparse

import h5py
import numpy as np
import pandas as pd


MAX_OBJECTS = 100
SEED = 42


def find_h5_files(root: Path):
    """Find all HDF5 files recursively."""
    files = list(root.rglob("*.h5"))
    files += list(root.rglob("*.hdf5"))
    return sorted(set(files))


def inspect_h5_structure(path: Path):
    """Print datasets contained in one HDF5 file."""
    print(f"\nStructure of: {path}")

    with h5py.File(path, "r") as f:

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(
                    f"  {name:40s} "
                    f"shape={str(obj.shape):20s} "
                    f"dtype={obj.dtype}"
                )

        f.visititems(visitor)


def find_dataset(f, candidates):
    """
    Find a dataset using several possible paths.

    This makes the script a little more tolerant to minor
    differences between ACRONYM releases.
    """
    for name in candidates:
        if name in f:
            return np.asarray(f[name])

    return None


def load_object(path: Path):
    """Load grasp transforms and success labels."""

    with h5py.File(path, "r") as f:

        transforms = find_dataset(
            f,
            [
                "grasps/transforms",
                "/grasps/transforms",
                "transforms",
            ],
        )

        success = find_dataset(
            f,
            [
                "grasps/qualities/flex/object_in_gripper",
                "/grasps/qualities/flex/object_in_gripper",
                "qualities/flex/object_in_gripper",
                "success",
            ],
        )

    if transforms is None:
        raise KeyError("Could not find grasp transforms.")

    if success is None:
        raise KeyError("Could not find success labels.")

    transforms = np.asarray(transforms)
    success = np.asarray(success).reshape(-1)

    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected transforms with shape (N,4,4), "
            f"got {transforms.shape}"
        )

    if len(transforms) != len(success):
        raise ValueError(
            f"Transforms ({len(transforms)}) and labels "
            f"({len(success)}) have different lengths."
        )

    return transforms, success


def summarize_object(path: Path):
    """Compute simple statistics for one object."""

    transforms, success = load_object(path)

    success_mask = success > 0

    n_total = len(success)
    n_success = int(success_mask.sum())
    n_failed = n_total - n_success

    translations = transforms[:, :3, 3]

    xyz_min = translations.min(axis=0)
    xyz_max = translations.max(axis=0)
    xyz_range = xyz_max - xyz_min

    return {
        "file": path.name,
        "path": str(path),
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "success_fraction": n_success / n_total if n_total else np.nan,
        "x_range": xyz_range[0],
        "y_range": xyz_range[1],
        "z_range": xyz_range[2],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing extracted ACRONYM files.",
    )

    parser.add_argument(
        "--max_objects",
        type=int,
        default=MAX_OBJECTS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(root)

    # ------------------------------------------------------------
    # Find files
    # ------------------------------------------------------------

    files = find_h5_files(root)

    print()
    print("=" * 70)
    print("ACRONYM DATA INSPECTION")
    print("=" * 70)
    print(f"Data directory : {root}")
    print(f"HDF5 files     : {len(files)}")

    if not files:
        print("\nNo .h5 or .hdf5 files found.")
        print("Check that acronym.tar.gz was extracted correctly.")
        return

    # ------------------------------------------------------------
    # Inspect one file first
    # ------------------------------------------------------------

    inspect_h5_structure(files[0])

    # ------------------------------------------------------------
    # Random object sample
    # ------------------------------------------------------------

    rng = np.random.default_rng(args.seed)

    n_select = min(args.max_objects, len(files))

    indices = rng.choice(
        len(files),
        size=n_select,
        replace=False,
    )

    selected = [files[i] for i in indices]

    print()
    print("=" * 70)
    print(f"ANALYZING {n_select} RANDOM OBJECTS")
    print("=" * 70)

    rows = []

    for i, path in enumerate(selected, start=1):

        try:
            row = summarize_object(path)
            rows.append(row)

            print(
                f"{i:3d}/{n_select} | "
                f"{path.name[:35]:35s} | "
                f"N={row['n_total']:5d} | "
                f"success={row['n_success']:5d} | "
                f"rate={row['success_fraction']:.3f}"
            )

        except Exception as e:
            print(
                f"{i:3d}/{n_select} | "
                f"{path.name[:35]:35s} | "
                f"ERROR: {e}"
            )

    if not rows:
        print("\nNo objects could be loaded successfully.")
        return

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------
    # Dataset summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Objects successfully read : {len(df)}")

    print()
    print("Candidate grasps/object")
    print(df["n_total"].describe())

    print()
    print("Successful grasps/object")
    print(df["n_success"].describe())

    print()
    print("Success fraction")
    print(df["success_fraction"].describe())

    # ------------------------------------------------------------
    # Important topology diagnostic
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("TOPOLOGY SAMPLE-SIZE CHECK")
    print("=" * 70)

    for threshold in [100, 250, 500, 1000]:
        n = int((df["n_success"] >= threshold).sum())

        print(
            f"Objects with >= {threshold:4d} successful grasps: "
            f"{n:3d}/{len(df)}"
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "acronym_object_summary.csv"

    df.to_csv(output_path, index=False)

    print()
    print(f"Saved: {output_path.resolve()}")

    # ------------------------------------------------------------
    # Show extremes
    # ------------------------------------------------------------

    print()
    print("Objects with fewest successful grasps:")
    print(
        df.nsmallest(5, "n_success")[
            ["file", "n_success", "success_fraction"]
        ].to_string(index=False)
    )

    print()
    print("Objects with most successful grasps:")
    print(
        df.nlargest(5, "n_success")[
            ["file", "n_success", "success_fraction"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()