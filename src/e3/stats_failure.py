#!/usr/bin/env python3

from pathlib import Path
import argparse
import h5py


# 50, 100, 150, ..., 1000
THRESHOLDS = list(range(50, 1001, 50))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "grasps_dir",
        type=Path,
        nargs="?",
        default=Path("grasps"),
        help="Directory containing ACRONYM .h5 grasp files "
             "(default: grasps)",
    )

    args = parser.parse_args()

    files = sorted(
        args.grasps_dir.rglob("*.h5")
    )

    if not files:
        raise RuntimeError(
            f"No .h5 files found under {args.grasps_dir}"
        )

    stats = []

    print(
        f"Found {len(files)} object files\n"
    )

    # ============================================================
    # COUNT SUCCESS / FAILURE FOR EVERY OBJECT
    # ============================================================

    for path in files:

        try:
            with h5py.File(path, "r") as f:

                labels = f[
                    "grasps/qualities/flex/object_in_gripper"
                ][:]

            # ACRONYM definition:
            #   1 = successful grasp
            #   0 = failed grasp

            n_success = int(
                (labels == 1).sum()
            )

            n_failure = int(
                (labels == 0).sum()
            )

            n_other = (
                    len(labels)
                    - n_success
                    - n_failure
            )

            if n_other > 0:
                print(
                    f"WARNING {path.name}: "
                    f"{n_other} labels are neither 0 nor 1"
                )

            stats.append(
                (
                    path.stem,
                    n_success,
                    n_failure,
                )
            )

            print(
                f"{path.stem:50s} "
                f"success={n_success:5d} "
                f"failure={n_failure:5d} "
                f"total={len(labels):5d}"
            )

        except Exception as e:

            print(
                f"ERROR {path}: {e}"
            )

    # ============================================================
    # SUMMARY
    # ============================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "ELIGIBLE OBJECTS"
    )

    print(
        "=" * 75
    )

    print(
        f"{'N':>6} "
        f"{'success >= N':>15} "
        f"{'failure >= N':>15} "
        f"{'BOTH >= N':>15}"
    )

    for n in THRESHOLDS:

        enough_success = sum(
            s >= n
            for _, s, f in stats
        )

        enough_failure = sum(
            f >= n
            for _, s, f in stats
        )

        enough_both = sum(
            (s >= n) and (f >= n)
            for _, s, f in stats
        )

        print(
            f"{n:6d} "
            f"{enough_success:15d} "
            f"{enough_failure:15d} "
            f"{enough_both:15d}"
        )


if __name__ == "__main__":
    main()