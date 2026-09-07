# e2/experiment.py

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from src.utils.grasp_io import (
    find_objects,
    load_successful_grasps,
)

from src.utils.geometry import distance_matrix

from src.utils.connectivity import (
    connectivity_signature,
    error_curve,
    first_stable_match,
)

from src.utils.samplers import (
    random_order,
    fps_order,
    connectivity_order,
)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "grasp_dir",
        type=str,
    )

    parser.add_argument(
        "--n_reference",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--max_points",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--n_objects",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--n_seeds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    rng = np.random.default_rng(
        args.seed
    )

    # ========================================================
    # FIND ELIGIBLE OBJECTS
    # ========================================================

    paths = find_objects(
        args.grasp_dir
    )

    eligible = []

    print(
        "Finding eligible objects..."
    )

    for path in paths:

        with h5py.File(
                path,
                "r",
        ) as f:

            labels = f[
                "grasps/qualities/flex/object_in_gripper"
            ][:]

        if np.sum(labels == 1) >= args.n_reference:

            eligible.append(
                path
            )

    if len(eligible) == 0:

        raise RuntimeError(
            "No objects have enough successful grasps."
        )

    n_objects = min(
        args.n_objects,
        len(eligible),
    )

    selected_paths = rng.choice(
        eligible,
        size=n_objects,
        replace=False,
    )

    # Same connectivity scales used in the paper
    eps_values = np.linspace(
        0.10,
        0.80,
        15,
    )

    targets = [
        50,
        100,
        200,
    ]

    targets = [
        x
        for x in targets
        if x <= args.max_points
    ]

    curve_rows = []
    match_rows = []

    # ========================================================
    # RESULTS DIRECTORY
    # ========================================================

    results_dir = (
            Path("results")
            / "e2_results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # OBJECT LOOP
    # ========================================================

    for object_number, path in enumerate(
            selected_paths,
            start=1,
    ):

        object_curve_rows = []
        object_match_rows = []

        print()

        print(
            "=" * 72
        )

        print(
            f"Object "
            f"{object_number}/{n_objects}: "
            f"{Path(path).name}"
        )

        # ----------------------------------------------------
        # Load all successful grasps for this object
        # ----------------------------------------------------

        successes = (
            load_successful_grasps(
                path
            )
        )

        object_rng = (
            np.random.default_rng(
                args.seed
                + object_number * 10000
            )
        )

        # ====================================================
        # FIXED REFERENCE SET
        # ====================================================

        reference_choice = (
            object_rng.choice(
                len(successes),
                size=args.n_reference,
                replace=False,
            )
        )

        reference_transforms = (
            successes[
                reference_choice
            ]
        )

        print(
            f"Reference successful grasps: "
            f"{args.n_reference}"
        )

        print(
            "Computing SE(3) distances..."
        )

        D = distance_matrix(
            reference_transforms
        )

        all_indices = np.arange(
            args.n_reference
        )

        # ====================================================
        # REFERENCE CONNECTIVITY SIGNATURE
        #
        # This is evaluation-only information.
        # It is never passed to connectivity_order().
        # ====================================================

        (
            reference_beta,
            reference_largest,
        ) = connectivity_signature(
            all_indices,
            D,
            eps_values,
        )

        # ====================================================
        # RANDOM
        # ====================================================

        print(
            f"Running random "
            f"({args.n_seeds} seeds)..."
        )

        random_curves = []

        for seed in range(
                args.n_seeds
        ):

            seed_rng = (
                np.random.default_rng(
                    args.seed
                    + object_number * 100000
                    + seed
                )
            )

            order = random_order(
                args.n_reference,
                seed_rng,
            )

            (
                total,
                _,
                _,
            ) = error_curve(
                order,
                D,
                eps_values,
                reference_beta,
                reference_largest,
                args.max_points,
            )

            random_curves.append(
                total
            )

        random_curves = np.asarray(
            random_curves
        )

        random_mean = (
            random_curves.mean(
                axis=0
            )
        )

        random_std = (
            random_curves.std(
                axis=0
            )
        )

        # ====================================================
        # FPS
        # ====================================================

        print(
            f"Running FPS "
            f"({args.n_seeds} seeds)..."
        )

        fps_curves = []

        for seed in range(
                args.n_seeds
        ):

            seed_rng = (
                np.random.default_rng(
                    args.seed
                    + object_number * 100000
                    + 10000
                    + seed
                )
            )

            order = fps_order(
                D,
                args.max_points,
                seed_rng,
            )

            (
                total,
                _,
                _,
            ) = error_curve(
                order,
                D,
                eps_values,
                reference_beta,
                reference_largest,
                args.max_points,
            )

            fps_curves.append(
                total
            )

        fps_curves = np.asarray(
            fps_curves
        )

        fps_mean = (
            fps_curves.mean(
                axis=0
            )
        )

        fps_std = (
            fps_curves.std(
                axis=0
            )
        )

        # ====================================================
        # CONNECTIVITY-AWARE
        # ====================================================

        print(
            f"Running online connectivity "
            f"({args.n_seeds} seeds)..."
        )

        connectivity_curves = []

        for seed in range(
                args.n_seeds
        ):

            seed_rng = (
                np.random.default_rng(
                    args.seed
                    + object_number * 100000
                    + 20000
                    + seed
                )
            )

            order = connectivity_order(
                D=D,
                eps_values=eps_values,
                max_points=args.max_points,
                rng=seed_rng,
                bootstrap=args.bootstrap,
            )

            (
                total,
                _,
                _,
            ) = error_curve(
                order,
                D,
                eps_values,
                reference_beta,
                reference_largest,
                args.max_points,
            )

            connectivity_curves.append(
                total
            )

        connectivity_curves = np.asarray(
            connectivity_curves
        )

        connectivity_mean = (
            connectivity_curves.mean(
                axis=0
            )
        )

        connectivity_std = (
            connectivity_curves.std(
                axis=0
            )
        )

        # ====================================================
        # SAVE FULL MEAN CURVES
        # ====================================================

        for step in range(
                1,
                args.max_points + 1,
        ):

            row = {
                "object":
                    Path(path).name,

                "step":
                    step,

                "random_error":
                    random_mean[
                        step - 1
                        ],

                "random_std":
                    random_std[
                        step - 1
                        ],

                "fps_error":
                    fps_mean[
                        step - 1
                        ],

                "fps_std":
                    fps_std[
                        step - 1
                        ],

                "connectivity_error":
                    connectivity_mean[
                        step - 1
                        ],

                "connectivity_std":
                    connectivity_std[
                        step - 1
                        ],
            }

            curve_rows.append(
                row
            )

            object_curve_rows.append(
                row
            )

        # ====================================================
        # RANDOM TARGETS
        # ====================================================

        for target in targets:

            target_error = (
                random_mean[
                    target - 1
                    ]
            )

            conn_points = (
                first_stable_match(
                    connectivity_mean,
                    target_error,
                )
            )

            fps_points = (
                first_stable_match(
                    fps_mean,
                    target_error,
                )
            )

            # ------------------------------------------------
            # CONNECTIVITY SAVING / SPEEDUP
            # ------------------------------------------------

            if np.isnan(
                    conn_points
            ):

                conn_saving = np.nan
                conn_speedup = np.nan

            else:

                conn_saving = (
                        1.0
                        -
                        conn_points / target
                )

                conn_speedup = (
                        target
                        /
                        conn_points
                )

            # ------------------------------------------------
            # FPS SAVING
            # ------------------------------------------------

            if np.isnan(
                    fps_points
            ):

                fps_saving = np.nan

            else:

                fps_saving = (
                        1.0
                        -
                        fps_points / target
                )

            # ------------------------------------------------
            # SAVE TARGET RESULT
            # ------------------------------------------------

            row = {
                "object":
                    Path(path).name,

                "random_target":
                    target,

                "random_error":
                    target_error,

                "connectivity_points":
                    conn_points,

                "connectivity_saving":
                    conn_saving,

                "connectivity_speedup":
                    conn_speedup,

                "fps_points":
                    fps_points,

                "fps_saving":
                    fps_saving,
            }

            match_rows.append(
                row
            )

            object_match_rows.append(
                row
            )

            # ------------------------------------------------
            # PRINT OBJECT RESULT
            # ------------------------------------------------

            print()

            print(
                f"Random@{target}"
            )

            print(
                f"  target error: "
                f"{target_error:.4f}"
            )

            if np.isnan(
                    conn_points
            ):

                print(
                    "  Connectivity: "
                    "not reached"
                )

            else:

                print(
                    f"  Connectivity: "
                    f"{int(conn_points)} points"
                )

                print(
                    f"  Saving: "
                    f"{100 * conn_saving:.1f}%"
                )

                print(
                    f"  Speedup: "
                    f"{conn_speedup:.2f}x"
                )

            if np.isnan(
                    fps_points
            ):

                print(
                    "  FPS: "
                    "not reached"
                )

            else:

                print(
                    f"  FPS: "
                    f"{int(fps_points)} points"
                )

        # ====================================================
        # SAVE COMPLETED OBJECT
        # ====================================================

        object_name = (
            Path(path).stem
        )

        object_dir = (
                results_dir
                / (
                    f"object_"
                    f"{object_number:03d}_"
                    f"{object_name}"
                )
        )

        object_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ====================================================
        # SAVE INDIVIDUAL SEEDS
        # ====================================================

        for seed in range(
                args.n_seeds
        ):

            seed_dir = (
                    object_dir
                    / f"seed_{seed + 1:02d}"
            )

            seed_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            steps = np.arange(
                1,
                args.max_points + 1,
                )

            pd.DataFrame({
                "step":
                    steps,

                "error":
                    random_curves[
                        seed
                    ],
            }).to_csv(
                seed_dir
                / "random.csv",
                index=False,
                )

            pd.DataFrame({
                "step":
                    steps,

                "error":
                    fps_curves[
                        seed
                    ],
            }).to_csv(
                seed_dir
                / "fps.csv",
                index=False,
                )

            pd.DataFrame({
                "step":
                    steps,

                "error":
                    connectivity_curves[
                        seed
                    ],
            }).to_csv(
                seed_dir
                / "connectivity.csv",
                index=False,
                )

        # ----------------------------------------------------
        # Object-level aggregate files
        # ----------------------------------------------------

        pd.DataFrame(
            object_curve_rows
        ).to_csv(
            object_dir
            / "curves.csv",
            index=False,
            )

        pd.DataFrame(
            object_match_rows
        ).to_csv(
            object_dir
            / "matches.csv",
            index=False,
            )

        # ====================================================
        # UPDATE GLOBAL CHECKPOINT
        # ====================================================

        pd.DataFrame(
            curve_rows
        ).to_csv(
            results_dir
            / "online_connectivity_curves.csv",
            index=False,
            )

        pd.DataFrame(
            match_rows
        ).to_csv(
            results_dir
            / "online_connectivity_matches.csv",
            index=False,
            )

        print()

        print(
            f"Checkpoint saved: "
            f"{object_dir}"
        )

    # ========================================================
    # FINAL DATAFRAMES
    # ========================================================

    curves_df = pd.DataFrame(
        curve_rows
    )

    matches_df = pd.DataFrame(
        match_rows
    )

    curves_path = (
            results_dir
            / "online_connectivity_curves.csv"
    )

    matches_path = (
            results_dir
            / "online_connectivity_matches.csv"
    )

    curves_df.to_csv(
        curves_path,
        index=False,
    )

    matches_df.to_csv(
        matches_path,
        index=False,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "FINAL SUMMARY"
    )

    print(
        "=" * 72
    )

    summary_rows = []

    for target in targets:

        x = matches_df[
            matches_df[
                "random_target"
            ] == target
            ]

        conn = (
            x[
                "connectivity_points"
            ]
            .dropna()
        )

        fps = (
            x[
                "fps_points"
            ]
            .dropna()
        )

        # ----------------------------------------------------
        # Connectivity summary
        # ----------------------------------------------------

        if len(conn) > 0:

            conn_mean = (
                conn.mean()
            )

            conn_median = (
                conn.median()
            )

            saving = (
                    100.0
                    *
                    (
                            1.0
                            -
                            conn_mean / target
                    )
            )

            speedup = (
                    target
                    /
                    conn_mean
            )

        else:

            conn_mean = np.nan
            conn_median = np.nan
            saving = np.nan
            speedup = np.nan

        # ----------------------------------------------------
        # FPS summary
        # ----------------------------------------------------

        if len(fps) > 0:

            fps_mean = (
                fps.mean()
            )

        else:

            fps_mean = np.nan

        # ----------------------------------------------------
        # Summary row
        # ----------------------------------------------------

        summary_rows.append({
            "random_target":
                target,

            "connectivity_mean_points":
                conn_mean,

            "connectivity_median_points":
                conn_median,

            "connectivity_saving_percent":
                saving,

            "connectivity_speedup":
                speedup,

            "connectivity_reached":
                len(conn),

            "objects":
                len(x),

            "fps_mean_points":
                fps_mean,

            "fps_reached":
                len(fps),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    print()

    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.2f}",
        )
    )

    summary_path = (
            results_dir
            / "online_connectivity_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ========================================================
    # OUTPUT PATHS
    # ========================================================

    print()

    print(
        "Saved:"
    )

    print(
        f"  {curves_path}"
    )

    print(
        f"  {matches_path}"
    )

    print(
        f"  {summary_path}"
    )


if __name__ == "__main__":
    main()