import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ============================================================
# 1. DATA
# ============================================================

def find_objects(folder):
    folder = Path(folder)
    return sorted(folder.rglob("*.h5"))


def load_grasp_candidates(path):
    """
    Load ALL candidate grasps and their hidden success labels.

    E3 setting:
        - geometry of every candidate grasp is known
        - success/failure labels are hidden from the selector
        - a label is revealed only after that candidate is queried
    """

    with h5py.File(path, "r") as f:

        transforms = f["grasps/transforms"][:]

        labels = f[
            "grasps/qualities/flex/object_in_gripper"
        ][:].astype(int)

    return transforms, labels


# ============================================================
# 2. SE(3) DISTANCE
# ============================================================

def distance_matrix(transforms):
    """
    Pairwise normalized SE(3) distance.

    d = sqrt(
        normalized_translation^2
        +
        normalized_rotation^2
    )
    """

    p = transforms[:, :3, 3]
    R = transforms[:, :3, :3]

    # --------------------------------------------------------
    # Translation
    # --------------------------------------------------------

    dp = np.linalg.norm(
        p[:, None, :] - p[None, :, :],
        axis=-1,
        )

    # --------------------------------------------------------
    # Rotation
    # --------------------------------------------------------

    relative = np.einsum(
        "aik,bil->abkl",
        R,
        R,
    )

    trace = np.trace(
        relative,
        axis1=-2,
        axis2=-1,
    )

    cosine = np.clip(
        (trace - 1.0) / 2.0,
        -1.0,
        1.0,
        )

    dr = np.arccos(cosine)

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    scale_p = np.median(
        dp[dp > 0]
    )

    scale_r = np.median(
        dr[dr > 0]
    )

    D = np.sqrt(
        (dp / scale_p) ** 2
        +
        (dr / scale_r) ** 2
    )

    return D.astype(
        np.float32
    )


# ============================================================
# 3. CONNECTIVITY SIGNATURE
# ============================================================

def connectivity_signature(
        indices,
        D,
        eps_values,
):
    """
    Compute:

        B(eps) = beta_0(eps) / N

        L(eps) = largest_component(eps) / N

    This is our multiscale connectivity map.
    """

    indices = np.asarray(
        indices,
        dtype=int,
    )

    n = len(indices)

    if n == 0:

        return (
            np.ones(len(eps_values)),
            np.zeros(len(eps_values)),
        )

    if n == 1:

        return (
            np.ones(len(eps_values)),
            np.ones(len(eps_values)),
        )

    sub = D[
        np.ix_(
            indices,
            indices,
        )
    ]

    beta_curve = []
    largest_curve = []

    for eps in eps_values:

        graph = csr_matrix(
            sub <= eps
        )

        n_components, labels = (
            connected_components(
                graph,
                directed=False,
            )
        )

        sizes = np.bincount(
            labels
        )

        beta_curve.append(
            n_components / n
        )

        largest_curve.append(
            sizes.max() / n
        )

    return (
        np.asarray(beta_curve),
        np.asarray(largest_curve),
    )


# ============================================================
# 4. STRUCTURAL ERROR
# ============================================================

def signature_error(
        selected,
        reference_beta,
        reference_largest,
        D,
        eps_values,
):
    """
    Used ONLY for evaluation.

    The selection algorithm never receives
    reference_beta or reference_largest.
    """

    beta, largest = (
        connectivity_signature(
            selected,
            D,
            eps_values,
        )
    )

    beta_error = np.mean(
        np.abs(
            beta - reference_beta
        )
    )

    largest_error = np.mean(
        np.abs(
            largest - reference_largest
        )
    )

    total_error = 0.5 * (
            beta_error
            +
            largest_error
    )

    return (
        float(total_error),
        float(beta_error),
        float(largest_error),
    )


# ============================================================
# 5. RANDOM
# ============================================================

def random_order(
        n,
        rng,
):
    return rng.permutation(n)


# ============================================================
# 6. FPS
# ============================================================

def fps_order(
        D,
        max_points,
        rng,
):
    """
    Geometry-only baseline.

    Repeatedly select the point farthest from
    the current selected set.
    """

    n = len(D)

    first = int(
        rng.integers(n)
    )

    selected = [
        first
    ]

    available = np.ones(
        n,
        dtype=bool,
    )

    available[first] = False

    # Distance of every point to nearest selected point
    nearest = D[:, first].copy()

    while len(selected) < max_points:

        candidates = np.flatnonzero(
            available
        )

        chosen = int(
            candidates[
                np.argmax(
                    nearest[candidates]
                )
            ]
        )

        selected.append(
            chosen
        )

        available[chosen] = False

        # Incremental update
        nearest = np.minimum(
            nearest,
            D[:, chosen],
        )

    return np.asarray(
        selected,
        dtype=int,
    )


# ============================================================
# 7. CURRENT CONNECTIVITY COMPONENTS
# ============================================================

def get_components(
        selected,
        D,
        eps,
):
    """
    Components of the CURRENTLY OBSERVED map.

    No reference structure is used.
    """

    selected = np.asarray(
        selected,
        dtype=int,
    )

    if len(selected) == 0:
        return []

    if len(selected) == 1:
        return [
            selected.copy()
        ]

    sub = D[
        np.ix_(
            selected,
            selected,
        )
    ]

    graph = csr_matrix(
        sub <= eps
    )

    _, labels = connected_components(
        graph,
        directed=False,
    )

    components = []

    for label in np.unique(
            labels
    ):

        components.append(
            selected[
                labels == label
                ]
        )

    return components


# ============================================================
# 8. ONLINE CONNECTIVITY SELECTION
# ============================================================

def choose_connectivity_point(
        observed_successes,
        available,
        D,
        eps_values,
):
    """
    Choose next point using ONLY the current observed map.

    Three ideas:

    1. FRONTIER
       Look just outside currently observed components.

    2. BRIDGE
       Look for points that could connect two currently
       separate components.

    3. NOVELTY
       Occasionally extend into poorly observed space.

    No reference connectivity signature is used.
    """

    candidates = np.flatnonzero(
        available
    )

    # --------------------------------------------------------
    # Distance from candidates to all observed points
    # --------------------------------------------------------

    candidate_to_selected = D[
        np.ix_(
            candidates,
            observed_successes,
        )
    ]

    nearest_selected = (
        candidate_to_selected.min(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Geometric novelty
    # --------------------------------------------------------

    novelty = (
        nearest_selected.copy()
    )

    if novelty.max() > 0:

        novelty = (
                novelty
                /
                novelty.max()
        )

    # --------------------------------------------------------
    # Structural score
    # --------------------------------------------------------

    structural_score = np.zeros(
        len(candidates),
        dtype=float,
    )

    for eps in eps_values:

        components = get_components(
            observed_successes,
            D,
            eps,
        )

        if len(components) == 0:
            continue

        # ----------------------------------------------------
        # For every candidate:
        # how many current components is it within eps of?
        # ----------------------------------------------------

        touched = np.zeros(
            len(candidates),
            dtype=int,
        )

        minimum_component_distance = (
            np.full(
                len(candidates),
                np.inf,
            )
        )

        for component in components:

            distances = D[
                np.ix_(
                    candidates,
                    component,
                )
            ]

            distance_to_component = (
                distances.min(
                    axis=1
                )
            )

            touched += (
                    distance_to_component
                    <= eps
            )

            minimum_component_distance = (
                np.minimum(
                    minimum_component_distance,
                    distance_to_component,
                )
            )

        # ====================================================
        # A. BRIDGE SCORE
        #
        # Candidate touches >= 2 currently separate components.
        #
        # This is directly informative about whether those
        # components should merge.
        # ====================================================

        bridge = np.maximum(
            touched - 1,
            0,
            ).astype(float)

        # ====================================================
        # B. FRONTIER SCORE
        #
        # Candidate lies just OUTSIDE the known structure.
        #
        # We want distance approximately eps.
        #
        # Too close:
        #     probably redundant.
        #
        # Extremely far:
        #     not informative about this component boundary.
        # ====================================================

        frontier = np.exp(
            -(
                     (
                             minimum_component_distance
                             - 1.25 * eps
                     )
                     /
                     (0.50 * eps + 1e-8)
             ) ** 2
        )

        # Frontier matters primarily when candidate
        # is not already inside a component.
        frontier[
            touched > 0
            ] *= 0.25

        # ====================================================
        # C. BOUNDARY EXTENSION
        #
        # Candidate touches exactly one component and lies
        # near its outer connectivity boundary.
        # ====================================================

        boundary = np.zeros(
            len(candidates),
            dtype=float,
        )

        one_component = (
                touched == 1
        )

        boundary[
            one_component
        ] = np.clip(
            minimum_component_distance[
                one_component
            ]
            /
            (eps + 1e-8),
            0.0,
            1.0,
            )

        # ----------------------------------------------------
        # Combine structural information at this scale
        # ----------------------------------------------------

        structural_score += (
                2.0 * bridge
                +
                1.0 * frontier
                +
                0.5 * boundary
        )

    # Average over scales
    structural_score /= len(
        eps_values
    )

    # Normalize
    if structural_score.max() > 0:

        structural_score = (
                structural_score
                /
                structural_score.max()
        )

    # --------------------------------------------------------
    # Final score
    #
    # Structural information dominates.
    # Novelty prevents collapse onto one local region.
    # --------------------------------------------------------

    score = (
            structural_score
            +
            0.20 * novelty
    )

    chosen_position = np.argmax(
        score
    )

    return int(
        candidates[
            chosen_position
        ]
    )


# ============================================================
# 9. ONLINE CONNECTIVITY ORDER
# ============================================================

def connectivity_active_order(
        D,
        hidden_labels,
        eps_values,
        max_queries,
        rng,
        bootstrap=5,
):
    """
    Genuine active grasp discovery.

    The selector knows:
        - geometry of all candidate grasps
        - which candidates have already been queried
        - success/failure outcomes of queried candidates
        - connectivity of OBSERVED SUCCESSFUL grasps

    The selector does NOT know:
        - labels of unqueried candidates
        - hidden full successful grasp set
        - reference connectivity curves
        - future structural error

    Returns
    -------
    queried_order : np.ndarray
        Candidate queried at every step.

    success_history : list
        observed successful indices after every query.

    outcomes : np.ndarray
        Revealed outcome for every query.
    """

    n = len(D)

    available = np.ones(
        n,
        dtype=bool,
    )

    queried_order = []
    observed_successes = []
    outcomes = []

    for step in range(max_queries):

        # --------------------------------------------------------
        # Bootstrap or no known successful grasp yet:
        # query randomly
        # --------------------------------------------------------

        if (
                step < bootstrap
                or len(observed_successes) == 0
        ):

            candidates = np.flatnonzero(
                available
            )

            chosen = int(
                rng.choice(candidates)
            )

        else:

            chosen = choose_connectivity_point(
                observed_successes=observed_successes,
                available=available,
                D=D,
                eps_values=eps_values,
            )

        # --------------------------------------------------------
        # Query candidate
        # --------------------------------------------------------

        available[chosen] = False

        outcome = int(
            hidden_labels[chosen]
        )

        queried_order.append(
            chosen
        )

        outcomes.append(
            outcome
        )

        # --------------------------------------------------------
        # Only successful queried grasps enter viable map
        # --------------------------------------------------------

        if outcome == 1:

            observed_successes.append(
                chosen
            )

    return (
        np.asarray(
            queried_order,
            dtype=int,
        ),
        np.asarray(
            outcomes,
            dtype=int,
        ),
    )

def active_error_curve(
        queried_order,
        outcomes,
        D,
        eps_values,
        reference_beta,
        reference_largest,
):
    """
    Structural reconstruction error as a function of QUERY COUNT.

    Only successful queried grasps contribute to the estimated
    viable-space connectivity structure.
    """

    total_errors = []
    beta_errors = []
    largest_errors = []
    success_counts = []

    observed_successes = []

    for step, (idx, outcome) in enumerate(
            zip(queried_order, outcomes),
            start=1,
    ):

        if outcome == 1:

            observed_successes.append(
                int(idx)
            )

        total, beta, largest = signature_error(
            observed_successes,
            reference_beta,
            reference_largest,
            D,
            eps_values,
        )

        total_errors.append(
            total
        )

        beta_errors.append(
            beta
        )

        largest_errors.append(
            largest
        )

        success_counts.append(
            len(observed_successes)
        )

    return (
        np.asarray(total_errors),
        np.asarray(beta_errors),
        np.asarray(largest_errors),
        np.asarray(success_counts),
    )
# ============================================================
# 10. ERROR CURVE
# ============================================================

def error_curve(
        order,
        D,
        eps_values,
        reference_beta,
        reference_largest,
        max_points,
):
    """
    Evaluation only.

    Reference structure appears here, AFTER the
    selection ordering has already been created.
    """

    total_errors = []
    beta_errors = []
    largest_errors = []

    for n_points in range(
            1,
            max_points + 1,
    ):

        selected = order[
            :n_points
        ]

        (
            total,
            beta,
            largest,
        ) = signature_error(
            selected,
            reference_beta,
            reference_largest,
            D,
            eps_values,
        )

        total_errors.append(
            total
        )

        beta_errors.append(
            beta
        )

        largest_errors.append(
            largest
        )

    return (
        np.asarray(total_errors),
        np.asarray(beta_errors),
        np.asarray(largest_errors),
    )


# ============================================================
# 11. STABLE TARGET MATCH
# ============================================================

def first_stable_match(
        errors,
        target_error,
        window=5,
):
    """
    First point where the method reaches the target
    and remains at or below it on average for a
    short window.

    This avoids counting one accidental crossing.
    """

    if len(errors) < window:
        return np.nan

    for end in range(
            window - 1,
            len(errors),
    ):

        recent = errors[
            end - window + 1:
            end + 1
        ]

        if np.mean(recent) <= target_error:

            return int(
                end + 1
            )

    return np.nan


# ============================================================
# 12. MAIN
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
        "--max_queries",
        type=int,
        default=300,
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

    # Same scales as before
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
        if x <= args.max_queries
    ]

    curve_rows = []
    match_rows = []

    # ========================================================
    # RESULTS DIRECTORY
    # ========================================================

    results_dir = (
            Path("results")
            / "active_connectivity_e3"
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

        transforms, hidden_labels = (
            load_grasp_candidates(
                path
            )
        )

        n_candidates = len(transforms)

        hidden_success_indices = np.flatnonzero(
            hidden_labels == 1
        )

        print(
            f"Candidate grasps: {n_candidates}"
        )

        print(
            f"Hidden successful grasps: "
            f"{len(hidden_success_indices)}"
        )

        print(
            "Computing SE(3) distances..."
        )

        D = distance_matrix(
            transforms
        )

        # ============================================================
        # HIDDEN ANSWER KEY
        #
        # This uses labels ONLY for evaluation.
        # It must never be passed to acquisition.
        # ============================================================

        (
            reference_beta,
            reference_largest,
        ) = connectivity_signature(
            hidden_success_indices,
            D,
            eps_values,
        )


        # ========================================================
        # RANDOM — 5 SEEDS
        # ========================================================

        print(
            f"Running random ({args.n_seeds} seeds)..."
        )

        random_curves = []

        for seed in range(args.n_seeds):

            seed_rng = np.random.default_rng(
                args.seed
                + object_number * 100000
                + seed
            )

            order = random_order(
                n_candidates,
                seed_rng,
            )[:args.max_queries]

            outcomes = hidden_labels[order]

            total, _, _, success_counts = active_error_curve(
                order,
                outcomes,
                D,
                eps_values,
                reference_beta,
                reference_largest,
            )

            random_curves.append(total)


        random_curves = np.asarray(
            random_curves
        )

        random_mean = random_curves.mean(axis=0)
        random_std = random_curves.std(axis=0)

        # ========================================================
        # FPS — 5 SEEDS
        # ========================================================

        print(
            f"Running FPS ({args.n_seeds} seeds)..."
        )

        fps_curves = []

        for seed in range(args.n_seeds):

            seed_rng = np.random.default_rng(
                args.seed
                + object_number * 100000
                + 10000
                + seed
            )

            order = fps_order(
                D,
                args.max_queries,
                seed_rng,
            )

            outcomes = hidden_labels[order]

            total, _, _, success_counts = active_error_curve(
                order,
                outcomes,
                D,
                eps_values,
                reference_beta,
                reference_largest,
            )

            fps_curves.append(total)



        fps_curves = np.asarray(
            fps_curves
        )

        fps_mean = fps_curves.mean(axis=0)
        fps_std = fps_curves.std(axis=0)

        # ========================================================
        # ONLINE CONNECTIVITY — 5 SEEDS
        # ========================================================

        print(
            f"Running online connectivity "
            f"({args.n_seeds} seeds)..."
        )

        connectivity_curves = []

        for seed in range(args.n_seeds):

            seed_rng = np.random.default_rng(
                args.seed
                + object_number * 100000
                + 20000
                + seed
            )

            order, outcomes = connectivity_active_order(
                D=D,
                hidden_labels=hidden_labels,
                eps_values=eps_values,
                max_queries=args.max_queries,
                rng=seed_rng,
                bootstrap=args.bootstrap,
            )

            total, beta, largest, success_counts = active_error_curve(
                order,
                outcomes,
                D,
                eps_values,
                reference_beta,
                reference_largest,
            )

            connectivity_curves.append(total)

        connectivity_curves = np.asarray(
            connectivity_curves
        )

        connectivity_mean = (
            connectivity_curves.mean(axis=0)
        )

        connectivity_std = (
            connectivity_curves.std(axis=0)
        )

        # ====================================================
        # SAVE FULL CURVES
        # ====================================================

        for step in range(
                1,
                args.max_queries + 1,
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

            curve_rows.append(row)
            object_curve_rows.append(row)

        # ====================================================
        # RANDOM @ 50 / 100 / 200
        # ====================================================

        for target in targets:

            target_error = (
                random_mean[
                    target - 1
                    ]
            )

            conn_points = first_stable_match(
                connectivity_mean,
                target_error,
            )

            fps_points = first_stable_match(
                fps_mean,
                target_error,
            )

            # ------------------------------------------------
            # Connectivity saving
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
            # FPS saving
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

            match_rows.append(row)
            object_match_rows.append(row)

            # =================================================
            # PRINT
            # =================================================

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

        object_name = Path(path).stem

        object_dir = (
                results_dir
                / f"object_{object_number:03d}_{object_name}"
        )

        object_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        # ====================================================
        # SAVE INDIVIDUAL SEEDS
        # ====================================================

        for seed in range(args.n_seeds):

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
                args.max_queries + 1,
                )

            pd.DataFrame({
                "step": steps,
                "error": random_curves[seed],
            }).to_csv(
                seed_dir / "random.csv",
                index=False,
                )

            pd.DataFrame({
                "step": steps,
                "error": fps_curves[seed],
            }).to_csv(
                seed_dir / "fps.csv",
                index=False,
                )

            pd.DataFrame({
                "step": steps,
                "error": connectivity_curves[seed],
            }).to_csv(
                seed_dir / "connectivity.csv",
                index=False,
                )
        pd.DataFrame(
            object_curve_rows
        ).to_csv(
            object_dir / "curves.csv",
            index=False,
            )

        pd.DataFrame(
            object_match_rows
        ).to_csv(
            object_dir / "matches.csv",
            index=False,
            )

        # ====================================================
        # UPDATE GLOBAL CHECKPOINT
        # ====================================================

        pd.DataFrame(
            curve_rows
        ).to_csv(
            results_dir / "online_connectivity_curves.csv",
            index=False,
            )

        pd.DataFrame(
            match_rows
        ).to_csv(
            results_dir / "online_connectivity_matches.csv",
            index=False,
            )

        print()
        print(
            f"Checkpoint saved: {object_dir}"
        )



    # ========================================================
    # SUMMARY
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

        conn = x[
            "connectivity_points"
        ].dropna()

        fps = x[
            "fps_points"
        ].dropna()

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

        if len(fps) > 0:

            fps_mean = (
                fps.mean()
            )

        else:

            fps_mean = np.nan

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
            f"{x:.2f}"
        )
    )

    summary_path = (
            results_dir
            /
            "online_connectivity_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

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