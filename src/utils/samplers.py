# utils/samplers.py

import numpy as np

from src.utils.connectivity import get_components


# ============================================================
# RANDOM
# ============================================================

def random_order(
        n,
        rng,
):
    """
    Uniform random ordering without replacement.
    """
    return rng.permutation(n)


# ============================================================
# FARTHEST-POINT SAMPLING
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
# CONNECTIVITY-AWARE ACQUISITION
# ============================================================

def choose_connectivity_point(
        selected,
        available,
        D,
        eps_values,
):
    """
    Choose the next point using only the currently observed map.

    Acquisition combines:

        1. Bridge
        2. Frontier
        3. Boundary extension
        4. Geometric novelty

    No reference connectivity signature is used.
    """

    candidates = np.flatnonzero(
        available
    )

    # --------------------------------------------------------
    # Distance from candidates to observed points
    # --------------------------------------------------------

    candidate_to_selected = D[
        np.ix_(
            candidates,
            selected,
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
            selected,
            D,
            eps,
        )

        if len(components) == 0:
            continue

        # ----------------------------------------------------
        # Candidate relationship to current components
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
        # BRIDGE
        # ====================================================

        bridge = np.maximum(
            touched - 1,
            0,
            ).astype(float)

        # ====================================================
        # FRONTIER
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

        # Frontier primarily matters outside an
        # already observed component.
        frontier[
            touched > 0
            ] *= 0.25

        # ====================================================
        # BOUNDARY EXTENSION
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

    # Average across connectivity scales
    structural_score /= len(
        eps_values
    )

    # Normalize structural acquisition score
    if structural_score.max() > 0:

        structural_score = (
                structural_score
                /
                structural_score.max()
        )

    # --------------------------------------------------------
    # Final acquisition score
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
# CONNECTIVITY-AWARE ORDER
# ============================================================

def connectivity_order(
        D,
        eps_values,
        max_points,
        rng,
        bootstrap=5,
):
    """
    Construct a connectivity-aware sampling order.

    The sampler knows:

        - candidate positions
        - pairwise geometry
        - points already selected
        - connectivity of selected points

    It does not know:

        - the reference connectivity signature
        - future structural error

    The first `bootstrap` points are sampled randomly.
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

    # --------------------------------------------------------
    # Random bootstrap
    # --------------------------------------------------------

    bootstrap_size = min(
        bootstrap,
        max_points,
    )

    if bootstrap_size > 1:

        extra = rng.choice(
            np.flatnonzero(
                available
            ),
            size=bootstrap_size - 1,
            replace=False,
        )

        for chosen in extra:

            selected.append(
                int(chosen)
            )

            available[
                chosen
            ] = False

    # --------------------------------------------------------
    # Continuous connectivity-aware reconstruction
    # --------------------------------------------------------

    while len(selected) < max_points:

        chosen = choose_connectivity_point(
            selected=selected,
            available=available,
            D=D,
            eps_values=eps_values,
        )

        selected.append(
            chosen
        )

        available[
            chosen
        ] = False

    return np.asarray(
        selected,
        dtype=int,
    )