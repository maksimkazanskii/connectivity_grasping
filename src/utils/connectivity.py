import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def connectivity_signature(
        indices,
        D,
        eps_values,
):
    """
    Compute the multiscale connectivity signature:

        B(eps) = beta_0(eps) / N
        L(eps) = largest_component(eps) / N
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


def signature_error(
        selected,
        reference_beta,
        reference_largest,
        D,
        eps_values,
):
    """
    Structural reconstruction error.

    Used only for evaluation. The sampling algorithm
    does not receive the reference signature.
    """

    beta, largest = connectivity_signature(
        selected,
        D,
        eps_values,
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


def get_components(
        selected,
        D,
        eps,
):
    """
    Return connected components of the currently
    observed grasp set at radius eps.

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


def first_stable_match(
        errors,
        target_error,
        window=5,
):
    """
    Return the first point at which the mean error
    over a short window reaches the target.

    This avoids counting a transient crossing.
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

def error_curve(
        order,
        D,
        eps_values,
        reference_beta,
        reference_largest,
        max_points,
):
    """
    Compute structural reconstruction error along
    a fixed sampling order.

    Evaluation only.
    """

    total_errors = []
    beta_errors = []
    largest_errors = []

    for n_points in range(
            1,
            max_points + 1,
    ):

        selected = order[:n_points]

        total, beta, largest = signature_error(
            selected,
            reference_beta,
            reference_largest,
            D,
            eps_values,
        )

        total_errors.append(total)
        beta_errors.append(beta)
        largest_errors.append(largest)

    return (
        np.asarray(total_errors),
        np.asarray(beta_errors),
        np.asarray(largest_errors),
    )