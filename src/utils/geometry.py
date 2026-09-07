# utils/geometry.py

import numpy as np


def distance_matrix(transforms):
    """
    Compute pairwise normalized SE(3) distances.

    d = sqrt(
        normalized_translation^2
        +
        normalized_rotation^2
    )

    Translation and rotation are normalized independently
    by their median nonzero pairwise distances.
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