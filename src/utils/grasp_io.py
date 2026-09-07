
from pathlib import Path

import h5py


def find_objects(folder):
    """
    Recursively find all ACRONYM HDF5 object files.
    """
    folder = Path(folder)
    return sorted(folder.rglob("*.h5"))


def load_successful_grasps(path):
    """
    Load successful grasps only.

    In E2, grasp viability is already known.
    The task is therefore not success discovery, but reconstruction
    of the connectivity structure of the viable grasp set.

    Returns
    -------
    transforms : np.ndarray
        Homogeneous 4x4 transforms of all successful grasps.
    """
    with h5py.File(path, "r") as f:
        transforms = f["grasps/transforms"][:]

        labels = f[
            "grasps/qualities/flex/object_in_gripper"
        ][:].astype(int)

    return transforms[labels == 1]