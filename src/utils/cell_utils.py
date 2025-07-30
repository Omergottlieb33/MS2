import numpy as np

def get_3d_bounding_box_corners(mask: np.ndarray) -> np.ndarray:
    """
    Calculates the 8 corners of the 3D bounding box for a given 3D mask.

    The mask is expected to be a 3D numpy array where non-zero values
    indicate the object's presence. The input array is assumed to have
    dimensions ordered as (z, y, x).

    Args:
        mask (np.ndarray): A 3D numpy array (z, y, x) representing the shape.

    Returns:
        np.ndarray: An array of shape (8, 3) containing the (x, y, z)
                    coordinates of the 8 corners of the bounding box.
                    Returns an empty array of shape (0, 3) if the mask is empty.
    """
    # Find the coordinates of all non-zero voxels.
    # np.where returns a tuple of arrays, one for each dimension (z, y, x).
    coords = np.where(mask > 0)
    if not coords[0].size:
        # No object found in the mask
        return np.empty((0, 3), dtype=int)

    z_coords, y_coords, x_coords = coords

    # Find the min and max for each dimension to define the bounding box
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)

    return z_min, y_min, x_min, z_max, y_max, x_max