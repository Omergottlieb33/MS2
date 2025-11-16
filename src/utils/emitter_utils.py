import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

def circular_z_score(theta_rad):
    theta = np.mod(theta_rad, 2*np.pi)
    s, c = np.sin(theta), np.cos(theta)
    mu = np.arctan2(s.mean(), c.mean())
    # shortest signed angular distance
    delta = np.arctan2(np.sin(theta - mu), np.cos(theta - mu))
    R = np.hypot(c.sum(), s.sum()) / theta.size
    sigma = np.sqrt(-2*np.log(max(R, 1e-12)))
    z = np.abs(delta) / max(sigma, 1e-12)
    return z, mu, sigma

def get_indices_in_mask(coordinates: np.ndarray, cell_mask: np.ndarray, remove_outline: bool = False, expand_pixels: int = 2) -> np.ndarray:
    """
    Finds the indices of coordinates that are inside a boolean cell mask.
    Optionally removes the outline (outermost pixel layer) of the mask first.

    Args:
        coordinates (np.ndarray): A NumPy array of shape (n, 2) with (x, y) coordinates.
        cell_mask (np.ndarray): A 2D NumPy array of dtype bool, where True indicates
                                the area of interest.
        remove_outline (bool): If True, the outer layer of pixels of the True
                               areas in the mask is removed before checking
                               for coordinates. Defaults to False.

    Returns:
        np.ndarray: A 1D NumPy array containing the indices of the coordinates
                    that fall within the True areas of the cell_mask.
    """
    # --- Input Validation ---
    if not isinstance(coordinates, np.ndarray) or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Coordinates must be a NumPy array of shape (n, 2).")
    if not isinstance(cell_mask, np.ndarray) or cell_mask.ndim != 2 or cell_mask.dtype != bool:
        raise ValueError("cell_mask must be a 2D boolean NumPy array.")

    # --- Mask Processing ---
    # If requested, remove the outer layer of pixels (outline) from the mask.
    # This process is called binary erosion.
    if remove_outline:
        # We use binary_erosion to shrink the True regions of the mask by one pixel
        # from all sides, effectively removing the outline.
        processed_mask = binary_erosion(cell_mask)
    else:
        processed_mask = cell_mask

    if expand_pixels and expand_pixels > 0:
        processed_mask = binary_dilation(
            processed_mask, iterations=expand_pixels)

    # --- Coordinate Processing ---
    # Round coordinates to the nearest integer to use them as indices.
    # Using np.round is safer than just casting to int, as it handles floating point values correctly.
    int_coords = coordinates.astype(int)

    # Extract x and y columns. Note that in image processing and array indexing,
    # x corresponds to columns (dimension 1) and y to rows (dimension 0).
    x_coords = int_coords[:, 0]
    y_coords = int_coords[:, 1]

    # Get the dimensions of the mask (height, width)
    mask_height, mask_width = processed_mask.shape

    # --- Boundary Check ---
    # Create a boolean mask to identify coordinates that are within the bounds of the cell_mask.
    # This is crucial to prevent IndexError when accessing the mask.
    in_bounds_mask = (
        (x_coords >= 0) & (x_coords < mask_width) &
        (y_coords >= 0) & (y_coords < mask_height)
    )

    # Get the original indices of the coordinates that are within bounds.
    # We will use these indices to filter our coordinates before checking the cell mask.
    original_indices_in_bounds = np.where(in_bounds_mask)[0]

    # If no coordinates are within the bounds, return an empty array.
    if len(original_indices_in_bounds) == 0:
        return np.array([], dtype=int)

    # Filter the coordinates to only include those within the mask's dimensions.
    x_in_bounds = x_coords[original_indices_in_bounds]
    y_in_bounds = y_coords[original_indices_in_bounds]

    # --- Mask Lookup ---
    # Use the valid, in-bounds coordinates to look up their values in the processed_mask.
    # NumPy's advanced indexing allows us to do this in a single, efficient operation.
    # The result is a boolean array indicating if each in-bounds point is in a True region.
    # Remember: array access is [row, column], which corresponds to [y, x].
    is_inside_mask = processed_mask[y_in_bounds, x_in_bounds]

    # --- Final Filtering ---
    # The final step is to select the original indices that correspond to a True value
    # in our 'is_inside_mask' array.
    final_indices = original_indices_in_bounds[is_inside_mask]

    return final_indices.astype(int)