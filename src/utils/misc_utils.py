import numpy as np
from scipy.ndimage import center_of_mass

def get_cell_centers_fast(masks_array):
    """
    Ultra-fast computation of cell centers using scipy.ndimage.center_of_mass.
    
    Parameters:
    - masks_array: 3D numpy array with labeled cells
    
    Returns:
    - centers: 2D numpy array with shape (n_cells, 4) where columns are [label, x, y, z]
    """
    
    # Get all unique labels (excluding background)
    labels = np.unique(masks_array)
    labels = labels[labels > 0]
    
    if len(labels) == 0:
        return np.empty((0, 4))
    
    # Compute centers of mass for all labels at once
    centers_of_mass = center_of_mass(masks_array > 0, masks_array, labels)
    
    # Convert to 4-column array [label, x, y, z]
    centers_array = []
    for i, label in enumerate(labels):
        if not np.isnan(centers_of_mass[i]).any():
            # Note: center_of_mass returns (z, y, x), so we need to reorder
            z, y, x = centers_of_mass[i]
            centers_array.append([int(label), float(x), float(y), float(z)])
    
    return np.array(centers_array)

def get_cell_centers(masks_array):
    labels = np.unique(masks_array)
    centers = {}
    for label in labels:
        if label == 0:
            continue
        mask = (masks_array == label).astype(np.uint8)
        z_coords, y_coords, x_coords = np.where(mask == 1)
        if len(x_coords) > 0:
            center_of_mass_x = np.mean(x_coords).astype(float)
            center_of_mass_y = np.mean(y_coords).astype(float)
            center_of_mass_z = np.mean(z_coords).astype(float)
            centers[int(label)] = (center_of_mass_x, center_of_mass_y, center_of_mass_z)
    return centers