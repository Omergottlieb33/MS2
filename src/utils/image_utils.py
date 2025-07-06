import numpy as np
import czifile

def load_czi_images(file_path):
    try:
        with czifile.CziFile(file_path) as czi:
            image_data = czi.asarray()
            print(f"Successfully loaded {file_path}")
            print(f"data shape: {image_data.shape}")
            return image_data

    except Exception as e:
        print(f"Error loading CZI file: {e}")
        return None

def get_labels_in_radius(binary_mask, segmentation_map, radius):
    """
    Get all labels from segmentation map that are within a radius of the binary mask.

    Args:
        binary_mask (np.ndarray): Binary mask (2D array with 0s and 1s)
        segmentation_map (np.ndarray): Segmentation map with numeric labels
        radius (int): Radius in pixels to expand around the binary mask

    Returns:
        set: Set of unique labels found within the expanded region
    """
    # Find bounding box of the binary mask
    rows, cols = np.where(binary_mask > 0)

    if len(rows) == 0:
        # Empty mask, return empty set
        return []

    # Get bounding box coordinates
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    # Expand bounding box by radius, considering image boundaries
    height, width = segmentation_map.shape

    expanded_min_row = max(0, min_row - radius)
    expanded_max_row = min(height - 1, max_row + radius)
    expanded_min_col = max(0, min_col - radius)
    expanded_max_col = min(width - 1, max_col + radius)

    # Extract the region from segmentation map
    region = segmentation_map[expanded_min_row:expanded_max_row + 1,
                              expanded_min_col:expanded_max_col + 1]

    # Get unique labels in the region (excluding 0 which is background)
    unique_labels = np.unique(region)
    labels_in_radius = unique_labels[unique_labels > 0].tolist()

    return labels_in_radius


def max_intensity_projection(stack):
    """
    Computes the maximum intensity projection along the z-axis.

    Parameters:
        stack (numpy.ndarray): A 4D numpy array of shape (Z, T, H, W),
                               where Z is the number of slices, T is time.

    Returns:
        numpy.ndarray: A 3D array (Z, H, W) containing the max projection over time.
    """
    return np.max(stack, axis=1)


def mean_intensity_projection(stack):
    """
    Computes the mean intensity projection along the z-axis.

    Parameters:
        stack (numpy.ndarray): A 4D numpy array of shape (Z, T, H, W),
                               where Z is the number of slices, T is time.

    Returns:
        numpy.ndarray: A 3D array (Z, H, W) containing the mean projection over time.
    """
    return np.mean(stack, axis=1)
