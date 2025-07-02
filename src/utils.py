import czifile
import numpy as np


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

def get_dice_coefficient(mask1, mask2, smooth=1e-7):
    """
    Calculate the Dice coefficient between two binary masks.
    
    The Dice coefficient is defined as:
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        mask1 (np.ndarray): First binary mask
        mask2 (np.ndarray): Second binary mask
        smooth (float): Small constant to avoid division by zero
        
    Returns:
        float: Dice coefficient between 0 and 1, where 1 indicates perfect overlap
    """
    # Ensure masks are boolean
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    
    # Calculate intersection and union
    intersection = np.sum(mask1 & mask2)
    total = np.sum(mask1) + np.sum(mask2)
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (total + smooth)
    
    return dice

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
        
