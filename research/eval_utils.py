import numpy as np


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
