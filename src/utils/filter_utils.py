from scipy.ndimage import median_filter
import numpy as np
from tqdm import tqdm

def median_filter_over_time(stack, kernel_size=3):
    t, z, h, w = stack.shape
    filtered_stack = np.zeros_like(stack)
    for i in tqdm(range(z), desc="Applying median filter over time"):
        filtered_stack[:, i, :, :] = median_filter(stack[:, i, :, :], size=kernel_size)
    return filtered_stack

def get_gene_expression_clusters(ms2_stack_t, ms2_stack_t_filtered, threshold):
    # Create a binary mask based on the threshold
    binary_mask = ms2_stack_t_filtered > threshold
    return ms2_stack_t*binary_mask  # Apply mask to the original stack