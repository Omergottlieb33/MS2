import re
import os
import json
import numpy as np
from tqdm import tqdm
from src.track import get_cell_centers, compute_cell_location, match_points_between_frames, \
create_tracklets, match_cells_by_iou_hungarian_local_optimized

def extract_time_number(filename):
    """Extract time number from filename like z_stack_t5_seg_masks.npz"""
    try:
        # Find the pattern t{number}_
        match = re.search(r't(\d+)_', filename)
        return int(match.group(1)) if match else 0
    except:
        return 0

def get_masks_paths(masks_dir:str) -> list:
    masks_paths = sorted([os.path.join(masks_dir, x) for x in os.listdir(masks_dir) 
                if x.startswith('z_stack_t') and x.endswith('_seg_masks.npz')], 
               key=extract_time_number)
    if masks_paths is None or len(masks_paths) == 0:
        raise ValueError(f'No masks found in {masks_dir} with expected naming convention.')
    return masks_paths

def get_adjaceny_graphs(masks:list, t:int) -> tuple:
    centers_list, g_list = [], [], []
    # get cell centers and adjacency graphs for each time point
    for i in tqdm(range(t),desc='calculating cell centers and adjacency graphs'):
        z_stack_seg_mask = masks[i]
        centers = get_cell_centers(z_stack_seg_mask)
        labels = np.unique(z_stack_seg_mask)
        g = compute_cell_location(centers=centers, labels=labels)
        centers_list.append(centers)
        g_list.append(g)
    return centers_list, g_list

def match_points_over_time_adjacency_graph(g_list:list, masks:list, t:int, distance_threshold=np.sqrt(3), degree_weight=0.3):
    matched_points = []
    for i in tqdm(range(t - 1), desc='matching points between frames'):
        g1 = g_list[i]
        g2 = g_list[i + 1]
        z_stack_seg_mask_t0 = masks[i]
        z_stack_seg_mask_t1 = masks[i + 1]
        matches = match_points_between_frames(
            g1, g2, z_stack_seg_mask_t0, z_stack_seg_mask_t1)
        matched_points.append(matches)
    return matched_points

def match_over_time_cell_iou(masks:list):
    """
    Match cells over time using IoU.
    Args:
        masks (list): List of segmentation masks for each time point.
    Returns:
        list: List of matched cells for each time point.
    """
    matched_cells = []
    for i in tqdm(range(len(masks) - 1), desc='matching cells by IoU'):
        z_stack_seg_mask_t0 = masks[i]
        z_stack_seg_mask_t1 = masks[i + 1]
        matches = match_cells_by_iou_hungarian_local_optimized(z_stack_seg_mask_t0, z_stack_seg_mask_t1)
        matched_cells.append(matches)
    return matched_cells



def cell_tracking(masks_dir:str, t:int) -> dict:
    """
    Perform cell tracking using adjacency graphs.
    Args:
        masks_dir (str): Directory containing segmentation masks.
        t (int): Number of time points to process. If None, processes all available masks.
        distance_threshold (float): Distance threshold for matching points between frames.
        degree_weight (float): Weight for degree similarity in matching score.
    Returns:
        dict: Tracklets mapping cell IDs across time points.
    """
    masks_paths = get_masks_paths(masks_dir)
    masks = []
    for i in range(len(masks_paths)):
        z_stack_seg_mask = np.load(masks_paths[i])['masks']
        masks.append(z_stack_seg_mask)
    if t<=0 or t> len(masks_paths):
        t = len(masks_paths) - 1
    # IoU matching
    matched_points = match_over_time_cell_iou(masks)
    # create tracklets
    tracklets = create_tracklets(matched_points)
    save_path = os.path.join(masks_dir, 'tracklets_matching_iou.json')
    with open(save_path, 'w') as f:
        json.dump(tracklets, f, indent=4)
    print(f'Tracklets saved to {save_path}')
    return tracklets

        

if __name__ == "__main__":
    czi_file_path = '/home/dafei/data/MS2/gRNA2_12.03.25-st-13-II---.czi'
    masks_dir = '/home/dafei/output/MS2/3d_cell_segmentation/gRNA2_12.03.25-st-13-II---/masks'
    t = 80  # Number of time points to process, set to None to process all available masks
    tracklets = cell_tracking(masks_dir, t=t)
    