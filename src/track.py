import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy.ndimage import center_of_mass
from scipy.optimize import linear_sum_assignment

def get_cell_centers(masks_array):
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

def centers_array_to_label_position_map(centers: np.ndarray) -> dict:
    # Create a dictionary to map labels to positions for fast lookup
    label_to_pos = {}
    for row in centers:
        label = int(row[0])
        pos = row[1:4]  # x, y, z coordinates
        label_to_pos[label] = pos
    return label_to_pos

def compute_cell_location(centers: np.ndarray, labels:np.array) -> nx.Graph:
    """
    Compute cell locations as a graph where nodes are cell labels and edges are distances between cells.
    """
    g = nx.Graph()
    
    label_to_pos = centers_array_to_label_position_map(centers)
    
    # Add nodes
    for label in labels:
        if label != 0 and label in label_to_pos:
            g.add_node(label)

    # Add edges with distances
    for i in labels:
        if i != 0 and i in label_to_pos:
            for j in labels:
                if j != 0 and j in label_to_pos and i != j:
                    pos1 = label_to_pos[i]
                    pos2 = label_to_pos[j]
                    distance = np.sqrt((pos1[0] - pos2[0])**2 +
                                       (pos1[1] - pos2[1])**2 +
                                       (pos1[2] - pos2[2])**2)
                    g.add_edge(i, j, weight=distance)
    
    return g
    

def match_points_between_frames(g1: nx.Graph, g2: nx.Graph, mask1: np.ndarray, mask2: np.ndarray, 
                               distance_threshold: float = np.sqrt(3)) -> dict:
    """
    Match points (cells) between consecutive frames using adjacency graphs and spatial proximity.
    
    Parameters:
        g1 (nx.Graph): Adjacency graph for frame 1
        g2 (nx.Graph): Adjacency graph for frame 2
        mask1 (np.ndarray): Segmentation mask for frame 1
        mask2 (np.ndarray): Segmentation mask for frame 2
        distance_threshold (float): Maximum distance for matching points
        
    Returns:
        dict: Mapping from frame2 cell IDs to frame1 cell IDs {cell_id_t2: cell_id_t1}
    """
    # --- 1. Get cell centers and volumes for both frames ---
    centers1 = get_cell_centers(mask1)
    centers2 = get_cell_centers(mask2)
    labels_to_pos1 = centers_array_to_label_position_map(centers1)
    labels_to_pos2 = centers_array_to_label_position_map(centers2)

    # Efficiently compute volumes (voxel counts) for all cells
    labels1_all = np.unique(mask1)
    labels1_all = labels1_all[labels1_all > 0]
    labels2_all = np.unique(mask2)
    labels2_all = labels2_all[labels2_all > 0]

    volumes1, volumes2 = {}, {}
    if len(labels1_all) > 0 and len(labels2_all) > 0:
        max_label = max(np.max(labels1_all), np.max(labels2_all))
        vols1_all = np.bincount(mask1.ravel(), minlength=max_label + 1)
        vols2_all = np.bincount(mask2.ravel(), minlength=max_label + 1)
        volumes1 = {int(label): vols1_all[label] for label in labels1_all}
        volumes2 = {int(label): vols2_all[label] for label in labels2_all}

    # Get valid cell labels (nodes) from graphs, excluding background (0)
    nodes1 = [n for n in g1.nodes() if n != 0 and n in labels_to_pos1 and n in volumes1]
    nodes2 = [n for n in g2.nodes() if n != 0 and n in labels_to_pos2 and n in volumes2]

    if not nodes1 or not nodes2:
        return {}

    # --- 2. Prepare data for vectorized calculations ---
    # For performance, extract positions and volumes into numpy arrays
    pos1 = np.array([labels_to_pos1[n] for n in nodes1])
    pos2 = np.array([labels_to_pos2[n] for n in nodes2])
    vol1 = np.array([volumes1[n] for n in nodes1])
    vol2 = np.array([volumes2[n] for n in nodes2])

    # --- 3. Calculate cost matrix with multiple metrics ---
    # Calculate the full pairwise distance matrix using vectorized operations (broadcasting).
    diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
    distance_cost = np.sqrt(np.sum(diff**2, axis=2))

    # Calculate a volume difference cost. This penalizes matches between cells of different sizes.
    # We normalize by the volume of the first cell to get a relative size change.
    vol_diff = np.abs(vol1[:, np.newaxis] - vol2[np.newaxis, :])
    volume_cost = vol_diff / (vol1[:, np.newaxis] + 1e-6) # Add epsilon to avoid division by zero

    # Combine costs with weights. These can be tuned.
    # Here, we prioritize distance but also strongly consider volume similarity.
    w_dist = 0.7
    w_vol = 0.3
    cost_matrix = (w_dist * (distance_cost / distance_threshold)) + (w_vol * volume_cost)

    # --- 4. Find optimal assignment using the Hungarian algorithm ---
    # that minimizes the total distance.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}
    # Create the matches dictionary from the optimal assignments, but only include
    # pairs where the distance is within the specified threshold.
    for r, c in zip(row_ind, col_ind):
        # The final check is still on the absolute physical distance, not the combined cost.
        if distance_cost[r, c] <= distance_threshold:
            cell1_label = nodes1[r]
            cell2_label = nodes2[c]
            matches[cell2_label] = cell1_label

    return matches

def get_border_labels(mask: np.ndarray, margin: int = 5) -> set:
    """Possible XY exits from mask contact, not just centroid proximity.

    Z contact alone is ambiguous in shallow stacks; leave it unresolved.
    """
    if margin < 1:
        return set()
    faces = [mask[:, :margin, :], mask[:, -margin:, :],
             mask[:, :, :margin], mask[:, :, -margin:]]
    return set(map(int, np.unique(np.concatenate([f.ravel() for f in faces])))) - {0}


def find_key_for_last_tracklet_value_optimized(tracklets, matches_dict, tracklet_id):
    """
    Optimized version using next() with generator expression for early termination.
    """
    last_value = tracklets[tracklet_id][-1]
    
    # Use next() with generator for immediate return on first match
    return next((key for key, value in matches_dict.items() if value == last_value), -1)

def create_tracklets(matches: list, skip_matches: list = None,
                     masks: list = None, border_margin: int = 5) -> dict:
    """Track every detection, preserving gaps without assuming biological death.

    With masks, skip recovery is reassigned over only missing sources and free
    targets. Border exits are annotated after recovery, never used to block it.
    Without masks, only labels present in the supplied matches are knowable.
    """
    n = len(masks) if masks is not None else len(matches) + 1
    if n == 0:
        return {}
    if len(matches) != n - 1:
        raise ValueError('Expected one match dictionary per consecutive frame pair')
    detections = [set() for _ in range(n)]
    if masks is not None:
        detections = [set(map(int, np.unique(m))) - {0} for m in masks]
    else:
        for t, pair in enumerate(matches):
            detections[t].update(map(int, pair.values()))
            detections[t + 1].update(map(int, pair))
        for t, pair in enumerate(skip_matches or []):
            detections[t].update(map(int, pair.values()))
            detections[t + 2].update(map(int, pair))
    tracks = {}
    for t in range(n):
        used = set()
        previous = {row[t-1]: tid for tid, row in tracks.items()
                    if t and row[t-1] > 0}
        if t:
            for target, source in matches[t-1].items():
                if source in previous and target in detections[t]:
                    tracks[previous[source]][t] = int(target)
                    used.add(int(target))
        if t >= 2 and skip_matches is not None:
            missing = {row[t-2]: tid for tid, row in tracks.items()
                       if row[t-2] > 0 and row[t-1] == -1}
            free = detections[t] - used
            if masks is not None and missing and free:
                pair = match_cells_by_iou_hungarian_local_optimized(
                    masks[t-2], masks[t], max_centroid_distance=15 * np.sqrt(2),
                    source_labels=missing, target_labels=free)
            else:
                pair = skip_matches[t-2] if t-2 < len(skip_matches) else {}
            for target, source in pair.items():
                if source in missing and target in free:
                    tracks[missing[source]][t] = int(target)
                    used.add(int(target))
        for label in sorted(detections[t] - used):
            tracks[len(tracks)] = [-1] * n
            tracks[len(tracks)-1][t] = int(label)
    if masks is not None:
        borders = [get_border_labels(m, border_margin) for m in masks]
        for row in tracks.values():
            last = max(i for i, label in enumerate(row) if label > 0)
            if last < n-1 and row[last] in borders[last]:
                row[last+1:] = [-2] * (n-last-1)
    return tracks


def match_cells_by_iou(mask1: np.ndarray, mask2: np.ndarray,
                      min_iou: float = 0.3) -> dict:
    """
    Match cells using Intersection over Union (IoU) metric.
    
    Parameters:
        mask1 (np.ndarray): Segmentation mask for frame 1
        mask2 (np.ndarray): Segmentation mask for frame 2
        min_iou (float): Minimum IoU threshold for valid matches
        
    Returns:
        dict: Mapping from frame2 cell IDs to frame1 cell IDs
    """
    cells1 = np.unique(mask1)[1:]
    cells2 = np.unique(mask2)[1:]
    
    if len(cells1) == 0 or len(cells2) == 0:
        return {}
    
    matches = {}
    
    for cell2 in cells2:
        cell2_mask = (mask2 == cell2)
        
        best_match = None
        best_iou = 0
        
        for cell1 in cells1:
            cell1_mask = (mask1 == cell1)
            
            # Calculate IoU
            intersection = np.sum(cell1_mask & cell2_mask)
            union = np.sum(cell1_mask | cell2_mask)
            
            if union > 0:
                iou = intersection / union
                
                if iou >= min_iou and iou > best_iou:
                    best_match = cell1
                    best_iou = iou
        
        if best_match is not None:
            matches[cell2] = best_match
    
    return matches

def match_cells_by_iou_hungarian_local(mask1: np.ndarray, mask2: np.ndarray,
                                     min_iou: float = 0.1,
                                     search_radius: int = 10) -> dict:
    """
    Fast IoU-based matching using Hungarian algorithm with local search optimization.
    
    Parameters:
        mask1 (np.ndarray): Segmentation mask for frame 1
        mask2 (np.ndarray): Segmentation mask for frame 2
        min_iou (float): Minimum IoU threshold for valid matches
        search_radius (int): Search radius around cell centroid in pixels
        
    Returns:
        dict: Mapping from frame2 cell IDs to frame1 cell IDs
    """
    
    # Get unique cell labels (excluding background)
    cells1 = np.unique(mask1)[1:]
    cells2 = np.unique(mask2)[1:]
    
    if len(cells1) == 0 or len(cells2) == 0:
        return {}
    
    # Pre-compute centroids for all cells
    centroids1 = {}
    centroids2 = {}
    
    for cell in cells1:
        cell_mask = (mask1 == cell)
        if np.any(cell_mask):
            centroid = center_of_mass(cell_mask)
            centroids1[cell] = tuple(int(c) for c in centroid)
    
    for cell in cells2:
        cell_mask = (mask2 == cell)
        if np.any(cell_mask):
            centroid = center_of_mass(cell_mask)
            centroids2[cell] = tuple(int(c) for c in centroid)
    
    # Filter cells that have valid centroids
    valid_cells1 = [c for c in cells1 if c in centroids1]
    valid_cells2 = [c for c in cells2 if c in centroids2]
    
    if len(valid_cells1) == 0 or len(valid_cells2) == 0:
        return {}
    
    # Create cost matrix
    n1, n2 = len(valid_cells1), len(valid_cells2)
    cost_matrix = np.full((n1, n2), 1.0)
    
    # Calculate local IoU for each pair
    for i, cell1 in enumerate(valid_cells1):
        centroid1 = centroids1[cell1]
        
        # Define local search region around cell1's centroid
        z1, y1, x1 = centroid1
        z_min = max(0, z1 - search_radius)
        z_max = min(mask1.shape[0], z1 + search_radius + 1)
        y_min = max(0, y1 - search_radius)
        y_max = min(mask1.shape[1], y1 + search_radius + 1)
        x_min = max(0, x1 - search_radius)
        x_max = min(mask1.shape[2], x1 + search_radius + 1)
        
        # Extract local regions
        local_mask1 = mask1[z_min:z_max, y_min:y_max, x_min:x_max]
        local_mask2 = mask2[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Create cell1 mask in local region
        cell1_local_mask = (local_mask1 == cell1)
        cell1_volume = np.sum(cell1_local_mask)
        
        if cell1_volume == 0:
            continue
        
        for j, cell2 in enumerate(valid_cells2):
            centroid2 = centroids2[cell2]
            
            # Quick distance check - skip if centroids are too far apart
            z2, y2, x2 = centroid2
            centroid_distance = np.sqrt((z1-z2)**2 + (y1-y2)**2 + (x1-x2)**2)
            if centroid_distance > search_radius * 2:
                cost_matrix[i, j] = 1.0
                continue
            
            # Create cell2 mask in local region
            cell2_local_mask = (local_mask2 == cell2)
            cell2_volume = np.sum(cell2_local_mask)
            
            if cell2_volume == 0:
                cost_matrix[i, j] = 1.0
                continue
            
            # Calculate IoU in local region
            intersection = np.sum(cell1_local_mask & cell2_local_mask)
            union = cell1_volume + cell2_volume - intersection
            
            if union > 0:
                iou = intersection / union
                cost_matrix[i, j] = 1.0 - iou
            else:
                cost_matrix[i, j] = 1.0
    
    # Apply Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Extract matches that meet IoU threshold
    matches = {}
    for i, j in zip(row_indices, col_indices):
        iou = 1.0 - cost_matrix[i, j]
        if iou >= min_iou:
            cell1 = valid_cells1[i]
            cell2 = valid_cells2[j]
            matches[cell2] = cell1
    
    return matches

def optional_assignment(cost, unmatched_cost=0.85):
    """Minimum-cost matching with a combined unmatched-pair cost threshold.

    Each source has its own dummy column. Free target columns implicitly cost
    zero; this is equivalent to charging half the threshold to each unmatched
    endpoint. Forbidden edges are excluded before solving.
    """
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or not np.isfinite(unmatched_cost) or unmatched_cost <= 0:
        raise ValueError('Expected a matrix and a positive finite unmatched cost')
    n, m = cost.shape
    if not n or not m:
        return []
    augmented = np.full((n, m+n), np.inf)
    augmented[:, :m] = np.where(np.isfinite(cost) & (cost < unmatched_cost), cost, np.inf)
    augmented[np.arange(n), m+np.arange(n)] = unmatched_cost
    rows, cols = linear_sum_assignment(augmented)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if c < m]


def match_cells_by_iou_hungarian_local_optimized(mask1: np.ndarray, mask2: np.ndarray,
                                               min_iou: float = 0.0,
                                               search_radius: int = 10,
                                               max_centroid_distance: float = None,
                                               use_2d_distance: bool = False,
                                               dist_weight: float = 0.3,
                                               voxel_scale=(2.52, 1.0, 1.0),
                                               unmatched_cost: float = 0.85,
                                               source_labels=None, target_labels=None) -> dict:
    """Exact full-volume IoU with scaled 3D distance and optional assignments.

    Distances are in XY-pixel units with the default microscope voxel scale.
    Zero-overlap links use the same continuous cost as overlapping links; weak
    candidates can remain unmatched. search_radius remains for API compatibility
    and sets the default distance gate, but no longer crops intersections.
    """
    if mask1.shape != mask2.shape or mask1.ndim != 3:
        raise ValueError('Masks must have identical (Z,Y,X) shapes')
    gate = search_radius * 2.5 if max_centroid_distance is None else max_centroid_distance
    if gate <= 0 or not 0 <= min_iou <= 1 or not 0 <= dist_weight <= 1:
        raise ValueError('Invalid distance gate, IoU threshold, or distance weight')
    scale = np.asarray(voxel_scale, dtype=float)
    if scale.shape != (3,) or not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError('voxel_scale must contain three positive finite values')
    labels1 = np.unique(mask1); labels1 = labels1[labels1 > 0]
    labels2 = np.unique(mask2); labels2 = labels2[labels2 > 0]
    if source_labels is not None:
        labels1 = np.intersect1d(labels1, list(source_labels))
    if target_labels is not None:
        labels2 = np.intersect1d(labels2, list(target_labels))
    if not len(labels1) or not len(labels2):
        return {}
    c1 = np.asarray(center_of_mass(mask1, mask1, labels1))
    c2 = np.asarray(center_of_mass(mask2, mask2, labels2))
    delta = (c1[:, None] - c2[None, :]) * scale
    if use_2d_distance:
        delta = delta[..., 1:]
    distances = np.linalg.norm(delta, axis=-1)
    v1 = np.bincount(mask1.ravel())[labels1]
    v2 = np.bincount(mask2.ravel())[labels2]
    # Accumulate exact intersections without a dense max_label**2 allocation.
    both = (mask1 > 0) & (mask2 > 0)
    stride = int(mask2.max()) + 1
    keys, counts = np.unique(mask1[both].astype(np.int64) * stride + mask2[both],
                             return_counts=True)
    overlap = dict(zip(keys.tolist(), counts.tolist()))
    intersections = np.array([[overlap.get(int(a)*stride+int(b), 0)
                               for b in labels2] for a in labels1], dtype=float)
    iou = intersections / (v1[:, None] + v2[None, :] - intersections)
    cost = (1-dist_weight) * (1-iou) + dist_weight * distances / gate
    cost[(distances > gate) | (iou < min_iou)] = np.inf
    return {int(labels2[j]): int(labels1[i])
            for i, j in optional_assignment(cost, unmatched_cost)}
