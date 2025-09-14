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

def find_key_for_last_tracklet_value_optimized(tracklets, matches_dict, tracklet_id):
    """
    Optimized version using next() with generator expression for early termination.
    """
    last_value = tracklets[tracklet_id][-1]
    
    # Use next() with generator for immediate return on first match
    return next((key for key, value in matches_dict.items() if value == last_value), -1)

def create_tracklets(matches:list) -> dict:
    if not matches:
        return {}

    matches_t0 = matches[0]
    
    # Initialize tracklets from first frame matches
    tracklets = {}
    for i, (label_t_plus1, label_t) in enumerate(matches_t0.items()):
        tracklets[i] = [int(label_t), int(label_t_plus1)]
    
    max_id = max(tracklets.keys()) if tracklets else -1

    for i in tqdm(range(1, len(matches)), desc='Creating tracklets'):
        matches_t = matches[i]
        
        # Track which keys from current matches have been used
        used_keys = set()
        next_labels = {}
        
        # First, determine the next label for all existing, active tracklets
        for tracklet_id, labels in tracklets.items():
            if labels[-1] != -1:  # If the track is active
                key = find_key_for_last_tracklet_value_optimized(tracklets, matches_t, tracklet_id)
                if key != -1:  # Match found
                    next_labels[tracklet_id] = int(key)
                    used_keys.add(key)
                else:  # No match found, terminate the track
                    next_labels[tracklet_id] = -1
            else:  # If the track was already terminated, keep it terminated
                next_labels[tracklet_id] = -1
        
        for tracklet_id, next_label in next_labels.items():
            tracklets[tracklet_id].append(next_label)

        # Create new tracklets for unmatched cells
        for key, value in matches_t.items():
            if key not in used_keys:
                max_id += 1
                # A new tracklet starts at time `i`, so pad with `i` placeholders.
                new_tracklet = [-1] * i + [int(value), int(key)]
                tracklets[max_id] = new_tracklet
    
    return tracklets

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

def match_cells_by_iou_hungarian_local_optimized(mask1: np.ndarray, mask2: np.ndarray,
                                               min_iou: float = 0.1,
                                               search_radius: int = 10,
                                               max_centroid_distance: float = None) -> dict: # type: ignore
    """
    Ultra-optimized local IoU matching with improved robustness and speed.

    This version is improved to be more robust to cell movement and significantly faster.

    Key improvements:
    1.  **Efficient Property Calculation:** Uses `scipy.ndimage.center_of_mass` and `np.bincount`
        to compute all cell centroids and volumes in a highly optimized, vectorized manner,
        avoiding slow Python loops.
    2.  **Robust Bounding Box:** The local region for IoU calculation is now a bounding box
        that encloses the centroids of *both* cells being compared. This makes the matching
        robust to cell movement, which was a primary failure point in the previous version.
    3.  **Clearer Parameters:** The relationship between `search_radius` and `max_centroid_distance`
        is critical. `max_centroid_distance` acts as a hard filter for candidate pairs, while
        `search_radius` defines the padding around the candidate pair's centroids to define
        the local region for IoU calculation.

    Parameters:
        mask1 (np.ndarray): Segmentation mask for frame 1.
        mask2 (np.ndarray): Segmentation mask for frame 2.
        min_iou (float): Minimum IoU threshold for a valid match. Defaults to 0.1.
        search_radius (int): Padding in pixels to add around the combined bounding box of two
                             candidate cell centroids to define the local search area. Defaults to 10.
        max_centroid_distance (float): The maximum distance between centroids for a pair of cells
                                       to be considered a potential match. If None, it defaults to
                                       a more generous value (`search_radius * 2.5`). A larger value
                                       allows for matching faster-moving cells.
        
    Returns:
        dict: Mapping from frame2 cell IDs to frame1 cell IDs.
    """
    
    if max_centroid_distance is None:
        # A more generous default than the previous version to account for movement.
        max_centroid_distance = search_radius * 2.5
    
    # Get unique cell labels (excluding background 0)
    labels1 = np.unique(mask1)
    labels1 = labels1[labels1 > 0]
    
    labels2 = np.unique(mask2)
    labels2 = labels2[labels2 > 0]
    
    if len(labels1) == 0 or len(labels2) == 0:
        return {}
    
    # --- 1. Optimized Property Calculation ---
    # Compute centroids for all labels at once. Note: center_of_mass returns (z, y, x).
    com1 = center_of_mass(mask1, mask1, labels1)
    com2 = center_of_mass(mask2, mask2, labels2)
    
    # Create a dictionary mapping label to centroid, filtering out any NaNs
    centroids1 = {int(label): center for label, center in zip(labels1, com1) if not np.isnan(center).any()}
    centroids2 = {int(label): center for label, center in zip(labels2, com2) if not np.isnan(center).any()}

    # Compute volumes (pixel counts) for all labels at once using np.bincount.
    max_label = max(np.max(labels1) if len(labels1) > 0 else 0, 
                    np.max(labels2) if len(labels2) > 0 else 0)
    vols1_all = np.bincount(mask1.ravel(), minlength=max_label + 1)
    vols2_all = np.bincount(mask2.ravel(), minlength=max_label + 1)
    
    volumes1 = {int(label): vols1_all[label] for label in centroids1.keys()}
    volumes2 = {int(label): vols2_all[label] for label in centroids2.keys()}

    valid_cells1 = sorted(list(centroids1.keys()))
    valid_cells2 = sorted(list(centroids2.keys()))
    
    if not valid_cells1 or not valid_cells2:
        return {}
    
    # --- 2. Pre-filter pairs based on centroid distance ---
    valid_pairs = []
    for i, cell1 in enumerate(valid_cells1):
        c1 = np.array(centroids1[cell1])
        for j, cell2 in enumerate(valid_cells2):
            c2 = np.array(centroids2[cell2])
            # Manual distance calculation for clarity and consistency
            distance = np.sqrt(np.sum((c1 - c2)**2))
            if distance <= max_centroid_distance:
                valid_pairs.append((i, j, cell1, cell2))
    
    # Create cost matrix (1 - IoU)
    n1, n2 = len(valid_cells1), len(valid_cells2)
    cost_matrix = np.full((n1, n2), 1.0)
    
    # --- 3. Calculate IoU only for valid pairs in a robust local region ---
    for i, j, cell1, cell2 in valid_pairs:
        c1 = centroids1[cell1]
        c2 = centroids2[cell2]
        
        # --- Robust Bounding Box Definition ---
        # Define a bounding box that encloses both centroids, plus padding.
        z_min = max(0, int(min(c1[0], c2[0]) - search_radius))
        z_max = min(mask1.shape[0], int(max(c1[0], c2[0]) + search_radius) + 1)
        y_min = max(0, int(min(c1[1], c2[1]) - search_radius))
        y_max = min(mask1.shape[1], int(max(c1[1], c2[1]) + search_radius) + 1)
        x_min = max(0, int(min(c1[2], c2[2]) - search_radius))
        x_max = min(mask1.shape[2], int(max(c1[2], c2[2]) + search_radius) + 1)
        
        # Extract local regions
        local_mask1 = mask1[z_min:z_max, y_min:y_max, x_min:x_max]
        local_mask2 = mask2[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Calculate intersection in the local region
        intersection = np.sum((local_mask1 == cell1) & (local_mask2 == cell2))
        
        if intersection > 0:
            # Get volumes from pre-computed values
            vol1 = volumes1[cell1]
            vol2 = volumes2[cell2]
            union = vol1 + vol2 - intersection
            
            if union > 0:
                iou = intersection / union
                cost_matrix[i, j] = 1.0 - iou
    
    # --- 4. Apply Hungarian algorithm to find optimal assignment ---
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Extract valid matches that meet the IoU threshold
    matches = {}
    for i, j in zip(row_indices, col_indices):
        iou = 1.0 - cost_matrix[i, j]
        if iou >= min_iou:
            cell1 = valid_cells1[i]
            cell2 = valid_cells2[j]
            matches[cell2] = cell1
    
    return matches