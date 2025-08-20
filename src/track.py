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
    # Get cell centers for both frames
    centers1 = get_cell_centers(mask1)
    centers2 = get_cell_centers(mask2)
    labels_to_pos1 = centers_array_to_label_position_map(centers1)
    labels_to_pos2 = centers_array_to_label_position_map(centers2)
    
    # Get valid cell labels (nodes) from graphs, excluding background (0)
    nodes1 = [n for n in g1.nodes() if n != 0 and n in labels_to_pos1]
    nodes2 = [n for n in g2.nodes() if n != 0 and n in labels_to_pos2]

    if not nodes1 or not nodes2:
        return {}

    # For performance, extract positions into numpy arrays
    pos1 = np.array([labels_to_pos1[n] for n in nodes1])
    pos2 = np.array([labels_to_pos2[n] for n in nodes2])

    # Calculate the full pairwise distance matrix using vectorized operations (broadcasting).
    # This is much faster than nested loops for large numbers of cells.
    diff = pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]
    #TODO: add cell graph degree to the distance metric, area metric, shape metric, etc.
    cost_matrix = np.sqrt(np.sum(diff**2, axis=2))

    # Use the Hungarian algorithm (linear_sum_assignment) to find the optimal assignment
    # that minimizes the total distance.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = {}
    # Create the matches dictionary from the optimal assignments, but only include
    # pairs where the distance is within the specified threshold.
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= distance_threshold:
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
                                               max_centroid_distance: float = None) -> dict:
    """
    Ultra-optimized local IoU matching with additional speedups.
    """
    from scipy.ndimage import center_of_mass
    
    if max_centroid_distance is None:
        max_centroid_distance = search_radius * 1.5
    
    # Get unique cell labels
    cells1 = np.unique(mask1)[1:]
    cells2 = np.unique(mask2)[1:]
    
    if len(cells1) == 0 or len(cells2) == 0:
        return {}
    
    # Compute centroids and volumes simultaneously
    centroids1, volumes1 = {}, {}
    centroids2, volumes2 = {}, {}
    
    for cell in cells1:
        cell_coords = np.where(mask1 == cell)
        if len(cell_coords[0]) > 0:
            centroid = (np.mean(cell_coords[0]), np.mean(cell_coords[1]), np.mean(cell_coords[2]))
            centroids1[cell] = centroid
            volumes1[cell] = len(cell_coords[0])
    
    for cell in cells2:
        cell_coords = np.where(mask2 == cell)
        if len(cell_coords[0]) > 0:
            centroid = (np.mean(cell_coords[0]), np.mean(cell_coords[1]), np.mean(cell_coords[2]))
            centroids2[cell] = centroid
            volumes2[cell] = len(cell_coords[0])
    
    valid_cells1 = list(centroids1.keys())
    valid_cells2 = list(centroids2.keys())
    
    if len(valid_cells1) == 0 or len(valid_cells2) == 0:
        return {}
    
    # Pre-filter pairs based on centroid distance
    valid_pairs = []
    for i, cell1 in enumerate(valid_cells1):
        c1 = centroids1[cell1]
        for j, cell2 in enumerate(valid_cells2):
            c2 = centroids2[cell2]
            distance = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)
            if distance <= max_centroid_distance:
                valid_pairs.append((i, j, cell1, cell2))
    
    # Create sparse cost matrix
    n1, n2 = len(valid_cells1), len(valid_cells2)
    cost_matrix = np.full((n1, n2), 1.0)
    
    # Calculate IoU only for valid pairs
    for i, j, cell1, cell2 in valid_pairs:
        c1 = centroids1[cell1]
        
        # Define local bounding box
        z1, y1, x1 = int(c1[0]), int(c1[1]), int(c1[2])
        z_min = max(0, z1 - search_radius)
        z_max = min(mask1.shape[0], z1 + search_radius + 1)
        y_min = max(0, y1 - search_radius)
        y_max = min(mask1.shape[1], y1 + search_radius + 1)
        x_min = max(0, x1 - search_radius)
        x_max = min(mask1.shape[2], x1 + search_radius + 1)
        
        # Extract local regions (much smaller than full masks)
        local_mask1 = mask1[z_min:z_max, y_min:y_max, x_min:x_max]
        local_mask2 = mask2[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Calculate intersection in local region
        intersection = np.sum((local_mask1 == cell1) & (local_mask2 == cell2))
        
        if intersection > 0:
            # Get volumes from pre-computed values
            vol1 = volumes1[cell1]
            vol2 = volumes2[cell2]
            union = vol1 + vol2 - intersection
            
            if union > 0:
                iou = intersection / union
                cost_matrix[i, j] = 1.0 - iou
    
    # Apply Hungarian algorithm
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Extract valid matches
    matches = {}
    for i, j in zip(row_indices, col_indices):
        iou = 1.0 - cost_matrix[i, j]
        if iou >= min_iou:
            cell1 = valid_cells1[i]
            cell2 = valid_cells2[j]
            matches[cell2] = cell1
    
    return matches