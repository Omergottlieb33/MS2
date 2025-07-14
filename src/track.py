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