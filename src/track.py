import numpy as np
from tqdm import tqdm
import networkx as nx
from scipy.ndimage import center_of_mass

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
    
    # Get nodes from graphs
    nodes1 = list(g1.nodes())
    nodes2 = list(g2.nodes())
    
    matches = {}
    used_nodes1 = set()
    
    # For each cell in frame 2, find the closest cell in frame 1
    for cell2 in nodes2:
        if cell2 == 0:  # Skip background
            continue
            
        center2 = labels_to_pos2[cell2]
        min_distance = float('inf')
        best_match = None
        
        for cell1 in nodes1:
            if cell1 == 0 or cell1 in used_nodes1:  # Skip background or already matched cells
                continue
                
            center1 = labels_to_pos1[cell1]
            
            # Calculate 3D Euclidean distance
            distance = np.sqrt((center1[0] - center2[0])**2 + 
                             (center1[1] - center2[1])**2 + 
                             (center1[2] - center2[2])**2)
            
            if distance < min_distance and distance <= distance_threshold:
                min_distance = distance
                best_match = cell1
        
        # Record the match if found
        if best_match is not None:
            matches[cell2] = best_match
            used_nodes1.add(best_match)
    
    return matches

def match_points_with_graph_features(g1: nx.Graph, g2: nx.Graph, mask1: np.ndarray, mask2: np.ndarray,
                                   distance_threshold: float = np.sqrt(3), degree_weight: float = 0.3) -> dict:
    """
    Enhanced point matching using both spatial proximity and graph features (node degree).
    
    Parameters:
        g1 (nx.Graph): Adjacency graph for frame 1
        g2 (nx.Graph): Adjacency graph for frame 2
        mask1 (np.ndarray): Segmentation mask for frame 1
        mask2 (np.ndarray): Segmentation mask for frame 2
        distance_threshold (float): Maximum distance for matching points
        degree_weight (float): Weight for degree similarity in matching score
        
    Returns:
        dict: Mapping from frame2 cell IDs to frame1 cell IDs
    """
    # Get cell centers for both frames
    centers1 = get_cell_centers(mask1)
    centers2 = get_cell_centers(mask2)
    labels_to_pos1 = centers_array_to_label_position_map(centers1)
    labels_to_pos2 = centers_array_to_label_position_map(centers2)
    
    # Get nodes and their degrees
    nodes1 = list(g1.nodes())
    nodes2 = list(g2.nodes())
    degrees1 = dict(g1.degree())
    degrees2 = dict(g2.degree())
    
    matches = {}
    used_nodes1 = set()
    
    # For each cell in frame 2, find the best match in frame 1
    for cell2 in nodes2:
        if cell2 == 0:  # Skip background
            continue
            
        center2 = labels_to_pos2[cell2]
        degree2 = degrees2[cell2]
        
        best_score = float('inf')
        best_match = None
        
        for cell1 in nodes1:
            if cell1 == 0 or cell1 in used_nodes1:  # Skip background or already matched cells
                continue
                
            center1 = labels_to_pos1[cell1]
            degree1 = degrees1[cell1]
            
            # Calculate spatial distance
            spatial_distance = np.sqrt((center1[0] - center2[0])**2 + 
                                     (center1[1] - center2[1])**2 + 
                                     (center1[2] - center2[2])**2)
            
            if spatial_distance > distance_threshold:
                continue
            
            # Calculate degree similarity (normalized)
            max_degree = max(degree1, degree2, 1)  # Avoid division by zero
            degree_diff = abs(degree1 - degree2) / max_degree
            
            # Combined matching score (lower is better)
            matching_score = (1 - degree_weight) * spatial_distance + degree_weight * degree_diff * distance_threshold
            
            if matching_score < best_score:
                best_score = matching_score
                best_match = cell1
        
        # Record the match if found
        if best_match is not None:
            matches[cell2] = best_match
            used_nodes1.add(best_match)
    
    return matches

def find_key_for_last_tracklet_value_optimized(tracklets, matches_dict, tracklet_id):
    """
    Optimized version using next() with generator expression for early termination.
    """
    last_value = tracklets[tracklet_id][-1]
    
    # Use next() with generator for immediate return on first match
    return next((key for key, value in matches_dict.items() if value == last_value), -1)

def create_tracklets(matches:list) -> dict:
    matches_t0 = matches[0]
    
    # Initialize tracklets from first frame matches
    tracklets = {}
    for i, (label_t_plus1, label_t) in enumerate(matches_t0.items()):
        tracklets[i] = [int(label_t), int(label_t_plus1)]
    
    for i in tqdm(range(1, len(matches)), desc='Creating tracklets'):
        matches_t = matches[i]
        
        # Track which keys from current matches have been used
        used_keys = set()
        
        # First, extend existing tracklets
        for tracklet_id, labels in tracklets.items():
            key = find_key_for_last_tracklet_value_optimized(tracklets, matches_t, tracklet_id)
            if key != -1:  # Match found
                labels.append(int(key))
                used_keys.add(key)
        
        # Create new tracklets for unmatched cells
        max_id = max(tracklets.keys()) if tracklets else -1
        for key, value in matches_t.items():
            if key not in used_keys:
                max_id += 1
                # A new tracklet starts at time `i`, so pad with `i` placeholders.
                new_tracklet = [-1] * i + [int(value), int(key)]
                tracklets[max_id] = new_tracklet
    
    return tracklets