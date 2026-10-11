import numpy as np
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

def get_border_labels(mask: np.ndarray, margin: int = 5) -> set:
    """Possible XY exits from mask contact, not just centroid proximity.

    Z contact alone is ambiguous in shallow stacks; leave it unresolved.
    """
    if margin < 1:
        return set()
    faces = [mask[:, :margin, :], mask[:, -margin:, :],
             mask[:, :, :margin], mask[:, :, -margin:]]
    return set(map(int, np.unique(np.concatenate([f.ravel() for f in faces])))) - {0}


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
