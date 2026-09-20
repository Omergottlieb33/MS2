"""Spatial selection of cells for MS2 analysis.

The user draws a closed boundary on the cell-channel projection at the first frame and
again at the last.  A tracklet is selected if it lies inside *either* boundary -- a cell
that has not appeared yet at the first frame still counts if it is inside the boundary at
the last one, because the identity being collected is the tracklet, not the per-frame label.
"""
import json
import os

import numpy as np
import tifffile
from matplotlib.path import Path

from src.track import get_cell_centers


def load_projection(path):
    """The (T, Y, X) cell-channel projection the boundaries are drawn on."""
    arr = tifffile.memmap(path)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f'expected a (T, Y, X) projection, got shape {arr.shape}')
    return arr


def cell_centroids(mask_volume):
    """Every label in a (Z, Y, X) volume with its centroid, as (labels, xy).

    get_cell_centers returns [label, x, y, z] for all labels in one pass, which is what
    makes a whole-frame selection cheap enough to run interactively.
    """
    centers = get_cell_centers(mask_volume)
    if len(centers) == 0:
        return np.empty(0, dtype=int), np.empty((0, 2))
    return centers[:, 0].astype(int), centers[:, 1:3]


def labels_inside(xy, polygon):
    """Boolean mask of which points fall within a closed polygon of (x, y) vertices."""
    if len(polygon) < 3 or len(xy) == 0:
        return np.zeros(len(xy), dtype=bool)
    return Path(np.asarray(polygon, dtype=float)).contains_points(xy)


def tracklets_in_polygon(tracks, mask_volume, polygon, t):
    """Tracklet ids whose cell centroid at frame t falls inside the polygon."""
    labels, xy = cell_centroids(mask_volume)
    hits = labels[labels_inside(xy, polygon)]
    found = {tracks.tracklet_of(int(label), t) for label in hits}
    found.discard(None)
    return found


def select_tracklets(tracks, masks, roi_first, roi_last, t_first, t_last):
    """Union of the tracklets caught by each boundary.

    Either polygon may be empty, in which case it simply contributes nothing -- selecting
    only on the last frame is a legitimate way to pick up cells that do not exist yet at
    the first one.
    """
    selected = set()
    for polygon, t in ((roi_first, t_first), (roi_last, t_last)):
        if not polygon:
            continue
        volume = masks.get(t)
        if volume is None:
            continue
        selected |= tracklets_in_polygon(tracks, volume, polygon, t)
    return sorted(selected)


def save_roi(path, roi_first, roi_last, t_first, t_last, projection=None, tracklets=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({
            't_first': t_first, 't_last': t_last,
            'roi_first': roi_first, 'roi_last': roi_last,
            'projection': projection, 'tracklets': tracklets,
        }, f, indent=2)
    return path


def load_roi(path):
    with open(path) as f:
        return json.load(f)


def select_from_roi_file(roi_path, tracklets_path, masks_dir):
    """Tracklet ids for a saved ROI.  The entry point ms2_gene_expression.py uses."""
    from src.viewer.loaders import MaskStore, TrackletStore
    roi = load_roi(roi_path)
    tracks = TrackletStore(tracklets_path)
    t_last = roi.get('t_last')
    if t_last is None or t_last < 0:
        t_last = tracks.n_frames - 1
    return select_tracklets(
        tracks, MaskStore(masks_dir),
        roi.get('roi_first'), roi.get('roi_last'),
        roi.get('t_first', 0), t_last)
