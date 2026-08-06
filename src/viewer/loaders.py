"""Data loading for the segmentation viewer.

The cell channel arrives either as a 4D (T, Z, Y, X) tif, or as the raw czi that
cell_3d_segmentation.py segments.  Masks are the per-timepoint npz files that
same script writes.  Tracklets are the json create_tracklets() produces.
"""
import functools
import json
import os

import numpy as np
import tifffile

from cell_tracking import evaluate_tracklets, extract_time_number, get_masks_paths
from src.utils.cell_utils import calculate_center_of_mass_3d
from src.utils.image_utils import load_czi_images

# Matches cell_3d_segmentation.py:25 -- image_data[ti, 1, :, :, :]
CELL_CHANNEL = 1


def _as_tzyx(arr, channel):
    """Squeeze singleton axes and reduce to (T, Z, Y, X), picking a channel if present."""
    arr = np.squeeze(arr)
    if arr.ndim == 5:
        arr = arr[:, channel]
    if arr.ndim != 4:
        raise ValueError(
            f'expected a 4D (T, Z, Y, X) or 5D (T, C, Z, Y, X) image, got shape {arr.shape}')
    return arr


def load_image(path, channel=CELL_CHANNEL):
    """Load the cell channel of a tif or czi as a (T, Z, Y, X) array."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        try:
            arr = tifffile.memmap(path)
        except (ValueError, MemoryError):
            arr = tifffile.imread(path)   # compressed tifs cannot be memory-mapped
    elif ext == '.czi':
        arr = load_czi_images(path)
        if arr is None:
            raise ValueError(f'could not read czi: {path}')
    else:
        raise ValueError(f'unsupported image format: {ext}')
    return _as_tzyx(arr, channel)


@functools.lru_cache(maxsize=8)
def _load_masks_file(path):
    """npz holds masks plus two flow arrays; indexing decompresses only the masks."""
    with np.load(path, allow_pickle=True) as f:
        return f['masks']


class MaskStore:
    """Per-timepoint masks, keyed by the t number in the filename rather than by
    position, so a missing timepoint does not shift every later frame."""

    def __init__(self, masks_dir):
        self.by_t = {extract_time_number(os.path.basename(p)): p
                     for p in get_masks_paths(masks_dir)}

    def get(self, t):
        """Masks for timepoint t as (Z, Y, X), or None if that timepoint is missing."""
        path = self.by_t.get(t)
        return _load_masks_file(path) if path is not None else None


def locate(volume, label):
    """Where a label sits in a (Z, Y, X) volume, as (z, cy, cx).

    Centroid rather than the slice holding the most pixels: these cells span most of the
    stack with a near-flat z profile, so the peak slice beats its neighbours by only a few
    percent and picks a different winner every frame on segmentation noise alone -- the z
    slider jumps around while the cell is not actually moving.  Averaging the whole profile
    is stable, and still follows real z motion.
    """
    com = calculate_center_of_mass_3d(volume == label)
    if com is None:
        return None
    cx, cy, cz = com    # returns (x, y, z), despite what its docstring claims
    return int(round(cz)), float(cy), float(cx)


# Sentinels written by create_tracklets(); see src/track.py:191.
GAP, EXITED = -1, -2


class TrackletStore:
    """Tracklets as {id: [label per frame]}, with the reverse (frame, label) -> id
    lookup that click-to-select needs."""

    SORTS = ('n_active', 'n_gaps', 'n_resurrections', 'span', 'tid')

    def __init__(self, path):
        with open(path) as f:
            raw = json.load(f)
        self.tracks = {int(k): v for k, v in raw.items()}
        self.n_frames = len(next(iter(self.tracks.values())))
        self.stats = evaluate_tracklets(raw)

        # (frame, label) -> tracklet is unique in practice; last write wins if it ever isn't.
        self.by_frame = [{} for _ in range(self.n_frames)]
        for tid, labels in self.tracks.items():
            for t, label in enumerate(labels):
                if label > 0:
                    self.by_frame[t][label] = tid

    def state_at(self, tid, t):
        """(label, state) for a track at a timepoint.  The two sentinels are kept
        apart: a gap is a tracking failure, a border exit is expected behaviour."""
        labels = self.tracks.get(tid)
        if labels is None or not 0 <= t < len(labels):
            return 0, 'outside'
        label = labels[t]
        if label > 0:
            return label, 'active'
        return 0, 'exited' if label == EXITED else 'gap'

    def tracklet_of(self, label, t):
        if not 0 <= t < self.n_frames:
            return None
        return self.by_frame[t].get(label)

    def timeline(self, tid):
        return [self.state_at(tid, t)[1] for t in range(self.n_frames)]

    def summary(self, sort='n_active', limit=200, ascending=False):
        if sort not in self.SORTS:
            sort = 'n_active'
        if sort == 'tid':
            # the index holds the raw json keys, which are strings -- sorting them
            # directly would put '10' before '2'
            order = np.argsort(self.stats.index.astype(int), kind='stable')
            if not ascending:
                order = order[::-1]
            top = self.stats.iloc[order[:limit]]
        else:
            top = self.stats.sort_values(sort, ascending=ascending).head(limit)
        return [{'tid': int(tid),
                 'span': int(r.span),
                 'n_active': int(r.n_active),
                 'n_gaps': int(r.n_gaps),
                 'n_resurrections': int(r.n_resurrections),
                 'border_exit': bool(r.border_exit)}
                for tid, r in top.iterrows()]
