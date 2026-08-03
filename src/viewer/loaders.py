"""Data loading for the segmentation viewer.

The cell channel arrives either as a 4D (T, Z, Y, X) tif, or as the raw czi that
cell_3d_segmentation.py segments.  Masks are the per-timepoint npz files that
same script writes.
"""
import functools
import os

import numpy as np
import tifffile

from cell_tracking import extract_time_number, get_masks_paths
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
