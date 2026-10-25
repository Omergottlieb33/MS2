"""ROI drawn before the pipeline: the SUM_C2 projection it is drawn on, boundaries on a chosen
frame window, the window reaching the activity step, and the viewer with no cells yet.

    python tests/test_roi.py
"""
import json
import os
import tempfile

import numpy as np

from fixtures import passed, write_recording

from ms2.pipeline import activity_window
from ms2.preprocess import sum_projection, write_projection
from ms2.roi_selection import load_projection, save_roi, select_from_roi_file
from ms2.viewer.app import Viewer

NAN = np.nan
# 4 cells on a 2x2 grid of 4x4 tiles; cell 3 (bottom right, x and y in 5..6) appears at frame 3
SIGNALS = [[1] * 5, [1] * 5, [1] * 5, [NAN, NAN, NAN, 1, 1]]
BOTTOM_RIGHT = [[4, 4], [8, 4], [8, 8], [4, 8]]


def test_projection_is_float32_sum_over_z_and_loads_back():
    stack = np.random.default_rng(0).integers(0, 255, (3, 4, 8, 9), dtype=np.uint8)
    proj = sum_projection(stack)
    assert proj.dtype == np.float32 and proj.shape == (3, 8, 9)
    assert np.array_equal(proj, stack.astype(np.float32).sum(axis=1))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'SUM_C2-rec.tif')
        write_projection(path, stack)
        assert np.array_equal(load_projection(path), proj)
    passed('SUM_C2 projection: float32 sum over Z, (T, Y, X) on disk')


def test_selection_uses_the_window_frames():
    with tempfile.TemporaryDirectory() as d:
        rec = write_recording(d, SIGNALS)
        roi = os.path.join(d, 'roi.json')
        save_roi(roi, BOTTOM_RIGHT, None, 0, 4)
        assert select_from_roi_file(roi, rec['tracklets'], rec['masks_dir']) == []
        save_roi(roi, BOTTOM_RIGHT, None, 3, 4)
        assert select_from_roi_file(roi, rec['tracklets'], rec['masks_dir']) == [3]
    passed('the start boundary is tested at the chosen start frame')


def test_activity_window_follows_the_roi():
    assert activity_window(None, 28) == (0, 27)
    with tempfile.TemporaryDirectory() as d:
        roi = os.path.join(d, 'roi.json')
        save_roi(roi, BOTTOM_RIGHT, None, 5, 20)
        assert activity_window(roi, 28) == (5, 20)
        with open(roi, 'w') as f:      # files from before windows: t_last -1 means the end
            json.dump({'roi_first': BOTTOM_RIGHT, 't_last': -1}, f)
        assert activity_window(roi, 28) == (0, 27)
    passed('activity runs over the ROI window, or everything without one')


def test_viewer_draws_without_image_or_tracks():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'SUM_C2-rec.tif')
        write_projection(path, np.ones((6, 2, 8, 8), np.uint8))
        v = Viewer(None, None, 1, projection_path=path)
        assert v.roi_frames() == (0, 5)
        assert v.roi_frames(2, 4) == (2, 4)
        for bad in ((4, 2), (0, 6), (-1, 3)):
            try:
                v.roi_frames(*bad)
            except ValueError:
                continue
            raise AssertionError(f'window {bad} accepted')
        assert v.roi_select(BOTTOM_RIGHT, None, 2, 4) is None
    passed('ROI-only viewer: window checked, no cell count before tracking')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('roi: all tests passed')


if __name__ == '__main__':
    main()
