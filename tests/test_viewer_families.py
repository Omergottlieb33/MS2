"""ms2/viewer: a results folder opens with its image, MS2 channel and cell families, in the older
hand-run layout too, and a family's cells are outlined with the emitter channel drawn over them.

    python tests/test_viewer_families.py
"""
import io
import json
import os
import shutil
import tempfile

import numpy as np
import tifffile
from PIL import Image

from fixtures import passed, write_recording

from ms2.viewer.app import open_results
from ms2.viewer.loaders import find_results
from ms2.viewer.render import GROUP

NAME = 'New-X'
# 0 silent, 1 a one-frame blip 5 above the floor, 2-4 active at three clearly different levels
SIGNALS = [[5, 5, 5, 5, 5, 5],
           [5, 5, 25, 5, 5, 5],
           [40, 40, 40, 40, 5, 5],
           [80, 80, 80, 80, 80, 5],
           [200, 200, 200, 200, 200, 200]]
BURST_CELL, BURST_T = 1, 2
MS2_PIXEL = (1, 5)     # (y, x) inside the burst cell's tile, see fixtures.write_recording


def old_layout(root):
    """A results folder as New-03-V has it: unify/, gene_expression_roi_selection/, and the tifs
    in the data folder the ROI's projection names."""
    rec = write_recording(os.path.join(root, 'build'), SIGNALS)
    rec_dir = os.path.join(root, 'outputs', NAME)
    data_dir = os.path.join(root, 'data', NAME)
    for d in ('unify', 'gene_expression_roi_selection'):
        os.makedirs(os.path.join(rec_dir, d))
    os.makedirs(data_dir)
    shutil.move(rec['masks_dir'], os.path.join(rec_dir, 'unify', 'masks_split'))
    shutil.move(rec['tracklets'], os.path.join(rec_dir, 'unify', 'tracklets_unified.json'))
    shutil.move(rec['csv'], os.path.join(rec_dir, 'gene_expression_roi_selection',
                                          'gene_expression_results.csv'))
    with open(os.path.join(rec_dir, 'unify', 'roi_selection.json'), 'w') as f:
        json.dump({'t_first': 0, 't_last': 5,
                   'projection': os.path.join(data_dir, f'SUM_C2-{NAME}.tif')}, f)

    n_t, (h, w) = len(SIGNALS[0]), (8, 12)          # the masks' 2 x 3 grid of 4x4 tiles
    tifffile.imwrite(os.path.join(data_dir, f'C2-{NAME}.tif'),
                     np.full((n_t, 2, h, w), 60, dtype=np.uint8))
    ms2 = np.zeros((n_t, 2, h, w), dtype=np.uint8)
    ms2[BURST_T, 0][MS2_PIXEL] = 100
    tifffile.imwrite(os.path.join(data_dir, f'C1-{NAME}_bg_removal.tif'), ms2)
    return rec_dir, data_dir


def rgb(png):
    return np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))


def test_finds_the_old_layout():
    with tempfile.TemporaryDirectory() as root:
        rec_dir, data_dir = old_layout(root)
        found = find_results(rec_dir)
        assert found['image'] == os.path.join(data_dir, f'C2-{NAME}.tif'), found
        assert found['ms2'] == os.path.join(data_dir, f'C1-{NAME}_bg_removal.tif'), found
        assert found['masks'].endswith(os.path.join('unify', 'masks_split')), found
        assert found['csv'].endswith('gene_expression_results.csv'), found
        assert found['peaks'] is None
    passed('old layout: image and MS2 channel found beside the ROI projection')


def test_families_and_frames():
    with tempfile.TemporaryDirectory() as root:
        rec_dir, _ = old_layout(root)
        v = open_results(rec_dir)
        fam = v.families
        assert [fam.cells[i]['level'] for i in range(5)] == [0, 1, 2, 3, 4]
        burst = fam.cells[BURST_CELL]
        assert burst['name'] == 'short burst / noise' and burst['peak_t'] == BURST_T
        assert burst['frames'] == 1 and burst['trace'][BURST_T] == 25
        assert [r['tid'] for r in fam.members(1)] == [BURST_CELL]

        # the family's cell outlined cyan; nothing else in it
        img = rgb(v.frame(BURST_T, 0, 0.4, 'outline', 1.0, 99.5, family=1))
        cyan = np.all(img == GROUP, axis=2)
        r, c = divmod(BURST_CELL, 3)
        assert cyan[4 * r + 1:4 * r + 3, 4 * c + 1:4 * c + 3].all()
        assert cyan.sum() == 4, cyan.sum()

        # and the emitter channel, summed over z, drawn over the cell at its burst
        img = rgb(v.frame(BURST_T, 0, 0.4, 'outline', 1.0, 99.5, family=1, show_ms2=True))
        y, x = MS2_PIXEL
        assert img[y, x, 0] == 255 and img[y, x, 1] == 255 and img[y, x, 2] < 60, img[y, x]
        # a frame with no emitter: the layer on changes nothing
        quiet = [rgb(v.frame(BURST_T - 1, 0, 0.4, 'outline', 1.0, 99.5, show_ms2=on))
                 for on in (True, False)]
        assert np.array_equal(*quiet)
    passed('families match the ladder; family outlined and MS2 drawn over it')


def test_missing_inputs_are_named():
    with tempfile.TemporaryDirectory() as root:
        try:
            open_results(root)
        except FileNotFoundError as exc:
            assert 'image' in str(exc) and 'tracklets' in str(exc), exc
        else:
            raise AssertionError('opened a folder with nothing in it')
    passed('an empty folder names what it lacks')


if __name__ == '__main__':
    test_finds_the_old_layout()
    test_families_and_frames()
    test_missing_inputs_are_named()
