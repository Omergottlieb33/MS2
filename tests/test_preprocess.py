"""src/preprocess.py: the background subtraction reproduces ImageJ's Subtract Background
(sliding paraboloid, radius 5) pixel for pixel.

    python tests/test_preprocess.py                 # quick checks
    python tests/test_preprocess.py <czi> <folder>  # every slice against a folder's ImageJ tifs

The folder holds C1-<rec>.tif, C2-<rec>.tif and the ImageJ output C1-<rec>_bg_remov*.tif.
"""
import glob
import os
import sys

import numpy as np
import tifffile

from fixtures import passed

from src.preprocess import load_czi, subtract_background, subtract_background_stack

REFERENCE = '/zjbd/zd1/shechtmanlab/omer/MS2/data/020626/STAGE-13/New-14-v2'


def test_flat_background_goes_to_one_and_spot_survives():
    img = np.full((64, 64), 10, np.uint8)
    img[30:33, 30:33] = 60
    out = subtract_background(img)
    assert out[5, 5] <= 1, out[5, 5]
    assert out[31, 31] > 40, out[31, 31]
    passed('flat background is removed, a 3x3 spot is kept')


def test_stack_matches_slice_by_slice():
    rng = np.random.default_rng(0)
    stack = rng.poisson(2, (2, 3, 40, 50)).astype(np.uint8)
    out = subtract_background_stack(stack, workers=2)
    for t in range(2):
        for z in range(3):
            assert np.array_equal(out[t, z], subtract_background(stack[t, z]))
    passed('threaded stack equals slice-by-slice')


def test_matches_imagej_on_reference_slices():
    ref = os.path.join(REFERENCE, 'C1-New-14-v2_bg_removed.tif')
    if not os.path.exists(ref):
        print('skip: reference data not mounted')
        return
    for k in (0, 13, 400, 791):
        raw = tifffile.imread(os.path.join(REFERENCE, 'C1-New-14-v2.tif'), key=k)
        assert np.array_equal(subtract_background(raw), tifffile.imread(ref, key=k)), k
    passed('identical to ImageJ on New-14-v2 slices')


def compare_folder(czi, folder):
    """Split channels from the czi and subtract the background, then compare every slice."""
    data = load_czi(czi)
    rec = os.path.splitext(os.path.basename(czi))[0]
    for ch, name in ((0, 'C1'), (1, 'C2')):
        ij = tifffile.imread(os.path.join(folder, f'{name}-{rec}.tif'))
        assert ij.shape == data[:, ch].shape, (name, ij.shape, data[:, ch].shape)
        assert np.array_equal(ij, data[:, ch]), f'{name} differs from czi channel {ch}'
        print(f'{name}: identical to czi channel {ch}, shape {ij.shape}')
    ref, = glob.glob(os.path.join(folder, f'C1-{rec}_bg_remov*.tif'))
    ij = tifffile.imread(ref)
    ours = subtract_background_stack(data[:, 0])
    diff = ours.astype(np.int16) - ij
    n_bad = np.count_nonzero(diff)
    print(f'{os.path.basename(ref)}: {ij.size} pixels, {n_bad} differ, '
          f'max |diff| {np.abs(diff).max()}, slices with any diff '
          f'{np.count_nonzero(diff.reshape(-1, *diff.shape[-2:]).any(axis=(1, 2)))}')
    assert n_bad == 0


def main():
    if len(sys.argv) == 3:
        compare_folder(sys.argv[1], sys.argv[2])
        print('preprocess: folder matches ImageJ')
        return
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('preprocess: all tests passed')


if __name__ == '__main__':
    main()
