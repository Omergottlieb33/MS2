"""ms2/cell_activity_compare.py: both tails of the spatial test, and the chi-square fallback
when the asymptotic approximation does not hold."""
import os
import tempfile

import numpy as np
import pandas as pd

from fixtures import passed, write_recording

from ms2.cell_activity import NOISE_FLOOR
import json

import tifffile

from ms2.cell_activity_compare import (N_PERMUTATIONS, alignment_matrix, cell_positions,
                                       cell_projection, compare_recordings, composition_test,
                                       nn_clustering)

ALPHA = 0.05


def lattice(side=14, spacing=1.0):
    """A dense, regular field of cells for the active set to be drawn from."""
    g = np.arange(side) * spacing
    return np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2).astype(float)


def test_overdispersed_active_cells_are_detected():
    """Active cells on a coarse sublattice sit further apart than any random draw of as many.

    The one-sided test this replaces could only ever report 'not clustered' here.
    """
    positions = lattice()
    active = np.zeros(len(positions), dtype=bool)
    coarse = (positions[:, 0] % 3 == 0) & (positions[:, 1] % 3 == 0)
    active[coarse] = True

    index, p_clustered, p_dispersed = nn_clustering(positions, active, np.random.default_rng(0))
    assert index > 1, index
    assert p_dispersed < ALPHA, p_dispersed
    assert p_clustered > ALPHA, p_clustered
    passed('regular spacing is flagged by the new tail and invisible to the old one')


def test_clustered_active_cells_are_still_detected():
    positions = lattice()
    center = positions.mean(axis=0)
    active = np.linalg.norm(positions - center, axis=1) < 3.0

    index, p_clustered, p_dispersed = nn_clustering(positions, active, np.random.default_rng(0))
    assert index < 1, index
    assert p_clustered < ALPHA, p_clustered
    assert p_dispersed > ALPHA, p_dispersed
    passed('the clustering tail still behaves as it did')


def test_too_few_active_cells_is_nan_on_both_tails():
    positions = lattice(side=4)
    active = np.zeros(len(positions), dtype=bool)
    active[0] = True
    assert all(np.isnan(v) for v in nn_clustering(positions, active, np.random.default_rng(0)))
    passed('one active cell has no nearest neighbour and says so')


def make_cells(counts):
    """A cells frame from {recording: {level: n}}."""
    rows = [{'recording': name, 'level': level}
            for name, levels in counts.items() for level, n in levels.items() for _ in range(n)]
    return pd.DataFrame(rows)


def composition_of(cells):
    table = pd.crosstab(cells['recording'], cells['level'])
    return table.loc[:, table.sum() > 0]


def test_low_expected_counts_take_the_permutation_path():
    # 2 x 3 table of 14 cells: several expected counts fall below 5
    cells = make_cells({'A': {0: 5, 1: 2, 2: 1}, 'B': {0: 4, 1: 1, 2: 1}})
    composition = composition_of(cells)
    expected = (np.outer(composition.sum(axis=1), composition.sum(axis=0))
                / composition.to_numpy().sum())
    assert expected.min() < 5, expected
    result = composition_test(cells, composition)
    assert result['test'] == 'chi-square (permutation)', result
    assert 1 / (1 + N_PERMUTATIONS) <= result['p'] <= 1.0, result
    passed('a sparse composition table is tested by permutation')


def test_healthy_counts_keep_the_asymptotic_test():
    cells = make_cells({'A': {0: 200, 1: 150, 2: 120}, 'B': {0: 180, 1: 40, 2: 210}})
    result = composition_test(cells, composition_of(cells))
    assert result['test'] == 'chi-square', result
    assert result['p'] < ALPHA, result   # these compositions really do differ
    passed('a well-filled table is left on the asymptotic test')


def test_the_permutation_p_reflects_the_data():
    """Identical compositions must not look significant, plainly different ones must."""
    same = make_cells({'A': {0: 5, 1: 2, 2: 1}, 'B': {0: 5, 1: 2, 2: 1}})
    apart = make_cells({'A': {0: 8, 1: 0, 2: 0}, 'B': {0: 0, 1: 4, 2: 4}})
    assert composition_test(same, composition_of(same))['p'] > ALPHA
    assert composition_test(apart, composition_of(apart))['p'] < ALPHA
    passed('the permutation p separates identical from disjoint compositions')


def test_a_ladder_per_recording_is_relative_to_its_own_recording():
    """Two recordings with the same three tiers, one 100x brighter.  The joint ladder puts the
    whole dim one at the bottom; a ladder per recording finds all three tiers in each."""
    tiers = np.repeat([15.0, 40.0, 100.0], 4)
    with tempfile.TemporaryDirectory() as tmp:
        recordings = {}
        for name, gain in (('dim', 1.0), ('bright', 100.0)):
            signals = np.zeros((len(tiers), 6))
            signals[:, 2] = NOISE_FLOOR + gain * tiers
            recordings[name] = write_recording(os.path.join(tmp, name), signals)
        cells = {}
        for ladder in ('joint', 'per_recording'):
            out = os.path.join(tmp, ladder)
            compare_recordings(recordings, out, ladder=ladder, flip_rotate=True)
            cells[ladder] = pd.read_csv(os.path.join(out, 'cells.csv'))
            for stem in ('heatmap_active_pooled', 'heatmap_active_dim', 'heatmap_active_bright'):
                assert os.path.exists(os.path.join(out, stem + '.png')), stem

    levels = {(ladder, name): set(c.loc[c['recording'] == name, 'level'])
              for ladder, c in cells.items() for name in recordings}
    assert levels['joint', 'dim'] == {2} and levels['joint', 'bright'] <= {3, 4}, levels
    assert levels['per_recording', 'dim'] == levels['per_recording', 'bright'] == {2, 3, 4}, levels
    assert (cells['joint']['active'] == cells['per_recording']['active']).all()
    passed('per-recording ladders span each recording; the active set is the same')


def test_alignment_turns_like_the_page():
    """Image x right, y down.  Clockwise on screen takes right to down; flip + 90 is the old
    --flip-rotate, image (x, y) at (-y, -x) in y-down coordinates."""
    assert np.allclose(alignment_matrix(rotate=90) @ [1, 0], [0, 1])
    assert np.allclose(alignment_matrix(flip=True) @ [1, 2], [-1, 2])
    assert np.allclose(alignment_matrix(flip=True, rotate=90) @ [3, 5], [-5, -3])
    # the page's vertical flip is flip with 180 - angle: the same as mirroring y on screen
    for f, a in ((False, 30), (True, -70)):
        assert np.allclose(np.diag([1, -1]) @ alignment_matrix(f, a), alignment_matrix(not f, 180 - a))
        assert np.allclose(np.diag([-1, 1]) @ alignment_matrix(f, a), alignment_matrix(not f, -a))
    passed('an alignment turns the cells as the page turns the picture')


def test_overlay_image_turns_as_the_cells_do():
    """orient_image (the activity overlay) and alignment_matrix (the comparison's maps) put a
    pixel in the same place, about the image centre, at right and at odd angles."""
    from ms2.cell_activity import orient_image
    h, w = 41, 61
    for flip, rotate in ((False, 90), (True, 90), (True, -58), (False, 180), (True, 33)):
        img = np.zeros((h, w), dtype=np.uint8)
        img[10:13, 45:48] = 255                    # a blob at (x, y) = (46, 11)
        out = orient_image(img, flip, rotate)
        ys, xs = np.nonzero(out)
        got = np.array([xs.mean() - (out.shape[1] - 1) / 2, ys.mean() - (out.shape[0] - 1) / 2])
        want = alignment_matrix(flip, rotate) @ [46 - (w - 1) / 2, 11 - (h - 1) / 2]
        assert np.allclose(got, want, atol=1.0), (flip, rotate, got, want)
    passed('the activity overlay turns exactly as the comparison turns the cells')


def test_recording_orientation_is_the_default_alignment():
    from ms2.cell_activity_compare import comparison_alignment
    recordings = {'a': {'orientation': {'flip': True, 'rotate': 90}}, 'b': {}}
    with tempfile.TemporaryDirectory() as out:
        assert comparison_alignment(out, {'b': {}}) is None
        assert comparison_alignment(out, recordings) == {'a': {'flip': True, 'rotate': 90}}
        with open(os.path.join(out, 'alignment.json'), 'w') as f:
            json.dump({'a': {'flip': False, 'rotate': 10}, 'b': {'flip': True, 'rotate': 0}}, f)
        assert comparison_alignment(out, recordings) == {'a': {'flip': False, 'rotate': 10},
                                                         'b': {'flip': True, 'rotate': 0}}
    passed("a recording's own orientation aligns it; the Compare page's alignment wins")


def test_projection_found_from_the_roi_and_aligned_map_drawn():
    """Hand-run layout: no preprocess/, the projection named by the ROI in masks/.  A saved
    alignment.json makes the comparison draw the aligned map."""
    with tempfile.TemporaryDirectory() as tmp:
        recordings = {}
        for name in ('a', 'b'):
            signals = np.zeros((12, 4))
            signals[[0, 1, 4, 5], 1] = NOISE_FLOOR + 50   # two rows: a hull with area
            rec = write_recording(os.path.join(tmp, name, 'unify'), signals)
            rec['masks_dir'] = os.path.join(tmp, name, 'unify', 'masks_split')
            os.rename(os.path.join(tmp, name, 'unify', 'masks'), rec['masks_dir'])
            projection = os.path.join(tmp, f'SUM_C2-{name}.tif')
            tifffile.imwrite(projection, np.random.default_rng(0).random((4, 16, 16), dtype=np.float32))
            os.makedirs(os.path.join(tmp, name, 'masks'))
            with open(os.path.join(tmp, name, 'masks', 'roi_selection.json'), 'w') as f:
                json.dump({'projection': projection}, f)
            assert cell_projection(rec) == projection
            recordings[name] = rec
        out = os.path.join(tmp, 'compare')
        os.makedirs(out)
        with open(os.path.join(out, 'alignment.json'), 'w') as f:
            json.dump({'a': {'flip': True, 'rotate': 90}, 'b': {'flip': False, 'rotate': 33}}, f)
        compare_recordings(recordings, out)
        assert os.path.exists(os.path.join(out, 'spatial_maps_aligned.png'))
    passed('the projection is found through the ROI; a saved alignment draws the aligned map')


def test_page_filters_roi_and_hand_set_cuts():
    """What the analysis page sets: an ROI tested on the window's first and last frame, a
    volume-CV ceiling, and the ladder given as rate cuts instead of fitted."""
    signals = np.zeros((12, 6))
    for i, rate in enumerate([0, 2, 6, 12, 30, 60, 0, 0, 0, 0, 0, 0]):
        signals[i, :] = NOISE_FLOOR + rate if rate else 0
    with tempfile.TemporaryDirectory() as tmp:
        rec = write_recording(os.path.join(tmp, 'a'), signals)
        # tiles of 4 px, 4 to a row: the outline holds the first row's six-pixel-wide strip of
        # cells 0..3 and cells 4..5 of the second row
        rec['roi_first'] = [[0, 0], [16, 0], [16, 4], [0, 4]]
        rec['roi_last'] = [[0, 4], [8, 4], [8, 8], [0, 8]]
        rec['max_volume_cv'] = 0.3
        out = os.path.join(tmp, 'out')
        compare_recordings({'a': rec, 'b': write_recording(os.path.join(tmp, 'b'), signals)}, out,
                           level_cuts=[10.0, 40.0])
        cells = pd.read_csv(os.path.join(out, 'cells.csv'))
    a = cells[cells['recording'] == 'a'].set_index('cell')
    assert sorted(a.index) == [f'cell_{i}' for i in range(6)], a.index      # the ROI's six
    assert (cells[cells['recording'] == 'b'].shape[0]) == 12                  # no ROI: all
    assert 'volume_cv' in cells and (cells['volume_cv'] == 0).all()
    # rate 2 and 6 never reach the blip height (10 above the floor): short burst / noise; then
    # the cuts 10 and 40 put 12 and 30 at moderate and 60 at high; 0 is silent
    assert a['level'].tolist() == [0, 1, 1, 3, 3, 4], a['level'].tolist()
    passed('the page\'s ROI, volume CV and hand-set cuts give the cells and levels it shows')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('cell_activity_compare: all tests passed')


def test_unstable_frac_counts_frames_off_the_median_volume():
    """One frame at a third of the volume is 1/5 of a 5-frame window and 1/50 of a 50-frame one,
    and a 1.4x wobble is never unstable."""
    def frac(volumes):
        props = {t: {1: (0.0, 0.0, 0.0, v)} for t, v in enumerate(volumes)}
        return cell_positions(props, {'1': [1] * len(volumes)}, ['cell_1'], range(len(volumes)))[1][0]

    assert frac([300, 900, 900, 900, 900]) == 0.2
    assert frac([300] + [900] * 49) == 0.02
    assert frac([900, 1260, 900, 643, 900]) == 0.0
    passed('unstable_frac is the share of frames more than 1.5x off the median volume')


if __name__ == '__main__':
    main()
