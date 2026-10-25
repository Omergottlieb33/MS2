"""ms2/cell_activity_segments.py: a segment's box in the aligned view maps onto the (-1..1)
grid with right-upper at (1, 1) whatever the turn, a bin holds the share of its cells firing
frame by frame, and the comparison writes its maps, correlations and boxes.

    python tests/test_cell_activity_segments.py
"""
import json
import os
import tempfile

import numpy as np
import pandas as pd

from fixtures import passed, write_recording

from ms2.cell_activity import NOISE_FLOOR
from ms2.cell_activity_compare import compare_recordings
from ms2.cell_activity_segments import (Layout, analyse_segments, plot_segment_gif,
                                        equal_count_bins, mean_cells_per_bin, normalise,
                                        segment_activity, segment_frame)

# 20 wide, 10 tall in the image; y points down, so (30, 10) is the image's right-upper corner
RECT = [[10, 10], [30, 10], [30, 20], [10, 20]]


def test_corners_follow_the_aligned_view():
    """Which image corner shows at the right-upper depends on the turn; that one is (1, 1)."""
    cases = {  # orientation: the image corner that shows at the aligned right-upper
        (False, 0): (30, 10),
        (True, 0): (10, 10),     # mirrored: the left edge is now on the right
        (False, 90): (10, 10),   # clockwise: the top edge is now the right edge
        (True, 90): (30, 10),
        (False, 180): (10, 20),
    }
    for (flip, rotate), corner in cases.items():
        frame = segment_frame(RECT, {'flip': flip, 'rotate': rotate})
        opposite = [40 - corner[0], 30 - corner[1]]
        uv = normalise([corner, opposite, [20, 15]], frame)
        assert np.allclose(uv, [[1, 1], [-1, -1], [0, 0]]), (flip, rotate, uv)
    # a turned box keeps its shape: rotated 90, the 20-wide side stands upright
    _, _, half = segment_frame(RECT, {'flip': False, 'rotate': 90})
    assert np.allclose(half, [5, 10]), half
    passed('the aligned right-upper corner is (1, 1), the centre (0, 0), under every turn')


def test_bins_hold_the_share_of_their_cells_firing():
    """Four cells, one per quadrant of a square segment; only the upper-right one fires, and
    the upper-left one is not tracked on frame 2."""
    signals = np.zeros((4, 3))
    signals[0] = NOISE_FLOOR + 30
    signals[1, 2] = np.nan
    with tempfile.TemporaryDirectory() as tmp:
        rec = write_recording(tmp, signals, masks=False)
        cells = pd.DataFrame({'cell': [f'cell_{i}' for i in range(4)],
                              'x': [7.5, 2.5, 2.5, 7.5], 'y': [2.5, 2.5, 7.5, 7.5]})
        entry = {'orientation': {'flip': False, 'rotate': 0},
                 'segments': {'2': [[0, 0], [10, 0], [10, 10], [0, 10]], '3': [[50, 50], [60, 50], [60, 60]]}}
        table = segment_activity('r', rec, entry, cells, bins=2, time_bins=1)
    t2 = table[table['segment'] == '2']
    frac = {(x, y, t): f for x, y, t, f in t2[['bin_x', 'bin_y', 't', 'frac_firing']].itertuples(index=False)}
    assert all(frac[1, 1, t] == 1 for t in range(3)), frac          # upper-right fires
    assert all(frac[0, 0, t] == 0 and frac[1, 0, t] == 0 for t in range(3))
    assert frac[0, 1, 0] == 0 and np.isnan(frac[0, 1, 2])            # untracked: no value
    assert t2['u_centre'].isin([-0.5, 0.5]).all() and len(t2) == 4 * 3
    # cells per bin, averaged over the frames: the upper-left cell is tracked on 2 of 3
    counts = mean_cells_per_bin(t2).set_index(['bin_x', 'bin_y'])['mean_cells']
    assert counts[1, 1] == 1 and np.isclose(counts[0, 1], 2 / 3), counts
    # a segment no cell falls in has only empty bins
    assert table.loc[table['segment'] == '3', 'frac_firing'].isna().all()
    passed('a bin is the share of its tracked cells firing, empty where none is tracked')


def test_comparison_of_segments():
    """Two identical recordings: every map and file is written, and their maps correlate 1."""
    signals = np.zeros((12, 6))
    signals[[0, 1, 4], 1:4] = NOISE_FLOOR + 50       # the upper-left cells fire mid-window
    signals[[6, 11], 4:] = NOISE_FLOOR + 50
    with tempfile.TemporaryDirectory() as tmp:
        recordings = {n: write_recording(os.path.join(tmp, n), signals) for n in ('a', 'b')}
        cmp_dir = os.path.join(tmp, 'compare')
        compare_recordings(recordings, cmp_dir)
        with open(os.path.join(cmp_dir, 'recordings.json'), 'w') as f:
            json.dump(recordings, f)
        whole = [[0, 0], [16, 0], [16, 12], [0, 12]]       # the fixture's 4 x 3 grid of cells
        with open(os.path.join(cmp_dir, 'segments.json'), 'w') as f:
            json.dump({n: {'t': 0, 'orientation': {'flip': False, 'rotate': 0},
                           'segments': {'2': whole, '4': whole}} for n in recordings}, f)
        for bad, why in (({'bins': 0}, 'bins'), ({'labels': ['7']}, 'not one of')):
            try:
                analyse_segments(cmp_dir, **bad)
            except ValueError as e:
                assert why in str(e), e
            else:
                raise AssertionError(f'{bad} accepted')
        written = analyse_segments(cmp_dir, bins=2, time_bins=2, labels=['2'])
        out = os.path.join(cmp_dir, 'segment_activity')
        assert sorted(os.path.basename(p) for p in written) == [
            'segment_2_activity.gif', 'segment_2_activity.png', 'segment_2_cells_per_bin.csv',
            'segment_2_cells_per_bin.png', 'segment_2_clusters.png', 'segment_2_clusters_correlation.csv',
            'segment_2_correlation.csv', 'segment_2_intensity.png', 'segment_2_intensity_correlation.csv',
            'segment_activity.csv', 'segment_boxes.png', 'segment_cells.csv'], written
        table = pd.read_csv(os.path.join(out, 'segment_activity.csv'))
        assert set(table['segment']) == {2} and set(table['time_bin']) == {0, 1}
        corr = pd.read_csv(os.path.join(out, 'segment_2_correlation.csv'))
        assert len(corr) == 1 and np.isclose(corr['r'][0], 1.0), corr
        with open(os.path.join(out, 'params.json')) as f:
            assert json.load(f) == {'bins': 2, 'cells_per_bin': None, 'time_bins': 2,
                                    'box_from': 'outline', 'segments': ['2']}
        # bins by cell count: 12 cells at about 4 a bin is 3 bins of 4
        analyse_segments(cmp_dir, time_bins=2, labels=['2'], cells_per_bin=4)
        table = pd.read_csv(os.path.join(out, 'segment_activity.csv'))
        first = table[table['t'] == table['t'].min()]
        assert first.groupby('recording')['bin'].nunique().tolist() == [3, 3], first
        counts = pd.read_csv(os.path.join(out, 'segment_2_cells_per_bin.csv'))
        assert (counts['mean_cells'] == 4).all(), counts       # every cell tracked throughout
        corr = pd.read_csv(os.path.join(out, 'segment_2_correlation.csv'))
        assert list(corr.columns) == ['recording_a', 'recording_b', 'n_raster_pixels', 'r']
        assert np.isclose(corr['r'][0], 1.0), corr
        with open(os.path.join(out, 'params.json')) as f:
            assert json.load(f)['cells_per_bin'] == 4
    passed('the segment comparison writes its table, maps, correlations and boxes')


def test_equal_count_bins():
    """Bins hold the target count, every cell lies in its own bin's rectangle, the rectangles
    tile the grid, and too few cells make one bin."""
    uv = np.random.default_rng(1).uniform(-1, 1, (24, 2)) ** 3     # denser in the middle
    for target, n_bins in ((4, 6), (2, 12), (6, 4)):
        ids, rects = equal_count_bins(uv, target)
        assert len(rects) == n_bins and (np.bincount(ids) == target).all(), (target, np.bincount(ids))
        u0, u1, v0, v1 = rects[ids, 2], rects[ids, 3], rects[ids, 4], rects[ids, 5]
        assert ((u0 <= uv[:, 0]) & (uv[:, 0] <= u1) & (v0 <= uv[:, 1]) & (uv[:, 1] <= v1)).all()
        assert np.isclose(((rects[:, 3] - rects[:, 2]) * (rects[:, 5] - rects[:, 4])).sum(), 4)
    ids, rects = equal_count_bins(uv[:3], 6)
    assert len(rects) == 1 and (ids == 0).all()
    # a prime number of bins (92 cells / 4 = 23) is laid out near square, not as one strip
    many = np.random.default_rng(2).uniform(-1, 1, (92, 2))
    ids, rects = equal_count_bins(many, 4)
    rows = np.bincount(rects[:, 0].astype(int))
    assert len(rects) == 23 and len(rows) == 5 and rows.max() - rows.min() <= 1, rows
    assert set(np.bincount(ids)) == {4}, np.bincount(ids)
    # dense middle, small bins: the bin holding the centre is smaller than an average one
    ids, rects = equal_count_bins(uv, 2)
    area = (rects[:, 3] - rects[:, 2]) * (rects[:, 5] - rects[:, 4])
    centre = [i for i, r in enumerate(rects) if r[2] <= 0 <= r[3] and r[4] <= 0 <= r[5]][0]
    assert area[centre] < area.mean(), (area[centre], area.mean())
    passed('equal-count bins hold the target, contain their cells and tile the grid')


def test_intensity_cluster_maps_and_the_tight_box():
    """Per cell: its total above the floor and its level, and the maps built from them; with
    box_from='cells' the grid is cut from the cells inside the outline, not the outline."""
    from ms2.cell_activity_segments import member_maps
    signals = np.zeros((4, 3))
    signals[0] = NOISE_FLOOR + np.array([5, 10, 15])         # total 30, upper-right
    signals[1, 0] = NOISE_FLOOR + 8                            # total 8, upper-left
    with tempfile.TemporaryDirectory() as tmp:
        rec = write_recording(tmp, signals, masks=False)
        cells = pd.DataFrame({'cell': [f'cell_{i}' for i in range(4)], 'level': [4, 2, 0, 0],
                              'x': [7.5, 2.5, 2.5, 7.5], 'y': [2.5, 2.5, 7.5, 7.5]})
        # the outline reaches far beyond the cells: (0, 0)..(20, 20)
        entry = {'orientation': {'flip': False, 'rotate': 0},
                 'segments': {'2': [[0, 0], [20, 0], [20, 20], [0, 20]]}}
        _, loose = segment_activity('r', rec, entry, cells, 2, 1, with_members=True)
        _, tight = segment_activity('r', rec, entry, cells, 2, 1, box_from='cells', with_members=True)
    # around the outline all four cells crowd one quadrant; around the cells they spread over four
    assert loose['bin'].nunique() == 1 and tight['bin'].nunique() == 4, (loose, tight)
    assert np.allclose(sorted(np.abs(tight[['u', 'v']].to_numpy().ravel())), 1)
    t = tight.set_index('cell')
    assert t.loc['cell_0', 'total'] == 30 and t.loc['cell_1', 'total'] == 8 and t.loc['cell_2', 'total'] == 0
    up_right = t.loc['cell_0', ['u', 'v']].tolist()
    assert up_right == [1.0, 1.0], up_right                     # image y points down: up is +1
    intensity = member_maps(tight, 'total', 2)['r']             # rows up from v = -1
    assert intensity[1, 1] == 30 and intensity[1, 0] == 8 and intensity[0, 0] == 0
    high = member_maps(tight, None, 2, level=4)['r']
    assert high[1, 1] == 1 and high[1, 0] == 0
    passed('intensity and cluster maps come from each cell; the tight box hugs the cells')


def test_gif_frames_at_each_tenth_of_the_window():
    """A 25-frame and an 11-frame recording: 11 GIF frames, each at the same share of both
    windows -- frame round(p * (n - 1)), so 0, 2, 5, .. 24 and 0, 1, .. 10."""
    import imageio.v2 as iio
    rows = [{'recording': n, 'segment': '2', 't': 100 + t, 'bin': 0, 'bin_x': 0, 'bin_y': 0,
             'u0': -1, 'u1': 1, 'v0': -1, 'v1': 1, 'n_cells': 1, 'n_firing': int(t % 2)}
            for n, k in (('long', 25), ('short', 11)) for t in range(k)]
    picked = []                    # each frame's panel titles, which name the frame shown
    from ms2 import cell_activity_segments as cas
    orig = cas.map_row
    cas.map_row = lambda maps, title, label, **kw: (picked.append(kw['titles']), orig(maps, title, label, **kw))[1]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 's.gif')
            table = pd.DataFrame(rows)
            plot_segment_gif(table, '2', ['long', 'short'], Layout(table, 1, None), path)
            assert len(iio.mimread(path)) == 11
    finally:
        cas.map_row = orig
    longs = [int(t['long'].split('t=')[1][:-1]) - 100 for t in picked]
    shorts = [int(t['short'].split('t=')[1][:-1]) - 100 for t in picked]
    assert longs == [0, 2, 5, 7, 10, 12, 14, 17, 19, 22, 24], longs
    assert shorts == list(range(11)), shorts
    passed('the GIF shows every recording at 0%, 10%, .. 100% of its own window')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('cell_activity_segments: all tests passed')


if __name__ == '__main__':
    main()
