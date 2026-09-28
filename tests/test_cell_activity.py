"""src/cell_activity.py: the onset measured against the cell's own life, the number of
activity levels (three by default, or the data's by silhouette), and a blip threshold that may
differ per recording."""
import numpy as np
import pandas as pd

from fixtures import passed

from src.cell_activity import (NOISE_FLOOR, activity_descriptors, activity_levels,
                               classify_cells)


def test_onset_tracked_ignores_frames_the_cell_was_not_tracked_in():
    n = 10
    signals, present = np.zeros((2, n)), np.zeros((2, n), dtype=bool)
    present[0] = True         # tracked throughout
    present[1, 5:] = True     # a daughter, first tracked halfway through the window
    signals[0, 5] = signals[1, 5] = 100.0

    desc = activity_descriptors(signals, present, NOISE_FLOOR)
    assert np.isclose(desc['onset'][0], desc['onset'][1]), desc['onset'].tolist()
    assert np.isclose(desc['onset_tracked'][0], 5 / 9), desc['onset_tracked'][0]
    # the daughter fires on its FIRST tracked frame; only onset_tracked says so
    assert np.isclose(desc['onset_tracked'][1], 0.0), desc['onset_tracked'][1]
    passed('onset cannot tell a late-born cell from a late firer, onset_tracked can')


def test_onset_tracked_equals_onset_for_a_cell_tracked_throughout():
    n = 8
    signals, present = np.zeros((3, n)), np.ones((3, n), dtype=bool)
    signals[[0, 1, 2], [0, 3, 7]] = 100.0
    desc = activity_descriptors(signals, present, NOISE_FLOOR)
    assert np.allclose(desc['onset'], desc['onset_tracked']), desc.to_string()
    passed('the two onsets agree when a cell is tracked through the whole window')


def test_silhouette_recovers_the_number_of_blobs():
    rng = np.random.default_rng(0)
    # separated in log space, which is where activity_levels clusters
    two = np.concatenate([rng.normal(10, 1, 40), rng.normal(10_000, 500, 40)])
    three = np.concatenate([rng.normal(10, 1, 40), rng.normal(1_000, 50, 40),
                            rng.normal(200_000, 5_000, 40)])
    assert len(np.unique(activity_levels(two, k=None))) == 2
    assert len(np.unique(activity_levels(three, k=None))) == 3
    passed('with k=None, k follows the data')


def test_the_default_is_three_levels_whatever_the_silhouette_says():
    rng = np.random.default_rng(0)
    two = np.concatenate([rng.normal(10, 1, 40), rng.normal(10_000, 500, 40)])
    levels = activity_levels(two)
    assert len(np.unique(levels)) == 3, np.unique(levels)
    assert all(two[levels == a].mean() < two[levels == b].mean() for a, b in ((0, 1), (1, 2)))
    assert len(np.unique(activity_levels([5.0, 5.0, 50.0]))) == 2  # only 2 distinct values
    passed('three ordered levels by default, even where the silhouette would pick 2')


def test_levels_rise_with_activity():
    rng = np.random.default_rng(0)
    score = np.concatenate([rng.normal(10, 1, 30), rng.normal(10_000, 500, 30)])
    levels = activity_levels(score)
    assert score[levels == 0].mean() < score[levels == 1].mean()
    passed('level 0 is the quietest group, as order_labels_by_activity promises')


def test_noise_peak_may_differ_per_row():
    # two cells with the SAME peak, judged by different recordings' blip thresholds
    rng = np.random.default_rng(0)
    peaks = np.concatenate([[15.0, 15.0], rng.uniform(100, 10_000, 30)])
    desc = pd.DataFrame({'peak': peaks, 'total': peaks * 4})
    per_row = np.concatenate([[10.0, 20.0], np.full(30, 10.0)])

    levels = classify_cells(desc, noise_peak=per_row)
    assert levels[0] >= 2, levels[:2]   # 15 clears its recording's threshold of 10
    assert levels[1] == 1, levels[:2]   # the same 15 is a blip where the threshold is 20
    passed('one pooled ladder, per-recording blip thresholds')


def test_a_scalar_noise_peak_still_works():
    rng = np.random.default_rng(0)
    peaks = np.concatenate([[5.0], rng.uniform(100, 10_000, 30)])
    desc = pd.DataFrame({'peak': peaks, 'total': peaks * 4})
    levels = classify_cells(desc, noise_peak=10.0)
    assert levels[0] == 1 and (levels[1:] >= 2).all(), levels
    passed('the scalar threshold path is unchanged')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('cell_activity: all tests passed')


if __name__ == '__main__':
    main()
