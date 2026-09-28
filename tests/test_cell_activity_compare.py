"""src/cell_activity_compare.py: both tails of the spatial test, and the chi-square fallback
when the asymptotic approximation does not hold."""
import os
import tempfile

import numpy as np
import pandas as pd

from fixtures import passed, write_recording

from src.cell_activity import NOISE_FLOOR
from src.cell_activity_compare import (N_PERMUTATIONS, compare_recordings, composition_test,
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
            compare_recordings(recordings, out, ladder=ladder)
            cells[ladder] = pd.read_csv(os.path.join(out, 'cells.csv'))

    levels = {(ladder, name): set(c.loc[c['recording'] == name, 'level'])
              for ladder, c in cells.items() for name in recordings}
    assert levels['joint', 'dim'] == {2} and levels['joint', 'bright'] <= {3, 4}, levels
    assert levels['per_recording', 'dim'] == levels['per_recording', 'bright'] == {2, 3, 4}, levels
    assert (cells['joint']['active'] == cells['per_recording']['active']).all()
    passed('per-recording ladders span each recording; the active set is the same')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('cell_activity_compare: all tests passed')


if __name__ == '__main__':
    main()
