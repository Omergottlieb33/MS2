"""src/cell_activity_calibration.py: a brighter recording puts the same floor lower in its
own background, which is the whole point of the diagnostic."""
import os
import shutil
import tempfile

import numpy as np

from fixtures import passed, write_recording

from src.cell_activity_calibration import calibration_row, run_calibration

# Same cells, one recording imaged BRIGHTER by a known factor and nothing else changed
BRIGHTER = 3.0
FLOOR = 20.0


def make_pair(tmp):
    rng = np.random.default_rng(0)
    dim = rng.uniform(5.0, 50.0, (20, 8))
    a = write_recording(os.path.join(tmp, 'dim'), dim)
    b = write_recording(os.path.join(tmp, 'bright'), dim * BRIGHTER)
    a['noise_floor'] = b['noise_floor'] = FLOOR
    return a, b


def test_floor_sits_lower_in_the_brighter_recording(tmp):
    a, b = make_pair(tmp)
    row_a, _ = calibration_row('dim', a)
    row_b, _ = calibration_row('bright', b)
    assert row_b['floor_percentile'] < row_a['floor_percentile'], (row_a, row_b)
    # the same constant keeps three times as much of the brighter recording
    assert row_b['frac_above_floor'] > row_a['frac_above_floor']
    passed('one floor, two recordings, two different percentiles')


def test_median_min_nonzero_follows_the_brightness(tmp):
    a, b = make_pair(tmp)
    row_a, _ = calibration_row('dim', a)
    row_b, _ = calibration_row('bright', b)
    assert np.isclose(row_b['median_min_nonzero'], BRIGHTER * row_a['median_min_nonzero']), (
        row_a['median_min_nonzero'], row_b['median_min_nonzero'])
    passed('median_min_nonzero scales with the recording, as a photometric measure should')


def test_zeros_are_counted_not_measured(tmp):
    signals = np.full((12, 8), 30.0)
    signals[:, :2] = 0.0  # tracked, no emitter accepted
    rec = write_recording(os.path.join(tmp, 'zeros'), signals)
    row, nonzero = calibration_row('Z', rec)
    assert np.isclose(row['frac_zero'], 0.25), row['frac_zero']
    assert (nonzero > 0).all() and len(nonzero) == 12 * 6, len(nonzero)
    passed('zeros are reported as frac_zero and kept out of the distribution')


def test_outputs_are_written(tmp):
    a, b = make_pair(tmp)
    out = os.path.join(tmp, 'calibration')
    table = run_calibration({'dim': a, 'bright': b}, out)
    assert os.path.isfile(os.path.join(out, 'calibration.csv'))
    assert os.path.isfile(os.path.join(out, 'background.png'))
    assert list(table['recording']) == ['dim', 'bright']
    passed('calibration.csv and background.png are written')


def main():
    tmp = tempfile.mkdtemp(prefix='ms2_calibration_')
    try:
        for name, test in sorted(globals().items()):
            if name.startswith('test_'):
                test(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('cell_activity_calibration: all tests passed')


if __name__ == '__main__':
    main()
