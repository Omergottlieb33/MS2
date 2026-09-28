"""src/cell_activity_validate.py: a good config is clean, and each corruption is caught."""
import copy
import os
import shutil
import tempfile

import numpy as np

from fixtures import passed, report_checks, write_recording

from src.cell_activity_validate import raise_on_errors, validate_config

def make_signals(n_cells=12, n_frames=6, gap_every=0):
    """n_cells x n_frames of plausible ellipse sums.  Cell 2 is never tracked over the first
    two frames; with gap_every set, every gap_every-th cell loses its first frame too, which
    is how a config with no cell tracked throughout is built."""
    rng = np.random.default_rng(0)
    signals = rng.uniform(5.0, 80.0, (n_cells, n_frames))
    signals[2, :2] = np.nan
    if gap_every:
        signals[::gap_every, 0] = np.nan
    return signals


SIGNALS = make_signals()


def test_good_config_is_clean(tmp):
    rec = write_recording(os.path.join(tmp, 'good'), SIGNALS)
    report = validate_config({'A': rec})
    assert report.empty, report.to_string()
    raise_on_errors(report)  # must not raise
    passed('a good config reports nothing')


def test_missing_keys(tmp):
    rec = write_recording(os.path.join(tmp, 'keys'), SIGNALS)
    del rec['masks_dir']
    report = validate_config({'A': rec})
    assert ('keys', 'A') in report_checks(report, 'error'), report.to_string()
    passed('a missing key is an error')


def test_window_past_the_matrix(tmp):
    rec = write_recording(os.path.join(tmp, 'window'), SIGNALS)
    rec['t_end'] = 99
    report = validate_config({'A': rec})
    assert ('window', 'A') in report_checks(report, 'error'), report.to_string()
    passed('a window past the matrix is an error')


def test_missing_mask(tmp):
    rec = write_recording(os.path.join(tmp, 'masks'), SIGNALS)
    os.remove(os.path.join(rec['masks_dir'], 'z_stack_t3_seg_masks.npz'))
    report = validate_config({'A': rec})
    assert ('masks', 'A') in report_checks(report, 'error'), report.to_string()
    passed('a missing mask timepoint is an error')


def test_tracklets_missing_a_cell(tmp):
    rec = write_recording(os.path.join(tmp, 'tracklets'), SIGNALS)
    with open(rec['tracklets']) as f:
        tracklets = f.read().replace('"2":', '"nope":')
    with open(rec['tracklets'], 'w') as f:
        f.write(tracklets)
    report = validate_config({'A': rec})
    assert ('tracklets', 'A') in report_checks(report, 'error'), report.to_string()
    passed('a cell with no tracklet is an error')


def test_out_of_range_min_presence(tmp):
    rec = write_recording(os.path.join(tmp, 'presence'), SIGNALS)
    rec['min_presence'] = 1.7
    report = validate_config({'A': rec})
    assert ('thresholds', 'A') in report_checks(report, 'error'), report.to_string()
    passed('min_presence outside 0..1 is an error')


def test_no_cell_survives_min_presence(tmp):
    # every cell loses a frame, so min_presence 1.0 leaves nothing to measure
    rec = write_recording(os.path.join(tmp, 'nocells'), make_signals(gap_every=1))
    rec['min_presence'] = 1.0
    report = validate_config({'A': rec})
    assert ('cells', 'A') in report_checks(report, 'error'), report.to_string()
    passed('a min_presence no cell meets is an error')


def test_few_surviving_cells_is_a_warning(tmp):
    rec = write_recording(os.path.join(tmp, 'fewcells'), make_signals(n_cells=4))
    report = validate_config({'A': rec})
    assert ('cells', 'A') in report_checks(report, 'warning'), report.to_string()
    assert not (report['level'] == 'error').any(), report.to_string()
    passed('a thin surviving cell set is a warning')


def test_unfixed_matrix_is_a_warning(tmp):
    rec = write_recording(os.path.join(tmp, 'unfixed'), SIGNALS,
                          name='gene_expression_results.csv')
    report = validate_config({'A': rec})
    assert ('csv', 'A') in report_checks(report, 'warning'), report.to_string()
    assert not (report['level'] == 'error').any(), report.to_string()
    passed('the unfixed matrix is a warning, not an error')


def test_mismatched_thresholds_across_recordings(tmp):
    a = write_recording(os.path.join(tmp, 'across_a'), SIGNALS)
    b = copy.deepcopy(a)
    a['noise_floor'], b['noise_floor'] = 20, 35
    report = validate_config({'A': a, 'B': b})
    checks = report_checks(report, 'warning')
    assert ('noise_floor', '<across recordings>') in checks, report.to_string()
    assert not (report['level'] == 'error').any(), report.to_string()
    passed('a noise_floor that differs between recordings is a warning')


def test_report_is_written_to_the_cache(tmp):
    rec = write_recording(os.path.join(tmp, 'cache'), SIGNALS)
    cache = os.path.join(tmp, 'cache_dir')
    validate_config({'A': rec}, cache)
    assert os.path.isfile(os.path.join(cache, 'validation.csv'))
    passed('validation.csv is written to the cache dir')


def main():
    tmp = tempfile.mkdtemp(prefix='ms2_validate_')
    try:
        for name, test in sorted(globals().items()):
            if name.startswith('test_'):
                test(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('cell_activity_validate: all tests passed')


if __name__ == '__main__':
    main()
