"""Check a src/cell_activity.py recording config before anything is measured.

The analyses read the expression matrix, the tracklets json and the masks directory of every
recording and assume the three agree.  When they don't the run dies deep inside itself -- after
cell_properties has walked every mask, which is minutes -- or, worse, doesn't die at all: an
unfixed expression matrix or a noise floor that means something different in each recording
produces numbers that look like an answer.

Every check runs on every recording, so one pass lists everything wrong rather than the first
thing wrong.  Rows are one of two levels:

    error     the run cannot proceed; raise_on_errors stops it here
    warning   the run proceeds but the result may not mean what it looks like -- almost all of
              these are comparability traps, where each recording is fine on its own and the
              comparison between them is not

    python -m src.cell_activity_validate --config recordings.json

recordings.json is the {name: recording dict} described in src/cell_activity.py.
"""
import argparse
import json
import os

import pandas as pd

from src.cell_activity import MIN_PRESENCE, NOISE_FLOOR, NOISE_PEAK
from src.gene_expression.expression_matrix import FIXED_NAME
from src.track_diagnostics import mask_paths_by_t

REQUIRED_KEYS = ('csv', 'tracklets', 'masks_dir', 't_start', 't_end')
# Below this many cells a recording's medians and permutation tests say very little.
FEW_CELLS = 10
# Windows further apart than this make duty and onset awkward to compare; rate normalises.
WINDOW_RATIO = 2.0
# The name of the recording the cross-recording checks are filed under.
ACROSS = '<across recordings>'


def row(recording, level, check, message):
    return {'recording': recording, 'level': level, 'check': check, 'message': message}


def check_structure(name, rec):
    """The keys and the paths they point at.  Every later check reads one of these, so when
    this returns an error the rest are skipped for this recording."""
    missing = [key for key in REQUIRED_KEYS if key not in rec]
    if missing:
        return [row(name, 'error', 'keys', f'missing {", ".join(missing)}')]

    rows = []
    for key, present in (('csv', os.path.isfile), ('tracklets', os.path.isfile),
                         ('masks_dir', os.path.isdir)):
        if not present(rec[key]):
            rows.append(row(name, 'error', 'paths', f'{key} does not exist: {rec[key]}'))
    if not all(isinstance(rec[k], int) for k in ('t_start', 't_end')):
        rows.append(row(name, 'error', 'window',
                        f't_start and t_end must be int, got {rec["t_start"]!r} '
                        f'and {rec["t_end"]!r}'))
    elif rec['t_start'] > rec['t_end']:
        rows.append(row(name, 'error', 'window',
                        f't_start {rec["t_start"]} is after t_end {rec["t_end"]}'))
    if os.path.basename(rec['csv']) != FIXED_NAME:
        rows.append(row(name, 'warning', 'csv',
                        f'{os.path.basename(rec["csv"])} is not {FIXED_NAME}; the unfixed matrix '
                        'holds most traces at the wrong timepoints, which a window cuts wrongly'))
    return rows


def check_thresholds(name, rec):
    """The optional numbers, which are silently wrong rather than loudly wrong when out of range."""
    rows = []
    for key in ('noise_floor', 'noise_peak'):
        value = rec.get(key)
        if value is not None and not (isinstance(value, (int, float)) and value >= 0):
            rows.append(row(name, 'error', 'thresholds', f'{key} must be a number >= 0, '
                                                         f'got {value!r}'))
    value = rec.get('min_presence')
    if value is not None and not (isinstance(value, (int, float)) and 0.0 <= value <= 1.0):
        rows.append(row(name, 'error', 'thresholds',
                        f'min_presence must be a number in 0..1, got {value!r}'))
    return rows


def load_matrix(name, rec):
    """The expression matrix indexed by timepoint, or the rows explaining why it can't be read."""
    try:
        df = pd.read_csv(rec['csv'])
    except Exception as e:
        return None, [row(name, 'error', 'csv', f'cannot be read: {e}')]
    if 'timepoint' not in df.columns:
        return None, [row(name, 'error', 'csv', 'has no timepoint column')]
    return df.set_index('timepoint').sort_index(), []


def check_window(name, rec, df):
    """The window against the matrix, and how many cells it leaves after min_presence."""
    lo, hi = int(df.index.min()), int(df.index.max())
    if not lo <= rec['t_start'] <= rec['t_end'] <= hi:
        return [row(name, 'error', 'window', f'{rec["t_start"]}..{rec["t_end"]} is not inside '
                                             f'the matrix\'s {lo}..{hi}')]

    min_presence = rec.get('min_presence', MIN_PRESENCE)
    presence = df.loc[rec['t_start']:rec['t_end']].notna().mean()
    n = int(((presence >= min_presence) & (presence > 0)).sum())
    if n == 0:
        return [row(name, 'error', 'cells', f'no cell of {len(presence)} is tracked through '
                                            f'{min_presence:.0%} of the window')]
    if n < FEW_CELLS:
        return [row(name, 'warning', 'cells', f'only {n} of {len(presence)} cells survive '
                                              f'min_presence {min_presence}')]
    return []


def check_tracklets(name, rec, columns):
    """Every cell column of the matrix needs its tracklet: cell_positions and render_overlay_gif
    look the id up without asking."""
    try:
        with open(rec['tracklets']) as f:
            tracklets = json.load(f)
    except Exception as e:
        return [row(name, 'error', 'tracklets', f'cannot be read: {e}')]

    misnamed = [c for c in columns if not c.startswith('cell_')]
    if misnamed:
        return [row(name, 'error', 'csv', f'{len(misnamed)} columns are not named cell_<id>, '
                                          f'e.g. {misnamed[:3]}')]
    missing = [c for c in columns if c[len('cell_'):] not in tracklets]
    if missing:
        return [row(name, 'error', 'tracklets', f'{len(missing)} of {len(columns)} cells have no '
                                                f'tracklet, e.g. {missing[:3]}')]
    return []


def check_masks(name, rec):
    """A mask file for every timepoint of the window."""
    try:
        available = set(mask_paths_by_t(rec['masks_dir']))
    except ValueError as e:  # get_masks_paths raises when nothing matches the naming convention
        return [row(name, 'error', 'masks', str(e))]
    missing = [t for t in range(rec['t_start'], rec['t_end'] + 1) if t not in available]
    if missing:
        return [row(name, 'error', 'masks', f'{len(missing)} of '
                                            f'{rec["t_end"] - rec["t_start"] + 1} window '
                                            f'timepoints have no mask, e.g. {missing[:5]}')]
    return []


def check_across_recordings(recordings):
    """The comparability traps: each recording is fine on its own and the comparison is not.

    A threshold that differs between recordings is only meaningful if the recordings are
    photometrically matched, which nothing verifies -- ellipse_sum is a raw AU sum.  Compare
    src/cell_activity_calibration.py, which measures where each floor actually sits.
    """
    rows = []
    for key, default, why in (
            ('noise_floor', NOISE_FLOOR, 'the same counts mean different things in each'),
            ('noise_peak', NOISE_PEAK, 'the blip level is drawn in a different place in each'),
            ('min_presence', MIN_PRESENCE, 'different cells enter the shared ladder from each')):
        values = {name: rec.get(key, default) for name, rec in recordings.items()}
        if len(set(values.values())) > 1:
            rows.append(row(ACROSS, 'warning', key, f'differs between recordings ({values}): '
                                                    f'{why}'))

    lengths = {name: rec['t_end'] - rec['t_start'] + 1 for name, rec in recordings.items()}
    if lengths and max(lengths.values()) > WINDOW_RATIO * min(lengths.values()):
        rows.append(row(ACROSS, 'warning', 'window',
                        f'window lengths differ by more than {WINDOW_RATIO:g}x ({lengths}): rate '
                        'normalises by tracked frames, duty and onset compare less cleanly'))
    return rows


def validate_config(recordings, cache_dir=None):
    """Every check over {name: recording dict}, as a table of recording, level, check, message.

    Written to cache_dir/validation.csv when cache_dir is given.
    """
    rows, sound = [], {}
    for name, rec in recordings.items():
        found = check_structure(name, rec) + check_thresholds(name, rec)
        if not any(r['level'] == 'error' for r in found):
            df, errors = load_matrix(name, rec)
            found += errors
            if df is not None:
                found += check_window(name, rec, df) + check_tracklets(name, rec, df.columns)
                found += check_masks(name, rec)
        if not any(r['level'] == 'error' for r in found):
            sound[name] = rec
        rows += found

    if len(sound) > 1:
        rows += check_across_recordings(sound)

    report = pd.DataFrame(rows, columns=['recording', 'level', 'check', 'message'])
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        report.to_csv(os.path.join(cache_dir, 'validation.csv'), index=False)
    return report


def raise_on_errors(report):
    """Print the report and stop if anything in it is an error."""
    for _, r in report.iterrows():
        print(f'{r["level"]:>7}  {r["recording"]}  [{r["check"]}]  {r["message"]}')
    errors = int((report['level'] == 'error').sum())
    if errors:
        raise ValueError(f'{errors} error(s) in the recording config; see the rows above')
    if report.empty:
        print('config ok')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='json of {name: recording dict}')
    p.add_argument('--out-dir', help='validation.csv is written here when given')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        recordings = json.load(f)
    raise_on_errors(validate_config(recordings, args.out_dir))


if __name__ == '__main__':
    main()
