"""Where each recording's noise floor actually sits in its own background.

ellipse_sum is a raw, uncalibrated AU sum (sum_pixels_in_sigma_ellipse in
ms2_gene_expression.py), so an absolute threshold like NOISE_FLOOR = 20 only means the same
thing in two recordings if the two were imaged under matched conditions -- laser power,
exposure, detector gain, depth, bleaching.  Nothing in the pipeline verifies that, and
src/cell_activity_compare.py puts recordings on one shared ladder regardless.

This does not calibrate anything.  It measures each recording's background distribution and
reports where its configured floor falls in it, so the assumption is visible before the
comparison is believed:

    floor_percentile    the share of nonzero tracked values the floor cuts away.  A floor at
                        the 8th percentile in one recording and the 40th in another is one
                        constant meaning two different things, and every number downstream --
                        level composition, rate, duty, onset -- inherits the difference.
    median_min_nonzero  the median over cells of each cell's smallest nonzero value, the
                        heuristic quoted in src/cell_activity.py's NOISE_FLOOR comment ("the
                        per-cell smallest non-zero value averages 22 in New-02"), computed per
                        recording rather than remembered from one.

Reads the matrix only, so it is cheap -- no masks are touched.

    python -m src.cell_activity_calibration --config recordings.json --out-dir .../calibration

recordings.json is the {name: recording dict} described in src/cell_activity.py.
"""
import argparse
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cell_activity import MIN_PRESENCE, NOISE_FLOOR, NOISE_PEAK, load_window, save_figure

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def calibration_row(name, rec):
    """One recording's background summary and the nonzero values it is measured over.

    Zeros are tracked frames where no emitter was accepted, so they carry no photometry; they
    are counted as frac_zero and kept out of the distribution.  See the module docstring for
    what to read in the row.
    """
    signals, present, cells, _ = load_window(
        rec['csv'], rec['t_start'], rec['t_end'], rec.get('min_presence', MIN_PRESENCE))
    tracked = signals[present]
    nonzero = tracked[tracked > 0]
    per_cell = np.array([s[p & (s > 0)].min() for s, p in zip(signals, present)
                         if (p & (s > 0)).any()])
    noise_floor = rec.get('noise_floor', NOISE_FLOOR)
    noise_peak = rec.get('noise_peak', NOISE_PEAK)

    above = np.clip(signals - noise_floor, 0.0, None)
    row = {'recording': name, 'n_cells': len(cells), 'n_tracked_frames': int(tracked.size),
           'frac_zero': float((tracked == 0).mean()) if tracked.size else np.nan,
           'median_min_nonzero': float(np.median(per_cell)) if per_cell.size else np.nan,
           'noise_floor': noise_floor,
           'floor_percentile': float((nonzero < noise_floor).mean() * 100) if nonzero.size
                               else np.nan,
           'frac_above_floor': float((tracked > noise_floor).mean()) if tracked.size else np.nan,
           'noise_peak': noise_peak,
           'frac_cells_clearing_peak': float((above.max(axis=1) >= noise_peak).mean())}
    for p in PERCENTILES:
        row[f'p{p}'] = float(np.percentile(nonzero, p)) if nonzero.size else np.nan
    return row, nonzero


def plot_background(distributions, floors, path_stem):
    """One ECDF per recording over the nonzero tracked values, its floor as a dashed line in the
    same colour.  Comparable floors sit at the same height on comparable curves."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    for (name, values), color in zip(distributions.items(), colors):
        if not len(values):
            continue
        ordered = np.sort(values)
        ax.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), color=color, lw=2,
                label=name)
        ax.axvline(floors[name], color=color, ls='--', lw=1.2)
    ax.set_xscale('log')
    ax.set(xlabel='ellipse sum of tracked, nonzero frames (AU)',
           ylabel='fraction of frames at or below', ylim=(0, 1),
           title='Background per recording; dashed = its configured noise floor')
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, path_stem, eps=False)


def run_calibration(recordings, out_dir):
    """Summarise every recording's background into out_dir/calibration.csv and background.png.
    Returns the table."""
    os.makedirs(out_dir, exist_ok=True)
    rows, distributions, floors = [], {}, {}
    for name, rec in recordings.items():
        row, distributions[name] = calibration_row(name, rec)
        rows.append(row)
        floors[name] = rec.get('noise_floor', NOISE_FLOOR)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, 'calibration.csv'), index=False)
    plot_background(distributions, floors, os.path.join(out_dir, 'background'))

    spread = table['floor_percentile'].max() - table['floor_percentile'].min()
    print(table.set_index('recording')[
        ['n_cells', 'median_min_nonzero', 'noise_floor', 'floor_percentile',
         'frac_above_floor']].to_string())
    if len(table) > 1 and spread > 10:
        print(f'warning: the configured noise floors sit {spread:.0f} percentiles apart in their '
              'own backgrounds -- cross-recording composition and rate are not on one scale')
    return table


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='json of {name: recording dict}')
    p.add_argument('--out-dir', required=True, help='calibration.csv and background.png go here')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        recordings = json.load(f)
    run_calibration(recordings, args.out_dir)


if __name__ == '__main__':
    main()
