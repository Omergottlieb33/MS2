"""Per-cell MS2 traces laid out on the absolute timeline.

process_cell returns a cell's ellipse sums indexed by absolute timepoint, but the expression
matrix used to pair them *positionally* with the frames where the cell is segmented.  Every
value after a missed detection -- and every value of a cell first seen after t=0 -- landed
later than it happened, and the tail was dropped.  In New-02-v3-ST11-12 that reproduces all
881 columns of gene_expression_results.csv, and 148 of the 151 cells with an emitter carry a
shifted trace.

Both paths of ms2_gene_expression.py (fresh fit and resume) now go through expression_series.
rebuild_results re-derives an existing matrix from the per-cell *_final.csv files, so old
outputs can be corrected without refitting:

    python -m src.gene_expression.expression_matrix \
        --results-dir .../gene_expression_roi_selection --tracklets .../masks/tracklets.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

RESULTS_NAME = 'gene_expression_results.csv'
FIXED_NAME = 'gene_expression_results_fixed.csv'


def final_csv_path(results_dir, cell_id):
    return os.path.join(results_dir, f'cell_{cell_id}_data_global_peaks_final.csv')


def amplitudes_by_timepoint(final_csv, n):
    """Length-n ellipse sums indexed by absolute timepoint, 0 where no emitter was accepted.
    The same layout process_cell returns."""
    df = pd.read_csv(final_csv)
    amplitudes = np.zeros(n)
    amplitudes[df['timepoint'].to_numpy(dtype=int)] = df['ellipse_sum'].to_numpy(dtype=float)
    return amplitudes


def expression_series(amplitudes, labels, n):
    """A cell's column of the expression matrix: amplitudes[t] where the tracklet has a label,
    NaN where it has none (-1 gap, -2 exited, or not yet born)."""
    series = np.full(n, np.nan)
    for t, label in enumerate(labels[:n]):
        if label > 0:
            series[t] = amplitudes[t]
    return series


def rebuild_results(results_dir, tracklets_path):
    """Write FIXED_NAME next to results_dir's expression matrix, rebuilt from the per-cell
    *_final.csv files.  Cell columns and timeline come from the existing matrix, so its cell
    selection (ROI or gap filter) is kept.  tracklets_path must be the file it was built from.
    """
    with open(tracklets_path) as f:
        tracklets = json.load(f)
    old = pd.read_csv(os.path.join(results_dir, RESULTS_NAME))
    n = len(old)
    fixed = {'timepoint': old['timepoint'].to_numpy()}
    for col in old.columns.drop('timepoint'):
        cell_id = col[len('cell_'):]
        amplitudes = amplitudes_by_timepoint(final_csv_path(results_dir, cell_id), n)
        fixed[col] = expression_series(amplitudes, tracklets[cell_id], n)
    out_path = os.path.join(results_dir, FIXED_NAME)
    pd.DataFrame(fixed).to_csv(out_path, index=False)
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', required=True,
                   help=f'folder holding {RESULTS_NAME} and the per-cell *_final.csv files')
    p.add_argument('--tracklets', required=True, help='tracklets json the matrix was built from')
    return p.parse_args()


def main():
    args = parse_args()
    print(f'fixed matrix -> {rebuild_results(args.results_dir, args.tracklets)}')


if __name__ == '__main__':
    main()
