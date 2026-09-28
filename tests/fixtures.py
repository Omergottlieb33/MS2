"""Tiny synthetic recordings for the tests, in the layout the real pipeline writes.

The repo has no test framework, so the tests are plain scripts of asserts run as

    python tests/test_cell_activity.py

Each builds what it needs here: an expression matrix named like
src/gene_expression/expression_matrix.py writes it, a tracklets json keyed the way
src/cell_activity.py reads it, and mask npz files named the way cell_tracking.py's
get_masks_paths expects.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gene_expression.expression_matrix import FIXED_NAME  # noqa: E402


def write_recording(out_dir, signals, n_frames=None, name=FIXED_NAME, masks=True):
    """A recording dict backed by real files.

    signals is (cells x frames); NaN marks a frame the cell is not tracked in, which is what
    the matrix holds and what load_window reads as untracked.
    """
    os.makedirs(out_dir, exist_ok=True)
    signals = np.asarray(signals, dtype=float)
    n_cells, n_frames = signals.shape[0], n_frames or signals.shape[1]

    csv_path = os.path.join(out_dir, name)
    frame = {'timepoint': np.arange(n_frames)}
    for i in range(n_cells):
        frame[f'cell_{i}'] = signals[i]
    pd.DataFrame(frame).to_csv(csv_path, index=False)

    # label 0 = not tracked, matching the tracklets the pipeline writes
    tracklets_path = os.path.join(out_dir, 'tracklets.json')
    with open(tracklets_path, 'w') as f:
        json.dump({str(i): [0 if np.isnan(v) else i + 1 for v in signals[i]]
                   for i in range(n_cells)}, f)

    # One 4x4 tile per cell on a roughly square grid, so the centroids span two dimensions --
    # a single row of cells gives ConvexHull a degenerate input.
    masks_dir = os.path.join(out_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    if masks:
        cols = int(np.ceil(np.sqrt(n_cells)))
        rows = int(np.ceil(n_cells / cols))
        for t in range(n_frames):
            block = np.zeros((1, 4 * rows, 4 * cols), dtype=np.int32)
            for i in range(n_cells):
                if not np.isnan(signals[i, t]):
                    r, c = divmod(i, cols)
                    block[0, 4 * r + 1:4 * r + 3, 4 * c + 1:4 * c + 3] = i + 1
            np.savez_compressed(os.path.join(masks_dir, f'z_stack_t{t}_seg_masks.npz'),
                                masks=block)

    return {'csv': csv_path, 'tracklets': tracklets_path, 'masks_dir': masks_dir,
            't_start': 0, 't_end': n_frames - 1}


def report_checks(report, level=None):
    """The (check, recording) pairs of a validation report, optionally at one level only."""
    if level is not None:
        report = report[report['level'] == level]
    return {(r['check'], r['recording']) for _, r in report.iterrows()}


def passed(name):
    print(f'  ok  {name}')
