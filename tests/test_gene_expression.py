"""ms2/gene_expression/run.py: a cell whose mask is one pixel thin on a frame is measured, not
a crash -- measuring every tracked cell, not only those in an ROI, meets such slivers.

    python tests/test_gene_expression.py
"""
import numpy as np

from fixtures import passed

from ms2.gene_expression.run import MS2GeneExpressionProcessor


def test_one_pixel_thin_mask_is_measured():
    masks = np.zeros((2, 2, 8, 8), dtype=np.uint16)          # t, z, y, x
    masks[0, :, 2:6, 2:6] = 1                                # a cell
    masks[1, 0, 3, 1:7] = 1                                  # a sliver one pixel tall
    ms2 = np.ones((2, 2, 8, 8), dtype=np.float32)
    ms2[0, :, 3, 3] = 9
    p = MS2GeneExpressionProcessor.__new__(MS2GeneExpressionProcessor)
    p.ms2_z_projections = ms2.sum(axis=1)
    p.load_masks = lambda t: masks[t]
    p._mask_buffer, p.cell_center_debug, p.cell_id = None, [], '1'
    p.cell_labels_by_timepoint = [1, 1]
    p._calculate_max_cell_intensity([0, 1])
    assert p.max_cell_intensity == 18, p.max_cell_intensity
    passed('a one-pixel-thin mask adds nothing to the intensity scale instead of crashing')


def test_unconverged_gaussian_fit_is_kept_as_nan():
    """A spot whose Gaussian fit does not converge is kept with NaN fit parameters: one such spot
    stopped the all-cells measurement of a whole recording."""
    import pandas as pd
    from ms2.gene_expression.ms2_peak_strategies import GlobalPeakStrategy
    masks = np.zeros((2, 8, 8), dtype=np.uint16)              # z, y, x
    masks[:, 2:6, 2:6] = 1
    ms2 = np.ones((8, 8), dtype=np.float32)
    ms2[3, 3] = 9
    s = GlobalPeakStrategy(ms2[None], prominence=1)
    s.df_peaks = pd.DataFrame({'timepoint': [0], 'cell_label': [1], 'x_peak': [3], 'y_peak': [3],
                               'slice_score': [20.0]})
    s._gaussian_fit = lambda image, centre: (None, None)      # as a fit that raised does
    p = MS2GeneExpressionProcessor.__new__(MS2GeneExpressionProcessor)
    p._load_data_at_timepoint = lambda t: (None, None, masks, ms2)
    p.cell_labels_by_timepoint = [1]
    s.pre_process_cell(p, [0])
    assert len(p.cell_df) == 1 and np.isnan(p.cell_df['gauss_sigma_x'].iloc[0]), p.cell_df
    # ...and is then not taken as the frame's emitter: NaN passed every threshold, and the
    # pixel sum it reached stopped the next recording's measurement
    row = p.cell_df.iloc[0].copy()
    row['circular_z_score'] = 0.0
    assert not MS2GeneExpressionProcessor._emitter_is_valid(row)
    row['gauss_sigma_x'] = row['gauss_sigma_y'] = 1.0
    assert MS2GeneExpressionProcessor._emitter_is_valid(row)
    passed('a spot whose Gaussian fit does not converge is kept with NaN parameters, not as an emitter')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('gene_expression: all tests passed')


if __name__ == '__main__':
    main()
