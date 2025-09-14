import gc
from src.gene_expression.ms2_visualization import MS2VisualizationManager
from src.gene_expression.ms2_peak_strategies import GlobalPeakStrategy
from src.utils.cell_utils import get_3d_bounding_box_corners, calculate_center_of_mass_3d
from cell_tracking import get_masks_paths

import os
import json
import argparse
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage

import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')


class MS2GeneExpressionProcessor:
    """
    Processes MS2 gene expression data for individual cells across timepoints.

    This class analyzes gene expression by fitting 2D Gaussians to MS2 signal
    within cell boundaries and tracks expression over time.
    """

    def __init__(self, tracklets, czi_file_path, masks_paths, ms2_background_removed, output_dir='output', plot=None,
                 ransac_mad_k_th=2, strategy: str | None = 'global', prominence: float | None = 20):
        """
        Initialize the MS2 gene expression processor.

        Args:
            tracklets (dict): Cell tracking data with cell IDs as keys
            czi_file_path (str): Path to the raw CZI image file.
            masks_paths (list): Paths to segmentation mask files
            ms2_background_removed (np.ndarray): MS2 channel background removed images
            output_dir (str): Directory for saving output files
        """
        self.tracklets = tracklets
        self.czi_file_path = czi_file_path
        self.mask_file_paths = masks_paths
        self.ms2_background_removed = ms2_background_removed
        self.ms2_z_projections = self.ms2_background_removed.sum(
            axis=1)  # Z projection
        self.output_dir = output_dir
        self.plot = plot
        self.ransac_mad_k_th = ransac_mad_k_th
        self.strategy_name = strategy  # TODO: redundant
        self.czi_reader = None
        self.prominence = prominence

        # Only initialize the CZI reader if segmentation plotting is enabled to save memory.
        if self.plot and self.plot.get('segmentation', False):
            try:
                from aicspylibczi import CziFile
                self.czi_reader = CziFile(self.czi_file_path)
            except ImportError:
                print(
                    "Warning: aicspylibczi is not installed. CZI data for segmentation plotting will not be loaded.")
            except Exception as e:
                print(
                    f"Warning: Could not open CZI file {self.czi_file_path}: {e}")
                self.czi_reader = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        if self.plot is None:
            self.visualizer = MS2VisualizationManager(
                output_dir, enabled=False)
        else:
            self.visualizer = MS2VisualizationManager(output_dir, enabled=True)
        # Initialize processing state variables
        self._reset_processing_state()
        self._init_strategy()

    def _init_strategy(self):
        self.strategy = self._build_strategy()
        # Match peaks to cell emitters
        emitter_cells_matches = os.path.join(
            os.getcwd(), f'peak_to_cell_matching_prominence_{self.prominence}.csv')
        if os.path.exists(emitter_cells_matches):
            self.strategy.emitter_cell_matching(self, emitter_cells_matches)
        else:
            self.strategy.emitter_cell_matching(self)

    def set_strategy(self, strategy: str):
        self.strategy_name = strategy
        self.strategy = self._build_strategy()

    def _build_strategy(self):
        if isinstance(self.strategy_name, str):
            if self.strategy_name == 'global':
                return GlobalPeakStrategy(self.ms2_z_projections, prominence=self.prominence)
            else:
                raise ValueError(f"Unknown strategy '{self.strategy_name}'")
        elif hasattr(self.strategy_name, "fit_timepoint"):
            return self.strategy_name
        else:
            raise ValueError("Invalid strategy object")

    def _reset_processing_state(self):
        """Reset variables used during processing."""
        self.ellipse_sums = []
        self.cell_noise = []
        self.visualization_figures = []
        self.segmentation_figures = []
        self.cell_labels_by_timepoint = []
        self.max_cell_intensity = 0
        self.current_cell_mask_projection = None
        self.current_cell_bbox_ms2 = None
        self.cell_center_of_mass = []
        self.cell_center_debug = []
        self.fit_gaussian_centers_list = []
        self.gaussian_fit_params = []
        self.cell_initial_center = []
        self.cell_df = pd.DataFrame(
            columns=['timepoint', 'x', 'y', 'intensity'])
        self.final_df = pd.DataFrame(
            columns=['timepoint', 'x', 'y', 'intensity'])
        # Also reset dataframes from strategies that might have been attached
        # to the instance from previous runs with different strategies.
        if hasattr(self, 'guessed_gaussian_df'):
            del self.guessed_gaussian_df

    def process_cell(self, cell_id, method: str | None = None):
        # Explicitly run garbage collection to free up memory from previous runs,
        # which can be helpful when processing many cells in a loop.
        gc.collect()

        method = method or self.strategy_name
        self.cell_id = cell_id
        print(f"Processing cell ID: {cell_id} with strategy '{method}'")

        # If the requested strategy is different from the current one, rebuild it.
        # TODO: redundant
        if self.strategy_name != method:
            self.set_strategy(method)

        # Reset state for new cell
        self._reset_processing_state()

        # Get cell tracking data
        self.cell_labels_by_timepoint = self.tracklets[str(cell_id)]
        valid_timepoints = self._get_valid_timepoints()
        if not valid_timepoints:
            print("No valid timepoints.")
            return

        self._calculate_max_cell_intensity(valid_timepoints)
        self.strategy.pre_process_cell(self, valid_timepoints)
        # init visualizer for cell
        self.visualizer.start_cell(cell_id, self.max_cell_intensity)

        # Process each timepoint
        for timepoint in tqdm(valid_timepoints, desc=f"Cell {cell_id}"):
            self._process_single_timepoint(timepoint)

        self._save_plots_and_animations(valid_timepoints)
        self._save_csv()
        return np.array(self.ellipse_sums), np.array(self.cell_noise), np.mean(np.array(self.cell_center_of_mass), axis=0)

    def _get_valid_timepoints(self):
        """Get timepoints where the cell is present (label != -1)."""
        return [t for t in range(len(self.cell_labels_by_timepoint))
                if self.cell_labels_by_timepoint[t] != -1]

    def _calculate_max_cell_intensity(self, valid_timepoints):
        """
        Calculate maximum MCP intensity across all timepoints for consistent visualization.

        Args:
            valid_timepoints (list): List of valid timepoint indices
        """
        max_intensity = 0

        for timepoint in valid_timepoints:
            # Load masks and get cell-specific data
            masks = np.load(self.mask_file_paths[timepoint])['masks']
            ms2_projection = self.ms2_z_projections[timepoint]
            cell_label = self.cell_labels_by_timepoint[timepoint]

            # Create cell mask and get bounding box
            cell_mask_3d = np.empty(masks.shape, dtype=np.uint8)
            np.equal(masks, cell_label, out=cell_mask_3d, casting='unsafe')
            z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)
            self.cell_center_debug.append(((x1 + x2) // 2, (y1 + y2) // 2))

            # Project cell mask to 2D
            cell_mask_2d = np.sum(cell_mask_3d, axis=0)
            # Normalize
            cell_mask_2d = (cell_mask_2d > 0).astype(np.uint8)

            cell_region_ms2 = ms2_projection[y1:y2, x1:x2]
            mask_region = cell_mask_2d[y1:y2, x1:x2]

            masked_ms2 = cell_region_ms2 * \
                mask_region.astype(ms2_projection.dtype)
            max_intensity = max(max_intensity, masked_ms2.max())

        self.max_cell_intensity = max_intensity

    def _process_single_timepoint(self, timepoint):
        z_stack, ms2_stack, masks, ms2_projection = self._load_data_at_timepoint(
            timepoint)
        cell_label = self.cell_labels_by_timepoint[timepoint]
        cell_mask_3d = np.empty(masks.shape, dtype=np.uint8)
        np.equal(masks, cell_label, out=cell_mask_3d, casting='unsafe')
        center = calculate_center_of_mass_3d(cell_mask_3d)
        if center is not None:
            self.cell_center_of_mass.append(center)
        current_cell_mask_projection = (
            np.sum(cell_mask_3d, axis=0) > 0).astype(np.uint8)

        # Ensure bbox and bbox image are set (used by ellipse sum)
        z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)
        self.current_cell_bbox_ms2 = ms2_projection[y1:y2, x1:x2]

        cell_df_t = self.cell_df[self.cell_df['timepoint'] == timepoint]
        if cell_df_t.empty:
            peak_xy = (0, 0)
            gaussian_params = None
        elif len(cell_df_t) == 1:
            row = cell_df_t.iloc[0]
            if row['diff_intensity_slice'] / row['intensity'] >= 0.5:
                peak_xy = (0, 0)
                gaussian_params = None
            else:
                peak_xy, gaussian_params = self._filter_emitter(row, x1, y1)
            if peak_xy is not None and gaussian_params is not None:
                self.final_df = pd.concat(
                    [self.final_df, row.to_frame().T], ignore_index=True)
        else:
            if len(cell_df_t) > 1:
                if (cell_df_t['diff_intensity_slice'].to_numpy() < 10).all():
                    row = cell_df_t.sort_values(['gauss_sigma_x', 'gauss_sigma_y', 'circular_z_score'],
                                                ascending=[False, False, True],
                                                na_position='last').iloc[0]
                else:
                    row = cell_df_t.sort_values(['diff_intensity_slice'],
                                                ascending=[True],
                                                na_position='last').iloc[0]
                peak_xy, gaussian_params = self._filter_emitter(row, x1, y1)
                if peak_xy is not None and gaussian_params is not None:
                    self.final_df = pd.concat(
                        [self.final_df, row.to_frame().T], ignore_index=True)

        if gaussian_params is not None:
            # TODO: debug sum pixels function
            _, ellipse_sum, noise = self.sum_pixels_in_sigma_ellipse(
                gaussian_params,
                image_region=self.current_cell_bbox_ms2,
                k=2,
                subtract_offset=True,
                clip_negative=True,
                pct_floor=0.85)
        else:
            masked_vals = self.current_cell_bbox_ms2 * \
                (current_cell_mask_projection[y1:y2, x1:x2] > 0)
            noise = np.median(masked_vals[masked_vals > 0])
            noise = np.median(np.abs(masked_vals[masked_vals > 0] - noise))
            noise = 1.4826 * noise
            ellipse_sum = 0.0  # TODO: calculate noise

        self.ellipse_sums.append(ellipse_sum)
        self.cell_noise.append(noise)
        self.gaussian_fit_params.append(gaussian_params)
        if self.plot:
            self.visualizer.add_timepoint(
                timepoint=timepoint,
                ms2_projection=ms2_projection,
                cell_mask_projection_2d=current_cell_mask_projection,
                cell_bbox_ms2=self.current_cell_bbox_ms2,
                bbox_coords=(z1, y1, x1, z2, y2, x2),
                gaussian_params=gaussian_params,
                method=self.strategy.name,
                peak_xy=peak_xy,
                add_segmentation_3d=True,
                z_stack=z_stack,
                masks=masks,
                cell_label=cell_label
            )

    @staticmethod
    def _get_gaussian_params(row, x1, y1):
        # Build gaussian params in bbox coordinates
        # Fallback to peak x/y if fitted centers are missing
        gx_abs = row['gauss_x0'] if pd.notna(
            row.get('gauss_x0', np.nan)) else row['x']
        gy_abs = row['gauss_y0'] if pd.notna(
            row.get('gauss_y0', np.nan)) else row['y']
        gaussian_params = {'x0': float(gx_abs) - x1,
                           'y0': float(gy_abs) - y1,
                           'sigma_x': float(row.get('gauss_sigma_x', np.nan)),
                           'sigma_y': float(row.get('gauss_sigma_y', np.nan)),
                           'amplitude': float(row.get('gauss_amplitude', np.nan)) if pd.notna(row.get('gauss_amplitude', np.nan)) else None,
                           'offset': float(row.get('gauss_offset', 0.0)) if pd.notna(row.get('gauss_offset', np.nan)) else 0.0
                           }
        return gaussian_params

    def _filter_emitter(self, row, x1, y1):
        """
        The following method filter emitter by the area of sigma ellipse or Z score
        """
        #TODO: add sigma less than 0.3 in an direction as filter
        if (row['gauss_sigma_x'] <= 0.48 and row['gauss_sigma_y'] <= 0.48) or row['circular_z_score'] > 3:
            peak_xy = (0, 0)
            gaussian_params = None
        else:
            peak_xy = row['initial_center']
            gaussian_params = self._get_gaussian_params(row, x1, y1)
        return peak_xy, gaussian_params

    def _load_data_at_timepoint(self, timepoint):
        z_stack = None
        # ms2_stack was loaded but never used in the calling function, so we can skip it.
        ms2_stack = None

        # Only load z_stack from the CZI file if the reader is available (i.e., plotting is on)
        if self.czi_reader:
            try:
                # Assuming C=1 is the channel for z_stack (e.g., phase contrast)
                # and we are interested in the first scene (S=0).
                z_stack_data, _ = self.czi_reader.read_image(
                    T=timepoint, C=1, S=0)
                # Squeeze to remove dimensions of size 1, to get (Z, Y, X)
                z_stack = np.squeeze(z_stack_data)
            except Exception as e:
                print(
                    f"Warning: Could not read timepoint {timepoint} from CZI file: {e}")

        masks = np.load(self.mask_file_paths[timepoint])['masks']
        ms2_projection = self.ms2_z_projections[timepoint]
        return z_stack, ms2_stack, masks, ms2_projection

    def sum_pixels_in_sigma_ellipse(self, gaussian_params,
                                    image_region=None,
                                    k=1.0,
                                    subtract_offset=False,
                                    clip_negative=True,
                                    k_bg=3.0,
                                    k_sigma=3.5,
                                    pct_floor=0.88
                                    ):
        """
        Sum pixel intensities inside the ellipse centered at (x0, y0) with semi-axes
        k*sigma_x and k*sigma_y (aligned with image axes).

        Args:
            gaussian_params (dict): Must contain 'x0','y0','sigma_x','sigma_y'. Optional 'offset'.
            image_region (np.ndarray|None): 2D image region to use. Defaults to self.current_cell_bbox_ms2.
            k (float): Scale factor for the ellipse radii (1.0 -> exactly sigma_x, sigma_y).
            subtract_offset (bool): If True, subtract gaussian_params['offset'] from pixels before summing.
            clip_negative (bool): If True, clip values below 0 after offset subtraction.

        Returns:
            tuple[float, int]: (sum_of_values, number_of_pixels_in_ellipse)
        """
        if gaussian_params is None:
            return 0.0, 0

        img = image_region if image_region is not None else self.current_cell_bbox_ms2
        if img is None:
            return 0.0, 0

        h, w = img.shape
        x0 = float(np.clip(gaussian_params.get('x0', 0.0), 0, w - 1))
        y0 = float(np.clip(gaussian_params.get('y0', 0.0), 0, h - 1))
        sx = float(gaussian_params.get('sigma_x', 0.0))
        sy = float(gaussian_params.get('sigma_y', 0.0))

        # Guard against invalid sigmas
        if not np.isfinite(sx) or not np.isfinite(sy) or sx <= 0 or sy <= 0:
            return 0.0, 0

        # Ellipse mask: ((x-x0)/(k*sx))^2 + ((y-y0)/(k*sy))^2 <= 1
        Y, X = np.ogrid[:h, :w]
        a2 = (k * sx) ** 2
        b2 = (k * sy) ** 2
        ellipse_mask = ((X - x0) ** 2) / a2 + ((Y - y0) ** 2) / b2 <= 1.0

        # Normalized radial distance in σ units
        # Annulus mask is confined in the bbox around cell contour
        r = np.sqrt(((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2)
        annulus_mask = (r >= k_bg) & (r <= 2*k_bg)

        vals = img[ellipse_mask].astype(np.float64, copy=False)
        annulus_vals = img[annulus_mask].astype(np.float64, copy=False)

        if subtract_offset:
            base = float(gaussian_params.get('offset', 0.0))
            vals = vals - base
            annulus_vals = annulus_vals - base
            if clip_negative:
                vals = np.clip(vals, 0.0, None)
                annulus_vals = np.clip(annulus_vals, 0.0, None)
         # Robust background stats
        if annulus_vals.size:
            bg_median = np.median(annulus_vals)
            bg_mad = np.median(np.abs(annulus_vals - bg_median))
            bg_sigma = 1.4826 * \
                bg_mad if bg_mad > 0 else (
                    np.std(annulus_vals) if annulus_vals.size > 1 else 1.0)
        else:
            bg_median, bg_sigma = (np.median(vals) if vals.size else 0.0, np.std(
                vals) if vals.size > 1 else 1.0)

        # Percentile floor inside ellipse (prevents threshold too low)
        pct_thr = np.percentile(vals, pct_floor * 100) if vals.size else 0.0

        # Base robust threshold
        thr_robust = bg_median + k_sigma * bg_sigma
        threshold = max(thr_robust, pct_thr)

        intensity = sum(vals[vals >= threshold])

        return vals, intensity, bg_sigma

    def process(self, cell_id):
        """Backward compatibility wrapper for process_cell."""
        self.process_cell(cell_id)

    def _save_plots_and_animations(self, valid_timepoints):
        if self.plot['emitter_fit']:
            self.visualizer.save_timepoint_animation(
                self.cell_id, valid_timepoints, self.ransac_mad_k_th)
        if self.plot['intensity']:
            self.visualizer.expression_plot(
                self.cell_id, self.ellipse_sums, 'Ellipse_sum')
        if self.plot['segmentation']:
            self.visualizer.save_segmentation_animation(
                self.cell_id, valid_timepoints)

    def _save_csv(self):
        if self.strategy_name == 'global' and hasattr(self, 'cell_df'):
            self.cell_df.to_csv(
                os.path.join(self.output_dir,
                             f"cell_{self.cell_id}_data_global_peaks.csv"),
                index=False
            )
            self.final_df.to_csv(
                os.path.join(self.output_dir,
                             f"cell_{self.cell_id}_data_global_peaks_final.csv"),
                index=False
            )
        if self.strategy_name == 'local' and hasattr(self, 'guessed_gaussian_df'):
            out_df = self.guessed_gaussian_df.copy()
            out_df.to_csv(
                os.path.join(self.output_dir,
                             f"cell_{self.cell_id}_data_local_peaks.csv"),
                index=False
            )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process MS2 gene expression data.")
    parser.add_argument("--czi_file_path", type=str, required=True,
                        help="Path to the CZI file.")
    parser.add_argument("--seg_maps_dir", type=str, required=True,
                        help="Path to the segmentation maps directory.")
    parser.add_argument("--tracklets_path", type=str, required=True,
                        help="Path to tracklet JSON file.")
    parser.add_argument("--ms2_background_removed", type=str, required=True,
                        help="Path to MS2 channel after background removal.")
    parser.add_argument("--output_dir", type=str, required=False, default='output',
                        help="Path to the output directory.")
    parser.add_argument('--prominence', type=float, required=False, default=18.0,
                        help="Maxima finder prominence")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load tracklets
    with open(args.tracklets_path, 'r') as f:
        tracklets = json.load(f)

    # Load MS2 z-projections
    ms2_background_removed = tifffile.imread(args.ms2_background_removed)

    masks_paths = get_masks_paths(args.seg_maps_dir)

    # Create processor instance
    processor = MS2GeneExpressionProcessor(
        tracklets=tracklets,
        czi_file_path=args.czi_file_path,
        masks_paths=masks_paths,
        ms2_background_removed=ms2_background_removed,
        output_dir=args.output_dir,
        plot={'emitter_fit': False, 'intensity': False, 'segmentation': False},
        ransac_mad_k_th=2.0,
        prominence=args.prominence
    )
    # Example: Process a specific cell  using 'global' strategy
    # amp = processor.process_cell(6, 'global')

    num_timepoints = ms2_background_removed.shape[0]
    expression_matrix = {
        'timepoint': list(range(num_timepoints))
    }
    # # process cells that have been tracked for all frames
    # valid_ids = [key for key, cell_labels in tracklets.items()
    #              if cell_labels.count(-1) < 20]
    valid_ids = np.arange(0, 50)
    non_zero_min = []
    cells_center_of_mass_df = pd.DataFrame(columns=['cell_id', 'x', 'y', 'z','noise'])
    for cell_id in tqdm(valid_ids):
        amp, noise, cell_center_of_mass = processor.process_cell(cell_id, 'global')
        cells_center_of_mass_df = pd.concat([cells_center_of_mass_df, pd.DataFrame([{
            'cell_id': cell_id,
            'x': cell_center_of_mass[0],
            'y': cell_center_of_mass[1],
            'z': cell_center_of_mass[2],
            'noise': np.median(noise)
        }])], ignore_index=True)
        non_zero_min.append(np.min(amp[amp > 0])
                            if np.any(amp > 0) else np.nan)
        # Reconstruct full-length vector aligned to all timepoints
        labels = tracklets[str(cell_id)]
        full_series = [np.nan] * num_timepoints
        valid_timepoints = [t for t, lbl in enumerate(labels) if lbl != -1]

        # Map returned amplitudes to their corresponding timepoints
        for tp, amp in zip(valid_timepoints, amp):
            full_series[tp] = amp

        expression_matrix[f'cell_{cell_id}'] = full_series
    noise_level = np.nanmean(non_zero_min) if non_zero_min else 0
    print(f"Noise level: {noise_level}")
    df = pd.DataFrame(expression_matrix)
    # # Replace zeros in cell columns with noise_level
    # cell_cols = [c for c in df.columns if c.startswith('cell_')]
    # if noise_level is not None and cell_cols:
    #     df[cell_cols] = df[cell_cols].replace(0, noise_level)
    out_path = os.path.join(processor.output_dir,
                            'gene_expression_results.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved expression matrix to {out_path}")
    cells_center_of_mass_df.to_csv(os.path.join(
        processor.output_dir, 'cells_center_of_mass.csv'), index=False)
    print(
        f"Saved cells center of mass to {os.path.join(processor.output_dir, 'cells_center_of_mass.csv')}")
