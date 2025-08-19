from ms2_visualization import MS2VisualizationManager
from ms2_peak_strategies import GlobalPeakStrategy, LocalPeakStrategy
from src.utils.cell_utils import get_3d_bounding_box_corners, calculate_center_of_mass_3d, estimate_emitter_2d_gaussian_with_fixed_offset, filter_ransac_poly, estimate_background_offset_annulus
from src.utils.image_utils import load_czi_images
from cell_tracking import get_masks_paths

from findmaxima2d import find_maxima, find_local_maxima
from scipy.ndimage import binary_erosion

import os
import json
import argparse
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
matplotlib.use('module://matplotlib_inline.backend_inline')


def find_global_peak(ms2_z_projections, prominence=20):
    df = pd.DataFrame(columns=['frame', 'x', 'y', 'timepoint', 'intensity'])
    for timepoint in tqdm(range(ms2_z_projections.shape[0]), desc="Finding global peaks"):
        local_max = find_local_maxima(ms2_z_projections[timepoint])
        y, x, regs = find_maxima(
            ms2_z_projections[timepoint], local_max, prominence)
        frame_df = pd.DataFrame(
            {'frame': timepoint, 'x': x, 'y': y, 'intensity': ms2_z_projections[timepoint][y, x]})
        df = pd.concat([df, frame_df], ignore_index=True)
    return df


def get_indices_in_mask(coordinates: np.ndarray, cell_mask: np.ndarray, remove_outline: bool = False) -> np.ndarray:
    """
    Finds the indices of coordinates that are inside a boolean cell mask.
    Optionally removes the outline (outermost pixel layer) of the mask first.

    Args:
        coordinates (np.ndarray): A NumPy array of shape (n, 2) with (x, y) coordinates.
        cell_mask (np.ndarray): A 2D NumPy array of dtype bool, where True indicates
                                the area of interest.
        remove_outline (bool): If True, the outer layer of pixels of the True
                               areas in the mask is removed before checking
                               for coordinates. Defaults to False.

    Returns:
        np.ndarray: A 1D NumPy array containing the indices of the coordinates
                    that fall within the True areas of the cell_mask.
    """
    # --- Input Validation ---
    if not isinstance(coordinates, np.ndarray) or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("Coordinates must be a NumPy array of shape (n, 2).")
    if not isinstance(cell_mask, np.ndarray) or cell_mask.ndim != 2 or cell_mask.dtype != bool:
        raise ValueError("cell_mask must be a 2D boolean NumPy array.")

    # --- Mask Processing ---
    # If requested, remove the outer layer of pixels (outline) from the mask.
    # This process is called binary erosion.
    if remove_outline:
        # We use binary_erosion to shrink the True regions of the mask by one pixel
        # from all sides, effectively removing the outline.
        processed_mask = binary_erosion(cell_mask)
    else:
        processed_mask = cell_mask

    # --- Coordinate Processing ---
    # Round coordinates to the nearest integer to use them as indices.
    # Using np.round is safer than just casting to int, as it handles floating point values correctly.
    int_coords = np.round(coordinates).astype(int)

    # Extract x and y columns. Note that in image processing and array indexing,
    # x corresponds to columns (dimension 1) and y to rows (dimension 0).
    x_coords = int_coords[:, 0]
    y_coords = int_coords[:, 1]

    # Get the dimensions of the mask (height, width)
    mask_height, mask_width = processed_mask.shape

    # --- Boundary Check ---
    # Create a boolean mask to identify coordinates that are within the bounds of the cell_mask.
    # This is crucial to prevent IndexError when accessing the mask.
    in_bounds_mask = (
        (x_coords >= 0) & (x_coords < mask_width) &
        (y_coords >= 0) & (y_coords < mask_height)
    )

    # Get the original indices of the coordinates that are within bounds.
    # We will use these indices to filter our coordinates before checking the cell mask.
    original_indices_in_bounds = np.where(in_bounds_mask)[0]

    # If no coordinates are within the bounds, return an empty array.
    if len(original_indices_in_bounds) == 0:
        return np.array([], dtype=int)

    # Filter the coordinates to only include those within the mask's dimensions.
    x_in_bounds = x_coords[original_indices_in_bounds]
    y_in_bounds = y_coords[original_indices_in_bounds]

    # --- Mask Lookup ---
    # Use the valid, in-bounds coordinates to look up their values in the processed_mask.
    # NumPy's advanced indexing allows us to do this in a single, efficient operation.
    # The result is a boolean array indicating if each in-bounds point is in a True region.
    # Remember: array access is [row, column], which corresponds to [y, x].
    is_inside_mask = processed_mask[y_in_bounds, x_in_bounds]

    # --- Final Filtering ---
    # The final step is to select the original indices that correspond to a True value
    # in our 'is_inside_mask' array.
    final_indices = original_indices_in_bounds[is_inside_mask]

    return final_indices


class MS2GeneExpressionProcessor:
    """
    Processes MS2 gene expression data for individual cells across timepoints.

    This class analyzes gene expression by fitting 2D Gaussians to MS2 signal
    within cell boundaries and tracks expression over time.
    """

    def __init__(self, tracklets, image_data, masks_paths, ms2_z_projections, output_dir='output', plot=True,
                 ransac_mad_k_th=2, strategy: str | None = 'global'):
        """
        Initialize the MS2 gene expression processor.

        Args:
            tracklets (dict): Cell tracking data with cell IDs as keys
            image_data (np.ndarray): Raw image data array
            masks_paths (list): Paths to segmentation mask files
            ms2_z_projections (np.ndarray): MS2 channel z-projected images
            output_dir (str): Directory for saving output files
        """
        self.tracklets = tracklets
        self.image_data = image_data
        self.mask_file_paths = masks_paths
        self.ms2_z_projections = ms2_z_projections
        self.output_dir = output_dir
        self.plot = plot
        self.ransac_mad_k_th = ransac_mad_k_th
        self.strategy_name = strategy

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        self.visualizer = MS2VisualizationManager(output_dir, enabled=plot)
        # Initialize processing state variables
        self._reset_processing_state()
        self.strategy = self._build_strategy()

    def set_strategy(self, strategy: str):
        self.strategy_name = strategy

    def _build_strategy(self):
        if isinstance(self.strategy_name, str):
            if self.strategy_name == 'global':
                return GlobalPeakStrategy(self.ms2_z_projections)
            elif self.strategy_name == 'local':
                return LocalPeakStrategy()
            else:
                raise ValueError(f"Unknown strategy '{self.strategy_name}'")
        elif hasattr(self.strategy_name, "fit_timepoint"):
            return self.strategy_name
        else:
            raise ValueError("Invalid strategy object")

    def _reset_processing_state(self):
        """Reset variables used during processing."""
        self.expression_amplitudes = []
        self.expression_amplitudes2 = []
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
        self.cell_df = pd.DataFrame(columns=['timepoint', 'x', 'y', 'intensity'])

    def process_cell(self, cell_id, method: str | None = None):
        method = method or self.strategy_name
        self.cell_id = cell_id
        print(f"Processing cell ID: {cell_id} with strategy '{method}'")

        # Reset state for new cell
        self._reset_processing_state()

        # Get cell tracking data
        self.cell_labels_by_timepoint = self.tracklets[str(cell_id)]
        valid_timepoints = self._get_valid_timepoints()

        if not valid_timepoints:
            print("No valid timepoints.")
            return

        self._calculate_max_cell_intensity(valid_timepoints)

        # build and run strategy pre-process
        self.strategy_name = method

        self.strategy.pre_process_cell(self, valid_timepoints)

        # init visualizer for cell
        self.visualizer.start_cell(cell_id, self.max_cell_intensity)

        # Process each timepoint
        for timepoint in tqdm(valid_timepoints, desc=f"Cell {cell_id}"):
            self._process_single_timepoint(timepoint)

        if self.plot:
            self.visualizer.save_timepoint_animation(
                self.cell_id, valid_timepoints, self.ransac_mad_k_th)
            self.visualizer.expression_plot(
                self.cell_id, self.expression_amplitudes2, 'Ellipse_sum')
            self.visualizer.expression_plot(
                self.cell_id, self.expression_amplitudes, 'Gaussian_Integral')

        if method == 'global' and hasattr(self, 'cell_df'):
            self.cell_df.to_csv(
                os.path.join(self.output_dir,
                             f"cell_{cell_id}_data_global_peaks.csv"),
                index=False
            )
        if method == 'local' and hasattr(self, 'guessed_gaussian_df'):
            out_df = self.guessed_gaussian_df.copy()
            out_df.to_csv(
                os.path.join(self.output_dir,
                             f"cell_{cell_id}_data_local_peaks.csv"),
                index=False
            )

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
            cell_mask_3d = (masks == cell_label).astype(np.uint8)
            z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)
            self.cell_center_debug.append(((x1 + x2) // 2, (y1 + y2) // 2))

            # Project cell mask to 2D
            cell_mask_2d = np.sum(cell_mask_3d, axis=0)
            # Normalize
            cell_mask_2d = (cell_mask_2d > 0).astype(np.uint8)

            # Expand mask slightly and extract MS2 signal
            expanded_mask = ndimage.binary_dilation(cell_mask_2d, iterations=0)
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
        cell_mask_3d = (masks == cell_label).astype(np.uint8)
        center = calculate_center_of_mass_3d(cell_mask_3d)
        if center is not None:
            self.cell_center_of_mass.append(center)
        current_cell_mask_projection = (
            np.sum(cell_mask_3d, axis=0) > 0).astype(np.uint8)

        gaussian_params, covariance_matrix, peak_xy = self.strategy.fit_timepoint(
            self, timepoint, cell_mask_3d, ms2_projection
        )
        self.gaussian_fit_params.append(gaussian_params)
        if gaussian_params is not None:
            _, ellipse_sum = self.sum_pixels_in_sigma_ellipse(
                gaussian_params, k=2, subtract_offset=True, clip_negative=True, pct_floor=0.85
            )
            intensity = gaussian_params['amplitude'] * 2 * np.pi * \
                gaussian_params['sigma_x'] * gaussian_params['sigma_y']
        else:
            ellipse_sum = 0.0
            intensity = 0.0

        self.expression_amplitudes.append(intensity)
        self.expression_amplitudes2.append(ellipse_sum)

        if self.plot:
            z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)
            self.visualizer.add_timepoint(
                timepoint=timepoint,
                ms2_projection=ms2_projection,
                cell_mask_projection_2d=current_cell_mask_projection,
                cell_bbox_ms2=self.current_cell_bbox_ms2,
                bbox_coords=(z1, y1, x1, z2, y2, x2),
                gaussian_params=gaussian_params,
                covariance_matrix=covariance_matrix,
                method=self.strategy.name,
                peak_xy=peak_xy,
                add_segmentation_3d=True,
                z_stack=z_stack,
                masks=masks,
                cell_label=cell_label
            )

        

    def _load_data_at_timepoint(self, timepoint):
        z_stack = self.image_data[0, timepoint, 1, :, :, :, 0]
        ms2_stack = self.image_data[0, timepoint, 0, :, :, :, 0]
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
            bg_sigma = 1.4826 * bg_mad if bg_mad > 0 else (np.std(annulus_vals) if annulus_vals.size > 1 else 1.0)
        else:
            bg_median, bg_sigma = (np.median(vals) if vals.size else 0.0, np.std(vals) if vals.size > 1 else 1.0)

        # Percentile floor inside ellipse (prevents threshold too low)
        pct_thr = np.percentile(vals, pct_floor * 100) if vals.size else 0.0

        # Base robust threshold
        thr_robust = bg_median + k_sigma * bg_sigma
        threshold = max(thr_robust, pct_thr)

        # threshold = np.median(annulus_vals) if annulus_vals.size else 0.0
        threshold_naive = np.percentile(vals, 99) if vals.size else 0.0

        intensity = sum(vals[vals >= threshold])

        return vals, intensity

    def process(self, cell_id):
        """Backward compatibility wrapper for process_cell."""
        self.process_cell(cell_id)


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
    parser.add_argument("--ms2_filtered_z_projection", type=str, required=True,
                        help="Path to MS2 z-projected images (TIFF format).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load tracklets
    with open(args.tracklets_path, 'r') as f:
        tracklets = json.load(f)

    # Load image data
    image_data = load_czi_images(args.czi_file_path)

    # Load MS2 z-projections
    ms2_z_projections = tifffile.imread(args.ms2_filtered_z_projection)

    masks_paths = get_masks_paths(args.seg_maps_dir)

    # Create processor instance
    processor = MS2GeneExpressionProcessor(
        tracklets=tracklets,
        image_data=image_data,
        masks_paths=masks_paths,
        ms2_z_projections=ms2_z_projections,
        output_dir='/home/dafei/output/MS2/3d_cell_segmentation/gRNA2_12.03.25-st-13-II---/v2/enlarge_mask_for_match/',
        ransac_mad_k_th=2.0
    )
    # debug cell
    processor.process_cell(0, method='global')
    # processor.process_cell(1, method='global')
    # processor.process_cell(6, method='global')
    # processor.process_cell(7, method='global')
    # processor.process_cell(10, method='global')
    
    # for id in range(0,10):
    #     processor.process_cell(id, 'global')
    # for id in range(0, 20):
    #     processor.process_cell(id)
    # ids = list(tracklets.keys())
    # num_rows = 80
    # initial_data = {'timepoints': np.arange(0, num_rows)}
    # df = pd.DataFrame(initial_data)
    # for cell_id in range(0, 50):
    #     processor.process_cell(cell_id)
    #     print(f"Finished processing cell {cell_id}")
    #     if len(processor.expression_amplitudes) < num_rows:
    #         processor.expression_amplitudes.extend([-1] * (num_rows - len(processor.expression_amplitudes)))  # Fill missing timepoints with -1

    #     df[f'cell_{cell_id}'] = processor.expression_amplitudes
    # df.to_csv(os.path.join(processor.output_dir, 'gene_expression_results.csv'), index=False)
    # print(f"Gene expression results saved to {processor.output_dir}/gene_expression_results.csv")
