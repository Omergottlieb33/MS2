from mpl_toolkits.mplot3d import Axes3D  # ensures 3D projection is registered
from src.utils.plot_utils import show_3d_segmentation_overlay_with_unique_colors, plot_2d_gaussian_with_size, plot_gaussian_initial_guess
from src.utils.cell_utils import get_3d_bounding_box_corners, calculate_center_of_mass_3d, gaussian_2d, filter_ransac_poly, estimate_background_offset_annulus
from src.utils.gif_utils import create_gif_from_figures, create_trajectory_gif
from src.utils.image_utils import load_czi_images
from cell_tracking import get_masks_paths

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


def estimate_emitter_2d_gaussian_with_fixed_offset(image, initial_position, initial_sigma=1.0, fixed_offset=None):
    """
    Estimates the parameters of a 2D Gaussian emitter in an image.

    Args:
        image (2D array): Input image containing the emitter.
        initial_position (tuple): Initial guess for the (x, y) position of the emitter.
        initial_sigma (float): Initial guess for the Gaussian sigma (default: 1.0).
        fixed_offset (float): If provided, offset will be fixed to this value.

    Returns:
        tuple[dict|None, np.ndarray|None]: (params, covariance)
            params keys: amplitude, x0, y0, sigma_x, sigma_y, offset
    """
    # Ensure float for stable fitting
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape

    # Create a meshgrid for the image (y first from np.indices, but we pass (x, y) to the model)
    y, x = np.indices(img.shape)

    # Clip initial position safely into the image bounds
    x0_guess = float(np.clip(initial_position[0], 0, w - 1))
    y0_guess = float(np.clip(initial_position[1], 0, h - 1))

    # Amplitude guess
    if fixed_offset is not None:
        amp_guess = float(max(img.max() - float(fixed_offset), np.finfo(np.float64).eps))
    else:
        amp_guess = float(max(img.max() - img.min(), np.finfo(np.float64).eps))

    # Reasonable sigma guess
    sx_guess = float(max(initial_sigma, 0.5))
    sy_guess = float(max(initial_sigma, 0.5))

    if fixed_offset is not None:
        # 5-parameter model with fixed offset
        def gaussian_2d_fixed_offset(xy, amplitude, x0, y0, sigma_x, sigma_y):
            x, y = xy
            return (amplitude * np.exp(-(((x - x0) ** 2) / (2.0 * sigma_x ** 2) +
                                         ((y - y0) ** 2) / (2.0 * sigma_y ** 2))) + float(fixed_offset)).ravel()

        initial_guess = (amp_guess, x0_guess, y0_guess, sx_guess, sy_guess)

        # Bounds
        amp_upper = max(img.max() - float(fixed_offset), np.finfo(np.float64).eps)
        bounds = (
            (0.0,       0.0,    0.0,   0.5, 0.5),            # lower
            (amp_upper, w - 1,  h - 1, 2.0, 2.0),            # upper
        )

        try:
            popt, pcov = curve_fit(
                gaussian_2d_fixed_offset, (x, y), img.ravel(),
                p0=initial_guess, bounds=bounds, maxfev=10000
            )
            params = {
                "amplitude": float(popt[0]),
                "x0": float(popt[1]),
                "y0": float(popt[2]),
                "sigma_x": float(popt[3]),
                "sigma_y": float(popt[4]),
                "offset": float(fixed_offset),
            }
            return params, pcov
        except (RuntimeError, ValueError):
            print("Gaussian fitting failed (fixed offset).")
            return None, None

    else:
        # 6-parameter model with variable offset
        offset_guess = float(img.min())

        initial_guess = (amp_guess, x0_guess, y0_guess, sx_guess, sy_guess, offset_guess)

        # Allow negative offsets; keep sigma and location bounded
        bounds = (
            (0.0,       0.0,    0.0,   0.5, 0.5, -np.inf),    # lower
            (img.max(), w - 1,  h - 1, 5.0, 5.0,  np.inf),    # upper
        )

        try:
            popt, pcov = curve_fit(
                gaussian_2d, (x, y), img.ravel(),
                p0=initial_guess, bounds=bounds, maxfev=10000
            )
            params = {
                "amplitude": float(popt[0]),
                "x0": float(popt[1]),
                "y0": float(popt[2]),
                "sigma_x": float(popt[3]),
                "sigma_y": float(popt[4]),
                "offset": float(popt[5]),
            }
            return params, pcov
        except (RuntimeError, ValueError):
            print("Gaussian fitting failed (variable offset).")
            return None, None


class MS2GeneExpressionProcessor:
    """
    Processes MS2 gene expression data for individual cells across timepoints.

    This class analyzes gene expression by fitting 2D Gaussians to MS2 signal
    within cell boundaries and tracks expression over time.
    """

    def __init__(self, tracklets, image_data, masks_paths, ms2_z_projections, output_dir='output', plot=True, ransac_mad_k_th=2):
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

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize processing state variables
        self._reset_processing_state()

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

    def process_cell(self, cell_id):
        """
        Process gene expression analysis for a specific cell.

        Args:
            cell_id (int): ID of the cell to analyze
        """
        self.cell_id = cell_id
        print(f"Processing cell ID: {self.cell_id}")

        # Reset state for new cell
        self._reset_processing_state()

        # Get cell tracking data
        self.cell_labels_by_timepoint = self.tracklets[str(self.cell_id)]
        valid_timepoints = self._get_valid_timepoints()

        self.previous_center = None

        if not valid_timepoints:
            print(f"No valid timepoints found for cell {self.cell_id}")
            return

        self._pre_process(valid_timepoints)

        # Process each timepoint
        for timepoint in tqdm(valid_timepoints, desc=f"Processing cell {self.cell_id}"):
            self._process_single_timepoint(timepoint)
        if self.plot:
            # Generate outputs
            self._create_expression_visualization(
                self.cell_id, valid_timepoints)
            self._create_expression_plot(self.cell_id)
            # self._create_segmentation_video(self.cell_id, valid_timepoints)
            # self._create_cell_center_of_mass_plot(self.cell_id, valid_timepoints)
        self.guessed_gaussian_df['gaussian_centers'] = self.fit_gaussian_centers_list
        gaussian_fit_params_df = pd.DataFrame(self.gaussian_fit_params)
        self.guessed_gaussian_df = pd.concat([self.guessed_gaussian_df, gaussian_fit_params_df], axis=1)
        self.guessed_gaussian_df.to_csv(os.path.join(
            self.output_dir, f"cell_{self.cell_id}_guessed_gaussian.csv"), index=False)

    def _get_valid_timepoints(self):
        """Get timepoints where the cell is present (label != -1)."""
        return [t for t in range(len(self.cell_labels_by_timepoint))
                if self.cell_labels_by_timepoint[t] != -1]

    def _pre_process(self, valid_timepoints):
        self._calculate_max_cell_intensity(valid_timepoints)
        # TODO: when there is nearly nos signal, no definitive center
        self._find_cell_gaussian_inliers(valid_timepoints)

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

    def _find_cell_gaussian_inliers(self, valid_timepoints):
        self.guessed_gaussian_df = pd.DataFrame(
            columns=['time', 'peak_x', 'peak_y', 'peak_value', 'cell_center_x', 'cell_center_y'])
        peak_coordinates_list = []
        for timepoint in tqdm(valid_timepoints, desc="Finding Gaussian inliers"):
            # Load masks and get cell-specific data
            z_stack, ms2_stack, masks, ms2_projection = self._load_data_at_timepoint(
                timepoint)
            cell_label = self.cell_labels_by_timepoint[timepoint]
            # Create cell mask and get bounding box
            cell_mask_3d = (masks == cell_label).astype(np.uint8)
            current_cell_mask_projection = np.sum(cell_mask_3d, axis=0)
            current_cell_mask_projection = (
                current_cell_mask_projection > 0).astype(np.uint8)
            # Get bounding box
            z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)
            current_cell_bbox_ms2 = ms2_projection[y1:y2, x1:x2]

            # Create expanded mask for peak detection
            expanded_mask = ndimage.binary_dilation(
                current_cell_mask_projection, iterations=0)

            # Extract MCP signal within expanded cell boundary
            mask_region = current_cell_mask_projection[y1:y2, x1:x2]
            masked_ms2_region = (current_cell_bbox_ms2 *
                                 mask_region.astype(ms2_projection.dtype))

            # Find peak position for initial Gaussian center guess
            peak_coordinates = peak_local_max(masked_ms2_region, num_peaks=1)
            peak_x, peak_y = peak_coordinates[0][1], peak_coordinates[0][0]
            # peak_coordinates_list.append((peak_x +(x1 - 1), peak_y +(y1 - 1)))
            peak_coordinates_list.append(
                (peak_x/masked_ms2_region.shape[1], peak_y/masked_ms2_region.shape[0]))
            data = {
                'time': timepoint,
                'peak_x': peak_x,
                'peak_y': peak_y,
                'peak_value': masked_ms2_region[peak_y, peak_x],
                'cell_center_x': x1 + masked_ms2_region.shape[1] / 2,
                'cell_center_y': y1 + masked_ms2_region.shape[0] / 2
            }
            new_row = pd.DataFrame([data])
            self.guessed_gaussian_df = pd.concat(
                [self.guessed_gaussian_df, new_row], ignore_index=True)
        self.guessed_gaussian_df['diff_peak_x'] = self.guessed_gaussian_df['peak_x'].diff(
        )
        self.guessed_gaussian_df['diff_peak_y'] = self.guessed_gaussian_df['peak_y'].diff(
        )
        self.guessed_gaussian_df['diff_x_center_cell'] = self.guessed_gaussian_df['cell_center_x'].diff(
        )
        self.guessed_gaussian_df['diff_y_center_cell'] = self.guessed_gaussian_df['cell_center_y'].diff(
        )
        peaks_array = np.array(peak_coordinates_list)
        self.inliers_ransac, mask_ransac, info_r = filter_ransac_poly(
            peaks_array, degree=2, residual_threshold=2.0, mad_k=self.ransac_mad_k_th)
        self.guessed_gaussian_df['is_inlier'] = pd.Series(
            mask_ransac, index=self.guessed_gaussian_df.index).astype(bool)
        self.inlier_center = np.mean(self.inliers_ransac, axis=0)
        if self.plot:
            plot_gaussian_initial_guess(
                peaks_array, self.inliers_ransac, output_path=f"{self.output_dir}/gaussian_initial_guess_{self.cell_id}_mad_k_{self.ransac_mad_k_th}.png")

    def _process_single_timepoint(self, timepoint):
        """
        Process MS2 gene expression for a single timepoint.

        Args:
            timepoint (int): Timepoint index to process
        """
        # Load image data
        z_stack, ms2_stack, masks, ms2_projection = self._load_data_at_timepoint(
            timepoint)

        # Get cell-specific data
        cell_label = self.cell_labels_by_timepoint[timepoint]
        cell_mask_3d = (masks == cell_label).astype(np.uint8)
        self.current_cell_mask_projection = np.sum(cell_mask_3d, axis=0)
        self.current_cell_mask_projection = (
            self.current_cell_mask_projection > 0).astype(np.uint8)
        # Fit Gaussian to MS2 signal
        gaussian_params, covariance_matrix = self._fit_gaussian_to_ms2_signal(
            cell_mask_3d, ms2_projection, timepoint
        )
        cov_err = np.diag(covariance_matrix)
        self.gaussian_fit_params.append(gaussian_params)
        if timepoint in [1,16,52]:
            print('debug')
        if gaussian_params is not None:
            vals, ellipse_sum = self.sum_pixels_in_sigma_ellipse(gaussian_params,k=3, subtract_offset=False, clip_negative=True)
            intensity = gaussian_params['amplitude']*2*np.pi*gaussian_params['sigma_x']*gaussian_params['sigma_y']
        else:
            ellipse_sum = 0
            intensity = 0
        self.expression_amplitudes.append(intensity)
        self.expression_amplitudes2.append(ellipse_sum)
        if self.plot:
            # Create visualization
            self._create_timepoint_visualization(
                ms2_projection, cell_mask_3d, gaussian_params, covariance_matrix, timepoint
            )

            # Create 3D segmentation overlay
            segmentation_figure = show_3d_segmentation_overlay_with_unique_colors(
                z_stack, masks, cell_label,
                return_fig=True, zoom_on_highlight=True
            )
            self.segmentation_figures.append(segmentation_figure)
        cell_center = calculate_center_of_mass_3d(cell_mask_3d)

        if cell_center is not None:
            self.cell_center_of_mass.append(cell_center)

    def _load_data_at_timepoint(self, timepoint):
        z_stack = self.image_data[0, timepoint, 1, :, :, :, 0]
        ms2_stack = self.image_data[0, timepoint, 0, :, :, :, 0]
        masks = np.load(self.mask_file_paths[timepoint])['masks']
        ms2_projection = self.ms2_z_projections[timepoint]
        return z_stack, ms2_stack, masks, ms2_projection

    def _closest_inlier_indices(self, idx: int):
        """
        Return the indices of the closest inliers backward and forward from idx.

        Returns:
            tuple[int|None, int|None]:
                (backward_idx, forward_idx) where
                - backward_idx is the nearest inlier index < idx (None if none),
                - forward_idx is the nearest inlier index > idx (None if none).
        """
        s = self.guessed_gaussian_df['is_inlier'].to_numpy(dtype=bool)
        n = len(s)
        if n == 0 or idx is None or idx < 0 or idx >= n:
            return None, None

        # backward
        back_rel = np.flatnonzero(s[:idx]) if idx > 0 else np.array([])
        backward_idx = int(back_rel[-1]) if back_rel.size else None

        # forward
        fwd_rel = np.flatnonzero(s[idx + 1:]) if idx < n - 1 else np.array([])
        forward_idx = int(idx + 1 + fwd_rel[0]) if fwd_rel.size else None

        return backward_idx, forward_idx

    def _fit_gaussian_to_ms2_signal(self, cell_mask_3d, ms2_projection, timepoint):
        """
        Fit 2D Gaussian to MS2 signal within cell boundaries.

        Args:
            cell_mask_3d (np.ndarray): 3D cell mask
            ms2_projection (np.ndarray): 2D MS2 z-projection

        Returns:
            tuple: (gaussian_parameters, covariance_matrix)
        """

        # Get bounding box
        z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)
        self.current_cell_bbox_ms2 = ms2_projection[y1:y2, x1:x2]
        row_t = self.guessed_gaussian_df.iloc[timepoint]
        #debug
        # condition for dealing with inliers and outlier peaks from RANSAC step
        if row_t['is_inlier'] or timepoint == len(self.guessed_gaussian_df) - 1:
            initial_center = (row_t['peak_x'], row_t['peak_y'])
        else:
            backward_idx, forward_idx = self._closest_inlier_indices(timepoint)
            if backward_idx is None:
                relevant_t = forward_idx
                rel_x = self.guessed_gaussian_df.iloc[relevant_t]['peak_x'] - self.guessed_gaussian_df.iloc[relevant_t]['diff_peak_x'] - \
                    self.guessed_gaussian_df.iloc[relevant_t]['diff_x_center_cell']
                rel_y = self.guessed_gaussian_df.iloc[relevant_t]['peak_y'] - self.guessed_gaussian_df.iloc[relevant_t]['diff_peak_y'] - \
                    self.guessed_gaussian_df.iloc[relevant_t]['diff_y_center_cell']
            elif forward_idx is None:
                relevant_t = backward_idx
                rel_x = self.guessed_gaussian_df.iloc[relevant_t]['peak_x'] + self.guessed_gaussian_df.iloc[relevant_t]['diff_peak_x'] + \
                    self.guessed_gaussian_df.iloc[relevant_t]['diff_x_center_cell']
                rel_y = self.guessed_gaussian_df.iloc[relevant_t]['peak_y'] + self.guessed_gaussian_df.iloc[relevant_t]['diff_peak_y'] + \
                    self.guessed_gaussian_df.iloc[relevant_t]['diff_y_center_cell']
            elif timepoint - backward_idx > forward_idx - timepoint:
                relevant_t = forward_idx
                rel_x = self.guessed_gaussian_df.iloc[relevant_t]['peak_x'] - self.guessed_gaussian_df.iloc[relevant_t]['diff_peak_x'] - \
                    self.guessed_gaussian_df.iloc[relevant_t]['diff_x_center_cell']
                rel_y = self.guessed_gaussian_df.iloc[relevant_t]['peak_y'] - self.guessed_gaussian_df.iloc[relevant_t]['diff_peak_y'] - \
                    self.guessed_gaussian_df.iloc[relevant_t]['diff_y_center_cell']
            elif timepoint - backward_idx < forward_idx - timepoint:
                relevant_t = backward_idx
                rel_x = self.guessed_gaussian_df.iloc[relevant_t]['peak_x'] + self.guessed_gaussian_df.iloc[relevant_t]['diff_peak_x'] + \
                    self.guessed_gaussian_df.iloc[relevant_t]['diff_x_center_cell']
                rel_y = self.guessed_gaussian_df.iloc[relevant_t]['peak_y'] + self.guessed_gaussian_df.iloc[relevant_t]['diff_peak_y'] + \
                    self.guessed_gaussian_df.iloc[relevant_t]['diff_y_center_cell']
            else:
                # default look at backward inlier
                relevant_t = backward_idx
                rel_x = self.guessed_gaussian_df.iloc[relevant_t]['peak_x'] + self.guessed_gaussian_df.iloc[relevant_t]['diff_peak_x'] + \
                    self.guessed_gaussian_df.iloc[relevant_t]['diff_x_center_cell']
                rel_y = self.guessed_gaussian_df.iloc[relevant_t]['peak_y'] + self.guessed_gaussian_df.iloc[relevant_t]['diff_peak_y'] + \
                    self.guessed_gaussian_df.iloc[relevant_t]['diff_y_center_cell']

            row_t_closest_inlier = self.guessed_gaussian_df.iloc[relevant_t]
            dist1 = np.sqrt((row_t['peak_x'] - rel_x) **
                            2 + (row_t['peak_y'] - rel_y)**2)
            dist = np.sqrt((row_t['peak_x'] - row_t_closest_inlier['peak_x'])
                           ** 2 + (row_t['peak_y'] - row_t_closest_inlier['peak_y'])**2)
            if dist1 < 5:
                initial_center = (row_t['peak_x'], row_t['peak_y'])
            else:
                initial_center = (
                    row_t_closest_inlier['peak_x'], row_t_closest_inlier['peak_y'])
        #TODO: add method to find Gaussian fit offset
        offset = estimate_background_offset_annulus(
            self.current_cell_bbox_ms2, initial_center
        )
        # Fit Gaussian
        gaussian_params, covariance_matrix = estimate_emitter_2d_gaussian_with_fixed_offset(
            self.current_cell_bbox_ms2, initial_center, fixed_offset=offset
        )

        self.fit_gaussian_centers_list.append(
            (gaussian_params['x0']+x1, gaussian_params['y0']+y1))

        return gaussian_params, covariance_matrix

    def _create_timepoint_visualization(self, ms2_projection, cell_mask_3d,
                                        gaussian_params, covariance_matrix, timepoint):
        """
        Create visualization for a single timepoint showing cell outline and Gaussian fit.

        Args:
            ms2_projection (np.ndarray): MS2 z-projection
            cell_mask_3d (np.ndarray): 3D cell mask
            gaussian_params (dict): Fitted Gaussian parameters
            covariance_matrix (np.ndarray): Parameter covariance matrix
        """
        # Get cell bounding box
        z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)

        # Create cell outline
        cell_outline = self._create_cell_outline()

        # Generate Gaussian visualization (limited radius for overlay)
        gaussian_visualization = self._create_gaussian_visualization(
            gaussian_params, covariance_matrix
        )

        # Create figure with two subplots: 2D overlay + 3D Gaussian
        fig = plt.figure(figsize=(12, 5))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

        # Left: MS2 image with outline, Gaussian overlay, and center marker
        ms2_region = ms2_projection[y1:y2, x1:x2]
        outline_rgba = np.zeros((*cell_outline.shape, 4))
        outline_rgba[cell_outline == 1] = [0, 1, 0, 1]  # Green outline
        outline_region = outline_rgba[y1:y2, x1:x2]

        ax_img.imshow(ms2_region, cmap='gray', vmin=0,
                      vmax=np.max(ms2_projection))
        ax_img.imshow(outline_region, alpha=0.25)

        # Mark Gaussian center if available
        if gaussian_params is not None:
            ax_img.plot(
                gaussian_params['x0'],
                gaussian_params['y0'],
                marker='o',
                markersize=4,
                markerfacecolor='blue',
                markeredgecolor='black',
                linewidth=0,
            )
        # Draw 1σ, 2σ, 3σ ellipses
        for k, color, lw in [(1, 'yellow', 1.0), (2, 'orange', 1.0), (3, 'red', 1.2)]:
            e = Ellipse(
                (gaussian_params['x0'], gaussian_params['y0']),
                width=2 * k * gaussian_params['sigma_x'],
                height=2 * k * gaussian_params['sigma_y'],
                angle=0.0,
                facecolor='none',
                edgecolor=color,
                linewidth=lw,
                alpha=0.9,
                zorder=5
            )
            ax_img.add_patch(e)
        peak = self.guessed_gaussian_df.iloc[timepoint,
                                             :]['peak_x'], self.guessed_gaussian_df.iloc[timepoint, :]['peak_y']
        ax_img.plot(peak[0], peak[1], marker='o', markersize=4,
                    markerfacecolor='red', markeredgecolor='black', linewidth=0)
        ax_img.set_title('MS2 gene expression')
        ax_img.axis('off')

        # Right: 3D Gaussian surface
        if gaussian_params is not None:
            # Footnote (LaTeX) with formula and fitted values
            latex_formula = r'$A \exp\!\left[-\left(\frac{(x-x_0)^2}{2\sigma_x^2} + \frac{(y-y_0)^2}{2\sigma_y^2}\right)\right] + b$'
            latex_values = (
                rf'$A={gaussian_params["amplitude"]:.2f},\ '
                rf'x_0={gaussian_params["x0"]:.2f},\ '
                rf'y_0={gaussian_params["y0"]:.2f},\ '
                rf'\sigma_x={gaussian_params["sigma_x"]:.2f},\ '
                rf'\sigma_y={gaussian_params["sigma_y"]:.2f},\ '
                rf'b={gaussian_params["offset"]:.2f}$'
            )
            # Make room at the bottom and place a footnote-like text
            plt.subplots_adjust(bottom=0.22)
            fig.text(
                0.01, 0.02,
                'Gaussian fit: ' + latex_formula + '\n' + latex_values,
                ha='left', va='bottom', fontsize=9
            )
            h, w = self.current_cell_bbox_ms2.shape
            X, Y = np.meshgrid(np.arange(w), np.arange(h))
            gaussian_full = plot_2d_gaussian_with_size(
                 gaussian_params['amplitude'],
                 gaussian_params['x0'],
                 gaussian_params['y0'],
                 gaussian_params['sigma_x'],
                 gaussian_params['sigma_y'],
                 gaussian_params['offset'],
                 w, h
             )
            surf = ax_3d.plot_surface(
                 X, Y, gaussian_full,
                 cmap='viridis',
                 edgecolor='none',
                 antialiased=True,
                 linewidth=0
             )
            # Lock intensity axis and color scale
            ax_3d.set_zlim(0, self.max_cell_intensity)
            surf.set_clim(0, self.max_cell_intensity)
            fig.colorbar(surf, ax=ax_3d, orientation='vertical')

            ax_3d.set_xlabel('x')
            ax_3d.set_ylabel('y')
            ax_3d.set_zlabel('intensity')
            ax_3d.view_init(elev=30, azim=230)
        else:
            ax_3d.text2D(0.1, 0.5, 'Gaussian fit failed',
                         transform=ax_3d.transAxes)
            ax_3d.set_axis_off()
        plt.tight_layout()
        self.visualization_figures.append(fig)

    def _create_gaussian_visualization(self, gaussian_params, covariance_matrix):
        """
        Create Gaussian visualization with limited radius.

        Args:
            gaussian_params (dict): Fitted Gaussian parameters
            covariance_matrix (np.ndarray): Parameter covariance matrix

        Returns:
            np.ndarray: Gaussian visualization array
        """
        if gaussian_params is not None:
            # Generate full Gaussian
            gaussian_full = plot_2d_gaussian_with_size(
                gaussian_params['amplitude'], gaussian_params['x0'], gaussian_params['y0'],
                gaussian_params['sigma_x'], gaussian_params['sigma_y'], gaussian_params['offset'],
                self.current_cell_bbox_ms2.shape[1], self.current_cell_bbox_ms2.shape[0]
            )

            # Limit Gaussian to reasonable radius (3 sigma)
            y_coords, x_coords = np.indices(self.current_cell_bbox_ms2.shape)
            distance_from_center = np.sqrt(
                (x_coords - gaussian_params['x0'])**2 +
                (y_coords - gaussian_params['y0'])**2
            )
            radius_mask = distance_from_center <= 3
            gaussian_limited = gaussian_full * radius_mask
            

        else:
            # Handle failed fitting
            gaussian_limited = np.zeros_like(self.current_cell_bbox_ms2)

        return gaussian_limited
    
    def sum_pixels_in_sigma_ellipse(self, gaussian_params, image_region=None, k=1.0, subtract_offset=False, clip_negative=True):
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
        r = np.sqrt(((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2)
        annulus_mask = (r >= 3.0) & (r <= 6.0)

        vals = img[ellipse_mask].astype(np.float64, copy=False)
        annulus_vals = img[annulus_mask].astype(np.float64, copy=False)
        
        if subtract_offset:
            base = float(gaussian_params.get('offset', 0.0))
            vals = vals - base
            annulus_vals = annulus_vals - base
            if clip_negative:
                vals = np.clip(vals, 0.0, None)
                annulus_vals = np.clip(annulus_vals, 0.0, None)
        #threshold = np.median(annulus_vals) if annulus_vals.size else 0.0
        threshold = np.percentile(vals, 99) if vals.size else 0.0

        intensity = sum(vals[vals>=threshold])

        return vals, intensity

    def _create_cell_outline(self):
        """Create 2D outline of the cell from the z-projected mask."""
        cell_binary = (self.current_cell_mask_projection > 0).astype(np.uint8)
        cell_outline = cell_binary - ndimage.binary_erosion(cell_binary)
        return cell_outline

    def _create_expression_visualization(self, cell_id, valid_timepoints):
        """Create animated visualization of gene expression over time."""
        output_path = os.path.join(
            self.output_dir, f"cell_{cell_id}_expression_animation_ransac_th{self.ransac_mad_k_th}.mp4")
        titles = [f'Timepoint {t+1}' for t in valid_timepoints]

        create_gif_from_figures(
            self.visualization_figures, output_path, fps=1, titles=titles
        )

    def _create_segmentation_video(self, cell_id, valid_timepoints):
        """Create animated visualization of 3D segmentation over time."""
        output_path = os.path.join(
            self.output_dir, f"cell_{cell_id}_segmentation.mp4")
        titles = [f"Timepoint {t}" for t in valid_timepoints]

        create_gif_from_figures(
            self.segmentation_figures, output_path, fps=1, titles=titles
        )

    def _create_expression_plot(self, cell_id):
        """
        Create and save gene expression plot over time.
        Plots odd and even timepoints separately (1-based indexing).
        """
        if not self.expression_amplitudes:
            print(f"No expression data to plot for cell {cell_id}")
            return

        t_axis = np.arange(len(self.expression_amplitudes)) + 1  # 1-based
        amps1 = np.asarray(self.expression_amplitudes, dtype=float)
        amps2 = np.asarray(self.expression_amplitudes2, dtype=float)

        even_mask = (t_axis % 2 == 0)
        odd_mask = ~even_mask

        fig, ax = plt.subplots(2, 1, figsize=(8, 5))

        # Subplot 2: Ellipse sum
        ax[0].plot(t_axis, amps2, color='tab:red')
        ax[0].axhline(amps2.mean(), color='black', linestyle='--',
                   label=f'Mean: {amps2.mean():.2f}')
        ax[0].set_title("Intensity = sum in ellipse")
        ax[0].set_xlabel("Time [30*sec]")
        ax[0].set_ylabel("Emitter Intensity [AU]")
        ax[0].grid(True, alpha=0.3)
        ax[0].legend(fontsize=8, loc='upper right')
        ax[1].plot(t_axis[odd_mask], amps2[odd_mask],
                marker='o', linestyle='-', color='tab:blue',
                label='Odd timepoints')
        ax[1].plot(t_axis[even_mask], amps2[even_mask],
                marker='s', linestyle='-', color='tab:orange',
                label='Even timepoints')
        ax[1].set_title("Intensity = sum in ellipse")
        ax[1].set_xlabel("Time [30*sec]")
        ax[1].set_ylabel("Emitter Intensity [AU]")
        ax[1].grid(True, alpha=0.3)
        ax[1].legend(fontsize=8, loc='upper right')

        plt.tight_layout()
        output_path = os.path.join(
            self.output_dir, f"cell_{cell_id}_expression_plot_ransac_th_{self.ransac_mad_k_th}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Expression plot saved to: {output_path}")

    def _create_cell_center_of_mass_plot(self, cell_id, valid_timepoints):
        positions = np.array(self.cell_center_of_mass)
        create_trajectory_gif(
            positions,
            os.path.join(self.output_dir,
                         f"cell_{cell_id}_center_of_mass.gif"),
            fps=1
        )

    # Keep the original method name for backward compatibility
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
        output_dir='/home/dafei/output/MS2/3d_cell_segmentation/gRNA2_12.03.25-st-13-II---/v2_gene_estimation/',
        ransac_mad_k_th=1.0
    )
    for id in range(0,50):
        processor.process_cell(id)
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
