import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.feature import peak_local_max
from scipy import ndimage
from findmaxima2d import find_maxima, find_local_maxima
import matplotlib.pyplot as plt

from src.utils.cell_utils import (
    get_3d_bounding_box_corners,
    estimate_emitter_2d_gaussian_with_fixed_offset,
    estimate_background_offset_annulus,
    filter_ransac_poly, calculate_center_of_mass_3d
)


def get_indices_in_mask(coordinates: np.ndarray, cell_mask: np.ndarray, remove_outline: bool = False, expand_pixels: int = 2) -> np.ndarray:
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

    if expand_pixels and expand_pixels > 0:
        processed_mask = binary_dilation(
            processed_mask, iterations=expand_pixels)

    # --- Coordinate Processing ---
    # Round coordinates to the nearest integer to use them as indices.
    # Using np.round is safer than just casting to int, as it handles floating point values correctly.
    int_coords = coordinates.astype(int)

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

    return final_indices.astype(int)


class PeakStrategy:
    """Interface for peak handling strategies."""
    name = "base"

    def pre_process_cell(self, processor, valid_timepoints):
        raise NotImplementedError

    def fit_timepoint(self, processor, timepoint, cell_mask_3d, ms2_projection):
        """
        Returns (gaussian_params, covariance_matrix, peak_xy)
        """
        raise NotImplementedError


class GlobalPeakStrategy(PeakStrategy):
    name = "global"

    def __init__(self, ms2_z_projections: np.ndarray, prominence=20):
        self.ms2_z_projections = ms2_z_projections
        self.prominence = prominence
        self.init()

    def init(self):
        self.df = pd.DataFrame(columns=['timepoint', 'x', 'y', 'intensity'])
        for timepoint in tqdm(range(self.ms2_z_projections.shape[0]), desc="Finding global peaks"):
            local_max = find_local_maxima(self.ms2_z_projections[timepoint])
            y, x, regs = find_maxima(
                self.ms2_z_projections[timepoint], local_max, self.prominence)
            frame_df = pd.DataFrame(
                {'timepoint': timepoint, 'x': x, 'y': y, 'intensity': self.ms2_z_projections[timepoint][y, x]})
            self.df = pd.concat([self.df, frame_df], ignore_index=True)

    def pre_process_cell(self, processor, valid_timepoints):
        # Include future gaussian param columns (they will be filled later)
        processor.cell_df = pd.DataFrame(columns=[
            'timepoint', 'x', 'y', 'intensity', 'dist_to_center', 'angle_to_center',
            'is_inlier', 'gauss_x0', 'gauss_y0', 'gauss_sigma_x',
            'gauss_sigma_y', 'gauss_theta', 'gauss_amplitude', 'gauss_offset'
        ])
        peak_coordinates_list = []
        peak_center_dist_angle = []
        for timepoint in valid_timepoints:
            # Load data
            _, _, masks, ms2_projection = processor._load_data_at_timepoint(
                timepoint)
            cell_label = processor.cell_labels_by_timepoint[timepoint]
            cell_mask_3d = (masks == cell_label).astype(np.uint8)
            center = calculate_center_of_mass_3d(cell_mask_3d)
            center = np.array((center[0], center[1]))
            z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)
            current_cell_mask_projection = (
                np.sum(cell_mask_3d, axis=0) > 0).astype(np.uint8)

            frame_df = self.df[self.df['timepoint'] == timepoint]
            if frame_df.empty:
                continue
            x = frame_df['x'].to_numpy()
            y = frame_df['y'].to_numpy()
            pts = np.vstack((x, y)).T

            # restrict to cell mask
            current_cell_mask_projection = current_cell_mask_projection.astype(
                bool)
            #TODO: add neighborhood mask consideration
            idx = get_indices_in_mask(pts, current_cell_mask_projection, False, expand_pixels=0)
            if idx.size == 0:
                continue
            relevant_pts = pts[idx].astype(int)
            intensities = ms2_projection[relevant_pts[:,
                                                      1], relevant_pts[:, 0]]
            best_local = idx[np.argmax(intensities)]
            peak_x, peak_y = pts[best_local]
            processor.cell_df.loc[len(processor.cell_df)] = {
                'timepoint': timepoint,
                'x': peak_x,
                'y': peak_y,
                'intensity': ms2_projection[peak_y, peak_x],
                'dist_to_center': np.linalg.norm(np.array([peak_x, peak_y]) - center),
                'angle_to_center': np.arctan2(peak_y - center[1], peak_x - center[0]),
                'is_inlier': None,
                'gauss_x0': None,
                'gauss_y0': None,
                'gauss_sigma_x': None,
                'gauss_sigma_y': None,
                'gauss_theta': None,
                'gauss_amplitude': None,
                'gauss_offset': None
            }
            # normalized (within cell bbox will be recomputed per frame)
            # Use simple dimension normalization (avoid bbox for global strategy)
            h, w = ms2_projection[y1:y2, x1:x2].shape
            peak_coordinates_list.append(((peak_x-x1) / w, (peak_y-y1) / h))
            peak_center_dist_angle.append((np.linalg.norm(np.array(
                [peak_x, peak_y]) - center), np.arctan2(peak_y - center[1], peak_x - center[0])))        
        #TODO: #3 debug angle distance clustering
        if len(peak_center_dist_angle):
            self._z_score_outlier_removal_thresholds(np.array(peak_center_dist_angle)[:, 1])
            peak_center_dist_angle_array = np.array(peak_center_dist_angle)

            degree = 3
            min_points_needed = degree + 1
            mask_ransac = None
            inliers_ransac = np.empty((0, 2))
            ransac_success = False

            if peak_center_dist_angle_array.shape[0] >= min_points_needed + 1:
                try:
                    inliers_ransac, mask_ransac, _ = filter_ransac_poly(
                        peak_center_dist_angle_array,
                        degree=degree,
                        residual_threshold=2.5,
                        mad_k=processor.ransac_mad_k_th
                    )
                    # treat empty or all-false as failure
                    if inliers_ransac.size > 0 and mask_ransac.sum() > 0:
                        ransac_success = True
                except ValueError:
                    pass  # fall through to NA assignment

            if not ransac_success:
                # set all to NA (nullable boolean)
                mask_ransac = np.array([pd.NA] * peak_center_dist_angle_array.shape[0], dtype=object)
                inliers_ransac = np.empty((0, 2))

            processor.cell_df['is_inlier'] = pd.Series(
                mask_ransac, index=processor.cell_df.index, dtype="boolean"
            )
            processor.cell_df['dist_to_center'] = peak_center_dist_angle_array[:, 0]
            processor.cell_df['angle_to_center'] = peak_center_dist_angle_array[:, 1]

            if processor.plot and ransac_success:
                from src.utils.plot_utils import plot_gaussian_initial_guess
                plot_gaussian_initial_guess(
                    peak_center_dist_angle_array,
                    inliers_ransac,
                    output_path=f"{processor.output_dir}/cell_{processor.cell_id}_gaussian_initial_guess_mad_k_{processor.ransac_mad_k_th}_center_dist_angle.png"
                )
        else:
            processor.cell_df['is_inlier'] = pd.Series(dtype="boolean")

        # keep same attribute for downstream compatibility
        processor.cell_initial_center = []

    def _z_score_outlier_removal_thresholds(self, angles):
        mean = np.mean(angles)
        std = np.std(angles)
        self.angle_outlier_threshold = [mean + 3 * std, mean - 3 * std]

    def fit_timepoint(self, processor, timepoint, cell_mask_3d, ms2_projection):
        z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)
        processor.current_cell_bbox_ms2 = ms2_projection[y1:y2, x1:x2]
        # locate peak for this frame
        row_match = processor.cell_df[processor.cell_df['timepoint'] == timepoint]
        #TODO: #2 handle no detection between frames
        if row_match.empty:
            processor.cell_initial_center.append((0, 0))
            return None, None, (0, 0)
        else:
            row = row_match.iloc[0]
            # Z score outlier removal condition
            if row['angle_to_center'] > self.angle_outlier_threshold[0] or row['angle_to_center'] < self.angle_outlier_threshold[1]:
                processor.cell_initial_center.append((0, 0))
                return None, None, (0, 0)
            else:
                initial_center = (row['x'] - x1, row['y'] - y1)
                processor.cell_initial_center.append(initial_center)

        offset = estimate_background_offset_annulus(
            processor.current_cell_bbox_ms2, initial_center
        )
        gaussian_params, covariance_matrix = estimate_emitter_2d_gaussian_with_fixed_offset(
            processor.current_cell_bbox_ms2, initial_center, initial_sigma=0.5, fixed_offset=offset
        )
        processor.fit_gaussian_centers_list.append(
            (int(gaussian_params['x0'] + x1), int(gaussian_params['y0'] + y1))
        )

        # --- NEW: store fitted gaussian parameters in cell_df ---
        idx = row_match.index[0]
        processor.cell_df.loc[idx, 'gauss_x0'] = gaussian_params.get(
            'x0', np.nan) + x1
        processor.cell_df.loc[idx, 'gauss_y0'] = gaussian_params.get(
            'y0', np.nan) + y1
        processor.cell_df.loc[idx, 'gauss_sigma_x'] = gaussian_params.get(
            'sigma_x', np.nan)
        processor.cell_df.loc[idx, 'gauss_sigma_y'] = gaussian_params.get(
            'sigma_y', np.nan)
        processor.cell_df.loc[idx, 'gauss_theta'] = gaussian_params.get(
            'theta', np.nan)
        processor.cell_df.loc[idx, 'gauss_amplitude'] = gaussian_params.get(
            'amplitude', np.nan)
        processor.cell_df.loc[idx, 'gauss_offset'] = gaussian_params.get(
            'offset', offset)

        return gaussian_params, covariance_matrix, initial_center

### Deprecated
class LocalPeakStrategy(PeakStrategy):
    name = "local"

    def pre_process_cell(self, processor, valid_timepoints):
        guessed = []
        records = []
        for timepoint in valid_timepoints:
            z_stack, ms2_stack, masks, ms2_projection = processor._load_data_at_timepoint(
                timepoint)
            cell_label = processor.cell_labels_by_timepoint[timepoint]
            cell_mask_3d = (masks == cell_label).astype(np.uint8)
            mask_proj = (np.sum(cell_mask_3d, axis=0) > 0).astype(np.uint8)
            z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)
            bbox = ms2_projection[y1:y2, x1:x2]

            # inside-mask restriction
            masked = bbox * mask_proj[y1:y2,
                                      x1:x2].astype(ms2_projection.dtype)
            peaks = peak_local_max(masked, num_peaks=1)
            if peaks.size == 0:
                continue
            peak_y, peak_x = peaks[0][0], peaks[0][1]
            records.append({
                'time': timepoint,
                'peak_x': peak_x,
                'peak_y': peak_y,
                'peak_value': masked[peak_y, peak_x],
                'cell_center_x': x1 + masked.shape[1] / 2,
                'cell_center_y': y1 + masked.shape[0] / 2
            })
            guessed.append(
                (peak_x / masked.shape[1], peak_y / masked.shape[0]))

        processor.guessed_gaussian_df = pd.DataFrame(records)
        if not processor.guessed_gaussian_df.empty:
            df = processor.guessed_gaussian_df
            df['diff_peak_x'] = df['peak_x'].diff()
            df['diff_peak_y'] = df['peak_y'].diff()
            df['diff_x_center_cell'] = df['cell_center_x'].diff()
            df['diff_y_center_cell'] = df['cell_center_y'].diff()
            peaks_array = np.array(guessed)
            inliers_ransac, mask_ransac, _ = filter_ransac_poly(
                peaks_array, degree=2, residual_threshold=2.0, mad_k=processor.ransac_mad_k_th
            )
            df['is_inlier'] = pd.Series(
                mask_ransac, index=df.index).astype(bool)
            processor.inlier_center = np.mean(inliers_ransac, axis=0)
            if processor.plot:
                from src.utils.plot_utils import plot_gaussian_initial_guess
                plot_gaussian_initial_guess(
                    peaks_array,
                    inliers_ransac,
                    output_path=f"{processor.output_dir}/gaussian_initial_guess_{processor.cell_id}_mad_k_{processor.ransac_mad_k_th}.png"
                )
        else:
            processor.guessed_gaussian_df = pd.DataFrame(
                columns=['time', 'peak_x', 'peak_y', 'peak_value',
                         'cell_center_x', 'cell_center_y',
                         'diff_peak_x', 'diff_peak_y',
                         'diff_x_center_cell', 'diff_y_center_cell',
                         'is_inlier']
            )

    def fit_timepoint(self, processor, timepoint, cell_mask_3d, ms2_projection):
        z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask_3d)
        processor.current_cell_bbox_ms2 = ms2_projection[y1:y2, x1:x2]

        df = processor.guessed_gaussian_df
        if df.empty or timepoint not in df['time'].values:
            return None, None, (0, 0)

        # row at index position (time may not be contiguous)
        row_idx = df.index[df['time'] == timepoint][0]
        row_t = df.loc[row_idx]

        if row_t['is_inlier'] or row_idx == len(df) - 1:
            initial_center = (row_t['peak_x'], row_t['peak_y'])
        else:
            backward_idx, forward_idx = self._closest_inlier_indices(
                df, row_idx)
            if backward_idx is None and forward_idx is None:
                initial_center = (row_t['peak_x'], row_t['peak_y'])
            else:
                initial_center = self._predict_from_inliers(
                    df, row_idx, backward_idx, forward_idx)

        offset = estimate_background_offset_annulus(
            processor.current_cell_bbox_ms2, initial_center
        )
        gaussian_params, covariance_matrix = estimate_emitter_2d_gaussian_with_fixed_offset(
            processor.current_cell_bbox_ms2, initial_center, fixed_offset=offset
        )
        processor.fit_gaussian_centers_list.append(
            (int(gaussian_params['x0'] + x1), int(gaussian_params['y0'] + y1))
        )
        return gaussian_params, covariance_matrix, initial_center

    def _closest_inlier_indices(self, df, idx: int):
        s = df['is_inlier'].to_numpy(dtype=bool)
        n = len(s)
        if idx < 0 or idx >= n:
            return None, None
        back_rel = np.flatnonzero(s[:idx]) if idx > 0 else np.array([])
        fwd_rel = np.flatnonzero(s[idx + 1:]) if idx < n - 1 else np.array([])
        back = int(back_rel[-1]) if back_rel.size else None
        fwd = int(idx + 1 + fwd_rel[0]) if fwd_rel.size else None
        return back, fwd

    def _predict_from_inliers(self, df, idx, back, fwd):
        row_t = df.iloc[idx]

        def extrapolate(reference_idx, sign=1):
            r = df.iloc[reference_idx]
            rel_x = r['peak_x'] + sign * r['diff_peak_x'] + \
                sign * r['diff_x_center_cell']
            rel_y = r['peak_y'] + sign * r['diff_peak_y'] + \
                sign * r['diff_y_center_cell']
            return rel_x, rel_y

        if back is None:
            rel_x, rel_y = extrapolate(fwd, sign=-1)
        elif fwd is None:
            rel_x, rel_y = extrapolate(back, sign=+1)
        elif idx - back > fwd - idx:
            rel_x, rel_y = extrapolate(fwd, sign=-1)
        elif idx - back < fwd - idx:
            rel_x, rel_y = extrapolate(back, sign=+1)
        else:
            rel_x, rel_y = extrapolate(back, sign=+1)

        # distance checks
        closest_inlier = df.iloc[back if back is not None else fwd]
        dist_model = np.hypot(row_t['peak_x'] - rel_x, row_t['peak_y'] - rel_y)
        dist_inlier = np.hypot(row_t['peak_x'] - closest_inlier['peak_x'],
                               row_t['peak_y'] - closest_inlier['peak_y'])
        if dist_model < 5:
            return (row_t['peak_x'], row_t['peak_y'])
        return (closest_inlier['peak_x'], closest_inlier['peak_y'])
