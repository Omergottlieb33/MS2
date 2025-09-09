import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from findmaxima2d import find_maxima, find_local_maxima

from src.utils.cell_utils import (
    get_3d_bounding_box_corners,
    estimate_emitter_2d_gaussian_with_fixed_offset,
    estimate_background_offset_annulus,
    filter_ransac_poly, calculate_center_of_mass_3d
)
from src.utils.emitter_utils import circular_z_score, get_indices_in_mask


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

    def pre_process_cell(self, processor, valid_timepoints):
        # Include future gaussian param columns (they will be filled later)
        processor.cell_df = pd.DataFrame(columns=[
            'timepoint', 'x', 'y', 'intensity','z_slice_score', 'dist_to_center', 'angle_to_center',
            'is_inlier', 'circular_z_score', 'gauss_x0', 'gauss_y0', 'gauss_sigma_x',
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
            mask = (self.df_peaks["timepoint"].isin([timepoint])) & (
                self.df_peaks["cell_label"] == cell_label)
            df_cell = self.df_peaks[mask]
            if df_cell.empty:
                continue
            relevant_pts1 = df_cell[['x_peak', 'y_peak']].to_numpy()
            intensities = ms2_projection[relevant_pts1[:,
                                                       1], relevant_pts1[:, 0]]
            peak_x, peak_y = relevant_pts1[np.argmax(intensities)]
            #TODO: Two peaks inside cell -> check Z score. if both Z score above TH check location of previous frame
            if df_cell['slice_score'].to_numpy()[np.argmax(intensities)] < 10:
                continue
            else:
                slice_score = df_cell['slice_score'].to_numpy()[np.argmax(intensities)]
            processor.cell_df.loc[len(processor.cell_df)] = {
                'timepoint': timepoint,
                'x': peak_x,
                'y': peak_y,
                'intensity': ms2_projection[peak_y, peak_x],
                'z_slice_score': slice_score,
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
        # TODO: add distance consideration to Z score
        if len(peak_center_dist_angle):
            self._z_score_outlier_removal_thresholds(
                np.array(peak_center_dist_angle)[:, 1])
            processor.cell_df['circular_z_score'] = circular_z_score(
                np.array(peak_center_dist_angle)[:, 1])[0]
            #TODO: optimise clustering, DBSCAN with circular coordinates
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
                mask_ransac = np.array(
                    [pd.NA] * peak_center_dist_angle_array.shape[0], dtype=object)
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
        if row_match.empty:
            processor.cell_initial_center.append((0, 0))
            return None, None, (0, 0)
        else:
            row = row_match.iloc[0]
            # TODO: optimise outlier removal, circular Z score with distance
            if row['circular_z_score'] > 3:
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

    def emitter_cell_matching(self, processor, csv_path=None):
        """
        Faster peak-to-cell matching:
        - Accumulate dict records instead of repeated DataFrame concat.
        - Skip frames with no peaks early.
        - Single groupby at end to retain highest slice_score per (timepoint, cell_label).
        """
        if csv_path is not None:
            self.df_peaks = pd.read_csv(csv_path)
            return
        records = []
        ms2_bg = processor.ms2_background_removed  # (T,Z,Y,X)
        for timepoint in tqdm(range(len(processor.ms2_background_removed)), desc="Matching peaks to cells"):
            z_projection = processor.ms2_z_projections[timepoint]
            local_max = find_local_maxima(z_projection)
            if local_max.size == 0:
                continue
            y, x, regs = find_maxima(z_projection, local_max, self.prominence)
            if x.size == 0:
                continue
            pts = np.vstack((x, y)).T  # (k,2)

            masks = np.load(processor.mask_file_paths[timepoint])['masks']
            df_t = self._match_cells_to_emitter(masks, pts, ms2_bg, timepoint)
            if not df_t.empty:
                records.extend(df_t.to_dict('records'))

        if not records:
            self.df_peaks = pd.DataFrame(
                columns=['timepoint', 'cell_label',
                         'x_peak', 'y_peak', 'slice_score']
            )
            return

        df_all = pd.DataFrame.from_records(records)
        # filter emitter peaks in overlaping cells
        df_filtered = df_all.loc[df_all.groupby(["timepoint", "x_peak", "y_peak"])[
            "slice_score"].idxmax()]
        self.df_peaks = df_filtered

    @staticmethod
    def _match_cells_to_emitter(masks, pts, tzyx, timepoint):
        """
        Vectorized matching of candidate peaks (pts) to 3D cell masks.
        For each cell label:
          - Select peaks inside (or slightly expanded) 2D projection.
          - Score each peak by sum over z of intensity confined to that cell's z mask.
        Returns a DataFrame with one row per (cell_label, peak) before later filtering.
        """
        if pts.size == 0:
            return pd.DataFrame(columns=['timepoint', 'cell_label', 'x_peak', 'y_peak', 'slice_score'])

        # Prepare
        intensity_stack = tzyx[timepoint]        # (Z,Y,X)
        labels = np.unique(masks)
        labels = labels[(labels != 0)]           # exclude background
        if labels.size == 0:
            return pd.DataFrame(columns=['timepoint', 'cell_label', 'x_peak', 'y_peak', 'slice_score'])

        # Buffers
        time_buf = []
        label_buf = []
        x_buf = []
        y_buf = []
        score_buf = []

        # Iterate labels (cannot trivially vectorize across variable-sized regions)
        for label in labels:
            mask_label = (masks == label)              # (Z,Y,X) bool
            # (Y,X) bool 2D projection
            proj = mask_label.any(axis=0)
            # Get indices of peaks within the mask and expanded mask
            idx = get_indices_in_mask(pts, proj, False, expand_pixels=2)
            if idx.size == 0:
                continue

            sel_pts = pts[idx]                         # (k,2)
            x_sel = sel_pts[:, 0]
            y_sel = sel_pts[:, 1]

            # Extract per-peak z profiles and mask support: (Z,k)
            z_profiles = intensity_stack[:, y_sel, x_sel]
            # Expand the 3d mask for peaks found in the expanded projection
            expanded_mask_label = binary_dilation(mask_label, iterations=2)
            z_mask = expanded_mask_label[:, y_sel, x_sel]
            # Weighted sum over z -> (k,)
            slice_score = (z_profiles * z_mask).sum(axis=0)

            k = slice_score.shape[0]
            time_buf.append(np.full(k, timepoint, dtype=np.int32))
            label_buf.append(np.full(k, label, dtype=np.int32))
            x_buf.append(x_sel.astype(np.int32))
            y_buf.append(y_sel.astype(np.int32))
            score_buf.append(slice_score.astype(np.float32))

        if not time_buf:
            return pd.DataFrame(columns=['timepoint', 'cell_label', 'x_peak', 'y_peak', 'slice_score'])

        # Concatenate buffers
        return pd.DataFrame({
            'timepoint': np.concatenate(time_buf),
            'cell_label': np.concatenate(label_buf),
            'x_peak': np.concatenate(x_buf),
            'y_peak': np.concatenate(y_buf),
            'slice_score': np.concatenate(score_buf)
        })
