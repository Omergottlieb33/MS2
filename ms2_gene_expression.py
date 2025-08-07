import os
import json
import argparse
import tifffile
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('module://matplotlib_inline.backend_inline')


from cell_tracking import get_masks_paths
from src.utils.image_utils import load_czi_images
from src.utils.gif_utils import create_gif_from_figures, create_extending_plot_gif
from src.utils.cell_utils import get_3d_bounding_box_corners
from src.utils.plot_utils import show_3d_segmentation_overlay_with_unique_colors, plot_2d_gaussian_with_size


def gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, offset):
    """
    2D Gaussian function.

    Args:
        xy: Tuple of (x, y) meshgrid coordinates.
        amplitude: Amplitude of the Gaussian.
        x0: X-coordinate of the center.
        y0: Y-coordinate of the center.
        sigma_x: Standard deviation in the x-direction.
        sigma_y: Standard deviation in the y-direction.
        offset: Constant offset.

    Returns:
        Flattened 2D Gaussian values.
    """
    x, y = xy
    return (amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                                 ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset).ravel()


def estimate_emitter_2d_gaussian(image, initial_position, initial_sigma=1.0):
    """
    Estimates the parameters of a 2D Gaussian emitter in an image.

    Args:
        image (2D array): Input image containing the emitter.
        initial_position (tuple): Initial guess for the (x, y) position of the emitter.
        initial_sigma (float): Initial guess for the Gaussian sigma (default: 1.0).

    Returns:
        dict: Estimated parameters of the Gaussian (amplitude, x0, y0, sigma_x, sigma_y, offset).
    """
    # Create a meshgrid for the image
    y, x = np.indices(image.shape)

    # Initial guesses for the parameters
    amplitude_guess = image.max() - image.min()
    x0_guess, y0_guess = initial_position
    offset_guess = image.min()
    initial_guess = (amplitude_guess, x0_guess, y0_guess,
                     initial_sigma, initial_sigma, offset_guess)

    # Define bounds for the parameters
    bounds = (
        (0, 0, 0, 0.5, 0.5, 0),  # Lower bounds
        (image.max(), image.shape[1], image.shape[0],
         3, 3, np.inf)  # Upper bounds
    )

    # Fit the 2D Gaussian model to the image
    try:
        popt, pcov = curve_fit(gaussian_2d, (x, y),
                               image.ravel(), p0=initial_guess, bounds=bounds)
        params = {
            "amplitude": popt[0],
            "x0": popt[1],
            "y0": popt[2],
            "sigma_x": popt[3],
            "sigma_y": popt[4],
            "offset": popt[5]
        }
        return params, pcov
    except RuntimeError:
        print("Gaussian fitting failed.")
        return None, None


class MS2GeneExpressionProcessor:
    """
    Processes MS2 gene expression data for individual cells across timepoints.
    
    This class analyzes gene expression by fitting 2D Gaussians to MS2 signal
    within cell boundaries and tracks expression over time.
    """
    
    def __init__(self, tracklets, image_data, masks_paths, ms2_z_projections, output_dir='debug'):
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
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize processing state variables
        self._reset_processing_state()
    
    def _reset_processing_state(self):
        """Reset variables used during processing."""
        self.expression_amplitudes = []
        self.visualization_figures = []
        self.segmentation_figures = []
        self.cell_labels_by_timepoint = []
        self.max_cell_intensity = 0
        self.current_cell_mask_projection = None
        self.current_cell_bbox_ms2 = None
    
    def process_cell(self, cell_id):
        """
        Process gene expression analysis for a specific cell.
        
        Args:
            cell_id (int): ID of the cell to analyze
        """
        print(f"Processing cell ID: {cell_id}")
        
        # Reset state for new cell
        self._reset_processing_state()
        
        # Get cell tracking data
        self.cell_labels_by_timepoint = self.tracklets[str(cell_id)]
        valid_timepoints = self._get_valid_timepoints()

        self.previous_center = None
        self.previous_center_global = None
        
        if not valid_timepoints:
            print(f"No valid timepoints found for cell {cell_id}")
            return
        
        # Calculate maximum intensity for consistent visualization
        self._calculate_max_cell_intensity(valid_timepoints)
        
        # Process each timepoint
        for timepoint in tqdm(valid_timepoints, desc=f"Processing cell {cell_id}"):
            self._process_single_timepoint(timepoint)
        
        # Generate outputs
        self._create_expression_visualization(cell_id, valid_timepoints)
        self._create_expression_plot(cell_id)
        self._create_segmentation_video(cell_id, valid_timepoints)
    
    def _get_valid_timepoints(self):
        """Get timepoints where the cell is present (label != -1)."""
        return [t for t in range(len(self.cell_labels_by_timepoint)) 
                if self.cell_labels_by_timepoint[t] != -1]
    
    def _calculate_max_cell_intensity(self, valid_timepoints):
        """
        Calculate maximum MS2 intensity across all timepoints for consistent visualization.
        
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
            
            # Project cell mask to 2D
            cell_mask_2d = np.sum(cell_mask_3d, axis=0)
            
            # Expand mask slightly and extract MS2 signal
            expanded_mask = ndimage.binary_dilation(cell_mask_2d, iterations=2)
            cell_region_ms2 = ms2_projection[y1-1:y2+1, x1-1:x2+1]
            mask_region = expanded_mask[y1-1:y2+1, x1-1:x2+1]
            
            masked_ms2 = cell_region_ms2 * mask_region.astype(ms2_projection.dtype)
            max_intensity = max(max_intensity, masked_ms2.max())
        
        self.max_cell_intensity = max_intensity
    
    def _process_single_timepoint(self, timepoint):
        """
        Process MS2 gene expression for a single timepoint.
        
        Args:
            timepoint (int): Timepoint index to process
        """
        # Load image data
        z_stack_brightfield = self.image_data[0, timepoint, 1, :, :, :, 0]
        ms2_stack = self.image_data[0, timepoint, 0, :, :, :, 0]
        masks = np.load(self.mask_file_paths[timepoint])['masks']
        ms2_projection = self.ms2_z_projections[timepoint]
        
        # Get cell-specific data
        cell_label = self.cell_labels_by_timepoint[timepoint]
        cell_mask_3d = (masks == cell_label).astype(np.uint8)
        self.current_cell_mask_projection = np.sum(cell_mask_3d, axis=0)
        
        # Fit Gaussian to MS2 signal
        gaussian_params, covariance_matrix = self._fit_gaussian_to_ms2_signal(
            cell_mask_3d, ms2_projection, timepoint
        )
        
        # Create visualization
        self._create_timepoint_visualization(
            ms2_projection, cell_mask_3d, gaussian_params, covariance_matrix
        )
        
        # Create 3D segmentation overlay
        segmentation_figure = show_3d_segmentation_overlay_with_unique_colors(
            z_stack_brightfield, masks, cell_label, 
            return_fig=True, zoom_on_highlight=True
        )
        self.segmentation_figures.append(segmentation_figure)
    
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
        self.current_cell_bbox_ms2 = ms2_projection[y1-1:y2+1, x1-1:x2+1]

        # Create expanded mask for peak detection
        expanded_mask = ndimage.binary_dilation(
            self.current_cell_mask_projection, iterations=0
        )
        
        # Extract MS2 signal within expanded cell boundary
        mask_region = expanded_mask[y1-1:y2+1, x1-1:x2+1]
        masked_ms2_region = (self.current_cell_bbox_ms2 * 
                           mask_region.astype(ms2_projection.dtype))
        
        # Find peak position for initial Gaussian center guess
        peak_coordinates = peak_local_max(masked_ms2_region, num_peaks=1)
        if len(peak_coordinates) > 0:
            peak_x, peak_y = peak_coordinates[0][1], peak_coordinates[0][0]
            initial_center = (peak_x, peak_y)
            # TODO: Handle peak jumping but dependes on first timepoint
            if timepoint == 0:
                global_peak_y = peak_y + (y1 - 1)
                global_peak_x = peak_x + (x1 - 1)
                self.previous_center = (peak_x, peak_y)
                self.previous_center_global = (global_peak_x, global_peak_y)
            else:
                global_peak_y = peak_y + (y1 - 1)
                global_peak_x = peak_x + (x1 - 1)
                distance = np.linalg.norm(np.array((global_peak_x, global_peak_y)) - np.array(self.previous_center_global))
                if distance > 10:  # If the peak is too far from the previous center
                    initial_center = self.previous_center
                else:
                    self.previous_center = (peak_x, peak_y)
                    self.previous_center_global = (global_peak_x, global_peak_y)
        else:
            # Fallback to center of bounding box
            initial_center = (self.current_cell_bbox_ms2.shape[1] // 2, 
                            self.current_cell_bbox_ms2.shape[0] // 2)
        
        # Fit Gaussian
        gaussian_params, covariance_matrix = estimate_emitter_2d_gaussian(
            self.current_cell_bbox_ms2, initial_center
        )
        if gaussian_params is not None:
            self.previous_center = (gaussian_params["x0"], gaussian_params["y0"])
        
        return gaussian_params, covariance_matrix
    
    def _create_timepoint_visualization(self, ms2_projection, cell_mask_3d, 
                                      gaussian_params, covariance_matrix):
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
        
        # Generate Gaussian visualization
        gaussian_visualization = self._create_gaussian_visualization(
            gaussian_params, covariance_matrix
        )
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        
        # Show MS2 image
        ms2_region = ms2_projection[y1-1:y2+1, x1-1:x2+1]
        ax.imshow(ms2_region, cmap='gray', vmin=0, vmax=np.max(ms2_projection))
        
        # Overlay cell outline in red
        outline_rgba = np.zeros((*cell_outline.shape, 4))
        outline_rgba[cell_outline == 1] = [1, 0, 0, 1]  # Red outline
        outline_region = outline_rgba[y1-1:y2+1, x1-1:x2+1]
        ax.imshow(outline_region, alpha=0.25)
        
        # Overlay Gaussian fit in hot colormap
        gaussian_img = ax.imshow(gaussian_visualization, cmap='hot',
                               interpolation='nearest', alpha=0.25, 
                               vmin=0, vmax=self.max_cell_intensity)
        
        # Add colorbar
        plt.colorbar(gaussian_img, ax=ax, fraction=0.046, pad=0.04, 
                    label='MS2 Intensity')
        
        ax.axis('off')
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
            
            # Store amplitude for expression tracking
            self.expression_amplitudes.append(gaussian_params['amplitude'])
        else:
            # Handle failed fitting
            gaussian_limited = np.zeros_like(self.current_cell_bbox_ms2)
            self.expression_amplitudes.append(0)
        
        return gaussian_limited
    
    def _create_cell_outline(self):
        """Create 2D outline of the cell from the z-projected mask."""
        cell_binary = (self.current_cell_mask_projection > 0).astype(np.uint8)
        cell_outline = cell_binary - ndimage.binary_erosion(cell_binary)
        return cell_outline
    
    def _create_expression_visualization(self, cell_id, valid_timepoints):
        """Create animated visualization of gene expression over time."""
        output_path = os.path.join(self.output_dir, f"cell_{cell_id}_expression_animation.mp4")
        titles = [f'Timepoint {t+1}' for t in valid_timepoints]
        
        create_gif_from_figures(
            self.visualization_figures, output_path, fps=1, titles=titles
        )
    
    def _create_segmentation_video(self, cell_id, valid_timepoints):
        """Create animated visualization of 3D segmentation over time."""
        output_path = os.path.join(self.output_dir, f"cell_{cell_id}_segmentation.mp4")
        titles = [f"Timepoint {t}" for t in valid_timepoints]
        
        create_gif_from_figures(
            self.segmentation_figures, output_path, fps=1, titles=titles
        )
    
    def _create_expression_plot(self, cell_id):
        """
        Create and save gene expression plot over time.
        
        Args:
            cell_id (int): Cell ID being processed
        """
        if not self.expression_amplitudes:
            print(f"No expression data to plot for cell {cell_id}")
            return
        
        mean_expression = np.mean(self.expression_amplitudes)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.expression_amplitudes, marker='o', linewidth=2, markersize=6)
        plt.axhline(mean_expression, color='black', linestyle='--',
                   label=f'Mean Expression: {mean_expression:.2f}')
        
        plt.legend()
        plt.title(f"Gene Expression Over Time for Cell ID {cell_id}")
        plt.xlabel("Timepoint")
        plt.ylabel("Gene Expression (MS2 Amplitude)")
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = os.path.join(self.output_dir, f"cell_{cell_id}_expression_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Expression plot saved to: {output_path}")

    # Keep the original method name for backward compatibility
    def process(self, cell_id):
        """Backward compatibility wrapper for process_cell."""
        self.process_cell(cell_id)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process MS2 gene expression data.")
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
        ms2_z_projections=ms2_z_projections
    )
    processor.process(1)  # Process the first cell as an example
    
    # cell_ids = range(0,20)
    # for cell_id in cell_ids:
    #     processor.process(cell_id)
    #     print(f"Finished processing cell {cell_id}")
    # print("All cells processed successfully.")
