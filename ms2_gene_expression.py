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
    except RuntimeError:
        print("Gaussian fitting failed.")
        params = None, None

    return params, pcov


class MS2GeneExpressionProcessor:
    def __init__(self, tracklets, cell_id, image_data, masks_paths, ms2_z_projection):
        """
        Initializes the processor with the required data.

        Args:
            tracklets (dict): Dictionary containing tracklet data.
            cell_id (int): Cell ID to process.
            image_data (np.ndarray): Image data array.
            masks_paths (list): List of paths to segmentation masks.
            ms2_z_projection (np.ndarray): MS2 z projection data.
        """
        self.tracklets = tracklets
        self.cell_id = cell_id
        self.image_data = image_data
        self.masks_paths = masks_paths
        self.ms2_z_projections = ms2_z_projection
        self.cell_labels = tracklets[str(cell_id)]
        self.valid_timepoints = [t for t in range(
            len(self.cell_labels)) if self.cell_labels[t] != -1]
        self.gene_expression_list = []
        self.segmentation_figures = []
        self.figures = []

    def process(self):
        """
        Processes all valid timepoints for the given cell.
        """
        print(f"Processing cell ID: {self.cell_id}")
        for t in tqdm(self.valid_timepoints):
            self.process_timepoint(t)
        create_gif_from_figures(self.figures,
                                f"cell_{self.cell_id}_gene_expression.gif",
                                fps=1,
                                titles=[f'Time {i+1}' for i in self.valid_timepoints])
        self.plot_gene_expression()
        create_gif_from_figures(self.segmentation_figures,
                                f"cell_{self.cell_id}_segmentation.gif",
                                fps=1,
                                titles=[f"Time {t}" for t in self.valid_timepoints])

    def process_timepoint(self, t):
        """
        Processes a single timepoint for the given cell.

        Args:
            t (int): Timepoint index.
        """
        z_stack = self.image_data[0, t, 1, :, :, :, 0]
        ms2_stack = self.image_data[0, t, 0, :, :, :, 0]
        masks = np.load(self.masks_paths[t])['masks']
        ms2_z_projection = self.ms2_z_projections[t]

        cell_mask = (masks == self.cell_labels[t]).astype(np.uint8)
        self.cell_mask_z_projection = np.sum(cell_mask, axis=0)
        popt, pcov = self.get_gaussian_params(cell_mask, ms2_z_projection)
        self.plot_cell_outline_and_gaussian(
            ms2_z_projection, cell_mask, popt, pcov)
        segmentation_overlay = show_3d_segmentation_overlay_with_unique_colors(
            z_stack, masks, self.cell_labels[t], return_fig=True, zoom_on_highlight=True)
        self.segmentation_figures.append(segmentation_overlay)

    def get_gaussian_params(self, cell_mask, ms2_z_projection):
        z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask)
        self.cell_bbox_ms2_z_projection = ms2_z_projection[y1-1:y2+1, x1-1:x2+1]
        expanded_mask = ndimage.binary_dilation(
            self.cell_mask_z_projection, iterations=2)

        masked_cell_ms2_z_projection = ms2_z_projection[y1-1:y2+1, x1-1:x2+1] * \
            expanded_mask[y1-1:y2+1, x1-1:x2 +
                          1].astype(ms2_z_projection.dtype)
        peaks = peak_local_max(masked_cell_ms2_z_projection, num_peaks=1)
        x, y = peaks[0][1], peaks[0][0]
        initial_position = (x, y)
        popt, pcov = estimate_emitter_2d_gaussian(
            self.cell_bbox_ms2_z_projection, initial_position)
        return popt, pcov

    def plot_cell_outline_and_gaussian(self, ms2_z_projection, cell_mask, popt, pcov):
        z1, y1, x1, z2, y2, x2 = get_3d_bounding_box_corners(cell_mask)
        cell_outline = self.get_cell_z_projection_outline()
        gaussian = self.plot_gaussian_fit(popt, pcov)
        y, x = np.indices(self.cell_bbox_ms2_z_projection.shape)
        distance_from_center = np.sqrt(
            (x - popt['x0'])**2 + (y - popt['y0'])**2)
        gaussian_mask = distance_from_center <= 3
        gaussian_limited = gaussian * gaussian_mask

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.imshow(ms2_z_projection[y1-1:y2+1, x1-1:x2+1], cmap='gray', vmin=0, vmax=np.max(ms2_z_projection))
        outline_rgba = np.zeros((*cell_outline.shape, 4))
        outline_rgba[cell_outline == 1] = [1, 0, 0, 1]
        ax.imshow(outline_rgba[y1-1:y2+1, x1-1:x2+1], alpha=0.25)
        img = ax.imshow(gaussian_limited, cmap='hot',
                         interpolation='nearest', alpha=0.25)
        plt.colorbar(img, ax=ax, fraction=0.046,
                      pad=0.04, label='MS2 Intensity')
        ax.axis('off')
        plt.tight_layout()
        self.figures.append(fig)


    def plot_gaussian_fit(self, popt, pcov):
        if popt is not None:
            covariance = np.sqrt(np.diag(pcov))
            gaussian = plot_2d_gaussian_with_size(
                popt['amplitude'], popt['x0'], popt['y0'],
                popt['sigma_x'], popt['sigma_y'], popt['offset'],
                self.cell_bbox_ms2_z_projection.shape[1], self.cell_bbox_ms2_z_projection.shape[0]
            )
            self.gene_expression_list.append(popt['amplitude'])
        else:
            gaussian = np.zeros_like(self.cell_bbox_ms2_z_projection)
            self.gene_expression_list.append(0)
        return gaussian

    def get_cell_z_projection_outline(self):
        cell_bbox_corners = (self.cell_mask_z_projection > 0).astype(np.uint8)
        outline_cell = cell_bbox_corners - \
            ndimage.binary_erosion(cell_bbox_corners)
        return outline_cell

    def plot_gene_expression(self):
        """
        Plots the gene expression over time for the processed cell.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.gene_expression_list, marker='o')
        plt.title(f"Gene Expression Over Time for Cell ID {self.cell_id}")
        plt.xlabel("Timepoint")
        plt.ylabel("Gene Expression (MS2 Intensity)")
        plt.ylim(bottom=0)  # Ensure the y-axis starts from 0
        plt.grid()
        plt.savefig(f"cell_{self.cell_id}_gene_expression_plot.png")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process MS2 gene expression data.")
    parser.add_argument('--czi_file_path', type=str, required=True,
                        help='Path to CZI file.')
    parser.add_argument('--seg_maps_dir', type=str, required=True,
                        help='Directory containing segmentation maps.')
    parser.add_argument('--tracklets_path', type=str,
                        required=True, help='Path to tracklets JSON file.')
    parser.add_argument('--ms2_filtered_z_projection', type=str, required=True,
                        help='Path to MS2 z projection file after background removal.')
    parser.add_argument('--cell_id', type=str,
                        required=True, help='Cell ID to process.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    image_data = load_czi_images(args.czi_file_path)
    masks_paths = get_masks_paths(args.seg_maps_dir)
    with open(args.tracklets_path, 'r') as f:
        tracklets = json.load(f)
    ms2_z_projection = tifffile.imread(args.ms2_filtered_z_projection)

    processor = MS2GeneExpressionProcessor(
        tracklets, args.cell_id, image_data, masks_paths, ms2_z_projection)
    processor.process()
