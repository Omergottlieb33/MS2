import colorsys
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from typing import Optional, List, Union

from cellpose.plot import mask_overlay
from cellpose.utils import masks_to_outlines
from src.utils.image_utils import enhance_cell_image_contrast


def draw_cell_outline_on_image(mask, image):
    if np.sum(mask) == 0:
        return image
    outlines = masks_to_outlines(mask)
    outX, outY = np.nonzero(outlines)
    imgout = image.copy()
    imgout[outX, outY] = np.array([255, 0, 0])  # pure red
    return imgout


def show_3d_segmentation_overlay(z_stack, masks, save_path=None, return_fig=False):
    frames = []
    for i in range(z_stack.shape[0]):
        img = z_stack[i]
        img = enhance_cell_image_contrast(img)
        maski = masks[i]
        overlay = mask_overlay(img, maski)
        frames.append(overlay)

    fig, ax = plt.subplots(1, len(frames), figsize=(20, 10))
    for i, frame in enumerate(frames):
        ax[i].imshow(frame)
        ax[i].axis('off')
        ax[i].set_title(f'Slice {i+1}')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved overlay images to {save_path}")
    if return_fig:
        return fig
    else:
        plt.show()
        plt.close(fig)


def create_colored_overlay_with_dominant_highlight(
    image: np.ndarray,
    mask: np.ndarray,
    highlight_label: int,
    highlight_color: List[int],
    background_color: List[int],
    color_scheme: str,
    highlight_alpha: float = 0.8,
    other_alpha: float = 0.4
) -> np.ndarray:
    """Overlay RGB segmentation on image, highlighting a specific label with different alpha."""
    rgb_image = convert_to_rgb(image)
    colored_mask = color_cells_with_unique_colors(
        mask, highlight_label, highlight_color, background_color, color_scheme
    )

    overlay = rgb_image.astype(np.float32)
    highlight_mask = mask == highlight_label
    other_mask = (mask > 0) & (mask != highlight_label)

    if np.any(highlight_mask):
        overlay[highlight_mask] = blend(
            overlay[highlight_mask], colored_mask[highlight_mask], highlight_alpha)
    if np.any(other_mask):
        overlay[other_mask] = blend(
            overlay[other_mask], colored_mask[other_mask], other_alpha)

    return overlay.astype(np.uint8)


def color_cells_with_unique_colors(
    mask: np.ndarray,
    highlight_label: int,
    highlight_color: List[int] = [255, 0, 0],
    background_color: List[int] = [0, 0, 0],
    color_scheme: str = 'hsv',
    highlight_brightness: float = 1.0,
    other_brightness: float = 0.7
) -> np.ndarray:
    """Assign unique colors to labeled regions, emphasizing the highlight_label."""
    unique_labels = np.unique(mask[mask > 0])
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_mask[mask == 0] = background_color

    if highlight_label in unique_labels:
        highlight_color_scaled = [
            min(255, int(c * highlight_brightness)) for c in highlight_color]
        colored_mask[mask == highlight_label] = highlight_color_scaled
        unique_labels = unique_labels[unique_labels != highlight_label]

    colors = get_color_list(len(unique_labels), color_scheme, other_brightness)
    for label, color in zip(unique_labels, colors):
        colored_mask[mask == label] = color

    return colored_mask


# === Utility Functions ===

def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel RGB uint8."""
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    return image.astype(np.uint8)


def blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """Blend two arrays with given alpha."""
    return (1 - alpha) * base + alpha * overlay


def get_color_list(n: int, scheme: str, brightness: float) -> List[List[int]]:
    """Get a list of n colors based on scheme."""
    if scheme == 'hsv':
        return generate_hsv_colors(n, brightness)
    elif scheme == 'random':
        return generate_random_colors(n, brightness)
    elif scheme == 'gradient':
        return generate_gradient_colors(n, brightness)
    else:
        return generate_hsv_colors(n, brightness)


def generate_hsv_colors(n: int, brightness: float = 1.0) -> List[List[int]]:
    return [
        [min(255, int(c * 255))
         for c in colorsys.hsv_to_rgb(i / n, 0.9, 0.9 * brightness)]
        for i in range(n)
    ]


def generate_random_colors(n: int, brightness: float = 1.0, seed: int = 42) -> List[List[int]]:
    np.random.seed(seed)
    colors = []
    for _ in range(n):
        r, g, b = np.random.randint(50, 255, size=3)
        r, g, b = [min(255, int(c * brightness)) for c in (r, g, b)]
        while r + g + b < 150:
            r, g, b = [min(255, c + 20) for c in (r, g, b)]
        colors.append([r, g, b])
    return colors


def generate_gradient_colors(n: int, brightness: float = 1.0) -> List[List[int]]:
    colors = []
    for i in range(n):
        ratio = i / max(1, n - 1)
        if ratio < 0.25:
            r, g, b = 0, int(255 * ratio / 0.25), 255
        elif ratio < 0.5:
            r, g, b = 0, 255, int(255 * (1 - (ratio - 0.25) / 0.25))
        elif ratio < 0.75:
            r, g, b = int(255 * (ratio - 0.5) / 0.25), 255, 0
        else:
            r, g, b = 255, int(255 * (1 - (ratio - 0.75) / 0.25)), 0
        colors.append([min(255, int(c * brightness)) for c in (r, g, b)])
    return colors


def show_3d_segmentation_overlay_with_unique_colors(z_stack, masks, highlight_label,
                                                    highlight_color=[
                                                        255, 0, 0],
                                                    background_color=[0, 0, 0],
                                                    color_scheme='hsv',
                                                    save_path=None, return_fig=False,
                                                    highlight_alpha=0.8, other_alpha=0.4,
                                                    zoom_on_highlight=False, zoom_padding=50):
    """
    Show 3D segmentation overlay with highlighted label in specific color and all other labels 
    colored uniquely using the color_cells_with_unique_colors function.

    Args:
        z_stack: 3D numpy array of images (z, h, w)
        masks: 3D numpy array of segmentation masks (z, h, w)
        highlight_label: integer label to highlight with specific color
        highlight_color: RGB color for highlighted label [R, G, B] (default: red)
        background_color: RGB color for background [R, G, B] (default: black)
        color_scheme: 'hsv', 'random', or 'gradient' for other cell coloring
        save_path: path to save the figure (optional)
        return_fig: if True, return the figure object instead of showing
        highlight_alpha: transparency for highlighted cell (0-1, higher = more opaque)
        other_alpha: transparency for other cells (0-1, lower = more transparent)
        zoom_on_highlight: if True, zoom in on the area around the highlighted cell
        zoom_padding: padding around the highlighted cell when zooming (in pixels)

    Returns:
        fig object if return_fig=True, otherwise None
    """
    # Calculate zoom region if zoom_on_highlight is True
    crop_coords = None
    if zoom_on_highlight:
        crop_coords = get_highlight_crop_coords(
            masks, highlight_label, zoom_padding)
        if crop_coords is None:
            print(
                f"Warning: Highlight label {highlight_label} not found in masks. Showing full image.")
            zoom_on_highlight = False

    frames = []
    for i in range(z_stack.shape[0]):
        img = z_stack[i]
        img = enhance_cell_image_contrast(img)
        maski = masks[i]

        # Apply cropping if zoom is enabled
        if zoom_on_highlight and crop_coords is not None:
            y_min, y_max, x_min, x_max = crop_coords
            img = img[y_min:y_max, x_min:x_max]
            maski = maski[y_min:y_max, x_min:x_max]

        # Create colored overlay with different alpha values for highlight vs other cells
        colored_overlay = create_colored_overlay_with_dominant_highlight(
            img, maski, highlight_label, highlight_color, background_color,
            color_scheme, highlight_alpha, other_alpha)
        frames.append(colored_overlay)

    # Handle single slice case
    if len(frames) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(frames[0])
        ax.axis('off')
        title = 'Slice 1'
        if zoom_on_highlight:
            title += f' (Zoomed on label {highlight_label})'
        ax.set_title(title)
    else:
        fig, ax = plt.subplots(1, len(frames), figsize=(4*len(frames), 8))
        for i, frame in enumerate(frames):
            ax[i].imshow(frame)
            ax[i].axis('off')
            title = f'Slice {i+1}'
            if zoom_on_highlight:
                title += f' (Zoomed on label {highlight_label})'
            ax[i].set_title(title)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved colored overlay images to {save_path}")

    if return_fig:
        return fig
    else:
        plt.show()
        plt.close(fig)


def get_highlight_crop_coords(masks, highlight_label, padding):
    """
    Calculate crop coordinates to zoom in on the highlighted label across all z-slices.

    Args:
        masks: 3D numpy array of segmentation masks (z, h, w)
        highlight_label: integer label to find
        padding: padding around the bounding box in pixels

    Returns:
        tuple: (y_min, y_max, x_min, x_max) crop coordinates or None if label not found
    """
    # Find all pixels with the highlight label across all z-slices
    highlight_pixels = masks == highlight_label

    if not np.any(highlight_pixels):
        return None

    # Get coordinates of all highlight pixels
    z_coords, y_coords, x_coords = np.where(highlight_pixels)

    # Calculate bounding box
    y_min = max(0, np.min(y_coords) - padding)
    y_max = min(masks.shape[1], np.max(y_coords) + padding + 1)
    x_min = max(0, np.min(x_coords) - padding)
    x_max = min(masks.shape[2], np.max(x_coords) + padding + 1)

    return (y_min, y_max, x_min, x_max)


def plot_masked_pixels_3d(
    image_tensor: np.ndarray,
    mask_tensor: np.ndarray,
    title: str = "3D Voxel Plot of Masked Pixels",
    point_size: int = 5,
    alpha: float = 0.7,
    cmap: str = 'viridis',
    save_path: Optional[str] = None,
    return_fig: bool = False,
    vmin: float = 0.0,
    vmax: float = 100.0,
    threshold: Optional[float] = None,
    use_threshold_coloring: bool = False
):
    """
    Visualizes pixels from a 3D image tensor that fall within a 3D mask as a 3D scatter plot.

    The color of each point in the scatter plot represents the pixel intensity from the image tensor.
    With threshold coloring: black (below threshold), green (above threshold), red (cell boundary).

    Args:
        image_tensor (np.ndarray): The 3D image data, expected shape (z, y, x).
        mask_tensor (np.ndarray): The 3D boolean or integer mask, same shape as image_tensor.
        title (str): The title for the plot.
        point_size (int): The size of the points in the scatter plot.
        alpha (float): The transparency of the points.
        cmap (str): The colormap for mapping pixel intensity to color (used when use_threshold_coloring=False).
        save_path (Optional[str]): If provided, saves the figure to this path.
        return_fig (bool): If True, returns the matplotlib figure object instead of showing it.
        vmin (float): Minimum value for colorbar range.
        vmax (float): Maximum value for colorbar range.
        threshold (Optional[float]): Threshold value for three-color scheme.
        use_threshold_coloring (bool): If True, use threshold-based coloring scheme.
    """
    if image_tensor.shape != mask_tensor.shape:
        raise ValueError("Image and mask tensors must have the same shape.")
    if image_tensor.ndim != 3:
        raise ValueError("Input tensors must be 3D (z, y, x).")

    coords = np.argwhere(mask_tensor > 0)
    if coords.shape[0] == 0:
        print("Mask is empty, nothing to plot.")
        return

    z_coords, y_coords, x_coords = coords[:, 0], coords[:, 1], coords[:, 2]
    intensities = image_tensor[z_coords, y_coords, x_coords]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if use_threshold_coloring and threshold is not None:
        # Three-color scheme based on threshold and boundary detection
        colors = get_threshold_based_colors(
            coords, mask_tensor, intensities, threshold
        )

        scatter = ax.scatter(x_coords, y_coords, z_coords,
                             c=colors, s=point_size, alpha=alpha)

        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='black', label=f'Background'),
            Patch(facecolor='green', label=f'MS2')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    else:
        # Original intensity-based coloring
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=intensities, cmap=cmap,
                             s=point_size, alpha=alpha, vmin=vmin, vmax=vmax)

        # Add colorbar with range info
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.1,
                                label='Pixel Intensity')
        colorbar.ax.text(0.5, 1.02, f'Range: {vmin:.2f} - {vmax:.2f}',
                         transform=colorbar.ax.transAxes,
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_zlabel("Z Coordinate")
    ax.set_title(title)
    ax.invert_zaxis()  # Match image array z-axis direction

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved 3D plot to {save_path}")
    if return_fig:
        return fig
    else:
        plt.show()
        plt.close(fig)


def get_threshold_based_colors(coords, mask_tensor, intensities, threshold):
    """
    Assign colors based on threshold and boundary detection.

    Args:
        coords: Array of coordinates where mask > 0
        mask_tensor: 3D mask array
        intensities: Intensity values at the coordinates
        threshold: Threshold value for classification

    Returns:
        Array of colors for each point
    """
    from scipy.ndimage import binary_erosion

    # Get the specific mask label (assuming single cell)
    mask_label = mask_tensor[coords[0, 0], coords[0, 1], coords[0, 2]]
    binary_mask = (mask_tensor == mask_label).astype(bool)

    # Create eroded mask to find boundary pixels
    # Use a smaller structuring element for 3D erosion
    eroded_mask = binary_erosion(binary_mask, iterations=1)
    boundary_mask = binary_mask & ~eroded_mask

    colors = []
    for i, (z, y, x) in enumerate(coords):
        if intensities[i] >= threshold:
            # Green for above threshold
            colors.append([0.0, 1.0, 0.0])  # Green
        else:
            # Black for below threshold
            colors.append([0.0, 0.0, 0.0])  # Black

    return np.array(colors)


def plot_2d_gaussian_with_size(amplitude, x0, y0, sigma_x, sigma_y, offset, width, height):
    """
    Plots a 2D Gaussian as a heatmap with specified width and height.

    Args:
        amplitude (float): Amplitude of the Gaussian.
        x0 (float): X-coordinate of the center.
        y0 (float): Y-coordinate of the center.
        sigma_x (float): Standard deviation in the x-direction.
        sigma_y (float): Standard deviation in the y-direction.
        offset (float): Constant offset.
        width (int): Width of the grid.
        height (int): Height of the grid.
    """
    # Create a grid of x and y values
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)

    # Calculate the 2D Gaussian
    gaussian = amplitude * np.exp(-(((x - x0) ** 2) / (2 * sigma_x ** 2) +
                                    ((y - y0) ** 2) / (2 * sigma_y ** 2))) + offset
    return gaussian

def plot_gaussian_initial_guess(emitter_peaks_array, inliers, output_path, origin_upper_right=True):
    emitter_peaks_array = np.asarray(emitter_peaks_array)
    inliers = np.asarray(inliers)

    fig, ax = plt.subplots(figsize=(6, 5))

    if origin_upper_right:
        ax.scatter(emitter_peaks_array[:, 0], emitter_peaks_array[:, 1], c='lightgray', s=10, label='all')
        ax.scatter(inliers[:, 0], inliers[:, 1], c='r', s=12, label='RANSAC inliers')

        # Set limits from data, then invert both axes
        if emitter_peaks_array.size:
            xmin, xmax = emitter_peaks_array[:, 0].min(), emitter_peaks_array[:, 0].max()
            ymin, ymax = emitter_peaks_array[:, 1].min(), emitter_peaks_array[:, 1].max()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        ax.invert_yaxis()  # Y increases downward
        ax.set_xlabel("X ")
        ax.set_ylabel("Y")
    else:
        # Standard Cartesian view (X→right, Y→up), keep your old negation for visual parity if desired
        ax.scatter(emitter_peaks_array[:, 0], emitter_peaks_array[:, 1], c='lightgray', s=10, label='all')
        ax.scatter(inliers[:, 0], inliers[:, 1], c='r', s=12, label='RANSAC inliers')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)

