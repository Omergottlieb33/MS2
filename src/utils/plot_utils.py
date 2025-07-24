import colorsys
import numpy as np
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

def show_3d_segmentation_overlay(z_stack, masks,save_path=None, return_fig=False):
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

def show_3d_segmentation_overlay_with_unique_colors(
    z_stack: np.ndarray,
    masks: np.ndarray,
    highlight_label: int,
    highlight_color: List[int] = [255, 0, 0],
    background_color: List[int] = [0, 0, 0],
    color_scheme: str = 'hsv',
    save_path: Optional[str] = None,
    return_fig: bool = False,
    highlight_alpha: float = 0.8,
    other_alpha: float = 0.4
):
    """Visualize a 3D stack with segmentation masks using distinct colors and a highlighted label."""
    frames = [
        create_colored_overlay_with_dominant_highlight(
            enhance_cell_image_contrast(z_stack[z]), masks[z],
            highlight_label, highlight_color, background_color,
            color_scheme, highlight_alpha, other_alpha
        )
        for z in range(z_stack.shape[0])
    ]

    fig, ax = plt.subplots(1, len(frames), figsize=(4 * len(frames), 8))
    if len(frames) == 1:
        ax.imshow(frames[0])
        ax.axis('off')
        ax.set_title('Slice 1')
    else:
        for i, frame in enumerate(frames):
            ax[i].imshow(frame)
            ax[i].axis('off')
            ax[i].set_title(f'Slice {i + 1}')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved colored overlay images to {save_path}")

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
        overlay[highlight_mask] = blend(overlay[highlight_mask], colored_mask[highlight_mask], highlight_alpha)
    if np.any(other_mask):
        overlay[other_mask] = blend(overlay[other_mask], colored_mask[other_mask], other_alpha)

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
        highlight_color_scaled = [min(255, int(c * highlight_brightness)) for c in highlight_color]
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
        [min(255, int(c * 255)) for c in colorsys.hsv_to_rgb(i / n, 0.9, 0.9 * brightness)]
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
