import numpy as np
import matplotlib.pyplot as plt
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

def show_3d_segmentation_overlay(z_stack, masks,save_path=None):
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
        ax[i].set_title(f'Frame {i+1}')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight')
        print(f"Saved overlay images to {save_path}")
    else:
        plt.show()
    plt.close(fig)