import numpy as np
from cellpose.utils import masks_to_outlines


def draw_cell_outline_on_image(mask, image):
    if np.sum(mask) == 0:
        return image
    outlines = masks_to_outlines(mask)
    outX, outY = np.nonzero(outlines)
    imgout = image.copy()
    imgout[outX, outY] = np.array([255, 0, 0])  # pure red
    return imgout