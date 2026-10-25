import numpy as np
import czifile
from cellpose.plot import image_to_rgb

def load_czi_images(file_path):
    try:
        with czifile.CziFile(file_path) as czi:
            image_data = czi.asarray()
            print(f"Successfully loaded {file_path}")
            print(f"data shape: {image_data.shape}")
            return image_data

    except Exception as e:
        print(f"Error loading CZI file: {e}")
        return None


def enhance_cell_image_contrast(image):
    if image.shape[0] < 4:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] < 3 or image.ndim < 3:
        image = image_to_rgb(image, channels=[0, 0])
    else:
        if image.max() <= 50.0:
            image = np.uint8(np.clip(image, 0, 1) * 255)
    return image
