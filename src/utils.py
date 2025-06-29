import czifile


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
