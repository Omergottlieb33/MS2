import io
import pickle
from PIL import Image
import matplotlib.pyplot as plt

def create_gif_from_overlay_sequence(base_images, gif_path,
                                   duration=500, figsize=(8, 8), titles=None):
    """
    Create a GIF from a sequence of overlay images using matplotlib.
    
    Args:
        base_images (list): List of base images (numpy arrays)
        overlay_images (list): List of overlay images (numpy arrays)
        gif_path (str): Output path for the GIF file
        alpha (float): Transparency of overlay
        base_cmap (str): Colormap for base images
        overlay_cmap (str): Colormap for overlay images
        duration (int): Duration between frames in milliseconds
        figsize (tuple): Figure size for each frame
        titles (list): Optional list of titles for each frame
    
    Returns:
        str: Path to the created GIF file
    """
    frames = []
    
    for i, base_img in enumerate(base_images):
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot base image
        if len(base_img.shape) == 3:
            ax.imshow(base_img)
        else:
            ax.imshow(base_img)
        
        # Set title if provided
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)
        
        ax.axis('off')
        
        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        frame = Image.open(buf)
        frames.append(frame.copy())
        
        plt.close(fig)
        buf.close()
    
    # Create GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    
    print(f"Overlay GIF saved to: {gif_path}")
    return gif_path

def save_tracked_cells_to_pickle(tracked_cells, file_path):
    """
    Save tracked cells to a pickle file.
    
    Args:
        tracked_cells (dict): Dictionary containing tracked cell data
        file_path (str): Path to save the pickle file
    """
    with open(file_path, 'wb') as f:
        pickle.dump(tracked_cells, f)
    print(f"Tracked cells saved to: {file_path}")