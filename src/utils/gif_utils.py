import io
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import imageio.v2 as iio


def create_gif_from_figures(figures, output_path='animation.gif', fps=5, titles=None):
    """
    Create an animation (GIF / MP4 / multi-page TIFF) from a list of matplotlib Figure objects.

    Ensures all frames:
      - Have identical (H, W)
      - Are uint8
      - Are 3-channel RGB (drops alpha if present)

    Parameters
    ----------
    figures : list[matplotlib.figure.Figure]
    output_path : str
    fps : int
    titles : list[str] | None
    """
    frames = []
    for i, fig in enumerate(figures):
        if titles and i < len(titles):
            # Put title (avoid changing final canvas size)
            fig.suptitle(titles[i], fontsize=14)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)  # ARGB
        # Convert ARGB -> RGBA then drop alpha
        rgba = np.empty_like(buf)
        rgba[..., 0] = buf[..., 1]  # R
        rgba[..., 1] = buf[..., 2]  # G
        rgba[..., 2] = buf[..., 3]  # B
        rgba[..., 3] = buf[..., 0]  # A
        rgb = rgba[..., :3]

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        frames.append(rgb)

    if not frames:
        print("No frames to write.")
        return

    # Enforce uniform size (resize any mismatches)
    target_h, target_w = frames[0].shape[:2]
    uniform = []
    resized = 0
    for f in frames:
        if f.shape[0] != target_h or f.shape[1] != target_w:
            resized += 1
            f_img = Image.fromarray(f)
            f_img = f_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
            f = np.array(f_img)
        uniform.append(f)
    frames = uniform
    if resized:
        print(f"Resized {resized} frame(s) to uniform {(target_h, target_w)}.")

    ext = os.path.splitext(output_path.lower())[1]

    if ext in ('.mp4', '.avi'):
        # MP4: some codecs dislike alpha & odd dims; we already have RGB & ensured size.
        # Ensure even dimensions (required by some encoders like libx264 with yuv420p).
        if target_h % 2 or target_w % 2:
            pad_h = target_h % 2
            pad_w = target_w % 2
            frames = [np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge') for f in frames]
            target_h += pad_h
            target_w += pad_w
            print(f"Padded frames to even size {(target_h, target_w)} for encoder.")

        try:
            with iio.get_writer(output_path, fps=fps, codec='libx264') as writer:
                for idx, frame in enumerate(frames):
                    if frame.shape[:2] != (target_h, target_w):
                        raise ValueError(f"Frame {idx} has shape {frame.shape[:2]} != {(target_h, target_w)} (after normalization).")
                    writer.append_data(frame)
        except Exception as e:
            print(f"Primary video write failed ({e}); retrying without explicit codec.")
            with iio.get_writer(output_path, fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)

    elif ext in ('.gif',):
        duration_ms = int(1000 / max(fps, 1))
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    elif ext in ('.tif', '.tiff'):
        try:
            import tifffile
            with tifffile.TiffWriter(output_path) as tif:
                for i, f in enumerate(frames):
                    tif.write(f, description="fps={}".format(fps) if i == 0 else None)
        except ImportError:
            with iio.get_writer(output_path, mode='I') as writer:
                for f in frames:
                    writer.append_data(f)
    else:
        print(f"Unknown extension '{ext}', defaulting to GIF.")
        duration_ms = int(1000 / max(fps, 1))
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            output_path if ext else output_path + '.gif',
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )

    print(f"Saved animation to {output_path}")

    # Close figures to free memory
    for fig in figures:
        plt.close(fig)


def create_trajectory_gif(positions, output_path='trajectory.gif', fps=2, 
                         figsize=(10, 8), point_size=50, line_width=2):
    """
    Create a GIF animation of a 3D trajectory where points are added progressively
    
    Parameters:
    - positions: numpy array of shape (n_timepoints, 3) containing x, y, z coordinates
    - output_path: path to save the GIF
    - fps: frames per second
    - figsize: figure size tuple
    - point_size: size of the markers
    - line_width: width of the connecting lines
    """
  
    
    frames = []
    
    # Get the full range for consistent axis limits
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
    
    # Add some padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    z_padding = (z_max - z_min) * 0.1
    
    for frame_idx in range(len(positions)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Get points up to current frame
        current_positions = positions[:frame_idx + 1]
        
        if len(current_positions) > 0:
            # Plot the trajectory line up to current point
            if len(current_positions) > 1:
                ax.plot(current_positions[:, 0], 
                       current_positions[:, 1], 
                       current_positions[:, 2], 
                       color='blue', linewidth=line_width, alpha=0.7)
            
            # Plot all points up to current frame
            ax.scatter(current_positions[:, 0], 
                      current_positions[:, 1], 
                      current_positions[:, 2], 
                      c=range(len(current_positions)), 
                      cmap='viridis', s=point_size, alpha=0.8)
            
            # Highlight the current (newest) point
            current_point = current_positions[-1]
            ax.scatter(current_point[0], current_point[1], current_point[2], 
                      c='red', s=point_size*1.5, marker='o', alpha=1.0)
        
        # Set consistent axis limits
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.set_zlim(z_min - z_padding, z_max + z_padding)
        
        # Labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f'Cell Trajectory - Time Point {frame_idx + 1}/{len(positions)}')
        
        # Convert figure to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        
        # Convert to PIL Image
        img = Image.open(buf)
        frames.append(img.copy())
        
        # Clean up
        plt.close(fig)
        buf.close()
    
    # Create and save the GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000/fps),
            loop=0
        )
        print(f"Trajectory GIF saved to {output_path}")
    else:
        print("No frames were created")

def create_extending_plot_gif(x_data, y_data, output_path='gene_Expression.gif', 
                             fps=2, figsize=(10, 6), line_color='blue', 
                             marker_color='red', line_width=2, marker_size=50,
                             xlabel='X', ylabel='Y', title='Extending Plot',
                             grid=True, show_current_point=True):
    """
    Create a GIF animation where a plot extends/grows progressively by adding one point at a time.
    
    Parameters:
    - x_data: array-like, x-coordinates of the data points
    - y_data: array-like, y-coordinates of the data points
    - output_path: path to save the GIF
    - fps: frames per second
    - figsize: figure size tuple
    - line_color: color of the line connecting points
    - marker_color: color of the current/newest point marker
    - line_width: width of the connecting line
    - marker_size: size of the current point marker
    - xlabel, ylabel, title: plot labels
    - grid: whether to show grid
    - show_current_point: whether to highlight the current point
    """    
    # Convert to numpy arrays
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    
    frames = []
    
    # Get the full range for consistent axis limits
    x_min, x_max = x_data.min(), x_data.max()
    y_min, y_max = y_data.min(), y_data.max()
    
    # Add some padding
    x_padding = (x_max - x_min) * 0.1 if x_max > x_min else 1
    y_padding = (y_max - y_min) * 0.1 if y_max > y_min else 1
    
    for i in range(1, len(x_data) + 1):
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get data up to current point
        current_x = x_data[:i]
        current_y = y_data[:i]
        
        # Plot the line up to current point
        if len(current_x) > 1:
            ax.plot(current_x, current_y, color=line_color, 
                   linewidth=line_width, alpha=0.8, marker='o', 
                   markersize=4, markerfacecolor=line_color, 
                   markeredgecolor='white', markeredgewidth=0.5)
        
        # Highlight the current (newest) point
        if show_current_point and len(current_x) > 0:
            ax.scatter(current_x[-1], current_y[-1], 
                      c=marker_color, s=marker_size, 
                      marker='o', alpha=1.0, zorder=5,
                      edgecolors='white', linewidth=1)
        
        # Set consistent axis limits
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Labels and formatting
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title} - Point {i}/{len(x_data)}')
        
        if grid:
            ax.grid(True, alpha=0.3)
        
        # Add point count annotation
        ax.text(0.02, 0.98, f'Points: {i}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Convert figure to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        
        # Convert to PIL Image
        img = Image.open(buf)
        frames.append(img.copy())
        
        # Clean up
        plt.close(fig)
        buf.close()
    
    # Create and save the GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000/fps),
            loop=0
        )
        print(f"Extending plot GIF saved to {output_path}")
    else:
        print("No frames were created")