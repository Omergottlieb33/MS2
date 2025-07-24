import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg



def create_gif_from_figures(figures, output_path='animation.gif', fps=5, titles=None):
    """
    Create a GIF from a list of matplotlib figures
    
    Parameters:
    - figures: list of matplotlib figure objects
    - output_path: path to save the GIF
    - fps: frames per second
    - titles: optional list of titles for each frame
    """    
    frames = []
    
    for i, fig in enumerate(figures):
        # Add title if provided
        if titles and i < len(titles):
             # Adjust the layout to make room for the title
            fig.subplots_adjust(top=0.9)  # Leave space at the top for title
            fig.suptitle(titles[i], fontsize=16, y=0.95)  # Position title higher
        
        # Convert figure to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
        
        # Convert to PIL Image
        img = Image.open(buf)
        frames.append(img.copy())
        buf.close()
    
    # Create and save the GIF
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000/fps),  # Convert fps to milliseconds
            loop=0
        )
        print(f"GIF saved to {output_path}")
        
        # Clean up figures to free memory
        for fig in figures:
            plt.close(fig)
    else:
        print("No frames were created")


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