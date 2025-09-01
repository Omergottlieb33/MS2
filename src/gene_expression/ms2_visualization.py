import os
import numpy as np
import matplotlib
from scipy import ndimage
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.utils.plot_utils import (
    show_3d_segmentation_overlay_with_unique_colors,
    plot_2d_gaussian_with_size
)
from src.utils.gif_utils import create_gif_from_figures, create_trajectory_gif


class MS2VisualizationManager:
    """
    Handles all visualization (figures, animations, plots) for MS2 processing.
    The processor pushes data here; this class is stateless w.r.t. computation.
    """

    def __init__(self, output_dir: str, enabled: bool = True, pad=5):
        self.output_dir = output_dir
        self.enabled = enabled
        self.pad = pad
        os.makedirs(self.output_dir, exist_ok=True)
        self._reset()

    def _reset(self):
        self.timepoint_figures = []
        self.segmentation_figures = []
        self.max_cell_intensity = 0
        self.current_cell_bbox_ms2 = None

    def start_cell(self, cell_id: int, max_cell_intensity: float):
        if not self.enabled:
            return
        self._reset()
        self.cell_id = cell_id
        self.max_cell_intensity = max_cell_intensity

    def add_timepoint(
        self,
        timepoint: int,
        ms2_projection,
        cell_mask_projection_2d,
        cell_bbox_ms2,
        bbox_coords,  # (z1,y1,x1,z2,y2,x2)
        gaussian_params: dict | None,
        covariance_matrix,
        method: str,
        peak_xy,
        add_segmentation_3d: bool,
        z_stack=None,
        masks=None,
        cell_label=None
    ):
        if not self.enabled:
            return
        z1, y1, x1, z2, y2, x2 = bbox_coords
        self.current_cell_bbox_ms2 = cell_bbox_ms2

        fig = plt.figure(figsize=(12, 5))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

        pad = 2
        # Safely crop the region with padding, ensuring we don't go out of bounds.
        h_full, w_full = ms2_projection.shape
        y_start = max(0, y1 - pad)
        y_end = min(h_full, y2 + pad)
        x_start = max(0, x1 - pad)
        x_end = min(w_full, x2 + pad)
        ms2_region = ms2_projection[y_start:y_end, x_start:x_end]

        # Outline
        cell_binary = (cell_mask_projection_2d > 0).astype(np.uint8)
        cell_outline = cell_binary - ndimage.binary_erosion(cell_binary)
        outline_rgba = np.zeros((*cell_binary.shape, 4))
        outline_rgba[cell_outline == 1] = [0, 1, 0, 1]
        outline_region = outline_rgba[y_start:y_end, x_start:x_end]

        # Use self.max_cell_intensity for vmax to ensure consistent brightness
        # scaling across all timepoints in the animation for this cell. This
        # makes the 2D plot consistent with the 3D plot's Z-axis.
        ax_img.imshow(ms2_region, cmap='gray', vmin=0, vmax=self.max_cell_intensity)
        ax_img.imshow(outline_region, alpha=0.25)

        # Gaussian center + σ ellipses
        if gaussian_params is not None:
            ax_img.plot(
                gaussian_params['x0']+pad,
                gaussian_params['y0']+pad,
                marker='o',
                markersize=4,
                markerfacecolor='blue',
                markeredgecolor='black',
                linewidth=0
            )
            for k, color, lw in [(1, 'yellow', 1.0), (2, 'orange', 1.0), (3, 'red', 1.2)]:
                e = Ellipse(
                    (gaussian_params['x0']+pad, gaussian_params['y0']+pad),
                    width=2 * k * gaussian_params['sigma_x'],
                    height=2 * k * gaussian_params['sigma_y'],
                    angle=0.0,
                    facecolor='none',
                    edgecolor=color,
                    linewidth=lw,
                    alpha=0.9,
                    zorder=5
                )
                ax_img.add_patch(e)

        if peak_xy and peak_xy != (0, 0):
            ax_img.plot(
                peak_xy[0]+pad, peak_xy[1]+pad,
                marker='o', markersize=4,
                markerfacecolor='red', markeredgecolor='black', linewidth=0
            )

        ax_img.set_title(f'MS2 expression t={timepoint} ({method})')
        ax_img.axis('off')

        # 3D Gaussian
        if gaussian_params is not None:
            h, w = cell_bbox_ms2.shape
            X, Y = np.meshgrid(np.arange(w), np.arange(h))
            Z = plot_2d_gaussian_with_size(
                gaussian_params['amplitude'],
                gaussian_params['x0'],
                gaussian_params['y0'],
                gaussian_params['sigma_x'],
                gaussian_params['sigma_y'],
                gaussian_params['offset'],
                w, h
            )
            surf = ax_3d.plot_surface(
                X, Y, Z, cmap='viridis',
                edgecolor='none', antialiased=True, linewidth=0
            )
            ax_3d.set_zlim(0, self.max_cell_intensity)
            surf.set_clim(0, self.max_cell_intensity)
            fig.colorbar(surf, ax=ax_3d, orientation='vertical')
            ax_3d.set_xlabel('x')
            ax_3d.set_ylabel('y')
            ax_3d.set_zlabel('intensity')
            ax_3d.view_init(elev=30, azim=230)
            # LaTeX Gaussian equation with current parameter values
            ax_3d.set_title(
                r'$G(x,y)=A e^{-\frac{(x-x_0)^2}{2\sigma_x^2}-\frac{(y-y_0)^2}{2\sigma_y^2}} + C$'
                '\n'
                rf'$A={gaussian_params["amplitude"]:.1f},\ x_0={gaussian_params["x0"]:.1f},\ y_0={gaussian_params["y0"]:.1f},\ '
                rf'\sigma_x={gaussian_params["sigma_x"]:.2f},\ \sigma_y={gaussian_params["sigma_y"]:.2f},\ C={gaussian_params["offset"]:.1f}$',
                fontsize=9, pad=10
            )
        else:
            ax_3d.text2D(0.1, 0.5, 'Gaussian fit failed', transform=ax_3d.transAxes)
            ax_3d.set_axis_off()

        plt.tight_layout()
        self.timepoint_figures.append(fig)

        if add_segmentation_3d and z_stack is not None:
            seg_fig = show_3d_segmentation_overlay_with_unique_colors(
                z_stack, masks, cell_label,
                return_fig=True, zoom_on_highlight=True
            )
            self.segmentation_figures.append(seg_fig)

    def save_timepoint_animation(self, cell_id: int, valid_timepoints, ransac_mad_k):
        if not self.enabled or not self.timepoint_figures:
            return
        out = os.path.join(
            self.output_dir,
            f"cell_{cell_id}_expression_animation_ransac_th{ransac_mad_k}.mp4"
        )
        titles = [f"Timepoint {t}" for t in valid_timepoints]
        create_gif_from_figures(self.timepoint_figures, out, fps=1, titles=titles)

    def save_segmentation_animation(self, cell_id: int, valid_timepoints):
        if not self.enabled or not self.segmentation_figures:
            return
        out = os.path.join(self.output_dir, f"cell_{cell_id}_segmentation.mp4")
        titles = [f"Timepoint {t}" for t in valid_timepoints]
        create_gif_from_figures(self.segmentation_figures, out, fps=1, titles=titles)

    def expression_plot(self, cell_id: int, intensities, title):
        if not self.enabled or not intensities:
            return
        t_axis = np.arange(len(intensities))
        amps = np.asarray(intensities, dtype=float)

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(t_axis, amps, color='tab:blue', linewidth=0.8)
        ax.scatter(t_axis, amps, s=18, color='tab:blue', edgecolors='none', zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Time [30*sec]")
        ax.set_ylabel("Emitter Intensity [AU]")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join(
            self.output_dir, f"cell_{cell_id}_expression_plot_v3_global_maxima_finder_{title}.png"
        )
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()

    def center_of_mass_trajectory(self, cell_id: int, positions):
        if not self.enabled or positions is None or len(positions) == 0:
            return
        out = os.path.join(self.output_dir, f"cell_{cell_id}_center_of_mass.gif")
        create_trajectory_gif(np.array(positions), out, fps=1)