import os
import warnings
import numpy as np
import matplotlib
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
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
        # Fixed-size ROI (set on first timepoint)
        self._roi_h = None
        self._roi_w = None

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
        method: str,
        peak_xy,
        add_segmentation_3d: bool,
        z_stack=None,
        masks=None,
        cell_label=None,
        animation_type: str = 'ms2_expression'
    ):
        if not self.enabled:
            return
        # Defensive shape checks to avoid 2D indexing on 1D arrays
        if ms2_projection is None or np.ndim(ms2_projection) != 2:
            raise ValueError(f"ms2_projection must be 2D (H,W). Got: {None if ms2_projection is None else ms2_projection.shape}")
        if cell_mask_projection_2d is None or np.ndim(cell_mask_projection_2d) != 2:
            raise ValueError(f"cell_mask_projection_2d must be 2D (H,W). Got: {None if cell_mask_projection_2d is None else cell_mask_projection_2d.shape}")
        # Normalize z_stack to a 2D projection if provided
        z_stack_projection = None
        if z_stack is not None:
            if np.ndim(z_stack) == 3:  # (Z,H,W)
                z_stack_projection = np.sum(z_stack, axis=0)
            elif np.ndim(z_stack) == 2:  # already (H,W)
                z_stack_projection = z_stack
            else:
                warnings.warn(f"z_stack has unsupported ndim={np.ndim(z_stack)}; skipping z-stack overlay.")
                z_stack_projection = None
        z1, y1, x1, z2, y2, x2 = bbox_coords
        self.current_cell_bbox_ms2 = cell_bbox_ms2

        fig = plt.figure(figsize=(12, 5))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_3d = fig.add_subplot(1, 2, 2, projection='3d')

        pad = 5
        # # Safely crop the region with padding, ensuring we don't go out of bounds.
        h_full, w_full = ms2_projection.shape
        # y_start = max(0, y1 - pad)
        # y_end = min(h_full, y2 + pad)
        # x_start = max(0, x1 - pad)
        # x_end = min(w_full, x2 + pad)
        # Initialize fixed ROI size on first timepoint
        if self._roi_h is None or self._roi_w is None:
            self._roi_h = min((y2 - y1) + 2 * pad, h_full)
            self._roi_w = min((x2 - x1) + 2 * pad, w_full)

        # Center ROI on current bbox center, but keep size fixed and clamp to image
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2
        y_start = int(cy - self._roi_h // 2)
        x_start = int(cx - self._roi_w // 2)
        y_start = max(0, min(y_start, h_full - self._roi_h))
        x_start = max(0, min(x_start, w_full - self._roi_w))
        y_end = y_start + self._roi_h
        x_end = x_start + self._roi_w
        ms2_region = ms2_projection[y_start:y_end, x_start:x_end]

        # Outline
        cell_binary = (cell_mask_projection_2d > 0).astype(np.uint8)
        cell_rgba = np.zeros((*cell_binary.shape, 4))
        cell_rgba[cell_binary == 1] = [0, 0, 1, 1]  # blue fill
        cell_region = cell_rgba[y_start:y_end, x_start:x_end]

        cell_outline = cell_binary - ndimage.binary_erosion(cell_binary)
        outline_rgba = np.zeros((*cell_binary.shape, 4))
        outline_rgba[cell_outline == 1] = [0, 0, 1, 1] # blue outline
        outline_region = outline_rgba[y_start:y_end, x_start:x_end]

        if animation_type != 'ms2_expression':
            cell_z_stack_projection = (
                z_stack_projection[y_start:y_end, x_start:x_end]
                if z_stack_projection is not None and np.ndim(z_stack_projection) == 2
                else None
            )
            color_quantiles = [(0,0,0),(0,0,0.5),(0,0,1)]
            blue_cmap = LinearSegmentedColormap.from_list('black_blue', color_quantiles, N=256)
            ax_img.imshow(cell_z_stack_projection, cmap=blue_cmap)
            ax_img.imshow(outline_region, alpha=0.3)
        # Use self.max_cell_intensity for vmax to ensure consistent brightness
        # scaling across all timepoints in the animation for this cell. This
        # makes the 2D plot consistent with the 3D plot's Z-axis.
        if animation_type == 'ms2_expression':
            ax_img.imshow(ms2_region, cmap='gray', vmin=0, vmax=self.max_cell_intensity)
            ax_img.imshow(outline_region, alpha=0.25)

            # Gaussian center + σ ellipses
            if gaussian_params is not None:
                # Highlight pixels within the 2σ ellipse (yellow)
                h_r, w_r = ms2_region.shape
                Xr, Yr = np.meshgrid(np.arange(w_r), np.arange(h_r))
                cx = gaussian_params['x0'] + pad
                cy = gaussian_params['y0'] + pad
                sx = max(gaussian_params['sigma_x'], 1e-6)
                sy = max(gaussian_params['sigma_y'], 1e-6)
                mask_2sigma = ((Xr - cx)**2 / (sx**2) + (Yr - cy)**2 / (sy**2)) <= 4.0

                highlight_rgba_region = np.zeros((h_r, w_r, 4), dtype=float)
                highlight_rgba_region[mask_2sigma] = [1, 1, 0, 1]  # yellow
                ax_img.imshow(highlight_rgba_region, alpha=0.3)  # draw highlight
                ax_img.imshow(outline_region, alpha=0.25)        # keep outline on top

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

        ax_img.set_title(f'MS2 expression t={timepoint}')
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

        if add_segmentation_3d and z_stack is not None and z_stack.shape == masks.shape:
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
            f"cell_{cell_id}_expression_animation.tif"
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
        t_axis = np.arange(len(intensities))*30/60  # minutes
        amps = np.asarray(intensities, dtype=float)
        amps_with_noise = amps.copy()
        amps_with_noise[amps_with_noise < 17] = 17
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(t_axis, amps_with_noise, color='tab:blue', linewidth=0.8)
        ax.scatter(t_axis, amps_with_noise, s=18, color='tab:blue', edgecolors='none', zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Time [min]")
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