import os
import json
import tifffile
import numpy as np
from tqdm import tqdm

from cell_tracking import get_masks_paths
from src.utils.image_utils import load_czi_images
from src.utils.gif_utils import create_gif_from_figures,create_extending_plot_gif
from src.utils.filter_utils import median_filter_over_time, get_gene_expression_clusters
from src.utils.plot_utils import show_3d_segmentation_overlay_with_unique_colors2, plot_masked_pixels_3d


def main(tracklets,cell_id, image_data, masks_paths, filtered_ms2_background_sub, th):
    cell_labels = tracklets[str(cell_id)]
    gene_expression_list = []
    figures_segmentation = []
    figures_3d_intensity = []
    max_intensity = np.max(image_data[0, :, 0, :, :, :, 0])
    print(f"Max intensity in the MS2 channel: {max_intensity}")
    valid_timepoints = [t for t in range(len(cell_labels)) if cell_labels[t]]
    for t in tqdm(valid_timepoints):
        z_stack_t = image_data[0, t, 1, :, :, :, 0]
        ms2_stack_t = image_data[0, t, 0, :, :, :, 0]
        mask_t = np.load(masks_paths[t])['masks']
        cell_mask_t = (mask_t == cell_labels[t]).astype(np.uint8)
        gene_clusters = get_gene_expression_clusters(ms2_stack_t, filtered_ms2_background_sub[t,:,:,:], th)
        cell_gene_expression_overlay = gene_clusters * cell_mask_t
        gene_expression = np.sum(cell_gene_expression_overlay)
        gene_expression_list.append(gene_expression)
        fig = show_3d_segmentation_overlay_with_unique_colors2(z_stack_t,
                                                            mask_t,
                                                            highlight_label=cell_labels[t],
                                                            highlight_color=[255, 0, 0],
                                                            color_scheme='hsv',
                                                            return_fig=True,
                                                            zoom_on_highlight=True,
                                                            zoom_padding=50)
        figures_segmentation.append(fig)
        fig_intensity = plot_masked_pixels_3d(gene_clusters, cell_mask_t,
                                            title=f"3D Intensity Plot for Cell ID {cell_labels[t]} at Time {t+1}",
                                            point_size=5, alpha=0.7,
                                            cmap='viridis', return_fig=True,
                                            vmin=0.0, vmax=max_intensity)
        fig_intensity = plot_masked_pixels_3d(
            image_tensor=gene_clusters,
            mask_tensor=cell_mask_t,
            title=f"3D Intensity Plot for Cell ID {cell_labels[t]} at Time {t+1}",
            point_size=5, alpha=0.7,
            save_path=None, return_fig=True,
            vmin=0.0, vmax=max_intensity,threshold=th,
            use_threshold_coloring=True
        )
        figures_3d_intensity.append(fig_intensity)
    create_extending_plot_gif(
    x_data=valid_timepoints,
    y_data=gene_expression_list,
    output_path=f'gene_expression_id_{cell_id}.gif',
    fps=1,
    figsize=(10, 6),
    line_color='blue',
    title=f'Gene expression for cell ID {cell_id}',
    xlabel='Time (frames)',
    ylabel= 'A.U')
    output_path1 = os.path.join(f'3d_tracking_id_{cell_id}.gif')
    create_gif_from_figures(figures_segmentation, output_path=output_path1, fps=1, 
                       titles=[f'Time {i+1}' for i in valid_timepoints])
    output_path2 = os.path.join(f'3d_intensity_id_{cell_id}_th_colored.gif')
    create_gif_from_figures(figures_3d_intensity, output_path=output_path2, fps=1, 
                       titles=f'3D Intensity Plot for Cell ID {cell_id}')


if __name__ == "__main__":
    czi_file_path = '/home/dafei/data/MS2/gRNA2_12.03.25-st-13-II---.czi'
    seg_maps_dir = '/home/dafei/output/MS2/3d_cell_segmentation/gRNA2_12.03.25-st-13-II---/masks'
    background_sub_ms2_channel = "/home/dafei/output/MS2/3d_cell_segmentation/gRNA2_12.03.25-st-13-II---/C1-gRNA2_12.03.25-st-13-II---_ms2_channel_background_sub.tif"
    tracklets_path = '/home/dafei/output/MS2/3d_cell_segmentation/gRNA2_12.03.25-st-13-II---/masks/tracklets_matching_iou.json'

    image_data = load_czi_images(czi_file_path)
    masks_paths = get_masks_paths(seg_maps_dir)
    ms2_background_sub = tifffile.imread(background_sub_ms2_channel)
    with open(tracklets_path, 'r') as f:
        tracklets = json.load(f)

    filtered_ms2_background_sub = median_filter_over_time(ms2_background_sub, kernel_size=3)
    th = np.percentile(filtered_ms2_background_sub, 99.999)
    print(f"Threshold for background subtraction: {th}")
    cell_id = 7  # Example cell ID, change as needed
    main(tracklets, cell_id, image_data, masks_paths, filtered_ms2_background_sub, th)