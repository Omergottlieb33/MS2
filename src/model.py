import numpy as np
from cellpose import models, core, plot
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from cellpose.utils import masks_to_outlines
import time
import os

from src.utils.image_utils import load_czi_images, max_intensity_projection, mean_intensity_projection, get_labels_in_radius
from src.utils.eval_utils import get_dice_coefficient


class MS2:
    def __init__(self,
                 czi_path,
                 fluorescence_th=30,
                 device='cuda:0',
                 flow_threshold=0.4,
                 cellprob_threshold=0.0,
                 tile_norm_blocksize=128):
        self.image_data = load_czi_images(czi_path)
        # image_data dimensions
        self.z_stacks = self.image_data.shape[1]
        # channel 0 fluorescence, channel 1 cells
        self.microscope_channels = np.arange(0, self.image_data.shape[2])
        self.num_frames = self.image_data.shape[3]
        self.img_size = self.image_data.shape[4], self.image_data.shape[5]
        self.fluorescence_th = fluorescence_th
        self.device = device
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        self.tile_norm_blocksize = tile_norm_blocksize
        self.init_model()

    def init_model(self):
        if core.use_gpu() == False:
            raise ImportError("No GPU access, change your runtime")
        self.model = models.CellposeModel(gpu=True, device=self.device)

    def segment_cells(self, image):
        mask, flows, styles = self.model.eval(image,
                                              batch_size=1,
                                              flow_threshold=self.flow_threshold,
                                              cellprob_threshold=self.cellprob_threshold,
                                              normalize={"tile_norm_blocksize": self.tile_norm_blocksize})
        return mask, flows, styles

    def plot_cell_segmentation(self, image, mask, flows):
        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, image, mask, flows[0])
        plt.tight_layout()
        plt.show()

    def get_fluorescence_image(self, z, t):
        """
        Get the fluorescence image for a given z-stack and timepoint.
        """
        return self.image_data[0, z, self.microscope_channels[0], t, :, :, 0]

    def get_fluorescence_images_over_time(self, z):
        """
        Get the fluorescence images for a given z-stack over all timepoints.
        """
        return [self.get_fluorescence_image(z, t) for t in range(self.num_frames)]

    def get_cell_image(self, z, t):
        """
        Get the cell image for a given z-stack and timepoint.
        """
        return self.image_data[0, z, self.microscope_channels[1], t, :, :, 0]

    def get_cell_images_over_time(self, z):
        """
        Get the cell images for a given z-stack over all timepoints.
        """
        return [self.get_cell_image(z, t) for t in range(self.num_frames)]

    def cell_activation(self, mask, fluoroscent):
        pass

    def segment_cells_over_time(self, z: int, tn=None):
        if tn is None:
            tn = self.num_frames
        imgs = [self.image_data[0, z, self.microscope_channels[1], t, :, :, 0]
                for t in tqdm(range(tn), desc="Collecting images")]
        masks, flows, styles = self.model.eval(imgs,
                                               batch_size=32,
                                               flow_threshold=self.flow_threshold,
                                               cellprob_threshold=self.cellprob_threshold,
                                               normalize={
                                                   "tile_norm_blocksize": self.tile_norm_blocksize},
                                               channels=[0, 0])
        return masks

    def segment_cells_images(self, images: list):
        if isinstance(images, np.ndarray):
            images = [images[i] for i in range(len(images))]
        if not isinstance(images, list):
            raise ValueError("Input images must be a list of numpy arrays.")
        masks, flows, styles = self.model.eval(images,
                                               batch_size=32,
                                               flow_threshold=self.flow_threshold,
                                               cellprob_threshold=self.cellprob_threshold,
                                               normalize={
                                                   "tile_norm_blocksize": self.tile_norm_blocksize},
                                               channels=[0, 0])
        return masks

    def segment_cells_at_single_timepoint(self, t):
        imgs = [self.image_data[0, z, self.microscope_channels[1], t, :, :, 0]
                for z in tqdm(range(self.z_stacks), desc="Collecting images")]
        masks, flows, styles = self.model.eval(imgs,
                                               batch_size=32,
                                               flow_threshold=self.flow_threshold,
                                               cellprob_threshold=self.cellprob_threshold,
                                               normalize={
                                                   "tile_norm_blocksize": self.tile_norm_blocksize},
                                               channels=[0, 0])
        return masks

    def get_cells_z_projection(self, method='mean'):
        """
        Get the maximum intensity projection of cell masks for a given z-stack.
        """
        stack = self.image_data[0, :, self.microscope_channels[1], :, :, :, 0]
        if method == 'mean':
            return mean_intensity_projection(stack)
        elif method == 'max':
            return max_intensity_projection(stack)

    def get_ms2_z_projection(self, method='max'):
        """
        Get the maximum intensity projection of fluorescence images for a given z-stack.
        """
        stack = self.image_data[0, :, self.microscope_channels[0], :, :, :, 0]
        if method == 'mean':
            return mean_intensity_projection(stack)
        elif method == 'max':
            return max_intensity_projection(stack)

    @staticmethod
    def init_cell_tracks(masks):
        # Dictionary to store tracks: {cell_id: [mask_at_t0, mask_at_t1, ...]}
        cell_tracks = {}
        next_cell_id = 1
        # Initialize tracks with cells from the first frame
        flag = True
        i = 0
        while flag:
            mask_0 = masks[i]
            max_label_0 = int(mask_0.max())
            if max_label_0 == 0:
                i += 1
                flag = True
            else:
                flag = False
        for label in range(i+1, max_label_0 + 1):
            cell_tracks[next_cell_id] = [label]
            next_cell_id += 1
        return cell_tracks, next_cell_id, i

    def cell_tracking(self, masks, dice_threshold=0.3):
        """
        Track cells over time using dice coefficient matching.

        Args:
            masks (list): List of 2D numpy arrays (height x width) for each timepoint.
            dice_threshold (float): Minimum dice coefficient for a valid match

        Returns:
            list: List of 3D tensors (num_frames x height x width) for each tracked cell.
                  Each tensor contains the cell mask at each timepoint, or zeros if no match.
        """

        # Get image dimensions
        height, width = masks[0].shape
        cell_tracks, next_cell_id, i = self.init_cell_tracks(masks)
        # Dictionary to store tracks: {cell_id: [mask_at_t0, mask_at_t1, ...]}
        # Track cells through subsequent frames
        for t in tqdm(range(i, len(masks) - 1), desc="Tracking cells over time"):
            mask_t = masks[t]
            mask_t_plus_1 = masks[t + 1]

            max_label1 = int(mask_t.max())
            max_label2 = int(mask_t_plus_1.max())

            # Keep track of which cells in t+1 have been matched
            matched_labels_t_plus_1 = set()

            # For each existing track, try to find a match in the next frame
            tracks_to_extend = {}

            # Find the best matches for cells in current frame
            for cell_id, track in cell_tracks.items():
                if len(track) == t + 1 - i:  # This track is active (has data up to current frame)
                    # Get the cell mask at current time
                    current_cell_mask = track[t-i]
                    # Find corresponding label in mask_t
                    relevant_labels_at_t = get_labels_in_radius(current_cell_mask, mask_t, radius=5)
                    # If no relevant labels found, continue to next cell
                    if len(relevant_labels_at_t) == 0:
                        # No relevant labels found, add empty mask
                        cell_tracks[cell_id].append(0)
                        continue
                    # Check if current cell mask matches any label in the current frame
                    current_label = None
                    for label in relevant_labels_at_t:
                        label_mask = (mask_t == label).astype(np.uint8)
                        if np.array_equal(current_cell_mask, label_mask):
                            current_label = label
                            break

                    if current_label is None:
                        # Cell not found in current frame, add empty mask
                        cell_tracks[cell_id].append(0)
                        continue

                    # # Find best match in next frame
                    best_dice_coeff = 0
                    best_match_label = None
                    relevant_labels_at_t_plus_1 = get_labels_in_radius(
                        current_cell_mask, mask_t_plus_1, radius=5)
                    for next_label in relevant_labels_at_t_plus_1:
                        if next_label in matched_labels_t_plus_1:
                            continue  # This label is already matched

                        cell_mask_t_plus_1 = (
                            mask_t_plus_1 == next_label).astype(np.uint8)
                        dice_coeff = get_dice_coefficient(
                            current_cell_mask, cell_mask_t_plus_1)

                        if dice_coeff > best_dice_coeff and dice_coeff >= dice_threshold:
                            best_dice_coeff = dice_coeff
                            best_match_label = next_label

                    if best_match_label is not None:
                        # Found a good match
                        best_match_mask = (
                            mask_t_plus_1 == best_match_label).astype(np.uint8)
                        tracks_to_extend[cell_id] = best_match_label
                        matched_labels_t_plus_1.add(best_match_label)
                    else:
                        # No good match found, add empty mask
                        tracks_to_extend[cell_id] = 0

            # Extend existing tracks
            for cell_id, next_label in tracks_to_extend.items():
                cell_tracks[cell_id].append(next_label)

            # Add new cells that weren't matched (new cell divisions or cells entering frame)
            for label in range(1, max_label2 + 1):
                if label not in matched_labels_t_plus_1:
                    # Fill previous timepoints with empty masks
                    new_track = [0 for _ in range(t + 1)]
                    new_track.append(label)
                    cell_tracks[next_cell_id] = new_track
                    next_cell_id += 1

        return cell_tracks
