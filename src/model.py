import numpy as np
from cellpose import models, core, plot
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from cellpose.utils import masks_to_outlines

from utils import load_czi_images


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

    def segment_cells(self, z, t, plot=False):
        image = self.image_data[0, z, self.microscope_channels[1], t, :, :, 0]
        mask, flows, styles = self.model.eval(image,
                                            batch_size=1,
                                            flow_threshold=self.flow_threshold,
                                            cellprob_threshold=self.cellprob_threshold,
                                            normalize={"tile_norm_blocksize": self.tile_norm_blocksize})
        if plot:
            fig = plt.figure(figsize=(12,5))
            plot.show_segmentation(fig, image, masks, flows[0])
            plt.tight_layout()
            plt.show()
        return mask
    
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
    
    def intensity_cell_matching(self, z, t, mask, fluoroscent):
        """
        Match the intensity of the fluorescence channel to the cell mask.
        """
        output = {'z': z, 't': t, 'cells': [], 'intensities': []}
        #TODO: develop background sepperation method for fluorescence channel to identify relevant intensities
        fluoroscent_th = fluoroscent > self.fluorescence_th
        num_labels = int(mask.max())
        for label in tqdm(range(1, num_labels)):
            cell_masks = mask==label
            cell_outlines = masks_to_outlines(cell_masks)
            loclaziation_mask = cell_masks * fluoroscent_th
            cell_intensity = np.sum(loclaziation_mask*fluoroscent)
            if cell_intensity > 0:
                output['cells'].append(label)
                output['intensities'].append(cell_intensity)
        output['cells'] = np.array(output['cells'])
        output['intensities'] = np.array(output['intensities'])
        return output
    
    def segment_cells_over_time(self, z):
        imgs = [self.image_data[0, z, self.microscope_channels[1], t, :, :, 0] for t in tqdm(range(self.num_frames), desc="Collecting images")]
        masks, flows, styles = self.model.eval(imgs,
                                               batch_size=32,
                                               flow_threshold=self.flow_threshold,
                                               cellprob_threshold=self.cellprob_threshold,
                                               normalize={"tile_norm_blocksize": self.tile_norm_blocksize},
                                               channels=[0,0])
        return masks
    
    def segment_cells_at_single_timepoint(self, t):
        imgs = [self.image_data[0, z, self.microscope_channels[1], t, :, :, 0] for z in tqdm(range(self.z_stacks), desc="Collecting images")]
        masks, flows, styles = self.model.eval(imgs,
                                               batch_size=32,
                                               flow_threshold=self.flow_threshold,
                                               cellprob_threshold=self.cellprob_threshold,
                                               normalize={"tile_norm_blocksize": self.tile_norm_blocksize},
                                               channels=[0,0])
        return masks
    
    def cell_tracking_v2(self, z):
        masks = self.segment_cells_over_time(z)
        # init_tracked_masks = [mask/max(mask) for mask in masks]  # Initialize tracked_masks with the masks from the first frame
        # for t in tqdm(range(1, self.num_frames), desc="Tracking cells over time"):
        #     current_mask = masks[t]
        #     for cell_mask in init_tracked_masks:
        #         best_overlap = 0
        #         for current_cell_mask in current_mask:
        #             normalized_cell_mask = current_cell_mask / max(current_cell_mask)
        #             overlap = np.sum(cell_mask * normalized_cell_mask)
        #             if overlap > best_overlap:
        #                 best_overlap = overlap
        #                 best_match = current_cell_mask
        #         if best_overlap > 0.5:
        #             pass


                

    
    def cell_tracking(self,z):
        masks = self.segment_cells_over_time(z)

        all_tracks_data = [] # To store links between frames

        for t in tqdm(range(self.num_frames - 1), desc="Processing frame pairs for tracking"):
            mask1 = masks[t]
            mask2 = masks[t+1]
            max_label1 = int(mask1.max())
            max_label2 = int(mask2.max())

            current_frame_links = []

            if max_label1 == 0 or max_label2 == 0:
                # No cells in one or both masks, so no links to find
                all_tracks_data.append({'frame_index_t': t, 'frame_index_t+1': t + 1, 'links': current_frame_links})
                continue

            # Identify pixels where both masks have a cell (non-zero label)
            # These are potential overlap regions.
            valid_overlap_pixels = (mask1 > 0) & (mask2 > 0)

            if not np.any(valid_overlap_pixels):
                # No overlapping pixels at all between any cells in mask1 and mask2
                all_tracks_data.append({'frame_index_t': t, 'frame_index_t+1': t + 1, 'links': current_frame_links})
                continue

            # Get the labels from mask1 and mask2 at these overlapping pixel locations
            m1_labels_at_overlap = mask1[valid_overlap_pixels]
            m2_labels_at_overlap = mask2[valid_overlap_pixels]

            # Convert labels to 0-indexed for use with np.bincount
            # (Cellpose labels are typically 1-indexed)
            m1_indices = m1_labels_at_overlap - 1
            m2_indices = m2_labels_at_overlap - 1

            # Create a 1D representation of label pairs (m1_idx, m2_idx)
            # This unique ID allows np.bincount to count occurrences of each pair.
            combined_indices = m1_indices * max_label2 + m2_indices

            # Calculate overlap counts for all (cell_in_mask1, cell_in_mask2) pairs
            # minlength ensures the output array is large enough for all possible combined_indices
            flat_overlap_counts = np.bincount(combined_indices,
                                               minlength=max_label1 * max_label2)

            # Reshape to an overlap matrix:
            # overlap_matrix[i, j] = overlap area between cell (i+1) in mask1 and cell (j+1) in mask2
            overlap_matrix = flat_overlap_counts.reshape((max_label1, max_label2))

            # For each cell in mask1, find the cell in mask2 with the maximum overlap
            best_match_indices_mask2 = np.argmax(overlap_matrix, axis=1) # 0-indexed label for mask2
            best_match_scores = np.max(overlap_matrix, axis=1)       # The actual max overlap score

            for i in range(max_label1): # i is the 0-indexed label for mask1
                if best_match_scores[i] > 0: # Consider only actual overlaps
                    label1 = i + 1 # Original 1-indexed label in mask1
                    label2_match = best_match_indices_mask2[i] + 1 # Original 1-indexed label in mask2
                    current_frame_links.append({
                        'from_label': label1,
                        'to_label': label2_match,
                        'overlap_score': int(best_match_scores[i]) # Convert score to int for consistency
                    })
            all_tracks_data.append({'frame_index_t': t, 'frame_index_t+1': t + 1, 'links': current_frame_links})
        
        # Collect matched labels into tracks.
        # Each track is a list of (frame_index, label) tuples.
        final_tracks = []
        # Maps a label in the *current* frame to its track ID (index in final_tracks)
        active_track_for_label = {}

        # Initialize tracks with all cells from the first frame
        if self.num_frames > 0:
            mask0 = masks[0]
            for label in range(1, int(mask0.max()) + 1):
                track_id = len(final_tracks)
                final_tracks.append([(0, label)])
                active_track_for_label[label] = track_id

        # Process links frame by frame to build and extend tracks
        for frame_links_info in all_tracks_data:
            t = frame_links_info['frame_index_t']
            links = frame_links_info['links']
            
            next_active_track_for_label = {}
            linked_to_labels = set()

            for link in links:
                from_label = link['from_label']
                to_label = link['to_label']
                
                if from_label in active_track_for_label:
                    # Continue an existing track
                    track_id = active_track_for_label[from_label]
                    final_tracks[track_id].append((t + 1, to_label))
                    next_active_track_for_label[to_label] = track_id
                    linked_to_labels.add(to_label)
            
            # Identify new cells in the next frame that weren't linked from the current one
            if t + 1 < self.num_frames:
                mask_t_plus_1 = masks[t+1]
                for label in range(1, int(mask_t_plus_1.max()) + 1):
                    if label not in linked_to_labels:
                        # This is a new cell, start a new track
                        track_id = len(final_tracks)
                        final_tracks.append([(t + 1, label)])
                        next_active_track_for_label[label] = track_id
            
            active_track_for_label = next_active_track_for_label

        return final_tracks, masks


if __name__ == "__main__":
    ms2 = MS2(czi_path='/home/dafei/data/MS2/New-03_I.czi',
              fluorescence_th=30,
              device=torch.device('cuda:0'),
              flow_threshold=0.4,
              cellprob_threshold=0.0,
              tile_norm_blocksize=128)
    z = 0
    tracks, masks = ms2.cell_tracking(z=0)
    print(f"Found {len(tracks)} cell tracks across {ms2.num_frames} frames.")
    if tracks:
        print(f"Example track 0: {tracks[0]}")
