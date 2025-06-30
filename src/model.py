import numpy as np
from cellpose import models, core, plot
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from cellpose.utils import masks_to_outlines

from src.utils import load_czi_images, get_dice_coefficient


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
    
    def segment_cells_over_time(self, z, tn=None):
        if tn is None:
            tn = self.num_frames
        imgs = [self.image_data[0, z, self.microscope_channels[1], t, :, :, 0] for t in tqdm(range(tn), desc="Collecting images")]
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
    
    @staticmethod
    def initiate_cell_tracks(masks):
        flag = True
        i=0
        cell_tracks = {}
        next_cell_id = 1
        while flag:
            mask_0 = masks[0]
            max_label_0 = int(mask_0.max())
            for label in range(1, max_label_0 + 1):
                cell_mask = (mask_0 == label).astype(np.uint8)
                if sum(cell_mask) == 0:
                    continue
                cell_tracks[next_cell_id] = [cell_mask]
                next_cell_id += 1
            if next_cell_id == 1:
                flag = True
                i+= 1
            elif next_cell_id > 1:
                flag = False
                break
        return cell_tracks, i, next_cell_id
    
    @staticmethod
    def get_cell_label_by_dice_coefficient(binary_mask, mask, dice_threshold=0.3):
        """
        Vectorized version to find the best matching cell label using dice coefficient.
        
        Args:
            binary_mask (np.ndarray): Binary mask of the cell to match
            mask (np.ndarray): Segmentation mask with labeled cells
            dice_threshold (float): Minimum dice coefficient threshold
            
        Returns:
            tuple: (best_label, best_dice) or (None, 0.0) if no match above threshold
        """
        max_label = int(mask.max())
        if max_label == 0:
            return None, 0.0
        
        # Convert binary_mask to boolean for faster operations
        binary_mask_bool = binary_mask.astype(bool)
        binary_mask_sum = np.sum(binary_mask_bool)
        
        if binary_mask_sum == 0:
            return None, 0.0
        
        # Create a 3D array where each slice is a binary mask for each label
        # Shape: (num_labels, height, width)
        labels = np.arange(1, max_label + 1)
        label_masks = mask[None, :, :] == labels[:, None, None]  # Broadcasting magic
        
        # Calculate intersections for all labels at once
        # Intersection = binary_mask & each_label_mask
        intersections = label_masks & binary_mask_bool[None, :, :]
        intersection_sums = np.sum(intersections, axis=(1, 2))  # Sum over height and width
        
        # Calculate sums for each label mask
        label_sums = np.sum(label_masks, axis=(1, 2))
        
        # Vectorized dice coefficient calculation
        # dice = 2 * intersection / (sum1 + sum2)
        # Add small epsilon to avoid division by zero
        epsilon = 1e-7
        total_sums = binary_mask_sum + label_sums
        dice_coeffs = (2.0 * intersection_sums + epsilon) / (total_sums + epsilon)
        
        # Find labels that meet the threshold
        valid_indices = dice_coeffs >= dice_threshold
        
        if not np.any(valid_indices):
            return None, 0.0
        
        # Find the best match among valid ones
        valid_dice = dice_coeffs[valid_indices]
        best_idx_in_valid = np.argmax(valid_dice)
        
        # Convert back to original label space
        valid_labels = labels[valid_indices]
        best_label = valid_labels[best_idx_in_valid]
        best_dice = valid_dice[best_idx_in_valid]
        
        return int(best_label), float(best_dice)
    
    def cell_tracking_2p1(self, z, dice_threshold=0.3):
        masks = self.segment_cells_over_time(z)
        # initiate tracking
        height, width = masks[0].shape
        cell_tracks, i, max_cell_id = self.initiate_cell_tracks(masks)
        # Track cells through subsequent frames
        for t in tqdm(range(i+1, self.num_frames), desc="Tracking cells over time"):
            mask_t = masks[t]
             # Find the best matches for cells in current frame
            for cell_id, track in cell_tracks.items():
                tracked_mask_t_minus_1 = track[t-1]  # Get the cell mask at current time
                label, dice_coeff = self.get_cell_label_by_dice_coefficient(tracked_mask_t_minus_1, mask_t, dice_threshold)
                if label is not None and dice_coeff >= dice_threshold:
                    # Found a match, extend the track
                    cell_mask_t = (mask_t == label).astype(np.uint8)
                    track.append(cell_mask_t) 
                else:
                    # No match, add an empty mask
                    track.append(np.zeros((height, width), dtype=np.uint8))
                    # If no match, we can also add a new cell mask
                    track[max_cell_id+1] = cell_mask_t
                    max_cell_id += 1
        return cell_tracks



    
    def cell_tracking_v2(self, z, dice_threshold=0.3):
        """
        Track cells over time using dice coefficient matching.
        
        Args:
            z (int): Z-stack index to process
            dice_threshold (float): Minimum dice coefficient for a valid match
            
        Returns:
            list: List of 3D tensors (num_frames x height x width) for each tracked cell.
                  Each tensor contains the cell mask at each timepoint, or zeros if no match.
        """
        masks = self.segment_cells_over_time(z)
        
        # Get image dimensions
        height, width = masks[0].shape
        
        # Dictionary to store tracks: {cell_id: [mask_at_t0, mask_at_t1, ...]}
        cell_tracks = {}
        next_cell_id = 1
        
        # Initialize tracks with cells from the first frame
        mask_0 = masks[0]
        max_label_0 = int(mask_0.max())
        for label in range(1, max_label_0 + 1):
            cell_mask = (mask_0 == label).astype(np.uint8)
            cell_tracks[next_cell_id] = [cell_mask]
            next_cell_id += 1
        
        # Track cells through subsequent frames
        for t in tqdm(range(self.num_frames - 1), desc="Tracking cells over time"):
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
                if len(track) == t + 1:  # This track is active (has data up to current frame)
                    current_cell_mask = track[t]  # Get the cell mask at current time
                    
                    # Find corresponding label in mask_t
                    current_label = None
                    for label in range(1, max_label1 + 1):
                        label_mask = (mask_t == label).astype(np.uint8)
                        if np.array_equal(current_cell_mask, label_mask):
                            current_label = label
                            break
                    
                    if current_label is None:
                        # Cell not found in current frame, add empty mask
                        cell_tracks[cell_id].append(np.zeros((height, width), dtype=np.uint8))
                        continue
                    
                    # Find best match in next frame
                    best_dice_coeff = 0
                    best_match_label = None
                    
                    for next_label in range(1, max_label2 + 1):
                        if next_label in matched_labels_t_plus_1:
                            continue  # This label is already matched
                            
                        cell_mask_t_plus_1 = (mask_t_plus_1 == next_label).astype(np.uint8)
                        dice_coeff = get_dice_coefficient(current_cell_mask, cell_mask_t_plus_1)
                        
                        if dice_coeff > best_dice_coeff and dice_coeff >= dice_threshold:
                            best_dice_coeff = dice_coeff
                            best_match_label = next_label
                    
                    if best_match_label is not None:
                        # Found a good match
                        best_match_mask = (mask_t_plus_1 == best_match_label).astype(np.uint8)
                        tracks_to_extend[cell_id] = best_match_mask
                        matched_labels_t_plus_1.add(best_match_label)
                    else:
                        # No good match found, add empty mask
                        tracks_to_extend[cell_id] = np.zeros((height, width), dtype=np.uint8)
            
            # Extend existing tracks
            for cell_id, next_mask in tracks_to_extend.items():
                cell_tracks[cell_id].append(next_mask)
            
            # Add new cells that weren't matched (new cell divisions or cells entering frame)
            for label in range(1, max_label2 + 1):
                if label not in matched_labels_t_plus_1:
                    # This is a new cell, start a new track
                    new_cell_mask = (mask_t_plus_1 == label).astype(np.uint8)
                    # Fill previous timepoints with empty masks
                    new_track = [np.zeros((height, width), dtype=np.uint8) for _ in range(t + 1)]
                    new_track.append(new_cell_mask)
                    cell_tracks[next_cell_id] = new_track
                    next_cell_id += 1
        
        # Convert tracks to 3D tensors and return as list
        tracked_cells = []
        for cell_id in sorted(cell_tracks.keys()):
            track = cell_tracks[cell_id]
            # Ensure all tracks have the same length (some might be shorter if cell appeared later)
            while len(track) < self.num_frames:
                track.append(np.zeros((height, width), dtype=np.uint8))
            
            # Convert to 3D tensor (num_frames x height x width)
            cell_tensor = np.stack(track, axis=0)
            tracked_cells.append(cell_tensor)
        
        return tracked_cells



if __name__ == "__main__":
    ms2 = MS2(czi_path='/home/dafei/data/MS2/New-03_I.czi',
              fluorescence_th=30,
              device=torch.device('cuda:0'),
              flow_threshold=0.4,
              cellprob_threshold=0.0,
              tile_norm_blocksize=128)
    z = 0
    
    # Use the new cell tracking function
    tracked_cells = ms2.cell_tracking_v2(z=0, dice_threshold=0.3)
    print(f"Found {len(tracked_cells)} tracked cells across {ms2.num_frames} frames.")
    
    # Print information about each tracked cell
    for i, cell_tensor in enumerate(tracked_cells):
        frames_with_cell = np.sum(np.sum(cell_tensor, axis=(1, 2)) > 0)
        print(f"Cell {i+1}: Present in {frames_with_cell}/{ms2.num_frames} frames, shape: {cell_tensor.shape}")
    
    # Optional: Also run the original tracking for comparison
    tracks, masks = ms2.cell_tracking(z=0)
    print(f"\nOriginal tracking found {len(tracks)} cell tracks.")
    if tracks:
        print(f"Example track 0: {tracks[0]}")
