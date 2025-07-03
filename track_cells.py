import os
import argparse
import time
import numpy as np
import torch
from src.model import MS2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--czi_path', type=str, required=True, help='Path to the CZI file')
    parser.add_argument('--fluorescence_th', type=int, default=30, help='Fluorescence threshold for segmentation')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run the model on (e.g., cuda:0, cpu)')
    parser.add_argument('--flow_threshold', type=float, default=0.4, help='Flow threshold for segmentation')
    parser.add_argument('--cellprob_threshold', type=float, default=0.0, help='Cell probability threshold for segmentation')
    parser.add_argument('--tile_norm_blocksize', type=int, default=128, help='Tile normalization block size')
    return parser.parse_args()

if __name__ == "__main__":
    ms2 = MS2(czi_path='/home/dafei/data/MS2/New-03_I.czi',
              fluorescence_th=30,
              device=torch.device('cuda:0'),
              flow_threshold=0.4,
              cellprob_threshold=0.0,
              tile_norm_blocksize=128)
    z = 0
    cell_images = ms2.get_cell_images_over_time(z=z)
    start_time = time.time()
    masks = ms2.segment_cells_images(cell_images)
    end_time = time.time()
    print(f"Cell segmentation computed in {end_time - start_time:.2f} seconds")
    start_time = time.time()
    tracked_cells = ms2.cell_tracking(masks, dice_threshold=0.3)
    end_time = time.time()
    print(f"Cell tracking computed in {end_time - start_time:.2f} seconds")
    output_dir = '/home/dafei/output/MS2/tracked_cells'
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'tracked_cells_z_projetion.npz')
    np.savez_compressed(save_path, tracked_cells=tracked_cells)
    print(f"Tracked cells saved to: {save_path}")