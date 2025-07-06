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
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output files')
    parser.add_argument('--z', type=int, default=0, help='Z-slice to process (default: 0)')
    parser.add_argument('--dice_threshold', type=float, default=0.3, help='Dice threshold for cell tracking')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    ms2 = MS2(czi_path=args.czi_path,
              fluorescence_th=args.fluorescence_th,
              device=torch.device(args.device),
              flow_threshold=args.flow_threshold,
              cellprob_threshold=args.cellprob_threshold,
              tile_norm_blocksize=args.tile_norm_blocksize)
    z = args.z
    print(f"Processing Z-slice: {z}")
    cell_images = ms2.get_cell_images_over_time(z=z)
    start_time = time.time()
    masks = ms2.segment_cells_images(cell_images)
    end_time = time.time()
    print(f"Cell segmentation computed in {end_time - start_time:.2f} seconds")
    start_time = time.time()
    tracked_cells = ms2.cell_tracking(masks, dice_threshold=args.dice_threshold)
    end_time = time.time()
    print(f"Cell tracking computed in {end_time - start_time:.2f} seconds")
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f'tracked_cells_z_projetion.npz')
    np.savez_compressed(save_path, tracked_cells=tracked_cells)
    print(f"Tracked cells saved to: {save_path}")