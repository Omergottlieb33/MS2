import os
import torch
import numpy as np
from tqdm import tqdm
from cellpose import models
from src.utils.image_utils import load_czi_images


def segment_3d_cells(input, output_dir,device):
    if not input.endswith('.czi'):
        raise ValueError("Input file must be a .czi file")
    file_name = os.path.basename(input).replace('.czi', '')
    save_dir = os.path.join(output_dir, file_name)
    masks_dir = os.path.join(save_dir, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    image_data = load_czi_images(input)
    t = np.linspace(0, image_data.shape[0]-1, image_data.shape[0], dtype=int)
    torch_device = torch.device(device)
    model = models.CellposeModel(gpu=True, device=torch_device)
    for ti in tqdm(t):
        save_path = os.path.join(masks_dir, f'z_stack_t{ti}_seg_masks.npz')
        if os.path.exists(save_path):
            print(f"Skipping {save_path}, already exists.")
            continue
        z_stack_t = image_data[ti, 1, :, :, :]
        masks, flows, _ = model.eval(
            z_stack_t, z_axis=0, channel_axis=1, batch_size=32, do_3D=True, flow3D_smooth=1)
        
        # Save compressed numpy array
        np.savez_compressed(save_path, masks=masks, flows_xyz_coord=flows[1], flows_circular_coord=flows[0])


if __name__ == "__main__":
    czi_file_path = "/zjbd/zd1/shechtmanlab/omer/MS2/data/020626/STAGE-14/New-15-ST14-V1.czi"
    output_dir = '/zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-14/New-15-ST14-V1'
    device = 'cuda:3'
    segment_3d_cells(czi_file_path, output_dir, device)
