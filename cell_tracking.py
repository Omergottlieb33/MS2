import re
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.track import create_tracklets, match_cells_by_iou_hungarian_local_optimized

def extract_time_number(filename):
    """Extract time number from filename like z_stack_t5_seg_masks.npz"""
    try:
        # Find the pattern t{number}_
        match = re.search(r't(\d+)_', filename)
        return int(match.group(1)) if match else 0
    except:
        return 0

def get_masks_paths(masks_dir:str) -> list:
    masks_paths = sorted([os.path.join(masks_dir, x) for x in os.listdir(masks_dir) 
                if x.startswith('z_stack_t') and x.endswith('_seg_masks.npz')], 
               key=extract_time_number)
    if masks_paths is None or len(masks_paths) == 0:
        raise ValueError(f'No masks found in {masks_dir} with expected naming convention.')
    return masks_paths

def get_num_frames(masks_dir: str) -> int:
    """Return number of frames in recording, validating that mask time indices are contiguous 0..T-1."""
    masks_paths = get_masks_paths(masks_dir)
    times = [extract_time_number(os.path.basename(p)) for p in masks_paths]
    t = len(masks_paths)
    if times != list(range(t)):
        missing = sorted(set(range(max(times) + 1)) - set(times))
        raise ValueError(f'Mask time indices in {masks_dir} are not contiguous 0..{t - 1}. '
                         f'Found {t} masks, max t={max(times)}, missing t={missing}')
    return t

def match_over_time_cell_iou(masks: list, min_iou: float = 0.0,
                             max_centroid_distance: float = 15,
                             use_2d_distance: bool = False) -> list:
    """
    Match cells over time using IoU.
    Args:
        masks (list): List of segmentation masks for each time point.
        min_iou: Minimum IoU to accept a match.
        max_centroid_distance: Max XY (or 3D) centroid distance to consider a pair.
        use_2d_distance: Opt in to XY-only distance; default uses scaled 3D distance.
    Returns:
        list: List of matched cells for each time point.
    """
    matched_cells = []
    for i in tqdm(range(len(masks) - 1), desc='matching cells by IoU'):
        matches = match_cells_by_iou_hungarian_local_optimized(
            masks[i], masks[i + 1],
            min_iou=min_iou,
            max_centroid_distance=max_centroid_distance,
            use_2d_distance=use_2d_distance,
        )
        matched_cells.append(matches)
    return matched_cells

def compute_skip_matches(masks: list, min_iou: float = 0.0,
                         max_centroid_distance: float = 15,
                         use_2d_distance: bool = False) -> list:
    """
    Compute IoU matches between frames separated by 2 time points.
    skip_matches[k] maps {frame[k+2]_label: frame[k]_label}.
    Used for second-chance matching in create_tracklets.
    """
    skip_matches = []
    for i in tqdm(range(len(masks) - 2), desc='computing skip-frame matches'):
        matches = match_cells_by_iou_hungarian_local_optimized(
            masks[i], masks[i + 2],
            min_iou=min_iou,
            max_centroid_distance=max_centroid_distance * np.sqrt(2),
            use_2d_distance=use_2d_distance,
        )
        skip_matches.append(matches)
    return skip_matches



def cell_tracking(masks_dir: str, t: int = None, output_path: str = None) -> dict:
    """Track up to t consecutive frames; None/nonpositive means all frames.

    Missing observations are gaps, not biological deaths. Fail on missing files
    or unreadable masks instead of silently changing the time axis.
    """
    paths = get_masks_paths(masks_dir)
    if t is not None and t > 0:
        paths = paths[:t]
    times = [extract_time_number(os.path.basename(p)) for p in paths]
    if times != list(range(len(times))):
        raise ValueError('Masks must start at t0 and contain every requested frame')
    masks = []
    for path in tqdm(paths, desc='loading masks'):
        with np.load(path, allow_pickle=True) as f:
            masks.append(f['masks'])
    matched = match_over_time_cell_iou(masks)
    # Recovery is computed against only missing sources/free targets by the builder.
    tracklets = create_tracklets(matched, skip_matches=[], masks=masks)
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.normpath(masks_dir)),
                                   'tracklets_v3.json')
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(tracklets, f, indent=4)
    print(f'Tracklets saved to {output_path}')
    return tracklets


def evaluate_tracklets(tracklets: dict) -> pd.DataFrame:
    """
    Compute per-track quality statistics for a tracklets dict.

    Sentinel values in tracklet lists:
        positive int — cell label at that frame
        -1           — cell not detected / failed to match
        -2           — cell exited the field of view (border exit)

    Returns:
        pd.DataFrame indexed by tracklet_id with columns:
            n_frames_total, birth_frame, death_frame, span,
            n_active, active_fraction, n_gaps, n_resurrections, border_exit
    """
    rows = []
    for tid, labels in tracklets.items():
        arr = np.array(labels, dtype=int)
        active = np.where(arr > 0)[0]   # positive labels only
        if len(active) == 0:
            continue
        birth, death = int(active[0]), int(active[-1])
        inner = arr[birth: death + 1]
        n_active = int(np.sum(inner > 0))
        n_gaps = int(np.sum(inner == -1))   # -1 gaps (not border exits)
        n_resurrections = int(np.sum((inner[:-1] == -1) & (inner[1:] > 0))) if len(inner) > 1 else 0
        border_exit = bool(arr[death + 1] == -2) if death + 1 < len(arr) else False
        span = death - birth + 1
        rows.append({
            'tracklet_id': tid,
            'n_frames_total': len(arr),
            'birth_frame': birth,
            'death_frame': death,
            'span': span,
            'n_active': n_active,
            'active_fraction': n_active / span,
            'n_gaps': n_gaps,
            'n_resurrections': n_resurrections,
            'border_exit': border_exit,
        })

    if not rows:
        print('No detected tracks to evaluate.')
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index('tracklet_id')

    total_active = df['n_active'].sum()
    total_gaps = df['n_gaps'].sum()
    fragmentation = total_gaps / total_active if total_active > 0 else 0.0

    print(f"=== Tracklet Evaluation ===")
    print(f"  Total tracks          : {len(df)}")
    print(f"  Mean active frames    : {df['n_active'].mean():.1f}  (median {df['n_active'].median():.1f})")
    print(f"  Mean active fraction  : {df['active_fraction'].mean():.3f}")
    print(f"  Tracks with gaps      : {(df['n_gaps'] > 0).sum()}  ({(df['n_gaps'] > 0).mean() * 100:.1f}%)")
    print(f"  Total gap frames      : {total_gaps}")
    print(f"  Fragmentation index   : {fragmentation:.4f}  (gaps / active frames)")
    print(f"  Resurrected tracks    : {(df['n_resurrections'] > 0).sum()}")
    print(f"  Border exits          : {df['border_exit'].sum()}")
    return df


if __name__ == "__main__":
    masks_dir = "/zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-13/New-01-ST13-V/New-01-v/masks/"
    t = get_num_frames(masks_dir)  # Number of time points to process, set to None to process all available masks
    tracklets = cell_tracking(masks_dir, t=t)
    print(f"Processed {t} frames, obtained {len(tracklets)} tracklets.")
    