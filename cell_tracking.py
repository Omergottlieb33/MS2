import re
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from src.track import get_cell_centers, compute_cell_location, match_points_between_frames, \
create_tracklets, match_cells_by_iou_hungarian_local_optimized

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

def get_adjaceny_graphs(masks:list, t:int) -> tuple:
    centers_list, g_list = [], [], []
    # get cell centers and adjacency graphs for each time point
    for i in tqdm(range(t),desc='calculating cell centers and adjacency graphs'):
        z_stack_seg_mask = masks[i]
        centers = get_cell_centers(z_stack_seg_mask)
        labels = np.unique(z_stack_seg_mask)
        g = compute_cell_location(centers=centers, labels=labels)
        centers_list.append(centers)
        g_list.append(g)
    return centers_list, g_list

def match_points_over_time_adjacency_graph(g_list:list, masks:list, t:int, distance_threshold=np.sqrt(3), degree_weight=0.3):
    matched_points = []
    for i in tqdm(range(t - 1), desc='matching points between frames'):
        g1 = g_list[i]
        g2 = g_list[i + 1]
        z_stack_seg_mask_t0 = masks[i]
        z_stack_seg_mask_t1 = masks[i + 1]
        matches = match_points_between_frames(
            g1, g2, z_stack_seg_mask_t0, z_stack_seg_mask_t1)
        matched_points.append(matches)
    return matched_points

def match_over_time_cell_iou(masks: list, min_iou: float = 0.3,
                             max_centroid_distance: float = 15,
                             use_2d_distance: bool = True) -> list:
    """
    Match cells over time using IoU.
    Args:
        masks (list): List of segmentation masks for each time point.
        min_iou: Minimum IoU to accept a match.
        max_centroid_distance: Max XY (or 3D) centroid distance to consider a pair.
        use_2d_distance: Use only Y, X for distance (recommended for 3D microscopy).
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

def compute_skip_matches(masks: list, min_iou: float = 0.3,
                         max_centroid_distance: float = 15,
                         use_2d_distance: bool = True) -> list:
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
            max_centroid_distance=max_centroid_distance,
            use_2d_distance=use_2d_distance,
        )
        skip_matches.append(matches)
    return skip_matches



def cell_tracking(masks_dir:str, t:int) -> dict:
    """
    Perform cell tracking using adjacency graphs.
    Args:
        masks_dir (str): Directory containing segmentation masks.
        t (int): Number of time points to process. If None, processes all available masks.
        distance_threshold (float): Distance threshold for matching points between frames.
        degree_weight (float): Weight for degree similarity in matching score.
    Returns:
        dict: Tracklets mapping cell IDs across time points.
    """
    masks_paths = get_masks_paths(masks_dir)
    masks = []
    for i in range(len(masks_paths)):
        try:
            z_stack_seg_mask = np.load(masks_paths[i], allow_pickle=True)['masks']
            masks.append(z_stack_seg_mask)
        except Exception as e:
            print(f"Error loading {masks_paths[i]}: {e}")
    if t<=0 or t> len(masks_paths):
        t = len(masks_paths) - 1
    # IoU matching
    matched_points = match_over_time_cell_iou(masks)
    skip_matches = compute_skip_matches(masks)
    # create tracklets with second-chance matching; pass masks for border-exit detection
    tracklets = create_tracklets(matched_points, skip_matches=skip_matches, masks=masks)
    save_dir = os.path.dirname(masks_dir)
    save_path = os.path.join(save_dir, 'tracklets_bug_fix.json')
    with open(save_path, 'w') as f:
        json.dump(tracklets, f, indent=4)
    print(f'Tracklets saved to {save_path}')
    return tracklets

        

def evaluate_matches(matched_cells: list, masks: list, jump_threshold: float = 20.0,
                     frame_gap: int = 1) -> pd.DataFrame:
    """
    Evaluate pairwise IoU matches by computing centroid jump distances.

    For every matched pair (cell at frame t → cell at frame t+frame_gap), reports
    the 3D and XY centroid displacement.  Flags matches whose 3D displacement
    exceeds `jump_threshold` as suspicious.

    Args:
        matched_cells: list of dicts {cell_t+gap_label: cell_t_label}, one per frame pair.
                       Pass matched_cells (frame_gap=1) or skip_matches (frame_gap=2).
        masks:         list of 3D (Z, Y, X) segmentation masks.
        jump_threshold: 3D centroid distance (pixels) above which a match is flagged.
        frame_gap:     number of frames between the two masks being compared (1 for
                       consecutive matches, 2 for skip/second-chance matches).

    Returns:
        pd.DataFrame with columns:
            frame, cell_t, cell_t1, dist_3d, dist_xy, flagged
    """
    from scipy.ndimage import center_of_mass as _com

    rows = []
    for frame_idx, match_dict in enumerate(matched_cells):
        if not match_dict:
            continue
        mask_t  = masks[frame_idx]
        mask_t1 = masks[frame_idx + frame_gap]

        labels_t  = list({int(v) for v in match_dict.values()})
        labels_t1 = list({int(k) for k in match_dict.keys()})

        coms_t  = {int(l): c for l, c in zip(labels_t,  _com(mask_t,  mask_t,  labels_t))}
        coms_t1 = {int(l): c for l, c in zip(labels_t1, _com(mask_t1, mask_t1, labels_t1))}

        for cell_t1, cell_t in match_dict.items():
            c0 = np.array(coms_t.get(int(cell_t),  [np.nan] * 3))
            c1 = np.array(coms_t1.get(int(cell_t1), [np.nan] * 3))
            if np.isnan(c0).any() or np.isnan(c1).any():
                continue

            dist_3d = float(np.linalg.norm(c1 - c0))
            dist_xy = float(np.linalg.norm(c1[1:] - c0[1:]))   # Y, X only

            rows.append({
                'frame':   frame_idx,
                'cell_t':  int(cell_t),
                'cell_t1': int(cell_t1),
                'dist_3d': round(dist_3d, 2),
                'dist_xy': round(dist_xy, 2),
                'flagged': dist_3d > jump_threshold,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No matches to evaluate.")
        return df

    flagged = df['flagged']
    print(f"=== Match Evaluation ===")
    print(f"  Total matches          : {len(df)}")
    print(f"  Mean  3D jump (px)     : {df['dist_3d'].mean():.2f}")
    print(f"  Median 3D jump (px)    : {df['dist_3d'].median():.2f}")
    print(f"  Max   3D jump (px)     : {df['dist_3d'].max():.2f}")
    print(f"  Flagged (>{jump_threshold:.0f} px)   : {flagged.sum()} ({flagged.mean()*100:.1f}%)")
    if flagged.any():
        worst = df.loc[df['dist_3d'].idxmax()]
        print(f"  Worst match            : frame {int(worst.frame)}  "
              f"cell {int(worst.cell_t)} → {int(worst.cell_t1)}  "
              f"({worst.dist_3d:.1f} px 3D, {worst.dist_xy:.1f} px XY)")
    return df


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


def visualize_cell_track(tracklet_id, tracklets: dict, masks: list, pad: int = 20, max_cols: int = 10):
    """
    Visualize a single cell's tracking across all time points.

    Top row: timeline bar — green = active, red = internal gap, gray = outside span.
    Remaining rows: max-Z-projection thumbnails at each active time point,
                    cropped to the cell's union bounding box (+pad pixels).

    Args:
        tracklet_id: int or str key into tracklets dict.
        tracklets: tracking dict (keys may be int or str).
        masks: list of 3D (Z, Y, X) segmentation masks, one per time point.
        pad: pixel padding around the cell bounding box for thumbnails.
        max_cols: maximum thumbnails per row.

    Returns:
        matplotlib Figure
    """
    # Support both int and str keys (JSON serialization converts keys to str)
    key = str(tracklet_id) if str(tracklet_id) in tracklets else tracklet_id
    if key not in tracklets:
        raise KeyError(f"tracklet_id {tracklet_id} not found in tracklets.")

    labels = np.array(tracklets[key], dtype=int)
    T = len(labels)
    active_frames = [t for t in range(T) if labels[t] > 0]   # positive labels only

    if not active_frames:
        print(f"Track {tracklet_id} has no active frames.")
        return None

    birth, death = active_frames[0], active_frames[-1]
    n_active = len(active_frames)

    # --- Union bounding box across all active frames ---
    H, W = masks[0].shape[-2], masks[0].shape[-1]
    y_min_u, x_min_u = H, W
    y_max_u, x_max_u = 0, 0
    for t in active_frames:
        mask3d = masks[t]
        cell_2d = (mask3d == labels[t]).any(axis=0) if mask3d.ndim == 3 else (mask3d == labels[t])
        ys, xs = np.where(cell_2d)
        if len(ys) == 0:
            continue
        y_min_u = min(y_min_u, int(ys.min()))
        y_max_u = max(y_max_u, int(ys.max()))
        x_min_u = min(x_min_u, int(xs.min()))
        x_max_u = max(x_max_u, int(xs.max()))

    y_min_c = max(0, y_min_u - pad)
    y_max_c = min(H, y_max_u + pad + 1)
    x_min_c = max(0, x_min_u - pad)
    x_max_c = min(W, x_max_u + pad + 1)

    # --- Figure layout using GridSpec ---
    n_rows = max(1, (n_active + max_cols - 1) // max_cols)
    fig_w = max(min(n_active, max_cols) * 1.6 + 0.5, 6)
    fig_h = (n_rows + 1) * 2.2
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(n_rows + 1, max_cols, figure=fig, hspace=0.5, wspace=0.15)

    # --- Timeline bar ---
    # Colors: green=active, red=internal gap (-1), blue=border exit (-2), gray=outside span
    ax_tl = fig.add_subplot(gs[0, :])
    color_img = np.zeros((1, T, 3))
    for t in range(T):
        if labels[t] > 0:
            color_img[0, t] = [0.2, 0.8, 0.2]       # active: green
        elif labels[t] == -2:
            color_img[0, t] = [0.2, 0.4, 0.9]       # border exit: blue
        elif birth < t < death:
            color_img[0, t] = [0.9, 0.2, 0.2]       # internal gap: red
        else:
            color_img[0, t] = [0.75, 0.75, 0.75]    # outside span: gray

    n_gaps = int(np.sum(labels[birth:death + 1] == -1))
    border_exit = bool(death + 1 < T and labels[death + 1] == -2)
    ax_tl.imshow(color_img, aspect='auto', interpolation='nearest')
    ax_tl.set_yticks([])
    ax_tl.set_xlabel('Frame', fontsize=9)
    ax_tl.set_title(
        f'Track {tracklet_id}  |  active: {n_active}/{T} frames  |  '
        f'birth: {birth}  death: {death}  |  '
        f'gaps: {n_gaps}{"  | border exit" if border_exit else ""}',
        fontsize=9
    )
    tick_step = max(1, T // 10)
    ax_tl.set_xticks(range(0, T, tick_step))
    ax_tl.set_xticklabels(range(0, T, tick_step), fontsize=7)

    # --- Thumbnails ---
    for idx, t in enumerate(active_frames):
        row = 1 + idx // max_cols
        col = idx % max_cols
        ax = fig.add_subplot(gs[row, col])

        label = labels[t]
        mask3d = masks[t]
        if mask3d.ndim == 3:
            cell_2d = (mask3d == label).any(axis=0)
            other_2d = (mask3d > 0).any(axis=0)
        else:
            cell_2d = (mask3d == label)
            other_2d = (mask3d > 0)

        cell_crop = cell_2d[y_min_c:y_max_c, x_min_c:x_max_c]
        other_crop = other_2d[y_min_c:y_max_c, x_min_c:x_max_c] & ~cell_crop

        rgb = np.zeros((y_max_c - y_min_c, x_max_c - x_min_c, 3), dtype=float)
        rgb[other_crop] = [0.45, 0.45, 0.45]   # other cells: mid-gray
        rgb[cell_crop] = [0.15, 0.85, 0.15]    # this cell: green

        ax.imshow(rgb, interpolation='nearest')
        ax.set_title(f't={t}', fontsize=7)
        ax.axis('off')

    return fig


if __name__ == "__main__":
    masks_dir = "/zjbd/zd1/shechtmanlab/omer/MS2/outputs/020626/STAGE-11/New-02-v3-ST11-12/New-02-v3-ST11-12/masks/"
    t = 109  # Number of time points to process, set to None to process all available masks
    tracklets = cell_tracking(masks_dir, t=t)
    