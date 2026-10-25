"""Plot one tracklet's cell over time; moved from ms2/tracking.py (used by notebooks/3d_cell_tracking)."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


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


