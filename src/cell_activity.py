"""Activity level of every cell inside one developmental-stage window of a recording.

The module form of notebooks/gene_expression_analysis_v2.ipynb, restricted to the frames
t_start..t_end (0-based, both included) so that embryos can be compared stage by stage --
src/cell_activity_compare.py builds on the functions here.  A cell's activity is its MS2
ellipse sum above the noise floor, and the levels form a ladder:

    0 silent                  never above the noise floor in the window
    1 short burst / noise     peak never clears NOISE_PEAK above the floor: 1-2 frame blips
    2-4 weak/moderate/high    KMeans on log total activity (it spans ~3 orders of magnitude),
                              renumbered so the level rises with activity

Recordings are described by a dict, the same one src/cell_activity_compare.py takes, stored
as json for the command line (noise_floor and min_presence are optional):

    {"New-02-ST11-12": {"csv": ".../gene_expression_results_fixed.csv",
                        "tracklets": ".../masks/tracklets_bug_fix2.json",
                        "masks_dir": ".../masks", "t_start": 0, "t_end": 60,
                        "noise_floor": 20, "min_presence": 0.5}}

    python -m src.cell_activity --config recordings.json --out-dir .../stage_activity

Point csv at gene_expression_results_fixed.csv (src/gene_expression/expression_matrix.py):
the original matrix holds most traces at the wrong timepoints, which a window then cuts wrongly.
"""
import argparse
import collections
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.track_diagnostics import load_masks, mask_paths_by_t
from src.utils.gif_utils import create_gif_from_figures

# Raw ellipse-sum counts; a frame at or below this is background.  The per-cell smallest
# non-zero value averages 22 in New-02.
NOISE_FLOOR = 20.0
# Counts above the floor.  Cells whose peak never clears this are 1-2 frame blips.
NOISE_PEAK = 10.0
# Levels the KMeans splits the cells above the blips into.
N_LEVELS = 3
# Fraction of the window's frames a cell must be tracked in to count.  Tracklet fragments
# would otherwise pad the silent level; 0 keeps every cell, as the notebook did.
MIN_PRESENCE = 0.5

LEVEL_NAMES = {0: 'silent', 1: 'short burst / noise', 2: 'weakly active',
               3: 'moderate active', 4: 'high active'}
LEVEL_COLORS = {0: (64, 64, 64), 1: (255, 0, 0), 2: (255, 165, 0),
                3: (255, 255, 0), 4: (255, 255, 255)}


def load_window(csv_path, t_start, t_end, min_presence=MIN_PRESENCE):
    """The cells tracked through enough of t_start..t_end (both included).

    Returns signals (cells x frames, untracked frames as 0 like the notebook), present (True
    where the cell is tracked), the cell column names and the window's timepoints.
    """
    df = pd.read_csv(csv_path).set_index('timepoint').sort_index()
    if not df.index.min() <= t_start <= t_end <= df.index.max():
        raise ValueError(f'window {t_start}..{t_end} is not inside {csv_path} '
                         f'({df.index.min()}..{df.index.max()})')
    window = df.loc[t_start:t_end]
    present = window.notna().to_numpy().T
    keep = (present.mean(axis=1) >= min_presence) & present.any(axis=1)
    return (window.fillna(0.0).to_numpy().T[keep], present[keep],
            window.columns[keep].tolist(), window.index.to_numpy())


def activity_descriptors(signals, present, noise_floor=NOISE_FLOOR):
    """Per-cell activity above the noise floor, rows in the order of signals.

    total, peak, frames   summed and maximum counts above the floor, frames above it
    rate, duty            total and frames per tracked frame, comparable across cells and
                          windows of different length
    onset                 first frame above the floor as a fraction of the window, 0..1
    """
    above = np.clip(signals - noise_floor, 0.0, None)
    fired = above > 0
    tracked = present.sum(axis=1)
    frames = fired.sum(axis=1)
    return pd.DataFrame({
        'total': above.sum(axis=1),
        'peak': above.max(axis=1),
        'frames': frames,
        'rate': above.sum(axis=1) / tracked,
        'duty': frames / tracked,
        'onset': np.where(frames > 0, fired.argmax(axis=1) / max(signals.shape[1] - 1, 1), np.nan),
    })


def order_labels_by_activity(labels, signals):
    """Renumber cluster ids so 0 = lowest mean signal ... k-1 = highest."""
    uniq = np.unique(labels)
    means = [signals[labels == k].mean() for k in uniq]
    remap = {old: new for new, old in enumerate(uniq[np.argsort(means)])}
    return np.array([remap[l] for l in labels])


def activity_levels(score, n_levels=N_LEVELS, seed=0):
    """Levels 0..k-1 of the scores: KMeans on log1p(score), renumbered lowest to highest."""
    score = np.log1p(np.asarray(score, dtype=float))
    k = min(n_levels, len(np.unique(score)))
    if k < n_levels:
        print(f'warning: {len(score)} cells with {k} distinct activities, '
              f'fitting {k} levels instead of {n_levels}')
    if k == 0:
        return np.zeros(0, dtype=int)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(score.reshape(-1, 1))
    return order_labels_by_activity(labels, score)


def classify_cells(desc, score='total', noise_peak=NOISE_PEAK, n_levels=N_LEVELS):
    """LEVEL_NAMES level of every row of desc, the ladder fitted on desc[score]."""
    levels = np.where(desc['peak'] > 0, 1, 0)
    ranked = (desc['peak'] >= noise_peak).to_numpy()
    levels[ranked] = 2 + activity_levels(desc[score].to_numpy()[ranked], n_levels)
    return levels


def normalise_signals(signals, noise_floor=NOISE_FLOOR):
    """Counts above the floor scaled by the recording's maximum, as the notebook fed PCA."""
    above = np.clip(signals - noise_floor, 0.0, None)
    return (above - above.min()) / above.max()


def save_figure(fig, path_stem, eps=True):
    fig.savefig(path_stem + '.png', dpi=150, bbox_inches='tight')
    if eps:
        fig.savefig(path_stem + '.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_clustered_heatmap(signals, levels, timepoints, title, path_stem):
    """Rows grouped by level, lowest first, with red lines between levels."""
    order = np.argsort(levels, kind='stable')
    fig, ax = plt.subplots(figsize=(20, 20))
    # 'none' keeps the eps at one pixel per cell and frame instead of a 110 MB resampled raster
    im = ax.imshow(signals[order], aspect='auto', cmap='viridis', interpolation='none',
                   extent=(timepoints[0] - 0.5, timepoints[-1] + 0.5, len(order), 0))
    for boundary in np.flatnonzero(np.diff(levels[order])) + 1:
        ax.axhline(boundary, color='red', lw=2.0)
    fig.colorbar(im, ax=ax, label='Expression')
    ax.set_yticks([])
    ax.set(title=title, xlabel='Timepoint', ylabel='Cells (sorted by level)')
    save_figure(fig, path_stem)


def plot_activity_ladder(total, levels, path_stem):
    fig, ax = plt.subplots(figsize=(7, 4))
    for level in np.unique(levels):
        m = levels == level
        ax.scatter(np.flatnonzero(m), total[m], s=18, label=LEVEL_NAMES[level])
    ax.set_yscale('log')
    ax.set(xlabel='active cell index', ylabel='total activity (counts above baseline)',
           title='Activity ladder over the active cells')
    ax.legend(fontsize=7)
    save_figure(fig, path_stem, eps=False)


def plot_pca(normalised, levels, timepoints, out_dir):
    """Diagnostic only: PCA encodes *when* a cell fired, which the activity ladder ignores."""
    if len(normalised) < 2:
        print('fewer than 2 active cells, skipping PCA')
        return
    pca = PCA(n_components=None, svd_solver='full')  # 'mle' invalid: n_samples < n_features
    Z = pca.fit_transform(normalised)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    for i, ax in enumerate(axes.ravel()):
        if i < len(pca.components_):
            ax.plot(timepoints, pca.components_[i])
            ax.set_title(f'PCA {i + 1} Component')
            ax.grid(True)
        else:
            ax.axis('off')
    fig.tight_layout()
    save_figure(fig, os.path.join(out_dir, 'pca_components'), eps=False)

    explained = pca.explained_variance_ratio_
    fig, ax = plt.subplots()
    ax.scatter(np.arange(1, len(explained) + 1), explained, marker='o', s=4)
    ax.set(xlabel='PC', ylabel='Explained variance ratio', title='Scree')
    save_figure(fig, os.path.join(out_dir, 'pca_scree'), eps=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    for level in np.unique(levels):
        m = levels == level
        ax.scatter(Z[m, 0], Z[m, 1], s=12, c=[np.array(LEVEL_COLORS[level]) / 255],
                   label=LEVEL_NAMES[level], edgecolors='black', linewidths=0.4)
    ax.set(title='PCA of temporal shape (colour = activity level)', xlabel='PC1', ylabel='PC2')
    ax.legend(title='Clusters', fontsize=6)
    save_figure(fig, os.path.join(out_dir, 'pca_scatter'), eps=False)


def cluster_overlay(masks, labels_by_level, alpha=0.6):
    """RGB max projection of the (Z,H,W) masks, each level's labels painted in its colour
    over black."""
    img = np.zeros(masks.shape[1:] + (3,), dtype=np.float32)
    for level in sorted(labels_by_level):
        painted = np.isin(masks, labels_by_level[level]).any(axis=0)
        img[painted] = alpha * np.array(LEVEL_COLORS[level], dtype=np.float32)
    return img.astype(np.uint8)


def render_overlay_gif(masks_dir, tracklets, cells, levels, signals, timepoints, noise_floor,
                       out_dir):
    """cluster_overlay.gif over the window.  A cell wears its level's colour on the frames its
    signal clears the noise floor and silent grey on the others; the first frame is also
    saved as a still."""
    paths = mask_paths_by_t(masks_dir)
    track_ids = [c[len('cell_'):] for c in cells]
    # Fixed legend: which levels light up changes from frame to frame
    legend = [Patch(facecolor=np.array(LEVEL_COLORS[l]) / 255, edgecolor='black',
                    label=LEVEL_NAMES[l]) for l in LEVEL_NAMES]
    figures = []
    for i, t in enumerate(timepoints):
        labels_by_level = collections.defaultdict(list)
        for tid, level, firing in zip(track_ids, levels, signals[:, i] > noise_floor):
            label = tracklets[tid][t]
            if label > 0:
                labels_by_level[level if firing else 0].append(label)
        img = cluster_overlay(load_masks(paths[t]), labels_by_level)

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(np.rot90(img, -1))
        plt.axis('off')
        plt.title(f'Cluster at timepoint {t}')
        plt.legend(handles=legend, loc='upper right', fontsize=6)
        plt.tight_layout()
        if i == 0:
            still = os.path.join(out_dir, f'cluster_overlay_t{t}')
            fig.savefig(still + '.png', dpi=300)
            fig.savefig(still + '.eps', format='eps', bbox_inches='tight', dpi=300)
        figures.append(fig)
    create_gif_from_figures(figures, os.path.join(out_dir, 'cluster_overlay.gif'), fps=1)


def analyze_recording(name, rec, out_dir):
    """Everything the notebook produced, for one recording dict (see the module docstring)
    and its window.  Writes into out_dir and returns the per-cell table."""
    os.makedirs(out_dir, exist_ok=True)
    noise_floor = rec.get('noise_floor', NOISE_FLOOR)
    signals, present, cells, timepoints = load_window(
        rec['csv'], rec['t_start'], rec['t_end'], rec.get('min_presence', MIN_PRESENCE))
    desc = activity_descriptors(signals, present, noise_floor)
    levels = classify_cells(desc)

    table = pd.concat([pd.DataFrame({'cell': cells, 'cluster': levels,
                                     'cluster_name': [LEVEL_NAMES[l] for l in levels]}),
                       desc], axis=1)
    table.sort_values('cluster', kind='stable').to_csv(
        os.path.join(out_dir, 'clusters.csv'), index=False)
    window = f't={timepoints[0]}..{timepoints[-1]}'
    counts = ', '.join(f'{LEVEL_NAMES[l]} {n}' for l, n in zip(*np.unique(levels, return_counts=True)))
    print(f'{name} {window}: {len(cells)} cells -- {counts}')

    shown = np.maximum(signals, noise_floor)  # background at the floor, as the notebook drew it
    plot_clustered_heatmap(shown, levels, timepoints, f'{name} {window}: all cells',
                           os.path.join(out_dir, 'heatmap_all'))
    active = levels > 0
    if active.any():
        plot_clustered_heatmap(shown[active], levels[active], timepoints,
                               f'{name} {window}: active cells',
                               os.path.join(out_dir, 'heatmap_active'))
        plot_activity_ladder(desc['total'].to_numpy()[active], levels[active],
                             os.path.join(out_dir, 'activity_ladder'))
        plot_pca(normalise_signals(signals, noise_floor)[active], levels[active], timepoints,
                 out_dir)

    with open(rec['tracklets']) as f:
        tracklets = json.load(f)
    render_overlay_gif(rec['masks_dir'], tracklets, cells, levels, signals, timepoints,
                       noise_floor, out_dir)
    return table


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='json of {name: recording dict}')
    p.add_argument('--out-dir', required=True, help='one sub-folder per recording is written here')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        recordings = json.load(f)
    for name, rec in recordings.items():
        analyze_recording(name, rec, os.path.join(args.out_dir, name))


if __name__ == '__main__':
    main()
