"""Compare stage-window cell activity across any number of recordings.

Every recording is cut to its own developmental-stage window and measured with the
definitions of ms2/cell_activity.py, then the recordings are set side by side.  Two choices
make the numbers comparable between embryos:

  levels   weak / moderate / high come from ONE KMeans ladder fitted on the pooled cells of
           all recordings, on activity per tracked frame (rate) so windows of different length
           compare.  ms2/cell_activity.py fits a ladder per recording, where "high" in one
           embryo is a different threshold than in the next.  Silent and short burst / noise
           use the same absolute thresholds in both.  Active = weakly active or above.
           --ladder per_recording fits one ladder per recording instead, to see how much of a
           level difference is the shared thresholds.  --ladder balanced fits the one ladder
           with every recording weighted equally, so the recording with the most cells does
           not set the thresholds for the rest.
  space    the embryo is cut off by the field of view and mounted at any rotation, so every
           spatial measure is orientation-free and relative to the imaged tissue -- the cells
           tracked in the window -- not to the whole embryo.

    python -m ms2.cell_activity_compare --config recordings.json --out-dir .../compare

recordings.json is the {name: recording dict} described in ms2/cell_activity.py.  An optional
max_unstable_frac in a recording dict also drops the cells with more than that share of their
tracked window frames unstable -- segmented volume more than VOLUME_JUMP times off the cell's
own median, from unstable segmentation or a mislinked tracklet.  0.2 keeps the cells whose
segmentation is stable in at least 80% of their frames.  Counting frames rather than taking
the volume's CV keeps one bad frame from weighing more in a short window than in a long one.
The tracklets of the cells kept are written to out_dir/tracklets/<name>_tracklets_kept.json.

Per recording (metrics.csv):
    n_<level>, frac_<level>, n_active, frac_active      composition on the shared ladder
    median_rate, median_peak, median_duty                strength of the active cells
    median_onset                                         first frame above the floor, 0..1 of window
    median_onset_tracked                                 the same, 0..1 of the cell's own tracked frames
    mean_frac_firing                                     mean share of cells above the floor per frame
    frac_active_{inner,middle,outer}                     tissue split into thirds by distance from
                                                         its centre, equal cell counts per third
    radial_bias                                          median r_norm, active minus all (>0 = edge)
    nn_index                                             nearest-neighbour distance of active cells
                                                         over random sets of as many cells (<1 =
                                                         clustered, >1 = spaced out)
    nn_p_clustered, nn_p_dispersed                       permutation p of each tail of nn_index
    hull_coverage                                        convex hull of active cells / of all cells

The config is checked before anything is measured and a background diagnostic is written to
out_dir/calibration first -- see ms2/cell_activity_validate.py and
ms2/cell_activity_calibration.py.  --skip-validate skips the first for a known-good config.

Embryos are mounted at any rotation, so the spatial maps of two recordings do not line up.
A recording's "orientation", {"flip": bool, "rotate": degrees} as the ROI page saves it, and
over it out_dir/alignment.json, {name: orientation} as the app's Compare page saves it, turn
each one about its tissue centre: mirrored left-right if flip, then rotated
clockwise.  It draws spatial_maps_aligned; the numbers do not change, as none depends on
orientation.  The page sets the turn on the recording's first window frame, from the
cell-channel projection: the recording's "projection" entry or else found beside its results
(cell_projection).
"""
import argparse
import glob
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.patches import Patch
from matplotlib.path import Path
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import chi2_contingency, false_discovery_control, kruskal, mannwhitneyu

from ms2.cell_activity import (K_LEVELS, LEVEL_NAMES, MIN_PRESENCE, NOISE_FLOOR, NOISE_PEAK,
                               activity_descriptors, classify_cells, levels_from_cuts, load_window,
                               save_figure)
from ms2.cell_activity_calibration import run_calibration
from ms2.cell_activity_validate import raise_on_errors, validate_config
from ms2.roi_selection import load_projection
from ms2.track_diagnostics import cell_properties

# Weakly active and above; below are silent cells and 1-2 frame blips.
ACTIVE_LEVEL = 2
LEVEL_KEYS = {0: 'silent', 1: 'noise', 2: 'weak', 3: 'moderate', 4: 'high', 5: 'very_high'}
# r_norm = distance from the tissue centre over this percentile of all cells' distances, so a
# few stray cells don't set the scale.
RADIUS_PERCENTILE = 95
SHELLS = ('inner', 'middle', 'outer')
N_PERMUTATIONS = 1000
# A frame is unstable when the cell's volume is this many times above or below its median.
VOLUME_JUMP = 1.5
PER_CELL_METRICS = ('rate', 'peak', 'duty', 'onset', 'r_norm')
ALIGNMENT_NAME = 'alignment.json'

# Recordings take categorical slots in fixed order; levels are neutral greys below
# ACTIVE_LEVEL and one blue ramp, light to dark, above it.
RECORDING_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                    '#e87ba4', '#008300', '#4a3aa7', '#e34948')
EXTRA_RECORDING_COLOR = '#8c8a83'
LEVEL_PLOT_COLORS = {0: '#d3d1cb', 1: '#9a988f', 2: '#86b6ef', 3: '#2a78d6', 4: '#104281',
                     5: '#071d3a'}
INK, MUTED = '#0b0b0b', '#52514e'


def cell_positions(props, tracklets, cells, timepoints):
    """Mean (y, x) of each cell over the window frames where its tracklet has a label, and the
    share of those frames whose volume is more than VOLUME_JUMP times off the cell's median."""
    w = cell_window(props, tracklets, cells, timepoints)
    return w['positions'], w['unstable_frac']


def cell_window(props, tracklets, cells, timepoints):
    """Per cell over its tracked window frames: mean (y, x), the share of frames whose volume
    is VOLUME_JUMP times off its median, its volume CV (std / mean), and its (x, y) on the
    window's first and last frame (NaN where untracked there) -- what an ROI drawn on those
    frames tests."""
    n = len(cells)
    out = {'positions': np.empty((n, 2)), 'unstable_frac': np.empty(n), 'vol_cv': np.empty(n),
           'at_start': np.full((n, 2), np.nan), 'at_end': np.full((n, 2), np.nan)}
    for i, cell in enumerate(cells):
        track = tracklets[cell[len('cell_'):]]
        p = np.array([props[t][track[t]] for t in timepoints if track[t] > 0])
        out['positions'][i] = p[:, 1:3].mean(axis=0)
        out['unstable_frac'][i] = (np.abs(np.log(p[:, 3] / np.median(p[:, 3]))) > np.log(VOLUME_JUMP)).mean()
        out['vol_cv'][i] = p[:, 3].std() / p[:, 3].mean()
        for key, t in (('at_start', timepoints[0]), ('at_end', timepoints[-1])):
            if track[t] > 0 and track[t] in props.get(t, {}):
                _, y, x, _ = props[t][track[t]]
                out[key][i] = x, y
    return out


def in_roi(rec, at_start, at_end):
    """Which cells an ROI drawn on the window's first and last frame takes in: inside the
    first outline on the first frame or the last one on the last (the pipeline's rule, see
    ms2/roi_selection.py).  Every cell when the recording has no outline."""
    first, last = rec.get('roi_first'), rec.get('roi_last')
    if not first and not last:
        return np.ones(len(at_start), dtype=bool)
    hit = np.zeros(len(at_start), dtype=bool)
    for outline, xy in ((first, at_start), (last, at_end)):
        if outline:
            ok = ~np.isnan(xy[:, 0])
            hit[ok] |= Path(np.asarray(outline, dtype=float)).contains_points(xy[ok])
    return hit


def radial_position(positions):
    """r_norm of every cell, and the tissue centre and radius it is measured against."""
    center = positions.mean(axis=0)
    distance = np.linalg.norm(positions - center, axis=1)
    radius = np.percentile(distance, RADIUS_PERCENTILE)
    return distance / radius, center, radius


def nn_clustering(positions, active, rng):
    """Mean nearest-neighbour distance among the active cells over its mean for random sets of
    as many cells of the tissue, and the permutation p-value of each tail.

    Both tails are reported because both are biology: active cells sitting closer together than
    chance (clustered, index < 1) and further apart than chance (overdispersed, index > 1, what
    lateral inhibition produces).  Testing only the clustering tail hides the second inside a
    non-significant result.
    """
    n = int(active.sum())
    if n < 2:
        return np.nan, np.nan, np.nan

    def mean_nn(points):
        return cKDTree(points).query(points, k=2)[0][:, 1].mean()

    observed = mean_nn(positions[active])
    null = np.array([mean_nn(positions[rng.choice(len(positions), n, replace=False)])
                     for _ in range(N_PERMUTATIONS)])
    return (observed / null.mean(),
            (1 + (null <= observed).sum()) / (1 + N_PERMUTATIONS),
            (1 + (null >= observed).sum()) / (1 + N_PERMUTATIONS))


def hull_coverage(positions, active):
    if active.sum() < 3:
        return np.nan
    return ConvexHull(positions[active]).volume / ConvexHull(positions).volume


def measure_recording(name, rec, cache_dir):
    """Per-cell table (descriptors, position, r_norm, shell) and per-frame firing of one
    recording, the tissue centre and radius, and (timepoints, traces) with the traces raised
    to the noise floor, rows in the order of the table."""
    noise_floor = rec.get('noise_floor', NOISE_FLOOR)
    signals, present, cells, timepoints = load_window(
        rec['csv'], rec['t_start'], rec['t_end'], rec.get('min_presence', MIN_PRESENCE))
    with open(rec['tracklets']) as f:
        tracklets = json.load(f)
    props = cell_properties(rec['masks_dir'], os.path.join(cache_dir, f'{name}_cell_properties.pkl'))
    w = cell_window(props, tracklets, cells, timepoints)
    # the filters the analysis page sets, in its order: stable volume, then the ROI
    keep = np.ones(len(cells), dtype=bool)
    if rec.get('max_unstable_frac') is not None:
        keep &= w['unstable_frac'] <= rec['max_unstable_frac']
        print(f'{name}: max_unstable_frac {rec["max_unstable_frac"]} keeps {keep.sum()}/{len(keep)} cells')
    if rec.get('max_volume_cv') is not None:
        keep &= w['vol_cv'] <= rec['max_volume_cv']
        print(f'{name}: max_volume_cv {rec["max_volume_cv"]} keeps {keep.sum()}/{len(keep)} cells')
    if rec.get('roi_first') or rec.get('roi_last'):
        keep &= in_roi(rec, w['at_start'], w['at_end'])
        print(f'{name}: the ROI keeps {keep.sum()}/{len(keep)} cells')
    signals, present = signals[keep], present[keep]
    positions, unstable_frac, vol_cv = w['positions'][keep], w['unstable_frac'][keep], w['vol_cv'][keep]
    cells = [c for c, k in zip(cells, keep) if k]
    r_norm, center, radius = radial_position(positions)

    table = activity_descriptors(signals, present, noise_floor)
    table.insert(0, 'cell', cells)
    table.insert(0, 'recording', name)
    # Carried per cell because the ladder is fitted once over the pooled recordings, so the
    # blip threshold has to travel with the row rather than be passed as one number
    table['noise_peak'] = rec.get('noise_peak', NOISE_PEAK)
    table['y'], table['x'], table['r_norm'] = positions[:, 0], positions[:, 1], r_norm
    table['unstable_frac'] = unstable_frac
    table['volume_cv'] = vol_cv
    table['shell'] = pd.qcut(r_norm, len(SHELLS), labels=SHELLS).astype(str)

    timecourse = pd.DataFrame({
        'recording': name, 't': timepoints,
        't_rel': (timepoints - timepoints[0]) / max(len(timepoints) - 1, 1),
        'frac_firing': (signals > noise_floor).sum(axis=0) / present.sum(axis=0)})
    return table, timecourse, (center, radius), (timepoints, np.maximum(signals, noise_floor))


def recording_metrics(cells, timecourse):
    active = cells['active'].to_numpy()
    positions = cells[['y', 'x']].to_numpy()
    row = {'recording': cells['recording'].iloc[0], 'n_frames': len(timecourse),
           'n_cells': len(cells)}
    for level, key in LEVEL_KEYS.items():
        row[f'n_{key}'] = int((cells['level'] == level).sum())
        row[f'frac_{key}'] = row[f'n_{key}'] / len(cells)
    row['n_active'], row['frac_active'] = int(active.sum()), active.mean()
    for metric in ('rate', 'peak', 'duty', 'onset', 'onset_tracked'):
        row[f'median_{metric}'] = cells.loc[active, metric].median()
    row['mean_frac_firing'] = timecourse['frac_firing'].mean()
    for shell in SHELLS:
        row[f'frac_active_{shell}'] = cells.loc[cells['shell'] == shell, 'active'].mean()
    row['radial_bias'] = cells.loc[active, 'r_norm'].median() - cells['r_norm'].median()
    # Seeded per recording so a recording's numbers don't depend on what it is compared with
    row['nn_index'], row['nn_p_clustered'], row['nn_p_dispersed'] = nn_clustering(
        positions, active, np.random.default_rng(0))
    row['hull_coverage'] = hull_coverage(positions, active)
    return row


def composition_test(cells, composition):
    """Chi-square of the level composition, permuted when the asymptotic approximation does not
    hold.  Below an expected count of 5 the chi-square distribution is a poor fit, so the
    recording labels are shuffled instead and the p read off the empirical distribution.
    Shuffling a label vector preserves both margins exactly, so no shuffled table can have an
    empty row or column."""
    chi2, p, _, expected = chi2_contingency(composition)
    if expected.min() >= 5:
        return {'test': 'chi-square', 'metric': 'level composition', 'statistic': chi2, 'p': p}

    rng = np.random.default_rng(0)
    labels, levels = cells['recording'].to_numpy(), cells['level'].to_numpy()
    null = np.array([chi2_contingency(pd.crosstab(rng.permutation(labels), levels))[0]
                     for _ in range(N_PERMUTATIONS)])
    return {'test': 'chi-square (permutation)', 'metric': 'level composition', 'statistic': chi2,
            'p': (1 + (null >= chi2).sum()) / (1 + N_PERMUTATIONS)}


def comparison_stats(cells):
    """Level composition by chi-square; the active cells' per-cell metrics by Kruskal-Wallis
    (3+ recordings) and pairwise Mann-Whitney, BH-corrected over all pairwise tests.  Cells
    are the samples, so these ask whether two embryos differ, not two conditions."""
    rows = []
    composition = pd.crosstab(cells['recording'], cells['level'])
    composition = composition.loc[:, composition.sum() > 0]
    if composition.shape[0] > 1 and composition.shape[1] > 1:
        rows.append(composition_test(cells, composition))

    active = {name: g for name, g in cells[cells['active']].groupby('recording', sort=False)}
    pairwise = []
    for metric in PER_CELL_METRICS:
        samples = {name: g[metric].dropna() for name, g in active.items()}
        samples = {name: s for name, s in samples.items() if len(s)}
        if len(samples) >= 3:
            stat, p = kruskal(*samples.values())
            rows.append({'test': 'kruskal-wallis', 'metric': metric, 'statistic': stat, 'p': p})
        for a, b in itertools.combinations(samples, 2):
            stat, p = mannwhitneyu(samples[a], samples[b])
            pairwise.append({'test': 'mann-whitney', 'metric': metric,
                             'recording_a': a, 'recording_b': b,
                             'n_a': len(samples[a]), 'n_b': len(samples[b]),
                             'median_a': samples[a].median(), 'median_b': samples[b].median(),
                             'statistic': stat, 'p': p})
    if pairwise:
        for row, p_adj in zip(pairwise, false_discovery_control([r['p'] for r in pairwise])):
            row['p_adj'] = p_adj
    return pd.DataFrame(rows + pairwise)


def recording_colors(names):
    if len(names) > len(RECORDING_COLORS):
        print(f'warning: {len(names)} recordings, only the first {len(RECORDING_COLORS)} get '
              'their own colour; the rest are grey (still named on the axes and panels)')
    return {name: RECORDING_COLORS[i] if i < len(RECORDING_COLORS) else EXTRA_RECORDING_COLOR
            for i, name in enumerate(names)}


def titled(title, name, prefix):
    """'<prefix> - <name>' above the figure's own title, or that title alone with no prefix."""
    return f'{prefix} - {name}\n{title}' if prefix else title


def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='y', color='#e6e5e1', lw=0.8)
    ax.set_axisbelow(True)


def plot_level_composition(metrics, path_stem, ladder='shared ladder', prefix=None):
    fig, ax = plt.subplots(figsize=(1.2 * len(metrics) + 3, 4.5))
    x = np.arange(len(metrics))
    bottom = np.zeros(len(metrics))
    for level, key in LEVEL_KEYS.items():
        if not metrics[f'n_{key}'].any():   # a level no recording reached stays off the legend
            continue
        frac = metrics[f'frac_{key}'].to_numpy()
        ax.bar(x, frac, 0.6, bottom=bottom, color=LEVEL_PLOT_COLORS[level],
               edgecolor='white', linewidth=1.5, label=LEVEL_NAMES[level])
        bottom += frac
    for xi, n in zip(x, metrics['n_cells']):
        ax.text(xi, 1.02, f'n={n}', ha='center', va='bottom', fontsize=8, color=MUTED)
    ax.set_xticks(x, metrics['recording'], rotation=30, ha='right')
    ax.set(ylabel='fraction of cells', ylim=(0, 1.1),
           title=titled(f'Activity level composition ({ladder})', 'Level composition', prefix))
    style_axes(ax)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=8)
    save_figure(fig, path_stem)


def plot_per_cell_metrics(cells, colors, path_stem, prefix=None):
    active = cells[cells['active']]
    names = list(colors)
    fig, axes = plt.subplots(2, 2, figsize=(1.2 * len(names) + 6, 7))
    labels = {'rate': 'counts above floor per tracked frame', 'peak': 'peak counts above floor',
              'duty': 'fraction of tracked frames firing', 'onset': 'onset (fraction of window)'}
    rng = np.random.default_rng(0)
    for ax, (metric, label) in zip(axes.ravel(), labels.items()):
        data = [active.loc[active['recording'] == n, metric].dropna().to_numpy() for n in names]
        ax.boxplot(data, positions=range(len(names)), widths=0.5, showfliers=False,
                   medianprops={'color': INK, 'lw': 1.5}, boxprops={'color': MUTED},
                   whiskerprops={'color': MUTED}, capprops={'color': MUTED})
        for i, (name, values) in enumerate(zip(names, data)):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(values)), values, s=18,
                       color=colors[name], edgecolors='white', linewidths=0.4, zorder=3)
        ax.set_xticks(range(len(names)), names, rotation=30, ha='right')
        ax.set_ylabel(label)
        style_axes(ax)
    fig.suptitle(titled('Active cells, per cell (box = quartiles, line = median)',
                        'Per-cell metrics', prefix))
    fig.tight_layout()
    save_figure(fig, path_stem)


def plot_timecourse(timecourse, colors, path_stem, prefix=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in colors.items():
        tc = timecourse[timecourse['recording'] == name]
        ax.plot(tc['t_rel'], tc['frac_firing'], color=color, lw=2, label=name)
    ax.set(xlabel='position in stage window (0 = t_start, 1 = t_end)',
           ylabel='fraction of tracked cells above floor',
           title=titled('Cells firing over the stage window', 'Timecourse', prefix))
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, path_stem)


def plot_radial_profile(metrics, colors, path_stem, prefix=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(SHELLS))
    for _, row in metrics.iterrows():
        ax.plot(x, [row[f'frac_active_{s}'] for s in SHELLS], color=colors[row['recording']],
                lw=2, marker='o', markersize=8, markeredgecolor='white', label=row['recording'])
    ax.set_xticks(x, [f'{s} third' for s in SHELLS])
    ax.set(ylabel='fraction of cells active',
           title=titled('Active cells from tissue centre to edge\n(relative to the imaged tissue)',
                        'Radial profile', prefix))
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, path_stem)


def plot_spatial_maps(cells, geometry, path_stem, prefix=None, flip_rotate=False, ncols=4):
    """One panel per recording, in the original image orientation (plot x = x, plot y = y with
    the axis pointing down).  flip_rotate mirrors the image left-right and then rotates it 90
    degrees clockwise: image (x, y) lands at plot (-y, x) with the y axis pointing up."""
    def plot_xy(x, y):
        return (-y, x) if flip_rotate else (x, y)

    names = list(geometry)
    ncols = min(len(names), ncols)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.3 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        c = cells[cells['recording'] == name]
        # only the levels some recording reached, so the shared legend lists no empty level
        for level in sorted(set(cells['level'])):
            m = c['level'] == level
            ax.scatter(*plot_xy(c.loc[m, 'x'], c.loc[m, 'y']), s=14 if level >= ACTIVE_LEVEL else 8,
                       color=LEVEL_PLOT_COLORS[level], edgecolors='white', linewidths=0.3,
                       label=LEVEL_NAMES[level], zorder=2 + level)
        (cy, cx), radius = geometry[name]
        cx, cy = plot_xy(cx, cy)
        ax.add_patch(plt.Circle((cx, cy), radius, fill=False, ls='--', color=MUTED, lw=1))
        ax.plot(cx, cy, marker='+', color=INK, markersize=10)
        ax.set_aspect('equal')
        if not flip_rotate:
            ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(f'{name}\nactive {int(c["active"].sum())}/{len(c)}', fontsize=9)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle(titled('Cells by shared activity level; + and dashed circle = centre and '
                        f'{RADIUS_PERCENTILE}th-percentile radius of the imaged tissue',
                        'Spatial map', prefix), fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, path_stem)


def cell_projection(rec):
    """The (T, Y, X) cell-channel projection of a recording: its "projection" entry, else the
    pipeline's preprocess/SUM_C2-*.tif in the recording's folder, else the projection its ROI
    was drawn on (the ROI beside its results, or the one its run_manifest names)."""
    if rec.get('projection'):
        return rec['projection']
    # masks_dir is <recording>/unified/masks_split, or unify/ in the older hand-run layout
    rec_dir = os.path.dirname(os.path.dirname(os.path.abspath(rec['masks_dir'])))
    hits = sorted(glob.glob(os.path.join(glob.escape(rec_dir), 'preprocess', 'SUM_C2-*.tif')))
    if hits:
        return hits[-1]
    rois = [os.path.join(rec_dir, *sub, 'roi_selection.json')
            for sub in ((), ('unified',), ('unify',), ('masks',))]
    manifest = os.path.join(rec_dir, 'run_manifest.json')
    if os.path.exists(manifest):
        with open(manifest) as f:
            rois.append(json.load(f).get('params', {}).get('roi') or '')
    for roi in rois:
        if os.path.isfile(roi):
            with open(roi) as f:
                path = json.load(f).get('projection')
            if path and os.path.exists(path):
                return path
    raise FileNotFoundError(f'no cell-channel projection found for the recording in {rec_dir}; '
                            'give its SUM_C2 tif as "projection" in recordings.json')


def stage_frame(rec, t=None):
    """The cell-channel projection at frame t, the window's first (t_start) by default."""
    return np.asarray(load_projection(cell_projection(rec))[rec['t_start'] if t is None else t],
                      dtype=float)


def alignment_matrix(flip=False, rotate=0.0):
    """Image (dx, dy) about the frame centre, y down, mirrored left-right if flip and then
    turned rotate degrees clockwise on screen -- what the page's CSS rotate() scaleX() shows."""
    a = np.radians(rotate)
    turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return turn @ np.diag([-1.0 if flip else 1.0, 1.0])


def plot_aligned_maps(cells, names, alignment, path_stem, prefix=None):
    """The spatial map of every recording turned as alignment says, about its tissue centre:
    the cells of the analysis only, drawn as plot_spatial_maps draws them, all on one scale."""
    xy = {}
    for name in names:
        c = cells[cells['recording'] == name]
        _, (cy, cx), _ = radial_position(c[['y', 'x']].to_numpy())
        xy[name] = (c[['x', 'y']].to_numpy() - [cx, cy]) @ alignment_matrix(**alignment.get(name, {})).T
    # one scale for every panel, so the embryos compare in size too
    half = np.max([np.abs(p).max(axis=0) for p in xy.values()], axis=0) * 1.05
    fig, axes = plt.subplots(1, len(names), squeeze=False,
                             figsize=(4 * len(names), 4 * half[1] / half[0] + 1.4))
    for ax, name in zip(axes[0], names):
        c, p = cells[cells['recording'] == name], xy[name]
        # short bursts are not in the active heatmap, so they show as silent; the legend is
        # always silent, weakly, moderate, high (plus very high when some cell reached it)
        shown = c['level'].replace(1, 0).to_numpy()
        for level in sorted({0, 2, 3, 4} | set(cells['level']) - {1}):
            sel = shown == level
            ax.scatter(p[sel, 0], p[sel, 1], s=14 if level >= ACTIVE_LEVEL else 8,
                       color=LEVEL_PLOT_COLORS[level], edgecolors='white', linewidths=0.3,
                       label=LEVEL_NAMES[level], zorder=2 + level)
        ax.plot(0, 0, marker='+', color=INK, markersize=10)
        ax.set(xlim=(-half[0], half[0]), ylim=(half[1], -half[1]))   # y down, as in the image
        ax.set_aspect('equal')
        ax.axis('off')
        a = alignment.get(name, {})
        turn = f'{"mirrored, " if a.get("flip") else ""}rotated {a.get("rotate", 0):g}°'
        ax.set_title(f'{name}\nactive {int(c["active"].sum())}/{len(c)} · {turn}', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle(titled('Cells by shared activity level, recordings aligned; + = centre of the '
                        'imaged tissue', 'Aligned spatial map', prefix), fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_figure(fig, path_stem)


def comparison_alignment(out_dir, recordings):
    """{name: {flip, rotate}}: each recording's own orientation, with out_dir/alignment.json
    over it, or None when neither turns anything."""
    alignment = {name: rec['orientation'] for name, rec in recordings.items() if rec.get('orientation')}
    path = os.path.join(out_dir, ALIGNMENT_NAME)
    if os.path.exists(path):
        with open(path) as f:
            alignment.update(json.load(f))
    return alignment or None


def align_comparison(out_dir, recordings=None, cells=None, prefix=None):
    """Draw out_dir/spatial_maps_aligned as comparison_alignment turns the recordings.
    recordings and cells default to the recordings.json and cells.csv in out_dir.  Returns
    the png's path."""
    if recordings is None:
        with open(os.path.join(out_dir, 'recordings.json')) as f:
            recordings = json.load(f)
    if cells is None:
        cells = pd.read_csv(os.path.join(out_dir, 'cells.csv'))
    stem = os.path.join(out_dir, 'spatial_maps_aligned')
    plot_aligned_maps(cells, list(recordings), comparison_alignment(out_dir, recordings) or {},
                      stem, prefix)
    return stem + '.png'


def plot_activity_heatmap(cells, traces, colors, path_stem, title):
    """The cells above silent, grouped by shared level (highest on top, red lines between),
    then by recording in config order, then by rate, highest first.  One recording is drawn on
    its own timepoints; several are aligned on frames from their window start, the frames past
    a shorter window grey, with each row's recording in the strip on the left."""
    # row = the cell's row in its recording's traces, which follow the recording's cells
    shown = cells.assign(row=cells.groupby('recording', sort=False).cumcount(),
                         order=cells['recording'].map({name: i for i, name in enumerate(traces)}))
    shown = shown[shown['level'] > 0]
    shown = shown.sort_values(['level', 'order', 'rate'], ascending=[False, True, False],
                              kind='stable')
    names = [name for name in traces if (shown['recording'] == name).any()]
    if not names:
        return
    width = max(traces[name][1].shape[1] for name in names)
    matrix = np.full((len(shown), width), np.nan)
    for i, (name, row) in enumerate(zip(shown['recording'], shown['row'])):
        trace = traces[name][1][row]
        matrix[i, :len(trace)] = trace

    pooled = len(traces) > 1
    fig, (strip, ax) = plt.subplots(1, 2, figsize=(12, min(4 + 0.04 * len(shown), 30)),
                                    gridspec_kw={'width_ratios': (1, 30), 'wspace': 0.02})
    if pooled:
        strip.imshow(np.array([[to_rgb(colors[n])] for n in shown['recording']]),
                     aspect='auto', interpolation='none', extent=(0, 1, len(shown), 0))
        strip.set_xticks([])
        strip.set_yticks([])
        strip.set_ylabel('cells')
    else:
        strip.axis('off')
    timepoints = traces[names[0]][0]
    x0 = timepoints[0] if not pooled else 0
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#e6e5e1')
    # 'none' keeps the eps at one pixel per cell and frame.  Log colours: a few bright cells
    # would otherwise flatten every weak trace to the floor colour.
    im = ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='none', norm=LogNorm(),
                   extent=(x0 - 0.5, x0 + width - 0.5, len(shown), 0))
    levels = shown['level'].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(levels)) + 1]
    for start, end in zip(starts, np.r_[starts[1:], len(levels)]):
        if start:
            ax.axhline(start, color='red', lw=1.5)
            if pooled:
                strip.axhline(start, color='red', lw=1.5)
        # axes fractions run bottom-up, rows top-down
        ax.text(1.01, 1 - (start + end) / 2 / len(levels), f'{LEVEL_NAMES[levels[start]]}\n'
                f'n={end - start}', transform=ax.transAxes, va='center', fontsize=8)
    ax.set_yticks([])
    ax.set(title=title, xlabel='frame from window start' if pooled else 'timepoint')
    fig.colorbar(im, ax=ax, label="expression, log scale (floor and below = floor)", pad=0.12)
    if pooled:
        fig.legend([Patch(color=colors[n]) for n in names], names, loc='lower center',
                   ncol=len(names), frameon=False, fontsize=8)
    save_figure(fig, path_stem)


STATS_README = """\
stats.csv: the sampling unit is the CELL, not the embryo.

Every test here pools the cells of a recording and treats them as independent samples.  Cells
within one embryo are not independent of each other, so a small p means "these two recordings
differ", not "these two stages differ" or "this condition has an effect".  One embryo per stage
cannot separate a stage effect from an embryo effect, however many cells it contributes -- more
cells only shrink the p-value of the difference between those two particular embryos.

To test a stage or a condition, image several embryos per group and test at the embryo level,
with each embryo's summary (the rows of metrics.csv) as one sample.
"""


def compare_recordings(recordings, out_dir, skip_validate=False, k=K_LEVELS, ladder='joint',
                       title=None, flip_rotate=False, level_cuts=None):
    """Measure every recording of {name: recording dict} in its window, put all on the shared
    ladder, and write the tables and figures into out_dir.  Returns the metrics table.  k is
    the number of ladder levels above the blips, None to choose it by silhouette.

    ladder='per_recording' fits a separate ladder to each recording instead, on the same rate,
    so "high" is relative to its own recording.  ladder='balanced' fits the one ladder with each
    cell weighted 1 / its recording's count of cells on the ladder, so every recording counts
    equally.  Only the
    split of the active cells into levels changes: which cells are active, and every per-cell
    metric, are the same in all modes.

    title, e.g. 'Stage 12', heads every figure as '<title> - <figure name>'.  flip_rotate
    mirrors the spatial maps left-right, then rotates them 90 degrees clockwise."""
    cache_dir = os.path.join(out_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    if not skip_validate:
        raise_on_errors(validate_config(recordings, cache_dir))
    run_calibration(recordings, os.path.join(out_dir, 'calibration'), title)
    tables, timecourses, geometry, traces = [], [], {}, {}
    for name, rec in recordings.items():
        table, timecourse, geometry[name], traces[name] = measure_recording(name, rec, cache_dir)
        tables.append(table)
        timecourses.append(timecourse)
    cells = pd.concat(tables, ignore_index=True)
    timecourse = pd.concat(timecourses, ignore_index=True)

    # One ladder over all recordings (or one per recording), on rate so different window
    # lengths compare.  The blip threshold is per row, so recordings may set their own
    # noise_peak.
    groups = ([cells.index] if ladder in ('joint', 'balanced')
              else list(cells.groupby('recording', sort=False).groups.values()))
    # 'balanced': each recording's cells on the ladder (peak >= noise_peak) sum to weight 1
    ranked = (cells['peak'] > 0) & (cells['peak'] >= cells['noise_peak'])
    weights = 1 / ranked.groupby(cells['recording']).transform('sum')
    cells['level'] = 0
    if level_cuts is not None:
        # the ladder set by hand on the analysis page: rate cuts between its levels
        cells['level'] = levels_from_cuts(cells, level_cuts, noise_peak=cells['noise_peak'].to_numpy())
    else:
        for idx in groups:
            cells.loc[idx, 'level'] = classify_cells(
                cells.loc[idx], score='rate', noise_peak=cells.loc[idx, 'noise_peak'].to_numpy(), k=k,
                weights=weights[idx].to_numpy() if ladder == 'balanced' else None)
    cells['level_name'] = cells['level'].map(LEVEL_NAMES)
    cells['active'] = cells['level'] >= ACTIVE_LEVEL
    metrics = pd.DataFrame([recording_metrics(cells[cells['recording'] == name],
                                              timecourse[timecourse['recording'] == name])
                            for name in recordings])

    cells.to_csv(os.path.join(out_dir, 'cells.csv'), index=False)
    os.makedirs(os.path.join(out_dir, 'tracklets'), exist_ok=True)
    for name, rec in recordings.items():
        with open(rec['tracklets']) as f:
            tracklets = json.load(f)
        kept = cells.loc[cells['recording'] == name, 'cell'].str[len('cell_'):]
        with open(os.path.join(out_dir, 'tracklets', f'{name}_tracklets_kept.json'), 'w') as f:
            json.dump({tid: tracklets[tid] for tid in kept}, f)
    timecourse.to_csv(os.path.join(out_dir, 'timecourse.csv'), index=False)
    metrics.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    comparison_stats(cells).to_csv(os.path.join(out_dir, 'stats.csv'), index=False)
    with open(os.path.join(out_dir, 'stats_readme.txt'), 'w') as f:
        f.write(STATS_README)

    colors = recording_colors(list(recordings))
    plot_level_composition(metrics, os.path.join(out_dir, 'level_composition'),
                           {'joint': 'shared ladder', 'balanced': 'shared ladder, recordings '
                            'weighted equally', 'per_recording': 'ladder per recording'}[ladder],
                           title)
    plot_per_cell_metrics(cells, colors, os.path.join(out_dir, 'per_cell_metrics'), title)
    plot_timecourse(timecourse, colors, os.path.join(out_dir, 'timecourse'), title)
    plot_radial_profile(metrics, colors, os.path.join(out_dir, 'radial_profile'), title)
    plot_spatial_maps(cells, geometry, os.path.join(out_dir, 'spatial_maps'), title, flip_rotate,
                      ncols=len(geometry))
    for name in recordings:
        plot_activity_heatmap(cells[cells['recording'] == name], {name: traces[name]}, colors,
                              os.path.join(out_dir, f'heatmap_active_{name}'),
                              titled(f'{name}: cells by shared activity level', 'Heatmap', title))
    plot_activity_heatmap(cells, traces, colors, os.path.join(out_dir, 'heatmap_active_pooled'),
                          titled('All recordings: cells by shared activity level, then recording',
                                 'Pooled heatmap', title))
    if comparison_alignment(out_dir, recordings):     # turned on the ROI or the Compare page
        align_comparison(out_dir, recordings, cells, title)
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='json of {name: recording dict}')
    p.add_argument('--out-dir', required=True, help='tables, figures and the positions cache')
    p.add_argument('--skip-validate', action='store_true',
                   help='skip the config checks, for a rerun of a known-good config')
    p.add_argument('--k', type=int, default=K_LEVELS,
                   help='ladder levels above the blips (default %(default)s)')
    p.add_argument('--silhouette', action='store_true',
                   help='choose k between 2 and N_LEVELS by silhouette instead of --k')
    p.add_argument('--ladder', choices=('joint', 'balanced', 'per_recording'), default='joint',
                   help='one ladder over all recordings (default), the same with every '
                        'recording weighted equally, or one ladder per recording')
    p.add_argument('--title', help="heads every figure as '<title> - <figure name>', "
                                   "e.g. 'Stage 12'")
    p.add_argument('--flip-rotate', action='store_true',
                   help='spatial maps mirrored left-right, then rotated 90 degrees clockwise')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        recordings = json.load(f)
    metrics = compare_recordings(recordings, args.out_dir, args.skip_validate,
                                 None if args.silhouette else args.k, args.ladder, args.title,
                                 args.flip_rotate)
    print(metrics.set_index('recording').T.to_string())


if __name__ == '__main__':
    main()
