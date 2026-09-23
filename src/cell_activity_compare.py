"""Compare stage-window cell activity across any number of recordings.

Every recording is cut to its own developmental-stage window and measured with the
definitions of src/cell_activity.py, then the recordings are set side by side.  Two choices
make the numbers comparable between embryos:

  levels   weak / moderate / high come from ONE KMeans ladder fitted on the pooled cells of
           all recordings, on activity per tracked frame (rate) so windows of different length
           compare.  src/cell_activity.py fits a ladder per recording, where "high" in one
           embryo is a different threshold than in the next.  Silent and short burst / noise
           use the same absolute thresholds in both.  Active = weakly active or above.
  space    the embryo is cut off by the field of view and mounted at any rotation, so every
           spatial measure is orientation-free and relative to the imaged tissue -- the cells
           tracked in the window -- not to the whole embryo.

    python -m src.cell_activity_compare --config recordings.json --out-dir .../compare

recordings.json is the {name: recording dict} described in src/cell_activity.py.

Per recording (metrics.csv):
    n_<level>, frac_<level>, n_active, frac_active      composition on the shared ladder
    median_rate, median_peak, median_duty                strength of the active cells
    median_onset                                         first frame above the floor, 0..1 of window
    mean_frac_firing                                     mean share of cells above the floor per frame
    frac_active_{inner,middle,outer}                     tissue split into thirds by distance from
                                                         its centre, equal cell counts per third
    radial_bias                                          median r_norm, active minus all (>0 = edge)
    nn_index, nn_p                                       nearest-neighbour distance of active cells
                                                         over random sets of as many cells (<1 =
                                                         clustered), one-sided permutation p
    hull_coverage                                        convex hull of active cells / of all cells
"""
import argparse
import itertools
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree
from scipy.stats import chi2_contingency, false_discovery_control, kruskal, mannwhitneyu

from src.cell_activity import (LEVEL_NAMES, MIN_PRESENCE, NOISE_FLOOR, activity_descriptors,
                               classify_cells, load_window, save_figure)
from src.track_diagnostics import cell_properties

# Weakly active and above; below are silent cells and 1-2 frame blips.
ACTIVE_LEVEL = 2
LEVEL_KEYS = {0: 'silent', 1: 'noise', 2: 'weak', 3: 'moderate', 4: 'high'}
# r_norm = distance from the tissue centre over this percentile of all cells' distances, so a
# few stray cells don't set the scale.
RADIUS_PERCENTILE = 95
SHELLS = ('inner', 'middle', 'outer')
N_PERMUTATIONS = 1000
PER_CELL_METRICS = ('rate', 'peak', 'duty', 'onset', 'r_norm')

# Recordings take categorical slots in fixed order; levels are neutral greys below
# ACTIVE_LEVEL and one blue ramp, light to dark, above it.
RECORDING_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                    '#e87ba4', '#008300', '#4a3aa7', '#e34948')
EXTRA_RECORDING_COLOR = '#8c8a83'
LEVEL_PLOT_COLORS = {0: '#d3d1cb', 1: '#9a988f', 2: '#86b6ef', 3: '#2a78d6', 4: '#104281'}
INK, MUTED = '#0b0b0b', '#52514e'


def cell_positions(props, tracklets, cells, timepoints):
    """Mean (y, x) of each cell over the window frames where its tracklet has a label."""
    positions = np.empty((len(cells), 2))
    for i, cell in enumerate(cells):
        track = tracklets[cell[len('cell_'):]]
        positions[i] = np.mean([props[t][track[t]][1:3] for t in timepoints if track[t] > 0],
                               axis=0)
    return positions


def radial_position(positions):
    """r_norm of every cell, and the tissue centre and radius it is measured against."""
    center = positions.mean(axis=0)
    distance = np.linalg.norm(positions - center, axis=1)
    radius = np.percentile(distance, RADIUS_PERCENTILE)
    return distance / radius, center, radius


def nn_clustering(positions, active, rng):
    """Mean nearest-neighbour distance among the active cells over its mean for random sets of
    as many cells of the tissue, and the one-sided permutation p-value of the clustering."""
    n = int(active.sum())
    if n < 2:
        return np.nan, np.nan

    def mean_nn(points):
        return cKDTree(points).query(points, k=2)[0][:, 1].mean()

    observed = mean_nn(positions[active])
    null = np.array([mean_nn(positions[rng.choice(len(positions), n, replace=False)])
                     for _ in range(N_PERMUTATIONS)])
    return observed / null.mean(), (1 + (null <= observed).sum()) / (1 + N_PERMUTATIONS)


def hull_coverage(positions, active):
    if active.sum() < 3:
        return np.nan
    return ConvexHull(positions[active]).volume / ConvexHull(positions).volume


def measure_recording(name, rec, cache_dir):
    """Per-cell table (descriptors, position, r_norm, shell) and per-frame firing of one
    recording, plus the tissue centre and radius."""
    noise_floor = rec.get('noise_floor', NOISE_FLOOR)
    signals, present, cells, timepoints = load_window(
        rec['csv'], rec['t_start'], rec['t_end'], rec.get('min_presence', MIN_PRESENCE))
    with open(rec['tracklets']) as f:
        tracklets = json.load(f)
    props = cell_properties(rec['masks_dir'], os.path.join(cache_dir, f'{name}_cell_properties.pkl'))
    positions = cell_positions(props, tracklets, cells, timepoints)
    r_norm, center, radius = radial_position(positions)

    table = activity_descriptors(signals, present, noise_floor)
    table.insert(0, 'cell', cells)
    table.insert(0, 'recording', name)
    table['y'], table['x'], table['r_norm'] = positions[:, 0], positions[:, 1], r_norm
    table['shell'] = pd.qcut(r_norm, len(SHELLS), labels=SHELLS).astype(str)

    timecourse = pd.DataFrame({
        'recording': name, 't': timepoints,
        't_rel': (timepoints - timepoints[0]) / max(len(timepoints) - 1, 1),
        'frac_firing': (signals > noise_floor).sum(axis=0) / present.sum(axis=0)})
    return table, timecourse, (center, radius)


def recording_metrics(cells, timecourse):
    active = cells['active'].to_numpy()
    positions = cells[['y', 'x']].to_numpy()
    row = {'recording': cells['recording'].iloc[0], 'n_frames': len(timecourse),
           'n_cells': len(cells)}
    for level, key in LEVEL_KEYS.items():
        row[f'n_{key}'] = int((cells['level'] == level).sum())
        row[f'frac_{key}'] = row[f'n_{key}'] / len(cells)
    row['n_active'], row['frac_active'] = int(active.sum()), active.mean()
    for metric in ('rate', 'peak', 'duty', 'onset'):
        row[f'median_{metric}'] = cells.loc[active, metric].median()
    row['mean_frac_firing'] = timecourse['frac_firing'].mean()
    for shell in SHELLS:
        row[f'frac_active_{shell}'] = cells.loc[cells['shell'] == shell, 'active'].mean()
    row['radial_bias'] = cells.loc[active, 'r_norm'].median() - cells['r_norm'].median()
    # Seeded per recording so a recording's numbers don't depend on what it is compared with
    row['nn_index'], row['nn_p'] = nn_clustering(positions, active, np.random.default_rng(0))
    row['hull_coverage'] = hull_coverage(positions, active)
    return row


def comparison_stats(cells):
    """Level composition by chi-square; the active cells' per-cell metrics by Kruskal-Wallis
    (3+ recordings) and pairwise Mann-Whitney, BH-corrected over all pairwise tests.  Cells
    are the samples, so these ask whether two embryos differ, not two conditions."""
    rows = []
    composition = pd.crosstab(cells['recording'], cells['level'])
    composition = composition.loc[:, composition.sum() > 0]
    if composition.shape[0] > 1 and composition.shape[1] > 1:
        chi2, p, _, _ = chi2_contingency(composition)
        rows.append({'test': 'chi-square', 'metric': 'level composition',
                     'statistic': chi2, 'p': p})

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


def style_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='y', color='#e6e5e1', lw=0.8)
    ax.set_axisbelow(True)


def plot_level_composition(metrics, path_stem):
    fig, ax = plt.subplots(figsize=(1.2 * len(metrics) + 3, 4.5))
    x = np.arange(len(metrics))
    bottom = np.zeros(len(metrics))
    for level, key in LEVEL_KEYS.items():
        frac = metrics[f'frac_{key}'].to_numpy()
        ax.bar(x, frac, 0.6, bottom=bottom, color=LEVEL_PLOT_COLORS[level],
               edgecolor='white', linewidth=1.5, label=LEVEL_NAMES[level])
        bottom += frac
    for xi, n in zip(x, metrics['n_cells']):
        ax.text(xi, 1.02, f'n={n}', ha='center', va='bottom', fontsize=8, color=MUTED)
    ax.set_xticks(x, metrics['recording'], rotation=30, ha='right')
    ax.set(ylabel='fraction of cells', ylim=(0, 1.1),
           title='Activity level composition (shared ladder)')
    style_axes(ax)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=8)
    save_figure(fig, path_stem)


def plot_per_cell_metrics(cells, colors, path_stem):
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
    fig.suptitle('Active cells, per cell (box = quartiles, line = median)')
    fig.tight_layout()
    save_figure(fig, path_stem)


def plot_timecourse(timecourse, colors, path_stem):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, color in colors.items():
        tc = timecourse[timecourse['recording'] == name]
        ax.plot(tc['t_rel'], tc['frac_firing'], color=color, lw=2, label=name)
    ax.set(xlabel='position in stage window (0 = t_start, 1 = t_end)',
           ylabel='fraction of tracked cells above floor',
           title='Cells firing over the stage window')
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, path_stem)


def plot_radial_profile(metrics, colors, path_stem):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(SHELLS))
    for _, row in metrics.iterrows():
        ax.plot(x, [row[f'frac_active_{s}'] for s in SHELLS], color=colors[row['recording']],
                lw=2, marker='o', markersize=8, markeredgecolor='white', label=row['recording'])
    ax.set_xticks(x, [f'{s} third' for s in SHELLS])
    ax.set(ylabel='fraction of cells active',
           title='Active cells from tissue centre to edge\n(relative to the imaged tissue)')
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, path_stem)


def plot_spatial_maps(cells, geometry, path_stem):
    """One panel per recording, rotated like the src/cell_activity.py overlays (rot90 -1:
    plot x = -y, plot y = x with the axis pointing down)."""
    names = list(geometry)
    ncols = min(len(names), 4)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.3 * nrows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        c = cells[cells['recording'] == name]
        for level in sorted(LEVEL_KEYS):
            m = c['level'] == level
            ax.scatter(-c.loc[m, 'y'], c.loc[m, 'x'], s=14 if level >= ACTIVE_LEVEL else 8,
                       color=LEVEL_PLOT_COLORS[level], edgecolors='white', linewidths=0.3,
                       label=LEVEL_NAMES[level], zorder=2 + level)
        (cy, cx), radius = geometry[name]
        ax.add_patch(plt.Circle((-cy, cx), radius, fill=False, ls='--', color=MUTED, lw=1))
        ax.plot(-cy, cx, marker='+', color=INK, markersize=10)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(f'{name}\nactive {int(c["active"].sum())}/{len(c)}', fontsize=9)
    for ax in axes.ravel()[len(names):]:
        ax.axis('off')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, fontsize=8)
    fig.suptitle('Cells by shared activity level; + and dashed circle = centre and '
                 f'{RADIUS_PERCENTILE}th-percentile radius of the imaged tissue', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, path_stem)


def compare_recordings(recordings, out_dir):
    """Measure every recording of {name: recording dict} in its window, put all on the shared
    ladder, and write the tables and figures into out_dir.  Returns the metrics table."""
    cache_dir = os.path.join(out_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    tables, timecourses, geometry = [], [], {}
    for name, rec in recordings.items():
        table, timecourse, geometry[name] = measure_recording(name, rec, cache_dir)
        tables.append(table)
        timecourses.append(timecourse)
    cells = pd.concat(tables, ignore_index=True)
    timecourse = pd.concat(timecourses, ignore_index=True)

    # One ladder over all recordings, on rate so different window lengths compare
    cells['level'] = classify_cells(cells, score='rate')
    cells['level_name'] = cells['level'].map(LEVEL_NAMES)
    cells['active'] = cells['level'] >= ACTIVE_LEVEL
    metrics = pd.DataFrame([recording_metrics(cells[cells['recording'] == name],
                                              timecourse[timecourse['recording'] == name])
                            for name in recordings])

    cells.to_csv(os.path.join(out_dir, 'cells.csv'), index=False)
    timecourse.to_csv(os.path.join(out_dir, 'timecourse.csv'), index=False)
    metrics.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    comparison_stats(cells).to_csv(os.path.join(out_dir, 'stats.csv'), index=False)

    colors = recording_colors(list(recordings))
    plot_level_composition(metrics, os.path.join(out_dir, 'level_composition'))
    plot_per_cell_metrics(cells, colors, os.path.join(out_dir, 'per_cell_metrics'))
    plot_timecourse(timecourse, colors, os.path.join(out_dir, 'timecourse'))
    plot_radial_profile(metrics, colors, os.path.join(out_dir, 'radial_profile'))
    plot_spatial_maps(cells, geometry, os.path.join(out_dir, 'spatial_maps'))
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='json of {name: recording dict}')
    p.add_argument('--out-dir', required=True, help='tables, figures and the positions cache')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        recordings = json.load(f)
    metrics = compare_recordings(recordings, args.out_dir)
    print(metrics.set_index('recording').T.to_string())


if __name__ == '__main__':
    main()
