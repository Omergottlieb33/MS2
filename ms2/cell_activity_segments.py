"""Body segments marked on every recording of a stage comparison.

The app's Segments page opens each recording of a comparison folder at its orientation (see
comparison_alignment in ms2/cell_activity_compare.py) and the user outlines segments 2..5 on
one frame of its window.  out_dir/segments.json holds them:

    {"New-02-v3": {"t": 45, "orientation": {"flip": true, "rotate": -58.0},
                   "segments": {"2": [[x, y], ...], "3": [[x, y], ...]}}}

Outlines are in the image's own pixel coordinates -- those of cells.csv x and y -- whatever
orientation they were drawn at, so the cells inside a segment are found without undoing any
turn.  A recording need not have every segment: the field of view can cut some off.

A recording's own segments, marked on the ROI page when the run is set up, sit beside its
ROI in a segments.json holding one such entry, {t, orientation, segments}.  A comparison
starts from them (ms2/pipeline_compare.py carries them into its config as "segments").

Segment activity -- the same segment set side by side across the embryos of a stage:

    python -m ms2.cell_activity_segments <comparison folder> --bins 8 --time-bins 4 [--segments 2 3]

Each segment's outline is turned to the aligned view and boxed upright; the box maps onto a
common grid, centre (0, 0), right-upper corner (1, 1), right-lower (1, -1), left-upper (-1, 1),
left-lower (-1, -1).  A cell is a point at its average position over the window (cells.csv x,
y), so a grid bin holds the same cells for the whole window: they move a few percent of a
segment over it.  A bin's value at a frame is the share of its cells tracked there whose signal
is above the noise floor -- empty bins are NaN -- so embryos of different cell densities
compare.  Frames line up by position in the stage window (0..1), as timecourse.png does, and
are pooled into time bins.

--cells-per-bin N replaces the grid with bins of uneven size holding about N cells each: the
segment's cells cut into columns of equal count by u, each column into rows of equal count by
v.  A segment with more cells is binned finer.  The bin layouts then differ between
recordings, so maps are painted on a common 100 x 100 raster to draw, average and correlate;
the bins' rectangles (u0, u1, v0, v1) are in the tables.  Written to
<comparison folder>/segment_activity/:

    segment_activity.csv          every frame x bin: n_cells, n_firing, frac_firing
    segment_<L>_activity.png/eps  each recording's whole-window map side by side and their mean,
                                  one 0..1 colour scale (the time bins are in the table)
    segment_<L>_activity.gif      the same maps at 0%, 10%, .. 100% of each recording's window
                                  (frame round(p * (n - 1)) of n), all recordings in each frame
    segment_<L>_cells_per_bin.png/csv  cells tracked per bin, averaged over the window
    segment_<L>_intensity.png     mean total intensity per cell in each bin (AU above the floor,
                                  summed over the window), with _intensity_correlation.csv
    segment_<L>_clusters.png      a row per cluster: each bin's share of cells at that level,
                                  every row in its cluster's colour, with _clusters_correlation.csv
    segment_cells.csv             every cell of every segment: its bin, (u, v), total, level
    segment_<L>_correlation.csv   Pearson r of each pair of recordings' whole-window maps,
                                  over the bins both have cells in
    segment_boxes.png             every recording aligned, its outlines, boxes and corners
    params.json                   bins, time bins, segments
"""
import argparse
import itertools
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from matplotlib import patheffects
from matplotlib.patches import Rectangle

SEGMENT_LABELS = ('2', '3', '4', '5')
SEGMENTS_NAME = 'segments.json'


def load_segments(out_dir):
    path = os.path.join(out_dir, SEGMENTS_NAME)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def clean_polygons(name, polygons):
    """{label: [[x, y], ...]} checked -- labels 2..5, 3 points or more -- in label order."""
    clean = {}
    for label, points in polygons.items():
        if label not in SEGMENT_LABELS:
            raise ValueError(f'{name}: segment {label!r} is not one of {", ".join(SEGMENT_LABELS)}')
        if len(points) < 3:
            raise ValueError(f'{name}: segment {label} needs at least 3 points')
        clean[label] = [[float(x), float(y)] for x, y in points]
    return dict(sorted(clean.items()))


def recording_segments_path(roi_path):
    """A recording's segments sit beside its ROI, apart from it: rewriting the ROI sends gene
    expression round again, marking segments should not."""
    return os.path.join(os.path.dirname(os.path.abspath(roi_path)), SEGMENTS_NAME)


def save_recording_segments(roi_path, t, orientation, polygons):
    """The segments marked on frame t of a recording, at the orientation they were drawn at."""
    path = recording_segments_path(roi_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'t': int(t), 'orientation': orientation or {'flip': False, 'rotate': 0},
                   'segments': clean_polygons(os.path.basename(os.path.dirname(path)), polygons)},
                  f, indent=2)
    return path


def load_recording_segments(roi_path):
    path = recording_segments_path(roi_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_segments(out_dir, segments, recordings):
    """Check segments ({name: {t, orientation, segments: {label: polygon}}}) against the
    comparison's recordings and write out_dir/segments.json.  Returns its path."""
    unknown = set(segments) - set(recordings)
    if unknown:
        raise ValueError(f'{", ".join(sorted(unknown))} not in this comparison')
    clean = {}
    for name in recordings:                 # in the comparison's order
        if name not in segments:
            continue
        entry = segments[name]
        rec = recordings[name]
        t = int(entry['t'])
        if not rec['t_start'] <= t <= rec['t_end']:
            raise ValueError(f'{name}: frame {t} is outside its window '
                             f'{rec["t_start"]}..{rec["t_end"]}')
        clean[name] = {'t': t, 'orientation': entry.get('orientation') or {'flip': False, 'rotate': 0},
                       'segments': clean_polygons(name, entry.get('segments', {}))}
    path = os.path.join(out_dir, SEGMENTS_NAME)
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)
    return path


ACTIVITY_DIR = 'segment_activity'


def segment_frame(polygon, orientation):
    """The upright box of a polygon in the aligned view: (matrix, centre, half-sizes).  The
    matrix turns image (x, y) as the comparison aligns the recording (see alignment_matrix),
    about the origin -- any fixed point does, the box is found in the same coordinates."""
    from ms2.cell_activity_compare import alignment_matrix
    m = alignment_matrix(**orientation)
    aligned = np.asarray(polygon, dtype=float) @ m.T
    lo, hi = aligned.min(axis=0), aligned.max(axis=0)
    return m, (lo + hi) / 2, (hi - lo) / 2


def normalise(xy, frame):
    """Image (x, y) to the segment grid: (u, v) with the box's right-upper corner at (1, 1).
    Up is +1 although image y points down, hence the sign on v."""
    m, centre, half = frame
    d = (np.asarray(xy, dtype=float) @ m.T - centre) / half
    return np.column_stack([d[:, 0], -d[:, 1]])


def inside(polygon, xy):
    return Path(np.asarray(polygon, dtype=float)).contains_points(np.asarray(xy, dtype=float))


def grid_bin(uv, bins):
    """Each (u, v) in [-1, 1] to its (ix, iy), iy counting up from the bottom row."""
    return np.clip(((np.asarray(uv) + 1) / 2 * bins).astype(int), 0, bins - 1)


def grid_bins(uv, bins):
    """A regular bins x bins grid: each cell's bin, and every bin as (col, row, u0, u1, v0,
    v1), row-major from the bottom-left."""
    ix, iy = grid_bin(uv, bins).T if len(uv) else (np.zeros(0, int), np.zeros(0, int))
    by, bx = np.divmod(np.arange(bins * bins), bins)
    edge = np.linspace(-1, 1, bins + 1)
    rects = np.column_stack([bx, by, edge[bx], edge[bx + 1], edge[by], edge[by + 1]])
    return iy * bins + ix, rects


def equal_count_bins(uv, target):
    """Bins of uneven size holding about `target` cells each: the cells cut into columns by u,
    then each column into rows by v, every bin the same count.  The bins needed, n/target,
    go into about sqrt(that) columns, some columns a row more than others, each column taking
    cells in proportion to its rows -- so any number of bins comes out near square, never as
    one strip of rows.  Edges sit halfway between neighbouring cells, the outer ones on the
    grid's border; dense areas get small bins."""
    n = len(uv)
    if not n:
        return np.zeros(0, dtype=int), np.array([(0, 0, -1.0, 1.0, -1.0, 1.0)])
    n_bins = max(1, int(round(n / target)))
    rows_per_col = [len(r) for r in np.array_split(np.arange(n_bins), max(1, round(np.sqrt(n_bins))))]
    ids, rects = np.zeros(n, dtype=int), []

    def cuts(values, shares):
        """Split order(values) into runs sized by shares; return the runs and their edges."""
        order = np.argsort(values, kind='stable')
        bounds = np.round(np.cumsum([0] + shares) / sum(shares) * len(values)).astype(int)
        runs = [order[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]
        edges = [-1.0] + [(values[a[-1]] + values[b[0]]) / 2 for a, b in zip(runs, runs[1:])] + [1.0]
        return runs, edges

    columns, u_edges = cuts(uv[:, 0], rows_per_col)
    for c, (col, k) in enumerate(zip(columns, rows_per_col)):
        rows, v_edges = cuts(uv[col, 1], [1] * k)
        for r, row in enumerate(rows):
            ids[col[row]] = len(rects)
            rects.append((c, r, u_edges[c], u_edges[c + 1], v_edges[r], v_edges[r + 1]))
    return ids, np.array(rects, dtype=float)


def box_frame(polygon, xy_inside, orientation, box_from='outline'):
    """The frame a segment's grid is cut from: the upright box around its outline, or with
    box_from='cells' around the cells inside it (around the outline under 2 cells) -- tight
    when the outline takes in more than the analysed cells."""
    if box_from == 'cells' and len(xy_inside) >= 2:
        m, centre, half = segment_frame(xy_inside, orientation)
        return m, centre, np.maximum(half, 1e-9)
    return segment_frame(polygon, orientation)


def segment_activity(name, rec, entry, rec_cells, bins, time_bins, labels=None,
                     cells_per_bin=None, box_from='outline', with_members=False):
    """The per-frame, per-bin table of one recording's segments (see the module docstring).
    rec_cells is this recording's rows of cells.csv: the comparison's own cells and their
    window-average positions.  cells_per_bin switches the bins×bins grid for equal-count bins;
    box_from='cells' cuts the grid from the cells inside each outline (see box_frame).
    with_members also returns one row per cell: its bin, its total signal above the floor over
    the window (what the intensity maps average) and its level, from rec_cells."""
    from ms2.cell_activity import NOISE_FLOOR, load_window
    noise_floor = rec.get('noise_floor', NOISE_FLOOR)
    # every tracked cell, then picked by name: cells.csv is the set the comparison kept
    signals, present, cell_names, timepoints = load_window(rec['csv'], rec['t_start'],
                                                           rec['t_end'], min_presence=0)
    row_of = {c: i for i, c in enumerate(cell_names)}
    t_rel = (timepoints - timepoints[0]) / max(len(timepoints) - 1, 1)
    t_bin = np.minimum((t_rel * time_bins).astype(int), time_bins - 1)
    xy = rec_cells[['x', 'y']].to_numpy()
    rows, members = [], []
    for label, polygon in entry['segments'].items():
        if labels and label not in labels:
            continue
        hit = inside(polygon, xy)
        cells = rec_cells['cell'].to_numpy()[hit]
        uv = normalise(xy[hit], box_frame(polygon, xy[hit], entry['orientation'], box_from))
        ids, rects = (equal_count_bins(uv, cells_per_bin) if cells_per_bin
                      else grid_bins(uv, bins))
        idx = np.array([row_of[c] for c in cells], dtype=int)
        tracked, firing = present[idx], (signals[idx] > noise_floor) & present[idx]
        n_cells = np.zeros((len(rects), len(timepoints)), dtype=int)
        n_firing = np.zeros_like(n_cells)
        np.add.at(n_cells, ids, tracked)
        np.add.at(n_firing, ids, firing)
        # one row per bin x frame
        k = np.repeat(np.arange(len(rects)), len(timepoints))
        j = np.tile(np.arange(len(timepoints)), len(rects))
        n, f = n_cells.ravel(), n_firing.ravel()
        with np.errstate(invalid='ignore', divide='ignore'):
            frac = np.where(n > 0, f / n, np.nan)
        above = np.clip(signals[idx] - noise_floor, 0, None) * present[idx]
        mr = rects[ids] if len(ids) else np.zeros((0, 6))
        members.append(pd.DataFrame({
            'recording': name, 'segment': label, 'cell': cells, 'bin': ids,
            'u0': mr[:, 2], 'u1': mr[:, 3], 'v0': mr[:, 4], 'v1': mr[:, 5],
            'u': uv[:, 0] if len(uv) else [], 'v': uv[:, 1] if len(uv) else [],
            'total': above.sum(axis=1),
            'level': rec_cells['level'].to_numpy()[hit] if 'level' in rec_cells else np.nan}))
        r = rects[k]
        rows.append(pd.DataFrame({
            'recording': name, 'segment': label, 't': timepoints[j].astype(int),
            't_rel': t_rel[j], 'time_bin': t_bin[j], 'bin': k,
            'bin_x': r[:, 0].astype(int), 'bin_y': r[:, 1].astype(int),
            'u0': r[:, 2], 'u1': r[:, 3], 'v0': r[:, 4], 'v1': r[:, 5],
            'u_centre': (r[:, 2] + r[:, 3]) / 2, 'v_centre': (r[:, 4] + r[:, 5]) / 2,
            'n_cells': n, 'n_firing': f, 'frac_firing': frac}))
    table = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if with_members:
        return table, pd.concat(members, ignore_index=True) if members else pd.DataFrame()
    return table


BIN_KEYS = ['bin', 'u0', 'u1', 'v0', 'v1']
RASTER = 100       # pixels per axis an equal-count map is painted on, to overlay and compare


def paint(bins_df, value, res):
    """A res x res map (rows up from v = -1) with each bin's rectangle filled with its value;
    NaN where no bin has one."""
    m = np.full((res, res), np.nan)
    edge = lambda w: int(np.clip(round((w + 1) / 2 * res), 0, res))
    for u0, u1, v0, v1, val in bins_df[['u0', 'u1', 'v0', 'v1', value]].itertuples(index=False):
        if np.isfinite(val):
            m[edge(v0):max(edge(v1), edge(v0) + 1), edge(u0):max(edge(u1), edge(u0) + 1)] = val
    return m


def pooled_bins(table):
    """Per recording and bin, the share firing over the frames in table, pooled as cell-frames
    (sum of firing / sum of tracked)."""
    s = table.groupby(['recording'] + BIN_KEYS, sort=False)[['n_firing', 'n_cells']].sum().reset_index()
    s['frac'] = np.where(s['n_cells'] > 0, s['n_firing'] / s['n_cells'].where(s['n_cells'] > 0), np.nan)
    return s


def pooled_maps(table, res):
    """{recording: res x res map} of pooled_bins: on a bins grid with res = bins, a pixel is a
    bin; equal-count bins are painted on a finer raster."""
    return {n: paint(g, 'frac', res) for n, g in pooled_bins(table).groupby('recording', sort=False)}


def map_correlations(maps, unit='n_bins'):
    """Pearson r of every pair of maps over the pixels both hold cells in (NaN under 3).  On a
    grid a pixel is a bin; on the raster of equal-count bins, larger bins weigh more."""
    rows = []
    for a, b in itertools.combinations(maps, 2):
        ok = np.isfinite(maps[a]) & np.isfinite(maps[b])
        x, y = maps[a][ok], maps[b][ok]
        r = (np.corrcoef(x, y)[0, 1] if ok.sum() >= 3 and x.std() > 0 and y.std() > 0
             else np.nan)
        rows.append({'recording_a': a, 'recording_b': b, unit: int(ok.sum()), 'r': r})
    return pd.DataFrame(rows, columns=['recording_a', 'recording_b', unit, 'r'])


def with_mean(maps, names, res):
    """maps in the recordings' order, a missing one all NaN, then 'mean of recordings' (a
    pixel no recording has cells in stays NaN)."""
    maps = {n: maps.get(n, np.full((res, res), np.nan)) for n in names}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        maps['mean of recordings'] = np.nanmean(np.stack(list(maps.values())), axis=0)
    return maps


def map_row(maps, title, colour_label, vmax=1.0, titles=None, outlines=None, labels=None,
            cmap='viridis'):
    """One figure: the maps side by side on one colour scale, NaN grey.  titles overrides the
    panel names; outlines ({name: bins rows}) draws equal-count bins' edges; labels ({name:
    [(u, v, text)]}) writes values into the bins."""
    cmap = (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap).copy()
    cmap.set_bad('#d3d1cb')
    fig, axes = plt.subplots(1, len(maps), squeeze=False, layout='constrained',
                             figsize=(2.6 * len(maps) + 1.2, 3.4))
    for ax, (name, m) in zip(axes[0], maps.items()):
        im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, origin='lower', extent=(-1, 1, -1, 1),
                       interpolation='none')
        if outlines and name in outlines:
            for u0, u1, v0, v1 in outlines[name][['u0', 'u1', 'v0', 'v1']].itertuples(index=False):
                ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, lw=0.4,
                                       color='white'))
        for u, v, text in (labels or {}).get(name, []):
            ax.text(u, v, text, ha='center', va='center', fontsize=6, color='white',
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground='black')])
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=6)
        ax.set_title((titles or {}).get(name, name), fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label=colour_label)
    fig.suptitle(title, fontsize=10)
    return fig


class Layout:
    """How a run bins its segments, for the figures: the raster each map is drawn on, the
    words that name the binning, and the bin edges to outline (equal-count bins only)."""

    def __init__(self, table, bins, cells_per_bin):
        self.res = RASTER if cells_per_bin else bins
        self.words = (f'bins of about {cells_per_bin} cells' if cells_per_bin
                      else f'{bins}x{bins} bins')
        self.unit = 'n_raster_pixels' if cells_per_bin else 'n_bins'
        self.outlines = ({n: g.drop_duplicates('bin') for n, g in table.groupby('recording', sort=False)}
                         if cells_per_bin else None)


def plot_segment_maps(table, label, names, layout, path_stem):
    """The whole-window map of every recording side by side, then their mean; the time bins
    stay in segment_activity.csv."""
    from ms2.cell_activity import save_figure
    fig = map_row(with_mean(pooled_maps(table, layout.res), names, layout.res),
                  f'Segment {label}, whole window: cells firing on the segment grid '
                  f'({layout.words}; grey = no cells)', 'share of the bin\'s cells firing',
                  outlines=layout.outlines)
    save_figure(fig, path_stem)


GIF_STEP = 0.1     # GIF frames every 10% of each recording's window


def plot_segment_gif(table, label, names, layout, path):
    """segment_<L>_activity.gif: one frame per 10% of the window, 0% to 100%.  A recording of
    n frames shows its frame round(p * (n - 1)) at p, so recordings of different lengths sit at
    the same point of their window; each panel names its frame."""
    from ms2.utils.gif_utils import create_gif_from_figures
    frames = {n: np.sort(table.loc[table['recording'] == n, 't'].unique()) for n in names}
    figures = []
    for p in np.round(np.arange(0, 1 + GIF_STEP / 2, GIF_STEP), 2):
        picked = {n: ts[int(np.floor(p * (len(ts) - 1) + 0.5))] for n, ts in frames.items()}
        at = table[table['t'] == table['recording'].map(picked)]
        figures.append(map_row(
            with_mean(pooled_maps(at, layout.res), names, layout.res),
            f'Segment {label} at {p:.0%} of the window: cells firing on the segment grid '
            f'({layout.words}; grey = no cells tracked)', 'share of the bin\'s cells firing',
            titles={n: f'{n} (t={t})' for n, t in picked.items()}, outlines=layout.outlines))
    create_gif_from_figures(figures, path, fps=1)
    for fig in figures:
        plt.close(fig)


def mean_cells_per_bin(table):
    """Per recording and bin: the cells tracked in it, averaged over the window's frames."""
    return (table.groupby(['recording', 'bin', 'bin_x', 'bin_y', 'u0', 'u1', 'v0', 'v1',
                           'u_centre', 'v_centre'], sort=False)
            ['n_cells'].mean().rename('mean_cells').reset_index())


def plot_cells_per_bin(counts, label, names, layout, path_stem):
    from ms2.cell_activity import save_figure
    shown = counts.assign(mean_cells=counts['mean_cells'].where(counts['mean_cells'] > 0))
    maps = with_mean({n: paint(g, 'mean_cells', layout.res)
                      for n, g in shown.groupby('recording', sort=False)}, names, layout.res)
    # the number in each bin, while there are few enough bins to read them
    labels = {n: [(u, v, f'{c:.1f}') for u, v, c in g[['u_centre', 'v_centre', 'mean_cells']]
                  .itertuples(index=False) if np.isfinite(c)]
              for n, g in shown.groupby('recording', sort=False) if len(g) <= 64}
    fig = map_row(maps, f'Segment {label}: cells tracked per bin, averaged over the window '
                  f'({layout.words}; grey = none)', 'cells per bin',
                  vmax=np.nanmax([np.nanmax(m) for m in maps.values()]), outlines=layout.outlines,
                  labels=labels)
    save_figure(fig, path_stem, eps=False)


def plot_segment_boxes(entries, cells, labels, path_stem, box_from='outline'):
    """Every recording in its aligned view: all its cells, each segment's outline, the box the
    grid is cut from, and the box's (u, v) corners -- to check the normalisation by eye."""
    from ms2.cell_activity import save_figure
    from ms2.cell_activity_compare import alignment_matrix
    fig, axes = plt.subplots(1, len(entries), figsize=(4 * len(entries), 4.2), squeeze=False)
    for ax, (name, entry) in zip(axes[0], entries.items()):
        c = cells[cells['recording'] == name]
        xy = c[['x', 'y']].to_numpy()
        ax.scatter(*(xy @ alignment_matrix(**entry['orientation']).T).T, s=3, color='#d3d1cb',
                   zorder=1)
        for label, polygon in entry['segments'].items():
            if labels and label not in labels:
                continue
            hit = inside(polygon, xy)
            m, centre, half = box_frame(polygon, xy[hit], entry['orientation'], box_from)
            ax.scatter(*(xy[hit] @ m.T).T, s=6, color='#2a78d6', zorder=2)
            p = np.vstack([polygon, polygon[:1]]) @ m.T
            ax.plot(p[:, 0], p[:, 1], color='#eb6834', lw=1.2, zorder=3)
            ax.add_patch(Rectangle(centre - half, *(2 * half), fill=False, ls='--', lw=1,
                                   color='#0b0b0b', zorder=4))
            ax.text(*centre, label, ha='center', va='center', fontsize=11, weight='bold', zorder=5)
            for su, sv in itertools.product((-1, 1), (-1, 1)):
                # aligned y points down, so v = +1 is the box's top edge
                ax.text(centre[0] + su * half[0], centre[1] - sv * half[1], f'({su},{sv})',
                        fontsize=5, color='#52514e', ha='center', va='center', zorder=5)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        ax.set_title(name, fontsize=9)
    fig.suptitle('Segments in the aligned view: outline (orange), grid box (dashed) and its '
                 '(u, v) corners, cells inside (blue)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, path_stem, eps=False)


# The analysis page's cluster colours: each cluster's maps run from near-black (0) to its colour.
CLUSTER_COLOURS = ('#bdbdbd', '#bc8f6f', '#4fc3f7', '#66bb6a', '#ef5350', '#ab47bc')


def member_maps(members, value, res, level=None):
    """{recording: res x res map} from the per-cell table: each bin's mean of value, or with
    level the share of the bin's cells at that level."""
    from ms2.cell_activity import LEVEL_NAMES  # noqa: F401  (levels are its keys)
    out = {}
    for name, g in members.groupby('recording', sort=False):
        per = g.groupby(BIN_KEYS, sort=False)
        v = (per['level'].apply(lambda x: (x == level).mean()) if level is not None
             else per[value].mean()).rename('value').reset_index()
        out[name] = paint(v, 'value', res)
    return out


def plot_member_maps(members, label, names, layout, path_stem):
    """segment_<L>_intensity: the mean total intensity per cell in each bin; and
    segment_<L>_clusters: a row per cluster, each bin's share of cells at that level, every row
    in its cluster's colour.  Returns {metric: {recording: map}} for the correlations."""
    from matplotlib.colors import LinearSegmentedColormap
    from ms2.cell_activity import LEVEL_NAMES, save_figure
    maps = {'intensity': member_maps(members, 'total', layout.res)}
    wm = with_mean(maps['intensity'], names, layout.res)
    vmax = np.nanmax([np.nanmax(m) if np.isfinite(m).any() else 0 for m in wm.values()]) or 1
    fig = map_row(wm, f'Segment {label}, whole window: mean intensity per cell ({layout.words}; '
                  'grey = no cells)', 'AU above the floor, summed over the window', vmax=vmax,
                  outlines=layout.outlines)
    save_figure(fig, path_stem + '_intensity')
    # silent .. high always; a fourth active tier only when the ladder has one
    levels = [l for l in sorted(LEVEL_NAMES) if l <= 4 or (members['level'] == l).any()]
    columns = names + ['mean of recordings']
    fig, axes = plt.subplots(len(levels), len(columns), squeeze=False, layout='constrained',
                             figsize=(2.3 * len(columns) + 1.6, 2.3 * len(levels) + 0.6))
    for r, level in enumerate(levels):
        maps[f'cluster:{LEVEL_NAMES[level]}'] = m = member_maps(members, None, layout.res, level)
        cmap = LinearSegmentedColormap.from_list(str(level), ['#18181c', CLUSTER_COLOURS[level]])
        cmap.set_bad('#d3d1cb')
        for c, (name, grid) in enumerate(with_mean(m, names, layout.res).items()):
            ax = axes[r, c]
            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, origin='lower', extent=(-1, 1, -1, 1),
                           interpolation='none')
            if layout.outlines and name in layout.outlines:
                for u0, u1, v0, v1 in layout.outlines[name][['u0', 'u1', 'v0', 'v1']].itertuples(index=False):
                    ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, fill=False, lw=0.4, color='white'))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=8)
            if c == 0:
                ax.set_ylabel(LEVEL_NAMES[level], fontsize=8, color=CLUSTER_COLOURS[level] if level > 0 else '#52514e')
        fig.colorbar(im, ax=axes[r, :], shrink=0.8, label='share', aspect=10)
    fig.suptitle(f'Segment {label}: share of each bin\'s cells in every cluster ({layout.words}; '
                 'grey = no cells)', fontsize=10)
    save_figure(fig, path_stem + '_clusters')
    return maps


def analyse_segments(cmp_dir, bins=8, time_bins=4, labels=None, cells_per_bin=None,
                     box_from='outline'):
    """Everything listed in the module docstring, for the segments of cmp_dir/segments.json
    (labels picks some of them).  cells_per_bin cuts each segment into bins of about that many
    cells instead of the bins x bins grid; box_from='cells' cuts each grid from the cells inside
    the outline (see box_frame).  Returns the paths written."""
    if bins < 1 or time_bins < 1 or (cells_per_bin is not None and cells_per_bin < 1):
        raise ValueError('bins, time bins and cells per bin must be 1 or more')
    if box_from not in ('outline', 'cells'):
        raise ValueError(f"box_from is 'outline' or 'cells', got {box_from!r}")
    unknown = set(labels or ()) - set(SEGMENT_LABELS)
    if unknown:
        raise ValueError(f'segment {", ".join(sorted(unknown))} is not one of {", ".join(SEGMENT_LABELS)}')
    from ms2.cell_activity_compare import comparison_alignment
    with open(os.path.join(cmp_dir, 'recordings.json')) as f:
        recordings = json.load(f)
    segments = load_segments(cmp_dir)
    if not segments:
        raise FileNotFoundError(f'no segments marked in {cmp_dir}: mark them on the Segments page')
    cells = pd.read_csv(os.path.join(cmp_dir, 'cells.csv'))
    alignment = comparison_alignment(cmp_dir, recordings) or {}
    # the orientation the outlines were drawn at, else the comparison's: both are 'aligned'
    entries = {n: {'orientation': alignment.get(n, {'flip': False, 'rotate': 0}), **segments[n]}
               for n in recordings if n in segments}
    names = list(entries)
    pairs = [segment_activity(n, recordings[n], entries[n], cells[cells['recording'] == n], bins,
                              time_bins, labels, cells_per_bin, box_from, with_members=True)
             for n in names]
    table = pd.concat([t for t, _ in pairs], ignore_index=True)
    members = pd.concat([m for _, m in pairs], ignore_index=True)
    if table.empty:
        raise ValueError(f'segment {", ".join(labels or SEGMENT_LABELS)} is not marked in {cmp_dir}')

    out = os.path.join(cmp_dir, ACTIVITY_DIR)
    os.makedirs(out, exist_ok=True)
    written = [os.path.join(out, 'segment_activity.csv'), os.path.join(out, 'segment_cells.csv')]
    table.to_csv(written[0], index=False)
    members.to_csv(written[1], index=False)
    for label in sorted(table['segment'].unique()):
        t = table[table['segment'] == label]
        shown = [n for n in names if (t['recording'] == n).any()]
        layout = Layout(t, bins, cells_per_bin)
        stem = os.path.join(out, f'segment_{label}_activity')
        plot_segment_maps(t, label, shown, layout, stem)
        corr = os.path.join(out, f'segment_{label}_correlation.csv')
        map_correlations(pooled_maps(t, layout.res), layout.unit).to_csv(corr, index=False)
        gif = os.path.join(out, f'segment_{label}_activity.gif')
        plot_segment_gif(t, label, shown, layout, gif)
        counts = mean_cells_per_bin(t)
        counts_csv = os.path.join(out, f'segment_{label}_cells_per_bin.csv')
        counts.to_csv(counts_csv, index=False)
        counts_stem = os.path.join(out, f'segment_{label}_cells_per_bin')
        plot_cells_per_bin(counts, label, shown, layout, counts_stem)
        written += [stem + '.png', gif, counts_stem + '.png', corr, counts_csv]
        # mean intensity per cell and the share in each cluster, with their correlations
        if 'level' in cells:
            base = os.path.join(out, f'segment_{label}')
            maps = plot_member_maps(members[members['segment'] == label], label, shown, layout, base)
            for metric, by_rec in maps.items():
                name = 'intensity' if metric == 'intensity' else 'clusters'
                part = map_correlations(by_rec, layout.unit).assign(metric=metric)
                path = os.path.join(out, f'segment_{label}_{name}_correlation.csv')
                if os.path.exists(path) and name == 'clusters' and metric != 'cluster:silent':
                    part = pd.concat([pd.read_csv(path), part], ignore_index=True)
                part.to_csv(path, index=False)
            written += [base + '_intensity.png', base + '_clusters.png',
                        os.path.join(out, f'segment_{label}_intensity_correlation.csv'),
                        os.path.join(out, f'segment_{label}_clusters_correlation.csv')]
    stem = os.path.join(out, 'segment_boxes')
    plot_segment_boxes(entries, cells, labels, stem, box_from)
    written.append(stem + '.png')
    with open(os.path.join(out, 'params.json'), 'w') as f:
        json.dump({'bins': None if cells_per_bin else bins, 'cells_per_bin': cells_per_bin,
                   'time_bins': time_bins, 'box_from': box_from,
                   'segments': sorted(table['segment'].unique())},
                  f, indent=2)
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmp_dir', help='comparison folder with recordings.json, cells.csv and segments.json')
    size = p.add_mutually_exclusive_group()
    size.add_argument('--bins', type=int, default=8, help='grid bins per axis (default %(default)s)')
    size.add_argument('--cells-per-bin', type=int,
                      help='instead of a grid, bins of uneven size holding about this many cells '
                           'each, so a segment with more cells is binned finer')
    p.add_argument('--time-bins', type=int, default=4,
                   help='parts the stage window is pooled into (default %(default)s)')
    p.add_argument('--segments', nargs='+', choices=SEGMENT_LABELS,
                   help='only these segments (default: every one marked)')
    p.add_argument('--box-from', choices=('outline', 'cells'), default='outline',
                   help="cut each segment's grid from its outline (default) or the cells inside it")
    args = p.parse_args()
    for path in analyse_segments(args.cmp_dir, args.bins, args.time_bins, args.segments,
                                 args.cells_per_bin, args.box_from):
        print(path)


if __name__ == '__main__':
    main()
