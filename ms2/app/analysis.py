"""The analysis page's data: every recording's pipeline results, and per stage window the cells
the page filters, clusters and maps live in the browser.

A result is one gene_expression*/ folder holding an expression matrix, with the tracks and
masks it was measured on.  Two layouts are found: the pipeline's (<rec>/unified/,
<rec>/gene_expression/) and the older hand-run one (<rec>/unify/, gene_expression_roi_selection/,
or a results folder whose roi_selection.json names tracks kept elsewhere).

A sample is a result and a window t_start..t_end.  Its data is one row per cell measured in
the matrix and tracked somewhere in the window: the trace over the window, the share of the
window it is tracked in, its volume CV (std / mean over its tracked window frames), its mean
position, and its position at t_start and t_end -- what an ROI drawn on those frames tests.
Positions and volumes come from the masks, read once per recording and cached.
"""
import glob
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from ms2.gene_expression.expression_matrix import FIXED_NAME, RESULTS_NAME

MAX_DEPTH = 6        # folders below a root searched for results
CACHE_DIR = '_cache'


def resolve(csv_path):
    """A result's inputs from its expression matrix: {csv, name, folder, rec_dir, tracklets,
    masks_dir, projection, roi}, None where not found."""
    from ms2.viewer.loaders import find_results
    gdir = os.path.dirname(os.path.abspath(csv_path))
    rec_dir = os.path.dirname(gdir)
    found = find_results(rec_dir)
    tracklets, masks_dir, roi = found['tracklets'], found['masks'], found['roi']
    roi_info = {}
    if roi:
        with open(roi) as f:
            roi_info = json.load(f)
    if not tracklets and roi_info.get('tracklets') and os.path.exists(roi_info['tracklets']):
        # a results folder of its own, measured on tracks kept with an earlier run
        tracklets = roi_info['tracklets']
        split = os.path.join(os.path.dirname(tracklets), 'masks_split')
        masks_dir = split if os.path.isdir(split) else None
    rec = {'csv': os.path.abspath(csv_path), 'name': os.path.basename(rec_dir),
           'folder': os.path.basename(gdir), 'rec_dir': rec_dir, 'tracklets': tracklets,
           'masks_dir': masks_dir, 'roi': roi,
           'roi_window': ([roi_info['t_first'], roi_info['t_last']]
                          if 't_first' in roi_info and 't_last' in roi_info else None),
           'projection': None}
    if masks_dir:
        from ms2.cell_activity_compare import cell_projection
        try:
            rec['projection'] = cell_projection(
                {'masks_dir': masks_dir, 'projection': roi_info.get('projection')
                 if roi_info.get('projection') and os.path.exists(roi_info['projection']) else None})
        except FileNotFoundError:
            pass
    return rec


def find_matrices(root, depth=MAX_DEPTH):
    """Every expression matrix under root, the fixed one where a folder has both."""
    found = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath[len(root):].count(os.sep) >= depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith(('.', '_')) and d != CACHE_DIR
                       and d not in ('masks', 'masks_split', 'preprocess', 'tracklets')]
        if os.path.basename(dirpath).startswith('gene_expression'):
            dirnames[:] = []                     # its per-cell files are not results
            for name in (FIXED_NAME, RESULTS_NAME):
                if name in filenames:
                    found.append(os.path.join(dirpath, name))
                    break
    return sorted(found)


def recording_info(rec):
    """What the browser lists for a result: frames, cells measured, tracks, and whether the
    page can analyse it (tracks, masks and an image found)."""
    with open(rec['csv']) as f:
        header = f.readline().strip().split(',')
        n_frames = sum(1 for _ in f)
    info = {**rec, 'n_frames': n_frames, 'n_cells': len(header) - 1, 'n_tracks': None,
            'problem': None}
    if not rec['tracklets'] or not rec['masks_dir']:
        info['problem'] = 'its tracks or masks were not found'
    elif not rec['projection']:
        info['problem'] = 'no cell-channel image was found to draw on'
    if rec['tracklets'] and os.path.exists(rec['tracklets']):
        with open(rec['tracklets']) as f:
            info['n_tracks'] = len(json.load(f))
    # the orientation saved with the recording's ROI, the page's starting point
    from ms2.roi_selection import load_orientation
    info['orientation'] = load_orientation(rec['roi']) if rec['roi'] else None
    info['height'] = info['width'] = None
    if rec['projection']:
        from ms2.roi_selection import load_projection
        info['height'], info['width'] = load_projection(rec['projection']).shape[1:]
    return info


def discover(roots):
    """Every result under roots, as recording_info rows; stage = the first folder below the
    root, e.g. STAGE-13."""
    rows = []
    for root in roots:
        for csv_path in find_matrices(root):
            try:
                info = recording_info(resolve(csv_path))
            except (OSError, ValueError, KeyError) as exc:
                info = {'csv': csv_path, 'name': os.path.basename(os.path.dirname(os.path.dirname(csv_path))),
                        'folder': os.path.basename(os.path.dirname(csv_path)), 'problem': str(exc)}
            rel = os.path.relpath(csv_path, root).split(os.sep)
            info['group'] = rel[0] if len(rel) > 3 else ''
            rows.append(info)
    return rows


def properties_cache(cache_root, masks_dir):
    """Where a recording's per-frame centroids and volumes are cached, keyed by its masks."""
    key = hashlib.sha1(os.path.abspath(masks_dir).encode()).hexdigest()[:16]
    return os.path.join(cache_root, CACHE_DIR, f'{key}_cell_properties.pkl')


def sample_data(rec, t_start, t_end, cache_root):
    """The cells of one window: see the module docstring.  Traces are rounded to 0.1 AU, with
    None where the cell is not tracked."""
    from ms2.cell_activity import load_window
    from ms2.track_diagnostics import cell_properties
    signals, present, cells, timepoints = load_window(rec['csv'], t_start, t_end, min_presence=0)
    with open(rec['tracklets']) as f:
        tracklets = json.load(f)
    props = cell_properties(rec['masks_dir'], properties_cache(cache_root, rec['masks_dir']))
    rows, traces = [], []
    for i, cell in enumerate(cells):
        track = tracklets.get(cell[len('cell_'):])
        if track is None:
            continue
        p = [props[t][track[t]] for t in timepoints
             if t < len(track) and track[t] > 0 and track[t] in props.get(t, {})]
        if not p:
            continue
        p = np.array(p)                       # z, y, x, volume per tracked window frame

        def at(t):
            if t < len(track) and track[t] > 0 and track[t] in props.get(t, {}):
                _, y, x, _ = props[t][track[t]]
                return [round(x, 1), round(y, 1)]
            return None
        def label(t):
            return int(track[t]) if t < len(track) and track[t] > 0 else None
        rows.append({'cell': cell, 'presence': round(float(present[i].mean()), 4),
                     'vol_cv': round(float(p[:, 3].std() / p[:, 3].mean()), 4),
                     'x': round(float(p[:, 2].mean()), 1), 'y': round(float(p[:, 1].mean()), 1),
                     'at_start': at(int(timepoints[0])), 'at_end': at(int(timepoints[-1])),
                     # its mask label on the window's first and last frame, to outline it there
                     'label_start': label(int(timepoints[0])), 'label_end': label(int(timepoints[-1]))})
        traces.append([round(float(v), 1) if ok else None for v, ok in zip(signals[i], present[i])])
    return {'timepoints': [int(t) for t in timepoints], 'cells': rows, 'traces': traces}


def label_png(masks_dir, t):
    """Frame t's segmentation projected over z -- each pixel the highest label above it -- as a
    PNG whose red and green channels hold the label (red * 256 + green), for the page to
    outline cells with."""
    import io
    from PIL import Image
    from ms2.viewer.loaders import MaskStore
    volume = MaskStore(masks_dir).get(t)
    if volume is None:
        raise FileNotFoundError(f'no masks for frame {t} in {masks_dir}')
    labels = volume.max(axis=0).astype(np.uint32)
    if labels.max() > 65535:
        raise ValueError(f'frame {t} has labels above 65535; the outline image holds 16 bits')
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1] = labels >> 8, labels & 255
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, 'PNG')
    return buf.getvalue()


def _walk_json(roots, filename, depth=MAX_DEPTH):
    """Every file of this name under roots, not inside per-cell or mask folders."""
    for root in roots:
        root = os.path.abspath(root)
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath[len(root):].count(os.sep) >= depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if not d.startswith(('.', '_')) and not d.startswith('gene_expression')
                           and d not in ('masks_split', 'preprocess', 'tracklets', 'calibration', 'cache')]
            if filename in filenames:
                yield os.path.join(dirpath, filename)


def outline_files(rec, roots):
    """The ROI and segment files that belong to a result, for the page to load one from:
    ROIs drawn on its tracks (an roi_selection.json naming them, or beside its results or its
    tracks), its own segments.json, and every comparison's segments.json with an entry for it."""
    tracks_rec = os.path.dirname(os.path.dirname(rec['tracklets'])) if rec.get('tracklets') else None
    near = {os.path.abspath(d) for d in (rec['rec_dir'], tracks_rec) if d}
    rois, segs = [], []
    for path in _walk_json(roots, 'roi_selection.json'):
        try:
            with open(path) as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        folder = os.path.dirname(path)
        mine = (r.get('tracklets') and rec.get('tracklets')
                and os.path.abspath(r['tracklets']) == os.path.abspath(rec['tracklets']))
        mine = mine or folder in near or os.path.dirname(folder) in near
        if mine and (r.get('roi_first') or r.get('roi_last')):
            rois.append({'path': path, 't_first': r.get('t_first'), 't_last': r.get('t_last')})
            beside = os.path.join(folder, 'segments.json')
            if os.path.exists(beside):
                with open(beside) as f:
                    if json.load(f).get('segments'):        # saved with the ROI, possibly empty
                        segs.append({'path': beside, 'name': None})
    for path in _walk_json(roots, 'segments.json'):
        folder = os.path.dirname(path)
        config = os.path.join(folder, 'recordings.json')
        if not os.path.exists(config):
            continue
        with open(config) as f:
            recordings = json.load(f)
        with open(path) as f:
            marked = json.load(f)
        for name, entry in recordings.items():
            if (os.path.abspath(entry.get('csv', '')) == rec['csv'] and name in marked
                    and marked[name].get('segments')):
                segs.append({'path': path, 'name': name})
    return {'rois': rois, 'segments': segs}


def read_outline_file(path, name=None):
    """An ROI file's outlines and frames, or the segments of one recording in a segments file
    (name picks its entry in a comparison's file)."""
    with open(path) as f:
        d = json.load(f)
    if os.path.basename(path) == 'roi_selection.json':
        return {'kind': 'roi', 'roi_first': d.get('roi_first'), 'roi_last': d.get('roi_last'),
                't_first': d.get('t_first'), 't_last': d.get('t_last')}
    entry = d[name] if name else d
    return {'kind': 'segments', 'segments': entry.get('segments', {}), 't': entry.get('t'),
            'orientation': entry.get('orientation')}


# ── saved analyses: <results root>/analyses/<name>/analysis.json, exported in place ──
ANALYSES_DIR = 'analyses'
ANALYSIS_NAME = 'analysis.json'
SAFE_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9 ._-]{0,79}')


def analysis_dir(root, name):
    name = (name or '').strip()
    if not SAFE_NAME.fullmatch(name):
        raise ValueError(f'{name!r} is not a usable name: letters, digits, space, . _ -, up to 80')
    return os.path.join(root, ANALYSES_DIR, name)


def save_analysis(root, name, analysis):
    """Write analysis.json -- the page's whole state -- into its own folder; returns the path."""
    import datetime
    folder = analysis_dir(root, name)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, ANALYSIS_NAME)
    with open(path, 'w') as f:
        json.dump({**analysis, 'name': name.strip(), 'saved': datetime.datetime.now().isoformat(timespec='seconds')},
                  f, indent=2)
    return path


def list_analyses(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(glob.escape(root), ANALYSES_DIR, '*', ANALYSIS_NAME))):
        try:
            with open(path) as f:
                a = json.load(f)
        except (OSError, ValueError):
            continue
        rows.append({'name': os.path.basename(os.path.dirname(path)), 'saved': a.get('saved'),
                     'mode': a.get('mode'), 'stage': a.get('stage'),
                     'samples': [f"{s.get('name')} · {s.get('stage')}" for s in a.get('samples', [])]})
    return sorted(rows, key=lambda r: r['saved'] or '', reverse=True)


def load_analysis(root, name):
    with open(os.path.join(analysis_dir(root, name), ANALYSIS_NAME)) as f:
        return json.load(f)


def sample_keys(analysis):
    """A unique, file-name-safe key per sample: the recording's name in a stage comparison,
    name_stageN for the stages of one recording."""
    keys, seen = [], set()
    for s in analysis['samples']:
        base = s['name'] if analysis.get('mode') != 'single' else f"{s['name']}_stage{s['stage']}"
        base = ''.join(c if c.isalnum() or c in '._-' else '_' for c in base)
        key, n = base, 2
        while key in seen:
            key, n = f'{base}_{n}', n + 1
        seen.add(key)
        keys.append(key)
    return keys


def export_config(analysis):
    """{key: recording dict} for ms2/cell_activity_compare.py: every sample's window, filters,
    ROI and orientation, and the noise floor and blip height set on the page."""
    f, c = analysis.get('filters', {}), analysis.get('cluster', {})
    config = {}
    for key, s in zip(sample_keys(analysis), analysis['samples']):
        rec = resolve(s['csv'])
        entry = {'csv': rec['csv'], 'tracklets': rec['tracklets'], 'masks_dir': rec['masks_dir'],
                 't_start': int(s['t0']), 't_end': int(s['t1']),
                 'min_presence': float(f.get('presence', 0.9)),
                 'noise_floor': float(c.get('floor', 20)), 'noise_peak': float(c.get('noise_peak', 10)),
                 'orientation': s.get('orientation') or {'flip': False, 'rotate': 0}}
        if f.get('vol_cv') is not None:           # off is saved as null (the page's infinity)
            entry['max_volume_cv'] = float(f['vol_cv'])
        roi = s.get('roi') or {}
        if roi.get('first') or roi.get('last'):
            entry['roi_first'], entry['roi_last'] = roi.get('first'), roi.get('last')
        if rec['projection']:
            entry['projection'] = rec['projection']
        config[key] = entry
    return config


def export(folder):
    """Every figure and table of a saved analysis, written into its folder: the comparison
    (cells, metrics, stats, heatmaps, aligned map) and, where segments are outlined, the segment
    histograms -- with the settings the page had."""
    from ms2.cell_activity_compare import compare_recordings
    from ms2.cell_activity_segments import analyse_segments
    with open(os.path.join(folder, ANALYSIS_NAME)) as f:
        analysis = json.load(f)
    config = export_config(analysis)
    with open(os.path.join(folder, 'recordings.json'), 'w') as f:
        json.dump(config, f, indent=2)
    c = analysis.get('cluster', {})
    title = (f"Stage {analysis.get('stage')}" if analysis.get('mode') != 'single'
             else analysis['samples'][0]['name'] if analysis['samples'] else None)
    cuts = None if c.get('auto', True) else [x for x in c.get('cuts', []) if x is not None]
    compare_recordings(config, folder, title=title, level_cuts=cuts)
    write_outlines(analysis, config, folder)
    segments = {key: {'t': config[key]['t_start'], 'orientation': config[key]['orientation'],
                      'segments': s['segments']}
                for key, s in zip(config, analysis['samples']) if s.get('segments')}
    if segments:
        with open(os.path.join(folder, 'segments.json'), 'w') as f:
            json.dump(segments, f, indent=2)
        h = analysis.get('hist', {})
        by_count = h.get('mode') == 'count'
        analyse_segments(folder, bins=int(h.get('size', 4)) if not by_count else 8, time_bins=4,
                         cells_per_bin=int(h.get('size', 4)) if by_count else None,
                         box_from=analysis.get('box_from', 'cells'))
    print('exported to', folder)


def write_outlines(analysis, config, folder):
    """outlines.json -- every sample's ROI and segment outlines, in image pixels -- and
    outlines_<sample>.png: its first window frame, turned as on the page, with them drawn on."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Affine2D
    from ms2.cell_activity import save_figure
    from ms2.cell_activity_compare import alignment_matrix, stage_frame
    seg_colours = {'2': '#e8a33d', '3': '#4caf50', '4': '#2a9fd6', '5': '#e87ba4'}
    outlines = {}
    for key, s in zip(config, analysis['samples']):
        roi = s.get('roi') or {}
        outlines[key] = {'recording': s['name'], 'stage': s.get('stage'), 'csv': s['csv'],
                         't_start': config[key]['t_start'], 't_end': config[key]['t_end'],
                         'orientation': config[key]['orientation'],
                         'roi_first': roi.get('first'), 'roi_last': roi.get('last'),
                         'segments': s.get('segments') or {}}
        if 'projection' not in config[key]:
            continue
        frame = stage_frame(config[key])
        h, w = frame.shape
        m = alignment_matrix(**config[key]['orientation'])
        affine = np.eye(3)
        affine[:2, :2] = m
        centre = np.array([(w - 1) / 2, (h - 1) / 2])
        fig, ax = plt.subplots(figsize=(6, 6))
        lo, hi = np.percentile(frame, [1, 99.5])
        ax.imshow(frame, cmap='gray', vmin=lo, vmax=hi, extent=(-w / 2, w / 2, h / 2, -h / 2),
                  transform=Affine2D(affine) + ax.transData)
        def draw(points, colour, label, dashed=False):
            p = (np.asarray(points, dtype=float) - centre) @ m.T
            p = np.vstack([p, p[:1]])
            ax.plot(p[:, 0], p[:, 1], color=colour, lw=1.6, ls='--' if dashed else '-', label=label)
        if roi.get('first'):
            draw(roi['first'], '#ffffff', f'ROI, start frame {config[key]["t_start"]}')
        if roi.get('last'):
            draw(roi['last'], '#bbbbbb', f'ROI, end frame {config[key]["t_end"]}', dashed=True)
        for label, pts in sorted((s.get('segments') or {}).items()):
            draw(pts, seg_colours.get(label, '#ffcc00'), f'segment {label}')
            c = ((np.asarray(pts, dtype=float) - centre) @ m.T).mean(axis=0)
            ax.text(*c, label, color=seg_colours.get(label, '#ffcc00'), fontsize=14, weight='bold',
                    ha='center', va='center')
        r = np.hypot(w, h) / 2
        ax.set(xlim=(-r, r), ylim=(r, -r))
        ax.set_aspect('equal')
        ax.axis('off')
        if ax.get_legend_handles_labels()[0]:
            # dark, so the white ROI line shows in it too
            ax.legend(loc='lower left', fontsize=7, facecolor='#202020', edgecolor='#555555',
                      labelcolor='white', framealpha=0.9)
        ax.set_title(f'{s["name"]} · stage {s.get("stage")} · frame {config[key]["t_start"]}', fontsize=9)
        save_figure(fig, os.path.join(folder, f'outlines_{key}'), eps=False)
    with open(os.path.join(folder, 'outlines.json'), 'w') as f:
        json.dump(outlines, f, indent=2)


# what a results zip holds: settings, outlines, figures and tables -- not the caches, the kept
# tracks or the eps copies of the figures
ZIP_SKIP_DIRS = {'cache', 'tracklets', CACHE_DIR}
ZIP_TYPES = ('.json', '.png', '.gif', '.csv', '.txt')
# the main results, at the top of the zip; everything else goes in work_product/
ZIP_MAIN = ('heatmap_active_pooled.png', 'level_composition.png', 'spatial_maps_aligned.png')
WORK_PRODUCT = 'work_product'


def zip_files(folder):
    """(path, name in the zip) of every file a results zip holds: the segment histograms
    (segment_activity/, without its GIFs) and ZIP_MAIN at the top, the rest -- settings,
    outlines, GIFs, other figures and tables -- in work_product/."""
    out = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [d for d in dirnames if d not in ZIP_SKIP_DIRS]
        for f in sorted(filenames):
            if not f.endswith(ZIP_TYPES) or f in ('job.json', 'exit_code'):
                continue
            path = os.path.join(dirpath, f)
            rel = os.path.relpath(path, folder)
            main = rel in ZIP_MAIN or (rel.startswith('segment_activity' + os.sep) and not f.endswith('.gif'))
            out.append((path, rel if main else os.path.join(WORK_PRODUCT, rel)))
    return sorted(out, key=lambda x: x[1])


def write_zip(folder, out):
    """The results zip of an exported analysis, written to out (a path or a file object);
    every entry sits under the analysis's name."""
    import zipfile
    name = os.path.basename(os.path.normpath(folder))
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for path, rel in zip_files(folder):
            z.write(path, os.path.join(name, rel))


def main():
    import argparse
    p = argparse.ArgumentParser(description='Export a saved analysis: python -m ms2.app.analysis export <folder>')
    p.add_argument('command', choices=['export'])
    p.add_argument('folder', help='the analysis folder holding analysis.json')
    export(p.parse_args().folder)


if __name__ == '__main__':
    main()
