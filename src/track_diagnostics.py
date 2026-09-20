"""Why the tracklets fragment: measurements over the masks and the tracks built from them.

Every number quoted when arguing about tracking quality should come from here rather than
from a one-off script, so it can be re-derived after a parameter change.  The five
analyses answer five separate questions:

    cell_properties      what is where -- centroid and volume of every label, cached
    split_merge_census   how unstable is the segmentation between consecutive frames
    motion_stats         how far do cells actually move, which sets every matching gate
    diagnose_deaths      when a track dies, what happened to the cell it was following
    stitch_feasibility   how many of those deaths a gap-bridging pass could repair

Run against the dataset the tracklets came from:

    python -m src.track_diagnostics --masks-dir .../masks --tracklets .../tracklets.json \
        --out-dir diagnostics --projection SUM_C2-....tif --pair 697 199
"""
import argparse
import collections
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, center_of_mass

from cell_tracking import evaluate_tracklets, extract_time_number, get_masks_paths
from src.viewer.loaders import EXITED

# A label smaller than this fraction of the frame's median is a segmentation fragment
# rather than a cell; 96% of labels that vanish between frames fall under it.
FRAGMENT_FRACTION = 0.4


def mask_paths_by_t(masks_dir):
    """{timepoint: path}, keyed by the t in the filename rather than by position."""
    return {extract_time_number(os.path.basename(p)): p for p in get_masks_paths(masks_dir)}


def load_masks(path):
    with np.load(path, allow_pickle=True) as f:
        return f['masks']


def cell_properties(masks_dir, cache_path=None):
    """{t: {label: (z, y, x, volume)}} for every label in every timepoint.

    One pass over the npz files, which is the expensive part of every analysis below --
    109 frames of 11x1024x1024 takes a couple of minutes -- so the result is pickled and
    reused.  Centroid and volume are computed the same way src/track.py already does it,
    with one center_of_mass call and one bincount per frame rather than per label.
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    props = {}
    for t, path in sorted(mask_paths_by_t(masks_dir).items()):
        m = load_masks(path)
        labels = np.unique(m)
        labels = labels[labels > 0]
        volumes = np.bincount(m.ravel())
        coms = center_of_mass(m > 0, m, labels) if len(labels) else []
        props[t] = {int(l): (float(c[0]), float(c[1]), float(c[2]), int(volumes[l]))
                    for l, c in zip(labels, coms) if not np.isnan(c).any()}

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(props, f)
    return props


def median_volume(props):
    """{t: median label volume}.  The cell-size prior, recomputed per frame because it
    drifts over the movie as the tissue develops."""
    return {t: float(np.median([p[3] for p in labels.values()])) if labels else 0.0
            for t, labels in props.items()}


def split_merge_census(masks_dir, props, min_fraction=0.25):
    """How much the labelling changes between consecutive frames, independent of tracking.

    A label at t+1 is a *child* of a label at t when it takes at least `min_fraction` of
    the parent's voxels, and a *parent* when it gives up that fraction of its own.  Two
    children means the segmentation split a cell, two parents means it merged two.  A
    label with no child has been lost, one with no parent is new.

    Returns (per_frame, lost_labels): counts per frame pair, and one row per lost label
    with its volume and z centroid -- the two numbers that say whether the losses are real
    cells or fragments at the edge of the stack.
    """
    by_t = mask_paths_by_t(masks_dir)
    ts = sorted(by_t)
    rows, lost_rows = [], []

    m0 = load_masks(by_t[ts[0]])
    for t, t1 in zip(ts, ts[1:]):
        m1 = load_masks(by_t[t1])
        labels0 = np.unique(m0)[1:]
        labels1 = np.unique(m1)[1:]
        v0 = np.bincount(m0.ravel())
        v1 = np.bincount(m1.ravel())

        # Joint histogram of the overlapping voxels: one pass instead of a pairwise loop.
        both = (m0 > 0) & (m1 > 0)
        stride = int(m1.max()) + 1
        key = m0[both].astype(np.int64) * stride + m1[both]
        pairs, overlap = np.unique(key, return_counts=True)
        parent = (pairs // stride).astype(int)
        child = (pairs % stride).astype(int)

        kept = parent[overlap >= min_fraction * v0[parent]]
        gave = child[overlap >= min_fraction * v1[child]]
        has_child, n_children = np.unique(kept, return_counts=True)
        has_parent, n_parents = np.unique(gave, return_counts=True)

        lost = np.setdiff1d(labels0, has_child)
        rows.append({
            't': t, 'n_labels': len(labels0), 'n_labels_next': len(labels1),
            'splits': int((n_children > 1).sum()), 'merges': int((n_parents > 1).sum()),
            'lost': len(lost), 'new': len(np.setdiff1d(labels1, has_parent)),
        })
        for label in lost:
            z, _, _, volume = props[t][int(label)]
            lost_rows.append({'t': t, 'label': int(label), 'volume': volume, 'z': z})

        m0 = m1

    return pd.DataFrame(rows), pd.DataFrame(lost_rows)


def load_tracklets(path):
    """{tracklet id: array of one label per frame}, with the int keys the json loses."""
    with open(path) as f:
        raw = json.load(f)
    return {int(k): np.asarray(v, dtype=int) for k, v in raw.items()}


def track_table(tracklets, props, ts):
    """One row per track: when it lives, how big it is, and how it ends.

    `mean_volume` is what separates the ~1060 fragment tracks from the real cells, and
    `exited` keeps border departures -- expected behaviour -- out of the failure counts.
    """
    rows = []
    for tid, labels in tracklets.items():
        active = np.where(labels > 0)[0]
        if not len(active):
            continue
        birth, death = int(active[0]), int(active[-1])
        volumes = [props[ts[i]][int(labels[i])][3] for i in active
                   if int(labels[i]) in props[ts[i]]]
        rows.append({
            'tid': tid, 'birth': birth, 'death': death, 'span': death - birth + 1,
            'n_active': len(active),
            'mean_volume': float(np.mean(volumes)) if volumes else 0.0,
            'exited': bool(death + 1 < len(labels) and labels[death + 1] == EXITED),
        })
    return pd.DataFrame(rows).set_index('tid')


def motion_stats(tracklets, props, ts):
    """Percentiles of the per-frame displacement of tracked cells.

    Measured only on consecutive frames where the track is active in both, so it describes
    how far a cell really travels.  This is what the matcher's distance gate should be set
    from: a gate far out in this tail is not tolerating motion, it is admitting junk.
    """
    dxy, dz, dvol = [], [], []
    for labels in tracklets.values():
        for i in range(len(labels) - 1):
            if labels[i] <= 0 or labels[i + 1] <= 0:
                continue
            a = props[ts[i]].get(int(labels[i]))
            b = props[ts[i + 1]].get(int(labels[i + 1]))
            if a is None or b is None:
                continue
            dxy.append(np.hypot(a[1] - b[1], a[2] - b[2]))
            dz.append(abs(a[0] - b[0]))
            dvol.append(abs(a[3] - b[3]) / max(a[3], 1))

    percentiles = [50, 75, 90, 95, 99, 99.9, 100]
    return pd.DataFrame({
        'dxy': np.percentile(dxy, percentiles),
        'dz': np.percentile(dz, percentiles),
        'rel_dvolume': np.percentile(dvol, percentiles),
    }, index=[f'p{p:g}' for p in percentiles]).assign(n_links=len(dxy))


def diagnose_deaths(tracklets, props, masks_dir, ts, min_volume=600, min_overlap=0.10):
    """For every track that dies mid-sequence, what became of the cell it was following.

    Follows the dying label into the next frame through the mask itself rather than
    through the tracker, then asks who owns the successor.  The answer separates the two
    failure modes that need completely different repairs: a `stolen` successor means the
    cell carries on under an id that already existed, which only merging duplicate ids can
    fix, while `vanished` means the object left the segmentation and needs bridging.

    Restricted to tracks of at least `min_volume` mean volume, since a fragment track
    dying is the segmentation being noisy rather than a tracking failure.
    """
    tracks = track_table(tracklets, props, ts)
    owner = collections.defaultdict(dict)          # frame index -> {label: tid}
    for tid, labels in tracklets.items():
        for i, label in enumerate(labels):
            if label > 0:
                owner[i][int(label)] = tid

    dying = tracks[(tracks.mean_volume >= min_volume)
                   & (tracks.death < len(ts) - 1)
                   & ~tracks.exited]
    by_death = collections.defaultdict(list)
    for tid, row in dying.iterrows():
        by_death[int(row.death)].append(tid)

    by_t = mask_paths_by_t(masks_dir)
    rows = []
    for i in sorted(by_death):
        m0, m1 = load_masks(by_t[ts[i]]), load_masks(by_t[ts[i + 1]])
        for tid in by_death[i]:
            label = int(tracklets[tid][i])
            hit = m0 == label
            n = int(hit.sum())
            counts = np.bincount(m1[hit].ravel())
            counts[0] = 0
            successor = int(counts.argmax()) if counts.size > 1 and counts.max() else 0
            fraction = counts[successor] / n if successor else 0.0

            if not successor or fraction < min_overlap:
                verdict, holder = 'vanished', None
            else:
                holder = owner[i + 1].get(successor)
                if holder is None:
                    verdict = 'orphan'
                elif tracks.at[holder, 'birth'] <= i:
                    verdict = 'stolen'
                elif tracks.at[holder, 'birth'] == i + 1:
                    verdict = 'new_track'
                else:
                    verdict = 'later_track'

            rows.append({
                'tid': tid, 'death_frame': i, 'label': label,
                'volume': props[ts[i]][label][3], 'z': props[ts[i]][label][0],
                'successor': successor, 'overlap': round(float(fraction), 3),
                'successor_volume': (props[ts[i + 1]][successor][3]
                                     if successor in props[ts[i + 1]] else 0),
                'successor_tid': holder, 'verdict': verdict,
            })
    return pd.DataFrame(rows)


def stitch_feasibility(tracklets, props, ts, max_dt=5, budget=7.0, max_dz=3.0,
                       min_volume=600):
    """How many mid-sequence deaths a gap-bridging pass could actually repair.

    Pairs each death with a later birth whose position is reachable -- `budget` pixels of
    XY travel per frame of gap, from the p95 of motion_stats -- and resolves the
    competition greedily, cheapest first, one partner each.  The gap between this count
    and the number of deaths is the part of the problem that bridging cannot reach.
    """
    tracks = track_table(tracklets, props, ts)
    full = tracks[tracks.mean_volume >= min_volume]
    deaths = full[(full.death < len(ts) - 1) & ~full.exited]
    births = collections.defaultdict(list)
    for tid, row in full[full.birth > 0].iterrows():
        births[int(row.birth)].append(tid)

    def centre(tid, i):
        return props[ts[i]].get(int(tracklets[tid][i]))

    candidates = []
    for tid, row in deaths.iterrows():
        i = int(row.death)
        a = centre(tid, i)
        if a is None:
            continue
        for dt in range(1, max_dt + 1):
            for other in births.get(i + dt, []):
                b = centre(other, i + dt)
                if b is None or other == tid:
                    continue
                dxy = np.hypot(a[1] - b[1], a[2] - b[2])
                if dxy <= budget * dt and abs(a[0] - b[0]) <= max_dz:
                    candidates.append((dxy / dt, tid, other, i, dt, dxy))

    candidates.sort()
    used_tail, used_head, pairs = set(), set(), []
    for score, tid, other, i, dt, dxy in candidates:
        if tid in used_tail or other in used_head:
            continue
        used_tail.add(tid)
        used_head.add(other)
        pairs.append({'tail': tid, 'head': other, 'death_frame': i, 'dt': dt,
                      'dxy': round(dxy, 2)})
    return pd.DataFrame(pairs), len(deaths)


def pair_overlay(tids, tracklets, props, masks_dir, ts, projection, out_path,
                 frames=None, pad=60):
    """Both tracks of a suspected duplicate id drawn on the cell channel over time.

    The counts say a pair is one cell split in two; this is what lets you see it.  Cropped
    to the union of the two cells so the nucleus fills the frame.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    a, b = (tracklets[t] for t in tids)
    active = [i for i in range(len(a)) if a[i] > 0 or b[i] > 0]
    if frames is None:
        frames = active[::max(1, len(active) // 9)][:9]

    by_t = mask_paths_by_t(masks_dir)
    fig, axes = plt.subplots(2, len(frames), figsize=(3 * len(frames), 6.5))
    for column, i in enumerate(frames):
        # Re-centre every panel: the tissue drifts far enough over the movie that one
        # fixed crop pushes the pair off the edge by the end.
        centres = [props[ts[i]][int(v[i])][1:3] for v in (a, b) if v[i] > 0]
        cy, cx = np.mean(centres, axis=0)
        y0, y1 = max(0, int(cy) - pad), int(cy) + pad
        x0, x1 = max(0, int(cx) - pad), int(cx) + pad

        m = load_masks(by_t[ts[i]])[:, y0:y1, x0:x1]
        img = np.asarray(projection[i])[y0:y1, x0:x1]
        lo, hi = np.percentile(img, [2, 99.5])
        grey = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.dstack([grey] * 3)
        for label, colour in ((a[i], (0, 1, 0)), (b[i], (1, 0, 0))):
            if label > 0:
                on = (m == label).any(axis=0)
                rgb[on] = 0.5 * np.array(colour) + 0.5 * rgb[on]
        axes[0, column].imshow(grey, cmap='gray')
        axes[0, column].set_title(f't={ts[i]}', fontsize=9)
        axes[1, column].imshow(rgb)
        axes[1, column].set_title(f'{tids[0]}={a[i]} (green)  {tids[1]}={b[i]} (red)',
                                  fontsize=7)
        for row in (0, 1):
            axes[row, column].axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=65)
    plt.close(fig)
    return out_path


def pair_evidence(tids, tracklets, props, masks_dir, ts):
    """Per-frame test of whether two tracks are one cell: do the labels touch, and is
    their combined volume one cell's worth rather than two?"""
    a, b = (tracklets[t] for t in tids)
    volumes = median_volume(props)
    by_t = mask_paths_by_t(masks_dir)
    rows = []
    for i in range(len(a)):
        if a[i] <= 0 or b[i] <= 0:
            continue
        m = load_masks(by_t[ts[i]])
        first, second = m == int(a[i]), m == int(b[i])
        pa, pb = props[ts[i]][int(a[i])], props[ts[i]][int(b[i])]
        rows.append({
            't': ts[i], f'label_{tids[0]}': int(a[i]), f'label_{tids[1]}': int(b[i]),
            'volume_sum': pa[3] + pb[3], 'median_cell_volume': volumes[ts[i]],
            'dxy': round(float(np.hypot(pa[1] - pb[1], pa[2] - pb[2])), 1),
            'touching': bool((binary_dilation(first) & second).any()),
        })
    return pd.DataFrame(rows)


def _table(frame, index=True):
    """A DataFrame as a markdown table.  Two lines of formatting rather than a dependency
    on tabulate, which this environment does not carry."""
    frame = frame.reset_index() if index else frame
    header = [str(c) for c in frame.columns]
    rows = [[f'{v:g}' if isinstance(v, float) else str(v) for v in row]
            for row in frame.itertuples(index=False)]
    return '\n'.join(['| ' + ' | '.join(header) + ' |',
                      '|' + '|'.join(['---'] * len(header)) + '|']
                     + ['| ' + ' | '.join(r) + ' |' for r in rows])


def write_report(out_dir, census, lost, motion, deaths, stitched, n_deaths, tracks,
                 cell_volume, n_frames, pair_rows=None, figure=None):
    """The findings as markdown, with the tables that back each one.

    `cell_volume` is the median volume of a *label*, which is what a track's mean volume
    has to be judged against.  The median over tracks is much lower and would flatter the
    numbers, because most tracks are the fragments being counted.
    """
    n = census[['splits', 'merges', 'lost', 'new']].sum()
    cells = census.n_labels.sum()
    verdicts = deaths.verdict.value_counts()
    fragments = tracks.mean_volume < FRAGMENT_FRACTION * cell_volume
    long_lived = int((tracks.n_active >= 0.9 * n_frames).sum())

    lines = [
        '# Tracking diagnostics', '',
        f'{len(tracks)} tracks over {n_frames} timepoints, '
        f'{census.n_labels.min()}-{census.n_labels.max()} labels per frame, median label '
        f'volume {cell_volume:.0f} voxels.  '
        f'Median track length {tracks.n_active.median():.0f} frames; only {long_lived} '
        f'tracks are active in at least 90% of frames.', '',
        '## Segmentation stability between consecutive frames', '',
        f'Per frame pair: {census.splits.mean():.1f} splits, {census.merges.mean():.1f} '
        f'merges, {census.lost.mean():.1f} labels lost, {census["new"].mean():.1f} new '
        f'({n.lost / cells:.2%} and {n["new"] / cells:.2%} of cells).', '',
        f'Of the {len(lost)} lost labels, {(lost.volume < 600).mean():.1%} are under 600 '
        f'voxels (median {lost.volume.median():.0f}) against a median cell volume of '
        f'{cell_volume:.0f}, and '
        f'{((lost.z < 1.5) | (lost.z > 8.5)).mean():.1%} sit within 1.5 slices of the top '
        'or bottom of the stack.  These are fragments, not cells.', '',
        _table(census.describe().loc[['mean', '50%', 'min', 'max']].round(1)), '',
        '## How far cells actually move', '',
        f'Measured over {motion.n_links.iloc[0]} consecutive links.', '',
        _table(motion.drop(columns='n_links').round(2)), '',
        '## What happens when a track dies', '',
        f'{len(deaths)} mid-sequence deaths of full-size tracks '
        '(mean volume >= 600, border exits excluded):', '',
        _table(verdicts.to_frame('count')), '',
        f'`stolen` means the cell carries on in the next frame under a track that already '
        f'existed -- a duplicate id, repairable only by merging.  Median dying volume '
        f'{deaths[deaths.verdict == "stolen"].volume.median():.0f} against successor '
        f'{deaths[deaths.verdict == "stolen"].successor_volume.median():.0f}: the '
        'segmentation had split one cell and merged it back.', '',
        '## How much gap bridging could repair', '',
        f'{len(stitched)} of {n_deaths} deaths pair with a reachable later birth.', '',
        '## Track population', '',
        f'{fragments.sum()} of {len(tracks)} tracks are fragments '
        f'(mean volume < {FRAGMENT_FRACTION:g} x {cell_volume:.0f}).', '',
        _table(tracks.groupby(pd.cut(tracks.n_active, [1, 5, 10, 20, 50, 100, 1000]),
                              observed=True).agg(
            n_tracks=('n_active', 'size'), median_volume=('mean_volume', 'median'),
        ).round(0)), '',
    ]
    if pair_rows is not None and len(pair_rows):
        lines += ['## Requested pair', '', _table(pair_rows, index=False), '']
    if figure:
        lines += [f'![pair]({os.path.basename(figure)})', '']

    path = os.path.join(out_dir, 'diagnostics.md')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--masks-dir', required=True,
                   help='directory of z_stack_t{N}_seg_masks.npz files')
    p.add_argument('--tracklets', required=True, help='tracklets json from create_tracklets()')
    p.add_argument('--out-dir', default='diagnostics', help='where tables and the report go')
    p.add_argument('--projection', help='(T, Y, X) cell-channel projection; enables --pair')
    p.add_argument('--pair', nargs=2, type=int, metavar=('TID', 'TID'),
                   help='two tracklet ids to examine as a suspected duplicate id')
    p.add_argument('--min-volume', type=float, default=600,
                   help='mean volume above which a track counts as a real cell')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    props = cell_properties(args.masks_dir,
                            os.path.join(args.out_dir, 'cell_properties.pkl'))
    ts = sorted(props)
    tracklets = load_tracklets(args.tracklets)
    print(f'{len(tracklets)} tracklets over {len(ts)} timepoints\n')

    tracks = track_table(tracklets, props, ts)
    evaluate_tracklets({str(k): v.tolist() for k, v in tracklets.items()})

    census, lost = split_merge_census(args.masks_dir, props)
    motion = motion_stats(tracklets, props, ts)
    deaths = diagnose_deaths(tracklets, props, args.masks_dir, ts,
                             min_volume=args.min_volume)
    stitched, n_deaths = stitch_feasibility(tracklets, props, ts,
                                            min_volume=args.min_volume)

    print('\n=== Segmentation churn per frame pair ===')
    print(census[['splits', 'merges', 'lost', 'new']].mean().round(1).to_string())
    print('\n=== Displacement of tracked cells ===')
    print(motion.drop(columns='n_links').round(2).to_string())
    print('\n=== Fate of the cell when a track dies ===')
    print(deaths.verdict.value_counts().to_string())
    print(f'\n{len(stitched)} of {n_deaths} deaths pair with a reachable later birth')

    for name, frame in (('census', census), ('lost_labels', lost), ('motion', motion),
                        ('deaths', deaths), ('tracks', tracks), ('stitches', stitched)):
        frame.to_csv(os.path.join(args.out_dir, f'{name}.csv'))

    pair_rows, figure = None, None
    if args.pair:
        pair_rows = pair_evidence(args.pair, tracklets, props, args.masks_dir, ts)
        print(f'\n=== Pair {args.pair[0]} vs {args.pair[1]} ===')
        print(pair_rows.to_string(index=False))
        if args.projection:
            from src.roi_selection import load_projection
            figure = pair_overlay(
                args.pair, tracklets, props, args.masks_dir, ts,
                load_projection(args.projection),
                os.path.join(args.out_dir, f'pair_{args.pair[0]}_{args.pair[1]}.png'))

    cell_volume = float(np.median(list(median_volume(props).values())))
    report = write_report(args.out_dir, census, lost, motion, deaths, stitched, n_deaths,
                          tracks, cell_volume, len(ts), pair_rows, figure)
    print(f'\nreport written to {report}')


if __name__ == '__main__':
    main()
