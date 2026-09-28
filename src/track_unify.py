"""Repair the tracklets, and the masks where the masks are what is broken.

src/track_diagnostics.py measures why tracks fragment; this acts on the two causes it
identifies.  Of the 760 mid-sequence deaths of full-size tracks in New-02-v3-ST11-12, 561
end with the cell's successor label already owned by another track, and the successor's
volume says those are two different accidents needing opposite repairs:

    successor >= 1.6 median cells   cellpose fused two real nuclei into one label.  Both
    (75% of cases)                  tracks are real -- merging them would delete a cell.
                                    Split the label with a seeded watershed and give each
                                    track its own piece.

    successor 0.6-1.6 median cells  the dying label is a fragment (median 0.31 cells)
    (22% of cases)                  reabsorbed into a normal cell.  This one really is a
                                    duplicate id: retire the fragment track.

Splitting changes voxels, so the corrected masks are written to a sibling directory and the
output tracklets index into that.  The originals are never touched.  Nothing is discarded
silently: every split, absorption and stitch is recorded in the log with the numbers that
justified it.

    python -m src.track_unify --masks-dir .../masks --tracklets .../tracklets.json \
        --out-dir .../unified
"""
import argparse
import collections
import json
import os

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects
from src.track import optional_assignment
from skimage.segmentation import watershed
from tqdm import tqdm

from cell_tracking import evaluate_tracklets
from src.track_diagnostics import (cell_properties, load_masks, load_tracklets,
                                   mask_paths_by_t, median_volume, track_table)
from src.viewer.loaders import GAP

# A label at least this many median cells is a clump of more than one nucleus.
CLUMP_FACTOR = 1.6
# A dying label this far below a median cell is a fragment, not a cell.
FRAGMENT_FACTOR = 0.5
# The band in which a label is one ordinary cell.
SINGLE_RANGE = (0.6, 1.6)
# How much of the dying label has to land in the successor for the link to be believed.
MIN_OVERLAP = 0.5
# A seed has to keep at least this much of itself on a label to still be following it.
MIN_HOLD = 0.25
# No link this repair creates may move a cell further than this in XY between frames.
# Measured motion is p99 = 10.9 px, p99.9 = 14.6; the old matcher's gate of 15 is what let
# ids walk onto neighbours in the first place, so do not inherit it.
MAX_STEP = 12.0
# Give up carrying tracks through a clump that never resolves; 16% of clumps outlast this.
MAX_CLUMP_FRAMES = 20
# A watershed piece this far below an equal share of the clump is not trusted.
MIN_PIECE_SHARE = 0.15
# A clump holding this many cells more than the tracks claiming it cannot be resolved by
# one piece per claimant: the surplus nuclei have no track to receive them, so the pieces
# come out several cells each.  Median clump is 2.20 cells against 2 claimants; this
# refuses the 11% where the segmentation fused a whole neighbourhood.
MAX_UNCLAIMED_CELLS = 1.0
# Gap stitching, from the measured motion: dXY p95 is 5.9 px per frame.
STITCH_MAX_DT, STITCH_BUDGET, STITCH_MAX_DZ = 5, 7.0, 3.0
# Flag a link this far out in the measured tails as a possible identity swap (p99).
SWAP_DZ, SWAP_DVOL = 3.0, 1.5


def _coords(volume, label):
    """Where a label sits, as (z, y, x) index arrays.  Seeds are carried between frames in
    this form: a few thousand indices rather than an 11 MB boolean per track per frame."""
    return np.where(volume == label)


def _overlapping_label(volume, seed):
    """The label covering most of a seed, and the fraction of the seed it covers."""
    if not len(seed[0]):
        return 0, 0.0
    counts = np.bincount(volume[seed])
    counts[0] = 0
    if not counts.size or not counts.max():
        return 0, 0.0
    best = int(counts.argmax())
    return best, float(counts[best] / len(seed[0]))


def _too_far(seed, target):
    """Did the cell move further between these two frames than a cell can?

    Both positions are computed from the voxels in hand rather than looked up, because the
    pieces this module cuts do not exist in the properties table built from the input masks.
    """
    if not len(seed[0]) or not len(target[0]):
        return True
    dy = target[1].mean() - seed[1].mean()
    dx = target[2].mean() - seed[2].mean()
    return float(np.hypot(dy, dx)) > MAX_STEP


def _owners(repaired, i):
    """{label: track} at one frame.  Rebuilt on demand because the repair edits tracks as
    it goes, and a stale map is how two tracks end up holding one label."""
    return {int(labels[i]): tid for tid, labels in repaired.items() if labels[i] > 0}


def _claim(repaired, births, tid, label, i, log, t):
    """Give a track a label at one frame, or refuse when another track holds it.

    Every assignment goes through here.  Two tracks naming one label breaks the assumption
    ms2_gene_expression.py and TrackletStore both make, and the branches below each had
    their own way of reaching that state.

    A holder that was *born* at this frame is not a competitor but the same cell under a
    fresh id -- the symptom being repaired -- so it is spliced in rather than deferred to.
    """
    holder = _owners(repaired, i).get(label)
    if holder is None or holder == tid:
        repaired[tid][i] = int(label)
        return True
    if births.get(holder) == i:
        _splice(repaired, tid, holder, i)
        log.append({'kind': 'clump_resolved_spliced', 'frame': t, 'track': tid,
                    'absorbed_track': holder, 'label': int(label)})
        return True
    log.append({'kind': 'clump_lost', 'frame': t, 'track': tid, 'label': int(label),
                'held_by': holder})
    return False


def _splice(repaired, keeper, donor, i):
    """Hand the donor's remaining life to the keeper and retire the donor.

    Used where a clump resolves: the label the revived track lands on is already the start
    of a fresh track, which is the same cell under a new id -- exactly the symptom being
    fixed.
    """
    for f in range(i, len(repaired[donor])):
        repaired[keeper][f] = repaired[donor][f]
    del repaired[donor]


def split_clump(volume, label, seeds, box, anisotropy):
    """Cut one fused label into a piece per seed with a seeded watershed.

    Watershed on the distance transform of the fused object, with the previous frame's
    cells as markers.  Validated by fusing correctly segmented touching pairs and splitting
    them back: reproduces the true partition to a median 99.3% of voxels.  `anisotropy` is
    the z:xy voxel ratio -- without it the distance transform treats a 0.5 um z step as a
    0.198 um one and the cut tilts.

    Returns {marker: (z, y, x) of that piece}, or None when the split is too lopsided to
    believe, in which case the caller leaves the label fused.
    """
    region = volume[box] == label
    origin = np.array([s.start for s in box])
    shape = np.array(region.shape)

    markers = np.zeros(region.shape, dtype=np.int32)
    for i, seed in enumerate(seeds, start=1):
        local = np.array(seed) - origin[:, None]
        inside = np.all((local >= 0) & (local < shape[:, None]), axis=0)
        local = local[:, inside]
        if local.size:
            markers[tuple(local)] = i
    markers[~region] = 0
    if len(set(np.unique(markers)) - {0}) < len(seeds):
        return None                      # a seed landed entirely outside the fused object

    distance = distance_transform_edt(region, sampling=anisotropy)
    cut = watershed(distance.max() - distance, markers=markers, mask=region)

    floor = MIN_PIECE_SHARE * region.sum() / len(seeds)
    pieces = {}
    for i in range(1, len(seeds) + 1):
        where = np.where(cut == i)
        if len(where[0]) < floor:
            return None                  # one piece came out too small to be a cell
        pieces[i] = tuple(w + origin[d] for d, w in enumerate(where))
    return pieces


def clump_events(tracklets, props, ts, volumes, masks_dir):
    """Classify every mid-sequence death by what its cell's successor turned out to be.

    The same walk diagnose_deaths does, kept separate because the repair needs the
    surviving track as well as the doomed one, and needs the events grouped by frame.
    """
    tracks = track_table(tracklets, props, ts)
    owner = collections.defaultdict(dict)
    for tid, labels in tracklets.items():
        for i, label in enumerate(labels):
            if label > 0:
                owner[i][int(label)] = tid

    dying = tracks[(tracks.death < len(ts) - 1) & ~tracks.exited]
    by_death = collections.defaultdict(list)
    for tid, row in dying.iterrows():
        by_death[int(row.death)].append(tid)

    paths = mask_paths_by_t(masks_dir)
    cell = float(np.median(list(volumes.values())))
    events, absorbed = collections.defaultdict(list), []
    for i in tqdm(sorted(by_death), desc='classifying deaths'):
        m0, m1 = load_masks(paths[ts[i]]), load_masks(paths[ts[i + 1]])
        for tid in by_death[i]:
            label = int(tracklets[tid][i])
            successor, fraction = _overlapping_label(m1, _coords(m0, label))
            if not successor or fraction < MIN_OVERLAP:
                continue
            holder = owner[i + 1].get(successor)
            if holder is None or tracks.at[holder, 'birth'] > i:
                continue

            cells = props[ts[i + 1]][successor][3] / volumes[ts[i + 1]]
            mine = props[ts[i]][label][3] / volumes[ts[i]]
            if cells >= CLUMP_FACTOR:
                events[i + 1].append({'tids': [tid, holder], 'label': successor,
                                      'cells': round(cells, 2)})
            elif (SINGLE_RANGE[0] <= cells < SINGLE_RANGE[1] and mine < FRAGMENT_FACTOR
                    and tracks.at[tid, 'mean_volume'] < FRAGMENT_FACTOR * cell):
                # The last condition judges the track over its whole life, not by the frame
                # it dies in.  A real cell whose label decays to a speck before vanishing
                # looks identical at the death frame to a fragment that shadowed a
                # neighbour all along -- and retiring the former deletes a cell.
                absorbed.append({'fragment': tid, 'host': holder, 'frame': ts[i],
                                 'fragment_cells': round(mine, 2),
                                 'host_cells': round(cells, 2),
                                 'track_mean_cells': round(
                                     tracks.at[tid, 'mean_volume'] / cell, 2),
                                 'overlap': round(fraction, 2)})
    return events, absorbed


def repair_clumps(tracklets, props, ts, volumes, masks_dir, out_masks_dir, anisotropy):
    """Carry both tracks through every fused label, splitting the masks as we go.

    One pass over the frames in order, because a clump lasting several frames has to be cut
    again each time and the seeds for frame f+1 are the pieces cut at frame f.  Only labels
    two tracks are actually contending for are touched; every other voxel of every frame is
    copied through unchanged.
    """
    events, absorbed = clump_events(tracklets, props, ts, volumes, masks_dir)
    repaired = {tid: [int(x) for x in labels] for tid, labels in tracklets.items()}
    births = {tid: int(np.argmax(np.asarray(labels) > 0)) for tid, labels in repaired.items()}
    paths = mask_paths_by_t(masks_dir)
    os.makedirs(out_masks_dir, exist_ok=True)

    log, open_repairs, following, previous = [], [], {}, None
    for i, t in enumerate(tqdm(ts, desc='splitting clumps')):
        volume = load_masks(paths[t])
        next_label = int(volume.max()) + 1
        boxes = None

        for event in events.get(i, []):
            # A track named by an event may already have been spliced into another one by
            # an earlier repair; the events were classified against the untouched input.
            tids = [x for x in event['tids'] if x in repaired]
            if len(tids) >= 2:
                open_repairs.append({'seeds': {}, 'started': i, 'tids': tids,
                                     'cells': event['cells']})

        still_open = []
        for repair in open_repairs:
            if not repair['seeds']:
                # First frame of the event: seed from the previous frame's cells, taken
                # from the corrected volume so a chained clump seeds off its own last cut.
                repair['seeds'] = {tid: _coords(previous, repaired[tid][i - 1])
                                   for tid in repair['tids']
                                   if tid in repaired and repaired[tid][i - 1] > 0}
            else:
                repair['seeds'] = {tid: seed for tid, seed in repair['seeds'].items()
                                   if tid in repaired}
            if len(repair['seeds']) < 2:
                continue

            claims = collections.defaultdict(list)
            for tid, seed in repair['seeds'].items():
                label, held = _overlapping_label(volume, seed)
                if label and held >= MIN_HOLD:
                    claims[label].append(tid)

            seeds_next, contended = {}, False
            for label, tids in claims.items():
                if len(tids) == 1:
                    tid, where = tids[0], _coords(volume, label)
                    if _too_far(repair['seeds'][tid], where):
                        log.append({'kind': 'link_too_far', 'frame': t, 'track': tid,
                                    'label': int(label)})
                        continue
                    if _claim(repaired, births, tid, label, i, log, t):
                        seeds_next[tid] = where
                    continue

                contended = True
                holder = _owners(repaired, i).get(label)
                if holder is not None and holder not in tids and births.get(holder) != i:
                    # An established track outside this repair owns the label; splitting it
                    # would take voxels the repair has no claim on.
                    log.append({'kind': 'clump_lost', 'frame': t, 'tids': tids,
                                'label': int(label), 'held_by': holder})
                    continue

                if props[t][label][3] < CLUMP_FACTOR * volumes[t]:
                    # Contended but too small to hold two cells: give it to whichever track
                    # covers it best and let the others end here.
                    best = max(tids, key=lambda x: _overlapping_label(volume,
                                                                      repair['seeds'][x])[1])
                    where = _coords(volume, label)
                    if _too_far(repair['seeds'][best], where):
                        log.append({'kind': 'link_too_far', 'frame': t, 'track': best,
                                    'label': int(label)})
                        continue
                    for other in tids:
                        if other != best and repaired[other][i] == label:
                            repaired[other][i] = GAP
                    if _claim(repaired, births, best, label, i, log, t):
                        seeds_next[best] = where
                    log.append({'kind': 'clump_collapsed', 'frame': t, 'label': int(label),
                                'kept': best, 'dropped': [x for x in tids if x != best]})
                    continue

                # Let whoever already holds the label keep it, so their later frames -- which
                # still name the original label -- stay valid.
                tids.sort(key=lambda x: holder != x)
                cells = props[t][label][3] / volumes[t]
                overfull = cells > len(tids) + MAX_UNCLAIMED_CELLS
                if overfull:
                    log.append({'kind': 'clump_overfull', 'frame': t, 'label': int(label),
                                'tids': tids, 'cells': round(cells, 2)})
                    pieces = None
                else:
                    if boxes is None:
                        boxes = find_objects(volume)
                    pieces = split_clump(volume, label, [repair['seeds'][x] for x in tids],
                                         boxes[label - 1], anisotropy)
                if pieces is None:
                    # The label stays fused, so whoever keeps it inherits a centroid that
                    # sits between two nuclei -- gate it like any other link.
                    keep, where = tids[0], _coords(volume, label)
                    if _too_far(repair['seeds'][keep], where):
                        log.append({'kind': 'link_too_far', 'frame': t, 'track': keep,
                                    'label': int(label)})
                        continue
                    if _claim(repaired, births, keep, label, i, log, t):
                        seeds_next[keep] = where
                    if not overfull:
                        log.append({'kind': 'split_refused', 'frame': t,
                                    'label': int(label), 'tids': tids, 'kept': keep})
                    continue

                for marker, tid in enumerate(tids, start=1):
                    where = pieces[marker]
                    if marker == 1:
                        if not _claim(repaired, births, tid, label, i, log, t):
                            continue                 # the first piece keeps the label
                    else:
                        new, next_label = next_label, next_label + 1
                        volume[where] = new
                        boxes = None                 # label bounds are stale after an edit
                        repaired[tid][i] = new
                    seeds_next[tid] = where
                log.append({'kind': 'clump_split', 'frame': t, 'label': int(label),
                            'tids': tids, 'cells': repair['cells'],
                            'pieces': [int(len(pieces[m][0])) for m in sorted(pieces)]})

            repair['seeds'] = seeds_next
            carried = i - repair['started']
            if not contended or len(seeds_next) < 2:
                # Resolved.  A track revived out of a clump usually lands on a label no
                # track owns, because the original tracker had already given up on that
                # cell -- so it has to be followed on until it rejoins the track graph,
                # otherwise the repair buys only the frames the clump lasted.
                for tid, seed in seeds_next.items():
                    if tid in repaired and all(x <= 0 for x in repaired[tid][i + 1:]):
                        following[tid] = seed
                continue
            if carried < MAX_CLUMP_FRAMES:
                still_open.append(repair)
            else:
                log.append({'kind': 'clump_abandoned', 'frame': t, 'tids': repair['tids'],
                            'frames_carried': carried})
        open_repairs = still_open

        for tid in list(following):
            if tid not in repaired:
                del following[tid]
                continue
            if any(tid in repair['seeds'] for repair in open_repairs):
                del following[tid]                   # back inside a clump; the repair has it
                continue
            label, held = _overlapping_label(volume, following[tid])
            if not label or held < MIN_HOLD:
                del following[tid]                   # cell gone; Pass C may bridge it
                continue
            where = _coords(volume, label)
            if _too_far(following[tid], where):
                log.append({'kind': 'link_too_far', 'frame': t, 'track': tid,
                            'label': int(label)})
                del following[tid]
                continue
            before = set(repaired)
            if not _claim(repaired, births, tid, label, i, log, t):
                del following[tid]
                continue
            if set(repaired) != before:
                del following[tid]                   # spliced back into an existing track
                continue
            following[tid] = where

        np.savez_compressed(os.path.join(out_masks_dir, os.path.basename(paths[t])),
                            masks=volume)
        previous = volume

    return repaired, log, absorbed


def absorb_fragments(repaired, absorbed, log):
    """Retire tracks that are a fragment of a cell another track already follows."""
    for entry in absorbed:
        if entry['fragment'] not in repaired:
            continue                                 # already consumed by a clump splice
        del repaired[entry['fragment']]
        log.append({'kind': 'fragment_absorbed', **entry})
    return repaired


def stitch_gaps(repaired, props, ts, volumes, log, anisotropy=2.52):
    """Re-link a track that ends with one that starts nearby a few frames later.

    Optional assignment with gap/volume penalties. Apply whole chains in reverse
    temporal order so deleting a donor never discards its selected successor.
    """
    tracklets = {tid: np.asarray(labels) for tid, labels in repaired.items()}
    if not tracklets:
        return repaired
    tracks = track_table(tracklets, props, ts)
    cell = float(np.median(list(volumes.values())))
    full = tracks[tracks.mean_volume >= FRAGMENT_FACTOR * cell]
    tails = full[(full.death < len(ts) - 1) & ~full.exited]
    heads = full[full.birth > 0]
    if tails.empty or heads.empty:
        return repaired

    def centre(tid, i):
        return props[ts[i]].get(int(tracklets[tid][i]))

    tail_ids, head_ids = list(tails.index), list(heads.index)
    cost = np.full((len(tail_ids), len(head_ids)), np.inf)
    for r, tid in enumerate(tail_ids):
        i = int(tails.at[tid, 'death'])
        a = centre(tid, i)
        if a is None:
            continue
        for c, other in enumerate(head_ids):
            if other == tid:
                continue
            dt = int(heads.at[other, 'birth']) - i
            if not 1 <= dt <= STITCH_MAX_DT:
                continue
            b = centre(other, i + dt)
            if b is None:
                continue
            # Diffusive uncertainty grows more slowly than a linear search radius.
            distance = float(np.linalg.norm((np.asarray(a[:3])-b[:3]) * [anisotropy, 1, 1]))
            gate = STITCH_BUDGET * np.sqrt(dt)
            volume_change = abs(np.log(max(a[3], 1) / max(b[3], 1)))
            if distance <= gate and abs(a[0] - b[0]) <= STITCH_MAX_DZ:
                cost[r, c] = distance / gate + 0.12*(dt-1) + 0.2*volume_change

    selected = optional_assignment(cost, unmatched_cost=1.0)
    # B->C must be applied before A->B, regardless of dictionary/track ID order.
    selected.sort(key=lambda rc: int(tails.at[tail_ids[rc[0]], 'death']), reverse=True)
    for r, c in selected:
        if not np.isfinite(cost[r, c]):
            continue
        tid, other = tail_ids[r], head_ids[c]
        if tid not in repaired or other not in repaired:
            continue
        death, birth = int(tails.at[tid, 'death']), int(heads.at[other, 'birth'])
        for f in range(death + 1, birth):
            repaired[tid][f] = GAP
        _splice(repaired, tid, other, birth)
        log.append({'kind': 'gap_stitched', 'tail': tid, 'head': other,
                    'death_frame': ts[death], 'birth_frame': ts[birth],
                    'cost': round(float(cost[r, c]), 4)})
    return repaired


def flag_swaps(repaired, props, ts, log):
    """Report links that jump further in z or change volume more than a real cell does."""
    for tid, labels in repaired.items():
        for i in range(len(labels) - 1):
            if labels[i] <= 0 or labels[i + 1] <= 0:
                continue
            a = props[ts[i]].get(int(labels[i]))
            b = props[ts[i + 1]].get(int(labels[i + 1]))
            if a is None or b is None:
                continue
            dz, dvol = abs(a[0] - b[0]), abs(a[3] - b[3]) / max(a[3], 1)
            if dz > SWAP_DZ or dvol > SWAP_DVOL:
                log.append({'kind': 'possible_swap', 'tid': tid, 'frame': ts[i + 1],
                            'dz': round(dz, 1), 'rel_dvolume': round(dvol, 1)})


def check(repaired, out_masks_dir, ts):
    """Refuse to ship a result that breaks the schema the consumers rely on.

    ms2_gene_expression.py indexes a tracklet by frame and looks that label up in that
    frame's mask (`:174`, `:232`), and TrackletStore assumes one track owns a label
    (src/viewer/loaders.py:146).  Checked here rather than discovered downstream.
    """
    lengths = {len(v) for v in repaired.values()}
    assert lengths == {len(ts)}, f'tracks of differing length: {lengths}'

    paths = mask_paths_by_t(out_masks_dir)
    for i, t in enumerate(tqdm(ts, desc='checking output')):
        present = set(np.unique(load_masks(paths[t])).tolist())
        seen = {}
        for tid, labels in repaired.items():
            label = int(labels[i])
            if label <= 0:
                continue
            assert label in present, f'track {tid} names label {label} absent at t={t}'
            assert label not in seen, \
                f'label {label} at t={t} claimed by tracks {seen[label]} and {tid}'
            seen[label] = tid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--masks-dir', required=True, help='the original z_stack_t{N} npz files')
    p.add_argument('--tracklets', required=True, help='tracklets json to repair')
    p.add_argument('--out-dir', required=True,
                   help='destination for masks_split/, the tracklets and the log')
    p.add_argument('--anisotropy', type=float, default=2.52,
                   help='z:xy voxel ratio, 0.5 um / 0.198 um for this microscope')
    p.add_argument('--properties-cache',
                   help='cell_properties.pkl from track_diagnostics, to skip recomputing it')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_masks = os.path.join(args.out_dir, 'masks_split')

    props = cell_properties(args.masks_dir,
                            args.properties_cache
                            or os.path.join(args.out_dir, 'cell_properties.pkl'))
    ts = sorted(props)
    volumes = median_volume(props)
    tracklets = load_tracklets(args.tracklets)

    print('--- before ---')
    evaluate_tracklets({str(k): v.tolist() for k, v in tracklets.items()})

    repaired, log, absorbed = repair_clumps(tracklets, props, ts, volumes,
                                            args.masks_dir, out_masks, args.anisotropy)
    repaired = absorb_fragments(repaired, absorbed, log)

    # The split moved voxels, so positions have to be re-read from what was written.
    props = cell_properties(out_masks, os.path.join(args.out_dir,
                                                    'cell_properties_split.pkl'))
    volumes = median_volume(props)
    repaired = stitch_gaps(repaired, props, ts, volumes, log, args.anisotropy)
    flag_swaps(repaired, props, ts, log)

    print('\n--- repairs ---')
    for kind, n in collections.Counter(e['kind'] for e in log).most_common():
        print(f'  {n:5d}  {kind}')

    check(repaired, out_masks, ts)
    print('\n--- after ---')
    evaluate_tracklets({str(k): list(map(int, v)) for k, v in repaired.items()})

    tracklets_path = os.path.join(args.out_dir, 'tracklets_unified.json')
    with open(tracklets_path, 'w') as f:
        json.dump({str(k): list(map(int, v)) for k, v in repaired.items()}, f, indent=4)
    with open(os.path.join(args.out_dir, 'tracklets_unified_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print(f'\ntracklets -> {tracklets_path}\nmasks     -> {out_masks}')


if __name__ == '__main__':
    main()
