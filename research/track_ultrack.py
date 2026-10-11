"""Track cells in 3D with ultrack, writing the tracklet json the rest of this repo reads.

The tracker in src/track.py matches one frame to the next and commits to that match, so a
segmentation mistake becomes a tracking mistake permanently -- which is what
src/track_diagnostics.py measured and src/track_unify.py repairs after the fact.  ultrack
instead builds a hierarchy of candidate segments per frame and chooses, with one global
integer program over the whole movie, the combination that tracks most consistently.  A
clump this repo can only cut open after the tracker has already broken on it, ultrack can
decline to create.

Two ways in:

    --masks-dir given    existing masks are passed as `labels`; ultrack still explores
                         alternatives around them
    no --masks-dir       segment first, either with cellpose using the tuned parameters
                         from cell_3d_segmentation.py (--mode labels, the default), or with
                         ultrack's own foreground/contour detection (--mode contours),
                         which is where its hypothesis hierarchy is richest

ultrack picks its own segments, so its labels do not correspond to any input mask labels.
The output is therefore a new mask directory alongside the tracklets, in the same
z_stack_t{N}_seg_masks.npz convention every other tool here expects:

    python -m src.track_ultrack --masks-dir .../masks --out-dir .../ultrack
    python -m src.track_ultrack --image cells.czi --mode contours --out-dir .../ultrack
"""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from cell_tracking import evaluate_tracklets
from src.track import get_border_labels
from src.track_diagnostics import load_masks, mask_paths_by_t
from src.viewer.loaders import EXITED, GAP

# z 0.5 um against xy 0.198 um in both New-02 and New-03; ultrack scales node distances by
# this, and a wrong ratio distorts every linking decision along z.
ANISOTROPY = 2.52
# Measured frame-to-frame displacement is p95 5.9 px, p99 10.9.  The old matcher's gate of
# 15 is what let ids walk onto neighbouring cells, so do not inherit it.
MAX_DISTANCE = 12.0
# A median cell is ~1170 voxels.  Below a third of that is a fragment, above ~3 cells is a
# clump rather than a hypothesis worth keeping.
MIN_AREA, MAX_AREA = 400, 4000
# Solve in overlapping windows so the ILP stays tractable across a long movie.
WINDOW_SIZE = 25


def build_config(out_dir, solver, window_size, min_area, max_area, max_distance, n_workers):
    """ultrack's config, set from this project's measurements rather than the defaults.

    `solver_name='CBC'` is deliberate.  gurobipy is installed but on a restricted licence
    ("non-production use only") that is size-limited and cannot take a problem this size.
    With the default '' python-mip tries Gurobi and raises InterfacingError, which ultrack
    catches and rescues into CBC -- naming CBC removes the dependency on that rescue.
    """
    from ultrack import MainConfig

    config = MainConfig()
    config.data_config.working_dir = out_dir          # the sqlite db lands here
    config.data_config.n_workers = n_workers
    config.segmentation_config.min_area = min_area
    config.segmentation_config.max_area = max_area
    config.segmentation_config.n_workers = n_workers
    config.linking_config.max_distance = max_distance
    config.linking_config.n_workers = n_workers
    config.tracking_config.solver_name = solver
    if window_size:
        config.tracking_config.window_size = window_size
    return config


def load_label_stack(masks_dir):
    """Existing per-timepoint masks as one (T, Z, Y, X) array, ordered by timepoint."""
    by_t = mask_paths_by_t(masks_dir)
    ts = sorted(by_t)
    stack = np.stack([load_masks(by_t[t]) for t in tqdm(ts, desc='loading masks')])
    return stack, ts


def segment_with_cellpose(image, device):
    """Segment each timepoint with the parameters tuned in cell_3d_segmentation.py.

    Imported rather than restated so the two entry points cannot drift apart: an A/B over
    six configurations picked min_size and the fixed 3D normalisation window, and repeating
    those numbers here would let one copy go stale.
    """
    import torch
    from cellpose import models

    from cell_3d_segmentation import MIN_SIZE, intensity_bounds

    # intensity_bounds indexes (T, C, Z, Y, X); the channel was already selected upstream,
    # so give it a single-channel axis and read channel 0.
    lo, hi = intensity_bounds(image[:, None], channel=0)
    print(f'fixed intensity window across all timepoints: {lo:.1f} - {hi:.1f}')
    model = models.CellposeModel(gpu=True, device=torch.device(device))

    labels = []
    for volume in tqdm(image, desc='cellpose'):
        masks, _flows, _ = model.eval(
            volume, z_axis=0, channel_axis=None, batch_size=32, do_3D=True,
            flow3D_smooth=1, min_size=MIN_SIZE,
            normalize={'norm3D': True, 'lowhigh': (lo, hi)})
        labels.append(np.asarray(masks, dtype=np.int32))
    return np.stack(labels)


def detect_contours(image):
    """ultrack's own foreground and contour estimate, no cellpose involved.

    Passing these instead of labels is what lets ultrack consider segmentations no single
    thresholding would produce -- the mode worth trying when cellpose itself is the thing
    introducing the clumps.
    """
    from ultrack.imgproc import detect_foreground, robust_invert

    voxel_size = (ANISOTROPY, 1.0, 1.0)
    foreground, contours = [], []
    for volume in tqdm(image, desc='foreground/contours'):
        foreground.append(detect_foreground(volume, voxel_size=voxel_size))
        contours.append(robust_invert(volume, voxel_size=voxel_size))
    return np.stack(foreground), np.stack(contours)


def write_masks(array, ts, out_masks_dir):
    """ultrack's chosen segments per timepoint, named so get_masks_paths finds them.

    Only the masks array is stored; the flow arrays in the original npz files are 99% of
    their size and nothing downstream reads them.
    """
    os.makedirs(out_masks_dir, exist_ok=True)
    for i, t in enumerate(tqdm(ts, desc='writing masks')):
        np.savez_compressed(os.path.join(out_masks_dir, f'z_stack_t{t}_seg_masks.npz'),
                            masks=np.asarray(array[i], dtype=np.uint16))


def build_tracklets(tracks_df, n_timepoints, out_masks_dir, ts, border_margin=5):
    """{track id: label per frame} in this repo's schema.

    ultrack relabels its segments by track id, so a track's label is its own id wherever it
    exists.  The two sentinels still have to be reconstructed, because the schema
    distinguishes them and both loaders.py and ms2_gene_expression.py act on that: -1 is a
    frame the track is missing from, -2 is a departure through the field-of-view border,
    which is expected behaviour rather than a tracking failure.
    """
    by_t = mask_paths_by_t(out_masks_dir)
    present = []
    borders = []
    for t in tqdm(ts, desc='border check'):
        volume = load_masks(by_t[t])
        present.append(set(np.unique(volume).tolist()) - {0})
        borders.append(get_border_labels(volume, border_margin))

    tracklets = {}
    for tid in sorted(tracks_df['track_id'].unique()):
        tid = int(tid)
        row = [tid if tid in present[i] else GAP for i in range(n_timepoints)]
        active = [i for i, v in enumerate(row) if v > 0]
        if not active:
            continue
        # Mark the frame after the track's last appearance as a border exit when it left
        # from the edge, matching create_tracklets' meaning of the sentinel.
        death = active[-1]
        if death + 1 < n_timepoints and tid in borders[death]:
            row[death + 1] = EXITED
        tracklets[tid] = row
    return tracklets


def check(tracklets, out_masks_dir, ts):
    """The invariants every consumer here assumes, asserted before anything ships.

    Same three as track_unify.check(): fixed length, every named label really present in
    that frame's mask, and no label claimed by two tracks at once
    (src/viewer/loaders.py:146 and ms2_gene_expression.py:174 both rely on it).
    """
    lengths = {len(v) for v in tracklets.values()}
    assert lengths == {len(ts)}, f'tracks of differing length: {lengths}'

    by_t = mask_paths_by_t(out_masks_dir)
    for i, t in enumerate(tqdm(ts, desc='checking output')):
        labels = set(np.unique(load_masks(by_t[t])).tolist())
        seen = {}
        for tid, row in tracklets.items():
            label = int(row[i])
            if label <= 0:
                continue
            assert label in labels, f'track {tid} names label {label} absent at t={t}'
            assert label not in seen, \
                f'label {label} at t={t} claimed by tracks {seen[label]} and {tid}'
            seen[label] = tid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', required=True,
                   help='destination for masks_ultrack/, the tracklets, and ultrack\'s db')
    p.add_argument('--masks-dir', help='existing z_stack_t{N} npz masks to track; '
                                       'without it the module segments first')
    p.add_argument('--image', help='4D (T, Z, Y, X) tif or a czi; required when '
                                   '--masks-dir is not given')
    p.add_argument('--mode', choices=('labels', 'contours'), default='labels',
                   help='how to segment when --masks-dir is absent: cellpose (labels) or '
                        'ultrack\'s own foreground/contour detection (contours)')
    p.add_argument('--channel', type=int, default=None,
                   help='cell channel for multi-channel input (default: '
                        'cell_3d_segmentation.CELL_CHANNEL)')
    p.add_argument('--device', default='cuda:0', help='torch device for cellpose')
    p.add_argument('--anisotropy', type=float, default=ANISOTROPY,
                   help='z:xy voxel ratio, 0.5 um / 0.198 um for this microscope')
    p.add_argument('--max-distance', type=float, default=MAX_DISTANCE,
                   help='linking gate in px; measured motion is p99 = 10.9')
    p.add_argument('--min-area', type=int, default=MIN_AREA)
    p.add_argument('--max-area', type=int, default=MAX_AREA)
    p.add_argument('--window-size', type=int, default=WINDOW_SIZE,
                   help='ILP window in frames; 0 solves the whole movie at once')
    p.add_argument('--solver', default='CBC', choices=('CBC', 'GUROBI', ''),
                   help='CBC by default: the installed Gurobi licence is size-restricted')
    p.add_argument('--n-workers', type=int, default=4)
    p.add_argument('--limit', type=int, help='only the first N timepoints, for a smoke test')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out_masks = os.path.join(args.out_dir, 'masks_ultrack')

    from ultrack import to_tracks_layer, track, tracks_to_zarr

    labels = foreground = contours = None
    if args.masks_dir:
        labels, ts = load_label_stack(args.masks_dir)
    else:
        if not args.image:
            raise SystemExit('--image is required when --masks-dir is not given')
        from cell_3d_segmentation import CELL_CHANNEL
        from src.viewer.loaders import load_image

        image = load_image(args.image,
                           CELL_CHANNEL if args.channel is None else args.channel)
        ts = list(range(len(image)))
        if args.mode == 'labels':
            labels = segment_with_cellpose(image, args.device)
        else:
            foreground, contours = detect_contours(image)

    if args.limit:
        ts = ts[:args.limit]
        labels = None if labels is None else labels[:args.limit]
        foreground = None if foreground is None else foreground[:args.limit]
        contours = None if contours is None else contours[:args.limit]

    n_t = len(ts)
    print(f'tracking {n_t} timepoints with ultrack '
          f'({"labels" if labels is not None else "foreground/contours"})')

    config = build_config(args.out_dir, args.solver, args.window_size, args.min_area,
                          args.max_area, args.max_distance, args.n_workers)
    track(config, labels=labels, foreground=foreground, contours=contours,
          scale=(args.anisotropy, 1.0, 1.0), overwrite='all')

    tracks_df, graph = to_tracks_layer(config)
    segments = tracks_to_zarr(config, tracks_df,
                              store_or_path=os.path.join(args.out_dir, 'segments.zarr'),
                              overwrite=True)
    write_masks(segments, ts, out_masks)

    tracklets = build_tracklets(tracks_df, n_t, out_masks, ts)
    check(tracklets, out_masks, ts)

    print('\n--- ultrack tracklets ---')
    evaluate_tracklets({str(k): v for k, v in tracklets.items()})

    tracklets_path = os.path.join(args.out_dir, 'tracklets_ultrack.json')
    with open(tracklets_path, 'w') as f:
        json.dump({str(k): [int(x) for x in v] for k, v in tracklets.items()}, f, indent=4)

    # Divisions are the one thing the current tracker cannot express at all, and the
    # tracklet schema has nowhere to put them -- so they go in a sidecar rather than being
    # dropped on the floor.
    lineage = {int(k): [int(x) for x in v] for k, v in graph.items()}
    with open(os.path.join(args.out_dir, 'lineage_ultrack.json'), 'w') as f:
        json.dump(lineage, f, indent=2)
    print(f'divisions (parent -> children): {len(lineage)}')
    print(f'\ntracklets -> {tracklets_path}\nmasks     -> {out_masks}')


if __name__ == '__main__':
    main()
