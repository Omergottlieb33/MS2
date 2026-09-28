"""Four per-track diagnostics for one tracklet JSON and its segmentation masks."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree

from src.track_diagnostics import load_masks, mask_paths_by_t
from src.tracking_evaluation import load_rows

DEFAULTS = dict(voxel_size_um=[0.5, 0.198, 0.198], max_step_um=3.0,
                neighbor_radius_um=10.0, neighbor_k=5)


def settings_checked(settings=None):
    result = {**DEFAULTS, **(settings or {})}
    scale = np.asarray(result['voxel_size_um'], dtype=float)
    if scale.shape != (3,) or not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError('voxel_size_um must contain three positive finite values')
    for key in ('max_step_um', 'neighbor_radius_um'):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if not isinstance(result['neighbor_k'], int) or isinstance(result['neighbor_k'], bool) or result['neighbor_k'] < 1:
        raise ValueError('neighbor_k must be a positive integer')
    return result


def infer_masks(tracklets):
    parent = Path(tracklets).resolve().parent
    for candidate in (parent/'masks_split', parent/'masks', parent, parent.parent/'masks'):
        if candidate.is_dir() and next(candidate.glob('z_stack_t*_seg_masks.npz'), None):
            return candidate
    raise ValueError('Cannot infer matching masks. Supply --masks-dir with the masks used by this tracklet JSON.')


def frame_properties(mask, spacing):
    labels, volumes = np.unique(mask, return_counts=True)
    volumes = volumes[labels > 0]; labels = labels[labels > 0]
    centres = center_of_mass(mask, mask, labels) if len(labels) else []
    return {int(label): dict(volume=float(volume*np.prod(spacing)),
                             centre=(np.asarray(c)*spacing).tolist())
            for label, volume, c in zip(labels, volumes, centres)}


def neighborhoods(props, owners, radius, k):
    """Include untracked nearby detections in the denominator, never as identities."""
    labels = list(props)
    if not labels:
        return {}
    xyz = np.array([props[label]['centre'] for label in labels])
    tree = cKDTree(xyz)
    result = {}
    for i, label in enumerate(labels):
        indices = tree.query_ball_point(xyz[i], radius)
        indices = sorted((j for j in indices if j != i),
                         key=lambda j: (float(np.linalg.norm(xyz[j]-xyz[i])), labels[j]))[:k]
        nearby = [labels[j] for j in indices]
        result[label] = dict(count=len(nearby), tracks={owners[l] for l in nearby if l in owners})
    return result


def diagnostics(rows, props, owners, settings):
    n = len(props)
    neighbors = [neighborhoods(p, o, settings['neighbor_radius_um'], settings['neighbor_k'])
                 for p, o in zip(props, owners)]
    output = []
    for tid, row in rows.items():
        active = np.flatnonzero(row > 0)
        if not len(active):
            continue
        volume = [None]*n; step = [None]*n; neighbor = [None]*n
        neighbor_retained = neighbor_total = neighbor_owned = 0
        neighbor_observations = 0
        for t in active:
            label = int(row[t])
            if label not in props[t]:
                raise ValueError(f'Track {tid} references absent label {label} at frame {t}')
            volume[t] = props[t][label]['volume']
            nb = neighbors[t][label]
            neighbor_owned += len(nb['tracks']); neighbor_observations += nb['count']
        for a, b in zip(active, active[1:]):
            if b != a+1:
                continue
            p = props[a][int(row[a])]; q = props[b][int(row[b])]
            step[b] = float(np.linalg.norm(np.asarray(q['centre'])-p['centre']))
            old = neighbors[a][int(row[a])]; new = neighbors[b][int(row[b])]
            if old['count']:
                kept = len(old['tracks'] & new['tracks'])
                neighbor[b] = kept/old['count']
                neighbor_retained += kept; neighbor_total += old['count']
        observed_volume = np.array([v for v in volume if v is not None])
        steps = [v for v in step if v is not None]
        output.append(dict(id=str(tid), active_frames=len(active), start=int(active[0]), end=int(active[-1]),
                           length_fraction=len(active)/n, gap_frames=int(active[-1]-active[0]+1-len(active)),
                           volume_cv=float(observed_volume.std()/observed_volume.mean()) if len(active)>1 else None,
                           median_volume=float(np.median(observed_volume)),
                           step_p95=float(np.percentile(steps,95)) if steps else None,
                           step_median=float(np.median(steps)) if steps else None,
                           step_tests=len(steps), large_steps=sum(v>settings['max_step_um'] for v in steps),
                           neighbor_retention=neighbor_retained/neighbor_total if neighbor_total else None,
                           neighbor_retained=neighbor_retained, neighbor_total=neighbor_total,
                           neighbor_tracked=neighbor_owned, neighbor_observations=neighbor_observations,
                           volume=volume, steps=step, neighbors=neighbor,
                           centers_xy=[(np.asarray(props[t][int(row[t])]['centre']) / settings['voxel_size_um'])[[2,1]].tolist()
                                       if row[t]>0 else None for t in range(n)]))
    return output


def fingerprint(tracklets, masks_dir, settings):
    paths = mask_paths_by_t(str(masks_dir))
    stamps = [(t, str(Path(p).resolve()), Path(p).stat().st_size, Path(p).stat().st_mtime_ns)
              for t, p in sorted(paths.items())]
    source = [Path(__file__), Path(__file__).with_name('tracking_evaluation.py'),
              Path(__file__).with_name('track_diagnostics.py')]
    value = [str(Path(tracklets).resolve()), hashlib.sha256(Path(tracklets).read_bytes()).hexdigest(),
             settings, stamps, [hashlib.sha256(p.read_bytes()).hexdigest() for p in source]]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evaluate(tracklets, masks_dir, settings=None, title=None, progress=print):
    settings = settings_checked(settings)
    paths = mask_paths_by_t(str(masks_dir)); ts = sorted(paths)
    if not ts or ts != list(range(len(ts))):
        raise ValueError('Masks must cover consecutive frames starting at t0')
    rows, owners = load_rows(tracklets, len(ts))
    props = []
    shape = None
    for t in ts:
        if t % 8 == 0:
            progress(f'Reading masks: frame {t+1}/{len(ts)}')
        mask = load_masks(paths[t])
        if mask.ndim != 3 or (shape is not None and mask.shape != shape):
            raise ValueError('Expected consistently shaped 3D masks')
        if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise ValueError('Masks must contain nonnegative integer labels')
        shape = mask.shape
        p = frame_properties(mask, np.asarray(settings['voxel_size_um']))
        if set(owners[t])-set(p):
            raise ValueError(f'Track references an absent mask label at frame {t}')
        props.append(p)
    progress('Computing track duration, volume, movement, and neighborhood consistency')
    return dict(schema='single-tracklets-v1', title=title or Path(tracklets).stem,
                frames=len(ts), settings=settings,
                paths=dict(tracklets=str(Path(tracklets).resolve()), masks=str(Path(masks_dir).resolve())),
                tracks=diagnostics(rows, props, owners, settings),
                empty_tracks=sum(not np.any(row > 0) for row in rows.values()),
                fingerprint=fingerprint(tracklets, masks_dir, settings))
