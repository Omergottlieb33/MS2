"""Reference-based, ground-truth-free tracking diagnostics.

All scores are consistency proxies. Reference masks define a shared evaluation
population, not biological ground truth. Run through src.tracking_dashboard.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, find_objects

from src.track import (create_tracklets, optional_assignment,
                       match_cells_by_iou_hungarian_local_optimized as match)
from src.track_diagnostics import load_masks, mask_paths_by_t

DEFAULTS = dict(min_volume=600, xy_margin=5, z_margin=1,
                voxel_size_um=[0.5, 0.198, 0.198],
                motion_threshold_um_per_frame=1.5, mapping_iou=0.25)


def fraction(numerator, denominator, scale=1):
    return scale * numerator / denominator if denominator else None


def features(mask):
    labels = np.unique(mask)
    labels = labels[labels > 0]
    volumes = np.bincount(mask.ravel())
    centres = center_of_mass(mask, mask, labels) if len(labels) else []
    boxes = find_objects(mask)
    return {int(label): dict(centre=list(map(float, c)), volume=int(volumes[label]),
                             box=[[int(s.start), int(s.stop)] for s in boxes[label-1]])
            for label, c in zip(labels, centres)}


def boundary(feature, shape, settings):
    box = feature['box']
    if any(box[d][0] < settings['xy_margin'] or
           box[d][1] > shape[d]-settings['xy_margin'] for d in (1, 2)):
        return 'xy'
    if settings['z_margin'] and (box[0][0] < settings['z_margin'] or
                               box[0][1] > shape[0]-settings['z_margin']):
        return 'axial_uncertain'
    return 'interior'


def map_labels(reference, candidate, threshold):
    """One-to-one full-volume IoU mapping: candidate label -> reference label."""
    if reference.shape != candidate.shape:
        raise ValueError('Candidate/reference mask shapes differ')
    a, va = np.unique(reference, return_counts=True)
    b, vb = np.unique(candidate, return_counts=True)
    va = va[a > 0]; a = a[a > 0]
    vb = vb[b > 0]; b = b[b > 0]
    if not len(a) or not len(b):
        return {}, dict(unmapped_reference=len(a), unmapped_output=len(b),
                        split_like=0, merge_like=0)
    both = (reference > 0) & (candidate > 0)
    stride = int(candidate.max())+1
    keys, counts = np.unique(reference[both].astype(np.int64)*stride+candidate[both],
                             return_counts=True)
    intersection = np.zeros((len(a), len(b)))
    intersection[np.searchsorted(a, keys//stride), np.searchsorted(b, keys % stride)] = counts
    iou = intersection/(va[:, None]+vb[None, :]-intersection)
    # nextafter permits pairs exactly at the requested IoU threshold.
    pairs = optional_assignment(np.where(iou >= threshold, 1-iou, np.inf),
                                np.nextafter(1-threshold, np.inf))
    mapping = {int(b[j]): int(a[i]) for i, j in pairs}
    return mapping, dict(unmapped_reference=len(a)-len(mapping),
                         unmapped_output=len(b)-len(mapping),
                         split_like=int(((intersection / va[:, None] >= .1).sum(axis=1) > 1).sum()),
                         merge_like=int(((intersection / vb[None, :] >= .1).sum(axis=0) > 1).sum()))


def load_rows(path, n):
    raw = json.loads(Path(path).read_text())
    rows = {}
    owners = [dict() for _ in range(n)]
    for tid, values in raw.items():
        row = np.asarray(values)
        if row.shape != (n,) or not np.issubdtype(row.dtype, np.integer):
            raise ValueError(f'Track {tid}: expected {n} integer entries')
        if np.any((row <= 0) & ~np.isin(row, [-1, -2])):
            raise ValueError(f'Track {tid}: unexpected missing-observation sentinel')
        rows[str(tid)] = row
        for t in np.flatnonzero(row > 0):
            label = int(row[t])
            if label in owners[t]:
                raise ValueError(f'Duplicate ownership at frame {t}, label {label}')
            owners[t][label] = str(tid)
    return rows, owners


def score_dataset(name, rows, owners, props, mappings, reference_props, shape, settings):
    n = len(props)
    eligible = [{label for label, f in p.items() if f['volume'] >= settings['min_volume']}
                for p in reference_props]
    frame_coverage = []
    exposure = dict(interior=0, axial_uncertain=0, xy=0)
    observed = 0
    for t in range(n):
        supported = {mappings[t][label] for label in owners[t]
                     if label in mappings[t] and mappings[t][label] in eligible[t]}
        observed += len(supported)
        frame_coverage.append(dict(t=t, covered=len(supported), eligible=len(eligible[t]),
                                   value=fraction(len(supported), len(eligible[t]))))
        if t < n-1:  # Final-frame observations cannot have an evaluable ending.
            for label in supported:
                exposure[boundary(reference_props[t][label], shape, settings)] += 1
    endings = dict(interior=0, axial_uncertain=0, xy=0, unmapped=0)
    events, residuals = [], []
    tests = flags = 0
    scale = np.asarray(settings['voxel_size_um'])
    for tid, row in rows.items():
        active = np.flatnonzero(row > 0)
        if not len(active):
            continue
        last = int(active[-1]); label = int(row[last])
        if last < n-1:
            ref = mappings[last].get(label)
            if ref is None:
                endings['unmapped'] += 1
            elif ref in eligible[last]:
                kind = boundary(reference_props[last][ref], shape, settings)
                endings[kind] += 1
                events.append(dict(kind=kind+'_ending', track=tid, frame=last,
                                   label=label, score=None))
        for a, b, c in zip(active, active[1:], active[2:]):
            if not all(mappings[t].get(int(row[t])) in eligible[t] for t in (a, b, c)):
                continue
            p, q, r = [np.asarray(props[t][int(row[t])]['centre'])*scale for t in (a, b, c)]
            residual = float(np.linalg.norm((r-q)/(c-b)-(q-p)/(b-a)))
            residuals.append(residual); tests += 1
            if residual > settings['motion_threshold_um_per_frame']:
                flags += 1
                events.append(dict(kind='motion', track=tid, frame=int(c), label=int(row[c]),
                                   score=residual, previous_frame=int(b), first_frame=int(a)))
    # Eligibility is chosen only from reference t0, never from output track size.
    # XY-interior cohort is retained even when touching Z in this shallow stack.
    cohort = sorted(label for label in eligible[0]
                    if boundary(reference_props[0][label], shape, settings) != 'xy')
    inverse = {ref: output for output, ref in mappings[0].items()}
    initial_tracks = {ref: owners[0].get(inverse.get(ref)) for ref in cohort}
    continuous = {ref: initial_tracks[ref] is not None for ref in cohort}
    curve = []
    for t in range(n):
        present = 0
        for ref, tid in initial_tracks.items():
            detected = tid is not None and rows[tid][t] > 0 and int(rows[tid][t]) in mappings[t]
            present += int(detected)
            continuous[ref] = continuous[ref] and detected
        curve.append(dict(t=t, observed=present, cohort=len(cohort),
                          value=fraction(present, len(cohort)),
                          uninterrupted=fraction(sum(continuous.values()), len(cohort))))
    denom = sum(map(len, eligible))
    return dict(name=name, tracks=len(rows), coverage=fraction(observed, denom),
                covered_observations=observed, eligible_observations=denom,
                exposure=exposure, endings=endings,
                interior_endings_per_1000=fraction(endings['interior'], exposure['interior'], 1000),
                axial_uncertain_per_1000=fraction(endings['axial_uncertain'], exposure['axial_uncertain'], 1000),
                motion_flags_per_1000=fraction(flags, tests, 1000),
                motion_flags=flags, motion_tests=tests,
                motion_p95=float(np.percentile(residuals, 95)) if residuals else None,
                cohort_size=len(cohort), cohort_curve=curve, frame_coverage=frame_coverage,
                events=events, integrity=dict(duplicate_claims=0, absent_labels=0))


def association_edges(rows):
    edges = set()
    for row in rows.values():
        active = np.flatnonzero(np.asarray(row) > 0)
        for a, b in zip(active, active[1:]):
            edges.add((int(a), int(row[a]), int(b), int(row[b])))
    return edges


def stability(base, variant):
    common = len(base & variant)
    return dict(baseline_links=len(base), variant_links=len(variant), common_links=common,
                retention=fraction(common, len(base)), precision=fraction(common, len(variant)),
                jaccard=fraction(common, len(base | variant)))


def stress_suite(masks, saved_rows, settings, options, progress=print):
    """Rerun the current default association pipeline, not historical/unified pipelines."""
    start = int(options.get('start', 0))
    count = min(int(options.get('frames', 8)), len(masks)-start)
    if start < 0 or count < 3 or start >= len(masks):
        raise ValueError('Stress window must contain at least 3 available frames')
    window = masks[start:start+count]
    gate = float(options.get('distance_gate', 15))
    def run(volumes, distance=gate):
        pairs = [match(a, b, max_centroid_distance=distance) for a, b in zip(volumes, volumes[1:])]
        return create_tracklets(pairs, skip_matches=[], masks=volumes)
    progress('Stress: baseline rerun')
    baseline = run(window)
    base_edges = association_edges(baseline)
    variants = {}
    def retained(rows, volumes):
        expected = {(t, int(label)) for t, m in enumerate(volumes) for label in np.unique(m) if label > 0}
        actual = {(t, int(label)) for row in rows.values() for t, label in enumerate(row) if label > 0}
        return fraction(len(expected & actual), len(expected))
    progress('Stress: reverse time')
    reverse = {k: list(reversed(v)) for k, v in run(list(reversed(window))).items()}
    variants['reverse_time'] = stability(base_edges, association_edges(reverse))
    variants['reverse_time']['detection_retention'] = retained(reverse, window)
    for factor in (0.9, 1.1):
        progress(f'Stress: consecutive-frame gate × {factor}')
        perturbed = run(window, gate*factor)
        variants[f'gate_{factor}'] = stability(base_edges, association_edges(perturbed))
        variants[f'gate_{factor}']['detection_retention'] = retained(perturbed, window)
    if not 0 <= float(options.get('drop_fraction', .1)) <= 1 or int(options.get('max_removals', 80)) < 0:
        raise ValueError('Invalid stress dropout fraction or removal limit')
    rng = np.random.default_rng(int(options.get('seed', 0)))
    volumes = [np.bincount(m.ravel()) for m in window]
    candidates = []
    for tid, row in baseline.items():
        valid = [t for t in range(1, count-1) if all(row[k] > 0 for k in (t-1, t, t+1))
                 and int(volumes[t][row[t]]) >= settings['min_volume']]
        if valid:
            candidates.append((tid, int(rng.choice(valid))))
    take = min(len(candidates), int(options.get('max_removals', 80)),
               int(np.ceil(len(candidates)*float(options.get('drop_fraction', .1)))))
    chosen = [candidates[i] for i in rng.choice(len(candidates), size=take, replace=False)] if take else []
    deleted = [set() for _ in window]
    for tid, t in chosen:
        deleted[t].add(baseline[tid][t])
    changed = [np.where(np.isin(m, list(deleted[t])), 0, m).astype(m.dtype)
               if deleted[t] else m for t, m in enumerate(window)]
    progress(f'Stress: remove {take} controlled observations')
    dropped = run(changed)
    projected = {k: [(-1 if v in deleted[t] else v) for t, v in enumerate(row)]
                 for k, row in baseline.items()}
    variants['dropout'] = stability(association_edges(projected), association_edges(dropped))
    owner = {(t, label): tid for tid, row in dropped.items() for t, label in enumerate(row) if label > 0}
    recovered = sum(owner.get((t-1, baseline[tid][t-1])) is not None and
                    owner.get((t-1, baseline[tid][t-1])) == owner.get((t+1, baseline[tid][t+1]))
                    for tid, t in chosen)
    variants['dropout'].update(removed=take, recovered=recovered, recovery=fraction(recovered, take))
    variants['dropout']['detection_retention'] = retained(dropped, changed)
    saved = {k: row[start:start+count] for k, row in saved_rows.items()}
    return dict(scope='Current default tracker on this mask set only; historical and unified pipelines are not rerun.',
                start=start, frames=count, seed=int(options.get('seed', 0)),
                baseline_saved_agreement=stability(association_edges(saved), base_edges),
                baseline_links=len(base_edges), variants=variants,
                parameter_scope='Only consecutive-frame distance gate is perturbed; skip recovery keeps its default gate.')


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for key in ('reference_masks',):
        config[key] = str((path.parent/config[key]).resolve())
    for item in config['datasets']:
        for key in ('masks', 'tracklets'):
            item[key] = str((path.parent/item[key]).resolve())
    config['settings'] = {**DEFAULTS, **config.get('settings', {})}
    s = config['settings']
    if s['min_volume'] < 0 or not 0 < s['mapping_iou'] < 1 or s['xy_margin'] < 0 or s['z_margin'] < 0:
        raise ValueError('Invalid eligibility or mapping settings')
    scale = np.asarray(s['voxel_size_um'], dtype=float)
    if scale.shape != (3,) or not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError('voxel_size_um must contain three positive finite values')
    if s['motion_threshold_um_per_frame'] <= 0:
        raise ValueError('Motion threshold must be positive')
    names = [x['name'] for x in config['datasets']]
    if not names or len(names) != len(set(names)):
        raise ValueError('Dataset names must be unique and nonempty')
    return config


def fingerprint(config):
    files = {}
    for directory in {config['reference_masks'], *(x['masks'] for x in config['datasets'])}:
        for path in mask_paths_by_t(directory).values():
            stat = Path(path).stat()
            files[str(path)] = [stat.st_size, stat.st_mtime_ns]
    for item in config['datasets']:
        files[item['tracklets']] = hashlib.sha256(Path(item['tracklets']).read_bytes()).hexdigest()
    for path in [Path(__file__), Path(__file__).with_name('track.py')]:
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps([config, files], sort_keys=True).encode()).hexdigest()


def evaluate(config, progress=print):
    s = config['settings']
    reference_paths = mask_paths_by_t(config['reference_masks'])
    ts = sorted(reference_paths); n = len(ts)
    if ts != list(range(n)) or not n:
        raise ValueError('Reference masks must cover consecutive frames starting at t0')
    datasets = []
    for spec in config['datasets']:
        paths = mask_paths_by_t(spec['masks'])
        if sorted(paths) != ts:
            raise ValueError(f"{spec['name']}: frame range differs from reference")
        rows, owners = load_rows(spec['tracklets'], n)
        datasets.append(dict(spec=spec, paths=paths, rows=rows, owners=owners,
                             props=[], mappings=[], mapping_counts=[]))
    reference_props = []
    shape = None
    for t in ts:
        if t % 8 == 0:
            progress(f'Evaluating frame {t+1}/{n}')
        cache = {}
        ref = load_masks(reference_paths[t])
        if ref.ndim != 3 or (shape is not None and tuple(ref.shape) != shape):
            raise ValueError('Expected consistent 3D mask shapes')
        shape = tuple(ref.shape)
        fp = features(ref); reference_props.append(fp)
        cache[str(Path(reference_paths[t]).resolve())] = (ref, fp)
        for d in datasets:
            path = str(Path(d['paths'][t]).resolve())
            if path not in cache:
                volume = load_masks(path)
                cache[path] = (volume, features(volume))
            volume, props = cache[path]
            if set(d['owners'][t])-set(props):
                raise ValueError(f"{d['spec']['name']}: track references missing label at t={t}")
            if path == str(Path(reference_paths[t]).resolve()):
                mapping = {label: label for label in fp}
                counts = dict(unmapped_reference=0, unmapped_output=0, split_like=0, merge_like=0)
            else:
                mapping, counts = map_labels(ref, volume, s['mapping_iou'])
            d['props'].append(props); d['mappings'].append(mapping); d['mapping_counts'].append(counts)
    scores = []
    for d in datasets:
        score = score_dataset(d['spec']['name'], d['rows'], d['owners'], d['props'],
                              d['mappings'], reference_props, shape, s)
        score['mapping_counts'] = {k: sum(c[k] for c in d['mapping_counts'])
                                   for k in d['mapping_counts'][0]}
        score['paths'] = {k: d['spec'][k] for k in ('masks', 'tracklets')}
        scores.append(score)
    stress = None
    options = config.get('stress', {})
    if options.get('enabled', False):
        target = next((d for d in datasets if d['spec']['name'] == options.get('dataset')), None)
        if target is None:
            raise ValueError('Stress dataset name does not match a configured dataset')
        # Current tracker uses its documented voxel ratio; avoid silently applying another calibration.
        if not np.allclose(np.asarray(s['voxel_size_um'])/s['voxel_size_um'][1], [2.52, 1, 1], rtol=.01):
            raise ValueError('Current default stress runner requires voxel ratio (2.52,1,1)')
        stress_start = int(options.get('start', 0))
        stress_n = int(options.get('frames', 8))
        # Load just the window, preserving explicit global start metadata.
        if stress_start < 0 or stress_n < 3 or stress_start+stress_n > n:
            raise ValueError('Stress window must fit within the recording and have >=3 frames')
        masks = [load_masks(target['paths'][t]) for t in range(stress_start, stress_start+stress_n)]
        rows = {k: row[stress_start:stress_start+stress_n] for k, row in target['rows'].items()}
        stress = stress_suite(masks, rows, s, {**options, 'start': 0}, progress)
        stress.update(start=stress_start, dataset=target['spec']['name'])
    return dict(schema_version=1, fingerprint=fingerprint(config), title=config.get('title', 'Tracking evaluation'),
                frames=n, settings=s, datasets=scores, stress=stress,
                reference_masks=config['reference_masks'],
                notes=[
                    'These metrics measure consistency, not identity accuracy; there is no ground truth.',
                    'Reference masks define fixed eligibility and the t0 XY-interior cohort.',
                    'Cross-mask comparisons use one-to-one IoU mapping; segmentation splits/merges can affect coverage.',
                    'Interior endings exclude both XY and Z boundary bands. Z-border endings are reported separately as uncertain.',
                    'Last-frame endings are excluded. No biological deaths are inferred; division events are not modeled.',
                    'Cohort curves keep a fixed denominator; later exits are not silently removed.',
                    'Motion residual is change in physical velocity across three observations, accounting for frame gaps.',
                    'Stress results apply only to the stated runner and window, not all saved tracking pipelines.',
                ])
