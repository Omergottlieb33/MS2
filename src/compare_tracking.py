"""Compare saved tracking outputs and verify mask/track consistency.

Usage: python -m src.compare_tracking --root RECORDING_ROOT --run RUN_DIRECTORY
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from src.track import get_border_labels
from src.track_diagnostics import cell_properties, load_masks, mask_paths_by_t

def summarize(track_path, masks_dir, cache_path):
    tracks = json.loads(Path(track_path).read_text())
    props = cell_properties(str(masks_dir), str(cache_path))
    ts = sorted(props)
    rows = np.asarray(list(tracks.values()), dtype=int)
    if not len(rows):
        rows = np.empty((0, len(ts)), dtype=int)
    if rows.shape[1] != len(ts):
        raise ValueError('Track lengths do not match masks')
    active = rows > 0
    lengths = active.sum(axis=1)
    gaps = unresolved = possible_exits = 0
    missing = duplicate = absent = 0
    jump_flags = links = zero_overlap = 0
    jumps = []
    paths = mask_paths_by_t(str(masks_dir))
    previous = None
    borders = []
    for i, t in enumerate(ts):
        m = load_masks(paths[t])
        borders.append(get_border_labels(m))
        labels = set(np.unique(m).tolist()) - {0}
        owned = rows[:,i][rows[:,i] > 0]
        missing += len(labels - set(owned))
        duplicate += len(owned) - len(set(owned))
        absent += len(set(owned) - labels)
        if previous is not None:
            both = (previous > 0) & (m > 0)
            stride = int(m.max()) + 1
            overlap = set(np.unique(previous[both].astype(np.int64)*stride + m[both]).tolist())
            for row in rows:
                a,b = row[i-1:i+1]
                if a <= 0 or b <= 0:
                    continue
                p,q = props[ts[i-1]][int(a)], props[t][int(b)]
                delta = np.asarray(q[:3])-p[:3]
                jumps.append(float(np.linalg.norm(delta*[2.52,1,1])))
                jump_flags += int(abs(delta[0]) > 3 or abs(q[3]-p[3])/max(p[3],1) > 1.5)
                links += 1
                zero_overlap += int(int(a)*stride+int(b) not in overlap)
        previous = m
    for row in rows:
        inds = np.flatnonzero(row > 0)
        if not len(inds):
            continue
        first,last = inds[[0,-1]]
        gaps += int(np.sum(row[first:last+1] == -1))
        if last < len(ts)-1:
            if row[last] in borders[last]:
                possible_exits += 1
            else:
                unresolved += 1
    if duplicate or absent:
        raise AssertionError(f'Duplicate claims={duplicate}, absent references={absent}')
    mean_volumes = [np.mean([props[ts[i]][int(label)][3] for i,label in enumerate(row) if label > 0])
                    if np.any(row > 0) else 0 for row in rows]
    full_size = lengths[np.asarray(mean_volumes) >= 600]
    return dict(full_size_tracks=len(full_size),
                full_size_mean_active=float(full_size.mean()) if len(full_size) else 0,
                frames=len(ts), tracks=len(rows), active_detections=int(active.sum()),
                mean_active_frames=float(lengths.mean()) if len(rows) else 0,
                median_active_frames=float(np.median(lengths)) if len(rows) else 0,
                full_movie_tracks=int(np.sum(lengths == len(ts))), singleton_tracks=int(np.sum(lengths == 1)),
                unowned_detections=missing, duplicate_claims=duplicate, absent_labels=absent,
                unresolved_interior_endings=unresolved, possible_xy_exits=possible_exits,
                internal_gap_frames=gaps, consecutive_links=links, zero_overlap_links=zero_overlap,
                heuristic_jump_flags=jump_flags,
                scaled_jump_p95=float(np.percentile(jumps,95)) if jumps else 0)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    specs = {
        'previous_unified': (args.root/'unify/tracklets_unified.json',args.root/'unify/masks_split'),
        'corrected_default': (args.run/'tracklets_default.json',args.root/'masks'),
        'corrected_unified': (args.run/'unified/tracklets_unified.json',args.run/'unified/masks_split'),
    }
    results={}
    for name,(tracks,masks) in specs.items():
        print('Auditing',name,flush=True)
        results[name]=summarize(tracks,masks,args.run/(name+'_audit_properties.pkl'))
    results['notes']=[
        'No biological deaths are assumed. Interior endings are unresolved identity/detection losses.',
        'Possible exits use the same XY mask-contact rule for every output; Z exits remain unresolved.',
        'Tracks include fragments and single-frame detections. Counts are not identity accuracy.',
        'Full-size subgroup uses mean segmented volume >= 600 voxels for all outputs; this is a heuristic, not ground truth.',
        'Unified masks differ. More unowned objects can result from deliberate fragment retirement.',
        'Zero overlap and jump flags are inspection candidates, not confirmed identity errors.',
    ]
    results['inputs']={name:dict(tracklets=str(t),masks=str(m),sha256=hashlib.sha256(t.read_bytes()).hexdigest())
                      for name,(t,m) in specs.items()}
    log=json.loads((args.run/'unified/tracklets_unified_log.json').read_text())
    results['repair_events']=dict(collections.Counter(e['kind'] for e in log))
    (args.run/'comparison.json').write_text(json.dumps(results,indent=2))
    keys=['tracks','full_size_tracks','full_size_mean_active','mean_active_frames','median_active_frames','full_movie_tracks','singleton_tracks',
          'active_detections','unowned_detections','unresolved_interior_endings','internal_gap_frames',
          'zero_overlap_links','heuristic_jump_flags','duplicate_claims','absent_labels']
    lines=['Tracking validation: New-03, 72 frames','',
           '| Metric | Previous unified | Corrected default | Corrected unified |',
           '|---|---:|---:|---:|']
    for key in keys:
        vals=[results[name][key] for name in specs]
        vals=[f'{v:.2f}' if isinstance(v,float) else str(v) for v in vals]
        lines.append('| '+key.replace('_',' ')+' | '+' | '.join(vals)+' |')
    lines+=['']+results['notes']+['','Visualization inputs:']
    for name,(t,m) in specs.items():
        lines+=['',name, '- Tracklets: '+str(t), '- Masks: '+str(m)]
    (args.run/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    main()
