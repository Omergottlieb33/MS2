"""Several recordings from their CZIs to the stage comparison.

    python -m ms2.pipeline_compare experiment.json [--device cuda:0] [--workers N]

experiment.json names the results root and, per recording, its czi and the stage window
(0-based frames, both included); roi, noise_floor, noise_peak and min_presence are optional,
and so is out, the results root of a recording run somewhere else than the experiment's out:

    {"out": "/zjbd/.../outputs/stage_comparison",
     "recordings": {
        "New-02-v3-ST11-12": {"czi": "/zjbd/.../New-02-v3-ST11-12.czi",
                              "t_start": 0, "t_end": 60, "roi": ".../roi_selection.json"},
        "New-03-v1-ST11-12": {"czi": "/zjbd/.../New-03-v1-ST11-12.czi",
                              "t_start": 0, "t_end": 75, "out": "/zjbd/.../outputs/stage_11"}}}

Every recording goes through ms2/pipeline.py into <its out>/<czi name>/ (finished steps are
skipped, so recordings already processed cost nothing).  The orientation saved with a
recording's ROI travels into the config, so the aligned spatial map needs no aligning, and
so do the segments marked there, which the Segments page starts from.  Then <out>/_comparison/ gets the
generated recordings.json and everything ms2/cell_activity_compare.py writes: cells.csv,
timecourse.csv, metrics.csv, stats.csv and the figures.
"""
import argparse
import json
import os

from ms2 import pipeline
from ms2.cell_activity_compare import compare_recordings

OPTIONAL_KEYS = ('noise_floor', 'noise_peak', 'min_presence')


def activity_config(experiment):
    """The {name: recording dict} ms2/cell_activity_compare.py reads, pointing into the
    pipeline's layout."""
    config = {}
    for name, entry in experiment['recordings'].items():
        rec = pipeline.Recording(entry['czi'], entry.get('out', experiment['out']))
        config[name] = {'csv': rec.expression_csv, 'tracklets': rec.unified_tracklets,
                        'masks_dir': rec.unified_masks,
                        't_start': entry['t_start'], 't_end': entry['t_end'],
                        **{k: entry[k] for k in OPTIONAL_KEYS if k in entry}}
        orientation = pipeline.load_orientation(rec)
        if orientation:
            config[name]['orientation'] = orientation
        if os.path.exists(rec.segments):
            with open(rec.segments) as f:
                config[name]['segments'] = json.load(f)
    return config


def run(experiment, device='cuda:0', workers=None, plots=False, ladder='joint'):
    for name, entry in experiment['recordings'].items():
        argv = [entry['czi'], '--out', entry.get('out', experiment['out']), '--device', device]
        if entry.get('roi'):
            argv += ['--roi', entry['roi']]
        if workers:
            argv += ['--workers', str(workers)]
        if plots:
            argv.append('--plots')
        args = pipeline.parse_args(argv)
        pipeline.run(args.czi, args.out, args)

    out_dir = os.path.join(experiment['out'], '_comparison')
    os.makedirs(out_dir, exist_ok=True)
    config = activity_config(experiment)
    with open(os.path.join(out_dir, 'recordings.json'), 'w') as f:
        json.dump(config, f, indent=2)
    return compare_recordings(config, out_dir, ladder=ladder)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('experiment', help='experiment json (see above)')
    p.add_argument('--device', default='cuda:0', help='torch device for Cellpose')
    p.add_argument('--workers', type=int, help='parallel workers (default: all cores)')
    p.add_argument('--plots', action='store_true', help='write the per-cell gene-expression figures')
    p.add_argument('--ladder', choices=('joint', 'per_recording'), default='joint',
                   help='one activity ladder over all recordings (default) or one per recording')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.experiment) as f:
        experiment = json.load(f)
    metrics = run(experiment, args.device, args.workers, args.plots, args.ladder)
    print(metrics.set_index('recording').T.to_string())


if __name__ == '__main__':
    main()
