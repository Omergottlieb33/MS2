"""One recording from its CZI to the per-cell activity table.

    python -m ms2.pipeline <rec>.czi --out <root> [--device cuda:0] [--roi roi.json]
                           [--stop-after STEP]

Steps, each writing into <root>/<rec>/ and marking itself done there, so a rerun resumes
where the last one stopped:

    preprocess        preprocess/C1-<rec>.tif, C2-<rec>.tif, C1-<rec>_bg_removed.tif, SUM_C2-<rec>.tif
    segment           masks/z_stack_t{t}_seg_masks.npz               (Cellpose, GPU)
    track             tracklets.json                                   (ms2/tracking.py)
    unify             unified/masks_split/, unified/tracklets_unified.json
    gene_expression   gene_expression/gene_expression_results.csv  (per-cell figures with --plots)
    activity          activity/clusters.csv and figures, over the ROI's frames (all without --roi)

To pick cells by ROI, draw the boundaries first -- in `ms2 app`, or with
`ms2 view --projection preprocess/SUM_C2-<rec>.tif` -- on the two frames that bound the
stage, and pass the saved file as --roi.  Gene expression keeps the cells inside either
boundary and activity is measured between the two frames.  Without --roi, gene expression
keeps cells with fewer than 20 missing frames and activity covers the whole recording.
"""
import argparse
import datetime
import json
import os
import subprocess

STEPS = ['preprocess', 'segment', 'track', 'unify', 'gene_expression', 'activity']


class Recording:
    def __init__(self, czi, root):
        self.czi = os.path.abspath(czi)
        self.name = os.path.splitext(os.path.basename(czi))[0]
        self.root = os.path.abspath(root)
        self.dir = os.path.join(self.root, self.name)
        self.preprocess = os.path.join(self.dir, 'preprocess')
        self.ms2_bg_removed = os.path.join(self.preprocess, f'C1-{self.name}_bg_removed.tif')
        self.projection = os.path.join(self.preprocess, f'SUM_C2-{self.name}.tif')
        self.roi = os.path.join(self.dir, 'roi_selection.json')    # where the app saves one
        self.masks = os.path.join(self.dir, 'masks')    # where segment_3d_cells writes
        self.tracklets = os.path.join(self.dir, 'tracklets.json')
        self.unified = os.path.join(self.dir, 'unified')
        self.unified_masks = os.path.join(self.unified, 'masks_split')
        self.unified_tracklets = os.path.join(self.unified, 'tracklets_unified.json')
        self.gene_expression = os.path.join(self.dir, 'gene_expression')
        self.expression_csv = os.path.join(self.gene_expression, 'gene_expression_results.csv')
        self.activity = os.path.join(self.dir, 'activity')

    def marker(self, step):
        return os.path.join(self.dir, f'.done_{step}')


def run_step(rec, step, args):
    # Imported per step: segmentation pulls in torch/cellpose, which the other steps don't need.
    if step == 'preprocess':
        from ms2.preprocess import preprocess
        preprocess(rec.czi, rec.preprocess, radius=args.bg_radius, workers=args.workers)
    elif step == 'segment':
        from ms2.segment import segment_3d_cells
        segment_3d_cells(rec.czi, rec.root, args.device)
    elif step == 'track':
        from ms2.tracking import cell_tracking
        cell_tracking(rec.masks, output_path=rec.tracklets)
    elif step == 'unify':
        from ms2.track_unify import unify
        unify(rec.masks, rec.tracklets, rec.unified)
    elif step == 'gene_expression':
        from ms2.gene_expression.run import run_gene_expression
        run_gene_expression(rec.czi, rec.unified_masks, rec.unified_tracklets, rec.ms2_bg_removed,
                            rec.gene_expression, args.prominence, roi=args.roi, plots=args.plots,
                            workers=args.workers or os.cpu_count())
    elif step == 'activity':
        import pandas as pd
        from ms2.cell_activity import analyze_recording
        t_start, t_end = activity_window(args.roi, len(pd.read_csv(rec.expression_csv)))
        analyze_recording(rec.name, {'csv': rec.expression_csv, 'tracklets': rec.unified_tracklets,
                                     'masks_dir': rec.unified_masks,
                                     't_start': t_start, 't_end': t_end},
                          rec.activity)


def activity_window(roi, n_frames):
    """The frames activity is measured over: the ROI's window when there is one, else all."""
    if roi:
        from ms2.roi_selection import load_roi
        r = load_roi(roi)
        t_last = r.get('t_last')
        return r.get('t_first', 0), n_frames - 1 if t_last is None or t_last < 0 else t_last
    return 0, n_frames - 1


def git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(czi, root, args):
    rec = Recording(czi, root)
    os.makedirs(rec.dir, exist_ok=True)
    last = STEPS.index(args.stop_after) if args.stop_after else len(STEPS) - 1
    for step in STEPS[:last + 1]:
        if os.path.exists(rec.marker(step)):
            print(f'[{rec.name}] {step}: done, skipping')
            continue
        print(f'[{rec.name}] {step}')
        run_step(rec, step, args)
        with open(rec.marker(step), 'w') as f:
            f.write(datetime.datetime.now().isoformat())
    with open(os.path.join(rec.dir, 'run_manifest.json'), 'w') as f:
        json.dump({'czi': rec.czi, 'git': git_hash(), 'finished': datetime.datetime.now().isoformat(),
                   'steps_done': [s for s in STEPS if os.path.exists(rec.marker(s))],
                   'params': {k: v for k, v in vars(args).items() if k not in ('czi', 'out')}},
                  f, indent=2)
    return rec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('czi')
    p.add_argument('--out', required=True, help='results root; the recording gets a sub-folder')
    p.add_argument('--device', default='cuda:0', help='torch device for Cellpose')
    p.add_argument('--roi', help='ROI json from the viewer; selects the cells for gene expression')
    p.add_argument('--stop-after', choices=STEPS)
    p.add_argument('--prominence', type=float, default=18.0, help='MS2 maxima-finder prominence')
    p.add_argument('--bg-radius', type=float, default=5, help='background subtraction radius')
    p.add_argument('--workers', type=int,
                   help='parallel workers for background subtraction and gene expression (default: all cores)')
    p.add_argument('--plots', action='store_true',
                   help='write the per-cell gene-expression figures (slow; off by default)')
    return p.parse_args(argv)


def main():
    args = parse_args()
    run(args.czi, args.out, args)


if __name__ == '__main__':
    main()
