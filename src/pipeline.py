"""One recording from its CZI to the per-cell activity table.

    python -m src.pipeline <rec>.czi --out <root> [--device cuda:0] [--roi roi.json]
                           [--stop-after STEP]

Steps, each writing into <root>/<rec>/ and marking itself done there, so a rerun resumes
where the last one stopped:

    preprocess        preprocess/C1-<rec>.tif, C2-<rec>.tif, C1-<rec>_bg_removed.tif
    segment           masks/z_stack_t{t}_seg_masks.npz               (Cellpose, GPU)
    track             tracklets.json                                   (cell_tracking.py)
    unify             unified/masks_split/, unified/tracklets_unified.json
    gene_expression   gene_expression/gene_expression_results.csv  (per-cell figures with --plots)
    activity          activity/clusters.csv and figures, over the whole recording

To pick cells by ROI: run with --stop-after unify, draw the boundaries in the viewer
(python -m src.viewer.app ... on unified/), then rerun with --roi <roi_selection.json>.
Without --roi, gene expression keeps cells with fewer than 20 missing frames.
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
        from src.preprocess import preprocess
        preprocess(rec.czi, rec.preprocess, radius=args.bg_radius, workers=args.workers)
    elif step == 'segment':
        from cell_3d_segmentation import segment_3d_cells
        segment_3d_cells(rec.czi, rec.root, args.device)
    elif step == 'track':
        from cell_tracking import cell_tracking
        cell_tracking(rec.masks, output_path=rec.tracklets)
    elif step == 'unify':
        from src.track_unify import unify
        unify(rec.masks, rec.tracklets, rec.unified)
    elif step == 'gene_expression':
        from ms2_gene_expression import run_gene_expression
        run_gene_expression(rec.czi, rec.unified_masks, rec.unified_tracklets, rec.ms2_bg_removed,
                            rec.gene_expression, args.prominence, roi=args.roi, plots=args.plots,
                            workers=args.workers or os.cpu_count())
    elif step == 'activity':
        import pandas as pd
        from src.cell_activity import analyze_recording
        n = len(pd.read_csv(rec.expression_csv))
        analyze_recording(rec.name, {'csv': rec.expression_csv, 'tracklets': rec.unified_tracklets,
                                     'masks_dir': rec.unified_masks, 't_start': 0, 't_end': n - 1},
                          rec.activity)


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
