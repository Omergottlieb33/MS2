"""src/pipeline.py: steps run in order, finished steps are skipped on a rerun, and
--stop-after stops there (the ROI pause).  The steps themselves are stubbed out.

    python tests/test_pipeline.py
"""
import json
import os
import tempfile

from fixtures import passed

import src.pipeline as pipeline


def run_stubbed(root, argv):
    calls = []
    real = pipeline.run_step
    pipeline.run_step = lambda rec, step, args: calls.append(step)
    try:
        args = pipeline.parse_args(['/data/rec-1.czi', '--out', root] + argv)
        rec = pipeline.run(args.czi, args.out, args)
    finally:
        pipeline.run_step = real
    return calls, rec


def test_stop_after_then_resume_runs_each_step_once():
    with tempfile.TemporaryDirectory() as root:
        calls, rec = run_stubbed(root, ['--stop-after', 'unify'])
        assert calls == ['preprocess', 'segment', 'track', 'unify'], calls
        calls, rec = run_stubbed(root, ['--roi', 'roi.json'])
        assert calls == ['gene_expression', 'activity'], calls
        with open(os.path.join(rec.dir, 'run_manifest.json')) as f:
            manifest = json.load(f)
        assert manifest['steps_done'] == pipeline.STEPS, manifest
        assert manifest['params']['roi'] == 'roi.json', manifest
    passed('--stop-after pauses, the rerun resumes at gene_expression')


def test_layout_is_named_after_the_czi():
    rec = pipeline.Recording('/data/New-03-v1.czi', '/out')
    assert rec.dir == '/out/New-03-v1'
    assert rec.masks == '/out/New-03-v1/masks'   # where segment_3d_cells(czi, root) writes
    assert rec.ms2_bg_removed == '/out/New-03-v1/preprocess/C1-New-03-v1_bg_removed.tif'
    passed('per-recording layout')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('pipeline: all tests passed')


if __name__ == '__main__':
    main()
