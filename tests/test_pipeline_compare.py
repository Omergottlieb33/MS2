"""ms2/pipeline_compare.py: every recording goes through the pipeline with its own ROI, and
the comparison config points into the pipeline's layout with each recording's window.  The
pipeline and the comparison are stubbed out.

    python tests/test_pipeline_compare.py
"""
import json
import os
import tempfile

from fixtures import passed

import ms2.pipeline_compare as pc


def test_runs_each_recording_and_writes_the_comparison_config():
    with tempfile.TemporaryDirectory() as out:
        experiment = {'out': out, 'recordings': {
            'A': {'czi': '/data/rec-a.czi', 't_start': 0, 't_end': 60, 'roi': '/r/a.json',
                  'noise_floor': 25},
            'B': {'czi': '/data/rec-b.czi', 't_start': 5, 't_end': 40, 'out': '/elsewhere'}}}
        runs, compared = [], []
        real_run, real_compare = pc.pipeline.run, pc.compare_recordings
        pc.pipeline.run = lambda czi, root, args: runs.append((czi, root, args.roi, args.device))
        pc.compare_recordings = lambda config, out_dir, ladder: compared.append((config, out_dir))
        try:
            pc.run(experiment, device='cuda:2')
        finally:
            pc.pipeline.run, pc.compare_recordings = real_run, real_compare

        assert runs == [('/data/rec-a.czi', out, '/r/a.json', 'cuda:2'),
                        ('/data/rec-b.czi', '/elsewhere', None, 'cuda:2')], runs
        config, out_dir = compared[0]
        assert out_dir == os.path.join(out, '_comparison')
        assert config['A'] == {
            'csv': os.path.join(out, 'rec-a', 'gene_expression', 'gene_expression_results.csv'),
            'tracklets': os.path.join(out, 'rec-a', 'unified', 'tracklets_unified.json'),
            'masks_dir': os.path.join(out, 'rec-a', 'unified', 'masks_split'),
            't_start': 0, 't_end': 60, 'noise_floor': 25}, config['A']
        assert (config['B']['t_start'], config['B']['t_end']) == (5, 40)
        assert config['B']['csv'] == '/elsewhere/rec-b/gene_expression/gene_expression_results.csv'
        with open(os.path.join(out_dir, 'recordings.json')) as f:
            assert json.load(f) == config
    passed('each recording runs with its ROI in its own results root; the config uses the '
           'pipeline layout and windows')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('pipeline_compare: all tests passed')


if __name__ == '__main__':
    main()
