"""ms2/app/jobs.py: detached jobs report done / failed / stopped from their folder alone, and
the Runs page finds recordings and their progress from the pipeline's markers.

    python tests/test_app.py
"""
import os
import sys
import tempfile
import time

from fixtures import passed

from ms2.app import jobs
from ms2.app.server import App, clean_paths
from ms2.pipeline import Recording


def wait(job_dir, timeout=10):
    end = time.time() + timeout
    while jobs.state(job_dir) == 'running' and time.time() < end:
        time.sleep(0.05)
    return jobs.state(job_dir)


def test_job_outcomes():
    with tempfile.TemporaryDirectory() as d:
        assert jobs.state(d) == 'none'
        jobs.start([sys.executable, '-c', 'print("hello")'], d, 'run.log')
        assert wait(d) == 'done'
        assert 'hello' in jobs.log_tail(d)
        jobs.start([sys.executable, '-c', 'raise SystemExit(3)'], d, 'run.log')
        assert wait(d) == 'failed'
        jobs.start([sys.executable, '-c', 'import time; time.sleep(60)'], d, 'run.log')
        assert jobs.state(d) == 'running'
        jobs.stop(d)
        assert wait(d) == 'stopped'
    passed('a job is done, failed or stopped, read back from its folder')


def test_log_keeps_last_progress_state():
    with tempfile.TemporaryDirectory() as d:
        jobs.start([sys.executable, '-c', 'print("a 10%\\ra 50%\\ra 100%")'], d, 'run.log')
        wait(d)
        assert jobs.log_tail(d).splitlines()[-1] == 'a 100%', jobs.log_tail(d)
    passed('progress bars show their last state')


def test_find_recordings_and_status():
    with tempfile.TemporaryDirectory() as root:
        rec = Recording('/data/rec-1.czi', root)
        os.makedirs(os.path.join(rec.dir, 'unified'))
        for step in ('preprocess', 'segment'):
            open(rec.marker(step), 'w').close()
        os.makedirs(os.path.join(root, 'empty'))
        os.makedirs(os.path.join(root, '_comparison'))
        open(os.path.join(root, '_comparison', 'job.json'), 'w').close()
        assert jobs.find_recordings(root) == [rec.dir]
        s = jobs.recording_status(rec.dir)
        assert s['name'] == 'rec-1' and s['steps_done'] == ['preprocess', 'segment'], s
        assert s['state'] == 'none' and s['roi'] is None
    passed('recordings are found by their markers; comparisons are not recordings')


def test_new_roi_reruns_gene_expression():
    with tempfile.TemporaryDirectory() as root:
        rec = Recording('/data/rec-1.czi', root)
        os.makedirs(rec.dir)
        for step in ('unify', 'gene_expression', 'activity'):
            open(rec.marker(step), 'w').close()
        os.utime(rec.marker('gene_expression'), (1, 1))
        with open(rec.roi, 'w') as f:
            f.write('{}')
        jobs.redo_after_roi_change(rec)
        assert os.path.exists(rec.marker('unify'))
        assert not os.path.exists(rec.marker('gene_expression'))
        assert not os.path.exists(rec.marker('activity'))
    passed('an ROI newer than gene expression sends it and activity round again')


def test_pipeline_command():
    cmd = jobs.pipeline_cmd('/d/r.czi', '/o', 'cuda:1', '/o/r/roi_selection.json', 18.0, 5, True)
    assert cmd[1:] == ['-m', 'ms2.pipeline', '/d/r.czi', '--out', '/o', '--device', 'cuda:1',
                       '--roi', '/o/r/roi_selection.json', '--prominence', '18.0',
                       '--bg-radius', '5', '--plots'], cmd
    assert jobs.czi_of('/o/r', {'cmd': cmd}) == '/d/r.czi'
    passed('pipeline command and the CZI it names')


def test_pasted_paths():
    q = clean_paths({'czi': ' "/data/a b.czi" ', 'out': "'/out'", 'device': '"cuda:0"'})
    assert q == {'czi': '/data/a b.czi', 'out': '/out', 'device': '"cuda:0"'}, q
    app = App(['/data'], '/data')
    app.check('/data/a b.czi')
    for bad, err in (('data/a.czi', ValueError), ('/etc/passwd', PermissionError)):
        try:
            app.check(bad)
        except err:
            continue
        raise AssertionError(f'{bad} accepted')
    passed('quotes around pasted paths are dropped; relative and outside paths refused')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('app: all tests passed')


if __name__ == '__main__':
    main()
