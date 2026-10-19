"""ms2/app/jobs.py: detached jobs report done / failed / stopped from their folder alone, and
the Runs page finds recordings and their progress from the pipeline's markers.

    python tests/test_app.py
"""
import io
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer

from fixtures import passed

from ms2.app import jobs
from ms2.app.server import App, Handler, clean_paths, code_version, save_upload, zip_results
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


def test_lab_login_and_folders():
    with tempfile.TemporaryDirectory() as ws:
        app = App([ws], ws, workspace=ws, password='s3cret')
        assert app.login(' Omer ', 's3cret') == 'omer'
        for name, pw in (('omer', 'wrong'), ('../x', 's3cret'), ('', 's3cret'), (None, None)):
            assert app.login(name, pw) is None, (name, pw)
        good = f'ms2login={app.cookie("omer")}'
        assert app.user_of(good) == 'omer'
        for bad in (None, '', 'ms2login=omer:00', 'ms2login=dana:' + app.cookie('omer').split(':')[1],
                    f'other={app.cookie("omer")}'):
            assert app.user_of(bad) is None, bad
        assert App([ws], ws, workspace=ws, password='new').user_of(good) is None
        app.check(os.path.join(ws, 'omer', 'runs', 'r'), user='omer')
        try:
            app.check(os.path.join(ws, 'dana', 'runs', 'r'), user='omer')
        except PermissionError:
            pass
        else:
            raise AssertionError("one user opened another's folder")
        assert app.out('omer') == os.path.join(os.path.realpath(ws), 'omer', 'runs')
    passed('lab login gives a folder name; a user stays inside their own folder')


def test_upload():
    with tempfile.TemporaryDirectory() as d:
        path = save_upload(io.BytesIO(b'x' * 100), 100, d, 'emb 1.czi')
        assert path == os.path.join(d, 'emb 1.czi') and os.path.getsize(path) == 100
        for bad in ('../a.czi', 'a.tif', '.czi', 'a\\b.czi'):
            try:
                save_upload(io.BytesIO(b'x'), 1, d, bad)
            except ValueError:
                continue
            raise AssertionError(f'{bad} accepted')
        try:
            save_upload(io.BytesIO(b'x' * 10), 100, d, 'short.czi')
        except ValueError:
            pass
        assert sorted(os.listdir(d)) == ['emb 1.czi'], os.listdir(d)
    passed('uploads land whole under their own name; a broken-off one leaves nothing')


def fake_results(root):
    rec = Recording('/data/rec-1.czi', root)
    os.makedirs(rec.activity)
    os.makedirs(rec.gene_expression)
    os.makedirs(rec.unified_masks)
    for f in (os.path.join(rec.dir, 'pipeline.log'), rec.roi,
              os.path.join(rec.activity, 'summary.csv'), rec.expression_csv,
              os.path.join(rec.gene_expression, 'cell_7_trace.csv'),
              os.path.join(rec.unified_masks, 'm.npz')):
        with open(f, 'w') as fh:
            fh.write('1')
    return rec


def test_zip_results():
    with tempfile.TemporaryDirectory() as root:
        rec = fake_results(root)
        buf = io.BytesIO()
        zip_results(rec.dir, buf)
        names = sorted(zipfile.ZipFile(buf).namelist())
        assert names == ['rec-1/activity/summary.csv', 'rec-1/gene_expression/gene_expression_results.csv',
                         'rec-1/pipeline.log', 'rec-1/roi_selection.json'], names
    passed('the results zip holds the logs, ROI and tables, not masks or per-cell files')


def test_lab_server():
    """The real server over http: login, upload, config and the results zip."""
    with tempfile.TemporaryDirectory() as ws:
        shared = os.path.join(ws, 'shared')
        os.makedirs(shared)
        with open(os.path.join(shared, 'report.html'), 'w') as f:
            f.write('<p>hi</p>')
        Handler.app = App([ws], ws, workspace=os.path.join(ws, 'users'), password='pw',
                          read_roots=[shared])
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_address[1]}'

        def request(path, cookie=None, method='GET', data=None):
            req = urllib.request.Request(base + path, data=data, method=method)
            if cookie:
                req.add_header('Cookie', cookie)
            return urllib.request.urlopen(req)

        def login(name, password):
            r = request('/api/login', method='POST',
                        data=json.dumps({'name': name, 'password': password}).encode())
            return r.headers['Set-Cookie'].split(';')[0]

        def refused(code, path, cookie=None, method='GET', data=None):
            try:
                request(path, cookie, method, data)
            except urllib.error.HTTPError as e:
                assert e.code == code, (path, e.code)
            else:
                raise AssertionError(f'{path} not refused')

        dana = login('dana', 'pw')
        def get(path, cookie=dana, method='GET', data=None):
            with request(path, cookie, method, data) as r:
                return r.read()
        try:
            refused(401, '/api/config')
            refused(401, '/api/config', 'ms2login=dana:forged')
            refused(401, '/api/upload?name=e.czi', None, 'PUT', b'abc')
            refused(403, '/api/login', None, 'POST', json.dumps({'name': 'dana', 'password': 'no'}).encode())
            with request('/') as r:                     # a page, not refused: sent to the form
                assert r.url.endswith('/login') and b'Lab password' in r.read()
            cfg = json.loads(get('/api/config'))
            assert cfg['lab'] and cfg['user'] == 'dana', cfg
            assert cfg['out_root'] == os.path.join(os.path.realpath(ws), 'users', 'dana', 'runs'), cfg
            up = json.loads(get('/api/upload?name=e.czi', method='PUT', data=b'abc'))
            assert open(up['path'], 'rb').read() == b'abc'
            assert cfg['version'] == code_version(), cfg['version']
            assert get('/static/explainer.webp')[:4] == b'RIFF'
            for bad in ('/static/../server.py', '/static/nothing.png'):
                try:
                    get(bad)
                except urllib.error.HTTPError as e:
                    assert e.code == 404, bad
                else:
                    raise AssertionError(f'{bad} served')
            assert get('/open' + shared + '/report.html') == b'<p>hi</p>'
            refused(403, '/open' + os.path.join(ws, 'users', 'x.html'), dana)   # someone else's folder
            refused(403, '/open/etc/hostname', dana)
            # a run added from another folder is processed there: that root must be the user's too
            elsewhere = {'out': cfg['out_root'], 'recordings': {'r': {
                'czi': os.path.join(cfg['out_root'], 'r.czi'), 'out': '/tmp', 't_start': 0, 't_end': 1}}}
            refused(403, '/api/compare/start', dana, 'POST', json.dumps(elsewhere).encode())
            rec = fake_results(cfg['out_root'])
            z = zipfile.ZipFile(io.BytesIO(get('/api/download?dir=' + rec.dir)))
            assert 'rec-1/activity/summary.csv' in z.namelist()
            refused(403, '/api/download?dir=' + rec.dir, login('omer', 'pw'))
            with request('/api/logout', dana, 'POST', b'{}') as r:
                assert 'Max-Age=0' in r.headers['Set-Cookie']
        finally:
            server.shutdown()
            server.server_close()
    passed('lab server: login form, upload, server files opened read-only, results zipped '
           'for their owner only, logout')


def test_align_endpoints():
    """The Compare page's alignment: each recording's first window frame, saved turns read
    back, a mismatched set of names refused, and the aligned map drawn."""
    import numpy as np
    import tifffile
    from fixtures import write_recording
    from ms2.cell_activity_compare import compare_recordings

    with tempfile.TemporaryDirectory() as root:
        recordings = {}
        for name in ('a', 'b'):
            signals = np.zeros((12, 4))
            signals[[0, 1, 4, 5], 1] = 100.0
            rec = write_recording(os.path.join(root, name), signals)
            rec['projection'] = os.path.join(root, f'SUM_C2-{name}.tif')
            tifffile.imwrite(rec['projection'], np.random.default_rng(0).random((4, 16, 16), dtype=np.float32))
            rec['t_start'] = 1
            recordings[name] = rec
        cmp_dir = os.path.join(root, '_comparison')
        compare_recordings(recordings, cmp_dir)
        with open(os.path.join(cmp_dir, 'recordings.json'), 'w') as f:
            json.dump(recordings, f)

        Handler.app = App([root], root)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_address[1]}'

        def call(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            with urllib.request.urlopen(urllib.request.Request(base + path, data=data)) as r:
                return r.read()
        try:
            state = json.loads(call('/api/align?dir=' + cmp_dir))
            assert [(r['name'], r['t'], r['flip'], r['rotate']) for r in state['recordings']] == \
                [('a', 1, False, 0), ('b', 1, False, 0)], state
            assert state['map'] is None
            assert call(f'/api/align/frame?dir={cmp_dir}&name=a')[:8] == b'\x89PNG\r\n\x1a\n'
            try:
                call('/api/align/save', {'dir': cmp_dir, 'alignment': {'a': {}}})
            except urllib.error.HTTPError as e:
                assert e.code == 400 and b'comparison' in e.read()
            else:
                raise AssertionError('an alignment missing a recording was saved')
            saved = json.loads(call('/api/align/save', {'dir': cmp_dir, 'alignment': {
                'a': {'flip': True, 'rotate': 90}, 'b': {'flip': False, 'rotate': -45}}}))
            assert saved['map'] == os.path.join(cmp_dir, 'spatial_maps_aligned.png')
            assert os.path.exists(saved['map'])
            state = json.loads(call('/api/align?dir=' + cmp_dir))
            assert [(r['flip'], r['rotate']) for r in state['recordings']] == [(True, 90), (False, -45)]
            assert state['map'] == saved['map']
        finally:
            server.shutdown()
            server.server_close()
    passed('alignment: first frames served, turns saved and read back, aligned map drawn')


def test_code_version():
    v = code_version()
    assert v['commit'] and len(v['full']) == 40 and v['full'].startswith(v['commit']), v
    assert isinstance(v['modified'], bool) and v['branch'], v
    with tempfile.TemporaryDirectory() as d:
        assert code_version(d) == {'commit': None}
    passed('the version is the checkout\'s commit, and unknown outside git')


def main():
    for name, test in sorted(globals().items()):
        if name.startswith('test_'):
            test()
    print('app: all tests passed')


if __name__ == '__main__':
    main()
