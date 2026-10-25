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
import urllib.parse
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


def test_analysis_only_server():
    """--analysis-only: the password alone logs in, only the Analysis page is served, and the
    folders it may read come from a file read again on every request."""
    with tempfile.TemporaryDirectory() as root:
        a, b, saves = (os.path.join(root, n) for n in ('a', 'b', 'saves'))
        for d in (a, b, saves):
            os.makedirs(d)
        for d in (a, b):
            with open(os.path.join(d, 'x.txt'), 'w') as f:
                f.write(d)
        listed = os.path.join(root, 'folders.txt')
        with open(listed, 'w') as f:
            f.write(f'# shared results\n{a}\n')
        Handler.app = App([root], saves, password='pw', roots_file=listed)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_address[1]}'

        def request(path, cookie=None, method='GET', data=None):
            req = urllib.request.Request(base + path, data=data, method=method)
            if cookie:
                req.add_header('Cookie', cookie)
            return urllib.request.urlopen(req)

        def refused(code, path, cookie=None, method='GET', data=None):
            try:
                request(path, cookie, method, data)
            except urllib.error.HTTPError as e:
                assert e.code == code, (path, e.code)
            else:
                raise AssertionError(f'{path} not refused')
        try:
            with request('/api/login') as r:
                assert json.loads(r.read()) == {'ask_name': False}
            with request('/') as r:                 # to the analysis page, then to the form
                assert r.url.endswith('/login')
            refused(403, '/api/login', None, 'POST', json.dumps({'password': 'no'}).encode())
            with request('/api/login', method='POST', data=json.dumps({'password': 'pw'}).encode()) as r:
                cookie = r.headers['Set-Cookie'].split(';')[0]
            with request('/', cookie) as r:
                assert r.url.endswith('/analysis') and b'MS2 analysis' in r.read()
            for path in ('/api/runs?root=' + a, '/api/ls?path=' + a, '/view', '/roi'):
                refused(404, path, cookie)
            refused(404, '/api/run/start', cookie, 'POST', b'{}')
            cfg = json.loads(request('/api/config', cookie).read())
            assert cfg['analysis_only'] and cfg['data_roots'] == [os.path.realpath(a)], cfg
            assert request('/file?path=' + os.path.join(a, 'x.txt'), cookie).read() == a.encode()
            refused(403, '/file?path=' + os.path.join(b, 'x.txt'), cookie)
            with open(listed, 'a') as f:            # a folder added while the page runs
                f.write(f'{b}  # later\n')
            assert request('/file?path=' + os.path.join(b, 'x.txt'), cookie).read() == b.encode()
            saved = json.loads(request('/api/analysis/save', cookie, 'POST', json.dumps(
                {'name': 's', 'analysis': {'samples': []}}).encode()).read())
            assert saved['path'] == os.path.join(saves, 'analyses', 's', 'analysis.json'), saved
            assert request('/file?path=' + saved['path'], cookie).read()   # exported files show
        finally:
            server.shutdown()
            server.server_close()
    passed('analysis-only server: password alone, nothing but the Analysis page, folders read '
           'from their file on every request, analyses saved in the shared folder')


def comparison_folder(root):
    """A finished comparison of two recordings, a and b, with 16x16 projections and windows
    starting at frame 1; its folder."""
    import numpy as np
    import tifffile
    from fixtures import write_recording
    from ms2.cell_activity_compare import compare_recordings

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
    return cmp_dir


def test_align_endpoints():
    """The Compare page's alignment: each recording's first window frame, saved turns read
    back, a mismatched set of names refused, and the aligned map drawn."""
    with tempfile.TemporaryDirectory() as root:
        cmp_dir = comparison_folder(root)
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


def test_segments_endpoints():
    """The Segments page: each recording's window, size and orientation, a full-resolution
    frame of the window, outlines checked and saved, and read back when it opens again."""
    with tempfile.TemporaryDirectory() as root:
        cmp_dir = comparison_folder(root)
        with open(os.path.join(cmp_dir, 'alignment.json'), 'w') as f:
            json.dump({'b': {'flip': True, 'rotate': 90}}, f)
        # a's segments were marked with its ROI: the page starts from them
        with open(os.path.join(cmp_dir, 'recordings.json')) as f:
            recordings = json.load(f)
        from_roi = {'t': 1, 'orientation': {'flip': False, 'rotate': 0}, 'segments': {'4': [[0, 0], [3, 0], [3, 3]]}}
        recordings['a']['segments'] = from_roi
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

        def refused(path, body, why):
            try:
                call(path, body)
            except urllib.error.HTTPError as e:
                assert e.code == 400 and why in e.read().decode(), why
            else:
                raise AssertionError(f'{why}: not refused')
        square = [[1, 1], [8, 1], [8, 8], [1, 8]]
        try:
            assert b'segment' in call('/segments?dir=' + cmp_dir)
            state = json.loads(call('/api/segments?dir=' + cmp_dir))
            assert state['labels'] == ['2', '3', '4', '5'] and state['segments'] == {'a': from_roi}
            a, b = state['recordings']
            assert (a['name'], a['t_start'], a['t_end'], a['width'], a['height']) == ('a', 1, 3, 16, 16), a
            assert a['orientation'] == {'flip': False, 'rotate': 0}
            assert b['orientation'] == {'flip': True, 'rotate': 90}
            from PIL import Image
            png = call(f'/api/align/frame?dir={cmp_dir}&name=a&t=3&full=1')
            assert Image.open(io.BytesIO(png)).size == (16, 16)
            refused(f'/api/align/frame?dir={cmp_dir}&name=a&t=0', None, 'outside')
            refused('/api/segments/save', {'dir': cmp_dir, 'segments': {'a': {'t': 2, 'segments': {'7': square}}}}, 'not one of')
            refused('/api/segments/save', {'dir': cmp_dir, 'segments': {'a': {'t': 2, 'segments': {'2': square[:2]}}}}, 'at least 3')
            refused('/api/segments/save', {'dir': cmp_dir, 'segments': {'c': {'t': 2, 'segments': {}}}}, 'not in this comparison')
            refused('/api/segments/save', {'dir': cmp_dir, 'segments': {'a': {'t': 9, 'segments': {}}}}, 'outside its window')
            saved = json.loads(call('/api/segments/save', {'dir': cmp_dir, 'segments': {
                'b': {'t': 1, 'orientation': {'flip': True, 'rotate': 90}, 'segments': {'3': square}},
                'a': {'t': 2, 'orientation': {'flip': False, 'rotate': 0}, 'segments': {'5': square, '2': square}}}}))
            with open(saved['path']) as f:
                on_disk = json.load(f)
            assert list(on_disk) == ['a', 'b'] and list(on_disk['a']['segments']) == ['2', '5'], on_disk
            assert on_disk['b'] == {'t': 1, 'orientation': {'flip': True, 'rotate': 90},
                                    'segments': {'3': [[1.0, 1.0], [8.0, 1.0], [8.0, 8.0], [1.0, 8.0]]}}
            assert json.loads(call('/api/segments?dir=' + cmp_dir))['segments'] == on_disk
        finally:
            server.shutdown()
            server.server_close()
    passed('segments: windows and orientations served, outlines checked, saved in order and read back')


def test_segment_activity_endpoints():
    """The Segment activity card: run on a comparison with segments marked, its figures and
    tables listed (figures first), and listed again when the card opens; bad input refused."""
    with tempfile.TemporaryDirectory() as root:
        cmp_dir = comparison_folder(root)
        whole = [[0, 0], [16, 0], [16, 12], [0, 12]]
        with open(os.path.join(cmp_dir, 'segments.json'), 'w') as f:
            json.dump({n: {'t': 1, 'orientation': {'flip': False, 'rotate': 0}, 'segments': {'2': whole}}
                       for n in ('a', 'b')}, f)
        Handler.app = App([root], root)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_address[1]}'

        def call(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            with urllib.request.urlopen(urllib.request.Request(base + path, data=data)) as r:
                return json.loads(r.read())
        try:
            assert call('/api/segments/activity?dir=' + cmp_dir) == {'files': []}
            for bad in ({'bins': 0}, {'segments': ['7']}, {'cells_per_bin': -2}):
                try:
                    call('/api/segments/activity', {'dir': cmp_dir, **bad})
                except urllib.error.HTTPError as e:
                    assert e.code == 400, bad
                else:
                    raise AssertionError(f'{bad} accepted')
            names = [f['name'] for f in call('/api/segments/activity', {
                'dir': cmp_dir, 'bins': 2, 'time_bins': 2, 'segments': ['2']})['files']]
            assert names == ['segment_2_activity.gif', 'segment_2_activity.png',
                             'segment_2_cells_per_bin.png', 'segment_2_clusters.png',
                             'segment_2_intensity.png', 'segment_boxes.png',
                             'segment_2_cells_per_bin.csv', 'segment_2_clusters_correlation.csv',
                             'segment_2_correlation.csv', 'segment_2_intensity_correlation.csv',
                             'segment_activity.csv', 'segment_cells.csv'], names
            assert [f['name'] for f in call('/api/segments/activity?dir=' + cmp_dir)['files']] == names
            call('/api/segments/activity', {'dir': cmp_dir, 'cells_per_bin': 4, 'segments': ['2']})
            with open(os.path.join(cmp_dir, 'segment_activity', 'params.json')) as f:
                assert json.load(f)['cells_per_bin'] == 4
        finally:
            server.shutdown()
            server.server_close()
    passed('segment activity: run from the card, results listed figures first, bad input refused')


def test_analysis_save_and_load_endpoints():
    """The analysis page's saved analyses over http: saved under the results folder, listed,
    loaded back as saved; a name that would leave its folder is refused."""
    with tempfile.TemporaryDirectory() as root:
        Handler.app = App([root], root)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_address[1]}'

        def call(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            with urllib.request.urlopen(urllib.request.Request(base + path, data=data)) as r:
                return json.loads(r.read())
        state = {'mode': 'compare', 'stage': '13', 'samples': [], 'filters': {'presence': 0.9, 'vol_cv': None}}
        try:
            saved = call('/api/analysis/save', {'name': 'stage 13', 'analysis': state})
            assert saved['path'] == os.path.join(root, 'analyses', 'stage 13', 'analysis.json')
            assert [r['name'] for r in call('/api/analysis/saved')] == ['stage 13']
            back = call('/api/analysis/load?name=' + urllib.parse.quote('stage 13'))
            assert back['filters'] == state['filters'] and back['stage'] == '13'
            assert call('/api/analysis/export?name=' + urllib.parse.quote('stage 13')) == {
                'state': 'none', 'log': '', 'files': []}
            try:                                     # nothing to zip before it is exported
                call('/api/analysis/zip?name=' + urllib.parse.quote('stage 13'))
            except urllib.error.HTTPError as e:
                assert e.code == 400 and b'not been exported' in e.read()
            else:
                raise AssertionError('a zip was served for an analysis never exported')
            try:
                call('/api/analysis/save', {'name': '../outside', 'analysis': state})
            except urllib.error.HTTPError as e:
                assert e.code == 400
            else:
                raise AssertionError('a name leaving the analyses folder was saved')
        finally:
            server.shutdown()
            server.server_close()
    passed('analyses save, list and load back over http; names stay inside their folder')


def test_roi_page_saves_the_orientation():
    """The ROI page's orientation lands beside the ROI, comes back when the page reopens, and
    sends a finished run's activity step round again to turn its figures."""
    import numpy as np
    import tifffile
    from ms2.viewer.app import Viewer

    with tempfile.TemporaryDirectory() as root:
        rec = Recording(os.path.join(root, 'rec.czi'), root)
        os.makedirs(rec.preprocess)
        tifffile.imwrite(rec.projection, np.zeros((3, 8, 8), dtype=np.float32))
        open(rec.marker('activity'), 'w').close()
        Handler.app = App([root], root)
        v = Viewer(None, None, 0, projection_path=rec.projection)
        v.roi_out = rec.roi
        Handler.app.viewers[''] = v
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f'http://127.0.0.1:{server.server_address[1]}'

        def call(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            with urllib.request.urlopen(urllib.request.Request(base + path, data=data)) as r:
                return json.loads(r.read())
        try:
            assert call('/roi/meta')['orientation'] is None
            call('/roi/orientation', {'path': rec.roi, 'flip': True, 'rotate': -58})
            assert call('/roi/meta')['orientation'] == {'flip': True, 'rotate': -58.0}
            assert not os.path.exists(rec.marker('activity'))
            assert not os.path.exists(rec.roi)          # turning the view leaves the ROI alone
            tri = [[0, 0], [5, 0], [5, 5]]
            call('/roi/save', {'path': rec.roi, 'roi_first': tri, 't_first': 1, 't_last': 2,
                               'orientation': {'flip': False, 'rotate': 90},
                               'segments': {'3': tri, '2': tri}})
            with open(rec.orientation) as f:
                assert json.load(f) == {'flip': False, 'rotate': 90.0}
            assert os.path.exists(rec.roi)
            # segments on the start frame, at the orientation they were drawn at, in label order
            marked = call('/roi/meta')['segments']
            assert marked['t'] == 1 and marked['orientation'] == {'flip': False, 'rotate': 90}, marked
            assert list(marked['segments']) == ['2', '3'] and marked['segments']['3'] == [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]
            # changed later without the ROI: the ROI file is left as it was
            before = os.path.getmtime(rec.roi)
            call('/roi/orientation', {'path': rec.roi, 'flip': True, 'rotate': 0, 't': 2,
                                      'segments': {'5': tri}})
            assert call('/roi/meta')['segments'] == {'t': 2, 'orientation': {'flip': True, 'rotate': 0},
                                                     'segments': {'5': [[0.0, 0.0], [5.0, 0.0], [5.0, 5.0]]}}
            assert os.path.getmtime(rec.roi) == before
            try:
                call('/roi/orientation', {'path': rec.roi, 'flip': True, 'rotate': 0, 't': 2,
                                          'segments': {'9': tri}})
            except urllib.error.HTTPError as e:
                assert e.code == 500
            else:
                raise AssertionError('segment 9 saved')
        finally:
            server.shutdown()
            server.server_close()
    passed('the ROI page saves the orientation and the segments beside the ROI, and redraws '
           'the activity figures')


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
