"""ms2 app: run the pipeline from the browser.

    ms2 app [--data-root DIR ...] [--out-root DIR] [--port 8765]
    ms2 app --workspace DIR --password-file FILE --host 0.0.0.0 [--port 8765]

Pick a CZI, draw the ROI on its first and last stage frames, start the run, follow it and
open the results -- all on the machine that holds the data and the GPUs.  From a laptop,
launchers/ starts this over ssh and opens the browser; on the machine itself, open
http://localhost:<port>.

Runs are detached: closing the browser, the app or the ssh session leaves them running,
and the Runs page picks them up again from their folders.

With --workspace the app is one shared page for the lab, left running on the server: people
log in with their name and the lab password, upload a CZI from their own computer, and
download the results as a zip.  Each name gets its own folder, DIR/<name>, with uploads/ and
runs/ in it, and may open nothing outside it.
"""
import argparse
import base64
import glob
import hmac
import json
import mimetypes
import os
import re
import subprocess
import sys
import zipfile
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ms2.app import jobs
from ms2.pipeline import Recording
from ms2.preprocess import make_projection
from ms2.viewer import app as viewer

STATIC = os.path.join(os.path.dirname(__file__), 'static')


class App:
    def __init__(self, data_roots, out_root, workspace=None, password=None):
        self.data_roots = [os.path.realpath(r) for r in data_roots]
        self.out_root = out_root
        self.workspace = workspace and os.path.realpath(workspace)
        self.password = password
        # per user ('' when the app is someone's own): the New-analysis form, kept while they
        # are on the ROI page, and the recording their viewer pages show
        self.forms = {}
        self.viewers = {}

    def roots(self, user=''):
        return [os.path.join(self.workspace, user)] if self.workspace else self.data_roots

    def out(self, user=''):
        return os.path.join(self.workspace, user, 'runs') if self.workspace else self.out_root

    def allowed(self, path, user=''):
        real = os.path.realpath(path)
        return any(real == r or real.startswith(r + os.sep) for r in self.roots(user))

    def check(self, *paths, user=''):
        for p in paths:
            if p and not os.path.isabs(p):
                raise ValueError(f'{p} is not a full path; it should start with /')
            if not p or not self.allowed(p, user):
                raise PermissionError(f'{p or "(empty path)"} is outside the data folders '
                                      f'{", ".join(self.roots(user))}')

    def login(self, header):
        """The user name from an HTTP Basic Authorization header, or None if it is refused."""
        try:
            kind, cred = (header or '').split(' ', 1)
            name, password = base64.b64decode(cred).decode().split(':', 1)
        except ValueError:
            return None
        name = name.strip().lower()
        if kind != 'Basic' or not USER_NAME.fullmatch(name) \
                or not hmac.compare_digest(password.encode(), self.password.encode()):
            return None
        return name


USER_NAME = re.compile(r'[a-z0-9][a-z0-9_-]{0,39}')   # becomes a folder name
PATH_FIELDS = ('czi', 'out', 'dir', 'path', 'root')


def clean_paths(fields):
    """Paths as typed or pasted: Windows' "Copy as path" wraps them in quotes."""
    return {k: v.strip().strip('"\'').strip() if k in PATH_FIELDS and isinstance(v, str) else v
            for k, v in fields.items()}


def gpus():
    """[{id, label}] with free memory, so people can steer clear of a busy card."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        out = ''
    found = []
    for line in out.strip().splitlines():
        i, name, used, total = [s.strip() for s in line.split(',')]
        free = (int(total) - int(used)) / 1024
        found.append({'id': f'cuda:{i}', 'label': f'GPU {i} · {name} · {free:.0f} GB free'})
    return found + [{'id': 'cpu', 'label': 'CPU (no GPU; segmentation is very slow)'}]


def listdir(path):
    entries = []
    for name in sorted(os.listdir(path), key=str.lower):
        if name.startswith('.'):
            continue
        full = os.path.join(path, name)
        if os.path.isdir(full):
            entries.append({'name': name, 'path': full, 'dir': True})
        elif name.lower().endswith('.czi'):
            entries.append({'name': name, 'path': full, 'dir': False})
    return entries


def results(rec_dir):
    """The files worth showing for a recording, newest-step first."""
    rec = jobs.recording(rec_dir)
    patterns = [os.path.join(rec.activity, '*'), os.path.join(rec.gene_expression, '*.csv'),
                os.path.join(rec.dir, 'dashboard', 'dashboard.html')]
    # the per-cell cell_<id>_*.csv files run to thousands; the summary tables are what people open
    files = [f for pat in patterns for f in sorted(glob.glob(pat))
             if os.path.isfile(f) and not os.path.basename(f).startswith('cell_')]
    return [{'name': os.path.relpath(f, rec_dir), 'path': f} for f in files]


def mount_viewer(handler, rec_dir, for_roi):
    """Point the user's viewer pages (/view, /roi) at one recording."""
    rec = jobs.recording(rec_dir)
    has = os.path.exists
    tracks = has(rec.unified_tracklets)
    image = os.path.join(rec.preprocess, f'C2-{rec.name}.tif')
    peaks = sorted(glob.glob(os.path.join(rec.gene_expression, 'peak_to_cell_matching_prominence_*.csv')))
    v = viewer.Viewer(
        None if for_roi or not has(image) else image,
        rec.unified_masks if tracks else None, viewer.CELL_CHANNEL,
        rec.unified_tracklets if tracks else None,
        None if for_roi or not has(rec.ms2_bg_removed) else rec.ms2_bg_removed,
        rec.projection if has(rec.projection) else None,
        None if for_roi or not peaks else peaks[-1])
    v.roi_out = rec.roi
    v.next_url = '/'
    handler.set_viewer(v)
    return v


def save_upload(rfile, length, folder, name):
    """Write an uploaded CZI into folder; a broken-off upload leaves nothing named .czi."""
    if not name.lower().endswith('.czi') or name.startswith('.') or re.search(r'[/\\\0]', name):
        raise ValueError(f'{name!r} is not a .czi file name')
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    part = path + '.part'
    try:
        with open(part, 'wb') as f:
            left = length
            while left:
                chunk = rfile.read(min(left, 8 << 20))
                if not chunk:
                    raise ValueError(f'the upload of {name} broke off; try again')
                f.write(chunk)
                left -= len(chunk)
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)
    return path


def zip_results(rec_dir, out):
    """Logs, ROI and the result tables of a recording, zipped into out (may be a socket)."""
    files = sorted(glob.glob(os.path.join(rec_dir, '*.log')))
    files += [os.path.join(rec_dir, n) for n in ('roi_selection.json', 'run_manifest.json')
              if os.path.exists(os.path.join(rec_dir, n))]
    files += [f['path'] for f in results(rec_dir)]
    name = os.path.basename(rec_dir)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, os.path.join(name, os.path.relpath(f, rec_dir)))


class Handler(viewer.Handler):
    """The viewer's handler plus the app's own pages and /api; one port for everything."""
    app = None
    user = ''

    def check(self, *paths):
        self.app.check(*paths, user=self.user)

    def check_path(self, path):
        self.check(path)            # the viewer's results-folder box stays inside the data roots

    def get_viewer(self):
        return self.app.viewers.get(self.user)

    def set_viewer(self, v):
        self.app.viewers[self.user] = v

    def authorized(self):
        """In lab mode, ask the browser for a name and the lab password until it gives them."""
        if self.app.password is None:
            return True
        self.user = self.app.login(self.headers.get('Authorization'))
        if self.user is not None:
            os.makedirs(os.path.join(self.app.workspace, self.user, 'runs'), exist_ok=True)
            return True
        self.user = ''
        body = (b'Log in with your name (letters, digits, - or _) as the user name '
                b'and the lab password.')
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="MS2", charset="UTF-8"')
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), 'application/json')

    def _error(self, exc):
        code = 403 if isinstance(exc, PermissionError) else 400 if isinstance(exc, (ValueError, KeyError, FileNotFoundError)) else 500
        body = json.dumps({'error': str(exc)}).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self.authorized():
            return
        url = urlparse(self.path)
        q = clean_paths({k: v[0] for k, v in parse_qs(url.query).items()})
        a = self.app
        try:
            if url.path == '/':
                with open(os.path.join(STATIC, 'index.html'), 'rb') as f:
                    self._send(f.read(), 'text/html; charset=utf-8')
            elif url.path == '/api/config':
                self._json({'data_roots': a.roots(self.user), 'out_root': a.out(self.user),
                            'gpus': gpus(), 'form': a.forms.get(self.user, {}),
                            'lab': bool(a.workspace), 'user': self.user})
            elif url.path == '/api/ls':
                self.check(q['path'])
                self._json({'path': os.path.realpath(q['path']),
                            'parent': os.path.dirname(os.path.realpath(q['path'])),
                            'entries': listdir(q['path'])})
            elif url.path == '/api/runs':
                self.check(q['root'])
                self._json([jobs.recording_status(d) for d in jobs.find_recordings(q['root'])])
            elif url.path == '/api/run':
                self.check(q['dir'])
                self._json({**jobs.recording_status(q['dir']), 'log': jobs.log_tail(q['dir']),
                            'results': results(q['dir']),
                            'dashboard': jobs.state(os.path.join(q['dir'], 'dashboard'))})
            elif url.path == '/api/compare':
                self.check(q['out'])
                d = os.path.join(q['out'], '_comparison')
                self._json({'state': jobs.state(d), 'log': jobs.log_tail(d),
                            'files': [{'name': os.path.basename(f), 'path': f}
                                      for f in sorted(glob.glob(os.path.join(d, '*')))
                                      if os.path.isfile(f) and not f.endswith(('job.json', 'exit_code'))]})
            elif url.path == '/api/download':
                self.check(q['dir'])
                name = os.path.basename(os.path.realpath(q['dir']))
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{name}_results.zip"')
                self.end_headers()          # no length: the zip is written as it is made
                zip_results(q['dir'], self.wfile)
            elif url.path == '/file':
                self.check(q['path'])
                kind = mimetypes.guess_type(q['path'])[0] or 'application/octet-stream'
                with open(q['path'], 'rb') as f:
                    self._send(f.read(), kind)
            elif url.path.startswith('/api/'):
                self.send_error(404)
            else:
                super().do_GET()                # /roi, /projection, /meta, /frame, ...
        except Exception as exc:
            self._error(exc)

    def do_PUT(self):
        if not self.authorized():
            return
        url = urlparse(self.path)
        try:
            if url.path != '/api/upload' or not self.app.workspace:
                return self.send_error(404)
            name = parse_qs(url.query)['name'][0]
            folder = os.path.join(self.app.workspace, self.user, 'uploads')
            self._json({'path': save_upload(self.rfile, int(self.headers['Content-Length']),
                                            folder, name)})
        except Exception as exc:
            self.close_connection = True    # the rest of a refused upload is still on its way
            self._error(exc)

    def do_POST(self):
        if not self.authorized():
            return
        url = urlparse(self.path)
        if not url.path.startswith('/api/'):
            return super().do_POST()            # /roi/preview, /roi/save
        a = self.app
        try:
            body = clean_paths(json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}'))
            if url.path == '/api/roi/open':
                # Draw the ROI before anything has run: only the projection is needed.
                self.check(body['czi'], body['out'])
                a.forms[self.user] = body
                rec = Recording(body['czi'], body['out'])
                if not os.path.exists(rec.projection):
                    make_projection(rec.czi, rec.preprocess)
                mount_viewer(self, rec.dir, for_roi=True)
                self._json({'url': '/roi'})
            elif url.path == '/api/roi/edit':
                self.check(body['dir'])
                rec = jobs.recording(body['dir'])
                if not os.path.exists(rec.projection):
                    make_projection(jobs.czi_of(rec.dir, jobs.read_job(rec.dir)), rec.preprocess)
                mount_viewer(self, rec.dir, for_roi=True)
                self._json({'url': '/roi'})
            elif url.path == '/api/view':
                self.check(body['dir'])
                mount_viewer(self, body['dir'], for_roi=False)
                self._json({'url': '/view'})
            elif url.path == '/api/run/start':
                self.check(body['czi'], body['out'])
                rec = Recording(body['czi'], body['out'])
                if jobs.state(rec.dir) == 'running':
                    raise ValueError(f'{rec.name} is already running')
                roi = rec.roi if body.get('use_roi', True) and os.path.exists(rec.roi) else None
                if roi:
                    jobs.redo_after_roi_change(rec)
                cmd = jobs.pipeline_cmd(rec.czi, rec.root, body.get('device', 'cuda:0'), roi,
                                        body.get('prominence'), body.get('bg_radius'),
                                        body.get('plots', False))
                jobs.start(cmd, rec.dir, 'pipeline.log')
                a.forms.pop(self.user, None)
                self._json({'dir': rec.dir})
            elif url.path == '/api/run/stop':
                self.check(body['dir'])
                jobs.stop(body['dir'])
                self._json({'ok': True})
            elif url.path == '/api/dashboard':
                self.check(body['dir'])
                rec = jobs.recording(body['dir'])
                cmd = [sys.executable, '-m', 'ms2.tracking_dashboard',
                       '--tracklets', rec.unified_tracklets, '--masks-dir', rec.unified_masks,
                       '--output-dir', os.path.join(rec.dir, 'dashboard'), '--evaluate-only']
                if os.path.exists(rec.roi):
                    cmd += ['--roi', rec.roi]
                jobs.start(cmd, os.path.join(rec.dir, 'dashboard'), 'dashboard.log')
                self._json({'ok': True})
            elif url.path == '/api/compare/start':
                self.check(body['out'], *[r['czi'] for r in body['recordings'].values()])
                d = os.path.join(body['out'], '_comparison')
                if jobs.state(d) == 'running':
                    raise ValueError('a comparison is already running in this folder')
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, 'experiment.json')
                with open(path, 'w') as f:
                    json.dump({'out': body['out'], 'recordings': body['recordings']}, f, indent=2)
                cmd = [sys.executable, '-m', 'ms2.pipeline_compare', path,
                       '--device', body.get('device', 'cuda:0'), '--ladder', body.get('ladder', 'joint')]
                jobs.start(cmd, d, 'compare.log')
                self._json({'ok': True})
            else:
                self.send_error(404)
        except Exception as exc:
            self._error(exc)

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', action='append',
                   help='folder the file browser may open (repeat for several); default: your home')
    p.add_argument('--out-root', help='default results folder')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--workspace', help='lab mode: one page for everyone, with a folder per user '
                                       'here for their uploads and runs (needs --password-file)')
    p.add_argument('--password-file', help='file holding the lab password, asked at login')
    p.add_argument('--host', default='127.0.0.1',
                   help='address to listen on; 0.0.0.0 to be reached from other computers')
    args = p.parse_args()
    if args.workspace and not args.password_file:
        p.error('--workspace needs --password-file: the name people log in with picks their folder')
    password = None
    if args.password_file:
        with open(args.password_file) as f:
            password = f.read().strip()
        if not password:
            p.error(f'{args.password_file} is empty')
    roots = args.data_root or [os.path.expanduser('~')]
    Handler.app = App(roots, args.out_root or roots[0], args.workspace, password)
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError:
        sys.exit(f'port {args.port} is in use -- another ms2 app is running, or pick another '
                 f'port with --port (and the same port in your launcher settings)')
    if args.workspace:
        print(f'ms2 app (lab mode) on port {args.port}, listening on {args.host}   '
              f'(user folders in {Handler.app.workspace})', flush=True)
    else:
        print(f'ms2 app on http://localhost:{args.port}   (data folders: {", ".join(Handler.app.data_roots)})',
              flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
