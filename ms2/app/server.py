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
runs/ in it, and may write nothing outside it.  --data-root folders, if given, may be read
too: pasting a server path opens that file in the browser.

    ms2 app --analysis-only FOLDERS_FILE --password-file FILE --out-root DIR --host 0.0.0.0

serves the Analysis page alone, behind the password only, over the result folders listed in
FOLDERS_FILE (one per line, read again on every request, so editing it needs no restart).
Everyone shares the analyses saved in DIR.
"""
import argparse
import glob
import hashlib
import hmac
import io
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd
from PIL import Image

from ms2.app import analysis, jobs
from ms2.cell_activity_compare import (ALIGNMENT_NAME, align_comparison, cell_projection,
                                       comparison_alignment, stage_frame)
from ms2.cell_activity_segments import (ACTIVITY_DIR, SEGMENT_LABELS, analyse_segments,
                                        load_segments, save_segments)
from ms2.pipeline import Recording
from ms2.preprocess import make_projection
from ms2.viewer import app as viewer

STATIC = os.path.join(os.path.dirname(__file__), 'static')
mimetypes.add_type('image/webp', '.webp')     # missing from older Pythons' table
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class App:
    def __init__(self, data_roots, out_root, workspace=None, password=None, read_roots=(),
                 roots_file=None):
        self.data_roots = [os.path.realpath(r) for r in data_roots]
        self.read_roots = [os.path.realpath(r) for r in read_roots]   # lab mode: open, not write
        self.out_root = out_root
        self.workspace = workspace and os.path.realpath(workspace)
        self.password = password
        self.roots_file = roots_file        # analysis-only mode: its folders replace data_roots
        # per user ('' when the app is someone's own): the New-analysis form, kept while they
        # are on the ROI page, and the recording their viewer pages show
        self.forms = {}
        self.viewers = {}
        self.version = code_version()

    def roots(self, user=''):
        if self.roots_file:
            return read_roots_file(self.roots_file)
        return [os.path.join(self.workspace, user)] if self.workspace else self.data_roots

    def out(self, user=''):
        return os.path.join(self.workspace, user, 'runs') if self.workspace else self.out_root

    def allowed(self, path, user='', roots=None):
        real = os.path.realpath(path)
        return any(real == r or real.startswith(r + os.sep) for r in roots or self.roots(user))

    def check_read(self, path, user=''):
        """A file to open in the browser: the user's own folders, their results folder, or the
        read-only ones."""
        if not os.path.isabs(path):
            raise ValueError(f'{path} is not a full path; it should start with /')
        if not (self.allowed(path, user, self.roots(user) + [os.path.realpath(self.out(user))])
                or self.read_roots and self.allowed(path, roots=self.read_roots)):
            raise PermissionError(f'{path} is outside the folders this page may open: '
                                  f'{", ".join(self.roots(user) + self.read_roots)}')

    def check(self, *paths, user=''):
        for p in paths:
            if p and not os.path.isabs(p):
                raise ValueError(f'{p} is not a full path; it should start with /')
            if not p or not self.allowed(p, user):
                raise PermissionError(f'{p or "(empty path)"} is outside the data folders '
                                      f'{", ".join(self.roots(user))}')

    def login(self, name, password):
        """The user's folder name, or None if the name or the password is refused."""
        name = 'visitor' if self.roots_file else (name or '').strip().lower()   # password only
        if not USER_NAME.fullmatch(name) \
                or not hmac.compare_digest((password or '').encode(), self.password.encode()):
            return None
        return name

    def _sign(self, name):
        return hmac.new(self.password.encode(), name.encode(), hashlib.sha256).hexdigest()

    def cookie(self, name):
        """The login cookie's value: the name, signed with the lab password.  Changing the
        password logs everyone out; nothing has to be kept on the server."""
        return f'{name}:{self._sign(name)}'

    def user_of(self, cookie_header):
        """The user a request's login cookie names, or None."""
        try:
            morsel = SimpleCookie(cookie_header or '').get(COOKIE)
        except Exception:
            return None
        name, _, sig = (morsel.value if morsel else '').partition(':')
        if USER_NAME.fullmatch(name) and hmac.compare_digest(sig, self._sign(name)):
            return name
        return None


COOKIE = 'ms2login'
PAGES = ('/', '/view', '/roi', '/segments', '/analysis')        # opened by hand: sent to the login form, not refused
USER_NAME = re.compile(r'[a-z0-9][a-z0-9_-]{0,39}')   # becomes a folder name
PATH_FIELDS = ('czi', 'out', 'dir', 'path', 'root')
ANALYSIS_PATHS = ('/analysis', '/login', '/file', '/api/login', '/api/logout', '/api/config')


def analysis_path(path):
    """What analysis-only mode serves: the Analysis page and what it calls."""
    return path in ANALYSIS_PATHS or path.startswith(('/api/analysis/', '/static/'))


def read_roots_file(path):
    """The folders listed in path, one per line; # starts a comment."""
    with open(path) as f:
        lines = [line.split('#')[0].strip() for line in f]
    roots = [os.path.realpath(line) for line in lines if line]
    if not roots:
        raise FileNotFoundError(f'{path} lists no folders')
    return roots


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


def code_version(repo=REPO):
    """The commit this app's code came from, read once at startup: a later git pull is not
    running until the app restarts.  commit is None outside a git checkout."""
    def git(*args):
        return subprocess.run(['git', '-C', repo, *args], capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
    try:
        short, full, date = git('log', '-1', '--format=%h|%H|%cI').split('|')
        return {'commit': short, 'full': full, 'date': date,
                'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
                'modified': bool(git('status', '--porcelain', '--untracked-files=no'))}
    except (OSError, subprocess.SubprocessError, ValueError):
        return {'commit': None}


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


def comparison_recordings(cmp_dir):
    with open(os.path.join(cmp_dir, 'recordings.json')) as f:
        return json.load(f)


def alignment_state(cmp_dir):
    """Each recording of a comparison with its window's first frame and its turn: the one
    saved for this comparison, else the one saved with the recording's ROI."""
    path = os.path.join(cmp_dir, ALIGNMENT_NAME)
    saved = {}
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
    rows = []
    for name, rec in comparison_recordings(cmp_dir).items():
        row = {'name': name, 't': rec['t_start'], 'flip': False, 'rotate': 0,
               **(rec.get('orientation') or {}), **saved.get(name, {})}
        try:
            cell_projection(rec)
        except FileNotFoundError as exc:
            row['error'] = str(exc)
        rows.append(row)
    aligned = os.path.join(cmp_dir, 'spatial_maps_aligned.png')
    return {'recordings': rows, 'map': aligned if os.path.exists(aligned) else None}


def segments_state(cmp_dir):
    """What the Segments page needs: each recording's window, image size and orientation (as
    the aligned map turns it), the segments saved for this comparison -- else those marked
    with the recording's ROI -- and the labels to mark."""
    from ms2.roi_selection import load_projection
    recordings = comparison_recordings(cmp_dir)
    alignment = comparison_alignment(cmp_dir, recordings) or {}
    rows = []
    for name, rec in recordings.items():
        row = {'name': name, 't_start': rec['t_start'], 't_end': rec['t_end'],
               'orientation': {'flip': False, 'rotate': 0, **alignment.get(name, {})}}
        try:
            row['height'], row['width'] = load_projection(cell_projection(rec)).shape[1:]
        except FileNotFoundError as exc:
            row['error'] = str(exc)
        rows.append(row)
    segments = {name: rec['segments'] for name, rec in recordings.items() if rec.get('segments')}
    segments.update(load_segments(cmp_dir))
    return {'recordings': rows, 'segments': segments, 'labels': SEGMENT_LABELS}


def frame_png(frame, size=512):
    """A cell-channel frame as a greyscale png, its contrast from the 1st-99.5th percentile;
    shrunk to fit size, or at full resolution when size is None."""
    lo, hi = np.percentile(frame, [1, 99.5])
    img = Image.fromarray((np.clip((frame - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8))
    if size:
        img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    return buf.getvalue()


_resolved = {}
KMEANS_LOCK = threading.Lock()


def warm_up():
    """Fit KMeans once before serving.  Its first fit makes scikit-learn's threadpoolctl walk
    the loaded libraries, holding the dynamic loader's lock while its ctypes callback waits for
    Python's lock -- and a request thread being started holds Python's lock while it waits for
    the loader's.  Done concurrently by request threads, the two deadlocked the whole server.
    After this fit the walk is cached and requests never make it."""
    from ms2.cell_activity import classify_cells
    desc = pd.DataFrame({'rate': np.linspace(1, 50, 30), 'peak': np.linspace(20, 90, 30)})
    with KMEANS_LOCK:
        classify_cells(desc, score='rate')


def analysis_recording(csv_path):
    """A result's inputs (analysis.resolve), remembered while its matrix is unchanged."""
    key = (os.path.abspath(csv_path), os.path.getmtime(csv_path))
    if key not in _resolved:
        _resolved[key] = analysis.resolve(csv_path)
    return _resolved[key]


def analysis_outputs(folder):
    """An exported analysis's figures and tables, figures first; segment ones after."""
    found = [f for pat in ('*', os.path.join(ACTIVITY_DIR, '*')) for f in glob.glob(os.path.join(glob.escape(folder), pat))
             if f.endswith(('.png', '.gif', '.csv')) and os.path.isfile(f)]
    found.sort(key=lambda f: (ACTIVITY_DIR in f, f.endswith('.csv'), f))
    return [{'name': os.path.relpath(f, folder), 'path': f} for f in found]


def activity_files(cmp_dir):
    """The segment activity results already in a comparison folder, figures first."""
    files = sorted(glob.glob(os.path.join(cmp_dir, ACTIVITY_DIR, '*')),
                   key=lambda f: (f.endswith('.csv'), f))
    return [{'name': os.path.basename(f), 'path': f} for f in files
            if f.endswith(('.png', '.gif', '.csv'))]


def save_alignment(cmp_dir, alignment):
    """Write alignment.json -- {name: {flip, rotate}}, every recording of the comparison -- and
    redraw the aligned spatial map.  Returns the map's path."""
    names = list(comparison_recordings(cmp_dir))
    if set(alignment) != set(names):
        raise ValueError(f'the alignment names {sorted(alignment)}, the comparison {sorted(names)}')
    clean = {n: {'flip': bool(alignment[n].get('flip')), 'rotate': float(alignment[n].get('rotate', 0))}
             for n in names}
    with open(os.path.join(cmp_dir, ALIGNMENT_NAME), 'w') as f:
        json.dump(clean, f, indent=2)
    return align_comparison(cmp_dir)


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

    def orientation_saved(self, roi_path):
        jobs.redo_after_orientation_change(jobs.recording(os.path.dirname(os.path.abspath(roi_path))))

    def get_viewer(self):
        return self.app.viewers.get(self.user)

    def set_viewer(self, v):
        self.app.viewers[self.user] = v

    def authorized(self, path):
        """In lab mode, every request but the login form's needs the login cookie.  In
        analysis-only mode, the rest of the app is not there at all."""
        if self.app.roots_file and not analysis_path(path):
            if path == '/':
                self.send_response(302)
                self.send_header('Location', '/analysis')
                self.end_headers()
            else:
                self.send_error(404)
            return False
        if self.app.password is None:
            return True
        user = self.app.user_of(self.headers.get('Cookie'))
        if user is not None:
            self.user = user
            if self.app.workspace:
                os.makedirs(os.path.join(self.app.workspace, user, 'runs'), exist_ok=True)
            return True
        if path in ('/login', '/api/login'):
            return True
        if self.command == 'GET' and path in PAGES:
            self.send_response(302)
            self.send_header('Location', '/login')
            self.end_headers()
        else:
            if self.command == 'PUT':
                self.close_connection = True   # don't read the upload that was refused
            self._error(PermissionError('you are logged out: reload the page to log in'), 401)
        return False

    def _login_reply(self, value, max_age):
        body = json.dumps({'ok': True}).encode()
        self.send_response(200)
        self.send_header('Set-Cookie', f'{COOKIE}={value}; Path=/; Max-Age={max_age}; '
                                       f'HttpOnly; SameSite=Lax')
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), 'application/json')

    def _error(self, exc, code=None):
        if code is None:
            code = 403 if isinstance(exc, PermissionError) else 400 if isinstance(exc, (ValueError, KeyError, FileNotFoundError)) else 500
        body = json.dumps({'error': str(exc)}).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if not self.authorized(url.path):
            return
        q = clean_paths({k: v[0] for k, v in parse_qs(url.query).items()})
        a = self.app
        try:
            if url.path == '/':
                with open(os.path.join(STATIC, 'index.html'), 'rb') as f:
                    self._send(f.read(), 'text/html; charset=utf-8')
            elif url.path == '/analysis':
                with open(os.path.join(STATIC, 'analysis.html'), 'rb') as f:
                    self._send(f.read(), 'text/html; charset=utf-8')
            elif url.path == '/api/analysis/recordings':
                roots = [q['root']] if q.get('root') else a.roots(self.user) + a.read_roots
                for r in roots:
                    a.check_read(r, self.user)
                self._json({'roots': roots, 'recordings': analysis.discover(roots)})
            elif url.path == '/api/analysis/resolve':
                # a result added by its path: the matrix, or the folder holding it
                path = q['path']
                a.check_read(path, self.user)
                if os.path.isdir(path):
                    found = analysis.find_matrices(path, depth=2)
                    if not found:
                        raise FileNotFoundError(f'no gene expression results in {path}')
                    path = found[0]
                self._json(analysis.recording_info(analysis_recording(path)))
            elif url.path == '/api/analysis/sample':
                a.check_read(q['csv'], self.user)
                rec = analysis_recording(q['csv'])
                if not rec['tracklets'] or not rec['masks_dir']:
                    raise FileNotFoundError(f'the tracks or masks of {q["csv"]} were not found')
                self._json(analysis.sample_data(rec, int(q['t0']), int(q['t1']),
                                                os.path.join(a.out(self.user), 'analyses')))
            elif url.path == '/api/analysis/labels':
                a.check_read(q['csv'], self.user)
                rec = analysis_recording(q['csv'])
                a.check_read(rec['masks_dir'], self.user)
                self._send(analysis.label_png(rec['masks_dir'], int(q['t'])), 'image/png', cache=True)
            elif url.path == '/api/analysis/files':
                a.check_read(q['csv'], self.user)
                roots = [q['root']] if q.get('root') else a.roots(self.user) + a.read_roots
                self._json(analysis.outline_files(analysis_recording(q['csv']), roots))
            elif url.path == '/api/analysis/file':
                a.check_read(q['path'], self.user)
                self._json(analysis.read_outline_file(q['path'], q.get('name') or None))
            elif url.path == '/api/analysis/saved':
                self._json(analysis.list_analyses(a.out(self.user)))
            elif url.path == '/api/analysis/load':
                self._json(analysis.load_analysis(a.out(self.user), q['name']))
            elif url.path == '/api/analysis/export':
                folder = analysis.analysis_dir(a.out(self.user), q['name'])
                self._json({'state': jobs.state(folder), 'log': jobs.log_tail(folder),
                            'files': analysis_outputs(folder)})
            elif url.path == '/api/analysis/zip':
                folder = analysis.analysis_dir(a.out(self.user), q['name'])
                if jobs.state(folder) == 'running':
                    raise ValueError(f'{q["name"]} is still being exported')
                if not os.path.exists(os.path.join(folder, 'cells.csv')):
                    raise FileNotFoundError(f'{q["name"]} has not been exported yet')
                zip_name = re.sub(r'[^A-Za-z0-9._-]+', '_', q['name']) + '_results.zip'
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{zip_name}"')
                self.end_headers()          # no length: the zip is written as it is made
                analysis.write_zip(folder, self.wfile)
            elif url.path == '/api/analysis/frame':
                a.check_read(q['csv'], self.user)
                rec = analysis_recording(q['csv'])
                if not rec['projection']:
                    raise FileNotFoundError(f'no cell-channel image found for {q["csv"]}')
                a.check_read(rec['projection'], self.user)
                from ms2.roi_selection import load_projection
                frames = load_projection(rec['projection'])
                t = int(q['t'])
                if not 0 <= t < len(frames):
                    raise ValueError(f'frame {t} is outside 0..{len(frames) - 1}')
                self._send(frame_png(np.asarray(frames[t], dtype=float), int(q.get('size', 512)) or None),
                           'image/png', cache=True)
            elif url.path == '/segments':
                with open(os.path.join(STATIC, 'segments.html'), 'rb') as f:
                    self._send(f.read(), 'text/html; charset=utf-8')
            elif url.path == '/login':
                with open(os.path.join(STATIC, 'login.html'), 'rb') as f:
                    self._send(f.read(), 'text/html; charset=utf-8')
            elif url.path == '/api/login':
                self._json({'ask_name': not a.roots_file})      # the form asks for a name or not
            elif url.path.startswith('/open/'):
                # the path is in the URL itself, so a page's relative links (its images,
                # scripts) resolve to their files next to it
                path = unquote(url.path[len('/open'):])
                a.check_read(path, self.user)
                if not os.path.isfile(path):
                    raise FileNotFoundError(f'{path} is not a file')
                kind = mimetypes.guess_type(path)[0] or 'application/octet-stream'
                with open(path, 'rb') as f:
                    self._send(f.read(), kind)
            elif url.path == '/api/config':
                self._json({'data_roots': a.roots(self.user), 'out_root': a.out(self.user),
                            'gpus': gpus(), 'form': a.forms.get(self.user, {}),
                            'lab': bool(a.workspace), 'user': self.user, 'version': a.version,
                            'analysis_only': bool(a.roots_file)})
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
            elif url.path == '/api/align':
                self.check(q['dir'])
                self._json(alignment_state(q['dir']))
            elif url.path == '/api/align/frame':
                self.check(q['dir'])
                rec = comparison_recordings(q['dir'])[q['name']]
                a.check_read(cell_projection(rec), self.user)
                t = int(q.get('t', rec['t_start']))
                if not rec['t_start'] <= t <= rec['t_end']:
                    raise ValueError(f'frame {t} is outside {q["name"]}\'s window')
                # full=1: every pixel, for outlines drawn at image precision
                self._send(frame_png(stage_frame(rec, t), None if q.get('full') else 512),
                           'image/png', cache=True)
            elif url.path == '/api/segments':
                self.check(q['dir'])
                self._json(segments_state(q['dir']))
            elif url.path == '/api/segments/activity':
                self.check(q['dir'])
                self._json({'files': activity_files(q['dir'])})
            elif url.path.startswith('/static/'):
                name = url.path[len('/static/'):]
                path = os.path.join(STATIC, name)
                if '/' in name or name.startswith('.') or not os.path.isfile(path):
                    return self.send_error(404)
                with open(path, 'rb') as f:
                    self._send(f.read(), mimetypes.guess_type(name)[0] or 'application/octet-stream',
                               cache=True)
            elif url.path == '/api/download':
                self.check(q['dir'])
                name = os.path.basename(os.path.realpath(q['dir']))
                self.send_response(200)
                self.send_header('Content-Type', 'application/zip')
                self.send_header('Content-Disposition', f'attachment; filename="{name}_results.zip"')
                self.end_headers()          # no length: the zip is written as it is made
                zip_results(q['dir'], self.wfile)
            elif url.path == '/file':
                a.check_read(q['path'], self.user)
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
        url = urlparse(self.path)
        if not self.authorized(url.path):
            return
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
        url = urlparse(self.path)
        if not self.authorized(url.path):
            return
        if not url.path.startswith('/api/'):
            return super().do_POST()            # /roi/preview, /roi/save
        a = self.app
        try:
            body = clean_paths(json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}'))
            if url.path == '/api/login':
                if a.password is None:
                    raise ValueError('this app has no login; it is not in lab mode')
                name = a.login(body.get('name'), body.get('password'))
                if name is None:
                    time.sleep(1)               # slows down guessing the password
                    raise PermissionError('wrong password, or a name that is not letters, '
                                          'digits, - or _')
                self._login_reply(a.cookie(name), 30 * 24 * 3600)
            elif url.path == '/api/logout':
                self._login_reply('', 0)
            elif url.path == '/api/roi/open':
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
                recs = body['recordings'].values()
                self.check(body['out'], *[r['czi'] for r in recs], *[r['out'] for r in recs if 'out' in r])
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
            elif url.path == '/api/analysis/save':
                self._json({'path': analysis.save_analysis(a.out(self.user), body['name'], body['analysis'])})
            elif url.path == '/api/analysis/export':
                # saved first, then exported in the background: the comparison takes a minute
                folder = os.path.dirname(analysis.save_analysis(a.out(self.user), body['name'], body['analysis']))
                if jobs.state(folder) == 'running':
                    raise ValueError(f'{body["name"]} is already being exported')
                jobs.start([sys.executable, '-m', 'ms2.app.analysis', 'export', folder], folder, 'export.log')
                self._json({'ok': True, 'folder': folder})
            elif url.path == '/api/analysis/ladder':
                # the automatic ladder: KMeans as classify_cells fits it, given as rate cuts;
                # one fit at a time (see warm_up)
                from ms2.cell_activity import classify_cells, ladder_cuts
                desc = pd.DataFrame({'rate': body['rate'], 'peak': body['peak']}, dtype=float)
                with KMEANS_LOCK:
                    levels = classify_cells(desc, score='rate', noise_peak=float(body['noise_peak']))
                self._json({'cuts': ladder_cuts(desc, levels)})
            elif url.path == '/api/segments/activity':
                # seconds: positions from cells.csv and traces from the expression tables
                self.check(body['dir'])
                analyse_segments(body['dir'], int(body.get('bins', 8)), int(body.get('time_bins', 4)),
                                 body.get('segments') or None,
                                 int(body['cells_per_bin']) if body.get('cells_per_bin') else None)
                self._json({'files': activity_files(body['dir'])})
            elif url.path == '/api/segments/save':
                self.check(body['dir'])
                self._json({'path': save_segments(body['dir'], body['segments'],
                                                  comparison_recordings(body['dir']))})
            elif url.path == '/api/align/save':
                self.check(body['dir'])
                self._json({'map': save_alignment(body['dir'], body['alignment'])})
            else:
                self.send_error(404)
        except Exception as exc:
            self._error(exc)

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', action='append',
                   help='folder the file browser may open (repeat for several); default: your home. '
                        'In lab mode: folders people may open files from, read-only')
    p.add_argument('--out-root', help='default results folder')
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--workspace', help='lab mode: one page for everyone, with a folder per user '
                                       'here for their uploads and runs (needs --password-file)')
    p.add_argument('--password-file', help='file holding the lab password, asked at login')
    p.add_argument('--analysis-only', metavar='FOLDERS_FILE',
                   help='serve the Analysis page alone, over the result folders listed in this '
                        'file, one per line (needs --password-file and --out-root)')
    p.add_argument('--host', default='127.0.0.1',
                   help='address to listen on; 0.0.0.0 to be reached from other computers')
    args = p.parse_args()
    if args.workspace and not args.password_file:
        p.error('--workspace needs --password-file: the name people log in with picks their folder')
    if args.analysis_only and not (args.password_file and args.out_root):
        p.error('--analysis-only needs --password-file, and --out-root for the saved analyses')
    password = None
    if args.password_file:
        with open(args.password_file) as f:
            password = f.read().strip()
        if not password:
            p.error(f'{args.password_file} is empty')
    roots = args.data_root or [os.path.expanduser('~')]
    # lab mode reads the --data-root folders only when they are given, never a default home
    Handler.app = App(roots, args.out_root or roots[0], args.workspace, password,
                      read_roots=(args.data_root or []) if args.workspace else [],
                      roots_file=args.analysis_only and os.path.realpath(args.analysis_only))
    if args.analysis_only:
        read_roots_file(args.analysis_only)     # a missing or empty list fails here, not at login
    warm_up()
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError:
        sys.exit(f'port {args.port} is in use -- another ms2 app is running, or pick another '
                 f'port with --port (and the same port in your launcher settings)')
    if args.analysis_only:
        print(f'ms2 analysis page on port {args.port}, listening on {args.host}   '
              f'(result folders listed in {args.analysis_only})', flush=True)
    elif args.workspace:
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
