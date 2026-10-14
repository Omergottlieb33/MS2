"""Browser-based viewer for 3D segmentation masks over the cell channel.

Renders one composited PNG per (t, z) server-side; the browser handles zoom and pan.
Run on the machine holding the data and reach it over an ssh port-forward:

    ssh -L 5000:localhost:5000 <server>
    python -m ms2.viewer.app --image cells.tif --masks-dir .../masks
"""
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from ms2.roi_selection import load_projection, save_roi, select_tracklets
from ms2.viewer.loaders import (CELL_CHANNEL, FamilyStore, MaskStore, Ms2Store, PeakStore,
                                TrackletStore, find_results, load_image, locate)
from ms2.viewer.render import composite, intensity_window

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

# Rendered frames are cached hard by the browser, keyed on their URL.  Change how a frame
# is drawn and the URL stays identical, so the stale picture is what you keep seeing.  The
# page appends this to every frame request, so restarting the server retires the old ones.
BUILD = str(int(time.time()))


class Viewer:
    def __init__(self, image_path, masks_dir, channel, tracklets_path=None, ms2_path=None,
                 projection_path=None, peaks_path=None):
        # No image: an ROI-only viewer on the projection, for drawing before the pipeline runs
        self.image = load_image(image_path, channel) if image_path else None
        self.masks = MaskStore(masks_dir) if masks_dir else None
        self.tracks = TrackletStore(tracklets_path) if tracklets_path else None
        self.ms2 = Ms2Store(ms2_path) if ms2_path else None
        self.peaks = PeakStore(peaks_path) if peaks_path else None
        self.projection = load_projection(projection_path) if projection_path else None
        self.projection_path = projection_path
        self.roi_out = None      # where /roi suggests saving; next to the tracklets if unset
        self.next_url = None     # where /roi offers to go after saving (the app's home)
        self.tracklets_path = tracklets_path
        self.families = None     # FamilyStore, when opened on a results folder
        self.results_dir = None
        if self.image is not None:
            self.n_t, self.n_z, self.height, self.width = self.image.shape
        self._windows = {}
        self._proj_windows = {}

    def window(self, t, lo_pct, hi_pct):
        key = (t, lo_pct, hi_pct)
        if key not in self._windows:
            self._windows[key] = intensity_window(self.image[t], lo_pct, hi_pct)
        return self._windows[key]

    def mask_slice(self, t, z):
        if self.masks is None:
            return None
        volume = self.masks.get(t)
        return None if volume is None or z >= volume.shape[0] else volume[z]

    def frame(self, t, z, alpha, mode, lo_pct, hi_pct, tid=None, show_ms2=False, ms2_thr=18.0,
              show_peaks=False, peak_score=0.0, family=None):
        t = max(0, min(self.n_t - 1, t))
        z = max(0, min(self.n_z - 1, z))
        lo, hi = self.window(t, lo_pct, hi_pct)
        highlight = None
        if tid is not None and self.tracks is not None:
            label, state = self.tracks.state_at(tid, t)
            highlight = label if state == 'active' else 0   # 0 -> dim everything, pick out nothing
        group = None
        if family is not None and self.families is not None and self.tracks is not None:
            group = [label for label in (self.tracks.state_at(f, t)[0]
                                         for f in self.families.tids(family)) if label > 0]
        # get() returns None past the end of a shorter MS2 file, which reads as no signal
        ms2 = self.ms2.get(t) if (show_ms2 and self.ms2 is not None) else None
        peaks = (self.peaks.get(t, peak_score)
                 if (show_peaks and self.peaks is not None) else None)
        return composite(self.image[t, z], self.mask_slice(t, z), lo, hi, alpha, mode,
                         highlight=highlight, ms2=ms2, ms2_thr=ms2_thr, peaks=peaks,
                         group=group)

    def roi_frames(self, t_first=None, t_last=None):
        """The two frames boundaries are drawn on; the whole projection unless a window is chosen."""
        n = len(self.projection)
        t_first = 0 if t_first is None else int(t_first)
        t_last = n - 1 if t_last is None else int(t_last)
        if not 0 <= t_first <= t_last < n:
            raise ValueError(f'frames {t_first}..{t_last} are outside 0..{n - 1}, or reversed')
        return t_first, t_last

    def projection_frame(self, t, lo_pct=1.0, hi_pct=99.5):
        t = max(0, min(len(self.projection) - 1, t))
        key = (t, lo_pct, hi_pct)
        if key not in self._proj_windows:
            self._proj_windows[key] = intensity_window(self.projection[t], lo_pct, hi_pct)
        lo, hi = self._proj_windows[key]
        return composite(self.projection[t], None, lo, hi)

    def roi_select(self, roi_first, roi_last, t_first=None, t_last=None):
        """Cell counts for a boundary pair, or None before tracking has produced any cells."""
        if self.tracks is None or self.masks is None:
            return None
        t_first, t_last = self.roi_frames(t_first, t_last)
        first = select_tracklets(self.tracks, self.masks, roi_first, None, t_first, t_last)
        last = select_tracklets(self.tracks, self.masks, None, roi_last, t_first, t_last)
        total = sorted(set(first) | set(last))
        return {'n_first': len(first), 'n_last': len(last), 'n_total': len(total),
                'only_last': len(set(last) - set(first)), 'tids': total}

    def label_at(self, t, z, y, x):
        """Label under a clicked pixel, its owning tracklet, and the slice's cell count."""
        mask = self.mask_slice(t, z)
        if mask is None or not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]):
            return {'label': 0, 'tid': None, 'n_cells': 0}
        label = int(mask[y, x])
        tid = self.tracks.tracklet_of(label, t) if (self.tracks and label) else None
        return {'label': label, 'tid': tid, 'n_cells': int((np.unique(mask) > 0).sum())}

    def track_at(self, tid, t):
        """Where the tracked cell is at t, for the follow behaviour."""
        label, state = self.tracks.state_at(tid, t)
        out = {'tid': tid, 't': t, 'label': label, 'state': state,
               'z': None, 'cy': None, 'cx': None}
        if state != 'active':
            return out
        volume = self.masks.get(t) if self.masks else None
        if volume is None:
            return out
        found = locate(volume, label)
        if found is None:
            # tracklet names a label the mask does not contain -- report it rather than
            # silently centring on nothing
            out['state'] = 'missing'
            return out
        out['z'], out['cy'], out['cx'] = found
        return out


def open_results(rec_dir):
    """A viewer on one recording's results folder (see find_results), with its cell families."""
    found = find_results(rec_dir)
    missing = [k for k in ('image', 'tracklets', 'masks') if not found[k]]
    if missing:
        raise FileNotFoundError(f'no {", ".join(missing)} found for {rec_dir}')
    v = Viewer(found['image'], found['masks'], CELL_CHANNEL, found['tracklets'], found['ms2'],
               peaks_path=found['peaks'])
    if found['csv']:
        v.families = FamilyStore(found['csv'], found['roi'])
    v.results_dir = os.path.abspath(rec_dir)
    return v


class Handler(BaseHTTPRequestHandler):
    viewer = None

    def check_path(self, path):
        """Refuse a folder the page may not open; anything goes when the viewer runs alone."""

    def get_viewer(self):
        return type(self).viewer

    def set_viewer(self, v):
        """One viewer for the whole server; the app keeps one per user instead."""
        type(self).viewer = v

    def _send(self, body, content_type, cache=False, status=200):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        if cache:
            self.send_header('Cache-Control', 'max-age=3600')
        self.end_headers()
        self.wfile.write(body)

    def _page(self, name):
        with open(os.path.join(STATIC_DIR, name), 'rb') as f:
            self._send(f.read(), 'text/html; charset=utf-8')

    def do_POST(self):
        url = urlparse(self.path)
        v = self.get_viewer()
        try:
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])) or b'{}')
            if url.path == '/roi/preview':
                out = v.roi_select(body.get('roi_first'), body.get('roi_last'),
                                   body.get('t_first'), body.get('t_last'))
                if out is not None:
                    out.pop('tids')      # the page only needs the counts
                self._send(json.dumps(out).encode(), 'application/json')
            elif url.path == '/roi/save':
                self.check_path(body['path'])
                t_first, t_last = v.roi_frames(body.get('t_first'), body.get('t_last'))
                path = save_roi(body['path'], body.get('roi_first'), body.get('roi_last'),
                                t_first, t_last, v.projection_path, v.tracklets_path)
                sel = v.roi_select(body.get('roi_first'), body.get('roi_last'), t_first, t_last)
                self._send(json.dumps({'path': path,
                                       'n_total': sel['n_total'] if sel else None}).encode(),
                           'application/json')
            elif url.path == '/results/open':
                # Windows' "Copy as path" wraps a path in quotes
                path = body.get('dir', '').strip().strip('"\'').strip()
                try:
                    self.check_path(path)
                    self.set_viewer(open_results(path))
                except (OSError, ValueError, KeyError) as exc:
                    self._send(json.dumps({'error': str(exc)}).encode(), 'application/json',
                               status=400)
                    return
                self._send(json.dumps({'ok': True}).encode(), 'application/json')
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_error(500, str(exc))

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        v = self.get_viewer()
        try:
            if url.path in ('/', '/view') and v.image is None:
                self.send_response(302)
                self.send_header('Location', '/roi')
                self.end_headers()
            elif url.path in ('/', '/view'):
                self._page('index.html')
            elif url.path == '/roi':
                self._page('roi.html')
            elif url.path == '/projection':
                self._send(v.projection_frame(
                    int(q['t']), float(q.get('lo', 1.0)), float(q.get('hi', 99.5))),
                    'image/png', cache=True)
            elif url.path == '/roi/meta':
                t_first, t_last = v.roi_frames()
                self._send(json.dumps({
                    't_first': t_first, 't_last': t_last, 'n_t': len(v.projection),
                    'height': v.projection.shape[1], 'width': v.projection.shape[2],
                    'has_tracks': v.tracks is not None and v.masks is not None,
                    'has_image': v.image is not None,
                    'next_url': v.next_url,
                    'build': BUILD,
                    'default_out': v.roi_out or os.path.join(
                        os.path.dirname(v.tracklets_path or '.'), 'roi_selection.json'),
                }).encode(), 'application/json')
            elif url.path == '/meta':
                self._send(json.dumps({
                    'n_t': v.n_t, 'n_z': v.n_z,
                    'height': v.height, 'width': v.width,
                    'has_masks': v.masks is not None,
                    'has_tracklets': v.tracks is not None,
                    'has_ms2': v.ms2 is not None,
                    'ms2_max': round(v.ms2.display_max) if v.ms2 else 0,
                    'build': BUILD,
                    'has_peaks': v.peaks is not None,
                    'n_peaks': v.peaks.n_peaks if v.peaks else 0,
                    'has_roi': v.projection is not None,
                    'results_dir': v.results_dir,
                    'families': v.families.summary() if v.families else None,
                }).encode(), 'application/json')
            elif url.path == '/family':
                self._send(json.dumps(v.families.members(int(q['level']))).encode(),
                           'application/json')
            elif url.path == '/frame':
                png = v.frame(
                    t=int(q['t']), z=int(q['z']),
                    alpha=float(q.get('alpha', 0.4)),
                    mode=q.get('mode', 'outline'),
                    lo_pct=float(q.get('lo', 1.0)),
                    hi_pct=float(q.get('hi', 99.5)),
                    tid=int(q['tid']) if q.get('tid', '') != '' else None,
                    show_ms2=q.get('ms2') == '1',
                    ms2_thr=float(q.get('ms2_thr', 18.0)),
                    show_peaks=q.get('peaks') == '1',
                    peak_score=float(q.get('peak_score', 0.0)),
                    family=int(q['family']) if q.get('family', '') != '' else None,
                )
                self._send(png, 'image/png', cache=True)
            elif url.path == '/label':
                out = v.label_at(
                    int(q['t']), int(q['z']), int(float(q['y'])), int(float(q['x'])))
                self._send(json.dumps(out).encode(), 'application/json')
            elif url.path == '/tracklets':
                self._send(json.dumps(v.tracks.summary(
                    sort=q.get('sort', 'n_active'),
                    limit=int(q.get('limit', 200)),
                    ascending=q.get('dir') == 'asc')).encode(), 'application/json')
            elif url.path == '/track':
                tid = int(q['tid'])
                out = v.track_at(tid, int(q['t']))
                out['timeline'] = v.tracks.timeline(tid) if q.get('timeline') else None
                # family and trace ride along with the timeline: both are fetched once per pick
                out['activity'] = (v.families.cells.get(tid)
                                   if q.get('timeline') and v.families else None)
                self._send(json.dumps(out).encode(), 'application/json')
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_error(500, str(exc))

    def log_message(self, *args):
        pass   # one line per frame request would drown the console while scrubbing


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', help="a recording's results folder: finds the image, MS2 "
                                     'channel, masks, tracks and peaks in it and sorts the '
                                     'cells into activity families; replaces the options below')
    p.add_argument('--image', help='4D (T, Z, Y, X) tif, or a czi; leave out to only draw an ROI')
    p.add_argument('--masks-dir', help='directory of z_stack_t{N}_seg_masks.npz files')
    p.add_argument('--tracklets', help='tracklets json from create_tracklets()')
    p.add_argument('--ms2', help='MS2 channel tif, (T, Z, Y, X) or already-projected (T, Y, X)')
    p.add_argument('--projection', help='(T, Y, X) cell-channel projection (SUM_C2 tif); '
                                        'enables the /roi boundary-drawing page')
    p.add_argument('--peaks', help='peak_to_cell_matching csv; rings the detected MS2 '
                                   'emitters, which the intensity threshold cannot '
                                   'separate when a nucleus has more than one')
    p.add_argument('--channel', type=int, default=CELL_CHANNEL,
                   help='cell channel index, used only for multi-channel input')
    p.add_argument('--port', type=int, default=5000)
    args = p.parse_args()
    if not (args.results or args.image or args.projection):
        p.error('give --results, or --image, --projection or both')
    return args


def main():
    args = parse_args()
    if args.results:
        Handler.viewer = open_results(args.results)
        found = find_results(args.results)
        args.image, args.masks_dir, args.tracklets, args.ms2, args.peaks = (
            found['image'], found['masks'], found['tracklets'], found['ms2'], found['peaks'])
    else:
        Handler.viewer = Viewer(args.image, args.masks_dir, args.channel, args.tracklets,
                                args.ms2, args.projection, args.peaks)
    v = Handler.viewer
    if v.image is not None:
        print(f'image {args.image}  ->  {v.n_t} timepoints, {v.n_z} z, {v.height}x{v.width}')
    print(f'masks {args.masks_dir or "(none)"}')
    if v.tracks:
        print(f'tracks {args.tracklets}  ->  {len(v.tracks.tracks)} tracklets, '
              f'{v.tracks.n_frames} frames')
    if v.ms2:
        kind = 'z-stack, summed' if v.ms2.data.ndim == 4 else 'already projected'
        print(f'ms2   {args.ms2}  ->  {v.ms2.n_frames} frames ({kind}), '
              f'display max {v.ms2.display_max:.0f}')
    if v.peaks:
        print(f'peaks {args.peaks}  ->  {v.peaks.n_peaks} emitters over '
              f'{len(v.peaks.by_t)} timepoints')
    if v.families:
        s = v.families.summary()
        print(f'families over t={s["window"][0]}..{s["window"][1]}: ' +
              ', '.join(f'{f["name"]} {f["n"]}' for f in s['levels']))
    elif args.results:
        print('families: no gene_expression_results.csv in the results folder')
    if v.projection is not None:
        print(f'proj  {args.projection}  ->  {len(v.projection)} frames; '
              f'boundaries drawn on frames chosen in the page')
    print(f'serving on http://localhost:{args.port}')
    if v.projection is not None:
        print(f'  boundary selection at http://localhost:{args.port}/roi')
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
