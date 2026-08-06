"""Browser-based viewer for 3D segmentation masks over the cell channel.

Renders one composited PNG per (t, z) server-side; the browser handles zoom and pan.
Run on the machine holding the data and reach it over an ssh port-forward:

    ssh -L 5000:localhost:5000 <server>
    python -m src.viewer.app --image cells.tif --masks-dir .../masks
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from src.viewer.loaders import CELL_CHANNEL, MaskStore, TrackletStore, load_image, locate
from src.viewer.render import composite, intensity_window

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')


class Viewer:
    def __init__(self, image_path, masks_dir, channel, tracklets_path=None):
        self.image = load_image(image_path, channel)
        self.masks = MaskStore(masks_dir) if masks_dir else None
        self.tracks = TrackletStore(tracklets_path) if tracklets_path else None
        self.n_t, self.n_z, self.height, self.width = self.image.shape
        self._windows = {}

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

    def frame(self, t, z, alpha, mode, lo_pct, hi_pct, tid=None):
        t = max(0, min(self.n_t - 1, t))
        z = max(0, min(self.n_z - 1, z))
        lo, hi = self.window(t, lo_pct, hi_pct)
        highlight = None
        if tid is not None and self.tracks is not None:
            label, state = self.tracks.state_at(tid, t)
            highlight = label if state == 'active' else 0   # 0 -> dim everything, pick out nothing
        return composite(self.image[t, z], self.mask_slice(t, z), lo, hi, alpha, mode,
                         highlight=highlight)

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


class Handler(BaseHTTPRequestHandler):
    viewer = None

    def _send(self, body, content_type, cache=False):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        if cache:
            self.send_header('Cache-Control', 'max-age=3600')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        v = self.viewer
        try:
            if url.path == '/':
                with open(os.path.join(STATIC_DIR, 'index.html'), 'rb') as f:
                    self._send(f.read(), 'text/html; charset=utf-8')
            elif url.path == '/meta':
                self._send(json.dumps({
                    'n_t': v.n_t, 'n_z': v.n_z,
                    'height': v.height, 'width': v.width,
                    'has_masks': v.masks is not None,
                    'has_tracklets': v.tracks is not None,
                }).encode(), 'application/json')
            elif url.path == '/frame':
                png = v.frame(
                    t=int(q['t']), z=int(q['z']),
                    alpha=float(q.get('alpha', 0.4)),
                    mode=q.get('mode', 'outline'),
                    lo_pct=float(q.get('lo', 1.0)),
                    hi_pct=float(q.get('hi', 99.5)),
                    tid=int(q['tid']) if q.get('tid', '') != '' else None,
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
    p.add_argument('--image', required=True, help='4D (T, Z, Y, X) tif, or a czi')
    p.add_argument('--masks-dir', help='directory of z_stack_t{N}_seg_masks.npz files')
    p.add_argument('--tracklets', help='tracklets json from create_tracklets()')
    p.add_argument('--channel', type=int, default=CELL_CHANNEL,
                   help='cell channel index, used only for multi-channel input')
    p.add_argument('--port', type=int, default=5000)
    return p.parse_args()


def main():
    args = parse_args()
    Handler.viewer = Viewer(args.image, args.masks_dir, args.channel, args.tracklets)
    v = Handler.viewer
    print(f'image {args.image}  ->  {v.n_t} timepoints, {v.n_z} z, {v.height}x{v.width}')
    print(f'masks {args.masks_dir or "(none)"}')
    if v.tracks:
        print(f'tracks {args.tracklets}  ->  {len(v.tracks.tracks)} tracklets, '
              f'{v.tracks.n_frames} frames')
    print(f'serving on http://localhost:{args.port}')
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
