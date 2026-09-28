"""Open a single-page dashboard for one tracklet JSON and its masks.

python -m src.tracking_dashboard --tracklets TRACKLETS.json --masks-dir MASKS
python -m src.tracking_dashboard --report outputs/tracklet_dashboard/evaluation.json
"""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.tracklet_quality import evaluate, fingerprint, infer_masks, settings_checked

ASSETS = Path(__file__).with_name('tracking_dashboard_static')


def standalone(report):
    template = (ASSETS/'index.html').read_text()
    data = json.dumps(report, allow_nan=False).replace('<', '\\u003c')
    return (template.replace('<link rel="stylesheet" href="/style.css">', '<style>'+(ASSETS/'style.css').read_text()+'</style>')
            .replace('<script src="/app.js"></script>',
                     '<script type="application/json" id="report-data">'+data+'</script><script>'+
                     (ASSETS/'app.js').read_text()+'</script>'))


def handler_for(report):
    page = standalone(report).encode()
    payload = json.dumps(report, allow_nan=False).encode()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/':
                body, mime = page, 'text/html; charset=utf-8'
            elif path == '/api/report':
                body, mime = payload, 'application/json'
            elif path == '/health':
                body, mime = b'{"status":"ok"}', 'application/json'
            else:
                self.send_error(404); return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
    return Handler


def validate_roi(data, report):
    """Reject malformed or mismatched ROI inputs before exporting a report."""
    if not isinstance(data, dict):
        raise ValueError('Expected an ROI JSON object')
    nonempty = False
    for key, frame in [('roi_first', data.get('t_first', 0)),
                       ('roi_last', data.get('t_last'))]:
        if key == 'roi_last' and (frame is None or isinstance(frame, (int, float)) and frame < 0):
            frame = report['frames'] - 1
        polygon = data.get(key) or []
        if not isinstance(polygon, list):
            raise ValueError('ROI polygons must be arrays')
        if not polygon:
            continue
        nonempty = True
        import math
        if len(polygon) < 3 or any(not isinstance(p, list) or len(p) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in p)
                for p in polygon):
            raise ValueError('Each ROI polygon needs at least three finite [x, y] vertices')
        if type(frame) is not int or not 0 <= frame < report['frames']:
            raise ValueError('ROI frame is outside this recording')
    if not nonempty:
        raise ValueError('The ROI contains no polygons')
    source = data.get('tracklets')
    if source and Path(source).is_absolute() and Path(source).parent != Path(report['paths']['tracklets']).parent:
        raise ValueError('ROI belongs to another tracklet folder; open the corresponding recording')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--config', type=Path)
    source.add_argument('--report', type=Path)
    source.add_argument('--tracklets', type=Path, help='One tracklet JSON to inspect')
    p.add_argument('--roi', type=Path, help='Saved roi_selection.json; selects tracks inside either endpoint polygon')
    p.add_argument('--masks-dir', type=Path, help='Matching segmentation masks; inferred when possible')
    p.add_argument('--voxel-size-um', type=float, nargs=3, metavar=('Z','Y','X'))
    p.add_argument('--max-step-um', type=float, help='Expected maximum consecutive-frame displacement')
    p.add_argument('--neighbor-radius-um', type=float)
    p.add_argument('--neighbors', type=int, help='Maximum number of nearby cells, default 5')
    p.add_argument('--output-dir', type=Path, default=Path('outputs/tracklet_dashboard'))
    p.add_argument('--recompute', action='store_true')
    p.add_argument('--evaluate-only', action='store_true')
    p.add_argument('--no-browser', action='store_true')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5011)
    args = p.parse_args()
    roi_path = args.roi
    if roi_path and not (args.tracklets or args.config or args.report):
        roi_data = json.loads(roi_path.read_text())
        if roi_data.get('tracklets'):
            args.tracklets = (roi_path.resolve().parent / roi_data['tracklets']).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir/'evaluation.json'
    if args.report:
        report = json.loads(args.report.read_text())
        if report.get('schema') != 'single-tracklets-v1':
            p.error('This is an older multi-result report. Use --tracklets FILE --masks-dir DIR to generate the single-page dashboard.')
    else:
        title = None
        settings = {}
        if args.tracklets:
            tracklets = args.tracklets.resolve()
            masks = args.masks_dir or infer_masks(tracklets)
        else:
            config_path = (args.config or Path(__file__).resolve().parents[1]/'configs/tracking_evaluation_new03.json').resolve()
            config = json.loads(config_path.read_text())
            if 'tracklets' not in config:
                p.error('Configuration must specify one tracklets path and its masks path; multiple datasets are no longer displayed.')
            tracklets = (config_path.parent/config['tracklets']).resolve()
            masks = args.masks_dir or ((config_path.parent/config['masks']).resolve() if config.get('masks') else infer_masks(tracklets))
            settings = config.get('settings', {})
            title = config.get('title')
            if roi_path is None and config.get('roi'):
                roi_path = (config_path.parent/config['roi']).resolve()
        for key, value in [('voxel_size_um', args.voxel_size_um), ('max_step_um', args.max_step_um),
                           ('neighbor_radius_um', args.neighbor_radius_um), ('neighbor_k', args.neighbors)]:
            if value is not None:
                settings[key] = value
        settings = settings_checked(settings)
        report = None
        if out.exists() and not args.recompute:
            cached = json.loads(out.read_text())
            if cached.get('schema') == 'single-tracklets-v1' and cached.get('fingerprint') == fingerprint(tracklets, masks, settings):
                report = cached
                if title:
                    report['title'] = title
                print('Using cached tracklet measurements.', flush=True)
        if report is None:
            report = evaluate(tracklets, masks, settings, title, progress=lambda msg: print(msg, flush=True))
    if roi_path:
        roi_data = json.loads(roi_path.read_text())
        try:
            validate_roi(roi_data, report)
        except ValueError as exc:
            p.error(str(exc))
        report['roi'] = dict(name=roi_path.name, data=roi_data)
        if any('centers_xy' not in t for t in report['tracks']):
            p.error('ROI filtering requires a regenerated report with centroid coordinates. Use --tracklets and --masks-dir.')
    elif not args.report:
        report.pop('roi', None)
    out.write_text(json.dumps(report, indent=2, allow_nan=False))
    (args.output_dir/'dashboard.html').write_text(standalone(report))
    print(f'Report: {out.resolve()}\nStandalone dashboard: {(args.output_dir/"dashboard.html").resolve()}', flush=True)
    if args.evaluate_only:
        return
    server = ThreadingHTTPServer((args.host, args.port), handler_for(report))
    host = '127.0.0.1' if args.host == '0.0.0.0' else args.host
    url = f'http://{host}:{server.server_port}'
    print(f'Dashboard: {url}\nRemote access: ssh -L {server.server_port}:localhost:{server.server_port} <server>', flush=True)
    if not args.no_browser:
        threading.Timer(.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
