"""Background jobs the app starts: a pipeline run, a comparison, a dashboard export.

A job is a detached process group, so it outlives the browser, the app and the ssh session
that started it.  Its folder holds job.json (pid, command, start time), its log, and, once
the command returns, exit_code -- which is how a finished job is told from a killed one
after the app that started it is gone.  The pid is matched with the kernel's start time
for it, so a pid reused by some later process is not mistaken for the job.
"""
import datetime
import json
import os
import signal
import subprocess
import sys
import threading

from ms2.pipeline import STEPS, Recording

TAG = 'ms2-job'    # argv[0] of the wrapping shell, to recognise it in ps


def start(cmd, job_dir, log_name):
    os.makedirs(job_dir, exist_ok=True)
    exit_file = os.path.join(job_dir, 'exit_code')
    if os.path.exists(exit_file):
        os.remove(exit_file)
    log_path = os.path.join(job_dir, log_name)
    with open(log_path, 'a') as log:
        log.write(f'\n=== {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {" ".join(cmd)}\n')
        log.flush()
        proc = subprocess.Popen(['/bin/sh', '-c', '"$@"; echo $? > "$MS2_EXIT_FILE"', TAG, *cmd],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True,
                                env={**os.environ, 'MS2_EXIT_FILE': exit_file,
                                     'PYTHONUNBUFFERED': '1'})
    with open(os.path.join(job_dir, 'job.json'), 'w') as f:
        json.dump({'pid': proc.pid, 'proc_start': proc_start(proc.pid), 'cmd': cmd,
                   'log': log_name, 'started': datetime.datetime.now().isoformat()}, f, indent=2)
    # Reap it when it ends, or it lingers as a zombie of the app and still looks alive.
    threading.Thread(target=proc.wait, daemon=True).start()
    return proc.pid


def read_job(job_dir):
    try:
        with open(os.path.join(job_dir, 'job.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _stat(pid):
    """(state, start time) of a process from /proc, or None once it is gone."""
    try:
        with open(f'/proc/{pid}/stat') as f:
            fields = f.read().rsplit(')', 1)[1].split()   # the name in (...) may hold spaces
    except OSError:
        return None
    return fields[0], fields[19]


def proc_start(pid):
    stat = _stat(pid)
    return stat and stat[1]


def alive(job):
    stat = _stat(job['pid'])
    return stat is not None and stat[0] != 'Z' and stat[1] == job.get('proc_start')


def state(job_dir):
    """'none', 'running', 'done', 'failed' (non-zero exit) or 'stopped' (killed)."""
    job = read_job(job_dir)
    if job is None:
        return 'none'
    try:
        with open(os.path.join(job_dir, 'exit_code')) as f:
            return 'done' if f.read().strip() == '0' else 'failed'
    except OSError:
        return 'running' if alive(job) else 'stopped'


def stop(job_dir):
    job = read_job(job_dir)
    if job and alive(job):
        os.killpg(job['pid'], signal.SIGTERM)


def log_tail(job_dir, n=40):
    """Last lines of the job log.  Progress bars rewrite a line with \\r; keep the last state."""
    job = read_job(job_dir)
    if job is None:
        return ''
    try:
        with open(os.path.join(job_dir, job['log']), 'rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 200_000))
            text = f.read().decode(errors='replace')
    except OSError:
        return ''
    lines = [line.rsplit('\r', 1)[-1] for line in text.split('\n')]
    return '\n'.join(lines[-n:])


def pipeline_cmd(czi, out, device, roi=None, prominence=None, bg_radius=None, plots=False):
    cmd = [sys.executable, '-m', 'ms2.pipeline', czi, '--out', out, '--device', device]
    if roi:
        cmd += ['--roi', roi]
    if prominence is not None:
        cmd += ['--prominence', str(prominence)]
    if bg_radius is not None:
        cmd += ['--bg-radius', str(bg_radius)]
    if plots:
        cmd.append('--plots')
    return cmd


def recording(rec_dir):
    """The pipeline layout of an existing recording folder (named after its CZI)."""
    return Recording(rec_dir + '.czi', os.path.dirname(os.path.abspath(rec_dir)))


def redo_after_roi_change(rec):
    """An ROI saved after gene expression ran selects different cells: run those steps again."""
    first = rec.marker('gene_expression')
    if os.path.exists(first) and os.path.getmtime(rec.roi) > os.path.getmtime(first):
        for step in ('gene_expression', 'activity'):
            if os.path.exists(rec.marker(step)):
                os.remove(rec.marker(step))


def redo_after_orientation_change(rec):
    """A new orientation turns the activity figures: draw them again on the next run."""
    if os.path.exists(rec.marker('activity')):
        os.remove(rec.marker('activity'))


def recording_status(rec_dir):
    """What the Runs page shows for one recording folder."""
    rec = recording(rec_dir)
    done = [s for s in STEPS if os.path.exists(rec.marker(s))]
    job = read_job(rec_dir) or {}
    roi = None
    if os.path.exists(rec.roi):
        with open(rec.roi) as f:
            r = json.load(f)
        roi = {'t_first': r.get('t_first'), 't_last': r.get('t_last')}
    return {'name': rec.name, 'dir': rec_dir, 'state': state(rec_dir), 'steps_done': done,
            'started': job.get('started'), 'czi': czi_of(rec_dir, job), 'roi': roi}


def czi_of(rec_dir, job=None):
    """The CZI a recording came from: its job command, else its run manifest."""
    cmd = (job or {}).get('cmd') or []
    if 'ms2.pipeline' in cmd:
        return cmd[cmd.index('ms2.pipeline') + 1]
    try:
        with open(os.path.join(rec_dir, 'run_manifest.json')) as f:
            return json.load(f).get('czi')
    except (OSError, ValueError):
        return None


def find_recordings(root, depth=3):
    """Recording folders under root: a pipeline marker or a job in them, or the projection
    the app makes for drawing an ROI before the run."""
    found = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath[len(root):].count(os.sep) >= depth:
            dirnames[:] = []
        names = set(filenames)
        name = os.path.basename(dirpath)
        if 'job.json' in names and name != '_comparison' or any(n.startswith('.done_') for n in names) \
                or os.path.exists(os.path.join(dirpath, 'preprocess', f'SUM_C2-{name}.tif')):
            found.append(dirpath)
            dirnames[:] = []          # a recording's own sub-folders are not recordings
        dirnames[:] = [d for d in dirnames if not d.startswith('.')]
    return sorted(found)
