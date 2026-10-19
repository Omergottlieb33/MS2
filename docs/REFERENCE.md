# MS2: reference

Details behind the [README](../README.md): the command line, what each pipeline step writes, the
activity analysis, and the viewer. Lab members using the app only need the
[user guide](USER_GUIDE.md).

## Command line

**Lab members:** see the [user guide](USER_GUIDE.md). Double-click a launcher and the app
opens in your browser. From there you pick a CZI, draw the ROI, start the run and open the
results, with no terminal needed.

**On the command line**, everything is one command, `ms2`:

```bash
ms2 app                                  # the browser app (what the launchers start)
ms2 run <rec>.czi --out <root> [--roi roi.json] [--device cuda:0] [--stop-after STEP]
ms2 compare experiment.json              # several recordings, then their comparison
ms2 view --image ... --masks-dir ...     # segmentation / tracking viewer
ms2 dashboard --tracklets ... --masks-dir ...   # tracking-quality dashboard
```

`ms2 <command> -h` lists each command's options.

### The lab page: one app for everyone

`ms2 app --workspace` runs a single app that the whole lab opens at `http://golgi:<port>`,
with no repository, launcher or SSH key on their side. People log in on the page with their name
and a shared lab password, and **Log out** in the header ends the session. They upload a CZI from their own computer, draw the
ROI, start the run, and download a zip of the logs, ROI and result tables (no masks or
TIFFs). Each name gets `<workspace>/<name>/uploads/` and `<workspace>/<name>/runs/`, and
can't write anything outside that folder. Every job runs under the account that started
the app.

**Open a file on the server** (Runs page) opens a pasted server path in a new tab. It works
for an HTML page with its images, a picture or a PDF. In lab mode the path may be in the
user's own folder, or under a `--data-root` given at startup, which is read-only. Without
`--data-root`, users can only open files in their own folder.

```bash
echo 'the-lab-password' > ~/.ms2_password && chmod 600 ~/.ms2_password
ms2 app --host 0.0.0.0 --port 8765 \
    --workspace /zjbd/zd1/shechtmanlab/omer/ms2_web --password-file ~/.ms2_password \
    --data-root /zjbd/zd1/shechtmanlab
```

The password travels as plain http, so this is for the lab network and VPN only. A login
lasts 30 days; changing the password file (and restarting) logs everyone out.

To keep it running, use a `systemd --user` service. It restarts after a crash or a reboot,
and needs `loginctl enable-linger <you>` once, from an admin. Save the following as
`~/.config/systemd/user/ms2-app.service`:

```ini
[Unit]
Description=MS2 lab page

[Service]
ExecStart=/zjbd/zd1/shechtmanlab/conda_envs/ms2/bin/ms2 app --host 0.0.0.0 --port 8765 --workspace /zjbd/zd1/shechtmanlab/omer/ms2_web --password-file %h/.ms2_password --data-root /zjbd/zd1/shechtmanlab
Restart=on-failure
KillMode=process

[Install]
WantedBy=default.target
```

Then run `systemctl --user enable --now ms2-app`. `KillMode=process` stops only the app on a
restart: runs are detached and keep going either way. Without linger, start the same command
in a `tmux` session; you have to start it again after a reboot.

The header shows the commit the app was started from; restart the app after a `git pull`.
The About tab's clip is `ms2/app/static/explainer.webp`. It is rendered from one finished
recording by `python -m ms2.app.make_explainer <rec dir>`, so rerun that if the pipeline's
look changes.

Not handled: there is no GPU queue, so two runs on one GPU can run out of memory (the GPU
list shows free memory). Uploads and runs are never deleted automatically.

### What a run does

`ms2 run` (`ms2/pipeline.py`) takes one recording through six steps, each into
`<root>/<rec>/`. Each step marks itself done, so a rerun resumes where the last one stopped:

| Step | Writes | |
|---|---|---|
| preprocess | `preprocess/C1-<rec>.tif`, `C2-<rec>.tif`, `C1-<rec>_bg_removed.tif`, `SUM_C2-<rec>.tif` | channel split; ImageJ's Subtract Background and Sum Slices, reproduced pixel for pixel |
| segment | `masks/z_stack_t{N}_seg_masks.npz` | Cellpose 3D, GPU |
| track | `tracklets.json` | IoU + Hungarian matching, with a second chance across a one-frame gap |
| unify | `unified/masks_split/`, `unified/tracklets_unified.json` | splits fused labels, retires fragments |
| gene_expression | `gene_expression/gene_expression_results.csv` | MS2 spots matched to cells and quantified per frame |
| activity | `activity/clusters.csv` and figures | the activity ladder (below) |

**ROI.** Boundaries are drawn on `SUM_C2-<rec>.tif` at two frames, the start and end of the
stage. Use `ms2 app`, or `ms2 view --projection <SUM_C2 tif>`. This can happen before
anything else has run, because the app builds the projection straight from the CZI. Pass the
saved `roi_selection.json` as `--roi`. Gene expression then keeps the tracklets inside either
boundary, and activity is measured between the two frames. Without `--roi`, gene expression
keeps cells with fewer than 20 missing frames, and activity covers the whole recording.

### Data

Acquisitions are Zeiss `.czi` time series with shape `(T, C, Z, Y, X)`. Channel `C=0` is MS2
and **`C=1` is the cell channel** used for segmentation and tracking. A representative dataset
is 109 timepoints × 2 channels × 11 z-slices × 1024 × 1024.

Tracklets are a JSON mapping each tracklet to its label at every frame, using two sentinels:
`-1` for a frame where the cell was not detected, and `-2` for a cell that left the field of
view.

```json
{ "0": [1, 1, 2, 3, -1, 4, -2, -2], "1": [...] }
```

## Activity analysis in detail: `ms2/cell_activity*.py`

`ms2 run` and `ms2 compare` run this for you. The modules below are for working on one stage
window by hand. They classify every tracked cell of the window into an activity ladder, and
compare those windows across embryos. All four modules take the same config: a JSON of
`{name: recording dict}`, one entry per recording.

```json
{
  "New-02-ST11-12": {
    "csv":        "/path/to/gene_expression/gene_expression_results_fixed.csv",
    "tracklets":  "/path/to/outputs/acquisition/masks/tracklets.json",
    "masks_dir":  "/path/to/outputs/acquisition/masks",
    "t_start":    0,
    "t_end":      60,
    "noise_floor":  20,
    "min_presence": 0.5
  }
}
```

`t_start`/`t_end` are 0-based and both included. `noise_floor` (a frame at or below it is
background), `noise_peak` (cells whose peak never clears it are 1–2 frame blips) and
`min_presence` (the share of the window a cell must be tracked in to count) are optional and fall
back to the constants at the top of `ms2/cell_activity.py`.

For gene-expression runs made before 2026-09-23, point `csv` at
`gene_expression_results_fixed.csv`. Those matrices hold most traces at the wrong timepoints,
which a stage window then cuts wrongly. `python -m ms2.gene_expression.expression_matrix`
rebuilds a corrected matrix from such a run. Runs since then write the corrected matrix
directly.

**Check the config first.** Every check runs on every recording, so one pass lists everything
wrong rather than the first thing wrong:

```bash
python -m ms2.cell_activity_validate --config recordings.json
```

Errors (a missing mask timepoint, a cell with no tracklet, a window outside the matrix) stop the
run. Warnings are comparability traps — the unfixed matrix, or a threshold that differs between
recordings — where each recording is fine on its own and the comparison between them is not.

**Check the photometry next.** `ellipse_sum` is a raw, uncalibrated AU sum, so an absolute
threshold only means the same thing in two recordings if the two were imaged under matched
conditions. This measures each recording's background and reports where its configured floor
falls in it:

```bash
python -m ms2.cell_activity_calibration --config recordings.json --out-dir .../calibration
```

Read `floor_percentile` in `calibration.csv`. A floor at the 8th percentile in one recording and
the 40th in another is one constant meaning two different things, and every number downstream —
level composition, rate, duty, onset — inherits the difference. This diagnoses the problem; it
does not fix it, which needs a reference standard per acquisition.

**Then run the analysis.** Per recording — heatmaps, the activity ladder, PCA of temporal shape
and a cluster-overlay GIF:

```bash
python -m ms2.cell_activity --config recordings.json --out-dir .../stage_activity
```

Across recordings, on one shared ladder fitted over all their pooled cells:

```bash
python -m ms2.cell_activity_compare --config recordings.json --out-dir .../compare
```

Both validate the config first; `--skip-validate` skips that for a rerun of a known-good one, and
`cell_activity_compare` also writes the calibration diagnostic into `<out-dir>/calibration`.

The comparison writes `cells.csv`, `metrics.csv`, `timecourse.csv` and `stats.csv` plus figures.
**Read `stats_readme.txt` before quoting a p-value**: the sampling unit is the cell, not the
embryo, so a small p means "these two recordings differ", not "these two stages differ". Testing a
stage or a condition needs several embryos per group, tested at the embryo level.

## Viewer

A browser-based viewer for inspecting segmentation and tracking. It renders each `(t, z)` slice
server-side and sends a PNG, so it works comfortably over SSH against data that stays on the
acquisition machine.

```bash
ms2 view \
  --image /path/to/acquisition.czi \        # or a 4D (T, Z, Y, X) tif
  --masks-dir /path/to/outputs/acquisition/masks \
  --tracklets /path/to/outputs/acquisition/tracklets.json \
  --port 5057
```

The app opens this viewer for a finished run from its **Runs & results** page.

Reach it through a tunnel:

```bash
ssh -L 5057:localhost:5057 <server>       # then open http://localhost:5057
```

`--masks-dir` and `--tracklets` are both optional: without them you get a plain image browser, and
without `--tracklets` you get segmentation overlays with no track selection.

### Using it

| | |
|---|---|
| **Time / Z** | `←` `→` step time, `↑` `↓` step z, or use the sliders |
| **Zoom / pan** | mouse wheel zooms at the cursor, drag to pan, `r` resets the view |
| **Overlay** | `space` toggles masks; the mode button switches outline ↔ filled |
| **Contrast** | percentile sliders, computed per timepoint so brightness does not flicker across z |
| **Select a track** | click any cell, or pick from the track list |
| **Clear** | `Esc` |

Each cell keeps the same colour across the whole movie, so a label that changes colour between
frames means the segmentation changed, not the cell.

Selecting a track dims every other cell to a faint outline and picks the tracked one out in red.
The viewer then follows that cell as you scrub time — jumping to the z-slice it sits on and keeping
it centred. Follow only re-centres when you move to a new timepoint, so you stay free to scroll z
and pan to look around the cell; the status line tells you which slice the cell is on if you have
scrolled away from it.

Frames where a track is lost are shown honestly rather than guessed at: the status line reads
`GAP — not detected` or `left field of view`, and no cell is highlighted. The timeline strip under
the track list shows the whole life of the selected track at a glance — green for detected frames,
red for gaps, grey after it leaves the field of view — and clicking it jumps to that timepoint.

The track list can be sorted by length, gap count, resurrections, span, or id in either direction.
Sorting by gap count is the fastest way to find where tracking is failing.

### Planned

- **Per-cell activity**: the MS2 expression trace for the selected cell, plotted alongside the
  timeline strip so transcriptional bursts can be read against the cell's own tracking history.

## Repository layout

```
ms2/            the package: pipeline steps, app/, viewer/, gene_expression/, cell_activity*
launchers/      double-click launchers for the app (Windows, Linux) and their settings
docs/           the user guide
configs/        example experiment and activity configs (paths on the lab server)
tests/          python tests/test_<name>.py, one script per module
research/       experiments, notebooks and notes; not maintained, not imported by ms2/
```

Tests are plain scripts: `for f in tests/test_*.py; do python $f; done`. The preprocessing
test also compares against ImageJ output when the reference data is mounted. Set
`MS2_TEST_DATA` to its folder.
