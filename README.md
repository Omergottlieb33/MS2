# MS2

Infrastructure for cell and fluorescence analysis in MS2 live-imaging research.

## Overview

The repository turns a multi-channel live-imaging acquisition into per-cell transcriptional
activity traces, in three stages:

1. **Segmentation** — Cellpose segments every cell in each 3D z-stack, one timepoint at a time.
2. **Tracking** — segmented cells are linked across time into tracklets, so a cell keeps one
   identity through the movie.
3. **Gene expression** — MS2 transcription-site spots are detected, matched to their parent cell,
   and quantified per cell over time.

A browser-based **viewer** is included for inspecting segmentation and tracking results visually.

### Data

Acquisitions are Zeiss `.czi` time series with shape `(T, C, Z, Y, X)`. Channel **`C=1` is the cell
channel** used for segmentation and tracking; the MS2 channel is supplied separately to the
expression stage as a background-subtracted TIF.

A representative dataset is 109 timepoints × 2 channels × 11 z-slices × 1024 × 1024.

Segmentation writes one compressed archive per timepoint:

```
<output_dir>/<acquisition>/masks/z_stack_t{N}_seg_masks.npz     # keys: masks, flows_xyz_coord, flows_circular_coord
```

Tracking writes a single JSON mapping each tracklet to its label at every frame, using two
sentinels — `-1` for a frame where the cell was not detected, `-2` for a cell that left the field
of view:

```json
{ "0": [1, 1, 2, 3, -1, 4, -2, -2], "1": [...] }
```

## Environment

Conda, Python 3.12, CUDA 12.4 build of PyTorch. Segmentation requires a GPU.

```bash
conda env create -f environment.yml
conda activate ms2
```

Key dependencies: `cellpose>=4.0`, `torch`, `czifile`, `aicspylibczi`, `tifffile`, `scikit-image`,
`networkx`, `scipy`, `findmaxima2d`.

Run everything from the repository root — the modules import as `src.*`, so the root must be on
`PYTHONPATH` (which it is by default when invoking from there).

## Running the pipeline

> **The end-to-end pipeline is still in development.** The three stages currently run as separate
> scripts, and the output of each must be passed by hand to the next. Two of the three take their
> parameters from hardcoded values in their `__main__` block rather than from the command line —
> edit those before running. Unified orchestration is planned.

### 1. Segmentation — `cell_3d_segmentation.py`

Edit the values at the bottom of the file, then run it:

```python
if __name__ == "__main__":
    czi_file_path = "/path/to/acquisition.czi"
    output_dir    = "/path/to/outputs/acquisition"
    device        = "cuda:3"
    segment_3d_cells(czi_file_path, output_dir, device)
```

```bash
python cell_3d_segmentation.py
```

Already-segmented timepoints are skipped, so an interrupted run can simply be restarted.

### 2. Tracking — `cell_tracking.py`

Cells are matched between consecutive frames by IoU with Hungarian assignment, with a second-chance
pass across a one-frame gap so a single missed detection does not end a track.

Edit the bottom of the file, then run:

```python
if __name__ == "__main__":
    masks_dir = "/path/to/outputs/acquisition/masks/"
    t = 109                      # number of timepoints; <=0 uses all available
    tracklets = cell_tracking(masks_dir, t=t)
```

```bash
python cell_tracking.py
```

Writes a tracklets JSON next to the masks directory. `evaluate_tracklets()` in the same module
prints track counts, gap totals and a fragmentation index for a result.

### 3. Gene expression — `ms2_gene_expression.py`

The only stage with a full command-line interface. The MS2 channel must already be
background-subtracted (rolling-ball or equivalent) and saved as a TIF.

```bash
python ms2_gene_expression.py \
  --czi_file_path /path/to/acquisition.czi \
  --seg_maps_dir /path/to/outputs/acquisition/masks \
  --tracklets_path /path/to/outputs/acquisition/tracklets.json \
  --ms2_background_removed /path/to/acquisition_ms2_bg_removed.tif \
  --output_dir /path/to/outputs/acquisition/gene_expression \
  --prominence 18.0
```

`--prominence` is the maxima-finder threshold for calling MS2 spots; it is the main knob to tune
against your signal-to-noise.

### 4. Activity analysis — `src/cell_activity*.py`

Classifies every tracked cell of one developmental-stage window into an activity ladder, and
compares those windows across embryos. All four modules take the same config: a JSON of
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
back to the constants at the top of `src/cell_activity.py`.

Point `csv` at `gene_expression_results_fixed.csv`, not `gene_expression_results.csv` — the
original matrix holds most traces at the wrong timepoints, which a stage window then cuts wrongly.
`src/gene_expression/expression_matrix.py` rebuilds a corrected matrix from an existing run.

**Check the config first.** Every check runs on every recording, so one pass lists everything
wrong rather than the first thing wrong:

```bash
python -m src.cell_activity_validate --config recordings.json
```

Errors (a missing mask timepoint, a cell with no tracklet, a window outside the matrix) stop the
run. Warnings are comparability traps — the unfixed matrix, or a threshold that differs between
recordings — where each recording is fine on its own and the comparison between them is not.

**Check the photometry next.** `ellipse_sum` is a raw, uncalibrated AU sum, so an absolute
threshold only means the same thing in two recordings if the two were imaged under matched
conditions. This measures each recording's background and reports where its configured floor
falls in it:

```bash
python -m src.cell_activity_calibration --config recordings.json --out-dir .../calibration
```

Read `floor_percentile` in `calibration.csv`. A floor at the 8th percentile in one recording and
the 40th in another is one constant meaning two different things, and every number downstream —
level composition, rate, duty, onset — inherits the difference. This diagnoses the problem; it
does not fix it, which needs a reference standard per acquisition.

**Then run the analysis.** Per recording — heatmaps, the activity ladder, PCA of temporal shape
and a cluster-overlay GIF:

```bash
python -m src.cell_activity --config recordings.json --out-dir .../stage_activity
```

Across recordings, on one shared ladder fitted over all their pooled cells:

```bash
python -m src.cell_activity_compare --config recordings.json --out-dir .../compare
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
python -m src.viewer.app \
  --image /path/to/acquisition.czi \        # or a 4D (T, Z, Y, X) tif
  --masks-dir /path/to/outputs/acquisition/masks \
  --tracklets /path/to/outputs/acquisition/tracklets.json \
  --port 5057
```

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

- **Per-cell activity** — the MS2 expression trace for the selected cell, plotted alongside the
  timeline strip so transcriptional bursts can be read against the cell's own tracking history.
- **MS2 channel display** — overlaying the MS2 channel and its detected spots on the cell channel,
  so a called spot can be checked against the cell it was assigned to.

## Citation

If you use this software, please cite the repository:

```bibtex
@software{gottlieb_ms2,
  author  = {Gottlieb, Omer},
  title   = {MS2: infrastructure for cell and fluorescence analysis},
  year    = {2025},
  url     = {https://github.com/Omergottlieb33/MS2},
  license = {MIT}
}
```

<!-- TODO: add the accompanying publication reference once available. -->

## License

MIT — see [LICENSE](LICENSE).
