# MS2 app: user guide

The MS2 app turns a live-imaging recording (a Zeiss `.czi` file) into a table of how strongly
each cell transcribes over time. Everything runs on the lab's GPU server (`golgi`). Your
computer only shows the app in a web browser.

**The quick way, if the lab page is running:** open `http://golgi:8765` (ask for the address
and the lab password). Log in with your name and the lab password (**Log out** is at the top). Then,
under **CZI file**, pick the CZI from your computer and carry on from
[Start](#2-start). When the run has finished, analyse it on the
[Analysis page](#analysing-the-results), and **Download results** on the Runs page gives you a zip of the logs and result
tables to save wherever you like. To view a page, picture or PDF that is on the server, paste
its full path into **Open a file on the server** on the Runs page. You need nothing else on your computer. The rest of this
section, and the setup below, are for running your own copy of the app.

To run your own copy you need: a user account on golgi, and the `launchers` folder from the
MS2 repository.

---

## One-time setup

### 1. Let your computer log in to golgi without a password

The launcher connects with an SSH key. You create it once.

**Windows:** open **PowerShell** (Start menu → type `powershell`) and run these two commands.
Replace `your-username` with your golgi username.

```powershell
ssh-keygen -t ed25519
```

Press **Enter** at every question.

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh your-username@golgi "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

This asks for your golgi password one last time.

**Linux:** in a terminal:

```bash
ssh-keygen -t ed25519            # press Enter at every question
ssh-copy-id your-username@golgi
```

To check it worked, run `ssh your-username@golgi`. It should log you in without asking for a
password. Type `exit` to leave.

### 2. Fill in your settings

Open `launchers/ms2_settings.txt` in a text editor (Notepad is fine) and change two lines:

- `SSH_USER=` your golgi username
- `PORT=` a number between 8700 and 8799 that nobody else in the lab uses. Your own
  number keeps your app separate from everyone else's.

Leave the other lines as they are, unless the person who installed MS2 tells you otherwise.

---

## Starting the app

- **Windows:** double-click **`MS2.bat`**.
- **Linux:** run `launchers/ms2.sh`.

A black window opens and connects. After a few seconds your browser opens the app.
**Keep the black window open while you work.** Closing it stops the app.

Closing the app does **not** stop analyses that are already running. You can close
everything, go home, and check the results tomorrow.

If the browser doesn't open by itself, open `http://localhost:8765` (use your own port
number).

---

## Analysing a recording

### 1. Choose the recording

On **New analysis**, click **Browse…** and go to your `.czi` file. Then check the **Results
folder**. Each recording gets its own sub-folder there, named after the file.

### 2. Start

- **Processor:** pick a GPU with the most free memory. The CPU option works but takes very long.
- **Advanced settings:** leave them alone unless you know why you are changing them.
  *MS2 spot prominence* is how bright a spot must be to count as transcription.

Click **Start analysis**. The app switches to **Runs & results**. The run measures every cell
tracked for 10 frames or more; you choose the stage, the region and the cells afterwards, on the
Analysis page (see *Analysing the results*).

---

## Following a run

**Runs & results** lists every recording in the results folder. The bar shows the six steps:

| Step | What happens | Typical time |
|---|---|---|
| preprocess | splits the channels, removes the MS2 background | minutes |
| segment | finds every cell in 3D (GPU) | the longest step |
| track | follows each cell from frame to frame | minutes |
| unify | repairs cells that were split or merged | minutes |
| gene_expression | measures the MS2 signal in each cell | minutes to an hour |
| activity | groups cells by activity level, draws figures | minutes |

The status labels mean:

- **running**: working now. The page refreshes by itself.
- **finished**: all done. The results are listed below the log.
- **failed**: something went wrong. Open the log (click the recording) and send its last lines
  to the MS2 maintainer.
- **stopped**: someone pressed Stop. **paused**: it was run only part of the way.
  **Resume** continues from the first unfinished step. Finished steps are not repeated.

Click a recording to see its log and buttons:

- **Open viewer**: browse the embryo frame by frame with the detected cells drawn on top.
- **Tracking dashboard**: a quality report on how well cells were followed over time.

## The results

In the recording's folder:

| File | What it is |
|---|---|
| `activity/clusters.csv` | **the main result**: one row per cell with its activity level (silent, short burst, weak, moderate, high), total and peak signal, how many frames it was active, and when it started |
| `activity/heatmap_*.png` | MS2 signal of every cell over time |
| `activity/activity_ladder.png` | how many cells are in each activity level |
| `activity/cluster_overlay.gif` | the embryo with each cell coloured by its activity level |
| `gene_expression/gene_expression_results.csv` | the MS2 signal of every cell at every frame |

PNG files open in the browser. Click a CSV to download it and open it in Excel.

---

## Analysing the results

The run measures every cell. Every decision about which cells to analyse and how is made
afterwards, on the **Analysis** page: the link at the top of the pipeline page, or
`http://localhost:8777/analysis`. Nothing there changes the run's results, so you can try
different settings as often as you like.

### 1. Samples and stages

The list on the left shows every result in the results folder, grouped by stage folder. Search
it, or paste a result's path under **add a result by its path**.

- **Stage comparison:** several embryos at one stage. Type the stage (for example 13), then
  click each recording to add it.
- **Single recording, several stages:** one embryo over its stages. Click the recording, then
  **+ add a stage** for each further stage and name each one.

For every sample, set where the stage **starts** and **ends** with the two sliders. Each shows
the frame it points at. Turn the embryo with ⟲/⟳ 90°, the angle slider and ⇋/⇵ flip, so all
embryos face the same way. Every image on the page zooms with the mouse wheel; drag to move.

The first time a recording is opened it takes about half a minute while its cells are read.

### 2. Cells: tracking filters

- **Tracked through at least …% of the window:** cells followed through less of the stage are
  left out.
- **Volume CV at most …:** cells whose segmented volume changes too much over the stage (an
  unstable segmentation) are left out.

### 3. Region and segments

One panel per sample shows the cells that pass the tracking filters, each outlined by its own
segmentation. Choose what to draw, click around it, then **close**:

- **ROI:** outline the region on the **start frame** and on the **end frame**. A cell is
  analysed if it is inside either outline on that frame. Analysed cells are outlined in blue,
  the others in grey. With no ROI, every cell that passes the filters is analysed.
- **2, 3, 4, 5:** outline each body segment.

**undo** removes the last point, **clear** starts over, **fit** shows the whole frame.
**load ROI from a file** and **load segments from a file** offer the ROIs and segments drawn
earlier for that recording. Nothing from earlier work is shown unless you load it.

**Box each segment** chooses what the segment's grid is cut from: the analysed cells inside the
outline (a tight box, the default) or the outline itself.

### 4. Activity clustering

- **Noise floor:** a cell fires on a frame when its MS2 signal is above it.
- **Blip height:** a cell whose highest signal never rises this far above the floor is a
  *short burst / noise*; it is not clustered.
- **The two rate cuts** split the rest into weakly, moderate and highly active. **Automatic**
  places them where KMeans does on all samples together; moving a cut sets it by hand.

The heatmap shows every cell above silent, all samples together, and redraws as you move a
slider. Use **zoom in** to read it.

### 5. Segment histograms

Each segment is stretched onto one square: centre (0, 0), right-upper corner (1, 1),
left-lower (−1, −1). The same segment of every sample then lies on the same square.

- **Value:** the share of cells firing, the mean intensity per cell (MS2 signal above the floor,
  summed over the window), or the share in each cluster (one row per cluster, each in its own
  colour).
- **Bins:** a grid, or about N cells per bin. A bin with only 1–2 cells can only read 0, ½ or
  1, so prefer about 4–6 cells per bin.
- **Time:** the whole window, or 0%, 10%, … 100% of each sample's window.

Hatched bins hold no cells. The table under the maps gives how alike each pair of samples' maps
are (Pearson r).

### 6. Save and export

Give the analysis a name and click **Save**: everything on the page, the outlines included, is
saved as `analyses/<name>/analysis.json` in the results folder. Saved analyses are listed below;
**load** reopens one exactly as saved.

**Save and export** also writes every figure and table into the same folder, with the page's
settings. It takes a few minutes; the files appear on the page as they are written:

| File | What it is |
|---|---|
| `cells.csv` | one row per analysed cell: its sample, activity level, rate, peak, position |
| `metrics.csv` | one row per sample: how many cells, how many in each level, and more |
| `heatmap_active_*.png` | the heatmap per sample and all samples together |
| `level_composition.png`, `per_cell_metrics.png`, `timecourse.png` | the samples compared |
| `spatial_maps_aligned.png` | every sample's cells, turned as set on the page |
| `segment_activity/segment_<n>_*.png` | the segment histograms: firing, intensity, clusters, and a GIF over time |
| `segment_activity/*_correlation.csv` | how alike the samples' maps are |

The p-values in `stats.csv` compare **cells**, not embryos. A small p-value means the samples
differ, not that the stages or conditions differ. To claim that, you need several embryos per
group.

---

## When something goes wrong

| What you see | What to do |
|---|---|
| `Permission denied (publickey,password)` | The SSH key isn't set up. Repeat **One-time setup, step 1**. |
| `Could not resolve hostname golgi` | You are outside the Technion network. Connect to the VPN, or set `HOST=132.68.106.73` in the settings. |
| `port … is in use` | Someone else uses your port number, or your app is already open in another window. Pick another `PORT`. |
| The browser shows "unable to connect" | The black window has closed. Start the app again. |
| A run **failed** at *segment* with "out of memory" | The GPU was busy. Pick another GPU on **New analysis** and press **Resume**. |
| "outside the data folders" | The app may only open files under the lab folder. Move the CZI there. |
