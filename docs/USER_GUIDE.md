# MS2 app: user guide

The MS2 app turns a live-imaging recording (a Zeiss `.czi` file) into a table of how strongly
each cell transcribes over time. Everything runs on the lab's GPU server (`golgi`). Your
computer only shows the app in a web browser.

**The quick way, if the lab page is running:** open `http://golgi:8765` (ask for the address
and the lab password). Log in with your name as the user name and the lab password. Then,
under **CZI file**, pick the CZI from your computer and carry on from
[Draw the region of interest](#2-draw-the-region-of-interest-optional-recommended). When the
run has finished, **Download results** on the Runs page gives you a zip of the logs and result
tables to save wherever you like. You need nothing else on your computer. The rest of this
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

### 2. Draw the region of interest (optional, recommended)

Click **Draw ROI**. The first time, this takes about a minute while the app prepares the image.
You will see the embryo twice, side by side:

- **Left:** move the **start frame** slider to the first frame of the developmental stage you
  want to analyse.
- **Right:** move the **end frame** slider to the last frame of that stage.
- On each side, click around the region to outline it, then click **close** to finish the
  shape. **undo** removes the last point and **clear** starts over. Zoom with the mouse wheel.
  Move the image by dragging with the right mouse button.
- Click **save roi**, then **continue →** to return.

A cell is analysed if it is inside **either** outline, the one at the start or the one at
the end. Activity is measured only between the start and end frames.

Without an ROI, the app analyses every cell that was tracked for most of the recording,
across all frames.

### 3. Start

- **Processor:** pick a GPU with the most free memory. The CPU option works but takes very long.
- **Advanced settings:** leave them alone unless you know why you are changing them.
  *MS2 spot prominence* is how bright a spot must be to count as transcription.

Click **Start analysis**. The app switches to **Runs & results**.

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
- **Edit ROI**: change the region or the frames. Then press **Resume**, and only the
  measurement steps run again.
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

## Comparing recordings

Several embryos at the same stage, for example. Put the recordings in the same results folder
and run each one first. Then open **Compare recordings**:

1. Tick the recordings to compare.
2. Check the start and end frames. They are filled in from each recording's ROI. Without an
   ROI you must type the end frame.
3. Click **Start comparison**. Results appear below and in the `_comparison` folder.

The p-values compare **cells**, not embryos. A small p-value means the two recordings differ, not
that the two stages or conditions differ. To claim that, you need several embryos per group.

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
