# MS2

Turns live-imaging recordings of embryos (Zeiss `.czi`, MS2 + cell channel) into per-cell
transcriptional activity.

## Overview

For each recording, the pipeline:

1. **Preprocesses**: splits the channels and removes the MS2 background.
2. **Segments**: finds every cell in 3D with Cellpose (GPU).
3. **Tracks**: follows each cell from frame to frame.
4. **Measures gene expression**: detects MS2 transcription spots and quantifies them per cell
   over time.
5. **Classifies activity**: groups cells from silent to highly active, and compares
   developmental stages across embryos.

Everything runs from a browser app on the lab's GPU server:
- Pick a CZI.
- Draw the region of interest on the first and last frames of the stage.
- Start the run.
- Follow its progress and open the results.

Lab members: see the [user guide](docs/USER_GUIDE.md). Command line and details:
[docs/REFERENCE.md](docs/REFERENCE.md).

## Install the environment

On the GPU server you need conda and an NVIDIA GPU (CUDA 12.4 build of PyTorch).

```bash
git clone https://github.com/Omergottlieb33/MS2.git
cd MS2
conda env create -f environment.yml          # creates the env "ms2" and installs this repo
conda activate ms2
```

To make one environment for the whole lab, create it in the shared folder instead:

```bash
conda env create -p /zjbd/zd1/shechtmanlab/conda_envs/ms2 -f environment.yml
```

The repo is installed in editable mode, so `git pull` updates the code for everyone using
the env.

## Run the app

The app runs on the server. Your computer reaches it through an SSH tunnel.

**With the launcher (Windows or Linux).** Do this once:
1. Set up an SSH key to the server. The [user guide](docs/USER_GUIDE.md) has the steps.
2. Fill in your username and a personal port in `launchers/ms2_settings.txt`.

Then double-click `launchers/MS2.bat` on Windows, or run `launchers/ms2.sh` on Linux. The
browser opens by itself.

**By hand.** From your computer:

```bash
ssh -L 8765:localhost:8765 your-username@golgi
```

Then, in that SSH session on the server:

```bash
conda activate ms2
ms2 app --data-root /zjbd/zd1/shechtmanlab --port 8765
```

Open http://localhost:8765 in your browser.

- **Port:** use your own port number so you don't collide with other lab members. Use the
  same number on both sides of `-L`, and in `--port`.
- **Data root:** `--data-root` is the folder the app may browse for CZI files and write
  results into.
- **Closing:** closing the app or the SSH session doesn't stop runs that have already started.

## License

MIT, see [LICENSE](LICENSE). To cite: Gottlieb, O. *MS2: infrastructure for cell and
fluorescence analysis* (2025), https://github.com/Omergottlieb33/MS2.
