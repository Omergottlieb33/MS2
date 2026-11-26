#!/usr/bin/env bash
# Build the ms2 conda environment from environment.yml, or update it if it already exists.
#
#   bash scripts/setup_env.sh [NAME | PATH]
#
#   NAME   a conda env name                        (default: ms2)
#   PATH   a folder, for a shared env, e.g. /zjbd/zd1/shechtmanlab/conda_envs/ms2
#
# The repo is installed in editable mode, so a later 'git pull' updates the code without
# running this again. Run it again when environment.yml or pyproject.toml change.
# At the end it checks that the ms2 package imports and whether PyTorch sees a GPU.

set -euo pipefail
. "$(dirname "$0")/_site_common.sh"

TARGET=${1:-ms2}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"   # environment.yml installs the repo with '-e .'

command -v conda >/dev/null || die "conda not found. Install miniconda or miniforge, or load conda into this shell."

if [[ "$TARGET" == */* ]]; then
    TARGET=$(realpath -m "$TARGET")
    WHERE=(-p "$TARGET")
    ENV_DIR=$TARGET
    exists() { [ -x "$TARGET/bin/python" ]; }
else
    WHERE=(-n "$TARGET")
    env_dir() { conda env list | awk -v n="$TARGET" '$1 == n {print $NF}'; }
    exists() { [ -n "$(env_dir)" ]; }
fi

if exists; then
    say "env $TARGET exists: updating it from environment.yml"
    conda env update "${WHERE[@]}" -f environment.yml
else
    say "creating env $TARGET from environment.yml (this takes a while)"
    conda env create "${WHERE[@]}" -f environment.yml
fi

[ -n "${ENV_DIR:-}" ] || ENV_DIR=$(env_dir)
PY=$ENV_DIR/bin/python
[ -x "$PY" ] || die "no python at $PY: the env wasn't built"

say "checking the env ($PY)"
"$PY" -c 'import ms2.app.server' || die "the ms2 package doesn't import: see the errors above"
if "$PY" -c 'import sys, torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)'; then
    say "PyTorch sees the GPU"
else
    warn "PyTorch doesn't see a GPU: segmentation will be slow. On a GPU server, check 'nvidia-smi' (needs a CUDA 12.4 driver)."
fi

say "done. Use it with: conda activate $TARGET"
