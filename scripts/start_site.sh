#!/usr/bin/env bash
# Start a lab site (Run pipeline + Analysis, behind the lab password) on a port.
#
#   bash scripts/start_site.sh PORT [extra 'ms2 app' options]
#
# The site runs in the background and keeps running after you log out. Stop it with
#   bash scripts/stop_site.sh PORT
# Its log: ~/ms2_site_<PORT>.log
#
# Settings: environment variables, or KEY=value lines in ~/.ms2_site.env
#   MS2_PYTHON            python of the ms2 environment
#                         (default: the active conda env, else the conda env "ms2",
#                          else the lab's shared env, else python3 in PATH)
#   MS2_PASSWORD_FILE     lab password file              (default ~/.ms2_password)
#   MS2_WORKSPACE         users' uploads and runs        (default ~/ms2_web)
#   MS2_ANALYSIS_FOLDERS  result folders listed for all  (default ~/ms2_analysis_folders.txt;
#                         if the file is missing, the Analysis page shows only users' runs)
#   MS2_ANALYSIS_SAVES    saved analyses                 (default ~/ms2_analysis_saves)
#   MS2_HOST              address to listen on           (default 0.0.0.0: other computers)

set -euo pipefail
. "$(dirname "$0")/_site_common.sh"

[ $# -ge 1 ] || die "usage: bash scripts/start_site.sh PORT [extra 'ms2 app' options]"
PORT=$1; shift
check_port "$PORT"

REPO=$(cd "$(dirname "$0")/.." && pwd)
PASSWORD_FILE=${MS2_PASSWORD_FILE:-$HOME/.ms2_password}
WORKSPACE=${MS2_WORKSPACE:-$HOME/ms2_web}
ANALYSIS_FOLDERS=${MS2_ANALYSIS_FOLDERS:-$HOME/ms2_analysis_folders.txt}
ANALYSIS_SAVES=${MS2_ANALYSIS_SAVES:-$HOME/ms2_analysis_saves}
HOST=${MS2_HOST:-0.0.0.0}
SHARED_ENV=/zjbd/zd1/shechtmanlab/conda_envs/ms2

# ---------------------------------------------------------------- port free?
if port_in_use "$PORT"; then
    pid=$(pid_on_port "$PORT")
    if [ -n "$pid" ]; then
        die "port $PORT is already in use by your process $pid: $(tr '\0' ' ' </proc/$pid/cmdline). Stop it with: bash scripts/stop_site.sh $PORT"
    fi
    die "port $PORT is in use by another user's program. Pick another port."
fi

# ---------------------------------------------------------------- python
find_python() {
    local c
    if [ -n "${MS2_PYTHON:-}" ]; then echo "$MS2_PYTHON"; return; fi
    [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ] && echo "$CONDA_PREFIX/bin/python"
    if command -v conda >/dev/null; then
        conda env list 2>/dev/null | awk '$1 == "ms2" {print $NF "/bin/python"}'
    fi
    for c in "$HOME"/miniconda3/envs/ms2 "$HOME"/anaconda3/envs/ms2 "$HOME"/miniforge3/envs/ms2 "$SHARED_ENV"; do
        echo "$c/bin/python"
    done
    command -v python3 || true
}
PY=""
while read -r c; do
    if [ -x "$c" ] && (cd "$REPO" && "$c" -c 'import ms2.app.server' 2>/dev/null); then PY=$c; break; fi
done < <(find_python)
[ -n "$PY" ] || die "no python with the ms2 package found. Activate the env ('conda activate ms2'), or set MS2_PYTHON=/path/to/env/bin/python"
say "python: $PY"

# ---------------------------------------------------------------- site files
if [ ! -s "$PASSWORD_FILE" ]; then
    # First start on this server: the person building the site chooses the lab password.
    [ -t 0 ] || die "no lab password in $PASSWORD_FILE. Make it with:
    echo 'the-lab-password' > $PASSWORD_FILE && chmod 600 $PASSWORD_FILE"
    say "no lab password on this server yet: choose the password users will log in with"
    while :; do
        read -rsp "password: " pw1; echo
        read -rsp "again:    " pw2; echo
        if [ -z "$pw1" ] || [[ "$pw1" =~ ^[[:space:]]|[[:space:]]$ ]]; then
            warn "empty, or starts or ends with a space: try again"
        elif [ "$pw1" != "$pw2" ]; then
            warn "the two don't match: try again"
        else
            break
        fi
    done
    (umask 077; printf '%s\n' "$pw1" >"$PASSWORD_FILE")
    unset pw1 pw2
    say "saved in $PASSWORD_FILE (only you can read it). To change it, edit the file and restart the site."
fi
mkdir -p "$WORKSPACE"

ARGS=(-m ms2.cli app --host "$HOST" --port "$PORT"
      --workspace "$WORKSPACE" --password-file "$PASSWORD_FILE")
if [ -s "$ANALYSIS_FOLDERS" ]; then
    mkdir -p "$ANALYSIS_SAVES"
    ARGS+=(--analysis-folders "$ANALYSIS_FOLDERS" --analysis-saves "$ANALYSIS_SAVES")
else
    warn "no $ANALYSIS_FOLDERS: the Analysis page shows only the users' own runs. List result folders there, one per line, and restart to add them."
fi
ARGS+=("$@")

# ---------------------------------------------------------------- start
LOG=$(site_log "$PORT")
say "starting the site on port $PORT (log: $LOG)"
(
    cd "$REPO"
    echo "---- started by start_site.sh at $(date) on $(git rev-parse --short HEAD 2>/dev/null || echo '?')" >>"$LOG"
    setsid nohup "$PY" "${ARGS[@]}" >>"$LOG" 2>&1 </dev/null &
)

if wait_up "$PORT" 120; then
    say "site is up: http://$(server_address):$PORT/   (pid $(pid_on_port "$PORT"))"
    say "stop it with: bash scripts/stop_site.sh $PORT"
else
    warn "the site didn't answer within 120 s. The end of its log:"
    tail -n 20 "$LOG" >&2
    exit 1
fi
