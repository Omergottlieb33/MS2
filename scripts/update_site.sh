#!/usr/bin/env bash
# Update the lab site on the server to the latest commit of a branch, and restart it.
#
#   bash scripts/update_site.sh [BRANCH]
#
# BRANCH defaults to the branch checked out now. The script:
#   1. stops if tracked files have uncommitted changes (they would block or be lost in a pull)
#   2. fetches, checks out BRANCH and pulls it (fast-forward only)
#   3. reinstalls the package if pyproject.toml or environment.yml changed
#   4. restarts the site, unless only page files (static html/js/css) changed. Page files are
#      read on every request, so a browser refresh shows them
#   5. waits until the site answers again
#
# How the site is restarted:
#   - a systemd --user service named $MS2_SERVICE (default ms2-site), if it exists, or else
#   - the process listening on each port in $MS2_PORTS (default 8790): it is stopped and started
#     again with the same command, folder and environment, in the background. Its log goes to
#     ~/ms2_site_<port>.log
# Pipeline runs already started keep going: they run in their own session.
#
# Settings (environment variables):
#   MS2_PORTS="8790 8800"   every port served from this checkout
#   MS2_SERVICE=ms2-site    systemd --user service name
#   MS2_FORCE_RESTART=1     restart even if only page files changed

set -euo pipefail
. "$(dirname "$0")/_site_common.sh"

PORTS=${MS2_PORTS:-8790}
SERVICE=${MS2_SERVICE:-ms2-site}
FORCE_RESTART=${MS2_FORCE_RESTART:-0}

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
say "repo: $PWD"

# ---------------------------------------------------------------- 1. clean tree
if ! git diff --quiet || ! git diff --cached --quiet; then
    git status --short --untracked-files=no
    die "tracked files have uncommitted changes. Commit them, or discard with 'git checkout -- <file>'."
fi

# ---------------------------------------------------------------- 2. pull
BRANCH=${1:-$(git rev-parse --abbrev-ref HEAD)}
[ "$BRANCH" = HEAD ] && die "detached HEAD: give the branch name, e.g. bash scripts/update_site.sh main"

OLD=$(git rev-parse HEAD)
say "fetching origin"
git fetch --prune origin
git rev-parse --verify --quiet "origin/$BRANCH" >/dev/null || die "no branch origin/$BRANCH"

if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
    say "checking out $BRANCH"
    if git rev-parse --verify --quiet "refs/heads/$BRANCH" >/dev/null; then
        git checkout "$BRANCH"
    else
        git checkout -b "$BRANCH" --track "origin/$BRANCH"
    fi
fi
say "pulling $BRANCH"
git pull --ff-only origin "$BRANCH" || die "can't fast-forward: the server's $BRANCH has commits not on GitHub. Look with 'git log origin/$BRANCH..$BRANCH'."
NEW=$(git rev-parse HEAD)

if [ "$OLD" = "$NEW" ]; then
    say "no new commits (already at $(git log -1 --format='%h %s'))"
    CHANGED=""
else
    say "updated $(git rev-parse --short "$OLD") -> $(git rev-parse --short "$NEW")"
    git log --format='    %h %s' "$OLD..$NEW" | head -20
    CHANGED=$(git diff --name-only "$OLD" "$NEW")
fi

# ---------------------------------------------------------------- find the running site
have_service() {
    command -v systemctl >/dev/null &&
        systemctl --user cat "$SERVICE" >/dev/null 2>&1
}

# python of the running site, for reinstalling into the same environment
site_python() {
    local port pid
    for port in $PORTS; do
        pid=$(pid_on_port "$port")
        if [ -n "$pid" ]; then readlink "/proc/$pid/exe"; return; fi
    done
    command -v python || true
}

# ---------------------------------------------------------------- 3. dependencies
if printf '%s\n' "$CHANGED" | grep -qxE 'pyproject\.toml|environment\.yml'; then
    PY=$(site_python)
    [ -n "$PY" ] || die "dependencies changed but no python found to reinstall with. Run 'conda activate ms2' and try again."
    say "dependencies changed: reinstalling with $PY"
    "$PY" -m pip install -e . || die "pip install failed: the site was not restarted"
    if printf '%s\n' "$CHANGED" | grep -qx 'environment\.yml'; then
        warn "environment.yml changed: new conda packages need 'conda env update -f environment.yml' (pip install doesn't add them)"
    fi
fi

# ---------------------------------------------------------------- 4. restart?
needs_restart() {
    [ "$FORCE_RESTART" = 1 ] && return 0
    # No new commits: the code may have been pulled by hand before this script, so restart.
    [ -z "$CHANGED" ] && return 0
    # Anything other than static page files or docs needs a restart.
    printf '%s\n' "$CHANGED" | grep -qvE '(/static/|\.md$|^notebooks/|^images/|^docs/)'
}

if ! needs_restart; then
    say "only page files changed: no restart needed, refresh the browser"
    exit 0
fi

restart_port() {   # restart the process on a port with its own command, folder and environment
    local port=$1 pid exe cwd log
    pid=$(pid_on_port "$port")
    if [ -z "$pid" ]; then
        warn "nothing of yours is listening on port $port: skipped. Start it with: bash scripts/start_site.sh $port"
        return 1
    fi
    exe=$(readlink "/proc/$pid/exe")
    cwd=$(readlink "/proc/$pid/cwd")
    local -a cmd env_vars
    mapfile -d '' cmd < "/proc/$pid/cmdline"
    mapfile -d '' env_vars < "/proc/$pid/environ"
    cmd[0]=$exe                       # absolute path, whatever PATH this shell has
    log="$HOME/ms2_site_$port.log"

    say "port $port: stopping pid $pid ($(printf '%q ' "${cmd[@]:1}"))"
    kill "$pid"
    local i
    for i in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        warn "pid $pid didn't stop: killing it"
        kill -9 "$pid"
        sleep 1
    fi

    say "port $port: starting again (log: $log)"
    (
        cd "$cwd"
        echo "---- restarted by update_site.sh at $(date) on $(git rev-parse --short HEAD)" >>"$log"
        setsid nohup env -i "${env_vars[@]}" "${cmd[@]}" >>"$log" 2>&1 </dev/null &
    )
}

FAILED=0
declare -A BEFORE
for port in $PORTS; do BEFORE[$port]=$(pid_on_port "$port"); done

if have_service; then
    say "restarting systemd service $SERVICE"
    systemctl --user restart "$SERVICE"
    systemctl --user --no-pager --lines=0 status "$SERVICE" || true
    sleep 2
fi
# Ports the service didn't restart (their old process is still alive, or there is no service).
for port in $PORTS; do
    old=${BEFORE[$port]}
    if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
        restart_port "$port" || FAILED=1
    elif [ -z "$old" ] && ! have_service; then
        restart_port "$port" || FAILED=1     # warns that nothing is listening
    fi
done

# ---------------------------------------------------------------- 5. check
for port in $PORTS; do
    if wait_up "$port"; then
        say "port $port: up at http://localhost:$port/"
    else
        warn "port $port: no answer after 60 s. See ~/ms2_site_$port.log or 'journalctl --user -u $SERVICE -n 50'"
        FAILED=1
    fi
done

[ "$FAILED" = 0 ] && say "done: site on $(git rev-parse --short HEAD) ($BRANCH)" || die "update finished with problems (see above)"
