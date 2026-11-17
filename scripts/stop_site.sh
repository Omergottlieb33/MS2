#!/usr/bin/env bash
# Stop the lab site running on a port.
#
#   bash scripts/stop_site.sh PORT [--force]
#
# Pipeline runs already started keep going: they run in their own session.
# If the site runs as a systemd --user service, the service is stopped (otherwise it would
# start the site again). --force stops whatever program of yours listens on the port, even
# if it isn't an ms2 site.

set -euo pipefail
. "$(dirname "$0")/_site_common.sh"

[ $# -ge 1 ] || die "usage: bash scripts/stop_site.sh PORT [--force]"
PORT=$1
FORCE=0
[ "${2:-}" = --force ] && FORCE=1
check_port "$PORT"

pid=$(pid_on_port "$PORT")
if [ -z "$pid" ]; then
    port_in_use "$PORT" && die "port $PORT is used by another user's program: only they (or an admin) can stop it."
    say "no site is running on port $PORT"
    exit 0
fi

cmd=$(tr '\0' ' ' </proc/"$pid"/cmdline)
say "port $PORT: pid $pid: $cmd"
if [ "$FORCE" = 0 ] && ! grep -qE 'ms2' <<<"$cmd"; then
    die "that isn't an ms2 site. To stop it anyway: bash scripts/stop_site.sh $PORT --force"
fi

# Started by a systemd --user service? Stop the service, or systemd may start it again.
unit=$(grep -oE '[^/]+\.service' /proc/"$pid"/cgroup 2>/dev/null | grep -vE '^(user@[0-9]+|session-.*)\.service$' | tail -1 || true)
if [ -n "$unit" ] && systemctl --user cat "$unit" >/dev/null 2>&1; then
    say "it runs as the systemd service $unit: stopping the service"
    systemctl --user stop "$unit"
    warn "$unit still starts at boot. To stop that too: systemctl --user disable $unit"
else
    kill "$pid"
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
        warn "pid $pid didn't stop within 10 s: killing it"
        kill -9 "$pid"
        sleep 1
    fi
fi

if [ -n "$(pid_on_port "$PORT")" ]; then
    die "something still listens on port $PORT"
fi
echo "---- stopped by stop_site.sh at $(date)" >>"$(site_log "$PORT")"
say "site on port $PORT stopped"
