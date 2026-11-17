# Shared by start_site.sh, stop_site.sh and update_site.sh. Sourced, not run.

say()  { printf '\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33mwarning: %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31merror: %s\033[0m\n' "$*" >&2; exit 1; }

# Personal settings (MS2_WORKSPACE, MS2_PYTHON, ...), if any: KEY=value lines.
if [ -f "$HOME/.ms2_site.env" ]; then . "$HOME/.ms2_site.env"; fi

check_port() {   # port
    [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -ge 1024 ] && [ "$1" -le 65535 ] ||
        die "port must be a number from 1024 to 65535, got '$1'"
}

# pid of this user's process listening on a TCP port (other users' pids aren't visible)
pid_on_port() {
    { ss -ltnp "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; } || true
}

# true if anyone (any user) listens on the port
port_in_use() {
    ss -ltnH "sport = :$1" 2>/dev/null | grep -q .
}

wait_up() {   # port [seconds]
    local i
    for i in $(seq 1 "${2:-60}"); do
        curl -s -o /dev/null --max-time 2 "http://localhost:$1/" && return 0
        sleep 1
    done
    return 1
}

site_log() { echo "$HOME/ms2_site_$1.log"; }

server_address() {   # first address of this machine, for the link to give users
    hostname -I 2>/dev/null | awk '{print $1}' | grep . || hostname
}
