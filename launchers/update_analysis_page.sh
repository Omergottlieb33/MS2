#!/bin/sh
# Runs on the server, from the analysis page's own checkout (see docs/REFERENCE.md): brings it
# up to the newest commit of BRANCH and restarts the page if that changed anything.
#     update_analysis_page.sh [BRANCH]          (default: main)
# Only fast-forwards: local edits or a rewritten branch stop it with an error in the journal,
# rather than being thrown away.
set -e
cd "$(dirname "$0")/.."
branch=${1:-main}
git fetch --quiet origin "$branch"
[ "$(git rev-parse HEAD)" = "$(git rev-parse FETCH_HEAD)" ] && exit 0
git merge --ff-only --quiet FETCH_HEAD
echo "updated to $(git log -1 --format='%h %s')"
systemctl --user restart ms2-analysis
