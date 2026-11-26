#!/usr/bin/env bash
# Show past results on the Analysis page of a running lab site: adds result folders to its list.
#
#   bash scripts/add_results.sh FOLDER [FOLDER ...]
#
# FOLDER is any folder holding results, e.g. /data/outputs (with STAGE-11/, STAGE-15/, ... inside):
# the page finds every recording below it. The list is ~/ms2_analysis_folders.txt (or
# MS2_ANALYSIS_FOLDERS, as in start_site.sh); the site reads it on every request, so the new
# results show after pressing Refresh on the page, without a restart.

set -euo pipefail
. "$(dirname "$0")/_site_common.sh"

[ $# -ge 1 ] || die "usage: bash scripts/add_results.sh FOLDER [FOLDER ...]"
LIST=${MS2_ANALYSIS_FOLDERS:-$HOME/ms2_analysis_folders.txt}
touch "$LIST"

ADDED=0
for dir in "$@"; do
    [ -d "$dir" ] || { warn "$dir is not a folder: skipped"; continue; }
    dir=$(realpath "$dir")
    if grep -qxF "$dir" "$LIST"; then
        say "$dir is already listed"
        continue
    fi
    # The results the page lists: a recording's gene_expression*/gene_expression_results*.csv
    n=$(find "$dir" -maxdepth 7 \( -name masks -o -name masks_split -o -name preprocess \
            -o -name tracklets \) -prune -o -path '*/gene_expression*/gene_expression_results*.csv' \
            -print 2>/dev/null | wc -l)
    if [ "$n" = 0 ]; then
        warn "no results in $dir (no gene_expression*/gene_expression_results*.csv below it): skipped"
        continue
    fi
    echo "$dir" >>"$LIST"
    say "added $dir ($n result files)"
    ADDED=1
done

[ "$ADDED" = 1 ] || exit 0
say "list: $LIST"

# A site started while the list was missing doesn't read it: it needs a restart.
sites=$(ps -u "$(id -un)" -o args= | grep -E 'ms2(\.cli)? app' | grep -v grep || true)
if [ -z "$sites" ]; then
    say "no site of yours is running: the results show once you start one (bash scripts/start_site.sh PORT)"
elif grep -qF -- "--analysis-folders" <<<"$sites"; then
    say "done: press Refresh in the Analysis page's recording list"
else
    warn "the running site was started without the list, so it doesn't read it. Restart it:
    bash scripts/stop_site.sh PORT && bash scripts/start_site.sh PORT"
fi
