#!/usr/bin/env bash
# Mirror this project to Snellius (~/dumbcoder), deleting remote files
# that no longer exist locally.
#
# Usage:
#   ./sync.sh          # show what would change, then mirror after confirmation
#   ./sync.sh -n        # dry run only (itemized list, no changes)
#   ./sync.sh -y        # mirror without asking for confirmation
set -euo pipefail

LOCAL="/Users/gidonkaminer/Documents/mol/s26/dumbcoder/"
REMOTE="snellius:~/dumbcoder"

# --delete mirrors local -> remote (removes stale remote files).
# Excludes keep junk and local-only state off the cluster.
RSYNC_OPTS=(
  -av --delete
  --exclude '.git/'
  --exclude '.claude/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '.DS_Store'
)

dry_run=0
assume_yes=0
while getopts "ny" opt; do
  case "$opt" in
    n) dry_run=1 ;;
    y) assume_yes=1 ;;
    *) echo "usage: $0 [-n] [-y]" >&2; exit 2 ;;
  esac
done

echo "Preview of changes (dry run):"
rsync "${RSYNC_OPTS[@]}" -n --itemize-changes "$LOCAL" "$REMOTE"

if [[ "$dry_run" -eq 1 ]]; then
  exit 0
fi

if [[ "$assume_yes" -ne 1 ]]; then
  read -r -p "Apply these changes (including deletions) to $REMOTE? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
fi

echo "Syncing..."
rsync "${RSYNC_OPTS[@]}" "$LOCAL" "$REMOTE"
echo "Done."
