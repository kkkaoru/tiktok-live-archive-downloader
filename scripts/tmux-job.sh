#!/usr/bin/env bash
# Run one long local processing command in a repository-local tmux server.
#
# Usage: scripts/tmux-job.sh WORK_DIR NAME COMMAND [ARGS...]
#
# The job writes WORK_DIR/NAME.log and, on exit, WORK_DIR/NAME.status with the
# command's exit code. Temporary files and caches stay under WORK_DIR/tmp. An
# existing status or log refuses a relaunch so an earlier outcome is inspected
# before any retry.
set -euo pipefail
umask 077

if [[ $# -lt 3 ]]; then
  echo "usage: $0 WORK_DIR NAME COMMAND [ARGS...]" >&2
  exit 2
fi
repo="$(cd "$(dirname "$0")/.." && pwd -P)"
work="$(cd "$1" && pwd -P)"
name="$2"
shift 2
case "$work" in
  "$repo"/edits/work/*) ;;
  *) echo "work directory must be under edits/work" >&2; exit 2 ;;
esac
if [[ ! "$name" =~ ^[a-z0-9][a-z0-9._-]{0,63}$ ]]; then
  echo "invalid job name" >&2
  exit 2
fi
for directory in "$work/tmp" "$work/cache"; do
  if [[ -L "$directory" || ( -e "$directory" && ! -d "$directory" ) ]]; then
    echo "scratch and cache must be real directories" >&2
    exit 2
  fi
done
if [[ -L "$work/tmux.sock" || ( -e "$work/tmux.sock" && ! -S "$work/tmux.sock" ) ]]; then
  echo "socket path must be an actual Unix socket or absent" >&2
  exit 2
fi
if [[ -e "$work/$name.log" || -L "$work/$name.log" || -e "$work/$name.status" || -L "$work/$name.status" ]]; then
  echo "job $name already has a log or status; inspect it first" >&2
  exit 3
fi
# Refuse new detached work below the free-space floor (30 GiB or 5%, whichever is larger).
# Skipped only when the repository environment is absent (isolated test roots).
python="$repo/.venv/bin/python"
if [[ -x "$python" ]]; then
  if ! disk="$(cd "$repo" && "$python" -m replay.disk_budget preflight --path "$work" 2>&1)"; then
    echo "$disk" >&2
    echo "free-space floor not met; retire approved intermediates before launching $name" >&2
    exit 4
  fi
fi
# Retain this reservation even after a failed dispatch: its outcome needs inspection.
if ! mkdir "$work/$name.launch"; then
  echo "job $name already has a launch reservation; inspect it first" >&2
  exit 3
fi
mkdir -p "$work/tmp" "$work/cache"
quoted="$(printf '%q ' "$@")"
inner="set -o noclobber; cd $(printf '%q' "$repo") && export TMPDIR=$(printf '%q' "$work/tmp") TMP=$(printf '%q' "$work/tmp") TEMP=$(printf '%q' "$work/tmp") XDG_CACHE_HOME=$(printf '%q' "$work/cache") && ( $quoted ) > $(printf '%q' "$work/$name.log") 2>&1; code=\$?; printf '%s\\n' \"\$code\" > $(printf '%q' "$work/$name.status"); exit \"\$code\""
# A short relative socket avoids the macOS Unix socket path-length limit.
cd "$work"
if tmux -f /dev/null -S tmux.sock has-session -t jobs 2>/dev/null; then
  tmux -f /dev/null -S tmux.sock new-window -d -t jobs -n "$name" /bin/bash -c "$inner"
else
  tmux -f /dev/null -S tmux.sock new-session -d -s jobs -n "$name" /bin/bash -c "$inner"
fi
echo "dispatched $name; inspect its log and status before claiming completion"
