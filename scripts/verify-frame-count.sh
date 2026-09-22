#!/usr/bin/env bash
# Generic video frame-count gate: sum video-stream nb_frames over media files
# and compare against an expected count. No pipeline knowledge; reusable for
# any future delivery/master verification.
# Usage: verify-frame-count.sh --expected N <media-file>...
set -euo pipefail
[[ "${1:-}" == --expected ]] || { printf 'Usage: %s --expected N <media-file>...\n' "$0" >&2; exit 2; }
expected=${2:?expected frame count}
shift 2
[[ "$expected" =~ ^[0-9]+$ ]] || { printf 'Non-numeric expected count: %s\n' "$expected" >&2; exit 2; }
(("$#" > 0)) || { printf 'No media files given\n' >&2; exit 2; }
total=0
for f in "$@"; do
  n=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$f")
  [[ "$n" =~ ^[0-9]+$ ]] || { printf 'FRAME_COUNT_UNREADABLE %s\n' "$f" >&2; exit 1; }
  total=$((total + n))
  printf '%s frames=%s\n' "$f" "$n"
done
printf 'SUM_FRAMES=%s EXPECTED=%s\n' "$total" "$expected"
[[ "$total" -eq "$expected" ]] || { printf 'FRAME_COUNT_MISMATCH\n' >&2; exit 1; }
printf 'FRAMES_VERIFIED %s\n' "$total"
