#!/usr/bin/env bash
# Ask this checkout's running TranscriptViewer to write an accessibility report of every visible
# window (JSON: elements, smallTargets, unlabeled) under edits/work/viewer-debug/<stamp>/, add a
# screenshot of each of those windows (screencapture -l: needs Screen Recording for the
# terminal; occluded windows are captured as drawn) and print the folder. Nothing is clicked or
# focused. The app keeps the latest 10 captures.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd -P)/edits/work/viewer-debug"
app="$(cd "$(dirname "$0")/.." && pwd -P)/TranscriptViewer/dist/TranscriptViewer.app"
if ! pgrep -f "$app/Contents/MacOS/TranscriptViewer" >/dev/null; then
  echo "TranscriptViewer from $app is not running" >&2
  exit 1
fi
newest() {
  local path
  path="$(find "$root" -mindepth 1 -maxdepth 1 2>/dev/null | sort | tail -n 1 || true)"
  echo "${path##*/}"
}
before="$(newest)"
osascript -l JavaScript >/dev/null <<'JS'
ObjC.import("Foundation");
$.NSDistributedNotificationCenter.defaultCenter.postNotificationNameObjectUserInfoDeliverImmediately(
  "works.kkkaoru.TranscriptViewer.debugCapture", $(), $(), true)
JS
for _ in $(seq 1 100); do
  latest="$(newest)"
  if [ -n "$latest" ] && [ "$latest" != "$before" ] && [ -f "$root/$latest/done.json" ]; then
    while IFS=$'\t' read -r number name; do
      screencapture -x -o -l "$number" "$root/$latest/$name.png" \
        || echo "screenshot of window $number failed (Screen Recording permission?)" >&2
    done <"$root/$latest/windows.tsv"
    echo "$root/$latest"
    exit 0
  fi
  sleep 0.1
done
echo "no capture within 10 s (is the app built with DebugCapture?)" >&2
exit 1
