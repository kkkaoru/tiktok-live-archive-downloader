#!/usr/bin/env bash
# Generic downward-expansion cleanup for separated vocals with a speech mask.
#
# A residual music bed left by vocal separation sits below speech level. This
# script derives, from the input's own loudness, the strongest expansion
# threshold that keeps speech untouched, applies it, and re-measures proof.
# No per-video tuning: every decision comes from replay.audio_expander.
#
# Usage: expand-vocals.sh <input.wav> <transcript.srt> <output.wav>
set -euo pipefail
umask 077
in=${1:?input wav required}
srt=${2:?transcript srt required}
out=${3:?output wav required}
[[ -f "$in" ]] || { printf 'EXPANDER_INPUT_MISSING %s\n' "$in" >&2; exit 2; }
[[ -f "$srt" ]] || { printf 'EXPANDER_TRANSCRIPT_MISSING %s\n' "$srt" >&2; exit 2; }
repo=$(cd "$(dirname "$0")/.." && pwd)
work="$out.expander-work"
if [[ -f "$out" && -f "$work/verify.json" ]] && jq -e '.passed == true' "$work/verify.json" >/dev/null; then
  printf 'EXPANDER_EXISTS %s\n' "$out"
  exit 0
fi
mkdir -p "$work"

measure() {
  ffmpeg -hide_banner -nostdin -v info -i "$1" -map 0:a:0 -af ebur128 -f null - 2> "$2"
}

plan_json="$work/plan.json"
measure "$in" "$work/before.log"
PYTHONPATH="$repo/src" uv run --locked --project "$repo" python - "$work/before.log" "$srt" "$plan_json" <<'PY'
import json
import sys
from pathlib import Path
from replay.audio_expander import plan_from_texts

log_path, srt_path, out_path = sys.argv[1:4]
try:
    plan = plan_from_texts(Path(log_path).read_text(), Path(srt_path).read_text())
except ValueError as exc:
    print(f"EXPANDER_PLAN_FAILED {exc}", file=sys.stderr)
    raise SystemExit(1)
Path(out_path).write_text(json.dumps(plan, indent=2) + "\n")
print(f"EXPANDER_PLANNED feasible={plan['feasible']} thresholdDb={plan['thresholdDb']:.2f}")
if not plan["feasible"]:
    print(f"EXPANDER_INFEASIBLE {plan['reason']}", file=sys.stderr)
    raise SystemExit(1)
PY
threshold_db=$(jq -r '.thresholdDb' "$plan_json")
[[ "$threshold_db" =~ ^-?[0-9]+(\.[0-9]+)?$ ]] || { printf 'EXPANDER_PLAN_UNREADABLE\n' >&2; exit 1; }
# Curve rendering lives in the tested module, so the shell never re-derives
# DSP domain rules (a stale inline copy once rejected valid deep thresholds).
curve_for() {
  PYTHONPATH="$repo/src" uv run --locked --project "$repo" python - "$1" <<'PY'
import sys
from replay.audio_expander import transfer_points

try:
    print(transfer_points(float(sys.argv[1])))
except ValueError as exc:
    print(f"EXPANDER_THRESHOLD_OUT_OF_RANGE {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}
points=$(curve_for "$threshold_db") || exit 1
printf 'EXPANDER_CURVE %s\n' "$points"

# One principled fallback: a real envelope follower may bite speech harder
# than the LUFS-median predictor, so a single 3 dB backoff retest is allowed.
# Anything else fails loudly instead of guessing further.
attempt=0
for backoff in 0 3; do
  attempt=$((attempt + 1))
  level=$(python3 -c 'import sys; print(float(sys.argv[1]) - float(sys.argv[2]))' "$threshold_db" "$backoff")
  curve=$(curve_for "$level") || exit 1
  ffmpeg -v error -xerror -nostdin -i "$in" -map 0:a:0 \
    -af "compand=attacks=0.02:decays=0.25:points=$curve" \
    -c:a pcm_s16le -ar 44100 "$work/expanded-attempt.wav" -y
  measure "$work/expanded-attempt.wav" "$work/after.log"
  PYTHONPATH="$repo/src" uv run --locked --project "$repo" python - "$work/before.log" "$work/after.log" "$srt" "$work/verify.json" <<'PY'
import json
import sys
from pathlib import Path
from replay.audio_expander import verify_from_texts

before_path, after_path, srt_path, out_path = sys.argv[1:5]
result = verify_from_texts(
    Path(before_path).read_text(), Path(after_path).read_text(), Path(srt_path).read_text()
)
Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
print(
    f"EXPANDER_ATTEMPT passed={result['passed']} "
    f"suppressionDb={result['suppressionDb']:.2f} speechLossDb={result['speechLossDb']:.2f}"
)
PY
  if jq -e '.passed == true' "$work/verify.json" >/dev/null; then
    mv "$work/expanded-attempt.wav" "$out"
    rm -f "$work/before.log" "$work/after.log"
    suppression=$(jq -r '.suppressionDb' "$work/verify.json")
    loss=$(jq -r '.speechLossDb' "$work/verify.json")
    printf 'EXPANSION_VERIFIED attempt=%s suppressionDb=%s speechLossDb=%s %s\n' \
      "$attempt" "$suppression" "$loss" "$out"
    exit 0
  fi
  printf 'EXPANDER_ATTEMPT_FAILED attempt=%s backoffDb=%s\n' "$attempt" "$backoff" >&2
done
printf 'EXPANSION_VERIFY_FAILED attempts=2 %s\n' "$in" >&2
exit 1
