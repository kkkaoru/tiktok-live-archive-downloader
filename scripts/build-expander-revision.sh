#!/usr/bin/env bash
# The legacy caller-named revision writer bypassed per-source delivery ownership.
# Keep the entry point fail-closed so old automation cannot recreate duplicates.
set -euo pipefail
printf '%s\n' \
  'Legacy expander revision publication is retired: it creates duplicate deliveries.' \
  'Use the shared dated, per-source-locked replay.audio_restore workflow instead.' >&2
exit 2
