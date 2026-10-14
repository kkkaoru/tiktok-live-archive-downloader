#!/bin/bash
# Build or test the shared native package with repository-local working storage.
set -euo pipefail
umask 077
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
WORK_NAME=${REPLAY_NATIVE_WORK_NAME:-native-swift}
case "$WORK_NAME" in
  .|..|*[!a-zA-Z0-9._-]*) printf 'Invalid native working-directory name.\n' >&2; exit 2 ;;
esac
WORK="$ROOT/edits/work/$WORK_NAME"
case "${1:-}" in
  build|test) ACTION=$1; shift ;;
  *) printf 'Usage: bash scripts/native-swift.sh build|test [SwiftPM options]\n' >&2; exit 2 ;;
esac
test "$(cd "$ROOT/native/apple-pro-apps" && pwd -P)" = "$ROOT/native/apple-pro-apps"
for directory in "$ROOT/edits" "$ROOT/edits/work" "$WORK" \
  "$WORK"/{tmp,cache,config,security,module-cache,build,profiles}; do
  if [ -L "$directory" ]; then
    printf 'Native working storage must not be a symlink.\n' >&2
    exit 1
  fi
  mkdir -p "$directory"
  test "$(cd "$directory" && pwd -P)" = "$directory"
done
export TMPDIR="$WORK/tmp" TMP="$WORK/tmp" TEMP="$WORK/tmp"
export XDG_CACHE_HOME="$WORK/cache"
export CLANG_MODULE_CACHE_PATH="$WORK/module-cache"
export SWIFTPM_MODULECACHE_OVERRIDE="$WORK/module-cache"
export LLVM_PROFILE_FILE="$WORK/profiles/%p-%m.profraw"
exec swift "$ACTION" --package-path "$ROOT/native/apple-pro-apps" \
  --scratch-path "$WORK/build" --cache-path "$WORK/cache" \
  --config-path "$WORK/config" --security-path "$WORK/security" \
  --force-resolved-versions -Xswiftc -warnings-as-errors "$@"
