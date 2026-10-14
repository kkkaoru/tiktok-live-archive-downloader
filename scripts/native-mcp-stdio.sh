#!/bin/bash
# Launch one existing repository-local MCP binary with private stdio diagnostics.
# Usage: bash scripts/native-mcp-stdio.sh WORK_NAME ABSOLUTE_BINARY [ARGS...]
# No scheduler, daemon, app operation or transport protocol is added. Each launch
# retains an exclusive private history directory; stdin/stdout and exit stay native.
exec 2>/dev/null
set -euo pipefail
umask 077

if [[ $# -lt 2 ]]; then exit 2; fi
stdio_work_name=$1
stdio_binary=$2
shift 2
case "$stdio_work_name" in
  .|..|''|*[!a-zA-Z0-9._-]*) exit 2 ;;
esac
if [[ ${#stdio_work_name} -gt 64 || -L "$0" ]]; then exit 2; fi
stdio_scripts=$(cd "$(dirname "$0")" && pwd -P)
stdio_repo=$(cd "$stdio_scripts/.." && pwd -P)
if [[ "$stdio_scripts" != "$stdio_repo/scripts" ]]; then exit 2; fi
case "$stdio_binary" in
  "$stdio_repo"/*) ;;
  *) exit 2 ;;
esac
if [[ -L "$stdio_binary" || ! -f "$stdio_binary" || ! -x "$stdio_binary" ]]; then exit 2; fi
stdio_binary_parent=$(cd "$(dirname "$stdio_binary")" && pwd -P)
if [[ "$stdio_binary_parent/$(basename "$stdio_binary")" != "$stdio_binary" ]]; then exit 2; fi
stdio_uid=$(id -u)

owned_metadata() {
  if [[ $(uname -s) == Darwin ]]; then
    stat -f '%u %Lp' "$1"
  else
    stat -c '%u %a' "$1"
  fi
}

read -r stdio_owner stdio_mode <<< "$(owned_metadata "$stdio_binary")"
if [[ "$stdio_owner" != "$stdio_uid" || $((8#$stdio_mode & 022)) -ne 0 ]]; then exit 2; fi
stdio_work="$stdio_repo/edits/work/$stdio_work_name"
stdio_history="$stdio_work/mcp-stdio"
for stdio_directory in "$stdio_repo/edits" "$stdio_repo/edits/work" "$stdio_work" "$stdio_history"; do
  if [[ -L "$stdio_directory" || ( -e "$stdio_directory" && ! -d "$stdio_directory" ) ]]; then
    exit 2
  fi
  if [[ ! -e "$stdio_directory" ]]; then
    # Another metadata-only launch may create the same parent concurrently.
    mkdir "$stdio_directory" || [[ -d "$stdio_directory" && ! -L "$stdio_directory" ]]
  fi
  if [[ $(cd "$stdio_directory" && pwd -P) != "$stdio_directory" ]]; then exit 2; fi
  read -r stdio_owner stdio_mode <<< "$(owned_metadata "$stdio_directory")"
  if [[ "$stdio_owner" != "$stdio_uid" || $((8#$stdio_mode & 022)) -ne 0 ]]; then exit 2; fi
done
if [[ "$stdio_mode" != 700 ]]; then exit 2; fi

# The explicit template confines mktemp itself; no OS-default scratch is used.
stdio_launch=$(mktemp -d "$stdio_history/launch.XXXXXXXX")
if [[ -L "$stdio_launch" || $(cd "$stdio_launch" && pwd -P) != "$stdio_launch" ]]; then exit 2; fi
set -o noclobber
exec 2>"$stdio_launch/stderr.log"
printf 'pid=%s\nbinary=%s\n' "$$" "$stdio_binary" >"$stdio_launch/launch.properties"
shasum -a 256 "$stdio_binary" >"$stdio_launch/binary.sha256"
mkdir "$stdio_launch/tmp" "$stdio_launch/cache" "$stdio_launch/module-cache" "$stdio_launch/profiles"
export TMPDIR="$stdio_launch/tmp" TMP="$stdio_launch/tmp" TEMP="$stdio_launch/tmp"
export XDG_CACHE_HOME="$stdio_launch/cache"
export CLANG_MODULE_CACHE_PATH="$stdio_launch/module-cache"
export SWIFTPM_MODULECACHE_OVERRIDE="$stdio_launch/module-cache"
export LLVM_PROFILE_FILE="$stdio_launch/profiles/%p-%m.profraw"
cd "$stdio_repo"
exec "$stdio_binary" "$@"
