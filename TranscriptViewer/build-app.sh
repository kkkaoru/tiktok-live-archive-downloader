#!/usr/bin/env bash
set -euo pipefail
umask 077
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
package="$root/TranscriptViewer"
work="$root/edits/work/transcript-viewer-build"
# The native runner owns a different package. Keep this existing installer as
# the Viewer entry point, with every SwiftPM/cache destination explicit.
for directory in "$package" "$package/.build" "$package/dist" \
    "$root/edits" "$root/edits/work" "$work" \
    "$work"/{tmp,cache,config,security,module-cache,profiles}; do
    if [[ -L "$directory" || ( -e "$directory" && ! -d "$directory" ) ]]; then
        printf 'Viewer working storage must be a real repository directory.\n' >&2
        exit 1
    fi
    mkdir -p "$directory"
    [[ "$(cd "$directory" && pwd -P)" == "$directory" ]] || {
        printf 'Viewer working storage escaped the checkout.\n' >&2; exit 1;
    }
done
export TMPDIR="$work/tmp" TMP="$work/tmp" TEMP="$work/tmp"
export XDG_CACHE_HOME="$work/cache"
export CLANG_MODULE_CACHE_PATH="$work/module-cache"
export SWIFTPM_MODULECACHE_OVERRIDE="$work/module-cache"
export LLVM_PROFILE_FILE="$work/profiles/%p-%m.profraw"
cd "$package"
lock="$PWD/dist/.TranscriptViewer.install.lock"
mkdir "$lock" || { printf 'Another app installation is active; inspect the lock.\n' >&2; exit 1; }
cleanup() {
    local exit_status=$?
    rmdir "$lock" || exit_status=1
    exit "$exit_status"
}
trap cleanup EXIT

build_options=(--package-path "$package" --scratch-path "$package/.build"
    --cache-path "$work/cache" --config-path "$work/config" --security-path "$work/security"
    --manifest-cache local --disable-automatic-resolution -c release -Xswiftc -warnings-as-errors)
swift build "${build_options[@]}" --product TranscriptViewer
bin="$(swift build "${build_options[@]}" --show-bin-path)/TranscriptViewer"
app="$PWD/dist/TranscriptViewer.app"
executable="$app/Contents/MacOS/TranscriptViewer"
[[ ! -L "$app" && ( ! -e "$app" || -d "$app" ) ]] || {
    printf 'App destination is not a regular bundle.\n' >&2; exit 1;
}
if [[ ! -f App/AppIcon.icns ]]; then
    bash App/generate-icon.sh >/dev/null
fi
stage="$(mktemp -d "$PWD/dist/.TranscriptViewer.stage.XXXXXX")"
staged_app="$stage/TranscriptViewer.app"
mkdir -p "$staged_app/Contents/MacOS" "$staged_app/Contents/Resources"
cp "$bin" "$staged_app/Contents/MacOS/TranscriptViewer"
cp App/Info.plist "$staged_app/Contents/Info.plist"
cp App/AppIcon.icns "$staged_app/Contents/Resources/AppIcon.icns"
chmod +x "$staged_app/Contents/MacOS/TranscriptViewer"
codesign --sign - "$staged_app" >/dev/null
codesign --verify --deep --strict "$staged_app"

# Stop only the process executing this checkout's old bundle, after the new one is signed.
pids=""
processes="$(ps -axo pid=,comm=)"
while read -r pid command; do
    if [[ "$command" == "$executable" && "$pid" =~ ^[0-9]+$ ]]; then pids+=" $pid"; fi
done <<< "$processes"
for pid in $pids; do
    if [[ "$(ps -p "$pid" -o comm=)" == "$executable" ]]; then kill -TERM "$pid"; fi
done
still_running=0
if [[ -n "$pids" ]]; then
    for ((attempt=0; attempt<50; attempt++)); do
        still_running=0
        for pid in $pids; do
            if [[ "$(ps -p "$pid" -o comm=)" == "$executable" ]]; then still_running=1; fi
        done
        if (( ! still_running )); then break; fi
        sleep 0.2
    done
fi
if (( still_running )); then
    printf 'Old app did not exit; signed staging bundle retained, original untouched.\n' >&2
    exit 1
fi

obsolete="$stage/obsolete.app"
if [[ -e "$app" ]]; then mv "$app" "$obsolete"; fi
[[ ! -e "$app" && ! -L "$app" ]] || {
    printf 'App destination appeared during installation; previous app and stage retained.\n' >&2
    exit 1
}
mv -n "$staged_app" "$app"
[[ ! -e "$staged_app" ]] || {
    printf 'App destination changed during installation; previous app and stage retained.\n' >&2
    exit 1
}
codesign --verify --deep --strict "$app"
if [[ -n "$pids" ]]; then
    open -g -a "$app"
    running=0
    for ((attempt=0; attempt<50; attempt++)); do
        processes="$(ps -axo pid=,comm=)"
        while read -r pid command; do
            if [[ "$command" == "$executable" ]]; then running=1; fi
        done <<< "$processes"
        if (( running )); then break; fi
        sleep 0.2
    done
    if (( ! running )); then
        printf 'New app was installed but did not start; previous bundle retained in staging.\n' >&2
        exit 1
    fi
fi
if [[ -d "$obsolete" ]]; then rm -r "$obsolete"; fi
rmdir "$stage"
printf '%s\n' "$app"
