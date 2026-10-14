#!/usr/bin/env bash
# Preserve the best available source streams on an external volume without re-encoding.
set -euo pipefail
umask 077
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# != 1 )); then
    printf '%s\n' 'Usage: bash scripts/download-youtube.sh YOUTUBE_URL' >&2
    exit 2
fi
url="$1"
if [[ "$url" =~ ^https://(www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{11})$ ]]; then
    url="https://www.youtube.com/watch?v=${BASH_REMATCH[2]}"
elif [[ ! "$url" =~ ^https://(www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]{11}$ ]]; then
    printf '%s\n' 'Expected a canonical YouTube watch or Shorts URL.' >&2
    exit 2
fi
if ! command -v node >/dev/null 2>&1; then
    printf '%s\n' 'Node.js 22+ is required for complete YouTube format extraction.' >&2
    exit 1
fi
ffmpeg_dir="$(brew --prefix ffmpeg-full 2>/dev/null)/bin"
if [[ ! -x "$ffmpeg_dir/ffmpeg" ]]; then
    printf '%s\n' 'Install ffmpeg-full for safe lossless remuxing and ASS subtitles.' >&2
    exit 1
fi
output="${YOUTUBE_DOWNLOAD_DIR:-/Volumes/m.2-ssd/movie/youtube}"
if [[ "$output" != /* || -L "$output" || ! -d "$(dirname "$output")" ]]; then
    printf '%s\n' 'Set YOUTUBE_DOWNLOAD_DIR under a mounted external volume.' >&2
    exit 1
fi
if [[ ! -e "$output" ]]; then
    mkdir -m 700 "$output"
fi
if [[ ! -d "$output" ]] || (( (8#$(stat -f '%Lp' "$output") & 077) != 0 )); then
    printf '%s\n' 'External download directory must be private and must not be a symlink.' >&2
    exit 1
fi
output="$(cd "$output" && pwd -P)"
repo_root="$(pwd -P)"
case "$output" in
    "$repo_root"/edits/work/*)
        exec uv run --locked --extra youtube python -m replay.youtube_delivery "$url" "$output" "$ffmpeg_dir"
        ;;
esac
if [[ "$(stat -f '%d' "$(dirname "$output")")" == "$(stat -f '%d' .)" ]]; then
    printf '%s\n' 'Set YOUTUBE_DOWNLOAD_DIR under a mounted external volume or edits/work/.' >&2
    exit 1
fi
if [[ -L youtube-originals ]]; then
    if [[ ! -d youtube-originals ]] || [[ "$(cd youtube-originals && pwd -P)" != "$output" ]]; then
        printf '%s\n' 'youtube-originals points elsewhere; migrate existing videos manually.' >&2
        exit 1
    fi
else
    if [[ -e youtube-originals ]] && ! rmdir youtube-originals 2>/dev/null; then
        printf '%s\n' 'youtube-originals contains data; migrate it manually.' >&2
        exit 1
    fi
    ln -s "$output" youtube-originals
fi
exec uv run --locked --extra youtube python -m replay.youtube_delivery "$url" "$output" "$ffmpeg_dir"
