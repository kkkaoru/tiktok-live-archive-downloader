#!/usr/bin/env bash
# Provision the locked CLI environment without copying credentials or logging in.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dev=false
browser=true
for argument in "$@"; do
    case "$argument" in
        --dev) dev=true ;;
        --no-browser) browser=false ;;
        --help)
            printf '%s\n' 'Usage: bash scripts/bootstrap.sh [--dev] [--no-browser]'
            exit 0
            ;;
        *) printf '%s\n' 'Unknown bootstrap option.' >&2; exit 2 ;;
    esac
done

if [[ "$(uname -s)" != Darwin ]]; then
    printf '%s\n' 'This bootstrap script requires macOS.' >&2
    exit 1
fi
if ! command -v brew >/dev/null 2>&1; then
    printf '%s\n' 'Install Homebrew from https://brew.sh and enable brew in PATH, then retry.' >&2
    exit 1
fi

cd "$root"
brew install uv ffmpeg
if "$browser"; then
    brew install --cask google-chrome
fi
if "$dev"; then
    brew install gitleaks
fi
IFS= read -r python_version < .python-version
uv python install "$python_version"
options=(--locked --extra web)
if "$dev"; then
    options+=(--extra capture)
fi
uv sync "${options[@]}"
umask 077
mkdir -p private downloads
chmod 700 private downloads
uv run --locked replay --help
printf '%s\n' 'Environment ready. Configure TIKTOK_SESSIONID privately or explicitly use refresh-session --login.'
printf '%s\n' 'No authentication, media transfer, or changes to existing job records were performed.'
