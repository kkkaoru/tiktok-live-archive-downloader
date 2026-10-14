#!/usr/bin/env bash
# Use this checkout's private, verified creator and session without putting either in argv.
set -euo pipefail
umask 077
cd "$(dirname "${BASH_SOURCE[0]}")/.."

case "${1:-}" in
    recordings|download-all) action="$1"; shift ;;
    *) printf '%s\n' 'Usage: bash scripts/replay-authorized.sh recordings|download-all [options]' >&2; exit 2 ;;
esac
for option in "$@"; do
    case "$option" in
        --user|--user=*|--anchor-id|--anchor-id=*|--store|--store=*|--host|--host=*)
            printf '%s\n' 'Use the private creator file and default store with this script.' >&2
            exit 2 ;;
    esac
done
if [[ -n "${TIKTOK_TARGET_USER:-}" ]]; then
    printf '%s\n' 'Unset TIKTOK_TARGET_USER before using the private creator file.' >&2
    exit 2
fi
if [[ ! -d private || -L private ]] || (( (8#$(stat -f '%Lp' private) & 077) != 0 )); then
    printf '%s\n' 'Private directory is missing, linked, or accessible to other users.' >&2
    exit 1
fi
for file in private/tiktok-sessionid.secret private/target-anchor-id; do
    if [[ ! -f "$file" || -L "$file" ]]; then
        printf 'Missing or linked private file: %s\n' "$file" >&2
        exit 1
    fi
    mode="$(stat -f '%Lp' "$file")"
    if (( (8#$mode & 077) != 0 )) || [[ "$(stat -f '%z' "$file")" -gt 33 ]]; then
        printf 'Unsafe private file permissions or size: %s\n' "$file" >&2
        exit 1
    fi
done
TIKTOK_SESSIONID=''
TIKTOK_TARGET_ANCHOR_ID=''
IFS= read -r TIKTOK_SESSIONID < private/tiktok-sessionid.secret || [[ -n "$TIKTOK_SESSIONID" ]]
IFS= read -r TIKTOK_TARGET_ANCHOR_ID < private/target-anchor-id || [[ -n "$TIKTOK_TARGET_ANCHOR_ID" ]]
session_bytes="$(stat -f '%z' private/tiktok-sessionid.secret)"
anchor_bytes="$(stat -f '%z' private/target-anchor-id)"
if [[ ! "$TIKTOK_SESSIONID" =~ ^[[:xdigit:]]{32}$ || ! "$TIKTOK_TARGET_ANCHOR_ID" =~ ^[0-9]{1,32}$ ]] ||
    (( session_bytes < 32 || session_bytes > 33 ||
       anchor_bytes < ${#TIKTOK_TARGET_ANCHOR_ID} || anchor_bytes > ${#TIKTOK_TARGET_ANCHOR_ID} + 1 )); then
    printf '%s\n' 'Invalid private session or creator selector.' >&2
    exit 1
fi
export TIKTOK_SESSIONID TIKTOK_TARGET_ANCHOR_ID
if [[ "$action" == download-all ]]; then
    if [[ ${TIKTOK_DOWNLOAD_DIR+x} ]]; then
        destination="${TIKTOK_DOWNLOAD_DIR}"
        if [[ "$destination" != /* || ! -d "$destination" || -L "$destination" ]] ||
            (( (8#$(stat -f '%Lp' "$destination") & 077) != 0 )); then
            printf '%s\n' 'TIKTOK_DOWNLOAD_DIR must be an existing private absolute directory, not a symlink.' >&2
            exit 1
        fi
        destination="$(cd "$destination" && pwd -P)"
        repo_root="$(pwd -P)"
        case "$destination" in
            "$repo_root"/edits/work/*)
                # Keep authorized acquisition inside this checkout without changing
                # the user's configured downloads symlink to an external volume.
                exec uv run --locked replay --store private/candidates download-all "$destination" "$@"
                ;;
        esac
        if [[ -L downloads ]]; then
            if [[ ! -d downloads ]] || [[ "$(cd downloads && pwd -P)" != "$destination" ]]; then
                printf '%s\n' 'downloads already links to a different directory; migrate data manually.' >&2
                exit 1
            fi
        else
            if [[ -e downloads ]]; then
                if [[ ! -d downloads ]] || ! rmdir downloads 2>/dev/null; then
                    printf '%s\n' 'downloads contains data; migrate it before configuring TIKTOK_DOWNLOAD_DIR.' >&2
                    exit 1
                fi
            fi
            ln -s "$destination" downloads
        fi
    fi
    if [[ -L downloads && ! -d downloads ]]; then
        printf '%s\n' 'downloads target is unavailable; mount the destination before downloading.' >&2
        exit 1
    fi
    exec uv run --locked replay download-all downloads "$@"
fi
exec uv run --locked replay recordings "$@"
