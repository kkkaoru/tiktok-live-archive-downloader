#!/usr/bin/env bash
# Scan only publishable Git content, never local recordings or credential stores.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
cd "$root"

if ! command -v gitleaks >/dev/null 2>&1; then
    printf '%s\n' 'Install the local scanner: brew install gitleaks' >&2
    exit 1
fi

check_path() {
    case "$1" in
        .gitignore|.gitleaks.toml|.python-version|Makefile|pyproject.toml|uv.lock|README.md|SECURITY.md|AGENTS.md) ;;
        .agents/skills/*/SKILL.md) ;;
        tests/stateful_guard_fixture.py) ;;
        src/replay/*.py|src/replay/py.typed|tests/test_*.py|docs/*.md|scripts/*.sh|.github/workflows/*.yml) ;;
        TranscriptViewer/Package.swift|TranscriptViewer/Package.resolved|TranscriptViewer/.swift-format|TranscriptViewer/build-app.sh) ;;
        TranscriptViewer/Sources/TranscriptCore/*.swift|TranscriptViewer/Sources/TranscriptViewer/*.swift|TranscriptViewer/Sources/TranscriptCheck/*.swift|TranscriptViewer/Sources/TranscriptSummaries/*.swift|TranscriptViewer/Sources/TranscriptSummaryCLI/*.swift) ;;
        TranscriptViewer/Tests/TranscriptCoreTests/*.swift) ;;
        TranscriptViewer/App/Info.plist|TranscriptViewer/App/generate-icon.py|TranscriptViewer/App/generate-icon.sh) ;;
        native/apple-pro-apps/Package.swift|native/apple-pro-apps/Package.resolved|native/apple-pro-apps/README.md) ;;
        native/apple-pro-apps/Sources/ProAppsCore/*.swift|native/apple-pro-apps/Sources/AppleProApps/*.swift) ;;
        native/apple-pro-apps/Tests/ProAppsCoreTests/*.swift|native/apple-pro-apps/Tests/ProAppsNativeTests/*.swift|native/apple-pro-apps/Tests/AppleProAppsTests/*.swift|native/apple-pro-apps/Tests/UIFixture/main.swift) ;;
        native/apple-pro-apps/Tests/ProAppsNativeTests/Fixtures/README.md|native/apple-pro-apps/Tests/ProAppsNativeTests/Fixtures/black.mp4|native/apple-pro-apps/Tests/ProAppsNativeTests/Fixtures/quadrants-30.mp4) ;;
        web-viewer/.gitignore|web-viewer/README.md|web-viewer/biome.json|web-viewer/bun.lock|web-viewer/package.json|web-viewer/tsconfig.json|web-viewer/vitest.config.ts) ;;
        web-viewer/cloudflare.config.ts|web-viewer/wrangler.config.ts|web-viewer/sync.config.example.json|web-viewer/public/index.html) ;;
        web-viewer/secrets.example.json|web-viewer/deploy.env.example) ;;
        web-viewer/src/*.js|web-viewer/src/*.ts|web-viewer/tools/*.py) ;;
        '') ;;
        *) printf 'Unapproved public path: %s\n' "$1" >&2; return 1 ;;
    esac
}

umask 077
work="$root/edits/work/public-check"
for directory in "$root/edits" "$root/edits/work" "$work"; do
    if [ -L "$directory" ]; then
        printf '%s\n' 'Publication working storage must not be a symlink.' >&2
        exit 1
    fi
    mkdir -p "$directory"
    test "$(cd "$directory" && pwd -P)" = "$directory"
done
temporary="$(mktemp -d "$work/index.XXXXXXXX")"
trap 'rm -rf "$temporary"' EXIT
# Collect first so Git failures cannot be hidden by process substitution.
git ls-files > "$temporary/index-paths"
git log --all --format= --name-only > "$temporary/history-paths"
git ls-files --stage > "$temporary/modes"
# Positive path policy complements .gitignore, which is not a security boundary.
while IFS= read -r path; do
    check_path "$path"
done < "$temporary/index-paths"
while IFS= read -r path; do
    check_path "$path"
done < "$temporary/history-paths"
if grep -Eq '^(120000|160000) ' "$temporary/modes"; then
    printf '%s\n' 'Public symlinks and submodules require explicit security review.' >&2
    exit 1
fi

# Export the index: this includes staged changes but cannot copy ignored private data.
mkdir "$temporary/index"
git checkout-index --all --prefix="$temporary/index/"
# Only these byte-verified synthetic videos may use the narrow media exceptions.
verify_fixture() {
    local file="$temporary/index/native/apple-pro-apps/Tests/ProAppsNativeTests/Fixtures/$1"
    local actual
    if [ -f "$file" ]; then
        actual="$(shasum -a 256 "$file")"
        if [ "${actual%% *}" != "$2" ]; then
            printf 'Synthetic fixture hash differs: %s\n' "$1" >&2
            return 1
        fi
    fi
}
verify_fixture black.mp4 7db0b4ade347132cf1ba46942efe01fc2723dfaf19618ad88cd718af5fe02398
verify_fixture quadrants-30.mp4 208c378209f9c8e0e96d9e4dda602f1dffdb03f0c4f01233047a7ec8556754c8
gitleaks dir "$temporary/index" --config "$root/.gitleaks.toml" --redact --no-banner
gitleaks git "$root" --config "$root/.gitleaks.toml" --log-opts=--all --redact --no-banner
printf '%s\n' 'Index and complete referenced history passed public-source checks.'
printf '%s\n' 'Review commit author/email metadata separately; scanning is not a guarantee.'
