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
        src/replay/*.py|src/replay/py.typed|tests/test_*.py|docs/*.md|scripts/*.sh|.github/workflows/*.yml) ;;
        '') ;;
        *) printf 'Unapproved public path: %s\n' "$1" >&2; return 1 ;;
    esac
}

temporary="$(mktemp -d)"
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
gitleaks dir "$temporary/index" --config "$root/.gitleaks.toml" --redact --no-banner
gitleaks git "$root" --config "$root/.gitleaks.toml" --log-opts=--all --redact --no-banner
printf '%s\n' 'Index and complete referenced history passed public-source checks.'
printf '%s\n' 'Review commit author/email metadata separately; scanning is not a guarantee.'
