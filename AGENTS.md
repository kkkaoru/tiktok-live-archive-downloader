# Agent working guide

Read [README.md](README.md), [SECURITY.md](SECURITY.md), and [docs/architecture.md](docs/architecture.md) before changing behavior. Use [docs/development.md](docs/development.md) for verification and [docs/operations.md](docs/operations.md) for operational troubleshooting.

## Scope and safety

- This is a macOS CLI for authorized, observed, finished TikTok LIVE replays. It is not a screen recorder or an access-control bypass.
- Work from the current checkout. Do not assume another checkout, a previous temporary directory, or a particular user's home exists.
- Treat `private/` and `downloads/` as user data, not source. Do not recursively read, index, attach, stage, clean, or delete them during ordinary coding work. They may contain credentials, signed URLs, account identifiers, footage, and device state.
- Do not read credential contents into tool output, prompts, logs, examples, or tests. For explicitly authorized live checks, load credentials privately into the child process environment; never interpolate their values into commands. Do not enable shell tracing or HTTP debug logging.
- Preserve media and `private/jobs/`. Never delete completion records or bypass deduplication just to make a download run. A missing or changed completed output is a stop condition, not permission to redownload.
- Do not launch login windows, reuse existing browser profiles, manipulate desktop input/focus, or operate connected devices as part of routine tests. Explicit login requires user authorization. Do not solve or bypass security challenges.
- Never weaken TLS checks, authenticated endpoint restrictions, media host scope, byte limits, or account-continuity checks to make a failing request pass. Do not guess room IDs or alter signed URLs.

## Where to make changes

| Area | Implementation | Tests |
| --- | --- | --- |
| CLI and options | `src/replay/cli.py` | `tests/test_cli.py` |
| Creator selection | `targets.py` | `test_targets.py` |
| Notice parsing and API access | `recordings.py`, `tiktok_api.py` | corresponding `test_*.py` |
| Session reissuance and explicit login | `session_refresh.py`, `web_auth.py` | corresponding `test_*.py` |
| Acquisition and completion records | `acquire.py`, `jobs.py` | corresponding `test_*.py` |
| Scoped storage and media transfer | `core.py`, `download.py`, `media.py` | corresponding tests and `test_ffmpeg.py` |
| Optional capture/import | `addon.py`, `api.py`, `android_log.py` | corresponding `test_*.py` |
| Setup and publication policy | `scripts/`, `.gitleaks.toml`, `pyproject.toml` | `test_bootstrap.py`, `test_publication.py` |

Unless fully qualified, implementation filenames above are under `src/replay/` and tests under `tests/`.

## Implementation requirements

- Use the configured Python patch version and `uv.lock`; do not introduce another environment manager. Install development dependencies with `uv sync --locked --extra capture --extra web`.
- Keep public APIs typed and run strict mypy. Validate external JSON at boundaries. Avoid unchecked `Any`, casts, mutable defaults, and broad exception handling that hides failures.
- Keep HTTP API authentication separate from profile lookup and CDN transfer. Normal listing, acquisition, and HTTP session reissuance must not acquire an Android dependency.
- Preserve atomic, private credential writes; fingerprint-bound metadata; stable-ID process locks; non-overwriting output publication; and QuickTime-compatible tagging without unnecessary re-encoding.
- Automatic output names must use validated API broadcast `start_time` in JST plus the stable ID. Do not infer dates from file metadata, notification times, or IDs. Explicitly authorized renames must update completed jobs without redownloading or overwriting another file.
- Add focused regression tests with synthetic inputs and mocked network responses for behavior changes. Use real local FFmpeg fixtures where appropriate, never private recordings as test fixtures.
- Run the full format/lint/type/test gate. Do not add skips, suppressions, disable hooks/signing, or lower coverage requirements to get a pass.

## Verification and handoff

```sh
make check
uv lock --check
uv build
```

`make check` requires FFmpeg/ffprobe and Gitleaks as well as Python development dependencies. It enforces at least 90% statement coverage in each production file, not just the aggregate.

Before publication, review and stage only the intended public paths, then run `make public-check`. It scans the index and referenced history, not arbitrary unstaged edits. New public paths may need a narrowly scoped addition to `scripts/check-public.sh` and the sdist include list. Never replace these lists with a broad exclusion of all tests or an unrestricted include.

Respect the user's Git scope: inspect existing changes first, do not absorb unrelated edits, and do not commit, push, rewrite history, or delete backups without authorization. Preserve configured hooks and signing. Review author/committer email metadata separately from secret scanning.

In the handoff, distinguish unit tests, local FFmpeg tests, live API listing, partial media receipt, complete downloads, and full decode. A mocked login test does not demonstrate real login; a reissued cookie does not demonstrate lifetime extension; a short download probe does not demonstrate a full archive. Report limitations and whether test media was removed.
