# Development and verification

## Reproduce the environment

Run commands from the checkout root. Install Homebrew yourself using its official instructions, then:

```sh
bash scripts/bootstrap.sh --dev --no-browser
make check
uv lock --check
uv build
```

`--no-browser` skips installation of Chrome, not the Python web extra. Chrome is not required for the mocked test suite. If system tools are already installed, `uv sync --locked --extra capture --extra web` is sufficient to provision Python dependencies.

The setup script's tests use fake system commands; they do not actually install Homebrew packages. Python and Python dependencies are pinned for the tested environment, but Homebrew tools and the isolated build backend are not a bit-for-bit operating-system lock.

## Optional local subtitle repair

`uv sync --locked --extra editing` installs NumPy and headless OpenCV for classical,
ROI-only subtitle repair kernels. These are also development dependencies so the
synthetic tests run without skips. They do not download AI models or send media
to a service, and normal downloader imports do not load them.

The repair and bounded-retry modules are building blocks, not an integrated
unattended editing CLI. Synthetic tests do not establish visual quality. The
same glyph-repair family is retained on retries; the last variant permits local
boundary softening rather than substituting a rectangular blur band.

## Quality gate

`make check` runs:

1. Shell syntax checks for the bootstrap and publication scripts.
2. Ruff format check and lint.
3. Strict mypy over `src` and `tests`.
4. Pytest with production-code coverage.
5. A minimum 90% statement-coverage check for every production file.

Use `uv run pytest tests/test_jobs.py` or another relevant test module while iterating, then run the full gate. Do not reduce gates, suppress errors, skip tests, or use real account data to make fixtures convenient.

Most network behavior is mocked. `test_ffmpeg.py` executes local FFmpeg/ffprobe against generated media. Publication tests use temporary Git repositories and the real local Gitleaks binary. No test needs an authenticated TikTok account, Android device, live browser login, or account secrets in CI.

## Evidence levels

| Check | What it establishes | What it does not establish |
| --- | --- | --- |
| Unit tests | Behavior on controlled inputs, including failures | Current service availability |
| CLI help | Installation and entry-point loading | Authentication or media access |
| Live `recordings` | Notice enumeration and replay resolution for that account | CDN transfer or full video integrity |
| Bounded media probe | Actual receipt of identifiable video bytes | Complete playback or a full archive |
| Full download and metadata inspection | Completed transfer/remux and expected tracks/tags | Successful decode of every frame |
| Full decode / native playback check | The specific decoding or playback check performed | Future URL validity or all possible players |

Only perform live checks when authorized. Record outcome summaries without credentials, signed URLs, private account details, or local personal paths. Keep account-specific evidence outside Git. A credential or browser state file is not by itself proof of replay permission.

## Safe bounded live probe

There is **no dedicated probe CLI command**. Do not describe `download-all` as a dry run: it can download every eligible recording.

For an authorized transfer-start check, resolve a replay from actual notifications and use the existing scoped downloader in a new private temporary directory. Use a small byte budget and one worker, inspect a received sample with local media tools, and clean up on both success and failure. Do not register a completed job for a partial sample or alter existing jobs to force reacquisition. A budget error is expected only if the configured limit was reached; it is not evidence of success unless video bytes were actually received and identified.

The media-byte budget does not include API bodies or playlist fetches, which have their own bounds. Downloader cleanup removes partial files, so inspect the sample before cleanup without logging the request URL. Confirm afterward that the temporary directory is gone and existing media/job records are unchanged. Do not publish the sample or its manifest.

## Publication and packaging

```sh
# First review and stage only the files intended for publication.
make public-check
uv build
```

The publication check exports the Git index to a temporary directory and runs Gitleaks on that export and all referenced history. Untracked/unstaged changes are not implicitly audited. It also rejects unapproved public paths and tracked symlinks/submodules. Commit email identities and human-readable content still require separate review.

The sdist explicitly includes public source, tests, scripts, and documentation. The wheel contains the runtime package. Check archive members when changing packaging; test with synthetic runtime files to ensure they are excluded, not with real secrets. Never upload `private/`, `downloads/`, captures, or test media as Actions artifacts.

CI runs the same checks on macOS with read-only repository permission, SHA-pinned actions, and no account credentials. A local green test run and a hosted CI run are separate results.
