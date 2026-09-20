# Layout and security boundaries

## Version-controlled

- `src/replay/`: CLI, runtime creator resolution, authenticated notice/replay API, session handling, finite HLS download and stable-ID jobs.
- `tests/`: deterministic mocked-network tests and local FFmpeg integration tests.
- `AGENTS.md`: portable coding-agent instructions, safety constraints and verification expectations.
- `docs/`: generic architecture, development and operations guides. Never include live account evidence or signed URLs.
- `.python-version`, `pyproject.toml`, `uv.lock`, `Makefile`: fixed Python patch version, locked Python dependencies and quality checks.
- `scripts/bootstrap.sh`: explicit Mac provisioning, without authentication or state migration.
- `scripts/check-public.sh`, `.gitleaks.toml`: positive publication-path policy plus local index/history secret scanning.
- `.github/workflows/ci.yml`: credential-free macOS verification with SHA-pinned actions.
- `SECURITY.md`: publication, credential and migration limitations. Build archives have an explicit source include list, independent of private runtime directories.

## Local only (ignored)

- `private/tiktok-sessionid.secret`: credential, mode 0600.
- `private/*.metadata.json`: fingerprint-bound expiry and pinned account identity.
- `private/jobs/`: persistent completion records and process locks. Preserve these when reorganizing files.
- `private/candidates/`: observed, scoped media URLs; these may contain signatures.
- `private/notes/`: investigation plans, account-specific verification reports and scratch notes.
- `private/legacy-investigation/` (optional): archived investigation materials and device state, not normal runtime dependencies. Historical paths inside these files need review before reusing old tools.
- Other `private/` contents: browser state, local runbooks, private configuration and diagnostic assets. Do not publish them.
- `downloads/`: completed media and local diagnostics. Existing media stays in place because job records refer to its paths.

Normal CLI commands do not migrate existing media or job state. When changing checkout locations, securely copy and verify the data before re-registering completed output paths; see [operations.md](operations.md). Local storage layout may include optional archives, but current runtime must not depend on a previous checkout.

## Network boundaries

1. Runtime `--user` resolution fetches only a canonical HTTPS TikTok profile with a new unauthenticated client, bounded to 2 MiB. It validates the username in the hydration object before accepting a numeric user ID. If HTTP omits identity state, an optional isolated headless Chrome loads the ordinary public page, blocking media and off-host document navigation. It does not receive account credentials, reuse browser profiles, automate macOS input or solve challenges. Missing/ambiguous identities and changed usernames fail explicitly; there is no identity guess based on display names.
2. Authenticated notice and replay requests use fixed HTTPS endpoints, bounded bodies and no redirects. Notifications are read with `is_mark_read=0`. Creator selection changes the exact anchor-ID filter, not account permissions.
3. Media requests use a separate client without API authentication. Hosts and redirects must remain in the explicit media scope. Automatic filenames use validated API `start_time` in JST plus the stable ID; metadata resolution and filename selection occur inside the ID lock, after the completed/busy check.
4. Session reissuance uses the observed Web SDK account-info request. Server-issued cookies, account continuity and notification access are checked before publishing a credential. Metadata is saved first and bound to a token fingerprint, so failed token publication cannot apply a different token's expiry to the previous credential.
5. Explicit web reauthentication uses a fresh browser context. It does not reuse the user's browser profile or automate macOS input. Third-party identity-provider state is not persisted.

## Media limits

Only completed, finite, unencrypted HLS with muxed audio/video is supported, including ordinary masters and fMP4 initialization. DRM, encrypted HLS, byte ranges, separate audio renditions, unfinished live streams and DASH are rejected. No replay ID guessing, signature modification, TLS verification bypass, posting or publishing is implemented.

## Supplemental tools

`list`, `download`, `import-har`, `import-android-log` operate on explicitly observed candidates. `capture` uses the optional mitmproxy dependency; `web-login` saves scoped browser state but does not by itself verify replay access. These are not dependencies of normal authenticated listing/download or HTTP cookie reissuance.
