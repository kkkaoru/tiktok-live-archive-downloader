# Layout and security boundaries

## Version-controlled

- `src/replay/`: CLI, runtime creator resolution, authenticated notice/replay API, session handling, finite HLS download and stable-ID jobs.
- `tests/`: deterministic mocked-network tests and local FFmpeg integration tests.
- `docs/`: generic technical documentation. Never include live account evidence or signed URLs.
- `pyproject.toml`, `uv.lock`, `Makefile`: reproducible dependencies and quality checks.

## Local only (ignored)

- `private/tiktok-sessionid.secret`: credential, mode 0600.
- `private/*.metadata.json`: fingerprint-bound expiry and pinned account identity.
- `private/jobs/`: persistent completion records and process locks. Preserve these when reorganizing files.
- `private/candidates/`: observed, scoped media URLs; these may contain signatures.
- `private/notes/`: investigation plans, account-specific verification reports and scratch notes.
- Other `private/` contents: browser state, device tools, captures and diagnostic assets retained locally for reproducibility.
- `downloads/`: completed media and local diagnostics. Existing media stays in place because job records refer to its paths.

The repository intentionally does not move existing media or job state into a new hierarchy. Local investigation notes previously at the repository root have been moved to `private/notes/`.

## Network boundaries

1. Runtime `--user` resolution fetches only a canonical HTTPS TikTok profile with a new unauthenticated client, bounded to 2 MiB. It validates the username in the hydration object before accepting a numeric user ID. If HTTP omits identity state, an optional isolated headless Chrome loads the ordinary public page, blocking media and off-host document navigation. It does not receive account credentials, reuse browser profiles, automate macOS input or solve challenges. Missing/ambiguous identities and changed usernames fail explicitly; there is no identity guess based on display names.
2. Authenticated notice and replay requests use fixed HTTPS endpoints, bounded bodies and no redirects. Notifications are read with `is_mark_read=0`. Creator selection changes the exact anchor-ID filter, not account permissions.
3. Media requests use a separate client without API authentication. Hosts and redirects must remain in the explicit media scope.
4. Session reissuance uses the observed Web SDK account-info request. Server-issued cookies, account continuity and notification access are checked before publishing a credential. Metadata is saved first and bound to a token fingerprint, so failed token publication cannot apply a different token's expiry to the previous credential.
5. Explicit web reauthentication uses a fresh browser context. It does not reuse the user's browser profile or automate macOS input. Third-party identity-provider state is not persisted.

## Media limits

Only completed, finite, unencrypted HLS with muxed audio/video is supported, including ordinary masters and fMP4 initialization. DRM, encrypted HLS, byte ranges, separate audio renditions, unfinished live streams and DASH are rejected. No replay ID guessing, signature modification, TLS verification bypass, posting or publishing is implemented.

## Supplemental tools

`list`, `download`, `import-har`, `import-android-log` operate on explicitly observed candidates. `capture` uses the optional mitmproxy dependency; `web-login` saves scoped browser state but does not by itself verify replay access. These are not dependencies of normal authenticated listing/download or HTTP cookie reissuance.
