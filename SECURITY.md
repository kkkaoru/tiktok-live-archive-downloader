# Security and publication policy

## Never publish

Do not commit or attach session tokens, cookies/browser state, HAR/packet captures, signed media URLs, device identifiers, screenshots, downloaded recordings, job metadata or personal investigation notes. Do not submit them in public GitHub issues, Actions logs or artifacts. A public repository does not imply permission to redistribute recordings.

`.gitignore` is a convenience, not a security boundary: it does not remove tracked files or past commits. `scripts/check-public.sh` checks a positive path list, rejects tracked symlinks/submodules, exports only the Git index and scans that export plus every referenced commit with local Gitleaks. The custom rules cover TikTok sessions, with exceptions limited to repeated-character test fixtures inside `tests/`.

Before publishing changes:

1. Review source, tests, documentation and dependency changes, then stage only intended files.
2. Run `make check` and `make public-check` with Gitleaks installed.
3. Review the staged diff and commit author/committer metadata. Prefer the GitHub-provided noreply email; do not assume commit signatures conceal email addresses.
4. Confirm the target repository owner/visibility and that no runtime artifacts are tracked.

The scan covers the index and history, not unrelated untracked working files. It is not proof that every possible secret or identifying detail has been removed. Avoid CI secrets for this project: tests use synthetic authentication and mocked network responses.

## Runtime boundaries

- API authentication comes from `TIKTOK_SESSIONID`, validated before use. Fixed HTTPS endpoints, TLS verification and redirect refusal bound authenticated API traffic.
- Creator lookup uses a separate unauthenticated HTTP client, with an optional fresh headless browser. It never imports existing browser profiles or passes the account session to profile lookup.
- Media transfer uses a separate client and explicit HTTPS host scope. Credentials do not follow cross-origin media redirects. Capture helpers are explicitly opt-in.
- Credential values and signed candidate URLs are hidden from normal CLI output and object representations. Errors from browser/network tools are sanitized. Do not enable third-party HTTP debug logging while using real credentials.
- Session reissuance checks account continuity and API access before atomic secret-file publication. Private storage uses 0700 directories and 0600 files. Cookie reissuance is not a promise of expiry extension or unattended recovery after revocation.
- Browser authentication is explicit. Human security checks are not bypassed, and third-party identity-provider storage is not retained.

## Moving to another Mac

The repository contains software, not an authenticated account or footage. Provision dependencies with `scripts/bootstrap.sh`; then configure a valid authorized session privately. If transferring credentials, use an appropriate secret manager or a secure transfer, not Git or public cloud artifacts. Keep the private directory outside untrusted synchronization.

Existing job records may contain absolute paths from the original Mac. Copying those records to a different path must not be treated as proof that media exists. The downloader rejects missing/changed completed files instead of silently redownloading them. Verify copied media and explicitly re-register its new path only after validation; do not delete job records to suppress the warning. Bootstrap does not rewrite or delete existing state.

## If a secret was exposed

Revoke or replace the credential first. Deleting the file or rewriting history alone does not invalidate a leaked token or guarantee removal from forks/caches. Coordinate any history rewrite before pushing it. For a security report, first contact the maintainer without including credentials; use a private reporting channel if one is available.
