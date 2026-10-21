# Secrets and environment

What stays out of Git, what belongs in Git, and where each value lives. `make public-check`
scans the index and referenced history (gitleaks) and only approves paths on its allowlist;
it is a safety net, not a substitute for keeping a secret out of the tree.

## Never in Git

| Value | Where it lives | Notes |
| --- | --- | --- |
| TikTok session (`TIKTOK_SESSIONID`) | `private/tiktok-sessionid.secret` (0600) | Loaded only by `scripts/replay-authorized.sh`; never in argv, chat, logs or `.env`. |
| Creator anchor ID (`TIKTOK_TARGET_ANCHOR_ID`) | `private/target-anchor-id` (0600) | Treated as a credential. |
| Recordings, transcripts, summaries | `downloads/`, `youtube-originals/`, `edits/`, `*.summaries/` | Personal data. Summaries and diagnostics must not be published. |
| Web viewer Worker secrets | `web-viewer/secrets.json` (0600) | `X_CLIENT_ID`, `X_CLIENT_SECRET`, `SESSION_SECRET`, `X_ADMIN_USERNAMES`, `TURNSTILE_SITE_KEY` / `TURNSTILE_SECRET_KEY` (share-link bot check), `CLIPS_API_TOKEN` (32+ chars; the Mac TranscriptViewer reads it to fetch and reload clip points, sent only to the first `VIEWER_DOMAIN`), optional `DEBUG_TOKEN` / `DEBUG_TOKEN_EXPIRES`. Template: `web-viewer/secrets.example.json`. |
| Legacy viewer login | `web-viewer/credentials.txt` | Basic-auth fallback from before X sign-in; delete when no longer needed. |
| Debug session cookie | `web-viewer/.debug-cookies` | One hour, read-only; recreated on demand. |
| Cloudflare CLI state | `.cloudflare/` (includes the account ID), `web-viewer/.cloudflare/` | CLI authentication itself is stored outside the checkout. |
| Viewer data and sync state | `web-viewer/data/`, `web-viewer/proxies/`, `web-viewer/sync-state.json` | Derived from recordings (transcripts, local paths). |
| Generic secret files | `.env`, `.env.*`, `*.secret`, `*.key`, `*.pem`, `*.p12`, `*.p8`, `.dev.vars`, `.netrc`, `.npmrc`, `secrets.json`, `credentials*.txt`, cookies, HAR | Covered by `.gitignore`. |

## Kept out of Git, but not secret

Deployment-specific settings that differ per machine: `web-viewer/deploy.env`
(`VIEWER_DOMAIN`, template `deploy.env.example`) and `web-viewer/sync.config.json`
(library roots, bucket, streamer; template `sync.config.example.json`). They hold local
paths and a personal domain, so they are ignored and shipped only as templates.

## In Git

Source, tests, lockfiles (`uv.lock`, `bun.lock`, `Package.resolved`), documentation,
Worker and Cloudflare configuration that only *names* environment variables, and the
`*.example.*` templates above. Templates carry names and empty or placeholder values only.

## Environment variables

| Variable | Used by | Source |
| --- | --- | --- |
| `TIKTOK_SESSIONID`, `TIKTOK_TARGET_ANCHOR_ID` | downloader CLI | Set in the child process by `scripts/replay-authorized.sh` from `private/`. |
| `TIKTOK_TARGET_USER`, `TIKTOK_DOWNLOAD_DIR` | downloader CLI / script | Optional; a download directory must be an existing private absolute directory. |
| `YOUTUBE_DOWNLOAD_DIR` | `scripts/download-youtube.sh` | Optional output directory. |
| `REPLAY_NATIVE_WORK_NAME`, `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`, `TRANSFORMERS_OFFLINE` | native build and ASR tooling | Non-secret switches. |
| `VIEWER_DOMAIN` | `web-viewer/cloudflare.config.ts` | `web-viewer/deploy.env`, read by `bun run deploy`. |
| `X_CLIENT_ID`, `X_CLIENT_SECRET`, `SESSION_SECRET`, `X_ADMIN_USERNAMES`, `DEBUG_TOKEN`, `DEBUG_TOKEN_EXPIRES`, `BASIC_USER`, `BASIC_PASS`, `TURNSTILE_SITE_KEY`, `TURNSTILE_SECRET_KEY`, `CLIPS_API_TOKEN` | viewer Worker | Uploaded as Worker secrets from `web-viewer/secrets.json`. |
| `UPLOAD_TOKEN` | temporary uploader Worker | Random per sync run; never stored. |

The downloader never reads a `.env` file automatically; do not add automatic loading.

## Known personal values still in tracked code

`/Volumes/m.2-ssd/movie` appears as a default in `scripts/download-youtube.sh` and as
`MEDIA_ROOT` in `src/replay/compressed_part_retirement.py`. It is a local volume name, not
a secret, but it ties the code to one machine; move it to an environment variable or
configuration when those files are next edited.

## Adding a new secret

1. Choose its home (a `private/` file for the downloader, `web-viewer/secrets.json` for
   the viewer) before writing any code that reads it.
2. Add its *name* to the matching template and to the tables above.
3. Confirm `git check-ignore -v <path>` reports the real file as ignored and the template
   as trackable, then run `make public-check`.
