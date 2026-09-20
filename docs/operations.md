# Operations and troubleshooting

See [README.md](../README.md) for setup and CLI examples, and [SECURITY.md](../SECURITY.md) for credential handling. Examples use placeholders, not real accounts or credentials. Resolve paths from the current checkout, not a previous temporary working directory.

## Normal workflow

1. Load a valid authorized session privately into `TIKTOK_SESSIONID` as described in the README. The CLI does not automatically read a `.env` file or a saved token file for ordinary API commands.
2. Select a creator with `--user`, `--anchor-id`, or `TIKTOK_TARGET_USER`. A display name is not a verified username. An explicit CLI selector takes precedence over the environment default.
3. Run `recordings` first. It resolves replay metadata but does not download video. Treat its output as private: it includes titles and identifiers even though it omits signed media URLs.
4. Use `download-all downloads` only when actual acquisition is intended. Keep the output directory and the store's parent consistent between runs.
5. Preserve completed media and job records together. Availability and completion are different: a local completed recording can remain valid after its remote URL expires.

`--store` is a global option and belongs before the subcommand. Its parent selects the job directory and the output of inline `--refresh-session`. Changing it unintentionally can make existing completions appear absent. A standalone `refresh-session` instead uses its `--output` option or the documented default.

## Filename convention

Normal `download-all` acquisitions use `YYYY-MM-DD_HH-mm-ss_JST_replay-ID.mp4`, for example `2024-01-01_09-00-00_JST_replay-123.mp4`. The prefix uses the replay API's `start_time` Unix seconds, converted to fixed Japan Standard Time (UTC+09:00), independent of the machine timezone. It is not the notification arrival time, filesystem date, or the API's separate `create_time` field. The stable replay ID avoids collisions for broadcasts starting within the same second.

An unknown or invalid start time blocks a new automatic download; no filename is guessed from a room ID. Already-completed or busy IDs are checked before API resolution, so historical filenames remain deduplicated even if remote metadata later disappears. The low-level `download` command continues to respect its explicit output argument, since an observed media URL alone does not establish a broadcast start time.

## Rename completed recordings

There is no automatic rename-on-startup or rename CLI command. Renaming is an explicitly authorized maintenance operation, not a reason to download again.

1. Match each stable ID to verified API start-time metadata and its existing completed job. Use `replay.recordings.recording_filename` to produce exactly the same name as future acquisitions.
2. Prepare the whole rename plan before changing files. Check that all originals are regular, complete files and all target names are absent, including dangling symlinks. Keep a private old/new path mapping and verification evidence.
3. Use `replay.jobs.rename_completed` under its per-ID lock, keeping each file in its original directory. It creates a non-overwriting hard link to the same inode, atomically updates the completed job, then removes the old name. This is a same-file rename, not a copy, download, or re-encode.
4. Check file identity, size, modification time and content, then verify `already_complete` and the deduplication path. Do not rename unrelated diagnostic clips as though they were full recordings.

A failed/interrupted operation may leave two names pointing to the same data; the job still points to an existing file. Inspect the private plan and current job before removing an alias or retrying. Never remove a record or overwrite a different destination to suppress the conflict. A successful repeat for the already-recorded name is a no-op.

## Common failures

| Symptom | Safe next step |
| --- | --- |
| Missing or invalid session environment variable | Configure the authorized session privately. Do not paste the token into a prompt or issue. |
| Session expired/rejected | Reload a newly issued credential if available. Use explicit reauthentication only when authorized; do not loop indefinitely or assume cookie expiry guarantees API access. |
| Refresh did not rotate the token or extend expiry | This can be a valid reissuance. Read the result fields; do not alter cookie lifetime locally or claim renewal. |
| Handle lookup unavailable/challenged | Check the web extra and Chrome installation, or use an independently verified numeric ID. Do not bypass the challenge or guess identity from display names. |
| Empty catalog | Check the chosen creator and account permissions. Only actual accessible notification history is enumerated. Do not invent missing room IDs. |
| API omitted its replay list / malformed pagination | Treat it as an API/schema failure, not an empty catalog. Add synthetic regression fixtures for investigated changes. |
| Media URL denied or expired | Resolve a fresh URL through the authorized API. Do not edit signatures, expiry parameters, or TLS settings. |
| Completed file missing/changed or output already exists | Stop and verify the file and job metadata. Do not delete records, overwrite outputs, or redownload automatically. |
| Busy recording | Another process may own its stable-ID lock. Inspect the process before retrying; do not unlink a lock to bypass it. |
| Download exceeded byte limit | Review available storage and the per-recording limit before authorizing a larger transfer. A partial file is not a completed recording. |
| Unsupported/encrypted/open-ended HLS | Stop. This tool does not support bypasses for those formats. |
| FFmpeg unavailable or remux failure | Check FFmpeg/ffprobe and local disk capacity. Preserve existing outputs; investigate with synthetic media first. |

Reload the secret file into the parent shell after a standalone session refresh. Child processes cannot change the parent's exported environment. Fresh visible Web login and expired-session recovery must not be assumed to work merely because their mocked tests pass.

## Move local data to a new checkout

There is **no automatic migration CLI command**. Bootstrap provisions software; Git clone/pull does not copy authentication, recordings, or job records.

1. Verify the new checkout and locked environment first. Do not copy `.git`, `.venv`, caches, or regenerated build artifacts from another checkout.
2. Stop related writers or acquire the existing session/job locks. Inventory data and check free space. Never overwrite pre-existing destination data without reconciling it.
3. Securely copy the active token together with its fingerprint-bound metadata, observed candidates, job records, and recordings. Keep directories private (0700) and credentials/metadata 0600. Do not put them in Git, public artifacts, or an untrusted synchronization location.
4. Compare source and destination sizes and cryptographic hashes. Verify copies are independent of source paths, including symlinks. APFS copy-on-write clones can avoid allocating the full data size again on a supported volume; they are not source symlinks or hardlinks and are not off-device backups. Do not assume this facility exists on every filesystem.
5. Completed jobs contain absolute output paths. After independently verifying the copied files, re-register each corresponding output using the existing `replay.jobs.register_completed` API and validate with `already_complete`. Do not register running/failed jobs as complete or create an empty job directory to hide an inconsistency.
6. Validate that token metadata still matches the copied token. Do not reset account-continuity metadata or manufacture a new expiry to make authentication pass.
7. From the destination checkout, use only destination authentication and store paths for an authorized live listing. If requested, perform a bounded transfer check as described in [development.md](development.md), then delete its temporary media.
8. Confirm all completed outputs point into the destination, private data is ignored/untracked, and normal runtime no longer depends on the old directory. Delete the original only with explicit authorization and after verification.

Historical investigation material can be retained separately under `private/legacy-investigation/`. Old scripts, device configurations, and logs may retain historical absolute paths; archiving them does not make those auxiliary environments immediately runnable. Never substitute old experimental credentials for the active credential.

Local migration runbooks and account-specific settings, if present, belong under `private/`; do not copy their contents into public documentation or an agent handoff.
