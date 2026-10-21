# Operations and troubleshooting

See [README.md](../README.md) for setup and CLI examples, and [SECURITY.md](../SECURITY.md) for credential handling. Examples use placeholders, not real accounts or credentials. Resolve paths from the current checkout, not a previous temporary working directory.

## Normal workflow

1. Load a valid authorized session privately into `TIKTOK_SESSIONID` as described in the README. The CLI does not automatically read a `.env` file or a saved token file for ordinary API commands.
2. Select a creator with `--user`, `--anchor-id`, `TIKTOK_TARGET_USER`, or `TIKTOK_TARGET_ANCHOR_ID`. A display name is not a verified username. An explicit CLI selector takes precedence; two environment selectors at once are rejected.
3. Run `recordings` first. It resolves replay metadata but does not download video. Treat its output as private: it includes titles and identifiers even though it omits signed media URLs.
4. Use `download-all downloads` only when actual acquisition is intended. Keep the output directory and the store's parent consistent between runs.
5. Preserve completed media and job records together. Availability and completion are different: a local completed recording can remain valid after its remote URL expires.

`--store` is a global option and belongs before the subcommand. Its parent selects the job directory and the output of inline `--refresh-session`. Changing it unintentionally can make existing completions appear absent. A standalone `refresh-session` instead uses its `--output` option or the documented default.

For a repeat run of a previously verified creator, keep the numeric ID in `private/target-anchor-id` and the active session in `private/tiktok-sessionid.secret`; both are ignored by Git and must be private regular files (0600) beneath a private directory (0700). Provision the ID **only after independently verifying it**; an interactive shell prompt avoids placing its value in shell history and `noclobber` refuses to replace an existing selection:

```sh
read -rs -p 'Verified numeric creator ID: ' target; printf '\n'
(umask 077; set -C; printf '%s\n' "$target" > private/target-anchor-id)
unset target
```

Use `bash scripts/replay-authorized.sh recordings` before `bash scripts/replay-authorized.sh download-all --workers 6 --max-gib 12`. This script refuses alternate creator/store/host flags and never places either private value in command arguments or normal output. It does not discover or authorize a new creator, read `.env`, log in, refresh automatically, or erase failed/completed jobs. `--refresh-session` is an explicit option if a still-valid session needs reissuance; refresh is not a guarantee of longer validity. If either file is absent, supply a verified selector or safely provision the credential; do not derive an identity from a completed replay ID.

## Selecting one observed replay

`recordings` includes `start_time`, the validated API broadcast-start Unix seconds
(or `null` when unavailable), without exposing creator IDs or media URLs. Convert
that value to JST to select a broadcast date; do not use notice times or ID order.
Then `bash scripts/replay-authorized.sh download-all --replay-id REPLAY_ID` limits
acquisition to that observed replay. The full notice pagination must still finish,
and an ID absent from the authorized creator's notices is rejected before media
resolution or output creation. Stable-ID locks, completion checks, filename rules,
media host scope and byte limits are unchanged. Omitting the option retains the
normal all-observed-recordings behavior.

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

## Completed recordings outside the download directory

If a completion record points outside `downloads/` (for example an encrypted scratch
volume), do not redownload it: the record is complete, so `download-all` skips it by
design. Mount the volume, then copy it into the download store and update its job:

```sh
uv run replay relocate-completed REPLAY_ID downloads
```

The copy is SHA-256 verified, published without overwriting an existing name, and
keeps the previous file. A 20 GiB reserve must remain on the destination.

## Required edit stages

Every edited replay has a per-recording `edits/replay-<id>/edit-config.json` written
from its completed job (`python -m replay.edit_completion init edits/replay-<id>
--engine ENGINE_DIR --executor-scope SCOPE_DIR`). It names the source, edit engine and
Executor scope and always enables all five stages: music removal, non-speech cuts,
masking of the source's burned-in captions, transcription, and burned dynamic captions.
Disabling a stage or adding footage-specific settings is rejected.

`python -m replay.edit_completion check edits/replay-<id>` exits 0 only when one dated
delivery passes every stage. It reads the verified music-removal report, the render's
cut plan, mask plan and frame audit, the caption plan and full-decode record. A render
with zero masks passes only when coarse OCR found no repeatable caption band; an OCR
failure is a stop, never "no captions". Native AVFoundation tools also reject media
whose resolved path lacks a `.mp4`/`.mov`/`.m4v` suffix, so offloaded proxies must keep
their extension.

An older delivery that misses a stage is recorded under `evidence/superseded/` by
`python -m replay.edit_completion supersede` (identity-bound, nothing deleted), which
allows the complete revision to take the dated delivery name.

## Offload generated videos to explicitly configured storage

For a recording with an existing edit config, `python -m replay.edit_completion
bind-storage EDIT_ROOT EXTERNAL_RECORDING_DIRECTORY` declares one existing, real,
absolute directory named after that recording. The previous config bytes remain
as a hash-named backup. This command does not move files or certify any delivery.

After an explicitly authorized, hash-verified copy, a local video symlink may point
to the **same relative path** below that declared storage root. Keep its suffix,
local sidecars, evidence and completion records. Dated-delivery discovery prefers
the local identity and counts the external mirror only once; unrelated links,
linked external targets and external paths with linked parents are not accepted.
All content hashes and required-stage checks still apply. A missing SSD does not
make an absent video complete or authorize replacement downloads.

Before migration, acquire the relevant recording lock, exclude active readers and
writers, check the mounted destination and free-space reserve, refuse existing
targets, preserve the source until independent copy/hash and post-link validation
succeed, and retain a recovery manifest. Do not offload working directories or
all evidence via directory symlinks. Cross-volume hardlinks are not supported.
This compatibility support is not an automatic relocation or background cleanup.

## Keeping disk and SSD use bounded

Free space is a shared resource. The floor is the larger of 30 GiB and 5% of the volume,
checked on the startup volume and on any offload volume.

- **Measure physical use, not file length.** `python -m replay.disk_budget report ROOT`
  lists allocated bytes by directory and extension. Each inode counts once, symlinks are
  not followed, `private/`, `downloads/` and `.git` are skipped, and sparse test files cost
  nothing, so a logical total can be many times the real one.
- **New large work is refused below the floor.** `scripts/tmux-job.sh` runs
  `python -m replay.disk_budget preflight --path WORK` before it reserves a launch and exits
  with code 4 when the floor is not met. `replay.delivery_pipeline` repeats the check at
  every step boundary. Pass `--need-gib N` to `preflight` when a job's peak extra
  allocation is known.
- **Retire only what an owner approved.** Nothing is deleted by filename pattern. The owner
  of a recording records one row per file (`path`, `kind`, `dev`, `inode`, `nlink`, `bytes`,
  `mtimeNs`, `ctimeNs`, `classification`, and `linkTarget` for symlinks) in a disposition
  file. Run
  `python -m replay.leaf_retirement FILE --sha256 HASH --root ROOT/ --log LOG` for a dry
  run, then again with `--apply`. Only `eligible` rows are considered; each is rechecked for
  canonical parent, identity, symlink target and open handles immediately before `unlink`,
  and any mismatch holds just that row. Directories are never removed and symlink targets
  are never followed.
- **Retire a finished delivery's render intermediates automatically.** `replay.delivery_run`
  runs `python -m replay.delivery_retention CONFIG --fcpxml TIMELINE --apply` after the render
  pipeline, timeline and summary succeed. It requires the dated delivery, transcript, a passed
  media verification (frame count, full decode, audio packets), no `assess_delivery` failure and
  the registered Viewer link. Targets come only from producer declarations (Compressor job and
  workflow `outputs`, native `render.response.json`) and cover both the startup disk (`work`) and
  the SSD (`storage`): alpha movies, native parts, `joined-video.mp4`, `av-master.mov`,
  `selected.wav` and the Compressor output. Media an FCPXML references, JSON, receipts, models, the
  delivery, its transcript and Viewer links are kept. Rows go through `leaf_retirement`; run
  `delivery_retention CONFIG` without `--apply` for a dry run. A `work/retention/retention-receipt.json`
  marks completion, and `delivery_run` and `pipeline_supervisor` then refuse to rerender, because a
  rerun would regenerate the removed media. This does not cover native-revision `part-*-master.mov`
  files, which still follow the `compressed_part_*` route below. Retirement also writes
  `work/retention/restore-recipe.json` and moves (never deletes) the checkpoint directories and
  JSON outputs of every producer plan that touched a retired file into `work/retention/archive/`.
  `delivery_retention CONFIG --restore` sets the retention receipt aside; `delivery_pipeline` then
  rebuilds the media from the original. Rebuilt movies are accepted by media verification, not by digest.
- **Applied retention offloads to R2 first.** Before anything is unlinked, `--apply` deploys the
  temporary token-gated Worker (`web-viewer/src/upload-worker.js`, through `replay.r2_worker`,
  deleted afterwards) and uploads every eligible row to the private `replay-intermediates`
  bucket as `intermediates/<recording>/<sha256>`, with the SHA-256 as object metadata and each
  multipart part checked against its MD5 ETag. The recording's manifest
  `edits/<recording>/evidence/intermediate-offload.json` records `root`, `relative`, size,
  mode, mtime, key and state (`pending`, `uploaded`, `retired`, `restored`); it never names an
  absolute media path. A failed upload stops retention before any unlink, and reruns reuse
  matching objects. The bucket's lifecycle rule deletes `intermediates/` objects 90 days after
  upload (`expiresAt`) and aborts unfinished multipart uploads after one day.
  `python -m replay.intermediate_offload status edits/<recording>` counts entries;
  `python -m replay.intermediate_offload restore edits/<recording> [--relative PATH]` downloads
  next to the target, verifies SHA-256, restores mode and mtime and links the file into place
  without overwriting. After expiry use `delivery_retention CONFIG --restore` to rebuild.
- **Quiet scratch under `edits/work` offloads the same way.** `python -m replay.work_offload`
  is a dry run; `--apply` uploads, then retires, every single-link regular file with at least
  1 MiB allocated (not mostly sparse) whose mtime and ctime are older than `--quiet-days`
  (default 7). Each `edits/work/<name>` is one unit with its own
  `<name>/evidence/intermediate-offload.json` under recording id `<name>`, so `status` and
  `restore edits/work/<name>` above apply. `.git`, offload manifests, locks, partial
  downloads, `r2-worker/` and the `work-offload/` log directory are never candidates. Open
  files are uploaded but held; a failing unit keeps every file and the sweep continues.
  Smaller non-empty files go through `python -m replay.work_pack sweep [--apply]`, which packs
  each unit's quiet small files into zstd (level 10) `.tar.zst` packs under
  `edits/work/work-offload/packs/<unit>/`: one group per first-level subdirectory, split at
  256 MiB uncompressed, so later edits and restores touch only their group. Packs are
  offloaded the same way, then the listed members and the local pack are retired; a rerun
  finishes an interrupted pack. The unit's `evidence/small-files-offload.json` lists each
  pack's members with SHA-256, size, mode and mtime;
  `python -m replay.work_pack restore edits/work/<unit> [--relative PATH]` links back every
  missing member (or only PATH, downloading only its pack) and never overwrites a file.
- **Prefer a verified replacement to a hold.** Large ProRes parts and whole masters are
  retired once a verified compact HEVC part set (`compressed_part_*`) and the dated delivery
  exist; keep clean sources, settings, clocks, receipts and failure evidence.
- **Report both numbers.** Logical bytes removed and the measured `df` increase can differ
  (hardlinks, APFS clones, concurrent jobs); state which is which.

## One-command recording pipeline

`python -m replay.recording_pipeline start <replay-id> --defaults edits/work/recording-pipeline/DEFAULTS-v1.json`
runs every stage of one recording detached (`scripts/tmux-job.sh`) and in parallel where the
graph allows: clocks, GPU separation, source-caption cleanup (STTN refill), ASR, YOLO tracking,
cut plan, paging, placement, Motion caption parts, Compressor alpha render, composition, finish
and `publish`. Each step writes a receipt bound to its inputs; a rerun skips finished steps and
resumes interrupted ones. A source whose first video frame is presented after zero is set up
on a `source_lead_in` copy (`<deliveryRoot>/source-lead-in/<recording>/`, same file name, with
`lead-in.json`): the video is encoded once at 10 Mbit/s with first-picture lead-in frames, every
source frame keeps its PTS and the audio is copied; the edit config is rebound to the copy and its
previous bytes kept as `edit-config.<sha>.json`. A single caption word too wide for one line is
split near its middle (a morpheme end first, then any grapheme) before it is scaled. A source whose cleanup profile finds no caption band (the bootstrap
refuses a blanket mask) gets a 240-frame profile survey under `caption-absence-v1/` instead; only
the verifier's own no-band refusal passes it, the source is composed unchanged, and the gate
re-runs that verification. `--force STEP` redoes a step and everything after it; `--force-all`
redoes the recording. `publish` hardlinks the movie into the recording's bound storage
(`revisions/pipeline-<sha12>/delivery/`) with its SRT and `metadata/pipeline-evidence.json`, which
`python -m replay.edit_completion check edits/replay-<id>` assesses in minutes
(`replay.pipeline_completion`). The last step, `upload`, runs the web viewer's `sync`
(`web-viewer/src/bin/sync.ts`, one at a time under `web-viewer/sync.lock`), so a published
recording reaches Cloudflare without a separate command. Keep Compressor open while it renders; it reports "job controller
is down" when the app quits after a submission.

## Resumable processing and measured phases

Keep configuration, logs and state in a private, purpose-named directory under
`edits/work/`. Before executing a generated multi-stage graph, run
`python -m replay.workflow WORKFLOW.json --check`. This checks dependencies;
it does not verify rendered media or certify a delivery.

For an existing `delivery_pipeline` configuration, the shared supervisor resumes
from completed step evidence rather than restarting the whole render:

```sh
bash scripts/tmux-job.sh edits/work/WORK pipeline \
  .venv/bin/python -B -m replay.pipeline_supervisor CONFIG.json
```

Inspect the retained launch reservation, log and status before retrying. Reusing
the same job name is deliberately refused. An unknown application submission
must be reconciled against its real job/output; a supervisor cannot turn an
uncertain submit into permission to submit again.

Measure a new command with the shared timer instead of inferring execution time
from file timestamps:

```sh
bash scripts/tmux-job.sh edits/work/WORK measured-stage \
  .venv/bin/python -B -m replay.phase_timer \
  edits/work/WORK/phases.jsonl verification -- COMMAND ARGUMENTS
```

The existing log parent must be a real private repository directory. Each JSON
line records the phase, wall-clock start/end, monotonic elapsed seconds and child
exit status. Declare the timer among workflow inputs when source pins apply.
Concurrent appenders use a bounded lease; replaced/missing existing logs and
partial final records are rejected. A logging failure after a child exits is not
evidence that the child did nothing: inspect outputs before any new attempt.

Registered native MCP providers can use `scripts/native-mcp-stdio.sh` with their
fixed binary so stderr and child scratch stay in the checkout. Retain stdin and
stdout for MCP transport, and inspect the actual provider's file descriptors;
setting scratch environment variables alone does not establish confinement.
These routines do not replace source hashes, native completion, frame/audio
verification or the final edit-completion checks.

### Running independent stages in parallel lanes

A workflow runs its stages one after another, so a manifest of many independent
windows (for example face tracking) uses one core of an otherwise idle machine.
Measured on one 2,656-frame, 90-second tracking window: 34 s alone, and four
windows at once took 37 s in total. A sequential 60-window head-tracking run took
3,300 s. Lengthening the redetection interval to 4 s instead saved only 24 % and
lost the face on 825 more frames, so it is not used.

```sh
.venv/bin/python -B -m replay.workflow_lanes WORKFLOW.json edits/work/WORK/lanes \
  --lanes 6 --launch tracking
```

`replay.workflow_lanes` splits the manifest into independent groups (stages joined
by a file dependency stay together, in order), balances them over up to 16 lanes
and writes one ordinary `lane-NN.json` with a private `state-NN` per lane.
`--launch PREFIX` starts every lane through `scripts/tmux-job.sh` as `PREFIX-laneNN`.
Each lane keeps the usual fingerprint, completion and verification rules, and a
lane output directory must be new or empty. Use it for work that has not started:
completed stages recorded in another state directory are not reused by fresh lane
states. Choose the lane count from free memory and cores, and inspect every lane's
retained log and status before claiming the whole workflow finished.

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


## Workflow failure records

`python -m replay.workflow_diagnostics PLAN --state DIR` runs a workflow exactly like
`python -m replay.workflow` but leaves `DIR/last-failure.json` on failure: the exception chain with
source locations, the redacted message of repository-authored errors and the OS errno. Plan
contents, command arguments and locals are never recorded. Prefer it for new long jobs so a failed
stage points at its private `run.log` instead of leaving only the exception class.

## Record mirror and shared-volume headroom

- `python -m replay.record_mirror DESTINATION` copies the small, unrecoverable per-recording
  records (`edit-config`, completion JSON, transcripts; never `evidence/` or `proxy/`) to a
  different volume, verified by size and never deleting. Current mirror:
  `/Volumes/m.2-ssd/movie/replay-records-backup` (rerun to refresh).
- `disk_budget.require_shared_headroom({path: peak_need})` sums the needs of paths on one
  device so scratch and output on the same SSD share a single free-space check.
- `native_delivery_completion.observe_identity` no longer compares `st_dev` (remounts renumber
  it); size, inode, mtime and ctime still bind the file, and `--verify-content` remains the
  cryptographic check after a copy to another volume.

### Large completed work lives in the external work root, not on extra volumes

Per-folder `replay-cold-*` APFS volumes mounted over `edits/work/<NAME>` were retired on
2026-10-17: macOS remounts them at `/Volumes/<name>` on every SSD reconnect, leaving the
`edits/work` paths empty, and each mount renumbers `st_dev`. Large completed or offloaded work
now lives as a plain directory in the approved external root
`/Volumes/m.2-ssd/movie/replay-work/<NAME>` (the root `scripts/tmux-job.sh` accepts), and
`edits/work/RELOCATED-TO-SSD.tsv` lists each moved folder's old path, new path, file count,
bytes and time. Receipts that still name `edits/work/<NAME>` must be read through that index;
moving between volumes gives files a new inode and ctime, so stat-bound pins recorded before
the move need re-pinning, while content sha256 still verifies.

- `python -m replay.work_roots resolve OLD_PATH...` prints where a recorded path lives now, and
  `python -m replay.work_roots check` fails if a relocated folder is missing or its old path
  reappeared. New code that validates a work directory should call
  `replay.work_roots.within_work` (either root) instead of hard-coding `edits/work`; hash-pinned
  modules keep their `edits/work` checks until their owners re-pin them.
- Do not create new per-folder volumes, bind-like mounts or symlinks; start large jobs in the
  external root directly.
- `native_invocation.identity()` no longer records `device` (decision 2026-10-16, user), so
  pins made after that survive remounts; pins recorded earlier still hold `device`.
- Re-pin (2026-10-17): `python -m replay.receipt_repin LIST --report OUT` re-verifies every
  pinned input of a finished receipt by sha256 and size at its current location (through
  `work_roots.resolve`) and writes `<receipt>.repin.json` with the current stat identity; the
  receipt is never edited, and the only accepted digest changes are the edits approved for
  this work (`native_invocation.py` option A, `native_delivery_completion.py`,
  `disk_budget.py`). Of the 7,541 receipts pinning the old digest, 6,730 were re-pinned. 811
  stay unresolved because an input really differs: 689 are pytest scratch receipts whose test
  media was deleted or mutated by the tests themselves, and 122 pin code that changed later
  (`separation_window_*.py`, `disjoint_wave_queue.py`, `Package.swift`, AudioClockProbe
  sources); on the user's instruction (2026-10-17) both groups were deleted, so no receipt is left
  unresolved. Reports:
  `edits/work/cold-consolidation/repin-report.json` and `repin-report-2.json`. Plans that will run again are re-bound with
  `replay.separation_plan_refresh` or `replay.metal_asr_config`.


## BGM library

BGM is kept apart from TikTok/YouTube replays in `<library>/<video id>/`, where `<library>` is the
`root` of ignored `edits/bgm-library.json` (`{"version": 1, "root": "/Volumes/m.2-ssd/movie/bgm-library"}`;
the Viewer and `replay.bgm_library` both read it, an unmounted volume is reported and never
recreated) or, without it, `edits/bgm-library`:
`<title> [<id>].m4a` (title, channel and artwork embedded, ready to drag into Final Cut Pro or
Logic Pro), `<title> [<id>].jpg` and an allowlisted `track.json` (title, channel, channel URL,
watch URL, duration, upload time; never signed media URLs). Save tracks with
`python -m replay.bgm_library add URL [URL ...]`; only canonical watch URLs are accepted, audio
only is downloaded (no video, no cookies), and a track appears only after its audio decodes.
The audio is YouTube's best stream (`-f ba`, usually Opus ~160k at 48 kHz) decoded once into
24-bit Apple Lossless, so no second lossy generation is added (about 375 MB per 30 min). Do not
let yt-dlp convert it (`-x --audio-format alac` produced AAC). An existing track is reported as
`present`; `--replace` re-downloads it and swaps it in only once the new one decodes. TranscriptViewer opens
the library in its own **BGM** window (toolbar or ⌘B): search title/channel, play audio with the
thumbnail shown, open the YouTube page, Reveal Finder, copy the path, or save a new URL.
`storage_disposition` always keeps `bgm-library`.

The BGM window's **デフォルトに設定** picks the BGM for videos without their own. Under the player
a searchable chooser (artwork, デフォルト, BGMなし, every track) sets one video's BGM; it starts and
pauses with the video, loops, and stays at 1.0x. 手動 is a fixed volume; 自動 keeps the BGM at the
chosen offset (default −22 dB) below the video's level at the playhead (±5 s power mean, pauses
under −50 dBFS held). The Viewer never decodes audio for this: the video's per-second levels are
`edits/excitement-cache/levels-<movie key>.json` (written by `replay.excitement catalog` or
`python -m replay.excitement levels MOVIE...`; the key is size plus first/last MiB, so moved or
offloaded copies match) and the BGM's level is `track.json` `loudness` (written on `add`/`add-loop`;
backfill with `python -m replay.bgm_library loudness`). Missing data shows 音量データなし and
assumes −20 dBFS talk and a −10 dBFS BGM. Both modes follow the video's own volume and mute. The
choice and volume are saved per recording ID (else movie file name) in the app's preferences, not
in `edits/`.

### Looping BGM from a talk stream

When a stream's BGM is a loop under continuous talk, rebuild it from the separation residual
instead of downloading anything. `python -m replay.bgm_loop residual OUT.wav --part MIX VOCALS
... --keep-seconds S` joins mix minus the separated vocals with exactly the decoded parts and
grid the separation used (verify each part's decode hash against its `sourceSHA256`).
`python -m replay.bgm_loop extract RESIDUAL VOCALS OUT.wav RECEIPT.json --repeats N` measures
the period by autocorrelation, aligns every repetition per 10 s segment (decoding joins and
clock drift included), rebuilds each 1 s block by a spectral median taken per frame over only
the copies in which nobody speaks at that moment (under speech the separation also strips parts
of the music, identically in every talked-over copy) and tiles N seamless periods. Frames with
fewer than three speech-free copies (`fewSpeechFreeBlockSeconds`) take, per frequency bin, the
third of the copies (3-8) with the least vocal-stem magnitude in that bin, because the separation
strips music only where it hears voice; on a held-out speech-free phase rebuilt from talked-over
copies only this cut the 2-16 kHz error by about 1 dB versus the quietest three copies broadband.
The receipt lists `uncleanBlockSeconds` (no two copies agree) and `singleCopyBlockSeconds` (no
median possible); check the result by
separating one period again and comparing its vocal energy with a raw repetition. When the
period holds a silence (a song that ends and is played again), the output starts at the attack
after it (`songStartFrame`); `python -m replay.bgm_loop retile MASTER.wav OUT.wav --repeats N`
re-tiles an exact one-period master the same way. A BGM mixed under talk is quiet: both commands
scale the period once in floating point so its 4x-oversampled true peak meets `--peak-db`
(default -1 dBFS, receipt `gainDB`). Decode the mix parts with `-c:a pcm_s32le`; the residual and
output are 32-bit float WAV (RF64 once past 4 GiB, as a multi-hour residual is), so the gain
amplifies no rounding noise (scaling a saved 16-bit master would). Publish it with
`python -m replay.bgm_library add-loop URL OUT.wav [--replace]`, which fetches only metadata and
the thumbnail and encodes 24-bit Apple Lossless (sample-exact length, so the file wraps gaplessly;
the Viewer's restart at the end of the item is not guaranteed gapless); `--replace` swaps an existing track only after the new one is staged and decodes.
The track is marked `loop` and the Viewer restarts it at its end.
