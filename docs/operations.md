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
  files, which still follow the `compressed_part_*` route below.
- **Prefer a verified replacement to a hold.** Large ProRes parts and whole masters are
  retired once a verified compact HEVC part set (`compressed_part_*`) and the dated delivery
  exist; keep clean sources, settings, clocks, receipts and failure evidence.
- **Report both numbers.** Logical bytes removed and the measured `df` increase can differ
  (hardlinks, APFS clones, concurrent jobs); state which is which.

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
