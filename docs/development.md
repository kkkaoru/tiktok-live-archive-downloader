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

## Sample-exact ASR refinement windows

For an overlong or zero-duration recognition group, do not invent word times or
repeat the same failed caption measurement. `python -m replay.asr_window SOURCE.wav
NEW_DIRECTORY --start-sample START --sample-count COUNT` extracts at most 60 seconds
from mono PCM16/16kHz audio into a new private directory. Its `window.json` binds
the source and extracted hashes and the exact half-open source sample range.
Run `replay.local_asr NEW_DIRECTORY/window.wav NEW_DIRECTORY/recognition` with the
pinned local model. Raw predictions and failed attempts remain evidence; replacing
caption tokens still requires clock projection and validation, not just successful
recognition. Neither command approves the transcript or changes existing plans.

`python -m replay.caption_asr_refine CONFIG.json NEW_DIRECTORY` can then create a
candidate token replacement for one unfinished measured-caption job. It pins the
extraction/ASR inputs, raw recognition, original job manifest, cuts and explicit
identity-clock evidence. Configure a whole-group source sample interval and the
existing segment's output frame offset. It rejects split boundary groups,
unsupported clocks, completed jobs and replacements affecting other jobs. The
original manifest and successful measurements stay unchanged. Outputs are only
`tokens.json` and a hash-bound `receipt.json`; scheduling a new measured candidate
and selecting it for combination are separate steps. Neither successful ASR nor
successful projection proves transcript correctness or approves delivery.

## Layered Motion caption finishing

The adopted default is style 20 in `src/replay/caption_style_defaults.py`:
sRGB `#2FB0EE` → `#E3F6FF`, held at the upper colour for 75% of the height,
then a linear gradient over the remaining 25%. `replay.motion_finish` selects
this preset when neither sRGB override is supplied. It requires explicit raw
`--top`/`--bottom` controls, a resolved `--font-name`/`--font-size`, and
`--stroke-scale`/`--shadow-scale` conversions from the SVG reference to the
prepared document. Do not infer these conversions from a colour label or treat
candidate scale inputs as calibration evidence. The shared
`selected_finish_style` factory uses the constants for every layer and shadow;
per-field geometry overrides cannot silently alter the selected preset.

This explicit user-selected palette is **not white-AAA** (its endpoint ratios
are approximately 2.451664:1 and 1.111553:1). Its candidate annotation records
these actual ratios, selection identity, raw controls and unverified render
flags, rather than inventing a 7:1 pass. The selected policy rejects any changed
colour, stop position or midpoint. Explicit alternate sRGB pairs retain the
historical strict 7:1 policy; the arithmetic and sampled AAA audit are unchanged.
Mixed/partial style configuration fails. Neither route proves rendered fidelity,
centering, face clearance or delivery; native output still requires verification.

The following is the explicit historical white-AAA override route:

`python -m replay.motion_finish PREPARED.moti NEW.moti --top R G B --bottom R G B
--top-srgb '#24556A' --bottom-srgb '#28596D'`
finishes an existing bounded two-object-per-stage Motion sequence. It preserves
text, font size, transforms and reveal clocks, and adds a synchronized foreground
with a thin black keyline, a white band, an outer black rim and shadow. The original
upper control must be explicitly replaced, with separately declared sRGB intent;
`--bright-start` (default 0.8) and `--bottom` control the
small highlight region. RGB values are Motion controls, not certified sRGB output.
Outline/shadow sizes are in the already prepared document's units.
`--font-name POSTSCRIPT --font-size MEASURED_POINTS` replaces the font on all
three objects together; both are required. A named weight can resolve from an
installed variable font without generating static files. Verify the actual font
file and native resolution first, then remeasure widths and Motion placement;
old-font calibration is not reusable merely because the point size is similar.
The helper
refuses malformed, unsynchronized, already-finished or oversized input and never
overwrites output. Run it **after** the historical two-object stage generator.
The FCP exporter accepts matching two- or three-object titles and duplicates the
caption into every object without style overrides. Reveal animation requires a
complete set of distinct published face/rim controls, plus the keyline control for
three objects; partial controls fail before changing the title. Finishing a Motion
sequence does not itself publish those controls or install an editable title.
Neither helper certifies contrast, placement, FCP editability or a completed delivery.

`replay.fcp_project --measured-captions candidate.json --measured-captions-sha256 HASH`
replaces the plan's ASS captions with a pinned `caption_measured_job` candidate.
The coordinate declaration, canvas, output frame rate, bounded text, ordered clocks
and in-canvas origins are checked. Explicit top-left canvas origins bypass the
legacy Hiragino/ASS baseline calculation, including for multiline captions.
Coordinate spaces cannot merge into one caption event or growth chain, and legacy
head relayout refuses measured origins. Source cuts, masks and audio are unchanged.
The plan still supplies its template caption style (`captionBurn.style` alone is
accepted without an obsolete ASS file). Callers must independently
verify upstream bindings and the installed template/calibration correspondence;
this adapter does not certify those, rendered AAA contrast or completion. Actual alpha bounds, colour-managed
compositing and final export must be verified separately.

## Safe-area caption wrapping for fixed-bottom captions

`replay.caption_safe_wrap` lays a title phrase out inside a caller-supplied
safe width using only its width, character count and character class: the
shared linguistic wrapper decides first at measured phrase boundaries,
character boundaries greedily pack per-character advances when the phrase has
no usable phrase structure (repetitions, overlong single tokens, half-width
kana, Latin runs), and a phrase still beyond the three-line budget rolls its
oldest characters out so the newest words stay inside the safe area. Every
rolled-out prefix is reported with its dropped characters; display clocks are
unchanged and no text, style or timing is invented. The advance estimate
overestimates narrow Latin widths so true renderer measurements fit inside
the wrapped lines; actual renderer measurement remains separately verified.

## Bottom-anchored multi-line captions

Rendered caption blocks grow downward from a fixed first-line top, so a
constant vertical anchor clips every multi-line block at the canvas edge
(measured: single-line boxes at y[828,1008], multi-line bottoms all at
y=1080 on the 1920x1080 output). `replay.motion_plan_batch` therefore
anchors the block bottom: each additional line shifts the placement up by
one measured advance (`MEASURED_LINE_ADVANCE_PX` = 138px at font 104,
glyph-top to glyph-top across two independent ProRes alpha renders, ±12px
glyph-dependent variation), scaled linearly with the render font size and
converted to canvas units. The shift applies both when paging cues and when
coalescing overlapping units into multi-line text. Verify after rendering:
multi-line bottoms must land back at the single-line bottom.

## YouTube Motion-overlay completion check

`python -m replay.youtube_edit_completion check edits/youtube-<id>` assesses
the YouTube route (Motion alpha titles composited over a cut program in one
ffmpeg pass, delivered as `delivery/<timestamp>_JST_<id>-edit-*/` with an
SSD-backed movie symlink, transcript and summaries) against the same five
required stages, using only measured evidence. Only the newest dated
delivery is the completion candidate; older generations are reported as
superseded and never count as complete. The parts manifest is chosen by
duration match *plus* a complete evidence set (wrap, tail-wrap and both
left-edge records), never by filename order. Stage rules: separation receipt
plus LC-AAC 44.1kHz stereo delivery audio whose multi-window volumedetect
levels match the cut program within 0.1dB; cut-plan kept totals versus
delivery duration with measured quiet peaks; a local Vision OCR survey of
the source (`evidence/source-masks-none/summary.json`, at least 120 evenly
spaced frames, no repeatable bottom-center caption band, raw frames bound by
digest); filtered word recognition projected through the cut plan onto the
cut clock before transcript comparison; overlay renders spot-verified
against their manifest plus wrap/left-edge evidence, delivery faststart and
a full decode, alpha bottoms of sampled 1/2/3-line captions inside the safe
area, a forensic multi-line burn-in proof (program/delivery
difference height at rewrapped phrase times), and final-pixel edge checks
(no caption pixels in the bottom guard band). Caption placement inside the
measured safe region is an additional explicit gate: only settled captions
(displayed longer than two frames) count, with a 5px shadow-tolerance inside
the 9px margin; transients are listed, never counted. Unproven stages are
reported as unmet, never as complete. The frozen `edit_completion.py` /
`edit_config.py` are only imported for reading.

## Measured captions to bounded Motion stage plans

`python -m replay.motion_measured COMBINED.json NEW_DIRECTORY --sha256 HASH
--width 1080 --height 1920 --rate 30 --frames FRAME_COUNT --expected-text Title`
exports plans consumed by `replay.motion_stages`, then `replay.motion_finish`.
It checks the combined candidate's pinned input files and common calibration,
keeps measured text and origins, and uses the existing FCP frame-sampling rule
(`start <= frame time < end`). A caption without a display frame or overlapping
frame intervals fails; no caption is silently discarded or retimed.

Each part has at most `--max-stages` stages (default 32; `replay.caption_parts` passes
`MAX_PART_SENTENCES` so a caption window renders as one Compressor part, since each Compressor
job has a ~4.5 s fixed cost) and covers only its own local interval. The
manifest records its global `offsetFrame`; gaps remain empty, and the caller
must retain that offset when compositing. Outputs are private and non-overwriting.
The existing XML, rendering and resource limits remain in force. This is candidate
plan preparation, not native rendering, colour/geometry acceptance, editable FCP
registration or a completed delivery.

## Cut-bounded native video composition

`python -m replay.native_edit_parts REQUEST.json NEW_DIRECTORY --sha256 HASH`
partitions a pinned video-only native request at its existing cuts. Each part has
one unchanged source selection and only the intersecting Motion video layers;
layer selections are clipped and rebased on the exact output-frame grid. Layer
order, opacity, geometry and static encoding settings are retained. This reduces
unrelated composition tracks without re-recognizing words or changing captions.
Timed effects, transitions, retiming, extra audio, off-grid boundaries and layers
extending beyond the complete source timeline fail instead of being dropped.
Native per-part duration/layer limits remain unchanged. Requests and their
frame-offset/hash manifest are private and non-overwriting.

The native service must still validate and render these requests. Verify every
part's clocks, streams and full decode, then use the existing `native_join` path
for lossless assembly and independently verify the assembled timeline. Audio is
processed separately on its verified sample clock; do not pad or stretch it to
match a video header. These requests do not certify source-caption repair,
rendered colour, A/V sync, editable FCP registration or delivery completion.

## Native-clock PCM selection jobs

`python -m replay.continuous_cut_job CONFIG.json NEW_DIRECTORY --sha256 HASH`
connects the existing `continuous_cuts` sample mapping and block-copy writer.
The version-1 configuration declares `coordinateSpace: native-presentation-seconds`,
SHA-256-bound `{path, sha256}` references for `source` (PCM WAV) and `clock`
(decoded audio-frame probe), optional upstream references in the `evidence` array,
and ordered `intervals` with `start`/`end` rational `{value, timescale}` records.
Paths stay inside the checkout; JSON, clock, waveform and list sizes are bounded.
The waveform is streamed, not loaded into memory or constrained by a caption-file
byte budget. All inputs and implementation bindings are checked again after copying.

The private, non-overwriting output contains `selected.wav` and `selection.json`.
Native timestamp gaps are not filled: the receipt retains exact sample spans and
rational audio/video-interval duration differences. No resampling, fades, padding,
or timestamp compensation is applied. Independently establish source correspondence,
compare selected PCM bytes, and check A/V sync after muxing; those approval flags
remain false. A failed job preserves its partial directory and cannot overwrite it.

## One-command recording delivery

`python -m replay.delivery_run DELIVERY.json --defaults DEFAULTS.json` runs the whole
render side for one recording. `DELIVERY.json` (version 1) names only `recording`
(`replay-<id>` or `youtube-<id>`), the dated `source`, source-clock transcript `labels`,
the frame `cuts`, either glob patterns `sourceRenderConfigs` of the caption-removed chunk
renders or, for one verified single-file clean source, the pinned `sourceCleanup` config
(`assemblyManifest`, `assemblyReceipt`, `cleanIndex`; its index becomes the render
config's `sourceChunks` and the config pins it), and an optional separated `vocals`. `DEFAULTS.json` names the machine's
measurement/foreground executables, calibration, Motion template and finish arguments,
Compressor presets, work/storage/delivery/viewer roots, separation runtime, summary CLI,
the caption fallbacks (`minimumTop`, `unknownTop`) and worker counts.

Each step is skipped when its output exists, so a rerun resumes: fitted 1080x1920 proxy
when needed (`replay.delivery_media proxy`), video/audio clocks, foreground plan, voice
separation (`replay.delivery_audio`), token projection, chunk index, render config;
then `replay.delivery_foreground` and `replay.delivery_measure` supervise foreground
observation and caption measurement together (failed attempts are kept as `.failed-sK`
and retried once); `replay.delivery_pipeline` renders (stopped workflow states are
recovered by `replay.stage_recovery`, completed receipts kept); `replay.delivery_fcpxml`
writes the editable timeline; the summary CLI summarizes the new SRT.

Shared fallbacks are recorded, never silent: `token-changes.json` (same-frame onsets
merged, coarse tokens' display capped at the policy context), `clamp.json` (support top
floored when the subject reaches the frame top, default top for unobserved frames), and
the combine's `hiddenPrefixesDropped` / `captionsWidenedToOneFrame`. Deliveries carry
`<stem>.verification.json` (full decode, frames versus the cut plan); caption, source-
caption removal and music-removal quality remain unapproved by these tools.

## Rust Vibrato caption morphology

The optional `editing` extra and development group pin `vibrato==0.2.3`, the
upstream Python binding to **Rust Vibrato 0.5.2**. There is no Sudachi fallback.
`replay.caption_linguistics` loads its dictionary once per process and caches
bounded analyses. `CaptionPolicy(linguistic_boundaries=True)` uses actual POS
boundaries for caption switches without changing native recognition timestamps.
The module's `wrap_two_lines` prefers phrase boundaries, then intact morphemes;
it rejects impossible widths rather than splitting words by character count.

Install the binding with the existing environment manager:

```sh
uv sync --locked --extra capture --extra web --extra editing
```

The dictionary is local user data, not a bundled package asset. Obtain the
[upstream v0.5.0 IPADIC archive](https://github.com/daac-tools/vibrato/releases/download/v0.5.0/ipadic-mecab-2_7_0.tar.xz)
(7,704,680 bytes). Its observed SHA-256 is
`4764f983b7c3a9e1cb6a5ee945e00558efd812980e0dad61224f63ee3b0475d9`.
Verify that checksum **before** extracting. Keep `COPYING` and `NOTICE` alongside
`system.dic.zst` under the current checkout's `edits/models/vibrato/`, then use
`zstd --decompress --keep` to create `edits/models/vibrato/system.dic` without
overwriting an existing file. The uncompressed dictionary is 47,788,814 bytes,
SHA-256 `f3ddd9cacbd6eff696e49bb9266824eed28dceaa89c55e75e4a13384cc7f9d02`.
These are pinned downloaded-content identities, not a publisher signature.
Runtime loading checks the size and checksum; it never downloads a dictionary.

Unit tests build a tiny synthetic dictionary through the actual Rust binding's
`Vibrato.from_textdict`, so they require neither recordings nor the downloaded
IPADIC dictionary. A successful POS test does not certify recognition accuracy,
rendered caption placement, colour contrast, or completed video delivery.

## Required-stage delivery discovery

`replay.edit_completion` discovers dated `.mp4`, `.mov` and `.m4v` deliveries
(case-insensitive extensions), excluding symlinks, directories and other recording
IDs. Discovery alone never approves a native render: the existing five required
stage checks still apply. A root with no recognized deliveries remains incomplete.
Native Motion/Compressor evidence is not interchangeable with legacy shared-render
receipts; MOV discovery does not establish such evidence compatibility.

An explicit `REVISION/metadata/native-evidence.json` selects the native completion
reader (`replay.native_completion`), without a legacy fallback on malformed or
stale evidence. Version 1 has backend `motion-compressor`, recording/config/source
fingerprints, and recording-relative `{path, sha256}` references named `manifest`,
`encoding`, `sourceEncoding`, `audio`, `cuts`, `masks`, `captions`, and `fcp`.
It supports the independently verified refined-audio/unchanged-video receipt chain:
whole and part movie hashes, exact declared frame counts, full-decode/clock/PCM
receipts, statistical music/speech measurements under the unchanged policy,
protected cut accounting, source-mask input/execution accounting, actual
Motion/Compressor alpha identities, observed caption layout, exact reconstruction
of the SRT from rational native caption plans, and observed editable FCP import.
Digest reuse is restricted to one check and file identities are checked again at
its end. Native account storage is not an evidence input.

Source-mask execution accounting alone **does not pass** source-removal quality.
`sourceCaptionQualityApproved: false` remains an explicit required-stage failure;
never flip it merely because a render or accounting audit completed. Even a true
flag requires a separate hash-bound `qualityEvidence` reference covering every
selected masked frame and binding every processed input. Its measured worst-case
residual/instability fractions and readable/technically-invalid frame counts must
pass the existing shared `decide_repair` thresholds without cosmetic fallback.
Missing measurements, sampled-only coverage and a bare approval flag cannot pass.
This reader consumes trusted local processing receipts, not cryptographically authenticated
attestations or new perceptual measurements. The same `assess` API can preflight
an existing private candidate before any dated delivery or Viewer publication.
Synthetic contract tests do not establish actual rendering, FCP appearance or
source-caption quality. No media is created, modified or published by this check.

YouTube originals use a separate version-2 `edit-config.json`, initialized with
`python -m replay.edit_completion init-youtube ROOT --source ORIGINAL --publication
PUBLICATION_JSON --engine ENGINE --executor-scope SCOPE`. The original's dated
stem must match the validated publication ID/time and the publication file must
reside beside that original. MKV/WebM originals are accepted as source evidence,
not asserted to be AVFoundation-compatible. Both media and publication bytes are
SHA-256-bound; same-size changes are rejected. Every required stage remains
literal `true`. This is local configuration, not acquisition or native-render
approval: an empty library or a MOV without required-stage evidence still fails
`check`. Existing TikTok version-1 configs and completed-job initialization remain
unchanged.

## Agent skills in Claude Code

Project skills live in `.agents/skills`. Claude Code discovers only
`.claude/skills`, so link it locally and start a new session:

```sh
mkdir -p .claude && ln -s ../.agents/skills .claude/skills
```

The link is ignored and must not be committed; `make public-check` rejects
symlinks. The `apple-*` skills call the Executor MCP namespace `apple-pro-apps`,
which must be registered separately; skill discovery does not authorize app actions.

## Optional local subtitle repair

`uv sync --locked --extra editing` installs NumPy and headless OpenCV for classical,
ROI-only subtitle repair kernels. These are also development dependencies so the
synthetic tests run without skips. They do not download AI models or send media
to a service, and normal downloader imports do not load them.

The repair and bounded-retry modules are building blocks, not an integrated
unattended editing CLI. Synthetic tests do not establish visual quality. The
same glyph-repair family is retained on retries; the last variant permits local
boundary softening rather than substituting a rectangular blur band. White
highlights alone cannot seed this source-specific pink-glyph mask. Additional
mask variants require independent real-output evidence before adoption.
`InpaintSettings(fill_enclosed_white=True)` is an opt-in diagnostic variant that
includes neutral-white interiors enclosed by the existing pink contours. It does
not seed standalone white highlights or fill dark letter counters, and defaults
remain unchanged. It does not distinguish non-text pink objects or reconstruct
occluded background detail; synthetic tests alone cannot approve its use in a delivery.
`registered_glyph_samples` is a separate opt-in building block for locally measured
source-to-donor and donor-to-source pixel flows. It copies only source-mask pixels
whose interpolation footprint stays in the donor, samples no donor-mask pixels,
and has at most half a pixel of round-trip error. Rejected pixels retain their
original content; they are not silently filled. Flow consistency is not proof of
correct correspondence or of a subtitle-free donor: independent registration,
occlusion, text-coverage and temporal-output audits remain required. Existing
spatial repair and retry defaults do not call this helper.
`InpaintSettings(algorithm="navier-stokes")` explicitly selects OpenCV's other
local inpainting kernel for controlled comparisons using the identical glyph
support. The default algorithm and automatic retries remain Telea. Neither
kernel's successful execution certifies visual quality or hidden detail recovery.
`render_repaired_video(..., measure=..., require_quality=True)` connects an
independent quality callback to those retries and refuses missing measurements
or cosmetic fallback. Receipts distinguish measured quality from passed quality;
they cover supplied mask spans only, not exhaustive subtitle detection. The
caller must still verify encoded output clocks/full decode and mask coverage.
When splitting one sequence for processing, pass `ramp_start_frame` and
`ramp_total_frames` on `VideoRepairJob` (or `repair_stream`) in the verified output
frame domain. This keeps the same softening ramp across artificial chunk seams;
it does not change local mask indices or video PTS. Defaults retain standalone
part behavior. Ramp continuity alone is not perceptual/temporal approval.

`python -m replay.subtitle_observations SOURCE TIMES.json NEW_EVIDENCE_DIR`
collects full-frame observations through the existing local Executor connection.
The time plan is a strictly increasing JSON array of source seconds. Unchanged,
validated batches can resume; failed attempts remain private and completed
missing/changed results are never recollected. Source/plan/code fingerprints and
requested clocks are checked. These sampled observations are not exhaustive
mask coverage or semantic subtitle approval.
The same collector's explicit `--head-tracking` mode requests `video_head_detect`
instead of OCR, once that native capability is independently verified and deployed.
`head_observations.head_frames` validates its source-pixel rectangles and actual
sample clocks, fits them into the existing vertical canvas, and selects the largest
face (otherwise the tallest body), matching the single-subject layout policy.
Missing detections remain absent; no identity or interpolation is invented. Tool
mode and parser fingerprints prevent treating an OCR revision as head evidence.
`head_observations.tracking_frames` separately validates raw `video_face_track`
results against an independently verified `VideoFrameClock` and display dimensions.
Every rational source PTS must match in order: duplicates, missing/extra frames,
shifted clocks, invalid confidence/rectangles and inconsistent detection flags
fail. Missing geometry stays empty, without body fallback or interpolated positions.
This parser does not collect evidence, bind source/tool hashes, certify hair
boundaries or approve captions; the caller must do those steps separately.

`python -m replay.tracking_observations collect CONFIG.json NEW_OUTPUT_DIR
--state-directory STATE_DIR --name window-0001` collects one verified window via
that parser and the existing `replay.stages` runner. Its version-1 JSON requires
absolute `source` and `nativeExecutable` paths, their independently verified
`sourceSHA256` and `nativeSHA256`, display `width`/`height`, and `clock` containing
integer `pts`, rational `timeBase` (for example `"1/30000"`) and `lastDuration`.
Optional `redetectIntervalSeconds` defaults to 0.5. Clocks must come from verified
source PTS, not average frame rate. The reduced clock denominator is passed as
native `sourceTimescale`; windows are bounded to 120 seconds and 7200 frames.

`tracking_plan.tracking_windows` partitions a complete, independently verified PTS
vector into those same bounded clocks without changing the per-clock limits. Every
source frame belongs to exactly one window; split extents use the following measured
PTS, and EOF uses the caller's independently verified terminal duration. It preserves
absolute VFR clocks, rejects invalid ordering and overlong single-frame extents, and
never estimates a frame rate or truncates a tail. Callers bind source/clock evidence,
reuse unchanged completed prefixes and invoke the existing stage runner. Window
partitioning is not continuous track identity, complete face geometry or edit approval.
With explicit `seed_pts`, windows instead share exactly one complete face-seed frame:
the latest eligible seed is used, with its full extent counted in both windows.
Seeds are exact decoded integer PTS, including the first requested frame, never
rounded sampling times or body-only detections. Missing/off-grid seeds or gaps
without an advancing seed fail. `maximum_seconds` can explicitly reduce, never
increase, the native duration budget. Both overlapping observations must be retained
and checked; candidate seeds do not establish successful native detection or identity.
The default unseeded partition remains unchanged.

`python -m replay.tracking_batch prepare MANIFEST NEW_DIRECTORY` prepares private,
non-overwriting window configs and a `workflow.json` for the existing workflow CLI.
The version-1 manifest supplies `source`, `sourceSHA256`, `nativeExecutable`,
`nativeSHA256`, `width`, `height`, `redetectIntervalSeconds`, rational-string
`maximumSeconds`, and `clock`, `seeds`, `eof` references (each `{path, sha256}`).
The clock file has `pts`, `timeBase`, `lastDuration`; the seed file supplies
`candidateSeedPTS` and matching `timeBase`; EOF evidence is an ffprobe `packets`
object whose last presentation packet supports the exact final PTS and duration.
The caller must independently verify full-source decoding, packet timebase/EOF
coverage, source provenance and candidate derivation before preparing this trusted
local manifest. Matching hashes preserve those inputs, not certify their producer.
The aggregate clock is bounded separately; per-window and workflow limits remain.
Run `python -m replay.workflow NEW_DIRECTORY/workflow.json --state STATE_DIR`
with project-local temporary environment variables. Sealed wrappers check planned
manifest/evidence/config/code hashes before and after native execution and reuse,
including before the first stage; modified workflow bytes also fail. Actual native
face detection is required at each window start. Missing later geometry remains
unknown, overlaps are retained in both responses, and no head/identity/placement
approval is inferred. Failed or incomplete attempts are preserved, not overwritten.

`tracking_stage` creates an ordinary stage with real observe/verify CLI commands
for the trusted workflow runner. Source, executable, configuration and shared
code hashes bind the request and raw response; verification runs again on reuse.
Changed/missing completed output stops, partial attempts stay intact, and unknown
outputs are not adopted implicitly. Linked output storage is rejected. This
performs no UI/app registration, camera capture, model download or fallback.
Finite, positive tracker boxes leaving the image are terminated as lost tracks,
not clipped into valid geometry. `outOfFrameTrackingObservations` preserves each
frame's rejection count and requires a redetection attempt; without valid geometry,
the frame remains missing. `outOfFrameDetectionObservations` separately records
finite, positive detector boxes outside the image; these are not accepted or seeded.
Any rejection leaves caption-placement coverage uncertain even if another valid
face exists in that frame. Malformed extents still fail.
Older successful responses implicitly have zero rejections (the older native
implementation aborted rather than returned such a window).
Window completeness is not full-video coverage, hair geometry, safe caption
placement or a completed edit stage. Native integration-test and deployment
requirements remain separate.

`foreground_observations.foreground_frames` validates supplemental, independently
collected foreground measurements: up to eight exact rational requested/actual
source times, native display/pixel budgets, explicit observation presence, ordered
local instance labels, source-sized union masks, integer nonzero-pixel counts and
unclipped integer bounds. The L8 mask threshold must remain one; a response cannot
approve caption placement. Missing observations, observations without instances,
and present instances with an empty quantized mask remain distinct. A bounding
rectangle is not filled-pixel coverage, labels are not persistent identity, and
neither masks nor ROI-conditioned face landmarks prove head/hair geometry or
subject absence. Callers bind the source, executable, request and response; this
parser alone neither collects frames nor completes an editing stage.

`python -m replay.foreground_collection collect CONFIG OUTPUT --state-directory
STATE --name WINDOW` collects this evidence with the existing stage runner. Its
version-1 configuration has exactly `version`, `source`, `sourceSHA256`,
`nativeExecutable`, `nativeSHA256`, `clock`, `width`, and `height`; `clock` contains
integer `pts`, rational-string `timeBase`, and integer `lastDuration`. Paths must
be absolute, and source-clock provenance is independently verified by the caller.
A window has at most 128 frames and spans at most 120 seconds. Each native call
receives at most eight exact rational samples, with a 60-second deadline; the
whole stage is bounded to 1,200 seconds. Requests retain the source tick scale,
without converting through floating-point seconds. Inputs and code are hashed
before and after collection, and every request, raw response and empty diagnostic
is verified again before reuse. Completed missing/changed evidence stops reuse;
partial attempts are retained rather than overwritten or adopted. `observe` is
new-output-only; `verify` is read-only. Collection never approves subject absence,
head geometry, exhaustive video coverage, caption placement or an edit stage.

`foreground_evidence.read_window` reads a completed collection against independently
pinned configuration and provenance SHA-256 values. It verifies the unchanged
producer inputs and parses the actual response bytes named by the receipt. It
never recollects missing evidence. `foreground_geometry.interval_top` requires the
complete supplied rational frame clock, includes the frame containing an interval
start, honors the exclusive end and terminal duration, and rejects required
unknown/empty support. Its placement helper checks a complete origin-relative
Motion envelope against an explicit region and fixed horizontal origin, without
legacy gauge limits, shrinking, clipping or semantic approval.

`caption_measured_plan.measured_plan` combines measured POS wrapping, existing word
clock resets and output-frame prefix accounting with per-growth-run envelope
placement. It requires a bounded output-domain token window and output frame
rate, a callback mapping output intervals through **verified** cuts to complete
source support, and independently bound Motion envelopes (text, style, template,
canvas and all animation states). A conservative common height can force earlier
timestamp-aligned switches; each growth run uses the union of its prefix envelopes
and its own support ceiling. Neither a guessed source/output mapping nor a native
font metric alone satisfies these callback contracts. An impossible union is an
error, not permission to change font size or discard words. These APIs create
candidates, not rendered-frame, semantic, contrast or delivery approval.

`python -m replay.caption_measured_job MANIFEST NEW_DIRECTORY --manifest-sha256 SHA256`
connects the reader, exact retained-clock map, calibrated native envelope and measured
planner. Run from the checkout root; explicit work/native input paths must be
absolute, checkout-local and unlinked, and the new private output belongs under
`edits/work/`. Original media referenced by an independently pinned foreground
receipt may remain on its existing external volume: the reader and final digest
checks verify that exact source without copying, moving or changing it.
Its pinned version-1 manifest requires exactly `tokens`, `spans`, `foreground`,
`native`, `calibration`, `outputFrameRate`, `region`, `centerX`, and `clearance`.
File references use `{path, sha256}`. The separate token document contains
`{version: 1, clock: "output", tokens: [{text, start, end, utterance}]}`. Each span
has `sourceStart`, `outputStart`, and `duration`, each `{value, timescale}` with
integer values. Each foreground reference contains pinned `config` and `provenance`
(the latter named `provenance.json`). The existing calibration supplies the exact
style and canvas; `region` is `[left, top, right, bottom]` in that same source canvas.
The shared 4096-token/120-second and integer 1–120 output-FPS limits remain.
At most 128 native calls share a cooperative 300-second planning budget; filesystem
operations are not forcibly timed out, so callers still need an outer deadline.
Requests and failures remain private. Conflicting pins fail rather than overwrite
one another. A fully written pending candidate is linked to `candidate.json` only
after final foreground/input checks, without overwriting an existing name; the
pending evidence remains. Every approval flag stays false: pins and an `output`
clock label are not proof of actual editor correspondence, complete Motion support,
semantic placement, AAA contrast, or delivery completion. Tests use synthetic
foreground receipts and mock native text measurement; they do not certify live
native IPC or real Motion rendering.

`caption_source_support.OutputSourceSupport` connects pinned `read_window` results
to that interval callback through explicit `RetainedSourceSpan` values. All source
starts, output starts and durations are exact fractions: a span is a verified
unit-speed translation, not a source frame number divided by nominal FPS. Within
one bounded 120-second output window, retained spans must be source-ordered and
output-contiguous. Up to 32 source windows must be ordered/disjoint and match the
same source, native producer and canvas identities. Queries split across cut and
evidence boundaries; removed portions are not queried, and any required gap or
unknown support fails. No source/output pixel transform, frame-rate conversion,
retiming, cut inference or actual editor correspondence is supplied. Bind the real
cut/clock evidence and reverify all receipts separately before publication.

`motion_native_envelope.CalibratedMotionEnvelope` supplies a provisional signed
origin-relative box from the existing `NativeCaptionLayout.measure` cache. Its
independently pinned calibration JSON has exactly `version: 1`, `canvas: [W,H]`,
`style` (the five `NativeCaptionStyle` fields), `nativeFont` (the seven native font
response fields), observed `padding`, `template` and `evidence`. References use
absolute `path` and `sha256`; 1–32 evidence files are required, each bounded to
64 MiB. The exact style/canvas must match; fresh native font metrics must equal
the calibration. Width is centered; top is `-padding-renderedAscent`; each text's
native glyph height plus observed Motion leading determines the bottom. A sampled
visible-alpha height must not replace this text-dependent measurement. Origin
`(x,y)` in top-left canvas units maps to Motion `(x-W/2,H/2-y)`.

`python -m replay.motion_native_envelope EXECUTABLE CALIBRATION NEW_DIRECTORY
--calibration-sha256 SHA256 --canvas W H --text TEXT [--text TEXT]` records up to
32 measured candidates with native requests/responses, fingerprints and explicit
false approval flags. It neither renders nor calls apps. Existing directories are
refused, failures remain, and source framing is untouched. Verify both producer
inputs and every consumed response before rendering/publishing a plan. The adapter
also checks each producer-pinned request's font/size/border/rim, including prewarmed
caches, and refuses incomplete attempts rather than scanning past a response gap.
It reads the frozen producer's attempt counter and input manifest without changing
that producer or adopting partial output. These
checks do not certify the supplied calibration's correctness, unseen text or
animation support: verify each actual Motion render independently before using
its envelope for approved placement. No source/output clock map is inferred.

`vertical_captions.layout(..., require_headroom=True)` uses the highest observed
head throughout each growing caption run, reserves the outer rim and head margin,
and refuses absent geometry or insufficient space rather than placing text on the
subject. This is sampled geometry, not exhaustive tracking or a calibration of
Motion's rendered text bounds. Its legacy default remains unchanged.

An explicitly authorized existing `apple-pro-apps` CLI can instead be selected
with `--native-executable /absolute/path/to/apple-pro-apps`; this never happens
as an automatic fallback. Its executable fingerprint is bound to the revision.
Native results stay raw JSON arrays, not fabricated Executor success receipts;
select the parser using `inputs.json`'s `backend`. Successful process exit,
requested clocks, and unchanged source/plan/executable are checked separately.

`subtitle_plan.observed_masks` connects explicitly selected observations to the
existing learned-band detector and native half-open repair plans, without a fixed
source canvas. Callers verify collection completeness and source provenance first,
and explicitly pass the evidence-selected `profile`. Learning counts lexical
variety without decimal digits or whitespace, so changing counters within fixed
Japanese/translated labels cannot alone seed subtitle bands. This affects only
profile diversity, never OCR text or boxes, and is not semantic approval.
Refinement does not relearn
subtitle bands from newly visible unrelated UI/text; out-of-profile lines remain
unclassified, not certified absent.
Actual decoded clocks, observed-empty cells, and unobserved gaps are preserved;
cells extend at most one second from an observation and stop at source duration.
Within active cells, geometry envelopes the immediate adjacent observations whose
bounded time supports touch, so a transiently truncated OCR line or held frame
cannot independently shrink the repair ROI. Empty cells and gaps stay unmasked.
This is sampled geometry, not a semantic or exhaustive mask-coverage certificate.
The shared `subtitle_repair` → `subtitle_stream` → `subtitle_video` path now obtains
frame-local character masks from `VNDetectTextRectanglesRequest` with
`reportCharacterBoxes=true`, through the native `image_text_rectangles_batch`
measurement operation and `subtitle_vision` (the single-image operation is retained). Vision's bottom-left normalized character boxes
are converted to top-left ROI pixels and rounded outward for rasterization. The
same detection is reused across all bounded repair retries. This does not use
recognized strings, pink-pixel thresholds, or a whole-line rectangle fallback;
the historical color helper remains available only for explicit legacy diagnostics.
Missing character geometry fails, and empty detections do not certify text absence.
Independent residual/readability/temporal quality checks remain mandatory for
strict delivery. An explicit detector callback can bind a separately verified
native executable and repository-local working directory. By default the stream
batches up to eight frames, reducing batch size to keep frame payloads within
32 MiB (an individually larger frame remains a singleton); source/output and mask
working buffers are additional. Only active subtitle ROIs are sent to one native
process/context, and decoded native images are released between requests. Results
must match request count and order. PNG I/O still occurs for each ROI. A separately
supplied single-image detector remains usable; it cannot be combined with a batch
detector. Measure representative throughput, memory and mask equivalence before
full-recording execution; batching does not establish footage quality or total ETA.

For detailed local diagnosis, enable DEBUG on **only** the `replay.subtitle_vision`
logger with a file handler in a private repository-local directory. Do not enable
root/HTTP debug logging. It records source/detector pixel hashes, PNG hashes and
lossless round-trip checks, channel ranges, stage timings, native observation and
character counts, request revision, bitmap bit depth/stride/color space, mask size,
and failure class. It never logs recognized text or image bytes. Expensive pixel
checks/hashes run only in DEBUG; retain these local diagnostic logs, not public logs.
Native measurement `diagnostics` is additive and optional when decoding old receipts.
Zero observations explicitly do not certify text absence.

An explicit `contrast_region` on `vision_text_mask` (or ordered `contrast_regions`
on `vision_text_masks`) may supply a caller-observed contrast reference on the exact
input image. It derives Otsu's threshold from that bounded region, then prepares the
whole detector input with the same threshold. A large surrounding scene must not
silently replace the reference histogram: that can erase the text/background contrast.
No region, threshold or clock is guessed, and uniform/out-of-image references fail.
Source pixels are unchanged, canvas coordinates are unchanged, and **only measured
Vision geometry**, never the thresholded pixels, forms removal masks. The default
raw-input path is unchanged. An observed reference and better detection are not full
coverage/readability/temporal acceptance; do not enable this on frames without bound
reference evidence or use it to bypass required quality checks.

For neural region repair, `vision_text_mask(s)(..., text_regions=True)` explicitly
selects the native measurement's `textBoxes`: actual `VNTextObservation.boundingBox`
values, not envelopes reconstructed from OCR or inferred between characters.
`text_region_mask` validates and rasterizes these at the same exact source dimensions,
with a 64-region bound and no fallback when the field is absent. Native legacy JSON
remains decodable with `textBoxes == nil`; default character-only processing is
unchanged. The broader measured support must be recorded and independently checked:
it does not certify complete text detection, acceptable reconstruction, temporal
stability, or permission to alter pixels outside that support. Empty detections
remain unknown even when an adjacent frame has a measured region.

`replay.subtitle_detector.LearnedTextDetector` is a separate, explicitly selected
local PP-OCRv3 DB detector using installed OpenCV DNN, not an automatic Vision
fallback. It requires an existing model and matching SHA-256, bounded to 32 MiB,
uses the [OpenCV Zoo preprocessing/decoding policy](https://github.com/opencv/opencv_zoo/tree/main/models/text_detection_ppocr),
and limits its worker to two CPU threads. Detection uses a separate copy matching
the configured network size, as required by the upstream wrapper. Its long edge
is 736 pixels; each dimension is rounded to a positive multiple of 32. This avoids
stretching wide subtitle bands into squares. Recorded grid sizes and fixed-point
projection preserve original source dimensions and subpixel polygon corners.
For non-square grids, a second square context view runs unconditionally with the
same pinned model and confidence policy. Only their validated measured support
is unioned; this is not an absence-triggered fallback or inferred mask extension.
Both views together retain the 64-region limit. Receipts count the square-context
regions separately; square inputs are not inferred twice.
DB's configured unclip expansion can cross its canvas. Convex polygon intersection
retains only measured support inside the canvas; individual vertices are never
clamped. Excessive expansion, malformed geometry and wholly off-canvas polygons
are rejected. Receipts retain observed-region and canvas-intersection counts;
intersections are not evidence of complete text coverage.
It sequentially handles at most eight ROIs, retains original dimensions and
source pixels, and rejects more than 64
polygons, malformed confidence or degenerate/out-of-bounds geometry. Masks come
from measured detector polygons, not OCR boxes. No inference-time download or
model Python code runs. Model changes during loading/rendering fail verification.

A `subtitle_candidate` configuration can explicitly set `learnedTextDetector`
with `backend: "ppocrv3-db"`, absolute `model` and `sha256`, alongside
`prepareVisionContext: false` and `visionTextRegions: true` (region support).
Receipts identify the actual detector and model binding, and set the unused
`visionTextRegions` receipt field to null. Native executable bindings remain for
the separate observation path; they are not evidence that Vision supplied these
masks. Existing Vision selection stays the default. Candidate rendering does not
approve coverage, reconstruction, temporal quality or any delivery stage.

`subtitle_audit.audit_mask_observations` independently checks complete line-box
containment at those actual clocks. `audit_output_text` compares matched source
and newly recognized output samples in the source-learned bands, reporting
readable lines and held-frame clock differences. Neither invents pixel-residual
or temporal-stability scores, and OCR absence is not perceptual approval.
`audit_pixel_support` counts supplied source/output evidence pixels and ROI leaks
on aligned decoded frames. Callers supply independently localized boolean masks;
it embeds no color detector, lowered confidence threshold or approval flag.
For a bounded CFR-frame audit, `subtitle_video.frame_observation_times` requests
interior local times. Add source offsets to those local times, rather than
subtracting large source clocks to reconstruct them. After verifying CFR output,
compare `observed_frame_indices` from actual decoded times with all expected
indices: repeated requests or held frames are not exhaustive frame coverage.
To avoid a full-sized lossless intermediate, `VideoRepairJob.normalized_start_frame`
explicitly selects a bounded window on the `fps` grid of a verified zero-based
source. It is not an original VFR frame ordinal. Whole-second seeking retains
keyframe preroll (including a frame held across a long gap), then FFmpeg normalizes
and trims exact frame indices. Normalization may repeat/drop source frames; callers
must validate the planned grid, extent and local mask mapping. A short input still
fails rather than padding a missing tail. The default prepared-CFR contract stays
unchanged. Do not use normalization when every original VFR frame must survive:
several input frames can fall in the same normalized clock cell.

For that case, pass `VideoRepairJob.source_clock=VideoFrameClock(...)` with verified
consecutive source PTS, rational time base and terminal presentation duration.
The shared adapter selects the exact source timestamp after whole-second preroll,
retains every decoded frame, and encodes explicit zero-rebased presentation clocks
through native FFmpeg. It does not add a decoder dependency or large intermediate.
The caller binds the clock vector to the unchanged source and independently checks
output PTS, frame count, terminal extent and full decode. Decode-order timestamps
and packet-duration representation can change after encoding. At split boundaries,
terminal duration extends to the next source PTS; concatenation must not accumulate
rounded-duration drift. `subtitle_clock.concat_durations` generates ffconcat
microsecond duration strings from cumulative exact boundaries on the declared
common mux grid; unrepresentable offsets are refused, not silently rounded off
the grid. This is not a join verification receipt: check every joined video PTS
and terminal extent, plus copied audio/subtitle payloads, clocks and dispositions.
At EOF, `VideoFrameClock.verify_terminal_packet_support` cross-checks the complete
bounded decoded tail against the original packets: identical presentation-frame
identities (including reordered packets) and an exact terminal duration. A shorter
container-level duration must not silently discard a decoded frame. Bind both
observations to the unchanged source and record any header discrepancy separately;
this input check does not certify a repaired output. All existing output frame,
terminal-extent and full-decode checks remain mandatory, without padding or relaxed
tolerances.

`native_join.verify_parts` also requires the selected video stream to have the
same stream index in every input. FFmpeg's concat demuxer associates streams by
index: an audio-first part followed by a video-first part can otherwise copy PCM
packets as video despite matching video codecs and time bases. A single-part
`native_join.join(..., audio=None)` losslessly remuxes a mismatched part into
video-first order; verify its copied packet payloads before the whole join. The
original stays untouched, and full joined clocks/decode still require verification.

For MP4 stream-copy assembly, `media.MP4_STREAM_COPY_OPTIONS` retains faststart
and disables the muxer's automatic `btrt` addition: mov_text already carries the
source atom in its extradata, so appending another changes subtitle configuration.
Keep comparing the complete extradata; do not ignore that field to accept a copy.
Raw decoding uses the filter time base so close VFR frames do not collide on a
nominal-fps encoder clock. Pass the same clock to `repair_spans(..., source_clock=...)`
so local mask times select actual frames rather than average-fps indices. Use its
`observation_times()` and `observed_indices()` for independently verified VFR OCR
coverage; duplicate or ambiguous actual clocks cannot establish full coverage.
This mode cannot be mixed with normalization or padded-tail trimming. The CFR-only
sampling/index helpers above must not be used to certify VFR observation coverage.
`subtitle_decode.verify_native_decoded_window` independently decodes a bounded
verified clock vector into native VideoToolbox surfaces. It retains private,
non-overwriting frame-clock evidence and diagnostics, rejects missing/extra/wrong
PTS or nonempty error output, and has no software fallback. Exhaustive disjoint
windows may use the existing two-worker scheduler. The framecrc checksums hash
AVFrame wrappers, **not pixel planes**; this verifies decoded clocks/counts, not
pixel equivalence or perceptual quality. Callers bind unchanged media and verify
terminal extent, audio and selectable captions separately.
Audio and selectable captions are not processed by this video adapter.
For platforms with verified VideoToolbox HEVC, select both `hardware_encode=True`
and `hardware_hevc=True` on `VideoRepairJob`; this tags output `hvc1`, with no
codec fallback if hardware fails. Optional `hevc_quality` (integer 1–100) selects
native constant-quality encoding instead of the unchanged 8 Mbps default. This
is a tuning input, not a quality measurement or permission to loosen storage
limits: compare decoded quality and actual size on verified evidence before
selecting it for a full revision. Probe clocks/frames/streams and fully decode
encoded output independently before accepting it.

For ASR of a continuous decoded PCM extraction, `audio_clock.AudioFrameClock`
projects word groups back through each original decoded audio frame's PTS,
rational time base and sample count. A constant first-audio offset is insufficient
when native timestamps contain gaps or overlaps. Bind the decoded-frame probe
and recognition WAV to the unchanged source; check conserved sample duration
within one recognition sample. Projection preserves unresolved zero-duration
points and rejects clock reversals; it does not resolve speech timing, certify
recognition accuracy, or modify the original audio.

### Quiet-only frame-cut planning

`quiet_cuts.quiet_cut_frames` consumes source-bound `MeasuredQuietSpan` evidence
and native `ClockLabel` candidates on an explicitly planned normalized frame grid.
It removes only quiet interiors of at least 0.3 seconds, rounds removal inward,
and protects recognized intervals with an outward-rounded 80 ms guard. Unresolved
points protect a frame too; this never assigns them a speech duration or repairs
recognition. Unclassified sound stays. Invalid clocks and an entirely removed
recording fail. This is not VAD classification or an edit-completion receipt.
Callers must independently verify the normalization, any explicit terminal video
hold/audio padding, the rendered cut boundaries, and final output clocks.

### Frame-aware caption context

`CaptionPolicy(frame_rate=...)` optionally anticipates width/lifetime resets when
an intermediate prefix would disappear before its first output frame. It starts
a new context at the existing token onset so the next growing prefix can retain
that token. It never changes native speech clocks, reveals future words early,
expands display budgets, or bridges sentence/ownership boundaries. The default
policy remains unchanged. Supply the actual planned output rate and independently
audit visibility after frame quantization; this is not proof of layout, accuracy,
or Motion fade support. A smooth final prefix still needs two visible frames;
`reset_lookahead_frames=2` anticipates budget resets within that horizon without
bridging semantic boundaries. Horizon and rewind decisions use the same bounded
rational `display_time` conversion as FCP/Motion events, so floating-point noise
at an exact frame boundary cannot invent another frame or suppress a reset.
This changes display-context decisions only; recognition clocks and the two-frame
smooth-prefix requirement remain unchanged. Audit remaining boundary conflicts independently.
`caption_display.unretained_frame_prefixes` separately checks prefixes with no own
output frame: only contiguous, same-owner additions of the actual next native token
may carry them into a later displayed prefix. A reset with coincidentally repeated
text, another owner, or a wholly unshown chain cannot establish retention. This is
lexical-prefix/frame accounting only, not fade, geometry or recognition approval.

### Native per-part Motion foreground requests

Source-caption production workflows declare `sourceCaptionProduction` with a
`bindings` object mapping consumer stage names to binding configuration paths.
Every binding path must be an explicit input of that stage. Native render,
native invocation, part render, delivery render and transformed-source consumers
require a binding when this production declaration is present. Semantic caption
planning is an intermediate stage and does not render the original footage.
External app render commands must also appear explicitly in `bindings`.

The usual input/output dependency graph orders repair producers before consumers;
`--check` validates declarations even when a producer has not created its binding
yet. Immediately before each consumer runs, including adoption or checkpoint
reuse, the shared binding validator checks the repair evidence, actual consumer
artifact and clean-media references against the consumer's explicit inputs and
command. A failed validation prevents the consumer from running. Workflows
without this declaration retain the generic runner contract; production workflow
owners must add the declaration when connecting external app processing.

The normal `delivery_render` composition/render builder requires `sourceCleanup`
in its render configuration, a `{path, sha256}` reference to the cleanup assembly
configuration (`assemblyManifest`, `assemblyReceipt`, `cleanIndex`). The original
source and selected clean index must match that configuration. Generated render
workflows always declare the binding and consumer input files; the pipeline also
rejects previously generated undeclared workflows. Resume preparation emits new
bindings, and direct retry validates them before launching the native process.
Historical receipts remain readable, but an old unbound seal cannot authorize a
new render. New repair and verified reuse use the same evidence contract.

Additional video layers are allowed only through optional `captionOverlays` in
the cleanup configuration: entries have `role: "motion-caption-alpha"` and an
`audit` path/hash reference to an existing `delivery-caption-audit` result with
`alphaOnlyAccepted: true`. Its part records must bind the alpha, Motion source,
plan and producer state, and prove full decode, exact frame clocks and measured
caption bounds/gaps. An arbitrary media path allowlist cannot authorize overlays.
The selected alpha intervals must lie within those verified clocks.

`python -m replay.native_render render-plan.json NEW_BATCHES.json --render-output
/absolute/new-output --foreground-manifest FOREGROUNDS.json` uses supplied per-part
foreground media instead of `styledCaptions`. Default requests remain unchanged.
The bounded regular manifest must contain exactly `version: 1`, `renderPlanSHA256`
(the SHA-256 of the exact plan file), and `parts`, in plan order. Each part has
exactly `startFrame`, `durationFrames`, `path` (absolute local media path) and
`sha256`. The two frame fields must match that part's global output interval;
the media itself must be independently rendered on a **zero-based local clock**.
Every part must be supplied; missing, extra, reordered or stale mappings fail
before any request file is written. Nonempty regular media files are fingerprinted;
changed or linked files are rejected. Source-cut selections remain unchanged,
and styled text is not drawn a second time. Fixed source-caption blur requests
are rejected; detected local source repair must precede this compositor.

This CLI only writes requests. The existing native single-pass renderer separately
requires matching canvas, duration and every decoded foreground frame from zero,
and refuses extra/missing/retimed frames rather than holding or stretching them.
Alpha is honored; opaque pixels cover the processed source, so inspect actual alpha
and caption quality separately. Rendering callers must bind the manifest and media
as stage inputs and verify their hashes again at execution/reuse: request generation
alone cannot prevent later file changes. Keep the **one editable full FCP/Motion
project** with synchronized text objects; generated native sidecars do not represent
foreground animation and are not a substitute. The manifest is not an editability,
caption-accuracy, placement or completion certificate. Full 60/90-second native
acceptance and all delivery-stage checks remain required.

### Bounded local word recognition

`python -m replay.local_asr INPUT.wav NEW_EVIDENCE_DIR` runs the already-installed
MLX Whisper runtime (the locked `youtube` extra) on any prepared recording. From
the checkout root, its default is the user-selected non-turbo
`mlx-community/whisper-large-v3-mlx`, revision
`49e6aa286ad60c14352c404340ded53710378a11`, under
`edits/models/mlx-whisper-large-v3/<revision>/`, with pinned config/weights hashes.
See README for those hashes and installation status. To reproduce an older model,
explicitly supply `--model LOCAL_MODEL_DIR --weights-sha256 SHA256
--config-sha256 SHA256`. It accepts exactly one local `weights.npz` or
`weights.safetensors`, and never substitutes a repository name or downloads a model.
The input must be complete mono PCM16/16kHz WAV, at most 60 seconds. It preserves
raw Japanese word-timed predictions and input/model fingerprints in a new private
directory. Failed or empty recognition is retained, not accepted as silence.
Unresolved zero-duration word groups remain explicitly counted, never stretched.
These are unreviewed continuous-PCM clocks, not presentation times: separately
bind the extraction to source decoded-frame evidence and use `AudioFrameClock`
before placing captions. This tool neither removes music nor certifies an edit.

`caption_io.native_sound_windows` parses explicit local CLI SoundAnalysis JSON
using the same score/clock limits as the Executor-backed `sound_windows` parser.
The caller must verify process success and unchanged source fingerprints; raw
measurements are not Executor receipts, and low/empty scores are never silence.

`caption_review.select_native_context(..., repair_target_onsets=True)` may repair
original onset reversals only when both sides lie wholly inside the explicit
replacement target. The default still rejects them. All alternative onsets and
untouched original onsets must remain ordered; shared anchors, native positive
intervals, display budgets and safe outer seams are unchanged. It selects whole
provider labels without sorting or inventing clocks, and does not certify wording.

## Optional voice-preserving BGM reduction

`src/replay/audio_restore.py` defines the complete local pipeline. The earlier
whole-mix level gate counted removed BGM as lost speech, so it reintroduced music to
meet a voice target; scalar time-domain blends cannot distinguish voice from music in
the same interval. The replacement applies a per-bin squared Wiener mask to the original
phase, then adapts locally against the source's pinned MLX Mel-RoFormer vocal stem. After AAC, it
re-separates the output and checks voice against that stem, while instrumental
residuals are measured in every 60-second window across the full video. All rules
are generic and per-video signal-derived; there are no recording-specific settings.
Install the locked optional dependency and run it with a new sibling revision path:

```sh
TMPDIR="$PWD/edits/.tmp-audio-restore" \
  uv run --no-dev --extra roformer python -m replay.audio_restore \
  edits/replay-<id>/revisions/<source-revision>/delivery/full-review.mp4 \
  edits/replay-<id>/revisions/audio-restore-v1
```

Only after that representative revision passes both quality gates, the tracked batch
orchestrator can process one unambiguous source delivery per replay sequentially.
Multiple input revisions require reconciliation before any processing starts. New
revision directories, videos, and subtitles use the validated broadcast timestamp
from source-bound `evidence/source-probe.json`; missing or mismatched dates stop
before directory creation. A per-source lock spans both restoration modes, and an
existing dated delivery blocks another revision family. The old
`scripts/build-expander-revision.sh` publisher is retired because it bypassed
this ownership and could overwrite files; old invocations fail before creating
anything. The expander remains part of the shared restoration workflow:

```sh
TMPDIR="$PWD/edits/.tmp-audio-restore" \
  uv run --no-dev --extra roformer python -m replay.audio_restore_batch \
  --revision-name audio-restore-mel-roformer-mlx
```

It refuses incomplete existing revisions and stops at the first failed gate.
The tool uses pinned `mlx-community/mel-roformer-mlx` vocals (8-second, half-overlapped windows) and a squared Wiener mask, then plans a downward expander from this candidate mix's ebur128/SRT speech and residual levels. FFmpeg `compand` caps the threshold at 9 dB below median speech and 60 dB above the measured residual floor; low-level inputs are shifted in float audio so the floor is at least -75 dB before processing, then shifted back. Its logs, plan, and verification stay under the new revision's `work/` and `metadata/` folders. The expander is an optional enhancement, not a condition for rejecting the whole candidate. It is used only if its measured residual suppression passes and the input's low-instrumental windows show preserved transcript speech (P05 loss within 1.5 dB and correlation at least 0.995); otherwise the pre-expander candidate proceeds to full AAC verification. The report records whether the stage was applied and the quiet-window voice comparison. The tool accepts only sources and new revisions inside this checkout's `edits/` tree.
It stores intermediate WAVs and `metadata/verification.json` beside the delivery, refuses
existing revisions. After a successor passes its gates, it writes a hash-bound
selection receipt and removes only the unchanged legacy processed input video.
Original recordings, source subtitles, plans, failed candidates, and other partial
work are retained. A failed check never authorizes retiring the input. MLX runs locally; its optional model
weights are not bundled in Git. The exact model revision and SHA-256 are checked against a previously downloaded local Hugging Face cache; processing never downloads weights on its own. A transcript is required for speech protection. The adaptive
blend protects the source-vocal level, then applies a uniform 0.5 dB voice-restoration gain.
The shared gates still check the decoded AAC output and keep the original 1.5 dB loss limit:
full-video AAC-decoded separated-voice level (P50 and P05 loss at most 1.5 dB,
correlation at least 0.995), instrumental reduction of at least 10 dB in every music-positive
60-second window, unchanged video frame count, and at most 30 minutes per audio hour. If no
instrumental content is found it records a no-op report without creating a delivery; any
failed gate stops before publishing the new video. These automated signal checks do not
replace listening review.

The strict 10 dB music-per-window rule can reject a model that its owner has
explicitly accepted by listening. Never label such a revision `verified` or claim
it met the numeric music/speech gates. With that explicit approval, the separate
`--listening-approved` workflow publishes only pinned MLX vocals (no Wiener
mask/expander), copies the original video stream, checks AAC duration and frame
count, decodes the entire resulting video/audio, and requires the entire run to
finish in **strictly less than 30 minutes per audio hour** before publishing.
It labels its report `listening-approved` and preserves the strict workflow:

```sh
UV_PROJECT_ENVIRONMENT="$PWD/edits/.venv-mel-roformer" HF_HUB_OFFLINE=1 \
  uv run --locked --no-dev --extra roformer python -m replay.audio_restore_batch \
  --listening-approved --device mps
```

The batch uses stable, dated, non-overwriting sibling revisions for each
source and recognizes prior completion records; it stops on the first technical
failure or throughput overrun. Reusing another revision name does not authorize a
second dated delivery for the same recording. Listen to
and review each new full-length output before choosing it as a delivery. The
short-window proxy is not a substitute for that review.

Generated media and work WAVs under `edits/` are local user data and must not be staged.
Only the processing code, tests, locked dependency metadata and generic documentation belong
in Git.

## Optional target-speaker extraction experiments

`uv sync --locked --no-dev --extra speaker-extraction --extra youtube` selects
PyTorch/torchaudio 2.2.2 with compatible NumPy 1.x for local reference-conditioned
speaker extraction experiments. This optional environment is incompatible with
the NumPy 2.x development, editing, and MLX RoFormer extras; restore the normal
verification environment with `uv sync --locked --extra capture --extra web`
before running the quality gate. No model weights or recordings are bundled.
Download model code only from a pinned revision, review it before importing, and
load checkpoints with `weights_only=True` and strict state-dictionary matching.
Evaluate held-out speaker references, synthetic mixtures, absent-target leakage,
and real overlapping speech before using any output for individual ASR. Merely
loading a model does not establish useful separation or correct speaker identity.

On macOS, add `--extra coreml-extraction` to that isolated environment for native
Core ML model evaluation. This only installs the locked runtime tooling, not a
model or a new extraction CLI. Use verified, pinned model packages, honor their
licenses, inspect their input contracts, and retain the same held-out/absence
checks. Core ML output arrays may have padded strides; preserve those strides
when converting results. Do not infer extraction quality from successful loading.

`uv sync --locked --no-dev --extra dialogue-separation --extra ja-alignment`
adds the pinned Japanese Nagisa tokenizer and PyKakasi reading dictionary for
local forced-alignment and orthography/pronunciation comparison experiments.
Dictionary readings are hypotheses, particularly for names; never substitute a
reading for an unknown acoustic label or claim it establishes a word's timing.
Use only reviewed, checksum-bound local models and local tokenizers with remote
code disabled. The installed aligner can interpolate non-monotonic timestamps:
reject those predictions instead of treating interpolation as measured clocks.
Preserve raw predictions, transcript text and the native timestamp quantization;
this dependency is not proof of correct timing or speaker attribution.

`uv sync --locked --no-dev --extra ctc-alignment` selects the existing modern
Torch version with Transformers for local CTC frame-alignment experiments.
Load only checksum-bound safetensors and explicit language adapters using local
files and remote code disabled; never fall back to unrestricted checkpoints.
Frame intervals must use the reviewed convolution stride/receptive field rather
than proportional text timing or a frames/audio-length ratio. Unknown vocabulary,
weak acoustic support and unsupported transcripts must remain unresolved, not be
forced into an approved subtitle.

`uv sync --locked --no-dev --extra usef-extraction` selects an isolated
PyTorch 2.14 / NumPy 1.x environment for reference-frame USEF experiments and
restricted checkpoint loading with explicit `safe_globals`. It is incompatible
with the old speaker-extraction, separation, editing, MLX and development extras.
Keep `weights_only=True`: static pickle-global inspection and a narrowly reviewed
NumPy metadata allowlist are required, never a fallback to unrestricted pickle.
Do not execute tagged upstream YAML constructors or import unreviewed training
modules. Convert validated tensor-only state to safetensors for other runtimes.
This tooling is not evidence of separation quality; preserve the same held-out,
solo, absent-target and actual-overlap gates.

`dialogue-separation` selects MLX Audio 0.5.7 for local two-speaker DialogueSidon
experiments without upgrading the existing RoFormer 0.4.3 workflow. These extras
are mutually exclusive. DialogueSidon is generative, has exactly two anonymous
output slots, and does not establish three-speaker separation; test identity,
leakage, and false speech before any transcription use. Its model weights carry
CC-BY-NC-4.0; retain attribution and do not assume commercial-use permission.

## Combining bounded measured caption candidates

`python -m replay.fcp_measured NEW.json --candidate PART.json SHA256 ...
--calibration CALIBRATION.json SHA256 --width 1080 --height 1920 --rate 30
--frames FRAME_COUNT` combines already
measured, globally timed parts without remeasuring. Parts must be ordered and
non-overlapping at existing policy-reset boundaries; the adapter does not sort,
trim holds, invent timing or approve segmentation. It requires matching policies,
valid canvas/clocks, nonempty provenance, unchanged input fingerprints and a
bounded total cue count. The caller identifies the actual common render calibration
(including its template, font/style and measured geometry); every part must bind
that exact file/hash. An unrelated shared input is not a calibration. Direct API
calls also enforce positive integral canvas/rate bounds and a positive rational
window of at most 120 seconds, rather than relying on CLI validation. Existing outputs are never overwritten. The result is
still a candidate: all geometry, editor correspondence, render, contrast and
delivery approval flags remain false. Independently verify complete requested
caption coverage and source-cut correspondence before rendering.

## Mandatory caption colour selection check

Before adopting any sRGB fill or gradient stop, run
`uv run --locked python -m replay.caption_contrast '#005A9C' '#24556A'`.
It reports every white-background contrast ratio and exits nonzero if **any**
colour falls below **7:1**, using unrounded WCAG sRGB luminance. This includes
small bright gradient toes; a white/black outline or shadow is not an exemption.
`require_white_contrast` exposes the same requirement to Python callers.
The sampled-output audit now requires 7:1 for upper, lower **and intermediate**
fill samples; the former 4.5:1 lower/intermediate thresholds are no longer used.

Motion controls are not measured sRGB. `CaptionAppearance.validate` requires
explicit `top_srgb` and `bottom_srgb` intent and automatically checks both against
white at 7:1 during creation and restyling. A component-wise **sRGB** envelope
bounds bounded RGB interpolation of that intent, not Motion's unknown conversion.
Raw `top`/`bottom` controls are checked for shape, finite values and bounds only:
the previous invented raw-linear contrast guarantee has been removed.
`motion_caption --fill-srgb '#24556A'` declares intent separately from `--fill`
raw controls. `motion_finish` requires `--top`, `--bottom`, `--top-srgb` and
`--bottom-srgb`; it does not infer intent from inherited controls. Python callers
likewise supply both intent stops and an explicit replacement top control.
There is no contrast exemption for a bright gradient toe.

Generated/restyled/finished candidate XML includes one versioned comment with
sRGB intent, distinct Motion stop controls/positions/midpoint, required 7:1 and
`renderedContrastVerified: false`. Shared text/position/sequence transformations
preserve this annotation; colour changes replace it. Hash the complete document,
including this annotation, when binding renderer configuration and output evidence.
The annotation is a declaration, **not evidence** of a control conversion or a
render approval. A native editor can discard comments: that creates different
bytes and cannot reuse a prior document fingerprint. Old templates and receipts
remain historical evidence, not retroactive approvals. Full rendered contrast,
intent correspondence, conversion, segmentation and coverage remain mandatory
before delivery; this candidate generator does not implement that delivery gate.

For signal-model sensitivity, `python -m replay.caption_signal FRAME.rgba64le
NEW_REPORT.json --width W --height H --intent '#24556A'` reads one full-range,
nonlinear BT.709 RGB-prime frame. Decode matrix/range only: leave transfer/primaries
unchanged and retain RGBA64LE precision. It reports camera inversion (the conventional
1.099/0.099 BT.709 curve), black-level-zero BT.1886 display gamma 2.4, and the
**historical, unverified for current apps** Apple video gamma 1.961 interpretation,
followed by standard sRGB encoding. No interpretation is selected automatically.
FFmpeg 9.0.2's zscale leaves zimg `scene_referred` at its default false, so its
BT.709-to-sRGB conversion uses a display EOTF, not camera inversion. Zimg also
uses continuity-adjusted coefficients; this diagnostic does not promise bit-exact
emulation of that library. A BT.709 tag does not resolve the viewing interpretation.

The report preserves 16-bit input codes and unrounded ratios. Its eight most
frequent opaque codes include decorations: they are **not** a glyph mask or
all-pixel/frame coverage. Selected intent is always checked against white at 7:1;
sample correspondence cannot approve a whole render. Check renderer configuration,
colour-managed viewing, gradient regions and frame coverage independently. Do not
choose the darkest interpretation merely to obtain a passing contrast ratio.

For a bounded colour-fidelity diagnostic, supply paired, independently converted
sRGB samples with `--rendered-colors '#24556A' '#28596D'` after the selected colours.
Every selected and supplied colour must still meet unrounded 7:1; paired channels
must match within one 8-bit code step (quantization allowance, not a perceptual
colour tolerance). Near-black samples cannot substitute for selected cyan merely
because they have stronger contrast. Missing pairs and invalid samples fail.
`sampled_colour_correspondence` exposes the same diagnostic to callers. Its result
covers supplied pairs only: it does not establish conversion, glyph segmentation,
stop identity, full-frame coverage or input/output provenance. CLI
`renderedContrastVerified` remains false even when the diagnostic passes. The
undocumented Motion-control conversion remains unresolved. Separating intent from
controls does not authorize accepting the dark default render or treating an
annotation or paired-sample diagnostic as full rendered verification.

## Mechanical job and workflow preflight

Use `scripts/tmux-job.sh WORK_DIR NAME COMMAND [ARGS...]` for detached trusted
commands under a stable, purpose-named `edits/work/` directory. It isolates tmux
configuration, uses explicit Bash and a short relative socket, and atomically
reserves the job name before dispatch. Logs/status are private and non-overwriting;
linked scratch/cache/socket/log/status destinations are refused. A failed dispatch
leaves `<name>.launch` intact: inspect it rather than deleting it or retrying with
another name. To inspect the socket, `cd WORK_DIR` then use `tmux -S tmux.sock`.
Dispatch success is not command completion. Supply an explicit command timeout;
this helper is not a process/resource scheduler or an arbitrary-command sandbox.

For trusted version-1 stage manifests, run:

```sh
python -m replay.workflow PLAN.json --state STATE_DIR --check
python -m replay.workflow PLAN.json --state STATE_DIR
```

The read-only check prints a stage-name dependency graph. Producer outputs matching
consumer inputs create edges. The entire graph rejects duplicate producers, cycles,
ancestor/descendant file declarations, linked outputs and nonregular existing outputs,
even when only a branch is selected. Artifacts and the control manifest must not
intersect the state directory; an output cannot overwrite the manifest. These checks
run before state creation. Existing hard-linked outputs require investigation rather
than automatic adoption or unlinking. The manifest remains version 1; its graph is
compiled once per invocation and shared paths are normalized once per compilation.

To run only a requested stage and **all** its ancestors, use repeatable targets:

```sh
python -m replay.workflow PLAN.json --state STATE_DIR --target captions --check
python -m replay.workflow PLAN.json --state STATE_DIR --target captions
```

Unknown targets fail; omitted targets mean the entire plan. Selection does not
bypass structural checks in other branches. Before executing any command, all
selected external inputs must exist as nonempty regular files; unselected external
inputs need not be ready. This permits work on an independent, explicitly selected
branch without mutating a blocked branch. Targeted success is not whole-plan or
whole-delivery completion, and a target cannot omit its declared prerequisites.

Execution preserves manifest priority among ready stages and uses the existing
fingerprint-bound stage locks, receipts and domain verifiers, including on reuse.
Missing/changed completed outputs still stop instead of authorizing a rerun.
A nonblocking `.workflow.lock` is held throughout the invocation in the state
directory, preventing another workflow from interleaving stages there. Lock files
are private, regular and single-link; they persist after release and must not be
deleted to force execution. Kernel ownership releases on process exit, not merely
on the appearance of a file. Separate state directories and direct stage callers
are not coordinated by this workflow lease; keep one state namespace for a job.
Independent nodes remain sequential, and a busy/failed stage stops that invocation.
No automatic retry, parallel executor, model download or quality-gate bypass is added.

The graph covers declared file dependencies, not hidden side effects, malicious
filesystem races or global output ownership across separate state directories.
Include helper code, model/config identities and required upstream evidence among
inputs, and use actual domain verification commands. `--check` reads metadata but
does not hash media or certify external input readiness, authorization, scratch
confinement of arbitrary commands, or final video quality. Changes to bound code or
configuration intentionally invalidate old fingerprints; never erase receipts or
bypass verification to make an old plan run.

## Quality gate

`make check` runs:

1. Shell syntax checks for the bootstrap and publication scripts.
2. Ruff format check and lint.
3. Strict mypy over `src` and `tests`.
4. Pytest with production-code coverage.
5. A minimum 90% statement-coverage check for every production file.

Use `uv run pytest tests/test_jobs.py` or another relevant test module while iterating, then run the full gate.
For isolated reports, set repository-local `COVERAGE_FILE` and run
`make check COVERAGE_JSON=edits/work/verification/coverage.json` after creating the
private parent directory. This redirects reports without changing the per-file gate.
The pytest recipe explicitly uses umask 022 so synthetic public-permission fixtures
are actually public, even when launched from the private job runner. This does not
change production file modes or the private parent directory/logs. Do not reduce gates, suppress errors, skip tests, or use real account data to make fixtures convenient.

`PYTEST_COMMAND` defaults to `uv run pytest -p replay.pytest_scratch`. A gate owner
may override this command to use the shared lifecycle controller at the same
pytest step; the existing coverage argument and per-file coverage check remain.
Keep the explicit basetemp under that gate's work directory and keep terminal,
coverage, source pins and failure evidence outside scratch. Cleanup must consume
an owner-approved, SHA-pinned disposition and fresh identity/open-handle checks;
a session-finish callback alone does not prove process termination or disuse.

Most network behavior is mocked. `test_ffmpeg.py` executes local FFmpeg/ffprobe against generated media. Publication tests use temporary Git repositories and the real local Gitleaks binary. No test needs an authenticated TikTok account, Android device, live browser login, or account secrets in CI.

## TranscriptViewer verification

```sh
cd TranscriptViewer
work="$(cd .. && pwd -P)/edits/work/transcript-verification"
mkdir -p "$work/tmp" "$work/cache"
export TMPDIR="$work/tmp" TMP="$work/tmp" TEMP="$work/tmp"
export CLANG_MODULE_CACHE_PATH="$work/cache/clang" SWIFTPM_MODULECACHE_OVERRIDE="$work/cache/swift"
cd ..
make swift-check
# Keep the report before the sanitizer build replaces the debug executables.
make swift-tsan
cd TranscriptViewer
swift build -c release -Xswiftc -warnings-as-errors
# Optional installation/restart, separate from the tests:
# bash build-app.sh
```

The optional Viewer installer explicitly binds both SwiftPM invocations to
`TranscriptViewer/.build` and uses a local manifest cache. Its temporary files,
cache, configuration, security state, module cache and profiles live under
`edits/work/transcript-viewer-build/`; linked or escaping working directories are
rejected before building. It retains the existing signed staging and exact-app
restart behavior, so do not use installation merely to check a build while an
agent owns the running Viewer. `tests/test_transcript_viewer_build.py` verifies
these command/environment boundaries with mocked tools and never operates apps.

`make swift-check` enforces strict Swift formatting and warning-free compilation,
then checks raw LLVM counters for at least 95% lines and functions in each
instrumented production source file. Both CLI executables are included alongside
the test binary; subprocess profiles are merged by SwiftPM. The report is saved
at `TranscriptViewer/.build/coverage.txt`. Available branch counters must also reach
95%; this Swift toolchain emits none, so branch coverage is explicitly unmeasured,
not 100%. Pure stored constants such as `Chrome.swift` have no executable coverage
map. No production file is configured as an exclusion. Gate regression tests reject
rounded-up percentages, invalid counters, and a failing file hidden by other files.
ViewInspector is an exact-pinned test-only dependency, not an app dependency.

Summary tests use synthetic transcripts: ten-minute boundaries, cross-boundary
utterances, missing intervals, cumulative captions, lossless balanced Unicode partitioning
(without isolated tiny tails or split graphemes),
hierarchical model requests, cancellation, stale results, provider failures,
cache invalidation after transcript changes, one bounded prose rewrite of a
model-generated draft (without dropping source input or stripping list markers),
refusal rejection, inline-bold normalization without stripping list markers,
numeric prose versus numbered lists, newest-first loaded-catalog batches, isolated failure counts,
queued-request cancellation, and duplicate-request cache reuse. Shared CLI/app tests
also cover command validation and catalog selection, persistent SHA/version/clock
validation, private permissions, corrupt/oversized/symlinked records, immutable
history retention on failure, cancellation, kernel lease contention/release, isolated
per-video failures and opt-in request/response diagnostics. Diagnostic ownership is
snapshotted across actor suspension; timing uses a monotonic clock. Filesystem fixtures
use the checkout's `edits/work/transcript-tests/` explicitly rather than Foundation's
OS temporary directory. Most tests inject model replies. The separate native
integration test actually generates and reloads a synthetic summary through the
shared CLI service, so this full gate requires an eligible Mac with Apple
Intelligence already enabled and its model available. Missing availability fails
rather than skips; tests do not enable settings or download assets. Native success
does not certify narrative quality. Never use private recordings as fixtures. Requests use fresh
sessions, bounded UTF-8 input and response budgets, plus exact token counts on
macOS 26.4+. The native provider uses Apple's documented
`permissiveContentTransformations` preset only for transforming supplied transcripts;
it is not a general-purpose chat entry point. Recognized String refusals fail rather
than being displayed as summaries. Regex-constrained and typed-repair experiments
were not adopted: the former failed in the native API and the latter can apply
default guardrails even in the text-transformation preset. Oversized or empty responses fail explicitly rather
than being published as successful summaries. Playback tests exercise AVPlayer's
selected/default rates while paused, playing, resuming and seeking without private
media. A generated silent FFmpeg fixture exercises native playback clocks and
subtitle suppression. SwiftUI tests assert actual control bindings, timestamp
jumps, search, cancellation, and error states; invisible native hosting and image
rendering verify player updates and timeline pixels without focusing a window.
Finder dispatch is injected and checked for the exact movie URL. These tests do
not verify audible pitch, real-media playback, an actual Finder reveal, or complete
visual/accessibility acceptance. Check those separately in the built app. The installer restarts only this checkout's running app after
signing the replacement; it does not modify videos or transcripts.

### Checking the running Viewer's screen

`scripts/viewer-debug-capture.sh` asks this checkout's running dist app (over a distributed
notification; nothing is clicked or focused) to write each visible window's accessibility report
to `edits/work/viewer-debug/<stamp>/<n>-<title>.json`, then screenshots those windows by window
number with `screencapture -l` and prints the folder. The report lists role, name, value and frame
(window points from the top left, as in the PNG) of every element, plus `smallTargets` (interactive,
below HIG's 28 pt macOS default) and `unlabeled`. Title-bar buttons, scroll-bar parts and zero-size
elements are left out. The app keeps the latest 10 captures. Screenshots need Screen Recording for the
terminal; an in-app drawing was rejected because it renders sidebar materials white. A rebuilt app
can wait on macOS's removable-volume prompt before its library loads: the user must answer it.

For an explicitly authorized existing-library run, build `transcript-summary` and
use `generate --library edits --diagnostics edits/work/summary-debug`. Add
`--include-text` only when local raw prompts/responses are needed; never publish
those files. `inspect --library edits` checks persisted output without inference.
Verify fresh-process reuse and the app's saved-result reload separately from native
generation. A valid JSON record and successful exit establish completed processing,
not factual accuracy. Production recordings are operational evidence, never test
fixtures. Do not approve macOS file-access or other permission prompts automatically.

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


## MPS-batched repair detector

`replay.subtitle_detector_mps.MpsTextDetector` runs the PP-OCRv3 DB network on Apple MPS from a
checksum-pinned `torch.export` program (`.pt2`, dynamic over batch 1–64 and 32-multiple grids
32–736) and lets OpenCV's own `TextDetectionModel_DB` decode the probability maps (embedded channel-0
network, production thresholds). It keeps the production two-view geometry and `polygon_mask`, so
masks are identical to `LearnedTextDetector` on measured real ROIs. It requires torch and an explicit
device, never falls back, and moves the CPU-loaded program to MPS. Build the pinned program once with
`edits/work/ocr-mps-detector-v1/build_exported_program.py` in an environment that has onnx and
onnx2torch; it bakes onnx2torch's constant Resize/Squeeze inputs (which `torch.export` cannot trace),
checks the program against onnx2torch on CPU and MPS, and writes `EXPORT-RECORD.json` (the deleted
TorchScript artifact it replaced matched it exactly and is kept there only by checksum). Pass the
recorded file and checksum as `detectorProgram` / `detectorProgramSHA256`. TorchScript (`torch.jit`)
is deprecated in torch 2.14 and is not loaded.


## Sampled-OCR source-caption repair (new generation)

`replay.subtitle_sampled_job CONFIG.json` repairs one source chunk without recognising every frame:

1. `subtitle_sampled_chunk.band_triggers` decodes only the subtitle band and marks abrupt changes
   (threshold = median + 6 MAD of the chunk's own frame deltas).
2. Native `video_text_recognize_session` reads every 6th frame plus the trigger frames only; each
   sampled frame's region comes from the existing `frame_region` rule.
3. The batched MPS detector (`subtitle_detector_mps`) supplies a per-frame region for every frame;
   `subtitle_sampled_admission` unites the bracketed sampled regions with the detector regions
   widened by 6 frames. Frames without any region stay unknown and unmodified.
4. `subtitle_video_fast.render_repaired_video_fast` repairs; the production detector inside it is the
   MPS one, whose masks are identical to `LearnedTextDetector` on measured real ROIs.
5. `subtitle_repair_audit` OCRs every 5th repaired frame (zero-based candidate clock). Any line with
   confidence >= 0.9 that overlaps that frame's search region fails the audit and escalates the whole
   chunk to a full-OCR attempt (`stride 1`); a second failure accepts nothing.

The receipt (`chunk-receipt.json`) records both attempts, the sampled frames, stage seconds and pins,
and states `regionsFromSampling`, `qualityApproved: false` and `deliveryComplete: false`. Region
fills are conservative search windows, not raw OCR; the repair stage still measures glyph pixels in
every frame. The native tool needs its checkpoint directories inside the checkout, so state stays under
`edits/work` while repaired videos go to the configured media root. Vision reports unreadable repair
smear with fixed confidences 0.3/0.5 and real text with 1.0, which is what the audit relies on; a
stride-5 audit samples frames and cannot prove absence.

### Sampled eligibility validator

`replay.subtitle_sampled_eligibility.validate_sampled_eligibility(receipt, admission, ...)` checks a
sampled-OCR chunk before a later stage consumes it: honest sampling provenance (`rawOcr` false,
`interpolated` true, fill rule, first and last frame sampled, no gap wider than the stride), exact
per-frame clock, `repair_spans` membership, unknown frames listed exactly, pins matching the source
hash, and an accepted final attempt whose audit passed with no residual lines. It certifies internal
consistency only; quality approval and delivery stay false.

### Sampled consumer adapter

`replay.subtitle_sampled_consumer.adapt(Adaptation(...))` validates each sampled chunk with the
sampled eligibility validator and writes `plan.json`, a pinned `settings.json` (same fields as the
base settings with the sampled run's native identity) and per-chunk `chunk.json`,
`candidate-config.json` and `repair/candidate.json` pointing at the accepted attempt's SSD media (no
copy). `subtitle_cleanup_assembly.prepare` then produces the manifest for
`delivery_source_assembly`. Quality, full decode and delivery flags remain false.

### Sampled survey evidence and its limits

`replay.subtitle_sampled_survey.survey(plan, assembly, output)` binds the per-chunk audits to the
assembled source by hash and writes `source-masks-none` evidence that discloses sampling, escalations
and rejected attempts. Its OCR-confidence audit cannot prove absence: a partially repaired caption can
be read below the 0.9 gate (Oct10 chunk 194). Do not publish its summary as completion evidence until
a colour-based residual check (unchanged caption-colour pixels inside detected text boxes) and review of
the flagged frames are attached.

### Colour residual gate and colour-key supplement (sampled path)

The OCR-confidence audit misses partially repaired captions (Vision read them at 0.5). The sampled path
adds an OCR-free check: `subtitle_colour_residual.unchanged_pink` counts caption-colour pixels that are
unchanged by the repair inside the original's recognised line boxes; `GATE` (40 pixels per audited
frame, one global constant) makes an attempt fail and escalate. `subtitle_colour_audit.colour_counts`
decodes the audited frames of source and candidate in lockstep. For chunks that reach the gate,
`subtitle_colour_support.ColourKeyDetector` ORs caption-fill pixels (plus outline) inside the line boxes
into the learned glyph mask (job config `colourKey`, receipt pin `colourKey: pink-key-v1`); scene pixels
of the same hue outside the boxes are untouched. Counts between 5 and 40 pixels remain a recorded
residual risk that needs human review; neither check approves quality.

### CTC label refinement (candidate only)

`replay.asr_ctc_label_refine` replaces a supported run of assembled ASR labels with CTC-aligned
onsets, and `replay.asr_ctc_label_refine_verify` proves the result differs from the pinned original
only as declared. Text is compared after both sides drop whitespace and the aligner's unvoiced
symbols; token and label edges that coincide form chunks (one token over several labels merges them,
several tokens over one label time it by the first token, crossing edges or an unvoiced-only label
inside a chunk are unmappable, unvoiced-only labels between chunks are carried unchanged). Ends are
derived (`derived-next-onset`, last label: original end if it shows a frame, else one frame, never past
the next original label) and recorded per label. A config without `block` and with only `deleteLabels`
removes labels that share their predecessor's onset. Nothing is approved; speech accuracy needs review.

With `block.mergeWeak` (opt-in) a run of weak units that lies between two supported contiguous units is
folded into the preceding supported label: the merged text is the exact concatenation of the original
labels, only the strong first token times it, the weak tokens are cited as `unusedWeak`, and a leading
weak label, a trailing weak label, a gap with another label, or a merged label longer than the focused
caption limit is never produced. The verifier requires `mergeWeak` to be declared for such citations.
Merged labels are human-review markers.

`replay.asr_window_substitute` (and `_verify`) replaces one collapsed label block, whose span must equal
a declared window, with the words of a bounded re-recognition (same pinned recognizer, input-PCM clock)
that lie wholly inside the window. Words crossing the window edge are listed as excluded and never
inserted, every other label is unchanged, and the recognizer's own word times are kept until a CTC
refinement re-times them. Excluded edge words can drop characters of the surrounding text; that needs
human review.

### Generic source-caption cleanup runner

`python -m replay.source_clean_run DESCRIPTION.json configs|run|assemble` cleans a whole recording from
one description (dated source and SHA-256, physical presentation clock, caption profile observations,
native tool, pinned detector, base settings, work directory under `edits/work/`, media root, ffmpeg/ffprobe,
chunk size, streams, free-space floor, sampling parameters). `configs` writes one chunk job per clock
chunk (colour key and colour gate on), `run` executes `replay.subtitle_sampled_job` with bounded
concurrent streams, skips chunks that have a receipt and stops starting chunks below the free-space
floor, and `assemble` adapts the validated chunks, prepares the assembly manifest, joins the media with
parallel probes and writes the pinned `source-cleanup.json`. It contains nothing recording-specific; the
per-recording caption profile and clock bootstrap are inputs. Outputs are candidates.

`source_clean_run BOOTSTRAP_DIR describe --defaults DEFAULTS.json` writes the description of a recording
from its bootstrap directory (`source.sha256`, `cleanup-bootstrap-v1/`) and one machine-wide defaults file
(native tool, pinned detector, ffmpeg/ffprobe, media root, budgets); the canvas is read with ffprobe.

`asr_ctc_label_refine` also accepts the opt-in block option `foldUnvoiced`. Unvoiced-only labels
(for example the long-vowel mark) that sit between two re-timed units are normally carried over at
their original onset, which can precede the measured onset of the label before them and is then
refused as reversed onsets. With `foldUnvoiced` such a label is instead appended, text and
provenance intact and listed under `foldedUnvoiced`, to the timed label before it; no time is
invented for a symbol the aligner does not align. The verifier requires the declaration, the exact
original labels and an unchanged recognized text. This is candidate timing evidence, not speech
accuracy approval.

`asr_window_substitute` also accepts the opt-in block option `foldShorterThan` (seconds, at most
0.1). A recognizer sometimes gives the pieces of one word a few milliseconds each, which paging
then refuses as a replacement chain delayed by more than one frame. With the option, each
re-recognized word shorter than the threshold is appended to the word before it: the text order is
kept, the earlier start and the later end are the recognizer's own times, and the folded words are
listed under `folded`. The verifier re-derives the folding. A short first word has nothing to fold
into and stays. This is candidate evidence, not speech accuracy approval.

The block option `dropSustained` (boolean) removes a block whose window holds only a sustained
vowel or noise: the re-recognized words crossing the window must all consist of the held characters
`あぁー～`, or the recognizer must hear nothing there. Any other recognized word makes the plan
refuse. The block is deleted without substitute labels and the verifier re-derives the decision.
This is candidate evidence; a person still confirms the deletion.
