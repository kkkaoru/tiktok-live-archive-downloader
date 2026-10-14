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

## Where to make changes

| Area | Implementation | Tests |
| --- | --- | --- |
| CLI and options | `src/replay/cli.py` | `tests/test_cli.py` |
| Creator selection | `targets.py` | `test_targets.py` |
| Notice parsing and API access | `recordings.py`, `tiktok_api.py` | corresponding `test_*.py` |
| Session reissuance and explicit login | `session_refresh.py`, `web_auth.py` | corresponding `test_*.py` |
| Acquisition and completion records | `acquire.py`, `jobs.py` | corresponding `test_*.py` |
| Per-recording edit config and required-stage gate | `edit_completion.py`, `delivery.py` | `test_edit_completion.py`, `test_delivery.py` |
| Scoped storage and media transfer | `core.py`, `download.py`, `media.py` | corresponding tests and `test_ffmpeg.py` |
| Optional capture/import | `addon.py`, `api.py`, `android_log.py` | corresponding `test_*.py` |
| Setup and publication policy | `scripts/`, `.gitleaks.toml`, `pyproject.toml` | `test_bootstrap.py`, `test_publication.py` |
| Shared native Mac processing and Vision tracking | `native/apple-pro-apps/Sources/` | `native/apple-pro-apps/Tests/` |

Unless fully qualified, implementation filenames are under `src/replay/` and tests
under `tests/`. Repository orchestration uses project skills in `.agents/skills/`:
`repo-jobs`, `disk-maintenance` and `replay-apple-workflow`. They call shared
implementations rather than owning alternate processing backends. Generic
`apple-*` skills describe app operations without depending on replay policy.

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

## Local audio revision processing

`replay.audio_restore` is an optional offline media tool, separate from downloader
acquisition. It accepts an existing processed delivery and creates a dated sibling
revision under `edits/`; it never reads `private/` or `downloads/` or overwrites an
original. Both restoration modes share a stable source-ID lock and refuse a second
dated delivery. Missing broadcast-date evidence stops before directory creation.
After the selected mode's checks pass and its receipt is saved, a hash-bound
selection record permits retiring only the unchanged legacy processed input video;
source transcripts, prior plans and partial work remain. The strict mode publishes
only after its transcript-based speech and instrumental checks pass. A per-bin squared Wiener mask is estimated from that video's Demucs vocal
stem and applied to the original phase; local blends use the separated voice, not
the music-bearing mix, as their level reference. Overlapping chunk estimates of
that same waveform use complementary sine-squared weights that sum to one;
equal-power weights would add an artificial 3 dB boost to coherent speech at seams.
An ebur128/SRT-derived downward
expander is tried on the adaptive mix with FFmpeg compand. Its application is
optional per video: if the measured reduction fails, or transcript speech in
that video's low-instrumental windows loses level/correlation, the unexpanded
candidate is kept instead of rejecting it. The measured floor and threshold are
level-shifted together in float audio and shifted back before verification.
A generic 0.5 dB voice gain is applied before encoding,
then speech is re-separated after AAC
and compared over the full transcript mask; instrumental residuals are checked in
every 60-second window across the entire output. Set `TORCH_HOME` to a repo-local
cache under `edits/` for processing; recordings and credentials are
not uploaded. Intermediate WAVs and verification reports remain with the local
revision and are not publication artifacts.

## Experimental editable caption chains

`replay.fcp_project --motion-reveal` is an opt-in candidate that replaces each strict
prefix-growth chain with one editable Motion title. Published controls are resolved
from the installed template, and paired linear keys preserve the original frame
boundaries. Unicode grapheme boundaries use the locked `regex` dependency; a stage
splitting a final composed character fails explicitly without normalization or fallback.
The default per-event output is unchanged. The candidate uses full-string anchoring,
requires the verified pop-in template configuration, and does not certify font fidelity,
fades, readability, performance, or delivery completion. Emoji glyph detail failed the
bounded native probe. A smaller title count is not performance acceptance.

`replay.motion_caption.animate_caption_template` writes the same explicit,
zero-based reveal samples into both text objects of a new in-memory Motion
candidate. It uses the observed ozml 5.13 scalar-curve format, exact rational key
times and collinear tangents by default; optional `smooth=True` uses zero-slope
cubic handles without adding keys or moving clocks. It refuses existing animation,
modified defaults, out-of-project clocks, oversized XML and unsupported layouts. It does not change
text, appearance, project duration or installed templates. Fade In is explicit:
fractions calibrated for pop-in do not automatically become word-aligned fades.
Compressor has rendered the candidate title and a copied reference curve directly,
including the fade/pop distinction, without FCP Share. A separate synthetic render
of generated keys from `chain_reveal_samples` retained all 120 frames at exact
30fps and all four planned prefix-change frames. This checks the shared clock,
not arbitrary text, layout or word-aligned fades. Those low-resolution probes are
not colour, alpha, performance or delivery acceptance. The automatic delivery
path still defaults to `styledCaptions`. `replay.native_render --foreground-manifest`
can instead wire explicitly supplied, hash-bound per-part Motion renders to the
existing native `foregroundVideoPath` compositor, without also drawing styled
captions. This does not generate or verify those Motion renders, nor does it replace
the separate editable full FCP project. The native foreground sidecar cannot
represent its animation; it must never be presented as that editable project.
A separate 1080x1920 synthetic Motion export used four sparse keys per sentence
with Fade In and zero-slope handles: all six reveal ramps increased alpha across
eight frames, with stable holds, empty gaps and exact 120-frame/4-second clocks.
This verifies the small smooth-animation contract, not ASR integration, arbitrary
glyphs, final colours, FCP-equivalent curves, loading performance or completion.

`smooth_chain_reveal_samples` generates sparse fade ramps from validated contiguous
caption-prefix clocks: at most two keys per recognized stage, shortened to that
stage's support, and no early reveal of a later prefix. The final ramp must reach
full reveal on a visible frame, not at the title's exclusive end; a one-frame
final stage fails rather than silently hiding its word. Fractions use complete
graphemes and reach 1 at the final prefix. This differs from the pop-in calibration.
`set_caption_reveal(..., smooth=True)` writes FCP's ease/smooth keys; callers must
explicitly configure and verify both text objects' Fade In. These shared helpers
are opt-in building blocks, not automatic full-workflow or real-ASR acceptance.
`build_fcpxml(..., motion_reveal=True, motion_smooth=True)` (CLI:
`--motion-reveal --motion-smooth`) now uses those sparse samples and smooth keys
for growing chains and singleton titles. The existing linear/pop-in mode remains
unchanged. Smooth mode requires a separately checked Fade In template; no mode
changes the installed template or certifies editable FCP playback or delivery.
For original-audio diagnostic timelines, explicit `source_audio=True` (CLI
`--source-audio`) retains audio on each source clip's own selection clock.
It cannot be combined with output-clock `voice`/`--audio`,
and a source without an audio stream fails. This mode does not remove music;
the default stays silent without processed voice, whose duration guard is unchanged.
Video-only source cuts additionally use `audio-channel-source`
with `active="0"`, as observed in FCP's own export: `srcEnable="video"` alone was
not preserved on import and allowed original audio to mix with processed voice.
The shared probe requires a single audio stream with a validated channel count
(1–64); missing or ambiguous channel metadata is rejected, never assumed stereo.
Connected processed voice is not deactivated. Structural tests and an exported
component setting are not audio acceptance: generated imports and rendered
waveforms still need independent checks.

`place_captions_above_heads` applies the same strict, sampled FFmpeg head-height
policy to an editable FCP `Timeline`: centre X, vary Y, keep each prefix-growth
run at one height. It preserves cuts, source masks, caption text/clocks and style.
Only the calibrated 1080×1920/30fps style and contiguous frame-aligned cuts from
one source are accepted. The caller must bind head samples to that source's
presentation clock; the adapter does not detect faces or certify the installed
Motion template's rendered bounds. Missing headroom fails rather than reverting
to a central diagnostic position. No existing export enables it implicitly.

`build_caption_sequence` assembles at most 32 non-overlapping sentences over at
most 120 seconds for a caption-only Motion probe. It preserves template styles,
remaps generated object IDs, uses sentence-local reveal keys and explicit global
frame support, and does not register a title or replace editable FCP media.
Unverified geometry, existing animation, clipped final reveal keys, out-of-range
IDs and oversized XML are not accepted as a successful rendering.
Opt-in `match_fcp_scale=True` converts the observed template's font size, outline
widths and rim shadow geometry by output-height/template-height. Default output
remains unchanged; placements stay in output units and colour, opacity and reveal
clocks are untouched. A bounded native comparison aligned sampled fill bounds
within two pixels, not every pixel, arbitrary layouts, contrast or delivery quality.

`replay.caption_native_layout.NativeCaptionLayout` connects the existing native
`caption_text_measure` CLI to `CaptionPolicy.measured_layout` and POS-aware line
breaks. It caches actual single-line bounds, batches bounded uncached requests,
and combines two lines using an explicitly supplied, independently observed Motion
line advance rather than assuming ASS leading matches Motion. Equal character
counts need not have equal width. Calls never shrink fonts or alter token clocks.
When a line overflows, the shared measured wrapper can prefetch all actual
POS-boundary line alternatives through `prepare_lines`. The integrated measured
planner uses this hook, so native requests pack uncached alternatives rather than
launching a new process for each split. Measurements, chosen breaks, 16-text/128-unit
native batches, cache limits and job call/deadline budgets remain unchanged.
Prefetch failure propagates; it does not authorize guessed widths or a fallback.
Its diagnostic CLI accepts an executable, new private directory, `--text` values,
font/outline/region parameters, `--line-advance`, and hash-bound `--evidence` paths.
Requests, responses and failures remain; existing directories are refused. Callers
verify bound inputs before publishing a plan. These are layout candidates, not
proof of calibration correctness, Motion positioning, face clearance or contrast.

`python -m replay.caption_pixels FRAME.rgba NEW_REPORT.json --width W --height H`
measures one bounded decoded RGBA8 frame: exact nonzero/opaque alpha support and
the eight most frequent opaque RGB code values. Sparse edge pixels are retained;
empty support remains explicit. It fingerprints input bytes, rejects changed or
incomplete frames, and never overwrites reports. Callers must independently bind
frame times and colour-managed decoding; this is not automatic sRGB conversion,
semantic glyph segmentation, contrast certification or delivery approval.

`python -m replay.motion_stages TEMPLATE PLAN.json NEW_OUTPUT.moti` prepares a
bounded caption-only sequence with one complete visible string per stage, rather
than anchoring each prefix against the future final string. Plans specify integer
`frames`, `width`, `height`, `rate`, `expectedText`, and `placements` containing
`text`, `startFrame`, `endFrame`, `x`, `y` in Motion origin units. Each stage uses
two identical full-reveal keys, no fade, and the unchanged 32-stage/120-second/XML
limits. This retains supplied line breaks and clocks; callers must derive them
from measured layout and recognized timings, bind inputs, and verify actual
centering, head clearance and contrast. It does not infer those properties or
replace the editable FCP project. Inputs and existing outputs are preserved.

`restyle_caption_template` copies the known two-object template with explicit
font size, gradient, border/rim and shadow controls. It preserves text, font face,
reveal keys, timing and transforms, and retains the version and XML-size guards.
Sizes remain template units before optional FCP canvas normalization. This is
experimental style preparation, not measured Motion colour, contrast or geometry;
verify native output before using it in a delivery. No installed template is written.

`replace_caption_text` copies both unanimated caption bodies and their single
style-run lengths only when the expected old text matches. The bounded ozml 5.13
contract supports BMP text, combining marks and newlines without normalization;
character tables, mixed styles, control characters and supplementary-plane text
are refused. Japanese multiline text with a combining dakuten rendered at the
same four exact frame boundaries in a 120-frame synthetic Motion export. This
neither proves arbitrary glyph coverage/layout nor authorizes flattened editing
projects; generate new reveal samples from the replacement text before rendering.
`position_caption_template` sets both default local origins using observed
version-matched Position controls, refusing pre-existing transforms and non-finite
or excessive coordinates. A 1920x1080 synthetic render moved every frame's alpha
bounds by (+200, -100) pixels for Motion coordinates (+200, +100), preserving
width, height and all 120 frame clocks. These are origin coordinates, not glyph
bounds or a head-tracking/baseline calibration; other canvases need verification.

## Shared transcript summaries

`TranscriptViewer/Sources/TranscriptCore` owns lossless transcript partitioning,
hierarchical narrative generation, response validation and timestamped sections.
`TranscriptSummaries` owns the Foundation Models provider, serialized in-process
service, persistent result format, opt-in diagnostics and testable CLI execution.
Both `TranscriptViewer` and `transcript-summary` depend on this same module; the
GUI's observable presentation state stays in `NarrativeSummaryModel.swift`.

A `<transcript>.summaries/` directory contains immutable JSON revisions and an
atomically replaced `current.json`. Loading validates the pipeline version,
SHA-256 of the complete derived transcript input, every interval clock and narrative
text. It does not load partial results or silently ignore malformed records.
Generation failure retains previous completed output. The version must be bumped
when changing the persisted processing contract. Directories and files are private;
symlinked/public storage directories are rejected. A stable `.generation.lock`
uses a nonblocking kernel lease to prevent the app and CLI from generating the same
transcript concurrently; ownership ends when its file handle closes, including on
process exit. Do not unlink the lock while a producer could own it. Completed
records remain readable during regeneration. Atomic pointers prevent partial reads,
and the old pointer is retained before replacement, including an explicitly
regenerated corrupt record. Different transcripts can still contend for the native
model across processes; avoid concurrent generation when investigating model-busy
failures.

Diagnostic directories have unique run IDs, private per-video input/outcome records,
and immutable request/response/error records. They contain paths and input hashes;
full model instructions, prompts and responses require explicit `--include-text`.
These records are user data, not distributable source or semantic approval receipts.

## Supplemental tools

`list`, `download`, `import-har`, `import-android-log` operate on explicitly observed candidates. `capture` uses the optional mitmproxy dependency; `web-login` saves scoped browser state but does not by itself verify replay access. These are not dependencies of normal authenticated listing/download or HTTP cookie reissuance.
