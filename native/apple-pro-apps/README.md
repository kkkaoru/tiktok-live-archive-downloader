# Shared native Mac processing

This is the canonical Swift package for reusable native processing. Historical
source snapshots remain unchanged for provenance, not as runtime dependencies.
The package accepts local paths and typed requests; it must not contain recording
IDs, broadcast dates, another checkout's path, or recording-specific branches.
The pinned `Package.resolved` and existing SwiftPM targets are retained.

## Ownership

| Work | Shared implementation |
| --- | --- |
| Native decoding, audio measurement, SoundAnalysis, OCR | `Sources/ProAppsCore/` |
| Sequential face detection and tracking | `FaceTrackProbe.swift`, `FaceTracking.swift` |
| Native edit recipes, encoding, overlays, Motion/FCP interchange | `Sources/ProAppsCore/` |
| Validated CLI/MCP dispatch | `Sources/AppleProApps/` |
| Source-bound collection, ASR, separation, cuts, captions, rendering and completion | repository `src/replay/` |
| Recording paths, evidence, plans, partial results and model assets | ignored repository-local data directories |

Use the existing `replay.workflow`/`replay.stages` runner for trusted execution
manifests, rather than adding a second scheduler. Retain its input/output binding,
verification and non-overwrite behavior. Model choice, speech/caption clocks,
source geometry, resource budgets and measured encoder capabilities are inputs to
shared policy, never tests for a particular recording name. An observation or
successful native render does not pass `replay.edit_completion` by itself.

## Build and tests

`app_open_document` accepts WAV, AIFF (`aif`/`aiff`), CAF, M4A and MP3 for
Logic Pro, in addition to its project and MIDI formats. The selected installed
edition receives the unchanged source through LaunchServices. This is delivery
only: inspect Logic's resulting project, regions and dialogs before playback.
No track placement, tempo adaptation, decoding or playback is implied by the
response; other applications retain their existing document allowlists.

From the checkout root:

```sh
bash scripts/native-swift.sh build --jobs 4
bash scripts/native-swift.sh test --jobs 4 --enable-code-coverage
```

The wrapper uses `edits/work/native-swift/` for scratch, caches and profiles and
keeps the lockfile authoritative. Set `REPLAY_NATIVE_WORK_NAME` to a simple basename
to use a separate child of `edits/work/`, preserving binaries bound to existing
evidence. Absolute paths, traversal, separators and linked working storage fail. Native writers must use
`Files.temporaryDirectory()`, not Foundation's environment-insensitive default.
Do not run old scratch scripts or infer deployment approval from a successful
build. Full tests require explicitly configured, verified local model fixtures;
WhisperKit test fixtures do not replace the selected non-turbo MLX caption model.
No model download, login, OS permission change or app registration is authorized
by these commands. Inspect integration-test requirements before execution.

## Read-only font identity supplier

Build the independent `font-identity-probe` target using the same wrapper and a
separate `REPLAY_NATIVE_WORK_NAME`; do not replace evidence-bound executables.
It reads one bounded JSON request from stdin and writes JSON to stdout:

```json
{"fontName":"Helvetica","pointSize":24,"texts":["Hello","字幕"]}
```

The requested PostScript name must resolve exactly. CoreText supplies the actual
font URL, family, face, variation axis identifiers/values and metrics, plus the
font identities and UTF-16 ranges of actual shaped glyph runs for each input
string. It does not guess URLs from names. Input limits are 64 KiB, 256 strings,
120 UTF-16 units per string and 4096 units total; controls/newlines are rejected.
Use separately supplied single lines. Sizes are points (1–1024), not ASS sizes.
There is no file writer or temporary-file API: request/response transport uses
stdio only. For instrumented builds, explicitly put `LLVM_PROFILE_FILE` inside
the repository. No registration, installation, download or app launch occurs.

Callers must hash the exact returned files read-only, validate their allowed
font-storage scope, bind request/response/source/build/tool identities and retain
an external deadline. These are current CoreText identities, **not proof that
Motion used the same font or that future caption reflow uses the same fallback**.
The existing `caption_text_measure` protocol remains unchanged. The separate
Python `replay.motion_resource_inventory.inventory` API inventories bounded XML
references recursively within explicit roots, preserving missing/symbolic/unknown
references. Its lexical inventory never certifies renderer resource closure.
A missing resource must not become an implicitly empty `resolvedAssets.resources`.

Focused tests: `scripts/native-swift.sh test --enable-code-coverage --filter
FontIdentityTests` (with the same isolated work name). Tests use OS fonts and
synthetic strings, never private recordings.

## Atomic popup selection

`ui_popup_select` takes the existing app, window and element locators plus an
exact `title`. It resolves one enabled `AXPopUpButton`, opens its menu once and
keeps that handle while waiting up to two seconds for a unique enabled direct
menu item. It presses at most once, then observes value/children for up to two
seconds without repeating the press. Each level has a 256-child limit. No app
activation, keyboard, filename assignment or Save action is included.

The `popupSelection` receipt records attempted opening/selection, bounded menu
child titles/roles, before/after values and distinct missing, ambiguous,
unreadable, stale-popup and backend-failure states. A post-dispatch error can
still have an effect: inspect the current app before any new attempt. A matched
popup value is not proof of a canonical directory, saved file or completed
export. Save workflows must separately verify the complete directory ancestry,
unused basename and resulting media. The existing `fcp-save-plan` remains the
read-only planner; this is a shared UI primitive, not another save backend.

## Bounded caption measurements

`caption_text_measure` accepts a complete single line up to 512 UTF-16 units;
two-line inputs retain the 120-unit limit. The request still permits at most
256 texts / 4096 total units / 16 megapixels. Each measured dimension is bounded
to 16384 pixels. The schema's UTF-8 byte ceiling is only a transport bound;
native UTF-16 validation remains authoritative. Long lines are measured intact,
without wrapping, truncation or invented timing. Bounds are not Motion layout
approval, and inputs exceeding the geometry budgets still fail.

## Degenerate native text envelopes

Vision can return a zero-area text envelope while still returning positive-area
character observations. `TextRectangleProbe` records these in
`diagnostics.degenerateTextObservationCount` rather than inflating the envelope.
Every character still passes the original finite, positive-area, in-bounds checks;
invalid nondegenerate envelopes still fail. In explicit text-region mode,
`replay.subtitle_vision` preserves the valid region support and unions only the
measured character pixels when a degenerate envelope is reported. It logs the
condition and refuses missing/empty/invalid character support. No OCR rectangle,
inferred line envelope, clipping, or absence/quality approval is substituted.
Independent removal and output-quality checks remain required.

## Exact-image text context

Video OCR (`measure-media video_text_recognize`) retains the display-oriented
decoded `CGImage` for full-frame recognition and uses CoreGraphics cropping for
an explicit region. It does not create a CoreImage rendering context. Every
compute stage exposed by Vision's text request selects an available CPU device;
an unavailable CPU stage fails rather than silently choosing GPU execution.
Japanese/English accurate recognition, disabled language correction, original
frame timestamps and full-frame top-left observation coordinates are retained.
Requested OCR seconds must be exactly representable on the existing 60000-tick
sample clock. The shared integer-time conversion avoids floating conversion to
the preceding tick; returned timestamps always remain AVFoundation's actual
timestamps. The configured CPU request is reused sequentially within the bounded
batch, without carrying request state between calls or increasing sample limits.
This compute policy does not certify recognition quality or exhaustive subtitle
absence. Repository-local scratch and actual process descriptors still need
verification by the invoking workflow.

`measure-media image_text_recognize_batch REQUEST.json` accepts `{"paths": [...]}`
with 1–8 explicit local PNG regions. Each image retains the two-megapixel budget;
recognition is sequential, Japanese/English, without language correction. Results
retain input order, top-left pixel geometry and explicit clipping. Limits are
64 lines per image and 32 KiB recognized text per batch; results are not truncated.
These are OCR contexts, not glyph-support masks, media clocks, exhaustive absence
proof or repair approval. No source modification, upload, screenshot or app launch
occurs. `replay.subtitle_prepared` uses same-frame observations only to prepare
bounded detector inputs; repair support still comes from measured Vision geometry.

## Sequential face observation

`measure-media video_face_track REQUEST.json` shares the validated CLI/MCP
implementation. The request has this shape:

```json
{
  "path": "/absolute/path/to/verified-source.mp4",
  "plan": {
    "startSeconds": 0,
    "durationSeconds": 1,
    "maximumFrames": 120,
    "redetectIntervalSeconds": 0.5,
    "sourceTimescale": 30000
  }
}
```

The values above illustrate a bounded request, not a footage-specific preset.
Choose windows and frame budgets from verified source clocks. `sourceTimescale`
is the verified rational source-clock denominator, not an inferred frame rate.
Omitting it retains the legacy edit grid; requests not exactly representable on
the selected grid fail before decoding rather than round across frame boundaries.
There is no camera input: AVAssetReader supplies display-oriented frames to Apple's
`VNDetectFaceRectanglesRequest`, `VNTrackObjectRequest` and
`VNSequenceRequestHandler` sequence. Every returned frame retains its rational
presentation timestamp. Redetection runs at the configured interval and immediately
when the observed face count drops. A still-unobserved interval does not trigger
redundant detection on every frame; `detectionAttempted` distinguishes attempts
from tracking-only frames. Tracking confidence at or below 0.3 is rejected.
When every request has terminated, the exhausted sequence is released instead
of advancing terminal-only requests on another frame. Missing geometry remains
missing; only a subsequent actual detection can seed a new sequence. Errors from
active native operations still propagate without retries or synthetic recovery.
A finite, positive tracking box that leaves the image terminates that track and
increments `outOfFrameTrackingObservations` on its actual source frame. It is not
clipped or returned as a face. Redetection is requested immediately; without a
valid detection, position remains unknown. The count includes rejected seeding
observations when a detector result initializes tracking on that same frame.
Finite, positive detector boxes extending outside the image are likewise not
accepted or seeded: `outOfFrameDetectionObservations` records them separately.
Malformed extents and absent framework result arrays still fail. Missing geometry
remains missing. Either rejection count signals incomplete geometry for caption
placement, even when another valid face measurement exists in that frame. Rectangles are
face measurements, not identity, hair boundaries, subject absence or safe caption
placement. The caller must bind results to unchanged source/tool hashes and
independently check coverage and caption geometry. Existing process/output limits
still apply; a long request can fail rather than silently truncate.
Invalid-geometry errors retain the detection/tracking method, normalized rectangle,
confidence and exact source PTS. They do not clamp boxes, substitute positions or
approve partial windows. Keep real-source diagnostics private alongside the
source-bound attempt; they are not public test fixtures or publication artifacts.

### Opt-in raw object evidence

The same `video_face_track` CLI/MCP request may explicitly set
`"objectEvidence": true` beside `path` and `plan`. Omitted or false retains the
ordinary array of `FaceTrackingFrame` objects and its original fields. True
returns an array of separate wrappers containing `faceFrame` and `objects`.
`faceFrame` uses the unchanged confidence, geometry, rejection, redetection and
termination policy; rejected rectangles never enter `faces`. The wrapper's
`faceObserved` describes that original face array, not object eligibility.

Each raw object records its actual source presentation, operation stage, native
observation UUID, confidence, unclipped normalized lower-left and display top-left
rectangle. `visibleIntersection` and ordered `clipSides` are separate geometric
derivations. `disposition` records accepted, confidenceRejected or
outOfFrameRejected under the unchanged policy. Detector evidence has no seed;
tracking evidence links to its actual detector seed UUID and source presentation.
This is a request-local appearance chain, not persistent identity. The source,
binary, request, native revision/environment and code identities must still be
bound by the caller's immutable evidence; this response does not hash media.

At most 96 raw objects can occur on one source frame (32 advancing, 32 detected,
32 newly seeded), and one opt-in request retains at most 7200 raw objects total.
Exceeding that additional evidence budget fails without truncation. Source-window,
frame, bitmap, CPU-only execution and disposable-child limits remain unchanged.
Malformed raw geometry/confidence fails rather than inventing values.

`objectEvidenceOnly` is true; `semanticHeadVerified` and
`captionPlacementApproved` remain false. This operation captures actual values
before rejection; it does not continue an ended tracker, provide head/hair Y,
reuse an old face position, repair a missing interval or approve caption placement.
Historical responses are unchanged and cannot retrospectively supply a discarded
rectangle or tracker state. Native tests/builds and an actual separately admitted
probe are required before deploying a new executable or accepting real evidence.

### Independent partial object evidence

`partialObjectEvidence: true` additionally requires `objectEvidence: true` and
an explicit `partialObjectPlan` containing `maximumSeeds` (1–256),
`minimumVisibleFraction` (greater than zero, at most one), and `boundaries`.
Each boundary has an exact rational `presentation` (`value`, `timescale`) and
a `reason` (`sourceCut`, `scene`, or `zoom`); at most 128 unique ordered boundaries
must match actual decoded PTS inside the window. An empty list means no caller
boundary was supplied, never proof that no scene/zoom change occurred.

After an ordinary confident tracker rejects a clipped current rectangle, an
independent CPU-only request may initialize appearance on that same current image
using its visible intersection. The input box and new seed UUID are explicitly
derived metadata; the original rejected rectangle, detector seed/clock and actual
new Vision observations remain separate. Every `partialObjects` record identifies
current source PTS, support/termination status, original rejection, derived seed
(including the current raw input), and current raw result when one exists.
Neither the intersection nor an old box is an accepted face or current observation.

Low confidence, full exit, insufficient visible support, ambiguous objects,
seed budget exhaustion, actual face reacquisition and explicit boundary/geometry
resets terminate this separate state. Clipping may reseed once on its current image;
a clipped initialization result is unsupported rather than recursively retried.
Old detector associations cannot cross a declared boundary. Missing native results
remain absent, and at most 7200 partial records are retained without truncation.
Legacy face frames/rejection counts/termination and omitted/false output are unchanged.
No head/hair Y, identity, UI exclusion or placement quality is inferred;
`semanticHeadVerified` and `captionPlacementApproved` remain false. This capability
supplies separate derived-policy candidate evidence for a later explicit consumer.

## Opt-in foreground measurement

`measure-media video_foreground_measure REQUEST.json` accepts a local `path` and
1–8 ordered `samples` of `{ "value": 1001, "timescale": 30000 }`. These are exact
rational source PTS, not rounded seconds or inferred frame numbers. Actual decoded
PTS must equal the request; source/display budgets and the measurement child
deadline remain unchanged. Direct CLI execution still requires an external deadline.

The opt-in operation uses Vision foreground-instance revision 1 and the same image
handler to produce a source-sized union of all observed instances. It returns local
instance labels and quantized L8 nonzero-pixel counts/tight top-left bounds at the
fixed threshold one. Bitmap layout, row storage and full-source extent are checked;
padding is excluded. Pixel storage is bounded to 64 MiB, separately from Vision's
internal memory. No mask image is exported or source file changed.

Absent observations, empty instance sets and empty quantized support remain distinct.
Labels are not persistent identity, and a mask rectangle is not filled-pixel coverage,
hair geometry, subject absence or safe caption placement. `captionPlacementApproved`
is always false. Existing tracking and rendering do not call this operation. Bind
source, executable, request and response hashes independently. Implementation and
synthetic verification are in progress; this is not deployment or delivery approval.

### Optional face-associated foreground support

`video_foreground_measure` additionally accepts `headFace` with `presentation`
(`value`/`timescale`), integer top-left `x`, `y`, `width`, `height`, and an explicit
`minimumFaceCoverage` fraction in `(0, 1]`. This opt-in permits exactly one sample
and requires the face and decoded frame to have equal rational source PTS. The
caller must bind the face detector's source identity; the operation does not
invent a face or verify the supplied detector result.

The alternative `headFaces` array accepts 1–8 objects of the same shape, one for
each `samples` entry in the same order. Each face presentation must equal its
sample's exact rational PTS; equivalent rational representations are accepted.
`headFace` and `headFaces` are mutually exclusive. Count, clock and ordering
validation happens before source reading. The existing single-face request is
unchanged. Frames are measured sequentially with one source generator and image
context, and each actual decoded frame still passes the original geometry and
mask-storage budgets. Face coordinates are never interpolated or copied between
frames. Requests without either face field retain their existing response shape;
absent face evidence is not an inferred head or approved caption position.

One Vision observation supplies both the union and each individual instance mask.
Four-connected components overlapping the face are associated mechanically.
Disconnected HUD support is excluded; ambiguous components, insufficient face
support and frame-edge contact at or above the face produce an unknown result.
Torso contact below the face does not imply head clipping. The optional `headTop`
report contains measured component bounds, face coverage and a candidate `topY`.
This is foreground geometry, **not semantic hair detection**: attached hands,
hats or props can alter the upper edge. `semanticHeadVerified` and
`captionPlacementApproved` remain false. No manual head apex or face-height
multiplier is used, and no position is copied to another frame.

Sufficiently face-supported `candidate` and `clippedSupport` responses retain up
to 256 measured `upperContour` pixels within the face's horizontal span,
preserving an actual minimum-Y sample when reducing the point count.
`interiorRegions` contains up to 256 16-by-16 tiles whose every pixel belongs to
that same component; these tiles lie within the face's horizontal span and above
its bottom. They support texture measurements without treating a bounding
rectangle as foreground. Ambiguous, absent and insufficiently face-supported
components still emit no contour or tiles.

The diagnostic `upperSupport` reports `status` (`unavailable`, `interiorMeasured`,
`canvasClipped`), nullable `minimumY`, contour top/bottom/side-edge contacts, and
nullable whole-component top/side-contact flags. An actual interior upper arc can
remain measurable when the component contacts a lateral frame edge elsewhere.
This does not turn the whole-head `clippedSupport` result into `candidate`:
`topY` remains null, and both semantic-head and caption-placement approvals remain
false. Contour points at a canvas edge are explicitly `canvasClipped`.
These foreground pixels support approximate upper geometry, not verified hair
semantics or a complete unclipped head. Attached hands, hats or props may affect
the measured arc. No manual apex, bounding-box corner, face-height multiplier or
placement approval is substituted.

Retained mask providers, including padding/backing storage, have a cumulative
64 MiB limit. Ambiguity needs at most two components per mask; component traversal
also uses bounded visited and queue storage. Vision's own allocation is separate.
Tests use synthetic masks, decoded fixtures and injected operations; real-source
observations require separate, bounded execution and provenance evidence.

## Sampled head-observation provenance

`video_head_detect` now includes `actualPresentationValue` and
`actualPresentationTimescale` from the decoded frame, plus `boundsClipped` for
any face or whole-person rectangle intersected with the image boundary. Existing
region values are unchanged. Strict consumers must match the rational source PTS
and reject clipped geometry for placement. An older report without these fields
cannot certify unclipped geometry or an exact source-frame match. Whole-person
bounds are geometric proxies, not verified hair boundaries or subject identity.

### Opt-in paired face-detector diagnosis

Set `compareFaceDetectors: true` on `video_head_detect` to compare legacy and
modern Vision detectors on the **same decoded CGImage**, without changing tracking.
The optional `faceDetectorComparison` report retains raw confidence, unclipped
normalized lower-left and display top-left rectangles, separate confidence/geometry
eligibility, actual request revisions, image producer and OS provenance. Default
head reports omit it. The comparison also retains generator bitmap metadata and
legacy detection on a baseline reader bitmap at the same rational PTS.
The baseline reader shares 32BGRA setup and preferred-transform application with
tracking, but retains the old default CIContext output conversion for comparison. Only the first sample in one exact sample-clock tick is
read; missing or mismatched samples fail without scanning for another frame.
Reader metadata includes its actual PTS, pixel format, transform, bitmap/color-space
properties, request revision and raw candidates. Cancellation after one sample is
not a full-stream decode. Neither producer evidence nor an empty result approves
placement. No additional modern detection, tracking or seeding is performed.
The opt-in additionally records `readerPreservingSourceColor`: a second independent
one-sample reader at that exact PTS, rendered as RGBA8 in the decoded CIImage's own
RGB color space. Missing/incompatible source color metadata fails rather than
assuming a profile. This controlled diagnostic records its output bitmap and raw
legacy detection separately. Normal sequential tracking now uses this same
source-profile-preserving RGBA8 conversion for both detection and tracking, rather
than implicitly converting to DeviceRGB. It requires a decoded RGB profile; no
profile is guessed and no detection threshold or clock policy changes. Diagnostic
agreement does not establish pixel equivalence, color accuracy or placement approval.
Requests must round-trip exactly on the existing sample clock
and match the actual rational PTS; unsuitable clocks fail instead of rounding.
Existing sample, image, observation and disposable-process limits still apply.
A mismatch, missing result array or framework error fails the request; an empty
successful result is not an error. Eligibility is not caption-placement approval.
Bind the request, response, source and independently built executable hashes as
usual. No detector confidence is copied from a synthesized tracking observation.

CLI failures from recognized native framework domains now retain only their domain
and numeric error code. Unknown domains are replaced with `unclassified`; localized
descriptions, paths and `userInfo` are never included. Tracking failures additionally
identify the exact operation (`faceDetection`, `trackingSeed`, `trackingAdvance`)
and validated rational source PTS. Cancellation and structured application errors
remain unchanged; a failed attempt is still a failure, not permission for an
automatic retry.

## Experimental reference-movie interchange

`reference-media REQUEST.json` creates a private, non-overwriting MOV header;
media samples stay in the original files. It is **not** a standalone delivery or
proof that a renderer composites multiple tracks. The request specifies a canonical
`rootDirectory`, an `outputPath` inside it, a positive integer `timescale`, and
`segments` (1–1024). Each segment supplies `sourcePath`, rational `sourceStart`,
`duration`, `outputStart` (`value`/`timescale`), and `layer` (0–15; higher is in
front). A base layer is required. Selections must be ordered without overlap on
each layer, exactly representable, positive in duration and within a six-hour
output. Sources must have one self-contained, unfragmented, unrotated video track,
a common canvas, and a compatible codec/media clock per layer. No audio, resizing,
source alteration or implicit rate conversion occurs.

All sources must resolve inside the explicit root. Nested references are refused
using both AVFoundation reference restrictions and the track's asynchronous
`isSelfContained` property: restrictions alone do not reject metadata-only copying.
The result lists dependencies and explicitly reports `selfContained: false` and
`rendered: false`. Retain and hash-bind the unchanged originals, use an external
process deadline, and verify actual target-renderer frames, layers and clocks before
using this intermediate in production. Full native and renderer acceptance remain
unverified; unit tests alone do not establish capacity savings or FCP editability.

## Save-panel planning without replacing the deployed FCP tools

`fcp_save_plan` reads a named Save window and returns one typed `UIRequest`; it
does not execute that action. `UIAction.open` maps to an element's advertised
`AXOpen`, which is distinct from `AXConfirm`. Each action must be followed by a
fresh observation. This permits column-folder opening when the actual cell or
text field exposes `AXOpen`, without keyboard or pointer synthesis.

The independent offline CLI is `apple-pro-apps fcp-save-plan /absolute/request.json`.
Its JSON request contains `context` (`directory`, `fileName`, `observedDirectory`,
`window: {title: ...}`, `saveDispatched`, optional `bundleID` and
`navigationAncestor`), `snapshotPath`, `snapshotSHA256`, and
`historicalInspection`. The snapshot is the decoded native `ui_inspect` payload
with `elements`, not an Executor envelope. Hash mismatch, noncanonical paths,
existing output, ambiguous controls and unsupported navigation are rejected.
Nonhistorical files must have been modified within 30 seconds; this file-age
check does not prove that the application still has that state. The caller must
capture fresh AX state and execute through the existing guarded UI tools.
Historical inspection always removes the proposed action (`next: null`).

`observedDirectory` and optional recent `navigationAncestor` must come from an
independently established full path. Popup basenames alone do not establish a
filesystem identity. Both internal and explicitly authorized external canonical
directories are supported. Setting the filename and pressing Save are separate
plans; `saveDispatched: true` prevents another save plan and directs observation
of the existing submission. Dispatch is not file or media completion.

Keep this helper separate from an already deployed server that provides live FCP
tools: this checkout does not yet contain that server's complete live API.
Do not replace that server merely to obtain Save planning or `AXOpen` support.

## Migration status

The prior package's source, tests, synthetic fixtures and lockfile have been
copied without copying recordings, dependency source trees or build products.
The archived source is preserved. Canonical tracking and scratch-policy changes
are under verification; full native coverage/sanitizer gates are **not yet
verified**. Inherited test product lookup and temporary storage now use the
configured build and scratch locations. Live integration-test authorization and
model-fixture requirements still need review before a full test run. Legacy
orchestrator callers still require migration to configured shared entry points.
This is not a completed replay delivery or proof that every old helper has been
replaced.
