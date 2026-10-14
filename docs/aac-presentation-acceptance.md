# AAC presentation evidence

`replay.aac_presentation_acceptance` provides pure timing assessment and a small,
read-only container timing observer. It never encodes, remuxes, deletes edit
lists, pads/trims audio, searches for an offset or opens an application. It does
not replace `youtube_delivery_validation`, `delivery_remux_audit`, the full AV
decoder, source provenance or delivery approval.

YouTube recommends MP4 without edit lists. Apple documents AAC encoder delay
using edit lists and sample groups. A correctly mapped edit list and conformity
to YouTube's recommendation are separate findings. The existing strict upload
inspector remains unchanged.

Sources: [YouTube upload recommendations](https://support.google.com/youtube/answer/1722171?hl=en),
[Apple's explicit encoder-delay structures](https://developer.apple.com/documentation/quicktime-file-format/using_track_structures_to_represent_encode_delay_explictly),
[Apple's AAC encoding background](https://developer.apple.com/documentation/quicktime-file-format/background_aac_encoding).

## Pure interfaces

`parse_container_headers(stream, size=...) -> Container` uses the existing
bounded BMFF header walker, skipping `mdat` payloads. It parses movie/media
timescales and durations, unique track IDs, audio/video handlers and signed
version-0/version-1 edit entries. Unknown versions, truncated fields, duplicated
required boxes/IDs, excessive counts and invalid durations fail explicitly.
Presence of `sgpd` and `sbgp` is reported; their contents and Apple's roll-group
interpretation are **not** certified by this observer.

`assess(container, Inputs(...)) -> dict[str, object]` independently reports:

- `youtubeNoEditListsAccepted`: no edit lists; other YouTube properties remain
  the responsibility of the original strict inspector.
- `packetDecodedPresentationAccepted`: all packet and decoded clocks are
  continuous and observed packet skip/discard, stream and decoded endpoints
  agree. Negative coded packet PTS may be valid. A packet whose duration already
  excludes trailing padding must not have padding subtracted again. The
  decoder-and-stream corroboration is the same rule used by
  `delivery_audio_quality.decode_aac`.
- `trackEditMappingAccepted`: one linear rate-1 edit per track, contained in its
  media support, and audio presentation duration matches the decoded clock.
  Video uses mdhd duration unless explicit pinned STTS/CTTS evidence corroborated
  with every original packet and decoded PTS is supplied.
  Multiple/empty/dwell/rate-changing edits remain unsupported. Demuxed decoded
  PTS already reflect edit/skip handling and are never shifted a second time.
- `originalPCMExtentAccepted`: explicit, pinned original PCM clock evidence
  matches the supplied source and its exact rational extent equals the decoded
  presentation. Specifying an expected count alone cannot pass.
- `originalPCMContentAccepted`: the original source and candidate are bound into
  an existing accepted `delivery-audio-quality` **quality-phase** receipt. It
  must retain all original checks, fixed speech/separation policy, full-program
  measured speech statistics and ordered complete quality windows. Length
  equality or one `accepted` boolean cannot establish content preservation.
- `presentationSyncAccepted`: all internal mapping and original PCM extent and
  content findings pass. This remains an evidence assessment, not a delivery,
  subjective quality or player approval.

`Evidence(reference, payload)` accepts immutable observation bytes and checks
their SHA256 whenever assessed. The caller must establish actual command/tool,
initial/terminal input pins, raw output provenance and the original domain
verification before using those bytes. A self-written JSON document with a
matching hash is not evidence that a process ran. The module neither creates
historical process records nor freshly hashes any whole media file.

`SourcePCM` requires a reference, observed sample rate/count/start and optional
`clock_evidence`. Missing evidence leaves extent acceptance false. The pinned
clock observation has this metadata contract:

```json
{
  "version": 1,
  "kind": "original-pcm-clock-observation",
  "source": {"path": "/absolute/original.wav", "sha256": "<actual SHA256>"},
  "sampleRate": 48000,
  "sampleCount": 1920,
  "start": "0"
}
```

This is an input format for observed facts, not a PCM collector or permission to
infer facts from an expected video duration. The parent must bind it to the
actual retained PCM/header/producer evidence. A 44.1kHz source is compared with
48kHz AAC using exact rational duration, without synthesizing a sample mapping
or resampling. Existing quality production must independently support the
chosen source and output rates; this adapter does not add such support.

## Admitted header observation

`observe_headers(Reference(...), expected_identity=FileIdentity(...))` is the
future small observation API for an already admitted reader. It opens one
canonical, unlinked regular file read only, checks the expected device/inode/
size/mtime/ctime against both path and FD, parses timing metadata, then checks
both identities again. It creates no files or directories. The returned
`timingPayloadSHA256` covers selected timing-box payloads with their offsets,
ends and kinds; it is not the complete file or complete header-region SHA256.

The supplied whole-file SHA is an **expected retained pin**. The report states
`expectedWholeSHA256FreshlyVerified: false` and `fullDecodePerformed: false`.
Parent-owned fresh whole-file pin/full AV checks remain required. Replacing the
path or modifying the file during observation fails; this observer is not a
lease against future mutations after it returns.

Observe current final MP4 headers only after resource/reader admission. Retired
whole ProRes masters are not dependencies; retain authoritative whole voice,
native parts, source recipes and existing raw packet/frame evidence.

All reports keep `qualityApproved`, `syncApproved` and `deliveryApproved` false.
An unexplained source-program deficit remains failed extent, even when packet
priming is internally consistent or shorter than a video frame. Terminal
content cannot be assumed silent. Actual end-content correspondence and target
player seeking remain independent evidence; no new encoding is prescribed by
this module.

## Public finite observation CLI

```sh
python -B -m replay.aac_presentation_acceptance observe REQUEST.json OUTPUT.json --sha256 REQUEST_SHA
python -B -m replay.aac_presentation_acceptance verify REQUEST.json OUTPUT.json --sha256 REQUEST_SHA
```

`observe` writes three exclusive, private files: `OUTPUT.json`,
`OUTPUT.json.decoded.json` and `OUTPUT.json.packets.json`. The parent directory
must already exist under checkout `edits/work`. Declare all three outputs in
`replay.workflow`; use `verify` as the read-only domain verifier. Existing or
partial output refuses producer re-entry. A complete existing result can only
be reused through the original workflow's exact output/input pins and the
read-only verifier. No header is regenerated by `verify` and neither action
launches a decoder, encoder or application.

The version-1 request has kind `aac-presentation-observation-request` and these
typed fields:

- `candidate`: the retained absolute final MP4 reference and expected whole SHA.
- `expectedIdentity`: five integral fields `[device, inode, size, mtimeNs, ctimeNs]`.
- `decodeReceipt` and `completedCheckpoint`: pinned original independent full
  decoder receipt and its complete workflow checkpoint. Initial and terminal
  candidate pins in the receipt must agree with each other and the request.
- `tools`: explicit `ffmpeg` and `ffprobe` references. Current tool bytes and
  executable status are checked; no historical tool-byte observation is created.
- `processes`: exactly `decode`, `streams`, `frames`, `packets`, each containing
  original `process`, `stdout` and `stderr` references. The actual retained
  command must match the existing `delivery_verification` command, with integral
  exit status zero, empty error output and the configured tool/candidate paths.
- `source`: optional `reference`, `sampleRate`, `sampleCount`, rational `start`,
  pinned `producer`, and optional `clockEvidence`. Source facts must match the
  producer's output/hash/sample clock. Missing clock evidence leaves exact source
  extent acceptance false. A source header or quality receipt is not fabricated.
- `contentComparison`: null or the pinned original quality-phase receipt.

The original verifier's `frames.stdout` records mixed `media_type` rows without
complete stream metadata. Its `packets.stdout` supplies stream indices and
packet counts, also lacking the full AAC stream contract. The CLI joins these
retained observations to the one actual audio stream in `streams.stdout`, which
must equal the original receipt's stream metadata. It selects audio frame rows
by `media_type` and packet rows by that observed stream index. It preserves
their order, PTS, sample counts, durations and side data. It does not add a
measured stream index to frame rows, sort rows or interpolate timings. Raw and
derived byte hashes, process records, source producer and checkpoint references
are retained separately. Static JSON with a hash is not an independent proof
that the original process ran; the caller must retain its original domain and
source closure verification.
The legacy decoder did not capture per-stdout hashes in its receipt. Newly
pinning retained raw bytes does not retroactively create those historical
observations; the report explicitly keeps
`historicalRawStdoutSHA256CapturedByOriginalDecoder` false.

Observation metadata is read through bounded, unlinked regular FDs with fresh
path/FD identity and SHA checks. Requests are at most 1MiB; each raw result is at
most 128MiB and each observed row table at most one million rows. The original
tables parse serially. Their JSON object graphs are released before the pure
assessor parses derived bytes. Terminal raw pin checks hash bytes without
parsing the tables again. This uses standard JSON parsing, **not a streaming
parser**. Encoded byte bounds are not an RSS measurement: the parent must budget
Python dictionaries, strings, integers, selected audio rows, serialized derived
bytes and the assessor's clock arrays when admitting a reader.

The header operation skips compressed media payloads, but the whole workflow
is not a header-only read: it also validates retained raw observation bytes.
Declaring the final movie as an ordinary workflow input invokes the existing
whole-file fingerprint boundaries. Count those reads separately from the
bounded timing-box read; neither the expected retained SHA nor a timing payload
hash is fresh whole-file verification. Preserve all original ELST entries and
all measured source deficits regardless of the presentation assessment.

## Video decode and composition support

`replay.video_composition_timing` provides a separate bounded table observer.
Apple distinguishes decode time from composition/presentation time with
`CT = DT + CTTS`. A positive video edit start can fit the composition range even
when that start plus duration exceeds the decode-duration-only mdhd envelope.
See [Apple composition offsets](https://developer.apple.com/documentation/quicktime-file-format/composition_offset_atom).

STTS version 0 and CTTS unsigned version 0 / signed version 1 are supported.
Runs/counts/byte lengths must agree, positive sample durations must sum exactly
to mdhd, and derived presentation intervals must be contiguous. Each table is
bounded to 8,000,008 bytes and one million samples. Absent CTTS means zero
composition offsets. Duplicate/malformed tables, gaps or overlaps, unsupported
edits and inconsistent original observations fail explicitly.

```sh
python -B -m replay.video_composition_timing observe ORIGINAL_REQUEST.json OUTPUT.json --sha256 REQUEST_SHA --aac-result PRIOR_RESULT.json --aac-result-sha256 PRIOR_SHA
python -B -m replay.video_composition_timing verify ORIGINAL_REQUEST.json OUTPUT.json --sha256 REQUEST_SHA --aac-result PRIOR_RESULT.json --aac-result-sha256 PRIOR_SHA
```

The original AAC request supplies candidate identity, retained full decoder
completion and exact process/tool/raw pins. The observer reads timing/sample
tables, then streams retained raw packet/frame JSON one row at a time with SHA
and initial/terminal FD identity checks. It checks every video packet
DTS/PTS/duration in decode order and every decoded PTS/duration in presentation
order against the tables. ELST media time is subtracted once; negative decode
pre-roll is retained. Original tables/raw rows are never sorted or rewritten;
separately derived presentation order is used only for interval/frame comparison.
No new decode or whole-file hash occurs. The whole hashes are retained expected
pins. Parent-owned matching identity/provenance and fresh input scopes remain
separate from table SHA observations.

`Inputs.video_composition` is optional immutable evidence. Without it, the
previous mdhd-only rejection remains. Evidence must bind the exact candidate and
complete container timing. With the optional prior AAC result, the CLI rechecks
its original pure domain and reports a separate AAC assessment using composition
support. PCM extent/content deficits are independent. Formal approvals stay
false. `verify` repeats the bounded table and retained-row domain observation
read only; it never repairs missing or changed outputs.
