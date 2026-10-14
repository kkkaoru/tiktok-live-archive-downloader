# Current compressed part observations

`replay.compressed_part_observation` independently observes an existing HEVC
video against its retained original production part. It does not encode, invoke
an Apple app, copy media, publish a delivery or retire an original. The historical
encoder remains unknown. A saved script, timing log or proposed workflow is not
an observation receipt.

The public API is `load(Reference) -> Request`,
`collect(Request, evidence_directory) -> dict[str, object]` and
`verify(Request, evidence_directory) -> dict[str, object]`. The CLI is:

```text
python -m replay.compressed_part_observation collect REQUEST EVIDENCE --sha256 REQUEST_SHA
python -m replay.compressed_part_observation verify REQUEST EVIDENCE --sha256 REQUEST_SHA
```

The request and existing parent directory are canonical repository work paths.
References use exact canonical paths and SHA256 values. Tool references must be
regular executable files. Original and compressed files must remain distinct.
The request has this schema; placeholders below describe data supplied by the
caller and are not executable configuration:

```json
{
  "version": 1,
  "kind": "compressed-part-observation-request",
  "source": {"path": "ABSOLUTE_ORIGINAL", "sha256": "SHA256"},
  "movie": {"path": "ABSOLUTE_COMPRESSED", "sha256": "SHA256"},
  "sourceNative": {"path": "ABSOLUTE_NATIVE_EVIDENCE", "sha256": "SHA256"},
  "tools": {
    "ffmpeg": {"path": "CANONICAL_FFMPEG", "sha256": "SHA256"},
    "ffprobe": {"path": "CANONICAL_FFPROBE", "sha256": "SHA256"}
  },
  "frames": 120,
  "rate": 30,
  "windowFrames": 30,
  "threads": 1,
  "videoConfiguration": {
    "codec_name": "hevc", "codec_tag_string": "hvc1",
    "profile": "Main", "level": 120,
    "width": 64, "height": 96, "pix_fmt": "yuv420p",
    "sample_aspect_ratio": "1:1", "field_order": "progressive",
    "color_range": "tv", "color_space": "bt709",
    "color_transfer": "bt709", "color_primaries": "bt709",
    "time_base": "1/3000", "extradata_hash": "SHA256:HEX"
  },
  "thresholds": {"ssim": 0.99, "psnr": 45}
}
```

`videoConfiguration` contains every field in the existing compressed delivery
codec contract. It must bind the expected common codec configuration; this
single-part adapter cannot prove equality across a batch. The source native
evidence must bind the unchanged original and its native completion, observed
Share idle state and successful full AV decode. The adapter preserves this
evidence; it does not replace the original native publication verifier.

`collect` uses the existing bounded `delivery_verification._run` implementation
for three read-only commands. It captures actual absolute argv, return code,
stdout and stderr. The first ffprobe scan software-decodes every video frame and
observes codec extradata. A packet-only ffprobe read observes the first coded
packet. One FFmpeg invocation compares the beginning, middle and end windows
against the original, at the same frame indices. It uses `yuv420p10le` on both
sides and saves complete per-frame SSIM/PSNR stats. The supplied window size is
bounded by the actual part extent; a short part can have overlapping windows.
Thread count is explicit. These calls are intended for a separately admitted
future configuration, not for rerunning completed production measurements.

The initial hash context freshly reads the complete request, original, compressed
file, native evidence, tools and code closure before any child starts. Terminal
verification freshly reads all those inputs again. Generated raw data is
cryptographically pinned independently. Exact zero-origin CFR, every decoded
PTS and duration, error flags, raw codec hash, HEVC NAL length and first IDR are
validated. Every metric row and its index must exist and contain finite values.
The output records arithmetic mean per decoded frame and minimum separately;
this PSNR aggregation is explicitly not an aggregate-MSE PSNR calculation.
Only the supplied numerical policy determines `numericalQualityAccepted`.
A numeric failure remains a candidly labelled candidate, never approved quality.

The existing runner does not record process PID/birth. The receipt truthfully
states `processIdentityRecorded:false`; it records no invented identity. Actual
current observation command capture is separate from encoding history. All
receipts retain `historicalGenerationVerified:false`, `candidateOnly:true`,
`sourceRetirementAllowed:false`, `finalCompressorRequired:true` and false quality,
sync and delivery approvals. Audio inside a part is not adopted or concatenated.

`verify` reads the completed immutable receipt and original raw/process evidence,
rechecks fresh current inputs and reproduces the derived domains. It never starts
a child or repairs missing artifacts. An existing directory without a complete
receipt is retained and rejected. Collection does not guess whether an interrupted
command succeeded and never overwrites or automatically retries it.

For a future workflow, declare the nonempty receipt, snapshots, process records,
video/packet raw JSON and metric stats as producer outputs. Use the `verify` CLI
as the domain verifier. The zero-length successful stderr/null stdout files are
bound within the receipt rather than declared as nonempty stage artifacts. Run
`replay.workflow --check` before execution. The existing planner accepts native
Compressor evidence only; this new result is deliberately a different schema.
A separately reviewed compact-input bridge is required before these candidates
can enter assembly. No planner, retirement or final-delivery gate is changed by
this module.

Verification uses synthetic metadata and a mocked existing runner. No production
movie, actual FFmpeg invocation, native app or SSD operation is part of those
tests. Mocked success verifies the adapter contract, not real media acceptance.
