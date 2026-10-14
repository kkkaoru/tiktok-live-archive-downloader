# Native part completion and publication

An Executor `fcp_export` response acknowledges dispatch, not completed media.
Use `replay.native_export_receipt CONFIG.json COMPLETION.json` after Share is
observed idle and the existing `replay.fcp_part_assembly probe` has produced its
exact frame-clock proof. This command performs FFprobe stream validation and a
full FFmpeg video/audio null decode. It does not operate applications or encode.

The configuration has `version: 1`, a positive `part`, `audioSampleRate`,
`audioChannels`, and SHA-256 references (`path`, `sha256`) named `assembly`,
`exportRequest`, `exportResponse`, `background`, `ffmpeg`, and `ffprobe`.
All paths are canonical absolute paths. The assembly and partition use the
existing `fcp_part_assembly` schemas. The request is the exact native tool input;
both response and background are retained Executor envelopes of the form
`{"ok":true,"data":{"isError":false,"structuredContent":{...}}}`.
The background observation must follow the final movie write and export response
and identify the same process. Export project, requested path, localized format
and Save-only action, ProRes profile, dimensions, frame rate, duration and PCM
audio clock must agree. Native `completionVerified: false` remains recorded;
only independent validation produces the completed receipt.
One optional standard `data` / `tmcd` timecode stream is accepted only with zero
origin and matching frame rate and duration. Other data, extra video/audio or
duplicate timecode tracks are rejected.

Use `replay.part_master_publication CONFIG.json` to copy a completed part into
an authorized existing destination directory. Its configuration contains:

```json
{
  "version": 1,
  "completion": {"path": "/absolute/completion.json", "sha256": "SHA256"},
  "destination": "/absolute/delivery/part.mov",
  "receiptOutput": "/absolute/evidence/published.json",
  "frameProofOutput": "/absolute/evidence/published-frame-proof.json"
}
```

The command checks capacity, records exclusive intent, copies to an exclusive
partial file, verifies source identity and both hashes, then publishes with an
exclusive hard link on the destination filesystem. Existing files are never
overwritten. The source remains intact. On failure, intent and partial evidence
remain for diagnosis; do not blindly retry or delete them. Only its successful
temporary link is removed. This is separate from acquisition-job relocation.

Create a new assembly configuration pointing to the destination movie and new
frame proof; preserve the original configuration and proof. Native progress
configuration can use a pinned `publication` receipt instead of `decodeStatus`
and `background`. This verifies original completion and identical copied bytes,
rather than incorrectly requiring the original idle observation to postdate the
copy. Legacy progress configurations remain supported.

Every part still needs the original independent verification. Assembly verifies
partition clock and caption lineage and disables part audio in favor of the
single whole processed voice. Its output is FCPXML, not a completed movie.
Whole render, final Compressor encode, final full decode, summary/Viewer delivery
and visual or speech quality acceptance remain separate stages. These receipts
always retain `qualityApproved: false`.

For a caption-position-only revision, `python -m replay.fcp_part_reuse CONFIG.json`
can create a new immutable frame proof for an unchanged published part. The
version-one configuration contains a positive `part` and pinned `priorAssembly`,
`targetAssembly` and `publication` references. Supply a pinned `retirement` when
the original staging copy has been retired; missing staging media alone never
authorizes reuse. The output is the absent proof path already declared by the
target assembly, and its movie must be the existing published movie.

Reuse verifies the original native completion and publication, identical ordered
frame ranges, preserved whole processed voice, unchanged whole text/media/clocks
and non-Y geometry, and the selected part's complete XML including its
position curves. Only the project display name is ignored. A changed part needs
a fresh native export and ordinary completion proof. Reuse cannot originate from
another reuse proof. Assembly revalidates these bindings and producer hashes;
the original proofs and publication records remain untouched. This saves native
rendering and copying, without granting visual, synchronization or delivery
approval.

The whole comparison preserves X even in parts that are not selected for reuse.
Constant X with linear keys may use different Y sample times or change between
static and animated placement. Moving X retains its key times and curves;
unknown channel attributes remain part of the comparison. Changed X, malformed
vectors, duplicate position channels and nonincreasing keys cannot pass as a
height-only change. A new producer version requires new immutable reuse proofs
and assembly lineage; it does not require rendering unchanged movie bytes again.
Position vectors accept finite decimal and scientific literals with bounded
length and exponent digits, matching the shared native placement serializer.
Exact rational X comparison retains tiny changes instead of rounding them to zero.
