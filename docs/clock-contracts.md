# Inspecting and reusing clock conversions

Use `replay.clock_contract` at the boundary between completed recognition/audio
processing and caption/FCP preparation. It verifies existing results; it does not
recognize speech, render audio, shift timestamps or apply a cut twice. Existing
`AudioFrameClock`, `asr_clock_projection.project_labels` and
`clocked_cut_audio.plan_slots` remain the mapping implementations.

The version1 configuration declares `operation`, `inputClock`, `outputClock`,
`previousApplications: 0`, and `artifacts`. Every artifact is a canonical
checkout-local `{"path": "absolute path", "sha256": "lowercase SHA256"}` binding.
For already authorized external audio, pass `--media-root /absolute/allowed/root`.
Only `inputAudio` and `outputAudio` may be read there; JSON evidence, configuration
and newly written receipts remain checkout-local. Canonical regular files and
hashes are still required; symlink escapes are rejected. Rechecking requires the
same explicit root. This option grants no permission beyond the user's authorization.
Prepare a new immutable
configuration whenever an input changes; never rewrite a completed receipt.

| Operation | Input → output | Required artifact roles |
|---|---|---|
| `recognition-source` | `continuous-decoded-pcm` → `source-presentation` | `producerConfig`, `inputLabels`, `outputLabels`, `audioClock` |
| `tempo-cut-audio` | `continuous-decoded-pcm` → `edited-cut-pcm` | `producerConfig`, `producerReceipt`, `inputAudio`, `outputAudio`, `audioClock`, `cuts` |

For recognition, the producer configuration is the existing
`asr_clock_projection` config. The input must be assembled
`input-pcm-not-projected` labels. Output source labels are recomputed and compared
exactly, including text and provenance. A nonzero native audio start is read from
decoded frame PTS; it is not added as an assumed constant shift. Source labels
can feed source-clock cut planning or the single source→edited caption projection.
Do not pass them back into `asr_clock_projection`.

For audio, use the existing tempo selection and receipt. The selected rational
intervals must equal the requested cut revision, whose lineage must bind the same
separated PCM and decoded clock. All cut sample placements are recomputed; actual
WAV rate, channel count and sample count must match the full output receipt. The
native decoded rate and resampled PCM rate may differ: correspondence uses the
existing rational planner. Edited audio goes to the FCP consumer as voice plus
its tempo receipt. Do not cut it again or project it through the source clock.

Run these commands with the repository environment and scratch paths described
in [operations](operations.md) and [repo-jobs](../.agents/skills/repo-jobs/SKILL.md):

```sh
python -m replay.clock_contract CONFIG.json --output NEW-RECEIPT.json
python -m replay.clock_contract CONFIG.json --check RECEIPT.json
python -m replay.clock_contract CONFIG.json --inspect
```

Declare configuration, artifacts and validator dependencies as workflow inputs;
declare the new receipt as output and `--check` as its domain verifier. Perform
`replay.workflow --check` before execution. `--check` rehashes and revalidates;
the existence of a receipt alone is not reusable completion. `--inspect` performs
the same checks and reports explicit origins/domains/rates, the next artifact
path and consumer, and first/middle/last cut correspondences from native seconds
through input PCM samples to edited samples. Rejections identify the configuration,
domains and relevant input/output paths. Fix the binding or rerun the affected
producer; do not change an expected hash merely to silence a rejection.

When cuts change, keep original raw recognition and source-presentation labels.
Create the new cut revision, reproject source labels through those new cuts once,
and regenerate tempo output from the separated continuous PCM using the new
intervals. Build a new contract and receipt. An old tempo receipt cannot be paired
with different cuts or audio, even if durations happen to match. Likewise a
nonzero `previousApplications`, swapped raw/projected inputs, wrong output sample
rate or changed artifact content is rejected.

For unsupported outer audio edges, `replay.cut_audio_support CONFIG OUTPUT` can
publish a separate cut revision using ceil(start) and floor(end) on the video
grid. It rejects removal intersecting any recognized interval or unresolved point.
It does not classify unsupported audio as silence or solve interior gaps. Preserve
the prior cuts and retain this new revision as a candidate.

`clockCorrespondenceVerified` establishes the tested mapping and identity only.
The contract deliberately retains `AVSyncVerified`, `speechQualityVerified` and
`deliveryComplete` as false. Full media decode, listening, semantic recognition,
caption review and final delivery checks remain distinct. Contract convenience
does not turn a structurally valid but repetitive or contradictory summary into
a faithful one.

Older protected frame plans with descriptive `grid`/`inputs` can be inspected with
an explicit `legacyCutBinding` object containing
`schema: "protected-frame-plan-completed-selection"` and `frameRate`.
This is accepted only when the completed tempo selection pins the exact old cuts,
source audio and decoded clock, and its receipt pins all those artifacts and the
selection. Every rational boundary must explicitly use the supplied frame rate;
every frame boundary, contiguous output offset and PCM sample slot is checked.
The descriptive grid string is never parsed to guess a rate. The old cut's audio
hash remains historical evidence, not a claim that it is the current stem.
This makes compatible older deliveries reusable without rewriting their records.
Other legacy schemas or missing producer pins remain unsupported. These schema
rules apply across recordings; a recording's date or ID never selects a policy.
