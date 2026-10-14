# Source-clock summary inputs and quality evidence

`transcript-summary generate|inspect|preview --source-labels JSON
--source-media FILE --clock-domain source-presentation` reads validated source
labels without moving their clocks or registering a Viewer candidate. Select
`--source-label-layout tokens` for recognition tokens: text concatenates without
cumulative-caption collapse, repetitions remain intact, and gaps over 1.5 seconds
insert an utterance boundary. The default `captions` retains the subtitle-input
contract. `preview` prints the exact ten-minute section inputs without inference.

`--caption-tokens MANIFEST` instead reads recognition tokens before progressive
caption display. The manifest pins the display SRT, segment index, reflow input
bindings and every token file. It declares `clockDomain: edited-output`; each
segment-local token receives its segment frame offset once. Repeated recognition
tokens are never collapsed. A manifest has its own summary fingerprint, distinct
from the related display SRT. Pinning both files does not prove their semantic
equivalence or authorize a summary-pointer replacement.

Narrative generation checks exact repeated sentences and narrowly matched
opposing assertions. These checks are rejection gates, not factual verification.
Different subjects, explicit corrections, quotations and temporal changes are
not indiscriminately rejected. Failed generation preserves the saved revision.
`--repair-sections START,START` requires matching source fingerprints and saved
section clocks, regenerates the selected sections plus overview, and preserves
the other sections. Changed structured input requires a new complete result.

## Explicit excerpts

`--excerpts-output NEW_JSON` selects a distinct source-excerpt artifact for a
single source-label input. It cannot be combined with force, repair selection or
preview. The existing Apple model selects integer IDs only; code constructs saved
text from the original source. Free prose, malformed arrays and out-of-range IDs
fail. Japanese word boundaries and actual utterance-gap boundaries bound the
candidate spans. A single oversized token is rejected instead of split internally.
Every source span enters a bounded pairwise selection process; at most three
excerpts remain per section, and the overview selects from these exact excerpts.

The artifact declares `mode: source-excerpts-v1`, the structured source fingerprint,
section clocks and half-open Swift Character ranges. `inspect` with the same
option verifies exact source slices, order, nonoverlap and overview membership.
Output is non-overwriting, in an already-private canonical directory; it never
replaces narrative `current.json`. Optional diagnostics bind requests and clocks;
`--include-text` stores private source/model text and must not be published.

These are excerpts, not rewritten narrative or semantic approval. Recognition
errors, meaningful source repetitions, source-section boundaries inside a spoken
word, selection-order bias and omission of context remain possible. The actual
Oct7 trial selected greetings in its overview, so exact grounding did not prove
good coverage of the important topics. No free-prose fallback is used to conceal
these limitations.

## Deterministic representative excerpts

Add `--excerpt-selection representative` with `--excerpts-output NEW_JSON` and
either a caption-token manifest or source labels in explicit `tokens` layout.
This selects a separate wrapper with `version: 1`,
`method: word-time-diversity-v1`, an explicit `clockDomain`, and the display label
**発話抜粋による要約・未検証**. No model is called. The original source-excerpt
validation remains responsible for exact text, character ranges, fingerprint,
section order and overview membership.

The selector uses Japanese word-tokenizer units, capped-by-logarithm document
frequency across distinct candidate texts, and vocabulary novelty. Common
function words and greetings are ignored only for scoring; quoted text is never
rewritten. Up to one candidate is selected per temporal third of a section.
Identical repeated candidates do not multiply frequency or occupy multiple slots.
The overview selects from the saved section excerpts. This is not POS tagging:
Japanese `NLTagger.lexicalClass` returned no content words in the measured runtime,
so the unsupported POS experiment was retained as a failed candidate, not silently
treated as successful lexical ranking.

The shared token-text layout maps each selected character range to its original
token clock envelope. `clocks` contains the exact source span and the minimum
start / maximum end of intersecting tokens. No time is inferred proportionally
from character counts. `inspect` requires the caller's clock domain and verifies
the envelope against the same token input. Boundary-spanning speech remains in
its starting ten-minute section. Word fragments and recognition errors in the
source remain possible; lexical/time coverage is not proof of semantic relevance
or unbiased representation.

The existing source-excerpt Viewer reader supports an explicitly pinned v1
source-clock selection. The new representative wrapper and caption-token relation
require a distinct consumer connection; generation alone does not prove Viewer
display. Copying source-clock results into an edited-clock summary store is
invalid. Existing narrative reuse requires an identical encoded
transcript fingerprint and section clocks. Source-to-cut endpoint mapping alone
does not prove subject identity or retained lexical coverage.

## Context-aware quotation summaries

The source-supported quotation pipeline produces **extractive summaries**: it
selects existing speech and adds reporting punctuation, not new factual prose.
New `SourceSupportedProse` records use schema version 2 and the explicit label
**発話抜粋による要約・未検証**. Version 1 keeps its original label and original
candidate validation; old immutable artifacts are not silently relabeled or
regenerated. The existing method family remains `source-supported-quotation-prose-v2`.

Version 2 ranks pinned POS content by text density with a greeting penalty rather
than rewarding the raw noun count in long mixed-topic passages. Nearby contiguous
clauses sharing content nouns are retained together when the next clause contains
a response marker. The actual source token clocks bound the combined quotation;
long time gaps and unrelated subjects are not joined. This is a bounded context
heuristic, not reliable speaker identification or semantic entailment. It keeps
questions, negation and source repetition unchanged and introduces no wording or
interpolated timestamps. It does not establish that every contrast was detected.
Nominal utterances remain eligible with less weight; they are not rejected merely
for lacking a predicate. A pair that exceeds the bounded model-input allowance
retains the complete response instead of truncating the counter-position.

`preview --caption-tokens MANIFEST --excerpt-selection supported-prose
--morphology JSON` prints fingerprint-bound legacy/contextual candidate ranges,
scores and original clock envelopes. It calls no model and writes no summary;
omit `--excerpts-output`. This diagnostic output is not a selected summary or
evidence of semantic acceptance.

New candidates require focused synthetic and bounded same-input comparison before
regeneration or Viewer adoption. The six prior artifacts remain readable during
that verification. Generation success or exact quotation support does not approve
ASR accuracy, complete topic coverage or overall summary quality.
