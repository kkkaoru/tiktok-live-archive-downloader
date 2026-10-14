# Joint measured caption fitting

`replay.caption_joint_fit.joint_fit` is a typed API for selecting a
simultaneous one-, two- or three-line layout together with its uniform scale.
`caption_anchor_fit.fit_group` now uses it for recognition-anchor groups after
the existing caller's eligibility and ordinary unscaled layout checks.

The API reuses the existing Japanese POS phrase boundaries and literal rhythmic
unit protections. It enumerates bounded combinations of those boundaries and
measures each whole rendered candidate. Candidates wider than the available
region are retained: a legal two-line candidate can require moderate scaling and
still provide larger glyphs than a much longer single line. Width and height are
considered together. Equal unscaled factors prefer fewer lines, then smaller
width. Equal scaled factors prefer smaller width, then fewer lines, preserving
the existing anchored group's tie behavior. Remaining ties retain boundary order.

Call `joint_fit(text, measure=..., available_width=..., available_height=...)`.
The measurement callable must return positive finite rendered width and height
for the whole candidate, including paint, line advance and animation clearance.
An optional `prepare_lines` callback receives the unique single lines once. The
immutable result retains all original characters in order, with only legal
newlines added, and reports the chosen dimensions, factor and search accounting.
No timing or style parameter is accepted or generated. Callers retain the
original recognized start/end, style 20, font, palette, center and bottom anchor.

`JointFitBudget` limits proposed candidates and measurement units. A measurement
unit is either one whole-candidate `measure` call or one unique line submitted to
`prepare_lines`. Internal work performed by those callbacks is not a certified
resource bound and must have its own native/cache limits. Overflow fails before
any preparation or measurement; the API never silently searches a truncated
prefix. Blank text, existing line breaks, NUL, nonpositive/nonfinite dimensions,
invalid line counts and invalid budgets are rejected. Candidate lines containing
only whitespace are excluded without trimming or changing source characters.

Caller-verified atomic words use `indivisible=True` and remain intact even if a
dictionary produces incidental phrase offsets inside an unknown run. Eligibility
and atomic provenance remain the existing caller's responsibility; morphology
alone cannot certify that every unknown word is safely divisible. Other text stays
intact when no existing phrase boundary is available. The API does not invent
character-count splits,
repetitions, word onset times, page times, suffix/particle splits or a new polite
chain exception. The legacy wrapper's extra polite-chain extraction is not copied
here. A future shared public candidate-enumeration helper could deduplicate the
small combinations loop after the existing pinned source boundary is released.

`readability_accepted` is always false. A maximum geometric fit is not a legibility
threshold or native paint certificate. In particular, fitting an intact long
atomic line can still produce unreadable tiny text. The result does not clamp its
font, discard characters or prematurely approve it. The existing 82-character
native observation and planned, unrendered 151-character XML remain separate.

The production adapter keeps the existing `CaptionScale` support/provenance and
original clocks. When no explicit height is supplied, it preserves the existing
capacity of three measured single-line heights. That one baseline measurement is
charged against the 4096-unit allowance, leaving at most 4095 units for joint
selection; the 1024-candidate limit stays unchanged. Genuine indivisible units
retain the existing separate `IndivisibleFit` path. No FCP or Motion style,
three-layer paint, center/bottom placement or native source is modified here.

New paging segments and head-placement candidates bind `caption_joint_fit.py`
as a declared producer input alongside `caption_anchor_fit.py`. A changed joint
source refuses segment reuse or head preparation through the existing input
hash checks. Historical producer manifests remain restricted to the original
layout producer source; they cannot exempt a changed joint-fit dependency.
Existing artifacts and their original input records are retained unchanged.

Production connection is not native acceptance. Retain synthetic regression
results and a new same-scene native comparison. Preserve text, recognized clocks, font, margins,
center/bottom/Y and settled entrance while varying scale alone for the paint
diagnosis. Record the blue fill, white rim, black rim and shadow footprint, then
check complete glyph visibility and readable three-layer composition separately
from XML geometry, native completion and full decode. Native acceptance and the
actual rendering belong to the production owner.
