"""Measured fallback for intact fine-ASR words, never coarse phrase timing."""

import hashlib
import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction

import regex

from replay import caption_linguistics
from replay.caption_anchor_fit import anchored_groups, fit_group
from replay.caption_measure_cache import CaptionMeasurer
from replay.caption_scale import CaptionScale
from replay.caption_style_defaults import MAX_CAPTION_LINES
from replay.caption_support_identity import (
    OriginalFragment,
    SupportAlias,
    SupportIdentity,
    bind_aliases,
    support_key,
)
from replay.caption_timing import CaptionPolicy, SpokenToken, lexical_token_sources


@dataclass(frozen=True)
class _Occurrence:
    unit: SpokenToken
    identity: SupportIdentity
    support: tuple[int, float]
    anchored: bool


class IndivisibleFit:
    def __init__(
        self,
        tokens: tuple[SpokenToken, ...],
        *,
        layout: CaptionMeasurer,
        maximum_height: float | None,
    ) -> None:
        self.layout = layout
        self.maximum_height = maximum_height
        self.evidence: dict[str, CaptionScale] = {}
        self.scoped_evidence: dict[tuple[str, float], CaptionScale] = {}
        self.occurrences = support_occurrences(tokens)

    def _occurrence(self, text: str, context: tuple[SpokenToken, ...] | None) -> _Occurrence | None:
        candidates = [
            item
            for item in self.occurrences
            if support_key(item.unit.text) == support_key(text)
            and (
                context is None
                or any(
                    token.start <= item.unit.start
                    and item.unit.end <= token.end
                    and caption_linguistics.recognition_continuity(token.utterance)
                    == item.unit.utterance
                    for token in context
                )
            )
        ]
        bind_aliases(tuple(SupportAlias(item.unit.text, item.identity) for item in candidates))
        return candidates[0] if candidates else None

    def eligible(self, text: str) -> bool:
        return self._occurrence(text, None) is not None

    def eligible_scoped(self, text: str, context: tuple[SpokenToken, ...]) -> bool:
        return self._occurrence(text, context) is not None

    def wrap(self, text: str) -> str:
        return self._wrap(text, None)

    def wrap_scoped(self, text: str, context: tuple[SpokenToken, ...]) -> str:
        return self._wrap(text, context)

    def _bounds(self, value: str) -> tuple[float, float]:
        if self.maximum_height is not None:
            return self.layout.measure(value)
        lines = value.split("\n")
        return max(self.layout.measure(line)[0] for line in lines), float(len(lines))

    def _phrase_lines(self, text: str) -> str:
        return caption_linguistics.wrap_measured_lines(
            text,
            measure=self._bounds,
            width=self.layout.width,
            height=MAX_CAPTION_LINES if self.maximum_height is None else self.maximum_height,
            maximum_lines=MAX_CAPTION_LINES,
            prepare_lines=self.layout.prepare_lines,
        )

    def _split_word(self, text: str) -> str:
        return caption_linguistics.wrap_split_word(
            text,
            measure=self._bounds,
            width=self.layout.width,
            height=MAX_CAPTION_LINES if self.maximum_height is None else self.maximum_height,
            prepare_lines=self.layout.prepare_lines,
        )

    def pages(self, text: str) -> tuple[str, ...]:
        """Full-size pages of a text too wide for one page, in reading order.

        For renderers that cannot scale a caption: each page is the longest remaining prefix
        (at a grapheme end) that still fits the line limit at the style's size.
        """
        rest, result = text.replace("\n", ""), []
        while rest:
            ends = sorted((match.end() for match in regex.finditer(r"\X", rest)), reverse=True)
            for end in ends:
                try:
                    page = self._phrase_lines(rest[:end])
                except caption_linguistics.CaptionLayoutError:
                    try:
                        page = self._split_word(rest[:end])
                    except caption_linguistics.CaptionLayoutError:
                        continue
                result.append(page)
                rest = rest[end:]
                break
            else:
                raise caption_linguistics.CaptionLayoutError("No grapheme fits the caption region")
        return tuple(result)

    def _wrap(self, text: str, context: tuple[SpokenToken, ...] | None) -> str:
        try:
            return self._phrase_lines(text)
        except caption_linguistics.CaptionLayoutError:
            occurrence = self._occurrence(text, context)
            if occurrence is None:
                raise
        # Splitting the word keeps the style's size; scaling is the fallback when no split fits.
        try:
            return self._split_word(text)
        except caption_linguistics.CaptionLayoutError:
            pass
        if occurrence.anchored:
            rendered, evidence = fit_group(
                text,
                layout=self.layout,
                maximum_height=self.maximum_height,
                support=occurrence.support,
            )
            if evidence is not None:
                self._record(rendered, evidence, context)
            return rendered
        width, height = self.layout.measure(text)
        count, maximum = occurrence.support
        evidence = CaptionScale(
            width,
            height,
            self.layout.width,
            height if self.maximum_height is None else self.maximum_height,
            count,
            maximum,
        )
        self._record(text, evidence, context)
        return text

    def _record(
        self, text: str, evidence: CaptionScale, context: tuple[SpokenToken, ...] | None
    ) -> None:
        if context is None:
            self.evidence[text] = evidence
        elif context:
            self.scoped_evidence[text, context[-1].start] = evidence


def support_occurrences(tokens: tuple[SpokenToken, ...]) -> list[_Occurrence]:
    """Audit every eligible unit using actual lexical source-index provenance."""
    policy = CaptionPolicy()
    words, sources = lexical_token_sources(tokens, word_mark_gap=policy.maximum_gap)
    starts = tuple(word.start for word in words)
    units = caption_linguistics.atomic_tokens(words, maximum_gap=policy.maximum_gap)
    result: list[_Occurrence] = []
    for unit in units:
        selected = [
            index
            for index in range(bisect_left(starts, unit.start), bisect_right(starts, unit.end))
            if words[index].end <= unit.end
            and caption_linguistics.recognition_continuity(words[index].utterance) == unit.utterance
        ]
        members = [words[index] for index in selected]
        if len(members) < 2 or "".join(word.text for word in members) != unit.text:
            continue
        maximum = max(word.end - word.start for word in members)
        anchored = unit.text in anchored_groups(members)
        intact = caption_linguistics._atomic_ends(unit.text) == frozenset({len(unit.text)})
        if maximum <= policy.maximum_context_seconds and (anchored or intact):
            indices = tuple(index for member in selected for index in sources[member])
            identity = _identity(tokens, unit, indices)
            bind_aliases((SupportAlias(unit.text, identity),))
            result.append(_Occurrence(unit, identity, (len(members), maximum), anchored))
    return result


def _identity(
    tokens: tuple[SpokenToken, ...], unit: SpokenToken, indices: tuple[int, ...]
) -> SupportIdentity:
    payload = [(token.text, token.start, token.end, token.utterance) for token in tokens]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    members = tuple(
        OriginalFragment(
            token.text, Fraction(str(token.start)), Fraction(str(token.end)), token.utterance, index
        )
        for index in indices
        for token in (tokens[index],)
    )
    return SupportIdentity(digest, unit.utterance, members, "token-sequence")
