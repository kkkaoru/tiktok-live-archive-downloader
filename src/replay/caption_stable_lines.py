"""Keep a page's line breaks once shown: growing captions never re-break text already on screen.

Paging lays out each growth stage on its own: ``caption_linguistics.wrap_measured_lines`` picks
the most balanced split among the morphological phrase boundaries, so adding words can move a
break inside text the viewer already read (``私のことを配信中に`` then ``私のことを\\n配信中に男
よりも``). While ``kept()`` is active, every stage is laid out under the layout of its displayed
prefix (the same context without its newest token): breaks inside that text stay exactly where
they were, and new breaks come from the same morphological candidates after it. When no such
layout fits, the layout fails like an over-budget page, so paging switches to a new page and the
shown text is not redisplayed with other breaks.

The layout modules are pinned by earlier evidence and stay unchanged; their module functions are
wrapped only inside ``kept()``.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol

from replay import caption_linguistics, caption_timing
from replay.caption_linguistics import CaptionLayoutError
from replay.caption_timing import CaptionPolicy, SpokenToken, punctuation_lines

BLOCKED = 1e12  # a finite measurement no caption region accepts
Measure = Callable[[str], tuple[float, float]]
Keep = tuple[int, tuple[int, ...]]  # shown characters and their break offsets
_KEEP: ContextVar[Keep | None] = ContextVar("caption_stable_lines_keep", default=None)
# Texts whose constrained measured wrap failed during the current layout. A fallback layout (an
# indivisible word fitted by scale) then does not count: it depends on context, so paging could
# accept a page that its final layout pass rejects.
_FAILED: ContextVar[list[str] | None] = ContextVar("caption_stable_lines_failed", default=None)


class Layout(Protocol):
    def __call__(
        self, text: str, policy: CaptionPolicy, context: tuple[SpokenToken, ...] = ()
    ) -> str: ...


class Wrap(Protocol):
    def __call__(self, text: str, *, measure: Measure, **options: object) -> str: ...


def breaks(layout: str) -> tuple[int, ...]:
    """Character offsets (in the text without newlines) where ``layout`` starts a new line."""
    offsets, total = [], 0
    for line in layout.split("\n")[:-1]:
        total += len(line)
        offsets.append(total)
    return tuple(offsets)


def allowed(candidate: str, keep: Keep) -> bool:
    """Whether ``candidate`` breaks the first ``shown`` characters exactly as they were shown."""
    shown, kept = keep
    return tuple(offset for offset in breaks(candidate) if offset < shown) == kept


def constrained(measure: Measure, text: str, keep: Keep) -> Measure:
    """``measure`` that makes every full layout of ``text`` re-breaking shown text not fit."""

    def bounds(value: str) -> tuple[float, float]:
        if value.replace("\n", "") == text and not allowed(value, keep):
            return BLOCKED, BLOCKED
        return measure(value)

    return bounds


def kept_wrap(original: Wrap) -> Wrap:
    def wrap(text: str, *, measure: Measure, **options: object) -> str:
        keep = _KEEP.get()
        if keep is None:
            return original(text, measure=measure, **options)
        try:
            return original(text, measure=constrained(measure, text, keep), **options)
        except CaptionLayoutError:
            failed = _FAILED.get()
            if failed is not None:
                failed.append(text)
            raise

    return wrap


def flat_text(tokens: tuple[SpokenToken, ...]) -> str:
    return punctuation_lines("".join(token.text for token in tokens)).replace("\n", "")


def displayed_prefix(flat: str, context: tuple[SpokenToken, ...]) -> tuple[SpokenToken, ...]:
    """The tokens shown just before ``flat``: those composing it, without the newest one.

    Paging's look-ahead passes a context that may run past ``flat`` (a particle tail it has not
    shown), so the prefix is found by the tokens whose text makes up ``flat``.
    """
    for count in range(len(context), 1, -1):
        if flat_text(context[:count]) == flat:
            return context[: count - 1]
    return ()


def kept_layout(original: Layout) -> Layout:
    """``original`` laid out under the layout of the context's displayed prefix (memoized)."""
    results: dict[tuple[int, str, tuple[SpokenToken, ...]], str | CaptionLayoutError] = {}

    def layout(text: str, policy: CaptionPolicy, context: tuple[SpokenToken, ...] = ()) -> str:
        flat = text.replace("\n", "")
        key = (id(policy), flat, context)
        if key not in results:
            results[key] = laid_out(text, flat, policy, context)
        result = results[key]
        if isinstance(result, CaptionLayoutError):
            raise result
        return result

    def laid_out(
        text: str, flat: str, policy: CaptionPolicy, context: tuple[SpokenToken, ...]
    ) -> str | CaptionLayoutError:
        keep = None
        prefix = displayed_prefix(flat, context)
        if prefix:
            shown = flat_text(prefix)
            try:
                keep = (len(shown), breaks(layout(shown, policy, prefix)))
            except CaptionLayoutError as error:
                return error  # every prefix of a page is shown, so this page cannot grow here
        failed: list[str] = []
        token, marker = _KEEP.set(keep), _FAILED.set(failed)
        try:
            result = original(text, policy, context)
        except CaptionLayoutError as error:
            return error
        finally:
            _KEEP.reset(token)
            _FAILED.reset(marker)
        if failed:
            return CaptionLayoutError("Shown line breaks leave no measured layout")
        return result

    return layout


@contextmanager
def kept() -> Generator[None]:
    """Paging inside the block keeps shown line breaks; the pinned functions are restored after."""
    timing, linguistics = caption_timing.__dict__, caption_linguistics.__dict__
    saved = timing["_measured_layout"], linguistics["wrap_measured_lines"]
    timing["_measured_layout"] = kept_layout(saved[0])
    linguistics["wrap_measured_lines"] = kept_wrap(saved[1])
    try:
        yield
    finally:
        timing["_measured_layout"], linguistics["wrap_measured_lines"] = saved
