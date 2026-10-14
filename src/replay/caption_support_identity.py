"""Outer-whitespace lookup aliases with exact original recognition identity.

No text rendering, token deduplication, timestamp projection or winner selection.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

MAX_TEXT = 8192
MAX_FRAGMENTS = 16384


def support_key(text: str) -> str:
    """Normalize only outer whitespace, never remove internal spaces or symbols."""
    if not text or len(text) > MAX_TEXT or "\0" in text:
        raise ValueError("Invalid caption support text")
    key = text.strip()
    if not key:
        raise ValueError("Empty normalized caption support key")
    return key


@dataclass(frozen=True, slots=True)
class OriginalFragment:
    text: str
    start: Fraction
    end: Fraction
    owner: str
    source_token_index: int

    def __post_init__(self) -> None:
        if (
            not self.text
            or len(self.text) > MAX_TEXT
            or not self.owner
            or not isinstance(self.start, Fraction)
            or not isinstance(self.end, Fraction)
            or not 0 <= self.start < self.end
            or type(self.source_token_index) is not int
            or self.source_token_index < 0
        ):
            raise ValueError("Invalid original caption fragment identity")


@dataclass(frozen=True, slots=True)
class SupportIdentity:
    input_sha256: str
    cut_identity: str
    fragments: tuple[OriginalFragment, ...]
    scope_kind: Literal["source-file", "token-sequence"] = "source-file"

    def __post_init__(self) -> None:
        if (
            len(self.input_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.input_sha256)
            or not self.cut_identity
            or not 1 <= len(self.fragments) <= MAX_FRAGMENTS
            or self.scope_kind not in {"source-file", "token-sequence"}
        ):
            raise ValueError("Invalid source/cut support identity")


@dataclass(frozen=True, slots=True)
class SupportAlias:
    text: str
    identity: SupportIdentity


def bind_aliases(entries: tuple[SupportAlias, ...]) -> dict[str, SupportIdentity]:
    """Refuse ambiguous normalization instead of choosing a different occurrence.

    Use per recognition group, not a whole transcript keyed only by its text.
    The original fragment texts and anchors are returned unchanged.
    """
    if len(entries) > MAX_FRAGMENTS:
        raise ValueError("Caption support alias budget exceeded")
    result: dict[str, SupportIdentity] = {}
    for entry in entries:
        key = support_key(entry.text)
        original_text = "".join(fragment.text for fragment in entry.identity.fragments)
        if key != support_key(original_text):
            raise ValueError("Caption alias differs from original recognized text")
        previous = result.get(key)
        if previous is not None and previous != entry.identity:
            raise ValueError("Ambiguous normalized caption support identity")
        result[key] = entry.identity
    return result
