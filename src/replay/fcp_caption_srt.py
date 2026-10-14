"""Extract display captions from a bounded, direct FCP edit on its output clock.

This is a sidecar export, not evidence that a renderer consumed the project.
Unsupported nesting, retiming, overlapping titles and divergent text layers fail.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from xml.parsers import expat

from replay.caption_font_supplier import MAX_FCPXML_BYTES, MAX_TITLES, _doctype
from replay.core import ReplayError
from replay.delivery_render import render_srt

Cue = tuple[Fraction, Fraction, str]
MAX_TEXT_CHARACTERS = 4096
SEQUENCE_PATH = ("fcpxml", "library", "event", "project", "sequence")
CLIP_PATH = (*SEQUENCE_PATH, "spine", "asset-clip")


def _time(attributes: dict[str, str], key: str) -> Fraction:
    value = attributes.get(key, "")
    if not value.endswith("s") or len(value) > 64:
        raise ReplayError(f"Missing or invalid FCP {key}")
    try:
        result = Fraction(value[:-1])
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError(f"Invalid FCP {key}") from error
    if result < 0:
        raise ReplayError(f"Negative FCP {key}")
    return result


class _Collector:
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.sequence_count = 0
        self.duration = Fraction(0)
        self.clip: dict[str, str] | None = None
        self.title: dict[str, str] | None = None
        self.body: str | None = None
        self.texts: list[str] = []
        self.cues: list[Cue] = []
        self.clip_end = Fraction(0)
        self.style_refs: set[str] = set()
        self.style_ids: set[str] = set()

    def start(self, name: str, attributes: dict[str, str]) -> None:
        path = (*self.stack, name)
        if len(path) > 64 or name == "timeMap":
            raise ReplayError("Unsupported FCP nesting, retiming or text style overrides")
        if not self.stack and name != "fcpxml":
            raise ReplayError("Expected FCPXML root")
        if self.body is not None and not (
            name == "text-style"
            and path == (*CLIP_PATH, "title", "text", "text-style")
            and set(attributes) == {"ref"}
            and attributes["ref"]
        ):
            raise ReplayError("Expected plain caption text")
        if self.body is not None:
            self.style_refs.add(attributes["ref"])
        if name == "text-style-def":
            identifier = attributes.get("id", "")
            if (
                path != (*CLIP_PATH, "title", name)
                or not identifier
                or identifier in self.style_ids
            ):
                raise ReplayError("Unsupported caption text style overrides")
            self.style_ids.add(identifier)
        if name == "sequence":
            self.sequence_count += 1
            if path != SEQUENCE_PATH or self.sequence_count != 1:
                raise ReplayError("Expected exactly one direct FCP sequence")
            self.duration = _time(attributes, "duration")
            if attributes.get("tcStart", "0s") != "0s":
                raise ReplayError("Nonzero sequence timecode is unsupported")
        if path == CLIP_PATH:
            if attributes.get("enabled", "1") != "1":
                raise ReplayError("Disabled primary clips are unsupported")
            offset = _time(attributes, "offset")
            duration = _time(attributes, "duration")
            if offset != self.clip_end or duration <= 0:
                raise ReplayError("Primary FCP clips must be contiguous and positive")
            self.clip_end = offset + duration
            self.clip = attributes
        if name == "title":
            if path != (*CLIP_PATH, "title") or len(self.cues) >= MAX_TITLES:
                raise ReplayError("Expected bounded direct clip titles")
            if attributes.get("enabled", "1") != "1":
                raise ReplayError("Disabled titles are unsupported")
            self.title = attributes
            self.texts = []
            self.style_refs = set()
            self.style_ids = set()
        if name == "text" and self.title is not None:
            if path != (*CLIP_PATH, "title", "text"):
                raise ReplayError("Expected direct title text")
            self.body = ""
        self.stack.append(name)

    def data(self, text: str) -> None:
        if self.body is not None:
            self.body += text
            if len(self.body) > MAX_TEXT_CHARACTERS:
                raise ReplayError("Caption text exceeds the supported bound")

    def end(self, name: str) -> None:
        if name == "text" and self.body is not None:
            self.texts.append(self.body)
            self.body = None
        if name == "title":
            self.finish_title()
            self.title = None
        if tuple(self.stack) == CLIP_PATH:
            self.clip = None
        self.stack.pop()

    def finish_title(self) -> None:
        if self.style_refs != self.style_ids:
            raise ReplayError("Caption text style references lack local definitions")
        if self.clip is None or self.title is None:
            raise ReplayError("Missing title clock context")
        if not 1 <= len(self.texts) <= 3 or len(set(self.texts)) != 1:
            raise ReplayError("Caption text layers must contain identical display text")
        text = self.texts[0]
        if (
            not text.strip()
            or "\n\n" in text
            or any(ord(character) < 32 and character != "\n" for character in text)
        ):
            raise ReplayError("Caption text cannot be represented as an SRT cue")
        clip_start = _time(self.clip, "offset")
        start = clip_start + _time(self.title, "offset") - _time(self.clip, "start")
        end = start + _time(self.title, "duration")
        if not clip_start <= start < end <= self.clip_end:
            raise ReplayError("Title extends beyond its selected source clip")
        if round(start * 1000) >= round(end * 1000):
            raise ReplayError("Title duration is below SRT millisecond resolution")
        self.cues.append((start, end, text))


def caption_cues(path: Path) -> tuple[Cue, ...]:
    """Return ordered exact rational output-clock cues; do not infer source timing."""
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FCPXML_BYTES:
        raise ReplayError("Expected bounded nonempty FCPXML")
    collector = _Collector()
    parser = expat.ParserCreate()
    parser.StartElementHandler = collector.start
    parser.CharacterDataHandler = collector.data
    parser.EndElementHandler = collector.end
    parser.StartDoctypeDeclHandler = _doctype
    try:
        with path.open("rb") as stream:
            parser.ParseFile(stream)
    except expat.ExpatError as error:
        raise ReplayError("Invalid FCPXML") from error
    if (
        collector.sequence_count != 1
        or collector.duration != collector.clip_end
        or not collector.cues
    ):
        raise ReplayError("FCP sequence must have complete clip coverage and captions")
    ordered = tuple(sorted(collector.cues))
    if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:], strict=False)):
        raise ReplayError("Overlapping title cues cannot describe one display caption")
    merged: list[Cue] = []
    for start, end, text in ordered:
        if merged and merged[-1][1] == start and merged[-1][2] == text:
            merged[-1] = (merged[-1][0], end, text)
        else:
            merged.append((start, end, text))
    return tuple(merged)


def srt_text(path: Path) -> str:
    """Serialize the final title display text, preserving line breaks and prefixes."""
    return render_srt(list(caption_cues(path)))
