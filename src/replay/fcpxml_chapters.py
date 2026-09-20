"""Frame-contiguous chapter sizing against actual serialized FCPXML budgets."""

from collections.abc import Callable
from dataclasses import dataclass
from xml.etree import ElementTree

MAX_XML_BYTES = 8 * 1024 * 1024
MAX_XML_ELEMENTS = 50000


@dataclass(frozen=True, slots=True)
class XMLChapter:
    first_frame: int
    end_frame: int
    content: bytes
    elements: int


def partition_chapters(frames: int, build: Callable[[int, int], bytes]) -> tuple[XMLChapter, ...]:
    """Bisect oversized owned documents, never omit frames or raise native limits.

    The builder must generate a complete self-contained project for a half-open
    output-frame range. Media and DTD validation remain separate requirements.
    """
    if type(frames) is not int or not 1 <= frames <= 648000:
        raise ValueError("Invalid chapter timeline")
    pending = [(0, frames)]
    result: list[XMLChapter] = []
    while pending:
        first, end = pending.pop()
        content = build(first, end)
        if b"<!DOCTYPE" in content or b"<!ENTITY" in content:
            raise ValueError("External or declared XML entities are forbidden")
        elements = 0
        if len(content) <= MAX_XML_BYTES:
            root = ElementTree.fromstring(content)
            if root.tag != "fcpxml":
                raise ValueError("Expected FCPXML project")
            elements = len(list(root.iter()))
        if len(content) <= MAX_XML_BYTES and elements <= MAX_XML_ELEMENTS:
            result.append(XMLChapter(first, end, content, elements))
            continue
        if end - first <= 1:
            raise ValueError("Single frame exceeds FCPXML budget")
        if len(result) + len(pending) >= 1023:
            raise ValueError("Chapter count budget exceeded")
        middle = (first + end) // 2
        pending.extend(((middle, end), (first, middle)))
    return tuple(result)
