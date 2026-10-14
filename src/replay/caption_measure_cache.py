"""Reuse fingerprint-bound native glyph measurements without replaying producers."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

from replay.caption_long_layout import MAX_SINGLE_LINE_UNITS, LongLineCaptionLayout
from replay.caption_native_layout import (
    MAX_CACHE,
    MAX_RESPONSE_BYTES,
    NativeCaptionLayout,
    _positive,
)
from replay.core import ReplayError
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import file_digest


class CaptionMeasurer(Protocol):
    width: float
    expected_point_size: float | None

    def prepare_lines(self, texts: Iterable[str]) -> None: ...

    def measure(self, text: str) -> tuple[float, float]: ...


class CachedCaptionLayout:
    """Old measurements remain data; missing text goes through the live measurer."""

    def __init__(self, native: NativeCaptionLayout) -> None:
        self.native = native
        self.width = native.width
        self.expected_point_size = native.expected_point_size
        self._cache: dict[str, tuple[float, float]] = {}
        self._pins: dict[str, str] = {}

    def reuse(self, *, directory: Path, inputs: Mapping[str, str]) -> None:
        directory = directory.resolve(strict=True)
        requests = sorted(
            Path(path)
            for path in inputs
            if Path(path).parent == directory and Path(path).name.endswith(".request.json")
        )
        if not requests or len(requests) > MAX_CACHE:
            raise ReplayError("Expected bounded pinned native measurement requests")
        for request in requests:
            response = request.with_name(
                request.name.removesuffix(".request.json") + ".response.json"
            )
            request_hash, response_hash = inputs[str(request)], inputs.get(str(response))
            if response_hash is None:
                raise ReplayError("Cached native request lacks its pinned response")
            raw = record(json.loads(_bound_bytes(request, request_hash, MAX_RESPONSE_BYTES)))
            data = record(json.loads(_bound_bytes(response, response_hash, MAX_RESPONSE_BYTES)))
            limit = 120
            if isinstance(self.native, LongLineCaptionLayout):
                texts = raw.get("texts")
                if isinstance(texts, list) and any(
                    isinstance(text, str) and len(text.encode("utf-16-le")) // 2 > limit
                    for text in texts
                ):
                    self.native.verify_long_inputs(inputs)
                    limit = MAX_SINGLE_LINE_UNITS
            self._adopt(raw, data, maximum_units=limit)
            self._pins.update({str(request): request_hash, str(response): response_hash})

    def _adopt(
        self, request: dict[str, object], response: dict[str, object], *, maximum_units: int
    ) -> None:
        style = self.native.style
        font = record(response.get("font"))
        texts, rows = request.get("texts"), response.get("bounds")
        if (
            request.get("fontName") != style.font_name
            or request.get("assFontSize") != style.ass_font_size
            or request.get("border") != style.border
            or request.get("rim") != style.rim
            or response.get("motionRenderVerified") is not False
            or font.get("postScriptName") != style.font_name
            or _positive(font.get("lineAdvance")) != style.ass_font_size
            or self.expected_point_size is None
            or abs(_positive(font.get("pointSize")) - self.expected_point_size) > 1e-6
            or not isinstance(texts, list)
            or not isinstance(rows, list)
            or not 1 <= len(texts) == len(rows) <= 16
        ):
            raise ReplayError("Cached native measurement style or font differs")
        for text, value in zip(texts, rows, strict=True):
            row = record(value)
            width, height = _positive(row.get("width")), _positive(row.get("height"))
            if (
                not isinstance(text, str)
                or not 1 <= len(text.encode("utf-16-le")) // 2 <= maximum_units
                or any(character in text for character in "\r\n\0")
                or row.get("text") != text
                or row.get("lineWidths") != [width]
                or (text in self._cache and self._cache[text] != (width, height))
                or len(self._cache) >= MAX_CACHE
            ):
                raise ReplayError("Cached native glyph geometry is inconsistent")
            self._cache[text] = (width, height)

    def prepare_lines(self, texts: Iterable[str]) -> None:
        missing = tuple(text for text in dict.fromkeys(texts) if text not in self._cache)
        if len(self._cache) + len(missing) > MAX_CACHE:
            raise ReplayError("Cached native measurements exceed resource budget")
        self.native.prepare_lines(missing)
        self._cache.update((text, self.native.measure(text)) for text in missing)

    def measure(self, text: str) -> tuple[float, float]:
        lines = text.split("\n")
        if not 1 <= len(lines) <= 3:
            raise ReplayError("Caption measurement requires one to three lines")
        self.prepare_lines(lines)
        sizes = tuple(self._cache[line] for line in lines)
        return max(size[0] for size in sizes), max(size[1] for size in sizes) + (
            len(lines) - 1
        ) * self.native.style.motion_line_advance

    def verify_inputs(self) -> dict[str, str]:
        if any(file_digest(Path(path)) != digest for path, digest in self._pins.items()):
            raise ReplayError("Reused native measurement changed")
        return self.native.verify_inputs() | self._pins
