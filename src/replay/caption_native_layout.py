"""Cached native font measurements connected to POS-aware caption layout.

This builds layout candidates, not Motion/headroom/contrast approval. The caller
supplies and binds an independently observed Motion line advance for this font.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from replay import caption_linguistics, caption_style_defaults
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import StageError, execute_command, file_digest

MAX_CACHE = 65_536
MAX_BATCH_UNITS = 128
MAX_RESPONSE_BYTES = 1_048_576


@dataclass(frozen=True, slots=True)
class NativeCaptionStyle:
    font_name: str
    ass_font_size: float
    border: float
    rim: float
    motion_line_advance: float

    def __post_init__(self) -> None:
        numbers = (self.ass_font_size, self.border, self.rim, self.motion_line_advance)
        if (
            not self.font_name
            or len(self.font_name) > 100
            or "\0" in self.font_name
            or any(isinstance(n, bool) or not math.isfinite(n) for n in numbers)
            or not 1 <= self.ass_font_size <= 256
            or not 0 <= self.border <= 16
            or not 0 <= self.rim <= 16
            or not self.ass_font_size <= self.motion_line_advance <= 4 * self.ass_font_size
        ):
            raise ReplayError("Invalid native caption style or observed line advance")


def _positive(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ReplayError("Native caption bounds must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 16_384:
        raise ReplayError("Native caption bounds must be finite and bounded")
    return result


class NativeCaptionLayout:
    """One private, non-overwriting measurement run; unchanged text is cached.

    Native batch limits stay unchanged. Small request batches reduce process
    overhead; character counts only bound request resources, never text fit.
    Failed/partial artifacts remain. Create a fresh directory for a new run.
    Call verify_inputs before publishing its plan. No app UI or fallback renderer.
    """

    def __init__(
        self,
        *,
        executable: Path,
        directory: Path,
        style: NativeCaptionStyle,
        width: float,
        height: float,
        evidence: tuple[Path, ...],
        expected_point_size: float | None = None,
        execute: Callable[[tuple[str, ...], Path, int], None] = execute_command,
    ) -> None:
        if any(
            isinstance(v, bool) or not math.isfinite(v) or not 0 < v <= 16_384
            for v in (width, height)
        ):
            raise ReplayError("Invalid caption display region")
        if not evidence:
            raise ReplayError("Bind observed font calibration evidence before layout")
        if expected_point_size is not None and (
            isinstance(expected_point_size, bool)
            or not math.isfinite(expected_point_size)
            or not 1 <= expected_point_size <= 1024
        ):
            raise ReplayError("Invalid expected native font point size")
        self.executable = executable.resolve(strict=True)
        self.directory = directory.absolute()
        self.style, self.width, self.height = style, width, height
        self.expected_point_size = expected_point_size
        self._execute = execute
        self._cache: dict[str, tuple[float, float]] = {}
        self._batch = 0
        sources = (self.executable, Path(__file__), Path(caption_linguistics.__file__), *evidence)
        self._inputs = {str(path.resolve()): file_digest(path) for path in sources}
        self.directory.mkdir(mode=0o700)
        self._save(
            self.directory / "inputs.json",
            {
                "inputs": self._inputs,
                "style": asdict(style),
                "width": width,
                "height": height,
                "expectedPointSize": expected_point_size,
                "motionRenderVerified": False,
            },
        )

    @staticmethod
    def _save(path: Path, value: object) -> None:
        with path.open("x", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)

    def verify_inputs(self) -> dict[str, str]:
        if any(file_digest(Path(path)) != digest for path, digest in self._inputs.items()):
            raise ReplayError("Native caption layout inputs changed")
        return dict(self._inputs)

    def _request(self, texts: tuple[str, ...]) -> None:
        stem = self.directory / f"batch-{self._batch:06d}"
        self._batch += 1
        request, response = stem.with_suffix(".request.json"), stem.with_suffix(".response.json")
        self._save(
            request,
            {
                "fontName": self.style.font_name,
                "assFontSize": self.style.ass_font_size,
                "border": self.style.border,
                "rim": self.style.rim,
                "texts": texts,
            },
        )
        request_hash = file_digest(request)
        command = (str(self.executable), "measure-media", "caption_text_measure", str(request))
        try:
            self._execute(command, response, 60)
        except StageError:
            # The tool can abort in exit()'s stdio flush after writing its answer (an
            # os_unfair_lock race seen twice in thousands of batches). A crashed run is never
            # used: keep its output as evidence and measure the batch once more.
            response.rename(stem.with_suffix(".crashed.json"))
            self._execute(command, response, 60)
        if file_digest(request) != request_hash:
            raise ReplayError("Native caption measurement request changed during execution")
        if response.is_symlink() or not 0 < response.stat().st_size <= MAX_RESPONSE_BYTES:
            raise ReplayError("Invalid native caption measurement response")
        data = record(json.loads(response.read_text(encoding="utf-8")))
        font = record(data.get("font"))
        bounds = data.get("bounds")
        if self.expected_point_size is not None and not math.isclose(
            _positive(font.get("pointSize")), self.expected_point_size, rel_tol=0, abs_tol=1e-6
        ):
            raise ReplayError("Native measurement point size differs from the Motion font")
        if (
            data.get("motionRenderVerified") is not False
            or font.get("postScriptName") != self.style.font_name
            or _positive(font.get("lineAdvance")) != self.style.ass_font_size
            or not isinstance(bounds, list)
            or len(bounds) != len(texts)
        ):
            raise ReplayError("Native caption response does not match its request")
        measured: dict[str, tuple[float, float]] = {}
        for text, raw in zip(texts, bounds, strict=True):
            item = record(raw)
            width, height = _positive(item.get("width")), _positive(item.get("height"))
            line_widths = item.get("lineWidths")
            if (
                item.get("text") != text
                or not isinstance(line_widths, list)
                or len(line_widths) != 1
                or _positive(line_widths[0]) != width
            ):
                raise ReplayError("Native caption response text or line geometry changed")
            measured[text] = (width, height)
        self._inputs[str(request)] = request_hash
        self._inputs[str(response)] = file_digest(response)
        self._cache.update(measured)

    def prepare_lines(self, texts: Iterable[str]) -> None:
        """Batch uncached single-line candidates without changing native limits."""
        pending = tuple(text for text in dict.fromkeys(texts) if text not in self._cache)
        if len(self._cache) + len(pending) > MAX_CACHE:
            raise ReplayError("Native caption measurement cache budget exceeded")
        batch: list[str] = []
        units = 0
        for text in pending:
            count = len(text.encode("utf-16-le")) // 2
            if not 1 <= count <= 120 or any(c in text for c in "\r\n\0"):
                raise ReplayError("Native caption measurement requires bounded single-line text")
            if batch and (units + count > MAX_BATCH_UNITS or len(batch) == 16):
                self._request(tuple(batch))
                batch, units = [], 0
            batch.append(text)
            units += count
        if batch:
            self._request(tuple(batch))

    def measure(self, text: str) -> tuple[float, float]:
        lines = text.split("\n")
        if not 1 <= len(lines) <= 3:
            raise ReplayError("Measured captions require one to three lines")
        self.prepare_lines(lines)
        sizes = [self._cache[line] for line in lines]
        return max(size[0] for size in sizes), max(size[1] for size in sizes) + (
            len(lines) - 1
        ) * self.style.motion_line_advance

    def __call__(self, text: str) -> str:
        return caption_linguistics.wrap_measured_lines(
            text,
            measure=self.measure,
            width=self.width,
            height=self.height,
            prepare_lines=self.prepare_lines,
            maximum_lines=3,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--text", action="append", required=True)
    parser.add_argument("--font-name", default=caption_style_defaults.FONT_POSTSCRIPT_NAME)
    parser.add_argument("--font-size", type=float, required=True)
    parser.add_argument("--border", type=float, required=True)
    parser.add_argument("--rim", type=float, required=True)
    parser.add_argument("--line-advance", type=float, required=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--height", type=float, required=True)
    args = parser.parse_args()
    layout = NativeCaptionLayout(
        executable=args.executable,
        directory=args.directory,
        style=NativeCaptionStyle(
            args.font_name, args.font_size, args.border, args.rim, args.line_advance
        ),
        width=args.width,
        height=args.height,
        evidence=tuple(args.evidence),
        execute=execute_command,
    )
    results: list[dict[str, object]] = []
    for text in args.text:
        wrapped = layout(text)
        width, height = layout.measure(wrapped)
        results.append({"sourceText": text, "text": wrapped, "width": width, "height": height})
    layout._save(
        layout.directory / "layouts.json",
        {
            "inputs": layout.verify_inputs(),
            "layouts": results,
            "motionRenderVerified": False,
            "captionGeometryApproved": False,
            "deliveryComplete": False,
        },
    )


if __name__ == "__main__":
    main()
