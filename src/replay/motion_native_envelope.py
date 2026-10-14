"""Calibrated Motion-origin candidates from the existing cached native measurer.

This does not certify unseen rendered text. Verify inputs before consuming the
plan, then independently check the actual Motion render and its full support.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from replay import foreground_evidence
from replay import tracking_observations as storage
from replay.caption_native_layout import (
    MAX_CACHE,
    MAX_RESPONSE_BYTES,
    NativeCaptionLayout,
    NativeCaptionStyle,
)
from replay.foreground_evidence import _bound_bytes
from replay.recordings import record
from replay.stages import execute_command, file_digest
from replay.vertical_captions import Box

MAX_EVIDENCE = 32
MAX_PADDING = 64
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_CANVAS_DIMENSION = 8192
MAX_PLAN_TEXTS = 32


@dataclass(frozen=True, slots=True)
class NativeFontMetrics:
    postscript_name: str
    family: str
    face: str
    point_size: float
    line_advance: float
    line_ascent: float
    rendered_ascent: float


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Caption calibration requires finite numbers")
    if not 0 <= value <= 1024 or not math.isfinite(value):
        raise ValueError("Caption calibration number exceeds its budget")
    return float(value)


def _text(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 200 or "\0" in value:
        raise ValueError("Invalid calibrated font name")
    return value


def _font(value: object) -> NativeFontMetrics:
    raw = record(value)
    if set(raw) != {
        "postScriptName",
        "family",
        "face",
        "pointSize",
        "lineAdvance",
        "lineAscent",
        "renderedAscent",
    }:
        raise ValueError("Invalid calibrated native font fields")
    result = NativeFontMetrics(
        _text(raw["postScriptName"]),
        _text(raw["family"]),
        _text(raw["face"]),
        _number(raw["pointSize"]),
        _number(raw["lineAdvance"]),
        _number(raw["lineAscent"]),
        _number(raw["renderedAscent"]),
    )
    if min(result.point_size, result.line_advance, result.line_ascent, result.rendered_ascent) <= 0:
        raise ValueError("Calibrated font metrics must be positive")
    return result


def _style(value: object) -> NativeCaptionStyle:
    raw = record(value)
    if set(raw) != {"font_name", "ass_font_size", "border", "rim", "motion_line_advance"}:
        raise ValueError("Invalid calibrated caption style fields")
    return NativeCaptionStyle(
        _text(raw["font_name"]),
        _number(raw["ass_font_size"]),
        _number(raw["border"]),
        _number(raw["rim"]),
        _number(raw["motion_line_advance"]),
    )


def _canvas(value: object) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Calibration requires width and height")
    width, height = storage._integer(value[0]), storage._integer(value[1])
    if not 0 < min(width, height) <= max(width, height) <= MAX_CANVAS_DIMENSION:
        raise ValueError("Calibration canvas exceeds its budget")
    return width, height


class CalibratedMotionEnvelope:
    """Bind a calibration to one NativeCaptionLayout without changing its producer.

    Calibration JSON version 1 contains exact style/nativeFont/canvas, observed padding,
    a template reference and 1..32 evidence references (absolute path + sha256).
    Metric origin is canvas-center-X and upward-positive Motion Y; the returned
    Box uses top-left canvas coordinates relative to that origin. Thus a canvas
    origin (x,y) maps to Motion (x-W/2,H/2-y). No source framing changes occur.
    """

    def __init__(
        self,
        layout: NativeCaptionLayout,
        *,
        calibration: Path,
        calibration_sha256: str,
        canvas: tuple[int, int],
    ) -> None:
        data = _bound_bytes(calibration, calibration_sha256, storage.CONFIG_BYTES)
        raw = record(json.loads(data))
        if (
            set(raw)
            != {"version", "style", "nativeFont", "canvas", "padding", "template", "evidence"}
            or type(raw["version"]) is not int
            or raw["version"] != 1
        ):
            raise ValueError("Invalid Motion envelope calibration")
        self._canvas = _canvas(list(canvas))
        if _canvas(raw["canvas"]) != self._canvas:
            raise ValueError("Motion calibration canvas does not match output")
        self._font = _font(raw["nativeFont"])
        self._padding = _number(raw["padding"])
        if (
            _style(raw["style"]) != layout.style
            or self._font.postscript_name != layout.style.font_name
            or self._font.line_advance != layout.style.ass_font_size
            or self._padding > MAX_PADDING
        ):
            raise ValueError("Motion calibration does not match native caption style")
        evidence = raw["evidence"]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= MAX_EVIDENCE:
            raise ValueError("Motion envelope requires bounded calibration evidence")
        self._layout = layout
        self._configuration = (layout.executable, layout.directory, layout.style)
        self._next_response = 0
        self._inputs = {
            str(calibration.resolve(strict=True)): calibration_sha256,
        }
        sources = (Path(__file__), Path(storage.__file__), Path(foreground_evidence.__file__))
        self._inputs.update({str(path.resolve()): file_digest(path) for path in sources})
        self._bind(raw["template"])
        for reference in evidence:
            self._bind(reference)
        self.verify_inputs()

    @property
    def canvas(self) -> tuple[int, int]:
        return self._canvas

    def _check_configuration(self) -> None:
        if (
            self._layout.executable,
            self._layout.directory,
            self._layout.style,
        ) != self._configuration:
            raise ValueError("Native measurement configuration changed")

    def _bind(self, value: object) -> None:
        raw = record(value)
        if set(raw) != {"path", "sha256"}:
            raise ValueError("Calibration references require a path and digest")
        path, digest = storage._path(raw["path"]), storage._digest(raw["sha256"])
        if str(path) in self._inputs:
            raise ValueError("Calibration evidence aliases another input")
        _bound_bytes(path, digest, MAX_EVIDENCE_BYTES)
        self._inputs[str(path)] = digest

    def _producer_bytes(self, path: Path) -> bytes:
        digest = self._layout._inputs.get(str(path))
        if digest is None:
            raise ValueError("Native measurement attempt lacks completed provenance")
        data = _bound_bytes(path, digest, MAX_RESPONSE_BYTES)
        self._inputs[str(path)] = digest
        return data

    def _check_request(self, path: Path) -> None:
        raw = record(json.loads(self._producer_bytes(path)))
        if set(raw) != {"fontName", "assFontSize", "border", "rim", "texts"}:
            raise ValueError("Invalid calibrated native measurement request")
        measured = (
            _text(raw["fontName"]),
            _number(raw["assFontSize"]),
            _number(raw["border"]),
            _number(raw["rim"]),
        )
        style = self._configuration[2]
        if measured != (style.font_name, style.ass_font_size, style.border, style.rim):
            raise ValueError("Native request settings differ from Motion calibration")

    def _read_fonts(self) -> None:
        # The frozen producer increments its counter even on failed requests.
        # Reject those gaps; do not mistake a missing filename for end of evidence.
        while self._next_response < self._layout._batch:
            if self._next_response >= MAX_CACHE:
                raise ValueError("Native font response budget exceeded")
            stem = self._layout.directory / f"batch-{self._next_response:06d}"
            self._check_request(stem.with_suffix(".request.json"))
            data = self._producer_bytes(stem.with_suffix(".response.json"))
            if _font(record(json.loads(data)).get("font")) != self._font:
                raise ValueError("Fresh native font metrics differ from Motion calibration")
            self._next_response += 1
        if self._next_response == 0:
            raise ValueError("Native font response evidence is missing")

    def __call__(self, text: str) -> Box:
        self._check_configuration()
        width, height = self._layout.measure(text)
        self._read_fonts()
        top = -self._padding - self._font.rendered_ascent
        return Box(-width / 2, top, width / 2, top + height)

    def verify_inputs(self) -> dict[str, str]:
        """Verify every consumed response and both producers before plan/render use."""
        self._check_configuration()
        if self._layout._batch or self._next_response:
            self._read_fonts()
        inputs = self._layout.verify_inputs()
        for name, digest in self._inputs.items():
            _bound_bytes(Path(name), digest, MAX_EVIDENCE_BYTES)
            if name in inputs and inputs[name] != digest:
                raise ValueError("Calibrated Motion envelope inputs changed")
            inputs[name] = digest
        return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--calibration-sha256", required=True)
    parser.add_argument("--canvas", type=int, nargs=2, required=True)
    parser.add_argument("--text", action="append", required=True)
    args = parser.parse_args()
    if not 1 <= len(args.text) <= MAX_PLAN_TEXTS:
        raise ValueError("Bounded Motion envelope plan text budget exceeded")
    raw = record(
        json.loads(_bound_bytes(args.calibration, args.calibration_sha256, storage.CONFIG_BYTES))
    )
    canvas = _canvas(args.canvas)
    layout = NativeCaptionLayout(
        executable=args.executable,
        directory=args.directory,
        style=_style(raw.get("style")),
        width=canvas[0],
        height=canvas[1],
        evidence=(args.calibration,),
        execute=execute_command,
    )
    envelope = CalibratedMotionEnvelope(
        layout,
        calibration=args.calibration,
        calibration_sha256=args.calibration_sha256,
        canvas=canvas,
    )
    results = [{"text": text, "bounds": asdict(envelope(text))} for text in args.text]
    layout._save(
        layout.directory / "envelopes.json",
        {
            "inputs": envelope.verify_inputs(),
            "canvas": canvas,
            "envelopes": results,
            "motionRenderVerified": False,
            "captionGeometryApproved": False,
            "deliveryComplete": False,
        },
    )


if __name__ == "__main__":
    main()
