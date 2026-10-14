"""Read-only full-frame geometry audit; never promote static bounds to style approval."""

import argparse
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay import (
    caption_style_defaults,
    motion_caption,
    motion_finish,
    motion_native_envelope,
    motion_stages,
    stages,
    workflow,
)
from replay.caption_native_layout import _positive
from replay.motion_native_envelope import _number
from replay.recordings import record
from replay.stages import _fingerprint, file_digest
from replay.workflow import load_stages

MAX_BYTES = 64 * 1024 * 1024
FRAME_HEADER = re.compile(r"frame:(\d+)\s+pts:(\d+)\s+pts_time:([0-9.]+)")
BOUND_KEYS = ("x1", "x2", "y1", "y2", "w", "h")


@dataclass(frozen=True)
class Plan:
    frames: int
    width: int
    height: int
    rate: int
    states: tuple[motion_caption.CaptionPlacement, ...]


class Inputs:
    """Freeze explicit repository-local inputs, including transitive receipts."""

    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}

    def path(self, value: object) -> Path:
        if not isinstance(value, str):
            raise ValueError("Expected an input path")
        path = Path(value).absolute()
        if path.resolve() != path or not path.is_relative_to(Path.cwd().resolve()):
            raise ValueError("Inputs and outputs must be unlinked checkout-local paths")
        return path

    def bind(self, path: Path) -> str:
        path = self.path(str(path))
        digest = file_digest(path)
        if self.hashes.setdefault(str(path), digest) != digest:
            raise ValueError("Input changed during verification")
        return digest

    def text(self, path: Path) -> str:
        self.bind(path)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Evidence exceeds byte budget")
        return path.read_text(encoding="utf-8")

    def json(self, path: Path) -> dict[str, object]:
        return record(json.loads(self.text(path)))

    def verify(self) -> None:
        for name in tuple(self.hashes):
            self.bind(Path(name))


def parse_plan(raw: dict[str, object]) -> Plan:
    if set(raw) != {"frames", "width", "height", "rate", "expectedText", "placements"}:
        raise ValueError("Invalid stage plan")
    frames, width, height, rate = (
        motion_stages._integer(raw[key]) for key in ("frames", "width", "height", "rate")
    )
    entries = raw["placements"]
    expected_text = motion_stages._text(raw["expectedText"])
    if (
        not expected_text
        or not 1 <= rate <= 60
        or not 1 <= frames <= 120 * rate
        or not 1 <= min(width, height) <= max(width, height) <= 8192
        or not isinstance(entries, list)
        or not 1 <= len(entries) <= motion_caption.MAX_SEQUENCE_SENTENCES
    ):
        raise ValueError("Invalid bounded plan dimensions or states")
    states = tuple(motion_stages._placement(entry, rate) for entry in entries)
    previous = 0
    for state in states:
        if (
            not previous <= state.start_frame < state.end_frame <= frames
            or not state.text.strip()
            or len(state.text) > 120
            or not 1 <= len(state.text.split("\n")) <= 2
            or not math.isfinite(state.x)
            or not math.isfinite(state.y)
        ):
            raise ValueError("Invalid state text, geometry or frame interval")
        previous = state.end_frame
    return Plan(frames, width, height, rate, states)


def check_clocks(probe: dict[str, object], md5: str, plan: Plan) -> Fraction:
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise ValueError("Missing probe streams")
    videos = [record(item) for item in streams if record(item).get("codec_type") == "video"]
    if len(videos) != 1:
        raise ValueError("Expected one video stream")
    video = videos[0]
    base = Fraction(str(video.get("time_base")))
    if (
        base <= 0
        or video.get("width") != plan.width
        or video.get("height") != plan.height
        or video.get("start_pts") != 0
        or video.get("nb_frames") != str(plan.frames)
        or video.get("nb_read_frames") != str(plan.frames)
        or Fraction(str(video.get("avg_frame_rate"))) != plan.rate
        or Fraction(str(video.get("duration_ts"))) * base != Fraction(plan.frames, plan.rate)
    ):
        raise ValueError("Probe does not match complete planned clock")
    lines = md5.splitlines()
    if (
        lines.count(f"#tb 0: 1/{plan.rate}") != 1
        or lines.count(f"#dimensions 0: {plan.width}x{plan.height}") != 1
    ):
        raise ValueError("framemd5 timebase or dimensions differ")
    rows = [line.split(",") for line in lines if line and not line.startswith("#")]
    if len(rows) != plan.frames:
        raise ValueError("Missing or extra decoded frames")
    for index, row in enumerate(rows):
        if (
            len(row) != 6
            or [int(value) for value in row[:4]] != [0, index, index, 1]
            or int(row[4]) <= 0
            or re.fullmatch(r"[0-9a-f]{32}", row[5].strip()) is None
        ):
            raise ValueError("Invalid decoded frame identity")
    return base


def parse_bounds(text: str, *, plan: Plan, base: Fraction) -> list[tuple[int, int, int, int]]:
    lines = text.splitlines()
    if len(lines) != plan.frames * 7:
        raise ValueError("Missing, extra or invalid alpha metadata")
    result: list[tuple[int, int, int, int]] = []
    for index in range(plan.frames):
        block = lines[index * 7 : (index + 1) * 7]
        header = FRAME_HEADER.fullmatch(block[0])
        if (
            header is None
            or int(header[1]) != index
            or int(header[2]) * base != Fraction(index, plan.rate)
            or Fraction(header[3])
            not in {
                Fraction(format(index / plan.rate, ".6g")),
                Fraction(format(index / plan.rate, ".6f")),
            }
        ):
            raise ValueError("Alpha metadata frame or PTS mismatch")
        values: dict[str, int] = {}
        for line in block[1:]:
            key, separator, value = line.partition("=")
            key = key.removeprefix("lavfi.bbox.")
            if (
                not line.startswith("lavfi.bbox.")
                or not separator
                or key not in BOUND_KEYS
                or key in values
            ):
                raise ValueError("Invalid or duplicate alpha bound key")
            values[key] = int(value)
        x1, x2, y1, y2, width, height = (values[key] for key in BOUND_KEYS)
        if (
            not 0 <= x1 <= x2 < plan.width
            or not 0 <= y1 <= y2 < plan.height
            or width != x2 - x1 + 1
            or height != y2 - y1 + 1
        ):
            raise ValueError("Invalid alpha support rectangle")
        result.append((x1, y1, x2 + 1, y2 + 1))
    return result


def check_template(
    document: str, *, plan: Plan, font: motion_native_envelope.NativeFontMetrics
) -> None:
    root = motion_caption._parse(document)
    settings = motion_caption._one(root, "scene/sceneSettings")
    expected_settings = {
        "width": plan.width,
        "height": plan.height,
        "duration": plan.frames,
        "frameRate": plan.rate,
        "NTSC": 0,
    }
    if any(settings.findtext(key) != str(value) for key, value in expected_settings.items()):
        raise ValueError("Template canvas or clock differs from plan")
    nodes = motion_caption._one(root, "scene/layer").findall("scenenode[@factoryID='16']")
    if len(nodes) != len(plan.states) * 3:
        raise ValueError("Template stage count differs")
    for index, node in enumerate(nodes):
        state = plan.states[index // 3]
        timing = motion_caption._one(node, "timing")
        expected = {
            "in": f"{state.start_frame} {plan.rate} 1 0",
            "out": f"{state.end_frame - 1} {plan.rate} 1 0",
            "offset": f"{state.start_frame} {plan.rate} 1 0",
        }
        position = motion_caption._one(
            node, "parameter[@id='1']/parameter[@id='100']/parameter[@id='101']"
        )
        reveal = motion_caption._one(node, "parameter[@id='2']/parameter[@id='319']")
        curve = motion_caption._one(reveal, "parameter[@id='321']/curve")
        if (
            timing.attrib != expected
            or node.findtext("parameter[@id='2']/parameter[@id='369']/text") != state.text
            or node.findtext("style/parameter[@id='83']/font") != font.postscript_name
            or float(motion_caption._one(node, "style/parameter[@id='3']").get("value", "nan"))
            != font.point_size
            or float(motion_caption._parameter(position, "1").get("value", "nan")) != state.x
            or float(motion_caption._parameter(position, "2").get("value", "nan")) != state.y
            or motion_caption._parameter(reveal, "322").get("value") != "0"
            or ET.tostring(curve)
            != ET.tostring(motion_caption._reveal_curve(state.samples, smooth=False))
        ):
            raise ValueError("Template text/font/geometry/static reveal differs from plan")


def native_sizes(
    request: dict[str, object], response: dict[str, object]
) -> dict[str, tuple[float, float]]:
    texts, bounds = request.get("texts"), response.get("bounds")
    if (
        response.get("motionRenderVerified") is not False
        or not isinstance(texts, list)
        or not isinstance(bounds, list)
        or not 1 <= len(texts) == len(bounds) <= 32
    ):
        raise ValueError("Invalid native request/response")
    result: dict[str, tuple[float, float]] = {}
    for text, value in zip(texts, bounds, strict=True):
        item = record(value)
        widths = item.get("lineWidths")
        if (
            not isinstance(text, str)
            or text in result
            or item.get("text") != text
            or not isinstance(widths, list)
            or len(widths) != len(text.split("\n"))
        ):
            raise ValueError("Native text/line correspondence differs")
        width, height = (_positive(item.get(key)) for key in ("width", "height"))
        if min(width, height) <= 0 or max(_positive(w) for w in widths) != width:
            raise ValueError("Invalid native dimensions")
        result[text] = width, height
    return result


def compare_geometry(
    *,
    plan: Plan,
    bounds: list[tuple[int, int, int, int]],
    sizes: dict[str, tuple[float, float]],
    calibration: dict[str, object],
) -> list[dict[str, object]]:
    font = motion_native_envelope._font(calibration.get("nativeFont"))
    style = motion_native_envelope._style(calibration.get("style"))
    padding = motion_native_envelope._number(calibration.get("padding"))
    if padding > motion_native_envelope.MAX_PADDING or calibration.get("canvas") != [
        plan.width,
        plan.height,
    ]:
        raise ValueError("Invalid calibration padding/canvas")
    results: list[dict[str, object]] = []
    covered: set[int] = set()
    for state in plan.states:
        lines = state.text.split("\n")
        if any(line not in sizes for line in lines):
            raise ValueError("Independent single-line metrics missing")
        width = max(sizes[line][0] for line in lines)
        height = (
            max(sizes[line][1] for line in lines) + (len(lines) - 1) * style.motion_line_advance
        )
        x, y = plan.width / 2 + state.x, plan.height / 2 - state.y
        top = y - padding - font.rendered_ascent
        envelope = (x - width / 2, top, x + width / 2, top + height)
        observed = bounds[state.start_frame : state.end_frame]
        passed = all(
            envelope[0] <= box[0]
            and envelope[1] <= box[1]
            and box[2] <= envelope[2]
            and box[3] <= envelope[3]
            for box in observed
        )
        covered.update(range(state.start_frame, state.end_frame))
        results.append(
            {
                "text": state.text,
                "startFrame": state.start_frame,
                "endFrame": state.end_frame,
                "framesChecked": len(observed),
                "expectedEnvelope": envelope,
                "observedBounds": sorted(set(observed)),
                "contained": passed,
            }
        )
    if covered != set(range(plan.frames)):
        raise ValueError(
            "Nonempty-alpha audit requires full state coverage; gaps need empty-alpha evidence"
        )
    return results


def bind_workflow(
    inputs: Inputs,
    *,
    workflow: Path,
    state: Path,
    outputs: tuple[Path, ...],
    required_inputs: tuple[Path, ...],
) -> None:
    inputs.bind(workflow)
    stages = load_stages(workflow)
    for output in outputs:
        producers = [stage for stage in stages if output in stage.outputs]
        if len(producers) != 1:
            raise ValueError("Missing unique evidence producer")
        stage = producers[0]
        if not set(required_inputs).issubset(stage.inputs):
            raise ValueError("Evidence producer omits a required source")
        for path in stage.inputs + stage.outputs:
            inputs.bind(path)
        receipt = inputs.json(state / f"{stage.name}.json")
        if (
            receipt.get("status") != "complete"
            or receipt.get("fingerprint") != _fingerprint(stage)
            or receipt.get("outputs") != {str(path): inputs.bind(path) for path in stage.outputs}
        ):
            raise ValueError("Stage input/output receipt mismatch")


def verify(config: Path, output: Path) -> dict[str, object]:
    inputs = Inputs()
    raw = inputs.json(config)
    paths = {key: inputs.path(value) for key, value in record(raw.get("paths")).items()}
    required = {
        "plan",
        "template",
        "twoLayer",
        "calibration",
        "nativeRequest",
        "nativeResponse",
        "paddedRequest",
        "paddedResponse",
        "probe",
        "framemd5",
        "bounds",
        "alpha",
        "renderReceipt",
        "prepareWorkflow",
        "prepareState",
        "verifyWorkflow",
        "verifyState",
    }
    if set(paths) != required:
        raise ValueError("Invalid verification input set")
    output = inputs.path(str(output))
    if output.exists():
        raise FileExistsError(output)
    for name, path in paths.items():
        if name not in {"prepareState", "verifyState"}:
            inputs.bind(path)
    for module in (
        motion_caption,
        motion_finish,
        motion_native_envelope,
        motion_stages,
        caption_style_defaults,
        stages,
        workflow,
    ):
        if module.__file__ is None:
            raise ValueError("Missing implementation path")
        inputs.bind(Path(module.__file__))
    inputs.bind(Path(__file__))
    plan = parse_plan(inputs.json(paths["plan"]))
    calibration = inputs.json(paths["calibration"])
    if type(calibration.get("version")) is not int or calibration["version"] != 1:
        raise ValueError("Unsupported historical calibration version")
    font = motion_native_envelope._font(calibration.get("nativeFont"))
    style = motion_native_envelope._style(calibration.get("style"))
    if font.postscript_name != style.font_name or font.line_advance != style.ass_font_size:
        raise ValueError("Calibration style/font correspondence differs")
    request, response = inputs.json(paths["nativeRequest"]), inputs.json(paths["nativeResponse"])
    padded_request, padded_response = (
        inputs.json(paths["paddedRequest"]),
        inputs.json(paths["paddedResponse"]),
    )
    for req, res, border, rim in (
        (request, response, 0, 0),
        (padded_request, padded_response, style.border, style.rim),
    ):
        if motion_native_envelope._font(res.get("font")) != font or (
            req.get("fontName"),
            req.get("assFontSize"),
            req.get("border"),
            req.get("rim"),
        ) != (style.font_name, style.ass_font_size, border, rim):
            raise ValueError("Native font/style reuse mismatch")
    plain, padded = native_sizes(request, response), native_sizes(padded_request, padded_response)
    if set(plain) != set(padded) or any(
        padded[text][0] < size[0] or padded[text][1] < size[1] for text, size in plain.items()
    ):
        raise ValueError("Padded metrics do not contain native glyph metrics")
    # Reuse only explicitly pinned native evidence, not old observed Motion bounds.
    evidence = calibration.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("Missing historical metric bindings")
    pins = {str(record(item).get("path")): record(item).get("sha256") for item in evidence}
    for key in ("nativeRequest", "nativeResponse", "paddedRequest", "paddedResponse", "plan"):
        if pins.get(str(paths[key])) != inputs.bind(paths[key]):
            raise ValueError("Historical native/plan pin mismatch")
    finish = record(raw.get("finish"))
    top, bottom = finish.get("top"), finish.get("bottom")
    if (
        not isinstance(top, list)
        or not isinstance(bottom, list)
        or len(top) != 3
        or len(bottom) != 3
    ):
        raise ValueError("Expected explicit Motion RGB controls")
    selected = motion_finish.selected_finish_style(
        top=(_number(top[0]), _number(top[1]), _number(top[2])),
        bottom=(_number(bottom[0]), _number(bottom[1]), _number(bottom[2])),
        font_name=font.postscript_name,
        font_size=font.point_size,
        stroke_scale=motion_native_envelope._number(finish.get("strokeScale")),
        shadow_scale=motion_native_envelope._number(finish.get("shadowScale")),
    )
    document = inputs.text(paths["template"])
    if document != motion_finish.finish_sequence(inputs.text(paths["twoLayer"]), style=selected):
        raise ValueError("Actual template differs from shared selected-style defaults")
    check_template(document, plan=plan, font=font)
    bind_workflow(
        inputs,
        workflow=paths["prepareWorkflow"],
        state=paths["prepareState"],
        outputs=(paths["template"],),
        required_inputs=(paths["plan"], paths["twoLayer"]),
    )
    bind_workflow(
        inputs,
        workflow=paths["verifyWorkflow"],
        state=paths["verifyState"],
        outputs=(paths["probe"], paths["framemd5"], paths["bounds"]),
        required_inputs=(paths["alpha"],),
    )
    render = inputs.json(paths["renderReceipt"])
    render_inputs = render.get("inputs")
    if (
        not isinstance(render_inputs, list)
        or render.get("status") != "Successful"
        or render.get("terminalInputsVerified") is not True
        or render.get("output") != str(paths["alpha"])
        or not any(
            record(item).get("path") == str(paths["template"])
            and record(item).get("sha256") == inputs.bind(paths["template"])
            for item in render_inputs
        )
    ):
        raise ValueError("Render source/terminal binding missing")
    base = check_clocks(inputs.json(paths["probe"]), inputs.text(paths["framemd5"]), plan)
    bounds = parse_bounds(inputs.text(paths["bounds"]), plan=plan, base=base)
    states = compare_geometry(plan=plan, bounds=bounds, sizes=padded, calibration=calibration)
    inputs.verify()
    result: dict[str, object] = {
        "version": 1,
        "scope": "geometry-only",
        "inputs": inputs.hashes,
        "inputSHA256VerifiedBeforeAndAfter": True,
        "states": states,
        "geometryContained": all(state["contained"] is True for state in states),
        "nativeMetricsReused": True,
        "paddingUnchanged": calibration["padding"],
        "staticNoFadeTemplateMatched": True,
        "selectedStyleXMLMatched": True,
        "newCalibrationIssued": False,
        "remaining": [
            "Rendered colour, white stroke thickness and gradient hold need separate evidence",
            "Static states do not verify animation, unseen text, head clearance, FCP or delivery",
            "Render receipt lacks a terminal output SHA; current alpha is bound by decode stages",
            "Historical font file/producer provenance is not re-observed by this audit",
        ],
        "deliveryComplete": False,
    }
    with output.open("x", encoding="utf-8") as stream:
        output.chmod(0o600)
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = verify(args.config, args.output)
    print(json.dumps({"scope": result["scope"], "geometryContained": result["geometryContained"]}))
    if result["geometryContained"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
