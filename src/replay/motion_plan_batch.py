"""Expand a bounded Motion title-plan bundle into resumable styled title assets.

The input bundle contains measured caption text and clocks. A caller supplies an
explicit normalized vertical anchor; the horizontal anchor is derived from the
bundle's measured safe-slot centre. Outputs are experimental Motion documents,
not proof of FCP import, compositing or delivery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path

from replay import caption_style_defaults, motion_finish, motion_stages
from replay.core import ReplayError
from replay.motion_caption import MAX_XML_CHARACTERS, _one, _parse
from replay.recordings import record

MAX_BUNDLE_BYTES = 64 * 1024 * 1024
# Measured line advance of rendered bottom-anchored captions: ProRes alpha
# renders at font 104 on the 1920x1080 output show single-line boxes at
# y[828,1008] with every multi-line top pinned at the same first-line top,
# so blocks grow downward by ~138px per line (glyph-top to glyph-top across
# two independent renders, ±12px glyph-dependent variation). The template
# carries no explicit leading, so the advance scales linearly with the
# render font size from this 104px reference measurement.
MEASURED_LINE_ADVANCE_PX = 138.0
MEASURED_LINE_ADVANCE_FONT_SIZE = 104.0


def _anchored_y(
    text: str,
    *,
    height: int,
    anchor_y_ratio: Fraction,
    font_size: float,
    output_height: int,
) -> float:
    """Bottom-anchored slot: extra lines grow upward from the anchor.

    The anchor marks the single-line position; each additional line shifts
    the block up by one measured advance (converted to canvas units) so the
    block bottom stays where the single-line bottom was verified.
    """
    if not isinstance(text, str) or not text:
        raise ReplayError("Caption text must be nonempty")
    if (
        not isinstance(height, int)
        or not isinstance(output_height, int)
        or height <= 0
        or output_height <= 0
        or not isinstance(font_size, float)
        or not font_size > 0
        or not isinstance(anchor_y_ratio, Fraction)
        or not 0 < anchor_y_ratio < 1
    ):
        raise ReplayError("Invalid bottom-anchor geometry")
    lines = text.count("\n") + 1
    advance = (
        MEASURED_LINE_ADVANCE_PX
        * (font_size / MEASURED_LINE_ADVANCE_FONT_SIZE)
        * (height / output_height)
    )
    return height * float(anchor_y_ratio) - (lines - 1) * advance


MAX_PARTS = 20_000
MAX_PLACEMENTS = 500_000
MAX_CUES_PER_TITLE = 32
MAX_TITLE_SECONDS = 120


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, *, limit: int) -> Path:
    absolute = path.absolute()
    if absolute.resolve() != absolute or absolute.is_symlink():
        raise ReplayError("Motion batch paths must not contain symlinks")
    try:
        info = absolute.stat()
    except OSError as error:
        raise ReplayError("Motion batch input is unavailable") from error
    if not absolute.is_file() or info.st_nlink != 1 or not 0 < info.st_size <= limit:
        raise ReplayError("Motion batch inputs must be bounded regular single-link files")
    return absolute


def _mapping(value: object, *, name: str, keys: set[str]) -> dict[str, object]:
    item = record(value)
    if set(item) != keys:
        raise ReplayError(f"Invalid Motion plan {name} fields")
    return item


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ReplayError(f"Invalid Motion plan {name}")
    return value


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError(f"Invalid Motion plan {name}")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayError(f"Invalid Motion plan {name}")
    return result


def _style_controls(template: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    root = _parse(template)
    nodes = root.findall(".//scenenode[@factoryID='16']")
    faces = [node for node in nodes if node.get("name") == "Caption Face"]
    if len(nodes) != 2 or len(faces) != 1:
        raise ReplayError("Motion template must contain one synchronized face/rim pair")
    node = faces[0]
    style = _one(node, "style")
    top = motion_finish._top_color(style)
    stops = _one(
        _one(_one(style, "parameter[@id='14']"), "parameter[@id='17']"), "parameter[@id='1']"
    ).findall("parameter[@factoryID='4']")
    if len(stops) != 2:
        raise ReplayError("Motion template must have two selected-style gradient stops")
    color = _one(stops[1], "parameter[@id='3']")
    bottom = tuple(
        float(_one(_one(color, f"parameter[@id='{identifier}']"), "curve").get("value", "nan"))
        for identifier in ("1", "2", "3")
    )
    return top, (bottom[0], bottom[1], bottom[2])


def _unique_json(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayError("Motion title-plan JSON contains duplicate keys")
        result[key] = value
    return result


def _planned_parts(
    title_parts: list[object],
    *,
    width: int,
    height: int,
    rate: int,
    center_x: float,
    anchor_y_ratio: Fraction,
    font_size: float,
    output_height: int,
) -> tuple[int, int, list[dict[str, object]]]:
    """Coalesce overlapping cue intervals globally, then repage Motion documents."""
    source: list[dict[str, object]] = []
    previous_offset = 0
    previous_start = -1
    expected_text: str | None = None
    source_count = 0
    for raw_part in title_parts:
        part = _mapping(
            raw_part,
            name="part",
            keys={
                "version",
                "width",
                "height",
                "rate",
                "frames",
                "offsetSeconds",
                "offsetFrame",
                "expectedText",
                "placements",
            },
        )
        offset = _integer(part["offsetFrame"], name="part offset")
        frames = _integer(part["frames"], name="part frames", minimum=1)
        dimensions = tuple(
            _integer(part[key], name="part dimension") for key in ("width", "height", "rate")
        )
        entries = part["placements"]
        reference = part["expectedText"]
        if (
            type(part["version"]) is not int
            or part["version"] != 1
            or dimensions != (width, height, rate)
            or offset < previous_offset
            or not math.isclose(
                _number(part["offsetSeconds"], name="part offset seconds"),
                offset / rate,
                rel_tol=0,
                abs_tol=1 / rate,
            )
            or not isinstance(entries, list)
            or not entries
            or not isinstance(reference, str)
            or not reference
            or (expected_text is not None and reference != expected_text)
        ):
            raise ReplayError("Motion title part clock/order differs from the bundle")
        expected_text = reference
        previous_offset = offset
        source_count += len(entries)
        local_start = -1
        for raw_cue in entries:
            cue = _mapping(
                raw_cue,
                name="placement",
                keys={"text", "startFrame", "endFrame", "x", "y"},
            )
            start = _integer(cue["startFrame"], name="caption start")
            end = _integer(cue["endFrame"], name="caption end", minimum=1)
            text = cue["text"]
            global_start, global_end = offset + start, offset + end
            if (
                not isinstance(text, str)
                or not text
                or start < local_start
                or not start < end <= frames
                or global_start < previous_start
            ):
                raise ReplayError("Motion caption text or interval is invalid")
            local_start = start
            previous_start = global_start
            source.append({"text": text, "startFrame": global_start, "endFrame": global_end})
    if not source or source_count > MAX_PLACEMENTS:
        raise ReplayError("Motion plan has no captions or exceeds its bounded cue count")

    coalesced: list[dict[str, object]] = []
    for cue in source:
        start = _integer(cue["startFrame"], name="caption start")
        end = _integer(cue["endFrame"], name="caption end", minimum=1)
        if coalesced and start < _integer(coalesced[-1]["endFrame"], name="caption end"):
            prior = coalesced[-1]
            prior["text"] = str(prior["text"]) + str(cue["text"])
            prior["endFrame"] = max(_integer(prior["endFrame"], name="caption end"), end)
        else:
            coalesced.append(dict(cue))

    planned: list[dict[str, object]] = []
    current: list[dict[str, object]] = []
    current_offset = 0

    def flush() -> None:
        nonlocal current
        if not current:
            return
        local = [
            {
                "text": cue["text"],
                "startFrame": _integer(cue["startFrame"], name="caption start") - current_offset,
                "endFrame": _integer(cue["endFrame"], name="caption end") - current_offset,
                "x": center_x,
                "y": _anchored_y(
                    str(cue["text"]),
                    height=height,
                    anchor_y_ratio=anchor_y_ratio,
                    font_size=font_size,
                    output_height=output_height,
                ),
            }
            for cue in current
        ]
        final = max(_integer(cue["endFrame"], name="caption end") for cue in current)
        planned.append(
            {
                "offsetFrame": current_offset,
                "frames": final - current_offset,
                "expectedText": expected_text,
                "placements": local,
            }
        )
        current = []

    for cue in coalesced:
        start = _integer(cue["startFrame"], name="caption start")
        if current and (
            len(current) >= MAX_CUES_PER_TITLE or start - current_offset >= MAX_TITLE_SECONDS * rate
        ):
            flush()
        if not current:
            current_offset = start
        current.append(cue)
    flush()
    return source_count, len(coalesced), planned


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.pending")
    if temporary.exists() or temporary.is_symlink():
        raise ReplayError("Unreconciled Motion batch manifest write exists")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def expand(
    template_path: Path,
    bundle_path: Path,
    output_dir: Path,
    *,
    anchor_y_ratio: Fraction,
) -> Path:
    """Prepare and hash all parts, safely reusing byte-identical prior outputs."""
    template_path = _regular(template_path, limit=MAX_XML_CHARACTERS)
    bundle_path = _regular(bundle_path, limit=MAX_BUNDLE_BYTES)
    if not isinstance(anchor_y_ratio, Fraction) or not 0 < anchor_y_ratio < 1:
        raise ReplayError("Motion batch requires an explicit interior vertical-anchor ratio")
    output_dir = output_dir.absolute()
    if output_dir.resolve() != output_dir or output_dir.is_symlink():
        raise ReplayError("Motion batch output must use a canonical, unlinked path")
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not output_dir.is_dir():
        raise ReplayError("Motion batch output must be a directory")

    raw = json.loads(bundle_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json)
    bundle = _mapping(
        raw,
        name="bundle",
        keys={"version", "canvas", "slot", "fontSize", "parts", "placements", "titleParts"},
    )
    if type(bundle["version"]) is not int or bundle["version"] != 1:
        raise ReplayError("Unsupported Motion title-plan bundle version")
    canvas = _mapping(bundle["canvas"], name="canvas", keys={"width", "height", "rate"})
    width = _integer(canvas["width"], name="canvas width", minimum=1)
    height = _integer(canvas["height"], name="canvas height", minimum=1)
    rate = _integer(canvas["rate"], name="frame rate", minimum=1)
    slot = _mapping(bundle["slot"], name="slot", keys={"centerX", "left", "right", "width"})
    center_x = _number(slot["centerX"], name="slot centre")
    left = _number(slot["left"], name="slot left")
    right = _number(slot["right"], name="slot right")
    slot_width = _number(slot["width"], name="slot width")
    if not 0 <= left < center_x < right <= width or not math.isclose(
        right - left, slot_width, rel_tol=0, abs_tol=1e-6
    ):
        raise ReplayError("Motion title slot is inconsistent with its canvas")
    font_size = _number(bundle["fontSize"], name="selected font size")
    if font_size <= 0:
        raise ReplayError("Motion title font size must be positive")
    part_count = _integer(bundle["parts"], name="part count", minimum=1)
    placement_count = _integer(bundle["placements"], name="placement count", minimum=1)
    title_parts = bundle["titleParts"]
    if (
        part_count > MAX_PARTS
        or placement_count > MAX_PLACEMENTS
        or not isinstance(title_parts, list)
        or len(title_parts) != part_count
    ):
        raise ReplayError("Motion title-part count differs from its bounded bundle")

    source_part_count = part_count
    source_placement_count = placement_count
    output_width, output_height = motion_stages._reference_canvas(width, height)
    total_placements, total_rendered_captions, planned = _planned_parts(
        title_parts,
        width=width,
        height=height,
        rate=rate,
        center_x=center_x,
        anchor_y_ratio=anchor_y_ratio,
        font_size=font_size,
        output_height=output_height,
    )
    if total_placements != source_placement_count:
        raise ReplayError("Motion title placement total differs from the source bundle")
    title_parts = [
        {
            "version": 1,
            "width": width,
            "height": height,
            "rate": rate,
            "frames": part["frames"],
            "offsetSeconds": _integer(part["offsetFrame"], name="part offset") / rate,
            "offsetFrame": part["offsetFrame"],
            "expectedText": part["expectedText"],
            "placements": part["placements"],
        }
        for part in planned
    ]
    part_count = len(title_parts)
    placement_count = total_rendered_captions

    template_text = template_path.read_text(encoding="utf-8")
    template_sha = _sha256(template_path)
    bundle_sha = _sha256(bundle_path)
    style_top, style_bottom = _style_controls(template_text)
    scale = font_size / caption_style_defaults.REFERENCE_FONT_SIZE
    style = motion_finish.selected_finish_style(
        top=style_top,
        bottom=style_bottom,
        font_size=font_size,
        stroke_scale=scale,
        shadow_scale=scale,
    )
    implementation = {
        path.name: _sha256(path)
        for path in (
            Path(motion_stages.__file__),
            Path(motion_finish.__file__),
            Path(__file__),
            Path(caption_style_defaults.__file__),
        )
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and manifest_path.is_symlink():
        raise ReplayError("Motion batch manifest cannot be a symlink")
    if manifest_path.exists():
        prior = record(json.loads(manifest_path.read_text(encoding="utf-8")))
        identity = prior.get("identity")
        expected_identity = {
            "templateSHA256": template_sha,
            "bundleSHA256": bundle_sha,
            "implementation": implementation,
            "anchorYRatio": f"{anchor_y_ratio.numerator}/{anchor_y_ratio.denominator}",
        }
        if identity != expected_identity:
            raise ReplayError("Motion batch resume inputs or implementation changed")
        completed = record(prior.get("completed", {}))
    else:
        completed = {}
        _atomic_json(
            manifest_path,
            {
                "version": 1,
                "candidateOnly": True,
                "identity": {
                    "templateSHA256": template_sha,
                    "bundleSHA256": bundle_sha,
                    "implementation": implementation,
                    "anchorYRatio": f"{anchor_y_ratio.numerator}/{anchor_y_ratio.denominator}",
                },
                "layout": {
                    "coordinateSpace": "canvas-top-left",
                    "horizontalCenterCanvas": center_x,
                    "outputWidth": output_width,
                    "outputHeight": output_height,
                    "horizontalCenterOutput": center_x * output_width / width,
                    "verticalAnchorCanvas": height * float(anchor_y_ratio),
                    "verticalAnchorOutput": output_height * float(anchor_y_ratio),
                    "fontSize": font_size,
                    "partCount": part_count,
                    "placementCount": placement_count,
                    "sourcePartCount": source_part_count,
                    "sourcePlacementCount": source_placement_count,
                },
                "completed": {},
                "motionRenderVerified": False,
                "nativeImportVerified": False,
                "compositeVerified": False,
                "deliveryComplete": False,
            },
        )

    total_placements = 0
    total_rendered_captions = 0
    previous_offset_end = 0
    for index, raw_part in enumerate(title_parts):
        part = _mapping(
            raw_part,
            name="part",
            keys={
                "version",
                "width",
                "height",
                "rate",
                "frames",
                "offsetSeconds",
                "offsetFrame",
                "expectedText",
                "placements",
            },
        )
        offset = _integer(part["offsetFrame"], name="part offset")
        frames = _integer(part["frames"], name="part frames", minimum=1)
        part_width = _integer(part["width"], name="part width", minimum=1)
        part_height = _integer(part["height"], name="part height", minimum=1)
        part_rate = _integer(part["rate"], name="part rate", minimum=1)
        entries = part["placements"]
        if (
            type(part["version"]) is not int
            or part["version"] != 1
            or (part_width, part_height, part_rate) != (width, height, rate)
            or offset < previous_offset_end
            or not isinstance(entries, list)
            or not entries
        ):
            raise ReplayError("Motion title part clock/order differs from the bundle")
        previous_offset_end = offset + frames
        total_placements += len(entries)
        normalized_entries: list[dict[str, object]] = []
        for entry in entries:
            cue = _mapping(
                entry,
                name="placement",
                keys={"text", "startFrame", "endFrame", "x", "y"},
            )
            start = _integer(cue["startFrame"], name="caption start")
            end = _integer(cue["endFrame"], name="caption end", minimum=1)
            text = cue["text"]
            if not isinstance(text, str) or not text or not start < end <= frames:
                raise ReplayError("Motion caption text or interval is invalid")
            if normalized_entries and start < _integer(
                normalized_entries[-1]["endFrame"], name="caption end"
            ):
                # Simultaneous/overlapping source words are one visible caption
                # unit. Keep the earliest recognized onset and union end; never
                # create artificial clocks or stack glyphs at the same position.
                previous = normalized_entries[-1]
                previous["text"] = str(previous["text"]) + text
                previous["endFrame"] = max(_integer(previous["endFrame"], name="caption end"), end)
                previous["y"] = _anchored_y(
                    str(previous["text"]),
                    height=height,
                    anchor_y_ratio=anchor_y_ratio,
                    font_size=font_size,
                    output_height=output_height,
                )
            else:
                normalized_entries.append(
                    {
                        "text": text,
                        "startFrame": start,
                        "endFrame": end,
                        "x": center_x,
                        "y": _anchored_y(
                            text,
                            height=height,
                            anchor_y_ratio=anchor_y_ratio,
                            font_size=font_size,
                            output_height=output_height,
                        ),
                    }
                )
        total_rendered_captions += len(normalized_entries)
        plan = {
            "frames": frames,
            "width": width,
            "height": height,
            "rate": rate,
            "expectedText": part["expectedText"],
            "placements": normalized_entries,
            "coordinateSpace": "canvas-top-left",
        }
        source = motion_stages.prepare(template_text, plan)
        finished = motion_finish.finish_sequence(source, style=style)
        payload = finished.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        filename = f"part-{index:04d}.moti"
        output = output_dir / filename
        if output.is_symlink():
            raise ReplayError("Motion batch output cannot be a symlink")
        prior_record = completed.get(filename)
        if output.exists():
            if not output.is_file() or output.stat().st_nlink != 1 or _sha256(output) != digest:
                raise ReplayError("Existing Motion part does not match its current plan")
            if prior_record is not None and record(prior_record).get("sha256") != digest:
                raise ReplayError("Motion batch receipt differs from its current output")
        else:
            with output.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        completed[filename] = {
            "sha256": digest,
            "bytes": len(payload),
            "offsetFrame": offset,
            "frames": frames,
            "placements": len(normalized_entries),
        }
        _atomic_json(
            manifest_path,
            {
                "version": 1,
                "candidateOnly": True,
                "identity": {
                    "templateSHA256": template_sha,
                    "bundleSHA256": bundle_sha,
                    "implementation": implementation,
                    "anchorYRatio": f"{anchor_y_ratio.numerator}/{anchor_y_ratio.denominator}",
                },
                "layout": {
                    "coordinateSpace": "canvas-top-left",
                    "horizontalCenterCanvas": center_x,
                    "outputWidth": output_width,
                    "outputHeight": output_height,
                    "horizontalCenterOutput": center_x * output_width / width,
                    "verticalAnchorCanvas": height * float(anchor_y_ratio),
                    "verticalAnchorOutput": output_height * float(anchor_y_ratio),
                    "fontSize": font_size,
                    "partCount": part_count,
                    "placementCount": total_placements,
                    "sourcePartCount": source_part_count,
                    "sourcePlacementCount": source_placement_count,
                    "renderedCaptionCount": total_rendered_captions,
                },
                "completed": completed,
                "motionRenderVerified": False,
                "nativeImportVerified": False,
                "compositeVerified": False,
                "deliveryComplete": False,
            },
        )
        print(f"PART {index + 1}/{part_count}: {filename} {digest}", flush=True)
    if total_placements != placement_count:
        raise ReplayError("Motion title placement total differs from the source bundle")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--anchor-y-ratio", required=True)
    args = parser.parse_args()
    try:
        ratio = Fraction(args.anchor_y_ratio)
        print(expand(args.template, args.bundle, args.output, anchor_y_ratio=ratio), flush=True)
    except (ReplayError, OSError, ValueError, ZeroDivisionError, KeyError) as error:
        print(f"MOTION_BATCH_ERROR {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
