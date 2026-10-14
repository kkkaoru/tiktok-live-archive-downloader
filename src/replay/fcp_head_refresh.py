"""Refresh bounded existing caption placement from pinned current observations."""

import argparse
import copy
import json
from collections.abc import Callable
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _pin
from replay.caption_style_defaults import HEAD_CAPTION_GAP_PX
from replay.core import ReplayError
from replay.fcp_caption_retarget import face_references
from replay.fcp_caption_srt import _time
from replay.fcp_caption_stability import stabilize_caption_y
from replay.fcp_head_near import FaceScaleSeries, read_face_scale, title_samples
from replay.fcp_head_quality import geometry_at, title_geometry_provenance
from replay.fcp_measured import CaptionAnchor
from replay.fcp_output_window import window
from replay.fcp_pre_layout import geometry_factor
from replay.fcp_project import CaptionTemplate, caption_template
from replay.fcp_screen_stack import ScreenStack, screen_stacks
from replay.fcp_tracking_caption import TrackingCalibration
from replay.fcp_visible_arc import VisibleArc, read_visible_arc
from replay.foreground_evidence import _bound_bytes
from replay.head_approx_placement import (
    MOTION_POLICIES,
    ApproxPlan,
    DerivedMotion,
    motion_scope,
    read_motion_binding,
    scoped_plan,
    unknown_stacks,
)
from replay.head_approx_placement import title_samples as approximate_title_samples
from replay.head_height_paging import make_layout
from replay.head_image_collection import read_image_binding
from replay.head_image_plan import MAX_WINDOWS, caption_spans
from replay.head_image_retained import read_image_binding as read_retained_image_binding
from replay.head_image_support import image_title_samples
from replay.head_reference import resolve_reference
from replay.recordings import record
from replay.stages import file_digest
from replay.tracking_observations import TrackingJob, load_job
from replay.tracking_retained import verify_catalog

MAX_REFRESH_TITLES = 20_000
MAX_MEASUREMENT_RECEIPT_BYTES = 64 * 1024 * 1024
MAX_MEASUREMENT_INPUTS = 262_144
MAX_FACE_WINDOWS = 512


def measurement_sources(config: dict[str, object]) -> tuple[Path, ...]:
    values = config.get("parts")
    if values is None:
        return (_pin(config.get("part")),)
    if "part" in config or not isinstance(values, list) or not 1 <= len(values) <= 512:
        raise ReplayError("Expected one bounded explicit measurement source list")
    paths = tuple(_pin(value) for value in values)
    if len(set(paths)) != len(paths):
        raise ReplayError("Duplicate measurement source")
    return paths


def measurement_pins(paths: tuple[Path, ...]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        receipt = record(
            json.loads(_bound_bytes(path, file_digest(path), MAX_MEASUREMENT_RECEIPT_BYTES))
        )
        pins = record(receipt.get("inputs"))
        if not pins or any(not isinstance(value, str) for value in pins.values()):
            raise ReplayError("Expected explicit native glyph measurement pins")
        if len(pins) > MAX_MEASUREMENT_INPUTS:
            raise ReplayError("Measurement receipt exceeds input count budget")
        for key, value in pins.items():
            if not isinstance(value, str) or (key in result and result[key] != value):
                raise ReplayError("Conflicting native measurement identity")
            result[key] = value
        if len(result) > MAX_MEASUREMENT_INPUTS:
            raise ReplayError("Measurement receipt union exceeds input count budget")
    return result


def placement_changes(before: ET.Element, after: ET.Element) -> list[dict[str, object]]:
    previous = before.findall(".//asset-clip/title")
    current = after.findall(".//asset-clip/title")
    if len(previous) != len(current):
        raise ReplayError("Caption count changed during refresh")
    rows: list[dict[str, object]] = []
    index = 0
    for clip in after.findall(".//asset-clip"):
        for title in clip.findall("title"):
            start = (
                _time(clip.attrib, "offset")
                + _time(title.attrib, "offset")
                - _time(clip.attrib, "start")
            )
            old = previous[index].find("adjust-transform")
            new = title.find("adjust-transform")
            if old is None or new is None:
                raise ReplayError("Missing placement transform")
            rows.append(
                {
                    "title": index,
                    "start": str(start),
                    "end": str(start + _time(title.attrib, "duration")),
                    "changed": ET.tostring(old) != ET.tostring(new),
                }
            )
            index += 1
    return rows


def partition_impact(
    manifest: Path, xml: Path, changes: list[dict[str, object]], *, start: Fraction
) -> list[dict[str, object]]:
    data = _json(manifest)
    if _pin(data.get("source")) != xml:
        raise ReplayError("Partition belongs to another source XML")
    parts = data.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= 512:
        raise ReplayError("Expected bounded partition records")
    result: list[dict[str, object]] = []
    for item in parts:
        part = record(item)
        left = _time({"start": str(part.get("globalStart"))}, "start")
        right = left + _time({"duration": str(part.get("duration"))}, "duration")
        affected = [
            row
            for row in changes
            if Fraction(str(row["start"])) + start < right
            and Fraction(str(row["end"])) + start > left
        ]
        result.append(
            {
                "path": str(_pin(part)),
                "titlesInWindow": len(affected),
                "changedTitles": sum(row["changed"] is True for row in affected),
                "renderReuseCandidate": not any(row["changed"] is True for row in affected),
            }
        )
    return result


def _without_positions(root: ET.Element) -> bytes:
    result = copy.deepcopy(root)
    for transform in result.findall(".//asset-clip/title/adjust-transform"):
        transform.attrib.pop("position", None)
        for position in transform.findall("param[@name='position']"):
            transform.remove(position)
    return ET.tostring(result)


def _source_identity(lineage: dict[str, object], job: TrackingJob) -> dict[str, str]:
    semantic = _pin(lineage.get("semanticXML"))
    if semantic.stat().st_size > 64 * 1024 * 1024:
        raise ReplayError("Source semantic XML exceeds byte budget")
    root = ET.fromstring(semantic.read_bytes())
    sources = root.findall(f"./resources/asset/media-rep[@src='{job.source.as_uri()}']")
    sequence = root.find("./library/event/project/sequence")
    formats = (
        []
        if sequence is None
        else root.findall(f"./resources/format[@id='{sequence.get('format')}']")
    )
    if (
        len(sources) != 1
        or len(formats) != 1
        or (formats[0].get("width"), formats[0].get("height")) != (str(job.width), str(job.height))
    ):
        raise ReplayError(
            "Face source identity or coordinate geometry differs from semantic source"
        )
    return {str(semantic): file_digest(semantic), str(job.source): job.source_hash}


def approximate_inputs(
    value: object, *, job: TrackingJob
) -> tuple[VisibleArc, tuple[Fraction, ...], dict[str, object] | None]:
    """Explicit approximation inputs never replace the dense current-pixel reader."""
    data = record(value)
    fields = set(data)
    if (
        data.get("policy") != "current-face-y-offset-candidate"
        or fields - {"policy", "image", "retainedImage", "zoomResetTimes"}
        or ("image" in data) == ("retainedImage" in data)
    ):
        raise ReplayError("Expected explicit approximate head policy and one pixel-seed binding")
    times = data.get("zoomResetTimes", [])
    if not isinstance(times, list) or len(times) > MAX_REFRESH_TITLES:
        raise ReplayError("Expected bounded explicit zoom-reset clocks")
    if any(not isinstance(time, str) for time in times):
        raise ReplayError("Expected explicit rational zoom-reset clocks")
    resets = tuple(_time({"time": str(time) + "s"}, "time") for time in times)
    if any(a >= b for a, b in pairwise(resets)):
        raise ReplayError("Zoom-reset clocks must be strictly increasing")
    if "retainedImage" in data:
        retained = read_retained_image_binding(
            data["retainedImage"], source_hash=job.source_hash, width=job.width, height=job.height
        )
        return retained.arc, resets, retained.receipt()
    return (
        read_image_binding(
            data["image"], source_hash=job.source_hash, width=job.width, height=job.height
        ),
        resets,
        None,
    )


def approximate_plans(
    root: ET.Element,
    *,
    faces: FaceScaleSeries,
    pixels: VisibleArc | None,
    resets: tuple[Fraction, ...],
    measured_height: Callable[[ET.Element], float],
    derived_motion: DerivedMotion | None = None,
) -> tuple[
    dict[ET.Element, ApproxPlan], dict[ET.Element, tuple[ScreenStack, ...]], list[dict[str, object]]
]:
    """Independent source-cut scopes retain raw clocks and bottom-stacked blocks."""
    plans: dict[ET.Element, ApproxPlan] = {}
    stacks: dict[ET.Element, tuple[ScreenStack, ...]] = {}
    reports: list[dict[str, object]] = []
    for clip in root.findall(".//asset-clip"):
        titles = clip.findall("title")
        if not titles:
            continue
        start = _time(clip.attrib, "start")
        end = start + _time(clip.attrib, "duration")
        if derived_motion is not None:
            plan = motion_scope(derived_motion.plan, start=start, end=end, height=faces.tops.height)
        elif pixels is not None:
            plan = scoped_plan(faces, pixels=pixels, start=start, end=end, resets=resets)
        else:
            raise ReplayError("Approximate placement requires its explicit source evidence")
        blocks = tuple(
            (
                _time(title.attrib, "offset"),
                _time(title.attrib, "offset") + _time(title.attrib, "duration"),
                measured_height(title),
            )
            for title in titles
        )
        reserved = unknown_stacks(plan, blocks)
        plans.update({title: plan for title in titles})
        stacks.update({title: reserved for title in titles})
        reports.append({"sourceStart": str(start), "sourceEnd": str(end), "plan": plan.encoded()})
    return plans, stacks, reports


def refresh(
    root: ET.Element,
    *,
    faces: FaceScaleSeries,
    calibration: TrackingCalibration | None,
    template: CaptionTemplate,
    measure: Callable[[str], tuple[float, float]],
    visible_arc: VisibleArc | None = None,
    current_upper_pixels: VisibleArc | None = None,
    approximate_pixels: VisibleArc | None = None,
    derived_motion: DerivedMotion | None = None,
    zoom_resets: tuple[Fraction, ...] = (),
) -> list[dict[str, object]]:
    """Replace only placement keys; no old Y is read or used as a seed."""
    unchanged = _without_positions(root)
    if (
        sum(
            item is not None
            for item in (visible_arc, current_upper_pixels, approximate_pixels, derived_motion)
        )
        > 1
    ):
        raise ReplayError(
            "Current upper pixels, approximate seeds and missing-face arcs are distinct inputs"
        )
    if approximate_pixels is None and derived_motion is None and calibration is None:
        raise ReplayError("Observed placement requires explicit coordinate calibration")
    titles = root.findall(".//asset-clip/title")
    if not titles or len(titles) > MAX_REFRESH_TITLES:
        raise ReplayError("Expected a bounded caption comparison window")
    arc_reports: list[dict[str, object]] = []
    plans: dict[ET.Element, ApproxPlan] = {}
    approximation_stacks: dict[ET.Element, tuple[ScreenStack, ...]] = {}
    if approximate_pixels is not None or derived_motion is not None:
        plans, approximation_stacks, reports = approximate_plans(
            root,
            faces=faces,
            pixels=approximate_pixels,
            resets=zoom_resets,
            derived_motion=derived_motion,
            measured_height=lambda title: (
                measure(title.findtext("text", ""))[1]
                * geometry_factor(title, template.pre_layout_geometry)
            ),
        )
        arc_reports.extend({"approximation": report} for report in reports)
    stacks = screen_stacks(
        root,
        gaps=faces.tops.derived_ranges,
        measured_height=lambda title: (
            measure(title.findtext("text", ""))[1]
            * geometry_factor(title, template.pre_layout_geometry)
        ),
        canvas_height=faces.tops.height,
    )
    for title in titles:
        if visible_arc is not None:
            start = _time(title.attrib, "offset")
            if not visible_arc.covers(start, start + _time(title.attrib, "duration")):
                raise ReplayError("Visible arc comparison exceeds measured frame cells")
        texts = ["".join(node.itertext()) for node in title.findall("text")]
        if len(texts) != 3 or len(set(texts)) != 1 or not texts[0]:
            raise ReplayError("Expected identical nonempty three-layer caption text")
        width, height = measure(texts[0])
        factor = geometry_factor(title, template.pre_layout_geometry)
        if not 0 < width * factor <= faces.tops.width:
            raise ReplayError("Measured caption width exceeds current canvas")
        if approximate_pixels is not None or derived_motion is not None:
            points = approximate_title_samples(
                title,
                plan=plans[title],
                caption_height=height * factor,
                canvas_height=faces.tops.height,
                stacks=approximation_stacks[title],
            )
        elif current_upper_pixels is not None and calibration is not None:
            points = image_title_samples(
                title,
                arc=current_upper_pixels,
                caption_height=height * factor,
                canvas_height=calibration.canvas_height,
            )
        elif calibration is not None:
            points = title_samples(
                title,
                faces=faces,
                calibration=calibration,
                caption_height=height * factor,
                screen_stacks=stacks[title],
            )
        else:
            raise ReplayError("Expected one explicit placement backend")
        transform = title.find("adjust-transform")
        if transform is None:
            raise ReplayError("Expected existing caption transform")
        transform.attrib.pop("position", None)
        for old in transform.findall("param[@name='position']"):
            transform.remove(old)
        animation = ET.SubElement(
            ET.SubElement(transform, "param", name="position"), "keyframeAnimation"
        )
        for time, vector in points:
            source_time = (
                _time(title.attrib, "offset")
                + Fraction(time.removesuffix("s"))
                - _time(title.attrib, "start")
            )
            boundary = source_time == _time(title.attrib, "offset") + _time(
                title.attrib, "duration"
            )
            arc = current_upper_pixels if current_upper_pixels is not None else visible_arc
            cell = None if arc is None else arc.at(source_time, ending=boundary)
            gap = HEAD_CAPTION_GAP_PX if calibration is None else calibration.gap
            applied = (
                cell is not None
                and cell.visible_top is not None
                and (
                    current_upper_pixels is not None
                    or not geometry_at(faces.tops, source_time).observed
                )
                and height * factor <= cell.visible_top - gap <= faces.tops.height
            )
            if applied and cell is not None and cell.visible_top is not None:
                bottom = cell.visible_top - gap
                vector = f"0 {50 - bottom / faces.tops.height * 100:.12g}"
                arc_reports.append(
                    {
                        "title": title.get("name"),
                        "sourceTime": str(source_time),
                        "sourceFrameStart": str(cell.start),
                        "sourceFrameEnd": str(cell.end),
                        "supportedPoints": cell.supported,
                        "unknownPoints": cell.unknown,
                        "visibleArcTop": cell.visible_top,
                        "captionBottom": bottom,
                        "provenance": "current-measured-upper-pixels-candidate"
                        if current_upper_pixels is not None
                        else "tracked-visible-arc-candidate",
                        "detectedFace": current_upper_pixels is not None
                        and geometry_at(faces.tops, source_time).observed,
                        "completeHeadVerified": False,
                        "headNearVerified": False,
                        "terminalBoundaryKey": boundary,
                    }
                )
            ET.SubElement(
                animation,
                "keyframe",
                time=time,
                value=vector,
                curve="linear",
            )
    stabilize_caption_y(root)
    if _without_positions(root) != unchanged:
        raise ReplayError("Placement refresh changed caption/media/clock content")
    return arc_reports


def bounded_image_references(
    index: Path, root: ET.Element, *, maximum: int | None = None
) -> list[tuple[Path, Path]]:
    """Reverify current comparison windows without merging a distant old reference."""
    spans = caption_spans(root)
    selected: list[tuple[Path, Path]] = []
    for config, evidence in face_references(index):
        job = load_job(config)
        left = job.clock.pts[0] * job.clock.time_base
        right = (job.clock.pts[-1] + job.clock.last_duration) * job.clock.time_base
        if any(left < end and start < right for start, end in spans):
            selected.append((config, evidence))
    limit = MAX_WINDOWS if maximum is None else maximum
    if not 1 <= limit <= MAX_FACE_WINDOWS or not 1 <= len(selected) <= limit:
        raise ReplayError("Current-image refresh exceeds its verified-window budget")
    return selected


def image_calibration(faces: FaceScaleSeries, arc: VisibleArc) -> TrackingCalibration:
    """Use current actual pixels and paired raw face solely for coordinate conversion."""
    cell = next((cell for cell in arc.cells if cell.visible_top is not None), None)
    if cell is None or cell.visible_top is None:
        raise ReplayError("Current-image conversion requires actual supported upper pixels")
    face = faces.observed_reference(cell.start)
    size = next((size for size in faces.sizes if size.time == cell.start), None)
    position = next((point for point in faces.tops.positions if point.time == cell.start), None)
    if size is None or position is None or not 0 < cell.visible_top <= face.top:
        raise ReplayError("Current upper pixels must precede their same-frame observed face")
    return TrackingCalibration(
        cell.start,
        cell.visible_top,
        1,
        0,
        HEAD_CAPTION_GAP_PX,
        faces.tops.height,
        (position.x, face.top, size.width, face.bottom - face.top),
    )


def run(config_path: Path, output: Path) -> Path:
    config = _json(config_path)
    if config.get("version") != 1 or output.exists() or output.absolute() != output.resolve():
        raise ReplayError("Expected version one and a new canonical output directory")
    names = (
        "xml",
        "lineage",
        "faceIndex",
        "template",
        "nativeExecutable",
        "measurementReference",
    )
    paths = {name: _pin(config.get(name)) for name in names}
    if config.get("longExecutable") is not None:
        paths["longExecutable"] = _pin(config["longExecutable"])
    parts = measurement_sources(config)
    typed_pins = measurement_pins(parts)
    partition = None if config.get("partition") is None else _pin(config["partition"])
    code = tuple(
        Path(__file__).with_name(name)
        for name in (
            "fcp_head_refresh.py",
            "fcp_head_near.py",
            "fcp_screen_stack.py",
            "head_near_placement.py",
            "caption_style_defaults.py",
            "fcp_head_quality.py",
            "fcp_caption_stability.py",
            "caption_y_stability.py",
            "head_reference.py",
            "head_height_paging.py",
            "fcp_visible_arc.py",
            "head_image_collection.py",
            "head_image_plan.py",
            "head_image_support.py",
            "head_approx_placement.py",
            "head_partial_pixel_track.py",
            "head_upper_texture.py",
            "head_contour_track.py",
            "head_texture_track.py",
            "head_seed.py",
            "head_image_retained.py",
            "tracking_retained.py",
            "producer_code_evidence.py",
            "input_hash_cache.py",
            "fcp_caption_srt.py",
            "fcp_project.py",
            "fcp_pre_layout.py",
            "fcp_output_window.py",
            "recordings.py",
            "artifact_json.py",
            "foreground_evidence.py",
            "tracking_observations.py",
            "head_observations.py",
            "subtitle_clock.py",
            "stages.py",
            "core.py",
        )
    )
    bound = {str(path): file_digest(path) for path in (config_path, *paths.values(), *parts, *code)}
    if partition is not None:
        bound[str(partition)] = file_digest(partition)
    lineage = _json(paths["lineage"])
    if (
        _pin(lineage.get("outputXML")) != paths["xml"]
        or lineage.get("captionClockEqual") is not True
    ):
        raise ReplayError("Expected matching verified clean-source lineage")
    if paths["xml"].stat().st_size > 64 * 1024 * 1024:
        raise ReplayError("Comparison XML exceeds byte budget")
    source = paths["xml"].read_bytes()
    root = window(
        ET.fromstring(source), Fraction(str(config.get("start"))), Fraction(str(config.get("end")))
    )
    before = copy.deepcopy(root)
    references = face_references(paths["faceIndex"])
    approximate = config.get("approxHead")
    if approximate is not None and (
        config.get("currentUpperPixels") is not None or config.get("visibleArc") is not None
    ):
        raise ReplayError("Approximate head policy cannot replace another explicit backend")
    if config.get("currentUpperPixels") is not None:
        references = bounded_image_references(paths["faceIndex"], root)
    elif approximate is not None:
        references = bounded_image_references(paths["faceIndex"], root, maximum=MAX_FACE_WINDOWS)
    retained_tracking = None
    if config.get("retainedTracking") is not None:
        descriptor = _pin(config["retainedTracking"])
        retained_tracking = verify_catalog(
            descriptor, sha256=file_digest(descriptor), references=references
        )
        bound.update(retained_tracking.inputs)
    job, faces = read_face_scale(
        references, missing_policy="hold", seed_overlap="next", retained=retained_tracking
    )
    bound.update(_source_identity(lineage, job))
    visible_arc = None
    current_upper_pixels = None
    approximate_pixels = None
    approximate_evidence = None
    derived_motion = None
    zoom_resets: tuple[Fraction, ...] = ()
    if approximate is not None:
        approximate_policy = record(approximate).get("policy")
        if isinstance(approximate_policy, str) and approximate_policy in MOTION_POLICIES:
            derived_motion = read_motion_binding(approximate, job=job)
            bound.update(derived_motion.inputs)
            approximate_evidence = derived_motion.receipt()
        else:
            approximate_pixels, zoom_resets, approximate_evidence = approximate_inputs(
                approximate, job=job
            )
            bound.update(approximate_pixels.inputs)
    if config.get("currentUpperPixels") is not None:
        current_upper_pixels = read_image_binding(
            config["currentUpperPixels"],
            source_hash=job.source_hash,
            width=job.width,
            height=job.height,
        )
        bound.update(current_upper_pixels.inputs)
    calibration = None
    if current_upper_pixels is not None:
        calibration = image_calibration(faces, current_upper_pixels)
    elif approximate_pixels is None and derived_motion is None:
        calibration = resolve_reference(config, faces)
        faces = faces.with_current_scale(reference_time=calibration.reference_time)
    if config.get("visibleArc") is not None:
        visible_arc = read_visible_arc(
            record(config["visibleArc"]),
            source_hash=job.source_hash,
            times=tuple(point.time for point in faces.tops.positions),
            width=job.width,
            height=job.height,
        )
        bound.update(visible_arc.inputs)
    output.mkdir(parents=True, mode=0o700)
    layout = make_layout(
        reference=paths["measurementReference"],
        executable=paths["nativeExecutable"],
        output=output / "metrics",
        pinned_inputs=typed_pins,
        cached_directory=parts[0].parent,
        long_executable=paths.get("longExecutable"),
    )
    template = caption_template(paths["template"], anchor=CaptionAnchor.BOTTOM)
    arc_reports = refresh(
        root,
        faces=faces,
        calibration=calibration,
        template=template,
        measure=layout.measure,
        visible_arc=visible_arc,
        current_upper_pixels=current_upper_pixels,
        approximate_pixels=approximate_pixels,
        derived_motion=derived_motion,
        zoom_resets=zoom_resets,
    )
    changes = placement_changes(before, root)
    impact = (
        []
        if partition is None
        else partition_impact(
            partition, paths["xml"], changes, start=Fraction(str(config.get("start")))
        )
    )
    project = root.find("./library/event/project")
    name = config.get("name")
    if project is None or not isinstance(name, str) or not 1 <= len(name) <= 200:
        raise ReplayError("Expected explicit bounded comparison project name")
    project.set("name", name)
    bound.update(layout.verify_inputs())
    if any(file_digest(Path(path)) != digest for path, digest in bound.items()):
        raise ReplayError("Comparison input changed during placement refresh")
    destination = output / "candidate.fcpxml"
    destination.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))
    placement_policy = "current-observed-biaxial-scale-no-history"
    if derived_motion is not None:
        placement_policy = derived_motion.plan.policy
    elif approximate_pixels is not None:
        placement_policy = "current-face-y-offset-candidate"
    elif current_upper_pixels is not None:
        placement_policy = "current-source-cell-upper-pixels"
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "version": 1,
                "candidateOnly": True,
                "headNearVerified": False,
                "inputs": bound,
                "output": {"path": str(destination), "sha256": file_digest(destination)},
                "placementPolicy": placement_policy,
                "captionMediaClockUnchanged": True,
                "placementChanges": changes,
                "changedTitles": sum(row["changed"] is True for row in changes),
                "partitionImpact": impact,
                "visibleArcPlacements": [row for row in arc_reports if "approximation" not in row],
                "approximateHeadPlans": [
                    row["approximation"] for row in arc_reports if "approximation" in row
                ],
                "retainedUpperPixelEvidence": approximate_evidence
                if derived_motion is None
                else None,
                "derivedFaceMotionEvidence": None
                if derived_motion is None
                else derived_motion.receipt(),
                "retainedTrackingEvidence": None
                if retained_tracking is None
                else retained_tracking.receipt(),
                "historicalMeasurementSources": [str(path) for path in parts],
                "geometry": title_geometry_provenance(root, faces.tops),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        receipt_path = args.output / "receipt.json"
        receipt = record(
            json.loads(
                _bound_bytes(receipt_path, file_digest(receipt_path), MAX_MEASUREMENT_RECEIPT_BYTES)
            )
        )
        inputs = record(receipt.get("inputs"))
        if (
            not 1 <= len(inputs) <= MAX_MEASUREMENT_INPUTS
            or receipt.get("captionMediaClockUnchanged") is not True
            or inputs.get(str(args.config)) != file_digest(args.config)
            or any(file_digest(Path(name)) != digest for name, digest in inputs.items())
        ):
            raise ReplayError("Refresh receipt input/domain mismatch")
        print(_pin(receipt.get("output")))
    else:
        print(run(args.config, args.output))


if __name__ == "__main__":
    main()
