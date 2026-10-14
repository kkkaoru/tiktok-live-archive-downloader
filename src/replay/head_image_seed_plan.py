"""Plan only required current-frame pixel seeds for explicit approximate captions."""

import argparse
import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
from itertools import batched, chain, groupby, pairwise
from pathlib import Path
from xml.etree import ElementTree as ET

from replay import tracking_observations as storage
from replay.artifact_json import _json, _pin
from replay.core import ReplayError
from replay.fcp_caption_retarget import face_references
from replay.fcp_caption_srt import _time
from replay.fcp_head_near import FaceScaleSeries, read_face_scale
from replay.fcp_visible_arc import VisibleArc
from replay.foreground_evidence import _bound_bytes
from replay.foreground_observations import MAX_SAMPLES
from replay.head_approx_placement import (
    MAX_CELLS,
    CurrentFaceCell,
    MeasurementRequest,
    current_cells,
    scoped_plan,
)
from replay.head_image_plan import MAX_CELLS as PAGE_CELLS
from replay.head_image_plan import ImageFrameCell, _fraction, _roi
from replay.head_image_retained import read_image_binding
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.stages import file_digest
from replay.tracking_retained import MAX_WINDOWS, verify_catalog
from replay.vertical_captions import Box

MAX_TITLES = 20_000
MAX_XML_BYTES = 64 * 1024 * 1024
MAX_PAGES = MAX_CELLS // PAGE_CELLS
MAX_SCOPED_CELLS = MAX_CELLS
NATIVE_CELLS_PER_SECOND = 6
NATIVE_FRAME_CELL_LIMIT = Fraction(1)
PROTOCOL = "current-upper-seed-requests-v1"
CODE_FILES = (
    "head_image_seed_plan.py",
    "head_approx_placement.py",
    "head_image_plan.py",
    "head_image_retained.py",
    "head_image_collection.py",
    "head_image_support.py",
    "fcp_caption_retarget.py",
    "fcp_caption_srt.py",
    "fcp_head_near.py",
    "fcp_tracking_caption.py",
    "fcp_head_quality.py",
    "fcp_visible_arc.py",
    "fcp_screen_stack.py",
    "head_near_placement.py",
    "head_observations.py",
    "frame_cell_extent.py",
    "caption_style_defaults.py",
    "fcp_project.py",
    "tracking_observations.py",
    "tracking_retained.py",
    "producer_code_evidence.py",
    "input_hash_cache.py",
    "subtitle_clock.py",
    "vertical_captions.py",
    "caption_io.py",
    "recordings.py",
    "artifact_json.py",
    "foreground_evidence.py",
    "foreground_observations.py",
    "stages.py",
    "core.py",
)
WORK_ROOT = Path(__file__).resolve().parents[2] / "edits" / "work"


@dataclass(frozen=True, slots=True)
class SourceScope:
    start: Fraction
    end: Fraction
    titles: tuple[tuple[Fraction, Fraction], ...]

    def __post_init__(self) -> None:
        if (
            not 0 <= self.start < self.end <= 86400
            or not self.titles
            or any(not self.start <= left < right <= self.end for left, right in self.titles)
            or any(a[1] >= b[0] for a, b in pairwise(self.titles))
        ):
            raise ReplayError("Seed scope requires ordered separate bounded title ranges")


@dataclass(frozen=True, slots=True)
class UnsupportedClockCell:
    source: CurrentFaceCell
    scope: SourceScope
    title_overlaps: tuple[tuple[Fraction, Fraction], ...]

    def encoded(self) -> dict[str, object]:
        return {
            "sourceFrameStart": str(self.source.start),
            "sourceFrameEnd": str(self.source.end),
            "scopeStart": str(self.scope.start),
            "scopeEnd": str(self.scope.end),
            "currentFaceObserved": self.source.face is not None,
            "titleOverlaps": [
                {"start": str(left), "end": str(right)} for left, right in self.title_overlaps
            ],
            "kind": "source-clock-support",
            "reason": "source-cell-exceeds-native-duration",
            "nativeCellBudgetSeconds": str(NATIVE_FRAME_CELL_LIMIT),
            "displayPersistenceVerified": False,
            "headNearVerified": False,
        }


@dataclass(frozen=True, slots=True)
class SeedPlan:
    cells: tuple[ImageFrameCell, ...]
    missing: tuple[MeasurementRequest, ...]
    unused_requests: int
    reused_seeds: int
    unsupported: tuple[UnsupportedClockCell, ...] = ()

    def cost(self) -> dict[str, object]:
        count = len(self.cells)
        return {
            "newRequestedCells": count,
            "missingFaceRequests": len(self.missing),
            "captionFreeRequests": self.unused_requests,
            "reusedPixelSeeds": self.reused_seeds,
            "unsupportedScopedClockCells": len(self.unsupported),
            "unsupportedCaptionClockCells": sum(
                bool(cell.title_overlaps) for cell in self.unsupported
            ),
            "chunksAt128Cells": (count + PAGE_CELLS - 1) // PAGE_CELLS,
            "nativeBatchesAt8Cells": (count + MAX_SAMPLES - 1) // MAX_SAMPLES,
            "nativeSecondsEstimate": str(Fraction(count, NATIVE_CELLS_PER_SECOND)),
            "throughputBasisCellsPerSecond": NATIVE_CELLS_PER_SECOND,
            "workflowElapsedEstimate": None,
            "nativeMeasurementsStarted": False,
            "candidateOnly": True,
            "headNearVerified": False,
        }


@dataclass(frozen=True, slots=True)
class ScopeRequests:
    cells: tuple[ImageFrameCell, ...]
    missing: tuple[MeasurementRequest, ...]
    unused: int
    reused: frozenset[Fraction]
    unsupported: tuple[UnsupportedClockCell, ...] = ()


def _title_ranges(titles: list[ET.Element]) -> tuple[tuple[Fraction, Fraction], ...]:
    spans = sorted(
        (
            _time(title.attrib, "offset"),
            _time(title.attrib, "offset") + _time(title.attrib, "duration"),
        )
        for title in titles
    )
    merged: list[tuple[Fraction, Fraction]] = []
    for left, right in spans:
        if left >= right:
            raise ReplayError("Caption source ranges must have positive duration")
        if merged and left <= merged[-1][1]:
            start, end = merged.pop()
            merged.append((start, max(end, right)))
        else:
            merged.append((left, right))
    return tuple(merged)


def source_scopes(root: ET.Element) -> tuple[SourceScope, ...]:
    """Full scopes use their own bounds; dense 512-title plans remain unchanged."""
    if root.find(".//timeMap") is not None:
        raise ReplayError("Seed planning requires direct source clocks without time maps")
    scopes: list[SourceScope] = []
    count = 0
    for clip in root.findall("./library/event/project/sequence/spine/asset-clip"):
        titles = clip.findall("title")
        if not titles:
            continue
        count += len(titles)
        start = _time(clip.attrib, "start")
        end = start + _time(clip.attrib, "duration")
        scopes.append(SourceScope(start, end, _title_ranges(titles)))
    if not scopes or not 1 <= count <= MAX_TITLES:
        raise ReplayError("Full seed planning exceeds its bounded title/scope budget")
    return tuple(scopes)


def selected_references(index: Path, scopes: tuple[SourceScope, ...]) -> list[tuple[Path, Path]]:
    selected: list[tuple[Path, Path]] = []
    for config, evidence in face_references(index):
        job = storage.load_job(config)
        left = job.clock.pts[0] * job.clock.time_base
        right = (job.clock.pts[-1] + job.clock.last_duration) * job.clock.time_base
        if any(left < scope.end and scope.start < right for scope in scopes):
            selected.append((config, evidence))
    if not 1 <= len(selected) <= MAX_WINDOWS or len(set(selected)) != len(selected):
        raise ReplayError("Seed scopes require bounded distinct verified tracking windows")
    return selected


def _required_time(request: MeasurementRequest, scope: SourceScope) -> Fraction | None:
    index = bisect_right(scope.titles, request.unsupported_start, key=lambda span: span[1])
    if index == len(scope.titles) or scope.titles[index][0] >= request.unsupported_end:
        return None
    return max(request.unsupported_start, scope.titles[index][0])


def _target(cells: tuple[CurrentFaceCell, ...], time: Fraction) -> ImageFrameCell:
    index = bisect_right(cells, time, key=lambda cell: cell.start) - 1
    if index < 0 or not cells[index].start <= time < cells[index].end:
        raise ReplayError("Required title clock lacks an actual current source-frame cell")
    cell = cells[index]
    if cell.face is None:
        raise ReplayError("A missing or ambiguous face cannot become a pixel-seed ROI")
    x, y, width, height = cell.face
    return ImageFrameCell(cell.start, cell.end, _roi(Box(x, y, x + width, y + height)))


def _bounded_scope_requests(
    faces: FaceScaleSeries,
    *,
    scope: SourceScope,
    pixels: VisibleArc,
    resets: tuple[Fraction, ...],
) -> ScopeRequests:
    raw_cells = current_cells(faces, start=scope.start, end=scope.end)
    approximate = scoped_plan(faces, pixels=pixels, start=scope.start, end=scope.end, resets=resets)
    reused = frozenset(
        cell.face_time for cell in approximate.cells if cell.policy == "pixel-seed-candidate"
    )
    targets: list[ImageFrameCell] = []
    missing: list[MeasurementRequest] = []
    unused = 0
    for request in approximate.requests:
        time = _required_time(request, scope)
        if time is None:
            unused += 1
            continue
        if request.face is None:
            missing.append(request)
            continue
        targets.append(_target(raw_cells, time))
    return ScopeRequests(tuple(targets), tuple(missing), unused, reused)


def _intersections(
    scope: SourceScope, *, start: Fraction, end: Fraction
) -> tuple[tuple[Fraction, Fraction], ...]:
    left_index = bisect_right(scope.titles, start, key=lambda span: span[1])
    right_index = bisect_left(scope.titles, end, key=lambda span: span[0])
    return tuple(
        (max(left, start), min(right, end)) for left, right in scope.titles[left_index:right_index]
    )


def _bounded_cell(cell: CurrentFaceCell) -> bool:
    return cell.end - cell.start <= NATIVE_FRAME_CELL_LIMIT


def _split_scope(
    cells: tuple[CurrentFaceCell, ...], scope: SourceScope
) -> tuple[tuple[SourceScope, ...], tuple[UnsupportedClockCell, ...]]:
    supported: list[SourceScope] = []
    unsupported: list[UnsupportedClockCell] = []
    for bounded, group in groupby(cells, key=_bounded_cell):
        run = tuple(group)
        if not bounded:
            unsupported.extend(
                UnsupportedClockCell(
                    cell, scope, _intersections(scope, start=cell.start, end=cell.end)
                )
                for cell in run
            )
            continue
        start, end = max(scope.start, run[0].start), min(scope.end, run[-1].end)
        titles = _intersections(scope, start=start, end=end)
        if titles:
            supported.append(SourceScope(start, end, titles))
    return tuple(supported), tuple(unsupported)


def _scope_requests(
    faces: FaceScaleSeries,
    *,
    scope: SourceScope,
    pixels: VisibleArc,
    resets: tuple[Fraction, ...],
) -> ScopeRequests:
    source = current_cells(faces, start=scope.start, end=scope.end)
    bounded, unsupported = _split_scope(source, scope)
    planned = tuple(
        _bounded_scope_requests(faces, scope=part, pixels=pixels, resets=resets) for part in bounded
    )
    return ScopeRequests(
        tuple(chain.from_iterable(item.cells for item in planned)),
        tuple(chain.from_iterable(item.missing for item in planned)),
        sum(item.unused for item in planned),
        frozenset(chain.from_iterable(item.reused for item in planned)),
        unsupported,
    )


def build_requests(
    faces: FaceScaleSeries,
    *,
    scopes: tuple[SourceScope, ...],
    pixels: VisibleArc,
    resets: tuple[Fraction, ...] = (),
) -> SeedPlan:
    """Follow raw scopes but measure only real cells required by rendered titles."""
    if not 1 <= len(scopes) <= MAX_TITLES or len(faces.tops.positions) > MAX_CELLS:
        raise ReplayError("Seed planning exceeds its current-cell/scope budget")
    positions = faces.tops.positions
    visits = sum(
        bisect_left(positions, scope.end, key=lambda point: point.time)
        - bisect_right(positions, scope.start, key=lambda point: point.time)
        + 1
        for scope in scopes
    )
    if visits > MAX_SCOPED_CELLS:
        raise ReplayError("Repeated source scopes exceed the finite current-cell work budget")
    if any(a >= b for a, b in pairwise(resets)):
        raise ReplayError("Zoom resets require strictly increasing exact source clocks")
    starts = {point.time for point in faces.tops.positions}
    if any(time not in starts for time in resets):
        raise ReplayError("Zoom resets must bind actual source-frame starts")
    planned = tuple(
        _scope_requests(faces, scope=scope, pixels=pixels, resets=resets) for scope in scopes
    )
    targets: dict[Fraction, ImageFrameCell] = {}
    for target in chain.from_iterable(item.cells for item in planned):
        prior = targets.get(target.start)
        if prior is not None and prior != target:
            raise ReplayError("Duplicate requested source clocks have conflicting current ROIs")
        targets[target.start] = target
    ordered = tuple(targets[time] for time in sorted(targets))
    if any(left.end > right.start for left, right in pairwise(ordered)):
        raise ReplayError("Requested real source-frame cells overlap")
    return SeedPlan(
        ordered,
        tuple(chain.from_iterable(item.missing for item in planned)),
        sum(item.unused for item in planned),
        len(frozenset(chain.from_iterable(item.reused for item in planned))),
        tuple(chain.from_iterable(item.unsupported for item in planned)),
    )


def _output(path: Path) -> Path:
    destination = storage._checked_output(path.absolute())
    if not destination.is_relative_to(WORK_ROOT) or destination == WORK_ROOT:
        raise ReplayError("Seed plan output must remain a repository work child")
    return destination


def _config(path: Path) -> tuple[dict[str, object], dict[str, Path], tuple[Fraction, ...]]:
    data = _json(path)
    required = {"version", "xml", "faceIndex", "native", "sandbox", "zoomResetTimes"}
    if (
        set(data) - {"retainedTracking", "retainedImage"} != required
        or type(data["version"]) is not int
        or data["version"] != 1
    ):
        raise ReplayError("Expected explicit bounded full seed-plan configuration")
    paths = {name: _pin(data[name]) for name in ("xml", "faceIndex", "native", "sandbox")}
    times = data["zoomResetTimes"]
    if not isinstance(times, list) or len(times) > MAX_TITLES:
        raise ReplayError("Expected bounded exact zoom reset clocks")
    return data, paths, tuple(_fraction(time) for time in times)


def _publish(
    plan: SeedPlan,
    *,
    output: Path,
    inputs: dict[str, str],
    job: storage.TrackingJob,
    native: Path,
    sandbox: Path,
    cache: InputHashCache,
) -> dict[str, object]:
    output.mkdir(mode=0o700)
    pages: list[dict[str, object]] = []
    for index, cells in enumerate(batched(plan.cells, PAGE_CELLS)):
        if index >= MAX_PAGES:
            raise ReplayError("Seed request pages exceed their finite budget")
        page = output / f"requests-{index:04d}.json"
        storage._write(page, {"version": 1, "cells": [cell.encoded() for cell in cells]})
        pages.append({"path": str(page), "sha256": file_digest(page), "cells": len(cells)})
    cache.verify()
    result: dict[str, object] = {
        "version": 1,
        "protocol": PROTOCOL,
        "inputs": inputs,
        "source": {"path": str(job.source), "sha256": job.source_hash},
        "native": {"path": str(native), "sha256": inputs[str(native)]},
        "sandbox": {"path": str(sandbox), "sha256": inputs[str(sandbox)]},
        "width": job.width,
        "height": job.height,
        "pages": pages,
        "cost": plan.cost(),
        "missingRequests": [request.encoded() for request in plan.missing],
        "unsupportedClockCells": [cell.encoded() for cell in plan.unsupported],
        "completeRequestedSeedsMeasured": False,
        "denseFrameCoverage": False,
        "candidateOnly": True,
        "headNearVerified": False,
    }
    storage._write(output / "requests.json", result)
    return result


def plan(config: Path, output: Path) -> dict[str, object]:
    """Zero native/GUI calls; preserve cryptographic inputs and immutable request pages."""
    config = storage._path(str(config.absolute()))
    output = _output(output)
    if output.exists():
        raise ReplayError("Seed request namespace must be absent")
    data, paths, resets = _config(config)
    root = ET.fromstring(_bound_bytes(paths["xml"], file_digest(paths["xml"]), MAX_XML_BYTES))
    scopes = source_scopes(root)
    references = selected_references(paths["faceIndex"], scopes)
    retained = None
    if data.get("retainedTracking") is not None:
        descriptor = _pin(data["retainedTracking"])
        retained = verify_catalog(descriptor, sha256=file_digest(descriptor), references=references)
    job, faces = read_face_scale(
        references, missing_policy="hold", seed_overlap="next", retained=retained
    )
    inputs = {
        str(path): file_digest(path)
        for path in (
            config,
            *paths.values(),
            *(Path(__file__).with_name(name) for name in CODE_FILES),
        )
    }
    inputs[str(job.source)] = job.source_hash
    if retained is not None:
        inputs.update(retained.inputs)
    else:
        for cfg, evidence in references:
            current = storage.load_job(cfg)
            inputs.update({str(cfg): current.config_hash, str(current.native): current.native_hash})
            inputs.update(
                {
                    str(path): file_digest(path)
                    for path in (
                        evidence / "request.json",
                        evidence / "response.json",
                        evidence / "provenance.json",
                    )
                }
            )
    pixels = VisibleArc((), {})
    if data.get("retainedImage") is not None:
        image = read_image_binding(
            data["retainedImage"], source_hash=job.source_hash, width=job.width, height=job.height
        )
        pixels = image.arc
        inputs.update(image.inputs)
    cache = InputHashCache((Path(name) for name in inputs), allow_uncached_overflow=True)
    cache.prime()
    with input_hash_context(cache):
        if any(file_digest(Path(name)) != digest for name, digest in inputs.items()):
            raise ReplayError("Seed-plan protected input content differs")
        result = build_requests(faces, scopes=scopes, pixels=pixels, resets=resets)
        return _publish(
            result,
            output=output,
            inputs=inputs,
            job=job,
            native=paths["native"],
            sandbox=paths["sandbox"],
            cache=cache,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = plan(arguments.config, arguments.output)
    print(json.dumps(result["cost"]))


if __name__ == "__main__":
    main()
