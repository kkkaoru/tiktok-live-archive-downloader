"""Replay predeclared measured-tile models from pinned correspondence metadata.

This leaf never opens source media, performs optical flow or adopts caption Y.
It uses the current typed fitting and approximation contracts unchanged.
"""

import argparse
import json
import math
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from time import monotonic

import numpy as np

from replay import head_partial_pixel_track as pixels
from replay.code_dependency_inventory import capture
from replay.core import ReplayError
from replay.delivery_mask_quality import integer, rows
from replay.foreground_evidence import _bound_bytes
from replay.head_seed import Coordinates
from replay.head_texture_track import Region, TextureFit, fit_texture_model
from replay.recordings import record
from replay.stages import _uncached_file_digest
from replay.subtitle_clock import VideoFrameClock

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "local-support-correspondence-replay-v1"
PARTITION_POLICY = "measured-seed-area-median-v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_TILES = 256


@dataclass(frozen=True, slots=True)
class LocalReplay:
    source_pts: int
    cut_y: int
    tile_owners: tuple[int, ...]
    feature_owners: tuple[tuple[int, ...], tuple[int, ...]]
    local_models: tuple[dict[str, object], ...]
    pair: dict[str, object] | None
    reason: str | None

    def audit(self) -> dict[str, object]:
        return {
            "sourcePTS": self.source_pts,
            "partitionPolicy": PARTITION_POLICY,
            "seedAreaMedianY": self.cut_y,
            "originalTileOwners": self.tile_owners,
            "originalFeatureOwners": self.feature_owners,
            "localModels": self.local_models,
            "localPairDecision": self.pair,
            "reason": self.reason,
            "continuousControlChainVerified": False,
            "currentMeasuredUpperPixels": False,
            "semanticHeadVerified": False,
            "headNearVerified": False,
            "captionPlacementApproved": False,
            "YAdopted": False,
        }


def _number(value: object) -> float:
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        raise ValueError("Finite numeric metadata is required")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Finite numeric metadata is required")
    return result


def _regions(value: object, *, width: int, height: int) -> tuple[Region, ...]:
    entries = rows(value)
    if not 1 <= len(entries) <= MAX_TILES:
        raise ValueError("Bounded measured tiles are required")
    result: list[Region] = []
    for entry in entries:
        values = tuple(integer(item) for item in rows(entry))
        if len(values) != 4:
            raise ValueError("Measured tile geometry differs")
        x, y, w, h = values
        if min(w, h) < 1 or x + w > width or y + h > height:
            raise ValueError("Measured tile exceeds its canvas")
        result.append((x, y, w, h))
    for index, tile in enumerate(result):
        if any(_overlap(tile, other) for other in result[:index]):
            raise ValueError("Measured tiles cannot overlap")
    return tuple(result)


def _overlap(one: Region, two: Region) -> bool:
    return max(one[0], two[0]) < min(one[0] + one[2], two[0] + two[2]) and max(
        one[1], two[1]
    ) < min(one[1] + one[3], two[1] + two[3])


def measured_partition(regions: tuple[Region, ...], *, height: int) -> tuple[int, tuple[int, ...]]:
    """Assign whole native tiles before fitting, using measured occupied area."""
    if not 16 <= height <= 16384 or not regions:
        raise ValueError("Bounded measured partition geometry is required")
    counts = np.zeros(height, np.int64)
    for x, y, w, h in regions:
        if min(x, y) < 0 or min(w, h) < 1 or y + h > height:
            raise ValueError("Measured partition tile is invalid")
        counts[max(y, 8) : min(y + h, height - 8)] += w
    total = int(counts.sum())
    if total == 0:
        raise ValueError("Measured partition has no supported area")
    cut = int(np.searchsorted(np.cumsum(counts), (total + 1) // 2))
    owners = tuple(int(y + h / 2 >= cut) for x, y, w, h in regions)
    if set(owners) != {0, 1}:
        raise ValueError("Two independent measured regions are required")
    return cut, owners


def _owner(point: Coordinates, regions: tuple[Region, ...], owners: tuple[int, ...]) -> int:
    if not np.isfinite(point).all() or np.any(point < 0):
        raise ValueError("Current feature coordinates must be finite and nonnegative")
    x, y = int(float(point[0])), int(float(point[1]))
    matches = tuple(
        owner
        for tile, owner in zip(regions, owners, strict=True)
        if tile[0] <= x < tile[0] + tile[2] and tile[1] <= y < tile[1] + tile[3]
    )
    if len(matches) != 1:
        raise ValueError("Current feature lacks unique measured tile ownership")
    return matches[0]


def _local_fit(
    fits: tuple[TextureFit, TextureFit], feature_owners: tuple[tuple[int, ...], ...], owner: int
) -> tuple[TextureFit | None, tuple[tuple[int, int], ...]]:
    indices = tuple(
        (cohort, index)
        for cohort, group in enumerate(feature_owners)
        for index, region in enumerate(group)
        if region == owner
    )
    if len(indices) < pixels.MIN_FEATURES:
        return None, indices
    source = np.asarray([fits[c].source[i] for c, i in indices], np.float32)
    current = np.asarray([fits[c].metrics.points[i] for c, i in indices], np.float32)
    first = fits[0].metrics
    metrics = replace(
        first,
        points=current,
        supported=np.asarray([fits[c].metrics.supported[i] for c, i in indices], np.bool_),
        forward_status=np.asarray(
            [fits[c].metrics.forward_status[i] for c, i in indices], np.bool_
        ),
        backward_status=np.asarray(
            [fits[c].metrics.backward_status[i] for c, i in indices], np.bool_
        ),
        in_bounds=np.asarray([fits[c].metrics.in_bounds[i] for c, i in indices], np.bool_),
        roundtrip_error=np.asarray(
            [fits[c].metrics.roundtrip_error[i] for c, i in indices], np.float32
        ),
        appearance_error=np.asarray(
            [fits[c].metrics.appearance_error[i] for c, i in indices], np.float32
        ),
    )
    return fit_texture_model(source, metrics), indices


def replay_local_models(
    *,
    fits: tuple[TextureFit, TextureFit],
    controls: tuple[Coordinates, Coordinates],
    original_regions: tuple[Region, ...],
    current_regions: tuple[Region, ...],
    source_pts: int,
    width: int,
    height: int,
    budget: float,
) -> LocalReplay:
    """Replay one transition; every original selected point retains its status."""
    if (
        not 16 <= width <= 16384
        or type(source_pts) is not int
        or source_pts < 0
        or not math.isfinite(budget)
        or not 0 < budget <= pixels.target_y_budget(height)
    ):
        raise ValueError("Bounded original geometry, clock and uncertainty are required")
    if len(original_regions) != len(current_regions):
        raise ValueError("Dropped measured tiles cannot invent an index association")
    cut, owners = measured_partition(original_regions, height=height)
    features = tuple(
        tuple(_owner(point, current_regions, owners) for point in fit.source) for fit in fits
    )
    if len(features) != 2:
        raise ValueError("Exactly two original independent cohorts are required")
    feature_owners = (features[0], features[1])
    if set(map(tuple, fits[0].source)).intersection(map(tuple, fits[1].source)):
        raise ValueError("Original correspondence cohorts must be independent")
    local = tuple(_local_fit(fits, feature_owners, owner) for owner in range(2))
    audits: list[dict[str, object]] = []
    decisions: list[pixels.ApproximateModelDecision] = []
    complete_fits: list[TextureFit] = []
    for owner, (fitted, indices) in enumerate(local):
        if fitted is None:
            audits.append(
                {
                    "region": owner,
                    "originalIndices": indices,
                    "selected": len(indices),
                    "reason": "local_texture_insufficient",
                }
            )
            continue
        decision = pixels.approximate_model(
            fitted,
            controls[owner],
            maximum_target_y_uncertainty_px=budget,
            width=width,
            height=height,
        )
        complete_fits.append(fitted)
        decisions.append(decision)
        audits.append(
            {
                "region": owner,
                "originalIndices": indices,
                "selected": len(indices),
                "independentlySupported": int(np.count_nonzero(fitted.metrics.supported)),
                "legacyReason": fitted.reason,
                "sourceSpan": fitted.source_span,
                "inlierSpan": fitted.inlier_span,
                "matrix": fitted.matrix,
                "strictInliers": fitted.inliers,
                "strictResiduals": fitted.residuals,
                "approximateHeadContract": decision.audit,
            }
        )
    pair: pixels.ModelPairDecision | None = None
    if len(complete_fits) == 2:
        pair = pixels._model_pair(
            (complete_fits[0], complete_fits[1]),
            controls,
            (decisions[0], decisions[1]),
            width=width,
            height=height,
        )
    reason = "local_texture_insufficient" if pair is None else pair.reason
    return LocalReplay(
        source_pts,
        cut,
        owners,
        feature_owners,
        tuple(audits),
        None if pair is None else pair.audit,
        reason,
    )


def _path(value: object) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Canonical metadata path is required")
    path = Path(value)
    if (
        not path.is_absolute()
        or path != path.resolve()
        or not path.is_relative_to(ROOT / "edits" / "work")
    ):
        raise ValueError("Metadata must stay in canonical repository work")
    return path


def _read_pin(value: object, inputs: dict[str, str]) -> object:
    reference = record(value)
    if set(reference) != {"path", "sha256"}:
        raise ValueError("Exact metadata reference is required")
    path = _path(reference["path"])
    digest = reference["sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("Metadata SHA256 is required")
    if str(path) in inputs and inputs[str(path)] != digest:
        raise ValueError("Conflicting metadata references")
    inputs[str(path)] = digest
    if sum(Path(name).stat().st_size for name in inputs) > MAX_METADATA_BYTES:
        raise ValueError("One-step metadata replay exceeds its byte budget")
    return json.loads(_bound_bytes(path, digest, MAX_JSON_BYTES))


def _bound_original(value: object, report: dict[str, object], inputs: dict[str, str]) -> object:
    reference = record(value)
    if record(report.get("inputs")).get(str(reference.get("path"))) != reference.get("sha256"):
        raise ValueError("Original producer metadata binding differs")
    return _read_pin(reference, inputs)


def run(config: Path, output: Path) -> dict[str, object]:
    """Write one exclusive metadata replay without touching source media."""
    started = monotonic()
    _path(str(config))
    _path(str(output))
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Absent metadata output in an existing directory is required")
    code = capture(("head_local_support_probe",))
    code_inputs = {str(item.path): item.sha256 for item in code}
    inputs = {str(config): _uncached_file_digest(config)}
    raw = record(json.loads(_bound_bytes(config, inputs[str(config)], MAX_JSON_BYTES)))
    if (
        set(raw) != {"version", "protocol", "producerConfig", "report", "sourcePTS", "contract"}
        or type(raw["version"]) is not int
        or raw["version"] != 1
        or raw["protocol"] != PROTOCOL
    ):
        raise ValueError("Unsupported local correspondence replay configuration")
    contract = record(raw["contract"])
    if (
        set(contract) != {"partitionPolicy", "maximumTargetYModelUncertaintyPx"}
        or contract["partitionPolicy"] != PARTITION_POLICY
    ):
        raise ValueError("Explicit unchanged local model contract is required")
    report = record(_read_pin(raw["report"], inputs))
    producer = record(_bound_original(raw["producerConfig"], report, inputs))
    for reference in rows(report.get("spatialModelRecords")):
        _read_pin(reference, inputs)
    observations = pixels.spatial_observations(report)
    pts = integer(raw["sourcePTS"])
    matches = tuple(i for i, item in enumerate(observations) if item.get("source_pts") == pts)
    if len(matches) != 1 or matches[0] == 0:
        raise ValueError("Exactly one actual non-seed correspondence transition is required")
    index = matches[0]
    current = observations[index]
    previous = observations[index - 1]
    if previous.get("reason") is not None:
        raise ValueError("Current tile association requires a fully supported predecessor")
    width, height = integer(producer.get("width")), integer(producer.get("height"))
    source = record(producer.get("source"))
    if record(report.get("inputs")).get(str(source.get("path"))) != source.get("sha256"):
        raise ValueError("Original source metadata binding differs")
    clock_data = record(producer.get("clock"))
    time_base = clock_data.get("timeBase")
    if not isinstance(time_base, str):
        raise ValueError("Original presentation clock is required")
    clock = VideoFrameClock(
        tuple(integer(p) for p in rows(clock_data.get("pts"))),
        Fraction(time_base),
        integer(clock_data.get("lastDuration")),
    )
    pixels.validate_clock(clock)
    if (
        pts not in clock.pts
        or observations[0].get("source_pts") != clock.pts[0]
        or type(current.get("face_observed")) is not bool
    ):
        raise ValueError("Original source/face presentation clock differs")
    face, flags = pixels.detector_seed(
        _bound_original(producer.get("detectorResponse"), report, inputs), clock=clock
    )
    if (
        tuple(_number(v) for v in rows(report.get("seedDetectorRegion"))) != face
        or flags[clock.pts.index(pts)] != current["face_observed"]
    ):
        raise ValueError("Original accepted seed or raw face status differs")
    seed = pixels.read_seed(
        _bound_original(producer.get("nativeResponse"), report, inputs),
        clock=clock,
        width=width,
        height=height,
        face=face,
    )
    if seed is None:
        raise ValueError("Original measured interior seed is unsupported")
    regions = _regions(report.get("seedAnchorRegions"), width=width, height=height)
    if regions != seed.anchor_regions:
        raise ValueError("Native measured tile ownership differs")
    refresh = record(previous.get("feature_refresh"))
    if (
        refresh.get("status") != "current-pixel-features-selected"
        or record(refresh.get("modelPairDecision")).get("accepted") is not True
    ):
        raise ValueError("Predecessor current geometry is not fully supported")
    current_regions = _regions(refresh.get("derivedSupportRegions"), width=width, height=height)
    models = rows(current.get("models"))
    if len(models) != 2:
        raise ValueError("Exactly two original model records are required")
    budget = _number(contract["maximumTargetYModelUncertaintyPx"])
    if budget != _number(
        record(producer.get("approximateYPolicy")).get("maximumTargetYModelUncertaintyPx")
    ):
        raise ValueError("Original target uncertainty budget cannot change")
    parsed = tuple(
        pixels._validated_model(
            model, maximum_target_y_uncertainty_px=budget, width=width, height=height
        )
        for model in models
    )
    result = replay_local_models(
        fits=(parsed[0][0], parsed[1][0]),
        controls=(parsed[0][1], parsed[1][1]),
        original_regions=regions,
        current_regions=current_regions,
        source_pts=pts,
        width=width,
        height=height,
        budget=budget,
    )
    if any(_uncached_file_digest(Path(name)) != digest for name, digest in inputs.items()):
        raise ValueError("Metadata or reader changed during replay")
    after_code = capture(("head_local_support_probe",))
    if {str(item.path): item.sha256 for item in after_code} != code_inputs:
        raise ValueError("Reader code membership or bytes changed during replay")
    artifact: dict[str, object] = {
        "version": 1,
        "protocol": PROTOCOL,
        "inputs": inputs,
        "readerCodeInputs": code_inputs,
        "elapsedSeconds": monotonic() - started,
        "source": source,
        "timeBase": time_base,
        "faceObserved": current["face_observed"],
        "originalGlobalModels": models,
        "replay": result.audit(),
        "opticalFlowExecuted": False,
        "sourceMediaRead": False,
        "nativeExecuted": False,
        "newFaceObservation": False,
    }
    encoded = json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_JSON_BYTES:
        raise ValueError("Local metadata output exceeds its byte budget")
    with output.open("xb") as handle:
        handle.write(encoded)
    output.chmod(0o600)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        artifact = run(args.config.absolute(), args.output.absolute())
    except (ValueError, OSError, ReplayError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "sourcePTS": record(artifact["replay"])["sourcePTS"],
                "reason": record(artifact["replay"])["reason"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
