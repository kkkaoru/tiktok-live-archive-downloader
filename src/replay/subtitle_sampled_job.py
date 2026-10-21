"""One-chunk job: sampled-OCR repair with real collaborators, audit gate and receipt.

``python -m replay.subtitle_sampled_job CONFIG.json`` reads a bounded JSON description of
one source chunk (clock, canvas, source, profile observations, native tool, pinned detector
artifacts), runs ``subtitle_sampled_run`` and prints the accepted attempt. Existing outputs
are never replaced; a failed audit after escalation leaves a receipt and a nonzero exit.
"""

import argparse
import glob
import json
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from replay import subtitle_band_audit as band_audit
from replay import subtitle_colour_audit as colour_audit
from replay import subtitle_colour_support as colour
from replay import subtitle_detector_mps as mps
from replay import subtitle_sampled_chunk as chunk
from replay import subtitle_sampled_run as runner
from replay import subtitle_sttn
from replay import subtitle_yolo_admission as yolo_plan
from replay import subtitle_yolo_regions as yolo
from replay.caption_io import FrameObservation, native_ocr_frames
from replay.local_asr import fingerprint
from replay.recordings import record
from replay.subtitle_cleanup_plan import CaptionSearchProfile, search_profile
from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image
from replay.subtitle_plan import repair_spans
from replay.subtitle_region_fill import Region
from replay.subtitle_repair_audit import Residual
from replay.subtitle_stream import SubtitleSpan
from replay.subtitle_sttn import refill_bands
from replay.subtitle_video import VideoRepairJob
from replay.subtitle_video_fast import (
    Schedule,
    SupportStore,
    render_refilled_video,
    render_repaired_video_fast,
)

MAX_CONFIG_BYTES = 8 * 1024 * 1024
RENDER_BATCH = 16
COLOUR_KEY_PIN = "pink-key-v1"


class Detects(Protocol):
    def detect(self, images: Sequence[Image]) -> tuple[Image, ...]: ...


def _span_origin(spans: Sequence[SubtitleSpan], index: int) -> tuple[int, int]:
    for span in spans:
        if span.start_frame <= index < span.end_frame:
            return span.x, span.y
    raise ValueError(f"Active frame {index} lies in no repair span")


def _entry_boxes(entry: object) -> list[colour.Box]:
    raw = record(entry).get("lineBoxes", [])
    if not isinstance(raw, list):
        raise ValueError("Admission line boxes must be a list")
    return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in raw]


def key_frames(
    admitted: Mapping[str, object], spans: Sequence[SubtitleSpan]
) -> list[colour.FrameBoxes]:
    """Per active frame in render order: ROI origin and the recognised line boxes."""
    entries = admitted["frames"]
    if not isinstance(entries, list):
        raise ValueError("Admission lacks per-frame entries")
    frames: list[colour.FrameBoxes] = []
    for index, raw in enumerate(entries):
        entry = record(raw)
        if entry.get("classification") != "overlay-subtitle":
            continue
        if "lineBoxes" not in entry:
            raise ValueError("Admission lacks recognised line boxes for colour keying")
        boxes = _entry_boxes(entry)
        region = entry.get("region")
        if not boxes and isinstance(region, dict):
            # The detector found a caption that OCR could not read (a lone digit): key its region.
            x, y = int(region["x"]), int(region["y"])
            boxes = [(x, y, x + int(region["width"]), y + int(region["height"]))]
        frames.append(colour.FrameBoxes(_span_origin(spans, index), tuple(boxes)))
    return frames


def with_residual(
    admitted: dict[str, object],
    boxes: Sequence[Sequence[yolo.Corners]],
    extra: Mapping[int, Sequence[colour.Box]],
) -> list[list[yolo.Corners]]:
    """Add residual boxes to the planned YOLO boxes and line boxes of their caption frames.

    A residual line beyond a frame's caption region (a tall block whose upper lines lie above the
    profiled band) widens that region and its mask too, or the repair would stay clipped to it.
    """
    entries, masks = admitted["frames"], admitted.get("masks")
    if not isinstance(entries, list) or not isinstance(masks, list):
        raise ValueError("Admission lacks per-frame entries")
    overlays = [
        index
        for index, entry in enumerate(entries)
        if record(entry).get("classification") == "overlay-subtitle"
    ]
    merged = [list(frame) for frame in boxes]
    for index, added in extra.items():
        if index not in overlays:
            continue
        entry = record(entries[index])
        entry["lineBoxes"] = [*_entry_boxes(entry), *(list(box) for box in added)]
        regions = [record(entry["region"]), record(record(masks[overlays.index(index)])["region"])]
        for region in regions:
            _widen(region, added)
        if index < len(merged):
            merged[index] += [(float(a), float(b), float(c), float(d)) for a, b, c, d in added]
    return merged


def _widen(region: dict[str, object], boxes: Sequence[colour.Box]) -> None:
    """Grow an {x, y, width, height} region in place to cover every box and one line beyond.

    The extra line height above and below lets the colour key's stacked-line rule reach a
    further caption line of the same block that OCR did not report.
    """
    x, y = int(str(region["x"])), int(str(region["y"]))
    right, bottom = x + int(str(region["width"])), y + int(str(region["height"]))
    left = min([x, *(box[0] for box in boxes)])
    top = min([y, *(box[1] - (box[3] - box[1]) for box in boxes)])
    right = max([right, *(box[2] for box in boxes)])
    bottom = max([bottom, *(box[3] + (box[3] - box[1]) for box in boxes)])
    region.update(x=left, y=top, width=right - left, height=bottom - top)


def box_frames(
    admitted: Mapping[str, object],
    spans: Sequence[SubtitleSpan],
    boxes: Sequence[Sequence[yolo.Corners]],
) -> list[yolo.PlannedBoxes]:
    """Per active frame in render order: ROI origin and that frame's YOLO caption boxes."""
    entries = admitted["frames"]
    if not isinstance(entries, list) or len(entries) != len(boxes):
        raise ValueError("YOLO boxes do not match the admitted frames")
    return [
        yolo.PlannedBoxes(_span_origin(spans, index), tuple(boxes[index]))
        for index, raw in enumerate(entries)
        if record(raw).get("classification") == "overlay-subtitle"
    ]


def _clock(value: object) -> VideoFrameClock:
    data = record(value)
    pts = data.get("pts")
    if not isinstance(pts, list) or not all(type(item) is int for item in pts):
        raise ValueError("Invalid source clock")
    base, last = data.get("timeBase"), data.get("lastDuration")
    if not isinstance(base, str) or type(last) is not int:
        raise ValueError("Invalid source clock")
    return VideoFrameClock(tuple(int(item) for item in pts), Fraction(base), last)


def _profile(directory: Path) -> CaptionSearchProfile:
    files = sorted(glob.glob(str(directory / "*.result.json")))
    if not files:
        raise ValueError("Missing subtitle profile observations")
    frames = tuple(
        frame
        for name in files
        for frame in native_ocr_frames(json.loads(Path(name).read_text(encoding="utf-8")))
    )
    return search_profile(frames)


def real_collaborators(
    config: Mapping[str, object],
    *,
    network: mps.Network,
    render: Callable[[VideoRepairJob, Detects], object] | None = None,
    caption_detector: yolo.Detector | None = None,
    caption_masks: bool = False,
    yolo_admission: bool = False,
    sttn: subtitle_sttn.Inpainter | None = None,
) -> tuple[runner.Collaborators, dict[str, str]]:
    """Build the planning, rendering and auditing collaborators from one chunk description."""
    source, native = Path(str(config["source"])), Path(str(config["native"]))
    clock = _clock(config["clock"])
    canvas = (int(str(config["width"])), int(str(config["height"])))
    root = Path(str(config["checkoutRoot"]))
    state_root = Path(str(config["output"]))
    media_root = Path(str(config.get("mediaOutput", config["output"])))
    profile = _profile(Path(str(config["profileDirectory"])))
    envelope = profile.measured_envelope
    band = chunk.ocr_band(envelope, canvas)
    radius = int(str(config.get("radius", 6)))
    workers = int(str(config.get("workers", 2)))

    if (caption_masks or yolo_admission) and caption_detector is None:
        raise ValueError("YOLO repair masks and admission need the YOLO caption detector")
    planned_boxes: list[list[yolo.Corners]] = []
    extra: dict[int, list[colour.Box]] = {}

    def residual(lines: Sequence[Residual], audit_stride: int) -> None:
        # A residual glyph persists between audited frames: cover the frames around each one.
        for line in lines:
            if line.box is None:
                continue
            x, y, w, h = line.box
            box = (int(x), int(y), int(np.ceil(x + w)), int(np.ceil(y + h)))
            for index in range(line.frame - audit_stride, line.frame + audit_stride + 1):
                extra.setdefault(index, []).append(box)

    def regions() -> list[Region | None]:
        if caption_detector is None:
            return chunk.detector_regions(
                source, clock, canvas=canvas, envelope=envelope, network=network
            )
        frames = yolo.chunk_frames(
            source, clock, canvas=canvas, ffmpeg=str(config.get("ffmpeg", "ffmpeg"))
        )
        planned_boxes[:] = yolo.chunk_boxes(
            frames, caption_detector, count=len(clock.pts), scale=yolo.canvas_scale(canvas)
        )
        return yolo.yolo_regions(planned_boxes, canvas=canvas, envelope=envelope)

    def plan(stride: int, directory: Path) -> dict[str, object]:
        if yolo_admission and stride != runner.FULL_STRIDE:
            # The escalation attempt (stride 1) keeps the full-OCR plan as the fallback.
            started = time.monotonic()
            regions()
            admitted = yolo_plan.admission(
                boxes=planned_boxes, clock=clock, profile=profile, radius=radius
            )
            admitted["stageSeconds"] = {"yoloBoxes": round(time.monotonic() - started, 3)}
            return admitted
        sampled_at = directory / "ocr"
        return chunk.plan_chunk(
            clock=clock,
            profile=profile,
            stride=stride,
            radius=radius,
            triggers=lambda: chunk.band_triggers(
                source, clock, canvas_width=canvas[0], envelope=envelope
            ),
            ocr=lambda indices: chunk.session_ocr(
                native,
                source,
                clock,
                indices,
                canvas=canvas,
                work=sampled_at,
                root=root,
                workers=workers,
                band=band,
            ),  # fmt: skip
            regions=regions,
        )

    def render_attempt(admitted: dict[str, object], directory: Path) -> Path:
        boxes = with_residual(admitted, planned_boxes, extra) if extra else planned_boxes
        (directory / "admission.json").write_text(json.dumps(admitted), encoding="utf-8")
        masks = admitted["masks"]
        spans = repair_spans(
            masks, frames=len(clock.pts), width=canvas[0], height=canvas[1], source_clock=clock
        )
        media = media_root / directory.name
        media.mkdir(parents=True, exist_ok=True, mode=0o700)
        video = media / "candidate.mp4"
        job = VideoRepairJob(
            source, video, canvas[0], canvas[1], len(clock.pts), spans, source_clock=clock,
            hardware_encode=sttn is not None, hardware_hevc=False, context_frames=0,
            context_donors_verified=False, decode_log=directory / "decode.log",
            ramp_start_frame=0, ramp_total_frames=None,
        )  # fmt: skip
        boxed: yolo.BoxMaskDetector | None = None
        detector: Detects
        if caption_masks:
            boxed = yolo.BoxMaskDetector(box_frames(admitted, spans, boxes))
            detector = boxed
        else:
            detector = mps.MpsTextDetector(network, max_batch=RENDER_BATCH)
        keyed: colour.ColourKeyDetector | None = None
        if config.get("colourKey") is True:
            planned = key_frames(admitted, spans)
            keyed = colour.ColourKeyDetector(detector.detect, planned)
            detector = keyed
        if render is not None:
            render(job, detector)
        elif sttn is not None:
            # Find the exact repair pixels with the usual renderer, then refill them with STTN.
            kept = (
                min((s.y for s in spans), default=0),
                max((s.y + s.height for s in spans), default=0),
            )
            store = SupportStore(canvas[0], kept)
            started = time.monotonic()
            render_repaired_video_fast(
                job,
                detect_batch=detector.detect,
                schedule=Schedule(workers=int(str(config.get("repairWorkers", 4))), window=32,
                                  detect_chunk=RENDER_BATCH),
                support_sink=store,
            )  # fmt: skip
            painter = sttn

            def refill(bands: Iterator[Image], rows: tuple[int, int]) -> NDArray[np.uint8]:
                return refill_bands(bands, store.bands(*rows), painter)

            repaired = time.monotonic()
            render_refilled_video(job, store, refill)
            # Split of the render between the mask repair pass and the STTN refill pass.
            (directory / "render-seconds.json").write_text(
                json.dumps({"repair": round(repaired - started, 3),
                            "sttnRefill": round(time.monotonic() - repaired, 3)}),
                encoding="utf-8",
            )  # fmt: skip
        else:
            render_repaired_video_fast(
                job,
                detect_batch=detector.detect,
                schedule=Schedule(workers=int(str(config.get("repairWorkers", 4))), window=32,
                                  detect_chunk=RENDER_BATCH),
            )  # fmt: skip
        if keyed is not None and keyed.consumed != len(planned):
            raise ValueError("Colour-key frames were not all consumed by the renderer")
        if boxed is not None and boxed.consumed != len(box_frames(admitted, spans, boxes)):
            raise ValueError("YOLO mask frames were not all consumed by the renderer")
        return video

    relative = VideoFrameClock(
        tuple(pts - clock.pts[0] for pts in clock.pts), clock.time_base, clock.last_duration
    )

    audit_tile = config.get("auditTile")
    if audit_tile is not None and (type(audit_tile) is not int or not 1 <= audit_tile <= 4):
        raise ValueError("auditTile must be an integer from 1 to 4")

    audited: dict[tuple[Path, tuple[int, ...]], list[Image]] = {}
    gated = config.get("colourGate") is True
    # The colour gate's source rows depend only on the source and the frames: decode them
    # while the band audit recognises the candidate.
    ahead: dict[tuple[Path, tuple[int, ...]], Future[list[Image]]] = {}
    decoder = ThreadPoolExecutor(max_workers=1)

    def source_bands(indices: tuple[int, ...]) -> list[Image]:
        return list(
            colour_audit._frames(
                str(config.get("ffmpeg", "ffmpeg")), source,
                float(clock.pts[0] * clock.time_base), indices, canvas,
                band_audit.audit_band(envelope, canvas),
            )
        )  # fmt: skip

    def audit(video: Path, indices: Sequence[int]) -> Mapping[int, FrameObservation]:
        # A repaired candidate's timeline starts at zero, not at the chunk's source time.
        work = state_root / video.parent.name / "audit"
        if audit_tile is None:
            return chunk.session_ocr(
                native, video, relative, indices, canvas=canvas,
                work=work, root=root, workers=workers, band=band,
            )  # fmt: skip

        def recognise(
            movie: Path, tiles: VideoFrameClock, height: int
        ) -> dict[int, FrameObservation]:
            return chunk.session_ocr(
                native, movie, tiles, range(len(tiles.pts)), canvas=(canvas[0], height),
                work=movie.parent / "ocr", root=root, workers=workers,
            )  # fmt: skip

        decoded: list[Image] = []
        ahead.clear()
        if gated:
            ahead[(video, tuple(indices))] = decoder.submit(source_bands, tuple(indices))
        observed = band_audit.band_audit(
            video, indices, clock=relative, canvas=canvas, envelope=envelope, tile=audit_tile,
            work=work, recognise=recognise, ffmpeg=str(config.get("ffmpeg", "ffmpeg")),
            decoded=decoded,
        )  # fmt: skip
        audited.clear()
        audited[(video, tuple(indices))] = decoded  # the colour gate reads these bands again
        return observed

    def colour_residual(
        admitted: dict[str, object], video: Path, indices: Sequence[int]
    ) -> list[int]:
        entries = admitted["frames"]
        if not isinstance(entries, list):
            raise ValueError("Admission lacks per-frame entries")
        boxes = [_entry_boxes(entry) for entry in entries]
        bands = audited.pop((video, tuple(indices)), None)
        pending = ahead.pop((video, tuple(indices)), None)
        return colour_audit.colour_counts(
            ffmpeg=str(config.get("ffmpeg", "ffmpeg")),
            source=source,
            source_start=float(clock.pts[0] * clock.time_base),
            cleaned=video,
            indices=indices,
            boxes=boxes,
            size=canvas,
            band=None if bands is None else band_audit.audit_band(envelope, canvas),
            cleaned_bands=bands,
            source_bands=None if pending is None else pending.result(),
        )

    def caption_coloured(residuals: Sequence[Residual]) -> list[bool]:
        lines = [(item.frame, item.box) for item in residuals if item.box is not None]
        if len(lines) != len(residuals):
            raise ValueError("Residual lines lack their boxes")
        counts = colour_audit.source_pink_counts(
            ffmpeg=str(config.get("ffmpeg", "ffmpeg")),
            source=source,
            source_start=float(clock.pts[0] * clock.time_base),
            lines=lines,
            size=canvas,
        )
        return [count >= colour_audit.MINIMUM_CAPTION_PINK for count in counts]

    pins = {"native": fingerprint(native), "source": str(config.get("sourceSHA256", ""))}
    if audit_tile is not None:
        pins["auditBand"] = f"tile{audit_tile}"
    if config.get("colourKey") is True:
        pins["colourKey"] = COLOUR_KEY_PIN
    gate = colour_residual if gated else None
    coloured = caption_coloured if gated else None
    return runner.Collaborators(plan, render_attempt, audit, gate, coloured, residual), pins


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.config.stat().st_size > MAX_CONFIG_BYTES:
        raise ValueError("Chunk configuration exceeds its byte budget")
    config = record(json.loads(arguments.config.read_text(encoding="utf-8")))
    network = mps.lazy_network(
        Path(str(config["detectorProgram"])), sha256=str(config["detectorProgramSHA256"])
    )
    region_pins: dict[str, str] = {}
    caption_detector: yolo.Detector | None = None
    caption_masks = yolo_admission = False
    if "regionDetector" in config:
        settings = yolo.settings_from(config["regionDetector"])
        caption_detector = yolo.load_detector(settings)
        caption_masks, yolo_admission = settings.masks, settings.admission
        masks = (":masks" if settings.masks else "") + (":admission" if settings.admission else "")
        region_pins = {"regionDetector": f"yolo:{settings.sha256}:{settings.imgsz}{masks}"}
    sttn_pins: dict[str, str] = {}
    painter: subtitle_sttn.Inpainter | None = None
    if "sttn" in config:
        settings_sttn = record(config["sttn"])
        painter = subtitle_sttn.Inpainter(
            Path(str(settings_sttn["directory"])),
            checkpoint_sha=str(settings_sttn["checkpointSHA256"]),
            architecture_sha=str(settings_sttn["architectureSHA256"]),
        )
        sttn_pins = {
            "sttn": str(settings_sttn["checkpointSHA256"]),
            "sttnAttention": subtitle_sttn.ATTENTION,
        }
    tools, pins = real_collaborators(
        config,
        network=network,
        caption_detector=caption_detector,
        caption_masks=caption_masks,
        yolo_admission=yolo_admission,
        sttn=painter,
    )
    receipt = runner.run(
        Path(str(config["output"])),
        tools,
        stride=int(str(config.get("stride", 6))),
        audit_stride=int(str(config.get("auditStride", runner.AUDIT_STRIDE))),
        pins=pins
        | {"detectorProgram": str(config["detectorProgramSHA256"])}
        | region_pins
        | sttn_pins,
    )
    print(json.dumps({"accepted": receipt["accepted"], "escalated": receipt["escalated"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
