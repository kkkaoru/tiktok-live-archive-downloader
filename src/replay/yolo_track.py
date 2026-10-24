"""Track the avatar face and the burned-in caption boxes over a video with the trained detector.

Frames are decoded once by ffmpeg, scaled to the training width (so the detector sees the same
pixels it was trained on) and passed to the pinned checkpoint in batches. By default the video is
decoded by VideoToolbox (the media engine) and the detector runs on the GPU (MPS); the hardware
decode yields the same bytes as the software decoder, and decoding runs ahead of inference so the
two overlap. The result is a JSON
evidence file that names the checkpoint and the video by sha256 and holds, per sampled frame, the
single face box chosen by one fixed rule (highest confidence; the number of faces is kept so
ambiguity stays visible) and the caption box count. Boxes are reported in source pixels. The
evidence is detection output, not a measurement: callers decide where it may replace Vision.

With ``--boxes`` every frame is detected once and the sampled rows are taken from that same pass
(the same predictions, so the same rows), while ``yolo_boxes`` streams each frame's caption boxes
into range files that the source-caption cleanup reads instead of detecting its chunks again.
"""

import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from bisect import bisect_left
from collections.abc import Callable, Generator, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from replay import packet_clock
from replay.core import ReplayError
from replay.delivery_media import read_clock
from replay.mps_graph_cache import released_mps_graphs
from replay.yolo_labels import CAPTION, FACE
from replay.yolo_train import check_checkpoint, check_settings

# v2: every row carries the frame's presentation timestamp. v1 derived time as frame / rate,
# which drifts on variable-frame-rate recordings (45 s by the end of 10-07).
PROTOCOL = "yolo-face-caption-track-v2"
CONFIDENCE_FLOOR = 0.25
TRAIN_WIDTH = 640
MAX_BATCH = 64
FACE_RULE = "highest-confidence face box; faceCount keeps the number of faces above the floor"
DECODERS = ("videotoolbox", "software")
PREFETCH_FRAMES = 128


@dataclass(frozen=True, slots=True)
class Clock:
    """Presentation timestamps of the video stream in display order, and their time base."""

    pts: tuple[int, ...]
    time_base: Fraction

    def first_index(self, seconds: Fraction) -> int:
        """Index of the first frame shown at or after `seconds` (where an input seek lands)."""
        return bisect_left(self.pts, seconds / self.time_base)


@dataclass(frozen=True, slots=True)
class Row:
    frame: int
    face: tuple[float, float, float, float, float] | None
    face_count: int
    captions: int


class ResultLike(Protocol):
    @property
    def boxes(self) -> Any: ...


class ModelLike(Protocol):
    def predict(
        self,
        source: list[NDArray[np.uint8]],
        *,
        imgsz: int,
        device: str,
        conf: float,
        verbose: bool,
    ) -> Sequence[ResultLike]: ...


Sink = Callable[[Sequence[tuple[int, NDArray[np.uint8]]], Sequence[ResultLike]], None]


class Adapter:
    """Hand `track` only results of the expected class, refusing anything else."""

    def __init__(self, predictor: Any, expected: type) -> None:
        self._predictor = predictor
        self._expected = expected

    def predict(
        self,
        source: list[NDArray[np.uint8]],
        *,
        imgsz: int,
        device: str,
        conf: float,
        verbose: bool,
    ) -> Sequence[ResultLike]:
        found = list(
            self._predictor.predict(source, imgsz=imgsz, device=device, conf=conf, verbose=verbose)
        )
        if not found or not all(isinstance(item, self._expected) for item in found):
            raise ReplayError("Unexpected prediction type")
        return found


def row_from(frame: int, boxes: Any, *, scale: float) -> Row:
    """One frame's row from an Ultralytics `Boxes` (or None); boxes are scaled to source pixels."""
    if boxes is None:
        return Row(frame, None, 0, 0)
    labels, scores, corners = boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
    faces = [
        (float(score), [float(value) * scale for value in corner])
        for label, score, corner in zip(labels, scores, corners, strict=True)
        if int(label) == FACE
    ]
    captions = sum(1 for label in labels if int(label) == CAPTION)
    if not faces:
        return Row(frame, None, 0, captions)
    score, corner = max(faces, key=lambda item: item[0])
    return Row(frame, (corner[0], corner[1], corner[2], corner[3], score), len(faces), captions)


def track(
    frames: Iterator[tuple[int, NDArray[np.uint8]]],
    model: ModelLike,
    *,
    scale: float,
    batch: int,
    device: str,
    imgsz: int,
    sink: Sink | None = None,
    keep: Callable[[int], bool] = lambda _number: True,
    breaks: int | None = None,
) -> tuple[list[Row], float]:
    """Run the detector over `frames` in batches; return the rows and the seconds spent in it.

    ``sink`` sees every batch with its results; only frames ``keep`` accepts become rows. With
    ``breaks`` no batch spans a multiple of it (a cleanup chunk boundary), so every batch is
    composed exactly as a chunk's own detection composes it.
    """
    if not 1 <= batch <= MAX_BATCH:
        raise ReplayError("Invalid batch size")
    rows: list[Row] = []
    seconds = 0.0
    pending: list[tuple[int, NDArray[np.uint8]]] = []

    def flush() -> None:
        nonlocal seconds
        if not pending:
            return
        started = time.monotonic()
        results = model.predict(
            [image for _, image in pending],
            imgsz=imgsz,
            device=device,
            conf=CONFIDENCE_FLOOR,
            verbose=False,
        )
        seconds += time.monotonic() - started
        if len(results) != len(pending):
            raise ReplayError("Detector returned a different number of results")
        if sink is not None:
            sink(pending, results)
        rows.extend(
            row_from(number, result.boxes, scale=scale)
            for (number, _), result in zip(pending, results, strict=True)
            if keep(number)
        )
        pending.clear()

    for item in frames:
        if breaks is not None and pending and item[0] % breaks == 0:
            flush()
        pending.append(item)
        if len(pending) == batch:
            flush()
    flush()
    return rows, seconds


def video_frames(
    source: Path,
    *,
    width: int,
    height: int,
    start: int,
    end: int,
    step: int,
    rate: int | Fraction,
    ffmpeg: str,
    decoder: str = "software",
    decode_log: Path | None = None,
) -> Generator[tuple[int, NDArray[np.uint8]]]:
    """Decode frames start, start+step, ... below `end`, scaled to the training width (BGR).

    The input is sought to `start / rate` and the output is limited to the wanted frame count, so
    ffmpeg stops reading when the range is done instead of decoding to the end of the file. With
    ``videotoolbox`` the frames stay on the media engine until the sampled ones are downloaded;
    the scaling is the same software filter, so the pixels match the software decoder's.
    ``decode_log`` receives a ``showinfo`` line per decoded frame: the PTS actually decoded.
    """
    if not 0 <= start < end or step < 1 or rate < 1:
        raise ReplayError("Invalid frame range or step")
    if decoder not in DECODERS:
        raise ReplayError(f"Unknown decoder: {decoder}")
    scaled = scaled_height(width, height)
    count = (end - start + step - 1) // step
    hardware = decoder == "videotoolbox"
    accelerated = ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
    download = "hwdownload,format=nv12," if hardware else ""
    shown = "" if decode_log is None else "showinfo,"
    command = [
        ffmpeg, "-v", "error" if decode_log is None else "info", "-nostdin",
        *(accelerated if hardware else []),
        "-ss", f"{float(start / rate):.6f}", "-i", str(source),
        "-vf", f"{shown}select='not(mod(n,{step}))',{download}scale={TRAIN_WIDTH}:{scaled}",
        "-frames:v", str(count), "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]  # fmt: skip
    size = TRAIN_WIDTH * scaled * 3
    with (
        nullcontext(None) if decode_log is None else decode_log.open("wb") as log,
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            stderr=None if decode_log is None else log,
        ) as process,
    ):
        if process.stdout is None:
            raise ReplayError("ffmpeg gave no output stream")
        number = start
        while len(block := process.stdout.read(size)) == size:
            yield number, np.frombuffer(block, dtype=np.uint8).reshape(scaled, TRAIN_WIDTH, 3)
            number += step
    if process.returncode != 0:
        raise ReplayError("ffmpeg failed while decoding frames")


def prefetched[T](frames: Generator[T], depth: int = PREFETCH_FRAMES) -> Generator[T]:
    """Yield `frames` while a thread keeps decoding up to `depth` frames ahead.

    The detector then waits for the decoder only when decoding is the slower side. A decoder
    failure is raised where the frame it failed to produce would have been yielded; when the
    consumer stops early the decoder is closed, which ends its ffmpeg process.
    """
    if depth < 1:
        raise ReplayError("Invalid prefetch depth")
    # One decoded frame, the decoder's failure, or the end of the stream.
    buffer: queue.Queue[T | BaseException | None] = queue.Queue(depth)
    stop = threading.Event()

    def put(item: T | BaseException | None) -> bool:
        while not stop.is_set():
            try:
                buffer.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for item in frames:
                if not put(item):
                    return
            put(None)
        except BaseException as error:  # handed to the consumer, which raises it
            put(error)
        finally:
            frames.close()

    worker = threading.Thread(target=produce, name="yolo-decode", daemon=True)
    worker.start()
    try:
        while (item := buffer.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        worker.join()


def scaled_height(width: int, height: int) -> int:
    """The even height `scale=TRAIN_WIDTH:-2` gives, as in the training images."""
    if width < 1 or height < 1:
        raise ReplayError("Invalid video size")
    return 2 * round(height * TRAIN_WIDTH / width / 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def presentation_clock(source: Path, ffprobe: str) -> Clock:
    """Packet timestamps of the first video stream, sorted into display order (no decoding).

    Packets the edit list discards (flag ``D``) give no frame and are left out.
    """
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=time_base:packet=pts,flags", "-of", "json", str(source)],
        capture_output=True, text=True, check=True, timeout=600,
    ).stdout  # fmt: skip
    data = json.loads(out)
    pts = sorted(int(packet["pts"]) for packet in data["packets"] if "D" not in packet["flags"])
    if not pts or len(set(pts)) != len(pts):
        raise ReplayError("The video stream has no or duplicate presentation timestamps")
    return Clock(tuple(pts), Fraction(str(data["streams"][0]["time_base"])))


def timed_rows(rows: Sequence[Row], clock: Clock, *, start: Fraction, step: int) -> list[Row]:
    """Rows renumbered to decoded-frame indices: the seek lands on the first frame at `start`."""
    first = clock.first_index(start)
    renumbered = [replace(row, frame=first + number * step) for number, row in enumerate(rows)]
    if renumbered and renumbered[-1].frame >= len(clock.pts):
        raise ReplayError("More decoded frames than presentation timestamps")
    return renumbered


def document(
    rows: Sequence[Row],
    *,
    clock: Clock,
    weights: tuple[Path, str],
    source: tuple[Path, str],
    size: tuple[int, int],
    rate: int | Fraction,
    step: int,
    timing: dict[str, float],
    device: str = "mps",
    decoder: str = "software",
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL,
        "provenance": "yolo",
        "device": device,
        "decoder": decoder,
        "weights": {"path": str(weights[0]), "sha256": weights[1]},
        "source": {"path": str(source[0]), "sha256": source[1]},
        "width": size[0],
        "height": size[1],
        "rate": rate.numerator if rate.denominator == 1 else str(rate),
        "step": step,
        "timeBase": str(clock.time_base),
        "faceRule": FACE_RULE,
        "confidenceFloor": CONFIDENCE_FLOOR,
        "timing": timing,
        "frames": [
            {
                "frame": row.frame,
                "pts": clock.pts[row.frame],
                "face": list(row.face) if row.face is not None else None,
                "faceCount": row.face_count,
                "captions": row.captions,
            }
            for row in rows
        ],
    }


def probe_video(source: Path, ffprobe: str) -> tuple[int, int, Fraction]:
    """(width, height, nominal frame rate) of the first video stream; it may be fractional."""
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "json", str(source)],
        capture_output=True, text=True, check=True, timeout=60,
    ).stdout  # fmt: skip
    stream = json.loads(out)["streams"][0]
    numerator, denominator = (int(part) for part in str(stream["r_frame_rate"]).split("/"))
    if numerator < 1 or denominator < 1:
        raise ReplayError("The video has no frame rate")
    return int(stream["width"]), int(stream["height"]), Fraction(numerator, denominator)


def parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--config-directory", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--decoder", choices=DECODERS, default="videotoolbox")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--boxes", type=Path, help="also stream every frame's caption boxes here")
    parser.add_argument("--frame-clock", type=Path, help="the decoded clock the boxes are keyed by")
    parser.add_argument(
        "--chunk-frames", type=int, default=900, help="with --boxes: the cleanup chunk size"
    )
    arguments = parser.parse_args(argv)
    if (arguments.boxes is None) != (arguments.frame_clock is None):
        parser.error("--boxes and --frame-clock go together")
    return arguments


def decoded_clock(decode_log: Path, frame_clock: Path, *, end: int) -> None:
    """Refuse a packet clock whose first ``end`` PTS are not the ones the decoder output."""
    observed = [
        int(value)
        for value in re.findall(rb"\bn:\s*\d+\s+pts:\s*(-?\d+)\s", decode_log.read_bytes())
    ]
    expected = list(read_clock(frame_clock).pts[:end])
    if observed[: len(expected)] != expected:
        raise packet_clock.refuse(
            frame_clock,
            consumer="yolo-track",
            reason=f"the {len(observed)} decoded PTS differ from the clock's first {end}",
        )


def run(args: argparse.Namespace, model: ModelLike, sink: Sink | None = None) -> dict[str, object]:
    width, height, rate = probe_video(args.source, args.ffprobe)
    clock = presentation_clock(args.source, args.ffprobe)
    started = time.monotonic()
    # A sink needs every frame: decode them all and keep the sampled ones as rows.
    decoded = args.step if sink is None else 1
    # A clock taken from packets is checked against the PTS this pass decodes from frame 0.
    checked = sink is not None and args.start == 0 and packet_clock.is_packet(args.frame_clock)
    decode_log = args.output.with_name(args.output.name + ".decode.log") if checked else None
    rows, inference = track(
        prefetched(video_frames(
            args.source, width=width, height=height, start=args.start, end=args.end,
            step=decoded, rate=rate, ffmpeg=args.ffmpeg, decoder=args.decoder,
            decode_log=decode_log,
        )),
        model, scale=width / TRAIN_WIDTH, batch=args.batch, device=args.device, imgsz=args.imgsz,
        sink=sink, keep=lambda number: (number - args.start) % args.step == 0,
        breaks=None if sink is None else args.chunk_frames,
    )  # fmt: skip
    if decode_log is not None:
        decoded_clock(decode_log, args.frame_clock, end=args.end)
    wall = time.monotonic() - started
    timing = {
        "wallSeconds": wall,
        "inferenceSeconds": inference,
        "decodeAndOtherSeconds": wall - inference,
        "framesPerSecond": len(rows) / wall if wall > 0 else 0.0,
    }
    return document(
        timed_rows(rows, clock, start=args.start / rate, step=args.step),
        clock=clock,
        weights=(args.weights, args.weights_sha256),
        source=(args.source, sha256(args.source)),
        size=(width, height),
        rate=rate,
        step=args.step,
        timing=timing,
        device=args.device,
        decoder=args.decoder,
    )


def require_mps() -> None:
    """Refuse to start on the GPU path when Metal is unavailable instead of running elsewhere."""
    import torch

    if not torch.backends.mps.is_available():
        raise ReplayError("The MPS (GPU) device is not available")


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite the output")
    check_checkpoint(args.weights, args.weights_sha256)
    check_settings(args.config_directory, args.work)
    os.environ["YOLO_CONFIG_DIR"] = str(args.config_directory)
    os.environ["ULTRALYTICS_OFFLINE"] = "1"
    if args.device == "mps":
        require_mps()
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.model import YOLO

    sink = None
    if args.boxes is not None:
        # Deferred: yolo_boxes builds on subtitle_yolo_regions, which imports this module.
        from replay import yolo_boxes

        sink = yolo_boxes.Writer(
            args.boxes, clock_path=args.frame_clock, source=args.source,
            weights_sha256=args.weights_sha256, imgsz=args.imgsz, ffprobe=args.ffprobe,
        )  # fmt: skip
    with released_mps_graphs():
        result = run(args, Adapter(YOLO(str(args.weights)), Results), sink)
    if sink is not None:
        sink.close()
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["timing"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
