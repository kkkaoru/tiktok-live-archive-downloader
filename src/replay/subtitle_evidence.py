"""Opt-in bounded pre-encode evidence, emitted by the actual repair writer.

This captures observations, not quality approval or reconstruction of old media.
The candidate receipt binds this manifest only after successful encoding.
"""

import hashlib
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image
from replay.subtitle_repair import RepairedFrame
from replay.subtitle_stream import SubtitleSpan

MAX_EVIDENCE_BYTES = 128 * 1024 * 1024


class RepairCapture:
    """Save at most eight explicitly selected active frames without overwrites."""

    def __init__(self, directory: Path, frames: object, clock: VideoFrameClock) -> None:
        if (
            not isinstance(frames, list)
            or not 1 <= len(frames) <= 8
            or any(type(frame) is not int or not 0 <= frame < len(clock.pts) for frame in frames)
        ):
            raise ValueError("Evidence requires one to eight valid frame indices")
        selected = tuple(int(frame) for frame in frames)
        if selected != tuple(sorted(set(selected))):
            raise ValueError("Evidence frames must be unique and ordered")
        if directory != directory.resolve() or not directory.is_relative_to(Path.cwd().resolve()):
            raise ValueError("Evidence directory must be checkout-local without symlinks")
        directory.mkdir(mode=0o700)
        self.directory = directory
        self.frames = selected
        self.clock = clock
        self.records: list[dict[str, object]] = []
        self.bytes = 0

    def __call__(
        self, frame: int, span: SubtitleSpan, source: Image, glyph: Image, result: RepairedFrame
    ) -> None:
        if frame not in self.frames:
            return
        if (
            len(self.records) >= len(self.frames)
            or frame != self.frames[len(self.records)]
            or source.dtype != np.uint8
            or source.shape != (span.height, span.width, 3)
            or result.image.dtype != np.uint8
            or result.image.shape != source.shape
            or glyph.dtype != np.uint8
            or glyph.shape != source.shape[:2]
            or result.support.dtype != np.uint8
            or result.support.shape != glyph.shape
            or (not result.attempts and result.unsupported is None)
            or (
                result.unsupported is not None
                and (
                    result.unsupported.dtype != np.uint8 or result.unsupported.shape != glyph.shape
                )
            )
        ):
            raise ValueError("Invalid writer evidence geometry or order")
        self.bytes += source.nbytes + result.image.nbytes + glyph.nbytes + result.support.nbytes
        if result.unsupported is not None:
            self.bytes += result.unsupported.nbytes
        if self.bytes > MAX_EVIDENCE_BYTES:
            raise ValueError("Writer evidence byte budget exceeded")
        path = self.directory / f"{frame:08d}.npz"
        with path.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            arrays = dict(source=source, candidate=result.image, glyph=glyph, repair=result.support)
            if result.unsupported is not None:
                arrays["unsupported"] = result.unsupported
            np.savez(stream, allow_pickle=False, **arrays)
        with path.open("rb") as stream:
            pin = hashlib.file_digest(stream, "sha256").hexdigest()
        self.records.append(
            {
                "frame": frame,
                "sourcePTS": self.clock.pts[frame],
                "candidatePTS": self.clock.pts[frame] - self.clock.pts[0],
                "timeBase": str(self.clock.time_base),
                "region": asdict(span),
                "path": str(path),
                "sha256": pin,
                "bytes": path.stat().st_size,
                "methods": [attempt.method.value for attempt in result.attempts],
                "glyphPixels": int(np.count_nonzero(glyph)),
                "repairPixels": int(np.count_nonzero(result.support)),
                "unsupportedPixels": (
                    int(np.count_nonzero(result.unsupported))
                    if result.unsupported is not None
                    else None
                ),
                "registeredCandidate": result.unsupported is not None,
            }
        )

    def manifest(self) -> dict[str, object]:
        if tuple(row["frame"] for row in self.records) != self.frames:
            raise ValueError("Selected evidence frames were not all written active frames")
        for row in self.records:
            with Path(str(row["path"])).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                    raise ValueError("Writer evidence changed before candidate completion")
        return {
            "version": 1,
            "kind": "source-mask-writer-evidence",
            "pixelFormat": "bgr24",
            "producerPolicy": (
                "registered-original-donors-candidate-v1"
                if any(row["registeredCandidate"] for row in self.records)
                else "subtitle-repair-with-retries-v1"
            ),
            "capturePoint": "after-encoder-pipe-write-before-lossy-encode",
            "frames": self.records,
            "temporalRegisteredSupport": None,
            "readableObservations": None,
            "qualityApproval": False,
        }
