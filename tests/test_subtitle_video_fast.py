"""Thread-parallel repair scheduling matches the original renderer (synthetic inputs only)."""

import hashlib
import subprocess
from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_inpaint import Image
from replay.subtitle_stream import SubtitleSpan, repair_stream
from replay.subtitle_video import VideoRepairJob, render_repaired_video, spatial_evidence
from replay.subtitle_video_fast import (
    Schedule,
    SupportStore,
    _decode_command,
    _encode_command,
    fast_repair_stream,
    render_refilled_video,
    render_repaired_video_fast,
)

SIZE = 64
SPAN = SubtitleSpan(2, 9, 8, 16, 40, 24)


def text_mask(roi: Image) -> Image:
    return np.asarray(roi[:, :, 2] > 200, dtype=np.uint8) * 255


def batch_mask(rois: tuple[Image, ...]) -> tuple[Image, ...]:
    return tuple(text_mask(roi) for roi in rois)


def synthetic_frames(count: int) -> list[Image]:
    rng = np.random.default_rng(7)
    frames: list[Image] = []
    for index in range(count):
        frame = rng.integers(0, 120, size=(SIZE, SIZE, 3), dtype=np.uint8)
        frame[20:28, 12 + index : 30 + index] = (30, 30, 250)
        frames.append(frame)
    return frames


def make_source(path: Path, frames: int) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=s={SIZE}x{SIZE}:r=30",
            "-frames:v",
            str(frames),
            "-c:v",
            "ffv1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def decoded_digest(path: Path) -> str:
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(raw).hexdigest()


def run_original(frames: list[Image], spans: tuple[SubtitleSpan, ...]) -> list[bytes]:
    written: list[bytes] = []
    repair_stream(
        iter(frames),
        spans=spans,
        expected_frames=len(frames),
        measure=spatial_evidence,
        write=lambda frame: written.append(frame.tobytes()),
        receipt=lambda index, span, result: None,
        quality_available=False,
        detect_batch=batch_mask,
    )
    return written


def run_fast(
    frames: list[Image], spans: tuple[SubtitleSpan, ...], schedule: Schedule
) -> tuple[list[bytes], list[int]]:
    written: list[bytes] = []
    seen: list[int] = []
    fast_repair_stream(
        iter(frames),
        spans=spans,
        expected_frames=len(frames),
        measure=spatial_evidence,
        write=lambda frame: written.append(frame.tobytes()),
        receipt=lambda index, span, result: seen.append(index),
        quality_available=False,
        detect_batch=batch_mask,
        schedule=schedule,
    )
    return written, seen


def test_parallel_stream_matches_original_bytes_and_receipt_order() -> None:
    frames = synthetic_frames(14)
    expected = run_original(frames, (SPAN,))
    written, seen = run_fast(frames, (SPAN,), Schedule(workers=4, window=5, detect_chunk=3))
    assert written == expected
    assert seen == [2, 3, 4, 5, 6, 7, 8]


def test_single_worker_and_large_window_match_original() -> None:
    frames = synthetic_frames(12)
    expected = run_original(frames, (SPAN,))
    written, seen = run_fast(frames, (SPAN,), Schedule(workers=1, window=32, detect_chunk=8))
    assert written == expected
    assert seen == [2, 3, 4, 5, 6, 7, 8]


def test_frames_without_spans_pass_through_unchanged() -> None:
    frames = synthetic_frames(6)
    written, seen = run_fast(frames, (), Schedule(workers=2, window=4, detect_chunk=2))
    assert written == [frame.tobytes() for frame in frames]
    assert seen == []


def test_evidence_sink_receives_ordered_frames() -> None:
    frames = synthetic_frames(8)
    events: list[tuple[int, int]] = []
    fast_repair_stream(
        iter(frames),
        spans=(SubtitleSpan(2, 8, 8, 16, 40, 24),),
        expected_frames=len(frames),
        measure=spatial_evidence,
        write=lambda frame: None,
        receipt=lambda index, span, result: None,
        quality_available=False,
        detect_batch=batch_mask,
        evidence=lambda index, span, roi, glyph, result: events.append(
            (index, int(glyph.sum() > 0))
        ),
        schedule=Schedule(workers=3, window=3, detect_chunk=2),
    )
    assert [index for index, _ in events] == [2, 3, 4, 5, 6, 7]


def test_single_roi_detector_is_supported() -> None:
    frames = synthetic_frames(10)
    written: list[bytes] = []
    fast_repair_stream(
        iter(frames),
        spans=(SPAN,),
        expected_frames=len(frames),
        measure=spatial_evidence,
        write=lambda frame: written.append(frame.tobytes()),
        receipt=lambda index, span, result: None,
        quality_available=False,
        detect=text_mask,
        schedule=Schedule(workers=2, window=4, detect_chunk=2),
    )
    assert written == run_original(frames, (SPAN,))


@pytest.mark.parametrize(
    "workers,window,chunk", [(0, 8, 1), (33, 8, 1), (2, 0, 1), (2, 129, 1), (2, 4, 0), (2, 4, 5)]
)
def test_schedule_rejects_invalid_values(workers: int, window: int, chunk: int) -> None:
    with pytest.raises(ValueError, match="Invalid"):
        Schedule(workers=workers, window=window, detect_chunk=chunk)


def test_detector_selection_is_exclusive_and_required() -> None:
    frames = synthetic_frames(10)
    arguments = {
        "spans": (SPAN,),
        "expected_frames": 10,
        "measure": spatial_evidence,
        "write": lambda frame: None,
        "receipt": lambda index, span, result: None,
        "quality_available": False,
    }
    with pytest.raises(ValueError, match="either a single or batch"):
        fast_repair_stream(iter(frames), detect=text_mask, detect_batch=batch_mask, **arguments)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="detector is required"):
        fast_repair_stream(iter(frames), **arguments)  # type: ignore[arg-type]


def test_overlapping_or_out_of_range_spans_are_rejected() -> None:
    frames = synthetic_frames(6)
    overlapping = (SubtitleSpan(0, 4, 8, 16, 40, 24), SubtitleSpan(3, 5, 8, 16, 40, 24))
    with pytest.raises(ValueError, match="Overlapping"):
        run_fast(frames, overlapping, Schedule())
    with pytest.raises(ValueError, match="out-of-range"):
        run_fast(frames, (SubtitleSpan(0, 7, 8, 16, 40, 24),), Schedule())


def test_decoder_excess_and_short_streams_are_rejected() -> None:
    frames = synthetic_frames(6)
    with pytest.raises(ValueError, match="excess"):
        fast_repair_stream(
            iter(frames),
            spans=(),
            expected_frames=4,
            measure=spatial_evidence,
            write=lambda frame: None,
            receipt=lambda index, span, result: None,
            detect_batch=batch_mask,
            schedule=Schedule(workers=1, window=8, detect_chunk=1),
        )
    with pytest.raises(ValueError, match="ended before"):
        fast_repair_stream(
            iter(frames[:3]),
            spans=(),
            expected_frames=5,
            measure=spatial_evidence,
            write=lambda frame: None,
            receipt=lambda index, span, result: None,
            detect_batch=batch_mask,
        )


def test_geometry_change_and_oversized_window_are_rejected() -> None:
    frames = synthetic_frames(4)
    frames[2] = np.zeros((SIZE, SIZE + 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="geometry"):
        run_fast(frames, (), Schedule(workers=1, window=4, detect_chunk=1))
    huge = [np.zeros((4096, 4096, 3), dtype=np.uint8)] * 2
    with pytest.raises(ValueError, match="memory budget"):
        fast_repair_stream(
            iter(huge),
            spans=(),
            expected_frames=2,
            measure=spatial_evidence,
            write=lambda frame: None,
            receipt=lambda index, span, result: None,
            detect_batch=batch_mask,
            schedule=Schedule(workers=1, window=16, detect_chunk=1),
        )


def test_detector_mask_count_mismatch_is_rejected() -> None:
    frames = synthetic_frames(10)
    with pytest.raises(ValueError, match="mask count"):
        fast_repair_stream(
            iter(frames),
            spans=(SPAN,),
            expected_frames=10,
            measure=spatial_evidence,
            write=lambda frame: None,
            receipt=lambda index, span, result: None,
            detect_batch=lambda rois: (),
            schedule=Schedule(workers=1, window=10, detect_chunk=6),
        )


def test_span_exceeding_canvas_and_non_bgr_frames_are_rejected() -> None:
    frames = synthetic_frames(4)
    with pytest.raises(ValueError, match="exceeds decoded canvas"):
        run_fast(frames, (SubtitleSpan(0, 3, 40, 16, 40, 24),), Schedule())
    gray = [np.zeros((SIZE, SIZE), dtype=np.uint8)] * 3
    with pytest.raises(ValueError, match="BGR"):
        fast_repair_stream(
            iter(gray),
            spans=(SubtitleSpan(0, 3, 8, 16, 40, 24),),
            expected_frames=3,
            measure=spatial_evidence,
            write=lambda frame: None,
            receipt=lambda index, span, result: None,
            detect_batch=batch_mask,
        )


def test_render_matches_original_video_and_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_source(source, 12)
    spans = (SubtitleSpan(2, 10, 8, 16, 40, 24),)
    original = tmp_path / "original.mp4"
    fast = tmp_path / "fast.mp4"
    first = render_repaired_video(
        VideoRepairJob(source, original, SIZE, SIZE, 12, spans), detect_batch=batch_mask
    )
    second = render_repaired_video_fast(
        VideoRepairJob(source, fast, SIZE, SIZE, 12, spans),
        detect_batch=batch_mask,
        schedule=Schedule(workers=3, window=4, detect_chunk=2),
    )
    assert decoded_digest(fast) == decoded_digest(original)
    assert (second.frames, second.softened_frames, second.attempts) == (
        first.frames,
        first.softened_frames,
        first.attempts,
    )
    assert (second.supported_pixels, second.unsupported_pixels) == (
        first.supported_pixels,
        first.unsupported_pixels,
    )
    assert (second.quality_measured, second.quality_passed) == (
        first.quality_measured,
        first.quality_passed,
    )


def test_render_with_measure_and_default_schedule_matches(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_source(source, 8)
    spans = (SubtitleSpan(1, 7, 8, 16, 40, 24),)
    original = tmp_path / "original.mp4"
    fast = tmp_path / "fast.mp4"
    render_repaired_video(
        VideoRepairJob(source, original, SIZE, SIZE, 8, spans),
        measure=spatial_evidence,
        detect_batch=batch_mask,
    )
    render_repaired_video_fast(
        VideoRepairJob(source, fast, SIZE, SIZE, 8, spans),
        measure=spatial_evidence,
        detect_batch=batch_mask,
    )
    assert decoded_digest(fast) == decoded_digest(original)


def test_render_refuses_missing_source_existing_output_and_temporal_context(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mkv"
    make_source(source, 4)
    output = tmp_path / "output.mp4"
    spans: tuple[SubtitleSpan, ...] = ()
    with pytest.raises(ValueError, match="Missing source"):
        render_repaired_video_fast(
            VideoRepairJob(tmp_path / "absent.mkv", output, SIZE, SIZE, 4, spans),
            detect_batch=batch_mask,
        )
    output.write_bytes(b"retained")
    with pytest.raises(FileExistsError):
        render_repaired_video_fast(
            VideoRepairJob(source, output, SIZE, SIZE, 4, spans), detect_batch=batch_mask
        )
    clock = VideoFrameClock((0, 1, 2, 3), Fraction(1, 30), 1)
    with pytest.raises(ValueError, match="original renderer"):
        render_repaired_video_fast(
            VideoRepairJob(
                source,
                tmp_path / "temporal.mp4",
                SIZE,
                SIZE,
                4,
                spans,
                context_frames=1,
                source_clock=clock,
            ),
            detect_batch=batch_mask,
        )


def test_decode_command_uses_source_clock_window() -> None:
    clock = VideoFrameClock((90, 91, 92, 93), Fraction(1, 30), 1)
    job = VideoRepairJob(Path("in.mkv"), Path("out.mp4"), SIZE, SIZE, 4, (), source_clock=clock)
    command = _decode_command(job)
    assert command[command.index("-ss") + 1] == "2"
    assert "-copyts" in command
    assert command[command.index("-v") + 1] == "info"
    assert command[command.index("-frames:v") + 1] == "4"


def test_decode_command_normalized_window_seeks_before_start() -> None:
    job = VideoRepairJob(
        Path("in.mkv"), Path("out.mp4"), SIZE, SIZE, 10, (), normalized_start_frame=100
    )
    command = _decode_command(job)
    assert command[command.index("-ss") + 1] == "2"
    assert "fps=30:start_time=0:eof_action=pass,trim=start_frame=40" in command


def test_decode_command_padded_tail_limits_frames_only() -> None:
    job = VideoRepairJob(
        Path("in.mkv"), Path("out.mp4"), SIZE, SIZE, 10, (), allow_padded_tail=True
    )
    command = _decode_command(job)
    assert "-ss" not in command
    assert command[command.index("-frames:v") + 1] == "10"


def test_encode_command_selects_hardware_and_quality_options() -> None:
    software = _encode_command(VideoRepairJob(Path("in.mkv"), Path("o.mp4"), SIZE, SIZE, 4, ()))
    hevc = _encode_command(
        VideoRepairJob(
            Path("in.mkv"),
            Path("o.mp4"),
            SIZE,
            SIZE,
            4,
            (),
            hardware_encode=True,
            hardware_hevc=True,
            hevc_quality=60,
        )
    )
    assert "libx264" in software
    assert software[software.index("-profile:v") + 1] == "high"
    assert hevc[hevc.index("-q:v") + 1] == "60"
    assert hevc[hevc.index("-profile:v") + 1] == "main"


def test_the_support_sink_records_exactly_the_repaired_pixels(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_source(source, 6)
    spans = (SubtitleSpan(1, 5, 8, 16, 40, 24),)
    store = SupportStore(SIZE, (16, 40))
    render_repaired_video_fast(
        VideoRepairJob(source, tmp_path / "fast.mp4", SIZE, SIZE, 6, spans),
        detect_batch=batch_mask,
        support_sink=store,
    )
    assert not (tmp_path / "fast.mp4").exists()
    assert len(store) == 6
    span = store.span()
    assert span is not None
    bands = store.bands(*span)
    assert not bands[0].any()
    assert bands[2].any()
    rows = np.nonzero(bands[2].any(axis=1))[0] + span[0]
    assert rows.min() >= 16 and rows.max() < 40
    assert len(list(store.sources(*span))) == 6
    assert list(store.sources(*span)) == []  # handed over once, then released


def test_a_store_without_repairs_has_no_span() -> None:
    store = SupportStore(8, (0, 8))
    store(np.zeros((8, 8), dtype=bool), np.zeros((8, 8, 3), dtype=np.uint8))
    assert store.span() is None


def test_a_repair_outside_the_kept_rows_is_refused() -> None:
    store = SupportStore(8, (2, 4))
    support = np.zeros((8, 8), dtype=bool)
    support[5, 1] = True
    with pytest.raises(ValueError, match="outside the kept source rows"):
        store(support, np.zeros((8, 8, 3), dtype=np.uint8))


def test_refill_replaces_only_supported_rows_and_keeps_the_frame_count(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_source(source, 6)
    store = SupportStore(SIZE, (10, 40))
    for index in range(6):
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        if index % 2:
            mask[20:30, 10:50] = True
        store(mask, np.full((SIZE, SIZE, 3), index, dtype=np.uint8))
    seen: list[tuple[int, int]] = []
    stored: list[int] = []

    def refill(bands: Iterator[Image], rows: tuple[int, int]) -> NDArray[np.uint8]:
        seen.append(rows)
        listed = list(bands)
        stored.extend(int(band[0, 0, 0]) for band in listed)
        assert listed[0].shape == (10, SIZE, 3)
        return np.stack([np.full_like(band, 255) for band in listed])

    output = tmp_path / "refilled.mp4"
    count = render_refilled_video(VideoRepairJob(source, output, SIZE, SIZE, 6, ()), store, refill)
    assert (count, seen, stored) == (6, [(20, 30)], [0, 1, 2, 3, 4, 5])
    assert output.stat().st_size > 0
    with pytest.raises(FileExistsError):
        render_refilled_video(VideoRepairJob(source, output, SIZE, SIZE, 6, ()), store, refill)


def test_refilled_bands_handed_over_in_order_encode_the_same_video(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_source(source, 6)
    outputs = []
    for streamed in (False, True):
        store = SupportStore(SIZE, (10, 40))
        for index in range(6):
            mask = np.zeros((SIZE, SIZE), dtype=bool)
            mask[20:30, 10:50] = True
            store(mask, np.full((SIZE, SIZE, 3), index, dtype=np.uint8))

        def refill(
            bands: Iterator[Image], rows: tuple[int, int], streamed: bool = streamed
        ) -> Iterable[Image]:
            filled = np.stack([np.full_like(band, 40 * i) for i, band in enumerate(bands)])
            return iter(list(filled)) if streamed else filled

        output = tmp_path / f"refilled-{streamed}.mp4"
        render_refilled_video(VideoRepairJob(source, output, SIZE, SIZE, 6, ()), store, refill)
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]


def test_refill_needs_one_support_per_frame_and_some_repair(tmp_path: Path) -> None:
    job = VideoRepairJob(tmp_path / "s.mkv", tmp_path / "o.mp4", SIZE, SIZE, 2, ())
    empty = SupportStore(SIZE, (0, 0))
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    empty(np.zeros((SIZE, SIZE), dtype=bool), frame)
    with pytest.raises(ValueError, match="One repair support per frame"):
        render_refilled_video(job, empty, lambda bands, rows: np.zeros(0, np.uint8))


def test_a_chunk_without_repaired_pixels_is_encoded_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    make_source(source, 3)
    empty = SupportStore(SIZE, (0, 0))
    for _ in range(3):
        empty(np.zeros((SIZE, SIZE), dtype=bool), np.zeros((SIZE, SIZE, 3), dtype=np.uint8))

    def refill(bands: Iterator[Image], rows: tuple[int, int]) -> NDArray[np.uint8]:
        raise AssertionError("nothing to refill")

    output = tmp_path / "unchanged.mp4"
    job = VideoRepairJob(source, output, SIZE, SIZE, 3, ())
    assert render_refilled_video(job, empty, refill) == 3
    assert output.stat().st_size > 0
