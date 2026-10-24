"""Face/caption tracking: the face rule, batching, scaling, the decoded frame stream and the CLI."""

import argparse
import json
import shutil
import subprocess
import sys
import types
from collections.abc import Generator, Iterator, Sequence
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import packet_clock
from replay import yolo_track as tracking
from replay.core import ReplayError
from replay.yolo_track import Row
from replay.yolo_train import REQUIRED_OFF


class Tensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def tolist(self) -> list[object]:
        return self.values


class Boxes:
    def __init__(self, items: list[tuple[int, float, list[float]]]) -> None:
        self.cls = Tensor([item[0] for item in items])
        self.conf = Tensor([item[1] for item in items])
        self.xyxy = Tensor([item[2] for item in items])


class Result:
    def __init__(self, boxes: Boxes | None) -> None:
        self.boxes = boxes


class FakeModel:
    """Answers each image by the value of its first pixel: 0 = nothing, 1 = one face."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def predict(
        self,
        source: list[NDArray[np.uint8]],
        *,
        imgsz: int,
        device: str,
        conf: float,
        verbose: bool,
    ) -> Sequence[Result]:
        self.batches.append(len(source))
        return [
            Result(Boxes([(0, 0.9, [10.0, 20.0, 30.0, 40.0])]) if int(image[0, 0, 0]) else None)
            for image in source
        ]


def image(value: int) -> NDArray[np.uint8]:
    return np.full((4, 4, 3), value, dtype=np.uint8)


def frames(values: list[int]) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    return iter([(index * 30, image(value)) for index, value in enumerate(values)])


def test_no_boxes_gives_an_empty_row() -> None:
    assert tracking.row_from(7, None, scale=1.0) == Row(7, None, 0, 0)


def test_only_captions_gives_a_row_without_a_face() -> None:
    boxes = Boxes([(1, 0.8, [0.0, 0.0, 1.0, 1.0]), (1, 0.6, [0.0, 0.0, 1.0, 1.0])])
    assert tracking.row_from(3, boxes, scale=1.0) == Row(3, None, 0, 2)


def test_the_highest_confidence_face_is_chosen_and_scaled_to_source_pixels() -> None:
    boxes = Boxes(
        [
            (0, 0.4, [1.0, 2.0, 3.0, 4.0]),
            (0, 0.9, [10.0, 20.0, 30.0, 40.0]),
            (1, 0.7, [0.0, 0.0, 5.0, 5.0]),
        ]
    )
    assert tracking.row_from(9, boxes, scale=1.6875) == Row(
        9, (16.875, 33.75, 50.625, 67.5, 0.9), 2, 1
    )


def test_batches_are_filled_in_order_and_the_remainder_is_flushed() -> None:
    model = FakeModel()
    rows, seconds = tracking.track(
        frames([1, 0, 1, 1, 0]), model, scale=2.0, batch=2, device="cpu", imgsz=640
    )
    assert model.batches == [2, 2, 1]
    assert [row.frame for row in rows] == [0, 30, 60, 90, 120]
    assert [row.face is not None for row in rows] == [True, False, True, True, False]
    assert rows[0].face == (20.0, 40.0, 60.0, 80.0, 0.9)
    assert seconds >= 0.0


def test_no_frames_gives_no_rows_and_no_prediction() -> None:
    model = FakeModel()
    assert tracking.track(iter([]), model, scale=1.0, batch=4, device="cpu", imgsz=640)[0] == []
    assert model.batches == []


def test_an_invalid_batch_is_refused() -> None:
    with pytest.raises(ReplayError, match="Invalid batch size"):
        tracking.track(iter([]), FakeModel(), scale=1.0, batch=0, device="cpu", imgsz=640)
    with pytest.raises(ReplayError, match="Invalid batch size"):
        tracking.track(iter([]), FakeModel(), scale=1.0, batch=65, device="cpu", imgsz=640)


def test_a_detector_that_drops_a_result_is_refused() -> None:
    class Short(FakeModel):
        def predict(
            self,
            source: list[NDArray[np.uint8]],
            *,
            imgsz: int,
            device: str,
            conf: float,
            verbose: bool,
        ) -> Sequence[Result]:
            return [Result(None)]

    with pytest.raises(ReplayError, match="different number of results"):
        tracking.track(frames([1, 1]), Short(), scale=1.0, batch=2, device="cpu", imgsz=640)


def test_the_adapter_returns_results_of_the_expected_class_only() -> None:
    class Predictor:
        def __init__(self, items: list[object]) -> None:
            self.items = items

        def predict(self, source: object, **options: object) -> list[object]:
            return self.items

    good = tracking.Adapter(Predictor([Result(None)]), Result)
    assert len(good.predict([image(0)], imgsz=640, device="cpu", conf=0.25, verbose=False)) == 1
    with pytest.raises(ReplayError, match="Unexpected prediction type"):
        tracking.Adapter(Predictor(["x"]), Result).predict(
            [image(0)], imgsz=640, device="cpu", conf=0.25, verbose=False
        )
    with pytest.raises(ReplayError, match="Unexpected prediction type"):
        tracking.Adapter(Predictor([]), Result).predict(
            [image(0)], imgsz=640, device="cpu", conf=0.25, verbose=False
        )


def test_the_scaled_height_matches_the_training_images() -> None:
    assert tracking.scaled_height(1080, 1920) == 1138
    assert tracking.scaled_height(640, 480) == 480
    with pytest.raises(ReplayError, match="Invalid video size"):
        tracking.scaled_height(0, 10)


def test_the_document_names_weights_video_rule_and_timing() -> None:
    rows = [Row(0, (1.0, 2.0, 3.0, 4.0, 0.5), 1, 2), Row(30, None, 0, 0)]
    clock = tracking.Clock(tuple(range(0, 3000 * 40, 3000)), Fraction(1, 90000))
    result = tracking.document(
        rows,
        clock=clock,
        weights=(Path("w.pt"), "aa"),
        source=(Path("v.mp4"), "bb"),
        size=(1080, 1920),
        rate=30,
        step=30,
        timing={"wallSeconds": 2.0},
    )
    assert result["protocol"] == "yolo-face-caption-track-v2"
    assert result["timeBase"] == "1/90000"
    assert result["provenance"] == "yolo"
    assert result["weights"] == {"path": "w.pt", "sha256": "aa"}
    assert result["source"] == {"path": "v.mp4", "sha256": "bb"}
    assert result["frames"] == [
        {"frame": 0, "pts": 0, "face": [1.0, 2.0, 3.0, 4.0, 0.5], "faceCount": 1, "captions": 2},
        {"frame": 30, "pts": 90000, "face": None, "faceCount": 0, "captions": 0},
    ]


def test_rows_are_renumbered_from_the_frame_the_seek_lands_on() -> None:
    # A variable-rate stream: frame 2 is shown late, so a seek to 0.1 s lands on frame 2.
    clock = tracking.Clock((0, 3000, 30000, 33000, 36000), Fraction(1, 90000))
    rows = [Row(3, None, 0, 0), Row(5, None, 0, 0)]
    assert [
        row.frame for row in tracking.timed_rows(rows, clock, start=Fraction(1, 10), step=2)
    ] == [
        2,
        4,
    ]
    with pytest.raises(ReplayError, match="More decoded frames"):
        tracking.timed_rows(rows, clock, start=Fraction(1, 10), step=3)


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe is required")
def test_the_presentation_clock_reads_sorted_packet_timestamps(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
         "-pix_fmt", "yuv420p", "-video_track_timescale", "90000", str(video)],
        check=True,
    )  # fmt: skip
    clock = tracking.presentation_clock(video, "ffprobe")
    assert clock.time_base == Fraction(1, 90000)
    assert clock.pts == tuple(range(0, 90000, 9000))
    assert clock.first_index(Fraction(1, 2)) == 5


def test_the_presentation_clock_leaves_out_packets_the_edit_list_discards(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=3",
         "-pix_fmt", "yuv420p", "-g", "30", str(video)],
        check=True,
    )  # fmt: skip
    trimmed = tmp_path / "trimmed.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "0.45", "-i", str(video), "-c", "copy", str(trimmed)],
        check=True,
    )  # fmt: skip
    clock = tracking.presentation_clock(trimmed, "ffprobe")
    assert clock.time_base == Fraction(1, 10240)
    assert clock.pts == tuple(range(0, 25600, 1024))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for the stream test")
def test_the_decoded_stream_samples_the_requested_frames_at_the_training_width(
    tmp_path: Path,
) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x480:rate=10:duration=3",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    got = list(
        tracking.video_frames(
            video, width=320, height=480, start=4, end=25, step=5, rate=10, ffmpeg="ffmpeg"
        )
    )
    assert [number for number, _ in got] == [4, 9, 14, 19, 24]
    assert got[0][1].shape == (960, 640, 3)
    assert tracking.probe_video(video, "ffprobe") == (320, 480, Fraction(10))


def test_an_invalid_range_or_step_is_refused() -> None:
    with pytest.raises(ReplayError, match="Invalid frame range or step"):
        next(
            tracking.video_frames(
                Path("x.mp4"), width=10, height=10, start=5, end=5, step=1, rate=10, ffmpeg="ffmpeg"
            )
        )
    with pytest.raises(ReplayError, match="Invalid frame range or step"):
        next(
            tracking.video_frames(
                Path("x.mp4"), width=10, height=10, start=0, end=5, step=0, rate=10, ffmpeg="ffmpeg"
            )
        )


def test_a_failing_decoder_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="ffmpeg failed"):
        list(
            tracking.video_frames(
                tmp_path / "missing.mp4",
                width=10,
                height=10,
                start=0,
                end=5,
                step=1,
                rate=10,
                ffmpeg="ffmpeg",
            )
        )


def test_a_fractional_frame_rate_is_kept_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Done:
        stdout = json.dumps({"streams": [{"width": 4, "height": 4, "r_frame_rate": "301/12"}]})

    monkeypatch.setattr(subprocess, "run", lambda *args, **options: Done())
    assert tracking.probe_video(tmp_path / "x.mp4", "ffprobe") == (4, 4, Fraction(301, 12))


def test_a_video_without_a_frame_rate_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Done:
        stdout = json.dumps({"streams": [{"width": 4, "height": 4, "r_frame_rate": "0/0"}]})

    monkeypatch.setattr(subprocess, "run", lambda *args, **options: Done())
    with pytest.raises(ReplayError, match="no frame rate"):
        tracking.probe_video(tmp_path / "x.mp4", "ffprobe")


def test_the_sha256_of_a_file_is_its_content_digest(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    assert tracking.sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for the run test")
def test_run_and_main_write_evidence_for_a_synthetic_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x480:rate=10:duration=2",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    settings = tmp_path / "config" / "Ultralytics"
    settings.mkdir(parents=True)
    work = tmp_path.resolve()
    (settings / "settings.json").write_text(
        json.dumps(
            {name: False for name in REQUIRED_OFF}
            | {"datasets_dir": str(work / "d"), "runs_dir": str(work / "r")}
        ),
        encoding="utf-8",
    )

    class Predictor:
        def __init__(self, path: str) -> None:
            self.path = path

        def predict(self, source: list[NDArray[np.uint8]], **options: object) -> list[Result]:
            return [Result(Boxes([(0, 0.8, [1.0, 2.0, 3.0, 4.0])])) for _ in source]

    results_module = types.ModuleType("ultralytics.engine.results")
    results_module.__dict__["Results"] = Result
    model_module = types.ModuleType("ultralytics.models.yolo.model")
    model_module.__dict__["YOLO"] = Predictor
    for name, module in (
        ("ultralytics", types.ModuleType("ultralytics")),
        ("ultralytics.engine", types.ModuleType("ultralytics.engine")),
        ("ultralytics.engine.results", results_module),
        ("ultralytics.models", types.ModuleType("ultralytics.models")),
        ("ultralytics.models.yolo", types.ModuleType("ultralytics.models.yolo")),
        ("ultralytics.models.yolo.model", model_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    output = tmp_path / "out.json"
    code = tracking.main(
        [str(video), str(output), "--weights", str(weights), "--weights-sha256",
         tracking.sha256(weights), "--config-directory", str(tmp_path / "config"),
         "--work", str(work), "--end", "20", "--step", "10", "--batch", "2", "--device", "cpu"]
    )  # fmt: skip
    written = json.loads(output.read_text(encoding="utf-8"))
    assert code == 0
    assert [item["frame"] for item in written["frames"]] == [0, 10]
    assert written["frames"][0]["face"] == [0.5, 1.0, 1.5, 2.0, 0.8]
    assert written["rate"] == 10
    assert written["step"] == 10
    assert (written["device"], written["decoder"]) == ("cpu", "videotoolbox")
    assert set(written["timing"]) == {
        "wallSeconds", "inferenceSeconds", "decodeAndOtherSeconds", "framesPerSecond",
    }  # fmt: skip
    clock = tmp_path / "video-clock.json"
    frames = [{"pts": 9000 * index} for index in range(20)]
    clock.write_text(json.dumps({"frames": frames, "streams": [{"time_base": "1/90000"}]}))
    boxed = tmp_path / "boxed.json"
    assert tracking.main(
        [str(video), str(boxed), "--weights", str(weights), "--weights-sha256",
         tracking.sha256(weights), "--config-directory", str(tmp_path / "config"),
         "--work", str(work), "--end", "20", "--step", "10", "--batch", "2", "--device", "cpu",
         "--boxes", str(tmp_path / "boxes"), "--frame-clock", str(clock)]
    ) == 0  # fmt: skip
    # Every frame was detected, yet the rows are those of the sampled pass.
    assert json.loads(boxed.read_text(encoding="utf-8"))["frames"] == written["frames"]
    assert json.loads((tmp_path / "boxes/complete.json").read_text())["frames"] == 20


def test_a_packet_clock_is_checked_against_the_decoded_pts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x480:rate=10:duration=2",
         "-pix_fmt", "yuv420p", "-bf", "2", str(video)],
        check=True,
    )  # fmt: skip
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"weights")
    settings = tmp_path / "config" / "Ultralytics"
    settings.mkdir(parents=True)
    work = tmp_path.resolve()
    (settings / "settings.json").write_text(
        json.dumps(
            {name: False for name in REQUIRED_OFF}
            | {"datasets_dir": str(work / "d"), "runs_dir": str(work / "r")}
        ),
        encoding="utf-8",
    )

    class Predictor:
        def __init__(self, path: str) -> None:
            self.path = path

        def predict(self, source: list[NDArray[np.uint8]], **options: object) -> list[Result]:
            return [Result(Boxes([])) for _ in source]

    results_module = types.ModuleType("ultralytics.engine.results")
    results_module.__dict__["Results"] = Result
    model_module = types.ModuleType("ultralytics.models.yolo.model")
    model_module.__dict__["YOLO"] = Predictor
    for name, module in (
        ("ultralytics", types.ModuleType("ultralytics")),
        ("ultralytics.engine", types.ModuleType("ultralytics.engine")),
        ("ultralytics.engine.results", results_module),
        ("ultralytics.models", types.ModuleType("ultralytics.models")),
        ("ultralytics.models.yolo", types.ModuleType("ultralytics.models.yolo")),
        ("ultralytics.models.yolo.model", model_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    clock = tmp_path / "work" / "video-clock.json"
    clock.parent.mkdir()
    assert packet_clock.produce(video, clock, ffprobe="ffprobe", ffmpeg="ffmpeg") == "packet-v1"
    arguments = [
        "--weights", str(weights), "--weights-sha256", tracking.sha256(weights),
        "--config-directory", str(tmp_path / "config"), "--work", str(work), "--end", "20",
        "--step", "10", "--batch", "2", "--device", "cpu", "--frame-clock", str(clock),
    ]  # fmt: skip
    output = tmp_path / "out.json"
    assert tracking.main([str(video), str(output), *arguments, "--boxes", str(tmp_path / "a")]) == 0
    assert (tmp_path / "out.json.decode.log").read_bytes().count(b" pts:") == 20
    data = json.loads(clock.read_text(encoding="utf-8"))
    data["frames"][3]["pts"] += 1
    clock.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="DECODED_CLOCK_MISMATCH yolo-track"):
        tracking.main(
            [str(video), str(tmp_path / "again.json"), *arguments, "--boxes", str(tmp_path / "b")]
        )
    assert packet_clock.refused(clock) == "the 20 decoded PTS differ from the clock's first 20"


def test_parse_defaults(tmp_path: Path) -> None:
    args = tracking.parse(
        [str(tmp_path / "v.mp4"), str(tmp_path / "o.json"), "--weights", "w",
         "--weights-sha256", "x", "--config-directory", "c", "--work", "w", "--end", "5",
         "--step", "1"]
    )  # fmt: skip
    assert isinstance(args, argparse.Namespace)
    assert (args.start, args.batch, args.device, args.imgsz) == (0, 16, "mps", 640)
    assert args.decoder == "videotoolbox"


def test_main_refuses_to_overwrite_before_loading_anything(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        tracking.main(
            [str(tmp_path / "v.mp4"), str(output), "--weights", "w", "--weights-sha256", "x",
             "--config-directory", str(tmp_path), "--work", str(tmp_path), "--end", "10",
             "--step", "5"]
        )  # fmt: skip


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for the stream test")
def test_the_hardware_decoder_gives_the_software_decoders_pixels(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x480:rate=10:duration=2",
         "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )  # fmt: skip
    decoded = {
        decoder: list(
            tracking.video_frames(
                video,
                width=320,
                height=480,
                start=2,
                end=20,
                step=3,
                rate=10,
                ffmpeg="ffmpeg",
                decoder=decoder,
            )
        )  # fmt: skip
        for decoder in tracking.DECODERS
    }
    hardware, software = decoded["videotoolbox"], decoded["software"]
    assert [n for n, _ in hardware] == [n for n, _ in software] == [2, 5, 8, 11, 14, 17]
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(hardware, software, strict=True))


def test_an_unknown_decoder_is_refused() -> None:
    with pytest.raises(ReplayError, match="Unknown decoder: gpu"):
        next(
            tracking.video_frames(
                Path("x.mp4"), width=10, height=10, start=0, end=5, step=1, rate=10,
                ffmpeg="ffmpeg", decoder="gpu",
            )
        )  # fmt: skip


def numbered(count: int, closed: list[bool]) -> Generator[tuple[int, NDArray[np.uint8]]]:
    try:
        for number in range(count):
            yield number, np.zeros((1, 1, 3), dtype=np.uint8)
    finally:
        closed.append(True)


def test_prefetching_keeps_the_order_and_closes_the_decoder() -> None:
    closed: list[bool] = []
    frames = numbered(10, closed)
    assert [n for n, _ in tracking.prefetched(frames, depth=2)] == list(range(10))
    assert closed == [True]


def test_stopping_early_closes_the_decoder() -> None:
    closed: list[bool] = []
    stream = tracking.prefetched(numbered(1000, closed), depth=1)
    assert next(stream)[0] == 0
    stream.close()
    assert closed == [True]


def test_a_decoder_failure_reaches_the_consumer() -> None:
    def failing() -> Generator[tuple[int, NDArray[np.uint8]]]:
        yield 0, np.zeros((1, 1, 3), dtype=np.uint8)
        raise ReplayError("ffmpeg failed while decoding frames")

    stream = tracking.prefetched(failing(), depth=4)
    assert next(stream)[0] == 0
    with pytest.raises(ReplayError, match="ffmpeg failed"):
        next(stream)


def test_an_invalid_prefetch_depth_is_refused() -> None:
    with pytest.raises(ReplayError, match="prefetch depth"):
        next(tracking.prefetched(numbered(1, []), depth=0))


@pytest.mark.parametrize("available", [True, False])
def test_the_gpu_path_requires_mps(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    torch = types.ModuleType("torch")
    torch.__dict__["backends"] = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: available)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    if available:
        tracking.require_mps()
    else:
        with pytest.raises(ReplayError, match="MPS"):
            tracking.require_mps()


def test_a_sink_sees_every_frame_while_only_kept_frames_become_rows() -> None:
    seen: list[list[int]] = []

    def sink(batch: Sequence[tuple[int, NDArray[np.uint8]]], results: Sequence[object]) -> None:
        seen.append([number for number, _ in batch])
        assert len(results) == len(batch)

    rows, _ = tracking.track(
        iter([(number, image(1)) for number in range(5)]), FakeModel(), scale=1.0, batch=2,
        device="cpu", imgsz=640, sink=sink, keep=lambda number: number % 2 == 0,
    )  # fmt: skip
    assert seen == [[0, 1], [2, 3], [4]]
    assert [row.frame for row in rows] == [0, 2, 4]


def test_boxes_need_the_frame_clock(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        tracking.parse(
            [str(tmp_path / "v.mp4"), str(tmp_path / "o.json"), "--weights", "w",
             "--weights-sha256", "x", "--config-directory", "c", "--work", "w", "--end", "5",
             "--step", "1", "--boxes", str(tmp_path / "b")]
        )  # fmt: skip


def test_breaks_keep_every_batch_inside_one_chunk() -> None:
    model = FakeModel()
    tracking.track(
        iter([(number, image(1)) for number in range(7)]), model, scale=1.0, batch=2,
        device="cpu", imgsz=640, breaks=3,
    )  # fmt: skip
    assert model.batches == [2, 1, 2, 1, 1]
