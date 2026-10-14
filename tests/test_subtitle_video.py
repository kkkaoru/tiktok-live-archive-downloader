"""Real synthetic FFmpeg streaming tests; no private media or network."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from replay.subtitle_clock import VideoFrameClock
from replay.subtitle_evidence import RepairCapture
from replay.subtitle_inpaint import Image
from replay.subtitle_retry import RepairEvidence
from replay.subtitle_stream import SubtitleSpan, repair_stream
from replay.subtitle_video import (
    VideoRepairJob,
    frame_observation_times,
    observed_frame_indices,
    render_repaired_video,
    spatial_evidence,
)


@pytest.fixture(autouse=True)
def synthetic_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    def detect(image: Image) -> Image:
        return np.asarray(image[:, :, 2] != image[:, :, 1], dtype=np.uint8) * 255

    monkeypatch.setattr(
        "replay.subtitle_stream.vision_text_masks", lambda images: tuple(map(detect, images))
    )


@pytest.mark.parametrize("width,frames,fps", [(0, 1, 30), (4097, 1, 30), (16, 1, 121)])
def test_invalid_video_budget(width: int, frames: int, fps: int) -> None:
    with pytest.raises(ValueError):
        VideoRepairJob(Path("source"), Path("output"), width, 4096, frames, (), fps)


@pytest.mark.parametrize("fault", ["outside", "existing", "alias", "missing-parent"])
def test_explicit_decode_log_requires_new_local_canonical_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    log = checkout / "decode.log"
    if fault == "outside":
        log = tmp_path / "decode.log"
    elif fault == "existing":
        log.write_text("retained", encoding="utf-8")
    elif fault == "alias":
        alias = checkout / "alias"
        alias.symlink_to(checkout, target_is_directory=True)
        log = alias / "decode.log"
    else:
        log = checkout / "missing/decode.log"
    with pytest.raises(ValueError, match="decode log"):
        VideoRepairJob(checkout / "source", tmp_path / "media.mp4", 16, 16, 1, (), decode_log=log)


def test_external_media_stream_keeps_actual_decoder_log_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    source = checkout / "source.mkv"
    media = tmp_path / "media.mp4"
    log = checkout / "decode.log"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=16x16:r=1",
            "-frames:v",
            "2",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    result = render_repaired_video(
        VideoRepairJob(
            source,
            media,
            16,
            16,
            2,
            (),
            source_clock=VideoFrameClock((0, 1000), Fraction(1, 1000), 1000),
            decode_log=log,
        )
    )
    assert result.frames == 2
    assert media.is_file()
    assert log.is_file()
    assert not media.with_suffix(".mp4.decode.log").exists()


def test_frame_requests_are_interior_local_clocks() -> None:
    assert frame_observation_times(frames=3, fps=30) == (
        0.008333333333333333,
        0.041666666666666664,
        0.075,
    )


@pytest.mark.parametrize("frames,fps", [(0, 30), (50001, 30), (1, 0), (1, 121), (True, 30)])
def test_invalid_frame_sampling_budget(frames: int, fps: int) -> None:
    with pytest.raises(ValueError, match="sampling"):
        frame_observation_times(frames=frames, fps=fps)


def test_duplicate_actual_frames_do_not_prove_complete_coverage() -> None:
    assert observed_frame_indices((0, 0, 1 / 30, 3 / 30), frames=4, fps=30) == (0, 1, 3)


def test_actual_cfr_frame_coverage_accepts_serialized_clock_precision() -> None:
    assert observed_frame_indices((0.066667, 0, 0.033333), frames=3, fps=30) == (0, 1, 2)


@pytest.mark.parametrize(
    "clock", [-1, -0.0000001, 0.1, 0.008333, 1e300, float("nan"), float("inf"), True]
)
def test_invalid_actual_frame_clock(clock: float) -> None:
    with pytest.raises(ValueError, match="clock"):
        observed_frame_indices((clock,), frames=3, fps=30)


def test_missing_actual_clocks_cannot_claim_frame_coverage() -> None:
    with pytest.raises(ValueError, match="observation count"):
        observed_frame_indices((), frames=3, fps=30)


def test_hevc_requires_explicit_hardware_selection() -> None:
    with pytest.raises(ValueError, match="explicit hardware"):
        VideoRepairJob(Path("source"), Path("output"), 128, 128, 1, (), hardware_hevc=True)


@pytest.mark.parametrize("quality", [0, 101, True])
def test_native_quality_rejects_invalid_values(quality: int) -> None:
    with pytest.raises(ValueError, match="Native quality"):
        VideoRepairJob(
            Path("source"),
            Path("output"),
            128,
            128,
            1,
            (),
            hardware_encode=True,
            hardware_hevc=True,
            hevc_quality=quality,
        )


def test_native_quality_cannot_silently_use_another_encoder() -> None:
    with pytest.raises(ValueError, match="Native quality"):
        VideoRepairJob(Path("source"), Path("output"), 128, 128, 1, (), hevc_quality=65)


@pytest.mark.parametrize("start,total", [(-1, 10), (1, None), (8, 10), (True, 12)])
def test_video_rejects_invalid_global_ramp(start: int, total: int | None) -> None:
    with pytest.raises(ValueError, match="ramp clocks"):
        VideoRepairJob(
            Path("source"),
            Path("output"),
            128,
            128,
            4,
            (),
            ramp_start_frame=start,
            ramp_total_frames=total,
        )


@pytest.mark.parametrize("hevc_quality", [None, 65])
def test_real_hardware_hevc_has_quicktime_tag_and_all_frames(
    tmp_path: Path, hevc_quality: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    stream = Mock(wraps=repair_stream)
    monkeypatch.setattr("replay.subtitle_video.repair_stream", stream)
    source, output = tmp_path / "source.mkv", tmp_path / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=128x128:r=30",
            "-frames:v",
            "4",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    receipt = render_repaired_video(
        VideoRepairJob(
            source,
            output,
            128,
            128,
            4,
            (),
            hardware_encode=True,
            hardware_hevc=True,
            hevc_quality=hevc_quality,
            ramp_start_frame=8,
            ramp_total_frames=20,
        )
    )
    assert stream.call_args.kwargs["ramp_start_frame"] == 8
    assert stream.call_args.kwargs["ramp_total_frames"] == 20
    assert receipt.frames == 4
    assert not receipt.quality_passed
    measured = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,codec_tag_string,nb_read_frames",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    assert json.loads(measured.stdout)["streams"] == [
        {
            "codec_name": "hevc",
            "codec_tag_string": "hvc1",
            "nb_read_frames": "4",
        }
    ]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(output),
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
    )


def test_source_replacement_refused() -> None:
    with pytest.raises(ValueError, match="replace source"):
        VideoRepairJob(Path("source"), Path("source"), 16, 16, 1, ())


def test_missing_input_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing"):
        render_repaired_video(VideoRepairJob(tmp_path / "absent", tmp_path / "out", 16, 16, 1, ()))


def test_existing_output_preserved(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.write_bytes(b"source")
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        render_repaired_video(VideoRepairJob(source, output, 16, 16, 1, ()))
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize("hardware", [False, True])
@pytest.mark.parametrize("measure_quality", [False, True])
def test_real_stream_repairs_and_decodes(
    tmp_path: Path, hardware: bool, measure_quality: bool
) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=gray:s=64x64:r=30,drawbox=x=20:y=20:w=8:h=8:color=magenta:t=fill",
            "-frames:v",
            "4",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    measure = Mock(return_value=RepairEvidence(True, 0.0, 0.0, False))
    capture = RepairCapture(
        tmp_path / "lossless", [0, 1, 2, 3], VideoFrameClock((0, 1, 2, 3), Fraction(1, 30), 1)
    )
    receipt = render_repaired_video(
        VideoRepairJob(
            source,
            output,
            64,
            64,
            4,
            (SubtitleSpan(0, 4, 8, 8, 48, 48),),
            hardware_encode=hardware,
        ),
        measure=measure if measure_quality else None,
        require_quality=measure_quality,
        evidence=capture,
    )
    assert capture.manifest()["qualityApproval"] is False
    assert len(capture.records) == 4
    assert receipt.frames == 4
    if measure_quality:
        assert receipt.softened_frames == 0
        assert receipt.quality_measured
        assert receipt.quality_passed
        assert measure.call_count == 4
    else:
        assert receipt.softened_frames == 4
        assert not receipt.quality_measured
        assert not receipt.quality_passed
        measure.assert_not_called()
    assert receipt.attempts == 4
    assert receipt.seconds > 0
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(output), "-f", "null", "-"],
        check=True,
        capture_output=True,
    )
    assert result.stderr == b""


@pytest.mark.parametrize("expected,message", [(1, "excess"), (3, "before expected")])
def test_wrong_frame_count_refused(tmp_path: Path, expected: int, message: str) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=gray:s=16x16:r=30",
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match=message):
        render_repaired_video(VideoRepairJob(source, output, 16, 16, expected, ()))


@pytest.mark.parametrize("start", [-1, True, 1.5, 2_592_000])
def test_invalid_normalized_window(start: int) -> None:
    with pytest.raises(ValueError, match="normalized video window"):
        VideoRepairJob(Path("source"), Path("out"), 32, 32, 2, (), normalized_start_frame=start)


def test_window_cannot_hide_native_padding() -> None:
    with pytest.raises(ValueError, match="normalized video window"):
        VideoRepairJob(
            Path("source"),
            Path("out"),
            32,
            32,
            2,
            (),
            normalized_start_frame=0,
            allow_padded_tail=True,
        )


@pytest.mark.parametrize("start", [0, 2, 94])
def test_normalized_window_selects_exact_cfr_marker(tmp_path: Path, start: int) -> None:
    source, output = tmp_path / "source.mkv", tmp_path / "window.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            f"color=black:s=32x32:r=30:d=4,drawbox=color=lime:t=fill:enable='eq(n,{start})'",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    receipt = render_repaired_video(
        VideoRepairJob(source, output, 32, 32, 2, (), normalized_start_frame=start)
    )
    assert receipt.frames == 2
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(output),
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    pixels = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(2, 32, 32, 3)
    assert float(pixels[0, :, :, 1].mean()) > 240
    assert float(pixels[1].mean()) < 8


def test_normalized_vfr_window_keeps_frame_held_before_seek(tmp_path: Path) -> None:
    source, output = tmp_path / "gap.mkv", tmp_path / "window.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=red:s=32x32:r=30:d=6,drawbox=color=lime:t=fill:enable='eq(n,0)'",
            "-vf",
            "select='eq(n,0)+gte(n,150)'",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    receipt = render_repaired_video(
        VideoRepairJob(source, output, 32, 32, 2, (), normalized_start_frame=94)
    )
    assert receipt.frames == 2
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(output),
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    pixels = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(2, 32, 32, 3)
    assert float(pixels[:, :, :, 1].mean()) > 240
    assert float(pixels[:, :, :, 0].mean()) < 8


def test_normalized_window_cannot_pad_missing_tail(tmp_path: Path) -> None:
    source = tmp_path / "short.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=32x32:r=30:d=1",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="before expected"):
        render_repaired_video(
            VideoRepairJob(
                source, tmp_path / "partial.mp4", 32, 32, 2, (), normalized_start_frame=29
            )
        )
    assert source.is_file()


@pytest.mark.parametrize(
    "frames,padded,normalized", [(2, False, None), (1, True, None), (1, False, 0)]
)
def test_presentation_window_rejects_mixed_modes(
    frames: int, padded: bool, normalized: int | None
) -> None:
    with pytest.raises(ValueError, match="presentation-clock window"):
        VideoRepairJob(
            Path("source"),
            Path("out"),
            32,
            32,
            frames,
            (),
            allow_padded_tail=padded,
            normalized_start_frame=normalized,
            source_clock=VideoFrameClock((0,), Fraction(1, 90000), 2970),
        )


@pytest.mark.parametrize("temporal", [False, True])
def test_vfr_repair_preserves_colliding_frames_gap_and_terminal_extent(
    tmp_path: Path, capfd: pytest.CaptureFixture[str], temporal: bool
) -> None:
    source, output = tmp_path / "jitter.mp4", tmp_path / "repaired.mp4"
    clock = VideoFrameClock((0, 1170, 2970, 2971, 11970, 551970), Fraction(1, 90000), 2910)
    with clock.encoding_options() as options:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=s=32x32:r=30:d=0.2",
                "-c:v",
                "libx264",
                "-bf",
                "0",
                *options,
                str(source),
            ],
            check=True,
            capture_output=True,
        )
    receipt = render_repaired_video(
        VideoRepairJob(
            source,
            output,
            32,
            32,
            6,
            (SubtitleSpan(0, 6, 0, 0, 32, 32),),
            source_clock=clock,
            context_frames=1 if temporal else 0,
        )
    )
    assert receipt.frames == 6
    if temporal:
        assert receipt.attempts == 0
        assert not receipt.quality_passed
    else:
        assert receipt.attempts == 6
    assert capfd.readouterr().err == ""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "stream=time_base:packet=pts,duration:packet_side_data=",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    data = json.loads(probe.stdout)
    assert data["streams"][0]["time_base"] == "1/90000"
    assert tuple(packet["pts"] for packet in data["packets"]) == (
        0,
        1170,
        2970,
        2971,
        11970,
        551970,
    )
    assert data["packets"][-1]["duration"] == 2910
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(output),
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-enc_time_base",
            "demux",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    assert len(decoded.stdout) == 18432
    assert decoded.stderr == b""


def test_vfr_window_seek_selects_exact_nonzero_source_pts(tmp_path: Path) -> None:
    source, output = tmp_path / "marker.mkv", tmp_path / "window.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=32x32:r=30:d=0.2,drawbox=color=lime:t=fill:enable='eq(n,5)'",
            "-vf",
            "settb=1/90000,setpts='if(lt(N,5),N*3000,551970)'",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    receipt = render_repaired_video(
        VideoRepairJob(
            source,
            output,
            32,
            32,
            1,
            (),
            source_clock=VideoFrameClock((551970,), Fraction(1, 90000), 2970),
        )
    )
    assert receipt.frames == 1
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(output),
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    pixels = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(32, 32, 3)
    assert float(pixels[:, :, 1].mean()) > 240
    assert float(pixels[:, :, 0].mean()) < 8


def test_explicit_native_padding_trim(tmp_path: Path) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "output.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=gray:s=16x16:r=30",
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    receipt = render_repaired_video(
        VideoRepairJob(source, output, 16, 16, 1, (), allow_padded_tail=True)
    )
    assert receipt.frames == 1
    assert receipt.attempts == 0


def test_strict_quality_requires_measurement_before_opening_files(tmp_path: Path) -> None:
    output = tmp_path / "not-created.mp4"
    with pytest.raises(ValueError, match="independent quality measurement"):
        render_repaired_video(
            VideoRepairJob(tmp_path / "absent", output, 16, 16, 1, ()), require_quality=True
        )
    assert not output.exists()


@pytest.mark.parametrize(
    "evidence",
    [RepairEvidence(True), RepairEvidence(True, 0.1, 0.0, True)],
)
@pytest.mark.parametrize("strict", [False, True])
def test_cosmetic_fallback_is_not_quality_approval(
    tmp_path: Path, evidence: RepairEvidence, strict: bool
) -> None:
    source, output = tmp_path / "source.mp4", tmp_path / "candidate.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=gray:s=32x32:r=30",
            "-frames:v",
            "2",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    job = VideoRepairJob(source, output, 32, 32, 2, (SubtitleSpan(0, 2, 0, 0, 32, 32),))
    measure = Mock(return_value=evidence)
    if strict:
        with pytest.raises(ValueError, match="measured quality gate"):
            render_repaired_video(job, measure=measure, require_quality=True)
        assert source.is_file()
    else:
        receipt = render_repaired_video(job, measure=measure)
        assert receipt.frames == 2
        decoded = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        assert decoded.stdout == b"2\n"
        assert receipt.softened_frames == 2
        assert not receipt.quality_passed
        if evidence.readable_text is None:
            assert not receipt.quality_measured
        else:
            assert receipt.quality_measured


def test_spatial_measurement_detects_one_unmasked_channel_change() -> None:
    source = np.zeros((32, 32, 3), dtype=np.uint8)[::2, ::2]
    result = source.copy()
    result[0, 0, 2] = 1
    evidence = spatial_evidence(source, result, np.zeros((16, 16), dtype=np.uint8))
    assert not evidence.technically_valid
    assert evidence.readable_text is None
    assert evidence.unstable_fraction is None


def test_spatial_measurement_does_not_claim_ocr() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    evidence = spatial_evidence(image, image, np.zeros((16, 16), dtype=np.uint8))
    assert evidence.technically_valid
    assert evidence.readable_text is None
    assert evidence.unstable_fraction is None
