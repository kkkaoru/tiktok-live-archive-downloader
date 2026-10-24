import json
import shutil
import wave
from fractions import Fraction
from math import ceil
from pathlib import Path

import pytest

from replay import continuous_quiet_plan as plan
from replay.audio_clock import AudioFrameClock
from replay.caption_projection import MeasuredQuietSpan
from replay.continuous_quiet_plan import main, prepare_config, run, source_quiet
from replay.core import ReplayError
from replay.quiet_cuts import quiet_cut_frames
from replay.stages import file_digest
from replay.waveform_quiet import measured_quiet_runs


def write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def bind(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def pcm(path: Path, channels: int) -> Path:
    with wave.open(str(path), "wb") as stream:
        stream.setparams((channels, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0" * (32000 * channels * 2))
    return path


@pytest.fixture
def config(tmp_path: Path) -> Path:
    mono, stereo = pcm(tmp_path / "mono.wav", 1), pcm(tmp_path / "stereo.wav", 2)
    audio = write(
        tmp_path / "audio.json",
        {
            "streams": [{"sample_rate": "16000", "time_base": "1/16000"}],
            "frames": [{"pts": 0, "nb_samples": 32000}],
        },
    )
    video = write(
        tmp_path / "video.json",
        {
            "streams": [{"time_base": "1/30"}],
            "frames": [{"pts": 0, "duration": 60}],
        },
    )
    projection = write(
        tmp_path / "projection.json", {"artifacts": {"pcm": bind(mono), "audioClock": bind(audio)}}
    )
    labels = write(
        tmp_path / "labels.json",
        {
            "clock": "source-presentation",
            "configSHA256": file_digest(projection),
            "parentSHA256": {str(mono): file_digest(mono)},
            "labels": [{"text": "声", "start": 0.9, "end": 1.1, "provenance": "original:0"}],
        },
    )
    lineage = write(
        tmp_path / "lineage.json",
        {
            "clock": "continuous-decoded-pcm",
            "stereoPCM": bind(stereo),
            "monoPCM": bind(mono),
            "audioClock": bind(audio),
            "transforms": ["concatenate-decoded-frames", "duration-preserving-resample"],
        },
    )
    return write(
        tmp_path / "config.json",
        {
            "version": 1,
            "fps": 30,
            "pcmClock": "continuous-decoded-pcm",
            "artifacts": {
                "stereoPCM": bind(stereo),
                "audioClock": bind(audio),
                "videoClock": bind(video),
                "projectionConfig": bind(projection),
                "labels": bind(labels),
                "lineageEvidence": bind(lineage),
            },
        },
    )


def test_real_stereo_measurement_protects_asr_despite_quiet(config: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    output = config.parent / "output"
    assert main([str(config), str(output), "--ffmpeg", ffmpeg]) == 0
    result = json.loads((output / "cuts.json").read_text())
    assert result["cuts"] == [
        {"source_start_frame": 24, "source_end_frame": 36, "output_start_frame": 0}
    ]
    assert result["deliveryComplete"] is False
    assert result["nonSpeechRemovalVerified"] is False


def test_opposite_phase_channels_do_not_cancel_into_quiet(config: Path) -> None:
    stereo = config.parent / "stereo.wav"
    with wave.open(str(stereo), "wb") as stream:
        stream.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes((b"\xe8\x03\x18\xfc") * 32000)
    lineage = config.parent / "lineage.json"
    evidence = json.loads(lineage.read_text())
    evidence["stereoPCM"] = bind(stereo)
    write(lineage, evidence)
    data = json.loads(config.read_text())
    data["artifacts"]["stereoPCM"] = bind(stereo)
    data["artifacts"]["lineageEvidence"] = bind(lineage)
    write(config, data)
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    cuts = run(config, config.parent / "out", ffmpeg=ffmpeg)
    assert len(cuts) == 1
    assert cuts[0].source_start_frame == 0
    assert cuts[0].source_end_frame == 60


def test_gap_is_not_claimed_quiet() -> None:
    clock = AudioFrameClock((1000, 3000), (1000, 1000), 10000, Fraction(1, 10000))
    assert source_quiet((MeasuredQuietSpan(0, 0.2),), clock) == (
        MeasuredQuietSpan(0.1, 0.2),
        MeasuredQuietSpan(0.3, 0.4),
    )


def test_timestamp_jitter_gaps_do_not_split_a_quiet_run() -> None:
    starts = tuple(index * 2064 for index in range(100))
    clock = AudioFrameClock(starts, (2048,) * 100, 48000, Fraction(1, 48000))
    result = source_quiet((MeasuredQuietSpan(0, 204800 / 48000),), clock)
    assert result == (MeasuredQuietSpan(0, (99 * 2064 + 2048) / 48000),)
    loud_after = source_quiet((MeasuredQuietSpan(0, 2048 / 48000),), clock)
    assert loud_after == (MeasuredQuietSpan(0, 2064 / 48000),)


def test_alternating_encoder_steps_join_like_constant_ones() -> None:
    starts = (0, 2064, 4128, 6144, 8208, 10272, 12288)
    clock = AudioFrameClock(starts, (2048,) * 7, 48000, Fraction(1, 48000))
    assert source_quiet((MeasuredQuietSpan(0, 7 * 2048 / 48000),), clock) == (
        MeasuredQuietSpan(0, 14336 / 48000),
    )


def test_flv_millisecond_steps_do_not_split_a_quiet_run() -> None:
    # 40/60 ms steps for 42.67 ms frames (13 to 2): overlaps and 17.33 ms gaps averaging out.
    steps = (1920,) * 6 + (2880,) + (1920,) * 7 + (2880,)
    starts = tuple(sum(steps[:index]) for index in range(16))
    clock = AudioFrameClock(starts, (2048,) * 16, 48000, Fraction(1, 48000))
    assert source_quiet((MeasuredQuietSpan(0, 16 * 2048 / 48000),), clock) == (
        MeasuredQuietSpan(0, 32768 / 48000),
    )


def test_a_dropped_frame_still_splits_a_quiet_run() -> None:
    starts = (0, 2048, 6144, 8192)
    clock = AudioFrameClock(starts, (2048,) * 4, 48000, Fraction(1, 48000))
    assert source_quiet((MeasuredQuietSpan(0, 4 * 2048 / 48000),), clock) == (
        MeasuredQuietSpan(0, 4096 / 48000),
        MeasuredQuietSpan(6144 / 48000, 10240 / 48000),
    )


def test_a_dropped_frame_among_flv_steps_still_splits_a_quiet_run() -> None:
    # The 80 ms step skips one frame: a 37.33 ms gap and a lasting drift of about one frame.
    starts = (0, 1920, 4800, 8640, 11520, 13440)
    clock = AudioFrameClock(starts, (2048,) * 6, 48000, Fraction(1, 48000))
    assert source_quiet((MeasuredQuietSpan(0, 6 * 2048 / 48000),), clock) == (
        MeasuredQuietSpan(0, 6848 / 48000),
        MeasuredQuietSpan(8640 / 48000, 15488 / 48000),
    )
    assert source_quiet((MeasuredQuietSpan(0, 2 * 2048 / 48000),), clock) == (
        MeasuredQuietSpan(0, 4800 / 48000),
    )


def test_drift_beyond_one_frame_stops_joining() -> None:
    # A clock that runs 1,000 samples ahead per frame passes one frame of drift at the third gap.
    starts = (0, 3048, 6096, 9144, 12192)
    clock = AudioFrameClock(starts, (2048,) * 5, 48000, Fraction(1, 48000))
    assert source_quiet((MeasuredQuietSpan(0, 5 * 2048 / 48000),), clock) == (
        MeasuredQuietSpan(0, 8144 / 48000),
        MeasuredQuietSpan(9144 / 48000, 14240 / 48000),
    )


def test_overlap_is_quiet_only_when_every_owner_is_quiet() -> None:
    clock = AudioFrameClock((0, 500), (1000, 1000), 10000, Fraction(1, 10000))
    assert source_quiet((MeasuredQuietSpan(0, 0.2),), clock) == (MeasuredQuietSpan(0, 0.15),)


def test_overlapping_nonquiet_frame_blocks_quiet_owner() -> None:
    clock = AudioFrameClock((0, 500), (1000, 1000), 10000, Fraction(1, 10000))
    assert source_quiet((MeasuredQuietSpan(0, 0.1),), clock) == (MeasuredQuietSpan(0, 0.05),)


def test_touching_supported_frames_merge() -> None:
    clock = AudioFrameClock((0, 1000), (1000, 1000), 10000, Fraction(1, 10000))
    assert source_quiet((MeasuredQuietSpan(0, 0.2),), clock) == (MeasuredQuietSpan(0, 0.2),)


def test_fractional_resample_tail_is_omitted_not_extended() -> None:
    clock = AudioFrameClock((0,), (48001,), 48000, Fraction(1, 48000))
    result = source_quiet((MeasuredQuietSpan(0, 44101 / 44100),), clock, measurement_rate=44100)
    assert result == (MeasuredQuietSpan(0, 48001 / 48000),)


def test_owner_sweep_advances_past_overlap_without_bridging_gap() -> None:
    clock = AudioFrameClock((0, 500, 2000), (1000, 1000, 1000), 10000, Fraction(1, 10000))
    result = source_quiet((MeasuredQuietSpan(0, 0.3),), clock)
    assert result == (MeasuredQuietSpan(0, 0.15), MeasuredQuietSpan(0.2, 0.3))


def test_three_owners_with_unknown_middle_owner_protect_overlap() -> None:
    clock = AudioFrameClock((0, 250, 500), (1000, 1000, 1000), 10000, Fraction(1, 10000))
    quiet = (MeasuredQuietSpan(0, 0.1), MeasuredQuietSpan(0.2, 0.3))
    assert source_quiet(quiet, clock) == (
        MeasuredQuietSpan(0, 0.025),
        MeasuredQuietSpan(0.125, 0.15),
    )
    assert source_quiet((MeasuredQuietSpan(0, 0.3),), clock) == (MeasuredQuietSpan(0, 0.15),)


def test_long_nonquiet_owner_blocks_nested_short_quiet_owners() -> None:
    clock = AudioFrameClock((0, 100, 200), (2000, 100, 100), 10000, Fraction(1, 10000))
    assert source_quiet((MeasuredQuietSpan(0.2, 0.22),), clock) == ()
    assert source_quiet((MeasuredQuietSpan(0, 0.22),), clock) == (MeasuredQuietSpan(0, 0.2),)


def test_no_measured_quiet_does_not_fabricate_silence() -> None:
    clock = AudioFrameClock((7380, 7880), (1000, 1000), 10000, Fraction(1, 10000))
    assert source_quiet((), clock) == ()


def test_many_micro_overlaps_form_usable_quiet_support_with_nonzero_origin() -> None:
    clock = AudioFrameClock(
        tuple(35424 + index * 1920 for index in range(10000)),
        (2048,) * 10000,
        48000,
        Fraction(1, 48000),
    )
    result = source_quiet((MeasuredQuietSpan(0, 20480000 / 48000),), clock)
    assert result == (MeasuredQuietSpan(0.738, (35424 + 9999 * 1920 + 2048) / 48000),)
    assert result[0].end - result[0].start > 0.3
    frames = ceil(result[0].end * 30)
    cuts = quiet_cut_frames(result, (), source_frames=frames)
    assert sum(cut.duration_frames for cut in cuts) < frames / 10


def test_invalid_measurement_rate_rejected() -> None:
    clock = AudioFrameClock((0,), (1000,), 10000, Fraction(1, 10000))
    with pytest.raises(ReplayError, match="sample rate"):
        source_quiet((), clock, measurement_rate=0)


@pytest.mark.parametrize(
    "quiet",
    [
        (MeasuredQuietSpan(0, 0.3),),
        (MeasuredQuietSpan(0.1, 0.15), MeasuredQuietSpan(0, 0.05)),
    ],
)
def test_bad_quiet_support_rejected(quiet: tuple[MeasuredQuietSpan, ...]) -> None:
    clock = AudioFrameClock((0,), (2000,), 10000, Fraction(1, 10000))
    with pytest.raises(ReplayError):
        source_quiet(quiet, clock)


@pytest.mark.parametrize(
    "field,value", [("fps", True), ("pcmClock", "source-presentation"), ("artifacts", {})]
)
def test_invalid_config(config: Path, field: str, value: object) -> None:
    data = json.loads(config.read_text())
    data[field] = value
    write(config, data)
    with pytest.raises(ReplayError):
        run(config, config.parent / "out", ffmpeg="unused")


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("lineageEvidence", "monoPCM", {}),
        ("labels", "configSHA256", "0" * 64),
        ("labels", "clock", "input-pcm-not-projected"),
        ("labels", "labels", [{"text": 2}]),
    ],
)
def test_clock_provenance_rejected(config: Path, name: str, field: str, value: object) -> None:
    data = json.loads(config.read_text())
    path = Path(data["artifacts"][name]["path"])
    document = json.loads(path.read_text())
    document[field] = value
    write(path, document)
    data["artifacts"][name] = bind(path)
    write(config, data)
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    with pytest.raises(ReplayError):
        run(config, config.parent / "out", ffmpeg=ffmpeg)


def test_existing_directory_rejected(config: Path) -> None:
    with pytest.raises(ReplayError, match="new canonical"):
        run(config, config.parent, ffmpeg="unused")


def test_bad_binding(config: Path) -> None:
    data = json.loads(config.read_text())
    data["artifacts"]["labels"] = {"path": 5}
    write(config, data)
    with pytest.raises(ReplayError, match="pinned input"):
        run(config, config.parent / "out", ffmpeg="unused")


def test_prepare_quiet_config_at_completed_artifact_boundary(config: Path) -> None:
    data = json.loads(config.read_text())
    artifacts = data.pop("artifacts")
    data["artifactPaths"] = {name: row["path"] for name, row in artifacts.items()}
    data["prepareKind"] = "quiet"
    template = write(config.parent / "template.json", data)
    output = config.parent / "prepared.json"
    assert main([str(template), str(output), "--prepare"]) == 0
    assert json.loads(output.read_text())["artifacts"] == artifacts
    with pytest.raises(ReplayError, match="new canonical"):
        prepare_config(template, output)


def test_prepare_lineage_materializes_explicit_roles(config: Path) -> None:
    template = write(
        config.parent / "template.json",
        {
            "prepareKind": "lineage",
            "clock": "continuous-decoded-pcm",
            "artifactPaths": {
                "stereoPCM": str(config.parent / "stereo.wav"),
                "monoPCM": str(config.parent / "mono.wav"),
                "audioClock": str(config.parent / "audio.json"),
            },
        },
    )
    output = config.parent / "prepared.json"
    prepare_config(template, output)
    assert json.loads(output.read_text())["monoPCM"] == bind(config.parent / "mono.wav")


@pytest.mark.parametrize(
    "kind,paths", [("unknown", {}), ("lineage", {"stereoPCM": 1, "monoPCM": 2, "audioClock": 3})]
)
def test_prepare_rejects_missing_or_nonpath_roles(config: Path, kind: str, paths: object) -> None:
    template = write(config.parent / "template.json", {"prepareKind": kind, "artifactPaths": paths})
    with pytest.raises(ReplayError):
        prepare_config(template, config.parent / "prepared.json")


def test_plan_needs_explicit_tool(config: Path) -> None:
    with pytest.raises(ReplayError, match="explicit FFmpeg"):
        main([str(config), str(config.parent / "out")])


def stem_records(tmp_path: Path, *, output_gain: float, input_gain: float) -> tuple[Path, str]:
    """A stem with the separation receipt that pins it and the preparation gain record."""
    stem = tmp_path / "pipeline/vocals.wav"
    stem.parent.mkdir()
    stem.write_bytes(b"stem")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    gain = prepared / "input-gain.json"
    gain.write_text(json.dumps({"gainDB": input_gain}), encoding="utf-8")
    (prepared / "pcm-preparation.json").write_text(
        json.dumps({"inputGain": {"path": str(gain), "sha256": file_digest(gain)}}),
        encoding="utf-8",
    )
    (stem.parent / "result.json").write_text(
        json.dumps({
            "modelInput": str(prepared / "model-input.wav"), "outputGainDB": output_gain,
            "outputs": [{"path": str(stem), "sha256": file_digest(stem), "bytes": 4}],
        }),
        encoding="utf-8",
    )  # fmt: skip
    return stem, file_digest(stem)


def peak_lines(peaks: list[float], *, window: int = 882) -> list[str]:
    lines: list[str] = []
    for index, peak in enumerate(peaks):
        lines += [f"frame:{index} pts:{index * window} pts_time:{index * window / 44100}",
                  f"lavfi.astats.Overall.Peak_level={peak}"]  # fmt: skip
    return lines


def test_the_quiet_threshold_follows_the_stems_recorded_gain(tmp_path: Path) -> None:
    stem, digest = stem_records(tmp_path, output_gain=-1.0, input_gain=-3.1)
    assert plan.stem_gain_db(stem, digest) == pytest.approx(-4.0)
    assert plan.quiet_threshold_dbfs(stem, digest) == pytest.approx(-64.0)
    peaks = [-30.0, -58.5, -61.0, -70.0, -59.5, -62.0, -20.0, -61.5, -61.5, -30.0]
    plain = measured_quiet_runs(peak_lines(peaks), sample_count=8820)
    gained = measured_quiet_runs(
        peak_lines([peak - 4.0 for peak in peaks]), sample_count=8820,
        threshold_dbfs=plan.quiet_threshold_dbfs(stem, digest),
    )  # fmt: skip
    assert gained == plain
    assert plain != measured_quiet_runs(
        peak_lines([peak - 4.0 for peak in peaks]), sample_count=8820
    )


def test_a_stem_without_gain_records_has_no_offset(tmp_path: Path) -> None:
    stem = tmp_path / "vocals.wav"
    stem.write_bytes(b"stem")
    assert plan.quiet_threshold_dbfs(stem, file_digest(stem)) == -60.0


def test_a_preparation_without_a_measured_gain_offsets_by_the_output_gain_only(
    tmp_path: Path,
) -> None:
    stem, digest = stem_records(tmp_path, output_gain=-1.0, input_gain=-0.1)
    (tmp_path / "prepared/pcm-preparation.json").write_text("{}", encoding="utf-8")
    assert plan.stem_gain_db(stem, digest) == pytest.approx(-1.0)


def test_unpinned_or_changed_gain_records_are_refused(tmp_path: Path) -> None:
    stem, digest = stem_records(tmp_path, output_gain=-1.0, input_gain=-3.1)
    with pytest.raises(ReplayError, match="does not pin this stem"):
        plan.stem_gain_db(stem, "0" * 64)
    (tmp_path / "prepared/input-gain.json").write_text('{"gainDB": -9.0}', encoding="utf-8")
    with pytest.raises(ReplayError, match="differs from its pin"):
        plan.stem_gain_db(stem, digest)


def test_a_gain_that_is_not_a_number_is_refused(tmp_path: Path) -> None:
    stem, digest = stem_records(tmp_path, output_gain=-1.0, input_gain=-0.1)
    result = tmp_path / "pipeline/result.json"
    data = json.loads(result.read_text(encoding="utf-8"))
    result.write_text(json.dumps({**data, "outputGainDB": True}), encoding="utf-8")
    with pytest.raises(ReplayError, match="not a finite number"):
        plan.stem_gain_db(stem, digest)


def model_input_records(tmp_path: Path, *, input_gain: float) -> tuple[Path, str]:
    """A separation model input (the quiet input of older transcription flows) and its receipt."""
    model_input = tmp_path / "model-input.wav"
    model_input.write_bytes(b"input")
    gain = tmp_path / "input-gain.json"
    gain.write_text(json.dumps({"gainDB": input_gain}), encoding="utf-8")
    (tmp_path / "pcm-preparation.json").write_text(
        json.dumps({
            "output": {"path": str(model_input), "sha256": file_digest(model_input)},
            "inputGain": {"path": str(gain), "sha256": file_digest(gain)},
        }),
        encoding="utf-8",
    )  # fmt: skip
    return model_input, file_digest(model_input)


def test_a_model_input_carries_only_its_extraction_gain(tmp_path: Path) -> None:
    model_input, digest = model_input_records(tmp_path, input_gain=-3.1)
    # A separation receipt beside it (the older flow's) is not consulted for the model input.
    (tmp_path / "result.json").write_text(json.dumps({"gainDB": -0.5}), encoding="utf-8")
    assert plan.stem_gain_db(model_input, digest) == pytest.approx(-3.0)
    assert plan.quiet_threshold_dbfs(model_input, digest) == pytest.approx(-63.0)


def test_a_model_input_without_a_preparation_receipt_has_no_offset(tmp_path: Path) -> None:
    model_input = tmp_path / "model-input.wav"
    model_input.write_bytes(b"input")
    (tmp_path / "result.json").write_text(json.dumps({"gainDB": -0.5}), encoding="utf-8")
    assert plan.quiet_threshold_dbfs(model_input, file_digest(model_input)) == -60.0


def test_a_model_input_its_receipt_does_not_pin_is_refused(tmp_path: Path) -> None:
    model_input, digest = model_input_records(tmp_path, input_gain=-3.1)
    with pytest.raises(ReplayError, match="does not pin"):
        plan.stem_gain_db(model_input, "0" * 64)
    (tmp_path / "input-gain.json").write_text('{"gainDB": -9.0}', encoding="utf-8")
    with pytest.raises(ReplayError, match="differs from its pin"):
        plan.stem_gain_db(model_input, digest)
