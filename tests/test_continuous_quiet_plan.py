import json
import shutil
import wave
from fractions import Fraction
from math import ceil
from pathlib import Path

import pytest

from replay.audio_clock import AudioFrameClock
from replay.caption_projection import MeasuredQuietSpan
from replay.continuous_quiet_plan import main, prepare_config, run, source_quiet
from replay.core import ReplayError
from replay.quiet_cuts import quiet_cut_frames
from replay.stages import file_digest


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
