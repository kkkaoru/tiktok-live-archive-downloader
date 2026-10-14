"""Post-assembly regression observations; no real recordings or app operations."""

import json
import shutil
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import pytest
from test_compressed_part_delivery import Example, pin, read, write
from test_compressed_part_delivery import example as example

from replay import aac_presentation_normalization as normalization
from replay import compressed_audio_verification as audit
from replay import compressed_part_delivery as delivery
from replay import delivery_verification as technical
from replay.audio_clock import AudioFrameClock
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest
from replay.whole_technical_batch import Reference


@dataclass(frozen=True)
class RealObservation:
    example: Example
    plan: delivery.Plan
    raw: dict[str, object]


@pytest.fixture
def real_observation(example: Example) -> RealObservation:
    """One tiny software-generated fixture, never production footage or app jobs."""
    ffmpeg_name, ffprobe_name = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg_name is not None and ffprobe_name is not None
    raw = read(example.config)
    tools = record(raw["tools"])
    tools["ffmpeg"] = pin(Path(ffmpeg_name).resolve(strict=True))
    tools["ffprobe"] = pin(Path(ffprobe_name).resolve(strict=True))
    raw["width"], raw["height"] = 64, 96
    contract_path = Path(str(record(raw["clockContract"])["path"]))
    contract = read(contract_path)
    xml = Path(str(record(contract["semanticXML"])["path"]))
    xml.write_text(
        xml.read_text(encoding="utf-8").replace(
            'width="640" height="960"', 'width="64" height="96"'
        ),
        encoding="utf-8",
    )
    contract["semanticXML"] = pin(xml)
    raw["clockContract"] = write(contract_path, contract)
    write(example.config, raw)
    plan = delivery.load(example.config)
    technical._run(
        [
            str(plan.tools["ffmpeg"].path),
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x96:rate=30:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-c:v",
            "libx265",
            "-threads",
            "1",
            "-x265-params",
            "pools=1:frame-threads=1:log-level=error",
            "-tag:v",
            "hvc1",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-af",
            "atrim=end_sample=40960",
            "-movflags",
            "+faststart",
            str(plan.output),
        ],
        directory=example.root,
        name="synthetic-media",
        timeout=30,
    )
    collected = technical._run(
        audit.observation_command(plan),
        directory=example.root,
        name="synthetic-observation",
        timeout=30,
    )
    return RealObservation(example, plan, read(collected))


@pytest.fixture
def h264_observation(real_observation: RealObservation) -> RealObservation:
    """A tiny explicit H264/AVC fixture, preserving the measured AAC bytes."""
    value = real_observation
    movie = value.plan.output.with_name("h264.mp4")
    technical._run(
        [
            str(value.plan.tools["ffmpeg"].path),
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(value.plan.output),
            "-c:v",
            "libx264",
            "-threads",
            "1",
            "-tag:v",
            "avc1",
            "-c:a",
            "copy",
            str(movie),
        ],
        directory=value.example.root,
        name="synthetic-h264",
        timeout=30,
    )
    value.example.change("output", str(movie))
    plan = delivery.load(value.example.config)
    collected = technical._run(
        audit.observation_command(plan),
        directory=value.example.root,
        name="synthetic-h264-observation",
        timeout=30,
    )
    return RealObservation(value.example, plan, read(collected))


def test_actual_h264_requires_explicit_matching_contract(
    h264_observation: RealObservation,
) -> None:
    value = h264_observation
    receipt, _, frames, packets = audit.whole_observations(
        value.plan, value.raw, video_contract="h264"
    )
    assert receipt["version"] == 3
    assert receipt["videoContract"] == "h264"
    assert receipt["decodePasses"] == 1
    assert len([row for row in audit._rows(frames, "frames") if row["media_type"] == "video"]) == 30
    assert len(audit._payloads(packets)) > 30
    with pytest.raises(ReplayError, match="Video codec/tag/geometry mismatch"):
        audit.whole_observations(value.plan, value.raw)


def test_actual_hevc_cannot_be_reinterpreted_as_h264(real_observation: RealObservation) -> None:
    with pytest.raises(ReplayError, match="Video codec/tag/geometry mismatch"):
        audit.whole_observations(real_observation.plan, real_observation.raw, video_contract="h264")


@pytest.mark.parametrize("damage", ["corrupt", "truncated"])
def test_h264_damage_still_fails_strict_decode(
    h264_observation: RealObservation, damage: str
) -> None:
    value = h264_observation
    data = bytearray(value.plan.output.read_bytes())
    packet = next(
        row
        for row in audit._rows(value.raw, "packets_and_frames")
        if row.get("type") == "packet" and row.get("stream_index") == 0
    )
    if damage == "corrupt":
        position, size = int(str(packet["pos"])), int(str(packet["size"]))
        data[position : position + size] = b"\0" * size
    else:
        del data[len(data) // 2 :]
    movie = value.plan.output.with_name("h264-" + damage + ".mp4")
    movie.write_bytes(data)
    with pytest.raises(ReplayError):
        baseline_decode(value, movie, "h264-baseline-" + damage)
    plan = replace(value.plan, output=movie)
    with pytest.raises(ReplayError):
        path = technical._run(
            audit.observation_command(plan),
            directory=value.example.root,
            name="h264-combined-" + damage,
            timeout=30,
        )
        audit.whole_observations(plan, read(path), video_contract="h264")


def test_actual_combined_probe_has_every_decoded_frame_and_packet(
    real_observation: RealObservation,
) -> None:
    receipt, streams, frames, packets = audit.whole_observations(
        real_observation.plan, real_observation.raw
    )
    assert receipt["decodePasses"] == 1
    assert receipt["fullDecode"] is True
    assert len(audit._rows(streams, "streams")) == 2
    assert len([row for row in audit._rows(frames, "frames") if row["media_type"] == "video"]) == 30
    assert len(audit._payloads(packets)) > 30


def baseline_decode(value: RealObservation, movie: Path, name: str) -> Path:
    """The previous full FFmpeg decoder's strict error policy, for real parity tests."""
    return technical._run(
        [
            str(value.plan.tools["ffmpeg"].path),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-max_error_rate",
            "0",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(movie),
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-f",
            "null",
            "-",
        ],
        directory=value.example.root,
        name=name,
        timeout=30,
    )


@pytest.mark.parametrize("damage", ["corrupt", "truncated"])
def test_actual_damage_is_rejected_by_old_and_combined_decoder(
    real_observation: RealObservation, damage: str
) -> None:
    value = real_observation
    data = bytearray(value.plan.output.read_bytes())
    packets = [
        row
        for row in audit._rows(value.raw, "packets_and_frames")
        if row.get("type") == "packet" and row.get("stream_index") == 0
    ]
    if damage == "corrupt":
        first = packets[0]
        position, size = int(str(first["pos"])), int(str(first["size"]))
        data[position : position + size] = b"\0" * size
    else:
        del data[len(data) // 2 :]
    movie = value.plan.output.with_name(damage + ".mp4")
    movie.write_bytes(data)
    with pytest.raises(ReplayError):
        baseline_decode(value, movie, "baseline-" + damage)
    plan = replace(value.plan, output=movie)
    with pytest.raises(ReplayError):
        path = technical._run(
            audit.observation_command(plan),
            directory=value.example.root,
            name="combined-" + damage,
            timeout=30,
        )
        audit.whole_observations(plan, read(path))


def test_actual_missing_audio_is_not_accepted(real_observation: RealObservation) -> None:
    value = real_observation
    movie = value.plan.output.with_name("noaudio.mp4")
    technical._run(
        [
            str(value.plan.tools["ffmpeg"].path),
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(value.plan.output),
            "-map",
            "0:v",
            "-c",
            "copy",
            str(movie),
        ],
        directory=value.example.root,
        name="make-noaudio",
        timeout=30,
    )
    with pytest.raises(ReplayError):
        baseline_decode(value, movie, "baseline-noaudio")
    plan = replace(value.plan, output=movie)
    path = technical._run(
        audit.observation_command(plan),
        directory=value.example.root,
        name="combined-noaudio",
        timeout=30,
    )
    with pytest.raises(ReplayError, match="one video and one audio"):
        audit.whole_observations(plan, read(path))


def test_actual_video_clock_gap_is_rejected(real_observation: RealObservation) -> None:
    value = real_observation
    movie = value.plan.output.with_name("gap.mp4")
    technical._run(
        [
            str(value.plan.tools["ffmpeg"].path),
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(value.plan.output),
            "-vf",
            "setpts=PTS+gte(N\\,15)/(30*TB)",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "libx265",
            "-threads",
            "1",
            "-x265-params",
            "pools=1:frame-threads=1:log-level=error",
            "-tag:v",
            "hvc1",
            "-c:a",
            "copy",
            str(movie),
        ],
        directory=value.example.root,
        name="make-gap",
        timeout=30,
    )
    baseline_decode(value, movie, "baseline-gap")  # Error-free decoding alone cannot prove clocks.
    plan = replace(value.plan, output=movie)
    path = technical._run(
        audit.observation_command(plan),
        directory=value.example.root,
        name="combined-gap",
        timeout=30,
    )
    with pytest.raises(ReplayError, match="rate mismatch|PTS|duration"):
        audit.whole_observations(plan, read(path))


def test_observe_reuse_and_derived_domains_decode_only_once(
    real_observation: RealObservation, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = real_observation
    original_run = technical._run
    commands: list[list[str]] = []

    def counted(command: list[str], *, directory: Path, name: str, timeout: int) -> Path:
        commands.append(command)
        return original_run(command, directory=directory, name=name, timeout=timeout)

    monkeypatch.setattr(technical, "_run", counted)
    directory = value.example.root / "single-observation"
    receipt = audit.observe(value.example.config, directory)
    assert audit.observe(value.example.config, directory) == receipt
    target = value.example.root / "single-technical.json"
    assert audit.observed_result(value.example.config, directory, target, audio=False) == receipt
    assert audit.observed_result(value.example.config, directory, target, audio=False) == receipt
    assert len(commands) == 1
    assert "-show_frames" in commands[0] and "-show_packets" in commands[0]


def bind_source_audio_observations(value: RealObservation, *, video_contract: str = "hevc") -> None:
    """Test source facts are the actual measured fixture frames/payloads."""
    _, streams, frames, packets = audit.whole_observations(
        value.plan, value.raw, video_contract=video_contract
    )
    audio = [row for row in audit._rows(streams, "streams") if row["codec_type"] == "audio"]
    value.example.audio(
        "decodedProbe",
        {
            "streams": audio,
            "frames": [
                row for row in audit._rows(frames, "frames") if row["media_type"] == "audio"
            ],
        },
    )
    value.example.audio("packetProbe", packets)
    value.example.change("allowClockCandidate", True)


@pytest.mark.parametrize("action", ["--technical-result", "--audio-result"])
def test_h264_cli_domains_reuse_one_bound_scan(
    h264_observation: RealObservation,
    action: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = h264_observation
    bind_source_audio_observations(value, video_contract="h264")
    directory = value.example.root / "h264-observed"
    assert (
        audit.main(
            [
                str(value.example.config),
                str(directory),
                "--observe",
                "--video-contract",
                "h264",
            ]
        )
        == 0
    )
    snapshot = read(directory / "input-snapshot.json")
    manifest = read(directory / "observation.json")
    assert snapshot["videoContract"] == manifest["videoContract"] == "h264"
    assert snapshot["version"] == manifest["version"] == 3
    original = (directory / "combined.stdout").read_bytes()

    def unexpected(*args: object, **kwargs: object) -> Path:
        pytest.fail("Contract checks and derived domains must never decode again")

    monkeypatch.setattr(technical, "_run", unexpected)
    with pytest.raises(ReplayError, match="bound to current inputs"):
        audit.observe(value.example.config, directory)
    target = value.example.root / "h264-derived.json"
    with pytest.raises(ReplayError, match="bound to current inputs"):
        audit.observed_result(value.example.config, directory, target, audio=True)
    assert not target.exists()
    args = [
        str(value.example.config),
        str(directory),
        str(target),
        action,
        "--video-contract",
        "h264",
    ]
    assert audit.main(args) == 0
    saved = read(target)
    assert saved["videoContract"] == "h264" and saved["version"] == 3
    assert saved["deliveryApproved"] is False
    assert audit.main(args) == 0
    assert (directory / "combined.stdout").read_bytes() == original
    assert '"videoContract": "h264"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "change",
    [
        {"videoContract": "hevc"},
        {"videoContract": "vp9"},
        {"version": 3.0},
        {"videoContract": None},
        {"unexpected": True},
    ],
)
def test_h264_saved_contract_mutation_is_not_redecoded(
    h264_observation: RealObservation,
    change: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = h264_observation
    directory = value.example.root / "changed-h264-contract"
    audit.observe(value.example.config, directory, video_contract="h264")
    manifest = read(directory / "observation.json")
    manifest.update(change)
    write(directory / "observation.json", manifest)

    def unexpected(*args: object, **kwargs: object) -> Path:
        pytest.fail("A changed selector cannot authorize another decode")

    monkeypatch.setattr(technical, "_run", unexpected)
    with pytest.raises(ReplayError, match="bound to current inputs"):
        audit.observe(value.example.config, directory, video_contract="h264")


def test_saved_snapshot_selector_cannot_be_rebound(
    h264_observation: RealObservation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = h264_observation
    directory = value.example.root / "rebound-h264-snapshot"
    audit.observe(value.example.config, directory, video_contract="h264")
    path = directory / "input-snapshot.json"
    snapshot = read(path)
    snapshot["videoContract"] = "hevc"
    manifest = read(directory / "observation.json")
    manifest["snapshot"] = write(path, snapshot)
    write(directory / "observation.json", manifest)

    def unexpected(*args: object, **kwargs: object) -> Path:
        pytest.fail("A repinned conflicting snapshot must not trigger decoding")

    monkeypatch.setattr(technical, "_run", unexpected)
    with pytest.raises(ReplayError, match="input snapshot changed"):
        audit.observe(value.example.config, directory, video_contract="h264")


def test_h264_interrupted_scan_cannot_adopt_under_other_contract(
    h264_observation: RealObservation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = h264_observation
    directory = value.example.root / "h264-interrupted"
    directory.mkdir()
    write(directory / "input-snapshot.json", audit._snapshot(value.plan, video_contract="h264"))
    technical._run(
        audit.observation_command(value.plan), directory=directory, name="combined", timeout=30
    )

    def unexpected(*args: object, **kwargs: object) -> Path:
        pytest.fail("Finished scans must be adopted without another decode")

    monkeypatch.setattr(technical, "_run", unexpected)
    with pytest.raises(ReplayError, match="during the one decode"):
        audit.observe(value.example.config, directory)
    assert not (directory / "observation.json").exists()
    assert (
        audit.observe(value.example.config, directory, video_contract="h264")["fullDecode"] is True
    )


def test_cli_derived_audio_uses_same_observation_without_extra_decoder(
    real_observation: RealObservation,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = real_observation
    bind_source_audio_observations(value)
    directory = value.example.root / "cli-observed"
    assert audit.main([str(value.example.config), str(directory), "--observe"]) == 0

    def unexpected(*args: object, **kwargs: object) -> Path:
        pytest.fail("Saved technical/audio domains must never launch another decoder")

    monkeypatch.setattr(technical, "_run", unexpected)
    for flag in ("--technical-result", "--audio-result"):
        target = value.example.root / (flag.removeprefix("--") + ".json")
        assert audit.main([str(value.example.config), str(directory), str(target), flag]) == 0
        if flag == "--audio-result":
            result = read(target)
            assert result["sourceClockPreserved"] is True
            assert result["orderedPayloadEqual"] is True
            assert result["candidateMuxReady"] is True
            assert result["syncApproved"] is False
    assert "decodePasses" in capsys.readouterr().out
    with pytest.raises(ReplayError, match="output path"):
        audit.main([str(value.example.config), str(directory), "--technical-result"])


@pytest.mark.parametrize(
    "change",
    [
        "manifest",
        "process-path",
        "process-return",
        "process-command",
        "stderr",
        "raw-path",
        "raw-bytes",
        "movie-bytes",
    ],
)
def test_saved_observation_rejects_changed_provenance_without_redecode(
    real_observation: RealObservation, change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = real_observation
    directory = value.example.root / "saved-observed"
    audit.observe(value.example.config, directory)
    manifest = read(directory / "observation.json")
    if change == "manifest":
        manifest["version"] = 4
    elif change == "process-path":
        manifest["process"] = pin(value.example.root / "synthetic-observation.process.json")
    elif change in {"process-return", "process-command"}:
        process_path = directory / "combined.process.json"
        process = read(process_path)
        process["returncode" if change == "process-return" else "command"] = 1
        manifest["process"] = write(process_path, process)
    elif change == "stderr":
        (directory / "combined.stderr").write_bytes(b"decoder error")
    elif change == "raw-path":
        manifest["raw"] = pin(value.example.root / "synthetic-observation.stdout")
    elif change == "raw-bytes":
        path = directory / "combined.stdout"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        value.plan.output.write_bytes(value.plan.output.read_bytes() + b"changed")
    write(directory / "observation.json", manifest)

    def unexpected(*args: object, **kwargs: object) -> Path:
        pytest.fail("Failed provenance is not permission to re-decode or overwrite")

    monkeypatch.setattr(technical, "_run", unexpected)
    with pytest.raises((ReplayError, ValueError)):
        audit.observe(value.example.config, directory)


def test_changed_original_is_rejected_before_observation(real_observation: RealObservation) -> None:
    value = real_observation
    master = value.example.root / "master-1.mov"
    master.write_bytes(b"changed native original")
    with pytest.raises(ReplayError, match="source/tool/evidence changed"):
        audit.observe(value.example.config, value.example.root / "changed-before")


def test_movie_change_during_scan_never_publishes_observation(
    real_observation: RealObservation, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = real_observation
    original_run = technical._run

    def changed(command: list[str], *, directory: Path, name: str, timeout: int) -> Path:
        output = original_run(command, directory=directory, name=name, timeout=timeout)
        value.plan.output.write_bytes(value.plan.output.read_bytes() + b"changed during decoder")
        return output

    monkeypatch.setattr(technical, "_run", changed)
    directory = value.example.root / "changed-during"
    with pytest.raises(ReplayError, match="during the one decode"):
        audit.observe(value.example.config, directory)
    assert not (directory / "observation.json").exists()


def test_derived_result_nonoverwrite_is_checked(real_observation: RealObservation) -> None:
    value = real_observation
    directory = value.example.root / "nonoverwrite-observed"
    audit.observe(value.example.config, directory)
    target = value.example.root / "wrong-derived.json"
    write(target, {"technicalAccepted": False})
    with pytest.raises(ReplayError, match="derived whole domain result differs"):
        audit.observed_result(value.example.config, directory, target, audio=False)


def test_completed_scan_interrupted_before_manifest_is_adopted_without_redecode(
    real_observation: RealObservation, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = real_observation
    directory = value.example.root / "interrupted-manifest"
    directory.mkdir()
    write(directory / "input-snapshot.json", audit._snapshot(value.plan))
    technical._run(
        audit.observation_command(value.plan), directory=directory, name="combined", timeout=30
    )

    def unexpected(*args: object, **kwargs: object) -> Path:
        pytest.fail("A complete input-bound scan is adopted without another decoder")

    monkeypatch.setattr(technical, "_run", unexpected)
    assert audit.observe(value.example.config, directory)["fullDecode"] is True
    assert (directory / "observation.json").is_file()


@pytest.mark.parametrize(
    "field,frames",
    [
        ("pts", [{"pts": 2112, "nb_samples": 960}, {"pts": 3072, "nb_samples": 1024}]),
        (
            "best_effort_timestamp",
            [
                {"best_effort_timestamp": 2112, "nb_samples": 960},
                {"best_effort_timestamp": 3072, "nb_samples": 1024},
            ],
        ),
    ],
)
def test_explicit_observed_audio_formats_preserve_the_complete_clock(
    field: str, frames: list[dict[str, object]]
) -> None:
    probe: dict[str, object] = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "sample_rate": "48000",
                "time_base": "1/48000",
            }
        ],
        "frames": frames,
    }
    assert audit._source_clock(
        probe, timestamp_field=audit._source_timestamp_field(field), channels=2
    ) == AudioFrameClock((2112, 3072), (960, 1024), 48000, Fraction(1, 48000))
    assert frames == probe["frames"]


@pytest.mark.parametrize(
    "frame",
    [
        {"nb_samples": 1024},
        {"best_effort_timestamp": True, "nb_samples": 1024},
        {"best_effort_timestamp": 0.0, "nb_samples": 1024},
        {"best_effort_timestamp": -1, "nb_samples": 1024},
        {"best_effort_timestamp": 0, "nb_samples": True},
        {"best_effort_timestamp": 0, "nb_samples": 0},
        {"best_effort_timestamp": 0},
        {"pts": 1, "best_effort_timestamp": 0, "nb_samples": 1024},
        {"pts": False, "best_effort_timestamp": 0, "nb_samples": 1024},
        {"pts": 0, "best_effort_timestamp": "0", "nb_samples": 1024},
    ],
)
def test_explicit_source_audio_rejects_missing_invalid_or_divergent_clock(
    frame: dict[str, object],
) -> None:
    probe: dict[str, object] = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "sample_rate": "48000",
                "time_base": "1/48000",
            }
        ],
        "frames": [frame],
    }
    with pytest.raises((ReplayError, ValueError)):
        audit._source_clock(probe, timestamp_field="best_effort_timestamp", channels=2)


@pytest.mark.parametrize(
    "stream",
    [
        {"codec_type": "video"},
        {"codec_name": "mp3"},
        {"channels": 1},
        {"channels": True},
        {"sample_rate": "48000.0"},
        {"sample_rate": "0"},
        {"time_base": "0/1"},
        {"time_base": "missing"},
    ],
)
def test_explicit_source_audio_requires_native_stream_rate_base_and_channels(
    stream: dict[str, object],
) -> None:
    probe = {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "sample_rate": "48000",
                "time_base": "1/48000",
                **stream,
            }
        ],
        "frames": [{"best_effort_timestamp": 2112, "nb_samples": 960}],
    }
    with pytest.raises((ReplayError, ValueError)):
        audit._source_clock(probe, timestamp_field="best_effort_timestamp", channels=2)


def test_default_pts_is_not_a_silent_best_effort_fallback() -> None:
    with pytest.raises(ReplayError, match="selected source audio timestamp"):
        audit._source_clock(
            {
                "streams": [{"sample_rate": "48000", "time_base": "1/48000"}],
                "frames": [{"best_effort_timestamp": 2112, "nb_samples": 960}],
            },
            timestamp_field="pts",
            channels=2,
        )
    with pytest.raises(ReplayError, match="timestamp formats disagree"):
        audit._source_clock(
            {
                "streams": [{"sample_rate": "48000", "time_base": "1/48000"}],
                "frames": [{"pts": 0, "best_effort_timestamp": 1, "nb_samples": 960}],
            },
            timestamp_field="pts",
            channels=2,
        )
    with pytest.raises(ReplayError, match="Unsupported explicit source"):
        audit._source_timestamp_field("inferred")


@pytest.fixture
def historical_reader_observation(real_observation: RealObservation) -> tuple[Path, Path]:
    value = real_observation
    bind_source_audio_observations(value)
    return bind_historical_observation(value)


def bind_historical_observation(value: RealObservation) -> tuple[Path, Path]:
    evidence = value.example.root / "historical-reader-observed"
    audit.observe(value.example.config, evidence)
    archive = value.example.root / "historical-reader.py.bytes"
    archive.write_bytes(b"Historical producer reader bytes; evidence only.\n")
    source = Path(audit.__file__).resolve()
    manifest = read(evidence / "observation.json")
    snapshot = read(evidence / "input-snapshot.json")
    inputs = manifest["inputs"]
    assert isinstance(inputs, list)
    changed = [
        {"path": str(source), "sha256": file_digest(archive)}
        if record(row)["path"] == str(source)
        else row
        for row in inputs
    ]
    snapshot["inputs"] = changed
    manifest["inputs"] = changed
    manifest["snapshot"] = write(evidence / "input-snapshot.json", snapshot)
    observation_ref = write(evidence / "observation.json", manifest)
    binding = value.example.root / "historical-reader-binding.json"
    write(
        binding,
        {
            "version": 1,
            "observation": observation_ref,
            "producerReader": {"sourcePath": str(source), "historicalBytes": pin(archive)},
            "currentReader": pin(source),
        },
    )
    return evidence, binding


def test_explicit_historical_reader_reuses_original_decode_and_keeps_provenance(
    real_observation: RealObservation,
    historical_reader_observation: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, binding = historical_reader_observation
    raw_before = (evidence / "combined.stdout").read_bytes()
    manifest_before = (evidence / "observation.json").read_bytes()

    def unexpected(*args: object, **kwargs: object) -> Path:
        pytest.fail("Historical reader compatibility must not decode again")

    monkeypatch.setattr(technical, "_run", unexpected)
    target = real_observation.example.root / "historical-reader-audio.json"
    with pytest.raises(ReplayError, match="bound to current inputs"):
        audit.observed_result(real_observation.example.config, evidence, target, audio=True)
    reference = delivery._ref(pin(binding))
    result = audit.observed_result(
        real_observation.example.config,
        evidence,
        target,
        audio=True,
        producer_binding=reference,
    )
    assert result["candidateMuxReady"] is True
    assert result["sourceClockPreserved"] is True
    assert result["sourceClockTimestampField"] == "pts"
    assert record(result["readerProvenance"])["decoderReused"] is True
    assert result["deliveryApproved"] is False
    assert (evidence / "combined.stdout").read_bytes() == raw_before
    assert (evidence / "observation.json").read_bytes() == manifest_before
    assert (
        audit.observed_result(
            real_observation.example.config,
            evidence,
            target,
            audio=True,
            producer_binding=reference,
        )
        == result
    )


def test_best_effort_historical_source_cli_preserves_actual_clock_without_decode(
    real_observation: RealObservation,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    value = real_observation
    bind_source_audio_observations(value)
    audio = record(read(value.example.config)["audio"])
    proof = read(Path(str(record(audio["evidence"])["path"])))
    source_path = Path(str(record(proof["decodedProbe"])["path"]))
    source = read(source_path)
    source["frames"] = [
        {
            **{key: item for key, item in row.items() if key != "pts"},
            "best_effort_timestamp": row["pts"],
        }
        for row in audit._rows(source, "frames")
    ]
    value.example.audio("decodedProbe", source)
    original = source_path.read_bytes()
    evidence, binding = bind_historical_observation(value)

    def unexpected(*args: object, **kwargs: object) -> Path:
        pytest.fail("The explicit clock reader only consumes retained observations")

    monkeypatch.setattr(technical, "_run", unexpected)
    target = value.example.root / "best-effort-historical-audio.json"
    assert (
        audit.main(
            [
                str(value.example.config),
                str(evidence),
                str(target),
                "--audio-result",
                "--source-timestamp-field",
                "best_effort_timestamp",
                "--producer-binding",
                str(binding),
                "--producer-binding-sha256",
                file_digest(binding),
            ]
        )
        == 0
    )
    result = read(target)
    assert result["candidateMuxReady"] is True
    assert result["sourceClockPreserved"] is True
    assert result["sourceClockTimestampField"] == "best_effort_timestamp"
    assert record(result["readerProvenance"])["decoderReused"] is True
    assert result["syncApproved"] is False
    assert source_path.read_bytes() == original
    assert "best_effort_timestamp" in capsys.readouterr().out


@pytest.mark.parametrize(
    "damage",
    [
        "archive",
        "reader",
        "observation",
        "source",
        "version",
        "extra",
        "movie",
        "tool",
        "raw",
        "snapshot",
        "process",
    ],
)
def test_explicit_historical_reader_cannot_hide_other_input_mutations(
    real_observation: RealObservation,
    historical_reader_observation: tuple[Path, Path],
    damage: str,
) -> None:
    evidence, binding_path = historical_reader_observation
    binding = read(binding_path)
    if damage == "archive":
        archive = Path(str(record(record(binding["producerReader"])["historicalBytes"])["path"]))
        archive.write_bytes(b"wrong archive")
    elif damage == "reader":
        binding["currentReader"] = pin(binding_path)
    elif damage == "observation":
        binding["observation"] = pin(binding_path)
    elif damage == "source":
        record(binding["producerReader"])["sourcePath"] = str(binding_path)
    elif damage == "version":
        binding["version"] = True
    elif damage == "extra":
        binding["extra"] = True
    elif damage == "movie":
        real_observation.plan.output.write_bytes(b"changed movie")
    elif damage == "tool":
        real_observation.example.change("rate", 60)
    elif damage == "raw":
        (evidence / "combined.stdout").write_bytes(b"changed raw")
    elif damage == "snapshot":
        (evidence / "input-snapshot.json").write_bytes(b"changed snapshot")
    else:
        (evidence / "combined.process.json").write_bytes(b"changed process")
    write(binding_path, binding)
    with pytest.raises((ReplayError, ValueError)):
        audit.observed_result(
            real_observation.example.config,
            evidence,
            real_observation.example.root / "refused-historical-audio.json",
            audio=True,
            producer_binding=delivery._ref(pin(binding_path)),
        )


@pytest.mark.parametrize(
    "options",
    [
        ["--producer-binding", "missing.json"],
        ["--producer-binding-sha256", "0" * 64],
        ["--observe", "--source-timestamp-field", "best_effort_timestamp"],
        ["--technical-result", "--source-timestamp-field", "best_effort_timestamp"],
    ],
)
def test_cli_requires_explicit_complete_derived_clock_binding(
    real_observation: RealObservation, options: list[str]
) -> None:
    with pytest.raises(ReplayError):
        audit.main(
            [
                str(real_observation.example.config),
                str(real_observation.example.root / "unused"),
                *options,
            ]
        )


@pytest.mark.parametrize(
    "artifact",
    ["combined.stdout", "input-snapshot.json", "combined.process.json", "combined.stderr"],
)
def test_historical_artifact_mutation_during_derived_domain_is_rejected(
    real_observation: RealObservation,
    historical_reader_observation: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    evidence, binding = historical_reader_observation
    original = audit.compare

    def changed(
        plan: delivery.Plan,
        receipt: dict[str, object],
        streams: dict[str, object],
        frames: dict[str, object],
        packets: dict[str, object],
        *,
        video_contract: str = "hevc",
        source_timestamp_field: str = "pts",
        normalization_binding: Reference | None = None,
    ) -> dict[str, object]:
        result = original(
            plan,
            receipt,
            streams,
            frames,
            packets,
            video_contract=video_contract,
            source_timestamp_field=source_timestamp_field,
            normalization_binding=normalization_binding,
        )
        (evidence / artifact).write_bytes(b"changed after initial validation")
        return result

    monkeypatch.setattr(audit, "compare", changed)
    output = real_observation.example.root / "changed-during-domain.json"
    with pytest.raises(ReplayError, match="artifacts changed|process failed"):
        audit.observed_result(
            real_observation.example.config,
            evidence,
            output,
            audio=True,
            producer_binding=delivery._ref(pin(binding)),
        )
    assert not output.exists()


def test_failed_scan_is_not_relaunched_or_adopted(
    real_observation: RealObservation, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = real_observation
    directory = value.example.root / "failed-scan"
    directory.mkdir()
    write(directory / "input-snapshot.json", audit._snapshot(value.plan))
    write(
        directory / "combined.process.json",
        {"command": audit.observation_command(value.plan), "returncode": 1},
    )
    (directory / "combined.stderr").write_bytes(b"failed decoder")

    def unexpected(*args: object, **kwargs: object) -> Path:
        pytest.fail("A failed decoder cannot be retried implicitly")

    monkeypatch.setattr(technical, "_run", unexpected)
    with pytest.raises(ReplayError, match="process failed"):
        audit.observe(value.example.config, directory)


@pytest.mark.parametrize(
    "content", ["metadata-only", "packets-only", "decoded-error", "unknown-type"]
)
def test_observation_requires_actual_full_decoded_evidence(
    real_observation: RealObservation, content: str
) -> None:
    value = real_observation
    raw = value.raw
    rows = list(audit._rows(raw, "packets_and_frames"))
    if content == "metadata-only":
        raw.pop("packets_and_frames")
    elif content == "packets-only":
        raw["packets_and_frames"] = [row for row in rows if row["type"] == "packet"]
    elif content == "decoded-error":
        next(row for row in rows if row["type"] == "frame")["decode_error_flags"] = 1
    else:
        rows[0]["type"] = "unknown"
    with pytest.raises(ReplayError):
        audit.whole_observations(value.plan, raw)


@dataclass(frozen=True)
class Verification:
    example: Example
    evidence: Path
    streams: dict[str, object]
    frames: dict[str, object]
    packets: dict[str, object]
    receipt: dict[str, object]


@pytest.fixture
def verification(example: Example, monkeypatch: pytest.MonkeyPatch) -> Verification:
    packet_path = example.root / "audio-packets.json"
    packets = read(packet_path)
    rows = packets["packets"]
    assert isinstance(rows, list)
    record(rows[0])["data_hash"] = "SHA256:" + "a" * 64
    record(rows[1])["data_hash"] = "SHA256:" + "b" * 64
    example.audio("packetProbe", packets)
    plan = delivery.load(example.config)
    plan.output.write_bytes(b"copy-assembled synthetic video/audio")
    evidence = example.root / "technical"
    evidence.mkdir()
    streams: dict[str, object] = {
        "streams": [{"codec_type": "audio", "sample_rate": "48000", "time_base": "1/48000"}]
    }
    frames: dict[str, object] = {
        "frames": [
            {"media_type": "audio", "pts": 0, "nb_samples": 24000},
            {"media_type": "audio", "pts": 24000, "nb_samples": 24000},
            {"media_type": "video", "pts": 0, "duration": 100},
        ]
    }
    receipt: dict[str, object] = {
        "technicalAccepted": True,
        "fullDecode": True,
        "inputs": [{"path": str(plan.output), "sha256": file_digest(plan.output)}],
    }
    write(evidence / "receipt.json", receipt)
    write(evidence / "streams.stdout", streams)
    write(evidence / "frames.stdout", frames)

    def run(command: list[str], *, directory: Path, name: str, timeout: int) -> Path:
        assert "-show_data_hash" in command
        assert "skip_samples,discard_padding" in " ".join(command)
        assert command[-1] == str(plan.output)
        assert timeout == 7200
        path = directory / f"{name}.stdout"
        write(path, packets)
        return path

    monkeypatch.setattr(technical, "_run", run)
    return Verification(example, evidence, streams, frames, packets, receipt)


def compare(value: Verification) -> dict[str, object]:
    return audit.compare(
        delivery.load(value.example.config),
        value.receipt,
        value.streams,
        value.frames,
        value.packets,
    )


def test_full_clock_and_payload_comparison_never_approves_sync(verification: Verification) -> None:
    result = compare(verification)
    assert result["audioMuxAccepted"] is True
    assert result["orderedPayloadEqual"] is True
    assert result["decodedSamples"] == 48000
    assert result["qualityApproved"] is False
    assert result["syncApproved"] is False
    assert result["deliveryApproved"] is False


@pytest.mark.parametrize("selector", ["videoContract", "expectedVideoContract"])
def test_explicit_h264_legacy_receipt_is_bound(verification: Verification, selector: str) -> None:
    value = verification
    value.receipt[selector] = "h264"
    result = audit.compare(
        delivery.load(value.example.config),
        value.receipt,
        value.streams,
        value.frames,
        value.packets,
        video_contract="h264",
    )
    assert result["audioMuxAccepted"] is True
    assert result["videoContract"] == "h264"
    with pytest.raises(ReplayError, match="contract differs"):
        compare(value)


def test_unlabelled_legacy_receipt_cannot_authorize_h264(verification: Verification) -> None:
    value = verification
    with pytest.raises(ReplayError, match="lacks an explicit"):
        audit.verify(
            value.example.config,
            value.evidence,
            value.example.root / "wrong-contract",
            video_contract="h264",
        )
    assert not (value.example.root / "wrong-contract").exists()
    assert compare(value)["videoContract"] == "hevc"


@pytest.mark.parametrize("value", [None, 3, "unknown", "H264"])
def test_selector_rejects_unknown_values_before_reading_config(value: object) -> None:
    with pytest.raises(ReplayError, match="Unsupported explicit"):
        audit._video_contract(value)


def test_version_three_receipt_requires_selector(verification: Verification) -> None:
    verification.receipt["version"] = 3
    with pytest.raises(ReplayError, match="lacks an explicit"):
        compare(verification)


def test_cli_unknown_selector_is_rejected_before_output(
    verification: Verification,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = verification.example.root / "unknown-contract"
    with pytest.raises(SystemExit) as stopped:
        audit.main(
            [
                str(verification.example.config),
                str(verification.evidence),
                str(output),
                "--video-contract",
                "vp9",
            ]
        )
    assert stopped.value.code == 2
    assert not output.exists()
    assert "invalid choice" in capsys.readouterr().err


def test_conflicting_receipt_selectors_are_rejected(verification: Verification) -> None:
    verification.receipt.update(videoContract="hevc", expectedVideoContract="h264")
    with pytest.raises(ReplayError, match="contract differs"):
        compare(verification)


@pytest.mark.parametrize("field", ["technicalAccepted", "fullDecode"])
def test_false_technical_result_remains_rejected(verification: Verification, field: str) -> None:
    verification.receipt[field] = False
    assert compare(verification)["route"] == "prores-fallback"


def test_different_decoded_segmentation_and_gap_are_rejected(verification: Verification) -> None:
    verification.frames["frames"] = [
        {"media_type": "audio", "pts": 0, "nb_samples": 24000},
        {"media_type": "audio", "pts": 24064, "nb_samples": 24000},
    ]
    result = compare(verification)
    assert result["audioMuxAccepted"] is False
    assert "PTS/sample counts" in str(result["reasons"])
    assert "end 751/750" in str(result["reasons"])


def test_same_payload_with_lost_skip_and_discard_is_rejected(verification: Verification) -> None:
    verification.packets["packets"] = [
        {"pts": 0, "duration": 24064, "data_hash": "SHA256:" + "a" * 64},
        {"pts": 24064, "duration": 24064, "data_hash": "SHA256:" + "b" * 64},
    ]
    result = compare(verification)
    assert result["orderedPayloadEqual"] is True
    assert result["audioMuxAccepted"] is False
    assert "priming/remainder" in str(result["reasons"])


def test_changed_packet_payload_is_rejected(verification: Verification) -> None:
    rows = verification.packets["packets"]
    assert isinstance(rows, list)
    record(rows[0])["data_hash"] = "SHA256:" + "c" * 64
    result = compare(verification)
    assert result["orderedPayloadEqual"] is False
    assert "ordered AAC" in str(result["reasons"])


@pytest.mark.parametrize("value", [None, "sha256:abc", "SHA256:" + "x" * 64])
def test_missing_or_invalid_payload_observations_are_not_accepted(
    verification: Verification, value: object
) -> None:
    rows = verification.packets["packets"]
    assert isinstance(rows, list)
    record(rows[0])["data_hash"] = value
    with pytest.raises(ReplayError, match="payload SHA256"):
        compare(verification)


@pytest.mark.parametrize("rows", [[], None, "rows", [None]])
def test_probe_row_boundaries_are_validated(rows: object) -> None:
    with pytest.raises((ReplayError, ValueError)):
        audit._rows({"packets": rows}, "packets")


@pytest.mark.parametrize(
    "change",
    [{"inputs": []}, {"inputs": None}, {"inputs": [{"path": "/different", "sha256": "a" * 64}]}],
)
def test_unbound_technical_result_cannot_be_reused(
    verification: Verification, change: dict[str, object]
) -> None:
    verification.receipt.update(change)
    with pytest.raises(ReplayError, match="bind"):
        compare(verification)


def test_multiple_audio_tracks_are_rejected(verification: Verification) -> None:
    verification.streams["streams"] = [
        {"codec_type": "audio", "sample_rate": "48000", "time_base": "1/48000"},
        {"codec_type": "audio", "sample_rate": "48000", "time_base": "1/48000"},
    ]
    with pytest.raises(ReplayError, match="exactly one"):
        compare(verification)


def test_missing_whole_audio_proof_is_rejected(verification: Verification) -> None:
    verification.example.change("audio", None)
    with pytest.raises(ReplayError, match="proof"):
        compare(verification)


def test_verify_collects_bounded_observations_and_exclusive_result(
    verification: Verification,
) -> None:
    output = verification.example.root / "audit"
    result = audit.verify(verification.example.config, verification.evidence, output)
    assert result["audioMuxAccepted"] is True
    assert read(output / "result.json")["deliveryApproved"] is False
    assert (output / "packets.stdout").is_file()
    with pytest.raises(FileExistsError):
        audit.verify(verification.example.config, verification.evidence, output)


def test_cli_reports_actual_result(
    verification: Verification, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        audit.main(
            [
                str(verification.example.config),
                str(verification.evidence),
                str(verification.example.root / "cli-audit"),
            ]
        )
        == 0
    )
    assert "audioMuxAccepted" in capsys.readouterr().out


def test_cli_rejected_clock_is_nonzero(
    verification: Verification, capsys: pytest.CaptureFixture[str]
) -> None:
    verification.frames["frames"] = [{"media_type": "audio", "pts": 0, "nb_samples": 46400}]
    write(verification.evidence / "frames.stdout", verification.frames)
    assert (
        audit.main(
            [
                str(verification.example.config),
                str(verification.evidence),
                str(verification.example.root / "failed-cli-audit"),
            ]
        )
        == 1
    )
    assert "prores-fallback" in capsys.readouterr().out


def test_probe_reader_rejects_duplicate_keys_and_large_artifact(
    verification: Verification, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = verification.example.root / "bad-probe.json"
    path.write_text('{"x":1,"x":1}', encoding="utf-8")
    with pytest.raises(ValueError):
        audit._probe(path)
    monkeypatch.setattr(audit, "MAX_PROBE_BYTES", 2)
    path.write_text(json.dumps({"frames": [1]}), encoding="utf-8")
    with pytest.raises(ValueError):
        audit._probe(path)


def test_candidate_allows_only_preserved_observed_endpoint_limit(
    verification: Verification,
) -> None:
    verification.example.change("allowClockCandidate", True)
    shifted_frames = [
        {"media_type": "audio", "pts": 64, "nb_samples": 24000},
        {"media_type": "audio", "pts": 24064, "nb_samples": 24000},
    ]
    verification.frames["frames"] = shifted_frames
    verification.example.audio("decodedProbe", {"frames": shifted_frames})
    verification.packets["packets"] = [
        {
            "pts": 0,
            "duration": 24064,
            "data_hash": "SHA256:" + "a" * 64,
            "side_data_list": [
                {"side_data_type": "Skip Samples", "skip_samples": 64, "discard_padding": 0}
            ],
        },
        {
            "pts": 24064,
            "duration": 24064,
            "data_hash": "SHA256:" + "b" * 64,
            "side_data_list": [
                {"side_data_type": "Skip Samples", "skip_samples": 0, "discard_padding": 64}
            ],
        },
    ]
    verification.example.audio("packetProbe", verification.packets)
    result = compare(verification)
    assert result["candidateMuxReady"] is True
    assert result["audioMuxAccepted"] is False
    assert result["route"] == "compressed-clock-candidate"
    assert result["syncApproved"] is False
    assert result["deliveryApproved"] is False
    assert "751/750" in str(result["reasons"])


@pytest.mark.parametrize("failure", ["gap", "payload", "technical", "priming"])
def test_candidate_cannot_bypass_mux_integrity(verification: Verification, failure: str) -> None:
    verification.example.change("allowClockCandidate", True)
    if failure == "gap":
        verification.frames["frames"] = [
            {"media_type": "audio", "pts": 0, "nb_samples": 24000},
            {"media_type": "audio", "pts": 24064, "nb_samples": 24000},
        ]
    elif failure == "payload":
        rows = verification.packets["packets"]
        assert isinstance(rows, list)
        record(rows[0])["data_hash"] = "SHA256:" + "c" * 64
    elif failure == "technical":
        verification.receipt["fullDecode"] = False
    else:
        verification.packets["packets"] = [
            {"pts": 0, "duration": 24064, "data_hash": "SHA256:" + "a" * 64},
            {"pts": 24064, "duration": 24064, "data_hash": "SHA256:" + "b" * 64},
        ]
    result = compare(verification)
    assert result["candidateMuxReady"] is False
    assert result["audioMuxAccepted"] is False


def normalized_fixture(
    verification: Verification,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[normalization.Normalization, Reference]:
    verification.example.audio(
        "decodedProbe",
        {
            "frames": [{"pts": 64, "nb_samples": 24000}, {"pts": 24064, "nb_samples": 24000}],
        },
    )
    plan = delivery.load(verification.example.config)
    assert plan.audio is not None
    proof = delivery._read(delivery._ref(plan.audio["evidence"]))
    value = normalization.Normalization(
        AudioFrameClock((64, 24064), (24000, 24000), 48000, Fraction(1, 48000)),
        AudioFrameClock((0, 24000), (24000, 24000), 48000, Fraction(1, 48000)),
        {
            "inputs": {
                "audio": plan.audio["movie"],
                "packetProbe": proof["packetProbe"],
                "decodedProbe": proof["decodedProbe"],
                "ffmpeg": plan.tools["ffmpeg"].json(),
            },
            "output": str(plan.output),
            "timestampField": "pts",
            "primingSamples": 64,
            "preRollTicks": 64,
        },
    )
    ref_path = verification.example.root / "normalization.json"
    write(ref_path, {"synthetic": True})
    reference = Reference(ref_path, file_digest(ref_path))

    def verify(_proof: Reference) -> normalization.Normalization:
        return value

    monkeypatch.setattr(normalization, "verify", verify)
    return value, reference


def test_explicit_normalization_preserves_source_under_declared_mapping(
    verification: Verification,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, reference = normalized_fixture(verification, monkeypatch)
    result = audit.compare(
        delivery.load(verification.example.config),
        verification.receipt,
        verification.streams,
        verification.frames,
        verification.packets,
        normalization_binding=reference,
    )
    assert result["audioMuxAccepted"] is True
    assert result["sourceClockPreserved"] is True
    assert result["presentationNormalization"] == {
        "proof": reference.json(),
        "primingSamples": 64,
        "preRollTicks": 64,
        "sourceClockPreservedUnderDeclaredMapping": True,
    }
    assert result["syncApproved"] is False
    assert value.source.pts == (64, 24064)
    assert compare(verification)["sourceClockPreserved"] is False


@pytest.mark.parametrize(
    "key", ["audio", "packetProbe", "decodedProbe", "ffmpeg", "output", "timestampField"]
)
def test_normalization_cannot_bind_another_source_or_producer(
    verification: Verification,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    value, reference = normalized_fixture(verification, monkeypatch)
    if key in {"output", "timestampField"}:
        value.document[key] = "different"
    else:
        record(value.document["inputs"])[key] = {"path": "/different", "sha256": "a" * 64}
    with pytest.raises(ReplayError, match="bind the adopted"):
        audit.compare(
            delivery.load(verification.example.config),
            verification.receipt,
            verification.streams,
            verification.frames,
            verification.packets,
            normalization_binding=reference,
        )


def test_normalization_rechecks_proof_at_end_of_derived_domain(
    verification: Verification,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, reference = normalized_fixture(verification, monkeypatch)

    def saved(
        _plan: delivery.Plan,
        _evidence: Path,
        *,
        video_contract: audit.VideoContract,
        producer_binding: Reference | None,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
        return verification.receipt, verification.streams, verification.frames, verification.packets

    monkeypatch.setattr(audit, "_saved_observation", saved)
    result = audit.observed_result(
        verification.example.config,
        verification.evidence,
        verification.example.root / "normalized-result.json",
        audio=True,
        normalization_binding=reference,
    )
    assert result["audioMuxAccepted"] is True
    assert value.normalized.pts == (0, 24000)
    with pytest.raises(ReplayError, match="timestamp field"):
        audit.observed_result(
            verification.example.config,
            verification.evidence,
            verification.example.root / "technical-normalized-result.json",
            audio=False,
            normalization_binding=reference,
        )


@pytest.mark.parametrize(
    "flags",
    [
        ["--normalization-binding", "/not-opened"],
        ["--normalization-sha256", "a" * 64],
        ["--normalization-binding", "/not-opened", "--normalization-sha256", "a" * 64, "--observe"],
    ],
)
def test_normalization_cli_requires_explicit_paired_audio_mode(flags: list[str]) -> None:
    with pytest.raises(ReplayError, match="normalization"):
        audit.main(["/unused", "/unused", *flags])
