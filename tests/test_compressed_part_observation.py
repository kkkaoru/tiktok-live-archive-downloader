"""Synthetic/mocked current observation contracts; no media or app processes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from replay import compressed_part_delivery as delivery
from replay import compressed_part_observation as observe
from replay import delivery_verification as technical
from replay import whole_technical_batch as metadata
from replay.core import ReplayError
from replay.input_hash_cache import Entry, InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest


def write(path: Path, value: object) -> metadata.Reference:
    path.write_text(json.dumps(value), encoding="utf-8")
    return metadata.Reference(path, file_digest(path))


def pin(path: Path) -> metadata.Reference:
    return metadata.Reference(path, file_digest(path))


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def dump(data: bytes) -> str:
    return "00000000: " + data.hex(" ", 2)


def conversion(world: World) -> observe.ConversionRequest:
    return observe.load_conversion(
        write(
            world.root / "conversion.json",
            {
                "version": 1,
                "kind": "compressed-part-conversion-request",
                "source": world.raw["source"],
                "sourceNative": world.raw["sourceNative"],
                "tool": record(world.raw["tools"])["ffmpeg"],
                "output": str(world.root / "converted.mp4"),
                "outputRoot": str(world.root),
                "frames": world.raw["frames"],
                "rate": world.raw["rate"],
                "width": 64,
                "height": 96,
                "bitrate": 10_000_000,
                "pixelFormat": "yuv420p",
                "timeoutSeconds": 7200,
            },
        )
    )


def producer(
    world: World, monkeypatch: pytest.MonkeyPatch, *, effect: Callable[[Path], None] | None = None
) -> None:
    def run(command: list[str], *, directory: Path, name: str, timeout: int) -> Path:
        if name != "convert":
            return world.run(command, directory=directory, name=name, timeout=timeout)
        world.calls.append(name)
        assert timeout == 7200
        assert command[command.index("-c:v") + 1] == "hevc_videotoolbox"
        assert command[command.index("-b:v") + 1] == "10000000"
        assert "-n" in command and "-an" in command and "-fps_mode" in command
        Path(command[-1]).write_bytes(b"synthetic converted output")
        (directory / "convert.stdout").write_bytes(b"")
        (directory / "convert.stderr").write_bytes(b"")
        write(directory / "convert.process.json", {"command": command, "returncode": 0})
        if effect is not None:
            effect(directory)
        return directory / "convert.stdout"

    monkeypatch.setattr(technical, "_run", run)


def converted_request(world: World, monkeypatch: pytest.MonkeyPatch) -> observe.Request:
    producer(world, monkeypatch)
    request = conversion(world)
    observe.convert(request, world.root / "producer")
    world.raw["kind"] = "converted-compressed-part-observation-request"
    world.raw["movie"] = pin(request.output).json()
    world.raw["converterExecution"] = pin(world.root / "producer/execution.json").json()
    return world.request()


@dataclass
class World:
    root: Path
    config: Path
    raw: dict[str, object]
    video: dict[str, object]
    first: dict[str, object]
    calls: list[str]
    effect: Callable[[Path], None] | None = None

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    def request(self) -> observe.Request:
        return observe.load(write(self.config, self.raw))

    def run(self, command: list[str], *, directory: Path, name: str, timeout: int) -> Path:
        self.calls.append(name)
        assert timeout == 7200
        payload: object = {
            "video": self.video,
            "first": self.first,
            "quality": None,
            "full-av": None,
        }[name]
        output = directory / f"{name}.stdout"
        output.write_text("" if payload is None else json.dumps(payload), encoding="utf-8")
        (directory / f"{name}.stderr").write_bytes(b"")
        write(directory / f"{name}.process.json", {"command": command, "returncode": 0})
        if name == "quality":
            (directory / "quality-0.ssim.stats").write_text("n:1 All:0.999\n", encoding="utf-8")
            (directory / "quality-1.ssim.stats").write_text("n:1 All:0.998\n", encoding="utf-8")
            (directory / "quality-2.ssim.stats").write_text("n:1 All:0.997\n", encoding="utf-8")
            (directory / "quality-0.psnr.stats").write_text("n:1 psnr_avg:50\n", encoding="utf-8")
            (directory / "quality-1.psnr.stats").write_text("n:1 psnr_avg:49\n", encoding="utf-8")
            (directory / "quality-2.psnr.stats").write_text("n:1 psnr_avg:48\n", encoding="utf-8")
            if self.effect is not None:
                self.effect(directory)
        return output


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    source, movie, ffmpeg, ffprobe = (
        tmp_path / name for name in ("original.mov", "compact.mp4", "ffmpeg", "ffprobe")
    )
    source.write_bytes(b"synthetic original bytes")
    movie.write_bytes(b"synthetic current compressed bytes")
    ffmpeg.write_bytes(b"fake executable ffmpeg")
    ffprobe.write_bytes(b"fake executable ffprobe")
    ffmpeg.chmod(0o700)
    ffprobe.chmod(0o700)
    source_native = write(
        tmp_path / "source-native.json",
        {
            "movie": pin(source).json(),
            "completionVerified": True,
            "shareObservedIdle": True,
            "fullAVDecodeExitCode": 0,
        },
    )
    extra = b"\x01" + bytes(20) + b"\x03\x00"
    video: dict[str, object] = {
        "codec_name": "hevc",
        "codec_tag_string": "hvc1",
        "profile": "Main",
        "level": 120,
        "width": 64,
        "height": 96,
        "pix_fmt": "yuv420p",
        "sample_aspect_ratio": "1:1",
        "field_order": "progressive",
        "color_range": "tv",
        "color_space": "bt709",
        "color_transfer": "bt709",
        "color_primaries": "bt709",
        "time_base": "1/3000",
        "extradata_hash": "SHA256:" + hashlib.sha256(extra).hexdigest(),
    }
    raw: dict[str, object] = {
        "version": 1,
        "kind": "compressed-part-observation-request",
        "source": pin(source).json(),
        "movie": pin(movie).json(),
        "sourceNative": source_native.json(),
        "tools": {"ffmpeg": pin(ffmpeg).json(), "ffprobe": pin(ffprobe).json()},
        "frames": 3,
        "rate": 30,
        "windowFrames": 1,
        "threads": 1,
        "videoConfiguration": video,
        "thresholds": {"ssim": 0.99, "psnr": 45},
    }
    stream = {
        **video,
        "index": 0,
        "codec_type": "video",
        "avg_frame_rate": "30/1",
        "start_pts": 0,
        "duration_ts": 300,
        "extradata_size": 23,
        "extradata": dump(extra),
    }
    frames = [
        {
            "media_type": "video",
            "stream_index": 0,
            "pts": index * 100,
            "best_effort_timestamp": index * 100,
            "duration": 100,
            "decode_error_flags": 0,
        }
        for index in range(3)
    ]
    data = b"\x00\x00\x00\x02\x28\x01"
    first: dict[str, object] = {
        "packets": [
            {
                "stream_index": 0,
                "pts": 0,
                "dts": -200,
                "size": 6,
                "flags": "K__",
                "data": dump(data),
                "data_hash": "SHA256:" + hashlib.sha256(data).hexdigest(),
            }
        ]
    }
    result = World(
        tmp_path, tmp_path / "request.json", raw, {"streams": [stream], "frames": frames}, first, []
    )
    monkeypatch.setattr(technical, "_run", result.run)
    return result


@pytest.mark.parametrize("frames,pixel", [(3, "yuv420p"), (2, "p010le")])
def test_converter_captures_real_process_and_never_reencodes_completed_output(
    world: World, monkeypatch: pytest.MonkeyPatch, frames: int, pixel: str
) -> None:
    producer(world, monkeypatch)
    request = conversion(world)
    raw = read(request.reference.path)
    raw.update(frames=frames, pixelFormat=pixel)
    request = observe.load_conversion(write(request.reference.path, raw))
    evidence = world.root / "producer"
    result = observe.convert(request, evidence)
    assert result["frames"] == frames
    assert result["historicalNativeJobVerified"] is False
    assert result["currentDeclaredInputsVerified"] is True
    assert result["sourceBeforeAfterIdentityEqual"] is True
    assert result["returncode"] == 0
    assert result["qualityApproved"] is False
    assert "nativeStatus" not in result
    previous = (evidence / "execution.json").stat()
    assert observe.convert(request, evidence) == result
    assert (evidence / "execution.json").stat() == previous
    assert world.calls == ["convert"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "unknown"),
        ("version", True),
        ("frames", 0),
        ("rate", 241),
        ("width", 0),
        ("height", 16385),
        ("bitrate", 0),
        ("timeoutSeconds", 7201),
        ("pixelFormat", "unknown"),
        ("unexpected", True),
    ],
)
def test_converter_rejects_invalid_intents_without_effect(
    world: World, field: str, value: object
) -> None:
    request = conversion(world)
    raw = read(request.reference.path)
    raw[field] = value
    with pytest.raises(ReplayError):
        observe.load_conversion(write(request.reference.path, raw))
    assert world.calls == []


@pytest.mark.parametrize("role", ["source", "tool", "sourceNative"])
def test_converter_refuses_changed_input_pin(
    world: World, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    producer(world, monkeypatch)
    request = conversion(world)
    ref = metadata._reference(read(request.reference.path)[role])
    ref.path.write_bytes(b"changed input")
    with pytest.raises(ReplayError, match="pin changed"):
        observe.convert(request, world.root / "producer")
    assert world.calls == []


def test_converter_partial_failure_is_preserved_and_not_retried(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(directory: Path) -> None:
        raise ReplayError("interrupted conversion")

    producer(world, monkeypatch, effect=fail)
    request = conversion(world)
    evidence = world.root / "producer"
    with pytest.raises(ReplayError, match="interrupted"):
        observe.convert(request, evidence)
    assert request.output.exists() and (evidence / "initial.json").exists()
    assert not (evidence / "execution.json").exists()
    with pytest.raises(OSError):
        observe.convert(request, evidence)
    assert world.calls == ["convert"]


def test_converter_changed_source_during_process_cannot_complete(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = metadata._reference(world.raw["source"])

    def change(directory: Path) -> None:
        source.path.write_bytes(b"source changed during process")

    producer(world, monkeypatch, effect=change)
    with pytest.raises(ReplayError, match="pin changed"):
        observe.convert(conversion(world), world.root / "producer")
    assert not (world.root / "producer/execution.json").exists()


@pytest.mark.parametrize("linked", [False, True])
def test_converter_refuses_occupied_or_linked_output_without_process(
    world: World, linked: bool
) -> None:
    request = conversion(world)
    if linked:
        request.output.symlink_to(metadata._reference(world.raw["movie"]).path)
    else:
        request.output.write_bytes(b"existing output")
    with pytest.raises(ReplayError):
        observe.convert(request, world.root / "producer")
    assert world.calls == []


@pytest.mark.parametrize(
    "name",
    ["initial.json", "terminal.json", "convert.process.json", "convert.stderr", "execution.json"],
)
def test_converter_verifier_rejects_tampered_evidence(
    world: World, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    producer(world, monkeypatch)
    request = conversion(world)
    evidence = world.root / "producer"
    observe.convert(request, evidence)
    receipt = pin(evidence / "execution.json")
    (evidence / name).write_bytes(b"tampered evidence")
    with pytest.raises((ReplayError, ValueError)):
        observe.verify_conversion(receipt)
    assert world.calls == ["convert"]


def test_converter_receipt_rejects_later_output_writer(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer(world, monkeypatch)
    request = conversion(world)
    evidence = world.root / "producer"
    observe.convert(request, evidence)
    request.output.write_bytes(b"later writer")
    with pytest.raises(ReplayError, match="output changed"):
        observe.verify_conversion(pin(evidence / "execution.json"))


def test_converter_same_bytes_changed_source_identity_is_rejected(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = metadata._reference(world.raw["source"])
    contents = source.path.read_bytes()

    def rewrite(directory: Path) -> None:
        source.path.write_bytes(contents)

    producer(world, monkeypatch, effect=rewrite)
    with pytest.raises(ReplayError, match="identity changed"):
        observe.convert(conversion(world), world.root / "producer")


def test_converter_transitive_code_mutation_is_rejected(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = world.root / "synthetic-transitive-reader.py"
    helper.write_bytes(b"initial reader")
    original = metadata.reader_code_paths

    def readers() -> tuple[Path, ...]:
        return (*original(), helper)

    monkeypatch.setattr(metadata, "reader_code_paths", readers)

    def change(directory: Path) -> None:
        helper.write_bytes(b"changed reader")

    producer(world, monkeypatch, effect=change)
    with pytest.raises(ReplayError, match="pin changed"):
        observe.convert(conversion(world), world.root / "producer")


def test_converter_missing_original_completion_cannot_start(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    native = metadata._reference(world.raw["sourceNative"])
    raw = read(native.path)
    raw["completionVerified"] = False
    world.raw["sourceNative"] = write(native.path, raw).json()
    producer(world, monkeypatch)
    with pytest.raises(ReplayError, match="original bound native"):
        observe.convert(conversion(world), world.root / "producer")
    assert world.calls == []


def test_converter_invalid_terminal_clock_is_not_adopted(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer(world, monkeypatch)
    evidence = world.root / "producer"
    observe.convert(conversion(world), evidence)
    receipt = pin(evidence / "execution.json")
    terminal = read(evidence / "terminal.json")
    terminal["endedAtNs"] = 0
    write(evidence / "terminal.json", terminal)
    with pytest.raises(ReplayError, match="timestamps"):
        observe.verify_conversion(receipt)


def test_converted_full_av_failure_cannot_create_observation(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = converted_request(world, monkeypatch)

    def failed(command: list[str], *, directory: Path, name: str, timeout: int) -> Path:
        result = world.run(command, directory=directory, name=name, timeout=timeout)
        if name == "full-av":
            write(directory / "full-av.process.json", {"command": command, "returncode": 1})
        return result

    monkeypatch.setattr(technical, "_run", failed)
    with pytest.raises(ReplayError, match="process"):
        observe.collect(request, world.evidence)
    assert not (world.evidence / "observation.json").exists()


def test_verified_converter_raw_observations_enter_existing_delivery_policy(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = converted_request(world, monkeypatch)
    observe.collect(request, world.evidence)
    evidence = world.root / "delivery-evidence.json"
    observe.delivery_evidence(request, world.evidence, evidence)
    part = delivery.Part(
        1,
        0,
        request.frames,
        request.source,
        request.movie,
        pin(evidence),
        delivery.CurrentOriginal(request.source),
    )
    plan = delivery.Plan(
        request.reference.path,
        request.rate,
        64,
        96,
        (part,),
        None,
        request.thresholds,
        request.reference,
        request.reference,
        world.root / "final.mp4",
        {"python": request.ffmpeg, "ffmpeg": request.ffmpeg, "ffprobe": request.ffprobe},
    )
    video, failures, windows = delivery._part_check(part, plan)
    assert video == request.video and failures == []
    assert len(windows) == 3
    _, failures, _ = delivery._part_check(
        part, replace(plan, thresholds={"ssim": 1.0, "psnr": 100.0})
    )
    assert len(failures) == 6
    assert request.source.path.exists()


def test_converter_cli_runs_once_and_reentry_is_readonly(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    producer(world, monkeypatch)
    request = conversion(world)
    evidence = world.root / "producer"
    assert (
        observe.main(
            [
                "convert",
                str(request.reference.path),
                str(evidence),
                "--sha256",
                request.reference.sha256,
            ]
        )
        == 0
    )
    assert record(json.loads(capsys.readouterr().out))["historicalNativeJobVerified"] is False
    receipt = pin(evidence / "execution.json")
    assert (
        observe.main(
            ["verify-conversion", str(receipt.path), str(evidence), "--sha256", receipt.sha256]
        )
        == 0
    )
    assert world.calls == ["convert"]
    with pytest.raises(SystemExit):
        observe.main(
            ["verify-conversion", str(receipt.path), str(world.root), "--sha256", receipt.sha256]
        )
    assert world.calls == ["convert"]


def test_converter_delivery_cli_requires_explicit_output(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request = converted_request(world, monkeypatch)
    observe.collect(request, world.evidence)
    args = [
        "delivery-evidence",
        str(request.reference.path),
        str(world.evidence),
        "--sha256",
        request.reference.sha256,
    ]
    with pytest.raises(SystemExit):
        observe.main(args)
    output = world.root / "delivery-evidence.json"
    assert observe.main([*args, "--output", str(output)]) == 0
    assert (
        record(json.loads(capsys.readouterr().out))["kind"] == "converted-compressed-part-evidence"
    )


def test_converted_observer_binds_producer_without_fabricating_native_status(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = converted_request(world, monkeypatch)
    result = observe.collect(request, world.evidence)
    assert result["conversionExecutionVerified"] is True
    assert result["historicalGenerationVerified"] is False
    assert (
        result["converterExecution"] == request.converter.json()
        if request.converter is not None
        else False
    )
    output = world.root / "delivery-evidence.json"
    evidence = observe.delivery_evidence(request, world.evidence, output)
    assert evidence["kind"] == "converted-compressed-part-evidence"
    assert "nativeStatus" not in evidence
    assert world.calls == ["convert", "video", "first", "quality", "full-av"]
    assert observe.verify(request, world.evidence) == result
    with pytest.raises(FileExistsError):
        observe.delivery_evidence(request, world.evidence, output)


@pytest.mark.parametrize("field,value", [("frames", 2), ("rate", 60)])
def test_converted_observer_rejects_clock_disagreement(
    world: World, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    converted_request(world, monkeypatch)
    world.raw[field] = value
    with pytest.raises(ReplayError):
        observe.collect(world.request(), world.evidence)
    assert not (world.evidence / "observation.json").exists()


def test_current_capture_keeps_generation_and_delivery_unapproved(world: World) -> None:
    result = observe.collect(world.request(), world.evidence)
    assert world.calls == ["video", "first", "quality"]
    assert result["videoClock"] == {"frames": 3, "start": "0", "end": "1/10", "exactCFR": True}
    assert result["firstVCL"] == 20
    assert result["historicalGenerationVerified"] is False
    assert result["processIdentityRecorded"] is False
    assert result["actualObservationCommandsCaptured"] is True
    assert result["numericalQualityAccepted"] is True
    assert result["qualityApproved"] is False
    assert result["syncApproved"] is False
    assert result["deliveryApproved"] is False
    assert result["sourceRetirementAllowed"] is False
    assert result["partAudioAdopted"] is False
    assert "nativeStatus" not in result
    assert result["qualityWindows"] == [
        {
            "name": "beginning",
            "startFrame": 0,
            "endFrame": 1,
            "ssim": 0.999,
            "psnr": 50.0,
            "ssimMinimum": 0.999,
            "psnrMinimum": 50.0,
            "thresholdsPassed": True,
        },
        {
            "name": "middle",
            "startFrame": 1,
            "endFrame": 2,
            "ssim": 0.998,
            "psnr": 49.0,
            "ssimMinimum": 0.998,
            "psnrMinimum": 49.0,
            "thresholdsPassed": True,
        },
        {
            "name": "end",
            "startFrame": 2,
            "endFrame": 3,
            "ssim": 0.997,
            "psnr": 48.0,
            "ssimMinimum": 0.997,
            "psnrMinimum": 48.0,
            "thresholdsPassed": True,
        },
    ]


def test_completed_reuse_does_not_decode_or_rewrite(world: World) -> None:
    request = world.request()
    observe.collect(request, world.evidence)
    prior = (world.evidence / "observation.json").stat()
    observe.collect(request, world.evidence)
    assert world.calls == ["video", "first", "quality"]
    assert (world.evidence / "observation.json").stat() == prior


def test_in_memory_threshold_change_cannot_replace_pinned_policy(world: World) -> None:
    request = world.request()
    request.thresholds["psnr"] = 0
    with pytest.raises(ReplayError, match="pinned JSON"):
        observe.collect(request, world.evidence)
    assert world.calls == []
    assert not world.evidence.exists()


def test_in_memory_frame_change_cannot_replace_pinned_clock(world: World) -> None:
    request = replace(world.request(), frames=1)
    with pytest.raises(ReplayError, match="pinned JSON"):
        observe.collect(request, world.evidence)
    assert world.calls == []
    assert not world.evidence.exists()


def test_verify_rejects_in_memory_policy_change_without_repair(world: World) -> None:
    request = world.request()
    observe.collect(request, world.evidence)
    request.thresholds["ssim"] = 0
    with pytest.raises(ReplayError, match="pinned JSON"):
        observe.verify(request, world.evidence)
    assert world.calls == ["video", "first", "quality"]


def test_fresh_start_and_end_read_both_current_media(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    original = InputHashCache._read

    def tracked(cache: InputHashCache, path: Path, entry: Entry, *, fresh: bool) -> str:
        calls.append((path.name, fresh))
        return original(cache, path, entry, fresh=fresh)

    monkeypatch.setattr(InputHashCache, "_read", tracked)
    observe.collect(world.request(), world.evidence)
    assert calls.count(("original.mov", True)) == 2
    assert calls.count(("compact.mp4", True)) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("kind", "unrelated"),
        ("frames", 0),
        ("frames", True),
        ("rate", 241),
        ("windowFrames", 0),
        ("threads", 257),
    ],
)
def test_request_rejects_wrong_kind_and_budgets(world: World, field: str, value: object) -> None:
    world.raw[field] = value
    with pytest.raises(ReplayError):
        world.request()
    assert world.calls == []


def test_request_rejects_undeclared_fields(world: World) -> None:
    world.raw["nativeStatus"] = "Successful"
    with pytest.raises(ReplayError, match="schema"):
        world.request()


@pytest.mark.parametrize(
    "thresholds",
    [
        {"ssim": 0.99},
        {"ssim": 2, "psnr": 45},
        {"ssim": 0.99, "psnr": -1},
        {"ssim": "0.99", "psnr": 45},
        {"ssim": 0.99, "psnr": True},
    ],
)
def test_thresholds_require_physical_explicit_policy(
    world: World, thresholds: dict[str, object]
) -> None:
    world.raw["thresholds"] = thresholds
    with pytest.raises(ReplayError):
        world.request()


def test_same_original_and_candidate_is_not_comparison(world: World) -> None:
    world.raw["movie"] = world.raw["source"]
    with pytest.raises(ReplayError, match="distinct"):
        world.request()


def test_tool_must_be_present_executable(world: World) -> None:
    (world.root / "ffmpeg").chmod(0o600)
    with pytest.raises(ReplayError, match="executable"):
        world.request()


def test_tools_must_be_distinct(world: World) -> None:
    tools = record(world.raw["tools"])
    world.raw["tools"] = {"ffmpeg": tools["ffmpeg"], "ffprobe": tools["ffmpeg"]}
    with pytest.raises(ReplayError, match="distinct"):
        world.request()


def test_configuration_must_be_complete(world: World) -> None:
    config = record(world.raw["videoConfiguration"])
    config.pop("extradata_hash")
    world.raw["videoConfiguration"] = config
    with pytest.raises(ReplayError, match="configuration"):
        world.request()


def test_configuration_cannot_call_missing_fact_observed(world: World) -> None:
    config = record(world.raw["videoConfiguration"])
    config["profile"] = None
    world.raw["videoConfiguration"] = config
    with pytest.raises(ReplayError, match="nonmissing"):
        world.request()


@pytest.mark.parametrize(
    "field,value",
    [
        ("codec_name", "h264"),
        ("codec_type", "audio"),
        ("extradata_hash", "SHA256:" + "0" * 64),
        ("avg_frame_rate", "24/1"),
        ("start_pts", 100),
        ("duration_ts", 299),
        ("index", True),
        ("extradata_size", 22),
        ("extradata", "00000001: 0000"),
    ],
)
def test_changed_or_missing_codec_facts_fail(world: World, field: str, value: object) -> None:
    streams = world.video["streams"]
    assert isinstance(streams, list)
    stream = record(streams[0])
    stream[field] = value
    world.video["streams"] = [stream]
    with pytest.raises((ReplayError, ValueError)):
        observe.collect(world.request(), world.evidence)
    assert not (world.evidence / "observation.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("pts", 1),
        ("duration", 99),
        ("stream_index", 1),
        ("media_type", "audio"),
        ("decode_error_flags", 1),
        ("best_effort_timestamp", 1),
    ],
)
def test_every_decoded_clock_and_error_flag_is_checked(
    world: World, field: str, value: object
) -> None:
    rows = world.video["frames"]
    assert isinstance(rows, list)
    first = record(rows[0])
    first[field] = value
    world.video["frames"] = [first, *rows[1:]]
    with pytest.raises(ReplayError):
        observe.collect(world.request(), world.evidence)


@pytest.mark.parametrize(
    "field,value", [("frames", []), ("streams", []), ("frames", None), ("error", {"code": -1})]
)
def test_metadata_only_or_reported_decoder_error_cannot_pass(
    world: World, field: str, value: object
) -> None:
    world.video[field] = value
    with pytest.raises(ReplayError):
        observe.collect(world.request(), world.evidence)


def test_no_first_packet_fails(world: World) -> None:
    world.first["packets"] = []
    with pytest.raises(ReplayError, match="Exactly one"):
        observe.collect(world.request(), world.evidence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pts", 100),
        ("stream_index", 2),
        ("size", 5),
        ("data_hash", "SHA256:" + "0" * 64),
        ("data", "bad dump"),
    ],
)
def test_first_packet_binding_is_strict(world: World, field: str, value: object) -> None:
    rows = world.first["packets"]
    assert isinstance(rows, list)
    packet = record(rows[0])
    packet[field] = value
    world.first["packets"] = [packet]
    with pytest.raises(ReplayError):
        observe.collect(world.request(), world.evidence)


def test_non_idr_rejects_even_with_valid_current_hash(world: World) -> None:
    data = b"\x00\x00\x00\x02\x02\x01"
    world.first["packets"] = [
        {
            "stream_index": 0,
            "pts": 0,
            "size": 6,
            "data": dump(data),
            "data_hash": "SHA256:" + hashlib.sha256(data).hexdigest(),
        }
    ]
    with pytest.raises(ReplayError, match="IDR"):
        observe.collect(world.request(), world.evidence)


def test_ffprobe_decimal_packet_size_is_bound_to_observed_bytes(world: World) -> None:
    rows = world.first["packets"]
    assert isinstance(rows, list)
    packet = record(rows[0])
    packet["size"] = "6"
    world.first["packets"] = [packet]
    result = observe.collect(world.request(), world.evidence)
    assert result["firstVCL"] == 20


@pytest.mark.parametrize("size", ["06", "nan", "-6", "99999999999", False, 0])
def test_packet_size_not_guessed_from_bad_representation(world: World, size: object) -> None:
    rows = world.first["packets"]
    assert isinstance(rows, list)
    packet = record(rows[0])
    packet["size"] = size
    world.first["packets"] = [packet]
    with pytest.raises(ReplayError):
        observe.collect(world.request(), world.evidence)


def test_different_geometry_rate_and_window_count_preserve_frame_clock(world: World) -> None:
    world.raw["frames"], world.raw["rate"], world.raw["windowFrames"] = 7, 24, 2
    configuration = record(world.raw["videoConfiguration"])
    configuration["width"], configuration["height"], configuration["time_base"] = 96, 64, "1/2400"
    world.raw["videoConfiguration"] = configuration
    streams = world.video["streams"]
    assert isinstance(streams, list)
    stream = record(streams[0])
    stream.update(configuration)
    stream["duration_ts"], stream["avg_frame_rate"] = 700, "24/1"
    world.video["streams"] = [stream]
    world.video["frames"] = [
        {"media_type": "video", "stream_index": 0, "pts": index * 100, "duration": 100}
        for index in range(7)
    ]

    def metrics(directory: Path) -> None:
        (directory / "quality-0.ssim.stats").write_text(
            "n:1 All:0.999\nn:2 All:0.997\n", encoding="utf-8"
        )
        (directory / "quality-1.ssim.stats").write_text(
            "n:1 All:0.999\nn:2 All:0.997\n", encoding="utf-8"
        )
        (directory / "quality-2.ssim.stats").write_text(
            "n:1 All:0.999\nn:2 All:0.997\n", encoding="utf-8"
        )
        (directory / "quality-0.psnr.stats").write_text(
            "n:1 psnr_avg:50\nn:2 psnr_avg:48\n", encoding="utf-8"
        )
        (directory / "quality-1.psnr.stats").write_text(
            "n:1 psnr_avg:50\nn:2 psnr_avg:48\n", encoding="utf-8"
        )
        (directory / "quality-2.psnr.stats").write_text(
            "n:1 psnr_avg:50\nn:2 psnr_avg:48\n", encoding="utf-8"
        )

    world.effect = metrics
    result = observe.collect(world.request(), world.evidence)
    assert result["videoClock"] == {"frames": 7, "start": "0", "end": "7/24", "exactCFR": True}
    assert result["numericalQualityAccepted"] is True
    assert result["qualityWindows"] == [
        {
            "name": "beginning",
            "startFrame": 0,
            "endFrame": 2,
            "ssim": 0.998,
            "psnr": 49.0,
            "ssimMinimum": 0.997,
            "psnrMinimum": 48.0,
            "thresholdsPassed": True,
        },
        {
            "name": "middle",
            "startFrame": 2,
            "endFrame": 4,
            "ssim": 0.998,
            "psnr": 49.0,
            "ssimMinimum": 0.997,
            "psnrMinimum": 48.0,
            "thresholdsPassed": True,
        },
        {
            "name": "end",
            "startFrame": 5,
            "endFrame": 7,
            "ssim": 0.998,
            "psnr": 49.0,
            "ssimMinimum": 0.997,
            "psnrMinimum": 48.0,
            "thresholdsPassed": True,
        },
    ]


@pytest.mark.parametrize(
    "line",
    [
        "",
        "n:2 All:0.999\n",
        "n:1 All:nan\n",
        "n:1 All:inf\n",
        "n:1 All:1.1\n",
        "n:1 All:-1\n",
        "n:1 Y:0.99\n",
        "n:1 All:garbage\n",
    ],
)
def test_raw_quality_window_is_not_invented_or_truncated(world: World, line: str) -> None:
    def mutate(directory: Path) -> None:
        (directory / "quality-1.ssim.stats").write_text(line, encoding="utf-8")

    world.effect = mutate
    with pytest.raises(ReplayError):
        observe.collect(world.request(), world.evidence)


def test_low_quality_remains_explicit_candidate(world: World) -> None:
    def mutate(directory: Path) -> None:
        (directory / "quality-2.psnr.stats").write_text("n:1 psnr_avg:30\n", encoding="utf-8")

    world.effect = mutate
    result = observe.collect(world.request(), world.evidence)
    assert result["numericalQualityAccepted"] is False
    assert result["candidateOnly"] is True
    assert result["deliveryApproved"] is False


def test_duplicate_metric_cannot_hide_bad_value(world: World) -> None:
    def mutate(directory: Path) -> None:
        (directory / "quality-0.ssim.stats").write_text("n:1 All:nan All:0.999\n", encoding="utf-8")

    world.effect = mutate
    with pytest.raises(ReplayError, match="duplicated"):
        observe.collect(world.request(), world.evidence)


@pytest.mark.parametrize(
    "field,value",
    [("command", ["ffprobe", "unbound.mp4"]), ("returncode", 1), ("returncode", True)],
)
def test_actual_failed_or_unbound_process_does_not_pass(
    world: World, field: str, value: object
) -> None:
    def mutate(directory: Path) -> None:
        process = read(directory / "video.process.json")
        process[field] = value
        write(directory / "video.process.json", process)

    world.effect = mutate
    with pytest.raises(ReplayError, match="process evidence"):
        observe.collect(world.request(), world.evidence)


def test_retained_stderr_rejects_silent_success(world: World) -> None:
    def mutate(directory: Path) -> None:
        (directory / "video.stderr").write_text("corrupt packet", encoding="utf-8")

    world.effect = mutate
    with pytest.raises(ReplayError, match="process evidence"):
        observe.collect(world.request(), world.evidence)


@pytest.mark.parametrize(
    "name", ["original.mov", "compact.mp4", "ffmpeg", "ffprobe", "source-native.json"]
)
def test_initial_input_mutation_prevents_children(world: World, name: str) -> None:
    request = world.request()
    (world.root / name).write_bytes(b"changed current input")
    with pytest.raises((ReplayError, ValueError)):
        observe.collect(request, world.evidence)
    assert world.calls == []
    assert not world.evidence.exists()


@pytest.mark.parametrize("name", ["original.mov", "compact.mp4", "ffmpeg", "request.json"])
def test_terminal_input_mutation_cannot_publish(world: World, name: str) -> None:
    def mutate(directory: Path) -> None:
        (world.root / name).write_bytes(b"changed during independent observation")

    world.effect = mutate
    with pytest.raises(ReplayError, match="changed"):
        observe.collect(world.request(), world.evidence)
    assert not (world.evidence / "observation.json").exists()


def test_original_native_full_proof_is_required(world: World) -> None:
    path = world.root / "source-native.json"
    raw = read(path)
    raw["fullAVDecodeExitCode"] = True
    world.raw["sourceNative"] = write(path, raw).json()
    with pytest.raises(ReplayError, match="native completion"):
        observe.collect(world.request(), world.evidence)
    assert world.calls == []


def test_interrupted_directory_is_preserved_without_rerun(world: World) -> None:
    world.evidence.mkdir()
    (world.evidence / "video.stdout").write_text("partial", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        observe.collect(world.request(), world.evidence)
    assert world.calls == []
    assert (world.evidence / "video.stdout").read_text(encoding="utf-8") == "partial"


def test_child_exception_preserves_partial_and_restores_context(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(command: list[str], *, directory: Path, name: str, timeout: int) -> Path:
        raise ReplayError("actual bounded child failure")

    monkeypatch.setattr(technical, "_run", fail)
    with pytest.raises(ReplayError, match="actual bounded child"):
        observe.collect(world.request(), world.evidence)
    assert (world.evidence / "initial.json").exists()
    request = world.request()
    with input_hash_context(InputHashCache([request.source.path])):
        assert request.source.sha256 == file_digest(request.source.path)


@pytest.mark.parametrize(
    "name", ["video.stdout", "quality-0.ssim.stats", "terminal.json", "observation.json"]
)
def test_reuse_rejects_changed_or_missing_current_evidence(world: World, name: str) -> None:
    request = world.request()
    observe.collect(request, world.evidence)
    (world.evidence / name).unlink()
    with pytest.raises((FileNotFoundError, ReplayError, ValueError)):
        observe.verify(request, world.evidence)
    assert world.calls == ["video", "first", "quality"]


def test_existing_receipt_does_not_hide_raw_mutation(world: World) -> None:
    request = world.request()
    observe.collect(request, world.evidence)
    (world.evidence / "quality-0.ssim.stats").write_text("n:1 All:0.998\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="Immutable"):
        observe.verify(request, world.evidence)


def test_initial_closure_is_not_substituted(world: World) -> None:
    request = world.request()
    observe.collect(request, world.evidence)
    result = read(world.evidence / "observation.json")
    result["initial"] = write(
        world.evidence / "initial.json",
        {"version": 1, "kind": "compressed-part-observation-inputs", "inputs": []},
    ).json()
    write(world.evidence / "observation.json", result)
    with pytest.raises(ReplayError, match="input closure"):
        observe.verify(request, world.evidence)


def test_directory_replacement_is_detected(world: World) -> None:
    def mutate(directory: Path) -> None:
        directory.rename(world.root / "retained-failed-evidence")
        directory.mkdir()

    world.effect = mutate
    with pytest.raises(ReplayError, match="directory changed"):
        observe.collect(world.request(), world.evidence)
    assert (world.root / "retained-failed-evidence" / "initial.json").exists()


def test_raw_mutation_during_parsing_prevents_receipt(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = observe._quality

    def change(
        request: observe.Request, directory: Path
    ) -> tuple[list[dict[str, object]], tuple[metadata.Reference, ...]]:
        result = original(request, directory)
        (directory / "video.stdout").write_text("changed raw bytes", encoding="utf-8")
        return result

    monkeypatch.setattr(observe, "_quality", change)
    with pytest.raises(ReplayError, match="during domain"):
        observe.collect(world.request(), world.evidence)
    assert not (world.evidence / "observation.json").exists()


def test_linked_input_is_rejected(world: World) -> None:
    alias = world.root / "linked.mov"
    alias.symlink_to(world.root / "original.mov")
    world.raw["source"] = {"path": str(alias), "sha256": file_digest(world.root / "original.mov")}
    with pytest.raises(ReplayError, match="symlinks"):
        world.request()


def test_collect_cli_and_verify_share_original_domain(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    request = world.request()
    assert (
        observe.main(
            [
                "collect",
                str(world.config),
                str(world.evidence),
                "--sha256",
                request.reference.sha256,
            ]
        )
        == 0
    )
    assert (
        observe.main(
            ["verify", str(world.config), str(world.evidence), "--sha256", request.reference.sha256]
        )
        == 0
    )
    assert '"historicalGenerationVerified": false' in capsys.readouterr().out
    assert world.calls == ["video", "first", "quality"]


def test_cli_failure_is_explicit(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    world.request()
    with pytest.raises(SystemExit, match="1"):
        observe.main(["verify", str(world.config), str(world.evidence), "--sha256", "0" * 64])
    assert "COMPRESSED_PART_OBSERVATION_ERROR" in capsys.readouterr().err
