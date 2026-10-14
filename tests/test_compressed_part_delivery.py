"""Synthetic observation artifacts only; no app, recordings or encoder jobs."""

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from replay import compressed_part_delivery as delivery
from replay import compressed_part_observation as observer
from replay import delivery_clock_contract as clock
from replay import whole_technical_batch as metadata
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import StageError, _fingerprint, file_digest, run_stage
from replay.workflow import load_stages, stage_graph


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def write(path: Path, value: object) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return pin(path)


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class Example:
    config: Path
    root: Path
    output: Path

    def change(self, field: str, value: object) -> None:
        raw = read(self.config)
        raw[field] = value
        write(self.config, raw)

    def part(self, number: int, role: str, change: dict[str, object]) -> Path:
        raw = read(self.config)
        parts = raw["parts"]
        assert isinstance(parts, list)
        part = record(parts[number - 1])
        evidence = Path(str(record(part["evidence"])["path"]))
        proof = read(evidence)
        path = Path(str(record(proof[role])["path"]))
        old = read(path)
        old.update(change)
        proof[role] = write(path, old)
        part["evidence"] = write(evidence, proof)
        write(self.config, raw)
        return path

    def audio(self, role: str, change: dict[str, object]) -> Path:
        raw = read(self.config)
        audio = record(raw["audio"])
        evidence = Path(str(record(audio["evidence"])["path"]))
        proof = read(evidence)
        path = Path(str(record(proof[role])["path"]))
        old = read(path)
        old.update(change)
        proof[role] = write(path, old)
        audio["evidence"] = write(evidence, proof)
        write(self.config, raw)
        return path


def contract(root: Path, storage: Path) -> dict[str, str]:
    original = root / "originals/original.mp4"
    original.parent.mkdir()
    original.write_bytes(b"original")
    edit_root = root / "edits/replay-123"
    edit_root.mkdir(parents=True)
    edit = edit_root / "edit-config.json"
    write(
        edit,
        {
            "version": 1,
            "replay_id": "123",
            "source": str(original),
            "source_size": 8,
            "engine": str(root),
            "executor_scope": str(root),
            "storage_root": str(storage),
            "stages": dict.fromkeys(
                (
                    "bgm_removal",
                    "silence_cut",
                    "source_caption_mask",
                    "transcription",
                    "dynamic_captions",
                ),
                True,
            ),
        },
    )
    work = root / "edits/work"
    cuts = work / "cuts.json"
    write(
        cuts,
        {
            "clock": "source-presentation",
            "fps": 30,
            "cuts": [{"source_start_frame": 100, "source_end_frame": 130, "output_start_frame": 0}],
        },
    )
    xml = work / "semantic.fcpxml"
    xml.write_text(
        '<fcpxml><resources><format id="r1" width="640" height="960" frameDuration="1/30s"/>'
        '</resources><library><event><project><sequence format="r1" duration="1s"><spine>'
        '<asset-clip offset="0s" start="10/3s" duration="1s"><title offset="10/3s" '
        'duration="1s"><text>Synthetic</text></title></asset-clip>'
        "</spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    return write(
        work / "clock.json",
        {
            "version": 1,
            "kind": "delivery-clock-expectation",
            "editConfig": pin(edit),
            "cuts": pin(cuts),
            "semanticXML": pin(xml),
        },
    )


def part(work: Path, number: int) -> dict[str, object]:
    master, movie = work / f"master-{number}.mov", work / f"part-{number}.m4v"
    master.write_bytes(f"native full part {number}".encode())
    movie.write_bytes(f"compressed full part {number}".encode())
    source, candidate = pin(master), pin(movie)
    fields: dict[str, object] = dict.fromkeys(delivery.VIDEO_FIELDS, "observed")
    fields.update(
        codec_name="hevc",
        codec_tag_string="hvc1",
        profile="Main",
        level=120,
        width=640,
        height=960,
        pix_fmt="yuv420p",
        time_base="1/3000",
        avg_frame_rate="30/1",
        start_pts=0,
        duration_ts=1500,
        extradata_hash="SHA256:" + "a" * 64,
    )
    evidence = {
        "version": 1,
        "kind": "compressed-part-evidence",
        "source": source,
        "movie": candidate,
        "sourceNative": write(
            work / f"source-{number}.json",
            {
                "movie": source,
                "completionVerified": True,
                "shareObservedIdle": True,
                "fullAVDecodeExitCode": 0,
            },
        ),
        "nativeStatus": write(
            work / f"native-{number}.json",
            {
                "status": "Successful",
                "source": str(master),
                "output": str(movie),
            },
        ),
        "fullAV": write(work / f"full-{number}.json", {"movie": candidate, "exitCode": 0}),
        "videoProbe": write(
            work / f"probe-{number}.json",
            {
                "movie": candidate,
                "streams": [fields],
                "frames": [{"pts": index * 100, "duration": 100} for index in range(15)],
            },
        ),
        "firstPacket": write(
            work / f"packet-{number}.json",
            {
                "movie": candidate,
                "nalLengthBytes": 4,
                "dataHex": "000000024001000000022801",
            },
        ),
        "quality": write(
            work / f"quality-{number}.json",
            {
                "source": source,
                "candidate": candidate,
                "windows": [
                    {"startFrame": 0, "endFrame": 15, "ssim": 0.99, "psnr": 50, "vmaf": 98}
                ],
            },
        ),
    }
    return {
        "number": number,
        "startFrame": (number - 1) * 15,
        "frames": 15,
        "master": source,
        "movie": candidate,
        "evidence": write(work / f"evidence-{number}.json", evidence),
    }


def audio(work: Path) -> dict[str, object]:
    pcm, movie = work / "whole.wav", work / "whole-audio.m4a"
    pcm.write_bytes(b"one whole pcm")
    movie.write_bytes(b"encoded once")
    source, candidate = pin(pcm), pin(movie)
    proof = {
        "source": source,
        "movie": candidate,
        "encodingPasses": 1,
        "nativeStatus": write(
            work / "audio-native.json",
            {
                "status": "Successful",
                "source": str(pcm),
                "output": str(movie),
            },
        ),
        "fullAV": write(work / "audio-full.json", {"movie": candidate, "exitCode": 0}),
        "decodedProbe": write(
            work / "audio-probe.json",
            {
                "movie": candidate,
                "streams": [{"sample_rate": "48000", "time_base": "1/48000"}],
                "frames": [{"pts": 0, "nb_samples": 24000}, {"pts": 24000, "nb_samples": 24000}],
            },
        ),
        "packetProbe": write(
            work / "audio-packets.json",
            {
                "movie": candidate,
                "streams": [
                    {
                        "codec_name": "aac",
                        "profile": "LC",
                        "sample_rate": "48000",
                        "channels": 2,
                        "time_base": "1/48000",
                    }
                ],
                "packets": [
                    {
                        "pts": -64,
                        "duration": 24064,
                        "side_data_list": [
                            {
                                "side_data_type": "Skip Samples",
                                "skip_samples": 64,
                                "discard_padding": 0,
                            }
                        ],
                    },
                    {
                        "pts": 24000,
                        "duration": 24064,
                        "side_data_list": [
                            {
                                "side_data_type": "Skip Samples",
                                "skip_samples": 0,
                                "discard_padding": 64,
                            }
                        ],
                    },
                ],
            },
        ),
    }
    return {
        "pcm": source,
        "movie": candidate,
        "evidence": write(work / "audio-evidence.json", proof),
    }


@pytest.fixture
def example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Example:
    monkeypatch.setattr(metadata, "ROOT", tmp_path)
    monkeypatch.setattr(clock, "ROOT", tmp_path)
    work, storage = tmp_path / "edits/work", tmp_path / "storage/replay-123"
    work.mkdir(parents=True)
    storage.mkdir(parents=True)
    tool = work / "executable"
    tool.write_bytes(b"synthetic executable")
    tool.chmod(0o700)
    fallback = write(
        work / "fallback.json",
        {
            "version": 1,
            "stages": [
                {
                    "name": "existing-prores",
                    "command": [str(tool)],
                    "inputs": [str(tool)],
                    "outputs": [str(storage / "fallback.mov")],
                    "timeout_seconds": 10,
                }
            ],
        },
    )
    path = work / "config.json"
    write(
        path,
        {
            "version": 1,
            "rate": 30,
            "width": 640,
            "height": 960,
            "parts": [part(work, 1), part(work, 2)],
            "audio": audio(work),
            "thresholds": {"ssim": 0.98, "psnr": 40, "vmaf": 95},
            "fallback": fallback,
            "clockContract": contract(tmp_path, storage),
            "output": str(storage / "delivery.mp4"),
            "tools": dict.fromkeys(("python", "ffmpeg", "ffprobe"), pin(tool)),
        },
    )
    return Example(path, work, work / "prepared")


@dataclass(frozen=True)
class Converted:
    request: observer.Request
    observed: dict[str, object]
    evidence: Path


@pytest.fixture
def converted(example: Example, monkeypatch: pytest.MonkeyPatch) -> Converted:
    plan = delivery.load(example.config)
    part = plan.parts[0]
    assert part.evidence is not None and part.movie is not None
    legacy = read(part.evidence.path)
    native = metadata._reference(legacy["sourceNative"])
    converter = metadata._reference(
        write(example.root / "execution.json", {"kind": "converter-fixture"})
    )
    probe = read(metadata._reference(legacy["videoProbe"]).path)
    streams = probe["streams"]
    assert isinstance(streams, list)
    video = {key: record(streams[0])[key] for key in delivery.VIDEO_FIELDS}
    request_ref = metadata._reference(
        write(example.root / "observer-request.json", {"fixture": "request"})
    )
    request = observer.Request(
        request_ref,
        part.master,
        part.movie,
        native,
        plan.tools["ffmpeg"],
        plan.tools["ffprobe"],
        part.frames,
        plan.rate,
        part.frames,
        1,
        video,
        {"ssim": 0.98, "psnr": 40},
        converter,
    )
    empty = example.root / "empty.stdout"
    empty.write_bytes(b"")
    empty_ref = {
        "path": str(empty),
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    initial, terminal = (
        write(example.root / "initial.json", {"fixture": "initial"}),
        write(example.root / "terminal.json", {"fixture": "terminal"}),
    )
    observed: dict[str, object] = {
        "request": request_ref.json(),
        "videoConfiguration": video,
        "qualityWindows": [
            {"startFrame": 0, "endFrame": part.frames, "ssim": 0.99, "psnr": 50, "vmaf": 98}
        ],
        "inputs": [request_ref.json(), native.json(), converter.json()],
        "artifacts": [legacy["videoProbe"], empty_ref],
        "initial": initial,
        "terminal": terminal,
    }
    observation = metadata._reference(write(example.root / "observation.json", observed))
    raw = {
        "version": 1,
        "kind": "converted-compressed-part-evidence",
        "source": part.master.json(),
        "movie": part.movie.json(),
        "sourceNative": native.json(),
        "converterExecution": converter.json(),
        "observation": observation.json(),
    }
    evidence = part.evidence.path
    config = read(example.config)
    rows = config["parts"]
    assert isinstance(rows, list)
    record(rows[0])["evidence"] = write(evidence, raw)
    write(example.config, config)

    def load(reference: metadata.Reference) -> observer.Request:
        assert reference == request_ref
        return request

    def verify(value: observer.Request, directory: Path) -> dict[str, object]:
        assert value == request and directory == example.root
        return observed

    # The independent raw/producer validator is exercised without mocks in its
    # own tests; this boundary fixture isolates assembly policy and pin propagation.
    monkeypatch.setattr(observer, "load", load)
    monkeypatch.setattr(observer, "verify", verify)
    return Converted(request, observed, evidence)


def test_converter_variant_uses_verified_observation_and_preserves_native_branch(
    example: Example, converted: Converted
) -> None:
    result = delivery.admit(delivery.load(example.config))
    assert result["route"] == "compressed-ready"
    assert result["deliveryApproved"] is False
    inputs = result["inputsVerified"]
    assert isinstance(inputs, list)
    assert converted.request.reference.json() in inputs
    assert pin(example.root / "initial.json") in inputs
    assert pin(example.root / "terminal.json") in inputs
    assert "nativeStatus" not in read(converted.evidence)


@pytest.mark.parametrize(
    "field,value", [("version", True), ("source", {"path": "wrong"}), ("unexpected", 0)]
)
def test_converter_variant_rejects_malformed_or_mismatched_provenance(
    example: Example, converted: Converted, field: str, value: object
) -> None:
    raw = read(converted.evidence)
    raw[field] = value
    plan = delivery.load(example.config)
    with pytest.raises(ReplayError, match="provenance differs"):
        delivery._converted_part(plan.parts[0], plan, raw)


@pytest.mark.parametrize("field,value", [("frames", 1), ("rate", 60), ("converter", None)])
def test_converter_variant_cannot_change_selected_clock_or_producer(
    example: Example,
    converted: Converted,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    request = converted.request
    changed = (
        replace(request, frames=1)
        if field == "frames"
        else replace(request, rate=60)
        if field == "rate"
        else replace(request, converter=None)
    )

    def load(reference: metadata.Reference) -> observer.Request:
        return changed

    monkeypatch.setattr(observer, "load", load)
    plan = delivery.load(example.config)
    with pytest.raises(ReplayError, match="selected delivery part"):
        delivery._part_check(plan.parts[0], plan)


def test_converter_quality_failure_keeps_prores_fallback(
    example: Example, converted: Converted
) -> None:
    windows = converted.observed["qualityWindows"]
    assert isinstance(windows, list)
    record(windows[0])["ssim"] = 0.8
    plan = delivery.load(example.config)
    reference = metadata._reference(read(converted.evidence)["observation"])
    raw = read(converted.evidence)
    raw["observation"] = write(reference.path, converted.observed)
    _, failures, measurements = delivery._converted_part(plan.parts[0], plan, raw)
    assert failures == ["ssim 0.8 < 0.98 at frames 0..15"]
    assert measurements[0]["ssim"] == 0.8
    assert plan.fallback.path.exists() and plan.parts[0].master.path.exists()


def test_ready_preserves_source_audio_and_quality_separation(example: Example) -> None:
    result = delivery.admit(delivery.load(example.config))
    assert result["route"] == "compressed-ready"
    assert result["assemblyReady"] is True
    assert result["deliveryApproved"] is False
    assert result["reasons"] == []
    inputs = result["inputsVerified"]
    verification = result["finalVerificationRequired"]
    assert isinstance(inputs, list) and len(inputs) > 20
    assert isinstance(verification, list) and "audio/video synchronization" in verification


def test_prepare_graph_maps_only_video_parts_and_single_whole_audio(example: Example) -> None:
    result = delivery.prepare(example.config, example.output)
    assert result["assemblyReady"] is True
    stages = load_stages(example.output / "workflow.json")
    assert stage_graph(stages) == {
        "compressed-part-admission": (),
        "compressed-lossless-join": ("compressed-part-admission",),
        "compressed-whole-technical": ("compressed-lossless-join",),
        "compressed-whole-audio": ("compressed-lossless-join", "compressed-whole-technical"),
    }
    command = stages[1].command
    assert command[command.index("-map") + 1] == "0:v:0"
    assert "1:a:0" in command
    assert "-shortest" not in command
    assert "-copyts" in command
    assert "-start_at_zero" not in command
    assert "-use_editlist" not in command
    assert command[command.index("-c") + 1] == "copy"
    assert "-clock-contract" not in stages[2].command
    assert "--observe" in stages[1].verify_command
    assert "--technical-result" in stages[2].command
    assert "--audio-result" in stages[3].command
    assert stages[2].verify_command == stages[2].command
    assert stages[3].verify_command == stages[3].command
    assert all("delivery_verification" not in " ".join(stage.command) for stage in stages)
    assert len(stages[1].outputs) == 5
    assert "compressed_part_delivery.py" in " ".join(map(str, stages[0].inputs))
    assert len((example.output / "parts.ffconcat").read_text().splitlines()) == 5
    assert read(example.output / "pins.json")["deliveryApproved"] is False


def test_pending_parts_do_not_launch_or_copy_completed_parts(example: Example) -> None:
    raw = read(example.config)
    parts = raw["parts"]
    assert isinstance(parts, list)
    record(parts[1]).update(movie=None, evidence=None)
    example.change("parts", parts)
    before = (example.root / "part-1.m4v").read_bytes()
    result = delivery.prepare(example.config, example.output)
    assert result["route"] == "pending"
    assert result["pendingParts"] == [2]
    assert not (example.output / "workflow.json").exists()
    assert (example.root / "part-1.m4v").read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("profile", "Main 10"),
        ("extradata_hash", "SHA256:" + "b" * 64),
        ("color_transfer", "smpte2084"),
        ("time_base", "1/6000"),
    ],
)
def test_mixed_codec_configuration_selects_fallback(
    example: Example, field: str, value: object
) -> None:
    probe = read(example.root / "probe-2.json")
    streams = probe["streams"]
    assert isinstance(streams, list)
    record(streams[0])[field] = value
    example.part(2, "videoProbe", {"streams": streams})
    result = delivery.assess(delivery.load(example.config))
    assert result["route"] == "prores-fallback"
    assert "mixed HEVC profiles" in str(result["reasons"])


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"codec_name": "h264"}, "HEVC/hvc1"),
        ({"codec_tag_string": "hev1"}, "HEVC/hvc1"),
        ({"avg_frame_rate": "25/1"}, "frame rate"),
        ({"width": 1280}, "geometry"),
        ({"duration_ts": 1499}, "duration/frame count"),
        ({"start_pts": 100}, "duration/frame count"),
    ],
)
def test_measured_stream_failures_select_fallback(
    example: Example, change: dict[str, object], reason: str
) -> None:
    probe = read(example.root / "probe-1.json")
    streams = probe["streams"]
    assert isinstance(streams, list)
    record(streams[0]).update(change)
    example.part(1, "videoProbe", {"streams": streams})
    result = delivery.assess(delivery.load(example.config))
    assert result["route"] == "prores-fallback"
    assert reason in str(result["reasons"])


@pytest.mark.parametrize(
    "frames",
    [
        [{"pts": index * 100, "duration": 100} for index in range(14)],
        [{"pts": index * 100 + (1 if index == 10 else 0), "duration": 100} for index in range(15)],
        [{"pts": index * 100, "duration": 99 if index == 10 else 100} for index in range(15)],
    ],
)
def test_complete_pts_and_duration_are_required(
    example: Example, frames: list[dict[str, int]]
) -> None:
    example.part(1, "videoProbe", {"frames": frames})
    assert delivery.assess(delivery.load(example.config))["route"] == "prores-fallback"


@pytest.mark.parametrize(
    "role,change,reason",
    [
        ("nativeStatus", {"status": "Failed"}, "success/output"),
        ("nativeStatus", {"source": "/different.mov"}, "source binding"),
        ("fullAV", {"exitCode": 1}, "full AV decode"),
        ("sourceNative", {"completionVerified": False}, "source full production part"),
        ("firstPacket", {"dataHex": "000000022a01"}, "HEVC IDR"),
    ],
)
def test_failed_native_decode_source_and_random_access_proofs(
    example: Example, role: str, change: dict[str, object], reason: str
) -> None:
    example.part(1, role, change)
    result = delivery.assess(delivery.load(example.config))
    assert result["route"] == "prores-fallback"
    assert reason in str(result["reasons"])


@pytest.mark.parametrize("metric,value", [("ssim", 0.8), ("psnr", 32), ("vmaf", 80)])
def test_quality_failure_keeps_actual_numbers_and_declared_fallback(
    example: Example, metric: str, value: float
) -> None:
    window: dict[str, object] = {
        "startFrame": 0,
        "endFrame": 15,
        "ssim": 0.99,
        "psnr": 50,
        "vmaf": 98,
    }
    window[metric] = value
    example.part(1, "quality", {"windows": [window]})
    result = delivery.prepare(example.config, example.output)
    assert result["route"] == "prores-fallback"
    assert f"{metric} {float(value)} <" in str(result["reasons"])
    assert result["fallback"] == read(example.config)["fallback"]
    assert not (example.output / "workflow.json").exists()


@pytest.mark.parametrize(
    "role,change,reason",
    [
        ("nativeStatus", {"status": "Failed"}, "native source/success"),
        ("fullAV", {"exitCode": True}, "full decode"),
        (
            "decodedProbe",
            {"frames": [{"pts": 0, "nb_samples": 24000}, {"pts": 24001, "nb_samples": 24000}]},
            "gaps or overlaps",
        ),
        (
            "decodedProbe",
            {"frames": [{"pts": 0, "nb_samples": 24000}, {"pts": 24000, "nb_samples": 22400}]},
            "46400/48000",
        ),
    ],
)
def test_whole_audio_clock_failure_is_not_hidden(
    example: Example, role: str, change: dict[str, object], reason: str
) -> None:
    example.audio(role, change)
    result = delivery.assess(delivery.load(example.config))
    assert result["route"] == "prores-fallback"
    assert reason in str(result["reasons"])


def test_removed_aac_priming_is_rejected_despite_same_encoded_media(example: Example) -> None:
    example.audio(
        "packetProbe",
        {
            "packets": [
                {"pts": 0, "duration": 24064},
                {"pts": 24064, "duration": 24064},
            ]
        },
    )
    result = delivery.assess(delivery.load(example.config))
    assert result["route"] == "prores-fallback"
    assert "priming/remainder" in str(result["reasons"])


def test_multiple_audio_encodes_are_rejected(example: Example) -> None:
    path = example.root / "audio-evidence.json"
    proof = read(path)
    proof["encodingPasses"] = 2
    raw = record(read(example.config)["audio"])
    raw["evidence"] = write(path, proof)
    example.change("audio", raw)
    assert "encoded once" in str(delivery.assess(delivery.load(example.config))["reasons"])


def test_absent_whole_audio_is_pending(example: Example) -> None:
    example.change("audio", None)
    result = delivery.assess(delivery.load(example.config))
    assert result["route"] == "pending"
    assert result["audioPending"] is True


@pytest.mark.parametrize(
    "name", ["master-1.mov", "part-1.m4v", "whole.wav", "whole-audio.m4a", "executable"]
)
def test_admission_hashes_media_and_tools_again(example: Example, name: str) -> None:
    path = example.root / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ReplayError, match="hash changed"):
        delivery.admit(delivery.load(example.config))


def test_nonexecutable_tool_is_rejected(example: Example) -> None:
    (example.root / "executable").chmod(0o600)
    with pytest.raises(ReplayError, match="executable"):
        delivery.admit(delivery.load(example.config))


def python_alias(example: Example) -> Path:
    alias = example.root.parents[1] / ".venv/bin/python"
    alias.parent.mkdir(parents=True)
    target = example.root / "executable"
    alias.symlink_to(target)
    example.change(
        "pythonInvocation",
        {"path": str(alias), "target": pin(target), "linkTarget": os.readlink(alias)},
    )
    return alias


def test_python_environment_alias_is_bound_separately_from_tool_bytes(example: Example) -> None:
    alias = python_alias(example)
    plan = delivery.load(example.config)
    assert plan.python_invocation == alias
    assert plan.tools["python"].path == example.root / "executable"
    delivery.admit(plan)
    delivery.prepare(example.config, example.output)
    stages = load_stages(example.output / "workflow.json")
    assert stages[0].command[0] == str(alias)
    assert alias not in stages[0].inputs
    assert plan.tools["python"].path in stages[0].inputs
    verified = delivery.admit(plan)["inputsVerified"]
    assert isinstance(verified, list) and plan.tools["python"].json() in verified


@pytest.mark.parametrize("change", ["target", "link", "regular", "outside", "extra"])
def test_python_alias_cannot_weaken_media_or_executable_binding(
    example: Example, change: str
) -> None:
    alias = python_alias(example)
    row = record(read(example.config)["pythonInvocation"])
    if change == "target":
        row["target"] = pin(example.root / "master-1.mov")
    elif change == "link":
        row["linkTarget"] = "unobserved-target"
    elif change == "regular":
        alias.unlink()
        alias.write_bytes(b"not an executable alias")
    elif change == "outside":
        row["path"] = str(example.root.parents[1].parent / "outside-python")
    else:
        row["unknown"] = True
    example.change("pythonInvocation", row)
    with pytest.raises(ReplayError, match="Python invocation"):
        delivery.load(example.config)


def test_loaded_python_alias_is_rechecked_before_admission(example: Example) -> None:
    alias = python_alias(example)
    plan = delivery.load(example.config)
    alias.unlink()
    alias.symlink_to(example.root / "master-1.mov")
    with pytest.raises(ReplayError, match="Python invocation"):
        delivery.admit(plan)


def test_real_admission_stage_fingerprints_and_executes_existing_venv_alias(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = Path(__file__).resolve().parents[1]
    alias = checkout / ".venv/bin/python"
    interpreter = alias.resolve(strict=True)
    raw = read(example.config)
    tools = record(raw["tools"])
    tools["python"] = pin(interpreter)
    raw["pythonInvocation"] = {
        "path": str(alias),
        "target": pin(interpreter),
        "linkTarget": os.readlink(alias),
    }
    write(example.config, raw)
    monkeypatch.setattr(metadata, "ROOT", checkout)
    delivery.prepare(example.config, example.output)
    stage = load_stages(example.output / "workflow.json")[0]
    assert len(_fingerprint(stage)) == 64
    assert interpreter in stage.inputs and alias not in stage.inputs
    assert stage.command[0] == str(alias)
    state = example.root / "real-stage-state"
    state.mkdir()
    assert run_stage(stage, state_directory=state) == "completed"
    assert read(example.output / "ready.json")["assemblyReady"] is True
    assert run_stage(stage, state_directory=state) == "reused"


@pytest.mark.parametrize("existing", ["delivery", "prepared"])
def test_output_nonoverwrite(example: Example, existing: str) -> None:
    path = example.output if existing == "prepared" else Path(str(read(example.config)["output"]))
    path.mkdir() if existing == "prepared" else path.write_bytes(b"existing")
    with pytest.raises((ReplayError, FileExistsError)):
        delivery.prepare(example.config, example.output)


def test_all_current_media_pins_are_workflow_inputs(example: Example) -> None:
    delivery.prepare(example.config, example.output)
    stages = load_stages(example.output / "workflow.json")
    assert example.root / "source-1.json" in stages[0].inputs
    assert example.root / "quality-2.json" in stages[0].inputs
    assert example.root / "audio-packets.json" in stages[1].inputs
    assert example.root / "semantic.fcpxml" in stages[2].inputs


def test_changed_measurement_pin_is_rejected_without_repair(example: Example) -> None:
    path = example.root / "quality-1.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ReplayError, match="fingerprint changed"):
        delivery.assess(delivery.load(example.config))


@pytest.mark.parametrize(
    "field,value", [("rate", 25), ("width", 1280), ("height", 1920), ("output", "/elsewhere.mp4")]
)
def test_clock_or_authorized_delivery_root_mismatch(
    example: Example, field: str, value: object
) -> None:
    example.change(field, value)
    with pytest.raises(ReplayError, match="pinned edited clock"):
        delivery.load(example.config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("rate", 0),
        ("height", 20000),
        ("parts", []),
        ("parts", {}),
    ],
)
def test_configuration_domain_is_bounded(example: Example, field: str, value: object) -> None:
    example.change(field, value)
    with pytest.raises(ReplayError):
        delivery.load(example.config)


@pytest.mark.parametrize(
    "metric,value",
    [
        ("ssim", 0),
        ("ssim", 1.1),
        ("psnr", -1),
        ("vmaf", 101),
        ("psnr", "40"),
        ("vmaf", float("nan")),
    ],
)
def test_explicit_quality_thresholds_cannot_be_invalid(
    example: Example, metric: str, value: object
) -> None:
    limits = record(read(example.config)["thresholds"])
    limits[metric] = value
    example.change("thresholds", limits)
    with pytest.raises(ReplayError):
        delivery.load(example.config)


@pytest.mark.parametrize(
    "change", [{"startFrame": 16}, {"number": True}, {"frames": 0}, {"movie": None}, {"unknown": 1}]
)
def test_order_partition_and_movie_proof_pair_are_required(
    example: Example, change: dict[str, object]
) -> None:
    raw = read(example.config)
    rows = raw["parts"]
    assert isinstance(rows, list)
    record(rows[1]).update(change)
    example.change("parts", rows)
    with pytest.raises(ReplayError):
        delivery.load(example.config)


@pytest.mark.parametrize(
    "change",
    [
        {"dataHex": "gggggggggggg"},
        {"dataHex": "000000ff2801"},
        {"dataHex": "000000022800"},
        {"dataHex": "0000000228010000"},
        {"dataHex": "000000024001"},
        {"nalLengthBytes": 2},
    ],
)
def test_first_packet_raw_nal_measurements_are_validated(
    example: Example, change: dict[str, object]
) -> None:
    example.part(1, "firstPacket", change)
    with pytest.raises(ReplayError):
        delivery.assess(delivery.load(example.config))


def test_admit_cli_produces_a_readiness_artifact(
    example: Example, capsys: pytest.CaptureFixture[str]
) -> None:
    path = example.root / "ready.json"
    assert delivery.main([str(example.config), "--admit-output", str(path)]) == 0
    assert read(path)["assemblyReady"] is True
    assert "compressed-ready" in capsys.readouterr().out


def test_cli_pending_is_not_reported_ready(
    example: Example, capsys: pytest.CaptureFixture[str]
) -> None:
    example.change("audio", None)
    assert delivery.main([str(example.config)]) == 1
    assert "pending" in capsys.readouterr().out


def test_prepare_cli_and_code_closure(example: Example, capsys: pytest.CaptureFixture[str]) -> None:
    assert delivery.main([str(example.config), "--prepare", str(example.output)]) == 0
    assert "compressed-ready" in capsys.readouterr().out
    assert delivery.code_inputs()[0].name == "compressed_part_delivery.py"
    assert all(path.is_file() for path in delivery.code_inputs())


def test_shared_workflow_admission_reuses_without_overwriting(example: Example) -> None:
    delivery.prepare(example.config, example.output)
    admission = load_stages(example.output / "workflow.json")[0]
    calls: list[str] = []

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        calls.append(command[-2])
        assert delivery.main(list(command[3:])) == 0
        log.write_text("synthetic admission operation", encoding="utf-8")

    state = example.root / "state"
    assert run_stage(admission, state_directory=state, execute=execute) == "completed"
    saved = (example.output / "ready.json").read_bytes()
    assert run_stage(admission, state_directory=state, execute=execute) == "reused"
    assert calls == ["--admit-output", "--verify-admission", "--verify-admission"]
    assert (example.output / "ready.json").read_bytes() == saved
    (example.root / "part-1.m4v").write_bytes(b"different artifact")
    with pytest.raises(StageError, match="changed"):
        run_stage(admission, state_directory=state, execute=execute)


def test_changed_saved_admission_is_rejected(example: Example) -> None:
    path = example.root / "saved.json"
    write(path, {"assemblyReady": True})
    with pytest.raises(ReplayError, match="Saved admission"):
        delivery.main([str(example.config), "--verify-admission", str(path)])


def test_frame_budget_is_distinct_from_timestamp_tick_budget(example: Example) -> None:
    first = read(example.root / "probe-1.json")
    streams = first["streams"]
    assert isinstance(streams, list)
    record(streams[0]).update(time_base="1/480000000", duration_ts=240000000)
    frames = [{"pts": index * 16000000, "duration": 16000000} for index in range(15)]
    example.part(1, "videoProbe", {"streams": streams, "frames": frames})
    example.part(2, "videoProbe", {"streams": streams, "frames": frames})
    assert delivery.assess(delivery.load(example.config))["assemblyReady"] is True


def test_declared_ssim_and_psnr_need_no_fabricated_vmaf(example: Example) -> None:
    example.change("thresholds", {"ssim": 0.98, "psnr": 40})
    example.part(
        1, "quality", {"windows": [{"startFrame": 0, "endFrame": 15, "ssim": 0.99, "psnr": 50}]}
    )
    result = delivery.assess(delivery.load(example.config))
    assert result["assemblyReady"] is True
    observations = result["observations"]
    assert isinstance(observations, list)
    assert "vmaf" not in str(observations[0])


@pytest.mark.parametrize("value", [{}, {"unknown": 1}])
def test_empty_or_unknown_quality_policy_is_rejected(
    example: Example, value: dict[str, object]
) -> None:
    example.change("thresholds", value)
    with pytest.raises(ReplayError, match="policy"):
        delivery.load(example.config)


def measured_endpoint_limit(example: Example) -> None:
    example.audio(
        "decodedProbe",
        {"frames": [{"pts": 64, "nb_samples": 24000}, {"pts": 24064, "nb_samples": 24000}]},
    )
    example.audio(
        "packetProbe",
        {
            "packets": [
                {
                    "pts": 0,
                    "duration": 24064,
                    "side_data_list": [
                        {"side_data_type": "Skip Samples", "skip_samples": 64, "discard_padding": 0}
                    ],
                },
                {
                    "pts": 24064,
                    "duration": 24064,
                    "side_data_list": [
                        {"side_data_type": "Skip Samples", "skip_samples": 0, "discard_padding": 64}
                    ],
                },
            ]
        },
    )


def test_explicit_candidate_preserves_known_endpoint_limit(example: Example) -> None:
    measured_endpoint_limit(example)
    assert delivery.assess(delivery.load(example.config))["route"] == "prores-fallback"
    example.change("allowClockCandidate", True)
    result = delivery.assess(delivery.load(example.config))
    assert result["route"] == "compressed-clock-candidate"
    assert result["assemblyReady"] is True
    assert result["strictAccepted"] is False
    assert result["deliveryApproved"] is False
    assert "751/750" in str(result["reasons"])


@pytest.mark.parametrize(
    "role,change",
    [
        (
            "decodedProbe",
            {"frames": [{"pts": 0, "nb_samples": 24000}, {"pts": 24064, "nb_samples": 24000}]},
        ),
        ("nativeStatus", {"status": "Failed"}),
        ("fullAV", {"exitCode": 1}),
        (
            "packetProbe",
            {"packets": [{"pts": 0, "duration": 24064}, {"pts": 24064, "duration": 24064}]},
        ),
    ],
)
def test_candidate_permission_cannot_hide_internal_or_native_failures(
    example: Example, role: str, change: dict[str, object]
) -> None:
    example.change("allowClockCandidate", True)
    example.audio(role, change)
    assert delivery.assess(delivery.load(example.config))["assemblyReady"] is False


def test_single_part_acceptance_binds_current_inputs_and_excludes_whole_audio(
    example: Example,
) -> None:
    result = delivery.part_acceptance(example.config, number=1)
    assert result["kind"] == "accepted-compressed-part"
    assert result["accepted"] is True
    inputs = result["inputs"]
    assert isinstance(inputs, list)
    assert pin(example.root / "master-1.mov") in inputs
    assert pin(example.root / "master-2.mov") not in inputs
    assert pin(example.root / "whole.wav") not in inputs


def test_single_part_acceptance_failure_keeps_metric_numbers(example: Example) -> None:
    example.part(
        1,
        "quality",
        {"windows": [{"startFrame": 0, "endFrame": 15, "ssim": 0.9, "psnr": 50, "vmaf": 98}]},
    )
    result = delivery.part_acceptance(example.config, number=1)
    assert result["accepted"] is False
    assert "ssim 0.9" in str(result["reasons"])


@dataclass(frozen=True)
class SyntheticRetirement:
    inventory: tuple[metadata.Reference, ...]
    rebuild_required: bool = True


def retired_configuration(
    example: Example, monkeypatch: pytest.MonkeyPatch, *, rebuild: bool = True
) -> None:
    replacement = example.root / "retained-clean.mov"
    replacement.write_bytes(b"retained recipe clean input")
    recipe = example.root / "retained-recipe.json"
    write(recipe, {"kind": "native-rerender-recipe", "fallback": "rebuild-required"})
    provenance = write(
        example.root / "retirement-proof.json", {"synthetic": "read-only API fixture"}
    )
    raw = read(example.config)
    parts = raw["parts"]
    assert isinstance(parts, list)
    row = record(parts[0])
    row["retirement"] = provenance
    expected_source, expected_movie = row["master"], row["movie"]
    example.change("parts", parts)

    def verify(
        ref: metadata.Reference, *, source: metadata.Reference, movie: metadata.Reference, part: int
    ) -> SyntheticRetirement:
        assert ref.json() == provenance
        assert source.json() == expected_source
        assert movie.json() == expected_movie
        assert part == 1
        return SyntheticRetirement(
            (delivery._ref(pin(replacement)), delivery._ref(pin(recipe))), rebuild
        )

    monkeypatch.setattr(delivery, "verify_retired", verify)
    (example.root / "master-1.mov").unlink()  # Only synthetic media inside this test fixture.


def test_verified_retired_source_uses_fresh_recipe_not_absent_original(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    retired_configuration(example, monkeypatch)
    plan = delivery.load(example.config)
    assert isinstance(plan.parts[0].original, delivery.RetiredOriginal)
    result = delivery.admit(plan)
    assert result["assemblyReady"] is True
    assert result["fallbackRebuildRequired"] is True
    inputs = result["inputsVerified"]
    assert isinstance(inputs, list)
    assert not any(record(row)["path"] == str(example.root / "master-1.mov") for row in inputs)
    assert pin(example.root / "retained-clean.mov") in inputs
    assert pin(example.root / "retained-recipe.json") in inputs
    assert plan.parts[0].master.path == example.root / "master-1.mov"


def test_retired_original_cannot_return_through_other_dependency(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    retired_configuration(example, monkeypatch)
    plan = delivery.load(example.config)
    historical = plan.parts[0].master
    plan = replace(plan, fallback=historical)
    with pytest.raises(ReplayError, match="materialized input dependency"):
        delivery._references(plan)


def test_retired_source_requires_native_rebuild_contract(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    retired_configuration(example, monkeypatch, rebuild=False)
    with pytest.raises(ReplayError, match="rebuild prerequisites"):
        delivery.load(example.config)


def test_new_quality_comparison_cannot_use_retired_original(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    retired_configuration(example, monkeypatch)
    with pytest.raises(ReplayError, match="materialized original"):
        delivery.part_acceptance(example.config, number=1)


def test_retired_partial_compression_is_not_a_missing_file_bypass(
    example: Example, monkeypatch: pytest.MonkeyPatch
) -> None:
    retired_configuration(example, monkeypatch)
    raw = read(example.config)
    rows = raw["parts"]
    assert isinstance(rows, list)
    record(rows[0]).update(movie=None, evidence=None)
    example.change("parts", rows)
    with pytest.raises(ReplayError, match="retain its current original"):
        delivery.load(example.config)
