import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from replay import native_export_receipt as native
from replay.core import ReplayError
from replay.stages import file_digest


@pytest.fixture
def plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    movie = tmp_path / "part.mov"
    movie.write_bytes(b"synthetic media")
    part = tmp_path / "part.fcpxml"
    part.write_text(
        '<fcpxml><library><event><project name="Synthetic Part"/></event></library></fcpxml>'
    )
    manifest = tmp_path / "partition.json"
    manifest.write_text(json.dumps({"parts": [{"path": str(part), "sha256": file_digest(part)}]}))
    assembly = tmp_path / "assembly.json"
    assembly.write_text(
        json.dumps({"partition": {"path": str(manifest), "sha256": file_digest(manifest)}})
    )
    proof = tmp_path / "proof.json"
    video = {
        "codec_type": "video",
        "codec_name": "prores",
        "profile": "LT",
        "width": 10,
        "height": 20,
        "avg_frame_rate": "30/1",
        "duration_ts": 2,
        "time_base": "1/30",
        "start_pts": 0,
    }
    proof.write_text(
        json.dumps(
            {
                "movie": {"path": str(movie), "sha256": file_digest(movie)},
                "part": 1,
                "observed": {"streams": [video]},
            }
        )
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "bundleID": "com.apple.FinalCutApp",
                "project": "Synthetic Part",
                "directory": str(tmp_path),
                "fileName": "part.mov",
                "format": "videoAndAudio",
                "codec": "Apple ProRes 422 LT",
                "allowForeground": True,
            }
        )
    )
    response = tmp_path / "response.json"
    response.write_text(
        json.dumps(
            {
                "ok": True,
                "data": {
                    "isError": False,
                    "structuredContent": {
                        "pid": 123,
                        "retrySafe": False,
                        "finalCut": {
                            "language": "ja",
                            "vocabularyVerified": True,
                            "project": {"name": "Synthetic Part"},
                            "export": {
                                "outputPath": str(movie),
                                "format": "ビデオとオーディオ",
                                "action": "保存のみ",
                                "codec": "Apple ProRes 422 LT",
                                "foregroundRestored": True,
                                "completionVerified": False,
                            },
                        },
                    },
                },
            }
        )
    )
    background = tmp_path / "background.json"
    background.write_text(
        json.dumps(
            {
                "ok": True,
                "data": {
                    "isError": False,
                    "structuredContent": {
                        "pid": 123,
                        "elements": [
                            {"role": "AXStaticText", "value": "Sharing"},
                            {"identifier": "_NS:166", "value": "Idle"},
                        ],
                    },
                },
            }
        )
    )
    ffmpeg, ffprobe = tmp_path / "ffmpeg", tmp_path / "ffprobe"
    ffmpeg.write_bytes(b"synthetic ffmpeg")
    ffprobe.write_bytes(b"synthetic ffprobe")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "part": 1,
                "audioSampleRate": 48000,
                "audioChannels": 2,
                **{
                    name: {"path": str(path), "sha256": file_digest(path)}
                    for name, path in (
                        ("assembly", assembly),
                        ("exportRequest", request),
                        ("exportResponse", response),
                        ("background", background),
                        ("ffmpeg", ffmpeg),
                        ("ffprobe", ffprobe),
                    )
                },
            }
        )
    )
    audio = {
        "codec_type": "audio",
        "codec_name": "pcm_s24le",
        "sample_rate": "48000",
        "channels": 2,
        "duration_ts": 3200,
        "time_base": "1/48000",
        "start_pts": 0,
    }
    monkeypatch.setattr(native, "verify", lambda config, *, part: proof)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps({"streams": [video, audio]}).encode(), b""
        ),
    )
    return config


def test_independent_decode_binds_dispatch_without_claiming_quality(plan: Path) -> None:
    output = plan.parent / "completion.json"
    native.complete(plan, output)
    receipt = json.loads(output.read_text())
    assert receipt["completionVerified"] is True
    assert receipt["nativeReportedCompletion"] is False
    assert receipt["qualityApproved"] is False
    assert receipt["project"] == "Synthetic Part"
    assert receipt["fullAVDecodeCommand"][1:6] == ["-nostdin", "-v", "error", "-xerror", "-i"]
    assert receipt["fullAVDecodeCommand"][7:] == [
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ReplayError, match="exists"):
        native.complete(plan, output)


@pytest.fixture
def recovery_plan(plan: Path) -> Path:
    config = json.loads(plan.read_text())
    response = Path(config["exportResponse"]["path"])
    original = json.loads(response.read_text())
    observation = plan.parent / "project-observation.json"
    observation.write_text(json.dumps(original))
    os.utime(observation, ns=(1, 1))
    response.write_text(
        json.dumps(
            {
                "ok": False,
                "error": {
                    "code": "mcp_tool_error",
                    "message": json.dumps({"retrySafe": False, "error": "AXError -25204"}),
                },
            }
        )
    )
    movie = plan.parent / "part.mov"
    movie.unlink()
    movie.write_bytes(b"synthetic media")
    background = Path(config["background"]["path"])
    background.touch()
    config["projectObservation"] = {"path": str(observation), "sha256": file_digest(observation)}
    config["exportResponse"]["sha256"] = file_digest(response)
    plan.write_text(json.dumps(config))
    return plan


def test_recovery_preserves_uncertain_dispatch(recovery_plan: Path) -> None:
    output = recovery_plan.parent / "recovered.json"
    native.complete(recovery_plan, output, recover=True)
    receipt = json.loads(output.read_text())
    assert receipt["nativeDispatchVerified"] is False
    assert receipt["recoveredAfterUncertainDispatch"] is True
    assert receipt["completionVerified"] is True
    assert receipt["qualityApproved"] is False
    assert receipt["nativeReportedCompletion"] is False
    with pytest.raises(ReplayError, match="did not succeed"):
        native.complete(recovery_plan, recovery_plan.parent / "normal.json")


def test_recovery_rejects_media_created_before_dispatch(recovery_plan: Path) -> None:
    config = json.loads(recovery_plan.read_text())
    response = Path(config["exportResponse"]["path"])
    data = response.read_bytes()
    response.unlink()
    response.write_bytes(data)
    Path(config["background"]["path"]).touch()
    with pytest.raises(ReplayError, match="pre-existing media"):
        native.complete(recovery_plan, recovery_plan.parent / "rejected.json", recover=True)


@pytest.mark.parametrize(
    "change", ["pid", "project", "old-media", "stale", "failure", "json", "retry"]
)
def test_recovery_rejects_unbound_evidence(recovery_plan: Path, change: str) -> None:
    config = json.loads(recovery_plan.read_text())
    if change == "old-media":
        os.utime(recovery_plan.parent / "part.mov", ns=(1, 1))
    elif change == "stale":
        os.utime(Path(config["background"]["path"]), ns=(1, 1))
    else:
        key = "projectObservation" if change in {"pid", "project"} else "exportResponse"
        path = Path(config[key]["path"])
        raw = json.loads(path.read_text())
        if change == "pid":
            raw["data"]["structuredContent"]["pid"] = 456
        elif change == "project":
            raw["data"]["structuredContent"]["finalCut"]["project"]["name"] = "Other"
        elif change == "failure":
            raw["ok"] = True
        elif change == "json":
            raw["error"]["message"] = "not json"
        else:
            raw["error"]["message"] = json.dumps({"retrySafe": True, "error": "AXError -25204"})
        path.write_text(json.dumps(raw))
        if key == "projectObservation":
            os.utime(path, ns=(1, 1))
        config[key]["sha256"] = file_digest(path)
        recovery_plan.write_text(json.dumps(config))
    with pytest.raises(ReplayError):
        native.complete(recovery_plan, recovery_plan.parent / "rejected.json", recover=True)


@pytest.mark.parametrize(
    "change",
    [
        "project",
        "codec",
        "format",
        "action",
        "language",
        "path",
        "claimed-complete",
        "pid",
        "failed",
        "native-error",
        "filename",
        "idle",
        "stale",
    ],
)
def test_rejects_unbound_or_unfinished_export(plan: Path, change: str) -> None:
    config = json.loads(plan.read_text())
    key = "exportResponse"
    raw = json.loads(Path(config[key]["path"]).read_text())
    export = raw["data"]["structuredContent"]["finalCut"]["export"]
    if change == "project":
        raw["data"]["structuredContent"]["finalCut"]["project"]["name"] = "Other"
    elif change == "codec":
        export["codec"] = "Other"
    elif change == "format":
        export["format"] = "Video Only"
    elif change == "action":
        export["action"] = "Open with application"
    elif change == "language":
        raw["data"]["structuredContent"]["finalCut"]["language"] = "unknown"
        export.pop("action")
        export.pop("format")
    elif change == "path":
        export["outputPath"] = str(plan.parent / "other.mov")
    elif change == "claimed-complete":
        export["completionVerified"] = True
    elif change == "pid":
        raw["data"]["structuredContent"]["pid"] = 456
    elif change == "failed":
        raw["ok"] = False
    elif change == "native-error":
        raw["data"]["isError"] = True
    elif change == "filename":
        key = "exportRequest"
        raw = json.loads(Path(config[key]["path"]).read_text())
        raw["fileName"] = "../escape.mov"
    else:
        key = "background"
        raw = json.loads(Path(config[key]["path"]).read_text())
        if change == "idle":
            raw["data"]["structuredContent"]["elements"][1]["value"] = "Running"
    path = Path(config[key]["path"])
    path.write_text(json.dumps(raw))
    if change == "stale":
        os.utime(path, (1, 1))
    config[key]["sha256"] = file_digest(path)
    plan.write_text(json.dumps(config))
    with pytest.raises(ReplayError):
        native.complete(plan, plan.parent / "completion.json")


@pytest.mark.parametrize(
    "change", ["empty", "one-kind", "clock", "codec", "size", "decode", "mutate"]
)
def test_requires_real_metadata_and_full_decode(
    plan: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    original = subprocess.run

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        result = original(command, capture_output=True, check=True, timeout=600)
        if Path(command[0]).name == "ffmpeg":
            if change == "decode":
                raise subprocess.CalledProcessError(1, command)
            if change == "mutate":
                (plan.parent / "part.mov").write_bytes(b"changed")
            return result
        value = json.loads(result.stdout)
        if change == "empty":
            value["streams"] = []
        elif change == "one-kind":
            value["streams"][1]["codec_type"] = "video"
        elif change == "clock":
            value["streams"][0]["time_base"] = "bad"
        elif change == "codec":
            value["streams"][0]["codec_name"] = "h264"
        if change == "size":
            return subprocess.CompletedProcess(command, 0, b" " * (native.MAX_METADATA_BYTES + 1))
        return subprocess.CompletedProcess(command, 0, json.dumps(value).encode())

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises((ReplayError, subprocess.CalledProcessError)):
        native.complete(plan, plan.parent / "completion.json")
    assert not (plan.parent / "completion.json").exists()


def test_cli_and_invalid_configuration(plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["native-export-receipt", str(plan), str(plan.parent / "result.json")]
    )
    native.main()
    raw = json.loads(plan.read_text())
    raw["part"] = True
    plan.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="configuration"):
        native.complete(plan, plan.parent / "other.json")


@pytest.mark.parametrize(
    "change",
    ["valid", "unknown", "extra-video", "extra-audio", "duplicate", "clock", "origin", "rate"],
)
def test_native_timecode_stream_contract(
    plan: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    original = subprocess.run

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        result = original(command, capture_output=True, check=True, timeout=600)
        if Path(command[0]).name != "ffprobe":
            return result
        raw = json.loads(result.stdout)
        timecode: dict[str, object] = {
            "codec_type": "data",
            "codec_tag_string": "tmcd",
            "start_pts": 0,
            "avg_frame_rate": "30/1",
            "duration_ts": 40,
            "time_base": "1/600",
        }
        if change == "unknown":
            timecode["codec_tag_string"] = "other"
        elif change in {"extra-video", "extra-audio"}:
            timecode["codec_type"] = change.removeprefix("extra-")
        elif change == "clock":
            timecode["duration_ts"] = 41
        elif change == "origin":
            timecode["start_pts"] = 1
        elif change == "rate":
            timecode["avg_frame_rate"] = "25/1"
        raw["streams"].append(timecode)
        if change == "duplicate":
            raw["streams"].append(timecode)
        return subprocess.CompletedProcess(command, 0, json.dumps(raw).encode())

    monkeypatch.setattr(subprocess, "run", run)
    output = plan.parent / "completion.json"
    if change == "valid":
        native.complete(plan, output)
        assert json.loads(output.read_text(encoding="utf-8"))["completionVerified"] is True
    else:
        with pytest.raises(ReplayError):
            native.complete(plan, output)
