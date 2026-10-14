import json
import subprocess
import sys
import wave
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_part_assembly as assembly
from replay.core import ReplayError
from replay.fcp_partition import write_parts


def _tool(config: Path, *, decoder_threads: int = 0) -> assembly.FrameProbeTool:
    return assembly.FrameProbeTool.from_reference(
        assembly._reference(config.parent / "ffprobe"), decoder_threads=decoder_threads
    )


def test_voice_receipt_bound_without_reusing_part_audio(config: Path) -> None:
    raw = json.loads(config.read_text())
    receipt = config.parent / "voice-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "outputSHA256": raw["voice"]["sha256"],
                "sampleRate": 48000,
                "channels": 1,
                "outputSamples": 3200,
                "qualityAccepted": False,
            }
        )
    )
    new = config.parent / "new-config.json"
    assembly.prepare(
        Path(raw["partition"]["path"]),
        config.parent / "media",
        Path(raw["voice"]["path"]),
        new,
        voice_receipt=receipt,
    )
    assembly.probe(new, part=1, tool=_tool(new))
    assembly.probe(new, part=2, tool=_tool(new))
    output = config.parent / "receipt-assembly.fcpxml"
    lineage = assembly.assemble(new, output, name="receipt")
    assembly.verify_assembly(new, output)
    assert json.loads(lineage.read_text())["voiceReceipt"] == assembly._reference(receipt)
    bad = json.loads(receipt.read_text())
    bad["outputSHA256"] = "0" * 64
    receipt.write_text(json.dumps(bad))
    with pytest.raises(ReplayError):
        assembly.verify(new, part=1)
    altered = json.loads(new.read_text())
    altered["voiceReceipt"] = assembly._reference(receipt)
    new.write_text(json.dumps(altered))
    with pytest.raises(ReplayError, match="Whole voice receipt"):
        assembly.verify(new, part=1)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tool = tmp_path / "ffprobe"
    tool.write_bytes(b"synthetic pinned executable")
    tool.chmod(0o700)
    semantic = tmp_path / "semantic.fcpxml"
    semantic.write_text(
        """<fcpxml version="1.11"><resources><format id="f" frameDuration="1/30s"
      width="1080" height="1920"/></resources><library><event><project><sequence format="f"
      duration="1/15s" tcStart="0s"><spine>
      <asset-clip offset="0s" start="100s" duration="1/30s"><title offset="100s" start="3600s"
      duration="1/30s"><text>first</text></title></asset-clip>
      <asset-clip offset="1/30s" start="200s" duration="1/30s"><title offset="200s" start="3600s"
      duration="1/30s"><text>second</text></title></asset-clip>
      </spine></sequence></project></event></library></fcpxml>""",
        encoding="utf-8",
    )
    manifest = write_parts(
        semantic, tmp_path / "parts", target_duration=Fraction(1, 30), maximum_titles=1
    )
    voice = tmp_path / "voice.wav"
    with wave.open(str(voice), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\x00\x00" * 3200)
    media = tmp_path / "media"
    media.mkdir()
    (media / "part-0001-master.mov").write_bytes(b"synthetic first master")
    (media / "part-0002-master.mov").write_bytes(b"synthetic second master")
    result = tmp_path / "assembly.json"
    assembly.prepare(manifest, media, voice, result)
    observed = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1080,
                "height": 1920,
                "start_pts": 0,
                "avg_frame_rate": "30/1",
                "duration_ts": 1,
                "time_base": "1/30",
                "nb_read_frames": "1",
            }
        ],
        "frames": [{"best_effort_timestamp": 0}],
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, json.dumps(observed), ""),
    )
    return result


def test_exact_assembly_one_whole_voice_and_semantic_srt_lineage(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _tool(config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assembly",
            "probe",
            str(config),
            "--part",
            "1",
            "--ffprobe",
            str(tool.path),
            "--ffprobe-sha256",
            tool.sha256,
            "--decoder-threads",
            "0",
        ],
    )
    assembly.main()
    assembly.probe(config, part=2, tool=_tool(config))
    monkeypatch.setattr(sys, "argv", ["assembly", "verify", str(config), "--part", "1"])
    assembly.main()
    output = config.parent / "assembled.fcpxml"
    monkeypatch.setattr(
        sys, "argv", ["assembly", "assemble", str(config), str(output), "--name", "whole"]
    )
    assembly.main()
    root = ET.parse(output).getroot()
    monkeypatch.setattr(sys, "argv", ["assembly", "verify-assembly", str(config), str(output)])
    assembly.main()
    clips = root.findall(".//spine/asset-clip")
    assert [
        (clip.get("offset"), clip.get("duration"), clip.get("srcEnable")) for clip in clips
    ] == [("0/1s", "1/30s", "video"), ("1/30s", "1/30s", "video")]
    assert root.findall(".//title") == []
    audio = root.findall(".//spine/asset-clip/asset-clip")
    assert len(audio) == 1
    assert audio[0].get("duration") == "1/15s"
    assert audio[0].get("start") == "0s"
    receipt = json.loads(output.with_suffix(".lineage.json").read_text())
    assert receipt["partAudioDisabled"] is True
    assert receipt["nativeAssemblyRenderVerified"] is False
    assert len(receipt["masterProofs"]) == 2
    assert receipt["semanticXML"]["path"].endswith("semantic.fcpxml")


def test_prepare_cli(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = config.parent
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "assembly",
            "prepare",
            str(root / "parts/manifest.json"),
            str(root / "media"),
            str(root / "voice.wav"),
            str(root / "new-config.json"),
        ],
    )
    assembly.main()
    assert (root / "new-config.json").is_file()


@pytest.mark.parametrize("part", [0, 3])
def test_missing_or_excess_part_index(config: Path, part: int) -> None:
    with pytest.raises(ReplayError, match="index"):
        assembly.probe(config, part=part, tool=_tool(config))
    with pytest.raises(ReplayError, match="index"):
        assembly.verify(config, part=part)


@pytest.mark.parametrize("value", [None, "relative", "\x00", "/a/../b"])
def test_path_validation(value: object) -> None:
    with pytest.raises(ReplayError, match="absolute"):
        assembly._path(value)


def test_probe_failure_is_not_accepted(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "decode failed"),
    )
    with pytest.raises(ReplayError, match="probe failed"):
        assembly.probe(config, part=1, tool=_tool(config))


@pytest.mark.parametrize(
    "kind", ["streams", "video", "clock", "geometry", "missing-frame", "timestamp"]
)
def test_incomplete_or_changed_frame_clock(config: Path, kind: str) -> None:
    plan, _, _ = assembly._config(config)
    video: dict[str, object] = {
        "codec_type": "video",
        "width": 1080,
        "height": 1920,
        "start_pts": 0,
        "avg_frame_rate": "30/1",
        "duration_ts": 1,
        "time_base": "1/30",
        "nb_read_frames": "1",
    }
    raw: dict[str, object] = {"streams": [video], "frames": [{"best_effort_timestamp": 0}]}
    if kind == "streams":
        raw["streams"] = None
    elif kind == "video":
        raw["streams"] = []
    elif kind == "clock":
        video["avg_frame_rate"] = "invalid"
    elif kind == "geometry":
        video["width"] = 720
    elif kind == "missing-frame":
        raw["frames"] = []
    else:
        raw["frames"] = [{"best_effort_timestamp": 1}]
    with pytest.raises(ReplayError):
        assembly._check_video(raw, plan, 0)


def test_tampered_movie_and_incomplete_binding_rejected(config: Path) -> None:
    assembly.probe(config, part=1, tool=_tool(config))
    assembly.probe(config, part=2, tool=_tool(config))
    (config.parent / "media/part-0001-master.mov").write_bytes(b"changed movie")
    with pytest.raises(ReplayError, match="hash changed"):
        assembly.assemble(config, config.parent / "bad.fcpxml", name="bad")
    raw = json.loads(config.read_text())
    raw["parts"] = []
    config.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="every ordered"):
        assembly.probe(config, part=1, tool=_tool(config))


def test_wrong_whole_voice_duration_rejected(config: Path) -> None:
    voice = config.parent / "voice.wav"
    with wave.open(str(voice), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\x00\x00" * 1600)
    raw = json.loads(config.read_text())
    raw["voice"] = assembly._reference(voice)
    config.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="processed voice"):
        assembly.assemble(config, config.parent / "bad-voice.fcpxml", name="bad voice")


def test_proof_cannot_be_reordered(config: Path) -> None:
    proof = assembly.probe(config, part=1, tool=_tool(config))
    raw = json.loads(proof.read_text())
    raw["part"] = 2
    proof.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="ordered partition"):
        assembly.assemble(config, config.parent / "wrong-order.fcpxml", name="wrong order")


@pytest.mark.parametrize("change", ["caption", "position", "template"])
def test_old_master_proof_rejected_after_semantic_revision(config: Path, change: str) -> None:
    proof = assembly.probe(config, part=1, tool=_tool(config))
    saved_proof = proof.read_bytes()
    original = json.loads(config.read_text())
    old_manifest = Path(original["partition"]["path"])
    saved_manifest = old_manifest.read_bytes()
    source = config.parent / "semantic.fcpxml"
    root = ET.parse(source).getroot()
    title = root.find(".//title")
    assert title is not None
    if change == "caption":
        text = title.find("text")
        assert text is not None
        text.text = "corrected phrase boundary"
    elif change == "position":
        ET.SubElement(title, "param", {"name": "Y", "key": "position", "value": "12"})
    else:
        resources = root.find("resources")
        assert resources is not None
        ET.SubElement(resources, "effect", {"id": "new-title", "uid": "new-immutable-template"})
        title.set("ref", "new-title")
    revised_source = config.parent / "revised-semantic.fcpxml"
    ET.ElementTree(root).write(revised_source)
    revised_manifest = write_parts(
        revised_source,
        config.parent / "revised-parts",
        target_duration=Fraction(1, 30),
        maximum_titles=1,
    )
    revised_config = config.parent / "revised-assembly.json"
    original["partition"] = assembly._reference(revised_manifest)
    revised_config.write_text(json.dumps(original))
    with pytest.raises(ReplayError, match="ordered partition"):
        assembly.verify(revised_config, part=1)
    assert proof.read_bytes() == saved_proof
    assert old_manifest.read_bytes() == saved_manifest


@pytest.mark.parametrize(
    "change", ["lineage", "duration", "format", "resources", "clip-audio", "voice-clock"]
)
def test_assembly_validator_rejects_clock_and_audio_changes(config: Path, change: str) -> None:
    assembly.probe(config, part=1, tool=_tool(config))
    assembly.probe(config, part=2, tool=_tool(config))
    output = config.parent / "assembled.fcpxml"
    receipt_path = assembly.assemble(config, output, name="whole")
    receipt = json.loads(receipt_path.read_text())
    root = ET.parse(output).getroot()
    if change == "lineage":
        receipt["semanticSRTSHA256"] = "0" * 64
    elif change == "duration":
        sequence = root.find(".//sequence")
        assert sequence is not None
        sequence.set("duration", "1s")
    elif change == "resources":
        resources = root.find("resources")
        assert resources is not None
        ET.SubElement(resources, "asset", id="extra")
    elif change == "format":
        format_node = root.find("./resources/format")
        assert format_node is not None
        format_node.set("frameDuration", "1/24s")
    elif change == "clip-audio":
        clip = root.find(".//spine/asset-clip")
        assert clip is not None
        clip.set("srcEnable", "all")
    else:
        voice = root.find(".//spine/asset-clip/asset-clip")
        assert voice is not None
        voice.set("start", "1s")
    ET.ElementTree(root).write(output)
    receipt["assemblyXML"] = assembly._reference(output)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ReplayError):
        assembly.verify_assembly(config, output)


@pytest.mark.parametrize("threads", [0, 1, 64])
def test_pinned_absolute_probe_records_actual_tool_and_decoder_policy(
    config: Path, monkeypatch: pytest.MonkeyPatch, threads: int
) -> None:
    commands: list[list[str]] = []
    observed = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1080,
                "height": 1920,
                "start_pts": 0,
                "avg_frame_rate": "30/1",
                "duration_ts": 1,
                "time_base": "1/30",
                "nb_read_frames": "1",
            }
        ],
        "frames": [{"best_effort_timestamp": 0}],
    }

    def execute(arguments: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        assert options == {"capture_output": True, "text": True, "check": False, "timeout": 1800}
        return subprocess.CompletedProcess(arguments, 0, json.dumps(observed), "")

    monkeypatch.setattr(subprocess, "run", execute)
    tool = _tool(config, decoder_threads=threads)
    proof = assembly.probe(config, part=1, tool=tool)
    command = commands[0]
    assert command[0] == str(tool.path)
    assert command[command.index("-threads") + 1] == str(threads)
    assert "-count_frames" in command and "-show_frames" in command
    assert command[command.index("-select_streams") + 1] == "v:0"
    data = json.loads(proof.read_text())
    assert data["observed"] == observed
    assert data["probeProvenance"] == tool.provenance()
    assert assembly.verify(config, part=1) == proof


@pytest.mark.parametrize("value", [-1, 65, True, "0", None])
def test_invalid_decoder_policy_cannot_select_a_fallback(config: Path, value: object) -> None:
    with pytest.raises(ReplayError):
        assembly.FrameProbeTool.from_reference(
            assembly._reference(config.parent / "ffprobe"), decoder_threads=value
        )


@pytest.mark.parametrize("kind", ["relative", "digest", "missing-digest", "linked"])
def test_invalid_probe_tool_binding_rejected(config: Path, kind: str) -> None:
    tool = config.parent / "ffprobe"
    value: dict[str, object] = {**assembly._reference(tool)}
    if kind == "relative":
        value["path"] = "ffprobe"
    elif kind == "digest":
        value["sha256"] = "x" * 64
    elif kind == "missing-digest":
        value.pop("sha256")
    else:
        link = config.parent / "linked-probe"
        link.symlink_to(tool)
        value["path"] = str(link)
    with pytest.raises(ReplayError):
        assembly.FrameProbeTool.from_reference(value)


@pytest.mark.parametrize("kind", ["changed", "missing", "not-executable"])
def test_tool_failure_before_probe_has_no_effects(
    config: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    tool = _tool(config)
    if kind == "changed":
        tool.path.write_bytes(b"changed executable")
    elif kind == "missing":
        tool.path.unlink()
    else:
        tool.path.chmod(0o600)

    def unexpected(*arguments: object, **options: object) -> None:
        pytest.fail("An invalid tool must not dispatch a probe")

    monkeypatch.setattr(subprocess, "run", unexpected)
    with pytest.raises(ReplayError):
        assembly.probe(config, part=1, tool=tool)
    assert not (config.parent / "assembly-part-0001-proof.json").exists()


@pytest.mark.parametrize("kind", ["tool", "movie"])
def test_mutation_during_probe_refuses_completed_proof(
    config: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    tool = _tool(config)
    observed = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1080,
                "height": 1920,
                "start_pts": 0,
                "avg_frame_rate": "30/1",
                "duration_ts": 1,
                "time_base": "1/30",
                "nb_read_frames": "1",
            }
        ],
        "frames": [{"best_effort_timestamp": 0}],
    }

    def execute(arguments: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        target = tool.path if kind == "tool" else config.parent / "media/part-0001-master.mov"
        target.write_bytes(b"changed during observation")
        return subprocess.CompletedProcess(arguments, 0, json.dumps(observed), "")

    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(ReplayError, match="changed"):
        assembly.probe(config, part=1, tool=tool)
    assert not (config.parent / "assembly-part-0001-proof.json").exists()


@pytest.mark.parametrize("stdout", ["{", "é" * 101])
def test_invalid_json_and_utf8_byte_budget_preserve_no_proof(
    config: Path, monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    monkeypatch.setattr(assembly, "MAX_JSON_BYTES", 200)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout, ""),
    )
    with pytest.raises(ReplayError, match="probe"):
        assembly.probe(config, part=1, tool=_tool(config))
    assert not (config.parent / "assembly-part-0001-proof.json").exists()


def test_existing_proof_is_not_probed_or_overwritten(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proof = assembly.probe(config, part=1, tool=_tool(config))
    saved = proof.read_bytes()

    def unexpected(*arguments: object, **options: object) -> None:
        pytest.fail("A completed proof cannot be overwritten by another invocation")

    monkeypatch.setattr(subprocess, "run", unexpected)
    with pytest.raises(ReplayError, match="absent"):
        assembly.probe(config, part=1, tool=_tool(config))
    assert proof.read_bytes() == saved
    assert assembly.verify(config, part=1) == proof


def test_historical_proof_keeps_unknown_tool_without_rewrite(config: Path) -> None:
    proof = assembly.probe(config, part=1, tool=_tool(config))
    data = json.loads(proof.read_text())
    data.pop("probeProvenance")
    proof.write_text(json.dumps(data))
    saved = proof.read_bytes()
    (config.parent / "ffprobe").write_bytes(b"later tool version")
    assert assembly._probe_provenance(data) is None
    assert assembly.verify(config, part=1) == proof
    assert proof.read_bytes() == saved


@pytest.mark.parametrize("kind", ["version", "threads", "digest"])
def test_malformed_new_tool_provenance_is_not_historical_unknown(config: Path, kind: str) -> None:
    proof = assembly.probe(config, part=1, tool=_tool(config))
    data = json.loads(proof.read_text())
    if kind == "version":
        data["probeProvenance"]["version"] = True
    elif kind == "threads":
        data["probeProvenance"]["decoderThreads"] = -1
    else:
        data["probeProvenance"]["executable"]["sha256"] = "not-a-pin"
    proof.write_text(json.dumps(data))
    with pytest.raises(ReplayError):
        assembly.verify(config, part=1)


@pytest.mark.parametrize(("rate", "width", "height"), [(24, 720, 1280), (60, 1920, 1080)])
@pytest.mark.parametrize("change", ["valid", "missing", "duplicate", "reordered", "wrong-geometry"])
def test_full_rational_pts_remain_required_for_different_media(
    config: Path, rate: int, width: int, height: int, change: str
) -> None:
    original, _, _ = assembly._config(config)
    frame = Fraction(1, rate)
    format_node = ET.Element("format", width=str(width), height=str(height))
    plan = assembly.Timeline(
        original.manifest,
        original.semantic,
        format_node,
        frame,
        frame * 3,
        ({"duration": f"3/{rate}s"},),
    )
    stamps = [0, 1, 2]
    if change == "missing":
        stamps.pop()
    elif change == "duplicate":
        stamps[1] = 0
    elif change == "reordered":
        stamps.reverse()
    observed = {
        "streams": [
            {
                "codec_type": "video",
                "width": width + (change == "wrong-geometry"),
                "height": height,
                "start_pts": 0,
                "avg_frame_rate": str(rate),
                "duration_ts": 3,
                "time_base": str(frame),
                "nb_read_frames": "3",
            }
        ],
        "frames": [{"best_effort_timestamp": stamp} for stamp in stamps],
    }
    if change == "valid":
        assembly._check_video(observed, plan, 0)
    else:
        with pytest.raises(ReplayError):
            assembly._check_video(observed, plan, 0)


def test_probe_timeout_keeps_no_completed_output(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def execute(arguments: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(arguments, 1800)

    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(subprocess.TimeoutExpired):
        assembly.probe(config, part=1, tool=_tool(config))
    assert not (config.parent / "assembly-part-0001-proof.json").exists()
