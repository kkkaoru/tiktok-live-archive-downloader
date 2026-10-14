import json
import sys
import wave
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_transformed_source as adapter
from replay.core import ReplayError
from replay.delivery_source_assembly import Chunk
from replay.fcp_part_assembly import _reference
from replay.fcp_project import GAUSSIAN_UID
from replay.subtitle_clock import VideoFrameClock


def test_proxy_candidate_binding_is_recorded_and_reverified(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = json.loads(config.read_text())
    original = config.parent / "original.mov"
    binding = {
        "policy": "aspect-fit-cfr-candidate",
        "original": _reference(original),
        "proxy": _reference(original),
        "proxyClock": _reference(original),
        "sourceConfig": _reference(original),
    }
    raw["proxyBinding"] = binding
    config.write_text(json.dumps(raw))
    monkeypatch.setattr(
        adapter,
        "bind_proxy",
        lambda root, config, **kwargs: (original, {"binding": binding}),
    )
    output = config.parent / "proxy.fcpxml"
    receipt_path = adapter.transform(config, output, name="proxy")
    adapter.verify(config, output)
    receipt = json.loads(receipt_path.read_text())
    receipt["proxyCorrespondence"]["binding"]["policy"] = "changed"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ReplayError, match="Proxy correspondence lineage"):
        adapter.verify(config, output)


def test_explicit_terminal_display_cell_keeps_physical_media_end(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        adapter,
        "clean_source",
        lambda raw: (
            config.parent / "original.mov",
            config.parent / "clean.mov",
            Fraction(149, 50),
        ),
    )
    output = config.parent / "terminal.fcpxml"
    with pytest.raises(ReplayError, match="physical coverage"):
        adapter.transform(config, output, name="terminal")
    raw = json.loads(config.read_text())
    raw["terminalFrameCell"] = True
    config.write_text(json.dumps(raw))
    receipt = adapter.transform(config, output, name="terminal")
    root = ET.parse(output).getroot()
    asset = root.find("./resources/asset[@id='s']")
    clip = root.find(".//spine/asset-clip")
    assert asset is not None and clip is not None
    assert Fraction(asset.attrib["duration"][:-1]) == Fraction(149, 50)
    assert Fraction(clip.attrib["duration"][:-1]) == 1
    boundary = json.loads(receipt.read_text())["terminalFrameCell"]
    assert boundary["physicalEnd"] == "149/50"
    assert boundary["displayGridEnd"] == "3"
    assert boundary["observedCoverageExtended"] is False


def test_trimmed_clean_source_preserves_voice_and_motion_phase(config: Path) -> None:
    raw = json.loads(config.read_text())
    raw["outputWindow"] = ["1/2s", "1s"]
    config.write_text(json.dumps(raw))
    output = config.parent / "window.fcpxml"
    adapter.transform(config, output, name="window")
    root = ET.parse(output).getroot()
    title = root.find(".//title")
    voice = root.find(".//spine/asset-clip/asset-clip")
    assert title is not None and voice is not None
    assert Fraction(title.attrib["start"][:-1]) == Fraction(7201, 2)
    assert Fraction(voice.attrib["start"][:-1]) == Fraction(1, 2)
    assert Fraction(voice.attrib["duration"][:-1]) == Fraction(1, 2)


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    original = tmp_path / "original.mov"
    original.write_bytes(b"original")
    clean = tmp_path / "clean.mov"
    clean.write_bytes(b"clean uncaptioned")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    evidence = tmp_path / "clock.json"
    evidence.write_text('{"clock":"verified"}')
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "ranges": [
                    {
                        "media": str(clean),
                        "sha256": _reference(clean)["sha256"],
                        "sourceFirstPTS": 0,
                        "sourceEndPTS": 90,
                        "frames": 90,
                        "timeBase": "1/30",
                    }
                ]
            }
        )
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "exactPresentationClockVerified": True,
                "compressedPacketsIdentical": True,
                "output": _reference(clean),
                "indexSHA256": _reference(index)["sha256"],
                "frames": 90,
                "inputs": {str(manifest): _reference(manifest)["sha256"]},
                "probeEvidence": {str(evidence): _reference(evidence)["sha256"]},
            }
        )
    )

    def chunks(path: Path, *, sha256: str, bindings: dict[str, str]) -> tuple[Chunk, ...]:
        bindings[str(path)] = sha256
        return (
            Chunk(
                clean,
                VideoFrameClock(tuple(range(90)), Fraction(1, 30), 1),
                (str(original), _reference(original)["sha256"]),
            ),
        )

    monkeypatch.setattr(adapter, "load_chunks", chunks)
    voice = tmp_path / "voice.wav"
    with wave.open(str(voice), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\0\0" * 48000)
    voice_receipt = tmp_path / "voice.json"
    voice_receipt.write_text(
        json.dumps(
            {
                "outputSHA256": _reference(voice)["sha256"],
                "sampleRate": 48000,
                "channels": 1,
                "outputSamples": 48000,
            }
        )
    )
    semantic = tmp_path / "semantic.fcpxml"
    semantic.write_text(f'''<fcpxml version="1.11"><resources>
      <format id="f" frameDuration="1/30s" width="1080" height="1920"/>
      <asset id="s" start="0s" duration="3s" hasVideo="1" hasAudio="1">
        <media-rep src="{original.as_uri()}"/></asset>
      <asset id="a" hasVideo="0" hasAudio="1"><media-rep src="{voice.as_uri()}"/></asset>
      <effect id="g" uid="{GAUSSIAN_UID}"/></resources>
      <library><event><project><sequence format="f" duration="1s" tcStart="0s"><spine>
      <asset-clip ref="s" offset="0s" start="2s" duration="1s">
        <asset-clip ref="a" srcEnable="audio" offset="2s" start="0s" duration="1s"/>
        <asset-clip ref="s" lane="1" srcEnable="video" offset="2s" start="2s" duration="1s">
          <adjust-crop><trim-rect/></adjust-crop>
          <audio-channel-source srcCh="1, 2" active="0"/><filter-video ref="g"/>
        </asset-clip>
        <title offset="2s" start="3600s" duration="1s"><text>hello</text></title>
      </asset-clip></spine></sequence></project></event></library></fcpxml>''')
    result = tmp_path / "config.json"
    result.write_text(
        json.dumps(
            {
                "version": 1,
                "removeSourceMasks": True,
                "semanticXML": _reference(semantic),
                "assemblyManifest": _reference(manifest),
                "assemblyReceipt": _reference(receipt),
                "cleanIndex": _reference(index),
                "voice": _reference(voice),
                "voiceReceipt": _reference(voice_receipt),
            }
        )
    )
    return result


def test_clean_replacement_keeps_titles_and_audio_clock(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = config.parent / "out.fcpxml"
    monkeypatch.setattr(sys, "argv", ["transform", str(config), str(out), "--name", "restored"])
    adapter.main()
    root = ET.parse(out).getroot()
    title = root.find(".//title")
    assert title is not None
    assert title.attrib == {"offset": "2s", "start": "3600s", "duration": "1s"}
    assert len(root.findall(".//spine/asset-clip/asset-clip")) == 1
    receipt = json.loads(out.with_suffix(".source-lineage.json").read_text())
    assert receipt["removedSourceMasks"] == 1
    assert receipt["sourceCaptionQualityAccepted"] is False
    monkeypatch.setattr(
        sys, "argv", ["transform", str(config), str(out), "--name", "restored", "--verify"]
    )
    adapter.main()
    out.write_bytes(out.read_bytes() + b" ")
    with pytest.raises(ReplayError, match="lineage changed"):
        adapter.verify(config, out)


@pytest.mark.parametrize(
    "field,value",
    [
        ("exactPresentationClockVerified", False),
        ("indexSHA256", "bad"),
        ("inputs", {}),
        ("probeEvidence", {}),
        ("frames", 89),
    ],
)
def test_reject_bad_assembly_evidence(config: Path, field: str, value: object) -> None:
    raw = json.loads(config.read_text())
    path = Path(raw["assemblyReceipt"]["path"])
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    raw["assemblyReceipt"] = _reference(path)
    config.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        adapter.transform(config, config.parent / "invalid.fcpxml", name="invalid")


@pytest.mark.parametrize(
    "mutation",
    [
        "intent",
        "voice",
        "range",
        "source",
        "overlay",
        "overlay-audio",
        "audio",
        "cut",
        "no-clean",
        "primary-audio",
    ],
)
def test_reject_substitution_or_unrecognized_structure(config: Path, mutation: str) -> None:
    raw = json.loads(config.read_text())
    if mutation == "intent":
        raw["removeSourceMasks"] = False
    elif mutation == "no-clean":
        raw.pop("assemblyReceipt")
    elif mutation == "voice":
        path = Path(raw["voiceReceipt"]["path"])
        path.write_text("{}")
        raw["voiceReceipt"] = _reference(path)
    elif mutation == "range":
        raw["outputWindow"] = [0, 1]
    else:
        path = Path(raw["semanticXML"]["path"])
        root = ET.parse(path).getroot()
        if mutation == "primary-audio":
            node = root.find(".//spine/asset-clip")
            assert node is not None
            ET.SubElement(node, "audio-channel-source", active="1")
        elif mutation == "source":
            node = root.find("./resources/asset/media-rep")
            assert node is not None
            node.set("src", (config.parent / "unrelated.mov").as_uri())
        elif mutation == "audio":
            node = root.find("./resources/asset[@id='a']")
            assert node is not None
            node.set("hasVideo", "1")
        elif mutation == "overlay":
            node = root.find(".//filter-video")
            assert node is not None
            node.set("ref", "unknown")
        elif mutation == "overlay-audio":
            node = root.find(".//audio-channel-source")
            assert node is not None
            node.set("active", "1")
        else:
            node = root.find("./resources/asset[@id='s']")
            assert node is not None
            node.set("start", "1s")
        ET.ElementTree(root).write(path)
        raw["semanticXML"] = _reference(path)
    config.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        adapter.transform(config, config.parent / "invalid.fcpxml", name="invalid")
