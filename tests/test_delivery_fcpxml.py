import json
import wave
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest

from replay import delivery_fcpxml as fcpxml
from replay.core import ReplayError
from replay.delivery_render import load_config


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def configured(tmp_path: Path) -> Path:
    captions, work, storage = tmp_path / "captions", tmp_path / "work", tmp_path / "storage"
    voice_dir = work / "audio-selected"
    voice_dir.mkdir(parents=True)
    with wave.open(str(voice_dir / "selected.wav"), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 56000)
    write(
        captions / "segments.json",
        {"segments": [{"segment": 0, "offsetFrame": 0, "frames": 120, "jobs": [0]}]},
    )
    plans = work / "motion" / "segment-0000" / "plans"
    write(
        plans / "manifest.json",
        {"parts": [{"path": str(plans / "part-0000.json"), "offsetFrame": 63, "frames": 30}]},
    )
    alpha = storage / "alpha" / "segment-0000-part-0000.mov"
    alpha.parent.mkdir(parents=True)
    alpha.write_bytes(b"alpha")
    cuts = write(
        tmp_path / "cuts.json",
        {
            "cuts": [
                {"source_start_frame": 30, "source_end_frame": 90, "output_start_frame": 0},
                {"source_start_frame": 110, "source_end_frame": 170, "output_start_frame": 60},
            ]
        },
    )
    chunks = write(
        tmp_path / "chunks.json",
        {
            "ranges": [
                {
                    "media": str(tmp_path / "chunk a.mp4"),
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": 900000,
                    "timeBase": "1/90000",
                }
            ]
        },
    )
    return write(
        tmp_path / "render-config.json",
        {
            "version": 1,
            "recording": "replay-1",
            "captions": str(captions),
            "work": str(work),
            "storage": str(storage),
            "rate": 30,
            "width": 1080,
            "height": 1920,
            "cuts": str(cuts),
            "sourceChunks": str(chunks),
            "motion": {"template": str(tmp_path / "t.moti"), "expectedText": "Title", "finish": []},
            "alphaPreset": str(tmp_path / "a.setting"),
            "encodePreset": str(tmp_path / "e.setting"),
            "native": str(tmp_path / "native"),
            "encoding": {},
            "audio": {"source": str(tmp_path / "v.wav"), "clock": str(tmp_path / "c.json")},
            "source": str(tmp_path / "2024-01-01_09-00-00_JST_replay-1.mp4"),
            "deliveryRoot": str(tmp_path / "deliveries" / "replay-1"),
            "finalPreset": str(tmp_path / "f.setting"),
            "calibration": str(tmp_path / "calibration.json"),
            "viewerRoot": str(tmp_path / "edits" / "replay-1"),
        },
    )


def test_build_connects_captions_and_voice_to_cut_spine(tmp_path: Path) -> None:
    text = fcpxml.build(load_config(configured(tmp_path)), title="Edited")
    root = ElementTree.fromstring(text.split("\n", 2)[2])
    spine = root.find("./library/event/project/sequence/spine")
    assert spine is not None
    clips = list(spine)
    assert [clip.get("start") for clip in clips] == ["1s", "11/3s"]
    assert [clip.get("offset") for clip in clips] == ["0s", "2s"]
    connected = clips[1].findall("asset-clip")
    assert [(item.get("lane"), item.get("offset")) for item in connected] == [("1", "113/30s")]
    voice = clips[0].findall("asset-clip")
    assert [(item.get("ref"), item.get("lane")) for item in voice] == [("voice", "-1")]
    assert voice[0].get("offset") == "1s"
    assert voice[0].get("duration") == "7/2s"
    voice_asset = root.find("./resources/asset[@id='voice']")
    assert voice_asset is not None
    assert voice_asset.get("audioRate") == "16000"
    assert voice_asset.get("audioChannels") == "1"
    assert voice_asset.get("duration") == "7/2s"
    asset = root.find("./resources/asset[@id='r2']/media-rep")
    assert asset is not None
    assert asset.get("src") == "file://" + str(tmp_path / "chunk%20a.mp4")


def test_cli_writes_once(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = configured(tmp_path)
    output = tmp_path / "edited.fcpxml"
    assert fcpxml.main([str(config), str(output), "--title", "Edited"]) == 0
    assert fcpxml.main([str(config), str(output), "--title", "Edited"]) == 1
    assert "DELIVERY_FCPXML_ERROR" in capsys.readouterr().err


def test_empty_voice_is_not_padded(tmp_path: Path) -> None:
    config = configured(tmp_path)
    with wave.open(str(tmp_path / "work" / "audio-selected" / "selected.wav"), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    with pytest.raises(ReplayError, match="Invalid selected voice format"):
        fcpxml.build(load_config(config), title="Edited")
