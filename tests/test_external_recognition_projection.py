"""External projection uses caller-declared lineage with measured durations."""

import hashlib
import json
import wave
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.external_recognition_projection import run

RATE = 16000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(frames: int, per: int) -> dict[str, object]:
    return {
        "streams": [{"sample_rate": "16000", "time_base": "1/16000"}],
        "frames": [
            {"pts": index * per, "nb_samples": per} for index in range(frames)
        ],
    }


def _write(tmp_path: Path, frames: int = 10, per: int = 1600) -> dict[str, str]:
    pcm = tmp_path / "mono.wav"
    with wave.open(str(pcm), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(RATE)
        stream.writeframes(b"\x00\x00" * (frames * per))
    clock = tmp_path / "audio-clock.json"
    clock.write_text(json.dumps(_probe(frames, per)), encoding="utf-8")
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "input-pcm-not-projected",
                "inputs": {str(pcm): _sha(pcm)},
                "samples": frames * per,
                "sampleRate": RATE,
            }
        ),
        encoding="utf-8",
    )
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "input-pcm-not-projected",
                "labels": [
                    {"text": "あ", "start": 0.0, "end": 0.5, "provenance": "w:0"},
                    {"text": "い", "start": 0.5, "end": 1.0, "provenance": "w:1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "inputClock": "continuous-decoded-pcm",
                "artifacts": {
                    name: {"path": str(path), "sha256": _sha(path)}
                    for name, path in (
                        ("labels", labels),
                        ("pcm", pcm),
                        ("audioClock", clock),
                        ("asrInputs", manifest),
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    return {"config": str(config), "labels": str(labels)}


def test_projects_with_measured_identity(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    output = tmp_path / "source-labels.json"
    assert run(Path(paths["config"]), output) == 2
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["clock"] == "source-presentation"
    assert result["lineageIsCallerDeclared"] is True
    assert result["humanReviewed"] is False
    assert [label["text"] for label in result["labels"]] == ["あ", "い"]
    assert result["labels"][0]["start"] == pytest.approx(0.0)
    assert result["labels"][1]["end"] == pytest.approx(1.0)
    assert result["unresolvedPoints"] == 0


def test_rejects_digest_mismatch_and_bad_clock(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    config = json.loads(Path(paths["config"]).read_text(encoding="utf-8"))
    config["artifacts"]["labels"]["sha256"] = "0" * 64
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ReplayError, match="digest"):
        run(bad, tmp_path / "out.json")
    labels = json.loads(Path(paths["labels"]).read_text(encoding="utf-8"))
    labels["clock"] = "source-presentation"
    Path(paths["labels"]).write_text(json.dumps(labels), encoding="utf-8")
    config = json.loads(Path(paths["config"]).read_text(encoding="utf-8"))
    config["artifacts"]["labels"]["sha256"] = _sha(Path(paths["labels"]))
    Path(paths["config"]).write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ReplayError, match="bind the declared PCM"):
        run(Path(paths["config"]), tmp_path / "out2.json")
