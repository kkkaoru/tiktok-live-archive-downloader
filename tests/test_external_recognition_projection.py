"""External projection uses caller-declared lineage with measured durations."""

import hashlib
import json
import wave
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.external_recognition_projection import main, run

RATE = 16000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(frames: int, per: int) -> dict[str, object]:
    return {
        "streams": [{"sample_rate": "16000", "time_base": "1/16000"}],
        "frames": [{"pts": index * per, "nb_samples": per} for index in range(frames)],
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


def _rewrite(path: Path, change: dict[str, object]) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(change)
    path.write_text(json.dumps(data), encoding="utf-8")


def _rebind(config: Path, name: str, artifact: Path) -> None:
    data = json.loads(config.read_text(encoding="utf-8"))
    data["artifacts"][name]["sha256"] = _sha(artifact)
    config.write_text(json.dumps(data), encoding="utf-8")


def test_refuses_an_existing_output(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    output = tmp_path / "exists.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="new output"):
        run(Path(paths["config"]), output)


def test_rejects_an_unsupported_clock_contract(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    _rewrite(Path(paths["config"]), {"version": 2})
    with pytest.raises(ReplayError, match="Unsupported projection clock"):
        run(Path(paths["config"]), tmp_path / "out.json")


def test_rejects_incomplete_lineage(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    config = Path(paths["config"])
    data = json.loads(config.read_text(encoding="utf-8"))
    del data["artifacts"]["pcm"]
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="complete caller-declared lineage"):
        run(config, tmp_path / "out.json")


def test_rejects_an_artifact_binding_with_extra_fields(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    config = Path(paths["config"])
    data = json.loads(config.read_text(encoding="utf-8"))
    data["artifacts"]["labels"]["note"] = "extra"
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError, match="explicit artifact path"):
        run(config, tmp_path / "out.json")


def test_rejects_non_numeric_recognition_clocks(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    labels = Path(paths["labels"])
    _rewrite(labels, {"labels": [{"text": "あ", "start": "0", "end": 0.5, "provenance": "w:0"}]})
    _rebind(Path(paths["config"]), "labels", labels)
    with pytest.raises(ReplayError, match="must be numeric"):
        run(Path(paths["config"]), tmp_path / "out.json")


def test_rejects_labels_without_text_or_provenance(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    labels = Path(paths["labels"])
    _rewrite(labels, {"labels": [{"text": 5, "start": 0.0, "end": 0.5, "provenance": "w:0"}]})
    _rebind(Path(paths["config"]), "labels", labels)
    with pytest.raises(ReplayError, match="text/provenance"):
        run(Path(paths["config"]), tmp_path / "out.json")


def test_rejects_an_empty_label_list(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    labels = Path(paths["labels"])
    _rewrite(labels, {"labels": []})
    _rebind(Path(paths["config"]), "labels", labels)
    with pytest.raises(ReplayError, match="bounded unprojected"):
        run(Path(paths["config"]), tmp_path / "out.json")


def test_rejects_pcm_that_differs_from_the_recognition_manifest(tmp_path: Path) -> None:
    paths = _write(tmp_path)
    manifest = tmp_path / "inputs.json"
    _rewrite(manifest, {"samples": 1})
    _rebind(Path(paths["config"]), "asrInputs", manifest)
    with pytest.raises(ReplayError, match="duration-preserving"):
        run(Path(paths["config"]), tmp_path / "out.json")


def test_main_prints_the_projected_label_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _write(tmp_path)
    assert main([paths["config"], str(tmp_path / "out.json")]) == 0
    assert capsys.readouterr().out == "Projected external recognition labels: 2\n"
