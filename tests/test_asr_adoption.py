import json
import wave
from pathlib import Path

import pytest

from replay import asr_adoption
from replay.core import ReplayError
from replay.local_asr import fingerprint
from replay.stages import _fingerprint
from replay.workflow import _stage


@pytest.fixture
def retained(tmp_path: Path) -> list[str]:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(bytes(32000))
    model = tmp_path / "model"
    model.mkdir()
    config, weights = model / "config.json", model / "weights.npz"
    config.write_text("{}", encoding="utf-8")
    weights.write_bytes(b"synthetic")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "inputs.json").write_text(
        json.dumps(
            {
                "version": 1,
                "source": str(source),
                "model": str(model),
                "source_sha256": fingerprint(source),
                "config_sha256": fingerprint(config),
                "weights_sha256": fingerprint(weights),
                "sample_rate": 16000,
                "samples": 16000,
                "clock": "continuous-input-pcm",
                "human_reviewed": False,
            }
        ),
        encoding="utf-8",
    )
    (evidence / "raw-transcript.json").write_text(
        json.dumps(
            {
                "language": "ja",
                "text": "test",
                "segments": [
                    {
                        "text": "test",
                        "start": 0.1,
                        "end": 0.8,
                        "words": [{"word": "test", "start": 0.1, "end": 0.8, "probability": 0.9}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (evidence / "validation.json").write_text(
        json.dumps(
            {
                "word_groups": 1,
                "unresolved_points": 0,
                "human_reviewed": False,
                "source_clock_projected": False,
            }
        ),
        encoding="utf-8",
    )
    code = [Path.cwd() / "src/replay/local_asr.py", Path.cwd() / "src/replay/caption_io.py"]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": 1,
                "stages": [
                    {
                        "name": "raw",
                        "command": ["/usr/bin/false"],
                        "inputs": [str(path) for path in [source, config, weights, *code]],
                        "outputs": [str(evidence)],
                        "timeout_seconds": 10,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "raw.json").write_text(
        json.dumps(
            {
                "status": "running",
                "fingerprint": _fingerprint(_stage(json.loads(plan.read_bytes())["stages"][0])),
            }
        ),
        encoding="utf-8",
    )
    pins = tmp_path / "pins.sha256"
    paths = [
        source,
        evidence / "inputs.json",
        evidence / "raw-transcript.json",
        evidence / "validation.json",
        *code,
    ]
    pins.write_text("".join(f"{fingerprint(path)}  {path}\n" for path in paths), encoding="utf-8")
    return [str(plan), str(state), "raw", str(evidence), str(pins)]


def test_readonly_adoption(retained: list[str]) -> None:
    state = Path(retained[1]) / "raw.json"
    before = state.read_bytes()
    assert asr_adoption.main(retained) == 0
    assert state.read_bytes() == before
    assert json.loads(before)["status"] == "running"


@pytest.mark.parametrize("changed", ["raw", "model", "receipt", "pins", "clock", "validation"])
def test_adoption_refuses_changed_evidence(retained: list[str], changed: str) -> None:
    root = Path(retained[0]).parent
    paths = {
        "raw": root / "evidence/raw-transcript.json",
        "model": root / "model/weights.npz",
        "receipt": root / "state/raw.json",
        "pins": root / "pins.sha256",
        "clock": root / "evidence/inputs.json",
        "validation": root / "evidence/validation.json",
    }
    paths[changed].write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError):
        asr_adoption.main(retained)
