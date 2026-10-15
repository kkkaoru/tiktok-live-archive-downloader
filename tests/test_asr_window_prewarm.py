"""Parallel prewarm yields the same window evidence the pinned serial run reuses."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import asr_window_prewarm as prewarm
from replay import isolated_cpu_asr as isolated
from replay.asr_runtime import ASRDevice
from replay.audio_windows import AudioWindow
from replay.core import ReplayError
from replay.local_asr import fingerprint
from replay.stages import execution_lock


class FakeRuntime:
    def __init__(self, device: ASRDevice) -> None:
        assert device == ASRDevice.CPU

    def predict(self, samples: NDArray[np.float32], model: Path, *, evidence: Path) -> object:
        assert samples.dtype == np.float32 and model.is_dir()
        evidence.mkdir()
        return {"segments": [], "text": ""}


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "edits/work"
    root.mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    (model / "weights.npz").write_bytes(b"weights")
    (model / "config.json").write_bytes(b"config")
    source = root / "source.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(16000 * 2 * 400))
    monkeypatch.setattr(isolated, "WhisperRuntime", FakeRuntime)
    return root, source, model


def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
    assert command[-2] == "--worker" and timeout == 300
    log.write_text("synthetic worker")
    isolated.worker(Path(command[-1]))


def serial(
    fixture: tuple[Path, Path, Path], policy: str, *, limit: int | None = None
) -> dict[str, object]:
    root, source, model = fixture
    result = isolated.run(
        source,
        root / "output",
        root / policy,
        model=model,
        weights_sha256=fingerprint(model / "weights.npz"),
        config_sha256=fingerprint(model / "config.json"),
        execute=execute,
        limit=limit,
    )
    loaded: dict[str, object] = json.loads(result.read_text())
    return loaded


def test_claims_hand_out_each_window_once_from_the_last() -> None:
    claims = prewarm._Claims(3)
    assert [claims.take(), claims.take(), claims.take(), claims.take()] == [2, 1, 0, -1]


def test_frontier_is_the_first_window_without_a_clock_receipt(tmp_path: Path) -> None:
    windows = [AudioWindow(0, 960000), AudioWindow(480000, 960000), AudioWindow(960000, 960000)]
    assert prewarm.frontier(tmp_path, windows) == 0
    (tmp_path / "000000000000.clock.json").write_text("{}")
    (tmp_path / "000000960000.clock.json").write_text("{}")
    assert prewarm.frontier(tmp_path, windows) == 1
    (tmp_path / "000000480000.clock.json").write_text("{}")
    assert prewarm.frontier(tmp_path, windows) == 3


def test_prewarm_stays_ahead_of_the_serial_run_which_reuses_it(
    fixture: tuple[Path, Path, Path],
) -> None:
    root, source, model = fixture
    serial(fixture, "one", limit=1)
    created = prewarm.prewarm(source, root / "output", workers=2, model=model, execute=execute)
    assert created == 9
    for start in (480000, 960000, 1440000):
        assert not (root / f"output/{start:012d}.clock.json").exists()
    assert (root / "output/000000480000.clock.json").exists() is False
    assert (root / "output/000005760000.clock.json").is_file()
    assert (root / "output/000005280000.cpu-runtime/runtime").is_dir()
    final = serial(fixture, "two")
    assert final["newWindows"] == 3
    assert final["recognitionComplete"] is True
    assert final["verifiedWindows"] == 13
    assert json.loads((root / "output/complete.json").read_text())["windows"] == 13


def test_exclusive_prewarm_finishes_every_window_for_a_stopped_serial_run(
    fixture: tuple[Path, Path, Path],
) -> None:
    root, source, model = fixture
    serial(fixture, "one", limit=1)
    created = prewarm.prewarm(
        source, root / "output", workers=2, model=model, execute=execute, exclusive=True
    )
    assert created == 12
    assert (root / "output/000000480000.clock.json").is_file()
    final = serial(fixture, "two")
    assert final["newWindows"] == 0
    assert final["recognitionComplete"] is True


def test_exclusive_prewarm_refuses_while_the_serial_run_holds_its_lease(
    fixture: tuple[Path, Path, Path],
) -> None:
    root, source, model = fixture
    serial(fixture, "one", limit=1)
    with execution_lock(root / "output", ".run") as acquired:
        assert acquired
        with pytest.raises(ReplayError, match="already has an owner"):
            prewarm.prewarm(
                source, root / "output", workers=1, model=model, execute=execute, exclusive=True
            )


def test_prewarm_needs_an_in_progress_serial_collection(fixture: tuple[Path, Path, Path]) -> None:
    root, source, model = fixture
    with pytest.raises(ReplayError, match="in-progress collection"):
        prewarm.prewarm(source, root / "output", workers=2, model=model, execute=execute)
    serial(fixture, "one")
    with pytest.raises(ReplayError, match="in-progress collection"):
        prewarm.prewarm(source, root / "output", workers=2, model=model, execute=execute)


def test_prewarm_rejects_a_model_that_differs_from_the_serial_run(
    fixture: tuple[Path, Path, Path],
) -> None:
    root, source, model = fixture
    serial(fixture, "one", limit=1)
    (model / "config.json").write_bytes(b"changed")
    with pytest.raises(ReplayError, match="differ from the serial ASR run"):
        prewarm.prewarm(source, root / "output", workers=2, model=model, execute=execute)


def test_prewarm_rejects_an_invalid_worker_count(fixture: tuple[Path, Path, Path]) -> None:
    root, source, model = fixture
    with pytest.raises(ReplayError, match="worker count"):
        prewarm.prewarm(source, root / "output", workers=0, model=model, execute=execute)
    with pytest.raises(ReplayError, match="worker count"):
        prewarm.prewarm(source, root / "output", workers=9, model=model, execute=execute)


def test_prewarm_rejects_an_invalid_deadline(fixture: tuple[Path, Path, Path]) -> None:
    root, source, model = fixture
    with pytest.raises(ReplayError, match="child deadline"):
        prewarm.prewarm(source, root / "output", workers=1, timeout=0, model=model, execute=execute)


def test_prewarm_rejects_stereo_pcm(fixture: tuple[Path, Path, Path]) -> None:
    root, _, model = fixture
    stereo = root / "stereo.wav"
    with wave.open(str(stereo), "wb") as stream:
        stream.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(bytes(16000 * 4))
    (root / "output").mkdir()
    (root / "output/inputs.json").write_text(
        json.dumps(
            {
                "inputs": {
                    str(stereo.resolve()): fingerprint(stereo),
                    str((model / "config.json").resolve()): fingerprint(model / "config.json"),
                    str((model / "weights.npz").resolve()): fingerprint(model / "weights.npz"),
                }
            }
        )
    )
    with pytest.raises(ReplayError, match="mono PCM16"):
        prewarm.prewarm(stereo, root / "output", workers=1, model=model, execute=execute)


def test_main_prints_the_created_window_count(
    fixture: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, source, model = fixture
    monkeypatch.setattr(prewarm, "prewarm", lambda *a, **k: 5)
    monkeypatch.setattr(
        "sys.argv",
        ["x", str(source), str(root / "output"), "--workers", "2", "--model", str(model)],
    )
    prewarm.main()
    assert capsys.readouterr().out.strip() == "ASR_PREWARM_CREATED 5"


def test_main_turns_a_refusal_into_a_clean_exit(
    fixture: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, source, model = fixture

    def refuse(*args: object, **kwargs: object) -> int:
        raise ReplayError("nope")

    monkeypatch.setattr(prewarm, "prewarm", refuse)
    monkeypatch.setattr(
        "sys.argv",
        ["x", str(source), str(root / "output"), "--workers", "2", "--model", str(model)],
    )
    with pytest.raises(SystemExit, match="ASR_PREWARM_ERROR nope"):
        prewarm.main()
