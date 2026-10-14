"""Synthetic CPU probe and descriptor-boundary tests; no MLX import."""

import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import audio_cpu_probe as probe
from replay.core import ReplayError
from replay.native_invocation import read


@pytest.mark.parametrize(
    ("snapshot", "expected"),
    [
        ("f5\nau\ntREG\nn/outside/cache\n", ["/outside/cache"]),
        ("f5\naw\ntREG\nnrelative\n", ["relative"]),
        ("f5\nar\ntREG\nn/outside/model\n", []),
        ("f5\nau\ntCHR\nn/dev/null\n", []),
        ("f5\nau\ntREG\nn/repo/cache/file\nf6\nn/outside/no-access\n", []),
    ],
)
def test_descriptor_records(snapshot: str, expected: list[str]) -> None:
    assert probe.external_writable_files(snapshot, Path("/repo")) == expected


def test_descriptor_link_escape(tmp_path: Path) -> None:
    (tmp_path / "escape").symlink_to("/outside")
    snapshot = f"f7\nau\ntREG\nn{tmp_path}/escape/cache\n"
    assert probe.external_writable_files(snapshot, tmp_path) == [f"{tmp_path}/escape/cache"]


def test_audit_retains_violation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            [], 0, "f5\nau\ntREG\nn/outside/cache\n", ""
        ),
    )
    with pytest.raises(ReplayError, match="outside checkout"):
        probe.audit_descriptors(tmp_path, "import")
    assert read(tmp_path / "import-descriptors.json")["externalWritableRegular"] == [
        "/outside/cache"
    ]


@pytest.fixture
def prepared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    source = work / "input.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(np.full((32, 2), 1000, dtype=np.int16).tobytes())
    return source, work / "probe"


def test_probe_uses_cpu_and_preserves_samples(
    monkeypatch: pytest.MonkeyPatch, prepared: tuple[Path, Path]
) -> None:
    source, directory = prepared
    events: list[str] = []

    def audit(directory: Path, phase: str) -> None:
        events.append(phase)

    def load(model: str, device: str) -> SimpleNamespace:
        assert model == "mlx-community/mel-roformer-mlx"
        assert device == "cpu"
        return SimpleNamespace(extract_vocals=lambda audio: audio * np.float32(0.5))

    monkeypatch.setattr(probe, "audit_descriptors", audit)
    monkeypatch.setattr(probe, "_import", lambda name: events.append(name))
    monkeypatch.setattr(probe, "_load_backend", load)
    assert (
        probe.main(
            [str(source), str(directory), "--samples", "16", "--input-sha256", probe.sha256(source)]
        )
        == 0
    )
    assert events == ["before-import", "mlx.core", "after-import", "after-load", "after-inference"]
    with wave.open(str(directory / "vocals.wav"), "rb") as stream:
        assert stream.getnframes() == 16
        assert stream.getnchannels() == 2
    assert read(directory / "input.json")["sourceSampleCount"] == 32
    assert read(directory / "input.json")["windowSamples"] == 352800
    assert read(directory / "input.json")["hopSamples"] == 176400
    assert read(directory / "result.json")["deliveryApproved"] is False
    assert (
        probe.main(
            [
                str(source),
                str(directory),
                "--samples",
                "16",
                "--input-sha256",
                probe.sha256(source),
                "--verify",
            ]
        )
        == 0
    )
    with pytest.raises(ReplayError, match="provenance changed"):
        probe.verify_probe(source, directory, samples=17, input_sha256=probe.sha256(source))
    (directory / "vocals.wav").write_bytes(b"changed")
    with pytest.raises(ReplayError, match="output changed"):
        probe.verify_probe(source, directory, samples=16, input_sha256=probe.sha256(source))


def test_probe_stops_before_model_on_import_cache(
    monkeypatch: pytest.MonkeyPatch, prepared: tuple[Path, Path]
) -> None:
    source, directory = prepared
    events: list[str] = []

    def audit(directory: Path, phase: str) -> None:
        events.append(phase)
        if phase == "after-import":
            raise ReplayError("external cache")

    monkeypatch.setattr(probe, "audit_descriptors", audit)
    monkeypatch.setattr(probe, "_import", lambda name: events.append(name))
    assert (
        probe.main(
            [str(source), str(directory), "--samples", "32", "--input-sha256", probe.sha256(source)]
        )
        == 1
    )
    assert events == ["before-import", "mlx.core", "after-import"]
    assert not (directory / "result.json").exists()


@pytest.mark.parametrize("samples", [0, 60 * 44100 + 1, 33])
def test_probe_rejects_sample_budget(prepared: tuple[Path, Path], samples: int) -> None:
    source, directory = prepared
    with pytest.raises(ReplayError):
        probe.probe(source, directory, samples=samples, input_sha256=probe.sha256(source))
    assert not directory.exists()


def test_probe_rejects_input_change(prepared: tuple[Path, Path]) -> None:
    source, directory = prepared
    with pytest.raises(ReplayError, match="hash changed"):
        probe.probe(source, directory, samples=16, input_sha256="0" * 64)


@pytest.mark.parametrize("change_input", [False, True])
def test_probe_rejects_bad_result_or_changed_input(
    monkeypatch: pytest.MonkeyPatch, prepared: tuple[Path, Path], change_input: bool
) -> None:
    source, directory = prepared

    def extract(audio: NDArray[np.float32]) -> NDArray[np.float32]:
        if change_input:
            source.write_bytes(b"changed")
            return audio
        return audio[:, :-1]

    monkeypatch.setattr(probe, "audit_descriptors", lambda *args: None)
    monkeypatch.setattr(probe, "_import", lambda name: None)
    monkeypatch.setattr(
        probe, "_load_backend", lambda *args: SimpleNamespace(extract_vocals=extract)
    )
    with pytest.raises(ReplayError):
        probe.probe(source, directory, samples=16, input_sha256=probe.sha256(source))
    assert not (directory / "vocals.wav").exists()
