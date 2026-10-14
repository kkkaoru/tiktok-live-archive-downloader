"""Known sample deltas, not real-media quality fixtures."""

import wave
from pathlib import Path

import numpy as np
import pytest

from replay import audio_pcm_compare as comparison
from replay.audio_cpu_probe import sha256
from replay.core import ReplayError
from replay.native_invocation import read


def write_pcm(path: Path, samples: list[list[int]]) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


@pytest.fixture
def paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    monkeypatch.chdir(tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    reference, candidate = work / "reference.wav", work / "candidate.wav"
    write_pcm(reference, [[0, 0], [100, -100]])
    write_pcm(candidate, [[3, 4], [100, -100]])
    return reference, candidate, work / "comparison.json"


def test_exact_error_metrics(paths: tuple[Path, Path, Path]) -> None:
    reference, candidate, output = paths
    assert (
        comparison.main(
            [
                str(reference),
                str(candidate),
                str(output),
                "--reference-sha256",
                sha256(reference),
                "--candidate-sha256",
                sha256(candidate),
            ]
        )
        == 0
    )
    result = read(output)
    assert result["maximumAbsoluteErrorPCM16"] == 4.0
    assert result["rmsErrorPCM16"] == 2.5
    assert result["maximumAbsoluteErrorFullScale"] == 0.0001220703125
    assert result["differentChannelSamples"] == 2
    assert result["samplesPerChannel"] == 2
    assert result["alignmentSearched"] is False
    assert result["qualityTolerance"] is None
    assert result["deliveryApproved"] is False


def test_identical_pcm_does_not_approve_quality(paths: tuple[Path, Path, Path]) -> None:
    reference, candidate, output = paths
    candidate.write_bytes(reference.read_bytes())
    comparison.compare(
        reference,
        candidate,
        output,
        reference_sha256=sha256(reference),
        candidate_sha256=sha256(candidate),
    )
    result = read(output)
    assert result["rmsErrorPCM16"] == 0.0
    assert result["musicRemovalApproved"] is False
    assert result["speechPreservationApproved"] is False


def test_changed_hash_is_failure(paths: tuple[Path, Path, Path]) -> None:
    reference, candidate, output = paths
    assert (
        comparison.main(
            [
                str(reference),
                str(candidate),
                str(output),
                "--reference-sha256",
                "0" * 64,
                "--candidate-sha256",
                sha256(candidate),
            ]
        )
        == 1
    )
    assert not output.exists()


def test_clock_mismatch_is_not_repaired(paths: tuple[Path, Path, Path]) -> None:
    reference, candidate, output = paths
    write_pcm(candidate, [[0, 0]])
    with pytest.raises(ReplayError, match="sample counts"):
        comparison.compare(
            reference,
            candidate,
            output,
            reference_sha256=sha256(reference),
            candidate_sha256=sha256(candidate),
        )
    assert not output.exists()


def test_change_during_measurement_rejected(
    monkeypatch: pytest.MonkeyPatch, paths: tuple[Path, Path, Path]
) -> None:
    reference, candidate, output = paths
    values = iter(["reference", "candidate", "changed"])
    monkeypatch.setattr(comparison, "sha256", lambda path: next(values))
    with pytest.raises(ReplayError, match="during measurement"):
        comparison.compare(
            reference, candidate, output, reference_sha256="reference", candidate_sha256="candidate"
        )
