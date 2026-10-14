import json
import sys
from pathlib import Path

import pytest

from replay import reflow_producer_evidence as producer
from replay.core import ReplayError
from replay.stages import file_digest


@pytest.fixture
def files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path]:
    monkeypatch.setattr(producer, "__file__", str(tmp_path / "reflow_producer_evidence.py"))
    source = tmp_path / "caption_native_paging.py"
    source.write_bytes(b"original producer")
    saved = tmp_path / "producer.bytes"
    saved.write_bytes(b"original producer")
    artifact = tmp_path / "layout.json"
    artifact.write_text(json.dumps({"inputs": {str(source): file_digest(source)}}))
    return artifact, source, saved, tmp_path / "evidence.json"


def test_saved_bytes_survive_code_change_and_cli(
    files: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, source, saved, output = files
    monkeypatch.setattr(sys, "argv", ["evidence", *map(str, files)])
    producer.main()
    source.write_bytes(b"new producer never executed here")
    pins = producer.read(output, artifact=artifact, inputs={str(source): file_digest(saved)})
    assert len(pins) == 1
    current = producer.evidence_inputs(
        output, artifact=artifact, inputs={str(source): file_digest(saved)}
    )
    assert len(current) == 2
    with pytest.raises(FileExistsError):
        producer.prepare(artifact=artifact, source=source, evidence=saved, output=output)


@pytest.mark.parametrize("target", ["artifact", "saved", "input"])
def test_tampering_rejected(files: tuple[Path, Path, Path, Path], target: str) -> None:
    artifact, source, saved, output = files
    producer.prepare(artifact=artifact, source=source, evidence=saved, output=output)
    inputs: dict[str, object] = {str(source): file_digest(saved)}
    if target == "artifact":
        artifact.write_bytes(b"changed artifact")
    elif target == "saved":
        saved.write_bytes(b"changed evidence")
    else:
        inputs[str(source)] = "0" * 64
    with pytest.raises(ReplayError, match="artifact|bytes"):
        producer.read(output, artifact=artifact, inputs=inputs)


@pytest.mark.parametrize("name", ["movie.mp4", "tokens.json", "measurement.json", "other.py"])
def test_nonproducer_inputs_cannot_be_exempted(
    files: tuple[Path, Path, Path, Path], name: str
) -> None:
    artifact, source, saved, output = files
    with pytest.raises(ReplayError, match="restricted"):
        producer.prepare(
            artifact=artifact, source=source.with_name(name), evidence=saved, output=output
        )


def test_nonstring_source_rejected() -> None:
    with pytest.raises(ReplayError, match="path"):
        producer._producer(None)


def test_wrong_saved_producer_rejected(files: tuple[Path, Path, Path, Path]) -> None:
    artifact, source, saved, output = files
    saved.write_bytes(b"different producer")
    with pytest.raises(ReplayError, match="match artifact"):
        producer.prepare(artifact=artifact, source=source, evidence=saved, output=output)
