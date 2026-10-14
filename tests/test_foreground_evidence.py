import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from replay import foreground_collection as collection
from replay.foreground_evidence import read_window
from replay.recordings import record
from replay.stages import file_digest


@pytest.fixture
def evidence(tmp_path: Path) -> tuple[Path, Path, str, str]:
    source = tmp_path / "synthetic.bin"
    source.write_bytes(b"synthetic source")
    native = tmp_path / "native"
    native.write_text(
        f"#!{sys.executable}\nimport json, sys\nfrom pathlib import Path\n"
        "r = json.loads(Path(sys.argv[3]).read_text())\n"
        "print(json.dumps([{'requested': s, 'actual': s, 'width': 16, 'height': 16, "
        "'observationPresent': False, 'instanceLabels': None, 'unionMask': None, "
        "'maskThreshold8Bit': 1, 'captionPlacementApproved': False} for s in r['samples']]))\n",
        encoding="utf-8",
    )
    native.chmod(0o700)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "source": str(source),
                "sourceSHA256": file_digest(source),
                "nativeExecutable": str(native),
                "nativeSHA256": file_digest(native),
                "clock": {
                    "pts": [0, 1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008],
                    "timeBase": "1/30000",
                    "lastDuration": 1001,
                },
                "width": 16,
                "height": 16,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "observations"
    collection.observe(config, output)
    return config, output, file_digest(config), file_digest(output / "provenance.json")


def test_reads_pinned_typed_frames_without_writes(evidence: tuple[Path, Path, str, str]) -> None:
    config, output, config_hash, receipt_hash = evidence
    before = (config.stat().st_mtime_ns, (output / "provenance.json").stat().st_mtime_ns)
    window = read_window(config, output, config_sha256=config_hash, provenance_sha256=receipt_hash)
    assert len(window.frames) == 9
    assert window.frames[0].time == Fraction(0)
    assert window.frames[8].time == Fraction(1001, 3750)
    assert window.frames[0].labels is None
    assert window.frames[0].mask is None
    assert window.job.width == 16
    assert window.job.height == 16
    assert window.provenance_sha256 == file_digest(output / "provenance.json")
    assert before == (config.stat().st_mtime_ns, (output / "provenance.json").stat().st_mtime_ns)


@pytest.mark.parametrize("pin", ["config", "receipt", "malformed"])
def test_refuses_unpinned_bytes(evidence: tuple[Path, Path, str, str], pin: str) -> None:
    config, output, config_hash, receipt_hash = evidence
    if pin == "config":
        config_hash = "0" * 64
    elif pin == "receipt":
        receipt_hash = "0" * 64
    else:
        receipt_hash = "not-a-digest"
    with pytest.raises(ValueError):
        read_window(config, output, config_sha256=config_hash, provenance_sha256=receipt_hash)


@pytest.mark.parametrize("path", ["batch-0000/response.json", "provenance.json"])
def test_mutation_after_verification_is_not_returned(
    evidence: tuple[Path, Path, str, str], monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    config, output, config_hash, receipt_hash = evidence
    verify = collection.verify

    def mutate(config: Path, output: Path) -> None:
        verify(config, output)
        target = output / path
        target.write_bytes(target.read_bytes() + b" ")

    monkeypatch.setattr(collection, "verify", mutate)
    with pytest.raises(ValueError, match="Pinned foreground evidence changed"):
        read_window(config, output, config_sha256=config_hash, provenance_sha256=receipt_hash)


def test_source_change_during_read_fails(
    evidence: tuple[Path, Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, config_hash, receipt_hash = evidence
    verify = collection.verify

    def mutate(config: Path, output: Path) -> None:
        verify(config, output)
        collection.load_job(config).source.write_bytes(b"changed")

    monkeypatch.setattr(collection, "verify", mutate)
    with pytest.raises(ValueError, match="changed"):
        read_window(config, output, config_sha256=config_hash, provenance_sha256=receipt_hash)


@pytest.mark.parametrize("batches", [None, []])
def test_pinned_snapshot_cannot_be_replaced_during_verification(
    evidence: tuple[Path, Path, str, str], monkeypatch: pytest.MonkeyPatch, batches: object
) -> None:
    config, output, config_hash, receipt_hash = evidence
    path = output / "provenance.json"
    correct = path.read_bytes()
    raw = record(json.loads(correct))
    path.write_text(json.dumps(raw | {"batches": batches}), encoding="utf-8")
    altered_hash = file_digest(path)
    verify = collection.verify

    def repair(config: Path, output: Path) -> None:
        (output / "provenance.json").write_bytes(correct)
        verify(config, output)

    monkeypatch.setattr(collection, "verify", repair)
    with pytest.raises(ValueError, match="batch count"):
        read_window(config, output, config_sha256=config_hash, provenance_sha256=altered_hash)
    assert receipt_hash == file_digest(path)


def test_changed_code_bindings_after_verify_fail(
    evidence: tuple[Path, Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, output, config_hash, receipt_hash = evidence
    bindings = collection._bindings
    calls = 0

    def changed(job: collection.ForegroundJob) -> dict[str, str]:
        nonlocal calls
        calls += 1
        result = bindings(job)
        if calls > 2:
            result["synthetic-code-change"] = "0" * 64
        return result

    monkeypatch.setattr(collection, "_bindings", changed)
    with pytest.raises(ValueError, match="during evidence reading"):
        read_window(config, output, config_sha256=config_hash, provenance_sha256=receipt_hash)


def test_missing_completed_response_is_never_recollected(
    evidence: tuple[Path, Path, str, str],
) -> None:
    config, output, config_hash, receipt_hash = evidence
    path = output / "batch-0001/response.json"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        read_window(config, output, config_sha256=config_hash, provenance_sha256=receipt_hash)
    assert not path.exists()
