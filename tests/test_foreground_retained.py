import json
import sys
from pathlib import Path

import pytest

from replay import foreground_collection as collection
from replay import foreground_retained as retained
from replay import tracking_observations as storage
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(retained, "ROOT", tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    source, native = work / "source", work / "native"
    source.write_bytes(b"synthetic media")
    native.write_bytes(b"not executed")
    native.chmod(0o700)
    config = work / "config.json"
    storage._write(
        config,
        {
            "version": 1,
            "source": str(source),
            "sourceSHA256": file_digest(source),
            "nativeExecutable": str(native),
            "nativeSHA256": file_digest(native),
            "clock": {"pts": [0], "timeBase": "1/30", "lastDuration": 1},
            "width": 16,
            "height": 16,
        },
    )
    job = collection.load_job(config)
    output = work / "output"
    batch = output / "batch-0000"
    batch.mkdir(parents=True)
    storage._write(batch / "request.json", job.request((0,)))
    storage._write(
        batch / "response.json",
        [
            {
                "requested": {"value": 0, "timescale": 30},
                "actual": {"value": 0, "timescale": 30},
                "width": 16,
                "height": 16,
                "observationPresent": False,
                "instanceLabels": None,
                "unionMask": None,
                "maskThreshold8Bit": 1,
                "captionPlacementApproved": False,
            }
        ],
    )
    (batch / "stderr.log").write_bytes(b"")
    archive = work / "stages.bytes"
    archive.write_bytes(b"raise RuntimeError('historical code must never run')\n")
    pins = collection._bindings(job)
    pins[str(Path(collection.__file__).with_name("stages.py"))] = file_digest(archive)
    storage._write(output / "inputs.json", pins)
    storage._write(output / "provenance.json", collection._receipt(job, output, pins))
    descriptor = work / "descriptor.json"
    storage._write(
        descriptor,
        {
            "version": 1,
            "protocol": "foreground-collection-v1",
            "archives": [
                {"role": "stages.py", "path": str(archive), "sha256": file_digest(archive)}
            ],
        },
    )
    return config, output, descriptor


def run(fixture: tuple[Path, Path, Path], *, metadata_only: bool = False) -> retained.Verification:
    config, output, descriptor = fixture
    return retained.verify(
        config,
        output,
        config_sha256=file_digest(config),
        provenance_sha256=file_digest(output / "provenance.json"),
        descriptor=descriptor,
        descriptor_sha256=file_digest(descriptor),
        metadata_only=metadata_only,
    )


@pytest.mark.parametrize("metadata_only", [False, True])
def test_retained_checks_are_new_readonly_evidence(
    fixture: tuple[Path, Path, Path],
    metadata_only: bool,
) -> None:
    config, output, descriptor = fixture
    before = (output / "provenance.json").read_bytes()
    with pytest.raises(ValueError, match="provenance changed"):
        collection.verify(config, output)
    result = run(fixture, metadata_only=metadata_only)
    assert len(result.window.frames) == 1
    assert result.window.frames[0].mask is None
    assert result.media_bytes_verified is not metadata_only
    assert result.receipt()["historicalExecutionReissued"] is False
    assert result.receipt()["captionPlacementApproved"] is False
    assert result.producer_code != result.receipt()["currentReaderCode"]
    assert (output / "provenance.json").read_bytes() == before
    assert descriptor.is_file()


@pytest.mark.parametrize(
    "target",
    [
        "source",
        "native",
        "stages.bytes",
        "output/inputs.json",
        "output/batch-0000/response.json",
        "output/batch-0000/request.json",
        "output/batch-0000/stderr.log",
    ],
)
def test_tampered_bytes_fail(fixture: tuple[Path, Path, Path], target: str) -> None:
    config, _, _ = fixture
    path = config.parent / target
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        run(fixture)


@pytest.mark.parametrize(
    "change",
    [
        "protocol",
        "version",
        "empty",
        "duplicate",
        "source",
        "native",
        "config",
        "model",
        "response",
        "output",
        "outside",
        "wrongArchive",
        "fields",
        "roleType",
    ],
)
def test_descriptor_rejects_unsafe_redirects(
    fixture: tuple[Path, Path, Path],
    change: str,
) -> None:
    config, _, descriptor = fixture
    data = record(json.loads(descriptor.read_bytes()))
    archive = {
        "role": "stages.py",
        "path": str(config.parent / "stages.bytes"),
        "sha256": file_digest(config.parent / "stages.bytes"),
    }
    if change == "protocol":
        data["protocol"] = "future-protocol"
    elif change == "version":
        data["version"] = True
    elif change == "empty":
        data["archives"] = []
    elif change == "duplicate":
        data["archives"] = [archive, archive]
    else:
        if change == "outside":
            archive["path"] = str(config.parents[2] / "outside")
        elif change == "wrongArchive":
            archive["path"] = str(config.parent / "native")
            archive["sha256"] = file_digest(config.parent / "native")
        elif change == "fields":
            archive["extra"] = "bad"
        elif change == "roleType":
            data["archives"] = [{"role": None}]
        else:
            archive["role"] = change
        if change != "roleType":
            data["archives"] = [archive]
    descriptor.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        run(fixture)


@pytest.mark.parametrize(
    "change", ["roles", "sourcePin", "batch", "clock", "approval", "threshold"]
)
def test_current_protocol_revalidated(
    fixture: tuple[Path, Path, Path],
    change: str,
) -> None:
    _, output, _ = fixture
    path = output / "provenance.json"
    data = record(json.loads(path.read_bytes()))
    if change == "roles":
        record(data["inputs"])["model"] = "0" * 64
    elif change == "sourcePin":
        record(data["inputs"])[str(output.parent / "source")] = "0" * 64
    elif change == "batch":
        data["frames"] = 2
    else:
        response = output / "batch-0000/response.json"
        entries = json.loads(response.read_bytes())
        if change == "clock":
            entries[0]["actual"]["value"] = 1
        elif change == "approval":
            entries[0]["captionPlacementApproved"] = True
        else:
            entries[0]["maskThreshold8Bit"] = 2
        response.write_text(json.dumps(entries), encoding="utf-8")
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises((ValueError, ReplayError)):
        run(fixture)


def test_cli_new_receipt(fixture: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    config, output, descriptor = fixture
    receipt = config.parent / "verification.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "foreground_retained",
            str(config),
            str(output),
            str(descriptor),
            str(receipt),
            "--config-sha256",
            file_digest(config),
            "--provenance-sha256",
            file_digest(output / "provenance.json"),
            "--descriptor-sha256",
            file_digest(descriptor),
        ],
    )
    retained.main()
    assert record(json.loads(receipt.read_bytes()))["mediaBytesVerified"] is True
    with pytest.raises(FileExistsError):
        retained.main()
