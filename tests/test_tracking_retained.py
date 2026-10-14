import json
import sys
from pathlib import Path

import pytest
from test_tracking_observations import job as job

from replay import tracking_observations as storage
from replay import tracking_retained as retained
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest


@pytest.fixture
def catalog(job: tuple[Path, Path]) -> tuple[Path, Path, Path]:
    config, output = job
    storage.observe(config, output)
    archive = config.parent / "historical-stages.bytes"
    archive.write_bytes(b"historical producer code, evidence only")
    provenance = output / "provenance.json"
    original = record(json.loads(provenance.read_text(encoding="utf-8")))
    record(original["inputs"])[str(retained.ROOT / "src/replay/stages.py")] = file_digest(archive)
    provenance.write_text(json.dumps(original), encoding="utf-8")
    descriptor = config.parent / "retained.json"
    descriptor.write_text(
        json.dumps(
            {
                "version": 1,
                "protocol": "tracking-observations-v1",
                "archives": [
                    {
                        "sourcePath": str(retained.ROOT / "src/replay/stages.py"),
                        "path": str(archive),
                        "sha256": file_digest(archive),
                    }
                ],
                "windows": [
                    {
                        "config": {"path": str(config), "sha256": file_digest(config)},
                        "provenance": {"path": str(provenance), "sha256": file_digest(provenance)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return descriptor, config, output


def test_explicit_historical_reuse_keeps_original_bytes_and_current_reader_separate(
    catalog: tuple[Path, Path, Path],
) -> None:
    descriptor, config, output = catalog
    original = (output / "provenance.json").read_bytes()
    raw = (output / "response.json").read_bytes()
    with pytest.raises(ValueError, match="provenance changed"):
        storage.verify(config, output)
    verified = retained.verify_catalog(
        descriptor, sha256=file_digest(descriptor), references=[(config, output)]
    )
    window = verified.window(config, output)
    assert window.frames == 1
    assert window.job.clock.pts == (1014,)
    assert (output / "provenance.json").read_bytes() == original
    assert (output / "response.json").read_bytes() == raw
    proof = verified.receipt()
    assert proof["mediaBytesVerified"] is True
    assert proof["nativeBytesVerified"] is True
    assert proof["historicalExecutionReissued"] is False
    assert proof["captionPlacementApproved"] is False
    assert proof["deliveryComplete"] is False
    assert (
        window.producer_code[str(retained.ROOT / "src/replay/stages.py")]
        != verified.inputs[str(retained.ROOT / "src/replay/stages.py")]
    )
    receipt = descriptor.parent / "verified.json"
    retained.write_receipt(receipt, verified)
    assert json.loads(receipt.read_text(encoding="utf-8"))["protocol"] == "tracking-observations-v1"
    with pytest.raises(FileExistsError):
        retained.write_receipt(receipt, verified)
    with pytest.raises(ValueError, match="bind this face window"):
        verified.window(config, descriptor.parent)


@pytest.mark.parametrize("references", [[], [(Path("/other-config"), Path("/other-output"))]])
def test_selected_face_index_must_match_catalog(
    catalog: tuple[Path, Path, Path], references: list[tuple[Path, Path]]
) -> None:
    descriptor, _, _ = catalog
    with pytest.raises(ValueError, match="selected face index"):
        retained.verify_catalog(descriptor, sha256=file_digest(descriptor), references=references)


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "type",
        "duplicate",
        "extra-role",
        "missing-pin",
        "bad-hash",
        "bad-path",
        "archive-sha",
        "wrong-provenance-name",
    ],
)
def test_catalog_and_archive_bindings_are_mandatory(
    catalog: tuple[Path, Path, Path], change: str
) -> None:
    descriptor, config, output = catalog
    document = record(json.loads(descriptor.read_text(encoding="utf-8")))
    rows = document["windows"]
    assert isinstance(rows, list)
    window = record(rows[0])
    changes: dict[str, object] = {
        "empty": [],
        "type": None,
        "duplicate": [window, window],
        "extra-role": [{**window, "extra": True}],
        "missing-pin": [{**window, "config": {"path": str(config)}}],
        "bad-hash": [{**window, "config": {"path": str(config), "sha256": "a" * 64}}],
        "bad-path": [{**window, "config": {"path": "relative", "sha256": file_digest(config)}}],
    }
    if change in changes:
        document["windows"] = changes[change]
    elif change == "archive-sha":
        archive = config.parent / "wrong-code.bytes"
        archive.write_bytes(b"another historical producer")
        document["archives"] = [
            {
                "sourcePath": str(retained.ROOT / "src/replay/stages.py"),
                "path": str(archive),
                "sha256": file_digest(archive),
            }
        ]
    else:
        replacement = output / "renamed.json"
        replacement.write_bytes((output / "provenance.json").read_bytes())
        window["provenance"] = {"path": str(replacement), "sha256": file_digest(replacement)}
    descriptor.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError):
        retained.verify_catalog(descriptor, sha256=file_digest(descriptor))


@pytest.mark.parametrize(
    "change",
    [
        "extra-input",
        "missing-input",
        "source-pin",
        "native-pin",
        "frame-count",
        "exit-code",
        "approval-number",
        "extra-field",
        "request-plan",
        "response-clock",
    ],
)
def test_historical_code_never_substitutes_data_or_domain_validation(
    catalog: tuple[Path, Path, Path], change: str
) -> None:
    descriptor, config, output = catalog
    provenance = output / "provenance.json"
    original = record(json.loads(provenance.read_text(encoding="utf-8")))
    inputs = record(original["inputs"])
    job = storage.load_job(config)
    if change == "extra-input":
        inputs[str(config.parent / "other.bin")] = "a" * 64
    elif change == "missing-input":
        inputs.pop(str(retained.ROOT / "src/replay/core.py"))
    elif change == "source-pin":
        inputs[str(job.source)] = "a" * 64
    elif change == "native-pin":
        inputs[str(job.native)] = "a" * 64
    elif change == "request-plan":
        request = output / "request.json"
        request.write_text("{}", encoding="utf-8")
        original["requestSHA256"] = file_digest(request)
    elif change == "response-clock":
        response = output / "response.json"
        response.write_text(
            response.read_text(encoding="utf-8").replace("1014", "1015"), encoding="utf-8"
        )
        original["responseSHA256"] = file_digest(response)
    else:
        changes: dict[str, dict[str, object]] = {
            "frame-count": {"frames": 2},
            "exit-code": {"nativeExitCode": 1},
            "approval-number": {"captionPlacementApproved": 0},
            "extra-field": {"extra": True},
        }
        original.update(changes[change])
    provenance.write_text(json.dumps(original), encoding="utf-8")
    document = record(json.loads(descriptor.read_text(encoding="utf-8")))
    rows = document["windows"]
    assert isinstance(rows, list)
    record(rows[0])["provenance"] = {"path": str(provenance), "sha256": file_digest(provenance)}
    descriptor.write_text(json.dumps(document), encoding="utf-8")
    expected_exception = ReplayError if change == "response-clock" else ValueError
    with pytest.raises(expected_exception):
        retained.verify_catalog(descriptor, sha256=file_digest(descriptor))


@pytest.mark.parametrize("artifact", ["source", "native", "config", "response", "provenance"])
def test_changed_protected_bytes_do_not_create_a_receipt(
    catalog: tuple[Path, Path, Path], artifact: str
) -> None:
    descriptor, config, output = catalog
    job = storage.load_job(config)
    path = {
        "source": job.source,
        "native": job.native,
        "config": config,
        "response": output / "response.json",
        "provenance": output / "provenance.json",
    }[artifact]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        retained.verify_catalog(descriptor, sha256=file_digest(descriptor))
    assert not (descriptor.parent / "verified.json").exists()


def test_native_must_remain_executable(catalog: tuple[Path, Path, Path]) -> None:
    descriptor, config, _ = catalog
    storage.load_job(config).native.chmod(0o600)
    with pytest.raises(ValueError, match="native executable"):
        retained.verify_catalog(descriptor, sha256=file_digest(descriptor))


def test_diagnostics_remain_empty_regular_evidence(catalog: tuple[Path, Path, Path]) -> None:
    descriptor, _, output = catalog
    (output / "stderr.log").write_text("unexpected diagnostics", encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostics"):
        retained.verify_catalog(descriptor, sha256=file_digest(descriptor))


def test_duplicate_json_keys_cannot_change_native_request() -> None:
    with pytest.raises(ValueError, match="keys must be unique"):
        retained._json(b'{"nativeExitCode":1,"nativeExitCode":0}')


def test_local_path_cannot_follow_an_escaping_link(tmp_path: Path) -> None:
    link = tmp_path / "linked"
    link.symlink_to(retained.ROOT / "src/replay")
    with pytest.raises(ValueError, match="canonical repository"):
        retained._local(link / "stages.py")


def test_conflicting_role_fingerprints_are_rejected() -> None:
    with pytest.raises(ValueError, match="Conflicting"):
        retained._bind({"/path": "a" * 64}, Path("/path"), "b" * 64)


def test_reader_cli_publishes_only_a_new_verified_receipt(
    catalog: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor, _, _ = catalog
    receipt = descriptor.parent / "verified-cli.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["retained", str(descriptor), str(receipt), "--sha256", file_digest(descriptor)],
    )
    retained.main()
    assert json.loads(receipt.read_text(encoding="utf-8"))["mediaBytesVerified"] is True


def test_missing_archive_for_changed_code_remains_a_strict_failure(
    catalog: tuple[Path, Path, Path],
) -> None:
    descriptor, config, _ = catalog
    archive = config.parent / "historical-core.bytes"
    archive.write_bytes((retained.ROOT / "src/replay/core.py").read_bytes())
    document = record(json.loads(descriptor.read_text(encoding="utf-8")))
    document["archives"] = [
        {
            "sourcePath": str(retained.ROOT / "src/replay/core.py"),
            "path": str(archive),
            "sha256": file_digest(archive),
        }
    ]
    descriptor.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit archive evidence"):
        retained.verify_catalog(descriptor, sha256=file_digest(descriptor))


def test_verification_receipt_requires_an_existing_work_parent(
    catalog: tuple[Path, Path, Path],
) -> None:
    descriptor, _, _ = catalog
    verified = retained.verify_catalog(descriptor, sha256=file_digest(descriptor))
    with pytest.raises(ValueError, match="repository work parents"):
        retained.write_receipt(descriptor.parent / "missing/receipt.json", verified)
