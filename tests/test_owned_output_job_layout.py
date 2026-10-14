"""Synthetic owned output roots never contain socket-producing job runtimes."""

import hashlib
import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch

from replay import owned_output_job_layout as layout


@pytest.fixture
def pins(tmp_path: Path, monkeypatch: MonkeyPatch) -> tuple[layout.MetadataPin, layout.MetadataPin]:
    monkeypatch.setattr(layout, "REPOSITORY", tmp_path)
    config = tmp_path / "config.json"
    request = tmp_path / "request.json"
    config.write_text(json.dumps({"outputDirectory": str(tmp_path / "owned")}), encoding="utf-8")
    request.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "measured-job-admission",
                "workDirectory": str(tmp_path / "runtime"),
                "scratchDirectory": str(tmp_path / "runtime/tmp"),
                "cacheDirectory": str(tmp_path / "runtime/cache"),
            }
        ),
        encoding="utf-8",
    )
    return layout.MetadataPin(
        config, hashlib.sha256(config.read_bytes()).hexdigest()
    ), layout.MetadataPin(request, hashlib.sha256(request.read_bytes()).hexdigest())


def test_distinct_runtime_and_regular_workflow_allowed(
    pins: tuple[layout.MetadataPin, layout.MetadataPin], tmp_path: Path
) -> None:
    (tmp_path / "owned/state").mkdir(parents=True)
    (tmp_path / "owned/workflow.json").write_text("{}", encoding="utf-8")
    result = layout.validate_layout(
        config=pins[0],
        request=pins[1],
        owned_field="outputDirectory",
        orchestration_roots=(tmp_path / "runtime",),
        reject_reciprocal=True,
    )
    assert result["layoutVerified"] is True
    assert result["runtimeObserved"] is False
    assert result["productionAdmitted"] is False
    assert not (tmp_path / "runtime").exists()


@pytest.mark.parametrize(
    "root", ["owned", "owned/smoke-jobs", "owned/admission-full-v1", "owned/state/jobs"]
)
def test_nested_runtime_rejected(
    pins: tuple[layout.MetadataPin, layout.MetadataPin], tmp_path: Path, root: str
) -> None:
    with pytest.raises(ValueError, match="overlaps"):
        layout.validate_layout(
            config=pins[0],
            request=pins[1],
            owned_field="outputDirectory",
            orchestration_roots=(tmp_path / root,),
        )


def test_reciprocal_optional(
    pins: tuple[layout.MetadataPin, layout.MetadataPin], tmp_path: Path
) -> None:
    assert (
        layout.validate_layout(
            config=pins[0],
            request=pins[1],
            owned_field="outputDirectory",
            orchestration_roots=(tmp_path,),
            reject_reciprocal=False,
        )["layoutVerified"]
        is True
    )
    with pytest.raises(ValueError, match="Owned output overlaps"):
        layout.validate_layout(
            config=pins[0],
            request=pins[1],
            owned_field="outputDirectory",
            orchestration_roots=(tmp_path,),
            reject_reciprocal=True,
        )


@pytest.mark.parametrize("value", ["relative", "/not-this-checkout", "", None, 1])
def test_invalid_directory(value: object, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(layout, "REPOSITORY", tmp_path)
    with pytest.raises(ValueError):
        layout._directory(value)


def test_symlink_and_regular_file_refused(
    pins: tuple[layout.MetadataPin, layout.MetadataPin], tmp_path: Path
) -> None:
    (tmp_path / "link").symlink_to(tmp_path / "owned", target_is_directory=True)
    with pytest.raises(ValueError, match="canonical"):
        layout._directory(str(tmp_path / "link/jobs"))
    with pytest.raises(ValueError, match="not a directory"):
        layout._directory(str(pins[0].path))
    with pytest.raises(ValueError, match="canonical"):
        layout._directory(str(tmp_path / "owned/../runtime"))


@pytest.mark.parametrize("body", ["[]", '{"x":1,"x":2}', "invalid", ""])
def test_bad_metadata(pins: tuple[layout.MetadataPin, layout.MetadataPin], body: str) -> None:
    path = pins[0].path
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        layout._read(layout.MetadataPin(path, hashlib.sha256(path.read_bytes()).hexdigest()))


def test_hash_size_and_schema(
    pins: tuple[layout.MetadataPin, layout.MetadataPin], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="SHA256"):
        layout._read(layout.MetadataPin(pins[0].path, "bad"))
    with pytest.raises(ValueError, match="changed"):
        layout._read(layout.MetadataPin(pins[0].path, "0" * 64))
    monkeypatch.setattr(layout, "MAX_METADATA_BYTES", 1)
    with pytest.raises(ValueError, match="bounded"):
        layout._read(pins[0])
    monkeypatch.setattr(layout, "MAX_METADATA_BYTES", 1024 * 1024)
    with pytest.raises(ValueError, match="Unsupported admission"):
        layout.validate_layout(
            config=pins[0],
            request=pins[0],
            owned_field="outputDirectory",
            orchestration_roots=(tmp_path / "jobs",),
        )
    with pytest.raises(ValueError, match="supported owned field"):
        layout.validate_layout(
            config=pins[0], request=pins[1], owned_field="unknown", orchestration_roots=()
        )


def test_admission_nested_itself_rejected(
    pins: tuple[layout.MetadataPin, layout.MetadataPin], tmp_path: Path
) -> None:
    path = pins[1].path
    raw = json.loads(path.read_text())
    raw["scratchDirectory"] = str(tmp_path / "owned/tmp")
    path.write_text(json.dumps(raw), encoding="utf-8")
    request = layout.MetadataPin(path, hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="Orchestration root overlaps"):
        layout.validate_layout(
            config=pins[0],
            request=request,
            owned_field="outputDirectory",
            orchestration_roots=(tmp_path / "external",),
        )


def test_cli(pins: tuple[layout.MetadataPin, layout.MetadataPin], tmp_path: Path) -> None:
    argv = [
        str(pins[0].path),
        "--config-sha256",
        pins[0].sha256,
        "--request",
        str(pins[1].path),
        "--request-sha256",
        pins[1].sha256,
        "--owned-field",
        "outputDirectory",
        "--orchestration-root",
        str(tmp_path / "jobs"),
    ]
    assert layout.main(argv) == 0
    assert layout.main([*argv, "--orchestration-root", str(tmp_path / "owned")]) == 1
