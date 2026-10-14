"""Synthetic archival-code checks never execute the preserved Python bytes."""

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import producer_code_evidence as evidence

PROTOCOL = "tracking-observations-v1"
ROLES = ("stages.py", "tracking_observations.py")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Files:
    root: Path
    descriptor: Path
    archive: Path
    source: Path
    document: dict[str, object]

    def write(self, document: object) -> None:
        self.descriptor.write_text(json.dumps(document), encoding="utf-8")

    def read(self) -> evidence.CodeEvidence:
        return evidence.read_archives(
            self.descriptor,
            _sha(self.descriptor.read_bytes()),
            protocol=PROTOCOL,
            source_names=ROLES,
            root=self.root,
            binding_fields=frozenset({"windows"}),
        )


@pytest.fixture
def files(tmp_path: Path) -> Files:
    source = tmp_path / "src" / "replay" / "stages.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"current producer code is separate")
    work = tmp_path / "edits" / "work"
    work.mkdir(parents=True)
    archive = work / "stages.bytes"
    archive.write_bytes(b"raise RuntimeError('archived code must never execute')\n")
    document: dict[str, object] = {
        "version": 1,
        "protocol": PROTOCOL,
        "archives": [
            {"sourcePath": str(source), "path": str(archive), "sha256": _sha(archive.read_bytes())}
        ],
        "windows": [
            {
                "config": {"path": str(work / "config.json"), "sha256": "a" * 64},
                "provenance": {"path": str(work / "provenance.json"), "sha256": "b" * 64},
            }
        ],
    }
    fixture = Files(tmp_path, work / "manifest.json", archive, source, document)
    fixture.write(document)
    return fixture


def test_historical_code_and_artifact_context_stay_bound_without_execution(files: Files) -> None:
    original = files.descriptor.read_bytes()
    result = files.read()
    assert result.document == files.document
    assert result.archives == {
        str(files.source): evidence.CodeArchive(files.archive, _sha(files.archive.read_bytes()))
    }
    assert result.inputs == {
        str(files.descriptor): _sha(original),
        str(files.archive): _sha(files.archive.read_bytes()),
    }
    files.source.write_bytes(b"new reader uses new code; old source need not execute")
    assert files.read() == result
    assert files.descriptor.read_bytes() == original


@pytest.mark.parametrize("target", ["archive", "descriptor", "binding"])
def test_changed_code_or_artifact_manifest_is_rejected(files: Files, target: str) -> None:
    pin = _sha(files.descriptor.read_bytes())
    if target == "archive":
        files.archive.write_bytes(b"different archived code")
    elif target == "descriptor":
        files.descriptor.write_bytes(files.descriptor.read_bytes() + b" ")
    else:
        files.write(files.document | {"windows": [{"provenance": "replaced artifact"}]})
    with pytest.raises(ValueError, match="Pinned producer evidence bytes changed"):
        evidence.read_archives(
            files.descriptor,
            pin,
            protocol=PROTOCOL,
            source_names=ROLES,
            root=files.root,
            binding_fields=frozenset({"windows"}),
        )


@pytest.mark.parametrize("version", [True, False, 0, 2, "1", None])
def test_version_is_an_exact_supported_integer(files: Files, version: object) -> None:
    files.write(files.document | {"version": version})
    with pytest.raises(ValueError, match="declared protocol"):
        files.read()


@pytest.mark.parametrize("mode", ["protocol", "missingBinding", "extraField", "notObject"])
def test_protocol_and_exact_binding_fields_are_required(files: Files, mode: str) -> None:
    documents: dict[str, object] = {
        "protocol": files.document | {"protocol": "other-domain-v1"},
        "missingBinding": {key: value for key, value in files.document.items() if key != "windows"},
        "extraField": files.document | {"metadataOnly": True},
        "notObject": [],
    }
    files.write(documents[mode])
    with pytest.raises(ValueError):
        files.read()


@pytest.mark.parametrize("location", ["manifest", "archive"])
def test_duplicate_json_keys_are_rejected(files: Files, location: str) -> None:
    data = files.descriptor.read_text(encoding="utf-8")
    if location == "manifest":
        data = data.replace('"version": 1', '"version": 1, "version": 1')
    else:
        data = data.replace('"sourcePath":', '"sha256": "' + "a" * 64 + '", "sourcePath":')
    files.descriptor.write_text(data, encoding="utf-8")
    with pytest.raises(ValueError, match="keys must be unique"):
        files.read()


@pytest.mark.parametrize("rows", [None, {}, [], [None], [1]])
def test_invalid_archive_array_is_rejected(files: Files, rows: object) -> None:
    files.write(files.document | {"archives": rows})
    with pytest.raises(ValueError):
        files.read()


@pytest.mark.parametrize(
    "mode",
    [
        "duplicate",
        "extra",
        "missing",
        "sourceType",
        "source",
        "media",
        "model",
        "native",
        "clock",
        "outside",
        "relative",
        "traversal",
        "nul",
        "newline",
        "empty",
        "archiveType",
        "sourceArchive",
        "manifestArchive",
        "badSHA",
        "nullSHA",
    ],
)
def test_archive_redirects_cannot_replace_noncode_inputs(files: Files, mode: str) -> None:
    row: dict[str, object] = {
        "sourcePath": str(files.source),
        "path": str(files.archive),
        "sha256": _sha(files.archive.read_bytes()),
    }
    if mode == "duplicate":
        rows: object = [row, row]
    else:
        if mode == "extra":
            row["allowMedia"] = True
        elif mode == "missing":
            row.pop("path")
        elif mode == "sourceType":
            row["sourcePath"] = 4
        elif mode in {"source", "media", "model", "native", "clock"}:
            row["sourcePath"] = str(
                files.root
                / {
                    "source": "other.py",
                    "media": "movie.mp4",
                    "model": "weights.npz",
                    "native": "native",
                    "clock": "clock.json",
                }[mode]
            )
        elif mode == "outside":
            row["path"] = str(files.root.parent / "outside.bytes")
        elif mode == "relative":
            row["path"] = "edits/work/stages.bytes"
        elif mode == "traversal":
            row["path"] = str(files.archive.parent / ".." / "work" / "stages.bytes")
        elif mode == "nul":
            row["path"] = str(files.archive) + "\0"
        elif mode == "newline":
            row["path"] = str(files.archive) + "\n"
        elif mode == "empty":
            row["path"] = ""
        elif mode == "archiveType":
            row["path"] = True
        elif mode == "sourceArchive":
            row["path"] = str(files.source)
        elif mode == "manifestArchive":
            row["path"] = str(files.descriptor)
        elif mode == "badSHA":
            row["sha256"] = "A" * 64
        elif mode == "nullSHA":
            row["sha256"] = None
        rows = [row]
    files.write(files.document | {"archives": rows})
    with pytest.raises(ValueError):
        files.read()


@pytest.mark.parametrize(
    "mode", ["symlink", "parentLink", "empty", "oversize", "directory", "fifo"]
)
def test_only_canonical_bounded_regular_archive_bytes_are_read(files: Files, mode: str) -> None:
    original = files.archive.read_bytes()
    files.archive.unlink()
    if mode == "symlink":
        target = files.archive.with_name("target.bytes")
        target.write_bytes(original)
        files.archive.symlink_to(target)
    elif mode == "parentLink":
        target = files.archive.parent / "real"
        target.mkdir()
        (target / "stages.bytes").write_bytes(original)
        linked = files.archive.parent / "linked"
        linked.symlink_to(target, target_is_directory=True)
        row = {
            "sourcePath": str(files.source),
            "path": str(linked / "stages.bytes"),
            "sha256": _sha(original),
        }
        files.write(files.document | {"archives": [row]})
    elif mode == "empty":
        files.archive.write_bytes(b"")
    elif mode == "oversize":
        files.archive.write_bytes(b"x" * (evidence.MAX_BYTES + 1))
    elif mode == "directory":
        files.archive.mkdir()
    else:
        os.mkfifo(files.archive)
    with pytest.raises(ValueError):
        files.read()


@pytest.mark.parametrize(
    "names", [(), ("stages.py", "stages.py"), ("movie.mp4",), ("../stages.py",)]
)
def test_consumer_allowlist_is_fixed_unique_nonempty_code_only(
    files: Files, names: tuple[str, ...]
) -> None:
    with pytest.raises(ValueError):
        evidence.read_archives(
            files.descriptor,
            _sha(files.descriptor.read_bytes()),
            protocol=PROTOCOL,
            source_names=names,
            root=files.root,
            binding_fields=frozenset({"windows"}),
        )


@pytest.mark.parametrize(
    "mode", ["root", "rootLink", "protocol", "emptyBindings", "reserved", "badField"]
)
def test_consumer_contract_cannot_be_omitted_or_escaped(files: Files, mode: str) -> None:
    root = files.root
    if mode == "root":
        root = Path("relative-root")
    elif mode == "rootLink":
        linked = files.root / "linked-root"
        linked.symlink_to(files.root, target_is_directory=True)
        root = linked
    fields = {
        "emptyBindings": frozenset(),
        "reserved": frozenset({"version"}),
        "badField": frozenset({"../windows"}),
    }.get(mode, frozenset({"windows"}))
    with pytest.raises(ValueError):
        evidence.read_archives(
            files.descriptor,
            _sha(files.descriptor.read_bytes()),
            protocol="invalid protocol" if mode == "protocol" else PROTOCOL,
            source_names=ROLES,
            root=root,
            binding_fields=fields,
        )


def test_midread_file_change_is_rejected(files: Files, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.lstat

    def change(path: Path) -> os.stat_result:
        if path == files.archive:
            path.write_bytes(path.read_bytes() + b"changed during read")
        return original(path)

    monkeypatch.setattr(Path, "lstat", change)
    with pytest.raises(ValueError, match="changed during reading"):
        evidence._bound_bytes(files.archive, _sha(files.archive.read_bytes()))


def test_descriptor_must_be_regular_pinned_work_evidence(files: Files) -> None:
    outside = files.root / "manifest.json"
    outside.write_bytes(files.descriptor.read_bytes())
    with pytest.raises(ValueError, match="manifests must stay under repository work"):
        evidence.read_archives(
            outside,
            _sha(outside.read_bytes()),
            protocol=PROTOCOL,
            source_names=ROLES,
            root=files.root,
            binding_fields=frozenset({"windows"}),
        )
    assert stat.S_ISREG(files.archive.stat().st_mode)
