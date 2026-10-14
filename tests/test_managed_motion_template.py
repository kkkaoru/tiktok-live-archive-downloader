"""Closed synthetic libraries only; no application calls or real managed writes."""

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest

from replay import managed_motion_template as m
from replay.native_sequence import ContractError
from replay.sequence_journal import decode, encode, publish


class Host:
    tick = 100
    live = "123:456"
    is_closed = True

    def now(self) -> int:
        return self.tick

    def birth(self, pid: int) -> str:
        assert pid == 42
        return self.live

    def closed(self, library: Path) -> bool:
        assert library.suffix == ".fcpbundle"
        return self.is_closed


@dataclass
class Case:
    root: Path
    library: Path
    target: Path
    config: Path
    observation: Path
    operation: Path
    host: Host

    def capture(self) -> None:
        m.capture(self.config, self.observation, host=self.host)

    def apply(self) -> None:
        m.apply(self.config, self.observation, self.operation, host=self.host)


@pytest.fixture
def case(tmp_path: Path) -> Case:
    library = tmp_path / "Lab.fcpbundle"
    target = library / "Titles/observed.moti"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"<old />")
    (library / "CurrentVersion.flexolibrary").write_bytes(b"database")
    (library / "reference.link").symlink_to("missing-original-media")
    old = tmp_path / "old.moti"
    old.write_bytes(target.read_bytes())
    new = tmp_path / "new.moti"
    new.write_bytes(b"<new />")
    control = tmp_path / "control.json"
    actor = {"pid": 42, "birth": "123:456"}
    control.write_bytes(
        encode(
            {
                "version": 1,
                "kind": "managed-template-exclusive-control",
                "library": str(library),
                "actor": actor,
                "soleOperator": True,
                "noQueuedLibraryOpen": True,
            }
        )
    )
    request = tmp_path / "close-request.json"
    request.write_bytes(encode({"bundleID": "com.apple.FinalCutApp", "library": "Lab"}))
    response = tmp_path / "close-response.json"
    response.write_bytes(
        encode(
            {
                "ok": True,
                "data": {
                    "isError": False,
                    "structuredContent": {"pid": 42, "finalCut": {"libraryClosed": True}},
                },
            }
        )
    )
    config = tmp_path / "config.json"
    config.write_bytes(
        encode(
            {
                "version": 1,
                "kind": "managed-motion-template-update",
                "library": str(library),
                "target": str(target),
                "backup": str(tmp_path / "backup.moti"),
                "original": m.reference(old),
                "replacement": m.reference(new),
                "actor": actor,
                "control": m.reference(control),
                "closeRequest": m.reference(request),
                "closeResponse": m.reference(response),
            }
        )
    )
    return Case(
        tmp_path, library, target, config, tmp_path / "capture.json", tmp_path / "operation", Host()
    )


def test_complete_update_and_readonly_reconciliation(case: Case) -> None:
    case.capture()
    case.apply()
    result = m.verify(case.config, case.operation, host=case.host)
    assert result["bodyUpdated"] is True
    assert result["qualityApproved"] is False
    assert case.target.read_bytes() == b"<new />"
    assert (case.root / "backup.moti").read_bytes() == b"<old />"
    assert m.verify(case.config, case.operation, host=case.host) == result
    with pytest.raises(ContractError):
        case.apply()


@pytest.mark.parametrize(
    "changed",
    [
        "target",
        "new.moti",
        "old.moti",
        "config.json",
        "control.json",
        "close-response.json",
        "CurrentVersion.flexolibrary",
    ],
)
def test_changed_inputs_never_overwrite(case: Case, changed: str) -> None:
    case.capture()
    path = case.target if changed == "target" else case.root / changed
    if changed == "CurrentVersion.flexolibrary":
        path = case.library / changed
    path.write_bytes(b"changed")
    with pytest.raises((ContractError, ValueError)):
        case.apply()
    assert not (case.root / "backup.moti").exists()


@pytest.mark.parametrize(
    "mode", ["open", "birth", "expired", "future", "backup", "linked", "symlink", "different"]
)
def test_capture_and_commit_refusals(case: Case, mode: str) -> None:
    case.capture()
    if mode == "open":
        case.host.is_closed = False
    elif mode == "birth":
        case.host.live = "different"
    elif mode == "expired":
        case.host.tick += m.MAX_AGE_NS + 1
    elif mode == "future":
        case.host.tick = 0
    elif mode == "backup":
        (case.root / "backup.moti").write_bytes(b"preserve")
    elif mode == "linked":
        os.link(case.target, case.root / "other-link")
    elif mode == "symlink":
        case.target.unlink()
        case.target.symlink_to(case.root / "old.moti")
    else:
        document = decode(case.config.read_bytes())
        document["kind"] = "different"
        case.config.write_bytes(encode(document))
    with pytest.raises((ContractError, FileExistsError)):
        case.apply()


def test_post_replace_failure_preserves_old_body_and_never_repeats(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    case.capture()
    original = os.replace

    def replace(source: Path, target: Path) -> None:
        original(source, target)
        (case.library / "CurrentVersion.flexolibrary").write_bytes(b"changed-by-actor")

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(ContractError, match="inventory changed"):
        case.apply()
    assert case.target.read_bytes() == b"<new />"
    assert (case.root / "backup.moti").read_bytes() == b"<old />"
    assert (case.operation / "intent.json").is_file()
    assert not (case.operation / "result.json").exists()
    with pytest.raises(ContractError):
        m.verify(case.config, case.operation, host=case.host)


def test_precommit_actor_reopen_retains_intent_and_backup(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    case.capture()
    closed = Mock(side_effect=[True, False])
    monkeypatch.setattr(case.host, "closed", closed)
    with pytest.raises(ContractError, match="handles"):
        case.apply()
    assert case.target.read_bytes() == b"<old />"
    assert (case.root / "backup.moti").is_file()


@pytest.mark.parametrize(
    "field,value", [("library", "relative"), ("target", "relative"), ("version", 2)]
)
def test_invalid_config(case: Case, field: str, value: object) -> None:
    config = decode(case.config.read_bytes())
    config[field] = value
    case.config.write_bytes(encode(config))
    with pytest.raises(ContractError):
        case.capture()


def test_original_close_wrong_pid_rejected(case: Case) -> None:
    response = case.root / "close-response.json"
    response.write_bytes(
        encode(
            {
                "ok": True,
                "data": {
                    "isError": False,
                    "structuredContent": {"pid": 43, "finalCut": {"libraryClosed": True}},
                },
            }
        )
    )
    config = decode(case.config.read_bytes())
    config["closeResponse"] = m.reference(response)
    case.config.write_bytes(encode(config))
    with pytest.raises(ContractError, match="close"):
        case.capture()


def test_generation_change_and_backup_change_refused(case: Case) -> None:
    case.capture()
    case.apply()
    (case.root / "backup.moti").write_bytes(b"wrong")
    with pytest.raises(ContractError, match="body differs"):
        m.verify(case.config, case.operation, host=case.host)
    intent = decode((case.operation / "intent.json").read_bytes())
    intent["code"] = {}
    (case.operation / "intent.json").write_bytes(encode(intent))
    with pytest.raises(ContractError, match="generation"):
        m.verify(case.config, case.operation, host=case.host)


@pytest.mark.parametrize("limit", ["MAX_FILES", "MAX_TOTAL"])
def test_snapshot_budgets(case: Case, monkeypatch: pytest.MonkeyPatch, limit: str) -> None:
    monkeypatch.setattr(m, limit, 1)
    with pytest.raises(ContractError, match="bound|budget"):
        case.capture()


def test_system_host_never_treats_unknown_lsof_as_closed(
    monkeypatch: pytest.MonkeyPatch, case: Case
) -> None:
    run = Mock(return_value=subprocess.CompletedProcess([], 1, b"", b""))
    monkeypatch.setattr(subprocess, "run", run)
    assert m.SystemHost().closed(case.library)
    run.return_value = subprocess.CompletedProcess([], 1, b"", b"permission denied")
    assert not m.SystemHost().closed(case.library)
    run.return_value = subprocess.CompletedProcess([], 0, b"p42", b"")
    assert not m.SystemHost().closed(case.library)
    monkeypatch.setattr(m, "ProcessHost", Mock(return_value=case.host))
    assert m.SystemHost().birth(42) == "123:456"
    assert m.SystemHost().now() > 0


def test_cli_uses_original_contract(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "SystemHost", lambda: case.host)
    assert m.main(["capture", str(case.config), str(case.observation)]) == 0
    assert (
        m.main(["apply", str(case.config), str(case.operation), "--capture", str(case.observation)])
        == 0
    )
    assert m.main(["verify", str(case.config), str(case.operation)]) == 0
    with pytest.raises(SystemExit):
        m.main(["apply", str(case.config), str(case.operation)])


def test_empty_database_sidecar_is_retained(case: Case) -> None:
    sidecar = case.library / "empty.lock"
    sidecar.touch()
    case.capture()
    case.apply()
    assert sidecar.read_bytes() == b""


def test_replacement_mutates_after_backup_no_commit(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    case.capture()
    original = publish

    def changed(path: Path, data: bytes) -> None:
        original(path, data)
        if path.name == "backup.moti":
            (case.root / "new.moti").write_bytes(b"different-source")

    monkeypatch.setattr(m, "publish", changed)
    with pytest.raises(ContractError, match="Reference changed"):
        case.apply()
    assert case.target.read_bytes() == b"<old />"
    assert (case.root / "backup.moti").read_bytes() == b"<old />"


def test_expiry_during_capture_and_before_commit(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(case.host, "now", Mock(side_effect=[100, m.MAX_AGE_NS + 101]))
    with pytest.raises(ContractError, match="expired"):
        case.capture()
    assert not case.observation.exists()
    monkeypatch.setattr(case.host, "now", lambda: 100)
    case.capture()
    monkeypatch.setattr(case.host, "now", Mock(side_effect=[100, m.MAX_AGE_NS + 101]))
    with pytest.raises(ContractError, match="expired"):
        case.apply()
    assert case.target.read_bytes() == b"<old />"


def test_partial_operation_not_repeated(case: Case) -> None:
    case.capture()
    case.operation.mkdir()
    (case.operation / "intent.json").write_bytes(b"historical partial")
    with pytest.raises(FileExistsError):
        case.apply()
    assert (case.operation / "intent.json").read_bytes() == b"historical partial"
    assert case.target.read_bytes() == b"<old />"


def test_invalid_control_and_paths(case: Case) -> None:
    control = case.root / "control.json"
    document = decode(control.read_bytes())
    document["soleOperator"] = False
    control.write_bytes(encode(document))
    config = decode(case.config.read_bytes())
    config["control"] = m.reference(control)
    case.config.write_bytes(encode(config))
    with pytest.raises(ContractError, match="Exclusive"):
        case.capture()


@pytest.mark.parametrize("value", [None, False, [], "", 1])
def test_malformed_actor_cannot_become_identity(case: Case, value: object) -> None:
    config = decode(case.config.read_bytes())
    config["actor"] = value
    case.config.write_bytes(encode(config))
    with pytest.raises(ContractError):
        case.capture()


def test_nine_mib_opaque_database_streams_without_json_reader(
    case: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = case.library / "CurrentVersion.fcpevent"
    chunk = b"opaque database\x00" * 4096
    digest = hashlib.sha256()
    with database.open("wb") as output:
        for _ in range(160):
            output.write(chunk)
            digest.update(chunk)
    assert database.stat().st_size > 9 * 1024 * 1024
    original_read = os.read
    requested: list[int] = []

    def read(descriptor: int, count: int) -> bytes:
        requested.append(count)
        return original_read(descriptor, count)

    monkeypatch.setattr(os, "read", read)
    case.capture()
    case.apply()
    assert m._blob(database, m.MAX_TOTAL) == digest.hexdigest()
    assert max(requested) <= 65536
    assert case.target.read_bytes() == b"<new />"


@pytest.mark.parametrize("change", ["grow", "replace", "hardlink"])
def test_opaque_blob_mutation_during_read_rejected(
    case: Case, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    path = case.library / "CurrentVersion.flexolibrary"
    original_read = os.read
    changed = False

    def read(descriptor: int, count: int) -> bytes:
        nonlocal changed
        if not changed:
            changed = True
            if change == "grow":
                with path.open("ab") as output:
                    output.write(b"growth")
            elif change == "replace":
                replacement = case.root / "replaced-db"
                replacement.write_bytes(path.read_bytes())
                os.replace(replacement, path)
            else:
                os.link(path, case.root / "linked-db")
        return original_read(descriptor, count)

    monkeypatch.setattr(os, "read", read)
    with pytest.raises(ContractError, match="identity changed|exceeds aggregate"):
        m._blob(path, path.stat().st_size)


def test_aggregate_sixty_four_mib_is_not_per_file_allowance(case: Case) -> None:
    database = case.library / "z-large.fcpevent"
    with database.open("wb") as output:
        output.truncate(m.MAX_TOTAL)
    with pytest.raises(ContractError, match="over-budget.*z-large.fcpevent.*bytes=67108864"):
        case.capture()
    assert not case.observation.exists()


def test_link_and_nonregular_opaque_blob_refused(case: Case) -> None:
    database = case.library / "CurrentVersion.flexolibrary"
    os.link(database, case.root / "linked-db")
    with pytest.raises(ContractError, match="Invalid"):
        m._blob(database, m.MAX_TOTAL)
    with pytest.raises(ContractError, match="Invalid"):
        m._blob(case.library, m.MAX_TOTAL)


def test_opaque_blob_opening_identity_change(case: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    path = case.library / "CurrentVersion.flexolibrary"
    original_open = os.open

    def opening(
        path_arg: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        replacement = case.root / "new-db"
        replacement.write_bytes(b"database")
        os.replace(replacement, path)
        return original_open(path_arg, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", opening)
    with pytest.raises(ContractError, match="Opening identity"):
        m._blob(path, m.MAX_TOTAL)
