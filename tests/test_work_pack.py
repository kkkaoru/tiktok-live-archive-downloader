import hashlib
import io
import json
import os
import tarfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from replay import intermediate_offload, leaf_retirement, r2_worker, work_offload, work_pack
from replay.core import ReplayError
from replay.r2_worker import RemoteObject

UPLOADED = datetime(2026, 10, 17, 3, 0, tzinfo=UTC)
LATER = datetime.now(UTC) + timedelta(days=1)
STAMP = "20261017T120000"


class FakeStore:
    """In-memory R2 double; `fail` makes every upload raise."""

    def __init__(self, *, fail: bool = False) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.fail = fail

    def head(self, key: str) -> RemoteObject | None:
        stored = self.objects.get(key)
        if stored is None:
            return None
        return RemoteObject(len(stored[0]), UPLOADED, dict(stored[1]))

    def put(self, key: str, path: Path, metadata: Mapping[str, str]) -> None:
        if self.fail:
            raise ReplayError("synthetic upload failure")
        self.objects[key] = (path.read_bytes(), dict(metadata))

    def get(self, key: str, destination: Path) -> int:
        data = self.objects[key][0]
        destination.write_bytes(data)
        return len(data)


def closed(paths: Sequence[str]) -> set[str]:
    return set()


def media(path: Path, data: bytes = b"small scratch") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def work_root(tmp_path: Path) -> Path:
    root = tmp_path / "work"
    root.mkdir()
    return Path(os.path.realpath(root))


def swept(root: Path, store: FakeStore, **options: object) -> list[work_pack.UnitResult]:
    probe = options.get("probe", closed)
    assert callable(probe)
    return work_pack.sweep(
        root, cutoff=LATER, store=store, log=root.parent / "log.jsonl", stamp=STAMP, probe=probe
    )


def test_scan_small_takes_only_small_non_empty_files(tmp_path: Path) -> None:
    unit = work_root(tmp_path) / "pilot"
    small = media(unit / "a" / "row.json")
    media(unit / "empty.json", b"")
    media(unit / "large.bin", b"x" * work_offload.MIN_BYTES)
    media(unit / "evidence" / work_offload.PACK_INDEX_NAME)
    media(unit / "evidence" / f".{work_offload.PACK_INDEX_NAME}.tmp")
    rows = work_offload.scan(unit, int(LATER.timestamp() * 1e9), small=True)
    assert [row.path for row in rows] == [str(small)]


def test_dry_run_plans_without_touching_anything(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    source = media(root / "pilot" / "row.json")
    [result] = work_pack.sweep(root, cutoff=LATER, store=None, log=tmp_path / "log", stamp=STAMP)
    assert (result.name, result.candidates, result.candidate_bytes) == ("pilot", 1, 13)
    assert source.exists()
    assert not work_pack.index_path(root / "pilot").exists()


def test_apply_packs_offloads_retires_and_restores(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    unit = root / "pilot"
    first = media(unit / "a" / "row.json")
    second = media(unit / "b" / "c" / "row.csv", b"other small scratch")
    os.chmod(second, 0o640)
    mtime = second.stat().st_mtime_ns
    store = FakeStore()
    top = media(unit / "top.md", b"# notes")
    [result] = swept(root, store)
    assert (result.packs, result.packed, result.retired, result.error) == (3, 3, 3, None)
    assert not first.exists() and not second.exists() and not top.exists()
    packs = work_pack.load(unit)
    assert [(pack.group, pack.relative) for pack in packs] == [
        (".", f"pilot/{STAMP}-0000-_root.tar.zst"),
        ("a", f"pilot/{STAMP}-0001-a.tar.zst"),
        ("b", f"pilot/{STAMP}-0002-b.tar.zst"),
    ]
    assert {pack.state for pack in packs} == {work_pack.State.RETIRED}
    assert {pack.compression for pack in packs} == {work_pack.ZSTD}
    assert [member.relative for member in packs[2].members] == ["b/c/row.csv"]
    assert not list((work_pack.pack_root(root) / "pilot").iterdir())
    assert intermediate_offload.status(unit, recording="pilot")["retired"] == 3
    assert all(key.startswith(f"{intermediate_offload.KEY_PREFIX}/pilot/") for key in store.objects)
    assert work_pack.restore(unit, work_root=root, store=store) == {
        "restored": 3,
        "alreadyPresent": 0,
    }
    assert first.read_bytes() == b"small scratch"
    assert (second.stat().st_mode & 0o777, second.stat().st_mtime_ns) == (0o640, mtime)
    assert {pack.state for pack in work_pack.load(unit)} == {work_pack.State.RESTORED}
    assert not list((work_pack.pack_root(root) / "pilot").iterdir())


def test_restore_relative_fetches_only_its_pack_and_member(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    unit = root / "pilot"
    wanted = media(unit / "a" / "one.json")
    sibling = media(unit / "a" / "two.json", b"sibling")
    other = media(unit / "b" / "row.json", b"other")
    store = FakeStore()
    swept(root, store)
    fetched: list[str] = []
    get = store.get

    def counted(key: str, destination: Path) -> int:
        fetched.append(key)
        return get(key, destination)

    store.get = counted  # type: ignore[method-assign]
    counts = work_pack.restore(unit, work_root=root, store=store, relative="a/one.json")
    assert counts == {"restored": 1, "alreadyPresent": 0}
    assert wanted.exists() and not sibling.exists() and not other.exists()
    assert len(fetched) == 1
    assert {pack.state for pack in work_pack.load(unit)} == {work_pack.State.RETIRED}
    assert work_pack.restore(unit, work_root=root, store=store, relative="a/one.json") == {
        "restored": 0,
        "alreadyPresent": 1,
    }
    with pytest.raises(ReplayError, match="No offloaded pack holds"):
        work_pack.restore(unit, work_root=root, store=store, relative="missing.json")


def test_chunks_split_a_group_at_the_pack_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unit = work_root(tmp_path) / "pilot"
    for name in ("1.json", "2.json", "3.json"):
        media(unit / "a" / name, b"12345")
    monkeypatch.setattr(work_pack, "PACK_BYTES", 10)
    rows = work_offload.scan(unit, int(LATER.timestamp() * 1e9), small=True)
    sizes = [(group, len(chunk)) for group, chunk in work_pack.chunks(unit, rows)]
    assert sizes == [("a", 2), ("a", 1)]


def test_slug_makes_group_names_safe() -> None:
    assert work_pack.slug(".") == "_root"
    assert work_pack.slug("日本 語/x") == "_____x"
    assert len(work_pack.slug("x" * 200)) == work_pack.SLUG_LIMIT


def test_rerun_finishes_an_uploaded_pack_and_discards_a_pending_one(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    unit = root / "pilot"
    left = media(unit / "a" / "row.json")
    changed = media(unit / "a" / "changed.json", b"packed bytes")
    store = FakeStore()
    rows = work_offload.scan(unit, int(LATER.timestamp() * 1e9), small=True)
    packs_root = work_pack.pack_root(root)
    archive = packs_root / "pilot" / "old.tar.zst"
    members, _kept = work_pack.build(unit, rows, archive)
    uploaded = work_pack.Pack(
        "pilot/old.tar.zst",
        leaf_retirement.sha256_of(archive),
        archive.stat().st_size,
        work_pack.State.UPLOADED,
        members,
        "a",
        work_pack.ZSTD,
    )
    intermediate_offload.offload(
        [work_offload.row_of(str(archive), os.lstat(archive))],
        recording="pilot",
        roots=(packs_root,),
        viewer_root=unit,
        store=store,
    )
    stale = packs_root / "pilot" / "stale.tar.zst"
    stale.write_bytes(b"half written")
    pending = replace(uploaded, relative="pilot/stale.tar.zst", state=work_pack.State.PENDING)
    work_pack.save(unit, [uploaded, pending])
    changed.write_bytes(b"edited after packing")
    [result] = swept(root, store)
    assert not left.exists() and not archive.exists() and not stale.exists()
    old, new = work_pack.load(unit)
    assert (old.relative, old.state) == ("pilot/old.tar.zst", work_pack.State.RETIRED)
    assert [member.relative for member in new.members] == ["a/changed.json"]
    assert new.members[0].sha256 == hashlib.sha256(b"edited after packing").hexdigest()
    assert (result.retired, result.error) == (2, None)
    work_pack.restore(unit, work_root=root, store=store, relative="a/changed.json")
    assert changed.read_bytes() == b"edited after packing"


def test_still_matching_rejects_a_changed_or_linked_member(tmp_path: Path) -> None:
    unit = work_root(tmp_path) / "pilot"
    source = media(unit / "row.json")
    info = source.stat()
    member = work_pack.Member("row.json", "0" * 64, info.st_size, 0o644, info.st_mtime_ns)
    assert work_pack._still_matching(unit, member) is None
    assert work_pack._still_matching(unit, replace(member, relative="gone.json")) is None
    assert work_pack._still_matching(unit, replace(member, size=1)) is None
    os.symlink(unit, tmp_path / "alias")
    assert work_pack._still_matching(tmp_path / "alias", member) is None


def test_legacy_uncompressed_pack_still_restores(tmp_path: Path) -> None:
    unit = work_root(tmp_path) / "pilot"
    archive = tmp_path / "legacy.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("row.json")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"abc"))
    digest = leaf_retirement.sha256_of
    member = work_pack.Member("row.json", hashlib.sha256(b"abc").hexdigest(), 3, 0o600, 5)
    pack = work_pack.Pack("x.tar", digest(archive), 1, work_pack.State.RETIRED, [member])
    assert work_pack._extract(pack, archive, unit, {"row.json"}) == (1, 0)
    assert (unit / "row.json").read_bytes() == b"abc"


def test_extract_rejects_a_pack_missing_listed_members(tmp_path: Path) -> None:
    archive = tmp_path / "empty.tar"
    with tarfile.open(archive, "w"):
        pass
    member = work_pack.Member("row.json", "", 0, 0, 0)
    pack = work_pack.Pack("x.tar", "", 0, work_pack.State.RETIRED, [member])
    with pytest.raises(ReplayError, match="lacks 1 listed members"):
        work_pack._extract(pack, archive, tmp_path / "unit", {"row.json"})


def test_download_refuses_missing_and_expired_entries(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    unit = root / "pilot"
    media(unit / "row.json")
    store = FakeStore()
    swept(root, store)
    [pack] = work_pack.load(unit)
    packs_root = work_pack.pack_root(root)
    with pytest.raises(ReplayError, match="has no offload entry"):
        work_pack._download(replace(pack, relative="x"), unit, packs_root, store, UPLOADED)
    expired = UPLOADED + intermediate_offload.LIFETIME
    with pytest.raises(ReplayError, match="expired from R2"):
        work_pack._download(pack, unit, packs_root, store, expired)


def test_restore_keeps_a_file_that_came_back(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    unit = root / "pilot"
    source = media(unit / "row.json")
    store = FakeStore()
    swept(root, store)
    media(source, b"recreated")
    assert work_pack.restore(unit, work_root=root, store=store) == {
        "restored": 0,
        "alreadyPresent": 1,
    }
    assert source.read_bytes() == b"recreated"


def test_restore_skips_packs_that_are_not_offloaded(tmp_path: Path) -> None:
    unit = work_root(tmp_path) / "pilot"
    work_pack.save(unit, [work_pack.Pack("pilot/x.tar", "0" * 64, 1, work_pack.State.PENDING, [])])
    counts = work_pack.restore(unit, work_root=unit.parent, store=FakeStore())
    assert counts == {"restored": 0, "alreadyPresent": 0}


def test_restore_rejects_a_corrupt_member(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    unit = root / "pilot"
    media(unit / "row.json")
    store = FakeStore()
    swept(root, store)
    [pack] = work_pack.load(unit)
    bad = work_pack.Member("row.json", "0" * 64, 13, 0o644, 0)
    work_pack.save(unit, [replace(pack, members=[bad])])
    with pytest.raises(ReplayError, match="does not match its SHA-256"):
        work_pack.restore(unit, work_root=root, store=store)
    assert not (unit / "row.json").exists()


def test_restore_rejects_a_pack_whose_tar_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = work_root(tmp_path)
    unit = root / "pilot"
    media(unit / "row.json")
    store = FakeStore()
    swept(root, store)
    [pack] = work_pack.load(unit)
    pack.sha256 = "0" * 64
    work_pack.save(unit, [pack])
    with pytest.raises(ReplayError, match="Restored pack"):
        work_pack.restore(unit, work_root=root, store=store)


def test_extract_rejects_a_non_regular_member(tmp_path: Path) -> None:
    archive = tmp_path / "pack.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("row.json")
        info.type = tarfile.DIRTYPE
        tar.addfile(info)
    pack = work_pack.Pack(
        "x.tar", "", 0, work_pack.State.RETIRED, [work_pack.Member("row.json", "", 0, 0, 0)]
    )
    with pytest.raises(ReplayError, match="not a regular file"):
        work_pack._extract(pack, archive, tmp_path / "unit", {"row.json"})


def test_member_changed_while_packing_is_not_listed_or_unlinked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = work_root(tmp_path)
    unit = root / "pilot"
    kept = media(unit / "a.json")
    changed = media(unit / "b.json")
    gone = media(unit / "c.json")
    scan = work_offload.scan

    calls: list[Path] = []

    def stale(path: Path, cutoff_ns: int, *, small: bool = False) -> list[leaf_retirement.Row]:
        rows = scan(path, cutoff_ns, small=small)
        calls.append(path)
        if len(calls) == 2:
            changed.write_bytes(b"rewritten while packing")
            gone.unlink()
        return rows

    monkeypatch.setattr(work_offload, "scan", stale)
    [result] = swept(root, FakeStore())
    assert (result.packed, result.retired) == (1, 1)
    assert [member.relative for member in work_pack.load(unit)[0].members] == ["a.json"]
    assert not kept.exists()
    assert changed.read_bytes() == b"rewritten while packing"


def test_member_whose_identity_changed_is_not_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = work_root(tmp_path)
    unit = root / "pilot"
    source = media(unit / "a.json")
    scan = work_offload.scan

    def touched(path: Path, cutoff_ns: int, *, small: bool = False) -> list[leaf_retirement.Row]:
        rows = scan(path, cutoff_ns, small=small)
        os.utime(source, ns=(1, 1))
        return rows

    monkeypatch.setattr(work_offload, "scan", touched)
    [result] = swept(root, FakeStore())
    assert (result.packed, result.retired) == (0, 0)
    assert source.exists()


def test_open_member_is_packed_but_held(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    source = media(root / "pilot" / "a.json")

    def everything_open(paths: Sequence[str]) -> set[str]:
        return set(paths)

    [result] = swept(root, FakeStore(), probe=everything_open)
    assert (result.packed, result.retired, result.held) == (1, 0, {"open handle": 1})
    assert source.exists()


def test_failed_upload_keeps_every_member(tmp_path: Path) -> None:
    root = work_root(tmp_path)
    source = media(root / "pilot" / "a.json")
    [result] = swept(root, FakeStore(fail=True))
    assert result.error == "synthetic upload failure"
    assert source.exists()
    assert work_pack.load(root / "pilot")[0].state is work_pack.State.PENDING


def test_pack_changed_before_upload_fails_the_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = work_root(tmp_path)
    source = media(root / "pilot" / "a.json")

    def skipped(*args: object, **kwargs: object) -> intermediate_offload.OffloadSummary:
        return intermediate_offload.OffloadSummary(0, 0, {"pack": "identity changed"})

    monkeypatch.setattr(intermediate_offload, "offload", skipped)
    [result] = swept(root, FakeStore())
    assert result.error is not None and "changed before it was confirmed" in result.error
    assert source.exists()


def test_index_round_trips_and_rejects_bad_documents(tmp_path: Path) -> None:
    unit = tmp_path / "pilot"
    pack = work_pack.Pack(
        "pilot/x.tar",
        "a" * 64,
        10,
        work_pack.State.UPLOADED,
        [work_pack.Member("row.json", "b" * 64, 3, 0o644, 7)],
    )
    work_pack.save(unit, [pack])
    assert work_pack.load(unit) == [pack]
    assert work_pack.load(tmp_path / "missing") == []
    path = work_pack.index_path(unit)
    cases: list[tuple[object, str]] = [
        ({"version": 2, "unit": "pilot", "packs": []}, "another unit"),
        ({"version": 1, "unit": "pilot", "packs": {}}, "packs array"),
        ({"version": 1, "unit": "pilot", "packs": [{"relative": 1}]}, "relative path"),
        (
            {"version": 1, "unit": "pilot", "packs": [{"relative": "x", "sha256": "", "bytes": 1}]},
            "members array",
        ),
        (
            {
                "version": 1,
                "unit": "pilot",
                "packs": [
                    {"relative": "x", "sha256": "", "bytes": 1, "members": [], "state": "lost"}
                ],
            },
            "unknown state",
        ),
        (
            {
                "version": 1,
                "unit": "pilot",
                "packs": [
                    {
                        "relative": "x",
                        "sha256": "",
                        "bytes": 1,
                        "state": "retired",
                        "members": [{"relative": 1}],
                    }
                ],
            },
            "relative path and SHA-256",
        ),
        (
            {
                "version": 1,
                "unit": "pilot",
                "packs": [
                    {
                        "relative": "x",
                        "sha256": "",
                        "bytes": 1,
                        "state": "retired",
                        "members": [{"relative": "r", "sha256": "", "bytes": "1"}],
                    }
                ],
            },
            "integer bytes",
        ),
    ]
    for document, message in cases:
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ReplayError, match=message):
            work_pack.load(unit)


def test_a_second_process_cannot_take_the_unit_lock(tmp_path: Path) -> None:
    unit = tmp_path / "pilot"
    with (
        work_pack.locked(unit),
        pytest.raises(ReplayError, match="Another process"),
        work_pack.locked(unit),
    ):
        pass


def test_report_totals_held_and_failures() -> None:
    results = [
        work_pack.UnitResult("a", 2, 10, packed=2, retired=1, retired_bytes=5, held={"x": 1}),
        work_pack.UnitResult("b", 1, 5, held={"x": 2}, error="boom"),
    ]
    summary = work_pack.report(results, cutoff=UPLOADED, applied=False)
    assert summary == {
        "mode": "dry-run",
        "cutoff": "2026-10-17T12:00:00+09:00",
        "units": 2,
        "candidates": 3,
        "candidateBytes": 15,
        "packs": 0,
        "packed": 2,
        "packedBytes": 0,
        "retired": 1,
        "retiredBytes": 5,
        "held": {"x": 3},
        "failed": {"b": "boom"},
    }


@contextmanager
def fake_session(store: FakeStore) -> Generator[FakeStore]:
    yield store


def test_main_dry_run_and_apply_and_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = work_root(tmp_path)
    source = media(root / "pilot" / "a.json")
    store = FakeStore()
    monkeypatch.setattr(r2_worker, "session", lambda bucket: fake_session(store))
    assert work_pack.main(["--work-root", str(root)]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"] == 0
    assert work_pack.main(["--work-root", str(root), "sweep", "--quiet-days", "-1"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"] == 1
    arguments = ["--work-root", str(root), "sweep", "--quiet-days", "-1", "--apply"]
    assert work_pack.main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["retired"] == 1
    assert not source.exists()
    assert work_pack.main(["--work-root", str(root), "restore", str(root / "pilot")]) == 0
    assert json.loads(capsys.readouterr().out) == {"alreadyPresent": 0, "restored": 1}
    assert source.exists()


def test_main_sweep_skips_named_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = work_root(tmp_path)
    active = media(root / "active" / "a.json")
    store = FakeStore()
    monkeypatch.setattr(r2_worker, "session", lambda bucket: fake_session(store))
    arguments = ["--work-root", str(root), "sweep", "--quiet-days", "-1", "--apply"]
    assert work_pack.main([*arguments, "--skip", "active"]) == 0
    assert json.loads(capsys.readouterr().out)["candidates"] == 0
    assert active.exists()
    assert store.objects == {}


def test_main_reports_failed_units_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = work_root(tmp_path)
    media(root / "pilot" / "a.json")
    monkeypatch.setattr(r2_worker, "session", lambda bucket: fake_session(FakeStore(fail=True)))
    arguments = ["--work-root", str(root), "sweep", "--quiet-days", "-1", "--apply"]
    assert work_pack.main(arguments) == 1
    assert json.loads(capsys.readouterr().out)["failed"] == {"pilot": "synthetic upload failure"}
    (root / "pilot" / "evidence" / work_offload.PACK_INDEX_NAME).write_text("[]", "utf-8")
    assert work_pack.main(["--work-root", str(root), "restore", str(root / "pilot")]) == 1
    assert "WORK_PACK_ERROR" in capsys.readouterr().err
