from pathlib import Path

import pytest

from replay import work_roots
from replay.core import ReplayError

HEADER = "name\told_path\tnew_path\tfiles\tbytes\tmoved_at\n"


def index(work: Path, body: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    (work / "RELOCATED-TO-SSD.tsv").write_text(HEADER + body, encoding="utf-8")


def test_external_work_defaults_to_the_ssd_root() -> None:
    assert work_roots.external_work({}) == Path("/Volumes/m.2-ssd/movie/replay-work")


def test_external_work_follows_the_environment() -> None:
    assert work_roots.external_work({"REPLAY_EXTERNAL_WORK_ROOT": "/x/y"}) == Path("/x/y")


def test_external_work_reads_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REPLAY_EXTERNAL_WORK_ROOT", "/from/env")
    assert work_roots.roots(work=Path("/w")) == (Path("/w"), Path("/from/env"))


def test_within_work_accepts_both_roots_and_rejects_others(tmp_path: Path) -> None:
    work = tmp_path / "work"
    external = tmp_path / "external"
    (work / "a").mkdir(parents=True)
    (external / "b").mkdir(parents=True)
    assert work_roots.within_work(work / "a", work=work, external=external) is True
    assert work_roots.within_work(external / "b", work=work, external=external) is True
    assert work_roots.within_work(tmp_path / "other", work=work, external=external) is False
    assert work_roots.within_work(work / ".." / "other", work=work, external=external) is False


def test_within_work_rejects_a_linked_leaf(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    (work / "link").symlink_to(work)
    assert work_roots.within_work(work / "link", work=work, external=tmp_path / "e") is False


def test_resolve_maps_a_relocated_path_and_keeps_others(tmp_path: Path) -> None:
    work = tmp_path / "work"
    external = tmp_path / "external"
    index(work, f"big\t{work}/big\t{external}/big\t3\t30\t2026-10-17T00:00:00Z\n")
    assert work_roots.resolve(work / "big" / "a" / "b.json", work=work, external=external) == (
        external / "big" / "a" / "b.json"
    )
    assert work_roots.resolve(work / "big", work=work, external=external) == external / "big"
    assert (
        work_roots.resolve(work / "bigger" / "x", work=work, external=external)
        == work / "bigger" / "x"
    )


def test_resolve_without_an_index_returns_the_path(tmp_path: Path) -> None:
    assert (
        work_roots.resolve(tmp_path / "w" / "x", work=tmp_path / "w", external=tmp_path)
        == tmp_path / "w" / "x"
    )


def test_relocations_reject_wrong_columns(tmp_path: Path) -> None:
    (tmp_path / "RELOCATED-TO-SSD.tsv").write_text("name\tpath\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="does not have the columns"):
        work_roots.relocations(work=tmp_path, external=tmp_path)


def test_relocations_reject_a_malformed_row(tmp_path: Path) -> None:
    index(tmp_path, "big\t/w/big\t/e/big\tmany\t30\tt\n")
    with pytest.raises(ReplayError, match="Malformed relocation row"):
        work_roots.relocations(work=tmp_path, external=Path("/e"))


def test_relocations_reject_a_target_outside_the_external_root(tmp_path: Path) -> None:
    index(tmp_path, "big\t/w/big\t/elsewhere/big\t3\t30\tt\n")
    with pytest.raises(ReplayError, match="must map a work folder into the external root"):
        work_roots.relocations(work=tmp_path, external=Path("/e"))


def test_relocations_reject_a_renamed_folder(tmp_path: Path) -> None:
    index(tmp_path, "big\t/w/other\t/e/big\t3\t30\tt\n")
    with pytest.raises(ReplayError, match="must map a work folder into the external root"):
        work_roots.relocations(work=tmp_path, external=Path("/e"))


def test_missing_reports_absent_targets_and_reappeared_sources(tmp_path: Path) -> None:
    work = tmp_path / "work"
    external = tmp_path / "external"
    (external / "ok").mkdir(parents=True)
    (external / "back").mkdir()
    (work / "back").mkdir(parents=True)
    index(
        work,
        f"ok\t{work}/ok\t{external}/ok\t1\t1\tt\n"
        f"gone\t{work}/gone\t{external}/gone\t1\t1\tt\n"
        f"back\t{work}/back\t{external}/back\t1\t1\tt\n",
    )
    assert work_roots.missing(work=work, external=external) == ["back", "gone"]


def test_main_resolve_prints_current_locations(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(work_roots, "resolve", lambda path: Path("/new") / path.name)
    assert work_roots.main(["resolve", "/old/a", "/old/b"]) == 0
    assert capsys.readouterr().out == "/new/a\n/new/b\n"


def test_main_check_reports_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(work_roots, "missing", lambda: [])
    assert work_roots.main(["check"]) == 0
    assert capsys.readouterr().out == "ok\n"


def test_main_check_lists_problems(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(work_roots, "missing", lambda: ["gone"])
    assert work_roots.main(["check"]) == 1
    assert capsys.readouterr().out == "gone\n"
