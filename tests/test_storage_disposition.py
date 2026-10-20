"""Heavy-media disposition: fixed rule order, reference holds and the written plan."""

import json
from pathlib import Path

import pytest

from replay import edit_completion, storage_disposition
from replay.core import ReplayError
from replay.storage_disposition import Entry


def entry(path: str, size: int = 1 << 30) -> Entry:
    return Entry(Path(path), size, size, 1, 2, 1, 3, 4, 0o100600)


def complete(owner: str) -> bool:
    return owner == "replay-1234567890123456"


@pytest.mark.parametrize(
    ("path", "size", "expected"),
    [
        ("/w/replay-1234567890123456/a.json", 1 << 30, ("keep", "small or non-media evidence")),
        ("/w/replay-1234567890123456/a.mov", 10, ("keep", "small or non-media evidence")),
        ("/w/models/replay-1234567890123456/a.pt", 1 << 30,
         ("keep", "protected, model or tool data")),
        ("/w/bgm-library/abcdefghijk/a.m4a", 1 << 30, ("keep", "protected, model or tool data")),
        ("/w/replay-1234567890123456/delivery/a.mp4", 1 << 30, ("keep", "dated delivery")),
        ("/w/replay-1234567890123456/source-proxy-native-clock.mp4", 1 << 30,
         ("keep", "reused caption proxy")),
        ("/w/scratch/a.mov", 1 << 30, ("hold", "no owning recording")),
        ("/w/replay-9999999999999999/a.mov", 1 << 30,
         ("keep", "recording is unfinished or unverifiable")),
        ("/w/replay-1234567890123456/named.mov", 1 << 30,
         ("hold", "named by a JSON receipt or workflow")),
        ("/w/replay-1234567890123456/free.mov", 1 << 30,
         ("eligible", "unreferenced media of a completed recording")),
    ],
)  # fmt: skip
def test_rules_apply_in_order(path: str, size: int, expected: tuple[str, str]) -> None:
    named = {"/w/replay-1234567890123456/named.mov"}
    assert storage_disposition.decide(entry(path, size), named, complete) == expected


@pytest.mark.parametrize(
    ("path", "owner"),
    [
        ("/w/replay-7696523254311209729/x.mov", "replay-7696523254311209729"),
        ("/w/7696523254311209729/x.mov", "replay-7696523254311209729"),
        ("/w/youtube-Lp6Wje9ktVs/x.mov", "youtube-Lp6Wje9ktVs"),
        ("/w/clean-replay-7696523254311209729-v1/x.mov", "replay-7696523254311209729"),
        ("/w/other/x.mov", None),
    ],
)
def test_the_owner_is_read_from_the_path(path: str, owner: str | None) -> None:
    assert storage_disposition.owner_of(Path(path)) == owner


def test_references_name_media_in_json_but_skip_plans_and_large_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "receipt.json").write_text('{"a": "/v/one.mov", "b": "/v/x.txt"}', "utf-8")
    (tmp_path / "dispositions.json").write_text('{"a": "/v/two.mov"}', "utf-8")
    (tmp_path / "big.json").write_text('{"a": "/v/three.mov"}' + " " * 9000, "utf-8")
    monkeypatch.setattr(storage_disposition, "MAX_JSON_BYTES", 8192)
    assert storage_disposition.references([tmp_path]) == {"/v/one.mov"}


def test_walking_skips_symlinks_and_protected_names(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/a.mov").write_bytes(b"a")
    (tmp_path / "link.mov").symlink_to(tmp_path / "sub/a.mov")
    (tmp_path / "private").mkdir()
    (tmp_path / "private/secret.mov").write_bytes(b"s")
    assert [item.path for item in storage_disposition.walk(tmp_path)] == [tmp_path / "sub/a.mov"]
    assert list(storage_disposition.walk(tmp_path / "missing")) == []


def test_the_completion_gate_is_memoised_and_failures_count_as_unfinished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []

    def report(root: Path) -> dict[str, object]:
        calls.append(root)
        if root.name == "replay-2":
            raise ReplayError("no config")
        return {"complete": True}

    monkeypatch.setattr(edit_completion, "completion_report", report)
    gate = storage_disposition.completes(tmp_path, frozenset({"replay-3"}))
    assert (gate("replay-1"), gate("replay-1"), gate("replay-2"), gate("replay-3")) == (
        True,
        True,
        False,
        True,
    )
    assert calls == [tmp_path / "replay-1", tmp_path / "replay-2"]


def test_the_plan_is_written_with_its_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "work/replay-1234567890123456"
    root.mkdir(parents=True)
    (root / "free.mov").write_bytes(b"f" * 8192)
    (root / "named.mov").write_bytes(b"n" * 8192)
    (root / "hard.mov").hardlink_to(root / "free.mov")
    (root / "receipt.json").write_text(json.dumps({"m": str(root / "named.mov")}), "utf-8")
    monkeypatch.setattr(storage_disposition, "MIN_BYTES", 1)
    monkeypatch.setattr(edit_completion, "completion_report", lambda _root: {"complete": True})
    out = tmp_path / "plan"
    code = storage_disposition.main(
        [str(tmp_path / "work"), "--edits", str(tmp_path / "edits"), "--out", str(out)]
    )
    assert code == 0
    rows = json.loads((out / "dispositions.json").read_text(encoding="utf-8"))["rows"]
    assert sorted((Path(r["path"]).name, r["classification"]) for r in rows) in (
        [("free.mov", "eligible"), ("named.mov", "hold")],
        [("hard.mov", "eligible"), ("named.mov", "hold")],
    )
    printed = json.loads(capsys.readouterr().out)
    assert printed["dispositions"] == str(out / "dispositions.json")
    assert sorted(printed["summary"]) == [
        "eligible: unreferenced media of a completed recording",
        "hold: named by a JSON receipt or workflow",
    ]


def test_media_a_delivery_symlink_publishes_is_kept(tmp_path: Path) -> None:
    render = tmp_path / "replay-1234567890123456/revisions/r/compressor-A"
    delivery = tmp_path / "replay-1234567890123456/revisions/r2/delivery"
    render.mkdir(parents=True)
    delivery.mkdir(parents=True)
    (render / "movie.mp4").write_bytes(b"m")
    (delivery / "movie.mp4").symlink_to(render / "movie.mp4")
    (delivery / "movie.srt").write_text("1", encoding="utf-8")
    delivered = storage_disposition.delivery_targets([tmp_path])
    assert delivered == {str((render / "movie.mp4").resolve())}
    selected = entry(str((render / "movie.mp4").resolve()))
    assert storage_disposition.decide(selected, set(), complete, delivered) == (
        "keep",
        "linked dated delivery",
    )
