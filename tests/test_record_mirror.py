from pathlib import Path

import pytest

from replay import record_mirror
from replay.core import ReplayError

REAL_VOLUME = record_mirror.volume


def put(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def other_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(record_mirror, "volume", lambda path: 2 if "mirror" in path.parts else 1)


def test_small_records_are_copied_and_large_data_is_not(tmp_path: Path) -> None:
    edits = tmp_path / "edits"
    put(edits / "replay-1" / "edit-config.json", "{}")
    put(edits / "replay-1" / "transcripts" / "a.srt", "1")
    put(edits / "replay-1" / "evidence" / "big.json", "{}")
    put(edits / "replay-1" / "movie.mov", "x")
    put(edits / "work" / "x.json", "{}")
    result = record_mirror.mirror(edits, tmp_path / "mirror")
    assert result == {"copied": 2, "unchanged": 0}
    assert (tmp_path / "mirror" / "replay-1" / "transcripts" / "a.srt").read_text() == "1"
    assert not (tmp_path / "mirror" / "replay-1" / "evidence").exists()
    assert not (tmp_path / "mirror" / "work").exists()


def test_a_second_run_skips_records_that_already_match(tmp_path: Path) -> None:
    edits = tmp_path / "edits"
    put(edits / "youtube-9" / "edit-config.json", "{}")
    record_mirror.mirror(edits, tmp_path / "mirror")
    assert record_mirror.mirror(edits, tmp_path / "mirror") == {"copied": 0, "unchanged": 1}


def test_oversized_files_and_links_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edits = tmp_path / "edits"
    put(edits / "replay-1" / "a.json", "{}")
    (edits / "replay-1" / "link.json").symlink_to(edits / "replay-1" / "a.json")
    monkeypatch.setattr(record_mirror, "MAX_FILE_BYTES", 1)
    assert record_mirror.mirror(edits, tmp_path / "mirror") == {"copied": 0, "unchanged": 0}


def test_same_volume_destination_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    edits = tmp_path / "edits"
    edits.mkdir()
    monkeypatch.setattr(record_mirror, "volume", lambda _path: 1)
    with pytest.raises(ReplayError, match="different volume"):
        record_mirror.mirror(edits, tmp_path / "mirror")


def test_volume_uses_the_nearest_existing_ancestor(tmp_path: Path) -> None:
    assert REAL_VOLUME(tmp_path / "missing" / "deeper") == tmp_path.stat().st_dev


def test_cli_prints_counts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    edits = tmp_path / "edits"
    put(edits / "replay-1" / "edit-config.json", "{}")
    assert record_mirror.main(["--edits", str(edits), str(tmp_path / "mirror")]) == 0
    assert capsys.readouterr().out.strip() == '{"copied": 1, "unchanged": 0}'


def test_cli_reports_a_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "edits").mkdir()
    monkeypatch.setattr(record_mirror, "volume", lambda _path: 1)
    assert record_mirror.main(["--edits", str(tmp_path / "edits"), str(tmp_path / "m")]) == 1
    assert "different volume" in capsys.readouterr().err
