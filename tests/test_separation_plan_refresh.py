"""Plan refresh: digests follow the files, the old namespace is untouched, odd inputs refused."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from replay import separation_plan_refresh as refresh
from replay.core import ReplayError


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    (tmp_path / "edits" / "work").mkdir(parents=True)
    return tmp_path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pinned(root: Path) -> tuple[Path, Path, Path]:
    one, two = root / "one.py", root / "two.py"
    one.write_bytes(b"one")
    two.write_bytes(b"two")
    config, template = root / "config.json", root / "template.json"
    rows = [
        {"path": str(one), "sha256": digest(b"one")},
        {"path": str(two), "sha256": digest(b"old two")},
    ]
    config.write_text(json.dumps({"codeInputs": rows, "directory": "x", "gainDB": -0.1}))
    template.write_text(
        json.dumps({"codeInputs": rows, "outputDirectory": "y", "source": {"path": "z"}})
    )
    return config, template, two


def test_changed_names_only_the_files_that_differ(root: Path) -> None:
    config, _, _ = pinned(root)
    rows = json.loads(config.read_text())["codeInputs"]
    assert refresh.changed(rows) == ["two.py"]


def test_refreshed_digests_follow_the_current_files(root: Path) -> None:
    config, _, _ = pinned(root)
    rows = refresh.refreshed(json.loads(config.read_text())["codeInputs"])
    assert [row["sha256"] for row in rows] == [digest(b"one"), digest(b"two")]


def test_a_missing_or_outside_file_is_refused(
    root: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "a.py"
    outside.write_bytes(b"a")
    with pytest.raises(ReplayError, match="missing or outside"):
        refresh.refreshed([{"path": str(outside), "sha256": "x"}])
    with pytest.raises(ReplayError, match="missing or outside"):
        refresh.refreshed([{"path": str(root / "absent.py"), "sha256": "x"}])
    with pytest.raises(ReplayError, match="non-empty"):
        refresh.refreshed([])
    with pytest.raises(ReplayError, match="non-empty"):
        refresh.refreshed("nope")
    with pytest.raises(ReplayError, match="Expected a pinned code list"):
        refresh.changed("nope")


def test_the_new_namespace_gets_refreshed_files_and_the_old_ones_are_untouched(root: Path) -> None:
    config, template, _ = pinned(root)
    before = (config.read_bytes(), template.read_bytes())
    new = root / "edits" / "work" / "fresh"
    plan = refresh.refresh(config, template, new, run=False)
    written = json.loads((new / "PCM-PREPARATION-CONFIG-v1.json").read_text())
    shaped = json.loads((new / "PIPELINE-PLAN-TEMPLATE-v1.json").read_text())
    assert plan == new / "PIPELINE-PLAN-v1.json"
    assert [row["sha256"] for row in written["codeInputs"]] == [digest(b"one"), digest(b"two")]
    assert written["directory"] == str(new)
    assert written["gainDB"] == -0.1
    assert shaped["outputDirectory"] == str(new / "pipeline")
    assert shaped["source"] == {"path": str(new / "model-input.wav"), "sha256": None}
    assert (config.read_bytes(), template.read_bytes()) == before
    assert (new / "tmp").is_dir()
    assert (new / "cache").is_dir()


def test_an_existing_or_misplaced_namespace_is_refused(
    root: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    config, template, _ = pinned(root)
    existing = root / "edits" / "work" / "taken"
    existing.mkdir()
    with pytest.raises(ReplayError, match="must not exist"):
        refresh.refresh(config, template, existing, run=False)
    with pytest.raises(ReplayError, match="under edits/work"):
        refresh.refresh(config, template, root / "elsewhere", run=False)
    with pytest.raises(ReplayError, match="must not exist"):
        refresh.refresh(config, template, root / "missing-parent" / "new", run=False)


def test_the_producers_run_with_a_local_scratch_and_offline_models(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, template, _ = pinned(root)
    calls: list[tuple[list[str], dict[str, str]]] = []

    class Done:
        returncode = 0

    def fake_run(command: list[str], **options: object) -> Done:
        environment = options["env"]
        assert isinstance(environment, dict)
        calls.append((command, environment))
        return Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    new = root / "edits" / "work" / "ran"
    refresh.refresh(config, template, new, run=True)
    assert [call[0][4] for call in calls] == ["extract", "prepare-plan"]
    assert calls[1][0][-1] == str(new / "PIPELINE-PLAN-v1.json")
    first = calls[0][1]
    assert first["TMPDIR"] == str(new / "tmp")
    assert first["XDG_CACHE_HOME"] == str(new / "cache")
    assert first["HF_HUB_OFFLINE"] == "1"
    assert first["HF_HUB_CACHE"].endswith("mel-roformer/hub")


def test_main_writes_without_running_when_asked(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config, template, _ = pinned(root)
    new = root / "edits" / "work" / "mainrun"
    assert refresh.main([str(config), str(template), str(new), "--no-run"]) == 0
    assert capsys.readouterr().out.strip() == str(new / "PIPELINE-PLAN-v1.json")
