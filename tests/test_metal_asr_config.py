"""The derived Metal ASR config pins the new plan and the current digest of every code file."""

import hashlib
import json
from pathlib import Path

import pytest

from replay import metal_asr_config as derived
from replay.core import ReplayError


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reference(tmp_path: Path) -> tuple[Path, Path, Path]:
    one, two = tmp_path / "one.py", tmp_path / "two.py"
    one.write_bytes(b"one")
    two.write_bytes(b"two")
    plan = tmp_path / "plan.json"
    plan.write_text("plan", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "separationPlan": {"path": "old", "sha256": "old"},
                "timeouts": {"whole": 5},
                "metalCodeInputs": {str(one): digest(b"one"), str(two): "stale"},
            }
        ),
        encoding="utf-8",
    )
    return config, plan, two


def test_the_plan_is_pinned_and_stale_code_digests_are_refreshed(tmp_path: Path) -> None:
    config, plan, two = reference(tmp_path)
    output = tmp_path / "out.json"
    written = derived.derive(config, plan, output)
    assert written["separationPlan"] == {"path": str(plan), "sha256": digest(b"plan")}
    assert written["timeouts"] == {"whole": 5}
    pinned = written["metalCodeInputs"]
    assert isinstance(pinned, dict)
    assert pinned[str(two)] == digest(b"two")
    assert json.loads(output.read_text(encoding="utf-8")) == written
    assert json.loads(config.read_text(encoding="utf-8"))["metalCodeInputs"][str(two)] == "stale"


def test_an_existing_output_a_relative_plan_or_a_missing_code_file_is_refused(
    tmp_path: Path,
) -> None:
    config, plan, two = reference(tmp_path)
    existing = tmp_path / "taken.json"
    existing.write_text("x", encoding="utf-8")
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        derived.derive(config, plan, existing)
    with pytest.raises(ReplayError, match="existing absolute path"):
        derived.derive(config, Path("plan.json"), tmp_path / "a.json")
    two.unlink()
    with pytest.raises(ReplayError, match="pinned code file is missing"):
        derived.derive(config, plan, tmp_path / "b.json")


def test_main_prints_the_output_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config, plan, _ = reference(tmp_path)
    output = tmp_path / "viamain.json"
    assert derived.main([str(config), str(plan), str(output)]) == 0
    assert capsys.readouterr().out.strip() == str(output)


def test_a_reference_that_is_not_an_object_or_pins_nothing_is_refused(tmp_path: Path) -> None:
    _, plan, _ = reference(tmp_path)
    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    with pytest.raises(ReplayError, match="must be a JSON object"):
        derived.derive(listed, plan, tmp_path / "c.json")
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"metalCodeInputs": {}}), encoding="utf-8")
    with pytest.raises(ReplayError, match="pins no code files"):
        derived.derive(empty, plan, tmp_path / "d.json")
