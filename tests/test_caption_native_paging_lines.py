"""The line-limit entry point runs paging or its verifier with a lower limit, then restores it."""

import runpy
import sys

import pytest

from replay import caption_indivisible_fit, caption_native_paging_lines, caption_style_defaults
from replay.core import ReplayError


def test_the_module_runs_with_the_requested_limit_and_its_own_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, int, int, list[str]]] = []

    def fake_run(name: str, **_options: object) -> dict[str, object]:
        limits = (
            caption_style_defaults.MAX_CAPTION_LINES,
            caption_indivisible_fit.__dict__["MAX_CAPTION_LINES"],
        )
        seen.append((name, *limits, list(sys.argv[1:])))
        return {}

    monkeypatch.setattr(runpy, "run_module", fake_run)
    caption_native_paging_lines.main(
        ["--maximum-lines", "2", "--module", "replay.caption_reflow_verify", "--", "r", "--x"]
    )
    assert seen == [("replay.caption_reflow_verify", 2, 2, ["r", "--x"])]
    assert caption_style_defaults.MAX_CAPTION_LINES == 3
    assert caption_indivisible_fit.__dict__["MAX_CAPTION_LINES"] == 3


def test_modules_imported_inside_are_restored_too(monkeypatch: pytest.MonkeyPatch) -> None:
    def importing(name: str, **_options: object) -> dict[str, object]:
        module = type(sys)("replay.fresh_limit_holder")
        module.__dict__["MAX_CAPTION_LINES"] = caption_style_defaults.MAX_CAPTION_LINES
        sys.modules["replay.fresh_limit_holder"] = module
        return {}

    monkeypatch.setattr(runpy, "run_module", importing)
    monkeypatch.delitem(sys.modules, "replay.fresh_limit_holder", raising=False)
    caption_native_paging_lines.main(["--maximum-lines", "1", "--", "a"])
    assert sys.modules["replay.fresh_limit_holder"].__dict__["MAX_CAPTION_LINES"] == 3
    del sys.modules["replay.fresh_limit_holder"]


def test_the_style_limit_is_restored_when_the_module_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing(name: str, **_options: object) -> dict[str, object]:
        raise ReplayError("paging failed")

    monkeypatch.setattr(runpy, "run_module", failing)
    with pytest.raises(ReplayError, match="paging failed"):
        caption_native_paging_lines.main(["--maximum-lines", "1", "--", "a"])
    assert caption_style_defaults.MAX_CAPTION_LINES == 3


def test_paging_is_the_default_module() -> None:
    assert caption_native_paging_lines.limit_of(["--maximum-lines", "2", "--", "a"]) == (
        2,
        "replay.caption_native_paging",
        ["a"],
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--maximum-lines", "2"], "Usage"),
        (["--lines", "2", "--", "a"], "Usage"),
        (["--maximum-lines", "x", "--", "a"], "Usage"),
        (["--maximum-lines", "2", "--module", "replay.cli", "--", "a"], "Unsupported module"),
        (["--maximum-lines", "4", "--", "a"], "within the style"),
        (["--maximum-lines", "0", "--", "a"], "within the style"),
    ],
)
def test_invalid_requests_are_refused(arguments: list[str], message: str) -> None:
    with pytest.raises(ReplayError, match=message):
        caption_native_paging_lines.limit_of(arguments)
