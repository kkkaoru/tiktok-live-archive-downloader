"""Caption parts: windows split at caption gaps, resumable plan export and Compressor jobs."""

import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import caption_parts as parts
from replay.core import ReplayError


def cue(start: float, end: float, text: str = "a") -> dict[str, object]:
    return {"text": text, "cue": {"start": start, "end": end}}


def test_short_captions_need_no_boundary() -> None:
    assert parts.boundaries([cue(0, 1), cue(2, 3)], 90) == []


def test_the_latest_gap_inside_the_limit_is_chosen() -> None:
    cues = [cue(0, 50), cue(60, 100), cue(105, 150)]
    assert parts.boundaries(cues, 150 * 30) == [3150]


def test_a_gap_shorter_than_two_frames_is_not_a_boundary() -> None:
    cues = [cue(0, 100), cue(100.01, 150)]
    with pytest.raises(ReplayError, match="No caption gap"):
        parts.boundaries(cues, 150 * 30)


def test_windows_hold_every_cue_once_with_their_frame_range() -> None:
    cues = [cue(0, 50), cue(60, 100), cue(105, 150)]
    found = parts.windows(cues)
    assert [(w.index, w.start, w.end, len(w.cues)) for w in found] == [
        (0, 0, 3150, 2),
        (1, 3150, 4501, 1),
    ]


def test_a_cue_crossing_a_window_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parts, "boundaries", lambda _cues, _total: [30])
    with pytest.raises(ReplayError, match="crosses the window boundary"):
        parts.windows([cue(0.5, 2)])


def test_a_shifted_window_has_relative_cue_times_and_no_planned_runs() -> None:
    window = parts.Window(1, 300, 600, (cue(12, 14),))
    document = {"version": 1, "candidates": [], "plannedRuns": {}}
    assert parts.shifted(document, window) == {
        "version": 1,
        "candidates": [{"text": "a", "cue": {"start": 2.0, "end": 4.0}}],
    }


def test_writing_once_accepts_identical_content_only(tmp_path: Path) -> None:
    path = tmp_path / "a.json"
    parts.write_once(path, "x")
    parts.write_once(path, "x")
    with pytest.raises(ReplayError, match="different content"):
        parts.write_once(path, "y")


class Producers:
    """Writes what motion_measured, motion_stages and motion_finish would."""

    def __init__(self) -> None:
        self.modules: list[str] = []

    def __call__(self, command: Sequence[str]) -> None:
        name = command[command.index("-m") + 1]
        self.modules.append(name)
        arguments = list(command[command.index("-m") + 2 :])
        if name == "replay.motion_measured":
            plans = Path(arguments[1])
            plans.mkdir()
            listed = [{"offsetFrame": 5, "frames": 40, "stages": 2}]
            (plans / "manifest.json").write_text(json.dumps({"parts": listed}), "utf-8")
        elif name == "replay.motion_reveal_stages":
            Path(arguments[2]).write_text(name, encoding="utf-8")
        else:
            Path(arguments[1]).write_text(name, encoding="utf-8")


def candidate(tmp_path: Path) -> Path:
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({"version": 1, "candidates": [cue(1, 2)]}), "utf-8")
    return path


def test_prepare_writes_parts_and_jobs_and_reuses_them(tmp_path: Path) -> None:
    run = Producers()
    arguments = (candidate(tmp_path), tmp_path / "out", tmp_path / "render")
    template, preset = Path("/t.moti"), Path("/p")
    result = parts.prepare(*arguments, template=template, finish=["--x"], preset=preset, run=run)
    assert run.modules == [
        "replay.motion_measured",
        "replay.motion_reveal_stages",
        "replay.motion_finish",
    ]
    assert json.loads(result.read_text(encoding="utf-8")) == [
        {
            "name": "w000-part-0000",
            "window": 0,
            "offsetFrame": 5,
            "frames": 40,
            "stages": 2,
            "moti": str(tmp_path / "out/moti/w000-part-0000.moti"),
            "renderDir": str(tmp_path / "render/w000-part-0000"),
        }
    ]
    jobs = json.loads((tmp_path / "out/jobs.json").read_text(encoding="utf-8"))
    assert jobs == {
        "jobs": [
            {
                "name": "w000-part-0000",
                "source": str(tmp_path / "out/moti/w000-part-0000.moti"),
                "preset": "/p",
                "output": str(
                    tmp_path / "render/w000-part-0000/compressor-batch-v1/motion-alpha.mov"
                ),
            }
        ]
    }
    again = Producers()
    parts.prepare(*arguments, template=template, finish=["--x"], preset=preset, run=again)
    assert again.modules == []


def test_an_export_cut_short_is_exported_again(tmp_path: Path) -> None:
    (tmp_path / "out/plans-w000").mkdir(parents=True)
    (tmp_path / "out/plans-w000/part-0000.json").write_text("{}", encoding="utf-8")
    run = Producers()
    parts.prepare(
        candidate(tmp_path), tmp_path / "out", tmp_path / "render",
        template=Path("/t"), finish=[], preset=Path("/p"), run=run,
    )  # fmt: skip
    assert run.modules[0] == "replay.motion_measured"


def test_a_manifest_without_parts_is_refused(tmp_path: Path) -> None:
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / "manifest.json").write_text("{}", encoding="utf-8")
    window = parts.Window(0, 0, 30, ())
    with pytest.raises(ReplayError, match="lists no parts"):
        parts.export_plans(window, tmp_path / "w.json", plans, Producers())


def test_a_candidate_without_cues_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text('{"candidates": []}', encoding="utf-8")
    with pytest.raises(ReplayError, match="no cues"):
        parts.prepare(
            path, tmp_path / "o", tmp_path / "r",
            template=Path("/t"), finish=[], preset=Path("/p"), run=Producers(),
        )  # fmt: skip


def test_the_cli_passes_finish_arguments_after_the_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: dict[str, object] = {}

    def fake(*arguments: object, **options: object) -> Path:
        seen.update(options)
        return tmp_path / "parts.json"

    monkeypatch.setattr(parts, "prepare", fake)
    code = parts.main(
        ["c.json", "out", "render", "--template", "t.moti", "--preset", "p", "--",
         "--font-size", "82"]
    )  # fmt: skip
    assert code == 0
    assert seen["finish"] == ["--font-size", "82"]
    assert capsys.readouterr().out.strip() == str(tmp_path / "parts.json")


def test_the_checked_runner_raises_on_failure() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        parts.run_checked(["false"])
