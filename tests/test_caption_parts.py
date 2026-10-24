"""Caption parts: windows split at caption gaps, resumable plan export and Compressor jobs."""

import json
import subprocess
import threading
import time
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


def test_speech_tokens_time_each_parts_reveal_from_its_first_frame(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    class Recording(Producers):
        def __call__(self, command: Sequence[str]) -> None:
            commands.append(list(command))
            super().__call__(command)

    tokens = tmp_path / "tokens.json"
    parts.prepare(
        candidate(tmp_path), tmp_path / "out", tmp_path / "render", template=Path("/t.moti"),
        finish=[], preset=Path("/p"), run=Recording(), tokens=tokens,
    )  # fmt: skip
    reveal = next(command for command in commands if "replay.motion_reveal_stages" in command)
    record = tmp_path / "out/moti/w000-part-0000.reveal.json"
    # The window starts on frame 0 and the part 5 frames into it.
    assert reveal[-6:] == ["--tokens", str(tokens), "--offset-frame", "5", "--record", str(record)]


def test_a_palette_restamps_each_finished_part_before_naming_it(tmp_path: Path) -> None:
    stamped = (
        '<x><!--replay-caption-colour-v1 {"intent": {"space": "sRGB", "top": [1, 1, 1], '
        '"bottom": [0, 0, 0]}}--></x>'
    )

    class Stamping(Producers):
        def __call__(self, command: Sequence[str]) -> None:
            super().__call__(command)
            if command[command.index("-m") + 1] == "replay.motion_finish":
                Path(command[command.index("-m") + 3]).write_text(stamped, encoding="utf-8")

    green = {"gradientTop": "#B6DF9E", "gradientBottom": "#6DBE3C"}
    parts.prepare(
        candidate(tmp_path), tmp_path / "out", tmp_path / "render",
        template=Path("/t"), finish=[], preset=Path("/p"), run=Stamping(), palette=green,
    )  # fmt: skip
    final = (tmp_path / "out/moti/w000-part-0000.moti").read_text(encoding="utf-8")
    assert '"intentPolicy": "recording-palette"' in final
    assert sorted(path.name for path in (tmp_path / "out/moti").iterdir()) == [
        "w000-part-0000.moti"
    ]


def test_a_window_is_exported_with_the_part_stage_limit(tmp_path: Path) -> None:
    commands: list[Sequence[str]] = []
    run = Producers()

    def recorded(command: Sequence[str]) -> None:
        commands.append(command)
        run(command)

    arguments = (candidate(tmp_path), tmp_path / "out", tmp_path / "render")
    parts.prepare(*arguments, template=Path("/t"), finish=[], preset=Path("/p"), run=recorded)
    assert list(commands[0][-2:]) == ["--max-stages", "512"]


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
    assert seen["palette"] is None
    parts.main(
        ["c.json", "out", "render", "--template", "t.moti", "--preset", "p",
         "--palette", "#B6DF9E", "#6DBE3C", "--"]
    )  # fmt: skip
    assert seen["palette"] == {"gradientTop": "#B6DF9E", "gradientBottom": "#6DBE3C"}


def test_the_checked_runner_raises_on_failure() -> None:
    with pytest.raises(subprocess.CalledProcessError):
        parts.run_checked(["false"])


def test_windows_run_concurrently_and_keep_their_order(tmp_path: Path) -> None:
    class Overlapping(Producers):
        """Earlier windows finish last; counts the exports running at once."""

        def __init__(self) -> None:
            super().__init__()
            self.lock = threading.Lock()
            self.running = 0
            self.peak = 0

        def __call__(self, command: Sequence[str]) -> None:
            if "replay.motion_measured" in command:
                with self.lock:
                    self.running += 1
                    self.peak = max(self.peak, self.running)
                window = int(Path(command[command.index("-m") + 2]).stem[-3:])
                time.sleep(0.05 * (3 - window))
                with self.lock:
                    self.running -= 1
            with self.lock:
                super().__call__(command)

    path = tmp_path / "candidate.json"
    cues = [cue(0, 1), cue(120, 121), cue(240, 241), cue(360, 361)]
    path.write_text(json.dumps({"version": 1, "candidates": cues}), "utf-8")
    run = Overlapping()
    result = parts.prepare(
        path, tmp_path / "out", tmp_path / "render",
        template=Path("/t.moti"), finish=[], preset=Path("/p"), run=run,
    )  # fmt: skip
    listed = json.loads(result.read_text(encoding="utf-8"))
    assert [part["name"] for part in listed] == [
        "w000-part-0000",
        "w001-part-0000",
        "w002-part-0000",
        "w003-part-0000",
    ]
    assert run.peak > 1


def test_a_finished_part_drops_its_staged_file_and_a_resumed_one_is_untouched(
    tmp_path: Path,
) -> None:
    arguments = (candidate(tmp_path), tmp_path / "out", tmp_path / "render")
    parts.prepare(*arguments, template=Path("/t"), finish=[], preset=Path("/p"), run=Producers())
    moti = tmp_path / "out/moti"
    assert sorted(path.name for path in moti.iterdir()) == ["w000-part-0000.moti"]
    final = moti / "w000-part-0000.moti"
    before = (final.read_bytes(), final.stat().st_mtime_ns)
    again = Producers()
    parts.prepare(*arguments, template=Path("/t"), finish=[], preset=Path("/p"), run=again)
    assert again.modules == []
    assert (final.read_bytes(), final.stat().st_mtime_ns) == before
    assert sorted(path.name for path in moti.iterdir()) == ["w000-part-0000.moti"]
