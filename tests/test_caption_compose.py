"""Composition groups cuts within request limits, reuses finished groups and joins once."""

import json
import subprocess
from pathlib import Path

import pytest

from replay import caption_compose as compose
from replay.core import ReplayError
from replay.native_join import PartFile


def part_files(tmp_path: Path, parts: list[tuple[str, int, int]]) -> Path:
    rows = []
    for name, offset, frames in parts:
        directory = tmp_path / "render" / name
        (directory / "compressor-1").mkdir(parents=True)
        (directory / "compressor-1/motion-alpha.mov").write_bytes(b"m" * frames)
        rows.append(
            {"name": name, "offsetFrame": offset, "frames": frames, "renderDir": str(directory)}
        )
    path = tmp_path / "parts.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def cut_file(tmp_path: Path, cuts: list[tuple[int, int, int]]) -> Path:
    rows = [
        {"source_start_frame": a, "source_end_frame": b, "output_start_frame": o}
        for a, b, o in cuts
    ]
    path = tmp_path / "cuts.json"
    path.write_text(json.dumps({"cuts": rows}), encoding="utf-8")
    return path


def size(path: Path) -> int:
    """Fake packet count: the byte size of a fake render."""
    return path.stat().st_size


def test_plan_writes_requests_with_clips_and_overlapping_layers(tmp_path: Path) -> None:
    parts = part_files(tmp_path, [("p0", 0, 60), ("p1", 90, 30)])
    cuts = cut_file(tmp_path, [(0, 60, 0), (100, 160, 60)])
    groups = compose.plan(
        cuts, parts, Path("/src.mov"), tmp_path / "out", tmp_path / "req", packets=size
    )
    assert json.loads(groups.read_text(encoding="utf-8")) == [
        {"group": 0, "name": "g000", "frames": 120, "directory": str(tmp_path / "out/g000")}
    ]
    body = json.loads((tmp_path / "req/g000.json").read_text(encoding="utf-8"))
    assert body["recipe"]["clips"][1] == {
        "sourcePath": "/src.mov",
        "selection": {"startSeconds": 100.5 / 30, "durationSeconds": 2.0, "rate": 1},
    }
    assert body["recipe"]["additionalVideo"][1] == {
        "sourcePath": str(tmp_path / "render/p1/compressor-1/motion-alpha.mov"),
        "selection": {"startSeconds": 0.0, "durationSeconds": 1.0, "rate": 1},
        "offsetSeconds": 3.0,
        "opacity": 1,
    }


def test_replanning_is_idempotent(tmp_path: Path) -> None:
    parts = part_files(tmp_path, [("p0", 0, 60)])
    cuts = cut_file(tmp_path, [(0, 60, 0)])
    arguments = (cuts, parts, Path("/src.mov"), tmp_path / "out", tmp_path / "req")
    first = compose.plan(*arguments, packets=size)
    assert compose.plan(*arguments, packets=size) == first
    (tmp_path / "req/g000.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="different plan"):
        compose.plan(*arguments, packets=size)


def test_long_cuts_split_at_the_request_duration(tmp_path: Path) -> None:
    pieces = compose.split_cut(compose.Cut(0, 20_000, 0), [])
    assert pieces == [compose.Cut(0, 18_000, 0), compose.Cut(18_000, 20_000, 18_000)]


def test_cuts_split_where_too_many_layers_overlap() -> None:
    parts = [compose.Part(f"p{i}", i * 10, 10, Path("/m")) for i in range(20)]
    stacked = [*parts, *(compose.Part(f"q{i}", 0, 200, Path("/m")) for i in range(15))]
    pieces = compose.split_cut(compose.Cut(0, 200, 0), stacked)
    assert pieces == [compose.Cut(0, 10, 0), compose.Cut(10, 20, 10)] + [
        compose.Cut(n, n + 10, n) for n in range(20, 200, 10)
    ]


def test_an_unsplittable_overlap_is_refused() -> None:
    parts = [compose.Part(f"q{i}", 0, 200, Path("/m")) for i in range(17)]
    with pytest.raises(ReplayError, match="overlap one frame"):
        compose.split_cut(compose.Cut(0, 200, 0), parts)


def test_groups_respect_the_clip_limit() -> None:
    cuts = [compose.Cut(n, n + 1, n) for n in range(61)]
    assert compose.group_bounds(cuts, []) == [(0, 60), (60, 61)]


def test_a_part_without_a_complete_render_is_refused(tmp_path: Path) -> None:
    parts = part_files(tmp_path, [("p0", 0, 60)])
    with pytest.raises(ReplayError, match="No complete render"):
        compose.load_parts(parts, lambda _path: 59)


def test_an_empty_cut_list_is_refused(tmp_path: Path) -> None:
    (tmp_path / "cuts.json").write_text('{"cuts": []}', encoding="utf-8")
    with pytest.raises(ReplayError, match="empty"):
        compose.load_cuts(tmp_path / "cuts.json")


def planned(tmp_path: Path) -> Path:
    parts = part_files(tmp_path, [("p0", 0, 60)])
    cuts = cut_file(tmp_path, [(0, 60, 0)])
    compose.plan(cuts, parts, Path("/src.mov"), tmp_path / "out", tmp_path / "req", packets=size)
    return tmp_path / "req"


def rendering(tmp_path: Path, calls: list[str]) -> compose.Call:
    def call(arguments: Path, response: Path) -> None:
        calls.append(arguments.name)
        output = tmp_path / "out/g000/edit-1/g000.mp4"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"v" * 60)
        response.write_text("{}", encoding="utf-8")

    return call


def test_render_runs_a_group_once_and_reuses_it(tmp_path: Path) -> None:
    requests = planned(tmp_path)
    calls: list[str] = []
    call = rendering(tmp_path, calls)
    assert compose.render(requests, call, workers=2, packets=size) == {"g000": "completed"}
    assert compose.render(requests, call, workers=2, packets=size) == {"g000": "reused"}
    assert calls == ["g000.json"]


def test_a_partial_group_is_cleared_and_rendered_again(tmp_path: Path) -> None:
    requests = planned(tmp_path)
    (tmp_path / "out/g000/edit-0").mkdir(parents=True)
    (tmp_path / "out/g000/edit-0/g000.mp4").write_bytes(b"v" * 10)
    (requests / "g000.response.json").write_text("{}", encoding="utf-8")
    calls: list[str] = []
    assert compose.render(requests, rendering(tmp_path, calls), workers=1, packets=size) == {
        "g000": "completed"
    }
    assert (tmp_path / "out/g000/edit-0").exists() is False


def test_a_group_rendering_wrong_frames_fails(tmp_path: Path) -> None:
    requests = planned(tmp_path)
    with pytest.raises(ReplayError, match="did not render its planned 60 frames"):
        compose.render(requests, lambda _a, _r: None, workers=1, packets=size)


def test_finish_encodes_audio_once_and_joins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests = planned(tmp_path)
    compose.render(requests, rendering(tmp_path, []), workers=1, packets=size)
    joined: list[tuple[list[PartFile], Path, Path | None]] = []

    def fake_join(parts: list[PartFile], *, rate: int, output: Path, audio: Path | None) -> int:
        joined.append((parts, output, audio))
        output.write_bytes(b"j" * 60)
        return 60

    commands: list[list[str]] = []

    def fake_run(command: list[str], **_k: object) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"a")

    monkeypatch.setattr(compose, "join", fake_join)
    monkeypatch.setattr(subprocess, "run", fake_run)
    output = tmp_path / "final/joined.mp4"
    output.parent.mkdir()
    tempo = tmp_path / "tempo.wav"
    assert compose.finish(requests, tempo, output, ffmpeg="ff", packets=size) == 60
    assert compose.finish(requests, tempo, output, ffmpeg="ff", packets=size) == 60
    assert len(joined) == 1
    assert joined[0][2] == tmp_path / "final/audio/tempo.m4a"
    assert commands == [
        ["ff", "-nostdin", "-v", "error", "-i", str(tempo), *compose.AUDIO_ARGUMENTS,
         str(tmp_path / "final/audio/.tempo.partial.m4a")],
    ]  # fmt: skip


def test_finish_refuses_a_missing_group(tmp_path: Path) -> None:
    requests = planned(tmp_path)
    output = tmp_path / "joined.mp4"
    (tmp_path / "audio").mkdir()
    (tmp_path / "audio/tempo.m4a").write_bytes(b"a")
    with pytest.raises(ReplayError, match="g000 is missing"):
        compose.finish(requests, tmp_path / "t.wav", output, ffmpeg="ff", packets=size)


def test_packet_count_of_an_unreadable_file_is_negative(tmp_path: Path) -> None:
    (tmp_path / "x.mp4").write_bytes(b"not video")
    assert compose.packet_count(tmp_path / "x.mp4") == -1


def test_the_media_edit_call_pins_the_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, **_k: commands.append(command))
    compose.media_edit_call(Path("/server"), "a" * 64)(Path("/g.json"), Path("/g.response.json"))
    assert commands[0][3:] == [
        "replay.apple_pro_apps_call", "media_edit", "/g.json", "/g.response.json",
        "--server", "/server", "--server-sha256", "a" * 64, "--timeout", "7200",
    ]  # fmt: skip


def test_the_cli_plans_renders_and_finishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parts = part_files(tmp_path, [("p0", 0, 60)])
    cuts = cut_file(tmp_path, [(0, 60, 0)])
    monkeypatch.setattr(compose, "packet_count", size)
    arguments = ["plan", str(cuts), str(parts), "/src.mov", str(tmp_path / "out"),
                 str(tmp_path / "req")]  # fmt: skip
    assert compose.main(arguments) == 0
    monkeypatch.setattr(compose, "render", lambda *_a, **_k: {"g000": "reused"})
    rendered = ["render", str(tmp_path / "req"), "--server", "/s", "--server-sha256", "a"]
    assert compose.main(rendered) == 0
    monkeypatch.setattr(compose, "finish", lambda *_a, **_k: 60)
    assert compose.main(["finish", str(tmp_path / "req"), "/t.wav", "/o.mp4"]) == 0
    assert capsys.readouterr().out.split("\n")[1:3] == ['{"g000": "reused"}', "60"]


def state_file(tmp_path: Path, status: str, output: Path) -> Path:
    states = tmp_path / "states"
    states.mkdir(exist_ok=True)
    (states / "p0.json").write_text(
        json.dumps({"status": status, "output": str(output)}), encoding="utf-8"
    )
    return states


def test_only_the_recorded_successful_render_is_used(tmp_path: Path) -> None:
    parts = part_files(tmp_path, [("p0", 0, 60)])
    movie = tmp_path / "render/p0/compressor-1/motion-alpha.mov"
    states = state_file(tmp_path, "Successful", movie)
    assert compose.load_parts(parts, size, states)[0].render == movie


def test_a_render_of_a_failed_job_is_refused(tmp_path: Path) -> None:
    parts = part_files(tmp_path, [("p0", 0, 60)])
    states = state_file(tmp_path, "submitted", tmp_path / "render/p0/compressor-1/motion-alpha.mov")
    with pytest.raises(ReplayError, match="No successful Compressor render"):
        compose.load_parts(parts, size, states)


def test_a_recorded_render_with_other_frames_is_refused(tmp_path: Path) -> None:
    parts = part_files(tmp_path, [("p0", 0, 60)])
    states = state_file(
        tmp_path, "Successful", tmp_path / "render/p0/compressor-1/motion-alpha.mov"
    )
    with pytest.raises(ReplayError, match="does not hold its planned frames"):
        compose.load_parts(parts, lambda _path: 59, states)


def test_a_refused_composition_is_retried_once_with_clips_ending_two_ticks_early(
    tmp_path: Path,
) -> None:
    requests = planned(tmp_path)
    calls: list[str] = []
    succeed = rendering(tmp_path, calls)

    def call(arguments: Path, response: Path) -> None:
        if not calls:
            calls.append(arguments.name)
            response.write_text('{"error": "exited 1"}', encoding="utf-8")
            raise subprocess.CalledProcessError(1, ["media_edit"])
        succeed(arguments, response)

    assert compose.render(requests, call, workers=1, packets=size) == {"g000": "completed"}
    assert calls == ["g000.json", "g000.retry.json"]
    original = json.loads((requests / "g000.json").read_text(encoding="utf-8"))
    retry = json.loads((requests / "g000.retry.json").read_text(encoding="utf-8"))
    assert original["recipe"]["clips"][0]["selection"]["durationSeconds"] == 2.0
    assert retry["recipe"]["clips"][0]["selection"]["durationSeconds"] == 2.0 - 2 / 60000
    assert (requests / "g000.failed-response.json").read_text() == '{"error": "exited 1"}'
