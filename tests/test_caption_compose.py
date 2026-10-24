"""Composition groups cuts within request limits, reuses finished groups and joins once."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from replay import caption_compose as compose
from replay.core import ReplayError
from replay.native_join import PartFile, VideoStream


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
    pieces = compose.split_cut(compose.Cut(0, 8_000, 0), [])
    assert pieces == [
        compose.Cut(0, 3_600, 0),
        compose.Cut(3_600, 7_200, 3_600),
        compose.Cut(7_200, 8_000, 7_200),
    ]


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
    joined: list[tuple[list[PartFile], Path, Path]] = []

    def fake_join(parts: list[PartFile], *, output: Path, audio: Path, ffmpeg: str) -> int:
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


def joinable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packets: int) -> list[list[str]]:
    """Fake probes and an ffmpeg that writes its last argument."""
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_k: object) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"j")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        compose, "verify_parts", lambda _p: VideoStream("hevc", Fraction(1, 30000), 0)
    )
    monkeypatch.setattr(compose, "quicktime_video_tags", lambda _p: ["-tag:v", "hvc1"])
    monkeypatch.setattr(
        compose, "probe_video", lambda _p: VideoStream("hevc", Fraction(1, 30000), packets)
    )
    (tmp_path / "out").mkdir()
    return commands


def test_the_join_reserves_the_index_at_the_front(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = joinable(tmp_path, monkeypatch, 90)
    parts = [PartFile(tmp_path / "g0.mp4", 60), PartFile(tmp_path / "g1.mp4", 30)]
    output = tmp_path / "out/joined.mp4"
    assert compose.join(parts, output=output, audio=tmp_path / "a.m4a", ffmpeg="ff") == 90
    assert output.read_bytes() == b"j"
    assert commands[0][0] == "ff"
    assert commands[0][commands[0].index("-map") :] == [
        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-video_track_timescale", "30000",
        "-tag:v", "hvc1", "-moov_size", "1057216", "-write_btrt", "0", commands[0][-1],
    ]  # fmt: skip
    assert list((tmp_path / "out").iterdir()) == [output]


def test_a_join_missing_frames_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    joinable(tmp_path, monkeypatch, 89)
    output = tmp_path / "out/joined.mp4"
    with pytest.raises(ReplayError, match="does not hold the planned frames"):
        compose.join(
            [PartFile(tmp_path / "g0.mp4", 90)], output=output, audio=tmp_path / "a", ffmpeg="ff"
        )
    assert not output.exists()


def test_a_join_never_overwrites(tmp_path: Path) -> None:
    (tmp_path / "joined.mp4").write_bytes(b"old")
    with pytest.raises(ReplayError, match="overwrite"):
        compose.join([], output=tmp_path / "joined.mp4", audio=tmp_path / "a", ffmpeg="ff")


def probed(monkeypatch: pytest.MonkeyPatch, stdout: str) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_k: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return commands


def test_the_rate_follows_the_source_scaled_to_the_output(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = {"streams": [{"bit_rate": "4849005", "width": 1080, "height": 1920}]}
    commands = probed(monkeypatch, json.dumps(stream))
    assert compose.bit_rate(Path("/o.mp4")) == 12_100_000
    assert commands[0][-1] == "/o.mp4"


def test_a_small_source_is_scaled_up_to_the_output_size(monkeypatch: pytest.MonkeyPatch) -> None:
    probed(
        monkeypatch, json.dumps({"streams": [{"bit_rate": "1000000", "width": 540, "height": 960}]})
    )
    assert compose.bit_rate(Path("/o.mp4")) == 10_000_000


@pytest.mark.parametrize(("rate", "expected"), [("619500", 10_000_000), ("10030467", 20_000_000)])
def test_the_rate_stays_within_its_bounds(
    monkeypatch: pytest.MonkeyPatch, rate: str, expected: int
) -> None:
    probed(
        monkeypatch, json.dumps({"streams": [{"bit_rate": rate, "width": 1080, "height": 1920}]})
    )
    assert compose.bit_rate(Path("/o.mp4")) == expected


@pytest.mark.parametrize(
    "stdout",
    ['{"streams": []}', '{"streams": [{"bit_rate": "N/A", "width": 1, "height": 1}]}', "x"],
)
def test_an_unreadable_source_rate_keeps_the_maximum(
    monkeypatch: pytest.MonkeyPatch, stdout: str
) -> None:
    probed(monkeypatch, stdout)
    assert compose.bit_rate(Path("/o.mp4")) == 20_000_000


def derivation(directory: Path, name: str, source: Path, output: Path) -> Path:
    directory.mkdir(parents=True)
    (directory / name).write_text(json.dumps({"source": str(source), "output": str(output)}))
    return output


def test_the_rate_follows_the_derivation_chain_to_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = tmp_path / "youtube/a.mp4"
    decodable = derivation(
        tmp_path / "decodable", "decodable.json", original, tmp_path / "decodable/a.mp4"
    )
    vertical = derivation(
        tmp_path / "vertical", "vertical.json", decodable, tmp_path / "vertical/a.mp4"
    )
    stream = {"streams": [{"bit_rate": "1693911", "width": 1920, "height": 1080}]}
    commands = probed(monkeypatch, json.dumps(stream))
    assert compose.original(vertical) == original
    assert compose.bit_rate(vertical) == 10_000_000
    assert commands[0][-1] == str(original)


def test_an_underived_source_is_its_own_original(tmp_path: Path) -> None:
    assert compose.original(tmp_path / "a.mp4") == tmp_path / "a.mp4"


@pytest.mark.parametrize(
    "text",
    ['{"output": "OUT"}', '{"source": "/o.mp4", "output": "/elsewhere.mp4"}', "not json", "[]"],
)
def test_an_invalid_derivation_record_keeps_the_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    source = tmp_path / "vertical/a.mp4"
    source.parent.mkdir()
    (source.parent / "vertical.json").write_text(text.replace("OUT", str(source)))
    commands = probed(monkeypatch, "{}")
    assert compose.original(source) is None
    assert compose.bit_rate(source) == 20_000_000
    assert commands == []


def test_two_derivation_records_beside_a_source_are_refused(tmp_path: Path) -> None:
    source = derivation(tmp_path / "d", "vertical.json", tmp_path / "o.mp4", tmp_path / "d/a.mp4")
    (tmp_path / "d/lead-in.json").write_text(
        json.dumps({"source": "/o.mp4", "output": str(source)})
    )
    assert compose.original(source) is None


def test_a_derivation_cycle_is_refused(tmp_path: Path) -> None:
    source = derivation(tmp_path / "d", "vertical.json", tmp_path / "d/a.mp4", tmp_path / "d/a.mp4")
    assert compose.original(source) is None


def test_the_plan_encodes_at_the_rate_of_the_rate_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts = part_files(tmp_path, [("p0", 0, 60)])
    asked: list[Path] = []

    def rate(path: Path) -> int:
        asked.append(path)
        return 8_200_000

    monkeypatch.setattr(compose, "bit_rate", rate)
    requests = tmp_path / "req"
    compose.plan(cut_file(tmp_path, [(0, 60, 0)]), parts, Path("/clean.mov"), tmp_path / "out",
                 requests, packets=size, rate_source=Path("/original.mp4"))  # fmt: skip
    body = json.loads((requests / "g000.json").read_text(encoding="utf-8"))
    assert body["recipe"]["video"]["encoding"] == {
        "codec": "hevc", "averageBitRate": 8_200_000, "allowFrameReordering": False
    }  # fmt: skip
    assert asked == [Path("/original.mp4")]


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
    encoded: list[tuple[Path, Path, str, compose.Loudness | None]] = []
    monkeypatch.setattr(
        compose, "encode_audio",
        lambda t, a, *, ffmpeg, loudness: encoded.append((t, a, ffmpeg, loudness)),
    )  # fmt: skip
    voiced = [*rendered, "--tempo", "/t.wav", "--audio", "/a.m4a", "--ffmpeg", "ff"]
    assert compose.main(voiced) == 0
    levelled = ["--loudness-target", "-18", "--true-peak-ceiling", "-1"]
    assert compose.main([*voiced, *levelled]) == 0
    assert encoded == [
        (Path("/t.wav"), Path("/a.m4a"), "ff", None),
        (Path("/t.wav"), Path("/a.m4a"), "ff", compose.Loudness(target=-18.0, ceiling=-1.0)),
    ]
    with pytest.raises(SystemExit):
        compose.main([*rendered, "--tempo", "/t.wav"])
    with pytest.raises(SystemExit):
        compose.main([*voiced, "--loudness-target", "-18"])
    finished: list[object] = []

    def finish(*_a: object, **keywords: object) -> int:
        finished.append(keywords["loudness"])
        return 60

    monkeypatch.setattr(compose, "finish", finish)
    assert compose.main(["finish", str(tmp_path / "req"), "/t.wav", "/o.mp4"]) == 0
    assert compose.main(["finish", str(tmp_path / "req"), "/t.wav", "/o.mp4", *levelled]) == 0
    assert finished == [None, compose.Loudness(target=-18.0, ceiling=-1.0)]
    assert capsys.readouterr().out.split("\n")[1:5] == [
        '{"g000": "reused"}', '{"g000": "reused"}', '{"g000": "reused"}', "60"
    ]  # fmt: skip


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


def test_a_retry_cuts_caption_layers_at_the_shorter_timeline(tmp_path: Path) -> None:
    requests = tmp_path / "requests"
    requests.mkdir()
    clip = {"sourcePath": "/s.mov", "selection": {"startSeconds": 0.0, "durationSeconds": 3.0}}
    layers = [
        {"sourcePath": "/a.mov", "selection": {"startSeconds": 0.0, "durationSeconds": 2.0},
         "offsetSeconds": 0.0, "opacity": 1},
        {"sourcePath": "/b.mov", "selection": {"startSeconds": 0.0, "durationSeconds": 4.0},
         "offsetSeconds": 2.0, "opacity": 1},
    ]  # fmt: skip
    body = {"recipe": {"clips": [clip, clip], "additionalVideo": layers}}
    (requests / "g001.json").write_text(json.dumps(body), encoding="utf-8")
    retry = json.loads(compose.trimmed(requests, "g001").read_text(encoding="utf-8"))
    durations = [
        layer["selection"]["durationSeconds"] for layer in retry["recipe"]["additionalVideo"]
    ]
    assert durations == [2.0, pytest.approx(4.0 - 4 / 60000)]


POLICY = compose.Loudness(target=-18.0, ceiling=-1.0)


def voice(
    path: Path, *, volume_db: float, burst: int = 4, width: float = 0.02, period: float = 0.5
) -> Path:
    """A 12 s stereo 44.1 kHz voice stand-in: a 330 Hz tone with a ``width`` s burst every
    ``period`` s, ``burst`` times its amplitude, like plosives and laughs over speech."""
    shape = f"if(lt(mod(t\\,{period})\\,{width})\\,{burst}\\,1)"
    source = (f"aevalsrc={0.5 / burst}*sin(2*PI*330*t)*{shape}:s=44100:d=12,"
              f"volume={volume_db}dB")  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", source, "-ac", "2",
         "-c:a", "pcm_f32le", str(path)],
        check=True,
    )  # fmt: skip
    return path


@pytest.mark.parametrize(("volume_db", "lufs"), [(-10.0, -30), (14.0, -6)])
def test_quiet_and_loud_voices_land_at_the_target_below_the_ceiling(
    tmp_path: Path, volume_db: float, lufs: int
) -> None:
    source = voice(tmp_path / "tempo.wav", volume_db=volume_db)
    integrated, peak = compose.measured(source, ffmpeg="ffmpeg")
    assert round(integrated) == lufs
    audio = compose.encode_audio(source, tmp_path / "audio/tempo.m4a", ffmpeg="ffmpeg",
                                 loudness=POLICY)  # fmt: skip
    delivered, delivered_peak = compose.measured(audio, ffmpeg="ffmpeg")
    assert abs(delivered - -18.0) <= 0.5
    assert delivered_peak <= -1.0
    record = json.loads((tmp_path / "audio/tempo.m4a.loudness.json").read_text("utf-8"))
    assert (record["integratedLufs"], record["truePeakDbtp"]) == (integrated, peak)
    assert record["firstGainDb"] == pytest.approx(-18.0 - integrated, abs=0.01)
    assert (record["targetLufs"], record["ceilingDbtp"]) == (-18.0, -1.0)
    assert "alimiter=limit=0.794328" in record["limiter"]


def test_a_quiet_voice_with_peaks_above_full_scale_is_lifted_and_limited(tmp_path: Path) -> None:
    # The flaw of a gain capped by the peak: this voice would have been made quieter.
    source = voice(tmp_path / "tempo.wav", volume_db=12.0, burst=16, width=0.001, period=2.0)
    integrated, peak = compose.measured(source, ffmpeg="ffmpeg")
    assert (integrated < -18.0, peak > 0.0) == (True, True)
    audio = compose.encode_audio(source, tmp_path / "tempo.m4a", ffmpeg="ffmpeg", loudness=POLICY)
    delivered, delivered_peak = compose.measured(audio, ffmpeg="ffmpeg")
    assert abs(delivered - -18.0) <= 1.0
    assert delivered_peak <= -1.0
    record = json.loads((tmp_path / "tempo.m4a.loudness.json").read_text("utf-8"))
    assert record["gainDb"] > 0


def test_the_limiter_chain_is_static_gain_then_an_oversampled_lookahead_limit() -> None:
    assert compose.level_chain(-9.4, POLICY) == (
        "aformat=sample_fmts=flt,volume=-9.40dB,aresample=176400,"
        "alimiter=limit=0.794328:attack=1:release=10:level=disabled:latency=enabled,"
        "aresample=44100"
    )


def test_silence_has_no_measurable_loudness(tmp_path: Path) -> None:
    silent = tmp_path / "silent.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "anullsrc=r=44100:cl=stereo", "-t", "3", str(silent)],
        check=True,
    )  # fmt: skip
    with pytest.raises(ReplayError, match="No measurable voice loudness"):
        compose.encode_audio(silent, tmp_path / "a.m4a", ffmpeg="ffmpeg", loudness=POLICY)
