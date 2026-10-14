import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.youtube_render import load_config, main, render, video_filter


def write_config(tmp_path: Path, **overrides: object) -> Path:
    base = tmp_path / "edits/youtube-abcdefghijk"
    (base / "inputs").mkdir(parents=True)
    (base / "inputs/cues.csv").write_text(
        "start,end,speaker,text\n1,2,left,左です\n1.5,3,middle,中です\n", encoding="utf-8"
    )
    (base / "inputs/cut-frames.json").write_text("[[10, 20], [30, 30]]\n", encoding="utf-8")
    publication = tmp_path / "publication.json"
    publication.write_text(json.dumps({"id": "abcdefghijk", "timestamp": 1790000000}))
    (tmp_path / "out").mkdir()
    config = {
        "version": 1,
        "video_id": "abcdefghijk",
        "source_video": str(tmp_path / "original.mkv"),
        "source_audio": str(tmp_path / "kept.wav"),
        "cut_frames": "inputs/cut-frames.json",
        "cues": "inputs/cues.csv",
        "publication": str(publication),
        "output_root": str(tmp_path / "out"),
        "ffmpeg": "/bin/ffmpeg",
        "ffprobe": "/bin/ffprobe",
        **overrides,
    }
    path = base / "render-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def fake_run(
    streams: list[str], commands: list[list[str]]
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[0] == "/bin/ffprobe":
            body = json.dumps({"streams": [{"codec_type": kind} for kind in streams]})
            return subprocess.CompletedProcess(command, 0, body, "")
        if "-filter_complex" in command:
            Path(command[-1]).write_bytes(b"movie")
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


def test_render_burns_ass_once_with_dated_review_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", fake_run(["video", "audio"], commands))
    output = render(load_config(write_config(tmp_path)))
    stem = "2026-09-21_23-13-20_JST_youtube-abcdefghijk-review"
    assert output == tmp_path / "out" / stem
    assert (output / f"{stem}.mov").read_bytes() == b"movie"
    assert "[左] 左です" in (output / f"{stem}.srt").read_text(encoding="utf-8")
    assert "[" not in (output / f"{stem}.ass").read_text(encoding="utf-8").split("[Events]")[1]
    encode = commands[0]
    assert encode[encode.index("-map") + 3] == "1:a:0"
    assert "-sn" in encode
    assert encode[encode.index("-c:a") + 1] == "copy"
    assert "mov_text" not in encode
    report = json.loads((output / "render.json").read_text(encoding="utf-8"))
    assert report["cut_spans"] == 2
    assert report["cues"] == 2
    assert report["captions"] == "burned once from ASS; no embedded subtitle track"
    with pytest.raises(FileExistsError, match="--replace"):
        render(load_config(tmp_path / "edits/youtube-abcdefghijk/render-config.json"))
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [stem]


def test_replace_keeps_the_old_delivery_until_the_new_render_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", fake_run(["video", "audio"], commands))
    config = load_config(write_config(tmp_path))
    output = render(config)
    stem = "2026-09-21_23-13-20_JST_youtube-abcdefghijk-review"
    (output / f"{stem}.mov").write_bytes(b"old movie")
    monkeypatch.setattr(subprocess, "run", fake_run(["video", "audio", "subtitle"], commands))
    with pytest.raises(ReplayError, match="exactly one video and one audio"):
        render(config, replace=True)
    assert (output / f"{stem}.mov").read_bytes() == b"old movie"
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        f".{stem}.rendering",
        stem,
    ]
    (tmp_path / "out" / f".{stem}.rendering").rename(tmp_path / "failed-render")
    monkeypatch.setattr(subprocess, "run", fake_run(["video", "audio"], commands))
    assert render(config, replace=True) == output
    assert (output / f"{stem}.mov").read_bytes() == b"movie"
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [stem]


def test_render_without_separate_audio_uses_source_audio_in_mp4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", fake_run(["audio", "video"], commands))
    output = render(load_config(write_config(tmp_path, source_audio=None)))
    assert (output / f"{output.name}.mp4").is_file()
    encode = commands[0]
    assert encode.count("-i") == 1
    assert "0:a:0" in encode
    assert encode[encode.index("-c:a") + 1] == "aac"


def test_render_refuses_an_embedded_subtitle_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", fake_run(["video", "audio", "subtitle"], commands))
    with pytest.raises(ReplayError, match="exactly one video and one audio"):
        render(load_config(write_config(tmp_path)))


def test_video_filter_cuts_then_burns_once() -> None:
    assert video_filter((), Path("/a/b.ass")) == "[0:v:0]ass=filename='/a/b.ass'[v]"
    assert video_filter(((1, 2), (5, 9)), Path("/a/x:y.ass")) == (
        "[0:v:0]select='not(between(n,1,2)+between(n,5,9))',setpts=N/(30*TB),"
        "ass=filename='/a/x\\:y.ass'[v]"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extra": 1}, "unknown, missing"),
        ({"version": 2}, "unknown, missing"),
        ({"video_id": ""}, "video ID"),
        ({"source_video": "relative.mkv"}, "absolute path"),
        ({"source_video": "/v/youtube-edits/x-edit-1.mp4"}, "original, not an edited"),
    ],
)
def test_config_rejects_invalid_or_edited_sources(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ReplayError, match=message):
        load_config(write_config(tmp_path, **overrides))


@pytest.mark.parametrize(
    "cuts",
    ['{"a": 1}', "[[5, 3]]", "[[1, 2], [2, 3]]", '[["1", 2]]', "[[1]]"],
)
def test_cut_frames_must_be_ordered_disjoint_pairs(tmp_path: Path, cuts: str) -> None:
    path = write_config(tmp_path)
    (path.parent / "inputs/cut-frames.json").write_text(cuts, encoding="utf-8")
    with pytest.raises(ReplayError, match="Cut frames"):
        load_config(path)


def test_cli_prints_the_new_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", fake_run(["video", "audio"], commands))
    path = str(write_config(tmp_path))
    assert main([path]) == 0
    assert capsys.readouterr().out.strip().endswith("_JST_youtube-abcdefghijk-review")
    assert main([path, "--replace"]) == 0
    assert capsys.readouterr().out.strip().endswith("_JST_youtube-abcdefghijk-review")
