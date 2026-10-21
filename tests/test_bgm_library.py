"""BGM library: canonical URLs, allowlisted records and staged, decode-checked downloads."""

import json
import shutil
import subprocess
import wave
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import bgm_library
from replay.core import ReplayError

MEASURE = bgm_library.loudness
URL = "https://www.youtube.com/watch?v=CYzeUrG3p9o"
METADATA = {
    "id": "CYzeUrG3p9o",
    "title": "Free BGM / one",
    "channel": "Sharou",
    "channel_url": "https://www.youtube.com/channel/UC1",
    "duration": 1837,
    "timestamp": 1778238018,
    "formats": [{"url": "https://signed.example/secret"}],
}


@pytest.fixture(autouse=True)
def stored_loudness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake media never decodes; the real measurement has its own tests."""
    monkeypatch.setattr(bgm_library, "loudness", lambda _audio: -12.0)


def fake_run(
    *,
    stdout: str = json.dumps(METADATA),
    code: int = 0,
    decode: bool = True,
    audio: tuple[str, ...] = (".source.webm",),
    calls: list[list[str]] | None = None,
) -> object:
    def run(command: Sequence[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(list(command))
        if command[0].endswith("yt-dlp"):
            staging = Path(command[command.index("-o") + 1]).parent
            for name in audio:
                (staging / name).write_bytes(b"a")
            (staging / "Free BGM one [CYzeUrG3p9o].jpg").write_bytes(b"j")
            return subprocess.CompletedProcess(command, code, stdout, "blocked")
        if "-c:a" in command:
            Path(command[-1]).write_bytes(b"lossless")
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "", "" if decode else "bad frame")

    return run


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/CYzeUrG3p9o",
        "https://www.youtube.com/watch?v=CYzeUrG3p9o&t=1",
        "https://www.youtube.com/watch?v=short",
    ],
)
def test_only_canonical_watch_urls_are_accepted(url: str) -> None:
    with pytest.raises(ReplayError):
        bgm_library.video_id(url)


def test_canonical_urls_give_their_id() -> None:
    assert bgm_library.video_id(URL) == "CYzeUrG3p9o"
    assert bgm_library.video_id("https://youtube.com/watch?v=_lF8lvEXwmc") == "_lF8lvEXwmc"


def test_records_keep_only_allowlisted_facts(tmp_path: Path) -> None:
    value = bgm_library.track_record(
        {"id": "CYzeUrG3p9o", "title": "T", "uploader": "U", "duration": True},
        identifier="CYzeUrG3p9o",
        audio=tmp_path / "a.m4a",
        thumbnail=tmp_path / "a.jpg",
    )
    assert value == {
        "version": 1,
        "id": "CYzeUrG3p9o",
        "title": "T",
        "channel": "U",
        "channelURL": None,
        "url": "https://www.youtube.com/watch?v=CYzeUrG3p9o",
        "duration": None,
        "timestamp": None,
        "audio": "a.m4a",
        "thumbnail": "a.jpg",
        "loop": False,
    }
    with pytest.raises(ReplayError, match="does not describe"):
        bgm_library.track_record(
            {"id": "other", "title": "T"},
            identifier="CYzeUrG3p9o",
            audio=tmp_path / "a.m4a",
            thumbnail=tmp_path / "a.jpg",
        )


def test_a_track_is_added_once_and_published_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", fake_run(calls=calls))
    root = tmp_path / "bgm"
    assert bgm_library.add(URL, root=root, ffmpeg="/bin/ffmpeg") == ("CYzeUrG3p9o", "added")
    record = json.loads((root / "CYzeUrG3p9o/track.json").read_text(encoding="utf-8"))
    assert sorted(path.name for path in (root / "CYzeUrG3p9o").iterdir()) == [
        "Free BGM one [CYzeUrG3p9o].jpg", "Free BGM one [CYzeUrG3p9o].m4a", "track.json"
    ]  # fmt: skip
    assert calls[0][calls[0].index("-f") + 1] == "ba"
    assert "--audio-format" not in calls[0]
    assert calls[1][calls[1].index("-c:a") : calls[1].index("-c:a") + 6] == [
        "-c:a", "alac", "-sample_fmt", "s32p", "-bits_per_raw_sample", "24"
    ]  # fmt: skip
    assert "title=Free BGM / one" in calls[1]
    assert "comment=Audio of https://www.youtube.com/watch?v=CYzeUrG3p9o" in calls[1]
    assert record["audio"] == "Free BGM one [CYzeUrG3p9o].m4a"
    assert record["thumbnail"] == "Free BGM one [CYzeUrG3p9o].jpg"
    assert record["channel"] == "Sharou"
    assert record["loudness"] == -12.0
    assert "secret" not in json.dumps(record)
    assert not (root / ".partial-CYzeUrG3p9o").exists()
    assert bgm_library.add(URL, root=root, ffmpeg="/bin/ffmpeg") == ("CYzeUrG3p9o", "present")
    assert bgm_library.add(URL, root=root, ffmpeg="/bin/ffmpeg", replace=True) == (
        "CYzeUrG3p9o", "replaced"
    )  # fmt: skip


@pytest.mark.parametrize(
    ("run", "message"),
    [
        (fake_run(code=1), "yt-dlp failed: blocked"),
        (fake_run(stdout="not json"), "invalid yt-dlp metadata"),
        (fake_run(decode=False), "track audio does not decode"),
        (fake_run(audio=(".source.webm", ".source.m4a")), "Expected one downloaded audio"),
    ],
)
def test_a_failed_download_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run: object, message: str
) -> None:
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ReplayError, match=message):
        bgm_library.add(URL, root=tmp_path, ffmpeg="/bin/ffmpeg")
    assert not (tmp_path / "CYzeUrG3p9o").exists()


def test_an_unexplained_track_directory_is_not_replaced(tmp_path: Path) -> None:
    (tmp_path / "CYzeUrG3p9o").mkdir()
    with pytest.raises(ReplayError, match="without track.json"):
        bgm_library.add(URL, root=tmp_path, ffmpeg="/bin/ffmpeg")


def test_decoding_reads_a_real_audio_file(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    audio = tmp_path / "tone.m4a"
    subprocess.run(
        [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "sine=duration=0.5",
         "-c:a", "aac", str(audio)],
        check=True,
    )  # fmt: skip
    assert bgm_library.decodes(audio, ffmpeg)
    (tmp_path / "broken.m4a").write_bytes(b"not audio")
    assert not bgm_library.decodes(tmp_path / "broken.m4a", ffmpeg)


def test_the_cli_reports_each_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(subprocess, "run", fake_run())
    code = bgm_library.main(
        ["add", URL, "https://youtu.be/x", "--root", str(tmp_path), "--replace"]
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert code == 1
    assert lines[0] == {"url": URL, "id": "CYzeUrG3p9o", "status": "added"}
    assert lines[1]["status"] == "failed"


def loop_wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as sink:
        sink.setnchannels(2)
        sink.setsampwidth(2)
        sink.setframerate(1000)
        sink.writeframes(bytes(4 * 1797_000))
    return path


def fake_loop_run(*, encode: int = 0) -> tuple[object, list[list[str]]]:
    calls: list[list[str]] = []

    def run(command: Sequence[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        if command[0].endswith("yt-dlp"):
            staging = Path(command[command.index("-o") + 1]).parent
            (staging / "Talk [CYzeUrG3p9o].jpg").write_bytes(b"j")
            return subprocess.CompletedProcess(command, 0, json.dumps(METADATA), "")
        if "-c:a" in command:
            Path(command[-1]).write_bytes(b"a")
            return subprocess.CompletedProcess(command, encode, "", "no encoder")
        return subprocess.CompletedProcess(command, 0, "", "")

    return run, calls


def test_a_rebuilt_loop_is_published_with_only_metadata_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run, calls = fake_loop_run()
    monkeypatch.setattr(subprocess, "run", run)
    audio = loop_wav(tmp_path / "loop.wav")

    code = bgm_library.main(
        ["add-loop", URL, str(audio), "--root", str(tmp_path / "bgm"), "--ffmpeg", "/bin/ffmpeg"]
    )

    record = json.loads((tmp_path / "bgm/CYzeUrG3p9o/track.json").read_text(encoding="utf-8"))
    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "url": URL,
        "id": "CYzeUrG3p9o",
        "status": "added",
    }
    assert record["title"] == "Free BGM / one（BGM・30分ループ）"
    assert record["duration"] == 1797.0
    assert record["loop"] is True
    assert record["audio"] == "Talk [CYzeUrG3p9o].m4a"
    assert "--skip-download" in calls[0]
    assert "title=Free BGM / one（BGM・30分ループ）" in calls[1]
    assert "artist=Sharou" in calls[1]
    assert "alac" in calls[1]


def test_a_loop_that_fails_to_encode_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = fake_loop_run(encode=1)
    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(ReplayError, match="encoding failed: no encoder"):
        bgm_library.add_loop(URL, loop_wav(tmp_path / "l.wav"), root=tmp_path, ffmpeg="ffmpeg")
    assert not (tmp_path / "CYzeUrG3p9o").exists()


def test_replace_swaps_a_loop_only_after_the_new_one_is_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = fake_loop_run()
    monkeypatch.setattr(subprocess, "run", run)
    audio = loop_wav(tmp_path / "loop.wav")
    root = tmp_path / "bgm"
    bgm_library.add_loop(URL, audio, root=root, ffmpeg="ffmpeg")
    (root / "CYzeUrG3p9o/old.txt").write_text("old", encoding="utf-8")

    assert bgm_library.add_loop(URL, audio, root=root, ffmpeg="ffmpeg") == (
        "CYzeUrG3p9o",
        "present",
    )
    assert bgm_library.add_loop(URL, audio, root=root, ffmpeg="ffmpeg", replace=True) == (
        "CYzeUrG3p9o", "replaced"
    )  # fmt: skip

    assert sorted(path.name for path in (root / "CYzeUrG3p9o").iterdir()) == [
        "Talk [CYzeUrG3p9o].jpg", "Talk [CYzeUrG3p9o].m4a", "track.json"
    ]  # fmt: skip
    assert not (root / ".replaced-CYzeUrG3p9o").exists()
    assert not (root / ".partial-CYzeUrG3p9o").exists()


def test_the_library_root_comes_from_its_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "bgm-library.json"
    assert bgm_library.library_root(config) == Path("edits/bgm-library")
    config.write_text('{"version": 1, "root": "/Volumes/ssd/bgm-library"}', encoding="utf-8")
    assert bgm_library.library_root(config) == Path("/Volumes/ssd/bgm-library")
    config.write_text('{"version": 1, "root": "bgm"}', encoding="utf-8")
    with pytest.raises(ReplayError, match="absolute root"):
        bgm_library.library_root(config)
    config.unlink()
    config.symlink_to(tmp_path / "elsewhere.json")
    (tmp_path / "elsewhere.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="plain file"):
        bgm_library.library_root(config)
    monkeypatch.setattr(bgm_library, "ROOT_CONFIG", tmp_path / "none.json")
    monkeypatch.setattr(subprocess, "run", fake_run())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "edits").mkdir()
    assert bgm_library.main(["add", URL]) == 0
    assert (tmp_path / "edits/bgm-library/CYzeUrG3p9o/track.json").is_file()


def test_an_unmounted_library_volume_is_never_created(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bgm_library.add(URL, root=tmp_path / "Volumes/ssd/bgm-library", ffmpeg="/bin/ffmpeg")
    assert not (tmp_path / "Volumes").exists()


def test_loudness_is_the_mean_level_of_the_audible_seconds(tmp_path: Path) -> None:
    tone = tmp_path / "tone.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-ac", "1", str(tone)],
        check=True,
    )  # fmt: skip
    silence = tmp_path / "silence.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "2",
         str(silence)],
        check=True,
    )  # fmt: skip
    assert MEASURE(tone, cache=tmp_path / "cache") == -21.1
    assert MEASURE(silence, cache=tmp_path / "cache") is None


def test_the_loudness_command_fills_only_missing_levels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "bgm"
    for identifier, value in [
        ("aaaaaaaaaaa", {"audio": "a.m4a"}),
        ("bbbbbbbbbbb", {"audio": "b.m4a", "loudness": -9.5}),
        ("ccccccccccc", {"audio": "../c.m4a"}),
    ]:
        (root / identifier).mkdir(parents=True)
        (root / identifier / "track.json").write_text(json.dumps(value), encoding="utf-8")

    code = bgm_library.main(["loudness", "--root", str(root)])

    assert code == 1
    assert capsys.readouterr().out.splitlines() == [
        "measured aaaaaaaaaaa -12.0",
        "present bbbbbbbbbbb",
        "failed ccccccccccc (invalid audio name)",
    ]
    assert json.loads((root / "aaaaaaaaaaa/track.json").read_text(encoding="utf-8")) == {
        "audio": "a.m4a",
        "loudness": -12.0,
    }
    assert sorted(path.name for path in (root / "aaaaaaaaaaa").iterdir()) == ["track.json"]
    assert bgm_library.measure_library(root, force=True)[1] == "measured bbbbbbbbbbb -12.0"
