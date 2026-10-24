"""Whole-movie decode split at a keyframe: as strict as one pass, plus a frame count."""

import json
import subprocess
from pathlib import Path

import pytest

from replay import movie_decode


def movie(path: Path, *, b_frames: int = 0, audio: bool = True, gop: int = 30) -> Path:
    """A 4 s, 30 fps H.264 movie (120 frames) with a keyframe every ``gop`` frames."""
    inputs = ["-f", "lavfi", "-i", "testsrc2=s=320x240:r=30"]
    if audio:
        inputs += ["-f", "lavfi", "-i", "sine=f=440:sample_rate=48000"]
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", *inputs, "-t", "4", "-c:v", "libx264",
         "-bf", str(b_frames), "-g", str(gop), "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )  # fmt: skip
    return path


def corrupt(path: Path, seconds: float) -> Path:
    """Scramble the payload of the video packet nearest ``seconds``."""
    listed = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time,pos,size", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    packets = json.loads(listed)["packets"]
    packet = min(packets, key=lambda item: abs(float(item["pts_time"]) - seconds))
    start, size = int(packet["pos"]), int(packet["size"])
    data = bytearray(path.read_bytes())
    data[start + 8 : start + size - 4] = bytes(
        byte ^ 0x5A for byte in data[start + 8 : start + size - 4]
    )
    path.write_bytes(bytes(data))
    return path


def test_a_clean_movie_splits_at_its_middle_keyframe_and_counts_every_frame(
    tmp_path: Path,
) -> None:
    clean = movie(tmp_path / "clean.mp4")
    assert movie_decode.split_point(clean, "ffprobe") == "2.000000"
    assert movie_decode.decodes(clean, ffmpeg="ffmpeg", ffprobe="ffprobe", frames=120) is True


def test_a_frame_missing_or_decoded_twice_at_the_seam_fails(tmp_path: Path) -> None:
    clean = movie(tmp_path / "clean.mp4")
    assert movie_decode.decodes(clean, ffmpeg="ffmpeg", ffprobe="ffprobe", frames=121) is False
    assert movie_decode.decodes(clean, ffmpeg="ffmpeg", ffprobe="ffprobe", frames=119) is False


def test_a_corrupt_packet_in_the_first_segment_fails(tmp_path: Path) -> None:
    broken = corrupt(movie(tmp_path / "a.mp4"), 0.5)
    assert movie_decode.decodes(broken, ffmpeg="ffmpeg", ffprobe="ffprobe", frames=120) is False


def test_a_corrupt_packet_in_the_second_segment_fails(tmp_path: Path) -> None:
    broken = corrupt(movie(tmp_path / "b.mp4"), 3.0)
    assert movie_decode.decodes(broken, ffmpeg="ffmpeg", ffprobe="ffprobe", frames=120) is False


def test_reordered_frames_or_no_audio_take_the_single_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reordered = movie(tmp_path / "bframes.mp4", b_frames=3)
    silent = movie(tmp_path / "silent.mp4", audio=False)
    assert movie_decode.split_point(reordered, "ffprobe") is None
    assert movie_decode.split_point(silent, "ffprobe") is None
    passes: list[Path] = []

    def single_pass(path: Path, _ffmpeg: str) -> bool:
        passes.append(path)
        return True

    monkeypatch.setattr(movie_decode, "single_pass", single_pass)
    assert movie_decode.decodes(reordered, ffmpeg="ffmpeg", ffprobe="ffprobe", frames=120) is True
    assert passes == [reordered]


def test_the_single_pass_reports_an_undecodable_file(tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a movie")
    assert movie_decode.single_pass(junk, "ffmpeg") is False
    assert movie_decode.single_pass(movie(tmp_path / "ok.mp4"), "ffmpeg") is True


def probed(stream: dict[str, object], packets: list[dict[str, object]]) -> object:
    def probe(_movie: Path, _ffprobe: str, streams: str, *arguments: str) -> dict[str, object]:
        if streams == "a:0":
            return {"streams": [{"index": 1}]}
        return {"packets": packets} if "-read_intervals" in arguments else {"streams": [stream]}

    return probe


def test_no_keyframe_or_reordered_packets_in_the_window_refuse_the_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = {"time_base": "1/15360", "has_b_frames": 0, "duration": "10.0"}
    monkeypatch.setattr(movie_decode, "_probe", probed(stream, [{"pts": 512, "flags": "__"}]))
    assert movie_decode.split_point(Path("m.mp4"), "ffprobe") is None
    monkeypatch.setattr(
        movie_decode,
        "_probe",
        probed(stream, [{"pts": 1024, "flags": "K_"}, {"pts": 512, "flags": "__"}]),
    )
    assert movie_decode.split_point(Path("m.mp4"), "ffprobe") is None
    monkeypatch.setattr(movie_decode, "_probe", probed(stream, [{"pts": 76800, "flags": "K_"}]))
    assert movie_decode.split_point(Path("m.mp4"), "ffprobe") == "5.000000"
    monkeypatch.setattr(movie_decode, "_probe", probed({"has_b_frames": 0}, []))
    assert movie_decode.split_point(Path("m.mp4"), "ffprobe") is None


class Child:
    """A finished process as ``subprocess.Popen`` reports it."""

    def __init__(self, code: int, out: str, err: str) -> None:
        self.returncode = code
        self.result = (out, err)

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        return self.result


def test_an_audio_error_or_a_missing_count_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(movie_decode, "split_point", lambda _m, _p: "1.000000")
    children = iter(
        [Child(0, "frame=30\n", ""), Child(0, "frame=60\n", ""), Child(0, "", "aac error")]
    )
    monkeypatch.setattr(movie_decode, "_start", lambda _command: next(children))
    assert movie_decode.decodes(Path("m.mp4"), ffmpeg="f", ffprobe="p", frames=90) is False
    children = iter([Child(0, "frame=30\n", ""), Child(0, "progress=end\n", ""), Child(0, "", "")])
    monkeypatch.setattr(movie_decode, "_start", lambda _command: next(children))
    assert movie_decode.decodes(Path("m.mp4"), ffmpeg="f", ffprobe="p", frames=90) is False


def test_a_segment_that_times_out_is_killed_and_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(movie_decode, "TIMEOUT_SECONDS", 0.1)
    child = movie_decode._start(["sleep", "5"])
    assert movie_decode._finish(child) == (False, "")
    assert child.returncode == -9


def test_the_second_segment_starts_exactly_at_a_keyframe_off_the_microsecond_grid(
    tmp_path: Path,
) -> None:
    # Keyframes every 31 frames sit at 31/30 s, 62/30 s ...: not whole microseconds.
    odd = movie(tmp_path / "odd.mp4", gop=31)
    split = movie_decode.split_point(odd, "ffprobe")
    assert split == "1.033333"
    before, after, _audio = movie_decode.commands(odd, "ffmpeg", "1.033333")
    first = subprocess.run(before, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(after, check=True, capture_output=True, text=True).stdout
    # Frame 31 (the keyframe) opens the second segment: neither one frame early nor late.
    assert movie_decode.decoded_frames(first) == 31
    assert movie_decode.decoded_frames(second) == 89
    shown = subprocess.run(
        ["ffmpeg", "-nostdin", "-copyts", "-ss", "1.033333", "-i", str(odd), "-map", "0:v:0",
         "-frames:v", "1", "-vf", "showinfo", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    ).stderr  # fmt: skip
    assert "n:   0 pts:  15872 pts_time:1.033333 " in shown
    assert movie_decode.decodes(odd, ffmpeg="ffmpeg", ffprobe="ffprobe", frames=120) is True
