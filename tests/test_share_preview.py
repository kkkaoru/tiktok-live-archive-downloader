import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import share_preview
from replay.core import ReplayError
from replay.share_preview import Preview


def make_movie(path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=180x320:duration=2"]
        + ["-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def write_estimate(transcript: Path, estimate: object) -> None:
    transcript.write_text("", encoding="utf-8")
    sidecar = transcript.with_name(transcript.name + ".excitement.json")
    sidecar.write_text(json.dumps(estimate), encoding="utf-8")


def test_peak_seconds_uses_the_combined_peak(tmp_path: Path) -> None:
    transcript = tmp_path / "a.srt"
    estimate = {"duration": 100, "binSeconds": 10, "combined": {"score": [0.1, 1.0, 0.5]}}
    write_estimate(transcript, estimate)
    assert share_preview.peak_seconds(transcript, duration=100.0) == 15.0


def test_peak_seconds_uses_speech_without_comments(tmp_path: Path) -> None:
    transcript = tmp_path / "a.srt"
    estimate = {"duration": 100, "binSeconds": 10, "combined": None, "speech": {"score": [1.0]}}
    write_estimate(transcript, estimate)
    assert share_preview.peak_seconds(transcript, duration=100.0) == 5.0


def test_peak_seconds_ignores_an_estimate_on_another_clock(tmp_path: Path) -> None:
    transcript = tmp_path / "a.srt"
    estimate = {"duration": 90, "binSeconds": 10, "speech": {"score": [1.0]}}
    write_estimate(transcript, estimate)
    assert share_preview.peak_seconds(transcript, duration=100.0) == 30.0


def test_peak_seconds_falls_back_without_scores(tmp_path: Path) -> None:
    transcript = tmp_path / "a.srt"
    write_estimate(transcript, {"duration": 100, "binSeconds": 10, "speech": {"score": []}})
    assert [
        share_preview.peak_seconds(transcript, duration=100.0),
        share_preview.peak_seconds(None, duration=50.0),
    ] == [30.0, 15.0]


def write_catalog(tmp_path: Path, videos: object) -> Path:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"videos": videos}), encoding="utf-8")
    return catalog


def test_previews_pick_the_local_playback_movie(tmp_path: Path) -> None:
    edited = tmp_path / "e.mp4"
    edited.write_bytes(b"e")
    original = tmp_path / "o.mp4"
    original.write_bytes(b"o")
    catalog = write_catalog(
        tmp_path,
        [
            {"id": "a", "movie": "e.mp4", "localPath": str(edited), "duration": 10},
            {"id": "b", "movie": None, "original": {"localPath": str(original)}, "duration": 20},
            {"id": "c", "movie": None, "original": None, "duration": 20},
            {"id": "d", "movie": "x.mp4", "localPath": str(tmp_path / "gone.mp4"), "duration": 9},
        ],
    )
    assert share_preview.previews(catalog) == [
        Preview("a", edited, 3.0),
        Preview("b", original, 6.0),
    ]


def test_previews_reject_bad_catalogs(tmp_path: Path) -> None:
    movie = tmp_path / "e.mp4"
    movie.write_bytes(b"e")
    bad_id = write_catalog(
        tmp_path, [{"id": "../x", "movie": "e", "localPath": str(movie), "duration": 1}]
    )
    with pytest.raises(ReplayError, match="invalid video id"):
        share_preview.previews(bad_id)
    no_list = tmp_path / "empty.json"
    no_list.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="no video list"):
        share_preview.previews(no_list)


def test_render_makes_one_card_per_movie_and_time(tmp_path: Path) -> None:
    movie = make_movie(tmp_path / "v.mp4")
    preview = Preview("v", movie, 1.0)
    card = share_preview.render(preview, cache=tmp_path / "cache", runner=share_preview._run)
    again = share_preview.render(preview, cache=tmp_path / "cache", runner=lambda _: 1)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0"]
        + [str(card)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert [again == card, probe.stdout.strip()] == [True, "1200,630"]


def test_render_reports_a_failed_frame(tmp_path: Path) -> None:
    movie = tmp_path / "v.mp4"
    movie.write_bytes(b"not a movie")
    with pytest.raises(ReplayError, match="Could not render a preview for v"):
        share_preview.render(Preview("v", movie, 1.0), cache=tmp_path, runner=lambda _: 1)


def publish_fixture(tmp_path: Path) -> Path:
    movie = make_movie(tmp_path / "v.mp4")
    return write_catalog(
        tmp_path, [{"id": "v", "movie": "v.mp4", "localPath": str(movie), "duration": 2}]
    )


def test_publish_uploads_only_changed_cards(tmp_path: Path) -> None:
    catalog = publish_fixture(tmp_path)
    calls: list[Sequence[str]] = []

    def runner(argv: Sequence[str]) -> int:
        calls.append(argv)
        return share_preview._run(argv) if argv[0] == "ffmpeg" else 0

    cache = tmp_path / "cache"
    dry = share_preview.publish(catalog, bucket="b", cache=cache, runner=runner, dry_run=True)
    first = share_preview.publish(catalog, bucket="b", cache=cache, runner=runner, dry_run=False)
    second = share_preview.publish(catalog, bucket="b", cache=cache, runner=runner, dry_run=False)
    uploads = [call[:5] for call in calls if call[0] == "cf"]
    assert [dry, first, second, uploads] == [
        ["would upload v"],
        ["uploaded v"],
        ["unchanged v"],
        [["cf", "r2", "objects", "put", "og/v.jpg"]],
    ]


def test_publish_reports_a_failed_upload(tmp_path: Path) -> None:
    catalog = publish_fixture(tmp_path)

    def runner(argv: Sequence[str]) -> int:
        return share_preview._run(argv) if argv[0] == "ffmpeg" else 1

    with pytest.raises(ReplayError, match="Upload failed for og/v.jpg"):
        share_preview.publish(
            catalog, bucket="b", cache=tmp_path / "cache", runner=runner, dry_run=False
        )


def test_main_prints_one_line_per_video(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    catalog = publish_fixture(tmp_path)
    code = share_preview.main(
        ["--catalog", str(catalog), "--bucket", "b", "--cache", str(tmp_path / "c"), "--dry-run"]
    )
    assert [code, capsys.readouterr().out] == [0, "would upload v\n"]
