import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "web-viewer" / "tools" / "build_catalog.py"
KASUMI = "UCEmlNV5-edGhNWYDMeGDgWw"
SRT = "1\n00:00:00,000 --> 00:00:01,000\nこんにちは\n"


def _youtube(library: Path, originals: Path, video_id: str, channel: str | None) -> None:
    stamp = "2026-10-22_00-00-20_JST_"
    delivery = library / f"youtube-{video_id}" / "delivery"
    delivery.mkdir(parents=True)
    (delivery / f"{stamp}youtube-{video_id}-review.mp4").write_bytes(b"m")
    (delivery / f"{stamp}youtube-{video_id}-review.srt").write_text(SRT, encoding="utf-8")
    original = originals / f"{stamp}youtube-{video_id}"
    original.mkdir(parents=True)
    (original / f"{stamp}youtube-{video_id}.mp4").write_bytes(b"o")
    if channel is not None:
        (original / "channel.json").write_text(channel, encoding="utf-8")


def test_youtube_videos_carry_their_channel_id(tmp_path: Path) -> None:
    library, originals, out = tmp_path / "lib", tmp_path / "yt", tmp_path / "out"
    _youtube(library, originals, "aaaaaaaaaaa", f'{{"channel_id":"{KASUMI}","channel":"A"}}')
    _youtube(library, originals, "bbbbbbbbbbb", '{"channel_id":"UCshort"}')
    _youtube(library, originals, "ccccccccccc", "not json")
    _youtube(library, originals, "ddddddddddd", None)
    tiktok = library / "replay-1" / "delivery"
    tiktok.mkdir(parents=True)
    (tiktok / "2026-10-01_00-00-00_JST_replay-1-review.mp4").write_bytes(b"m")
    (tiktok / "2026-10-01_00-00-00_JST_replay-1-review.srt").write_text(SRT, encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(library),
            "--out",
            str(out),
            "--no-duration",
            "--streamer",
            "既定",
            "--youtube-originals",
            str(originals),
        ],
        check=True,
        capture_output=True,
    )
    catalog = json.loads((out / "catalog.json").read_text(encoding="utf-8"))
    channels = {
        video["title"]: (video["streamer"], video.get("channelId")) for video in catalog["videos"]
    }
    assert channels == {
        "youtube-aaaaaaaaaaa": ("既定", "UCEmlNV5-edGhNWYDMeGDgWw"),
        "youtube-bbbbbbbbbbb": ("既定", None),
        "youtube-ccccccccccc": ("既定", None),
        "youtube-ddddddddddd": ("既定", None),
        "replay-1": ("既定", None),
    }
    assert catalog["issues"] == [
        "no channel: youtube-bbbbbbbbbbb (default streamer)",
        "no channel: youtube-ccccccccccc (default streamer)",
        "no channel: youtube-ddddddddddd (default streamer)",
    ]


def test_kept_youtube_videos_gain_their_channel_id(tmp_path: Path) -> None:
    library, keep, out = tmp_path / "lib", tmp_path / "keep", tmp_path / "out"
    library.mkdir()
    keep.mkdir()
    original = tmp_path / "yt" / "o"
    original.mkdir(parents=True)
    (original / "channel.json").write_text(
        '{"channel_id":"UCEmlNV5-edGhNWYDMeGDgWw"}', encoding="utf-8"
    )
    published = {
        "videos": [
            {
                "id": "k",
                "title": "youtube-kkkkkkkkkkk",
                "movie": "k.mp4",
                "original": {"localPath": str(original / "o.mp4")},
                "broadcastStartJST": None,
                "size": 1,
            }
        ]
    }
    (keep / "catalog.json").write_text(json.dumps(published), encoding="utf-8")
    (keep / "k.json").write_text('{"id":"k","cues":[]}', encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(library),
            "--out",
            str(out),
            "--no-duration",
            "--streamer",
            "既定",
            "--keep",
            str(keep),
        ],
        check=True,
        capture_output=True,
    )
    catalog = json.loads((out / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["videos"] == [
        {
            "id": "k",
            "title": "youtube-kkkkkkkkkkk",
            "movie": "k.mp4",
            "original": {"localPath": str(original / "o.mp4")},
            "broadcastStartJST": None,
            "size": 1,
            "kept": True,
            "channelId": "UCEmlNV5-edGhNWYDMeGDgWw",
        }
    ]


def test_a_retired_revisions_transcript_does_not_hide_the_playable_delivery(
    tmp_path: Path,
) -> None:
    library, out = tmp_path / "lib", tmp_path / "out"
    stem = "2026-10-08_00-00-00_JST_replay-2-review"
    for revision in ("old", "new"):
        delivery = library / "replay-2" / "revisions" / revision / "delivery"
        delivery.mkdir(parents=True)
        (delivery / f"{stem}.srt").write_text(SRT, encoding="utf-8")
    (library / "replay-2" / "revisions" / "new" / "delivery" / f"{stem}.mp4").write_bytes(b"m")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(library), "--out", str(out),
         "--no-duration", "--streamer", "既定"],
        check=True,
        capture_output=True,
    )  # fmt: skip
    catalog = json.loads((out / "catalog.json").read_text(encoding="utf-8"))
    assert [Path(video["localPath"]).parent.parent.name for video in catalog["videos"]] == ["new"]
    assert catalog["issues"] == []


def test_an_offloaded_delivery_publishes_fresh_data_without_its_movie_path(
    tmp_path: Path,
) -> None:
    library, keep, out = tmp_path / "lib", tmp_path / "keep", tmp_path / "out"
    stem = "2026-10-01_00-00-00_JST_replay-1-review"
    delivery = library / "replay-1" / "delivery"
    delivery.mkdir(parents=True)
    (delivery / f"{stem}.srt").write_text(SRT, encoding="utf-8")
    summaries = delivery / f"{stem}.srt.summaries"
    summaries.mkdir()
    summary = {"version": 1, "overview": "新しい要約",
               "sections": [{"startSeconds": 0, "endSeconds": 1, "text": "挨拶"}]}  # fmt: skip
    (summaries / "current.json").write_text(json.dumps(summary), encoding="utf-8")
    movie = str(delivery / f"{stem}.mp4")
    record = {"version": 1, "files": [{"role": "delivery", "localPath": movie, "bytes": 9}]}
    (library / "replay-1" / "offload.json").write_text(json.dumps(record), encoding="utf-8")
    keep.mkdir()
    published = {"videos": [{"id": stem, "title": "replay-1", "movie": f"{stem}.mp4",
                             "duration": 12.5}]}  # fmt: skip
    (keep / "catalog.json").write_text(json.dumps(published), encoding="utf-8")
    (keep / f"{stem}.json").write_text(f'{{"id":"{stem}","cues":[]}}', encoding="utf-8")
    subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(library), "--out", str(out),
         "--streamer", "既定", "--keep", str(keep)],
        check=True, capture_output=True,
    )  # fmt: skip
    catalog = json.loads((out / "catalog.json").read_text(encoding="utf-8"))
    assert catalog["videos"] == [
        {
            "id": stem, "streamer": "既定", "title": "replay-1",
            "broadcastStartJST": "2026-10-01 00:00:00 JST", "movie": f"{stem}.mp4",
            "size": 9, "original": None, "duration": 12.5,
            "transcriptPath": str(delivery / f"{stem}.srt"), "cueCount": 1,
            "hasSummary": True, "localPath": None,
        }
    ]  # fmt: skip
    assert catalog["issues"] == []
    data = json.loads((out / f"{stem}.json").read_text(encoding="utf-8"))
    assert data["summary"] == {
        "overview": "新しい要約", "sections": [{"start": 0, "end": 1, "text": "挨拶"}]
    }  # fmt: skip
