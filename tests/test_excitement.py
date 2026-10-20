import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

from replay import excitement
from replay.core import ReplayError
from replay.excitement import Comment, Utterance

SRT = """1
00:00:01,000 --> 00:00:02,000
[alice] こん

2
00:00:02,000 --> 00:00:03,000
[alice] こんにちは

3
00:00:20,000 --> 00:00:21,000
やあ
"""

CHAT_LINES = [
    {
        "replayChatItemAction": {
            "videoOffsetTimeMsec": "0",
            "actions": [
                {
                    "addChatItemAction": {
                        "item": {
                            "liveChatViewerEngagementMessageRenderer": {
                                "message": {"runs": [{"text": "system"}]}
                            }
                        }
                    }
                }
            ],
        }
    },
    {
        "replayChatItemAction": {
            "videoOffsetTimeMsec": "1500",
            "actions": [
                {
                    "addChatItemAction": {
                        "item": {
                            "liveChatTextMessageRenderer": {
                                "message": {"runs": [{"text": "わこ"}, {"emoji": {"emojiId": "x"}}]}
                            }
                        }
                    }
                }
            ],
        }
    },
    {
        "replayChatItemAction": {
            "videoOffsetTimeMsec": "2500",
            "actions": [
                {
                    "addChatItemAction": {
                        "item": {
                            "liveChatPaidMessageRenderer": {"message": {"runs": [{"text": "応援"}]}}
                        }
                    }
                }
            ],
        }
    },
    {
        "replayChatItemAction": {
            "videoOffsetTimeMsec": "3000",
            "actions": [{"addChatItemAction": {"item": {"liveChatMembershipItemRenderer": {}}}}],
        }
    },
    {
        "replayChatItemAction": {
            "videoOffsetTimeMsec": "4000",
            "actions": [{"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {}}}}],
        }
    },
    {
        "replayChatItemAction": {
            "videoOffsetTimeMsec": "99000",
            "actions": [
                {
                    "addChatItemAction": {
                        "item": {
                            "liveChatTextMessageRenderer": {"message": {"runs": [{"text": "late"}]}}
                        }
                    }
                }
            ],
        }
    },
    {"clickTrackingParams": "x"},
]


def write_chat(path: Path) -> Path:
    path.write_text("\n".join(json.dumps(line) for line in CHAT_LINES) + "\n\n", encoding="utf-8")
    return path


def make_movie(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=3",
            "-ac",
            "1",
            str(path),
        ],
        check=True,
    )
    return path


def test_read_utterances_merges_cumulative_captions_and_strips_speakers(tmp_path: Path) -> None:
    transcript = tmp_path / "a.srt"
    transcript.write_text(SRT, encoding="utf-8")
    assert excitement.read_utterances(transcript) == [
        Utterance(1.0, 3.0, "こんにちは"),
        Utterance(20.0, 21.0, "やあ"),
    ]


def test_read_utterances_accepts_zero_length_srt_cues(tmp_path: Path) -> None:
    transcript = tmp_path / "a.srt"
    transcript.write_text("1\n00:28:31,166 --> 00:28:31,166\nああ\n", encoding="utf-8")
    assert excitement.read_utterances(transcript) == [Utterance(1711.166, 1711.166, "ああ")]


def test_read_utterances_reads_csv(tmp_path: Path) -> None:
    transcript = tmp_path / "a.csv"
    transcript.write_text(
        "start_seconds,end_seconds,start,end,text\n0.5,1.5,a,b,はい\n\n", encoding="utf-8"
    )
    assert excitement.read_utterances(transcript) == [Utterance(0.5, 1.5, "はい")]


def test_read_utterances_rejects_csv_without_header(tmp_path: Path) -> None:
    transcript = tmp_path / "a.csv"
    transcript.write_text("x,y\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="header"):
        excitement.read_utterances(transcript)


def test_bin_seconds_targets_240_bins_within_bounds() -> None:
    assert [
        excitement.bin_seconds(100),
        excitement.bin_seconds(4800),
        excitement.bin_seconds(15205),
    ] == [10, 20, 60]


def test_speech_characters_spreads_text_over_bins() -> None:
    utterances = [Utterance(5.0, 15.0, "あい うえ"), Utterance(25.0, 25.0, "おか")]
    assert excitement.speech_characters(utterances, width=10, bins=3) == [2.0, 2.0, 2.0]


def test_speech_loudness_averages_spoken_seconds_only() -> None:
    levels = [-10.0, -20.0, -30.0, -40.0]
    assert excitement.speech_loudness(levels, [Utterance(0.5, 2.0, "a")], width=2, bins=2) == [
        -15.0,
        None,
    ]


def test_read_chat_counts_viewer_comments_inside_the_video(tmp_path: Path) -> None:
    chat = write_chat(tmp_path / "chat.json")
    assert excitement.read_chat(chat, duration=60) == [
        Comment(1.5, 3),
        Comment(2.5, 2),
        Comment(4.0, 0),
    ]


def test_ranks_average_ties_and_rank_missing_lowest() -> None:
    assert excitement.ranks([3.0, None, 3.0, 1.0]) == [
        0.8333333333333334,
        0.0,
        0.8333333333333334,
        0.3333333333333333,
    ]


def test_ranks_of_a_constant_measure_are_zero() -> None:
    assert excitement.ranks([2.0, 2.0, 2.0]) == [0.0, 0.0, 0.0]


def test_relative_score_scales_the_peak_to_one() -> None:
    assert excitement.relative_score([0.0, 1.0, 2.0]) == [0.0769, 0.4615, 1.0]


def test_relative_score_of_a_flat_measure_is_zero() -> None:
    assert excitement.relative_score([5.0, 5.0]) == [0.0, 0.0]


def test_peaks_keep_strong_separated_bins() -> None:
    score = [0.9, 1.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.0]
    assert excitement.peaks(score, width=10) == [10.0, 60.0]


def test_peaks_stop_at_zero_scores() -> None:
    assert excitement.peaks([0.0, 0.0], width=10) == []


def test_peaks_cap_the_count() -> None:
    score = [1.0, 0.0, 0.0, 0.0, 0.0] * 6
    assert excitement.peaks(score, width=1) == [0.0, 5.0, 10.0, 15.0, 20.0]


def test_estimate_without_comments_has_speech_only() -> None:
    result = excitement.estimate(
        utterances=[Utterance(0.0, 5.0, "あい")], duration=20.0, levels=None, comments=None
    )
    assert result == {
        "version": 1,
        "binSeconds": 10,
        "duration": 20.0,
        "speech": {
            "score": [1.0, 0.3333],
            "characters": [2.0, 0.0],
            "loudnessDb": None,
            "peaks": [0.0],
        },
        "comments": None,
        "combined": None,
    }


def test_estimate_combines_speech_and_comments() -> None:
    result = excitement.estimate(
        utterances=[Utterance(0.0, 5.0, "あい")],
        duration=20.0,
        levels=[-20.0] * 20,
        comments=[Comment(15.0, 4), Comment(16.0, 2)],
    )
    assert result == {
        "version": 1,
        "binSeconds": 10,
        "duration": 20.0,
        "speech": {
            "score": [1.0, 0.3333],
            "characters": [2.0, 0.0],
            "loudnessDb": [-20.0, None],
            "peaks": [0.0],
        },
        "comments": {
            "score": [0.3333, 1.0],
            "count": [0, 2],
            "characters": [0, 6],
            "peaks": [10.0],
        },
        "combined": {"score": [1.0, 1.0], "peaks": [0.0]},
    }


def test_build_writes_once_and_skips_unchanged_inputs(tmp_path: Path) -> None:
    transcript = tmp_path / "a.srt"
    transcript.write_text(SRT, encoding="utf-8")
    movie = make_movie(tmp_path / "a.m4a")
    chat = write_chat(tmp_path / "chat.json")
    written = excitement.build(
        excitement.Sources(transcript, 30.0, movie, chat), cache=tmp_path / "cache"
    )
    again = excitement.build(
        excitement.Sources(transcript, 30.0, movie, chat), cache=tmp_path / "cache"
    )
    saved = json.loads((tmp_path / "a.srt.excitement.json").read_text(encoding="utf-8"))
    assert [
        written,
        again,
        saved["speech"]["loudnessDb"][0] is not None,
        saved["comments"]["count"],
        saved["inputs"]["version"],
    ] == [
        tmp_path / "a.srt.excitement.json",
        None,
        True,
        [3, 0, 0],
        1,
    ]


def test_build_reports_undecodable_movie(tmp_path: Path) -> None:
    transcript = tmp_path / "a.srt"
    transcript.write_text(SRT, encoding="utf-8")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a movie")
    with pytest.raises(ReplayError, match="ffmpeg failed"):
        excitement.build(excitement.Sources(transcript, 30.0, broken), cache=tmp_path)


def test_fetch_chat_rejects_invalid_id(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="Invalid YouTube"):
        excitement.fetch_chat(youtube_id="../x", cache=tmp_path)


def test_fetch_chat_returns_downloaded_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def download(command: Sequence[str], **_: object) -> None:
        Path(command[-2].replace("%(ext)s", "live_chat.json")).write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", download)
    assert (
        excitement.fetch_chat(youtube_id="t__ULrIUydE", cache=tmp_path / "cache")
        == tmp_path / "cache/chat-youtube-t__ULrIUydE.live_chat.json"
    )


def test_fetch_chat_reports_missing_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_, **__: None)
    with pytest.raises(ReplayError, match="no live chat"):
        excitement.fetch_chat(youtube_id="t__ULrIUydE", cache=tmp_path)


def write_catalog(tmp_path: Path) -> Path:
    good = tmp_path / "2026-10-19_18-30-18_JST_youtube-t__ULrIUydE-review.srt"
    good.write_text(SRT, encoding="utf-8")
    edited = tmp_path / "edited.srt"
    edited.write_text(SRT, encoding="utf-8")
    bad = tmp_path / "bad.srt"
    bad.write_text("garbage", encoding="utf-8")
    cut = tmp_path / "cut.srt"
    cut.write_text(SRT, encoding="utf-8")
    edit = tmp_path / "edits/y_JST_youtube-3MpwpTk1wxs-review"
    edit.mkdir(parents=True)
    (edit / "y_JST_youtube-3MpwpTk1wxs-review.srt").write_text(SRT, encoding="utf-8")
    (edit / "filter.txt").write_text(
        "[0:v]select='between(t,0,1)+between(t,2,10)',setpts=N/FRAME_RATE/TB[v]", encoding="utf-8"
    )
    videos = [
        {
            "id": "2026-10-19_18-30-18_JST_youtube-t__ULrIUydE-review",
            "movie": None,
            "original": {"localPath": str(tmp_path / "missing.mp4")},
            "transcriptPath": str(good),
            "cueCount": 3,
            "duration": 30.0,
        },
        {
            "id": "x_JST_youtube-Lp6Wje9ktVs-review",
            "movie": "e.mp4",
            "localPath": None,
            "original": None,
            "transcriptPath": str(edited),
            "cueCount": 3,
            "duration": 30.0,
        },
        {
            "id": "y_JST_youtube-3MpwpTk1wxs-review",
            "movie": "y.mp4",
            "localPath": None,
            "original": None,
            "transcriptPath": str(cut),
            "cueCount": 3,
            "duration": 30.0,
        },
        {"id": "nodur", "movie": None, "transcriptPath": str(cut), "cueCount": 3},
        {
            "id": "bad",
            "movie": None,
            "original": None,
            "transcriptPath": str(bad),
            "cueCount": 1,
            "duration": 30.0,
        },
        {
            "id": "raw",
            "movie": None,
            "original": None,
            "transcriptPath": str(good),
            "cueCount": 0,
            "duration": 30.0,
        },
    ]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"videos": videos}), encoding="utf-8")
    return catalog


def test_build_catalog_uses_chat_only_on_the_original_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = write_catalog(tmp_path)

    def fetch(*, youtube_id: str, cache: Path) -> Path:
        cache.mkdir(exist_ok=True)
        return write_chat(cache / f"chat-youtube-{youtube_id}.live_chat.json")

    monkeypatch.setattr(excitement, "fetch_chat", fetch)
    options = excitement.CatalogOptions(tmp_path / "cache", fetch=True, edits=tmp_path / "edits")
    lines = excitement.build_catalog(catalog, options=options)
    saved = json.loads((tmp_path / "cut.srt.excitement.json").read_text(encoding="utf-8"))
    assert [lines, saved["comments"]["count"]] == [
        [
            "written 2026-10-19_18-30-18_JST_youtube-t__ULrIUydE-review volume=no comments=yes",
            "written x_JST_youtube-Lp6Wje9ktVs-review volume=no comments=no",
            "written y_JST_youtube-3MpwpTk1wxs-review volume=no comments=edited",
            "failed nodur (Catalog video has no duration)",
            "failed bad (Malformed SRT cue)",
        ],
        # the 99 s comment lies after the last kept range and lands on the edit's end
        [4, 0, 0],
    ]


def test_build_catalog_rejects_a_catalog_without_videos(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="no video list"):
        excitement.build_catalog(catalog, options=excitement.CatalogOptions(tmp_path))


def test_main_builds_one_transcript(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    transcript = tmp_path / "a.srt"
    transcript.write_text(SRT, encoding="utf-8")
    first = excitement.main(
        ["--cache", str(tmp_path), "build", "--transcript", str(transcript), "--duration", "30"]
    )
    second = excitement.main(
        ["--cache", str(tmp_path), "build", "--transcript", str(transcript), "--duration", "30"]
    )
    assert [first, second, capsys.readouterr().out] == [
        0,
        0,
        f"{tmp_path}/a.srt.excitement.json\nunchanged\n",
    ]


def test_main_catalog_fails_when_a_video_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    catalog = write_catalog(tmp_path)
    code = excitement.main(
        ["--cache", str(tmp_path / "cache"), "catalog", "--catalog", str(catalog)]
    )
    assert [code, capsys.readouterr().out] == [
        1,
        "written 2026-10-19_18-30-18_JST_youtube-t__ULrIUydE-review volume=no comments=no\n"
        "written x_JST_youtube-Lp6Wje9ktVs-review volume=no comments=no\n"
        "written y_JST_youtube-3MpwpTk1wxs-review volume=no comments=no\n"
        "failed nodur (Catalog video has no duration)\n"
        "failed bad (Malformed SRT cue)\n",
    ]


def test_cached_levels_decode_each_movie_content_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    movie = make_movie(tmp_path / "a.m4a")
    first = excitement.cached_levels(movie, cache=tmp_path / "cache")
    copy = tmp_path / "moved.m4a"
    copy.write_bytes(movie.read_bytes())

    def no_decode(_: Path) -> list[float]:
        raise AssertionError("decoded again")

    monkeypatch.setattr(excitement, "_pcm_seconds", no_decode)
    assert [len(first), excitement.cached_levels(copy, cache=tmp_path / "cache") == first] == [
        3,
        True,
    ]


def test_cached_levels_redecode_a_damaged_cache_entry(tmp_path: Path) -> None:
    movie = make_movie(tmp_path / "a.m4a")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"levels-{excitement.movie_key(movie)}.json").write_text('"bad"', encoding="utf-8")
    assert len(excitement.cached_levels(movie, cache=cache)) == 3


def test_new_transcript_version_reuses_cached_loudness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = tmp_path / "a.srt"
    transcript.write_text(SRT, encoding="utf-8")
    movie = make_movie(tmp_path / "a.m4a")
    excitement.build(excitement.Sources(transcript, 30.0, movie), cache=tmp_path)
    transcript.write_text(SRT.replace("やあ", "やあやあ"), encoding="utf-8")

    def no_decode(_: Path) -> list[float]:
        raise AssertionError("decoded again")

    monkeypatch.setattr(excitement, "_pcm_seconds", no_decode)
    written = excitement.build(excitement.Sources(transcript, 30.0, movie), cache=tmp_path)
    saved = json.loads(written.read_text(encoding="utf-8")) if written else {}
    assert saved["speech"]["characters"] == [5.0, 0.0, 4.0]


def test_to_edited_clock_moves_comments_through_kept_ranges() -> None:
    comments = [Comment(0.5, 1), Comment(1.5, 2), Comment(2.5, 3), Comment(12.0, 4)]
    assert excitement.to_edited_clock(comments, [(1.0, 2.0), (3.0, 10.0)]) == [
        Comment(0.0, 1),
        Comment(0.5, 2),
        Comment(1.0, 3),
        Comment(8.0, 4),
    ]


def test_read_keep_spans_rejects_a_filter_without_ordered_ranges(tmp_path: Path) -> None:
    cut_filter = tmp_path / "filter.txt"
    cut_filter.write_text("select='between(t,5,6)+between(t,1,2)'", encoding="utf-8")
    with pytest.raises(ReplayError, match="ordered kept ranges"):
        excitement.read_keep_spans(cut_filter)


def test_edit_cut_filter_requires_the_same_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "a.srt"
    transcript.write_text(SRT, encoding="utf-8")
    edit = tmp_path / "edits/v"
    edit.mkdir(parents=True)
    (edit / "filter.txt").write_text("select='between(t,0,1)'", encoding="utf-8")
    (edit / "v.srt").write_text("other version", encoding="utf-8")
    assert [
        excitement.edit_cut_filter(None, video_id="v", transcript=transcript),
        excitement.edit_cut_filter(tmp_path / "edits", video_id="v", transcript=transcript),
    ] == [None, None]


def test_main_builds_with_a_cut_filter(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    transcript = tmp_path / "a.srt"
    transcript.write_text(SRT, encoding="utf-8")
    chat = write_chat(tmp_path / "chat.json")
    cut_filter = tmp_path / "filter.txt"
    cut_filter.write_text("select='between(t,2,40)'", encoding="utf-8")
    code = excitement.main(
        ["--cache", str(tmp_path), "build", "--transcript", str(transcript), "--duration", "30"]
        + ["--chat", str(chat), "--cut-filter", str(cut_filter)]
    )
    saved = json.loads((tmp_path / "a.srt.excitement.json").read_text(encoding="utf-8"))
    assert [code, saved["comments"]["count"], capsys.readouterr().out] == [
        0,
        [3, 0, 0],
        f"{tmp_path}/a.srt.excitement.json\n",
    ]
