import json
from pathlib import Path

import pytest

from replay.caption_timing import SpokenToken
from replay.core import ReplayError
from replay.speaker_subtitles import SpeakerCue
from replay.youtube_caption_align import align, main, split_at_words


def test_reviewed_text_is_cut_at_recognized_word_clocks() -> None:
    words = [
        SpokenToken("今日は", 1.0, 1.5, "u"),
        SpokenToken("いい", 1.5, 1.8, "u"),
        SpokenToken("天気", 1.8, 2.4, "u"),
    ]
    assert split_at_words("明日は雨です", words) is None
    assert split_at_words("今日はいい天気", words) == [
        SpokenToken("今日は", 1.0, 1.5, "u"),
        SpokenToken("いい", 1.5, 1.8, "u"),
        SpokenToken("天気", 1.8, 2.4, "u"),
    ]


def test_small_corrections_keep_the_reviewed_text() -> None:
    words = [
        SpokenToken("ウルフィンが", 1.0, 1.6, "u"),
        SpokenToken("結構", 1.6, 2.0, "u"),
        SpokenToken("やばんだ", 2.0, 2.5, "u"),
        SpokenToken("よね", 2.5, 2.8, "u"),
    ]
    pieces = split_at_words("ウルフィンが結構やばいんだよね", words)
    assert pieces is not None
    assert "".join(piece.text for piece in pieces) == "ウルフィンが結構やばいんだよね"
    assert [piece.start for piece in pieces] == [1.0, 1.6, 2.0, 2.5]


def test_empty_inputs_do_not_align() -> None:
    assert split_at_words("", [SpokenToken("a", 0, 1, "u")]) is None
    assert split_at_words("text", []) is None


def test_inserted_word_inside_the_match_extends_the_previous_piece() -> None:
    words = [
        SpokenToken("ありがとう", 1.0, 1.6, "u"),
        SpokenToken("x", 1.6, 1.7, "u"),
        SpokenToken("ございます", 1.7, 2.2, "u"),
        SpokenToken("ん", 2.2, 2.3, "u"),
    ]
    assert split_at_words("ありがとうございます", words) == [
        SpokenToken("ありがとう", 1.0, 1.7, "u"),
        SpokenToken("ございます", 1.7, 2.2, "u"),
    ]


def test_aligned_cues_grow_and_unaligned_cues_stay_whole() -> None:
    cues = (
        SpeakerCue(1.0, 2.4, "unassigned", "今日はいい天気"),
        SpeakerCue(5.0, 9.0, "unassigned", "全く別の文章です"),
    )
    words = (
        SpokenToken("今日は", 1.0, 1.5, "x"),
        SpokenToken("いい", 1.5, 1.8, "x"),
        SpokenToken("天気", 1.8, 2.4, "x"),
        SpokenToken("雑音", 5.0, 5.5, "x"),
    )
    result, aligned = align(cues, words)
    assert aligned == 1
    assert [cue.text for cue in result] == [
        "今日は",
        "今日はいい",
        "今日はいい天気",
        "全く別の文章です",
    ]
    assert result[-1] == SpeakerCue(5.0, 9.0, "unassigned", "全く別の文章です")


def test_cue_whose_pieces_exceed_the_display_budget_stays_whole() -> None:
    long = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ"
    cues = (SpeakerCue(1.0, 3.0, "unassigned", long),)
    words = (SpokenToken(long, 1.0, 3.0, "x"),)
    assert align(cues, words) == (cues, 0)


def test_cli_reports_alignment(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "cues.csv").write_text(
        "start,end,speaker,text\n1,2,unassigned,こんにちは\n", encoding="utf-8"
    )
    (tmp_path / "words.json").write_text(
        json.dumps(
            {
                "language": "ja",
                "text": "こんにちはね",
                "segments": [
                    {
                        "start": 1,
                        "end": 2,
                        "text": "こんにちはね",
                        "words": [
                            {"word": "こんにちは", "start": 1.0, "end": 1.8, "probability": 0.9},
                            {"word": "ね", "start": 1.9, "end": 1.9, "probability": 0.9},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out.csv"
    arguments = [str(tmp_path / "cues.csv"), str(tmp_path / "words.json"), "10", str(output)]
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {"reviewed": 1, "aligned": 1, "cues": 1}
    assert output.read_text(encoding="utf-8") == (
        "start,end,speaker,text\n1.000,1.880,unassigned,こんにちは\n"
    )
    with pytest.raises(ReplayError, match="overwrite"):
        main(arguments)


def test_sub_millisecond_align_captions_are_not_emitted() -> None:
    cues = (SpeakerCue(1.0, 2.0, "unassigned", "あいう"),)
    words = (SpokenToken("あ", 1.0, 1.5, "x"), SpokenToken("いう", 1.0004, 2.0, "x"))
    result, aligned = align(cues, words)
    assert aligned == 1
    assert result == (SpeakerCue(1.0004, 2.08, "unassigned", "あいう"),)


def test_neighbouring_words_are_trimmed_and_word_fragments_rejoin() -> None:
    words = [
        SpokenToken("前の", 0.5, 0.9, "u"),
        SpokenToken("ゲ", 1.0, 1.2, "u"),
        SpokenToken("ーム", 1.2, 1.5, "u"),
        SpokenToken("たのしい", 1.5, 2.0, "u"),
        SpokenToken("次の", 2.1, 2.4, "u"),
    ]
    assert split_at_words("ゲームたのしい", words) == [
        SpokenToken("ゲーム", 1.0, 1.5, "u"),
        SpokenToken("たのしい", 1.5, 2.0, "u"),
    ]
    assert split_at_words("全然ちがう", [SpokenToken("ゲーム", 1.0, 1.5, "u")]) is None
