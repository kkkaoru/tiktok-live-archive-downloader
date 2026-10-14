import subprocess
from pathlib import Path

import pytest

from replay.core import ReplayError
from replay.speaker_subtitles import (
    SINGLE_FILL,
    SpeakerCue,
    main,
    read_cues,
    render_ass,
    render_srt,
    wrap_two_lines,
)


def test_two_line_layout_preserves_existing_short_layout() -> None:
    assert wrap_two_lines("短い字幕", 18) == "短い字幕"
    assert wrap_two_lines("一行目\n二行目", 18) == "一行目\n二行目"


def test_two_line_layout_rebalances_soft_paragraphs_without_losing_text() -> None:
    text = "abcd\nefghijklmnop"
    assert wrap_two_lines(text, 8) == "abcdefgh\nijklmnop"
    punctuated = "abcdefg,hijklmnopqrs"
    wrapped = wrap_two_lines(punctuated, 10)
    assert wrapped.replace("\n", "") == punctuated
    assert len(wrapped.splitlines()) == 2
    assert all(len(line) <= 10 for line in wrapped.splitlines())


@pytest.mark.parametrize("text", ["abcdefghij、klmnopqrs", "a" * 21])
def test_two_line_layout_does_not_weaken_break_or_width_guards(text: str) -> None:
    with pytest.raises(ValueError, match="Caption cannot fit two lines"):
        wrap_two_lines(text, 10)


@pytest.mark.parametrize("width", [0, 81, True])
def test_two_line_layout_validates_width(width: int) -> None:
    with pytest.raises(ValueError, match="Invalid two-line width"):
        wrap_two_lines("text", width)


def test_overlapping_speakers_keep_lanes_with_purple_green_and_light_blue_fills() -> None:
    result = render_ass(
        (
            SpeakerCue(1, 3, "right", "右"),
            SpeakerCue(1, 2, "left", "左"),
            SpeakerCue(1.5, 3, "middle", "中"),
        )
    )
    assert "Style: left,Hiragino Sans,38," in result
    assert (
        "Dialogue: 0,0:00:01.00,0:00:02.00,left,,0,0,0,,"
        "{\\an2\\pos(650,940)\\bord7\\shad0\\3c&H000000&\\1a&HFF&}左"
    ) in result
    assert (
        "Dialogue: 1,0:00:01.00,0:00:02.00,left,,0,0,0,,"
        "{\\an2\\pos(650,940)\\bord5\\shad0\\3c&HFFFFFF&\\1a&HFF&}左"
    ) in result
    assert (
        "Dialogue: 2,0:00:01.00,0:00:02.00,left,,0,0,0,,"
        "{\\an2\\pos(650,940)\\bord0\\shad0\\1c&HFF8CC4&\\clip(0,0,1920,1080)}左"
    ) in result
    assert "\\1c&HFFD46C&\\clip(0,0,1920,1080)}右" in result
    assert "\\1c&H8CDC7C&\\clip(0,0,1920,1080)}中" in result
    assert "\\pos(1650,940)" in result
    assert "\\pos(1140,940)" in result
    assert "[" not in result.split("[Events]")[1]


def test_wrapped_lane_caption_uses_one_solid_fill() -> None:
    result = render_ass(
        (
            SpeakerCue(1, 2, "middle", "一二三四五六七八九十一二三四"),
            SpeakerCue(1, 2, "left", "左"),
        )
    )
    clips = [
        line.split("\\1c")[1].split("}")[0]
        for line in result.splitlines()
        if line.startswith("Dialogue: 2") and ",middle," in line
    ]
    assert clips == ["&H8CDC7C&\\clip(0,0,1920,1080)"]


def test_overlapping_cues_of_one_speaker_stack_in_the_lowest_free_slot() -> None:
    result = render_ass(
        (
            SpeakerCue(1, 3, "right", "先の発話"),
            SpeakerCue(2, 4, "right", "重なる発話"),
            SpeakerCue(3.5, 5, "right", "三つ目"),
            SpeakerCue(1, 2, "left", "左"),
        )
    )
    borders = [line for line in result.splitlines() if line.startswith("Dialogue: 0")]
    assert [line.split("\\pos(")[1].split(")")[0] for line in borders] == [
        "650,940",
        "1650,940",
        "1650,880",
        "1650,940",
    ]


def test_overlapping_single_speaker_cues_stack_upwards() -> None:
    result = render_ass(
        (SpeakerCue(1, 3, "unassigned", "一つ目"), SpeakerCue(2, 4, "unassigned", "二つ目"))
    )
    borders = [line for line in result.splitlines() if line.startswith("Dialogue: 0")]
    assert [line.split("\\pos(")[1].split(")")[0] for line in borders] == ["960,1040", "960,963"]


def test_single_fill_has_aaa_specified_contrast_against_white() -> None:
    rgb = tuple(component / 255 for component in SINGLE_FILL)
    linear = tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
    luminance = sum(c * weight for c, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))
    ratio = 1.05 / (luminance + 0.05)
    assert ratio >= 7
    assert ratio == pytest.approx(7.1367, abs=0.0001)
    assert SINGLE_FILL == (0x00, 0x5A, 0x9C)


def test_single_or_unknown_speaker_is_blue_cyan_without_a_name() -> None:
    result = render_ass((SpeakerCue(1, 2, "unassigned", "ひとり"),))
    assert "Style: single,Hiragino Sans,52," in result
    assert (
        "Dialogue: 2,0:00:01.00,0:00:02.00,single,,0,0,0,,"
        "{\\an2\\pos(960,1040)\\bord0\\shad0\\1c&H9C5A00&\\clip(0,0,1920,1080)}ひとり"
    ) in result
    assert render_srt((SpeakerCue(1, 2, "left", "ひとり"),)) == (
        "1\n00:00:01,000 --> 00:00:02,000\nひとり\n"
    )
    long = render_ass((SpeakerCue(1, 2, "unassigned", "あ" * 30),))
    assert "あ" * 24 + "\\N" + "あ" * 6 in long
    tail = render_ass((SpeakerCue(1, 2, "unassigned", "あ" * 24 + "よ"),))
    assert "あ" * 13 + "\\N" + "あ" * 11 + "よ" in tail


def test_srt_keeps_concurrent_speakers_for_transcript_viewer() -> None:
    result = render_srt((SpeakerCue(1, 3, "right", "右"), SpeakerCue(1, 2, "left", "左")))
    assert result == (
        "1\n00:00:01,000 --> 00:00:02,000\n[左] 左\n\n2\n00:00:01,000 --> 00:00:03,000\n[右] 右\n"
    )


def test_serial_srt_preserves_concurrent_display_at_native_boundaries() -> None:
    result = render_srt(
        (SpeakerCue(1, 3, "right", "右"), SpeakerCue(1, 2, "left", "左")),
        serial=True,
    )
    assert result == (
        "1\n00:00:01,000 --> 00:00:02,000\n[左] 左\n[右] 右\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n[右] 右\n"
    )


def test_serial_srt_removes_ended_cues_before_same_clock_starts() -> None:
    result = render_srt(
        (SpeakerCue(1, 2, "left", "先"), SpeakerCue(2, 3, "right", "後")),
        serial=True,
    )
    assert result == (
        "1\n00:00:01,000 --> 00:00:02,000\n[左] 先\n\n2\n00:00:02,000 --> 00:00:03,000\n[右] 後\n"
    )
    assert render_srt((), serial=True) == ""


def test_serial_srt_refuses_collapsed_clock() -> None:
    with pytest.raises(ValueError, match="clock precision"):
        render_srt((SpeakerCue(0, 0.001, "left", "短すぎる"),), serial=True)


def test_real_mov_text_retains_both_overlapping_captions(tmp_path: Path) -> None:
    sidecar = tmp_path / "states.srt"
    sidecar.write_text(
        render_srt(
            (SpeakerCue(1, 3, "right", "右"), SpeakerCue(1, 2, "left", "左")),
            serial=True,
        ),
        encoding="utf-8",
    )
    movie = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=10:d=4",
            "-i",
            str(sidecar),
            "-map",
            "0:v:0",
            "-map",
            "1:0",
            "-c:v",
            "libx264",
            "-c:s",
            "mov_text",
            str(movie),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    extracted = tmp_path / "extracted.srt"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(movie),
            "-map",
            "0:s:0",
            str(extracted),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert extracted.read_text(encoding="utf-8") == (
        "1\n00:00:01,000 --> 00:00:02,000\n[左] 左\n[右] 右\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n[右] 右\n\n"
    )


def test_unattributed_overlap_is_grey_and_never_named() -> None:
    cues = (
        SpeakerCue(1, 2, "middle", "本人"),
        SpeakerCue(1, 2, "left", "相手"),
        SpeakerCue(1, 2, "unassigned", "声が重なる"),
    )
    result = render_ass(cues)
    assert "\\pos(960,825)\\bord0\\shad0\\1c&HC8C8C8&" in result
    assert render_srt(cues) == (
        "1\n00:00:01,000 --> 00:00:02,000\n[左] 相手\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n[中央] 本人\n\n"
        "3\n00:00:01,000 --> 00:00:02,000\n声が重なる\n"
    )


def test_long_captions_wrap_within_a_lane_or_fail_closed() -> None:
    result = render_ass(
        (SpeakerCue(0, 1, "left", "1234567890123"), SpeakerCue(9, 10, "middle", "他"))
    )
    assert "1234567\\N890123" in result
    with pytest.raises(ValueError, match="Split long speech"):
        render_ass((SpeakerCue(0, 1, "left", "a" * 37), SpeakerCue(9, 10, "middle", "他")))


def test_wrap_prefers_punctuation_without_changing_speech_clocks() -> None:
    result = render_ass(
        (
            SpeakerCue(1.25, 3.75, "left", "今日は、皆さんに感謝します。"),
            SpeakerCue(9, 10, "middle", "他"),
        )
    )
    assert "0:00:01.25,0:00:03.75,left" in result
    assert "今日は、\\N皆さんに感謝します。" in result


def test_wrap_does_not_start_a_line_with_closing_marks() -> None:
    result = render_ass(
        (SpeakerCue(0, 2, "right", "123456789012、続きです"), SpeakerCue(9, 10, "middle", "他"))
    )
    assert "1234567\\N89012、続きです" in result


def test_wrap_keeps_open_quote_with_following_text() -> None:
    result = render_ass(
        (SpeakerCue(0, 2, "middle", "12345678901「何ですか」"), SpeakerCue(9, 10, "left", "他"))
    )
    assert "12345678901\\N「何ですか」" in result


def test_wrap_preserves_external_leading_marks_and_spaces() -> None:
    result = render_ass(
        (SpeakerCue(0, 2, "left", "、先頭の記号も残す\n  続き"), SpeakerCue(9, 10, "middle", "他"))
    )
    assert "、先頭の記号も残す\\N  続き" in result


def test_transcript_text_cannot_inject_ass_overrides() -> None:
    result = render_ass((SpeakerCue(3600, 3601, "left", "a{\\pos(1,1)}\r\nb"),))
    assert "1:00:00.00,1:00:01.00" in result
    assert "a｛＼pos(1,1)｝\\Nb" in result
    assert "{\\pos(1,1)}" not in result


@pytest.mark.parametrize(
    ("start", "end", "speaker", "text"),
    [
        (0, 0, "left", "x"),
        (-1, 2, "left", "x"),
        (0, 2, "unknown", "x"),
        (0, 2, "right", "  "),
        (float("nan"), 2, "left", "x"),
    ],
)
def test_invalid_cues_rejected(start: float, end: float, speaker: str, text: str) -> None:
    with pytest.raises(ValueError, match="Invalid speaker cue"):
        SpeakerCue(start, end, speaker, text)


def test_csv_validation_and_non_overwriting_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_file = tmp_path / "cues.csv"
    csv_file.write_text("start,end,speaker,text\n0.1,1.4,left,hello\n", encoding="utf-8")
    assert read_cues(csv_file) == (SpeakerCue(0.1, 1.4, "left", "hello"),)
    output = tmp_path / "captions.ass"
    srt = tmp_path / "captions.srt"
    monkeypatch.setattr(
        "sys.argv", ["speaker_subtitles", str(csv_file), str(output), "--srt", str(srt)]
    )
    assert main() == 0
    assert output.read_text(encoding="utf-8").startswith("[Script Info]\n")
    assert srt.read_text(encoding="utf-8") == ("1\n00:00:00,100 --> 00:00:01,400\nhello\n")
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        main()


def test_csv_bad_header_and_bad_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cues.csv"
    path.write_text("start,end,speaker\n0,1,left\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="columns"):
        read_cues(path)
    path.write_text("start,end,speaker,text\n0,nonsense,left,hello\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="Invalid speaker cue"):
        read_cues(path)
    path.write_text("start,end,speaker,text\n", encoding="utf-8")
    with pytest.raises(ReplayError, match="empty"):
        read_cues(path)
    monkeypatch.setattr("sys.argv", ["speaker_subtitles", str(path), str(tmp_path / "bad.srt")])
    with pytest.raises(ReplayError, match="ASS"):
        main()
