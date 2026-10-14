"""Whole phrase display retains recognition clocks and canonical style geometry."""

import json
from pathlib import Path

import pytest

from replay.caption_linguistics import CaptionLayoutError
from replay.caption_phrase_ass import (
    Appearance,
    main,
    phrase_cues,
    reflow_existing,
    render_ass,
    restyle,
)
from replay.caption_timing import CaptionCue, SpokenToken
from replay.core import ReplayError


def test_phrase_not_prefix_and_no_midword_switch() -> None:
    tokens = (
        SpokenToken("フ", 1, 1.1, "cut0"),
        SpokenToken("ォロワー", 1.1, 1.5, "cut0"),
        SpokenToken("さん", 1.5, 1.7, "cut0"),
    )
    cues = phrase_cues(tokens, ((0, 300, 0),), layout=lambda text: text)
    assert len(cues) == 1
    assert cues[0].text == "フォロワーさん"
    assert cues[0].start == 1
    assert cues[0].end == pytest.approx(1.82)


def test_real_cut_resets_phrase_and_hold() -> None:
    tokens = (SpokenToken("はい", 0, 0.2, "cut0"), SpokenToken("どうも", 0.3, 0.8, "cut1"))
    assert phrase_cues(tokens, ((0, 9, 0), (9, 90, 1)), layout=lambda text: text) == (
        CaptionCue("はい", 0, 0.3),
        CaptionCue("どうも", 0.3, 0.92),
    )


def test_overwide_phrase_pages_at_recognized_onset() -> None:
    def layout(text: str) -> str:
        if len(text) > 3:
            raise CaptionLayoutError("wide")
        return text

    assert phrase_cues(
        (SpokenToken("はい", 0, 0.7, "cut0"), SpokenToken("どうも", 1, 1.8, "cut0")),
        ((0, 90, 0),),
        layout=layout,
    ) == (CaptionCue("はい", 0, 0.82), CaptionCue("どうも", 1, 1.92))


def test_overwide_atomic_unit_is_rejected() -> None:
    def layout(text: str) -> str:
        raise CaptionLayoutError("wide")

    with pytest.raises(ReplayError, match="Intact phrase"):
        phrase_cues((SpokenToken("フォロワーさん", 0, 1, "cut0"),), ((0, 90, 0),), layout=layout)


def test_missing_cut_is_rejected() -> None:
    with pytest.raises(ReplayError, match="retained cut"):
        phrase_cues((SpokenToken("はい", 5, 6, "cut0"),), ((0, 90, 0),), layout=lambda text: text)


def test_invalid_minimum_is_rejected() -> None:
    with pytest.raises(ReplayError, match="minimum"):
        phrase_cues((), (), layout=lambda text: text, minimum_seconds=0)


def test_ass_style_gradient_and_bottom_stack_without_animation() -> None:
    ass = render_ass((CaptionCue("一行\n二行\n三行", 1, 2),), Appearance(52, 1326, 1040, 745, 1906))
    assert "NotoSansJP-Thin_Bold,52" in ass
    assert "YCbCr Matrix: TV.709" in ass
    assert "\\pos(1326.000,936.000)" in ass
    assert "\\pos(1326.000,988.000)" in ass
    assert "\\pos(1326.000,1040.000)" in ass
    assert "\\bord8.509" in ass
    assert "\\bord10.282" in ass
    assert "\\1c&HEEB02F&" in ass
    assert "\\1c&HFFF6E3&" in ass
    assert "\\t(" not in ass
    assert "\\fad" not in ass
    assert ass.count("Dialogue:") == 108


@pytest.mark.parametrize("text", ["a\n\nb", "a\nb\nc\nd", " "])
def test_ass_rejects_empty_or_excess_lines(text: str) -> None:
    with pytest.raises(ReplayError, match="unsupported line"):
        render_ass((CaptionCue(text, 1, 2),), Appearance(52, 1326, 1040, 745, 1906))


@pytest.mark.parametrize("size,bottom", [(260, 1040), (52, 1080), (float("nan"), 1040)])
def test_appearance_rejects_invalid_extent(size: float, bottom: float) -> None:
    with pytest.raises(ReplayError, match="appearance"):
        Appearance(size, 1326, bottom, 745, 1906)


def test_cli_writes_plan_srt_and_ass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Measured:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __call__(self, text: str) -> str:
            return text

        def verify_inputs(self) -> dict[str, str]:
            return {}

    monkeypatch.setattr("replay.caption_phrase_ass.NativeCaptionLayout", Measured)
    cues = tmp_path / "cues.json"
    cuts = tmp_path / "cuts.json"
    cues.write_text(
        json.dumps(
            {"version": 1, "clock": "output", "cues": [{"text": "はい", "start": 0, "end": 1}]}
        )
    )
    cuts.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "source-presentation",
                "fps": 30,
                "cuts": [
                    {"source_start_frame": 0, "source_end_frame": 90, "output_start_frame": 0}
                ],
            }
        )
    )
    out = tmp_path / "plan"
    assert (
        main(
            [
                str(cues),
                str(cuts),
                str(out),
                "--native",
                str(cues),
                "--evidence",
                str(cuts),
                "--font-size",
                "52",
                "--center-x",
                "1326",
                "--bottom-y",
                "1040",
                "--safe-left",
                "745",
                "--safe-right",
                "1906",
            ]
        )
        == 0
    )
    assert "はい" in (out / "captions.srt").read_text()
    assert json.loads((out / "plan.json").read_text())["phraseCues"] == 1
    assert (out / "captions.ass").is_file()


def test_restyle_preserves_exact_phrase_text_and_clock(tmp_path: Path) -> None:
    source = tmp_path / "plan.json"
    source.write_text(json.dumps({"captions": [{"text": "字幕", "start": 1.25, "end": 2.5}]}))
    out = tmp_path / "new"
    restyle(source, out, Appearance(104, 1326, 1032, 745, 1906))
    assert json.loads((out / "plan.json").read_text())["captions"] == [
        {"text": "字幕", "start": 1.25, "end": 2.5}
    ]
    assert "00:00:01,250 --> 00:00:02,500" in (out / "captions.srt").read_text()


@pytest.mark.parametrize(
    "rows", [[], [{"text": "a", "start": True, "end": 2}], [{"text": "a", "start": 2, "end": 1}]]
)
def test_restyle_rejects_invalid_external_clocks(tmp_path: Path, rows: object) -> None:
    source = tmp_path / "plan.json"
    source.write_text(json.dumps({"captions": rows}))
    with pytest.raises(ReplayError):
        restyle(source, tmp_path / "out", Appearance(104, 1326, 1032, 745, 1906))


def test_actual_glyph_bounds_drive_gradient_and_shadow_clearance(tmp_path: Path) -> None:
    plan, bounds = tmp_path / "plan.json", tmp_path / "bounds.json"
    plan.write_text(json.dumps({"captions": [{"text": "字幕", "start": 1, "end": 2}]}))
    bounds.write_text(
        json.dumps(
            {
                "fontSize": 104,
                "records": {"字幕": {"inkTopOffsetPx": -80, "inkBottomOffsetPx": -14}},
            }
        )
    )
    out = tmp_path / "new"
    restyle(plan, out, Appearance(104, 1326, 1032, 745, 1906), bounds_path=bounds)
    assert json.loads((out / "plan.json").read_text())["bottomAnchorPx"] == 1026
    ass = (out / "captions.ass").read_text()
    assert "\\pos(1326.000,1026.000)" in ass
    assert "\\blur4.727" in ass
    assert "\\bord2.364" in ass


@pytest.mark.parametrize(
    "record", [{}, {"字幕": {"inkTopOffsetPx": "bad", "inkBottomOffsetPx": -14}}]
)
def test_bad_glyph_bounds_are_rejected(tmp_path: Path, record: object) -> None:
    plan, bounds = tmp_path / "plan.json", tmp_path / "bounds.json"
    plan.write_text(json.dumps({"captions": [{"text": "字幕", "start": 1, "end": 2}]}))
    bounds.write_text(json.dumps({"fontSize": 104, "records": record}))
    with pytest.raises(ReplayError):
        restyle(plan, tmp_path / "out", Appearance(104, 1326, 1032, 745, 1906), bounds_path=bounds)


def test_missing_or_invalid_glyph_ink_fails() -> None:
    with pytest.raises(ReplayError):
        render_ass(
            (CaptionCue("字幕", 1, 2),), Appearance(104, 1326, 1032, 745, 1906), glyph_bounds={}
        )
    with pytest.raises(ReplayError):
        render_ass(
            (CaptionCue("字幕", 1, 2),),
            Appearance(104, 1326, 1032, 745, 1906),
            glyph_bounds={"字幕": (-500, -14)},
        )


def test_reflow_preserves_full_word_and_original_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Measured:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __call__(self, text: str) -> str:
            assert text == "インフルエンスです"
            return "インフルエンス\nです"

        def verify_inputs(self) -> None:
            pass

    monkeypatch.setattr("replay.caption_phrase_ass.NativeCaptionLayout", Measured)
    plan, bounds = tmp_path / "plan.json", tmp_path / "bounds.json"
    plan.write_text(
        json.dumps({"captions": [{"text": "イン\nフルエンスです", "start": 1, "end": 2}]})
    )
    bounds.write_text(
        json.dumps(
            {"records": {"イン": {"inkWidthPx": 200}, "フルエンスです": {"inkWidthPx": 400}}}
        )
    )
    out = tmp_path / "out"
    reflow_existing(
        plan,
        out,
        appearance=Appearance(104, 1326, 1032, 745, 1906),
        native=tmp_path / "native",
        evidence=plan,
        bounds_path=bounds,
    )
    assert json.loads((out / "plan.json").read_text())["captions"] == [
        {"text": "インフルエンス\nです", "start": 1, "end": 2}
    ]
