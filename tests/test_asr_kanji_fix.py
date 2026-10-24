"""Glossary then zenz re-selection of same-reading kanji, on synthetic labels and candidates."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from vibrato import Vibrato

from replay import asr_glossary as glossary
from replay import asr_kanji_fix as fix
from replay import asr_whisper_alternatives as alternatives
from replay import caption_linguistics
from replay.core import ReplayError


@pytest.fixture(autouse=True)
def synthetic_dictionary(monkeypatch: pytest.MonkeyPatch) -> Vibrato:
    tokenizer = Vibrato.from_textdict(
        (
            "機械,0,0,0,名詞,一般,*,*,*,*,機械,キカイ,キカイ\n"
            "機会,0,0,0,名詞,一般,*,*,*,*,機会,キカイ,キカイ\n"
            "きかい,0,0,0,名詞,一般,*,*,*,*,きかい,キカイ,キカイ\n"
            "奇怪,0,0,0,名詞,一般,*,*,*,*,奇怪,キカイ,キカイ\n"
            "を,0,0,0,助詞,格助詞,一般,*,*,*,を,ヲ,ヲ\n"
            "買う,0,0,0,動詞,自立,*,*,五段,基本形,買う,カウ,カウ\n"
            "飼う,0,0,0,動詞,自立,*,*,五段,基本形,飼う,カウ,カウ\n"
            "星野,0,0,0,名詞,固有名詞,人名,姓,*,*,星野,ホシノ,ホシノ\n"
            "同,0,0,0,名詞,一般,*,*,*,*,同,ドウ,ドー\n"
            "説,0,0,0,名詞,一般,*,*,*,*,説,セツ,セツ\n"
            "が,0,0,0,助詞,格助詞,一般,*,*,*,が,ガ,ガ\n"
            "と,0,0,0,助詞,格助詞,一般,*,*,*,と,ト,ト\n"
        ),
        "1 1\n0 0 0\n",
        "DEFAULT 0 1 0\nSPACE 0 1 0\n0x0020 SPACE\n",
        "DEFAULT,0,0,10000,名詞,一般\nSPACE,0,0,10000,記号,空白\n",
        max_grouping_len=1,
    )
    monkeypatch.setattr(caption_linguistics, "_tokenizer", lambda: tokenizer)
    return tokenizer


def guard(*terms: str) -> fix.Guard:
    return glossary.Matcher(glossary.Glossary(protected=terms)).guarded


def candidate(
    text: str, original: float = -1.0, alternative: float = -1.5
) -> alternatives.Candidate:
    return alternatives.Candidate(0, 1, (1,), text, 0.5, original, alternative)


def segment(
    text: str,
    *candidates: alternatives.Candidate,
    chunk: str = "c",
    index: int = 0,
    start: float = 1.0,
) -> alternatives.Segment:
    return alternatives.Segment(chunk, index, start, start + 1.0, text, candidates)


def scores(by_surface: dict[str, float]) -> fix.Score:
    def score(items: Sequence[tuple[str, str, str]]) -> list[float]:
        return [by_surface.get(surface, 0.0) for _reading, _context, surface in items]

    return score


def test_reading_and_protected_morphemes() -> None:
    tokenizer = caption_linguistics._tokenizer()
    assert fix.analysed("星野が機械を買うOK", tokenizer, guard("買う")) == (
        "ホシノガキカイヲカウOK",
        [(6, 8), (0, 2)],
    )
    assert fix.analysed("謎", tokenizer, guard()) == ("謎", [(0, 1)])
    assert fix.analysed("ぬ", tokenizer, guard()) == ("ヌ", [(0, 1)])


@pytest.mark.parametrize(
    ("changed", "kind"),
    [
        ("機会を買う", "kanji-homophone"),
        ("きかいを買う", "orthography"),
        ("キカイを買う", "orthography"),
        ("機械を買う!", "punct"),
        ("機械を買う", "none"),
    ],
)
def test_difference_kinds(changed: str, kind: str) -> None:
    assert fix.classify("機械を買う", changed, []) == (kind, False)


def test_script_only_difference() -> None:
    assert fix.classify("かう", "カウ", []) == ("script", False)


def test_difference_inside_a_protected_span_is_flagged() -> None:
    assert fix.classify("機械を買う", "機会を買う", [(0, 2)]) == ("kanji-homophone", True)


def test_window_keeps_twelve_characters_each_side_and_moves_the_rest_to_context() -> None:
    original = "あ" * 20 + "機械" + "い" * 20
    changed = "あ" * 20 + "機会" + "い" * 20
    # The edit is 械 -> 会 (機 is shared), so the window starts 12 characters before 械.
    assert fix.window(original, changed, "前") == (
        "あ" * 11 + "機械" + "い" * 12,
        "あ" * 11 + "機会" + "い" * 12,
        "前" + "あ" * 9,
    )
    assert fix.window("a b", "ab", "x y") == ("ab", "ab", "xy")


def test_best_same_reading_kanji_swap_above_tau_is_accepted() -> None:
    segments = [
        segment(
            "機械を買う", candidate("機会を買う"), candidate("奇怪を買う"), candidate("機会を飼う!")
        )
    ]
    decisions, stats = fix.decide(
        segments,
        scores({"機会を買う": 4.0, "奇怪を買う": 1.0}),
        guard(),
        caption_linguistics._tokenizer(),
    )
    assert [(item.text, item.zenz, item.whisper, item.combined) for item in decisions] == [
        ("機会を買う", 4.0, -0.5, 3.5)
    ]
    assert stats == {"candidates": 3, "sameReading": 2, "kanjiHomophone": 2}  # "!" reads too


def test_spelling_variant_that_zenz_prefers_blocks_the_kanji_swap() -> None:
    segments = [segment("機械を買う", candidate("機会を買う"), candidate("きかいを買う"))]
    decisions, _ = fix.decide(
        segments,
        scores({"機会を買う": 4.0, "きかいを買う": 9.0}),
        guard(),
        caption_linguistics._tokenizer(),
    )
    assert decisions == []


def test_below_tau_protected_or_different_reading_is_not_accepted() -> None:
    segments = [
        segment("機械を買う", candidate("機会を買う")),
        segment("星野が機械", candidate("星野が機会"), index=1),
        segment("同説", candidate("動説"), index=2),
    ]
    decisions, stats = fix.decide(
        segments,
        scores({"機会を買う": 2.4, "星野が機会": 9.0}),
        guard("機械"),
        caption_linguistics._tokenizer(),
    )
    assert decisions == []
    assert stats == {"candidates": 3, "sameReading": 2, "kanjiHomophone": 2}


def test_duplicate_alternatives_keep_the_more_probable_path() -> None:
    segments = [
        segment(
            "機械を買う",
            candidate("機会を買う", -1.0, -3.0),
            candidate("機会を買う"),
            candidate("機会を買う", -1.0, -4.0),
        )
    ]
    decisions, _ = fix.decide(
        segments, scores({"機会を買う": 3.0}), guard(), caption_linguistics._tokenizer()
    )
    assert [item.whisper for item in decisions] == [-0.5]


def decision(item: alternatives.Segment, text: str) -> fix.Decision:
    return fix.Decision(item, text, 4.0, -0.5, 0.5, "kanji-homophone", False)


def label(text: str, start: float, end: float, provenance: str) -> dict[str, object]:
    return {"text": text, "start": start, "end": end, "provenance": provenance}


def test_decision_maps_through_the_chunk_text_to_labels() -> None:
    first, second = segment("あの", chunk="c"), segment("機械を", chunk="c", index=1)
    rows = [label("あの機", 1.0, 1.5, "c:0"), label("械を", 1.5, 2.0, "c:1")]
    edits, refused = fix.mapped([decision(second, "機会を")], [first, second], rows)
    assert edits == [
        glossary.Edit(
            "c",
            3,
            4,
            "械",
            "会",
            "zenz",
            {"zenz": 4.0, "whisper": -0.5, "combined": 3.5, "margin": 0.5},
        )
    ]
    assert refused == []


def test_decision_whose_text_left_the_labels_is_refused() -> None:
    item = segment("機械を", chunk="c")
    rows = [label("機会を", 1.0, 2.0, "c:0")]
    assert fix.mapped([decision(item, "機会を")], [item], rows) == (
        [],
        [{"start": 1.0, "text": "機会を", "reason": "unmapped"}],
    )
    assert fix.mapped([decision(item, "機会を")], [item], [label("x", 0.0, 1.0, "d:0")]) == (
        [],
        [{"start": 1.0, "text": "機会を", "reason": "no-labels"}],
    )


def write(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def test_correct_applies_glossary_then_zenz_and_writes_a_receipt(tmp_path: Path) -> None:
    labels = write(
        tmp_path / "labels.json",
        {
            "version": 1,
            "labels": [
                label("同説が", 1.0, 1.5, "c:0"),
                label("機械", 1.5, 2.0, "c:1"),
                label("と", 3.0, 3.2, "d:0"),
            ],
        },
    )
    terms = write(
        tmp_path / "glossary.json",
        {"version": 1, "replacements": [{"reading": "どうせつ", "word": "同接"}]},
    )
    replayed = [segment("同説が機械", candidate("同説が機会"), chunk="c")]
    stats = {"positions": 5, "agreements": 5, "skippedWindows": 0, "chunks": 1}
    rescoring = fix.Rescoring(
        tmp_path / "chunks", tmp_path / "mono.wav", tmp_path / "w", tmp_path / "z"
    )
    report = fix.correct(
        labels,
        tmp_path / "out.json",
        tmp_path / "receipt.json",
        [terms],
        rescoring,
        replay=lambda request: (replayed, stats),
        scorer=lambda path: scores({"同説が機会": 4.0}),
    )
    assert json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) == {
        "version": 1,
        "labels": [
            label("同接が", 1.0, 1.5, "c:0"),
            label("機会", 1.5, 2.0, "c:1"),
            label("と", 3.0, 3.2, "d:0"),
        ],
    }
    assert report["edits"] == [
        {
            "start": 1.0,
            "end": 1.5,
            "chunk": "c",
            "before": "同説",
            "after": "同接",
            "source": "glossary",
            "reading": "どうせつ",
            "rule": "reading",
        },
        {
            "start": 1.5,
            "end": 2.0,
            "chunk": "c",
            "before": "械",
            "after": "会",
            "source": "zenz",
            "zenz": 4.0,
            "whisper": -0.5,
            "combined": 3.5,
            "margin": 0.5,
        },
    ]
    assert report["labels"] == {"input": 3, "output": 3}
    assert json.loads((tmp_path / "receipt.json").read_text(encoding="utf-8")) == report


def test_glossary_only_run_needs_no_models(tmp_path: Path) -> None:
    labels = write(tmp_path / "labels.json", {"labels": [label("同説", 1.0, 2.0, "c:0")]})
    terms = write(
        tmp_path / "g.json",
        {"version": 1, "replacements": [{"reading": "どうせつ", "word": "同接会"}]},
    )
    code = fix.main(
        [str(labels), str(tmp_path / "o.json"), str(tmp_path / "r.json"), "--glossary", str(terms)]
    )
    assert code == 0
    assert json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))["labels"] == [
        label("同接会", 1.0, 2.0, "c:0")
    ]
    assert "rescoring" not in json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))


def test_existing_output_is_never_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = write(tmp_path / "labels.json", {"labels": [label("あ", 1.0, 2.0, "c:0")]})
    (tmp_path / "o.json").write_text("{}", encoding="utf-8")
    assert fix.main([str(labels), str(tmp_path / "o.json"), str(tmp_path / "r.json")]) == 1
    assert "ASR_KANJI_FIX_ERROR" in capsys.readouterr().err


@pytest.mark.parametrize("document", [[], {"labels": {}}, {"labels": [{"text": 1}]}])
def test_malformed_labels_are_rejected(tmp_path: Path, document: object) -> None:
    labels = write(tmp_path / "labels.json", document)
    with pytest.raises(ReplayError):
        fix.correct(labels, tmp_path / "o.json", tmp_path / "r.json", [], None)
