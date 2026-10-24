"""Reading-matched glossary corrections of recognition labels; synthetic dictionary and labels."""

import json
import timeit
from pathlib import Path

import pytest
from vibrato import Vibrato

from replay import asr_glossary as glossary
from replay import caption_linguistics
from replay.core import ReplayError


@pytest.fixture(autouse=True)
def synthetic_dictionary(monkeypatch: pytest.MonkeyPatch) -> Vibrato:
    tokenizer = Vibrato.from_textdict(
        (
            "同,0,0,0,名詞,一般,*,*,*,*,同,ドウ,ドー\n"
            "節,0,0,0,名詞,一般,*,*,*,*,節,フシ,フシ\n"
            "説,0,0,0,名詞,一般,*,*,*,*,説,セツ,セツ\n"
            "接,0,0,0,名詞,一般,*,*,*,*,接,セツ,セツ\n"
            "総,0,0,0,接頭詞,名詞接続,*,*,*,*,総,ソウ,ソー\n"
            "市長,0,0,0,名詞,一般,*,*,*,*,市長,シチョウ,シチョー\n"
            "視聴,0,0,0,名詞,サ変接続,*,*,*,*,視聴,シチョウ,シチョー\n"
            "が,0,0,0,助詞,格助詞,一般,*,*,*,が,ガ,ガ\n"
            "増え,0,0,0,動詞,自立,*,*,一段,連用形,増える,フエ,フエ\n"
            "た,0,0,0,助動詞,*,*,*,特殊・タ,基本形,た,タ,タ\n"
            "星野,0,0,0,名詞,固有名詞,人名,姓,*,*,星野,ホシノ,ホシノ\n"
        ),
        "1 1\n0 0 0\n",
        "DEFAULT 0 1 0\nSPACE 0 1 0\n0x0020 SPACE\n",
        "DEFAULT,0,0,10000,名詞,一般\nSPACE,0,0,10000,記号,空白\n",
        max_grouping_len=1,
    )
    monkeypatch.setattr(caption_linguistics, "_tokenizer", lambda: tokenizer)
    return tokenizer


TERMS = glossary.parse(
    {
        "version": 1,
        "replacements": [
            {"reading": "ドウセツ", "word": "同接", "variants": ["同節"]},
            {"reading": "そうしちょう", "word": "総視聴"},
        ],
        "protected": ["星野同説"],
    }
)


def label(text: str, start: float, end: float, provenance: str) -> dict[str, object]:
    return {"text": text, "start": start, "end": end, "provenance": provenance}


def test_reading_is_normalized_to_hiragana() -> None:
    assert glossary.hiragana("ドウセツＡ") == "どうせつA"


def test_parse_keeps_replacements_variants_and_protected_terms() -> None:
    assert (
        glossary.Glossary(
            (
                glossary.Replacement("どうせつ", "同接", ("同節",)),
                glossary.Replacement("そうしちょう", "総視聴"),
            ),
            ("星野同説",),
        )
        == TERMS
    )
    assert TERMS.guarded() == ("星野同説", "同接", "総視聴")


@pytest.mark.parametrize(
    "document",
    [
        {"version": 2},
        {"version": 1, "extra": []},
        {"version": 1, "replacements": {}},
        {"version": 1, "replacements": [{"reading": "どうせつ"}]},
        {"version": 1, "replacements": [{"reading": "どうせつ", "word": "同接", "x": 1}]},
        {"version": 1, "replacements": [{"reading": 1, "word": "同接"}]},
        {"version": 1, "replacements": [{"reading": "ど", "word": "同接"}]},
        {"version": 1, "replacements": [{"reading": "dou", "word": "同接"}]},
        {"version": 1, "replacements": [{"reading": "どうせつ", "word": "同 接"}]},
        {
            "version": 1,
            "replacements": [{"reading": "どうせつ", "word": "同接", "variants": ["どう"]}],
        },
        {
            "version": 1,
            "replacements": [{"reading": "どうせつ", "word": "同接", "variants": ["同接"]}],
        },
        {
            "version": 1,
            "replacements": [{"reading": "どうせつ", "word": "同接", "variants": "同節"}],
        },
        {"version": 1, "protected": ["星"]},
        {"version": 1, "protected": [1]},
        [],
    ],
)
def test_invalid_glossaries_are_rejected(document: object) -> None:
    with pytest.raises(ReplayError):
        glossary.parse(document)


def test_files_merge_later_reading_wins_and_protected_terms_accumulate(tmp_path: Path) -> None:
    shared = tmp_path / "common.json"
    channel = tmp_path / "channel.json"
    shared.write_text(
        json.dumps(
            {
                "version": 1,
                "replacements": [{"reading": "どうせつ", "word": "同説"}],
                "protected": ["星野"],
            }
        ),
        encoding="utf-8",
    )
    channel.write_text(
        json.dumps(
            {
                "version": 1,
                "replacements": [{"reading": "どうせつ", "word": "同接"}],
                "protected": ["星野", "月見"],
                "note": "channel",
            }
        ),
        encoding="utf-8",
    )
    assert glossary.load([shared, channel]) == glossary.Glossary(
        (glossary.Replacement("どうせつ", "同接"),), ("星野", "月見")
    )


def test_reading_match_over_whole_morphemes_replaces_a_wrong_homophone() -> None:
    assert glossary.find("c", "同説が増えた", TERMS) == [
        glossary.Edit(
            "c", 0, 2, "同説", "同接", "glossary", {"reading": "どうせつ", "rule": "reading"}
        )
    ]


def test_curated_variant_matches_where_the_dictionary_reads_otherwise() -> None:
    assert glossary.find("c", "同節が増えた", TERMS) == [
        glossary.Edit(
            "c", 0, 2, "同節", "同接", "glossary", {"reading": "どうせつ", "rule": "variant"}
        )
    ]


def test_compound_reading_spans_several_morphemes() -> None:
    assert glossary.find("c", "総市長が増えた", TERMS) == [
        glossary.Edit(
            "c",
            0,
            3,
            "総市長",
            "総視聴",
            "glossary",
            {"reading": "そうしちょう", "rule": "reading"},
        )
    ]


def test_a_different_word_with_part_of_the_reading_is_left_alone() -> None:
    assert glossary.find("c", "市長が増えた", TERMS) == []


def test_kana_spelling_and_the_correct_word_are_not_edited() -> None:
    assert glossary.find("c", "どうせつが同接が増えた", TERMS) == []


def test_protected_term_is_never_edited() -> None:
    assert glossary.find("c", "星野同説が増えた", TERMS) == []


def test_empty_glossary_finds_nothing() -> None:
    assert glossary.find("c", "同説", glossary.Glossary()) == []
    assert glossary.plan([label("同説", 0.0, 1.0, "c:0")], glossary.Glossary()) == []


def test_same_length_edit_rewrites_characters_in_place_and_keeps_every_timing() -> None:
    rows = [
        label("同", 1.0, 1.2, "c:0"),
        {**label("説が", 1.2, 1.6, "c:1"), "onsetSnap": {"originalStart": 1.25}},
        label("増えた", 1.6, 2.0, "c:2"),
        label("総市長", 3.0, 3.5, "d:0"),
    ]
    edits = glossary.plan(rows, TERMS)
    assert glossary.apply(rows, edits) == [
        label("同", 1.0, 1.2, "c:0"),
        {**label("接が", 1.2, 1.6, "c:1"), "onsetSnap": {"originalStart": 1.25}},
        label("増えた", 1.6, 2.0, "c:2"),
        label("総視聴", 3.0, 3.5, "d:0"),
    ]


def test_length_change_merges_touched_labels_from_first_start_to_last_end() -> None:
    rows = [
        label("あ同", 1.0, 1.2, "c:0"),
        label("説い", 1.2, 1.6, "c:1"),
        label("う", 1.6, 2.0, "c:2"),
    ]
    edit = glossary.Edit("c", 1, 3, "同説", "同接会", "glossary")
    assert glossary.apply(rows, [edit]) == [
        label("あ同接会い", 1.0, 1.6, "c:0"),
        label("う", 1.6, 2.0, "c:2"),
    ]


def test_two_edits_in_one_label_both_apply() -> None:
    rows = [label("同説と同説", 1.0, 2.0, "c:0"), label("た", 2.0, 2.2, "c:1")]
    edits = [
        glossary.Edit("c", 0, 2, "同説", "同接", "glossary"),
        glossary.Edit("c", 3, 5, "同説", "同接会", "glossary"),
    ]
    assert glossary.apply(rows, edits) == [
        label("同接と同接会", 1.0, 2.0, "c:0"),
        label("た", 2.0, 2.2, "c:1"),
    ]


def test_edit_that_no_longer_matches_its_labels_is_rejected() -> None:
    with pytest.raises(ReplayError, match="does not match"):
        glossary.apply(
            [label("同接", 0.0, 1.0, "c:0")], [glossary.Edit("c", 0, 2, "同説", "同接", "g")]
        )


def test_overlapping_edits_are_rejected() -> None:
    edits = [
        glossary.Edit("c", 0, 2, "同説", "同接", "g"),
        glossary.Edit("c", 1, 2, "説", "接", "g"),
    ]
    with pytest.raises(ReplayError, match="overlap"):
        glossary.apply([label("同説", 0.0, 1.0, "c:0")], edits)


def test_edit_of_an_unknown_chunk_is_rejected() -> None:
    with pytest.raises(ReplayError, match="chunk"):
        glossary.apply(
            [label("同説", 0.0, 1.0, "c:0")], [glossary.Edit("x", 0, 2, "同説", "同接", "g")]
        )


def test_labels_need_chunk_provenance() -> None:
    with pytest.raises(ReplayError, match="provenance"):
        glossary.chunks([{"text": "同説", "start": 0.0, "end": 1.0}])


def test_trie_first_entry_of_a_key_wins_and_walks_stop_on_a_miss() -> None:
    trie = glossary.Trie([("ab", 1), ("abc", 2), ("ab", 3)])
    assert list(trie.ends("xabcd", 1)) == [3, 4]
    assert trie.values[trie.step(0, "ab") or 0] == 1
    assert trie.step(0, "ax") is None
    assert list(trie.ends("ax", 0)) == []


def test_guarded_spans_cover_every_occurrence() -> None:
    matcher = glossary.Matcher(glossary.Glossary(protected=("星野", "星野同")))
    assert matcher.guarded("星野同説と星野") == [(0, 2), (0, 3), (5, 7)]


def test_compiled_matcher_is_reused_until_a_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"version": 1, "protected": ["星野"]}), encoding="utf-8")
    first = glossary.compiled([path])
    assert glossary.compiled([path]) is first
    path.write_text(json.dumps({"version": 1, "protected": ["月見"]}), encoding="utf-8")
    assert glossary.compiled([path]).glossary.protected == ("月見",)


def synthetic(entries: int) -> glossary.Matcher:
    """``entries`` distinct readings and protected names that never occur in the text below."""
    kana = "ぁあぃいぅうぇえぉお"
    readings = [
        "".join(kana[(number // 10**place) % 10] for place in range(5)) + "ゐ"
        for number in range(entries)
    ]
    return glossary.Matcher(
        glossary.Glossary(
            tuple(glossary.Replacement(reading, "語" + reading) for reading in readings),
            tuple("名" + reading for reading in readings),
        )
    )


def test_matching_cost_does_not_grow_with_the_list(synthetic_dictionary: Vibrato) -> None:
    text = "同説が増えた総市長が増えた" * 200
    small, large = synthetic(100), synthetic(10_000)
    fastest_small = min(
        timeit.repeat(lambda: small.find("c", text, synthetic_dictionary), number=1, repeat=5)
    )
    fastest_large = min(
        timeit.repeat(lambda: large.find("c", text, synthetic_dictionary), number=1, repeat=5)
    )
    # A per-entry scan would be ~100x slower at 10k entries; the tries walk the same paths.
    assert fastest_large < 2.0 * fastest_small
