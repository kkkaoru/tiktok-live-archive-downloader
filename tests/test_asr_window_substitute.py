"""A collapsed label run is replaced only by wholly-inside re-recognized words and is provable."""

import hashlib
import json
from pathlib import Path

import pytest

from replay import asr_window_substitute as substitute
from replay import asr_window_substitute_verify as verify
from replay.caption_repair import ClockLabel

WINDOW = (10.0, 20.0)


def label(text: str, start: float, end: float) -> ClockLabel:
    return ClockLabel(text, start, end, f"p:{text}")


def labels() -> tuple[ClockLabel, ...]:
    return (
        label("A", 8.0, 9.0),
        label("い", 10.0, 10.2),
        label("っしゃ", 10.2, 19.9),
        label("い", 19.9, 20.0),
        label("B", 20.0, 21.0),
    )


def words() -> tuple[substitute.Word, ...]:
    return (
        substitute.Word("前", 9.5, 10.5),
        substitute.Word("あ", 11.0, 12.0),
        substitute.Word("い", 12.0, 13.0),
        substitute.Word("う", 19.0, 19.9),
        substitute.Word("後", 19.9, 20.4),
    )


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def world(
    tmp_path: Path,
    *,
    clip_words: tuple[substitute.Word, ...] | None = None,
    fold: float | str | None = None,
    drop: bool | str | None = None,
) -> tuple[Path, Path]:
    document = {
        "version": 1,
        "clock": "input-pcm-not-projected",
        "parentSHA256": {},
        "humanReviewed": False,
        "labels": [
            {"text": t.text, "start": t.start, "end": t.end, "provenance": t.provenance}
            for t in labels()
        ],
    }
    original = tmp_path / "labels.json"
    original.write_text(json.dumps(document), encoding="utf-8")
    clip = tmp_path / "clip.json"
    clip.write_text(
        json.dumps(
            {
                "clock": "input-pcm-not-projected",
                "words": [
                    {"word": f" {w.text}", "start": w.start, "end": w.end}
                    for w in (clip_words or words())
                ],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "artifacts": {"labels": pin(original), "clip": pin(clip)},
                "block": {
                    "firstLabel": 1,
                    "stopLabel": 4,
                    "start": 10.0,
                    "end": 20.0,
                    **({} if fold is None else {"foldShorterThan": fold}),
                    **({} if drop is None else {"dropSustained": drop}),
                },
            }
        ),
        encoding="utf-8",
    )
    return config, tmp_path / "out.json"


def test_plan_inserts_only_words_wholly_inside_the_window_and_lists_crossing_words() -> None:
    outcome = substitute.plan(labels(), words(), first=1, stop=4, window=WINDOW, source="abc")
    assert [(c.text, c.start, c.end, c.provenance) for c in outcome.inserted] == [
        ("あ", 11.0, 12.0, "clip-reasr:abc:0"),
        ("い", 12.0, 13.0, "clip-reasr:abc:1"),
        ("う", 19.0, 19.9, "clip-reasr:abc:2"),
    ]
    assert [(w.text, w.start) for w in outcome.excluded] == [("前", 9.5), ("後", 19.9)]
    assert [c.text for c in outcome.removed] == ["い", "っしゃ", "い"]


def test_the_window_must_equal_the_span_of_the_declared_block() -> None:
    with pytest.raises(ValueError, match="not the span of the declared label block"):
        substitute.plan(labels(), words(), first=1, stop=4, window=(10.0, 19.0), source="x")


def test_a_bad_block_or_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="bounded label block"):
        substitute.plan(labels(), words(), first=3, stop=3, window=WINDOW, source="x")
    with pytest.raises(ValueError, match="bounded label block"):
        substitute.plan(labels(), words(), first=1, stop=4, window=(20.0, 10.0), source="x")


def test_a_window_without_inside_words_is_rejected() -> None:
    outside = (substitute.Word("x", 30.0, 31.0),)
    with pytest.raises(ValueError, match="No re-recognized word lies inside"):
        substitute.plan(labels(), outside, first=1, stop=4, window=WINDOW, source="x")


def test_inside_words_must_not_end_before_they_start_or_overlap() -> None:
    backwards = (substitute.Word("x", 12.0, 11.0),)
    with pytest.raises(ValueError, match="ends before it starts"):
        substitute.plan(labels(), backwards, first=1, stop=4, window=WINDOW, source="x")
    overlapping = (substitute.Word("x", 11.0, 13.0), substitute.Word("y", 12.0, 14.0))
    with pytest.raises(ValueError, match="words overlap"):
        substitute.plan(labels(), overlapping, first=1, stop=4, window=WINDOW, source="x")


def test_inserted_words_must_not_overlap_the_unchanged_neighbours() -> None:
    items = (
        label("A", 8.0, 10.5),
        label("い", 10.0, 20.0),
        label("B", 20.0, 21.0),
    )
    with pytest.raises(ValueError, match="left neighbour"):
        substitute.plan(
            items, (substitute.Word("x", 10.0, 11.0),), first=1, stop=2, window=WINDOW, source="x"
        )
    tight = (label("い", 10.0, 20.0), label("B", 19.5, 21.0))
    with pytest.raises(ValueError, match="right neighbour"):
        substitute.plan(
            tight, (substitute.Word("x", 19.0, 20.0),), first=0, stop=1, window=WINDOW, source="x"
        )


def test_words_must_use_the_input_clock_and_be_ordered_and_well_formed() -> None:
    with pytest.raises(ValueError, match="input PCM clock"):
        substitute.read_words({"clock": "other", "words": []})
    with pytest.raises(ValueError, match="no words"):
        substitute.read_words({"clock": "input-pcm-not-projected", "words": []})
    with pytest.raises(ValueError, match="Invalid re-recognized word"):
        substitute.read_words(
            {"clock": "input-pcm-not-projected", "words": [{"word": " ", "start": 0, "end": 1}]}
        )
    with pytest.raises(ValueError, match="not ordered"):
        substitute.read_words(
            {
                "clock": "input-pcm-not-projected",
                "words": [
                    {"word": "a", "start": 2, "end": 3},
                    {"word": "b", "start": 1, "end": 2},
                ],
            }
        )
    with pytest.raises(ValueError, match="finite number"):
        substitute.read_words(
            {
                "clock": "input-pcm-not-projected",
                "words": [{"word": "a", "start": True, "end": 3}],
            }
        )


def test_run_writes_a_candidate_that_the_verifier_proves(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    assert substitute.run(config, output) == 5
    assert verify.verify(output) == {"labels": 5, "removed": 3, "inserted": 3, "excluded": 2}
    document = json.loads(output.read_text(encoding="utf-8"))
    assert [row["text"] for row in document["labels"]] == ["A", "あ", "い", "う", "B"]
    assert (document["qualityApproved"], document["speechAccuracyVerified"]) == (False, False)
    assert document["windowSubstitution"]["candidateOnly"] is True


def test_run_refuses_existing_output_changed_inputs_and_bad_configuration(
    tmp_path: Path,
) -> None:
    config, output = world(tmp_path)
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="new output"):
        substitute.run(config, output)
    output.unlink()
    document = json.loads(config.read_text(encoding="utf-8"))
    document["version"] = 2
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported window substitution"):
        substitute.run(config, output)
    document["version"] = 1
    document["artifacts"] = {"labels": document["artifacts"]["labels"]}
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned labels and a pinned re-recognition"):
        substitute.run(config, output)


def test_run_rejects_a_changed_pinned_input(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    (tmp_path / "clip.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned substitution input changed"):
        substitute.run(config, output)


def test_run_rejects_non_integer_block_coordinates(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8").replace('"firstLabel": 1', '"firstLabel": "1"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact integer block coordinates"):
        substitute.run(config, output)


def written(tmp_path: Path) -> Path:
    config, output = world(tmp_path)
    substitute.run(config, output)
    return output


def test_verifier_rejects_a_change_outside_the_block(tmp_path: Path) -> None:
    output = written(tmp_path)
    output.write_text(
        output.read_text(encoding="utf-8").replace('"start": 8.0', '"start": 7.0', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="differ from the declared substitution"):
        verify.verify(output)


def test_verifier_rejects_undeclared_document_fields(tmp_path: Path) -> None:
    output = written(tmp_path)
    output.write_text(
        output.read_text(encoding="utf-8").replace('"version": 1,', '"version": 1, "extra": 1,', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Undeclared document fields"):
        verify.verify(output)


def test_verifier_rejects_a_changed_pinned_clip(tmp_path: Path) -> None:
    output = written(tmp_path)
    (tmp_path / "clip.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned verification input changed"):
        verify.verify(output)


def test_verifier_rejects_invalid_declarations(tmp_path: Path) -> None:
    output = written(tmp_path)
    output.write_text(
        output.read_text(encoding="utf-8").replace('"firstLabel": 1', '"firstLabel": "1"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid declared block"):
        verify.verify(output)


def test_verifier_rejects_missing_labels(tmp_path: Path) -> None:
    output = written(tmp_path)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["labels"] = "none"
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Refined labels are missing"):
        verify.verify(output)


def split_word() -> tuple[substitute.Word, ...]:
    return (
        substitute.Word("あ", 11.0, 12.0),
        substitute.Word("ぁ", 12.0, 12.002),
        substitute.Word("い", 12.002, 13.0),
    )


def test_fold_shorter_than_appends_a_tiny_word_to_the_word_before_it() -> None:
    outcome = substitute.plan(
        labels(),
        split_word(),
        first=1,
        stop=4,
        window=(10.0, 20.0),
        source="s",
        fold_shorter_than=0.0333,
    )
    assert [(c.text, c.start, c.end) for c in outcome.inserted] == [
        ("あぁ", 11.0, 12.002),
        ("い", 12.002, 13.0),
    ]
    assert outcome.folded == (substitute.Word("ぁ", 12.0, 12.002),)


def test_without_the_option_every_recognized_word_is_kept() -> None:
    outcome = substitute.plan(
        labels(), split_word(), first=1, stop=4, window=(10.0, 20.0), source="s"
    )
    assert [c.text for c in outcome.inserted] == ["あ", "ぁ", "い"]
    assert outcome.folded == ()


def test_a_short_first_word_has_nothing_to_fold_into() -> None:
    tiny = (substitute.Word("ぁ", 11.0, 11.001), substitute.Word("い", 11.001, 12.0))
    outcome = substitute.plan(
        labels(), tiny, first=1, stop=4, window=(10.0, 20.0), source="s", fold_shorter_than=0.0333
    )
    assert [c.text for c in outcome.inserted] == ["ぁ", "い"]
    assert outcome.folded == ()


def test_run_with_fold_shorter_than_is_recorded_and_proved(tmp_path: Path) -> None:
    config, output = world(tmp_path, clip_words=split_word(), fold=0.0333)
    assert substitute.run(config, output) == 4
    assert verify.verify(output) == {"labels": 4, "removed": 3, "inserted": 2, "excluded": 0}
    meta = json.loads(output.read_text(encoding="utf-8"))["windowSubstitution"]
    assert meta["block"]["foldShorterThan"] == 0.0333
    assert meta["folded"] == [{"text": "ぁ", "start": 12.0, "end": 12.002}]


def test_run_rejects_a_fold_threshold_that_is_not_a_small_number_of_seconds(
    tmp_path: Path,
) -> None:
    config, output = world(tmp_path, clip_words=split_word(), fold=0.5)
    with pytest.raises(ValueError, match="foldShorterThan must be"):
        substitute.run(config, output)


def test_run_rejects_a_fold_threshold_that_is_not_a_number(tmp_path: Path) -> None:
    config, output = world(tmp_path, clip_words=split_word(), fold="x")
    with pytest.raises(ValueError, match="foldShorterThan must be"):
        substitute.run(config, output)


def test_verifier_rejects_folded_words_that_differ_from_the_declaration(tmp_path: Path) -> None:
    config, output = world(tmp_path, clip_words=split_word(), fold=0.0333)
    substitute.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["windowSubstitution"]["folded"] = []
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Folded words differ"):
        verify.verify(output)


def test_verifier_rejects_an_invalid_declared_fold_threshold(tmp_path: Path) -> None:
    config, output = world(tmp_path, clip_words=split_word(), fold=0.0333)
    substitute.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["windowSubstitution"]["block"]["foldShorterThan"] = "x"
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid declared fold threshold"):
        verify.verify(output)


def held_vowel() -> tuple[substitute.Word, ...]:
    return (substitute.Word("あ", 10.0, 10.04), substitute.Word("ーー", 10.04, 19.9))


def test_drop_sustained_removes_the_block_when_only_a_held_vowel_is_recognized() -> None:
    outcome = substitute.plan(
        labels(), held_vowel(), first=1, stop=4, window=WINDOW, source="s", drop_sustained=True
    )
    assert outcome.inserted == ()
    assert [c.text for c in outcome.removed] == ["い", "っしゃ", "い"]


def test_drop_sustained_refuses_a_window_with_recognized_words() -> None:
    with pytest.raises(ValueError, match="not only a sustained vowel"):
        substitute.plan(
            labels(), words(), first=1, stop=4, window=WINDOW, source="s", drop_sustained=True
        )


def test_run_with_drop_sustained_is_recorded_and_proved(tmp_path: Path) -> None:
    config, output = world(tmp_path, clip_words=held_vowel(), drop=True)
    assert substitute.run(config, output) == 2
    assert verify.verify(output) == {"labels": 2, "removed": 3, "inserted": 0, "excluded": 0}
    document = json.loads(output.read_text(encoding="utf-8"))
    assert [row["text"] for row in document["labels"]] == ["A", "B"]
    assert document["windowSubstitution"]["block"]["dropSustained"] is True


def test_run_rejects_a_drop_flag_that_is_not_a_boolean(tmp_path: Path) -> None:
    config, output = world(tmp_path, clip_words=held_vowel(), drop="yes")
    with pytest.raises(ValueError, match="dropSustained must be a boolean"):
        substitute.run(config, output)


def test_verifier_rejects_a_drop_flag_that_is_not_a_boolean(tmp_path: Path) -> None:
    config, output = world(tmp_path, clip_words=held_vowel(), drop=True)
    substitute.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["windowSubstitution"]["block"]["dropSustained"] = "yes"
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid declared drop flag"):
        verify.verify(output)


def test_drop_sustained_allows_a_window_the_recognizer_heard_nothing_in() -> None:
    quiet = (substitute.Word("前", 5.0, 6.0), substitute.Word("後", 25.0, 26.0))
    outcome = substitute.plan(
        labels(), quiet, first=1, stop=4, window=WINDOW, source="s", drop_sustained=True
    )
    assert outcome.inserted == ()
    assert outcome.excluded == ()


def test_drop_sustained_refuses_a_word_that_only_crosses_the_window_edge() -> None:
    edge = (substitute.Word("前", 9.5, 10.5),)
    with pytest.raises(ValueError, match="not only a sustained vowel"):
        substitute.plan(
            labels(), edge, first=1, stop=4, window=WINDOW, source="s", drop_sustained=True
        )
