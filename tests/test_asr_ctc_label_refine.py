"""CTC label refinement replaces only a supported run, derives ends honestly and is verifiable."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay import asr_ctc_label_refine as refine
from replay import asr_ctc_label_refine_verify as verify
from replay.caption_repair import ClockLabel
from replay.recordings import record

FRAME = Fraction(1, 30)
OFFSET = Fraction(1, 2)


def label(text: str, start: float, end: float) -> ClockLabel:
    return ClockLabel(text, start, end, f"p:{text}")


def token(text: str, onset: float, *, weak: bool = False) -> refine.Token:
    return refine.Token(text, onset, -1.5, weak)


def labels() -> tuple[ClockLabel, ...]:
    return (
        label("A", 0.0, 1.0),
        label("あ", 1.0, 1.4),
        label("い", 1.4, 1.8),
        label("う", 1.8, 2.2),
        label("え", 2.2, 2.6),
        label("B", 2.6, 3.0),
    )


def aligned() -> list[refine.Token]:
    return [
        token("あ", 1.6),
        token("い", 2.0),
        token("う", 2.5),
        token("え", 2.9),
    ]


def test_supported_run_is_replaced_with_derived_ends() -> None:
    outcome = refine.refine(labels(), aligned(), first=1, stop=5, offset=OFFSET, frame=FRAME)
    rows = [(r["text"], r["start"], r["end"]) for r in outcome.selected]
    assert rows == [("あ", 1.1, 1.5), ("い", 1.5, 2.0), ("う", 2.0, 2.4), ("え", 2.4, 2.6)]
    assert [r["timing"] for r in outcome.selected][-1] == {
        "onset": "ctc-peak",
        "end": "original-asr-end",
    }
    assert [r["text"] for r in outcome.labels] == ["A", "あ", "い", "う", "え", "B"]


def test_a_weak_token_keeps_its_original_label_and_splits_the_run() -> None:
    weak = [token("あ", 1.6), token("い", 2.0, weak=True), token("う", 2.5), token("え", 2.9)]
    outcome = refine.refine(labels(), weak, first=1, stop=5, offset=OFFSET, frame=FRAME)
    assert [r["text"] for r in outcome.selected] == ["う", "え"]
    assert outcome.labels[1] == {"text": "あ", "start": 1.0, "end": 1.4, "provenance": "p:あ"}


def test_last_token_lasts_one_frame_when_the_original_end_is_too_early() -> None:
    items = (label("あ", 1.0, 1.01), label("B", 2.0, 3.0))
    outcome = refine.refine(items, [token("あ", 1.5)], first=0, stop=1, offset=OFFSET, frame=FRAME)
    assert outcome.selected[0]["timing"] == {"onset": "ctc-peak", "end": "onset-plus-one-frame"}
    assert outcome.selected[0]["end"] == pytest.approx(1.0 + 1 / 30)


def test_a_run_that_cannot_fit_is_reported_not_replaced() -> None:
    items = (label("あ", 1.0, 1.01), label("B", 1.01, 3.0))
    outcome = refine.refine(items, [token("あ", 1.5)], first=0, stop=1, offset=OFFSET, frame=FRAME)
    assert outcome.report["replaced"] is False
    assert [r["text"] for r in outcome.labels] == ["あ", "B"]


def test_several_tokens_over_one_label_time_it_by_the_first_token() -> None:
    items = (label("しい", 1.0, 1.4), label("B", 1.4, 2.0))
    pieces = [token("し", 1.6), token("い", 1.8)]
    outcome = refine.refine(items, pieces, first=0, stop=1, offset=OFFSET, frame=FRAME)
    assert [(r["text"], r["start"]) for r in outcome.selected] == [("しい", 1.1)]
    assert outcome.selected[0]["ctc"] == {
        "tokenIndices": [0, 1],
        "meanLogProbabilities": [-1.5, -1.5],
        "weakSupport": False,
    }


def test_a_weak_token_inside_a_group_makes_the_whole_label_weak() -> None:
    items = (label("しい", 1.0, 1.4), label("B", 1.4, 2.0))
    pieces = [token("し", 1.6), token("い", 1.8, weak=True)]
    outcome = refine.refine(items, pieces, first=0, stop=1, offset=OFFSET, frame=FRAME)
    assert outcome.report["replaced"] is False


def test_merged_labels_become_one_aligned_word() -> None:
    items = (label("ら", 1.0, 1.2), label("れる", 1.2, 1.6), label("B", 1.6, 2.0))
    outcome = refine.refine(
        items, [token("られる", 1.6)], first=0, stop=2, offset=OFFSET, frame=FRAME
    )
    assert [(r["text"], r["provenance"]) for r in outcome.selected] == [("られる", "p:ら")]


def test_text_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="differs from the declared label block"):
        refine.refine(labels(), [token("x", 1.6)], first=1, stop=5, offset=OFFSET, frame=FRAME)


def test_invalid_block_is_rejected() -> None:
    with pytest.raises(ValueError, match="bounded label block"):
        refine.refine(labels(), aligned(), first=3, stop=3, offset=OFFSET, frame=FRAME)


def test_duplicate_onset_removal_is_declared_and_checked() -> None:
    items = (label("a", 1.0, 1.2), label("a", 1.0, 1.3), label("B", 2.0, 3.0))
    outcome = refine.refine(
        items, [token("B", 2.5)], first=2, stop=3, offset=OFFSET, frame=FRAME, delete=(1,)
    )
    assert [r["text"] for r in outcome.labels] == ["a", "B"]
    assert [d.end for d in outcome.deleted] == [1.3]
    distinct = (label("a", 1.0, 1.2), label("b", 1.5, 1.7), label("B", 2.0, 3.0))
    with pytest.raises(ValueError, match="identical onset"):
        refine.refine(
            distinct, [token("B", 2.5)], first=2, stop=3, offset=OFFSET, frame=FRAME, delete=(1,)
        )
    with pytest.raises(ValueError, match="out of range"):
        refine.refine(
            items, [token("B", 2.5)], first=2, stop=3, offset=OFFSET, frame=FRAME, delete=(9,)
        )


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def world(tmp_path: Path, *, deletes: list[int] | None = None) -> tuple[Path, Path]:
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
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "model": "m",
                "tokens": [
                    {
                        "token": t.text,
                        "peakStart": t.onset,
                        "lastFrameStart": t.onset,
                        "meanLogProbability": t.mean_log_probability,
                        "weakSupport": t.weak,
                    }
                    for t in aligned()
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
                "artifacts": {"labels": pin(original), "candidate": pin(candidate)},
                "clock": {"sourceAudioStart": "1/2", "frame": "1/30"},
                "block": {"firstLabel": 1, "stopLabel": 5},
                "deleteLabels": deletes or [],
            }
        ),
        encoding="utf-8",
    )
    return config, tmp_path / "refined.json"


def test_run_writes_a_candidate_that_the_verifier_proves(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    assert refine.run(config, output) == 6
    proof = verify.verify(output)
    assert proof == {"labels": 6, "replaced": 4, "carried": 0, "deleted": 0, "frame": "1/30"}
    document = json.loads(output.read_text(encoding="utf-8"))
    assert (document["qualityApproved"], document["speechAccuracyVerified"]) == (False, False)
    assert document["ctcRefinement"]["candidateOnly"] is True


def test_verifier_rejects_a_change_outside_the_declared_block(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["labels"][0]["start"] = 0.5
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="differs outside the declared edits"):
        verify.verify(output)


def test_verifier_rejects_a_label_that_cites_a_weak_or_other_token(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["ctcRefinement"]["selected"][1]["ctc"]["meanLogProbabilities"] = [-0.1]
    document["labels"][2]["ctc"]["meanLogProbabilities"] = [-0.1]
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="supported CTC token"):
        verify.verify(output)


def test_verifier_rejects_undeclared_document_fields(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["clock"] = "projected"
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Undeclared document fields"):
        verify.verify(output)


def test_verifier_rejects_a_label_shorter_than_one_frame(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["ctcRefinement"]["selected"][0]["end"] = (
        document["ctcRefinement"]["selected"][0]["start"] + 0.01
    )
    document["labels"][1]["end"] = document["labels"][1]["start"] + 0.01
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="less than one frame"):
        verify.verify(output)


def test_verifier_rejects_a_changed_pinned_input(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    refine.run(config, output)
    (tmp_path / "candidate.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned verification input changed"):
        verify.verify(output)


def test_run_refuses_existing_output_and_bad_configuration(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    output.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="new output"):
        refine.run(config, output)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["extra"] = 1
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported CTC refinement configuration"):
        refine.run(config, tmp_path / "other.json")


def test_run_applies_declared_duplicate_removals(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["deleteLabels"] = [1]
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the replaced run"):
        refine.run(config, output)


def test_candidate_tokens_must_be_well_formed() -> None:
    with pytest.raises(ValueError, match="no tokens"):
        refine.tokens({"tokens": []})
    with pytest.raises(ValueError, match="Invalid CTC token"):
        refine.tokens({"tokens": [{"token": "a", "peakStart": 1.0, "meanLogProbability": -1.0}]})
    with pytest.raises(ValueError, match="finite number"):
        refine.tokens(
            {
                "tokens": [
                    {
                        "token": "a",
                        "peakStart": True,
                        "meanLogProbability": -1.0,
                        "weakSupport": False,
                    }
                ]
            }
        )


def refined(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config, output = world(tmp_path)
    refine.run(config, output)
    return output, record(json.loads(output.read_text(encoding="utf-8")))


def rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return [record(item) for item in value]


def edits(document: dict[str, object]) -> tuple[list[dict[str, object]], dict[str, object]]:
    return rows(document["labels"]), record(document["ctcRefinement"])


def save(output: Path, document: dict[str, object]) -> None:
    output.write_text(json.dumps(document), encoding="utf-8")


def test_verifier_rejects_removed_labels_that_are_not_an_original_block(tmp_path: Path) -> None:
    output, document = refined(tmp_path)
    _, meta = edits(document)
    rows(meta["removed"])[0]["text"] = "x"
    save(output, document)
    with pytest.raises(ValueError, match="not a contiguous original block"):
        verify.verify(output)


def test_verifier_accepts_a_refinement_that_replaced_nothing(tmp_path: Path) -> None:
    output, document = refined(tmp_path)
    original = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    _, meta = edits(document)
    meta["removed"] = []
    meta["selected"] = []
    document["labels"] = original["labels"]
    save(output, document)
    assert verify.verify(output)["replaced"] == 0


def test_verifier_rejects_a_label_count_that_differs_from_the_edits(tmp_path: Path) -> None:
    output, document = refined(tmp_path)
    document["labels"] = rows(document["labels"])[:-1]
    save(output, document)
    with pytest.raises(ValueError, match="count differs"):
        verify.verify(output)


def test_verifier_rejects_replaced_labels_that_differ_from_the_declared_selection(
    tmp_path: Path,
) -> None:
    output, document = refined(tmp_path)
    labels_, _ = edits(document)
    labels_[2]["text"] = "ん"
    save(output, document)
    with pytest.raises(ValueError, match="differ from the declared selection"):
        verify.verify(output)


def test_verifier_rejects_deletions_that_do_not_match_the_original(tmp_path: Path) -> None:
    output, document = refined(tmp_path)
    _, meta = edits(document)
    meta["deletedIndices"] = [5]
    save(output, document)
    with pytest.raises(ValueError, match="Declared deletions differ"):
        verify.verify(output)


def test_verifier_rejects_overlapping_replaced_labels(tmp_path: Path) -> None:
    output, document = refined(tmp_path)
    labels_, meta = edits(document)
    selected = rows(meta["selected"])
    for index in (1, 2):
        labels_[index]["end"] = float(str(labels_[index]["end"])) + 0.3
        selected[index - 1]["end"] = float(str(selected[index - 1]["end"])) + 0.3
    save(output, document)
    with pytest.raises(ValueError, match="overlaps"):
        verify.verify(output)


def test_verifier_rejects_a_label_without_a_ctc_citation(tmp_path: Path) -> None:
    output, document = refined(tmp_path)
    labels_, meta = edits(document)
    record(rows(meta["selected"])[0]["ctc"])["tokenIndices"] = [99]
    record(labels_[1]["ctc"])["tokenIndices"] = [99]
    save(output, document)
    with pytest.raises(ValueError, match="cites no CTC token"):
        verify.verify(output)


def test_unvoiced_symbols_inside_labels_do_not_block_the_match() -> None:
    items = (label("ミルフィー", 1.0, 1.6), label("B", 2.0, 3.0))
    outcome = refine.refine(
        items, [token("ミルフィ", 1.6)], first=0, stop=1, offset=OFFSET, frame=FRAME
    )
    assert [(r["text"], r["start"]) for r in outcome.selected] == [("ミルフィー", 1.1)]


def test_unvoiced_only_label_between_tokens_is_carried_unchanged() -> None:
    items = (
        label("あ", 1.0, 1.4),
        label("、", 1.4, 1.5),
        label("い", 1.5, 1.9),
        label("B", 2.5, 3.0),
    )
    outcome = refine.refine(
        items, [token("あ", 1.6), token("い", 2.0)], first=0, stop=3, offset=OFFSET, frame=FRAME
    )
    assert [(r["text"], r.get("carried", False)) for r in outcome.selected] == [
        ("あ", False),
        ("、", True),
        ("い", False),
    ]
    assert outcome.selected[1]["start"] == 1.4


def test_unvoiced_only_label_inside_one_token_makes_it_unmappable() -> None:
    items = (label("あ", 1.0, 1.4), label("、", 1.4, 1.5), label("い", 1.5, 1.9))
    assert refine.units(items, [token("あい", 1.6)]) == [None]


def test_token_edges_crossing_label_edges_are_unmappable() -> None:
    items = (label("あい", 1.0, 1.4), label("う", 1.4, 1.8))
    assert refine.units(items, [token("あ", 1.5), token("いう", 1.7)]) == [None]


def test_skip_reports_which_rule_refused_the_run() -> None:
    items = (label("あ", 1.0, 1.01), label("B", 1.01, 3.0))
    outcome = refine.refine(items, [token("あ", 1.5)], first=0, stop=1, offset=OFFSET, frame=FRAME)
    assert outcome.report["failures"] == {"Aligned token would display for less than one frame": 1}
    assert (outcome.report["weakTokens"], outcome.report["unmappedChunks"]) == (0, 0)


def test_deletion_only_configuration_needs_no_candidate(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    doc = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    doc["labels"][2]["start"] = doc["labels"][1]["start"]
    (tmp_path / "labels.json").write_text(json.dumps(doc), encoding="utf-8")
    data = json.loads(config.read_text(encoding="utf-8"))
    data["artifacts"] = {"labels": pin(tmp_path / "labels.json")}
    data["block"] = None
    data["deleteLabels"] = [2]
    config.write_text(json.dumps(data), encoding="utf-8")
    assert refine.run(config, output) == 5
    assert verify.verify(output) == {
        "labels": 5,
        "replaced": 0,
        "carried": 0,
        "deleted": 1,
        "frame": "1/30",
    }


def test_deletion_only_configuration_must_declare_deletions(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["artifacts"] = {"labels": pin(tmp_path / "labels.json")}
    data["block"] = None
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="must declare deletions"):
        refine.run(config, output)


def test_verifier_rejects_a_deleted_label_that_does_not_share_the_onset(tmp_path: Path) -> None:
    config, output = world(tmp_path)
    doc = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
    doc["labels"][2]["start"] = doc["labels"][1]["start"]
    (tmp_path / "labels.json").write_text(json.dumps(doc), encoding="utf-8")
    data = json.loads(config.read_text(encoding="utf-8"))
    data["artifacts"] = {"labels": pin(tmp_path / "labels.json")}
    data["block"] = None
    data["deleteLabels"] = [2]
    config.write_text(json.dumps(data), encoding="utf-8")
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["ctcRefinement"]["deletedIndices"] = [3]
    document["ctcRefinement"]["deleted"] = [doc["labels"][3]]
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="predecessor's onset"):
        verify.verify(output)


def weak_middle() -> list[refine.Token]:
    return [
        token("あ", 1.6),
        token("い", 2.0, weak=True),
        token("う", 2.5),
        token("え", 2.9),
    ]


def test_merge_weak_folds_a_bridged_weak_label_into_the_strong_one_before_it() -> None:
    outcome = refine.refine(
        labels(), weak_middle(), first=1, stop=5, offset=OFFSET, frame=FRAME, merge_weak=True
    )
    rows = [(r["text"], r["start"], r["end"]) for r in outcome.selected]
    assert rows == [("あい", 1.1, 2.0), ("う", 2.0, 2.4), ("え", 2.4, 2.6)]
    first = record(outcome.selected[0]["ctc"])
    assert first["tokenIndices"] == [0, 1]
    assert first["unusedWeak"] == [1]
    assert first["weakSupport"] is False
    assert "unusedWeak" not in record(outcome.selected[1]["ctc"])
    assert [r["text"] for r in outcome.labels] == ["A", "あい", "う", "え", "B"]


def test_without_merge_weak_the_weak_label_still_splits_the_run() -> None:
    outcome = refine.refine(labels(), weak_middle(), first=1, stop=5, offset=OFFSET, frame=FRAME)
    assert [r["text"] for r in outcome.selected] == ["う", "え"]


def test_a_leading_weak_label_is_never_folded_or_timed() -> None:
    leading = [
        token("あ", 1.6, weak=True),
        token("い", 2.0),
        token("う", 2.5),
        token("え", 2.9),
    ]
    outcome = refine.refine(
        labels(), leading, first=1, stop=5, offset=OFFSET, frame=FRAME, merge_weak=True
    )
    assert [r["text"] for r in outcome.selected] == ["い", "う", "え"]
    assert [r["text"] for r in outcome.labels] == ["A", "あ", "い", "う", "え", "B"]


def test_a_trailing_weak_label_with_no_strong_label_after_it_is_not_folded() -> None:
    trailing = [
        token("あ", 1.6),
        token("い", 2.0),
        token("う", 2.5),
        token("え", 2.9, weak=True),
    ]
    outcome = refine.refine(
        labels(), trailing, first=1, stop=5, offset=OFFSET, frame=FRAME, merge_weak=True
    )
    assert [r["text"] for r in outcome.selected] == ["あ", "い", "う"]


def test_a_merged_label_that_would_be_too_coarse_is_not_produced() -> None:
    gap = [token("あ", 1.6), token("い", 5.0, weak=True), token("う", 5.5), token("え", 5.9)]
    outcome = refine.refine(
        labels(), gap, first=1, stop=5, offset=OFFSET, frame=FRAME, merge_weak=True
    )
    assert [(r["text"], r["start"], r["end"]) for r in outcome.selected] == [("あい", 1.1, 1.8)]
    mapped = refine.fold_weak(refine.units(labels()[1:5], gap), gap)
    times = {0: 1.1, 1: 5.0}
    with pytest.raises(ValueError, match="too coarse for focused captions"):
        refine.selected_labels(labels(), gap, mapped, times, (0, 2), first=1, frame=FRAME, limit=5)


def test_a_gap_holding_another_label_blocks_the_fold() -> None:
    strong = Unit(0, 1, 0, 1, 1)
    weak = Unit(2, 3, 1, 2, 2)
    after = Unit(3, 4, 2, 3, 3)
    tokens_ = [token("あ", 1.0), token("い", 2.0, weak=True), token("う", 3.0)]
    assert refine.fold_weak([strong, weak, after], tokens_) == [strong, weak, after]
    assert refine.fold_weak([strong, None, after], tokens_) == [strong, None, after]


Unit = refine.Unit


def merged_world(tmp_path: Path) -> tuple[Path, Path]:
    config, output = world(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.unlink()
    candidate.write_text(
        json.dumps(
            {
                "model": "m",
                "tokens": [
                    {
                        "token": t.text,
                        "peakStart": t.onset,
                        "lastFrameStart": t.onset,
                        "meanLogProbability": t.mean_log_probability,
                        "weakSupport": t.weak,
                    }
                    for t in weak_middle()
                ],
            }
        ),
        encoding="utf-8",
    )
    document = json.loads(config.read_text(encoding="utf-8"))
    document["artifacts"]["candidate"] = pin(candidate)
    document["block"]["mergeWeak"] = True
    config.unlink()
    config.write_text(json.dumps(document), encoding="utf-8")
    return config, output


def test_merged_refinement_is_written_and_proved_by_the_verifier(tmp_path: Path) -> None:
    config, output = merged_world(tmp_path)
    assert refine.run(config, output) == 5
    assert verify.verify(output) == {
        "labels": 5,
        "replaced": 3,
        "carried": 0,
        "deleted": 0,
        "frame": "1/30",
    }
    assert json.loads(output.read_text(encoding="utf-8"))["ctcRefinement"]["mergeWeak"] is True


def test_verifier_rejects_unused_weak_tokens_when_merge_weak_was_not_declared(
    tmp_path: Path,
) -> None:
    config, output = merged_world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["ctcRefinement"]["mergeWeak"] = False
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="without mergeWeak"):
        verify.verify(output)


def test_verifier_rejects_a_merged_label_that_times_with_a_weak_token(tmp_path: Path) -> None:
    config, output = merged_world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    del document["ctcRefinement"]["selected"][0]["ctc"]["unusedWeak"]
    document["labels"][1]["ctc"] = document["ctcRefinement"]["selected"][0]["ctc"]
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match its supported CTC token"):
        verify.verify(output)


def test_verifier_rejects_marking_a_supported_token_as_unused(tmp_path: Path) -> None:
    config, output = merged_world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["ctcRefinement"]["selected"][0]["ctc"]["unusedWeak"] = [0, 1]
    document["labels"][1]["ctc"] = document["ctcRefinement"]["selected"][0]["ctc"]
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="supported token as unused"):
        verify.verify(output)


def test_verifier_rejects_malformed_unused_weak_citations(tmp_path: Path) -> None:
    config, output = merged_world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["ctcRefinement"]["selected"][0]["ctc"]["unusedWeak"] = "x"
    document["labels"][1]["ctc"] = document["ctcRefinement"]["selected"][0]["ctc"]
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid unused weak tokens"):
        verify.verify(output)


def test_run_rejects_unknown_block_options_and_non_boolean_merge_weak(tmp_path: Path) -> None:
    config, output = merged_world(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["block"]["mergeWeak"] = "yes"
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="mergeWeak and foldUnvoiced must be booleans"):
        refine.run(config, output)
    document["block"] = {"firstLabel": 1, "stopLabel": 5, "other": True}
    config.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported block option"):
        refine.run(config, output)


def long_vowel_labels() -> tuple[ClockLabel, ...]:
    return (
        label("A", 0.0, 1.0),
        label("あ", 1.0, 1.4),
        label("ー", 1.4, 1.5),
        label("い", 1.5, 1.9),
        label("B", 2.5, 3.0),
    )


def late_tokens() -> list[refine.Token]:
    return [token("あ", 2.0), token("い", 2.4)]


def test_a_carried_symbol_that_would_reverse_onsets_is_still_refused_by_default() -> None:
    with pytest.raises(ValueError, match="reverse word onsets"):
        refine.refine(
            long_vowel_labels(), late_tokens(), first=1, stop=4, offset=OFFSET, frame=FRAME
        )


def test_fold_unvoiced_appends_the_symbol_to_the_timed_label_before_it() -> None:
    outcome = refine.refine(
        long_vowel_labels(),
        late_tokens(),
        first=1,
        stop=4,
        offset=OFFSET,
        frame=FRAME,
        fold_unvoiced=True,
    )
    assert [(r["text"], r["start"], r.get("carried")) for r in outcome.selected] == [
        ("あー", 1.5, None),
        ("い", 1.9, None),
    ]
    assert outcome.selected[0]["foldedUnvoiced"] == [
        {"text": "ー", "start": 1.4, "end": 1.5, "provenance": "p:ー"}
    ]
    assert [r["text"] for r in outcome.labels] == ["A", "あー", "い", "B"]


def folded_world(tmp_path: Path) -> tuple[Path, Path]:
    document = {
        "version": 1,
        "clock": "input-pcm-not-projected",
        "parentSHA256": {},
        "humanReviewed": False,
        "labels": [
            {"text": t.text, "start": t.start, "end": t.end, "provenance": t.provenance}
            for t in long_vowel_labels()
        ],
    }
    original = tmp_path / "labels.json"
    original.write_text(json.dumps(document), encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "model": "m",
                "tokens": [
                    {
                        "token": t.text,
                        "peakStart": t.onset,
                        "lastFrameStart": t.onset,
                        "meanLogProbability": t.mean_log_probability,
                        "weakSupport": t.weak,
                    }
                    for t in late_tokens()
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
                "artifacts": {"labels": pin(original), "candidate": pin(candidate)},
                "clock": {"sourceAudioStart": "1/2", "frame": "1/30"},
                "block": {"firstLabel": 1, "stopLabel": 4, "foldUnvoiced": True},
                "deleteLabels": [],
            }
        ),
        encoding="utf-8",
    )
    return config, tmp_path / "refined.json"


def test_folded_refinement_is_written_and_proved_by_the_verifier(tmp_path: Path) -> None:
    config, output = folded_world(tmp_path)
    assert refine.run(config, output) == 4
    assert verify.verify(output) == {
        "labels": 4,
        "replaced": 2,
        "carried": 0,
        "deleted": 0,
        "frame": "1/30",
    }
    assert json.loads(output.read_text(encoding="utf-8"))["ctcRefinement"]["foldUnvoiced"] is True


def test_verifier_rejects_folded_labels_when_fold_unvoiced_was_not_declared(
    tmp_path: Path,
) -> None:
    config, output = folded_world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["ctcRefinement"]["foldUnvoiced"] = False
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="without foldUnvoiced"):
        verify.verify(output)


def test_verifier_rejects_a_folded_label_whose_text_lacks_the_folded_symbol(
    tmp_path: Path,
) -> None:
    config, output = folded_world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["ctcRefinement"]["selected"][0]["text"] = "あ"
    document["labels"][1]["text"] = "あ"
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="does not end with its folded unvoiced text"):
        verify.verify(output)


def test_verifier_rejects_a_dropped_folded_symbol(tmp_path: Path) -> None:
    config, output = folded_world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["ctcRefinement"]["selected"][0]["text"] = "あ"
    document["ctcRefinement"]["selected"][0]["foldedUnvoiced"] = []
    document["labels"][1]["text"] = "あ"
    document["labels"][1]["foldedUnvoiced"] = []
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="not carried over unchanged"):
        verify.verify(output)


def test_verifier_rejects_a_malformed_folded_list(tmp_path: Path) -> None:
    config, output = folded_world(tmp_path)
    refine.run(config, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["ctcRefinement"]["selected"][0]["foldedUnvoiced"] = "x"
    document["labels"][1]["foldedUnvoiced"] = "x"
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="without foldUnvoiced"):
        verify.verify(output)


def test_a_label_over_a_strong_and_a_weak_token_is_not_folded_as_a_weak_unit() -> None:
    items = (
        label("A", 0.0, 1.0),
        label("あ", 1.0, 1.4),
        label("いう", 1.4, 2.2),
        label("え", 2.2, 2.6),
        label("B", 2.6, 3.0),
    )
    tokens_ = [token("あ", 1.6), token("い", 2.0), token("う", 2.2, weak=True), token("え", 2.9)]
    mapped = refine.units(items[1:4], tokens_)
    assert refine.fold_weak(mapped, tokens_) == mapped
