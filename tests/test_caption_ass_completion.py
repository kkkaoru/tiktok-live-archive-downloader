"""ASS quality evidence must be measured and tied to the exact delivery."""

import json
from pathlib import Path

import pytest

from replay import caption_ass_completion as gate
from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import SpeakerCue, render_srt
from replay.stages import file_digest


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps({"animation": "none", "captions": [{"text": "はい", "start": 0, "end": 1}]})
    )
    cues = tmp_path / "cues.json"
    cues.write_text(
        json.dumps(
            {"version": 1, "clock": "output", "cues": [{"text": "はい", "start": 0, "end": 1}]}
        )
    )
    cuts = tmp_path / "cuts.json"
    cuts.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "source-presentation",
                "fps": 30,
                "cuts": [
                    {"source_start_frame": 0, "source_end_frame": 30, "output_start_frame": 0}
                ],
            }
        )
    )
    movie, ass = tmp_path / "movie.mp4", tmp_path / "captions.ass"
    movie.write_bytes(b"synthetic movie")
    ass.write_text("synthetic ass")
    transcript = tmp_path / "captions.srt"
    transcript.write_text(render_srt((SpeakerCue(0, 1, "unassigned", "はい"),)))
    energy = tmp_path / "energy.json"
    energy.write_text(
        json.dumps(
            {
                "planSHA256": file_digest(plan),
                "allCuesMeasured": True,
                "medianAbsoluteLagSeconds": 0.01,
                "p90AbsoluteLagSeconds": 0.1,
            }
        )
    )
    proof = tmp_path / "proof.json"
    gate.main(
        [
            str(proof),
            "--plan",
            str(plan),
            "--cues",
            str(cues),
            "--cuts",
            str(cuts),
            "--movie",
            str(movie),
            "--ass",
            str(ass),
            "--transcript",
            str(transcript),
            "--energy",
            str(energy),
        ]
    )
    return proof


def test_missing_measurements_never_pass(evidence: Path) -> None:
    failure, facts, placement = gate.assess(
        evidence, evidence.parent / "movie.mp4", evidence.parent / "captions.srt"
    )
    assert (
        failure == "missing measured color evidence; missing measured stability evidence; "
        "missing measured appearance evidence; missing measured safety evidence"
    )
    assert facts["transcriptionFailure"] is None
    assert facts["semanticAlignmentVerified"] is False
    assert placement["violations"] == [
        "missing measured color evidence",
        "missing measured stability evidence",
        "missing measured appearance evidence",
        "missing measured safety evidence",
    ]


def test_changed_media_is_rejected(evidence: Path) -> None:
    (evidence.parent / "movie.mp4").write_bytes(b"changed")
    with pytest.raises(ReplayError, match="identity"):
        gate.assess(evidence, evidence.parent / "movie.mp4", evidence.parent / "captions.srt")


@pytest.mark.parametrize("value", [True, "1", float("nan")])
def test_nonfinite_or_untyped_measurement_rejected(value: object) -> None:
    with pytest.raises(ReplayError, match="finite"):
        gate.number(value)


@pytest.mark.parametrize("value", [None, [], [3]])
def test_missing_measurement_rows_rejected(value: object) -> None:
    with pytest.raises((ReplayError, ValueError)):
        gate.rows(value)


def test_unbound_path_rejected() -> None:
    with pytest.raises(ReplayError, match="path/hash"):
        gate.bound_path({"path": "file"})


def test_cli_does_not_overwrite(evidence: Path) -> None:
    with pytest.raises(ReplayError, match="replace"):
        gate.main(
            [
                str(evidence),
                "--movie",
                str(evidence),
                "--transcript",
                str(evidence),
                "--plan",
                str(evidence),
                "--cues",
                str(evidence),
                "--cuts",
                str(evidence),
                "--ass",
                str(evidence),
                "--energy",
                str(evidence),
            ]
        )


@pytest.mark.parametrize(
    "kind,field,value",
    [
        ("color", "deltaE76", 7),
        ("stability", "existingGlyphMovementPx", 1),
        ("appearance", "relativeReferenceError", 0.2),
        ("color", "deltaE76", 0),
    ],
)
def test_measured_budget_and_exact_render_identity(
    evidence: Path, kind: str, field: str, value: float
) -> None:
    document = gate.read_document(evidence)
    measurement = evidence.parent / "measurement.json"
    measurement.write_text(
        json.dumps(
            {
                "movieSHA256": file_digest(evidence.parent / "movie.mp4"),
                "samples": [{field: value}],
            }
        )
    )
    document["measurements"] = {
        kind: {"path": str(measurement), "sha256": file_digest(measurement)}
    }
    evidence.write_text(json.dumps(document))
    failure, _, _ = gate.assess(
        evidence, evidence.parent / "movie.mp4", evidence.parent / "captions.srt"
    )
    assert failure is not None
    assert (f"measured {kind} exceeds its budget" in failure) == (value != 0)


@pytest.mark.parametrize("bottom,identity", [(1050, "valid"), (1060, "valid"), (1050, "wrong")])
def test_measured_geometry_checks_media_and_safe_region(
    evidence: Path, bottom: float, identity: str
) -> None:
    document = gate.read_document(evidence)
    measurement = evidence.parent / "geometry.json"
    measurement.write_text(
        json.dumps(
            {
                "movieSHA256": file_digest(evidence.parent / "movie.mp4")
                if identity == "valid"
                else "wrong",
                "safeLeft": 745,
                "safeRight": 1906,
                "safeBottom": 1056,
                "samples": [{"left": 800, "right": 1800, "bottom": bottom}],
            }
        )
    )
    document["measurements"] = {
        "geometry": {"path": str(measurement), "sha256": file_digest(measurement)}
    }
    evidence.write_text(json.dumps(document))
    failure, _, _ = gate.assess(
        evidence, evidence.parent / "movie.mp4", evidence.parent / "captions.srt"
    )
    assert failure is not None
    assert ("escapes" in failure) == (bottom > 1056)
    assert ("different media" in failure) == (identity == "wrong")


@pytest.mark.parametrize(
    "field,value",
    [
        ("animation", "growing"),
        ("backend", "motion"),
    ],
)
def test_wrong_backend_or_animation_never_passes(evidence: Path, field: str, value: str) -> None:
    document = gate.read_document(evidence)
    if field == "backend":
        document[field] = value
        evidence.write_text(json.dumps(document))
        with pytest.raises(ReplayError, match="backend"):
            gate.assess(evidence, evidence.parent / "movie.mp4", evidence.parent / "captions.srt")
        return
    plan = evidence.parent / "plan.json"
    document_plan = gate.read_document(plan)
    document_plan[field] = value
    plan.write_text(json.dumps(document_plan))
    record(record(document["artifacts"])["plan"])["sha256"] = file_digest(plan)
    evidence.write_text(json.dumps(document))
    failure, _, _ = gate.assess(
        evidence, evidence.parent / "movie.mp4", evidence.parent / "captions.srt"
    )
    assert failure is not None and "animation" in failure and "different phrase plan" in failure


def test_changed_delivered_transcript_fails(evidence: Path) -> None:
    transcript = evidence.parent / "captions.srt"
    transcript.write_text("wrong text")
    document = gate.read_document(evidence)
    record(record(document["artifacts"])["transcript"])["sha256"] = file_digest(transcript)
    evidence.write_text(json.dumps(document))
    failure, _, _ = gate.assess(evidence, evidence.parent / "movie.mp4", transcript)
    assert failure is not None and "SRT differs" in failure


def test_bounded_document_rejects_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.touch()
    with pytest.raises(ReplayError, match="bounded"):
        gate.read_document(empty)


@pytest.mark.parametrize(
    "field,value",
    [("allCuesMeasured", False), ("medianAbsoluteLagSeconds", 0.2), ("p90AbsoluteLagSeconds", 0.5)],
)
def test_energy_support_and_lag_budget_fail(evidence: Path, field: str, value: object) -> None:
    energy = evidence.parent / "energy.json"
    measured = gate.read_document(energy)
    measured[field] = value
    energy.write_text(json.dumps(measured))
    proof = gate.read_document(evidence)
    record(record(proof["artifacts"])["energy"])["sha256"] = file_digest(energy)
    evidence.write_text(json.dumps(proof))
    failure, _, _ = gate.assess(
        evidence, evidence.parent / "movie.mp4", evidence.parent / "captions.srt"
    )
    assert failure is not None and "onset lag" in failure


def test_short_nonterminal_phrase_fails(evidence: Path) -> None:
    plan = evidence.parent / "plan.json"
    document = gate.read_document(plan)
    document["captions"] = [
        {"text": "はい", "start": 0, "end": 0.1},
        {"text": "はい", "start": 1, "end": 2},
    ]
    plan.write_text(json.dumps(document))
    proof = gate.read_document(evidence)
    record(record(proof["artifacts"])["plan"])["sha256"] = file_digest(plan)
    evidence.write_text(json.dumps(proof))
    failure, _, _ = gate.assess(
        evidence, evidence.parent / "movie.mp4", evidence.parent / "captions.srt"
    )
    assert failure is not None and "six displayed frames" in failure and "text differs" in failure


def test_complete_measured_evidence_can_pass_without_semantic_claim(evidence: Path) -> None:
    movie = evidence.parent / "movie.mp4"
    color, stability = evidence.parent / "color.json", evidence.parent / "stability.json"
    appearance, geometry = evidence.parent / "appearance.json", evidence.parent / "geometry.json"
    color.write_text(
        json.dumps(
            {
                "movieSHA256": file_digest(movie),
                "samples": [
                    {"endpoint": "top", "lineCount": 1, "deltaE76": 0.5},
                    {"endpoint": "bottom", "lineCount": 1, "deltaE76": 1.0},
                    {"endpoint": "top", "lineCount": 2, "deltaE76": 0.5},
                    {"endpoint": "bottom", "lineCount": 2, "deltaE76": 1.0},
                    {"endpoint": "top", "lineCount": 3, "deltaE76": 0.5},
                    {"endpoint": "bottom", "lineCount": 3, "deltaE76": 1.0},
                ],
            }
        )
    )
    stability.write_text(
        json.dumps({"movieSHA256": file_digest(movie), "samples": [{"existingGlyphMovementPx": 0}]})
    )
    appearance.write_text(
        json.dumps(
            {"movieSHA256": file_digest(movie), "samples": [{"relativeReferenceError": 0.02}]}
        )
    )
    geometry.write_text(
        json.dumps(
            {
                "movieSHA256": file_digest(movie),
                "safeLeft": 745,
                "safeRight": 1906,
                "safeBottom": 1056,
                "samples": [{"left": 800, "right": 1800, "bottom": 1040}],
            }
        )
    )
    proof = gate.read_document(evidence)
    proof["measurements"] = {
        "color": {"path": str(color), "sha256": file_digest(color)},
        "stability": {"path": str(stability), "sha256": file_digest(stability)},
        "appearance": {"path": str(appearance), "sha256": file_digest(appearance)},
        "geometry": {"path": str(geometry), "sha256": file_digest(geometry)},
    }
    evidence.write_text(json.dumps(proof))
    failure, facts, placement = gate.assess(evidence, movie, evidence.parent / "captions.srt")
    assert failure is None
    assert facts["semanticAlignmentVerified"] is False
    assert placement == {"violations": []}
