import json
import sys
import wave
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

from replay import mixed_tempo_audio as job
from replay.audio_pcm_alignment import MappedSpan
from replay.clocked_cut_audio import CutSlot
from replay.stages import file_digest


def test_center_support_does_not_approve_missing_context() -> None:
    slot = CutSlot(F(1), F(2), 0, 12000, ((8000, 16000),))
    choice = job.choose_cut(
        slot,
        support=(MappedSpan(8000, 16000, 8400),),
        reference_samples=24000,
        candidate_samples=24400,
        context_samples=4000,
    )
    assert choice.mapped is None
    assert choice.reason == "unmeasured-center-or-context"


@pytest.mark.parametrize(
    "spans,reference,candidate,reason",
    [
        (((0, 100), (200, 300)), 1000, 1000, "noncontiguous-reference"),
        (((0, 100),), 1000, 1000, "unverified-source-edge-context"),
        (((400, 800),), 1000, 850, "candidate-context-out-of-bounds"),
    ],
)
def test_fallback_reasons(
    spans: tuple[tuple[int, int], ...], reference: int, candidate: int, reason: str
) -> None:
    choice = job.choose_cut(
        CutSlot(F(0), F(1), 0, 600, spans),
        support=(MappedSpan(0, 1000, 100),),
        reference_samples=reference,
        candidate_samples=candidate,
        context_samples=100,
    )
    assert choice.mapped is None
    assert choice.reason == reason


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    source, stem, fallback = (
        tmp_path / "reference.wav",
        tmp_path / "stem.wav",
        tmp_path / "fallback.wav",
    )
    with wave.open(str(source), "wb") as writer:
        writer.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(np.tile(np.array([1000, -1000], dtype="<i2"), (24000, 1)).tobytes())
    with wave.open(str(stem), "wb") as writer:
        writer.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(np.tile(np.array([2000, -2000], dtype="<i2"), (24400, 1)).tobytes())
    with wave.open(str(fallback), "wb") as writer:
        writer.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(np.tile(np.array([3000, -3000], dtype="<i2"), (16000, 1)).tobytes())
    clock = tmp_path / "clock.json"
    clock.write_text(
        json.dumps(
            {
                "streams": [{"sample_rate": "8000", "time_base": "1/8000"}],
                "frames": [{"pts": 0, "nb_samples": 24000}],
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "version": 1,
                "coordinateSpace": "native-presentation-seconds",
                "source": {"path": str(source), "sha256": file_digest(source)},
                "clock": {"path": str(clock), "sha256": file_digest(clock)},
                "intervals": [
                    {"start": {"value": 1, "timescale": 2}, "end": {"value": 3, "timescale": 2}},
                    {"start": {"value": 2, "timescale": 1}, "end": {"value": 3, "timescale": 1}},
                ],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "mapping.json"
    report.write_text(
        json.dumps(
            {
                "kind": "selected-pcm-correspondence-diagnostic",
                "inputs": {
                    str(source): file_digest(source),
                    str(stem): file_digest(stem),
                    str(selection): file_digest(selection),
                    str(clock): file_digest(clock),
                },
                "sampleRate": 8000,
                "mappingAccepted": False,
                "interpolationOutsideMeasuredWindows": False,
                "measuredSupport": [{"start": 4000, "end": 12000, "candidate_start": 4400}],
            }
        ),
        encoding="utf-8",
    )
    fallback_receipt = tmp_path / "fallback-receipt.json"
    fallback_receipt.write_text(
        json.dumps(
            {
                "kind": "context-atempo-exact-cut-slot-candidate",
                "output": str(fallback),
                "outputSHA256": file_digest(fallback),
                "cuts": [
                    {
                        "source_start_sample": 4000,
                        "source_end_sample": 12000,
                        "output_start_sample": 0,
                        "output_end_sample": 8000,
                    },
                    {
                        "source_start_sample": 16000,
                        "source_end_sample": 24000,
                        "output_start_sample": 8000,
                        "output_end_sample": 16000,
                    },
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
                "reference": {"path": str(source), "sha256": file_digest(source)},
                "candidateMixture": {"path": str(stem), "sha256": file_digest(stem)},
                "candidateStem": {"path": str(stem), "sha256": file_digest(stem)},
                "fallback": {"path": str(fallback), "sha256": file_digest(fallback)},
                "fallbackReceipt": {
                    "path": str(fallback_receipt),
                    "sha256": file_digest(fallback_receipt),
                },
                "selection": {"path": str(selection), "sha256": file_digest(selection)},
                "mapping": {"path": str(report), "sha256": file_digest(report)},
                "evidence": [{"path": str(clock), "sha256": file_digest(clock)}],
                "maximumRateChange": 0.5,
            }
        ),
        encoding="utf-8",
    )
    return config


def test_cli_preflight_and_exact_mixed_output(case: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = Path("edits/work/candidate.wav")
    preflight = Path("edits/work/preflight.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mixed",
            str(case),
            str(output),
            str(preflight),
            "--ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "--check",
        ],
    )
    job.main()
    receipt = json.loads(preflight.read_text(encoding="utf-8"))
    assert receipt["newStemCuts"] == 1
    assert receipt["fallbackCuts"] == 1
    assert receipt["cuts"][1]["sourceClock"] == "selected-output"
    assert output.exists() is False
    job.run(case, output, Path("edits/work/receipt.json"), ffmpeg=Path("/opt/homebrew/bin/ffmpeg"))
    with wave.open(str(output), "rb") as reader:
        assert reader.getnframes() == 16000
        values = np.frombuffer(reader.readframes(16000), dtype="<i2").reshape(-1, 2)
    assert np.unique(values[:8000], axis=0).tolist() == [[2000, -2000]]
    assert np.unique(values[8000:], axis=0).tolist() == [[3000, -3000]]
    final = json.loads(Path("edits/work/receipt.json").read_text(encoding="utf-8"))
    assert final["qualityAccepted"] is False
    assert final["cuts"][1]["sourceSpans"] == [[8000, 16000]]


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("version", 2, "Unsupported"),
        ("evidence", [], "evidence"),
        ("maximumRateChange", True, "rate policy"),
        ("reference", {}, "pinned input"),
        ("mapping", {"path": "absent", "sha256": "bad"}, "No such file"),
    ],
)
def test_bad_config(case: Path, field: str, value: object, reason: str) -> None:
    raw = json.loads(case.read_text(encoding="utf-8"))
    raw[field] = value
    case.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match=reason):
        job.prepare(case)


def test_changed_pin(case: Path) -> None:
    raw = json.loads(case.read_text(encoding="utf-8"))
    raw["reference"]["sha256"] = "bad"
    case.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        job.prepare(case)


@pytest.mark.parametrize(
    "field,value", [("outputSHA256", "bad"), ("cuts", [{"source_start_sample": 0}, {}])]
)
def test_fallback_clock_provenance_required(case: Path, field: str, value: object) -> None:
    config = json.loads(case.read_text(encoding="utf-8"))
    path = Path(config["fallbackReceipt"]["path"])
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt[field] = value
    path.write_text(json.dumps(receipt), encoding="utf-8")
    config["fallbackReceipt"]["sha256"] = file_digest(path)
    case.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Fallback receipt"):
        job.prepare(case)


def test_output_escape_and_existing(case: Path) -> None:
    with pytest.raises(ValueError, match="work directory"):
        job.run(case, Path("candidate.wav"), Path("edits/work/r.json"), ffmpeg=Path("ffmpeg"))
    with pytest.raises(ValueError, match="checkout"):
        job.prepare(case.parent.parent / "outside.json")


def test_noncontiguous_output_rejected(case: Path) -> None:
    prepared = job.prepare(case)
    slot = CutSlot(F(0), F(1), 1, 8001, ((0, 8000),))
    altered = job.Prepared(
        (job.Choice(slot, None, "unknown"),),
        prepared.stem,
        prepared.fallback,
        prepared.rate,
        prepared.channels,
        prepared.pins,
    )
    with pytest.raises(ValueError, match="Noncontiguous"):
        job.render(altered, Path("edits/work/bad.wav"), ffmpeg=Path("ffmpeg"))


def test_empty_render_rejected(case: Path) -> None:
    prepared = job.prepare(case)
    empty = job.Prepared(
        (), prepared.stem, prepared.fallback, prepared.rate, prepared.channels, prepared.pins
    )
    with pytest.raises(ValueError, match="Empty"):
        job.render(empty, Path("edits/work/empty.wav"), ffmpeg=Path("ffmpeg"))


def test_pitch_preserved_new_cut_followed_by_exact_stereo_fallback(case: Path) -> None:
    prepared = job.prepare(case)
    time = np.arange(32000) / 8000
    tones = np.column_stack((np.sin(2 * np.pi * 440 * time), np.sin(2 * np.pi * 660 * time)))
    with wave.open(str(prepared.stem), "wb") as writer:
        writer.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(np.rint(tones * 20000).astype("<i2").tobytes())
    first = CutSlot(F(1), F(5, 2), 0, 12000, ((8000, 16000),))
    second = CutSlot(F(3), F(7, 2), 12000, 16000, ((24000, 28000),))
    selected = job.choose_cut(
        first,
        support=(MappedSpan(0, 32000, 0),),
        reference_samples=32000,
        candidate_samples=32000,
        context_samples=4000,
    )
    altered = job.Prepared(
        (selected, job.Choice(second, None, "unmeasured-center-or-context")),
        prepared.stem,
        prepared.fallback,
        8000,
        2,
        prepared.pins,
    )
    rows = job.render(
        altered, Path("edits/work/pitch.wav"), ffmpeg=Path("/opt/homebrew/bin/ffmpeg")
    )
    with wave.open("edits/work/pitch.wav", "rb") as reader:
        values = np.frombuffer(reader.readframes(16000), dtype="<i2").reshape(-1, 2)
    frequencies = np.fft.rfftfreq(12000, 1 / 8000)
    assert abs(frequencies[np.argmax(abs(np.fft.rfft(values[:12000, 0])))] - 440) < 2
    assert abs(frequencies[np.argmax(abs(np.fft.rfft(values[:12000, 1])))] - 660) < 2
    assert np.unique(values[12000:], axis=0).tolist() == [[3000, -3000]]
    assert rows[1]["sourceSpans"] == ((12000, 16000),)


@pytest.mark.parametrize(
    "damage,reason",
    [("length", "exact selected"), ("format", "formats differ"), ("kind", "measurement report")],
)
def test_lineage_mismatch(case: Path, damage: str, reason: str) -> None:
    raw = json.loads(case.read_text(encoding="utf-8"))
    if damage == "kind":
        path = Path(raw["mapping"]["path"])
        report = json.loads(path.read_text(encoding="utf-8"))
        report["kind"] = "unverified"
        path.write_text(json.dumps(report), encoding="utf-8")
        raw["mapping"]["sha256"] = file_digest(path)
    else:
        path = Path(raw["fallback"]["path"])
        with wave.open(str(path), "wb") as writer:
            writer.setparams(
                (2, 2, 16000 if damage == "format" else 8000, 0, "NONE", "not compressed")
            )
            writer.writeframes(bytes(400))
        raw["fallback"]["sha256"] = file_digest(path)
    case.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        job.prepare(case)
