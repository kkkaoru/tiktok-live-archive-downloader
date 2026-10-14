import argparse
import json
import sys
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import audio_pcm_alignment as alignment
from replay.clocked_cut_audio import CutSlot
from replay.stages import file_digest


def test_context_requests_clip_real_edges_and_leave_identity_unchanged() -> None:
    slots = (
        CutSlot(Fraction(0), Fraction(1), 0, 1200, ((0, 1000),)),
        CutSlot(Fraction(1), Fraction(2), 1200, 2200, ((1000, 2000),)),
        CutSlot(Fraction(2), Fraction(3), 2200, 3400, ((2000, 3000),)),
    )
    result = alignment.context_slots(slots, samples=3000, context_samples=400)
    assert tuple(slot.spans for slot in result) == (((0, 1400),), ((1000, 2000),), ((1600, 3000),))


@pytest.mark.parametrize(
    "samples,margin,spans",
    [(0, 0, ((0, 1),)), (10, -1, ((0, 1),)), (10, 1, ((0, 20),)), (10, 1, ((0, 1), (2, 3)))],
)
def test_invalid_context_requests(
    samples: int, margin: int, spans: tuple[tuple[int, int], ...]
) -> None:
    slot = CutSlot(Fraction(0), Fraction(1), 0, 100, spans)
    with pytest.raises(ValueError, match="Context|context"):
        alignment.context_slots((slot,), samples=samples, context_samples=margin)


def write_pcm(path: Path, values: NDArray[np.float64]) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(values.shape[1])
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(np.rint(values * 32768).astype("<i2").tobytes())


def test_unique_stereo_match_preserves_opposite_channel_signal() -> None:
    signal = np.random.default_rng(1).normal(0, 0.1, 1000)
    reference = np.column_stack((signal, -signal))
    search = np.concatenate((np.zeros((300, 2)), reference * 0.5 + 0.05, np.zeros((400, 2))))
    result = alignment.match_window(reference, search, rate=8000)
    assert result.offset_samples == 300
    assert result.correlation == pytest.approx(1)
    assert result.reference_rms > 0.09
    assert result.candidate_rms > 0.04


@pytest.mark.parametrize("kind", ["silence", "dc", "unrelated", "periodic", "repeated", "edge"])
def test_rejects_unproved_correspondence(kind: str) -> None:
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 0.1, (1000, 1))
    search = rng.normal(0, 0.1, (4000, 1))
    reason = "low-correlation"
    if kind == "silence":
        reference[:] = 0
        reason = "reference-low-energy"
    elif kind == "dc":
        reference[:] = 0.2
        reason = "reference-low-energy"
    elif kind == "periodic":
        reference[:, 0] = np.sin(np.arange(1000) * 2 * np.pi / 400) * 0.1
        search[:, 0] = np.sin((np.arange(4000) - 123) * 2 * np.pi / 400) * 0.1
        reason = "ambiguous-match"
    elif kind == "repeated":
        search[500:1500] = reference
        search[2000:3000] = reference
        reason = "ambiguous-match"
    elif kind == "edge":
        search[:1000] = reference
        reason = "search-boundary"
    with pytest.raises(alignment.AlignmentRejected, match=reason):
        alignment.match_window(reference, search, rate=8000)


@pytest.mark.parametrize("rate", [0, 7999, True])
def test_bad_rate(rate: int) -> None:
    with pytest.raises(ValueError, match="windows"):
        alignment.match_window(np.zeros((10, 1)), np.zeros((20, 1)), rate=rate)


@pytest.mark.parametrize(
    "reference,search",
    [
        (np.zeros(10), np.zeros((20, 1))),
        (np.zeros((10, 2)), np.zeros((20, 1))),
        (np.zeros((10, 1)), np.zeros((10, 1))),
        (np.full((10, 1), np.nan), np.zeros((20, 1))),
    ],
)
def test_bad_arrays(reference: NDArray[np.float64], search: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="windows"):
        alignment.match_window(reference, search, rate=8000)


@pytest.mark.parametrize("value", [0, float("nan"), 11])
def test_bad_policy(value: float) -> None:
    with pytest.raises(ValueError, match="policy"):
        alignment.AlignmentPolicy(search_seconds=value)


def test_diagnostic_cli_and_immutable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    source = tmp_path / "source.wav"
    candidate = tmp_path / "candidate.wav"
    signal = np.random.default_rng(3).normal(0, 0.1, (80000, 1))
    write_pcm(source, signal)
    write_pcm(candidate, np.concatenate((np.zeros((400, 1)), signal)))
    result = tmp_path / "edits/work/result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alignment",
            str(source),
            str(candidate),
            str(result),
            "--reference-sha256",
            file_digest(source),
            "--candidate-sha256",
            file_digest(candidate),
            "--anchor",
            "4:4",
            "--anchor",
            "5:5",
        ],
    )
    alignment.main()
    raw = json.loads(result.read_text(encoding="utf-8"))
    assert raw["allAnchorsMatched"] is True
    assert raw["monotonic"] is True
    assert raw["mappingAccepted"] is False
    assert raw["anchors"][0]["candidateMinusReferenceSamples"] == 400
    with pytest.raises(FileExistsError):
        alignment.main()


def test_nonmonotonic_matches_are_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    source = tmp_path / "source.wav"
    candidate = tmp_path / "candidate.wav"
    signal = np.random.default_rng(4).normal(0, 0.1, (24000, 1))
    changed = signal.copy()
    changed[4000:12000], changed[12000:20000] = signal[12000:20000], signal[4000:12000]
    write_pcm(source, signal)
    write_pcm(candidate, changed)
    output = tmp_path / "edits/work/result.json"
    alignment.diagnose(
        source,
        candidate,
        output,
        anchors=((1, 2), (2, 1)),
        reference_sha256=file_digest(source),
        candidate_sha256=file_digest(candidate),
        policy=alignment.AlignmentPolicy(search_seconds=0.1),
    )
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["rejection"] == "non-monotonic-correspondence"
    assert raw["mappingAccepted"] is False


def test_unsupported_window_saved_without_fake_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    source = tmp_path / "source.wav"
    write_pcm(source, np.zeros((8000, 1)))
    output = tmp_path / "edits/work/result.json"
    alignment.diagnose(
        source,
        source,
        output,
        anchors=((0, 0),),
        reference_sha256=file_digest(source),
        candidate_sha256=file_digest(source),
    )
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["anchors"][0]["reason"] == "unsupported-window"
    assert raw["allAnchorsMatched"] is False


def test_bad_paths_and_anchors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="paths"):
        alignment.diagnose(
            tmp_path / "a",
            tmp_path / "b",
            tmp_path / "out",
            anchors=((1, 1),),
            reference_sha256="a",
            candidate_sha256="b",
        )
    with pytest.raises(ValueError, match="anchors"):
        alignment.diagnose(
            tmp_path / "a",
            tmp_path / "b",
            tmp_path / "edits/work/out",
            anchors=((2, 2), (1, 1)),
            reference_sha256="a",
            candidate_sha256="b",
        )
    with pytest.raises(ValueError, match="hash"):
        (tmp_path / "a").write_bytes(b"a")
        alignment.diagnose(
            tmp_path / "a",
            tmp_path / "a",
            tmp_path / "edits/work/out",
            anchors=((1, 1),),
            reference_sha256="wrong",
            candidate_sha256="wrong",
        )


def test_format_and_truncated_pcm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "short.wav"
    write_pcm(path, np.zeros((10, 1)))
    with wave.open(str(path), "rb") as reader:
        monkeypatch.setattr(reader, "readframes", lambda count: b"")
        with pytest.raises(ValueError, match="Truncated"):
            alignment._read_window(reader, start=0, count=10)
        monkeypatch.setattr(reader, "getsampwidth", lambda: 3)
        with pytest.raises(ValueError, match="PCM16"):
            alignment._format(reader)


def test_bad_anchor_syntax() -> None:
    with pytest.raises(ValueError):
        alignment._parse_anchor("x:y")
    with pytest.raises(argparse.ArgumentTypeError, match="reference:candidate"):
        alignment._parse_anchor("1")


def test_time_warp_does_not_pass_as_constant_shift() -> None:
    time = np.arange(4000) / 8000
    reference = (np.sin(2 * np.pi * (200 * time + 800 * time**2)) * 0.1)[:, None]
    warped_time = np.arange(4400) / 8800
    warped = (np.sin(2 * np.pi * (200 * warped_time + 800 * warped_time**2)) * 0.1)[:, None]
    search = np.concatenate((np.zeros((500, 1)), warped, np.zeros((500, 1))))
    with pytest.raises(alignment.AlignmentRejected, match="low-correlation"):
        alignment.match_window(reference, search, rate=8000)


def test_input_change_and_format_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    source = tmp_path / "source.wav"
    candidate = tmp_path / "candidate.wav"
    write_pcm(source, np.zeros((8000, 1)))
    write_pcm(candidate, np.zeros((8000, 2)))
    output = tmp_path / "edits/work/result.json"
    with pytest.raises(ValueError, match="channels"):
        alignment.diagnose(
            source,
            candidate,
            output,
            anchors=((0.5, 0.5),),
            reference_sha256=file_digest(source),
            candidate_sha256=file_digest(candidate),
        )
    with pytest.raises(ValueError, match="Conflicting"):
        alignment.diagnose(
            source,
            source,
            output,
            anchors=((0.5, 0.5),),
            reference_sha256="a",
            candidate_sha256="b",
        )
    digests = iter(("good", "changed"))
    monkeypatch.setattr(alignment, "file_digest", lambda path: next(digests))
    with pytest.raises(ValueError, match="changed"):
        alignment.diagnose(
            source,
            source,
            output,
            anchors=((0.5, 0.5),),
            reference_sha256="good",
            candidate_sha256="good",
        )


@pytest.fixture
def mapping_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path]:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "edits/work").mkdir(parents=True)
    source, candidate = tmp_path / "source.wav", tmp_path / "candidate.wav"
    signal = np.random.default_rng(8).normal(0, 0.1, (80000, 1))
    write_pcm(source, signal)
    write_pcm(candidate, np.concatenate((np.zeros((400, 1)), signal)))
    seed = tmp_path / "edits/work/seed.json"
    alignment.diagnose(
        source,
        candidate,
        seed,
        anchors=((0, 0), (4, 4)),
        reference_sha256=file_digest(source),
        candidate_sha256=file_digest(candidate),
    )
    clock = tmp_path / "clock.json"
    clock.write_text(
        json.dumps(
            {
                "streams": [{"sample_rate": "8000", "time_base": "1/8000"}],
                "frames": [{"pts": 0, "nb_samples": 40000}, {"pts": 40000, "nb_samples": 40000}],
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
                    {"start": {"value": 3, "timescale": 4}, "end": {"value": 3, "timescale": 1}},
                    {"start": {"value": 4, "timescale": 1}, "end": {"value": 6, "timescale": 1}},
                ],
            }
        ),
        encoding="utf-8",
    )
    return source, candidate, seed, selection


def test_mapping_cli_builds_measured_pieces(
    mapping_case: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, candidate, seed, selection = mapping_case
    output = seed.with_name("mapping.json")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alignment",
            str(source),
            str(candidate),
            str(output),
            "--reference-sha256",
            file_digest(source),
            "--candidate-sha256",
            file_digest(candidate),
            "--selection",
            str(selection),
            "--selection-sha256",
            file_digest(selection),
            "--seed-report",
            str(seed),
        ],
    )
    alignment.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["allCutsMeasured"] is True
    assert result["fullyMeasuredCuts"] == 2
    assert result["cuts"][0]["mappings"] == [
        {
            "referenceStart": 6000,
            "referenceEnd": 24000,
            "candidateStart": 6400,
            "candidateEnd": 24400,
        }
    ]
    assert result["mappingAccepted"] is False
    assert result["interpolationOutsideMeasuredWindows"] is False


def test_mapping_budget_leaves_unknown(mapping_case: tuple[Path, Path, Path, Path]) -> None:
    source, candidate, seed, selection = mapping_case
    output = seed.with_name("limited.json")
    alignment.map_selection(
        source,
        candidate,
        output,
        selection=selection,
        selection_sha256=file_digest(selection),
        seed_reports=(seed,),
        reference_sha256=file_digest(source),
        candidate_sha256=file_digest(candidate),
        maximum_windows=1,
        maximum_rate_change=0.5,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert len(result["observations"]) == 1
    assert result["allCutsMeasured"] is False
    assert result["cuts"][0]["unknownReferenceSpans"] == [[14000, 24000]]
    resumed = seed.with_name("resumed.json")
    alignment.map_selection(
        source,
        candidate,
        resumed,
        selection=selection,
        selection_sha256=file_digest(selection),
        seed_reports=(output,),
        reference_sha256=file_digest(source),
        candidate_sha256=file_digest(candidate),
        maximum_windows=10,
        maximum_rate_change=0.5,
    )
    continued = json.loads(resumed.read_text(encoding="utf-8"))
    assert continued["allCutsMeasured"] is True
    assert continued["observations"][0]["requestedReferenceStartSample"] == 14000


def test_invalid_resumed_report() -> None:
    with pytest.raises(ValueError, match="report"):
        alignment._resumed_support(
            {"inputs": {}}, pins={}, rate=8000, reference_samples=1000, candidate_samples=1000
        )
    with pytest.raises(ValueError, match="bounds"):
        alignment._resumed_support(
            {
                "inputs": {},
                "sampleRate": 8000,
                "mappingAccepted": False,
                "interpolationOutsideMeasuredWindows": False,
                "measuredSupport": [{"start": 1, "end": 20, "candidate_start": -1}],
            },
            pins={},
            rate=8000,
            reference_samples=1000,
            candidate_samples=1000,
        )


@pytest.mark.parametrize(
    "spans,reason",
    [
        ((alignment.MappedSpan(0, 10, 0), alignment.MappedSpan(5, 15, 7)), "inconsistent"),
        ((alignment.MappedSpan(0, 10, 100), alignment.MappedSpan(20, 30, 50)), "non-monotonic"),
        ((alignment.MappedSpan(-1, 10, 0),), "Invalid"),
    ],
)
def test_support_rejections(spans: tuple[alignment.MappedSpan, ...], reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        alignment.merge_support(spans)


def test_partial_cut_support_and_merge() -> None:
    slot = CutSlot(Fraction(0), Fraction(1), 0, 50, ((10, 60),))
    spans = (
        alignment.MappedSpan(0, 5, 100),
        alignment.MappedSpan(20, 30, 120),
        alignment.MappedSpan(30, 40, 130),
        alignment.MappedSpan(80, 90, 180),
    )
    result = alignment.cut_support(slot, spans)
    assert result["unknownReferenceSpans"] == [[10, 20], [40, 60]]
    assert result["mappings"] == [
        {"referenceStart": 20, "referenceEnd": 40, "candidateStart": 120, "candidateEnd": 140}
    ]


def test_scan_low_energy_and_empty_seed(tmp_path: Path) -> None:
    source, candidate = tmp_path / "source.wav", tmp_path / "candidate.wav"
    write_pcm(source, np.zeros((32000, 1)))
    write_pcm(candidate, np.zeros((32000, 1)))
    slot = CutSlot(Fraction(1), Fraction(2), 0, 8000, ((8000, 16000),))
    with wave.open(str(source), "rb") as left, wave.open(str(candidate), "rb") as right:
        support, rows = alignment.scan_required_support(
            left, right, slots=(slot,), seeds=(alignment.MappedSpan(0, 100, 0),), maximum_windows=1
        )
        assert rows[0]["reason"] == "reference-low-energy"
        assert support == (alignment.MappedSpan(0, 100, 0),)
        with pytest.raises(ValueError, match="seeds"):
            alignment.scan_required_support(left, right, slots=(slot,), seeds=(), maximum_windows=1)
        with pytest.raises(ValueError, match="budget"):
            alignment.scan_required_support(left, right, slots=(slot,), seeds=(), maximum_windows=0)


@pytest.mark.parametrize(
    "field,value", [("inputs", {}), ("policy", {}), ("anchors", None), ("anchors", [])]
)
def test_seed_document_rejections(
    mapping_case: tuple[Path, Path, Path, Path], field: str, value: object
) -> None:
    source, candidate, seed, _ = mapping_case
    raw = json.loads(seed.read_text(encoding="utf-8"))
    raw[field] = value
    seed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        alignment._seed_support(
            (seed,),
            pins={str(source): file_digest(source), str(candidate): file_digest(candidate)},
            rate=8000,
            reference_samples=80000,
            candidate_samples=80400,
        )


@pytest.mark.parametrize("field,value", [("referenceStartSample", "bad"), ("correlation", 0.1)])
def test_seed_measurement_rejections(
    mapping_case: tuple[Path, Path, Path, Path], field: str, value: object
) -> None:
    source, candidate, seed, _ = mapping_case
    raw = json.loads(seed.read_text(encoding="utf-8"))
    raw["anchors"][1][field] = value
    seed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        alignment._seed_support(
            (seed,),
            pins={str(source): file_digest(source), str(candidate): file_digest(candidate)},
            rate=8000,
            reference_samples=80000,
            candidate_samples=80400,
        )


@pytest.mark.parametrize("field,value", [("source", {}), ("clock", {}), ("intervals", None)])
def test_selection_rejections(
    mapping_case: tuple[Path, Path, Path, Path], field: str, value: object
) -> None:
    source, _, _, selection = mapping_case
    raw = json.loads(selection.read_text(encoding="utf-8"))
    raw[field] = value
    selection.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        alignment._selection_slots(
            selection,
            sha256=file_digest(selection),
            reference=source,
            reference_sha256=file_digest(source),
            rate=8000,
            samples=80000,
            maximum_rate_change=0.5,
        )


def test_cli_requires_diagnostic_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["alignment", "a", "b", "c", "--reference-sha256", "a", "--candidate-sha256", "b"],
    )
    with pytest.raises(SystemExit):
        alignment.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alignment",
            "a",
            "b",
            "c",
            "--reference-sha256",
            "a",
            "--candidate-sha256",
            "b",
            "--selection",
            "s",
        ],
    )
    with pytest.raises(SystemExit):
        alignment.main()
