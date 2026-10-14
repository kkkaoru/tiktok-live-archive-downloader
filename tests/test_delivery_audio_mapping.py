import json
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import pytest

from replay import delivery_audio_mapping as mapping
from replay.audio_clock import AudioFrameClock
from replay.continuous_cuts import retained_pcm_spans, write_pcm_spans
from replay.delivery_audio_mapping import cut_intervals, integer, placement, spans_from_json
from replay.stages import file_digest


def test_exact_mapping_and_gap_drift() -> None:
    clock = AudioFrameClock((0, 110, 200), (100, 100, 100), 10000, Fraction(1, 10000))
    result = placement(clock, ((0, 300),), rate=10000, cuts=((Fraction(0), Fraction(3, 100)),))
    assert result["nativeTimesPreserved"] is False
    assert result["durationDifference"] == "0"
    assert result["maximumAbsoluteDrift"] == "1/1000"
    assert result["pieces"] == [
        {
            "sourceStartSample": 0,
            "sourceEndSample": 100,
            "outputStartSample": 0,
            "outputEndSample": 100,
            "nativeStart": "0",
            "nativeEnd": "1/100",
            "cutIndex": 0,
            "expectedOutputStart": "0",
            "actualOutputStart": "0",
            "drift": "0",
            "withinNativeSupport": True,
        },
        {
            "sourceStartSample": 100,
            "sourceEndSample": 200,
            "outputStartSample": 100,
            "outputEndSample": 200,
            "nativeStart": "11/1000",
            "nativeEnd": "21/1000",
            "cutIndex": 0,
            "expectedOutputStart": "11/1000",
            "actualOutputStart": "1/100",
            "drift": "-1/1000",
            "withinNativeSupport": True,
        },
        {
            "sourceStartSample": 200,
            "sourceEndSample": 300,
            "outputStartSample": 200,
            "outputEndSample": 300,
            "nativeStart": "1/50",
            "nativeEnd": "3/100",
            "cutIndex": 0,
            "expectedOutputStart": "1/50",
            "actualOutputStart": "1/50",
            "drift": "0",
            "withinNativeSupport": True,
        },
    ]


def test_cut_mapping_and_unmapped_edges() -> None:
    clock = AudioFrameClock((0,), (300,), 10000, Fraction(1, 10000))
    exact = placement(
        clock,
        ((0, 100), (200, 300)),
        rate=10000,
        cuts=((Fraction(0), Fraction(1, 100)), (Fraction(1, 50), Fraction(3, 100))),
    )
    assert exact["nativeTimesPreserved"] is True
    assert exact["cuts"] == [
        {
            "cutIndex": 0,
            "expectedOutputStart": "0",
            "expectedOutputEnd": "1/100",
            "firstSelectedSampleDrift": "0",
            "lastSelectedSampleDrift": "0",
            "minimumDrift": "0",
            "maximumDrift": "0",
            "outputEndMinusExpectedCutEnd": "0",
        },
        {
            "cutIndex": 1,
            "expectedOutputStart": "1/100",
            "expectedOutputEnd": "1/50",
            "firstSelectedSampleDrift": "0",
            "lastSelectedSampleDrift": "0",
            "minimumDrift": "0",
            "maximumDrift": "0",
            "outputEndMinusExpectedCutEnd": "0",
        },
    ]
    bad = placement(
        clock, ((0, 300),), rate=10000, cuts=((Fraction(1, 100), Fraction(201, 20000)),)
    )
    assert bad["nativeTimesPreserved"] is False
    assert bad["selectedSamples"] == 300


@pytest.mark.parametrize("value", [True, -1, 1.0, None])
def test_bad_integer(value: object) -> None:
    with pytest.raises(ValueError):
        integer(value)


@pytest.mark.parametrize("value", [[], None, [1], [[1]], [[3, 2]], [[0, 3], [2, 4]]])
def test_bad_spans(value: object) -> None:
    with pytest.raises(ValueError):
        spans_from_json(value)


def test_parse_cuts() -> None:
    assert spans_from_json([[0, 2], [2, 4]]) == ((0, 2), (2, 4))
    assert cut_intervals(
        {"cuts": [{"source_start_frame": 0, "source_end_frame": 3, "output_start_frame": 0}]}, 30
    ) == ((Fraction(0), Fraction(1, 10)),)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [{"source_start_frame": 2, "source_end_frame": 1}],
        [{"source_start_frame": 0, "source_end_frame": 2, "output_start_frame": 1}],
    ],
)
def test_bad_cuts(value: object) -> None:
    with pytest.raises(ValueError):
        cut_intervals({"cuts": value}, 30)


def test_fractional_resampler_clock_is_not_approved() -> None:
    result = placement(
        AudioFrameClock((0,), (1,), 48000, Fraction(1, 48000)),
        ((0, 1),),
        rate=44100,
        cuts=((Fraction(0), Fraction(1)),),
    )
    assert result["sourceFrameBoundariesSampleAligned"] is False
    assert result["nativeTimesPreserved"] is False
    with pytest.raises(ValueError, match="exceed"):
        placement(
            AudioFrameClock((0,), (1,), 44100, Fraction(1, 44100)),
            ((0, 2),),
            rate=44100,
            cuts=((Fraction(0), Fraction(1)),),
        )


def metadata_fixture(
    root: Path, *, document: str = "", changes: dict[str, object] | None = None
) -> Path:
    """Synthetic metadata only: referenced waveform files intentionally do not exist."""

    def save(name: str, data: dict[str, object]) -> dict[str, str]:
        if document == name:
            data.update(changes or {})
        path = root / f"{name}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return {"path": str(path), "sha256": file_digest(path)}

    source = {"path": str(root / "absent-source.wav"), "sha256": "a" * 64}
    clock = save(
        "clock",
        {
            "streams": [{"sample_rate": "10000", "time_base": "1/10000"}],
            "frames": [{"pts": 0, "nb_samples": 100}, {"pts": 100, "nb_samples": 100}],
        },
    )
    cuts = save("cuts", {"cuts": [{"source_start_frame": 0, "source_end_frame": 2}]})
    producer = save(
        "producer",
        {
            "version": 1,
            "source": source,
            "clock": clock,
            "coordinateSpace": "native-presentation-seconds",
            "evidence": [],
            "intervals": [
                {"start": {"value": 0, "timescale": 100}, "end": {"value": 2, "timescale": 100}}
            ],
        },
    )
    pins = {ref["path"]: ref["sha256"] for ref in (source, clock, producer)}
    pcm: dict[str, object] = {
        "spans": [[0, 200]],
        "sampleRate": 10000,
        "sourceSamples": 200,
        "selectedSamples": 200,
        "channels": 2,
        "sampleWidth": 2,
    }
    selection = save(
        "selection",
        {
            "version": 1,
            "inputs": pins,
            "output": str(root / "absent-selected.wav"),
            "outputSHA256": "b" * 64,
            **pcm,
        },
    )
    audit = save(
        "audit",
        {
            "version": 2,
            "kind": "independent-pcm-selection-clock-audit",
            "inputs": {
                **pins,
                selection["path"]: selection["sha256"],
                str(root / "absent-selected.wav"): "b" * 64,
            },
            "selectedPCMBytesEqual": True,
            "comparedBytes": 800,
            "mismatchedBytes": 0,
            **pcm,
        },
    )
    render = save(
        "render",
        {
            "audio": {"source": source["path"], "clock": clock["path"]},
            "cuts": cuts["path"],
            "rate": 100,
        },
    )
    config = save(
        "config",
        {
            "version": 1,
            "renderConfig": render,
            "cuts": cuts,
            "selectionConfig": producer,
            "selection": selection,
            "originalClock": clock,
            "pcmAudit": audit,
        },
    )
    return Path(config["path"])


def test_metadata_cli_without_waveforms(tmp_path: Path) -> None:
    config = metadata_fixture(tmp_path)
    output = tmp_path / "report.json"
    assert mapping.main([str(config), str(output), "--sha256", file_digest(config)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["measurements"]["mapping"]["nativeTimesPreserved"] is True
    assert report["measurements"]["mapping"]["minimumDrift"] == "0"
    assert report["acceptedQuality"] is False
    assert report["currentPCMBytesRechecked"] is False
    assert report["aacSampleTimeMapping"] is None
    assert str(tmp_path / "absent-source.wav") not in report["inputs"]
    assert not (tmp_path / "absent-source.wav").exists()
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        mapping.analyze_metadata(config, output, sha256=file_digest(config))


@pytest.mark.parametrize(
    ("document", "changes", "message"),
    [
        ("config", {"version": True}, "version 1 mapping"),
        ("selection", {"version": 2}, "version 1 selection"),
        ("audit", {"version": True}, "version 2 PCM"),
        ("render", {"cuts": "/absent.json"}, "checkout"),
        ("producer", {"source": {"path": "/absent.wav", "sha256": "a" * 64}}, "checkout"),
        ("selection", {"inputs": {}}, "Selection does not bind"),
        ("audit", {"inputs": {}}, "Audit does not bind"),
        ("audit", {"selectedPCMBytesEqual": False}, "did not pass"),
        ("audit", {"mismatchedBytes": 1}, "did not pass"),
        ("producer", {"coordinateSpace": "output"}, "native intervals"),
        ("producer", {"intervals": []}, "native intervals"),
        ("audit", {"spans": [[0, 199]]}, "Audit spans differ"),
        ("selection", {"selectedSamples": 199}, "sample count differs"),
        ("audit", {"sourceSamples": 199}, "PCM metadata differs"),
        ("audit", {"comparedBytes": 799}, "byte extent differs"),
        ("clock", {"frames": [{"pts": 100, "nb_samples": 200}]}, "native cut projection"),
    ],
)
def test_metadata_rejects_inconsistent_evidence(
    tmp_path: Path, document: str, changes: dict[str, object], message: str
) -> None:
    config = metadata_fixture(tmp_path, document=document, changes=changes)
    with pytest.raises(ValueError, match=message):
        mapping.analyze_metadata(config, tmp_path / "report.json", sha256=file_digest(config))
    assert not (tmp_path / "report.json").exists()


def test_metadata_pins_and_mutation(tmp_path: Path) -> None:
    config = metadata_fixture(tmp_path)
    with pytest.raises(ValueError):
        mapping.metadata_reference({"path": str(config), "sha256": "a" * 64}, {})
    with pytest.raises(ValueError, match="Conflicting"):
        mapping.metadata_reference(
            {"path": str(config), "sha256": file_digest(config)}, {str(config): "b" * 64}
        )
    original = mapping.placement

    def mutate(*args: object, **kwargs: object) -> dict[str, object]:
        config.write_text("{}", encoding="utf-8")
        return {"nativeTimesPreserved": True}

    with (
        patch.object(mapping, "placement", side_effect=mutate),
        pytest.raises(ValueError, match="changed during"),
    ):
        mapping.analyze_metadata(config, tmp_path / "report.json", sha256=file_digest(config))
    assert mapping.placement is original


@pytest.mark.parametrize(("second_pts", "drift"), [(110, "-1/1000"), (90, "1/1000")])
def test_two_span_writer_discards_native_gap_or_overlap(
    tmp_path: Path, second_pts: int, drift: str
) -> None:
    import wave

    clock = AudioFrameClock((0, second_pts), (100, 100), 10000, Fraction(1, 10000))
    cuts = ((Fraction(0), Fraction(second_pts + 100, 10000)),)
    assert retained_pcm_spans(clock, cuts, output_rate=10000, output_samples=200) == ((0, 200),)
    source, selected = tmp_path / "source.wav", tmp_path / "selected.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((1, 2, 10000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * 100 + b"\x02\x00" * 100)
    assert write_pcm_spans(source, selected, ((0, 100), (100, 200))) == 200
    with wave.open(str(selected), "rb") as audio:
        assert audio.getnframes() == 200
        assert audio.readframes(200) == b"\x01\x00" * 100 + b"\x02\x00" * 100
    result = placement(clock, ((0, 100), (100, 200)), rate=10000, cuts=cuts)
    assert result["nativeTimesPreserved"] is False
    assert result["durationDifference"] == drift
    # Contract: timestamped second support is second_pts/10000, not sample ordinal 100/10000.
    assert result["maximumAbsoluteDrift"] == "1/1000"


def test_native_gap_placement_contract() -> None:
    clock = AudioFrameClock((0, 110), (100, 100), 10000, Fraction(1, 10000))
    result = placement(
        clock, ((0, 100), (100, 200)), rate=10000, cuts=((Fraction(0), Fraction(21, 1000)),)
    )
    pieces = result["pieces"]
    assert isinstance(pieces, list)
    assert pieces[1]["expectedOutputStart"] == "11/1000"
    assert pieces[1]["actualOutputStart"] == "1/100"
    # A future timestamped writer must preserve this measured 10-sample hole,
    # not pad the tail or reinterpret missing native support as observed silence.
    assert Fraction(pieces[1]["expectedOutputStart"]) - Fraction(
        pieces[0]["nativeEnd"]
    ) == Fraction(1, 1000)


def test_two_span_fractional_boundary_contract() -> None:
    clock = AudioFrameClock((0, 1024), (1024, 1024), 48000, Fraction(1, 48000))
    result = placement(
        clock,
        ((0, 941), (941, 1882)),
        rate=44100,
        cuts=((Fraction(0), Fraction(2048, 48000)),),
    )
    assert result["sourceFrameBoundariesSampleAligned"] is False
    assert result["nativeTimesPreserved"] is False
    assert result["durationDifference"] == "1/110250"
    assert result["maximumAbsoluteDrift"] == "0"
    # Zero signed drift under rate projection is not a measured resampler clock.


def test_report_scope_and_metadata_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = metadata_fixture(tmp_path)
    with pytest.raises(ValueError, match="Report must remain"):
        mapping.analyze_metadata(config, Path.cwd() / "mapping.json", sha256=file_digest(config))
    monkeypatch.setattr(mapping, "MAX_METADATA_BYTES", 1)
    with pytest.raises(ValueError):
        mapping.metadata_reference({"path": str(config), "sha256": file_digest(config)}, {})
