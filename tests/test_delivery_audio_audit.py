import json
import wave
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from replay import delivery_audio_audit as audit
from replay.audio_clock import AudioFrameClock
from replay.recordings import record
from replay.stages import file_digest


@dataclass
class Fixture:
    config: Path
    selection: Path
    output: Path
    source: Path
    selected: Path
    clock: Path

    def run(self) -> dict[str, object]:
        audit.audit(
            self.config,
            self.selection,
            self.output,
            config_sha256=file_digest(self.config),
            selection_sha256=file_digest(self.selection),
        )
        return record(json.loads(self.output.read_text(encoding="utf-8")))

    def repin(self) -> None:
        config = record(json.loads(self.config.read_text(encoding="utf-8")))
        config["source"] = {"path": str(self.source), "sha256": file_digest(self.source)}
        config["clock"] = {"path": str(self.clock), "sha256": file_digest(self.clock)}
        self.config.write_text(json.dumps(config), encoding="utf-8")
        receipt = record(json.loads(self.selection.read_text(encoding="utf-8")))
        receipt["inputs"] = {
            str(self.config): file_digest(self.config),
            str(self.source): file_digest(self.source),
            str(self.clock): file_digest(self.clock),
        }
        receipt["outputSHA256"] = file_digest(self.selected)
        self.selection.write_text(json.dumps(receipt), encoding="utf-8")


def replace(path: Path, key: str, value: object) -> None:
    data = record(json.loads(path.read_text(encoding="utf-8")))
    data[key] = value
    path.write_text(json.dumps(data), encoding="utf-8")


def wav(path: Path, *, count: int = 8000, rate: int = 8000, width: int = 2) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, width, rate, 0, "NONE", "not compressed"))
        writer.writeframes(b"\x01" * count * width)


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    source, selected = tmp_path / "source.wav", tmp_path / "selected.wav"
    wav(source)
    wav(selected)
    clock = tmp_path / "clock.json"
    clock.write_text(
        json.dumps(
            {
                "streams": [{"sample_rate": "8000", "time_base": "1/8000"}],
                "frames": [{"pts": 0, "nb_samples": 4000}, {"pts": 6000, "nb_samples": 4000}],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "coordinateSpace": "native-presentation-seconds",
                "source": {},
                "clock": {},
                "evidence": [],
                "intervals": [
                    {"start": {"value": 0, "timescale": 1}, "end": {"value": 5, "timescale": 4}}
                ],
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "version": 1,
                "inputs": {},
                "output": str(selected),
                "outputSHA256": "",
                "sourceSamples": 8000,
                "selectedSamples": 8000,
                "sampleRate": 8000,
                "channels": 1,
                "sampleWidth": 2,
                "spans": [[0, 8000]],
                "paddingSamples": 0,
                "resamplingPerformed": False,
                "fadesApplied": False,
                "nativeIntervalDuration": "5/4",
                "audioDuration": "1",
                "durationDifference": "-1/4",
            }
        ),
        encoding="utf-8",
    )
    result = Fixture(config, selection, tmp_path / "audit.json", source, selected, clock)
    result.repin()
    return result


def test_gap_and_unchanged_inputs(fixture: Fixture) -> None:
    before = (
        fixture.source.read_bytes(),
        fixture.selected.read_bytes(),
        fixture.config.read_bytes(),
        fixture.selection.read_bytes(),
    )
    result = fixture.run()
    assert result["comparedBytes"] == 16000
    assert result["selectedPCMBytesEqual"] is True
    assert result["durationDifference"] == "-1/4"
    assert result["intervals"] == [
        {
            "videoStart": "0",
            "videoEnd": "5/4",
            "videoOutputStart": "0",
            "videoDuration": "5/4",
            "nativeGaps": [["1/2", "3/4"]],
            "nativeGapDuration": "1/4",
            "nativeOverlaps": [],
            "nativeOverlapDuration": "0",
            "exactMappedSamplesBeforeRounding": "8000",
            "mappedSamples": 8000,
            "mappedDuration": "1",
            "roundingAndDeduplicationSamples": "0",
            "roundingAndClampSamples": "0",
            "deduplicatedSamples": 0,
            "outputClockStatus": "contiguous-cut",
            "outputStartSample": 0,
            "outputEndSample": 8000,
            "outputClockOffsetStart": "0",
            "outputClockOffsetEnd": "-1/4",
        }
    ]
    assert result["AVSyncVerified"] is False
    assert result["sourceCorrespondenceVerified"] is False
    assert result["musicGatePassed"] is False
    assert result["speechGatePassed"] is False
    assert result["deliveryComplete"] is False
    assert (
        fixture.source.read_bytes(),
        fixture.selected.read_bytes(),
        fixture.config.read_bytes(),
        fixture.selection.read_bytes(),
    ) == before
    assert fixture.output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        fixture.run()


def test_contiguous_noninteger_cuts_and_deduplication(fixture: Fixture) -> None:
    replace(
        fixture.clock, "frames", [{"pts": 0, "nb_samples": 4000}, {"pts": 4000, "nb_samples": 4000}]
    )
    replace(
        fixture.config,
        "intervals",
        [
            {"start": {"value": 0, "timescale": 1}, "end": {"value": 1, "timescale": 30}},
            {"start": {"value": 1, "timescale": 30}, "end": {"value": 1, "timescale": 1}},
        ],
    )
    fixture.repin()
    result = fixture.run()
    intervals = result["intervals"]
    assert isinstance(intervals, list)
    assert record(intervals[0])["mappedSamples"] == 267
    assert record(intervals[0])["roundingAndDeduplicationSamples"] == "1/3"
    assert record(intervals[0])["outputClockOffsetEnd"] == "1/24000"
    assert record(intervals[1])["mappedSamples"] == 7733
    assert record(intervals[1])["roundingAndDeduplicationSamples"] == "-1/3"
    assert record(intervals[1])["outputClockOffsetStart"] == "1/24000"
    assert record(intervals[1])["outputClockOffsetEnd"] == "0"
    assert result["receiptClockDiscrepancies"] == ["nativeIntervalDuration", "durationDifference"]


def test_byte_mismatch_is_evidence_not_approval(fixture: Fixture) -> None:
    with fixture.selected.open("r+b") as stream:
        stream.seek(44)
        stream.write(b"\xff")
    fixture.repin()
    result = fixture.run()
    assert result["selectedPCMBytesEqual"] is False
    assert result["mismatchedBytes"] == 1
    assert result["firstMismatchedOutputByte"] == 0


@pytest.mark.parametrize("name", ["source", "selected", "config", "clock", "selection"])
def test_changed_pin(fixture: Fixture, name: str) -> None:
    config_hash, receipt_hash = file_digest(fixture.config), file_digest(fixture.selection)
    path = getattr(fixture, name)
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="Pinned input changed"):
        audit.audit(
            fixture.config,
            fixture.selection,
            fixture.output,
            config_sha256=config_hash,
            selection_sha256=receipt_hash,
        )
    assert not fixture.output.exists()


@pytest.mark.parametrize("name", ["source", "selected", "config"])
def test_changed_during_comparison(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    original = audit._compare

    def compare(source: Path, selected: Path, spans: list[tuple[int, int]]) -> dict[str, object]:
        result = original(source, selected, spans)
        with getattr(fixture, name).open("ab") as stream:
            stream.write(b" ")
        return result

    monkeypatch.setattr(audit, "_compare", compare)
    with pytest.raises(ValueError, match="changed during execution"):
        fixture.run()
    assert not fixture.output.exists()


@pytest.mark.parametrize("name", ["source", "selected"])
def test_truncated_wav(fixture: Fixture, name: str) -> None:
    path = getattr(fixture, name)
    with path.open("r+b") as stream:
        stream.truncate(100)
    fixture.repin()
    with pytest.raises(ValueError, match="Truncated WAV"):
        fixture.run()


@pytest.mark.parametrize(
    "rate,width,count,error",
    [
        (8001, 2, 8000, "format mismatch"),
        (8000, 1, 8000, "PCM16"),
        (8000, 2, 7999, "count differs"),
        (7999, 2, 8000, "Unsupported WAV"),
    ],
)
def test_wav_mismatch(fixture: Fixture, rate: int, width: int, count: int, error: str) -> None:
    wav(fixture.selected, rate=rate, width=width, count=count)
    fixture.repin()
    with pytest.raises(ValueError, match=error):
        fixture.run()


@pytest.mark.parametrize(
    "key,value",
    [
        ("selectedSamples", 7999),
        ("sampleRate", True),
        ("spans", [[0, 7999]]),
        ("spans", [[False, 8000]]),
        ("paddingSamples", 1),
        ("fadesApplied", True),
        ("version", True),
        ("inputs", {}),
    ],
)
def test_invalid_receipt(fixture: Fixture, key: str, value: object) -> None:
    replace(fixture.selection, key, value)
    with pytest.raises(ValueError):
        fixture.run()


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", True),
        ("coordinateSpace", "output"),
        ("evidence", None),
        ("intervals", []),
        (
            "intervals",
            [{"start": {"value": 2, "timescale": 1}, "end": {"value": 1, "timescale": 1}}],
        ),
        (
            "intervals",
            [{"start": {"value": 0, "timescale": 1}, "end": {"value": 1, "timescale": 1}}] * 2,
        ),
    ],
)
def test_invalid_config(fixture: Fixture, key: str, value: object) -> None:
    replace(fixture.config, key, value)
    fixture.repin()
    with pytest.raises(ValueError):
        fixture.run()


def test_receipt_missing_config_binding(fixture: Fixture) -> None:
    replace(fixture.selection, "inputs", {str(fixture.source): file_digest(fixture.source)})
    with pytest.raises(ValueError, match="does not bind"):
        fixture.run()


def test_wrong_clock_binding(fixture: Fixture) -> None:
    replace(fixture.clock, "frames", [{"pts": 0, "nb_samples": 8002}])
    fixture.repin()
    with pytest.raises(ValueError, match="WAV count does not match"):
        fixture.run()


def test_overlapping_native_frames_preserve_distinct_samples(fixture: Fixture) -> None:
    replace(
        fixture.clock, "frames", [{"pts": 0, "nb_samples": 4000}, {"pts": 3999, "nb_samples": 4000}]
    )
    fixture.repin()
    result = fixture.run()
    assert result["selectedPCMBytesEqual"] is True
    assert result["version"] == 2
    intervals = result["intervals"]
    assert isinstance(intervals, list)
    assert record(intervals[0])["nativeOverlapDuration"] == "1/8000"
    assert record(intervals[0])["nativeGapDuration"] == "2001/8000"
    assert result["AVSyncVerified"] is False


def test_bounded_streaming_and_cli(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    original = wave.Wave_read.readframes
    reads: list[int] = []

    def readframes(self: wave.Wave_read, count: int) -> bytes:
        reads.append(count)
        return original(self, count)

    monkeypatch.setattr(audit, "BLOCK_FRAMES", 97)
    monkeypatch.setattr(wave.Wave_read, "readframes", readframes)
    monkeypatch.setattr(
        "sys.argv",
        [
            "audio-audit",
            str(fixture.config),
            str(fixture.selection),
            str(fixture.output),
            "--config-sha256",
            file_digest(fixture.config),
            "--selection-sha256",
            file_digest(fixture.selection),
        ],
    )
    audit.main()
    assert max(reads) == 97
    assert len(reads) == 332


def test_cut_gaps_edges_and_separate_spans(fixture: Fixture) -> None:
    replace(
        fixture.config,
        "intervals",
        [
            {"start": {"value": 1, "timescale": 4}, "end": {"value": 3, "timescale": 5}},
            {"start": {"value": 1, "timescale": 1}, "end": {"value": 3, "timescale": 2}},
        ],
    )
    wav(fixture.selected, count=4000)
    replace(fixture.selection, "selectedSamples", 4000)
    replace(fixture.selection, "spans", [[2000, 4000], [6000, 8000]])
    fixture.repin()
    result = fixture.run()
    assert result["selectedPCMBytesEqual"] is True
    intervals = result["intervals"]
    assert isinstance(intervals, list)
    assert record(intervals[0])["nativeGaps"] == [["1/2", "3/5"]]
    assert record(intervals[1])["nativeGaps"] == [["5/4", "3/2"]]
    assert record(intervals[1])["outputClockOffsetStart"] == "-1/10"
    assert result["durationDifference"] == "-7/20"


def test_budget_and_output_scope(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture.output = Path.cwd() / "audit-rejected.json"
    with pytest.raises(ValueError, match="edits/work"):
        fixture.run()
    fixture.output = fixture.config.parent / "audit.json"
    monkeypatch.setattr(audit, "MAX_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="budget"):
        fixture.run()


def test_intersection_budget(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "MAX_PIECES", 1)
    with pytest.raises(ValueError, match="intersection budget"):
        fixture.run()


def test_conflicting_pin(fixture: Fixture) -> None:
    replace(fixture.config, "evidence", [{"path": str(fixture.source), "sha256": "0" * 64}])
    fixture.repin()
    with pytest.raises(ValueError, match="Conflicting"):
        fixture.run()


def test_valid_evidence_pin(fixture: Fixture) -> None:
    replace(
        fixture.config,
        "evidence",
        [{"path": str(fixture.source), "sha256": file_digest(fixture.source)}],
    )
    fixture.repin()
    assert fixture.run()["selectedPCMBytesEqual"] is True


def test_truncation_during_compare(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    original = audit._compare

    def compare(source: Path, selected: Path, spans: list[tuple[int, int]]) -> dict[str, object]:
        with selected.open("r+b") as stream:
            stream.truncate(100)
        return original(source, selected, spans)

    monkeypatch.setattr(audit, "_compare", compare)
    with pytest.raises(ValueError, match="Truncated PCM during"):
        fixture.run()


def test_nonuniform_source_selection_uses_actual_offsets(fixture: Fixture) -> None:
    pcm = b"".join(i.to_bytes(2, "little") for i in range(8000))
    with wave.open(str(fixture.source), "wb") as writer:
        writer.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(pcm)
    with wave.open(str(fixture.selected), "wb") as writer:
        writer.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(pcm[4000:8000] + pcm[12000:16000])
    replace(
        fixture.config,
        "intervals",
        [
            {"start": {"value": 1, "timescale": 4}, "end": {"value": 1, "timescale": 2}},
            {"start": {"value": 1, "timescale": 1}, "end": {"value": 5, "timescale": 4}},
        ],
    )
    replace(fixture.selection, "selectedSamples", 4000)
    replace(fixture.selection, "spans", [[2000, 4000], [6000, 8000]])
    fixture.repin()
    result = fixture.run()
    assert result["selectedPCMBytesEqual"] is True
    assert result["comparedBytes"] == 8000
    assert result["durationDifference"] == "0"


@pytest.mark.parametrize("rate,samples", [(48000, 6144), (44100, 5645), (44100, 5644)])
def test_realistic_overlap_gap_projection_accounting(rate: int, samples: int) -> None:
    clock = AudioFrameClock((23088, 25008, 27888), (2048, 2048, 2048), 48000, Fraction(1, 48000))
    spans, diagnostics, accounting = audit._clock_diagnostics(
        clock, ((Fraction(23088, 48000), Fraction(29936, 48000)),), rate=rate, samples=samples
    )
    assert len(spans) == 1
    assert diagnostics[0]["nativeOverlaps"] == [["521/1000", "1571/3000"]]
    assert diagnostics[0]["nativeGaps"] == [["1691/3000", "581/1000"]]
    assert diagnostics[0]["outputClockStatus"] == "contiguous-cut"
    if rate == 48000:
        assert spans == [(0, 6144)]
        assert accounting == {
            "nativeGapDuration": "13/750",
            "nativeOverlapDuration": "1/375",
            "exactMappedSamplesBeforeRounding": "6144",
            "roundingAndClampSamples": "0",
            "deduplicatedSamples": 0,
            "roundingAndDeduplicationSamples": "0",
            "accountedDurationDifference": "-11/750",
            "durationDifferenceResidual": "0",
        }
    elif samples == 5645:
        assert spans == [(0, 5645)]
        assert accounting == {
            "nativeGapDuration": "13/750",
            "nativeOverlapDuration": "1/375",
            "exactMappedSamplesBeforeRounding": "28224/5",
            "roundingAndClampSamples": "11/5",
            "deduplicatedSamples": 2,
            "roundingAndDeduplicationSamples": "1/5",
            "accountedDurationDifference": "-3233/220500",
            "durationDifferenceResidual": "0",
        }
    else:
        assert spans == [(0, 5644)]
        assert accounting == {
            "nativeGapDuration": "13/750",
            "nativeOverlapDuration": "1/375",
            "exactMappedSamplesBeforeRounding": "28224/5",
            "roundingAndClampSamples": "6/5",
            "deduplicatedSamples": 2,
            "roundingAndDeduplicationSamples": "-4/5",
            "accountedDurationDifference": "-1619/110250",
            "durationDifferenceResidual": "0",
        }


@pytest.mark.parametrize("rate,samples", [(48000, 6144), (44100, 5645)])
def test_overlap_across_cut_boundary_has_no_contiguous_clock(rate: int, samples: int) -> None:
    clock = AudioFrameClock((23088, 25008, 27888), (2048, 2048, 2048), 48000, Fraction(1, 48000))
    spans, diagnostics, accounting = audit._clock_diagnostics(
        clock,
        (
            (Fraction(23088, 48000), Fraction(25100, 48000)),
            (Fraction(25100, 48000), Fraction(29936, 48000)),
        ),
        rate=rate,
        samples=samples,
    )
    assert len(spans) == 1
    if rate == 48000:
        assert [item["mappedSamples"] for item in diagnostics] == [2104, 4040]
    else:
        assert [item["mappedSamples"] for item in diagnostics] == [1935, 3710]
    assert [item["outputClockStatus"] for item in diagnostics] == [
        "unsupported-interleaved-cut-ownership",
        "unsupported-interleaved-cut-ownership",
    ]
    assert [item["outputStartSample"] for item in diagnostics] == [None, None]
    assert [item["outputEndSample"] for item in diagnostics] == [None, None]
    assert [item["outputClockOffsetStart"] for item in diagnostics] == [None, None]
    assert [item["outputClockOffsetEnd"] for item in diagnostics] == [None, None]
    assert accounting["durationDifferenceResidual"] == "0"


def test_many_projected_edges_share_one_sample_owned_by_earlier_cut() -> None:
    clock = AudioFrameClock((0, 1, 2), (1, 1, 1), 48000, Fraction(1, 48000))
    spans, diagnostics, accounting = audit._clock_diagnostics(
        clock,
        ((Fraction(0), Fraction(1, 48000)), (Fraction(1, 48000), Fraction(3, 48000))),
        rate=8000,
        samples=1,
    )
    assert spans == [(0, 1)]
    assert [item["mappedSamples"] for item in diagnostics] == [1, 0]
    assert [item["deduplicatedSamples"] for item in diagnostics] == [0, 2]
    assert [item["outputClockStatus"] for item in diagnostics] == [
        "contiguous-cut",
        "contiguous-cut",
    ]
    assert accounting["roundingAndClampSamples"] == "5/2"
    assert accounting["roundingAndDeduplicationSamples"] == "1/2"
    assert accounting["durationDifferenceResidual"] == "0"


def test_nested_overlaps_count_excess_multiplicity_not_pairwise_gaps() -> None:
    clock = AudioFrameClock((0, 2, 4), (10, 3, 2), 8000, Fraction(1, 8000))
    spans, diagnostics, accounting = audit._clock_diagnostics(
        clock, ((Fraction(0), Fraction(10, 8000)),), rate=8000, samples=15
    )
    assert spans == [(0, 15)]
    assert diagnostics[0]["nativeGaps"] == []
    assert diagnostics[0]["nativeOverlaps"] == [["1/4000", "1/1600"], ["1/2000", "3/4000"]]
    assert accounting == {
        "nativeGapDuration": "0",
        "nativeOverlapDuration": "1/1600",
        "exactMappedSamplesBeforeRounding": "15",
        "roundingAndClampSamples": "0",
        "deduplicatedSamples": 0,
        "roundingAndDeduplicationSamples": "0",
        "accountedDurationDifference": "1/1600",
        "durationDifferenceResidual": "0",
    }


@pytest.mark.parametrize("reorder_by_cut", [False, True])
def test_interleaved_overlap_compares_nonuniform_source_bytes(
    fixture: Fixture, reorder_by_cut: bool
) -> None:
    replace(
        fixture.clock,
        "frames",
        [{"pts": 0, "nb_samples": 4000}, {"pts": 3000, "nb_samples": 4000}],
    )
    replace(
        fixture.config,
        "intervals",
        [
            {"start": {"value": 0, "timescale": 1}, "end": {"value": 7, "timescale": 16}},
            {"start": {"value": 7, "timescale": 16}, "end": {"value": 7, "timescale": 8}},
        ],
    )
    pcm = b"".join(i.to_bytes(2, "little") for i in range(8000))
    with wave.open(str(fixture.source), "wb") as writer:
        writer.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(pcm)
    with wave.open(str(fixture.selected), "wb") as writer:
        writer.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        writer.writeframes(
            pcm[:7000] + pcm[8000:9000] + pcm[7000:8000] + pcm[9000:] if reorder_by_cut else pcm
        )
    fixture.repin()
    result = fixture.run()
    assert result["comparisonScope"] == "source-order-union-of-frame-intersections"
    if reorder_by_cut:
        assert result["selectedPCMBytesEqual"] is False
        assert result["firstMismatchedOutputByte"] == 7000
    else:
        assert result["selectedPCMBytesEqual"] is True
        assert result["firstMismatchedOutputByte"] is None
    assert result["comparedBytes"] == 16000
    assert result["durationDifference"] == "1/8"
    assert result["AVSyncVerified"] is False
    assert result["sourceCorrespondenceVerified"] is False
    assert result["musicGatePassed"] is False
    assert result["speechGatePassed"] is False
    assert result["deliveryComplete"] is False
    intervals = result["intervals"]
    assert isinstance(intervals, list)
    assert record(intervals[0])["mappedSamples"] == 4000
    assert record(intervals[1])["mappedSamples"] == 4000
    assert record(intervals[0])["outputClockOffsetEnd"] is None
    assert record(intervals[1])["outputClockOffsetEnd"] is None
    assert record(result["durationAccounting"])["durationDifferenceResidual"] == "0"


def test_projected_overlap_compares_separate_nonuniform_spans(fixture: Fixture) -> None:
    replace(
        fixture.clock,
        "frames",
        [{"pts": 0, "nb_samples": 4000}, {"pts": 3000, "nb_samples": 4000}],
    )
    replace(
        fixture.config,
        "intervals",
        [
            {"start": {"value": 1, "timescale": 4}, "end": {"value": 7, "timescale": 16}},
            {"start": {"value": 1, "timescale": 2}, "end": {"value": 7, "timescale": 8}},
        ],
    )
    pcm = b"".join(i.to_bytes(2, "little") for i in range(44100))
    with wave.open(str(fixture.source), "wb") as writer:
        writer.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        writer.writeframes(pcm)
    with wave.open(str(fixture.selected), "wb") as writer:
        writer.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
        writer.writeframes(pcm[22050:38588] + pcm[44100:49614] + pcm[55124:88200])
    replace(fixture.selection, "sampleRate", 44100)
    replace(fixture.selection, "sourceSamples", 44100)
    replace(fixture.selection, "selectedSamples", 27564)
    replace(fixture.selection, "spans", [[11025, 19294], [22050, 24807], [27562, 44100]])
    fixture.repin()
    result = fixture.run()
    assert result["selectedPCMBytesEqual"] is True
    assert result["comparedBytes"] == 55128
    assert result["durationAccounting"] == {
        "nativeGapDuration": "0",
        "nativeOverlapDuration": "1/16",
        "exactMappedSamplesBeforeRounding": "55125/2",
        "roundingAndClampSamples": "3/2",
        "deduplicatedSamples": 0,
        "roundingAndDeduplicationSamples": "3/2",
        "accountedDurationDifference": "3677/58800",
        "durationDifferenceResidual": "0",
    }
    intervals = result["intervals"]
    assert isinstance(intervals, list)
    assert record(intervals[0])["outputClockStatus"] == "contiguous-cut"
    assert record(intervals[0])["mappedSamples"] == 11026
    assert record(intervals[1])["mappedSamples"] == 16538
    assert result["AVSyncVerified"] is False


def test_symlink_output_rejected(fixture: Fixture) -> None:
    fixture.output.symlink_to(fixture.source)
    with pytest.raises(ValueError, match="Linked"):
        fixture.run()
