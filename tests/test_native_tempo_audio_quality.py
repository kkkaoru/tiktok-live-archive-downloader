"""Synthetic original tempo records, never private media or model fixtures."""

import hashlib
import importlib
import json
import os
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import delivery_audio_quality as dq
from replay import native_tempo_audio_quality as quality
from replay.audio_clock import AudioFrameClock
from replay.process_family_observation import SystemHost
from replay.recordings import record


@pytest.fixture
def selection() -> dict[str, object]:
    return {
        "version": 1,
        "coordinateSpace": "native-presentation-seconds",
        "source": {"path": "/source.wav", "sha256": "a" * 64},
        "clock": {"path": "/clock.json", "sha256": "b" * 64},
        "intervals": [{"start": {"value": 0, "timescale": 1}, "end": {"value": 2, "timescale": 1}}],
        "evidence": [],
    }


@pytest.fixture
def receipt() -> dict[str, object]:
    return {
        "version": 1,
        "kind": "context-atempo-exact-cut-slot-candidate",
        "inputs": {"/selection.json": "c" * 64, "/source.wav": "a" * 64, "/clock.json": "b" * 64},
        "sampleRate": 8000,
        "channels": 2,
        "maximumRateChangeAllowed": 0.5,
        "contextSeconds": "1/2",
        "maximumTailErrorSeconds": "1/10",
        "outputSHA256": "d" * 64,
        "outputStartSample": 0,
        "outputEndSample": 16000,
        "outputSamples": 16000,
        "cuts": [
            {
                "source_start_sample": 0,
                "source_end_sample": 16000,
                "output_start_sample": 0,
                "output_end_sample": 16000,
                "source_context_samples": 0,
                "synthetic_left_context_samples": 0,
                "synthetic_right_context_samples": 0,
                "filter_output_samples": 16000,
                "expected_filter_output_samples": 16000,
                "center_crop_start_sample": 0,
                "center_crop_end_sample": 16000,
                "tail_error_samples": 0,
                "process_seconds": 0.0,
                "filter_applied": False,
            }
        ],
    }


@pytest.fixture
def mapping(selection: dict[str, object], receipt: dict[str, object]) -> quality.TempoInputs:
    return quality.recorded_mapping(
        selection,
        receipt,
        AudioFrameClock((0,), (16000,), 8000, Fraction(1, 8000)),
        selection_path="/selection.json",
        selection_sha256="c" * 64,
        source_samples=16000,
    )


def test_noop_original_mapping(mapping: quality.TempoInputs) -> None:
    assert mapping.output_samples == 16000
    assert mapping.sample_rate == 8000
    assert mapping.slots[0].spans == ((0, 16000),)


def test_tempo_context_preserves_original_crop(
    selection: dict[str, object], receipt: dict[str, object]
) -> None:
    cuts = receipt["cuts"]
    assert isinstance(cuts, list)
    row = record(cuts[0])
    row.update(
        source_end_sample=12000,
        source_context_samples=4000,
        synthetic_left_context_samples=4000,
        filter_output_samples=26667,
        expected_filter_output_samples=26667,
        center_crop_start_sample=5333,
        center_crop_end_sample=21333,
        filter_applied=True,
    )
    result = quality.recorded_mapping(
        selection,
        receipt,
        AudioFrameClock((4000,), (16000,), 8000, Fraction(1, 8000)),
        selection_path="/selection.json",
        selection_sha256="c" * 64,
        source_samples=16000,
    )
    assert result.slots[0].spans == ((0, 12000),)
    assert result.output_samples == 16000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_end_sample", 15999),
        ("center_crop_start_sample", 1),
        ("filter_output_samples", 15999),
        ("tail_error_samples", 1),
        ("filter_applied", True),
        ("process_seconds", -1),
        ("process_seconds", "0"),
        ("process_seconds", float("inf")),
        ("output_start_sample", True),
    ],
)
def test_changed_measurement_rejected(
    selection: dict[str, object], receipt: dict[str, object], field: str, value: object
) -> None:
    cuts = receipt["cuts"]
    assert isinstance(cuts, list)
    record(cuts[0])[field] = value
    with pytest.raises(ValueError):
        quality.recorded_mapping(
            selection,
            receipt,
            AudioFrameClock((0,), (16000,), 8000, Fraction(1, 8000)),
            selection_path="/selection.json",
            selection_sha256="c" * 64,
            source_samples=16000,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("kind", "direct-source-spans"),
        ("inputs", {}),
        ("channels", 0),
        ("outputStartSample", 1),
        ("outputEndSample", 15999),
        ("outputSamples", 15999),
        ("cuts", []),
        ("contextSeconds", 0.5),
    ],
)
def test_bad_receipt(
    selection: dict[str, object], receipt: dict[str, object], field: str, value: object
) -> None:
    receipt[field] = value
    with pytest.raises(ValueError):
        quality.recorded_mapping(
            selection,
            receipt,
            AudioFrameClock((0,), (16000,), 8000, Fraction(1, 8000)),
            selection_path="/selection.json",
            selection_sha256="c" * 64,
            source_samples=16000,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("intervals", []),
        ("evidence", None),
        ("evidence", [{"path": "/other.json", "sha256": "f" * 64}]),
        ("source", {"path": "relative.wav", "sha256": "a" * 64}),
    ],
)
def test_bad_selection(
    selection: dict[str, object], receipt: dict[str, object], field: str, value: object
) -> None:
    selection[field] = value
    with pytest.raises(ValueError):
        quality.recorded_mapping(
            selection,
            receipt,
            AudioFrameClock((0,), (16000,), 8000, Fraction(1, 8000)),
            selection_path="/selection.json",
            selection_sha256="c" * 64,
            source_samples=16000,
        )


def test_independent_matched_transform(mapping: quality.TempoInputs) -> None:
    quality.matched_transform(
        mapping, replace(mapping, source_sha256="e" * 64, output_sha256="f" * 64)
    )


@pytest.mark.parametrize(
    "change", ["source_sha256", "clock_sha256", "sample_rate", "channels", "source_samples"]
)
def test_mismatched_transform(mapping: quality.TempoInputs, change: str) -> None:
    other = replace(mapping, source_sha256="e" * 64, output_sha256="f" * 64)
    changes = {
        "source_sha256": replace(other, source_sha256="a" * 64),
        "clock_sha256": replace(other, clock_sha256="e" * 64),
        "sample_rate": replace(other, sample_rate=16000),
        "channels": replace(other, channels=1),
        "source_samples": replace(other, source_samples=15999),
    }
    with pytest.raises(ValueError):
        quality.matched_transform(mapping, changes[change])


def test_full_cut_protection_does_not_interpolate(mapping: quality.TempoInputs) -> None:
    assert quality.protect_whole_cuts(mapping, ((100, 200),)) == (
        quality.ProtectedCut(0, 0, 16000, 0, 16000),
    )


@pytest.mark.parametrize(
    "intervals", [(), ((-1, 10),), ((20, 10),), ((0, 16001),), ((20, 30), (10, 20))]
)
def test_bad_protection(
    mapping: quality.TempoInputs, intervals: tuple[tuple[int, int], ...]
) -> None:
    with pytest.raises(ValueError):
        quality.protect_whole_cuts(mapping, intervals)


def test_removed_speech_rejected(mapping: quality.TempoInputs) -> None:
    modified = replace(mapping, slots=(replace(mapping.slots[0], spans=((100, 15000),)),))
    with pytest.raises(ValueError, match="removed"):
        quality.protect_whole_cuts(modified, ((0, 200),))
    with pytest.raises(ValueError, match="lacks"):
        quality.protect_whole_cuts(modified, ((14000, 15500),))


def test_exact_other_rate_requires_rational_extent(mapping: quality.TempoInputs) -> None:
    quality.require_exact_decode(
        mapping, quality.DecodedExtent(48000, 96000, Fraction(0), Fraction(2), True, 0, 0)
    )


@pytest.mark.parametrize(
    "extent",
    [
        quality.DecodedExtent(48000, 94464, Fraction(0), Fraction(246, 125), True, 0, 0),
        quality.DecodedExtent(48000, 96000, Fraction(0), Fraction(2), True, 64, 0),
        quality.DecodedExtent(48000, 96000, Fraction(0), Fraction(2), True, 0, 64),
        quality.DecodedExtent(48000, 96000, Fraction(0), Fraction(2), False, 0, 0),
        quality.DecodedExtent(0, 96000, Fraction(0), Fraction(2), True, 0, 0),
    ],
)
def test_incomplete_or_unsupported_aac(
    mapping: quality.TempoInputs, extent: quality.DecodedExtent
) -> None:
    with pytest.raises(ValueError):
        quality.require_exact_decode(mapping, extent)


def test_pcm_budget_counts_stereo_and_real_headers() -> None:
    assert (
        quality.pcm_budget(
            rate=44100, channels=2, sample_bytes=2, seconds=Fraction(8), files=12, header_bytes=78
        )
        == 16935336
    )


@pytest.mark.parametrize("seconds", [Fraction(0), Fraction(1, 3), Fraction(21601)])
def test_bad_budget(seconds: Fraction) -> None:
    with pytest.raises(ValueError):
        quality.pcm_budget(
            rate=8000, channels=2, sample_bytes=2, seconds=seconds, files=1, header_bytes=44
        )


@pytest.fixture
def final_receipt() -> dict[str, object]:
    identity = {
        "path": "/movie.mp4",
        "bytes": 10,
        "sha256": "e" * 64,
        "device": 1,
        "inode": 2,
        "mtimeNs": 3,
        "ctimeNs": 4,
    }
    return {
        "version": 1,
        "fullDecode": True,
        "inputs": [identity],
        "inputsAfter": [identity],
        "streams": {
            "streams": [
                {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2}
            ]
        },
        "audioDecodedClock": {"start": "0", "end": "2", "span": "2", "sampleCount": 96000},
    }


def test_recorded_extent_not_current_crypto(
    mapping: quality.TempoInputs, final_receipt: dict[str, object]
) -> None:
    result = quality.recorded_aac_extent(
        final_receipt, movie=("/movie.mp4", "e" * 64), inputs=mapping
    )
    assert result["completeExtent"] is True
    assert result["currentMediaCryptoVerified"] is False
    assert result["primingAndContinuityVerified"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", 2),
        ("fullDecode", False),
        ("inputsAfter", []),
        ("inputs", []),
        ("streams", {}),
        ("streams", {"streams": []}),
        ("audioDecodedClock", {"start": "0", "end": "2", "span": "2", "sampleCount": 123}),
    ],
)
def test_final_receipt_mismatch(
    mapping: quality.TempoInputs, final_receipt: dict[str, object], key: str, value: object
) -> None:
    final_receipt[key] = value
    with pytest.raises(ValueError):
        quality.recorded_aac_extent(final_receipt, movie=("/movie.mp4", "e" * 64), inputs=mapping)


def test_wrong_movie(mapping: quality.TempoInputs, final_receipt: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="different movie"):
        quality.recorded_aac_extent(final_receipt, movie=("/wrong.mp4", "e" * 64), inputs=mapping)


@pytest.fixture
def mapping_request(
    tmp_path: Path,
    selection: dict[str, object],
    receipt: dict[str, object],
    final_receipt: dict[str, object],
) -> tuple[Path, str]:
    clock = tmp_path / "clock.json"
    clock.write_text(
        json.dumps(
            {
                "streams": [{"sample_rate": "8000", "time_base": "1/8000"}],
                "frames": [{"pts": 0, "nb_samples": 16000}],
            }
        ),
        encoding="utf-8",
    )
    clock_sha = hashlib.sha256(clock.read_bytes()).hexdigest()
    selection["clock"] = {"path": str(clock), "sha256": clock_sha}
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    selection_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    receipt["inputs"] = {
        str(selection_path): selection_sha,
        str(clock): clock_sha,
        "/source.wav": "a" * 64,
    }
    receipt_path = tmp_path / "tempo.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    final_path = tmp_path / "final.json"
    final_path.write_text(json.dumps(final_receipt), encoding="utf-8")
    config = {
        "version": 1,
        "selection": {"path": str(selection_path), "sha256": selection_sha},
        "tempoReceipt": {
            "path": str(receipt_path),
            "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        },
        "clock": {"path": str(clock), "sha256": clock_sha},
        "sourceSamples": 16000,
        "finalReceipt": {
            "path": str(final_path),
            "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        },
        "movie": {"path": "/movie.mp4", "sha256": "e" * 64},
    }
    path = tmp_path / "request.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_immutable_mapping_only(
    mapping_request: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = mapping_request
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", ["adapter", str(path), str(output), "--sha256", digest])
    quality.main()
    quality.run(path, output, sha256=digest, verify=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["recordedTempoSlotsVerified"] is True
    assert result["measurementInputsReady"] is False
    assert result["qualityApproved"] is False
    assert result["mediaBytesRead"] == 0
    with pytest.raises(FileExistsError):
        quality.run(path, output, sha256=digest)
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        quality.run(path, output, sha256=digest, verify=True)


def test_request_hash_changed(mapping_request: tuple[Path, str]) -> None:
    path, digest = mapping_request
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        quality.inspect_request(path, sha256=digest)


def test_quality_inputs_can_pass_without_quality_promotion(mapping: quality.TempoInputs) -> None:
    mixture = replace(mapping, source_sha256="e" * 64, output_sha256="f" * 64)
    assert quality.prepare_quality_inputs(
        mapping,
        mixture,
        decoded=quality.DecodedExtent(8000, 16000, Fraction(0), Fraction(2), True, 0, 0),
        protected_source_samples=((200, 300),),
    ) == (quality.ProtectedCut(0, 0, 16000, 0, 16000),)


def test_32ms_recorded_gap_is_not_padded(
    mapping: quality.TempoInputs, final_receipt: dict[str, object]
) -> None:
    final_receipt["audioDecodedClock"] = {
        "start": "0",
        "end": "246/125",
        "span": "246/125",
        "sampleCount": 94464,
    }
    result = quality.recorded_aac_extent(
        final_receipt, movie=("/movie.mp4", "e" * 64), inputs=mapping
    )
    assert result["completeExtent"] is False
    assert result["endDifferenceSeconds"] == "-4/125"
    assert result["decodedSamples"] == 94464


def test_budget_standard_wav_header() -> None:
    assert (
        quality.pcm_budget(
            rate=44100, channels=2, sample_bytes=2, seconds=Fraction(8), files=12, header_bytes=44
        )
        == 16934928
    )


def test_nonunity_transform_cannot_be_replaced_by_direct_span(mapping: quality.TempoInputs) -> None:
    mixture = replace(
        mapping,
        source_sha256="e" * 64,
        output_sha256="f" * 64,
        slots=(replace(mapping.slots[0], spans=((0, 12000),)),),
    )
    with pytest.raises(ValueError, match="transforms"):
        quality.matched_transform(mapping, mixture)


def test_rate_change_bound_rejects_old_clock(
    selection: dict[str, object], receipt: dict[str, object]
) -> None:
    receipt["maximumRateChangeAllowed"] = 0.01
    with pytest.raises(ValueError, match="rate change"):
        quality.recorded_mapping(
            selection,
            receipt,
            AudioFrameClock((4000,), (16000,), 8000, Fraction(1, 8000)),
            selection_path="/selection.json",
            selection_sha256="c" * 64,
            source_samples=16000,
        )


def test_changed_cut_endpoint(selection: dict[str, object], receipt: dict[str, object]) -> None:
    selection["intervals"] = [
        {"start": {"value": 0, "timescale": 1}, "end": {"value": 1, "timescale": 1}}
    ]
    with pytest.raises(ValueError, match="complete selected"):
        quality.recorded_mapping(
            selection,
            receipt,
            AudioFrameClock((0,), (16000,), 8000, Fraction(1, 8000)),
            selection_path="/selection.json",
            selection_sha256="c" * 64,
            source_samples=16000,
        )


def test_changed_full_decode_identity(
    mapping: quality.TempoInputs, final_receipt: dict[str, object]
) -> None:
    final_receipt["inputsAfter"] = [{"path": "/movie.mp4", "sha256": "e" * 64, "inode": 999}]
    with pytest.raises(ValueError, match="identities"):
        quality.recorded_aac_extent(final_receipt, movie=("/movie.mp4", "e" * 64), inputs=mapping)


def test_duplicate_final_inputs(
    mapping: quality.TempoInputs, final_receipt: dict[str, object]
) -> None:
    rows = final_receipt["inputs"]
    assert isinstance(rows, list)
    rows.append(rows[0])
    final_receipt["inputsAfter"] = list(rows)
    with pytest.raises(ValueError, match="identity"):
        quality.recorded_aac_extent(final_receipt, movie=("/movie.mp4", "e" * 64), inputs=mapping)


def test_skip_unprotected_cuts(mapping: quality.TempoInputs) -> None:
    prior = replace(mapping.slots[0], spans=((0, 100),), output_end=100)
    middle = replace(mapping.slots[0], spans=((100, 200),), output_start=100, output_end=200)
    last = replace(mapping.slots[0], spans=((200, 300),), output_start=200, output_end=300)
    modified = replace(mapping, slots=(prior, middle, last))
    assert quality.protect_whole_cuts(modified, ((125, 150),)) == (
        quality.ProtectedCut(1, 100, 200, 100, 200),
    )


@pytest.fixture
def audio_probe() -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "profile": "LC",
                "sample_rate": "8000",
                "channels": 2,
                "time_base": "1/8000",
                "start_pts": 0,
                "duration_ts": 16000,
            }
        ],
        "packets_and_frames": [
            {
                "type": "packet",
                "pts": -1024,
                "duration": 1024,
                "side_data_list": [
                    {"side_data_type": "Skip Samples", "skip_samples": 1024, "discard_padding": 0}
                ],
            },
            {"type": "packet", "pts": 0, "duration": 16000},
            {"type": "frame", "pts": 0, "nb_samples": 8000},
            {"type": "frame", "pts": 8000, "nb_samples": 8000},
        ],
    }


def test_audio_only_honest_packet_skip(
    receipt: dict[str, object], audio_probe: dict[str, object]
) -> None:
    result = quality.assess_audio_only(receipt, audio_probe)
    assert result["packetPresentationMatchesSelected"] is True
    assert result["decodedSamplesMinusSource"] == 0
    assert result["packetStartAfterDecoderSkip"] == "0"
    assert result["packetDecodedPresentationAccepted"] is True
    assert result["videoContractSupported"] is False
    assert result["containerEditListVerified"] is False
    assert result["qualityApproved"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("streams", []),
        ("packets_and_frames", []),
        ("packets_and_frames", [{"type": "guess"}]),
        ("packets_and_frames", [{"type": "frame"}]),
    ],
)
def test_audio_observation_bad_rows(
    receipt: dict[str, object], audio_probe: dict[str, object], key: str, value: object
) -> None:
    audio_probe[key] = value
    with pytest.raises(ValueError):
        quality.assess_audio_only(receipt, audio_probe)


@pytest.mark.parametrize(
    ("key", "value"),
    [("version", 2), ("sampleRate", 1), ("outputSamples", 0), ("outputEndSample", 1)],
)
def test_audio_original_bad_clock(
    receipt: dict[str, object], audio_probe: dict[str, object], key: str, value: object
) -> None:
    receipt[key] = value
    with pytest.raises(ValueError):
        quality.assess_audio_only(receipt, audio_probe)


def test_audio_rate_not_silently_resampled(
    receipt: dict[str, object], audio_probe: dict[str, object]
) -> None:
    streams = audio_probe["streams"]
    assert isinstance(streams, list)
    record(streams[0])["sample_rate"] = "48000"
    with pytest.raises(ValueError, match="rate or channels"):
        quality.assess_audio_only(receipt, audio_probe)


def test_audio_only_metadata_cli(
    tmp_path: Path, receipt: dict[str, object], audio_probe: dict[str, object]
) -> None:
    producer = tmp_path / "producer.json"
    producer.write_text(json.dumps(receipt), encoding="utf-8")
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(audio_probe), encoding="utf-8")
    candidate = tmp_path / "synthetic.m4a"
    candidate.write_bytes(b"synthetic media bytes, not used as actual observations")
    request_path = tmp_path / "audio-request.json"
    request_path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "native-tempo-audio-only-request",
                "sourceProducer": {
                    "path": str(producer),
                    "sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
                },
                "candidate": str(candidate),
                "observation": str(probe),
                "outputRoot": str(tmp_path),
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(request_path.read_bytes()).hexdigest()
    result = tmp_path / "audio-result.json"
    quality.run(request_path, result, sha256=digest, audio_only=True)
    quality.run(request_path, result, sha256=digest, audio_only=True, verify=True)
    document = json.loads(result.read_text(encoding="utf-8"))
    assert document["packetEndMinusSourceEnd"] == "0"
    candidate.write_bytes(b"changed synthetic media")
    with pytest.raises(ValueError, match="changed"):
        quality.run(request_path, result, sha256=digest, audio_only=True, verify=True)


def packet_fixture(scale: int = 3000) -> bytes:
    multiplier = scale // 3000
    header = [
        "#format: frame checksums",
        "#version: 2",
        "#hash: SHA256",
        "#extradata 0, 36, " + "a" * 64,
        "#software: Lavf63.1.102",
        f"#tb 0: 1/{scale}",
        "#media_type 0: video",
        "#codec_id 0: h264",
        "#dimensions 0: 1080x1920",
        "#sar 0: 1/1",
        "#stream#, dts,        pts, duration,     size, hash",
    ]
    for dts, pts in ((-200, 0), (-100, 400), (0, 200)):
        header.append(
            f"0, {dts * multiplier}, {pts * multiplier}, {100 * multiplier}, 42, " + "b" * 64
        )
    return ("\n".join(header) + "\n").encode("ascii")


def test_copy_packet_timebase_change_is_exact() -> None:
    report = quality.compare_packet_hashes(
        packet_fixture(), packet_fixture(12000), expected_frames=3
    )
    assert report["rationalClocksEqual"] is True
    assert report["orderedPayloadsEqual"] is True
    assert report["fullDecodeVerified"] is False
    assert report["qualityApproved"] is False


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (b"#hash: SHA256", b"#hash: MD5"),
        (b"#codec_id 0: h264", b"#codec_id 0: hevc"),
        (b"a" * 64, b"c" * 64),
        (b"b" * 64, b"d" * 64),
        (b"0, -200, 0, 100, 42", b"0, -200, 1, 100, 42"),
        (b"0, -200, 0, 100, 42", b"0, -200, 0, 100, 41"),
        (b"0, -200, 0, 100, 42", b"0, -200, 0, 0, 42"),
        (b"0, -200, 0, 100, 42", b"1, -200, 0, 100, 42"),
        (b"#tb 0: 1/3000", b"#tb 0: 2/1"),
    ],
)
def test_copy_packet_tamper_rejected(before: bytes, after: bytes) -> None:
    with pytest.raises(ValueError):
        quality.compare_packet_hashes(
            packet_fixture(), packet_fixture().replace(before, after, 1), expected_frames=3
        )


def test_copy_packets_incomplete_or_duplicate_decode_clock() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        quality.compare_packet_hashes(packet_fixture(), packet_fixture(), expected_frames=4)
    duplicate = packet_fixture().replace(b"0, -100, 400", b"0, -200, 400")
    with pytest.raises(ValueError, match="strictly increasing"):
        quality.compare_packet_hashes(duplicate, duplicate, expected_frames=3)
    with pytest.raises(ValueError, match="packet count"):
        quality.compare_packet_hashes(b"", b"", expected_frames=True)
    with pytest.raises(ValueError, match="byte budget"):
        quality.compare_packet_hashes(b"", b"", expected_frames=3)


def test_packet_pair_immutable_cli(tmp_path: Path) -> None:
    left, right = tmp_path / "source.txt", tmp_path / "candidate.txt"
    left.write_bytes(packet_fixture())
    right.write_bytes(packet_fixture(12000))
    config: dict[str, object] = {
        "version": 1,
        "kind": "copy-video-packet-comparison",
        "expectedFrames": 3,
        "sourceObservation": {
            "path": str(left),
            "sha256": hashlib.sha256(left.read_bytes()).hexdigest(),
        },
        "candidateObservation": {
            "path": str(right),
            "sha256": hashlib.sha256(right.read_bytes()).hexdigest(),
        },
    }
    request_path = tmp_path / "packet-request.json"
    request_path.write_text(json.dumps(config))
    digest = hashlib.sha256(request_path.read_bytes()).hexdigest()
    output = tmp_path / "packet-result.json"
    quality.run(request_path, output, sha256=digest, packet_pair=True)
    quality.run(request_path, output, sha256=digest, packet_pair=True, verify=True)
    with pytest.raises(ValueError, match="exactly one"):
        quality.run(request_path, output, sha256=digest, packet_pair=True, audio_only=True)
    right.write_bytes(packet_fixture())
    with pytest.raises(ValueError):
        quality.run(request_path, output, sha256=digest, packet_pair=True, verify=True)
    config["version"] = 2
    request_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unsupported"):
        quality.inspect_packet_pair(
            request_path, sha256=hashlib.sha256(request_path.read_bytes()).hexdigest()
        )


@pytest.fixture
def pcm_clock_request(tmp_path: Path) -> tuple[Path, Path, Path]:
    import wave

    pcm = tmp_path / "source.wav"
    with wave.open(str(pcm), "wb") as writer:
        writer.setnchannels(2)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(bytes(32000))
    source_sha = hashlib.sha256(pcm.read_bytes()).hexdigest()
    producer = tmp_path / "producer.json"
    producer.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "context-atempo-exact-cut-slot-candidate",
                "output": str(pcm),
                "outputSHA256": source_sha,
                "sampleRate": 8000,
                "channels": 2,
                "outputStartSample": 0,
                "outputEndSample": 8000,
                "outputSamples": 8000,
            }
        )
    )
    request_path = tmp_path / "clock-request.json"
    request_path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "original-pcm-clock-request",
                "source": {"path": str(pcm), "sha256": source_sha},
                "producer": {
                    "path": str(producer),
                    "sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
                },
            }
        )
    )
    return request_path, pcm, producer


def test_pcm_header_clock_actual_bound(
    pcm_clock_request: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    request_path, pcm, _ = pcm_clock_request
    digest = hashlib.sha256(request_path.read_bytes()).hexdigest()
    output = tmp_path / "clock-result.json"
    quality.run(request_path, output, sha256=digest, pcm_clock=True)
    quality.run(request_path, output, sha256=digest, pcm_clock=True, verify=True)
    report = json.loads(output.read_text())
    assert report["sampleCount"] == 8000
    assert report["sourceSHAFreshlyVerified"] is True
    assert report["qualityApproved"] is False
    pcm.write_bytes(b"changed")
    with pytest.raises(ValueError, match="original SHA"):
        quality.inspect_pcm_clock(request_path, sha256=digest)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", True),
        ("outputStartSample", False),
        ("outputSamples", 7999),
        ("sampleRate", 44100),
        ("channels", 1),
        ("kind", "invented"),
    ],
)
def test_pcm_header_rejects_producer_mismatch(
    pcm_clock_request: tuple[Path, Path, Path], key: str, value: object
) -> None:
    request_path, _, producer = pcm_clock_request
    raw = json.loads(producer.read_text())
    raw[key] = value
    producer.write_text(json.dumps(raw))
    config = json.loads(request_path.read_text())
    config["producer"]["sha256"] = hashlib.sha256(producer.read_bytes()).hexdigest()
    request_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="original producer"):
        quality.inspect_pcm_clock(
            request_path, sha256=hashlib.sha256(request_path.read_bytes()).hexdigest()
        )


def test_pcm_header_rejects_truncated_payload(pcm_clock_request: tuple[Path, Path, Path]) -> None:
    request_path, pcm, _ = pcm_clock_request
    pcm.write_bytes(pcm.read_bytes()[:-4])
    config = json.loads(request_path.read_text())
    config["source"]["sha256"] = hashlib.sha256(pcm.read_bytes()).hexdigest()
    request_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="ends before"):
        quality.inspect_pcm_clock(
            request_path, sha256=hashlib.sha256(request_path.read_bytes()).hexdigest()
        )


def test_pcm_request_schema_rejected(pcm_clock_request: tuple[Path, Path, Path]) -> None:
    request_path, _, _ = pcm_clock_request
    config = json.loads(request_path.read_text())
    config["version"] = True
    request_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Unsupported"):
        quality.inspect_pcm_clock(
            request_path, sha256=hashlib.sha256(request_path.read_bytes()).hexdigest()
        )


@pytest.fixture
def pair_request(mapping_request: tuple[Path, str], tmp_path: Path) -> Path:
    original = json.loads(mapping_request[0].read_text())
    selection = json.loads(Path(original["selection"]["path"]).read_text())
    receipt = json.loads(Path(original["tempoReceipt"]["path"]).read_text())
    selection["source"] = {"path": "/mixture.wav", "sha256": "f" * 64}
    selected = tmp_path / "mixture-selection.json"
    selected.write_text(json.dumps(selection))
    selected_pin = {
        "path": str(selected),
        "sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
    }
    receipt["inputs"] = {
        str(selected): selected_pin["sha256"],
        "/mixture.wav": "f" * 64,
        original["clock"]["path"]: original["clock"]["sha256"],
    }
    receipt["outputSHA256"] = "e" * 64
    produced = tmp_path / "mixture-receipt.json"
    produced.write_text(json.dumps(receipt))
    path = tmp_path / "pair-request.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "tempo-transform-pair-request",
                "clock": original["clock"],
                "sourceSamples": 16000,
                "reference": {
                    "selection": original["selection"],
                    "receipt": original["tempoReceipt"],
                },
                "mixture": {
                    "selection": selected_pin,
                    "receipt": {
                        "path": str(produced),
                        "sha256": hashlib.sha256(produced.read_bytes()).hexdigest(),
                    },
                },
            }
        )
    )
    return path


def test_transform_pair_cli(
    pair_request: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "pair-output.json"
    digest = hashlib.sha256(pair_request.read_bytes()).hexdigest()
    monkeypatch.setattr(
        sys,
        "argv",
        ["adapter", str(pair_request), str(output), "--sha256", digest, "--transform-pair"],
    )
    quality.main()
    quality.run(pair_request, output, sha256=digest, transform_pair=True, verify=True)
    result = json.loads(output.read_text())
    assert result["sameTransformVerified"] is True
    assert result["mediaBytesRead"] == 0
    assert result["qualityApproved"] is False
    output.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        quality.run(pair_request, output, sha256=digest, transform_pair=True, verify=True)


@pytest.mark.parametrize("mutation", ["kind", "clock", "same", "conflict", "receipt"])
def test_transform_pair_rejects_bad_bindings(pair_request: Path, mutation: str) -> None:
    data = json.loads(pair_request.read_text())
    if mutation == "kind":
        data["kind"] = "unknown"
    elif mutation == "same":
        data["mixture"] = data["reference"]
    elif mutation == "conflict":
        data["mixture"]["selection"] = dict(data["reference"]["selection"], sha256="f" * 64)
    elif mutation == "receipt":
        Path(data["mixture"]["receipt"]["path"]).write_text("{}")
    else:
        path = Path(data["mixture"]["selection"]["path"])
        selection = json.loads(path.read_text())
        selection["clock"]["sha256"] = "a" * 64
        path.write_text(json.dumps(selection))
        data["mixture"]["selection"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    pair_request.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        quality.inspect_transform_pair(
            pair_request, sha256=hashlib.sha256(pair_request.read_bytes()).hexdigest()
        )


@pytest.fixture
def quality_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import wave

    from replay.stages import _fingerprint
    from replay.workflow import _stage

    def pin(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    sources = {}
    for role in ("voice", "mixture", "decoded"):
        path = tmp_path / (role + ".source")
        path.write_text(role)
        sources[role] = pin(path)
    pair = {
        "outputSamples": 441,
        "sampleRate": 44100,
        "channels": 2,
        "referenceOutputSHA256": sources["voice"]["sha256"],
        "mixtureOutputSHA256": sources["mixture"]["sha256"],
    }
    monkeypatch.setattr(quality, "inspect_transform_pair", lambda *args, **kwargs: pair)
    original = tmp_path / "pair-original.json"
    original.write_text("{}")
    observed = tmp_path / "pair-observed.json"
    observed.write_text(json.dumps(pair))
    composition = tmp_path / "composition.json"
    composition.write_text(
        json.dumps(
            {
                "kind": "video-composition-timing-observation",
                "aacAssessment": {
                    "originalPCMExtentAccepted": True,
                    "packetDecodedPresentationAccepted": True,
                    "trackEditMappingAccepted": True,
                    "decodedSamples": 441,
                    "decodedStart": "0",
                    "decodedEnd": "1/100",
                    "candidate": sources["decoded"],
                    "source": sources["voice"],
                },
            }
        )
    )
    executable = tmp_path / "ffmpeg"
    executable.write_text("synthetic executable pin")
    copies = {}
    for role in sources:
        output = tmp_path / (role + ".wav")
        with wave.open(str(output), "wb") as stream:
            stream.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
            stream.writeframes(bytes(441 * 4))
        stage = {
            "name": role,
            "command": list(
                quality.pcm16_command(str(executable), sources[role]["path"], str(output))
            ),
            "inputs": [str(executable), sources[role]["path"]],
            "outputs": [str(output)],
            "timeout_seconds": 30,
        }
        graph = tmp_path / (role + ".graph.json")
        graph.write_text(json.dumps({"version": 1, "stages": [stage]}))
        checkpoint = tmp_path / (role + ".checkpoint.json")
        checkpoint.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "fingerprint": _fingerprint(_stage(stage)),
                    "outputs": {str(output): pin(output)["sha256"]},
                }
            )
        )
        copies[role] = {
            "source": sources[role],
            "output": pin(output),
            "workflow": pin(graph),
            "checkpoint": pin(checkpoint),
            "stage": role,
        }
    config = {
        "version": 1,
        "kind": "native-tempo-quality-request",
        "pairRequest": pin(original),
        "pairResult": pin(observed),
        "composition": pin(composition),
        "mediaRoot": str(tmp_path),
        "copies": copies,
        "ffmpeg": pin(executable),
        "protection": "all-selected-samples-conservative",
        "modelConfig": pin(executable),
        "modelWeights": pin(executable),
        "hfCache": str(tmp_path),
        "runtime": [pin(executable)],
    }
    request = tmp_path / "quality-request.json"
    request.write_text(json.dumps(config))
    return request


def test_quality_consumer_preserves_existing_gate(
    quality_request: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dq, "load_model", lambda *args: object())
    monkeypatch.setattr(quality, "observe_quality_resources", lambda *args: None)

    def measure(**kwargs: object) -> dict[str, object]:
        assert kwargs["spans"] == ((0, 441),)
        assert kwargs["speech"] == ((0, 441),)
        return {"speechGatePassed": False, "musicGatePassed": True}

    monkeypatch.setattr(dq, "quality_windows", measure)
    output = tmp_path / "quality-result.json"
    digest = hashlib.sha256(quality_request.read_bytes()).hexdigest()
    quality.run(quality_request, output, sha256=digest, quality=True)
    result = json.loads(output.read_text())
    assert result["measurements"]["speechGatePassed"] is False
    assert result["qualityApproved"] is False
    assert result["originalModelRuntimeVerified"] is False
    with pytest.raises(ValueError, match="read-only"):
        quality.run(quality_request, output, sha256=digest, quality=True, verify=True)
    with pytest.raises(ValueError, match="new"):
        quality.measure_quality(quality_request, output, sha256=digest)


@pytest.mark.parametrize(
    "mutation",
    [
        "kind",
        "protection",
        "root",
        "clock",
        "extent",
        "pair",
        "output",
        "source",
        "stage",
        "graph",
        "status",
        "command",
        "runtime",
    ],
)
def test_quality_contract_rejects_tamper(quality_request: Path, mutation: str) -> None:
    data = json.loads(quality_request.read_text())

    def edit(pin: dict[str, str], change: object) -> None:
        path = Path(pin["path"])
        path.write_text(json.dumps(change))
        pin["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    if mutation in ("kind", "protection"):
        data[mutation] = "unsupported"
    elif mutation == "root":
        data["mediaRoot"] = str(quality_request / "missing")
    elif mutation == "runtime":
        data["runtime"] = []
    elif mutation in ("clock", "extent"):
        c = json.loads(Path(data["composition"]["path"]).read_text())
        c["aacAssessment"][
            "decodedSamples" if mutation == "clock" else "originalPCMExtentAccepted"
        ] = 0
        edit(data["composition"], c)
    elif mutation == "pair":
        edit(data["pairResult"], {})
    elif mutation == "source":
        data["copies"]["voice"]["source"] = data["copies"]["mixture"]["source"]
    elif mutation == "output":
        data["copies"]["voice"]["output"] = data["copies"]["mixture"]["source"]
    elif mutation == "stage":
        data["copies"]["voice"]["stage"] = "missing"
    elif mutation in ("graph", "command"):
        pin = data["copies"]["voice"]["workflow"]
        g = json.loads(Path(pin["path"]).read_text())
        if mutation == "graph":
            g["stages"] = []
        else:
            g["stages"][0]["command"].insert(1, "-shortest")
        edit(pin, g)
    else:
        pin = data["copies"]["voice"]["checkpoint"]
        c = json.loads(Path(pin["path"]).read_text())
        c["status"] = "running"
        edit(pin, c)
    quality_request.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError, EOFError)):
        quality.quality_inputs(
            quality_request, sha256=hashlib.sha256(quality_request.read_bytes()).hexdigest()
        )


def test_three_stereo_measurement_files_budget() -> None:
    assert quality.pcm_budget(
        rate=44100, channels=2, sample_bytes=2, seconds=Fraction(17), files=3, header_bytes=78
    ) == 3 * (17 * 44100 * 4 + 78)


class FakeObservedBackend:
    sources = ("vocals", "instrumental")

    def extract_vocals(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return chunk

    def extract_instrumental(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        return chunk


def test_observed_backend_preserves_exact_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phases: list[str] = []
    monkeypatch.setattr(
        quality, "observe_quality_resources", lambda path, phase: phases.append(phase)
    )
    backend = quality.ObservedBackend(FakeObservedBackend(), tmp_path)
    chunk = np.ones((2, 4), dtype=np.float32)
    assert backend.extract_vocals(chunk) is chunk
    assert backend.extract_instrumental(chunk) is chunk
    assert backend.sources == ("vocals", "instrumental")
    assert phases == ["after-inference-000000", "after-instrumental-000001"]


def test_quality_resource_observation_distinguishes_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(quality, "audit_descriptors", lambda *args: None)
    monkeypatch.setattr(os, "getpid", lambda: 42)
    monkeypatch.setattr(SystemHost, "birth", lambda *args: "123:456")
    monkeypatch.setattr(SystemHost, "footprint", lambda *args: 9876)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(
            get_active_memory=lambda: 100, get_peak_memory=lambda: 200, get_cache_memory=lambda: 50
        ),
    )
    quality.observe_quality_resources(tmp_path, "before-load")
    value = json.loads((tmp_path / "before-load-memory.json").read_text())
    assert value == {
        "version": 1,
        "pid": 42,
        "birth": "123:456",
        "physicalFootprintBytes": 9876,
        "mlx": {"activeBytes": 100, "peakBytes": 200, "cacheBytes": 50},
        "peakReset": False,
        "hardAllocationLimitVerified": False,
    }


@pytest.mark.parametrize("invalid", [-1, True, "100"])
def test_quality_resource_rejects_invalid_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: object
) -> None:
    monkeypatch.setattr(quality, "audit_descriptors", lambda *args: None)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(get_active_memory=lambda: invalid),
    )
    with pytest.raises(ValueError, match="Invalid MLX"):
        quality.observe_quality_resources(tmp_path, "before-load")


def test_quality_resource_rejects_external_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject(*args: object) -> None:
        raise ValueError("external write")

    monkeypatch.setattr(quality, "audit_descriptors", reject)
    with pytest.raises(ValueError, match="external write"):
        quality.observe_quality_resources(tmp_path, "after-load")
    assert not (tmp_path / "after-load-memory.json").exists()


def test_quality_resource_rejects_process_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    births = iter(("1:2", "1:3"))
    monkeypatch.setattr(quality, "audit_descriptors", lambda *args: None)
    monkeypatch.setattr(SystemHost, "birth", lambda *args: next(births))
    monkeypatch.setattr(SystemHost, "footprint", lambda *args: 9876)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(
            get_active_memory=lambda: 1, get_peak_memory=lambda: 2, get_cache_memory=lambda: 0
        ),
    )
    with pytest.raises(ValueError, match="identity changed"):
        quality.observe_quality_resources(tmp_path, "after-load")
