"""Synthetic timing boxes and immutable raw AAC clocks; no decoder or app calls."""

import hashlib
import io
import json
import struct
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import pytest

from replay import aac_presentation_acceptance as aac
from replay.core import ReplayError
from replay.recordings import record


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def clock_box(kind: bytes, *, duration: int = 1920, version: int = 0) -> bytes:
    payload = (
        bytes(12) + struct.pack(">II", 48000, duration)
        if version == 0
        else b"\1" + bytes(19) + struct.pack(">IQ", 48000, duration)
    )
    return box(kind, payload)


def edit_box(
    *, version: int = 0, duration: int = 1920, media: int = 64, rate: int = 65536
) -> bytes:
    entry = (
        struct.pack(">Iii", duration, media, rate)
        if version == 0
        else struct.pack(">Qqi", duration, media, rate)
    )
    return box(b"elst", bytes([version]) + bytes(3) + struct.pack(">I", 1) + entry)


def track(
    *,
    identifier: int,
    handler: bytes,
    edit: bytes = b"",
    duration: int = 2048,
    version: int = 0,
    roll: bool = False,
) -> bytes:
    header = (
        bytes(12) + struct.pack(">I", identifier)
        if version == 0
        else b"\1" + bytes(19) + struct.pack(">I", identifier)
    )
    groups = (
        box(b"minf", box(b"stbl", box(b"sgpd", bytes(4)) + box(b"sbgp", bytes(4)))) if roll else b""
    )
    media = box(
        b"mdia",
        clock_box(b"mdhd", duration=duration, version=version)
        + box(b"hdlr", bytes(8) + handler + bytes(8))
        + groups,
    )
    return box(b"trak", box(b"tkhd", header) + (box(b"edts", edit) if edit else b"") + media)


def movie(
    *, edit: bytes | None = None, version: int = 0, roll: bool = False, audio_duration: int = 2048
) -> bytes:
    audio = track(
        identifier=2,
        handler=b"soun",
        edit=edit_box(version=version) if edit is None else edit,
        version=version,
        roll=roll,
        duration=audio_duration,
    )
    video = track(identifier=1, handler=b"vide", duration=1920, version=version)
    return (
        box(b"ftyp", b"isom" + bytes(4))
        + box(b"moov", clock_box(b"mvhd", version=version) + video + audio)
        + box(b"mdat", b"synthetic")
    )


def timing(payload: bytes) -> aac.Container:
    return aac.parse_container_headers(io.BytesIO(payload), size=len(payload))


def test_composition_evidence_cannot_bind_another_container() -> None:
    parsed = timing(movie())
    video = replace(parsed.tracks[0], edits=(aac.Edit(1920, 64, Fraction(1)),))
    tracks = (video, parsed.tracks[1])
    container = aac.Container(
        parsed.timescale,
        parsed.duration,
        tracks,
        parsed.timing_payload_sha256,
        aac._facts_hash(parsed.timescale, parsed.duration, tracks, parsed.timing_payload_sha256),
    )
    bad = evidence("composition.json", {"timing": parsed.json()})
    inputs = aac.Inputs(
        aac.Reference(Path("/synthetic/final.mp4"), "a" * 64),
        decoded(),
        packets(),
        video_composition=bad,
    )
    with pytest.raises(ReplayError, match="different container"):
        aac.assess(container, inputs)


def evidence(name: str, document: object) -> aac.Evidence:
    payload = json.dumps(document, sort_keys=True).encode()
    return aac.Evidence(
        aac.Reference(Path("/synthetic") / name, hashlib.sha256(payload).hexdigest()), payload
    )


def audio_stream(*, duration: int = 1920) -> dict[str, object]:
    return {
        "codec_type": "audio",
        "codec_name": "aac",
        "profile": "LC",
        "sample_rate": "48000",
        "channels": 2,
        "time_base": "1/48000",
        "start_pts": 0,
        "duration_ts": duration,
    }


def decoded(*, second_pts: int = 960, second_samples: int = 960) -> aac.Evidence:
    return evidence(
        "frames.json",
        {
            "streams": [audio_stream()],
            "frames": [
                {"pts": 0, "nb_samples": 960},
                {"pts": second_pts, "nb_samples": second_samples},
            ],
        },
    )


def packets(*, duration: int = 1024, discard: int = 64) -> aac.Evidence:
    return evidence(
        "packets.json",
        {
            "streams": [audio_stream()],
            "packets": [
                {
                    "pts": -64,
                    "duration": 1024,
                    "side_data_list": [
                        {"side_data_type": "Skip Samples", "skip_samples": 64, "discard_padding": 0}
                    ],
                },
                {
                    "pts": 960,
                    "duration": duration,
                    "side_data_list": [
                        {
                            "side_data_type": "Skip Samples",
                            "skip_samples": 0,
                            "discard_padding": discard,
                        }
                    ],
                },
            ],
        },
    )


@pytest.fixture
def inputs() -> aac.Inputs:
    return aac.Inputs(aac.Reference(Path("/synthetic/final.mp4"), "a" * 64), decoded(), packets())


def content_report(inputs: aac.Inputs, source: aac.SourcePCM) -> aac.Evidence:
    return evidence(
        "quality.json",
        {
            "version": 1,
            "kind": "delivery-audio-quality",
            "phase": "quality",
            "accepted": True,
            "inputs": [inputs.candidate.json(), source.reference.json()],
            "checks": {
                "selectedPCMReceiptBound": True,
                "nativeTimesPreserved": True,
                "aacSampleTimeMapping": True,
                "speechGatePassed": True,
                "musicGatePassed": True,
                "inputsStable": True,
            },
            "measurements": {
                "aac": {
                    "selectedSamples": source.samples,
                    "decodedSamples": 1920,
                    "sampleRate": 48000,
                    "firstDecodedPTS": "0",
                },
                "quality": {
                    "policy": {
                        "minimum_instrumental_drop_db": 10.0,
                        "maximum_speech_loss_db": 1.5,
                        "quiet_instrumental_db": -60.0,
                    },
                    "minimumSpeechCorrelation": 0.995,
                    "speech": {
                        "correlation": 1.0,
                        "level_delta_db_p50": 0.0,
                        "level_delta_db_p05": 0.0,
                    },
                    "protectedSamples": source.samples,
                    "speechGatePassed": True,
                    "musicGatePassed": True,
                    "windows": [
                        {
                            "startSample": 0,
                            "endSample": source.samples,
                            "protectedSamples": source.samples,
                            "speechGatePassed": True,
                            "speech": {
                                "correlation": 1.0,
                                "level_delta_db_p50": 0.0,
                                "level_delta_db_p05": 0.0,
                            },
                            "music": {"passed": True},
                        }
                    ],
                },
            },
        },
    )


def source_pcm(*, rate: int = 48000, samples: int = 1920) -> aac.SourcePCM:
    reference = aac.Reference(Path("/synthetic/original.wav"), "b" * 64)
    measured = evidence(
        "original-clock.json",
        {
            "version": 1,
            "kind": "original-pcm-clock-observation",
            "source": reference.json(),
            "sampleRate": rate,
            "sampleCount": samples,
            "start": "0",
        },
    )
    return aac.SourcePCM(reference, rate, samples, clock_evidence=measured)


def test_edits_are_preserved_and_do_not_certify_source_sync(inputs: aac.Inputs) -> None:
    result = aac.assess(timing(movie(roll=True)), inputs)
    assert result["packetDecodedPresentationAccepted"] is True
    assert result["trackEditMappingAccepted"] is True
    assert result["youtubeNoEditListsAccepted"] is False
    assert result["originalPCMExtentAccepted"] is False
    assert result["originalPCMContentAccepted"] is False
    assert result["presentationSyncAccepted"] is False
    assert result["decodedSamples"] == 1920
    assert result["decodedStart"] == "0"
    assert result["decodedEnd"] == "1/25"
    assert result["primingAppliedAgain"] is False
    assert result["syncApproved"] is False
    assert result["deliveryApproved"] is False
    assert timing(movie(roll=True)).tracks[1].roll_metadata_present is True


def test_version_one_64_bit_clocks_and_signed_edit_fields() -> None:
    parsed = timing(movie(version=1))
    assert parsed.timescale == 48000
    assert parsed.duration == 1920
    assert parsed.tracks[1].edits == (aac.Edit(1920, 64, Fraction(1)),)


def test_no_editlist_does_not_imply_source_acceptance(inputs: aac.Inputs) -> None:
    parsed = timing(movie(edit=b"", audio_duration=1920))
    result = aac.assess(parsed, inputs)
    assert result["youtubeNoEditListsAccepted"] is True
    assert result["trackEditMappingAccepted"] is True
    assert result["presentationSyncAccepted"] is False


def test_original_pcm_extent_is_required_even_when_content_report_claims_pass(
    inputs: aac.Inputs,
) -> None:
    source = source_pcm(samples=3456)
    result = aac.assess(
        timing(movie()),
        replace(inputs, source=source, content_comparison=content_report(inputs, source)),
    )
    assert result["originalPCMExtentAccepted"] is False
    assert result["originalPCMContentAccepted"] is False
    assert result["presentationSyncAccepted"] is False
    assert result["decodedSamplesMinusOriginalProjection"] == "-1536"


def test_extent_alone_does_not_certify_original_content(inputs: aac.Inputs) -> None:
    source = source_pcm()
    result = aac.assess(timing(movie()), replace(inputs, source=source))
    assert result["originalPCMExtentAccepted"] is True
    assert result["originalPCMContentAccepted"] is False
    assert result["presentationSyncAccepted"] is False


def test_existing_full_quality_evidence_is_bound_to_original_and_candidate(
    inputs: aac.Inputs,
) -> None:
    source = source_pcm()
    result = aac.assess(
        timing(movie()),
        replace(inputs, source=source, content_comparison=content_report(inputs, source)),
    )
    assert result["originalPCMExtentAccepted"] is True
    assert result["originalPCMContentAccepted"] is True
    assert result["presentationSyncAccepted"] is True
    assert result["youtubeNoEditListsAccepted"] is False
    assert result["qualityApproved"] is False


def test_rational_source_rate_conversion_uses_actual_duration(inputs: aac.Inputs) -> None:
    source = source_pcm(rate=44100, samples=1764)
    result = aac.assess(timing(movie()), replace(inputs, source=source))
    assert result["originalPCMExtentAccepted"] is True
    assert result["decodedSamplesMinusOriginalProjection"] == "0"
    assert result["originalPCMContentAccepted"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("accepted", False),
        ("phase", "mapping"),
        ("kind", "invented"),
        ("version", 2),
        ("inputs", []),
    ],
)
def test_quality_claims_and_wrong_input_binding_do_not_pass(
    inputs: aac.Inputs, field: str, value: object
) -> None:
    source = source_pcm()
    report = content_report(inputs, source).document()
    report[field] = value
    result = aac.assess(
        timing(movie()),
        replace(inputs, source=source, content_comparison=evidence("quality.json", report)),
    )
    assert result["originalPCMContentAccepted"] is False


def test_changed_candidate_cannot_reuse_content_evidence(inputs: aac.Inputs) -> None:
    source = source_pcm()
    result = aac.assess(
        timing(movie()),
        replace(
            inputs,
            candidate=aac.Reference(Path("/synthetic/final.mp4"), "c" * 64),
            source=source,
            content_comparison=content_report(inputs, source),
        ),
    )
    assert result["originalPCMContentAccepted"] is False


@pytest.mark.parametrize("media,rate", [(-1, 65536), (64, 0), (64, 32768), (64, -65536)])
def test_empty_dwell_and_unknown_rates_remain_unsupported(
    inputs: aac.Inputs, media: int, rate: int
) -> None:
    result = aac.assess(timing(movie(edit=edit_box(media=media, rate=rate))), inputs)
    assert result["trackEditMappingAccepted"] is False
    assert result["presentationSyncAccepted"] is False


def test_multiple_edits_are_not_silently_joined(inputs: aac.Inputs) -> None:
    edits = box(
        b"elst",
        bytes(4) + struct.pack(">I", 2) + struct.pack(">IiiIii", 960, 64, 65536, 960, 1024, 65536),
    )
    result = aac.assess(timing(movie(edit=edits)), inputs)
    assert result["trackEditMappingAccepted"] is False


def test_edit_exceeding_actual_media_is_rejected(inputs: aac.Inputs) -> None:
    result = aac.assess(timing(movie(edit=edit_box(media=129))), inputs)
    assert result["trackEditMappingAccepted"] is False


def test_late_decoded_gap_is_not_hidden_by_equal_end_and_sample_summary(inputs: aac.Inputs) -> None:
    result = aac.assess(
        timing(movie()), replace(inputs, decoded=decoded(second_pts=961, second_samples=959))
    )
    assert result["packetDecodedPresentationAccepted"] is False
    assert result["decodedEnd"] == "1/25"
    assert result["packetReasons"] == ["Decoded AAC clock has gaps or overlaps"]


def test_late_decoded_overlap_is_rejected(inputs: aac.Inputs) -> None:
    result = aac.assess(
        timing(movie()), replace(inputs, decoded=decoded(second_pts=959, second_samples=961))
    )
    assert result["packetDecodedPresentationAccepted"] is False


def test_negative_coded_packet_pts_with_declared_skip_is_valid(inputs: aac.Inputs) -> None:
    assert aac.assess(timing(movie()), inputs)["packetDecodedPresentationAccepted"] is True


def test_double_priming_application_does_not_pass(inputs: aac.Inputs) -> None:
    raw = decoded().document()
    raw["frames"] = [{"pts": 64, "nb_samples": 960}, {"pts": 1024, "nb_samples": 960}]
    result = aac.assess(timing(movie()), replace(inputs, decoded=evidence("frames.json", raw)))
    assert result["packetDecodedPresentationAccepted"] is False
    assert result["trackEditMappingAccepted"] is False


def test_packet_duration_already_clipped_must_not_discard_twice(inputs: aac.Inputs) -> None:
    result = aac.assess(timing(movie()), replace(inputs, packets=packets(duration=960)))
    assert result["packetDecodedPresentationAccepted"] is True
    assert result["primingAppliedAgain"] is False


def test_removed_skip_discard_payload_copy_is_not_accepted(inputs: aac.Inputs) -> None:
    raw = packets().document()
    raw["packets"] = [{"pts": 0, "duration": 1024}, {"pts": 1024, "duration": 1024}]
    result = aac.assess(timing(movie()), replace(inputs, packets=evidence("packets.json", raw)))
    assert result["packetDecodedPresentationAccepted"] is False


def test_packet_gap_is_not_hidden_by_total_duration(inputs: aac.Inputs) -> None:
    raw = packets().document()
    raw["packets"] = [{"pts": -64, "duration": 1023}, {"pts": 960, "duration": 1024}]
    result = aac.assess(timing(movie()), replace(inputs, packets=evidence("packets.json", raw)))
    assert result["packetDecodedPresentationAccepted"] is False


def test_interior_skip_and_discard_are_rejected(inputs: aac.Inputs) -> None:
    raw = packets().document()
    raw["packets"] = [
        {
            "pts": -64,
            "duration": 1024,
            "side_data_list": [
                {"side_data_type": "Skip Samples", "skip_samples": 64, "discard_padding": 1}
            ],
        },
        {
            "pts": 960,
            "duration": 1024,
            "side_data_list": [
                {"side_data_type": "Skip Samples", "skip_samples": 1, "discard_padding": 64}
            ],
        },
    ]
    result = aac.assess(timing(movie()), replace(inputs, packets=evidence("packets.json", raw)))
    assert result["packetDecodedPresentationAccepted"] is False


def test_probe_payload_mutation_after_pin_is_rejected(inputs: aac.Inputs) -> None:
    with pytest.raises(ReplayError, match="bytes differ"):
        aac.assess(
            timing(movie()),
            replace(inputs, decoded=replace(inputs.decoded, payload=inputs.decoded.payload + b" ")),
        )


@pytest.mark.parametrize("payload", [b"", b"{", b"\xff"])
def test_bad_json_cannot_claim_observation_success(payload: bytes) -> None:
    pin = aac.Reference(Path("/synthetic/invalid.json"), hashlib.sha256(payload).hexdigest())
    with pytest.raises(ReplayError):
        aac.Evidence(pin, payload).document()


@pytest.mark.parametrize("duration,media", [(0, 64), (1920, -2)])
def test_invalid_elst_entries_fail_closed(duration: int, media: int) -> None:
    with pytest.raises(ReplayError, match="Invalid edit-list"):
        timing(movie(edit=edit_box(duration=duration, media=media)))


@pytest.mark.parametrize(
    "payload",
    [b"\2" + bytes(7), bytes(8), bytes(4) + struct.pack(">I", 33), bytes(4) + struct.pack(">I", 1)],
)
def test_unknown_empty_oversized_or_truncated_editlists(payload: bytes) -> None:
    with pytest.raises(ReplayError):
        timing(movie(edit=box(b"elst", payload)))


def test_duplicate_track_ids_are_rejected() -> None:
    payload = box(
        b"moov",
        clock_box(b"mvhd")
        + track(identifier=1, handler=b"vide")
        + track(identifier=1, handler=b"soun"),
    )
    with pytest.raises(ReplayError, match="Duplicate timing-track IDs"):
        timing(payload)


def test_duplicate_editlists_are_rejected() -> None:
    with pytest.raises(ReplayError, match="Duplicate edit lists"):
        timing(movie(edit=edit_box() + edit_box()))


def test_unknown_handler_is_not_an_audio_guess() -> None:
    payload = box(b"moov", clock_box(b"mvhd") + track(identifier=1, handler=b"meta"))
    with pytest.raises(ReplayError, match="handler"):
        timing(payload)


def test_multiple_audio_tracks_cannot_pass_single_audio_contract(inputs: aac.Inputs) -> None:
    payload = box(
        b"moov",
        clock_box(b"mvhd")
        + track(identifier=1, handler=b"vide", duration=1920)
        + track(identifier=2, handler=b"soun", edit=edit_box())
        + track(identifier=3, handler=b"soun", edit=edit_box()),
    )
    result = aac.assess(timing(payload), inputs)
    assert result["trackEditMappingAccepted"] is False


def test_mixed_probe_streams_are_selected_by_observed_audio_index(inputs: aac.Inputs) -> None:
    decoded_raw = {
        "streams": [{"codec_type": "video", "index": 0}, {**audio_stream(), "index": 1}],
        "frames": [
            {"stream_index": 0, "pts": 0},
            {"stream_index": 1, "pts": 0, "nb_samples": 1920},
        ],
    }
    result = aac.assess(
        timing(movie()), replace(inputs, decoded=evidence("frames.json", decoded_raw))
    )
    assert result["packetDecodedPresentationAccepted"] is True


def test_header_observer_is_readonly_and_does_not_claim_whole_sha(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.mp4"
    payload = movie()
    path.write_bytes(payload)
    identity = aac.FileIdentity.from_stat(path.stat())
    result = aac.observe_headers(aac.Reference(path, "f" * 64), expected_identity=identity)
    assert result["expectedWholeSHA256FreshlyVerified"] is False
    assert result["fullDecodePerformed"] is False
    assert result["identityBefore"] == result["identityAfter"]
    assert path.read_bytes() == payload
    assert sorted(child.name for child in tmp_path.iterdir()) == ["synthetic.mp4"]


def test_header_observer_rejects_changed_initial_identity(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.mp4"
    path.write_bytes(movie())
    identity = aac.FileIdentity.from_stat(path.stat())
    with pytest.raises(ReplayError, match="expected identity"):
        aac.observe_headers(
            aac.Reference(path, "f" * 64),
            expected_identity=replace(identity, size=identity.size + 1),
        )


def test_header_observer_detects_terminal_file_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "synthetic.mp4"
    path.write_bytes(movie())
    identity = aac.FileIdentity.from_stat(path.stat())
    original = aac.parse_container_headers

    def changed(stream: BinaryIO, *, size: int) -> aac.Container:
        result = original(stream, size=size)
        with path.open("ab") as output:
            output.write(b"changed")
        return result

    monkeypatch.setattr(aac, "parse_container_headers", changed)
    with pytest.raises(ReplayError, match="changed during"):
        aac.observe_headers(aac.Reference(path, "f" * 64), expected_identity=identity)


def test_header_observer_detects_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "synthetic.mp4"
    path.write_bytes(movie())
    identity = aac.FileIdentity.from_stat(path.stat())
    original = aac.parse_container_headers

    def changed(stream: BinaryIO, *, size: int) -> aac.Container:
        result = original(stream, size=size)
        replacement = tmp_path / "replacement.mp4"
        replacement.write_bytes(movie())
        replacement.replace(path)
        return result

    monkeypatch.setattr(aac, "parse_container_headers", changed)
    with pytest.raises(ReplayError, match="changed during"):
        aac.observe_headers(aac.Reference(path, "f" * 64), expected_identity=identity)


def test_header_observer_rejects_links(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.mp4"
    path.write_bytes(movie())
    alias = tmp_path / "alias.mp4"
    alias.symlink_to(path)
    with pytest.raises(ReplayError, match="links"):
        aac.observe_headers(
            aac.Reference(alias, "f" * 64),
            expected_identity=aac.FileIdentity.from_stat(path.stat()),
        )


@pytest.mark.parametrize("value", [0, -1, True])
def test_invalid_container_size_is_rejected(value: int) -> None:
    with pytest.raises(ReplayError, match="size"):
        aac.parse_container_headers(io.BytesIO(movie()), size=value)


@pytest.mark.parametrize(
    "path,sha",
    [("relative", "a" * 64), ("/synthetic/../escape", "a" * 64), ("/synthetic/good", "short")],
)
def test_bad_evidence_reference_is_rejected(path: str, sha: str) -> None:
    with pytest.raises(ReplayError, match="reference"):
        aac.Reference(Path(path), sha)


@pytest.mark.parametrize(
    "rate,samples,start",
    [(0, 1920, Fraction(0)), (48000, 0, Fraction(0)), (48000, 1920, Fraction(-1))],
)
def test_bad_original_pcm_clock_is_rejected(rate: int, samples: int, start: Fraction) -> None:
    with pytest.raises(ReplayError, match="PCM"):
        aac.SourcePCM(
            aac.Reference(Path("/synthetic/original.wav"), "b" * 64), rate, samples, start
        )


def test_expected_pcm_count_without_observed_clock_is_not_extent_acceptance(
    inputs: aac.Inputs,
) -> None:
    source = aac.SourcePCM(aac.Reference(Path("/synthetic/original.wav"), "b" * 64), 48000, 1920)
    result = aac.assess(
        timing(movie()),
        replace(inputs, source=source, content_comparison=content_report(inputs, source)),
    )
    assert result["originalPCMExtentAccepted"] is False
    assert result["originalPCMContentAccepted"] is False
    assert result["presentationSyncAccepted"] is False


def test_mutated_original_pcm_count_cannot_hide_terminal_shortfall(inputs: aac.Inputs) -> None:
    source = replace(source_pcm(samples=3456), samples=1920)
    with pytest.raises(ReplayError, match="clock evidence differs"):
        aac.assess(timing(movie()), replace(inputs, source=source))


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("kind", "expected-count"),
        ("source", {}),
        ("sampleRate", 44100),
        ("sampleCount", 3456),
        ("start", "1/25"),
    ],
)
def test_missing_or_wrong_original_clock_binding_is_rejected(
    inputs: aac.Inputs, field: str, value: object
) -> None:
    source = source_pcm()
    original = source.clock_evidence
    assert original is not None
    raw = original.document()
    raw[field] = value
    with pytest.raises(ReplayError, match="clock evidence differs"):
        aac.assess(
            timing(movie()),
            replace(inputs, source=replace(source, clock_evidence=evidence("clock.json", raw))),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("windows", []),
        ("policy", {}),
        ("minimumSpeechCorrelation", 0.1),
        ("speech", None),
        ("protectedSamples", 0),
        ("speechGatePassed", False),
        ("musicGatePassed", False),
    ],
)
def test_bare_or_incomplete_quality_claim_cannot_pass(
    inputs: aac.Inputs, field: str, value: object
) -> None:
    source = source_pcm()
    raw = content_report(inputs, source).document()
    quality = record(record(raw["measurements"])["quality"])
    quality[field] = value
    result = aac.assess(
        timing(movie()),
        replace(inputs, source=source, content_comparison=evidence("quality.json", raw)),
    )
    assert result["originalPCMContentAccepted"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("correlation", 0.8),
        ("level_delta_db_p50", -2.0),
        ("level_delta_db_p05", -2.0),
        ("correlation", True),
        ("correlation", "1"),
        ("correlation", float("nan")),
    ],
)
def test_changed_measured_speech_stats_cannot_reuse_pass_flag(
    inputs: aac.Inputs, field: str, value: object
) -> None:
    source = source_pcm()
    raw = content_report(inputs, source).document()
    speech = record(record(record(raw["measurements"])["quality"])["speech"])
    speech[field] = value
    result = aac.assess(
        timing(movie()),
        replace(inputs, source=source, content_comparison=evidence("quality.json", raw)),
    )
    assert result["originalPCMContentAccepted"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("startSample", 1),
        ("endSample", 960),
        ("endSample", 1921),
        ("protectedSamples", 1921),
        ("speechGatePassed", False),
        ("speech", None),
        ("music", {"passed": False}),
    ],
)
def test_full_quality_windows_must_preserve_coverage_and_gates(
    inputs: aac.Inputs, field: str, value: object
) -> None:
    source = source_pcm()
    raw = content_report(inputs, source).document()
    quality = record(record(raw["measurements"])["quality"])
    windows = quality["windows"]
    assert isinstance(windows, list)
    record(windows[0])[field] = value
    result = aac.assess(
        timing(movie()),
        replace(inputs, source=source, content_comparison=evidence("quality.json", raw)),
    )
    assert result["originalPCMContentAccepted"] is False


def test_empty_mask_window_is_not_a_speech_comparison_but_can_be_observed(
    inputs: aac.Inputs,
) -> None:
    source = source_pcm()
    raw = content_report(inputs, source).document()
    quality = record(record(raw["measurements"])["quality"])
    quality["protectedSamples"] = 960
    quality["windows"] = [
        {
            "startSample": 0,
            "endSample": 960,
            "protectedSamples": 0,
            "speechGatePassed": None,
            "speech": None,
            "music": {"passed": True},
        },
        {
            "startSample": 960,
            "endSample": 1920,
            "protectedSamples": 960,
            "speechGatePassed": True,
            "speech": {"correlation": 1.0, "level_delta_db_p50": 0.0, "level_delta_db_p05": 0.0},
            "music": {"passed": True},
        },
    ]
    result = aac.assess(
        timing(movie()),
        replace(inputs, source=source, content_comparison=evidence("quality.json", raw)),
    )
    assert result["originalPCMContentAccepted"] is True


def test_empty_mask_window_cannot_claim_speech_measurement(inputs: aac.Inputs) -> None:
    source = source_pcm()
    raw = content_report(inputs, source).document()
    quality = record(record(raw["measurements"])["quality"])
    quality["windows"] = [
        {
            "startSample": 0,
            "endSample": 1920,
            "protectedSamples": 0,
            "speechGatePassed": True,
            "speech": {"correlation": 1.0},
            "music": {"passed": True},
        }
    ]
    result = aac.assess(
        timing(movie()),
        replace(inputs, source=source, content_comparison=evidence("quality.json", raw)),
    )
    assert result["originalPCMContentAccepted"] is False


@pytest.mark.parametrize(
    "change", [{"codec_name": "opus"}, {"profile": "HE"}, {"time_base": "1/44100"}, {"channels": 1}]
)
def test_wrong_packet_codec_or_clock_is_not_aac_evidence(
    inputs: aac.Inputs, change: dict[str, object]
) -> None:
    raw = packets().document()
    raw["streams"] = [{**audio_stream(), **change}]
    with pytest.raises(ReplayError, match="contracts differ"):
        aac.assess(timing(movie()), replace(inputs, packets=evidence("packets.json", raw)))


@pytest.mark.parametrize("rows", [[], "not-list", [{"pts": True, "duration": 1024}]])
def test_missing_or_nonintegral_packet_clock_is_rejected(inputs: aac.Inputs, rows: object) -> None:
    raw = packets().document()
    raw["packets"] = rows
    with pytest.raises(ReplayError):
        aac.assess(timing(movie()), replace(inputs, packets=evidence("packets.json", raw)))


@pytest.mark.parametrize(
    "side",
    [
        "not-list",
        [{"side_data_type": "Skip Samples", "skip_samples": 65537}],
        [{"side_data_type": "Skip Samples"}, {"side_data_type": "Skip Samples"}],
    ],
)
def test_invalid_or_ambiguous_skip_samples_is_rejected(inputs: aac.Inputs, side: object) -> None:
    raw = packets().document()
    raw["packets"] = [
        {"pts": -64, "duration": 1024, "side_data_list": side},
        {"pts": 960, "duration": 1024},
    ]
    with pytest.raises(ReplayError):
        aac.assess(timing(movie()), replace(inputs, packets=evidence("packets.json", raw)))


def test_typed_container_mutation_cannot_bypass_parser() -> None:
    with pytest.raises(ReplayError, match="typed container"):
        replace(timing(movie()), timescale=0)


def test_typed_track_mutation_cannot_bypass_parser() -> None:
    with pytest.raises(ReplayError, match="typed track"):
        replace(timing(movie()).tracks[0], timescale=0)


def test_typed_edit_mutation_cannot_bypass_parser() -> None:
    with pytest.raises(ReplayError, match="typed edit"):
        aac.Edit(1920, -2, Fraction(1))


def test_nonregular_container_identity_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="regular file"):
        aac.FileIdentity.from_stat(tmp_path.stat())


def test_duplicate_observation_json_keys_are_rejected() -> None:
    payload = b'{"sampleCount":1920,"sampleCount":3456}'
    pin = aac.Reference(Path("/synthetic/duplicate.json"), hashlib.sha256(payload).hexdigest())
    with pytest.raises(ReplayError, match="Duplicate"):
        aac.Evidence(pin, payload).document()


@dataclass(frozen=True)
class LegacyFixture:
    request: aac.Reference
    document: dict[str, object]
    output: Path
    directory: Path


def local_bytes(path: Path, payload: bytes) -> aac.Reference:
    path.write_bytes(payload)
    return aac.Reference(path, hashlib.sha256(payload).hexdigest())


def local_json(path: Path, document: dict[str, object]) -> aac.Reference:
    return local_bytes(path, json.dumps(document, sort_keys=True).encode())


@pytest.fixture
def legacy(tmp_path: Path) -> LegacyFixture:
    candidate = local_bytes(tmp_path / "original.mp4", movie())
    identity = aac.FileIdentity.from_stat(candidate.path.stat())
    ffmpeg = local_bytes(tmp_path / "ffmpeg", b"synthetic-ffmpeg")
    ffprobe = local_bytes(tmp_path / "ffprobe", b"synthetic-ffprobe")
    ffmpeg.path.chmod(0o700)
    ffprobe.path.chmod(0o700)
    audio = {**audio_stream(), "index": 1}
    streams: dict[str, object] = {"streams": [{"codec_type": "video", "index": 0}, audio]}
    frames: dict[str, object] = {
        "frames": [
            {"media_type": "video", "pts": 0},
            {"media_type": "audio", "pts": 0, "nb_samples": 960},
            {"media_type": "audio", "pts": 960, "nb_samples": 960},
        ]
    }
    raw_packets = {
        "streams": [{"index": 0, "nb_read_packets": "1"}, {"index": 1, "nb_read_packets": "2"}],
        "packets": [
            {"stream_index": 0, "pts": 0, "duration": 1920},
            {
                "stream_index": 1,
                "pts": -64,
                "duration": 1024,
                "side_data_list": [
                    {"side_data_type": "Skip Samples", "skip_samples": 64, "discard_padding": 0}
                ],
            },
            {
                "stream_index": 1,
                "pts": 960,
                "duration": 1024,
                "side_data_list": [
                    {"side_data_type": "Skip Samples", "skip_samples": 0, "discard_padding": 64}
                ],
            },
        ],
    }
    binding = {
        **candidate.json(),
        "device": identity.device,
        "inode": identity.inode,
        "bytes": identity.size,
        "mtimeNs": identity.modified_ns,
        "ctimeNs": identity.changed_ns,
    }
    receipt = local_json(
        tmp_path / "receipt.json",
        {
            "version": 1,
            "fullDecode": True,
            "technicalAccepted": True,
            "inputs": [binding],
            "inputsAfter": [binding],
            "tools": {"ffmpeg": str(ffmpeg.path), "ffprobe": str(ffprobe.path)},
            "streams": streams,
            "audioDecodedClock": {"count": 2, "sampleCount": 1920},
            "audioPacketClock": {"count": 2},
        },
    )
    checkpoint = local_json(
        tmp_path / "checkpoint.json",
        {"status": "complete", "outputs": {str(receipt.path): receipt.sha256}},
    )
    prefix = [
        str(ffprobe.path),
        "-v",
        "error",
        "-threads",
        "0",
        "-protocol_whitelist",
        "file,pipe",
        "-of",
        "json",
    ]
    commands = {
        "decode": [
            str(ffmpeg.path),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-max_error_rate",
            "0",
            "-protocol_whitelist",
            "file,pipe",
            "-i",
            str(candidate.path),
            "-map",
            "0:v",
            "-map",
            "0:a",
            "-f",
            "null",
            "-",
        ],
        "streams": prefix + ["-show_streams", "-show_format", str(candidate.path)],
        "frames": prefix
        + [
            "-show_frames",
            "-show_entries",
            "frame=media_type,pts,duration,pkt_duration,nb_samples",
            str(candidate.path),
        ],
        "packets": prefix
        + [
            "-show_packets",
            "-count_packets",
            "-show_entries",
            "packet=stream_index,pts,dts,duration:stream=index,nb_read_packets",
            str(candidate.path),
        ],
    }
    raw_outputs = {
        "decode": local_bytes(tmp_path / "decode.stdout", b""),
        "streams": local_json(tmp_path / "streams.stdout", streams),
        "frames": local_json(tmp_path / "frames.stdout", frames),
        "packets": local_json(tmp_path / "packets.stdout", raw_packets),
    }
    processes = {
        role: {
            "process": local_json(
                tmp_path / (role + ".process.json"), {"returncode": 0, "command": commands[role]}
            ).json(),
            "stdout": raw_outputs[role].json(),
            "stderr": local_bytes(tmp_path / (role + ".stderr"), b"").json(),
        }
        for role in ("decode", "streams", "frames", "packets")
    }
    source = local_bytes(tmp_path / "source.wav", b"synthetic-authoritative-pcm")
    producer = local_json(
        tmp_path / "source-producer.json",
        {
            "output": str(source.path),
            "outputSHA256": source.sha256,
            "sampleRate": 48000,
            "outputSamples": 1920,
            "outputStartSample": 0,
            "outputEndSample": 1920,
        },
    )
    source_clock = local_json(
        tmp_path / "source-clock.json",
        {
            "version": 1,
            "kind": "original-pcm-clock-observation",
            "source": source.json(),
            "sampleRate": 48000,
            "sampleCount": 1920,
            "start": "0",
        },
    )
    document: dict[str, object] = {
        "version": 1,
        "kind": "aac-presentation-observation-request",
        "candidate": candidate.json(),
        "expectedIdentity": identity.json(),
        "decodeReceipt": receipt.json(),
        "completedCheckpoint": checkpoint.json(),
        "tools": {"ffmpeg": ffmpeg.json(), "ffprobe": ffprobe.json()},
        "processes": processes,
        "source": {
            "reference": source.json(),
            "sampleRate": 48000,
            "sampleCount": 1920,
            "start": "0",
            "producer": producer.json(),
            "clockEvidence": source_clock.json(),
        },
        "contentComparison": None,
    }
    request = local_json(tmp_path / "request.json", document)
    return LegacyFixture(request, document, tmp_path / "result.json", tmp_path)


def repin_request(fixture: LegacyFixture, document: dict[str, object]) -> aac.Reference:
    return local_json(fixture.request.path, document)


def test_public_cli_uses_actual_legacy_rows_without_new_decoder(legacy: LegacyFixture) -> None:
    assert (
        aac.main(
            [
                "observe",
                str(legacy.request.path),
                str(legacy.output),
                "--sha256",
                legacy.request.sha256,
            ]
        )
        == 0
    )
    report = record(json.loads(legacy.output.read_bytes()))
    result = record(report["assessment"])
    assert result["packetDecodedPresentationAccepted"] is True
    assert result["trackEditMappingAccepted"] is True
    assert result["originalPCMExtentAccepted"] is True
    assert result["originalPCMContentAccepted"] is False
    assert result["presentationSyncAccepted"] is False
    assert report["freshWholeCandidateSHA256Verified"] is False
    assert record(report["normalization"])["newDecoderRun"] is False
    normalized = record(
        json.loads(legacy.output.with_name("result.json.decoded.json").read_bytes())
    )
    assert normalized["frames"] == [
        {"media_type": "audio", "pts": 0, "nb_samples": 960},
        {"media_type": "audio", "pts": 960, "nb_samples": 960},
    ]
    assert (
        aac.main(
            [
                "verify",
                str(legacy.request.path),
                str(legacy.output),
                "--sha256",
                legacy.request.sha256,
            ]
        )
        == 0
    )


def test_verify_does_not_reobserve_headers_or_write(
    legacy: LegacyFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    aac.collect_presentation(legacy.request, legacy.output)
    before = legacy.output.stat()

    def forbidden(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("unexpected new header read")

    monkeypatch.setattr(aac, "observe_headers", forbidden)
    aac.verify_presentation(legacy.request, legacy.output)
    assert legacy.output.stat() == before


@pytest.mark.parametrize(
    "filename",
    [
        "frames.stdout",
        "packets.stdout",
        "streams.stdout",
        "frames.process.json",
        "frames.stderr",
        "ffprobe",
        "source-producer.json",
        "source-clock.json",
        "request.json",
    ],
)
def test_public_collection_rejects_changed_evidence_before_header(
    legacy: LegacyFixture, filename: str
) -> None:
    path = legacy.directory / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ReplayError):
        aac.collect_presentation(legacy.request, legacy.output)
    assert legacy.output.exists() is False


@pytest.mark.parametrize(
    "change",
    [
        {"version": 2},
        {"kind": "invented"},
        {"processes": {}},
        {"expectedIdentity": []},
        {"expectedIdentity": [1, 2, 0, 3, 4]},
        {"candidate": {"path": 7, "sha256": "f" * 64}},
    ],
)
def test_invalid_public_request_is_rejected(
    legacy: LegacyFixture, change: dict[str, object]
) -> None:
    reference = repin_request(legacy, {**legacy.document, **change})
    with pytest.raises(ReplayError):
        aac.collect_presentation(reference, legacy.output)


def test_public_request_size_is_bounded(
    legacy: LegacyFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aac, "MAX_REQUEST_BYTES", 8)
    with pytest.raises(ReplayError, match="request exceeds"):
        aac.load_observation_request(legacy.request)


def test_public_source_must_bind_original_producer(legacy: LegacyFixture) -> None:
    source = record(legacy.document["source"])
    source["sampleCount"] = 3456
    reference = repin_request(legacy, {**legacy.document, "source": source})
    with pytest.raises(ReplayError, match="PCM facts differ"):
        aac.collect_presentation(reference, legacy.output)


def test_public_missing_source_remains_unverified(legacy: LegacyFixture) -> None:
    reference = repin_request(legacy, {**legacy.document, "source": None})
    aac.collect_presentation(reference, legacy.output)
    report = record(json.loads(legacy.output.read_bytes()))
    assert record(report["assessment"])["originalPCMExtentAccepted"] is False


def test_public_missing_source_clock_preserves_extent_failure(legacy: LegacyFixture) -> None:
    source = record(legacy.document["source"])
    source["clockEvidence"] = None
    reference = repin_request(legacy, {**legacy.document, "source": source})
    aac.collect_presentation(reference, legacy.output)
    report = record(json.loads(legacy.output.read_bytes()))
    assert record(report["assessment"])["originalPCMExtentAccepted"] is False


def test_public_failed_process_and_bad_command_are_rejected(legacy: LegacyFixture) -> None:
    processes = record(legacy.document["processes"])
    frame = record(processes["frames"])
    frame["process"] = local_json(
        legacy.directory / "frames.process.json", {"returncode": 1, "command": []}
    ).json()
    reference = repin_request(
        legacy, {**legacy.document, "processes": {**processes, "frames": frame}}
    )
    with pytest.raises(ReplayError, match="unsuccessful"):
        aac.collect_presentation(reference, legacy.output)


def test_public_probe_stderr_is_not_ignored(legacy: LegacyFixture) -> None:
    processes = record(legacy.document["processes"])
    frame = record(processes["frames"])
    frame["stderr"] = local_bytes(legacy.directory / "frames.stderr", b"decoder error").json()
    reference = repin_request(
        legacy, {**legacy.document, "processes": {**processes, "frames": frame}}
    )
    with pytest.raises(ReplayError, match="reported decoder"):
        aac.collect_presentation(reference, legacy.output)


def test_public_refuses_uncertain_partial_outputs(legacy: LegacyFixture) -> None:
    legacy.output.with_name("result.json.decoded.json").write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        aac.collect_presentation(legacy.request, legacy.output)
    assert legacy.output.exists() is False


def test_public_terminal_raw_mutation_is_rejected(
    legacy: LegacyFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = aac.observe_headers

    def change(
        reference: aac.Reference, *, expected_identity: aac.FileIdentity
    ) -> dict[str, object]:
        result = original(reference, expected_identity=expected_identity)
        (legacy.directory / "frames.stdout").write_bytes(b'{"frames":[]}')
        return result

    monkeypatch.setattr(aac, "observe_headers", change)
    with pytest.raises(ReplayError):
        aac.collect_presentation(legacy.request, legacy.output)
    assert legacy.output.exists() is False


def test_public_verify_rejects_changed_derived_bytes(legacy: LegacyFixture) -> None:
    aac.collect_presentation(legacy.request, legacy.output)
    legacy.output.with_name("result.json.decoded.json").write_bytes(b"{}")
    with pytest.raises(ReplayError, match="bytes changed"):
        aac.verify_presentation(legacy.request, legacy.output)


def test_public_verify_rejects_changed_report(legacy: LegacyFixture) -> None:
    aac.collect_presentation(legacy.request, legacy.output)
    report = record(json.loads(legacy.output.read_bytes()))
    report["syncApproved"] = True
    legacy.output.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ReplayError, match="domain differs"):
        aac.verify_presentation(legacy.request, legacy.output)


def test_public_verify_rejects_changed_candidate(legacy: LegacyFixture) -> None:
    aac.collect_presentation(legacy.request, legacy.output)
    (legacy.directory / "original.mp4").write_bytes(movie() + b"changed")
    with pytest.raises(ReplayError, match="candidate changed"):
        aac.verify_presentation(legacy.request, legacy.output)


def test_public_cli_reports_validation_failure(legacy: LegacyFixture) -> None:
    legacy.request.path.write_bytes(b"changed")
    with pytest.raises(SystemExit) as error:
        aac.main(
            [
                "observe",
                str(legacy.request.path),
                str(legacy.output),
                "--sha256",
                legacy.request.sha256,
            ]
        )
    assert error.value.code == 1


def test_public_boolean_exit_status_is_not_success(legacy: LegacyFixture) -> None:
    processes = record(legacy.document["processes"])
    frame = record(processes["frames"])
    process_ref = record(frame["process"])
    document = record(json.loads(Path(str(process_ref["path"])).read_bytes()))
    document["returncode"] = False
    frame["process"] = local_json(legacy.directory / "frames.process.json", document).json()
    reference = repin_request(
        legacy, {**legacy.document, "processes": {**processes, "frames": frame}}
    )
    with pytest.raises(ReplayError, match="unsuccessful"):
        aac.collect_presentation(reference, legacy.output)


def test_public_collection_rejects_evidence_link(legacy: LegacyFixture) -> None:
    original = legacy.directory / "frames.stdout"
    moved = legacy.directory / "retained-frames.stdout"
    original.rename(moved)
    original.symlink_to(moved)
    with pytest.raises(ReplayError, match="traverse links"):
        aac.collect_presentation(legacy.request, legacy.output)


def test_public_verify_rejects_result_link_before_read(legacy: LegacyFixture) -> None:
    aac.collect_presentation(legacy.request, legacy.output)
    retained = legacy.directory / "retained-result.json"
    legacy.output.rename(retained)
    legacy.output.symlink_to(retained)
    with pytest.raises(ReplayError, match="traverse links"):
        aac.verify_presentation(legacy.request, legacy.output)


def test_public_tool_must_remain_executable(legacy: LegacyFixture) -> None:
    (legacy.directory / "ffprobe").chmod(0o600)
    with pytest.raises(ReplayError, match="configured pin"):
        aac.collect_presentation(legacy.request, legacy.output)


def test_public_metadata_boundary_outside_work_is_rejected() -> None:
    with pytest.raises(ReplayError, match="checkout edits/work"):
        aac._read(aac.Reference(Path(aac.__file__).resolve(), "a" * 64))


def test_public_output_outside_work_is_rejected(legacy: LegacyFixture) -> None:
    with pytest.raises(ReplayError, match="outputs must remain"):
        aac.collect_presentation(legacy.request, aac.ROOT / "not-a-work-result.json")


def test_terminal_pin_check_does_not_reparse_large_tables(
    legacy: LegacyFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = aac.load_observation_request(legacy.request)
    original = aac.Evidence.document

    def guarded(self: aac.Evidence) -> dict[str, object]:
        if self.reference.path.name in ("frames.stdout", "packets.stdout"):
            raise AssertionError("terminal boundary must hash bytes, not create JSON objects")
        return original(self)

    monkeypatch.setattr(aac.Evidence, "document", guarded)
    aac._terminal_pins(request)


def test_public_raw_role_cannot_rebind_an_unrelated_stdout(legacy: LegacyFixture) -> None:
    processes = record(legacy.document["processes"])
    frame = record(processes["frames"])
    frame["stdout"] = local_bytes(
        legacy.directory / "unrelated.stdout", (legacy.directory / "frames.stdout").read_bytes()
    ).json()
    reference = repin_request(
        legacy, {**legacy.document, "processes": {**processes, "frames": frame}}
    )
    with pytest.raises(ReplayError, match="receipt namespace"):
        aac.collect_presentation(reference, legacy.output)


def test_editlist_cannot_be_borrowed_from_media_container() -> None:
    malformed = box(
        b"trak",
        box(b"tkhd", bytes(12) + struct.pack(">I", 2))
        + box(
            b"mdia", clock_box(b"mdhd") + box(b"hdlr", bytes(8) + b"soun" + bytes(8)) + edit_box()
        ),
    )
    payload = box(b"moov", clock_box(b"mvhd") + malformed)
    with pytest.raises(ReplayError, match="edts"):
        timing(payload)


def test_editlist_at_wrong_depth_cannot_pass_own_track() -> None:
    malformed = box(
        b"trak",
        box(b"tkhd", bytes(12) + struct.pack(">I", 2))
        + box(b"mdia", clock_box(b"mdhd") + box(b"hdlr", bytes(8) + b"soun" + bytes(8)))
        + box(b"edts", box(b"mdia", edit_box())),
    )
    payload = box(b"moov", clock_box(b"mvhd") + malformed)
    with pytest.raises(ReplayError, match="outside its own"):
        timing(payload)


def test_valid_numeric_container_mutation_is_rejected_against_original_parse() -> None:
    with pytest.raises(ReplayError, match="changed in memory"):
        replace(timing(movie()), duration=1921)
