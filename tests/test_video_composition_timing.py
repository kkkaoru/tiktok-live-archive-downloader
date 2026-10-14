"""Synthetic reordered video and streamed legacy observations, without media tools."""

import hashlib
import io
import json
import os
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from replay import aac_presentation_acceptance as aac
from replay import video_composition_timing as video
from replay.core import ReplayError
from replay.recordings import record


def box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def table(kind: bytes, entries: tuple[tuple[int, int], ...], version: int = 0) -> bytes:
    return box(
        kind,
        bytes([version])
        + bytes(3)
        + struct.pack(">I", len(entries))
        + b"".join(
            struct.pack(">Ii" if version == 1 else ">II", count, value) for count, value in entries
        ),
    )


def clock(kind: bytes, scale: int, duration: int) -> bytes:
    return box(kind, bytes(12) + struct.pack(">II", scale, duration))


def track(
    identifier: int,
    handler: bytes,
    scale: int,
    duration: int,
    edit: int,
    span: int,
    tables: bytes = b"",
) -> bytes:
    return box(
        b"trak",
        box(b"tkhd", bytes(12) + struct.pack(">I", identifier))
        + box(
            b"edts",
            box(b"elst", bytes(4) + struct.pack(">I", 1) + struct.pack(">Iii", span, edit, 65536)),
        )
        + box(
            b"mdia",
            clock(b"mdhd", scale, duration)
            + box(b"hdlr", bytes(8) + handler + bytes(8))
            + box(b"minf", box(b"stbl", tables)),
        ),
    )


def movie(
    *, stts: bytes | None = None, ctts: bytes | None = None, duration: int = 400, edit: int = 200
) -> bytes:
    tables = (table(b"stts", ((4, 100),)) if stts is None else stts) + (
        table(b"ctts", ((1, 200), (1, 300), (1, 100), (1, 200))) if ctts is None else ctts
    )
    return box(
        b"moov",
        clock(b"mvhd", 3000, 400)
        + track(1, b"vide", 3000, duration, edit, 400, tables)
        + track(2, b"soun", 48000, 2048, 64, 120),
    ) + box(b"mdat", b"unused")


def parse(payload: bytes) -> video.Composition:
    container = aac.parse_container_headers(io.BytesIO(payload), size=len(payload))
    return video.parse_composition(io.BytesIO(payload), size=len(payload), container=container)


def reference(path: Path, payload: bytes) -> aac.Reference:
    path.write_bytes(payload)
    return aac.Reference(path, hashlib.sha256(payload).hexdigest())


def document(path: Path, data: object) -> aac.Reference:
    return reference(path, json.dumps(data).encode())


@pytest.fixture
def native_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> aac.ObservationRequest:
    monkeypatch.setattr(aac, "ROOT", tmp_path)
    root = tmp_path / "edits/work"
    root.mkdir(parents=True)
    candidate = reference(root / "final.mp4", movie())
    identity = aac.FileIdentity.from_stat(candidate.path.stat())
    tools = [reference(root / name, name.encode()) for name in ("ffmpeg", "ffprobe")]
    for tool in tools:
        tool.path.chmod(0o700)
    audio = {
        "index": 1,
        "codec_type": "audio",
        "codec_name": "aac",
        "profile": "LC",
        "sample_rate": "48000",
        "channels": 2,
        "time_base": "1/48000",
        "start_pts": 0,
        "duration_ts": 1920,
    }
    streams = {"streams": [{"index": 0, "codec_type": "video", "time_base": "1/3000"}, audio]}
    frames = [
        {"media_type": "video", "pts": pts, "duration": 100} for pts in (0, 100, 200, 300)
    ] + [
        {"media_type": "audio", "pts": 0, "nb_samples": 960},
        {"media_type": "audio", "pts": 960, "nb_samples": 960},
    ]
    packets = [
        {"stream_index": 0, "dts": dts, "pts": pts, "duration": 100}
        for dts, pts in ((-200, 0), (-100, 200), (0, 100), (100, 300))
    ] + [
        {
            "stream_index": 1,
            "pts": -64,
            "dts": -64,
            "duration": 1024,
            "side_data_list": [{"side_data_type": "Skip Samples", "skip_samples": 64}],
        },
        {
            "stream_index": 1,
            "pts": 960,
            "dts": 960,
            "duration": 1024,
            "side_data_list": [{"side_data_type": "Skip Samples", "discard_padding": 64}],
        },
    ]
    binding = {
        **candidate.json(),
        "device": identity.device,
        "inode": identity.inode,
        "bytes": identity.size,
        "mtimeNs": identity.modified_ns,
        "ctimeNs": identity.changed_ns,
    }
    receipt = document(
        root / "receipt.json",
        {
            "version": 1,
            "fullDecode": True,
            "technicalAccepted": True,
            "inputs": [binding],
            "inputsAfter": [binding],
            "tools": {"ffmpeg": str(tools[0].path), "ffprobe": str(tools[1].path)},
            "streams": streams,
            "audioDecodedClock": {"count": 2, "sampleCount": 1920},
            "audioPacketClock": {"count": 2},
        },
    )
    checkpoint = document(
        root / "checkpoint.json",
        {"status": "complete", "outputs": {str(receipt.path): receipt.sha256}},
    )
    provisional = aac.ObservationRequest(
        aac.Reference(root / "native_request.json", "a" * 64),
        candidate,
        identity,
        receipt,
        checkpoint,
        (tools[0], tools[1]),
        (),
        None,
        None,
        None,
    )
    commands = aac._commands(provisional)
    raw = {
        "decode": b"",
        "streams": json.dumps(streams).encode(),
        "frames": json.dumps({"frames": frames}).encode(),
        "packets": json.dumps({"packets": packets}).encode(),
    }
    processes = {
        role: {
            "process": document(
                root / (role + ".process.json"), {"command": commands[role], "returncode": 0}
            ).json(),
            "stdout": reference(root / (role + ".stdout"), payload).json(),
            "stderr": reference(root / (role + ".stderr"), b"").json(),
        }
        for role, payload in raw.items()
    }
    pinned = document(
        root / "native_request.json",
        {
            "version": 1,
            "kind": "aac-presentation-observation-request",
            "candidate": candidate.json(),
            "expectedIdentity": identity.json(),
            "decodeReceipt": receipt.json(),
            "completedCheckpoint": checkpoint.json(),
            "tools": {"ffmpeg": tools[0].json(), "ffprobe": tools[1].json()},
            "processes": processes,
            "source": None,
            "contentComparison": None,
        },
    )
    return aac.load_observation_request(pinned)


def test_reordered_table_range_preserves_decode_order() -> None:
    composition = parse(movie())
    assert [row.decode for row in composition.samples] == [0, 100, 200, 300]
    assert [row.presentation for row in composition.samples] == [200, 400, 300, 500]
    assert composition.range_json()["presentationEnd"] == 600


def test_signed_v1_negative_offset() -> None:
    assert parse(movie(ctts=table(b"ctts", ((4, -100),), 1))).samples[0].presentation == -100


def test_no_ctts_means_decode_equals_presentation() -> None:
    assert all(s.decode == s.presentation for s in parse(movie(ctts=b"")).samples)


@pytest.mark.parametrize(
    "stts,ctts,duration",
    [
        (table(b"stts", ((0, 100),)), b"", 400),
        (table(b"stts", ((4, 0),)), b"", 400),
        (table(b"stts", ((4, 100),), 1), b"", 400),
        (table(b"stts", ((4, 100),)), table(b"ctts", ((3, 200),)), 400),
        (table(b"stts", ((4, 100),)), table(b"ctts", ((4, 200),), 2), 400),
        (table(b"stts", ((4, 100),)), table(b"ctts", ((1, 200), (3, 100))), 400),
        (table(b"stts", ((4, 100),)), b"", 401),
        (b"", b"", 400),
        (table(b"stts", ((4, 100),)) * 2, b"", 400),
        (table(b"stts", ((4, 100),)), table(b"ctts", ((4, 200),)) * 2, 400),
    ],
)
def test_malformed_tables_fail(stts: bytes, ctts: bytes, duration: int) -> None:
    with pytest.raises(ReplayError):
        parse(movie(stts=stts, ctts=ctts, duration=duration))


def test_observe_and_original_pcm_failure(native_request: aac.ObservationRequest) -> None:
    result = video.observe(native_request)
    assert record(result["corroboration"])["decodedFrameCount"] == 4
    assert result["newDecode"] is False
    derived = aac._derived_pair(
        native_request, aac._output_paths(native_request.reference.path.parent / "dummy.json")
    )
    timing = aac.parse_container_headers(io.BytesIO(movie()), size=len(movie()))
    plain = aac.assess(timing, aac.Inputs(native_request.candidate, *derived))
    payload = aac._json_bytes(result)
    evidence = aac.Evidence(
        aac.Reference(
            native_request.reference.path.parent / "composition.json",
            hashlib.sha256(payload).hexdigest(),
        ),
        payload,
    )
    revised = aac.assess(
        timing, aac.Inputs(native_request.candidate, *derived, video_composition=evidence)
    )
    assert plain["trackEditMappingAccepted"] is False
    assert revised["trackEditMappingAccepted"] is True
    assert revised["presentationSyncAccepted"] is False


@pytest.mark.parametrize(
    "key,value",
    [
        ("presentationEnd", 199),
        ("trackID", 9),
        ("timescale", 44100),
        ("decodeDuration", 401),
        ("tableSHA256", "bad"),
    ],
)
def test_changed_composition_fact_fails(
    native_request: aac.ObservationRequest, key: str, value: object
) -> None:
    result = video.observe(native_request)
    row = record(result["composition"])
    row[key] = value
    with pytest.raises(ReplayError):
        video.validate(result, candidate=native_request.candidate, track=parse(movie()).track)


@pytest.mark.parametrize(
    "role,key,value",
    [
        ("packets", "dts", 0),
        ("packets", "pts", 200),
        ("frames", "pts", 200),
        ("frames", "duration", True),
    ],
)
def test_raw_mapping_mutation_rejected(
    native_request: aac.ObservationRequest, role: str, key: str, value: object
) -> None:
    process = next(p for p in native_request.processes if p.role == role)
    data = record(json.loads(process.stdout.path.read_bytes()))
    rows = data[role]
    assert isinstance(rows, list)
    record(rows[0])[key] = value
    changed = document(process.stdout.path, data)
    altered = replace(
        native_request,
        processes=tuple(
            replace(p, stdout=changed) if p.role == role else p for p in native_request.processes
        ),
    )
    with pytest.raises(ReplayError):
        video.corroborate(parse(movie()), altered)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"frames":[{},]}',
        b'{"frames":[]}garbage',
        b'{"frames":[],"frames":[]}',
        b'{"frames":[{"pts":NaN}]}',
        b'{"unknown":[]}',
        b'{"frames":[{}],}',
        b'{"frames":[',
        b'{"frames":[{"pts":' + b"1" * 70000 + b"}]}",
    ],
)
def test_streaming_json_rejects_invalid(
    native_request: aac.ObservationRequest, payload: bytes
) -> None:
    ref = reference(native_request.reference.path.parent / "malformed.json", payload)
    with pytest.raises((ReplayError, ValueError)):
        list(video.raw_rows(ref, "frames"))


def test_incremental_chunks_and_terminal_pin(
    native_request: aac.ObservationRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(video, "CHUNK_BYTES", 7)
    ref = document(
        native_request.reference.path.parent / "chunk.json", {"frames": [{"pts": 102456}]}
    )
    assert list(video.raw_rows(ref, "frames")) == [{"pts": 102456}]
    changed = replace(ref, sha256="f" * 64)
    with pytest.raises(ReplayError):
        list(video.raw_rows(changed, "frames"))


def test_cli_result_and_readonly_verifier(native_request: aac.ObservationRequest) -> None:
    output = native_request.reference.path.parent / "observed.json"
    args = [
        str(native_request.reference.path),
        str(output),
        "--sha256",
        native_request.reference.sha256,
    ]
    assert video.main(["observe", *args]) == 0
    before = output.stat()
    assert video.main(["verify", *args]) == 0
    assert output.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(SystemExit):
        video.main(["observe", *args])
    output.write_bytes(b"{}")
    with pytest.raises(SystemExit):
        video.main(["verify", *args])


def test_cli_reuses_verified_prior_aac(native_request: aac.ObservationRequest) -> None:
    prior = native_request.reference.path.parent / "prior.json"
    aac.collect_presentation(native_request.reference, prior)
    result = video.observe(
        native_request,
        prior_result=aac.Reference(prior, hashlib.sha256(prior.read_bytes()).hexdigest()),
    )
    assert record(result["aacAssessment"])["trackEditMappingAccepted"] is True
    assert record(result["aacAssessment"])["originalPCMExtentAccepted"] is False


def test_candidate_identity_or_request_mutation_rejected(
    native_request: aac.ObservationRequest,
) -> None:
    with pytest.raises(ReplayError):
        video.observe(replace(native_request, identity=replace(native_request.identity, size=1)))
    os.utime(native_request.candidate.path, ns=(1, 1))
    with pytest.raises(ReplayError):
        video.observe(native_request)


def test_streamed_raw_mutation_and_link_fail_closed(native_request: aac.ObservationRequest) -> None:
    ref = document(
        native_request.reference.path.parent / "rows.json", {"frames": [{"pts": 0}, {"pts": 1}]}
    )
    rows = video.raw_rows(ref, "frames")
    assert next(rows) == {"pts": 0}
    ref.path.write_bytes(b"{}")
    with pytest.raises(ReplayError):
        list(rows)
    link = ref.path.with_name("rows-link.json")
    link.symlink_to(ref.path)
    with pytest.raises(ReplayError):
        list(video.raw_rows(replace(ref, path=link), "frames"))


def test_raw_count_byte_and_row_bounds(
    native_request: aac.ObservationRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    ref = document(
        native_request.reference.path.parent / "bounded.json", {"frames": [{"pts": 0}] * 3}
    )
    monkeypatch.setattr(video, "MAX_SAMPLES", 1)
    with pytest.raises(ReplayError, match="count"):
        list(video.raw_rows(ref, "frames"))
    monkeypatch.setattr(aac, "MAX_PROBE_BYTES", 3)
    with pytest.raises(ReplayError, match="budget"):
        list(video.raw_rows(ref, "frames"))


@pytest.mark.parametrize("entries", [((0, 100),), ((1000001, 100),)])
def test_table_run_count_budget(entries: tuple[tuple[int, int], ...]) -> None:
    with pytest.raises(ReplayError):
        parse(movie(stts=table(b"stts", entries)))


def test_flags_lengths_and_missing_video_are_not_accepted() -> None:
    bad = box(b"stts", b"\x00\x00\x00\x01" + struct.pack(">I", 1) + struct.pack(">II", 4, 100))
    with pytest.raises(ReplayError):
        parse(movie(stts=bad))
    with pytest.raises(ReplayError):
        parse(movie(stts=box(b"stts", bytes(4) + struct.pack(">I", 2))))
    container = aac.parse_container_headers(io.BytesIO(movie()), size=len(movie()))
    audio = container.tracks[1]
    alone = aac.Container(
        container.timescale,
        container.duration,
        (audio,),
        container.timing_payload_sha256,
        aac._facts_hash(
            container.timescale, container.duration, (audio,), container.timing_payload_sha256
        ),
    )
    with pytest.raises(ReplayError):
        video.parse_composition(io.BytesIO(movie()), size=len(movie()), container=alone)


def test_actual_candidate_mutation_during_observation(
    native_request: aac.ObservationRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = video.corroborate

    def mutate(
        composition: video.Composition, request: aac.ObservationRequest
    ) -> dict[str, object]:
        result = original(composition, request)
        os.utime(request.candidate.path, ns=(1, 1))
        return result

    monkeypatch.setattr(video, "corroborate", mutate)
    with pytest.raises(ReplayError):
        video.observe(native_request)


def test_cli_prior_pair_and_changed_prior_domain(native_request: aac.ObservationRequest) -> None:
    args = [
        str(native_request.reference.path),
        str(native_request.reference.path.parent / "out.json"),
        "--sha256",
        native_request.reference.sha256,
    ]
    with pytest.raises(SystemExit):
        video.main(["observe", *args, "--aac-result", "absent.json"])
    prior = document(native_request.reference.path.parent / "bad-prior.json", {"request": {}})
    with pytest.raises(ReplayError):
        video.observe(native_request, prior_result=prior)
