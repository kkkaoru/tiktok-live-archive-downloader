"""Synthetic measured priming proofs and a real tiny AAC copy-mux fixture."""

import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_compressed_part_delivery import Example, pin, read, write
from test_compressed_part_delivery import example as example

from replay import aac_presentation_normalization as normal
from replay import compressed_part_delivery as delivery
from replay import delivery_verification as technical
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest
from replay.whole_technical_batch import Reference


@dataclass(frozen=True)
class Fixture:
    config: Path
    proof: Path
    example: Example

    def change(self, key: str, value: object) -> None:
        raw = read(self.config)
        raw[key] = value
        write(self.config, raw)

    def probe(self, role: str, change: dict[str, object]) -> None:
        raw = read(self.config)
        path = Path(str(record(raw[role])["path"]))
        probe = read(path)
        probe.update(change)
        raw[role] = write(path, probe)
        write(self.config, raw)


@pytest.fixture
def measured(example: Example) -> Fixture:
    stream = {
        "codec_name": "aac",
        "codec_type": "audio",
        "profile": "LC",
        "channels": 2,
        "sample_rate": "48000",
        "time_base": "1/48000",
        "initial_padding": 64,
    }
    decoded = write(
        example.root / "decoded-normalization.json",
        {
            "streams": [stream],
            "frames": [
                {"pts": 64, "nb_samples": 24000},
                {"pts": 24064, "nb_samples": 24000},
            ],
        },
    )
    packets = write(
        example.root / "packets-normalization.json",
        {
            "streams": [stream],
            "packets": [
                {
                    "pts": 0,
                    "dts": 0,
                    "duration": 24064,
                    "data_hash": "SHA256:" + "a" * 64,
                    "side_data_list": [
                        {"side_data_type": "Skip Samples", "skip_samples": 64, "discard_padding": 0}
                    ],
                },
                {
                    "pts": 24064,
                    "dts": 24064,
                    "duration": 24064,
                    "data_hash": "SHA256:" + "b" * 64,
                    "side_data_list": [
                        {"side_data_type": "Skip Samples", "skip_samples": 0, "discard_padding": 64}
                    ],
                },
            ],
        },
    )
    tag = write(
        example.root / "tag-normalization.json",
        {
            "format": {"tags": {"iTunSMPB": "00000000 00000040 00000040 000000000000BB80"}},
        },
    )
    config = example.root / "normalization-config.json"
    write(
        config,
        {
            "version": 1,
            "audio": pin(example.root / "whole-audio.m4a"),
            "video": pin(example.root / "master-1.mov"),
            "packetProbe": packets,
            "decodedProbe": decoded,
            "tagProbe": tag,
            "ffmpeg": pin(example.root / "executable"),
            "output": str(example.root / "normalized.mp4"),
            "timestampField": "pts",
            "videoTag": "avc1",
            "videoTrackTimescale": 3000,
        },
    )
    return Fixture(config, example.root / "normalization-proof.json", example)


def test_measured_derivation_and_fresh_readonly_reentry(measured: Fixture) -> None:
    result = normal.prepare(measured.config, measured.proof)
    assert result.normalized.pts == (0, 24000)
    assert result.normalized.samples == (24000, 24000)
    assert result.document["preRollTicks"] == 64
    assert result.document["primingSamples"] == 64
    assert result.document["deliveryApproved"] is False
    assert "setts=pts=PTS-64:dts=DTS-64" in str(result.document["producerCommand"])
    assert (
        normal.verify(Reference(measured.proof, file_digest(measured.proof))).normalized
        == result.normalized
    )


def test_best_effort_format_is_explicit_and_integral(measured: Fixture) -> None:
    measured.change("timestampField", "best_effort_timestamp")
    measured.probe(
        "decodedProbe",
        {
            "frames": [
                {"best_effort_timestamp": 64, "nb_samples": 24000},
                {"best_effort_timestamp": 24064, "nb_samples": 24000},
            ]
        },
    )
    assert normal.prepare(measured.config, measured.proof).normalized.pts == (0, 24000)


def test_decoded_profile_omission_keeps_observed_packet_lc_required(measured: Fixture) -> None:
    reference = record(read(measured.config)["decodedProbe"])
    streams = read(Path(str(reference["path"])))["streams"]
    assert isinstance(streams, list)
    record(streams[0]).pop("profile")
    measured.probe("decodedProbe", {"streams": streams})
    assert normal.prepare(measured.config, measured.proof).normalized.pts == (0, 24000)


@pytest.mark.parametrize("role,profile", [("packetProbe", None), ("decodedProbe", "HE")])
def test_missing_packet_profile_or_conflicting_decoded_profile_is_rejected(
    measured: Fixture, role: str, profile: object
) -> None:
    reference = record(read(measured.config)[role])
    streams = read(Path(str(reference["path"])))["streams"]
    assert isinstance(streams, list)
    if profile is None:
        record(streams[0]).pop("profile")
    else:
        record(streams[0])["profile"] = profile
    measured.probe(role, {"streams": streams})
    with pytest.raises(ReplayError, match="stream/rate/timebase/channel"):
        normal.prepare(measured.config, measured.proof)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("timestampField", []),
        ("timestampField", "invented"),
        ("videoTag", "unknown"),
        ("videoTrackTimescale", True),
        ("videoTrackTimescale", 0),
        ("output", "relative.mp4"),
    ],
)
def test_invalid_config_is_refused(measured: Fixture, field: str, value: object) -> None:
    measured.change(field, value)
    with pytest.raises((ReplayError, ValueError)):
        normal.prepare(measured.config, measured.proof)
    assert not measured.proof.exists()


@pytest.mark.parametrize(
    "frames",
    [
        [],
        [{"pts": True, "nb_samples": 24000}],
        [{"nb_samples": 24000}],
        [{"pts": 64, "best_effort_timestamp": 65, "nb_samples": 24000}],
        [{"pts": True, "best_effort_timestamp": 1, "nb_samples": 24000}],
        [{"pts": 64, "nb_samples": 0}],
        [{"pts": 64, "nb_samples": 24000}, {"pts": 24065, "nb_samples": 24000}],
    ],
)
def test_missing_divergent_or_gapped_frames_are_refused(measured: Fixture, frames: object) -> None:
    measured.probe("decodedProbe", {"frames": frames})
    with pytest.raises((ReplayError, ValueError)):
        normal.prepare(measured.config, measured.proof)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "bad",
        "00000000 00000000 00000040 000000000000BB80",
        "00000000 00000040 0000003F 000000000000BB80",
        "00000000 00000040 00000040 000000000000BB81",
    ],
)
def test_tag_missing_or_disagreement_is_refused(measured: Fixture, value: object) -> None:
    measured.probe("tagProbe", {"format": {"tags": {"iTunSMPB": value}}})
    with pytest.raises(ReplayError):
        normal.prepare(measured.config, measured.proof)


@pytest.mark.parametrize(
    "role", ["audio", "video", "packetProbe", "decodedProbe", "tagProbe", "ffmpeg"]
)
def test_mutated_input_cannot_reuse_proof(measured: Fixture, role: str) -> None:
    normal.prepare(measured.config, measured.proof)
    ref = record(read(measured.config)[role])
    path = Path(str(ref["path"]))
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ReplayError):
        normal.verify(Reference(measured.proof, file_digest(measured.proof)))


def test_normalized_packet_preroll_cannot_apply_twice(measured: Fixture) -> None:
    packets = read(Path(str(record(read(measured.config)["packetProbe"])["path"])))
    rows = packets["packets"]
    assert isinstance(rows, list)
    record(rows[0])["pts"] = -64
    record(rows[0])["dts"] = -64
    measured.probe("packetProbe", packets)
    with pytest.raises(ReplayError, match="already normalized"):
        normal.prepare(measured.config, measured.proof)


def test_internal_skip_and_invalid_payload_are_refused(measured: Fixture) -> None:
    packets = read(Path(str(record(read(measured.config)["packetProbe"])["path"])))
    rows = packets["packets"]
    assert isinstance(rows, list)
    record(rows[-1])["data_hash"] = "unobserved"
    measured.probe("packetProbe", packets)
    with pytest.raises(ReplayError):
        normal.prepare(measured.config, measured.proof)


def test_changed_proof_and_existing_output_are_refused(measured: Fixture) -> None:
    normal.prepare(measured.config, measured.proof)
    proof = read(measured.proof)
    proof["preRollTicks"] = 65
    write(measured.proof, proof)
    with pytest.raises(ReplayError, match="proof differs"):
        normal.verify(Reference(measured.proof, file_digest(measured.proof)))
    (measured.example.root / "normalized.mp4").write_bytes(b"occupied")
    with pytest.raises(ReplayError, match="already exists"):
        normal.prepare(measured.config, measured.example.root / "other-proof.json")


def test_cli_prepare_verify_and_failure(
    measured: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    assert normal.main(["prepare", str(measured.config), str(measured.proof)]) == 0
    assert (
        normal.main(["verify", str(measured.proof), "--sha256", file_digest(measured.proof)]) == 0
    )
    assert normal.main(["verify", str(measured.proof), "--sha256", "a" * 64]) == 1
    assert "normalization failed" in capsys.readouterr().out


def atom(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I4s", len(data) + 8, kind) + data


def test_real_apple_style_aac_copy_mux_excludes_observed_priming(measured: Fixture) -> None:
    """Custom tag insertion is synthetic fixture creation, never a media producer."""
    ffmpeg_name, ffprobe_name = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg_name is not None and ffprobe_name is not None
    ffmpeg, ffprobe = str(Path(ffmpeg_name).resolve()), str(Path(ffprobe_name).resolve())
    root = measured.example.root
    original, zero, video = root / "encoded.aac", root / "apple-source.m4a", root / "video.mp4"
    technical._run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=1",
            "-ac",
            "2",
            "-c:a",
            "aac",
            str(original),
        ],
        directory=root,
        name="synthetic-aac",
        timeout=30,
    )
    technical._run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-copyts",
            "-i",
            str(original),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "disabled",
            "-use_editlist",
            "0",
            str(zero),
        ],
        directory=root,
        name="synthetic-zero-preroll",
        timeout=30,
    )
    blob = zero.read_bytes()
    moov = blob.rfind(b"moov") - 4
    assert moov >= 0 and int.from_bytes(blob[moov : moov + 4], "big") == len(blob) - moov
    tag = b"00000000 00000400 00000080 000000000000BB80"
    custom = atom(
        b"----",
        atom(b"mean", b"\0" * 4 + b"com.apple.iTunes")
        + atom(b"name", b"\0" * 4 + b"iTunSMPB")
        + atom(b"data", b"\0\0\0\1" + b"\0" * 4 + tag),
    )
    udta = atom(
        b"udta",
        atom(
            b"meta",
            b"\0" * 4
            + atom(b"hdlr", b"\0" * 8 + b"mdir" + b"\0" * 12 + b"\0")
            + atom(b"ilst", custom),
        ),
    )
    zero.write_bytes(blob[:moov] + atom(b"moov", blob[moov + 8 :] + udta))
    technical._run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x96:rate=30:duration=1",
            "-c:v",
            "libx264",
            "-threads",
            "1",
            str(video),
        ],
        directory=root,
        name="synthetic-video",
        timeout=30,
    )
    packets = technical._run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_packets",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            str(zero),
        ],
        directory=root,
        name="synthetic-packets",
        timeout=30,
    )
    frames = technical._run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_frames",
            "-of",
            "json",
            str(zero),
        ],
        directory=root,
        name="synthetic-frames",
        timeout=30,
    )
    config = read(measured.config)
    config.update(
        audio=pin(zero),
        video=pin(video),
        ffmpeg=pin(Path(ffmpeg)),
        packetProbe=pin(packets),
        decodedProbe=pin(frames),
    )
    config["tagProbe"] = write(
        root / "synthetic-tag.json",
        {
            "format": {"tags": {"iTunSMPB": tag.decode()}},
        },
    )
    write(measured.config, config)
    value = normal.prepare(measured.config, measured.proof)
    command = value.document["producerCommand"]
    assert isinstance(command, list) and all(isinstance(word, str) for word in command)
    technical._run(
        [str(word) for word in command], directory=root, name="synthetic-mux", timeout=30
    )
    output = technical._run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_frames",
            "-show_packets",
            "-show_data_hash",
            "sha256",
            "-of",
            "json",
            str(config["output"]),
        ],
        directory=root,
        name="synthetic-final-observation",
        timeout=30,
    )
    raw = read(output)
    rows = raw["packets_and_frames"]
    assert isinstance(rows, list)
    audio_frames = [
        record(row)
        for row in rows
        if record(row).get("type") == "frame" and record(row).get("media_type") == "audio"
    ]
    assert audio_frames[0]["pts"] == 0
    assert sum(int(str(row["nb_samples"])) for row in audio_frames) == 48000
    audio_packets = [
        record(row)
        for row in rows
        if record(row).get("type") == "packet" and record(row).get("stream_index") == 1
    ]
    source_packets = record(read(packets))["packets"]
    assert isinstance(source_packets, list)
    assert [row["data_hash"] for row in audio_packets] == [
        record(row)["data_hash"] for row in source_packets
    ]
    assert delivery._skip(audio_packets[0], "skip_samples") == 1024
