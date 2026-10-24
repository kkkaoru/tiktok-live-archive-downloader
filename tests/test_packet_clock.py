"""The packet clock is written only when headers prove one frame per packet; else decoded."""

import json
import subprocess
from pathlib import Path

import pytest

from replay import delivery_media, packet_clock
from replay.core import ReplayError

HEVC_IDR = b"""\
[trace_headers @ 0x1] 0          nal_unit_type                                          100000 = 32
[trace_headers @ 0x1] 2001       field_seq_flag                                              0 = 0
[trace_headers @ 0x1] 3001       output_flag_present_flag                                    0 = 0
[trace_headers @ 0x1] 4001       nal_unit_type                                          010011 = 19
"""


def encode(path: Path, *options: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=2",
         "-pix_fmt", "yuv420p", *options, str(path)],
        check=True,
    )  # fmt: skip
    return path


def rows(path: Path) -> list[tuple[int, int]]:
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    return [(frame["pts"], frame["duration"]) for frame in frames]


def test_discarded_packets_are_dropped_and_the_rest_sorted_into_display_order() -> None:
    probe = {
        "streams": [{"time_base": "1/90000"}],
        "packets": [
            {"pts": 0, "duration": 3000, "flags": "K__"},
            {"pts": 6000, "duration": 3000, "flags": "___"},
            {"pts": 3000, "duration": 3000, "flags": "___"},
            {"pts": 9000, "duration": 3000, "flags": "_D_"},
        ],
    }
    assert packet_clock.packet_clock(probe) == {
        "frames": [
            {"pts": 0, "duration": 3000},
            {"pts": 3000, "duration": 3000},
            {"pts": 6000, "duration": 3000},
        ],
        "streams": [{"time_base": "1/90000"}],
        "clockProducer": {"method": "packet-v1", "discardedPackets": 1},
    }


def test_packets_without_a_frame_clock_give_none() -> None:
    base = [{"time_base": "1/90000"}]
    assert packet_clock.packet_clock({"streams": base, "packets": [{"pts": 0}]}) is None
    assert (
        packet_clock.packet_clock(
            {"streams": base, "packets": [{"pts": 0, "duration": 0, "flags": "K__"}]}
        )
        is None
    )
    assert (
        packet_clock.packet_clock(
            {"streams": base, "packets": [{"pts": 0, "duration": 1}, {"pts": 0, "duration": 1}]}
        )
        is None
    )
    assert packet_clock.packet_clock({"streams": base * 2, "packets": []}) is None
    assert packet_clock.packet_clock({"streams": base, "packets": []}) is None


def test_traced_headers_prove_or_refuse_one_frame_per_packet() -> None:
    assert packet_clock.header_refusal("hevc", HEVC_IDR, 0) is None
    assert packet_clock.header_refusal("hevc", HEVC_IDR, 12) == (
        "12 bytes of in-band parameter sets or BLA_W_LP pictures"
    )
    assert packet_clock.header_refusal("hevc", HEVC_IDR.replace(b"= 19", b"= 21"), 0) == (
        "the first picture is not IDR or a BLA without RASL"
    )
    assert packet_clock.header_refusal("hevc", HEVC_IDR.replace(b"0 = 0\n", b"1 = 1\n", 1), 0) == (
        "HEVC field sequence"
    )
    marked = HEVC_IDR.replace(
        b"0 = 0\n[trace_headers @ 0x1] 4001", b"1 = 1\n[trace_headers @ 0x1] 4001"
    )
    assert packet_clock.header_refusal("hevc", marked, 0) == (
        "HEVC pictures may be marked not for output"
    )
    fields = b"""\
[trace_headers @ 0x1] 9          frame_mbs_only_flag                                         0 = 0
[trace_headers @ 0x1] 20         nal_unit_type                                           00101 = 5
"""
    assert packet_clock.header_refusal("h264", fields, 0) == "H.264 field coding"
    assert packet_clock.header_refusal("h264", fields.replace(b"0 = 0", b"1 = 1"), 0) is None


def test_a_proven_source_gets_the_decoded_clock_from_its_packets(tmp_path: Path) -> None:
    source = encode(tmp_path / "clip.mp4", "-c:v", "libx264", "-bf", "2")
    decoded, packets = tmp_path / "decoded.json", tmp_path / "packets.json"
    assert delivery_media.main(["clock", str(source), str(decoded)]) == 0
    assert packet_clock.produce(source, packets, ffprobe="ffprobe", ffmpeg="ffmpeg") == "packet-v1"
    assert rows(packets) == rows(decoded)
    assert len(rows(packets)) == 20
    assert packet_clock.is_packet(packets) is True
    assert packet_clock.is_packet(decoded) is False


def test_packets_an_edit_list_discards_are_left_out_of_the_clock(tmp_path: Path) -> None:
    source = encode(tmp_path / "clip.mp4", "-c:v", "libx264", "-g", "30")
    trimmed = tmp_path / "trimmed.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", "0.45", "-i", str(source), "-c", "copy", str(trimmed)],
        check=True,
    )  # fmt: skip
    decoded, packets = tmp_path / "decoded.json", tmp_path / "packets.json"
    assert delivery_media.main(["clock", str(trimmed), str(decoded)]) == 0
    assert packet_clock.produce(trimmed, packets, ffprobe="ffprobe", ffmpeg="ffmpeg") == "packet-v1"
    assert rows(packets) == rows(decoded)
    assert json.loads(packets.read_text(encoding="utf-8"))["clockProducer"] == {
        "method": "packet-v1",
        "discardedPackets": 5,
    }


def test_field_coded_and_non_mp4_sources_get_the_decoded_clock(tmp_path: Path) -> None:
    interlaced = encode(tmp_path / "fields.mp4", "-c:v", "libx264", "-flags", "+ildct+ilme")
    assert packet_clock.produce(
        interlaced, tmp_path / "a.json", ffprobe="ffprobe", ffmpeg="ffmpeg"
    ) == ("decoded (H.264 field coding)")
    matroska = encode(tmp_path / "clip.mkv", "-c:v", "libx264")
    assert packet_clock.produce(
        matroska, tmp_path / "b.json", ffprobe="ffprobe", ffmpeg="ffmpeg"
    ) == ("decoded (not an MP4/MOV demuxer)")
    other = encode(tmp_path / "clip.mov", "-c:v", "mpeg4")
    assert (
        packet_clock.main(
            [str(other), str(tmp_path / "c.json"), "--ffprobe", "ffprobe", "--ffmpeg", "ffmpeg"]
        )
        == 0
    )
    assert packet_clock.is_packet(tmp_path / "c.json") is False


def test_a_consumer_refusal_makes_the_next_clock_a_decoded_one(tmp_path: Path) -> None:
    source = encode(tmp_path / "clip.mp4", "-c:v", "libx264")
    clock = tmp_path / "video-clock.json"
    error = packet_clock.refuse(clock, consumer="yolo-track", reason="PTS differ")
    assert str(error) == (
        "DECODED_CLOCK_MISMATCH yolo-track: PTS differ; the packet clock is refused, "
        "force clocks to decode it"
    )
    packet_clock.refuse(clock, consumer="cleanup chunk-00001", reason="later")
    assert packet_clock.refused(clock) == "PTS differ"
    assert packet_clock.produce(source, clock, ffprobe="ffprobe", ffmpeg="ffmpeg") == (
        "decoded (refused: PTS differ)"
    )
    assert packet_clock.is_packet(clock) is False


def test_a_failed_clock_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert packet_clock.main([str(tmp_path / "missing.mp4"), str(tmp_path / "out.json")]) == 1
    assert capsys.readouterr().err.startswith("PACKET_CLOCK_ERROR ")
    (tmp_path / "packet-clock-refused.json").write_text('{"reason": "x"}', encoding="utf-8")
    with pytest.raises(ReplayError, match="decoded clock failed"):
        packet_clock.produce(tmp_path / "missing.mp4", tmp_path / "out.json", ffprobe="ffprobe",
                             ffmpeg="ffmpeg")  # fmt: skip
