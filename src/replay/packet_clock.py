"""The source video clock from container packets when headers prove one packet is one frame.

A full decode (``delivery_media clock``) costs 5-1,160 s per recording; the packets' presentation
timestamps and durations, sorted into display order without the packets the edit list discards,
equalled that decoded clock on all 22 recordings measured (2026-10-24) in 0.1-18 s. They are
equal exactly when every packet yields one output frame, so the packet clock is written only
when the source proves that:

- an MP4/MOV demuxer (edit lists are applied and their discards flagged);
- progressive frames: H.264 ``frame_mbs_only_flag`` 1, HEVC ``field_seq_flag`` 0;
- every coded picture is output: HEVC ``output_flag_present_flag`` 0;
- the first picture is an IDR, or an HEVC BLA without skipped leading pictures (no CRA: its
  RASL pictures would be skipped);
- no parameter set or HEVC ``BLA_W_LP`` inside the packets, so the traced sample-entry
  parameter sets hold for the whole stream;
- integer, unique PTS and positive durations on every kept packet.

Anything else, or a refusal recorded by a consumer that decoded frames against this clock, gets
today's decoded clock unchanged. Consumers (``yolo_track``, the cleanup chunks) compare the PTS
they actually decode with the clock and call ``refuse`` on a difference; recovery is ``--force
clocks``, which re-derives the clock by decoding.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from replay import delivery_media
from replay.core import ReplayError
from replay.recordings import record

METHOD = "packet-v1"
REFUSAL = "packet-clock-refused.json"
MISMATCH = "DECODED_CLOCK_MISMATCH"
CONTAINER = "mov,mp4"
# Slice NAL types, and those a first picture may have: HEVC BLA_W_RADL, BLA_N_LP, IDR_W_RADL,
# IDR_N_LP; H.264 the IDR slice.
SLICES = {"hevc": range(32), "h264": range(1, 6)}
FIRST_PICTURES = {"hevc": frozenset({17, 18, 19, 20}), "h264": frozenset({5})}
# In-band units that would void the sample-entry proofs (HEVC VPS/SPS/PPS and BLA_W_LP).
IN_BAND = {"hevc": "16|32-34", "h264": "7-8"}
TRACED = re.compile(rb"^\[trace_headers[^\]]*\]\s+\d+\s+(\w+)\s+[01]+ = (-?\d+)\s*$", re.MULTILINE)
PROBE_TIMEOUT_SECONDS = 1800


def packet_clock(probe: object) -> dict[str, object] | None:
    """The ffprobe-shaped clock from ``-show_entries stream=...:packet=pts,duration,flags``.

    None when a kept packet lacks an integer PTS or a positive integer duration, or PTS repeat.
    """
    data = record(probe)
    streams, packets = data.get("streams"), data.get("packets")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(packets, list):
        return None
    base = record(streams[0]).get("time_base")
    rows: list[tuple[int, int]] = []
    for packet in map(record, packets):
        if "D" in str(packet.get("flags", "")):
            continue
        pts, duration = packet.get("pts"), packet.get("duration")
        if type(pts) is not int or type(duration) is not int or duration <= 0:
            return None
        rows.append((pts, duration))
    rows.sort()
    if not rows or not isinstance(base, str) or len({pts for pts, _ in rows}) != len(rows):
        return None
    return {
        "frames": [{"pts": pts, "duration": duration} for pts, duration in rows],
        "streams": [{"time_base": base}],
        "clockProducer": {
            "method": METHOD,
            "discardedPackets": len(packets) - len(rows),
        },
    }


def header_refusal(codec: str, traced: bytes, in_band: int) -> str | None:
    """Why the traced first-packet headers do not prove one frame per packet (None: proven)."""
    if in_band:
        return f"{in_band} bytes of in-band parameter sets or BLA_W_LP pictures"
    values: dict[str, set[int]] = {}
    for name, value in TRACED.findall(traced):
        values.setdefault(name.decode(), set()).add(int(value))
    slices = {value for value in values.get("nal_unit_type", set()) if value in SLICES[codec]}
    rules = (
        (
            bool(slices) and slices <= FIRST_PICTURES[codec],
            "the first picture is not IDR or a BLA without RASL",
        ),
        (codec != "h264" or values.get("frame_mbs_only_flag") == {1}, "H.264 field coding"),
        (values.get("field_seq_flag", {0}) == {0}, "HEVC field sequence"),
        (
            codec != "hevc" or values.get("output_flag_present_flag") == {0},
            "HEVC pictures may be marked not for output",
        ),
    )
    return next((reason for holds, reason in rules if not holds), None)


def _packets(source: Path, ffprobe: str) -> subprocess.Popen[bytes]:
    entries = "stream=codec_name,time_base:format=format_name:packet=pts,duration,flags"
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", entries,
               "-of", "json", str(source)]  # fmt: skip
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)


def _in_band(source: Path, ffmpeg: str, codec: str) -> subprocess.Popen[bytes]:
    units = f"filter_units=pass_types={IN_BAND[codec]}"
    command = [ffmpeg, "-v", "error", "-nostdin", "-i", str(source), "-map", "0:v:0", "-c", "copy",
               "-bsf:v", units, "-f", "rawvideo", "pipe:1"]  # fmt: skip
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)


def _traced(source: Path, ffmpeg: str) -> bytes:
    command = [ffmpeg, "-hide_banner", "-nostdin", "-v", "trace", "-i", str(source), "-map",
               "0:v:0", "-c", "copy", "-bsf:v", "trace_headers", "-frames:v", "1", "-f", "null",
               "-"]  # fmt: skip
    traced = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, check=True)
    return traced.stderr


def _codec(source: Path, ffprobe: str) -> str:
    command = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=codec_name", "-of", "csv=p=0", str(source)]  # fmt: skip
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def derive(source: Path, *, ffprobe: str, ffmpeg: str) -> tuple[dict[str, object] | None, str]:
    """The packet clock and ``"proven"``, or None and why the decoded clock is needed."""
    codec = _codec(source, ffprobe)
    if codec not in IN_BAND:
        return None, f"codec {codec} is not H.264 or HEVC"
    # Both read the whole file; run together they share the read.
    with _packets(source, ffprobe) as packets, _in_band(source, ffmpeg, codec) as scan:
        probed, _ = packets.communicate(timeout=PROBE_TIMEOUT_SECONDS)
        units, _ = scan.communicate(timeout=PROBE_TIMEOUT_SECONDS)
    if packets.returncode != 0 or scan.returncode != 0:
        raise ReplayError("The packet probe failed")
    data = record(json.loads(probed))
    if CONTAINER not in str(record(data.get("format")).get("format_name", "")):
        return None, "not an MP4/MOV demuxer"
    refusal = header_refusal(codec, _traced(source, ffmpeg), len(units))
    if refusal is not None:
        return None, refusal
    clock = packet_clock(data)
    return (clock, "proven") if clock is not None else (None, "invalid packet timestamps")


def refused(clock: Path) -> str | None:
    """The reason a consumer recorded against this recording's packet clock, if any."""
    marker = clock.parent / REFUSAL
    if not marker.exists():
        return None
    return str(record(json.loads(marker.read_text(encoding="utf-8"))).get("reason"))


def refuse(clock: Path, *, consumer: str, reason: str) -> ReplayError:
    """Record that a consumer's decoded PTS differ from the packet clock; the error to raise."""
    marker = clock.parent / REFUSAL
    if not marker.exists():
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"version": 1, "consumer": consumer, "reason": reason}, stream)
            stream.write("\n")
    return ReplayError(
        f"{MISMATCH} {consumer}: {reason}; the packet clock is refused, force clocks to decode it"
    )


def is_packet(clock: Path) -> bool:
    producer = record(json.loads(clock.read_text(encoding="utf-8"))).get("clockProducer")
    return isinstance(producer, dict) and producer.get("method") == METHOD


def produce(source: Path, output: Path, *, ffprobe: str, ffmpeg: str) -> str:
    """Write the packet clock when proven, else the decoded clock; return what was written."""
    reason = refused(output)
    if reason is None:
        clock, why = derive(source, ffprobe=ffprobe, ffmpeg=ffmpeg)
    else:
        clock, why = None, f"refused: {reason}"
    if clock is None:
        code = delivery_media.main(["clock", str(source), str(output), "--ffprobe", ffprobe])
        if code != 0:
            raise ReplayError("The decoded clock failed")
        return f"decoded ({why})"
    with output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(clock, stream, indent=1)
        stream.write("\n")
    delivery_media.read_clock(output)
    return METHOD


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)
    try:
        print(produce(args.source, args.output, ffprobe=args.ffprobe, ffmpeg=args.ffmpeg))
    except (ReplayError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"PACKET_CLOCK_ERROR {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
