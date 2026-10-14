"""Audit an existing Compressor-to-delivery remux, without decoding or publishing.

Only ordered compressed payloads and their rational clocks are certified. Native
success is taken from the retained state receipt, not a new Compressor API call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from itertools import chain
from pathlib import Path

from replay import delivery_verification as verifier
from replay.compressor_batch import JOB_ID
from replay.core import ReplayError
from replay.delivery_output_names import recorded_stem_matches
from replay.delivery_render import Config, configured_movie, load_config

STREAM_FIELDS = (
    "index",
    "id",
    "codec_type",
    "codec_name",
    "profile",
    "level",
    "width",
    "height",
    "pix_fmt",
    "sample_aspect_ratio",
    "field_order",
    "color_range",
    "color_space",
    "color_transfer",
    "color_primaries",
    "chroma_location",
    "sample_fmt",
    "sample_rate",
    "channels",
    "channel_layout",
    "bits_per_sample",
    "bits_per_raw_sample",
    "extradata_size",
    "disposition",
    "side_data_list",
)
SHA256 = re.compile(r"SHA256:[0-9a-fA-F]{64}")


def _local(path: Path) -> Path:
    path = verifier._canonical(path)
    if not path.is_relative_to(verifier.REPOSITORY):
        raise ReplayError("Configuration, state and candidate must be repository-local")
    return path


def _completed(
    config: Config, jobs: Path, state: Path, *, historical: bool = False
) -> tuple[Path, str]:
    rows = verifier._rows(verifier._json(jobs), "jobs")
    if historical:
        output = rows[0].get("output")
        if not isinstance(output, str) or not Path(output).is_absolute():
            raise ReplayError("Historical job must bind an absolute recorded output")
        expected = verifier._canonical(Path(output))
        if expected.parent != config.storage / "final" or not recorded_stem_matches(
            expected, config.source, config.recording, historical=True
        ):
            raise ReplayError("Historical output does not belong to this recording")
    else:
        expected = configured_movie(config)
    if len(rows) != 1 or rows[0] != {
        "name": "final",
        "source": str(config.storage / "av-master.mov"),
        "preset": str(config.final_preset),
        "output": str(expected),
    }:
        raise ReplayError("Final job does not match render configuration")
    saved = verifier._json(state)
    job_id = saved.get("jobID")
    if saved.get("status") != "Successful" or not isinstance(job_id, str):
        raise ReplayError("Missing native Successful receipt/jobID")
    if JOB_ID.fullmatch(job_id) is None:
        raise ReplayError("Invalid native jobID")
    if saved.get("retry_count", 0) != 0 or "requested_output" in saved:
        raise ReplayError("Retry output requires explicit reconciliation")
    output = saved.get("output")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise ReplayError("Missing absolute completed output path")
    source = verifier._canonical(Path(output))
    matches = [
        path
        for path in expected.parent.glob(f"{expected.stem}.*")
        if path.is_file() and path.stat().st_size > 0
    ]
    if matches != [source]:
        raise ReplayError("Missing, changed or ambiguous completed output path")
    size = saved.get("bytes")
    if type(size) is not int or size <= 0 or source.stat().st_size != size:
        raise ReplayError("Changed completed output size")
    return source, job_id


def _historical_bindings(
    descriptor: Path, jobs: Path, state: Path, bindings: dict[Path, dict[str, object]]
) -> None:
    """Verify explicit retained job/state hashes before permitting a legacy name."""
    descriptor = _local(descriptor)
    bindings[descriptor] = verifier._fingerprint(descriptor)
    raw = verifier._json(descriptor)
    if (
        set(raw) != {"version", "kind", "jobs", "state"}
        or type(raw.get("version")) is not int
        or raw["version"] != 1
        or raw["kind"] != "historical-compressor-output"
    ):
        raise ReplayError("Unknown historical Compressor binding schema")
    for key, path in (("jobs", jobs), ("state", state)):
        pin = raw.get(key)
        if pin != {"path": str(path), "sha256": bindings[path]["sha256"]}:
            raise ReplayError("Historical job/state binding changed")


def _probe(movie: Path, *, tool: Path, evidence: Path, name: str, timeout: int) -> Path:
    return verifier._run(
        [
            str(tool),
            "-v",
            "error",
            "-protocol_whitelist",
            "file,pipe",
            "-of",
            "json",
            "-show_streams",
            "-show_packets",
            "-show_data",
            "-show_data_hash",
            "sha256",
            "-show_entries",
            "stream:packet=stream_index,pts,dts,duration,size,flags,data_hash,side_data_list",
            str(movie),
        ],
        directory=evidence,
        name=name,
        timeout=timeout,
    )


def _extra(stream: dict[str, object]) -> bytes:
    text = stream.get("extradata")
    if not isinstance(text, str):
        raise ReplayError("Missing codec extradata dump")
    data = bytearray()
    for line in text.strip().splitlines():
        offset, separator, body = line.partition(": ")
        if not separator or int(offset, 16) != len(data):
            raise ReplayError("Malformed codec extradata dump")
        data.extend(bytes.fromhex(body[:39]))
    if len(data) != verifier._int(stream.get("extradata_size")) or "SHA256:" + hashlib.sha256(
        data
    ).hexdigest() != stream.get("extradata_hash"):
        raise ReplayError("Codec extradata dump/hash mismatch")
    return bytes(data)


def _hvc1_extra(original: dict[str, object], remux: dict[str, object]) -> bool:
    # hvc1 muxing may only set completeness on VPS/SPS/PPS arrays in hvcC.
    data = bytearray(_extra(original))
    if len(data) < 23 or data[0] != 1:
        raise ReplayError("Invalid hvcC header")
    cursor = 23
    for _ in range(data[22]):
        if cursor + 3 > len(data):
            raise ReplayError("Truncated hvcC array")
        if data[cursor] & 0x3F in (32, 33, 34):
            data[cursor] |= 0x80
        count = int.from_bytes(data[cursor + 1 : cursor + 3], "big")
        cursor += 3
        for _ in range(count):
            if cursor + 2 > len(data):
                raise ReplayError("Truncated hvcC NAL length")
            length = int.from_bytes(data[cursor : cursor + 2], "big")
            cursor += 2 + length
            if length == 0 or cursor > len(data):
                raise ReplayError("Truncated hvcC NAL payload")
    if cursor != len(data):
        raise ReplayError("Trailing hvcC bytes")
    return data == _extra(remux)


def compare(
    source: dict[str, object],
    candidate: dict[str, object],
    *,
    allow_hvc1: bool = False,
) -> list[dict[str, object]]:
    """Compare each A/V stream in demux order; container interleaving may differ."""
    left = verifier._rows(source, "streams")
    right = verifier._rows(candidate, "streams")
    if len(left) != 2 or len(right) != 2:
        raise ReplayError("Exactly one video and one audio stream required")
    kinds = [row.get("codec_type") for row in left]
    if kinds.count("video") != 1 or kinds.count("audio") != 1:
        raise ReplayError("Missing video/audio stream")
    packets_left = verifier._rows(source, "packets")
    packets_right = verifier._rows(candidate, "packets")
    indices = [verifier._int(row.get("index")) for row in left]
    if len(set(indices)) != 2 or any(
        verifier._int(row.get("stream_index")) not in indices
        for row in chain(packets_left, packets_right)
    ):
        raise ReplayError("Unknown or duplicate stream index")
    results: list[dict[str, object]] = []
    for original, remux in zip(left, right, strict=True):
        if any(original.get(key) != remux.get(key) for key in STREAM_FIELDS):
            raise ReplayError("Codec/stream identity mismatch")
        if not isinstance(original.get("codec_name"), str):
            raise ReplayError("Missing codec identity")
        extra = original.get("extradata_hash")
        if not isinstance(extra, str) or SHA256.fullmatch(extra) is None:
            raise ReplayError("Missing codec extradata SHA256")
        tag, new_tag = original.get("codec_tag_string"), remux.get("codec_tag_string")
        changed_tag = tag != new_tag
        if not isinstance(tag, str) or (
            changed_tag
            and not (
                allow_hvc1
                and original.get("codec_name") == "hevc"
                and tag == "hev1"
                and new_tag == "hvc1"
            )
        ):
            raise ReplayError("Codec tag mismatch (hev1 to hvc1 requires opt-in)")
        if extra != remux.get("extradata_hash") and not (
            changed_tag and _hvc1_extra(original, remux)
        ):
            raise ReplayError("Codec extradata mismatch")
        scale_left = verifier._scale(original.get("time_base"))
        scale_right = verifier._scale(remux.get("time_base"))
        index = original["index"]
        a = [row for row in packets_left if row.get("stream_index") == index]
        b = [row for row in packets_right if row.get("stream_index") == index]
        if not a or len(a) != len(b):
            raise ReplayError("Missing packets or packet count mismatch")
        for before, after in zip(a, b, strict=True):
            digest = before.get("data_hash")
            if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
                raise ReplayError("Missing packet payload SHA256")
            if (
                any(
                    before.get(key) != after.get(key)
                    for key in (
                        "data_hash",
                        "size",
                        "flags",
                        "side_data_list",
                    )
                )
                or verifier._int(before.get("size")) <= 0
            ):
                raise ReplayError("Ordered packet payload/side-data mismatch")
            for clock in ("pts", "dts", "duration"):
                if verifier._int(before.get(clock)) * scale_left != (
                    verifier._int(after.get(clock)) * scale_right
                ):
                    raise ReplayError("Packet rational clock mismatch")
            if verifier._int(before.get("duration")) <= 0:
                raise ReplayError("Packet duration must be positive")
        results.append(
            {
                "index": index,
                "type": original["codec_type"],
                "packets": len(a),
                "sourceTimeBase": str(scale_left),
                "candidateTimeBase": str(scale_right),
                "hvc1TagChanged": changed_tag,
                "orderedPayloadsEqual": True,
                "rationalClocksEqual": True,
            }
        )
    return results


def audit(
    config_path: Path,
    movie: Path,
    evidence: Path,
    *,
    ffprobe: Path,
    allow_hvc1: bool = False,
    timeout: int = 7200,
    historical_bindings: Path | None = None,
) -> bool:
    """Bind existing files before/after inspection; reserve new evidence or raise."""
    evidence = _local(evidence)
    evidence.mkdir(mode=0o700)
    receipt: dict[str, object] = {
        "version": 1,
        "remuxIdentityAccepted": False,
        "fullDecode": False,
        "qualityApproval": False,
        "syncApproval": False,
        "allowHvc1TagChange": allow_hvc1,
        "nativeStatusSource": "existing Successful state receipt; no native call",
        "historicalNaming": historical_bindings is not None,
    }
    bindings: dict[Path, dict[str, object]] = {}
    try:
        if not 1 <= timeout <= verifier.MAX_SECONDS:
            raise ReplayError("Invalid process timeout")
        config_path, movie = _local(config_path), verifier._canonical(movie)
        bindings[config_path] = verifier._fingerprint(config_path)
        config = load_config(config_path)
        root = verifier._canonical(config.delivery_root)
        verifier._canonical(config.storage)
        if not movie.is_relative_to(root) or movie == root:
            raise ReplayError("Candidate is outside configured delivery root")
        jobs = _local(config.work / "final-job.json")
        state = _local(config.work / "final-job-state" / "final.json")
        bindings[jobs] = verifier._fingerprint(jobs)
        bindings[state] = verifier._fingerprint(state)
        if historical_bindings is not None:
            _historical_bindings(historical_bindings, jobs, state, bindings)
        else:
            bindings[config.final_preset] = verifier._fingerprint(config.final_preset)
        source, job_id = _completed(config, jobs, state, historical=historical_bindings is not None)
        if source == movie:
            raise ReplayError("Candidate must be distinct from Compressor output")
        bindings[source] = verifier._fingerprint(source)
        bindings[movie] = verifier._fingerprint(movie)
        receipt.update(
            nativeJobID=job_id, nativeStatus="Successful", source=str(source), candidate=str(movie)
        )
        if not ffprobe.is_absolute() or not ffprobe.is_file() or not os.access(ffprobe, os.X_OK):
            raise ReplayError("FFprobe must be an explicit absolute executable")
        receipt["ffprobe"] = str(ffprobe.resolve())
        original_probe = _probe(
            source, tool=ffprobe, evidence=evidence, name="source", timeout=timeout
        )
        remux_probe = _probe(
            movie, tool=ffprobe, evidence=evidence, name="candidate", timeout=timeout
        )
        receipt["streams"] = compare(
            verifier._json(original_probe), verifier._json(remux_probe), allow_hvc1=allow_hvc1
        )
        receipt["remuxIdentityAccepted"] = True
    except verifier.ERRORS as exc:
        receipt["error"] = str(exc)
    finally:
        receipt["inputs"] = list(bindings.values())
        try:
            after = {path: verifier._fingerprint(path) for path in bindings}
            receipt["inputsAfter"] = list(after.values())
            if after != bindings:
                raise ReplayError("Bound input changed during remux audit")
        except verifier.ERRORS as exc:
            receipt["remuxIdentityAccepted"] = False
            receipt["bindingError"] = str(exc)
        verifier._write(evidence / "receipt.json", receipt)
    return receipt["remuxIdentityAccepted"] is True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("movie", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--ffprobe", type=Path, required=True)
    parser.add_argument("--allow-hvc1-tag-change", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--historical-bindings", type=Path)
    args = parser.parse_args(argv)
    try:
        return (
            0
            if audit(
                args.config,
                args.movie,
                args.evidence,
                ffprobe=args.ffprobe,
                allow_hvc1=args.allow_hvc1_tag_change,
                timeout=args.timeout,
                historical_bindings=args.historical_bindings,
            )
            else 1
        )
    except verifier.ERRORS as exc:
        print(json.dumps({"remuxIdentityAccepted": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
