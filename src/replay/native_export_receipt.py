"""Verify a native export dispatch against independently decoded part media."""

import argparse
import json
import math
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from replay.artifact_json import _json, _path, _pin
from replay.core import ReplayError
from replay.fcp_native_parts import _idle
from replay.fcp_part_assembly import verify
from replay.recordings import record
from replay.stages import file_digest

FORMAT_LABELS = {"ja": "ビデオとオーディオ", "en": "Video and Audio"}
SAVE_ACTIONS = {"ja": "保存のみ", "en": "Save only"}
PRORES_PROFILES = {
    "Apple ProRes 422 Proxy": "Proxy",
    "Apple ProRes 422 LT": "LT",
    "Apple ProRes 422": "Standard",
    "Apple ProRes 422 HQ": "HQ",
    "Apple ProRes 4444": "4444",
    "Apple ProRes 4444 XQ": "XQ",
}
MAX_METADATA_BYTES = 1024 * 1024
DECODE_TIMEOUT = 7200


def native_payload(raw: dict[str, object]) -> dict[str, object]:
    """Decode the actual Executor MCP envelope; reject failed or ambiguous wrappers."""
    if raw.get("ok") is not True or raw.get("error") is not None:
        raise ReplayError("Native call did not succeed")
    data = record(raw.get("data"))
    if data.get("isError") is not False:
        raise ReplayError("Native MCP result is not explicitly successful")
    return record(data.get("structuredContent"))


def save_new(path: Path, value: dict[str, object]) -> None:
    """Write private evidence exclusively; failures retain their partial evidence."""
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _dispatch(request: dict[str, object], payload: dict[str, object], project: str) -> Path:
    native = record(payload.get("finalCut"))
    exported = record(native.get("export"))
    codec = request.get("codec")
    name = request.get("fileName")
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ReplayError("Export filename is invalid")
    destination = _path(str(_path(request.get("directory")) / name))
    pid = payload.get("pid")
    if (
        request.get("bundleID") != "com.apple.FinalCutApp"
        or request.get("project") != project
        or record(native.get("project")).get("name") != project
        or native.get("vocabularyVerified") is not True
        or native.get("language") not in FORMAT_LABELS
        or request.get("format") != "videoAndAudio"
        or request.get("allowForeground") is not True
        or not isinstance(codec, str)
        or codec not in PRORES_PROFILES
        or exported.get("codec") != codec
        or exported.get("format") != FORMAT_LABELS.get(str(native.get("language")))
        or exported.get("action") != SAVE_ACTIONS.get(str(native.get("language")))
        or exported.get("outputPath") != str(destination)
        or exported.get("foregroundRestored") is not True
        or exported.get("completionVerified") is not False
        or payload.get("retrySafe") is not False
        or type(pid) is not int
        or pid <= 0
    ):
        raise ReplayError("Native dispatch does not bind the requested part export")
    return destination


def _metadata(raw: dict[str, object], expected: dict[str, object]) -> None:
    streams = raw.get("streams")
    if not isinstance(streams, list) or not 2 <= len(streams) <= 3:
        raise ReplayError("Part master requires video/audio and at most one timecode stream")
    videos = [record(item) for item in streams if record(item).get("codec_type") == "video"]
    audios = [record(item) for item in streams if record(item).get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ReplayError("Part master stream types differ")
    timecodes = [
        record(item) for item in streams if record(item).get("codec_type") not in {"video", "audio"}
    ]
    video, audio = videos[0], audios[0]
    try:
        valid = (
            video.get("codec_name") == "prores"
            and video.get("profile") == expected["profile"]
            and video.get("width") == expected["width"]
            and video.get("height") == expected["height"]
            and video.get("start_pts") == audio.get("start_pts") == 0
            and Fraction(str(video.get("avg_frame_rate"))) == expected["rate"]
            and Fraction(str(video.get("duration_ts"))) * Fraction(str(video.get("time_base")))
            == expected["duration"]
            and str(audio.get("codec_name")).startswith("pcm_")
            and audio.get("sample_rate") == str(expected["audioRate"])
            and audio.get("channels") == expected["channels"]
            and Fraction(str(audio.get("duration_ts"))) * Fraction(str(audio.get("time_base")))
            == expected["duration"]
            and all(
                item.get("codec_type") == "data"
                and item.get("codec_tag_string") == "tmcd"
                and item.get("start_pts") == 0
                and Fraction(str(item.get("avg_frame_rate"))) == expected["rate"]
                and Fraction(str(item.get("duration_ts"))) * Fraction(str(item.get("time_base")))
                == expected["duration"]
                for item in timecodes
            )
        )
    except (ValueError, ZeroDivisionError) as error:
        raise ReplayError("Invalid part master metadata clock") from error
    if not valid:
        raise ReplayError("Part master codec, geometry or AV clock differs")


def _birth_ns(value: os.stat_result) -> int:
    birth: object = getattr(value, "st_birthtime", None)
    if type(birth) is not float or not math.isfinite(birth) or birth <= 0:
        raise ReplayError("Recovery requires filesystem birth timestamps")
    return int(birth * 1_000_000_000)


def _recovered_dispatch(
    paths: dict[str, Path], request: dict[str, object], project: str
) -> tuple[Path, int, dict[str, object]]:
    """Bind an uncertain dispatch using filesystem chronology, never a forged success."""
    raw = _json(paths["exportResponse"])
    error = record(raw.get("error"))
    detail = error.get("message")
    if (
        raw.get("ok") is not False
        or error.get("code") != "mcp_tool_error"
        or not isinstance(detail, str)
    ):
        raise ReplayError("Recovery requires the original uncertain native failure")
    try:
        failure = record(json.loads(detail))
    except json.JSONDecodeError as error:
        raise ReplayError("Recovery failure detail is not structured") from error
    if failure.get("retrySafe") is not False or "AXError -25204" not in str(failure.get("error")):
        raise ReplayError("Recovery is restricted to an uncertain busy Accessibility response")
    opened = native_payload(_json(paths["projectObservation"]))
    native = record(opened.get("finalCut"))
    pid = opened.get("pid")
    name = request.get("fileName")
    if (
        type(pid) is not int
        or pid <= 0
        or native.get("vocabularyVerified") is not True
        or native.get("language") not in FORMAT_LABELS
        or record(native.get("project")).get("name") != project
        or request.get("project") != project
        or request.get("bundleID") != "com.apple.FinalCutApp"
        or request.get("format") != "videoAndAudio"
        or request.get("allowForeground") is not True
        or not isinstance(request.get("codec"), str)
        or request.get("codec") not in PRORES_PROFILES
        or not isinstance(name, str)
        or Path(name).name != name
        or name in {"", ".", ".."}
    ):
        raise ReplayError("Recovery project and request do not identify one native export")
    movie = _path(str(_path(request.get("directory")) / name))
    stats = {
        key: path.stat()
        for key, path in paths.items()
        if key in {"projectObservation", "exportResponse", "background"}
    }
    media = movie.stat()
    response = stats["exportResponse"]
    if not (
        stats["projectObservation"].st_mtime_ns
        <= _birth_ns(response)
        <= _birth_ns(media)
        <= media.st_mtime_ns
        <= stats["background"].st_mtime_ns
        and _birth_ns(response) <= response.st_mtime_ns <= stats["background"].st_mtime_ns
    ):
        raise ReplayError("Recovery chronology rejects pre-existing media or stale observations")
    evidence: dict[str, object] = {
        "projectObservedMtimeNs": stats["projectObservation"].st_mtime_ns,
        "responseBirthNs": _birth_ns(response),
        "responseMtimeNs": response.st_mtime_ns,
        "movieBirthNs": _birth_ns(media),
        "movieMtimeNs": media.st_mtime_ns,
        "movieDevice": media.st_dev,
        "movieInode": media.st_ino,
        "movieSize": media.st_size,
        "idleMtimeNs": stats["background"].st_mtime_ns,
        "limitations": (
            "Birth times use float-second filesystem precision, not exact nanoseconds. "
            "Chronology does not prove dispatch or visual correctness."
        ),
    }
    return movie, pid, evidence


def complete(config_path: Path, output: Path, *, recover: bool = False) -> None:
    """Run only metadata/full AV verification; never dispatch apps or encode media."""
    config_path, output = _path(str(config_path)), _path(str(output))
    if output.exists() or output.is_symlink():
        raise ReplayError("Export completion receipt already exists")
    config = _json(config_path)
    part = config.get("part")
    audio_rate, channels = config.get("audioSampleRate"), config.get("audioChannels")
    if (
        config.get("version") != 1
        or type(part) is not int
        or part < 1
        or type(audio_rate) is not int
        or audio_rate <= 0
        or type(channels) is not int
        or not 1 <= channels <= 64
    ):
        raise ReplayError("Invalid part completion configuration")
    paths = {
        key: _pin(config.get(key))
        for key in (
            "assembly",
            "exportRequest",
            "exportResponse",
            "background",
            "ffmpeg",
            "ffprobe",
        )
    }
    if recover:
        paths["projectObservation"] = _pin(config.get("projectObservation"))
    proof_path = verify(paths["assembly"], part=part)
    proof = _json(proof_path)
    assembly = _json(paths["assembly"])
    partition_path = _pin(assembly.get("partition"))
    parts = _json(partition_path).get("parts")
    if not isinstance(parts, list) or not 1 <= part <= len(parts):
        raise ReplayError("Partition part is absent")
    part_xml = _pin(parts[part - 1])
    project = ET.parse(part_xml).getroot().find("./library/event/project")
    if project is None or not project.get("name"):
        raise ReplayError("Partition project name is absent")
    project_name = project.attrib["name"]
    request = _json(paths["exportRequest"])
    recovery: dict[str, object] | None = None
    pid: object
    if recover:
        movie, pid, recovery = _recovered_dispatch(paths, request, project_name)
    else:
        response = native_payload(_json(paths["exportResponse"]))
        movie = _dispatch(request, response, project_name)
        pid = response["pid"]
    if record(proof.get("movie")).get("path") != str(movie):
        raise ReplayError("Export path differs from frame-verified movie")
    background = native_payload(_json(paths["background"]))
    _idle(background)
    if background.get("pid") != pid or paths["background"].stat().st_mtime_ns < max(
        movie.stat().st_mtime_ns, paths["exportResponse"].stat().st_mtime_ns
    ):
        raise ReplayError("Idle observation does not follow this process's completed export")
    observed = record(proof.get("observed"))
    streams = observed.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise ReplayError("Frame proof must contain exactly one video stream")
    video = record(streams[0])
    expected: dict[str, object] = {
        "profile": PRORES_PROFILES[str(request["codec"])],
        "width": video.get("width"),
        "height": video.get("height"),
        "rate": Fraction(str(video.get("avg_frame_rate"))),
        "duration": Fraction(str(video.get("duration_ts"))) * Fraction(str(video.get("time_base"))),
        "audioRate": audio_rate,
        "channels": channels,
    }
    inputs = {
        str(path): file_digest(path)
        for path in (config_path, proof_path, partition_path, part_xml, *paths.values())
    }
    metadata = subprocess.run(
        [str(paths["ffprobe"]), "-v", "error", "-show_streams", "-of", "json", str(movie)],
        capture_output=True,
        check=True,
        timeout=600,
    )
    if len(metadata.stdout) > MAX_METADATA_BYTES:
        raise ReplayError("Part metadata exceeds budget")
    parsed = record(json.loads(metadata.stdout))
    _metadata(parsed, expected)
    command = [
        str(paths["ffmpeg"]),
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(movie),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-f",
        "null",
        "-",
    ]
    subprocess.run(command, capture_output=True, check=True, timeout=DECODE_TIMEOUT)
    if file_digest(movie) != record(proof.get("movie")).get("sha256") or any(
        file_digest(Path(path)) != digest for path, digest in inputs.items()
    ):
        raise ReplayError("Export evidence changed during verification")
    if recover and _recovered_dispatch(paths, request, project_name)[2] != recovery:
        raise ReplayError("Recovery filesystem identity changed during verification")
    save_new(
        output,
        {
            "version": 1,
            "kind": "verified-native-part",
            "movie": proof["movie"],
            "part": part,
            "project": project_name,
            "frameProof": {"path": str(proof_path), "sha256": inputs[str(proof_path)]},
            "inputs": inputs,
            "observed": parsed,
            "fullAVDecodeCommand": command,
            "fullAVDecodeExitCode": 0,
            "shareObservedIdle": True,
            "nativeReportedCompletion": False,
            "completionVerified": True,
            "qualityApproved": False,
            "nativeDispatchVerified": not recover,
            "recoveredAfterUncertainDispatch": recover,
            "recoveryEvidence": recovery,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    complete(args.config, args.output, recover=args.recover)


if __name__ == "__main__":
    main()
