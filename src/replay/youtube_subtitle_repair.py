"""Run local, config-defined source-context recognition without changing any video."""

import argparse
import importlib
import json
import re
import wave
from hashlib import file_digest
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record
from replay.youtube_subtitles import (
    UnrenderableSegment,
    _duration,
    _job_data,
    _number,
    _run,
    _text,
    load_job,
    segment_cues,
)


def run_repairs(config: Path) -> Path:
    data = record(json.loads(config.read_text(encoding="utf-8")))
    if set(data) != {
        "version",
        "source_config",
        "source_job",
        "output_name",
        "evidence",
        "windows",
    }:
        raise ReplayError("Unknown or incomplete repair configuration")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ReplayError("Unsupported repair configuration version")
    job = load_job(Path(_text(data["source_config"])))
    source_job = Path(_text(data["source_job"])).resolve(strict=True)
    if source_job.parent != job.output_root.resolve() or not source_job.is_dir():
        raise ReplayError("Repair source job outside configured output root")
    if json.loads((source_job / "metadata/job.json").read_text(encoding="utf-8")) != _job_data(job):
        raise ReplayError("Repair job/source configuration mismatch")
    evidence = Path(_text(data["evidence"])).resolve(strict=True)
    if not evidence.is_relative_to(source_job / "metadata") or not evidence.is_file():
        raise ReplayError("Repair evidence outside source job")
    name = _text(data["output_name"])
    if re.fullmatch(r"[a-z][a-z0-9-]{0,63}", name) is None:
        raise ReplayError("Invalid repair output name")
    windows = data["windows"]
    if not isinstance(windows, list) or not 0 < len(windows) <= 2048:
        raise ReplayError("Expected bounded nonempty repair windows")
    duration = _duration(job)
    for raw in windows:
        item = record(raw)
        if set(item) != {"start", "end", "target_start", "target_end", "reason"}:
            raise ReplayError("Invalid repair window fields")
        start, end = _number(item["start"]), _number(item["end"])
        target_start, target_end = _number(item["target_start"]), _number(item["target_end"])
        if not 0 <= start <= target_start < target_end <= end <= duration or end - start > 60:
            raise ReplayError("Repair window outside measured source extent")
        if len(_text(item["reason"])) > 200:
            raise ReplayError("Repair reason exceeds text budget")
    with (job.model / "weights.npz").open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != job.weights_sha256:
            raise ReplayError("Local repair model checksum changed")
    transcribe: object = importlib.import_module("mlx_whisper").transcribe
    if not callable(transcribe):
        raise ReplayError("Installed repair recognizer is not callable")
    output = source_job / "metadata" / name
    output.mkdir(mode=0o700, exist_ok=True)
    manifest = output / "plan.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != data:
            raise ReplayError("Existing repair plan changed; choose a new output name")
    else:
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
    for index, raw in enumerate(windows):
        item = record(raw)
        cached = output / f"clip-{index:05}.json"
        if cached.exists():
            stored = record(json.loads(cached.read_text(encoding="utf-8")))
            if stored.get("window") != item or stored.get("source") != str(job.source):
                raise ReplayError("Cached repair input mismatch")
            continue
        start, end = _number(item["start"]), _number(item["end"])
        audio = output / f"clip-{index:05}.wav"
        if not audio.exists():
            _run(
                [
                    str(job.ffmpeg),
                    "-nostdin",
                    "-v",
                    "error",
                    "-n",
                    "-ss",
                    str(start),
                    "-i",
                    str(job.source),
                    "-t",
                    str(end - start),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(audio),
                ]
            )
        with wave.open(str(audio), "rb") as measured:
            if (measured.getframerate(), measured.getnchannels(), measured.getsampwidth()) != (
                16000,
                1,
                2,
            ):
                raise ReplayError("Unexpected repair PCM clock")
            measured_duration = measured.getnframes() / 16000
        if abs(measured_duration - (end - start)) > 0.001:
            raise ReplayError("Incomplete or changed repair audio")
        recognition: object = transcribe(
            str(audio),
            path_or_hf_repo=str(job.model),
            language="ja",
            verbose=False,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        try:
            segment_cues(recognition, offset=start, duration=measured_duration)
        except UnrenderableSegment:
            renderable = False
        else:
            renderable = True
        with cached.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "source": str(job.source),
                    "window": item,
                    "source_offset": start,
                    "measured_duration": measured_duration,
                    "recognition": recognition,
                    "all_segments_renderable": renderable,
                    "status": "native-ASR-hypothesis-awaiting-reconciliation",
                },
                stream,
                ensure_ascii=False,
            )
        print(f"Recognized repair context {index + 1}/{len(windows)}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(run_repairs(args.config))


if __name__ == "__main__":
    main()
