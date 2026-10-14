"""Render a config-selected set of actual native ASR segments, never invented clocks."""

import argparse
import json
import wave
from hashlib import file_digest
from operator import attrgetter
from pathlib import Path

from replay.core import ReplayError
from replay.recordings import record
from replay.speaker_subtitles import SpeakerCue
from replay.youtube_delivery import new_edit_directory
from replay.youtube_subtitles import (
    _duration,
    _job_data,
    _number,
    _text,
    load_job,
    render_subtitles,
    segment_cues,
)


def finalize(config: Path) -> Path:
    data = record(json.loads(config.read_text(encoding="utf-8")))
    if set(data) != {"version", "source_config", "source_job", "review", "selection"}:
        raise ReplayError("Unknown or incomplete native selection configuration")
    if type(data["version"]) is not int or data["version"] != 1:
        raise ReplayError("Unsupported native selection version")
    job = load_job(Path(_text(data["source_config"])))
    source_job = Path(_text(data["source_job"])).resolve(strict=True)
    if source_job.parent != job.output_root.resolve():
        raise ReplayError("Selection source job outside configured output root")
    metadata = source_job / "metadata"
    if json.loads((metadata / "job.json").read_text(encoding="utf-8")) != _job_data(job):
        raise ReplayError("Selection source configuration mismatch")
    review = Path(_text(data["review"])).resolve(strict=True)
    if not review.is_relative_to(metadata) or not review.is_file():
        raise ReplayError("Selection review outside source job")
    selection = data["selection"]
    if not isinstance(selection, list) or not 0 < len(selection) <= 50000:
        raise ReplayError("Expected bounded nonempty native segment references")
    duration = _duration(job)
    cues: list[SpeakerCue] = []
    seen: set[tuple[Path, int]] = set()
    for raw in selection:
        item = record(raw)
        if set(item) != {"record", "segment"}:
            raise ReplayError("Selection permits references only, not supplied text or clocks")
        reference = (source_job / _text(item["record"])).resolve(strict=True)
        if not reference.is_relative_to(metadata) or not reference.is_file():
            raise ReplayError("Native recognition reference outside source job")
        measured = record(json.loads(reference.read_text(encoding="utf-8")))
        if "source" in measured and measured["source"] != str(job.source):
            raise ReplayError("Selected recognition input source mismatch")
        recognition = record(measured.get("recognition"))
        segments = recognition.get("segments")
        number = item["segment"]
        if (
            type(number) is not int
            or not isinstance(segments, list)
            or not 0 <= number < len(segments)
            or len(segments) > 50000
        ):
            raise ReplayError("Invalid native segment index")
        identity = (reference, number)
        if identity in seen:
            raise ReplayError("Duplicate native segment reference")
        seen.add(identity)
        offset = _number(measured.get("source_offset"))
        audio = reference.with_suffix(".wav").resolve(strict=True)
        if not audio.is_relative_to(metadata):
            raise ReplayError("Selected recognition audio outside source job")
        with wave.open(str(audio), "rb") as source:
            if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (
                16000,
                1,
                2,
            ):
                raise ReplayError("Unexpected selected native input PCM clock")
            extent = source.getnframes() / 16000
        if not 0 <= offset < duration or offset + extent > duration + 0.001:
            raise ReplayError("Selected native audio outside input video")
        chosen = segment_cues(
            {"language": recognition.get("language"), "segments": [segments[number]]},
            offset=offset,
            duration=extent,
        )
        if not chosen or chosen[0].end > duration:
            raise ReplayError("Empty or out-of-video native segment")
        cues.extend(chosen)
    with (job.model / "weights.npz").open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != job.weights_sha256:
            raise ReplayError("Selected model provenance changed")
    publication: object = json.loads(job.publication.read_text(encoding="utf-8"))
    output = new_edit_directory(job.output_root, video_id=job.video_id, metadata=publication)
    (output / "metadata").mkdir(mode=0o700)
    with (output / "metadata/job.json").open("x", encoding="utf-8") as stream:
        json.dump(_job_data(job), stream, indent=2)
    with (output / "metadata/native-selection.json").open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
    return render_subtitles(
        job, output=output, cues=tuple(sorted(cues, key=attrgetter("start", "end")))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(finalize(args.config))


if __name__ == "__main__":
    main()
