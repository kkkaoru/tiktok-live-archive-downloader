"""Consume completed stateful checkpoints, never requested or historical support."""

import argparse
import json
import subprocess
import wave
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from replay.clocked_pcm import reader_placements, write_clocked_pcm
from replay.native_invocation import canonical, durable, pin, read, require, rows
from replay.recordings import record
from replay.stages import _fingerprint, file_digest
from replay.stateful_audio import RATE, checkpoints, integer, time, verify
from replay.workflow import load_stages


class Arguments(argparse.Namespace):
    plan: Path
    sha256: str
    state: Path
    index: int
    output: Path


def _observed_buffers(request: Path, index: int) -> tuple[list[object], list[object]]:
    """Intersect verified actual ledger buffers, retaining original decode ordinals."""
    data = read(request)
    targets = checkpoints(data)
    require(type(index) is int and 0 <= index < len(targets), "Unknown checkpoint")
    start, count = targets[index]
    end = start + Fraction(count, RATE)
    buffers: list[object] = []
    lineage: list[object] = []
    with (canonical(data.get("outputDirectory")) / "ledger.jsonl").open("rb") as stream:
        for line in stream:
            row = record(json.loads(line))
            pts = time(row.get("sourcePTS"))
            lower = max(start, pts)
            upper = min(end, pts + time(row.get("duration")))
            if lower >= upper:
                continue
            offset, samples = (lower - pts) * RATE, (upper - lower) * RATE
            require(offset.denominator == samples.denominator == 1, "Partial source sample")
            buffers.append(
                {
                    "value": lower.numerator,
                    "timescale": lower.denominator,
                    "durationValue": samples.numerator,
                    "durationTimescale": RATE,
                    "samples": samples.numerator,
                }
            )
            lineage.append(
                {
                    "bufferOrdinal": row.get("bufferOrdinal"),
                    "sampleOrdinal": integer(row.get("sampleOrdinal")) + int(offset),
                    "sourcePTS": str(lower),
                    "samples": int(samples),
                }
            )
    return buffers, lineage


def copy_support(*, plan: Path, plan_sha256: str, state: Path, index: int, output: Path) -> Path:
    """Pin completion and independent MOV clocks/PCM, then use the exact-copy writer.

    Only this checkpoint's unprojected PCM is accepted. No model lineage, offset
    repair, full-source EOF claim or editor playback approval is inferred.
    """
    plan, state, output = (canonical(str(path)) for path in (plan, state, output))
    require(output.is_relative_to(Path.cwd() / "edits/work"), "Output must stay local")
    require(file_digest(plan) == plan_sha256, "Workflow plan changed")
    stages = load_stages(plan)
    require(len(stages) == 1, "Expected single stateful observation stage")
    stage = stages[0]
    guarded = len(stage.command) == 6 and stage.command[1:3] == ("-m", "replay.stateful_guard")
    require(
        stage.name == "source-pts-observation" and (guarded or len(stage.command) == 2),
        "Unsupported observation command",
    )
    request = canonical(stage.command[3] if guarded else stage.command[1])
    native = canonical(stage.command[4] if guarded else stage.command[0])
    guard_report = canonical(stage.command[5]) if guarded else None
    command = stage.verify_command
    require(
        len(command) == 9
        and command[1:4] == ("-m", "replay.stateful_audio", "verify")
        and command[4:6] == (str(request), str(native)),
        "Unbound verifier",
    )
    ffmpeg, ffprobe = canonical(command[6]), canonical(command[7])
    data = read(request)
    directory = canonical(data.get("outputDirectory"))
    expected_outputs = (
        directory / "stateful.json",
        directory / "ledger.jsonl",
        *(directory / f"checkpoint-{n}.mov" for n in range(len(checkpoints(data)))),
    )
    budget = record(record(data.get("statefulSelection")).get("budget"))
    require(guarded == ("maxRSSBytes" in budget), "Missing watchdog command")
    if guard_report is not None:
        require(guard_report == canonical(command[8]) / "guard.json", "Unbound watchdog result")
        expected_outputs += (guard_report,)
    require(stage.outputs == expected_outputs, "Incomplete observation outputs")
    required = {
        request,
        native,
        canonical(data.get("source")),
        ffmpeg,
        ffprobe,
        Path.cwd() / "src/replay/stateful_audio.py",
        Path.cwd() / "src/replay/workflow.py",
        Path.cwd() / "src/replay/stages.py",
        Path.cwd() / "src/replay/native_invocation.py",
        Path.cwd() / "src/replay/recordings.py",
        Path(command[0]).resolve(),
        Path.cwd() / "native/apple-pro-apps/Package.swift",
        *(Path.cwd() / "native/apple-pro-apps/Sources/AudioClockProbe").glob("*.swift"),
    }
    if guarded:
        required.update((Path.cwd() / "src/replay/stateful_guard.py", Path("/bin/ps").resolve()))
    require(required <= set(stage.inputs), "Missing implementation/source/tool bindings")
    receipt = state / f"{stage.name}.json"
    expected = {
        "status": "complete",
        "fingerprint": _fingerprint(stage),
        "outputs": {str(path): file_digest(path) for path in stage.outputs},
    }
    require(read(receipt) == expected, "Observation is incomplete or changed")
    own_code = (
        Path(__file__).resolve(),
        Path.cwd() / "src/replay/clocked_pcm.py",
        Path.cwd() / "src/replay/continuous_cuts.py",
    )
    bindings = {
        path: pin(path) for path in (plan, receipt, *stage.inputs, *stage.outputs, *own_code)
    }
    verify(request, ffmpeg, ffprobe, guard_report=guard_report)
    buffers, lineage = _observed_buffers(request, index)
    start, count = checkpoints(data)[index]
    spans = reader_placements(
        {"mode": "track", "buffers": buffers},
        sample_rate=RATE,
        source_frames=count,
        origin=start,
        input_clock="reader-pcm-ordinal",
    )
    require(
        sum(span.sample_count for span in spans) == count
        and all(span.output_time == Fraction(span.source_start_sample, RATE) for span in spans),
        "Requested support differs from observation",
    )
    movie = directory / f"checkpoint-{index}.mov"
    probe = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_frames",
            "-show_streams",
            "-of",
            "json",
            str(movie),
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    metadata = record(json.loads(probe.stdout))
    streams = rows(metadata.get("streams"))
    require(len(streams) == 1, "Ambiguous MOV stream")
    clock = Fraction(str(record(streams[0]).get("time_base")))
    mov_buffers: list[object] = []
    for value in rows(metadata.get("frames")):
        frame = record(value)
        pts = integer(frame.get("pts")) * clock
        duration = integer(frame.get("duration"), 1) * clock
        mov_buffers.append(
            {
                "value": pts.numerator,
                "timescale": pts.denominator,
                "durationValue": duration.numerator,
                "durationTimescale": duration.denominator,
                "samples": integer(frame.get("nb_samples"), 1),
            }
        )
    mov_spans = reader_placements(
        {"mode": "track", "buffers": mov_buffers},
        sample_rate=RATE,
        source_frames=count,
        origin=Fraction(0),
        input_clock="reader-pcm-ordinal",
    )
    require(
        all(span.output_time == Fraction(span.source_start_sample, RATE) for span in mov_spans),
        "MOV does not expose the complete zero-based checkpoint clock",
    )
    pcm = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(movie),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-c:a",
            "pcm_s16le",
            "-",
        ],
        capture_output=True,
        check=True,
        timeout=30,
    )
    require(not pcm.stderr and len(pcm.stdout) == count * 4, "Incomplete independent PCM")
    output.mkdir(mode=0o700)
    wav = output / "readback.wav"
    with wave.open(str(wav), "wb") as writer:
        writer.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        writer.writeframes(pcm.stdout)
    wav.chmod(0o600)
    placement = write_clocked_pcm(
        wav,
        output / "supports",
        source_sha256=file_digest(wav),
        spans=spans,
        end=Fraction(count, RATE),
    )
    require(all(pin(path) == bound for path, bound in bindings.items()), "Bound inputs changed")
    result = output / "binding.json"
    durable(
        result,
        {
            "version": 1,
            "kind": "stateful-observed-checkpoint",
            "checkpoint": index,
            "inputs": list(bindings.values()),
            "ledgerSupport": lineage,
            "movReadback": metadata,
            "readback": pin(wav),
            "placement": pin(placement),
            "historicalModelApproved": False,
            "editorPlaybackApproved": False,
            "fullSourceEOFObserved": False,
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    args = Arguments()
    parser.parse_args(argv, namespace=args)
    print(
        copy_support(
            plan=args.plan,
            plan_sha256=args.sha256,
            state=args.state,
            index=args.index,
            output=args.output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
