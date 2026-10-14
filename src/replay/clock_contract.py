"""Verify explicit one-time clock conversions without modifying media or timestamps.

The same config is used to create or recheck a receipt. Reuse requires identical
artifact pins and a fresh domain check; execution success is not A/V acceptance.
"""

import argparse
import json
import os
import wave
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from replay.asr_clock_projection import _labels as pcm_labels
from replay.asr_clock_projection import project_labels
from replay.audio_clock import AudioFrameClock
from replay.clocked_cut_audio import plan_slots
from replay.clocked_tempo_audio import source_path
from replay.continuous_quiet_plan import _labels as source_labels
from replay.core import ReplayError
from replay.cut_audio_support import _cuts, _local, _read
from replay.input_hash_snapshot import InputHashSnapshot
from replay.recordings import record
from replay.stages import file_digest

RECOGNITION_ROLES = frozenset({"producerConfig", "inputLabels", "outputLabels", "audioClock"})
TEMPO_ROLES = frozenset(
    {
        "producerConfig",
        "producerReceipt",
        "inputAudio",
        "outputAudio",
        "audioClock",
        "cuts",
    }
)


def _artifact_binding(value: object, *, media_root: Path | None = None) -> tuple[Path, str]:
    """Validate local evidence or explicitly authorized read-only audio."""
    item = record(value)
    name, digest = item.get("path"), item.get("sha256")
    if set(item) != {"path", "sha256"} or not isinstance(name, str) or not isinstance(digest, str):
        raise ReplayError("Expected explicit clock artifact pins")
    path = Path(name).absolute()
    if media_root is not None and not path.is_relative_to(Path.cwd()):
        return source_path(name, media_root), digest
    if (
        path != path.resolve(strict=True)
        or not path.is_relative_to(Path.cwd())
        or not path.is_file()
    ):
        raise ReplayError("Clock artifacts must be canonical checkout files")
    return path, digest


def _matches(value: object, expected: tuple[Path, str]) -> bool:
    item = record(value)
    return item.get("path") == str(expected[0]) and item.get("sha256") == expected[1]


def _json_artifact(bindings: dict[str, tuple[Path, str]], role: str) -> dict[str, object]:
    return _read(_local(bindings[role][0]))


def _recognition(
    bindings: dict[str, tuple[Path, str]], clock: AudioFrameClock
) -> dict[str, object]:
    config = _json_artifact(bindings, "producerConfig")
    artifacts = record(config.get("artifacts"))
    if (
        config.get("inputClock") != "continuous-decoded-pcm"
        or config.get("transforms")
        != ["concatenate-decoded-frames", "duration-preserving-resample"]
        or not _matches(artifacts.get("labels"), bindings["inputLabels"])
        or not _matches(artifacts.get("audioClock"), bindings["audioClock"])
    ):
        raise ReplayError("Recognition producer does not bind this unprojected input and clock")
    original = pcm_labels(_json_artifact(bindings, "inputLabels"))
    output = _json_artifact(bindings, "outputLabels")
    parents = record(output.get("parentSHA256"))
    if (
        output.get("configSHA256") != bindings["producerConfig"][1]
        or parents.get(str(bindings["inputLabels"][0])) != bindings["inputLabels"][1]
        or parents.get(str(bindings["audioClock"][0])) != bindings["audioClock"][1]
        or source_labels(output) != project_labels(original, clock)
    ):
        raise ReplayError("Source labels are not the exact single decoded-clock projection")
    return {
        "inputClock": "continuous-decoded-pcm",
        "outputClock": "source-presentation",
        "inputOrigin": "0",
        "outputOrigin": "0",
        "sampleRate": clock.sample_rate,
        "mapping": "decoded-frame-piecewise",
        "labelCount": len(original),
        "cutApplications": 0,
        "nextInput": {
            "path": str(bindings["outputLabels"][0]),
            "clock": "source-presentation",
            "consumers": ["replay.cut_audio_support", "source-to-edited caption projection"],
            "forbidden": "Do not apply asr_clock_projection again",
        },
    }


def _rational(value: object) -> Fraction:
    item = record(value)
    numerator, denominator = item.get("value"), item.get("timescale")
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise ReplayError("Expected an exact rational cut boundary")
    return Fraction(numerator, denominator)


def _wave(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as reader:
        if reader.getsampwidth() != 2 or reader.getcomptype() != "NONE":
            raise ReplayError("Clock contract requires PCM16 WAV")
        return reader.getframerate(), reader.getnchannels(), reader.getnframes()


def _tempo(
    bindings: dict[str, tuple[Path, str]],
    clock: AudioFrameClock,
    legacy_grid: object = None,
) -> dict[str, object]:
    config = _json_artifact(bindings, "producerConfig")
    receipt = _json_artifact(bindings, "producerReceipt")
    cuts = _json_artifact(bindings, "cuts")
    fps = cuts.get("fps")
    legacy = legacy_grid is not None
    if legacy:
        grid = record(legacy_grid)
        fps = grid.get("frameRate")
        evidence = config.get("evidence")
        if (
            set(grid) != {"schema", "frameRate"}
            or grid.get("schema") != "protected-frame-plan-completed-selection"
            or not isinstance(cuts.get("grid"), str)
            or not isinstance(cuts.get("audioSHA256"), str)
            or not isinstance(cuts.get("inputs"), dict)
            or "fps" in cuts
            or "clock" in cuts
            or not isinstance(evidence, list)
            or not any(_matches(row, bindings["cuts"]) for row in evidence)
        ):
            raise ReplayError("Legacy cuts require an explicit completed-selection binding")
    if (
        config.get("coordinateSpace") != "native-presentation-seconds"
        or not _matches(config.get("source"), bindings["inputAudio"])
        or not _matches(config.get("clock"), bindings["audioClock"])
        or type(fps) is not int
        or not 1 <= fps <= 120
        or (not legacy and cuts.get("clock") != "source-presentation")
    ):
        raise ReplayError("Tempo producer must bind native cuts and the original continuous PCM")
    rows = config.get("intervals")
    if not isinstance(rows, list) or len(rows) > 3000:
        raise ReplayError("Invalid bounded tempo selection")
    if legacy and any(
        record(record(row).get(boundary)).get("timescale") != fps
        for row in rows
        for boundary in ("start", "end")
    ):
        raise ReplayError("Explicit legacy grid differs from producer rational timescales")
    intervals = tuple(
        (_rational(record(row).get("start")), _rational(record(row).get("end"))) for row in rows
    )
    original = _cuts(cuts)
    expected = tuple(
        (Fraction(c.source_start_frame, fps), Fraction(c.source_end_frame, fps)) for c in original
    )
    cursor = 0
    for cut in original:
        if cut.output_start_frame != cursor:
            raise ReplayError("Tempo cut output positions must be contiguous from zero")
        cursor += cut.duration_frames
    if intervals != expected:
        raise ReplayError("Tempo selection differs from the requested cut revision")
    cut_parents = record(cuts.get("parentSHA256", {}))
    if not legacy and any(
        cut_parents.get(str(bindings[role][0])) != bindings[role][1]
        for role in ("inputAudio", "audioClock")
    ):
        raise ReplayError("Cut revision does not bind this separated audio and decoded clock")
    inputs = record(receipt.get("inputs"))
    required = ("producerConfig", "inputAudio", "audioClock", "cuts")
    if any(inputs.get(str(bindings[role][0])) != bindings[role][1] for role in required):
        raise ReplayError("Tempo receipt does not pin these exact input artifacts")
    rate, channels, samples = _wave(bindings["inputAudio"][0])
    out_rate, out_channels, out_samples = _wave(bindings["outputAudio"][0])
    bound = receipt.get("maximumRateChangeAllowed")
    if isinstance(bound, bool) or not isinstance(bound, (int, float)):
        raise ReplayError("Tempo receipt has no explicit rate-change policy")
    slots = plan_slots(
        clock, intervals, rate=rate, samples=samples, maximum_rate_change=float(bound)
    )
    measurements = receipt.get("cuts")
    if not isinstance(measurements, list) or len(measurements) != len(slots):
        raise ReplayError("Tempo receipt does not cover every cut")
    for slot, value in zip(slots, measurements, strict=True):
        row = record(value)
        if (
            row.get("source_start_sample") != slot.spans[0][0]
            or row.get("source_end_sample") != slot.spans[-1][1]
            or row.get("output_start_sample") != slot.output_start
            or row.get("output_end_sample") != slot.output_end
        ):
            raise ReplayError("Tempo sample placements differ from the decoded clock/cuts")
    if (
        receipt.get("kind") != "context-atempo-exact-cut-slot-candidate"
        or receipt.get("output") != str(bindings["outputAudio"][0])
        or receipt.get("outputSHA256") != bindings["outputAudio"][1]
        or receipt.get("sampleRate") != rate
        or receipt.get("channels") != channels
        or receipt.get("outputStartSample") != 0
        or receipt.get("outputEndSample") != slots[-1].output_end
        or receipt.get("outputSamples") != slots[-1].output_end
        or (out_rate, out_channels, out_samples) != (rate, channels, slots[-1].output_end)
    ):
        raise ReplayError("Edited PCM output does not match its full clock receipt")
    return {
        "inputClock": "continuous-decoded-pcm",
        "outputClock": "edited-cut-pcm",
        "inputOrigin": "0",
        "outputOrigin": "0",
        "sampleRate": rate,
        "channels": channels,
        "mapping": "decoded-frame-and-local-tempo",
        "fps": fps,
        "outputFrames": cursor,
        "outputSamples": out_samples,
        "cutApplications": 1,
        "cutIntervals": [asdict(cut) for cut in original],
        "cutsSHA256": bindings["cuts"][1],
        "cutClockBinding": "completed-selection-and-receipt" if legacy else "cut-parent-pins",
        "historicalCutAudioSHA256": cuts.get("audioSHA256") if legacy else None,
        "nextInput": {
            "path": str(bindings["outputAudio"][0]),
            "clock": "edited-cut-pcm",
            "receipt": str(bindings["producerReceipt"][0]),
            "consumers": ["replay.fcp_transformed_source voice/voice_receipt"],
            "forbidden": "Do not apply source cuts or audio-clock projection again",
        },
        "inspectionAnchors": [
            {
                "cutIndex": index,
                "sourceStart": str(slots[index].native_start),
                "sourceEnd": str(slots[index].native_end),
                "inputStartSample": slots[index].spans[0][0],
                "inputEndSample": slots[index].spans[-1][1],
                "outputStartSample": slots[index].output_start,
                "outputEndSample": slots[index].output_end,
            }
            for index in dict.fromkeys((0, len(slots) // 2, len(slots) - 1))
        ],
    }


def verify(config: Path, *, media_root: Path | None = None) -> dict[str, object]:
    """Recompute clock correspondence and rehash all artifacts before reuse."""
    config = _local(config)
    document = _read(config)
    operation = document.get("operation")
    refs = record(document.get("artifacts"))
    expected = RECOGNITION_ROLES if operation == "recognition-source" else TEMPO_ROLES
    target = "source-presentation" if operation == "recognition-source" else "edited-cut-pcm"
    if (
        type(document.get("version")) is not int
        or document.get("version") != 1
        or operation not in ("recognition-source", "tempo-cut-audio")
        or type(document.get("previousApplications")) is not int
        or document.get("previousApplications") != 0
        or set(refs) != expected
        or document.get("inputClock") != "continuous-decoded-pcm"
        or document.get("outputClock") != target
    ):
        raise ReplayError(
            f"Expected one explicit unapplied clock conversion contract at {config}; "
            f"received {document.get('inputClock')} -> {document.get('outputClock')}"
        )
    if media_root is not None and (
        not media_root.is_absolute()
        or media_root != media_root.resolve(strict=True)
        or not media_root.is_dir()
    ):
        raise ReplayError("Media root must be an explicit canonical absolute directory")
    bindings = {
        role: _artifact_binding(
            value, media_root=media_root if role in {"inputAudio", "outputAudio"} else None
        )
        for role, value in refs.items()
    }
    pins = {str(path): digest for path, digest in bindings.values()}
    pins[str(config)] = file_digest(config)
    pins[str(Path(__file__).resolve())] = file_digest(Path(__file__))
    for name in (
        "asr_clock_projection.py",
        "audio_clock.py",
        "clocked_cut_audio.py",
        "clocked_tempo_audio.py",
        "continuous_quiet_plan.py",
        "cut_audio_support.py",
    ):
        dependency = Path(__file__).with_name(name).resolve()
        pins[str(dependency)] = file_digest(dependency)
    snapshot = InputHashSnapshot.capture(
        (*bindings.values(), *((Path(path), digest) for path, digest in pins.items()))
    )
    clock = AudioFrameClock.from_probe(_json_artifact(bindings, "audioClock"))
    try:
        detail = (
            _recognition(bindings, clock)
            if operation == "recognition-source"
            else _tempo(bindings, clock, document.get("legacyCutBinding"))
        )
    except (ReplayError, ValueError) as error:
        source_role = "inputLabels" if operation == "recognition-source" else "inputAudio"
        output_role = "outputLabels" if operation == "recognition-source" else "outputAudio"
        raise ReplayError(
            f"Clock contract {config}: continuous-decoded-pcm {bindings[source_role][0]} "
            f"-> {target} {bindings[output_role][0]} rejected: {error}"
        ) from error
    snapshot.finish()
    return {
        "version": 1,
        "kind": "verified-clock-contract",
        "operation": operation,
        "applicationCount": 1,
        "authorizedReadOnlyMediaRoot": str(media_root) if media_root is not None else None,
        "legacyCutBinding": document.get("legacyCutBinding"),
        "inputs": pins,
        **detail,
        "sourceAudioFirstTime": str(clock.pts[0] * clock.time_base),
        "sourceAudioLastTime": str(
            clock.pts[-1] * clock.time_base + Fraction(clock.samples[-1], clock.sample_rate)
        ),
        "reuse": "same-config-and-pins-after-fresh-domain-verification",
        "clockCorrespondenceVerified": True,
        "AVSyncVerified": False,
        "speechQualityVerified": False,
        "deliveryComplete": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--media-root", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--check", type=Path)
    group.add_argument("--inspect", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.config, media_root=args.media_root)
    if args.inspect:
        print(
            json.dumps(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"inputs", "cutIntervals"}
                },
                indent=2,
                allow_nan=False,
            )
        )
    elif args.check is not None:
        if _read(_local(args.check)) != result:
            raise ReplayError(
                "Clock receipt cannot be reused for this configuration or current inputs"
            )
    else:
        output = _local(args.output, output=True)
        with output.open("x", encoding="utf-8") as stream:
            os.fchmod(stream.fileno(), 0o600)
            json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
