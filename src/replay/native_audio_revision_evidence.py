"""Bind recorded exact-audio revisions without changing native producer history.

This metadata consumer establishes numerical evidence, not fresh media identity,
native publication, original model lineage, caption paint, or delivery approval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

from replay import native_completion as legacy
from replay import native_delivery_evidence as evidence
from replay.bgm_separation import MODEL_SAMPLE_RATE, SeparationPolicy
from replay.native_delivery_completion import STRICT_GAPS
from replay.stages import file_digest

KIND = "native-audio-revision-proof-request"
ROLES = {
    "descriptor",
    "fullAV",
    "composition",
    "quality",
    "qualityCheckpoint",
    "packets",
    "packetWorkflow",
    "packetCheckpoints",
    "comparisonCheckpoint",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def numerical_quality(value: object, total: int) -> dict[str, object]:
    """Apply the existing numerical thresholds to every contiguous output window."""
    data = evidence.mapping(value)
    policy = SeparationPolicy()
    require(
        data.get("policy") == asdict(policy)
        and data.get("minimumSpeechCorrelation") == legacy.MINIMUM_CORRELATION,
        "Numerical policy differs from the required gate",
    )
    require(
        data.get("speechGatePassed") is True
        and data.get("musicGatePassed") is True
        and legacy._speech(data.get("speech"), policy),
        "Full speech/music gate failed",
    )
    rows = data.get("windows")
    require(isinstance(rows, list) and 0 < len(rows) <= 360, "Invalid quality windows")
    if not isinstance(rows, list):
        raise ValueError("Invalid quality windows")
    cursor = 0
    for item in rows:
        row = evidence.mapping(item)
        end = min(cursor + 60 * MODEL_SAMPLE_RATE, total)
        require(
            cursor < end
            and row.get("startSample") == cursor
            and row.get("endSample") == end
            and row.get("protectedSamples") == end - cursor,
            "Quality must cover every selected sample without padding or gaps",
        )
        require(
            row.get("speechGatePassed") is True and legacy._speech(row.get("speech"), policy),
            "A protected window fails the original speech gate",
        )
        before = legacy._number(row.get("beforeInstrumentalDBFS"))
        after = legacy._number(row.get("afterInstrumentalDBFS"))
        require(
            evidence.mapping(row.get("music")).get("passed") is True
            and (
                before <= policy.quiet_instrumental_db
                or before - after >= policy.minimum_instrumental_drop_db
            ),
            "A window fails the original music gate",
        )
        cursor = end
    require(cursor == total == data.get("protectedSamples"), "Incomplete full quality coverage")
    return data


def packet_binding(
    data: dict[str, object],
    context: dict[str, object],
    candidate: dict[str, str],
    reader: evidence.MetadataReader,
) -> None:
    """Keep original packet producer commands and completed output identities."""
    packets = reader.read(data["packets"])
    require(
        packets.get("packets") == context.get("frames")
        and all(
            packets.get(key) is True
            for key in ("orderedPayloadsEqual", "rationalClocksEqual", "extradataAndGeometryEqual")
        ),
        "Video payload or clocks changed",
    )
    declared = reader.read(packets.get("request"))
    require(packets.get("evidence") == declared, "Packet comparison request differs")
    workflow = reader.read(data["packetWorkflow"])
    stages, checkpoints = workflow.get("stages"), data["packetCheckpoints"]
    require(isinstance(stages, list) and isinstance(checkpoints, list), "Missing packet stages")
    if not isinstance(stages, list) or not isinstance(checkpoints, list):
        raise ValueError("Missing packet stages")
    require(len(stages) == len(checkpoints) == 2, "Packet stage coverage differs")
    for stage_value, checkpoint_ref, movie, key in zip(
        stages,
        checkpoints,
        (evidence.reference(context.get("finalMovie")), candidate),
        ("sourceObservation", "candidateObservation"),
        strict=True,
    ):
        stage = evidence.mapping(stage_value)
        observation = evidence.reference(declared.get(key))
        command = stage.get("command")
        require(isinstance(command, list), "Missing packet command")
        if not isinstance(command, list):
            raise ValueError("Missing packet command")
        require(
            command.count("-i") == 1
            and command[command.index("-i") + 1 :]
            == [
                movie["path"],
                "-map",
                "0:v:0",
                "-c:v",
                "copy",
                "-threads",
                "1",
                "-hash",
                "sha256",
                "-f",
                "framehash",
                observation["path"],
            ]
            and movie["path"] in evidence.strings(stage.get("inputs"))
            and stage.get("outputs") == [observation["path"]],
            "Original packet producer movie/command differs",
        )
        checkpoint = reader.read(checkpoint_ref)
        if checkpoint.get("status") == "complete":
            require(
                checkpoint.get("outputs") == {observation["path"]: observation["sha256"]},
                "Original packet output changed",
            )
        else:
            require(
                checkpoint.get("status") == "running"
                and key == "candidateObservation"
                and stage.get("verify_command")
                == [
                    "/usr/bin/cmp",
                    evidence.reference(declared.get("sourceObservation"))["path"],
                    observation["path"],
                ],
                "Unsupported incomplete packet producer",
            )
    comparison = reader.read(data["comparisonCheckpoint"])
    packet_ref = evidence.reference(data["packets"])
    require(
        comparison.get("status") == "complete"
        and comparison.get("outputs") == {packet_ref["path"]: packet_ref["sha256"]},
        "Rational packet comparison incomplete or changed",
    )


def assess(request: object, reader: evidence.MetadataReader) -> dict[str, object]:
    """Verify only pinned metadata, preserving original and derived identities."""
    data = reader.read(request)
    require(
        set(data) == ROLES | {"version", "kind"}
        and data.get("version") == 1
        and data.get("kind") == KIND,
        "Unknown revision proof request",
    )
    original = reader.read(data["descriptor"])
    context = evidence.mapping(original.get("context"))
    frames, fps = evidence.integer(context.get("frames")), evidence.integer(context.get("fps"))
    require(frames > 0 and fps > 0, "Invalid original frame clock")
    samples = Fraction(frames * MODEL_SAMPLE_RATE, fps)
    require(samples.denominator == 1, "Nonintegral original audio extent")
    composition = reader.read(data["composition"])
    require(
        composition.get("kind") == "video-composition-timing-observation",
        "Unknown composition proof",
    )
    candidate = evidence.reference(composition.get("candidate"))
    packet_binding(data, context, candidate, reader)
    aac = evidence.mapping(composition.get("aacAssessment"))
    require(
        aac.get("candidate") == candidate
        and aac.get("source") == context.get("voice")
        and all(
            aac.get(key) is True
            for key in (
                "originalPCMExtentAccepted",
                "packetDecodedPresentationAccepted",
                "trackEditMappingAccepted",
            )
        )
        and aac.get("primingAppliedAgain") is False
        and aac.get("decodedStart") == "0"
        and aac.get("decodedSamples") == samples.numerator
        and Fraction(evidence.string(aac.get("decodedEnd"))) == Fraction(frames, fps),
        "AAC, original PCM, or edit-list clocks differ",
    )
    technical = reader.read(data["fullAV"])
    require(
        technical.get("technicalAccepted") is True
        and technical.get("fullDecode") is True
        and technical.get("inputs") == technical.get("inputsAfter")
        and technical.get("audioEndMinusVideoEnd") == "0"
        and technical.get("audioSpanMinusVideoDuration") == "0",
        "Full AV evidence is incomplete or changed",
    )
    expected = evidence.mapping(technical.get("expected"))
    require(expected.get("frames") == frames and expected.get("rate") == fps, "Video clock differs")
    inputs = technical.get("inputs")
    require(isinstance(inputs, list), "Missing full AV input bindings")
    if not isinstance(inputs, list):
        raise ValueError("Missing full AV input bindings")
    movie_rows = [
        evidence.mapping(row)
        for row in inputs
        if evidence.mapping(row).get("path") == candidate["path"]
    ]
    require(
        len(movie_rows) == 1 and movie_rows[0].get("sha256") == candidate["sha256"],
        "Full AV movie differs",
    )
    quality = reader.read(data["quality"])
    require(
        quality.get("kind") == "native-tempo-full-audio-quality-observation"
        and quality.get("currentModelDevice") == "mps"
        and quality.get("protection") == "all-selected-samples-conservative"
        and quality.get("protectedSamples") == samples.numerator,
        "Unsupported quality observation",
    )
    quality_request = reader.read(quality.get("request"))
    require(
        quality_request.get("composition") == data["composition"], "Quality composition differs"
    )
    pins = evidence.mapping(quality.get("inputs"))
    require(
        evidence.mapping(pins.get(candidate["path"])).get("sha256") == candidate["sha256"],
        "Quality movie identity differs",
    )
    checkpoint = reader.read(data["qualityCheckpoint"])
    qref = evidence.reference(data["quality"])
    require(
        checkpoint.get("status") == "complete"
        and evidence.mapping(checkpoint.get("outputs")) == {qref["path"]: qref["sha256"]},
        "Quality producer is incomplete or output changed",
    )
    numerical = numerical_quality(quality.get("measurements"), samples.numerator)
    failures = {stage: list(gaps) for stage, gaps in STRICT_GAPS.items()}
    failures["bgm_removal"] = ["separation model/source lineage"]
    return {
        "version": 1,
        "kind": "native-audio-revision-numerical-proof",
        "request": evidence.reference(request),
        "originalDescriptor": data["descriptor"],
        "originalMovie": context.get("finalMovie"),
        "candidate": candidate,
        "frames": frames,
        "sampleRate": MODEL_SAMPLE_RATE,
        "samples": samples.numerator,
        "numericalAudioMeasurementVerified": True,
        "numericalMeasurement": numerical,
        "fullAVRecordedVerified": True,
        "originalVideoPacketBindingVerified": True,
        "originalPacketStageCompletionPromoted": False,
        "exactAACExtentRecordedVerified": True,
        "currentMediaCryptoVerified": False,
        "recordedStagePromotion": False,
        "originalModelRuntimeVerified": False,
        "ASRProtectionAccuracyVerified": False,
        "CPUExactDeviceEquivalenceVerified": False,
        "strictFailures": failures,
        "nativeDerivedRevisionBinding": "unsupported: original descriptor final movie is unchanged",
        "qualityApproved": False,
        "syncApproved": False,
        "strictComplete": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = args.output.absolute()
    require(output.parent.resolve(strict=True) == output.parent, "Linked output directory")
    require(output.is_relative_to(Path.cwd() / "edits/work"), "Output must be repository-local")
    result = assess(
        {"path": str(args.request.absolute()), "sha256": args.sha256}, evidence.MetadataReader()
    )
    if args.verify:
        saved = evidence.MetadataReader().read({"path": str(output), "sha256": file_digest(output)})
        require(saved == result, "Revision proof changed")
        return
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
