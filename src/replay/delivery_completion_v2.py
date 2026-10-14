"""Fail-closed completion for direct Compressor encoding followed by packet remux.

Receipts are local evidence, not signed attestations. This reader never measures
media or turns missing producer observations into approvals. Viewer and summaries
remain separate delivery requirements.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from replay import caption_resource_inputs
from replay import native_completion as v1
from replay.bgm_separation import MODEL_SAMPLE_RATE, SeparationPolicy
from replay.compressor_batch import JOB_ID
from replay.core import ReplayError
from replay.delivery_output_names import recorded_stem_matches
from replay.edit_completion import REQUIRED_STAGES, load_config
from replay.recordings import record
from replay.subtitle_retry import RepairEvidence, RepairMethod, decide_repair

REFERENCES = (
    "renderConfig",
    "technical",
    "remux",
    "audioQuality",
    "maskQuality",
    "captionAudit",
    "cutExecution",
    "fcp",
)
ERRORS = (ReplayError, ValueError, OSError, ZeroDivisionError)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def path_value(value: object) -> Path:
    require(isinstance(value, str) and bool(value), "Missing artifact path")
    path = Path(str(value))
    require(
        path.is_absolute() and path == path.resolve(), "Artifact path must be absolute and unlinked"
    )
    return path


@dataclass
class Evidence(v1.Evidence):
    """Add only the explicitly bound recording's sibling render directory."""

    artifact_root: Path | None = None
    caption_tools: set[Path] = field(default_factory=set)
    caption_resources: dict[Path, str] = field(default_factory=dict)

    def digest(self, path: Path) -> str:
        resolved = path.resolve(strict=True)
        require(path.absolute() == resolved, "Linked evidence path is not permitted")
        if resolved in self.caption_tools or resolved in self.caption_resources:
            scoped = v1.Evidence(self.root, resolved, None, self.hashes, self.snapshots)
            digest = scoped.digest(path)
            require(
                resolved not in self.caption_resources
                or self.caption_resources[resolved] == digest,
                "Caption resource fingerprint changed",
            )
            return digest
        if self.artifact_root is not None and resolved.is_relative_to(self.artifact_root):
            scoped = v1.Evidence(
                self.root, self.source, self.artifact_root, self.hashes, self.snapshots
            )
            return scoped.digest(path)
        return super().digest(path)

    def reference(self, descriptor: object) -> tuple[Path, dict[str, object]]:
        ref = record(descriptor)
        require(set(ref) == {"path", "sha256"}, "Expected path/sha256 receipt reference")
        path = path_value(ref.get("path"))
        require(self.writable(path), "Receipt must be repository-local or in recording storage")
        require(self.digest(path) == ref.get("sha256"), "Receipt SHA256 changed")
        return path, self.read(path)

    def writable(self, path: Path) -> bool:
        return path.is_relative_to(self.root.parents[1]) or (
            self.storage is not None and path.is_relative_to(self.storage)
        )

    def inputs(self, data: dict[str, object], *required: Path) -> None:
        rows = v1._rows(data.get("inputs"))
        found: set[Path] = set()
        for raw in rows:
            item = record(raw)
            path = path_value(item.get("path"))
            require(path not in found, "Duplicate input binding")
            require(
                self.digest(path) == item.get("sha256")
                and type(item.get("bytes")) is int
                and item["bytes"] == path.stat().st_size,
                "Processing input changed",
            )
            found.add(path)
        require(
            all(path.resolve() in found for path in required),
            "Receipt omits required input binding",
        )
        if "inputsAfter" in data:
            require(data["inputsAfter"] == rows, "Processing inputs mutated")


def _configure(store: Evidence, data: dict[str, object]) -> tuple[Path, dict[str, object]]:
    path, render = store.reference(data.get("renderConfig"))
    require(
        render.get("recording") == store.root.name
        and path_value(render.get("source")) == store.source,
        "Render config recording/source mismatch",
    )
    artifact = path_value(render.get("storage"))
    if not store.writable(artifact):
        require(store.storage is not None, "External render storage requires recording storage")
        expected = (store.storage or store.root).parent / "work" / store.root.name / "render"
        require(
            artifact == expected and artifact.is_dir(),
            "Render storage is not this recording's exact sibling render root",
        )
        store.artifact_root = artifact
    return path, render


def _flags(data: dict[str, object], *keys: str) -> None:
    require(
        all(data.get(key) is True for key in keys),
        "Missing or unverified checks: " + ", ".join(keys),
    )


def _quality(
    store: Evidence, data: dict[str, object], kind: str, *paths: Path
) -> dict[str, object]:
    require(
        type(data.get("version")) is int and data["version"] == 1 and data.get("kind") == kind,
        "Unknown quality receipt schema",
    )
    store.inputs(data, *paths)
    _flags(data, "accepted")
    require(isinstance(data.get("limitations"), list), "Missing quality limitations")
    checks = record(data.get("checks"))
    require(
        bool(checks) and all(value is True for value in checks.values()),
        "Quality checks are incomplete",
    )
    return record(data.get("measurements"))


def _technical(
    store: Evidence,
    data: dict[str, object],
    *,
    render_path: Path,
    render: dict[str, object],
    movie: Path,
) -> int:
    store.inputs(data, render_path, movie, path_value(render.get("cuts")))
    _flags(data, "technicalAccepted", "fullDecode")
    expected = record(data.get("expected"))
    frames = v1._integer(expected.get("frames"))
    require(
        frames > 0
        and all(expected.get(key) == render.get(key) for key in ("rate", "width", "height")),
        "Technical expected clocks/canvas differ",
    )
    clock = record(data.get("videoClock"))
    require(
        clock.get("frames") == frames and clock.get("exactCFR") is True,
        "Final video clock is unverified",
    )
    return frames


def _remux(
    store: Evidence,
    data: dict[str, object],
    *,
    render_path: Path,
    render: dict[str, object],
    movie: Path,
) -> None:
    work = path_value(render.get("work"))
    jobs_path, state_path = work / "final-job.json", work / "final-job-state" / "final.json"
    compressor = path_value(data.get("source"))
    require(path_value(data.get("candidate")) == movie, "Remux candidate identity mismatch")
    store.inputs(
        data,
        render_path,
        movie,
        compressor,
        jobs_path,
        state_path,
        path_value(render.get("storage")) / "av-master.mov",
    )
    recording = render.get("recording")
    require(isinstance(recording, str), "Missing render recording identity")
    require(
        compressor != movie
        and compressor.parent == path_value(render.get("storage")) / "final"
        and recorded_stem_matches(
            compressor, path_value(render.get("source")), str(recording), historical=True
        ),
        "Remux source must be actual Compressor output with pinned recording identity, "
        "not PCM master",
    )
    _flags(data, "remuxIdentityAccepted")
    state = store.read(state_path)
    job_id = state.get("jobID")
    require(
        isinstance(job_id, str)
        and JOB_ID.fullmatch(job_id) is not None
        and state.get("status") == data.get("nativeStatus") == "Successful"
        and job_id == data.get("nativeJobID")
        and state.get("output") == str(compressor)
        and type(state.get("bytes")) is int
        and state["bytes"] == compressor.stat().st_size
        and state.get("retry_count", 0) == 0
        and "requested_output" not in state,
        "Direct CLI state/output identity mismatch",
    )
    jobs = v1._rows(store.read(jobs_path).get("jobs"))
    require(
        jobs
        == [
            {
                "name": "final",
                "source": str(path_value(render.get("storage")) / "av-master.mov"),
                "preset": render.get("finalPreset"),
                "output": str(compressor),
            }
        ],
        "Direct CLI job does not match render configuration",
    )
    streams = [record(item) for item in v1._rows(data.get("streams"))]
    require(
        len(streams) == 2 and {item.get("type") for item in streams} == {"video", "audio"},
        "Remux must verify both video and audio",
    )
    for stream in streams:
        _flags(stream, "orderedPayloadsEqual", "rationalClocksEqual")
        require(v1._integer(stream.get("packets")) > 0, "Missing remux packets")


def _cuts(
    store: Evidence,
    data: dict[str, object],
    *,
    render_path: Path,
    render: dict[str, object],
    movie: Path,
    frames: int,
) -> None:
    store.inputs(data, render_path, store.source, movie, path_value(render.get("cuts")))
    _flags(data, "selectionPlanMatchesProtectedCuts", "recordedExecutionMatchesPlan")
    require(
        v1._integer(data.get("selectedFrames")) == frames
        and v1._integer(data.get("sourceFrames")) - frames
        == v1._integer(data.get("removedFrames"))
        > 0,
        "Cut execution accounting differs",
    )


def _audio(
    store: Evidence,
    data: dict[str, object],
    *,
    render_path: Path,
    render: dict[str, object],
    movie: Path,
    frames: int,
) -> None:
    measured = _quality(
        store,
        data,
        "delivery-audio-quality",
        render_path,
        movie,
        path_value(record(render.get("audio")).get("source")),
    )
    _flags(
        record(data.get("checks")),
        "selectedPCMReceiptBound",
        "nativeTimesPreserved",
        "aacSampleTimeMapping",
        "speechGatePassed",
        "musicGatePassed",
        "inputsStable",
    )
    mapping, aac = record(measured.get("mapping")), record(measured.get("aac"))
    total = v1._integer(mapping.get("selectedSamples"))
    require(
        total == Fraction(frames * MODEL_SAMPLE_RATE, v1._integer(render.get("rate")))
        and aac.get("decodedSamples") == total
        and aac.get("audioPCMIdentical") is False,
        "AAC/native extent differs or lossy AAC claimed PCM-identical",
    )
    _flags(mapping, "nativeTimesPreserved")
    require(mapping.get("maximumAbsoluteDrift") == "0", "Per-cut native audio drift remains")
    quality = record(measured.get("quality"))
    policy = SeparationPolicy()
    require(
        quality.get("policy") == asdict(policy)
        and quality.get("minimumSpeechCorrelation") == v1.MINIMUM_CORRELATION,
        "Audio policy differs from v1",
    )
    _flags(quality, "speechGatePassed", "musicGatePassed")
    cursor = protected = 0
    for raw in v1._rows(quality.get("windows")):
        window = record(raw)
        end = min(cursor + 60 * MODEL_SAMPLE_RATE, total)
        require(
            window.get("startSample") == cursor and window.get("endSample") == end and end > cursor,
            "Music windows omit output coverage",
        )
        count = v1._integer(window.get("protectedSamples"))
        require(0 <= count <= end - cursor, "Invalid protected speech coverage")
        if count:
            _flags(window, "speechGatePassed")
            require(
                v1._speech(window.get("speech"), policy),
                "Post-AAC speech fails unchanged v1 thresholds",
            )
        before, after = (
            v1._number(window.get("beforeInstrumentalDBFS")),
            v1._number(window.get("afterInstrumentalDBFS")),
        )
        _flags(record(window.get("music")), "passed")
        require(
            before <= policy.quiet_instrumental_db
            or before - after >= policy.minimum_instrumental_drop_db,
            "Post-AAC music fails unchanged v1 threshold",
        )
        protected += count
        cursor = end
    require(
        cursor == total and protected == quality.get("protectedSamples") and protected > 0,
        "Audio quality omits output/protected speech",
    )


def _mask(store: Evidence, data: dict[str, object], *, render_path: Path, frames: int) -> None:
    measured = _quality(store, data, "delivery-mask-quality", render_path, store.source)
    _flags(
        record(data.get("checks")),
        "recordedMaskExecutionMatchesPlan",
        "executionRowsMatchProtectedCuts",
        "actualCompositorProvenanceVerified",
        "selectedEmptyZero",
        "qualityComplete",
        "losslessEvidenceVerified",
    )
    require(
        v1._integer(measured.get("selectedFrames")) == frames
        and v1._integer(measured.get("selectedMaskFrames"))
        == v1._integer(measured.get("measuredFrames"))
        > 0,
        "Mask measurements omit selected frames",
    )
    require(
        all(
            type(measured.get(key)) is int and measured[key] == 0
            for key in (
                "missingMaskFrames",
                "extraMaskFrames",
                "selectedEmpty",
                "technicallyInvalidFrames",
                "readableFrames",
            )
        ),
        "Mask execution/readability coverage failed",
    )
    evidence = RepairEvidence(
        technically_valid=True,
        residual_fraction=v1._number(measured.get("maximumResidualFraction")),
        unstable_fraction=v1._number(measured.get("maximumUnstableFraction")),
        readable_text=False,
    )
    require(
        decide_repair(RepairMethod.GLYPH, evidence).accepted,
        "Mask quality fails unchanged v1 thresholds",
    )


def validate_native_measurement(
    kind: str,
    *,
    store: Evidence,
    receipt: dict[str, object],
    descriptor: Path,
    context: dict[str, object],
    movie: Path,
) -> None:
    """Reuse numeric v2 measurements with an already validated native context.

    This does not admit a native descriptor or invent a render producer. Receipt
    inputs must explicitly bind that native descriptor, actual source and movie.
    Alpha-specific remux/paint contracts are deliberately not dispatched here.
    """
    require(
        kind in {"cutExecution", "audioQuality", "maskQuality"},
        "Unsupported native measurement kind",
    )
    frames = v1._integer(context.get("frames"))
    require(frames > 0, "Native measurement requires a positive edited extent")
    render: dict[str, object] = {
        "cuts": record(context.get("cuts", {})).get("path"),
        "audio": {"source": record(context.get("voice", {})).get("path")},
        "rate": context.get("fps"),
    }
    if kind == "cutExecution":
        _cuts(store, receipt, render_path=descriptor, render=render, movie=movie, frames=frames)
    elif kind == "audioQuality":
        _audio(store, receipt, render_path=descriptor, render=render, movie=movie, frames=frames)
    else:
        _mask(store, receipt, render_path=descriptor, frames=frames)


def _caption_tools(store: Evidence, data: dict[str, object]) -> None:
    """Authorize only named executable pins, never external media/resource directories."""
    tools = record(data.get("measurementTools", {}))
    require(set(tools) <= {"ffmpeg", "ffprobe"}, "Unknown caption measurement tool")
    pins: list[tuple[str, dict[str, object]]] = [(name, record(pin)) for name, pin in tools.items()]
    for raw in v1._rows(record(data.get("measurements")).get("parts")):
        part = record(raw)
        if "producerState" not in part:
            continue
        reference = record(part["producerState"])
        path = path_value(reference.get("path"))
        require(store.digest(path) == reference.get("sha256"), "Caption producer state changed")
        state = store.read(path)
        runtime = v1._rows(state.get("runtimeInputs"))
        require(
            len(runtime) == 6
            and state.get("submissionVersion") == 2
            and state.get("status") == "Successful"
            and state.get("jobID") == part.get("jobID")
            and state.get("terminalInputsVerified") is True
            and state.get("terminalRuntimeVerified") is True,
            "Caption runtime pins incomplete",
        )
        command = state.get("command")
        require(
            isinstance(command, list) and command[:1] == [record(runtime[0]).get("path")],
            "Caption runtime executable differs from submit command",
        )
        pins.extend((("Compressor", record(runtime[0])), ("python", record(runtime[1]))))
    for name, pin in pins:
        tool = path_value(pin.get("path"))
        require(
            (
                tool == Path(sys.executable).resolve(strict=True)
                if name == "python"
                else tool.name == name
            )
            and "private" not in tool.parts
            and tool.is_file()
            and os.access(tool, os.X_OK),
            "Caption tool is not the named regular executable",
        )
        store.caption_tools.add(tool)
        require(
            store.digest(tool) == pin.get("sha256") and tool.stat().st_size == pin.get("bytes"),
            "Caption executable fingerprint changed",
        )


def _captions(
    store: Evidence, data: dict[str, object], *, render_path: Path, movie: Path, frames: int
) -> None:
    # Data authorization lasts only for the caption domain, including on failure.
    try:
        motion = record(store.read(render_path).get("motion", {}))
        if "resourceEvidence" in motion:
            for raw in v1._rows(record(data.get("measurements")).get("parts")):
                part = record(raw)
                state_ref = record(part.get("producerState"))
                state_path = path_value(state_ref.get("path"))
                require(
                    store.digest(state_path) == state_ref.get("sha256"), "Caption state changed"
                )
                resources = caption_resource_inputs.validate(
                    config=render_path,
                    plan=path_value(record(part.get("plan")).get("path")),
                    source=path_value(record(part.get("source")).get("path")),
                    state=store.read(state_path),
                )
                store.caption_resources.update(resources.pins)
        _caption_quality(store, data, render_path=render_path, movie=movie, frames=frames)
    finally:
        store.caption_resources.clear()


def _caption_quality(
    store: Evidence, data: dict[str, object], *, render_path: Path, movie: Path, frames: int
) -> None:
    _caption_tools(store, data)
    measured = _quality(store, data, "delivery-caption-audit", render_path, movie)
    _flags(
        record(data.get("checks")),
        "measuredPlanCorrespondence",
        "orderedFrameTiming",
        "canonicalSRTFrameEquivalent",
        "candidateSRTExactMatch",
        "fullAlphaCoverage",
        "boundsAndGapsVerified",
        "historicalSubmissionBound",
        "recognitionCutProjectionVerified",
        "canonicalSRTActualDisplayVerified",
        "historicalToolCodeBound",
        "finalCompositionVerified",
    )
    require(
        v1._integer(measured.get("wholeVideoFrames")) == frames
        and v1._integer(measured.get("expectedAlphaParts"))
        == v1._integer(measured.get("auditedAlphaParts"))
        > 0,
        "Caption audit omits alpha parts",
    )
    parts = v1._rows(measured.get("parts"))
    require(len(parts) == measured["expectedAlphaParts"], "Caption part list incomplete")
    for raw in parts:
        part = record(raw)
        require(part.get("backend") == "direct-compressor-cli", "Not a direct CLI caption receipt")
        _flags(part, "historicalSubmissionBound")
        checks = record(part.get("checks"))
        _flags(
            checks,
            "boundsAndGapsVerified",
            "gapsVerified",
            "visibleStatesVerified",
            "fullStateConstraintCoverage",
        )
        require(
            v1._integer(checks.get("fullCaptionStates")) > 0
            and v1._integer(checks.get("sampleCount"))
            == v1._integer(checks.get("constraintFrames"))
            > 0
            and v1._integer(checks.get("measuredFrames")) == v1._integer(part.get("frames")) > 0,
            "Caption audit omits full states",
        )
    canonical = record(measured.get("canonicalSRT"))
    path = path_value(canonical.get("path"))
    srt = movie.with_suffix(".srt")
    store.inputs(data, path, srt)
    require(
        store.digest(path) == canonical.get("sha256") == store.digest(srt),
        "Movie sidecar differs from frame-derived canonical SRT",
    )


def _fcp(store: Evidence, data: dict[str, object], *, render_path: Path, movie: Path) -> None:
    xml = path_value(data.get("xml"))
    store.inputs(data, render_path, movie, xml)
    _flags(data, "importObserved", "noModalErrorObserved")
    require(store.digest(xml) == data.get("xmlSHA256"), "Observed FCP XML changed")


def assess(
    index: Path, *, delivery: Path, root: Path, source: Path, storage: Path | None = None
) -> dict[str, str]:
    """Return concrete stage failures; malformed shared provenance fails all stages."""
    store = Evidence(root.resolve(), source.resolve(strict=True), storage)
    data = store.read(index)
    require(
        type(data.get("version")) is int
        and data["version"] == 2
        and data.get("backend") == "motion-compressor-direct"
        and data.get("recording") == root.name
        and data.get("configurationSHA256") == store.digest(root / "edit-config.json")
        and data.get("sourceSHA256") == store.digest(source)
        and data.get("finalMovieSHA256") == store.digest(delivery),
        "V2 index does not bind current recording/config/source/final movie",
    )
    render_path, render = _configure(store, data)
    movie = delivery.resolve(strict=True)
    failures: dict[str, str] = {}
    frames = 0
    for key in ("technical", "remux"):
        try:
            receipt = store.reference(data.get(key))[1]
            if key == "technical":
                frames = _technical(
                    store, receipt, render_path=render_path, render=render, movie=movie
                )
            else:
                _remux(store, receipt, render_path=render_path, render=render, movie=movie)
        except ERRORS as exc:
            failures.update(dict.fromkeys(REQUIRED_STAGES, f"{key}: {exc}"))
    for key, stages in (
        ("cutExecution", ("silence_cut",)),
        ("audioQuality", ("bgm_removal",)),
        ("maskQuality", ("source_caption_mask",)),
        ("captionAudit", ("transcription", "dynamic_captions")),
        ("fcp", ("dynamic_captions",)),
    ):
        try:
            receipt = store.reference(data.get(key))[1]
            if key == "cutExecution":
                _cuts(
                    store,
                    receipt,
                    render_path=render_path,
                    render=render,
                    movie=movie,
                    frames=frames,
                )
            elif key == "audioQuality":
                _audio(
                    store,
                    receipt,
                    render_path=render_path,
                    render=render,
                    movie=movie,
                    frames=frames,
                )
            elif key == "maskQuality":
                _mask(store, receipt, render_path=render_path, frames=frames)
            elif key == "captionAudit":
                _captions(store, receipt, render_path=render_path, movie=movie, frames=frames)
            else:
                _fcp(store, receipt, render_path=render_path, movie=movie)
        except ERRORS as exc:
            for stage in stages:
                failures[stage] = failures.get(stage, "") + f" {key}: {exc}"
    store.verify_unchanged()
    return failures


def assemble(root: Path, movie: Path, output: Path, references: dict[str, Path]) -> Path:
    """Write a new index, including incomplete shells; assembly is not acceptance."""
    config = load_config(root)
    store = Evidence(root.resolve(), config.source, config.storage_root)
    output = path_value(str(output.absolute()))
    require(
        store.writable(output) and not output.is_relative_to(root.parents[1] / "private"),
        "Index destination is outside recording/repository storage",
    )
    require(
        set(references) <= set(REFERENCES) and "renderConfig" in references,
        "Unknown or missing renderConfig reference",
    )
    data: dict[str, object] = {
        "version": 2,
        "backend": "motion-compressor-direct",
        "recording": root.name,
        "configurationSHA256": store.digest(root / "edit-config.json"),
        "sourceSHA256": store.digest(config.source),
        "finalMovieSHA256": store.digest(movie),
    }
    for key, path in references.items():
        require(
            store.writable(path.resolve()),
            "Receipt destination is outside recording/repository storage",
        )
        data[key] = {"path": str(path.absolute()), "sha256": store.digest(path)}
    _configure(store, data)
    store.verify_unchanged()
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent) as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        os.chmod(stream.name, 0o600)
        os.link(stream.name, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("assemble", "check"))
    parser.add_argument("root", type=Path)
    parser.add_argument("movie", type=Path)
    parser.add_argument("index", type=Path)
    for key in REFERENCES:
        parser.add_argument("--" + key, type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.root)
        if args.operation == "assemble":
            refs = {
                key: value for key in REFERENCES if isinstance(value := getattr(args, key), Path)
            }
            assemble(args.root, args.movie, args.index, refs)
        failures = assess(
            args.index,
            delivery=args.movie,
            root=args.root,
            source=config.source,
            storage=config.storage_root,
        )
        print(
            json.dumps(
                {
                    "failures": failures,
                    "accepted": not failures,
                    "viewer": "not assessed",
                    "summaries": "not assessed",
                }
            )
        )
        return int(bool(failures))
    except ERRORS as exc:
        print(json.dumps({"accepted": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
