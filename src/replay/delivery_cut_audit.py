"""Read-only cut execution audit. No decoding, native calls or inferred approvals.

sourceFrames means fully supported nominal source-grid cells (floor(end * fps)),
NOT decoded frames. A fractional terminal exposure is reported separately, never
extended, clamped or labelled removed silence. SHA256s measured here are audit-time
bindings; only retained stage fingerprints establish recorded execution inputs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from pathlib import Path

from replay import (
    compressor_batch,
    mechanical_join,
    mechanical_plan_provenance,
    native_invocation,
)
from replay import delivery_completion_v2 as adapter
from replay import delivery_render as render
from replay.delivery_audio_mapping import cut_intervals, integer
from replay.delivery_output_names import recorded_stem_matches
from replay.delivery_render_resume import fingerprint
from replay.native_edit_parts import partition
from replay.recordings import record
from replay.stages import Stage, StageError, file_digest
from replay.workflow import parse_stages

ERRORS = (*adapter.ERRORS, StageError)
# Native EditPlan.timeScale; reject unrepresentable rational selections.
NATIVE_TIMESCALE = 60_000


@dataclass
class Bindings(adapter.Evidence):
    """Hash each file once per side of this audit; stat is only a mutation guard."""

    identities: dict[Path, tuple[int, ...]] = field(default_factory=dict)
    native_receipts: list[dict[str, object]] = field(default_factory=list)
    native_movies: list[Path] = field(default_factory=list)

    @staticmethod
    def stat_identity(path: Path) -> tuple[int, ...]:
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def digest(self, path: Path) -> str:
        adapter.require(path == path.resolve(), "Linked/noncanonical audit input")
        # Reuse the adapter's authorized recording/storage boundary checks.
        before = self.stat_identity(path)
        value = super().digest(path)
        identity = self.stat_identity(path)
        adapter.require(before == identity, "Input changed during hashing")
        adapter.require(
            self.identities.setdefault(path, identity) == identity, "Audit input mutated"
        )
        return value

    def rows(self) -> list[dict[str, object]]:
        return [
            {"path": str(path), "sha256": sha, "bytes": self.identities[path][2]}
            for path, sha in self.hashes.items()
            if path in self.identities
        ]

    def after(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for path in self.identities:
            sha = file_digest(path)
            adapter.require(self.stat_identity(path) == self.identities[path], "Input changed")
            rows.append({"path": str(path), "sha256": sha, "bytes": path.stat().st_size})
        adapter.require(rows == self.rows(), "Input SHA256 changed during audit")
        return rows

    def pin(self, path: Path, sha: object) -> Path:
        adapter.require(self.digest(path) == sha, f"SHA256 binding differs: {path}")
        return path

    def bound(self, data: dict[str, object], *paths: Path) -> None:
        """Accept existing mapping receipts as well as completion-v2 input rows."""
        inputs = data.get("inputs")
        if not isinstance(inputs, dict):
            self.inputs(data, *paths)
            return
        pins = record(inputs)
        for path in paths:
            self.pin(path, pins.get(str(path)))


@dataclass(frozen=True)
class EvidencePaths:
    source_clock: Path | None = None
    clock_receipt: Path | None = None
    assembly_receipt: Path | None = None
    protection: Path | None = None
    remux_receipt: Path | None = None


def _required(path: Path | None, name: str) -> Path:
    adapter.require(path is not None, f"Missing explicit {name}")
    return adapter.path_value(str(path))


def source_clock(store: Bindings, config: render.Config, paths: EvidencePaths) -> dict[str, object]:
    clock_path = _required(paths.source_clock, "--source-clock (existing decoded video clock)")
    receipt = store.read(_required(paths.clock_receipt, "--clock-receipt binding original + clock"))
    store.bound(receipt, config.source, clock_path)
    raw = store.read(clock_path)
    streams = render._list(raw.get("streams"))
    adapter.require(len(streams) == 1, "Expected one original video clock")
    base = record(streams[0]).get("time_base")
    adapter.require(isinstance(base, str), "Missing original rational time base")
    scale = Fraction(str(base))
    adapter.require(scale > 0, "Invalid original time base")
    frames = render._list(raw.get("frames"))
    adapter.require(0 < len(frames) <= 4_000_000, "Missing/bounded original frames")
    end = 0
    for value in frames:
        frame = record(value)
        pts, duration = integer(frame.get("pts")), integer(frame.get("duration"), minimum=1)
        adapter.require(pts == end, "Original source gap/overlap/nonzero origin is unsupported")
        end = pts + duration
    extent = end * scale
    slots = (extent * config.rate).__floor__()
    return {
        "sourceFrames": slots,
        "sourceGridSlots": slots,
        "sourceGridRounding": (
            "floor: zero-origin contiguous decoded support must cover the entire cell"
        ),
        "decodedSourceFrames": len(frames),
        "nativeSourceEnd": str(extent),
        "terminalPartialSeconds": str(extent - Fraction(slots, config.rate)),
        "sourceFramesKind": "fullySupportedNominalSourceGridSlots",
    }


def protection(store: Bindings, config: render.Config, paths: EvidencePaths, slots: int) -> None:
    """Require explicit intervals, not the old plan's unverified normalization flag.

    Protection evidence v1 records source-grid protectedSpeechWithASRGuard and
    removedNonSpeech intervals as [start, end] rows, plus inputs binding the cuts
    and their ASR/quiet evidence. This is evidence consumption, not ASR approval.
    """
    cuts = store.read(config.cuts)
    facts = store.read(_required(paths.protection, "--protection (speech/quiet interval evidence)"))
    adapter.require(
        facts.get("version") == 1
        and facts.get("coordinateSpace") == "source-grid-slots"
        and facts.get("rate") == config.rate,
        "Unsupported speech protection evidence contract",
    )
    plan_inputs = record(cuts.get("inputs"))
    adapter.require(bool(plan_inputs), "Cut plan has no ASR/quiet input bindings")
    pinned = [store.pin(adapter.path_value(name), sha) for name, sha in plan_inputs.items()]
    store.bound(facts, config.cuts, *pinned)
    intervals = cut_intervals(cuts, config.rate)
    selected = [(int(start * config.rate), int(end * config.rate)) for start, end in intervals]
    removed: list[list[int]] = []
    cursor = 0
    for start, end in selected:
        adapter.require(end <= slots, "Selection extends beyond fully supported source cells")
        if cursor < start:
            removed.append([cursor, start])
        cursor = end
    if cursor < slots:
        removed.append([cursor, slots])
    adapter.require(facts.get("removedNonSpeech") == removed, "Removed non-speech intervals differ")
    protected = render._list(facts.get("protectedSpeechWithASRGuard"))
    adapter.require(bool(protected), "Missing protected speech/ASR guard intervals")
    for value in protected:
        row = render._list(value)
        adapter.require(len(row) == 2, "Invalid protected interval")
        start, end = integer(row[0]), integer(row[1])
        adapter.require(
            start < end and any(left <= start < end <= right for left, right in selected),
            "Protected speech/ASR guard is not fully retained",
        )
    adapter.require(
        facts.get("allSpeechAndASRGuardsIncluded") is True
        and facts.get("removedIntervalsVerifiedNonSpeech") is True,
        "Speech/quiet evidence is incomplete",
    )


def selection_plan(store: Bindings, config: render.Config) -> list[tuple[Path, int]]:
    store.read(config.cuts)
    store.read(config.chunks)
    cut_intervals(store.read(config.cuts), config.rate)
    expected = render.clips(
        render.read_cuts(config.cuts), render.read_chunks(config.chunks), config.rate
    )
    adapter.require(
        all(
            (clip.start * NATIVE_TIMESCALE).denominator == 1
            and (Fraction(clip.frames, config.rate) * NATIVE_TIMESCALE).denominator == 1
            for clip in expected
        ),
        "Exact source selection is not representable on the native 60000-tick clock",
    )
    manifests = sorted(config.composition.glob("parts-[0-9][0-9][0-9][0-9]/manifest.json"))
    requests = sorted(config.composition.glob("request-[0-9][0-9][0-9][0-9].json"))
    adapter.require(
        bool(manifests) and len(manifests) == len(requests), "Missing partition inventory"
    )
    actual_clips: list[object] = []
    parts: list[tuple[Path, int]] = []
    for manifest_path, source in zip(manifests, requests, strict=True):
        manifest = store.read(manifest_path)
        reference = record(manifest.get("sourceRequest"))
        adapter.require(reference.get("path") == str(source), "Source request ordering differs")
        store.pin(source, reference.get("sha256"))
        request = store.read(source)
        recipe = record(request.get("recipe"))
        video = record(recipe.get("video"))
        adapter.require(
            video.get("frameRate") == config.rate
            and video.get("width") == config.width
            and video.get("height") == config.height
            and video.get("encoding") == config.encoding,
            "Native request canvas/rate/encoding differs from configuration",
        )
        actual_clips.extend(render._list(recipe.get("clips")))
        generated = partition(request)
        rows = render._list(manifest.get("parts"))
        adapter.require(len(rows) == len(generated), "Partition count differs")
        for value, (offset, frames, planned) in zip(rows, generated, strict=True):
            row = record(value)
            path = adapter.path_value(row.get("path"))
            adapter.require(path.is_relative_to(manifest_path.parent), "Part escapes manifest")
            store.pin(path, row.get("sha256"))
            actual = store.read(path)
            adapter.require(
                row.get("offsetFrame") == offset
                and row.get("frames") == frames
                and {**actual, "outputDirectory": planned["outputDirectory"]} == planned,
                "Actual partition changed source selection/order/coverage",
            )
            parts.append((path, frames))
    # Compare the producer's exact float encoding of known rational times. Never
    # Fraction(str(float)) or tolerance-based reconstruction of source boundaries.
    adapter.require(
        actual_clips == [render._clip_json(clip, config.rate) for clip in expected],
        "Native requests differ from exact rational cut-plan mapping",
    )
    return parts


def completed(store: Bindings, stage: Stage, plan: Path) -> dict[str, object]:
    saved = store.read(plan.with_name(f"{plan.stem}-state") / f"{stage.name}.json")
    inputs = {str(path): store.digest(path) for path in stage.inputs}
    outputs = {str(path): store.digest(path) for path in stage.outputs}
    adapter.require(
        saved
        == {"status": "complete", "fingerprint": fingerprint(stage, inputs), "outputs": outputs},
        f"Completed record input/output SHA256 differs: {stage.name}",
    )
    return saved


def execution(store: Bindings, config: render.Config, parts: list[tuple[Path, int]]) -> list[Path]:
    plans = sorted(config.composition.glob("render-[0-9]*-[0-9]*.json"))
    stages: dict[str, tuple[Stage, Path]] = {}
    for plan in plans:
        for stage in parse_stages(store.read(plan)):
            adapter.require(stage.name not in stages, "Duplicate render stage")
            stages[stage.name] = (stage, plan)
    adapter.require(len(stages) == len(parts) > 0, "Render workflow coverage differs")
    errors: list[str] = []
    for index, (request, frames) in enumerate(parts):
        entry = stages.get(f"render-{index:06d}")
        if entry is None:
            raise ValueError("Missing ordered render stage")
        stage, plan = entry
        try:
            native_execution(store, config, stage=stage, plan=plan, request=request, frames=frames)
        except ERRORS as error:
            errors.append(f"{stage.name}: {error}")
    adapter.require(not errors, "; ".join(errors))
    return store.native_movies


def native_execution(
    store: Bindings, config: render.Config, *, stage: Stage, plan: Path, request: Path, frames: int
) -> None:
    invocation, response, binding = native_invocation.paths(request)
    richer = stage.command == native_invocation.command(config.native, request, config.storage)
    if richer:
        adapter.require(
            stage.outputs == (invocation, response, binding)
            and stage.verify_command
            == native_invocation.command(config.native, request, config.storage, verify=True)
            and Path(native_invocation.__file__).resolve() in stage.inputs
            and not stage.adopt_existing,
            "Runtime native wrapper contract differs",
        )
    adapter.require(
        (
            richer
            or stage.command
            == (
                "/bin/bash",
                "-c",
                'exec "$1" edit-media "$2" > "$3"',
                "native-render",
                str(config.native),
                str(request),
                str(response),
            )
        )
        and request in stage.inputs
        and config.native in stage.inputs
        and response in stage.outputs,
        "Recorded native command/request/response differs",
    )
    saved = completed(store, stage, plan)
    native = store.read(response)
    movie = adapter.path_value(native.get("outputPath"))
    adapter.require(movie.is_relative_to(config.storage), "Native movie leaves configured storage")
    store.digest(movie)
    adapter.require(
        type(native.get("frameCount")) is int
        and native.get("frameCount") == frames
        and all(
            type(native.get(key)) in (int, float)
            and native.get(key) == float(Fraction(frames, config.rate))
            for key in ("expectedDurationSeconds", "actualDurationSeconds")
        ),
        "Actual native receipt frame count/duration differs",
    )
    recorded_sha = record(saved.get("outputs")).get(str(movie))
    if richer:
        # Only the producer's predeclared runtime sidecar can supply this binding.
        facts = native_invocation.verify(config.native, request, config.storage)
        store.read(invocation)
        store.read(binding)
        recorded_sha = record(facts.get("output")).get("sha256")
    store.native_movies.append(movie)
    store.native_receipts.append(
        {
            "stage": stage.name,
            "actualNativeReceipt": str(response),
            "receiptSHA256": store.digest(response),
            "movie": str(movie),
            "movieSHA256AtAudit": store.digest(movie),
            "bytesAtAudit": movie.stat().st_size,
            "movieSHA256Recorded": recorded_sha,
            "executionContract": "native-invocation-v1" if richer else "legacy-direct-native",
            "runtimeBinding": str(binding) if richer else None,
            "stageFingerprint": saved.get("fingerprint"),
        }
    )
    # The normal historical runner bound only the JSON response, not its movie.
    # A new digest cannot retroactively fill that missing execution observation.
    adapter.require(
        recorded_sha == store.digest(movie),
        f"Completed record lacks actual native movie SHA256: {stage.name}",
    )
    recipe = record(store.read(request).get("recipe"))
    sources = [
        adapter.path_value(record(item).get("sourcePath"))
        for item in (
            *render._list(recipe.get("clips")),
            *render._list(recipe.get("additionalVideo", [])),
        )
    ]
    adapter.require(
        all(path in stage.inputs for path in sources),
        "Native completion fingerprint omits source/overlay media hashes",
    )


def assembly(store: Bindings, config: render.Config, paths: EvidencePaths) -> None:
    facts = store.read(_required(paths.assembly_receipt, "--assembly-receipt"))
    adapter.require(
        facts.get("indexSHA256") == store.digest(config.chunks)
        and facts.get("exactPresentationClockVerified") is True
        and facts.get("compressedPacketsIdentical") is True,
        "Assembly index/clock/payload evidence differs",
    )
    chunks = render.read_chunks(config.chunks)
    adapter.require(len(chunks) == 1, "Explicit continuous source assembly required")
    output = record(facts.get("output"))
    adapter.require(output.get("path") == str(chunks[0].media), "Assembly output identity differs")
    store.pin(chunks[0].media, output.get("sha256"))
    probes = record(facts.get("probeEvidence"))
    adapter.require(bool(probes), "Assembly lacks retained probe evidence")
    receipt_path = _required(paths.assembly_receipt, "--assembly-receipt")
    frame_path, stream_path = (
        receipt_path.parent / "output-frames.json",
        receipt_path.parent / "output-streams.json",
    )
    store.pin(frame_path, probes.get(str(frame_path)))
    store.pin(stream_path, probes.get(str(stream_path)))
    original = store.read(_required(paths.source_clock, "--source-clock"))
    source_streams = render._list(original.get("streams"))
    output_streams = render._list(store.read(stream_path).get("streams"))
    adapter.require(len(source_streams) == len(output_streams) == 1, "Ambiguous assembly clocks")
    before_scale = Fraction(str(record(source_streams[0]).get("time_base")))
    after_scale = Fraction(str(record(output_streams[0]).get("time_base")))
    before = render._list(original.get("frames"))
    after = render._list(store.read(frame_path).get("frames"))
    adapter.require(len(before) == len(after) > 0, "Assembly decoded count differs from original")
    for a, b in zip(before, after, strict=True):
        left, right = record(a), record(b)
        adapter.require(
            all(
                integer(left.get(key)) * before_scale == integer(right.get(key)) * after_scale
                for key in ("pts", "duration")
            ),
            "Assembly clock differs from original source",
        )
    last = record(before[-1])
    end = (integer(last.get("pts")) + integer(last.get("duration"))) * before_scale
    adapter.require(
        chunks[0].start == 0 and chunks[0].end == end, "Assembly index native extent differs"
    )


def join_inputs(
    stage: Stage,
    base: tuple[Path, ...],
    *,
    tools: tuple[mechanical_plan_provenance.Pin, ...] | None,
) -> bool:
    """New sealed plans may add only the validated code, tools and consumer contract."""
    if tools is None:
        return stage.inputs == base
    expected = {
        *base,
        *mechanical_plan_provenance.code_paths(),
        *(item.path for item in tools),
        Path(stage.verify_command[3]),
    }
    return stage.inputs[: len(base)] == base and set(stage.inputs) == expected


def manifest_join(
    store: Bindings,
    config: render.Config,
    *,
    stage: Stage,
    tools: tuple[mechanical_plan_provenance.Pin, ...] | None,
) -> None:
    """Validate the original indirect media bindings without inventing stage inputs."""
    manifest = config.work / "join-manifest.json"
    joined = config.storage / "joined-video.mp4"
    adapter.require(
        stage.command
        == (
            sys.executable,
            "-m",
            "replay.mechanical_join",
            "join-manifest",
            str(manifest),
            "--output",
            str(joined),
            "--rate",
            str(config.rate),
        )
        and join_inputs(
            stage, (manifest, config.cuts, config.chunks, *config.cut_evidence), tools=tools
        )
        and stage.outputs == (joined,),
        "Recorded manifest join command/input/output differs",
    )
    parts = mechanical_join.parse_manifest(store.read(manifest))
    rendered = render.rendered_parts(config)
    adapter.require(
        len(parts) == len(rendered) == len(store.native_movies),
        "Manifest join part count differs from completed native inventory",
    )
    for part, (_, frames), movie in zip(parts, rendered, store.native_movies, strict=True):
        adapter.require(
            part.path == movie and part.frames == frames,
            "Manifest join part order/path/frame count differs",
        )
        store.pin(part.path, part.sha256)


def downstream(store: Bindings, config: render.Config, *, join: bool) -> None:
    joined, master = config.storage / "joined-video.mp4", config.storage / "av-master.mov"
    selected = config.work / "audio-selected/selected.wav"
    name = "join" if join else "final-mux"
    required = store.native_movies if join else [joined, selected]
    output = joined if join else master
    plan = config.work / f"{name}-workflow.json"
    stages = parse_stages(store.read(plan))
    adapter.require(len(stages) == 1 and stages[0].name == name, "Unexpected downstream stage")
    stage = stages[0]
    tools = None
    if join and stage.verify_command[1:3] == (
        "-m",
        mechanical_plan_provenance.VERIFY_MODULE,
    ):
        # Scope only the validator's exact Python/FFmpeg pins before completed()
        # hashes them. This is not media acceptance or historical pin invention.
        tools = mechanical_plan_provenance.validate_join_stage(stage)
        store.caption_resources.update({item.path: item.sha256 for item in tools})
    completed(store, stage, plan)
    if join and stage.command[1:4] == ("-m", "replay.mechanical_join", "join-manifest"):
        manifest_join(store, config, stage=stage, tools=tools)
        return
    adapter.require(
        bool(required)
        and all(path in stage.inputs for path in required)
        and stage.outputs == (output,),
        f"{name} completion does not bind every upstream media input",
    )
    if join:
        join_plan = config.work / "join-plan.json"
        if stage.command[1:4] == ("-m", "replay.delivery_render", "join-files"):
            adapter.require(
                stage.command[1:]
                == (
                    "-m",
                    "replay.delivery_render",
                    "join-files",
                    str(join_plan),
                    "--output",
                    str(joined),
                    "--rate",
                    str(config.rate),
                )
                and stage.inputs[: len(required) + 1] == (join_plan, *required)
                and (
                    tools is None
                    or join_inputs(
                        stage,
                        (join_plan, *required, config.cuts, config.chunks, *config.cut_evidence),
                        tools=tools,
                    )
                )
                and render._json(join_plan)
                == [
                    {"durationFrames": frames, "sourcePath": str(movie)}
                    for (_, frames), movie in zip(
                        render.rendered_parts(config), required, strict=True
                    )
                ],
                "Recorded ordered join media/plan/command differs",
            )
            return
        links = config.storage / "join-inputs"
        adapter.require(
            stage.command[1:]
            == (
                "-m",
                "replay.native_join",
                str(join_plan),
                str(links),
                str(joined),
                "--rate",
                str(config.rate),
            )
            and join_plan in stage.inputs,
            "Recorded join command/plan differs",
        )
        store.digest(join_plan)
        adapter.require(
            render._json(join_plan)
            == [{"durationFrames": frames} for _, frames in render.rendered_parts(config)],
            "Join plan order/durations differ",
        )
        adapter.require(
            sorted(links.glob("part-*.mp4"))
            == [links / f"part-{i:06d}.mp4" for i in range(len(required))],
            "Join input inventory differs",
        )
        for i, media in enumerate(required):
            alias = links / f"part-{i:06d}.mp4"
            adapter.require(
                alias in stage.inputs,
                "Join completion omits ordered input file bindings; "
                "current symlinks are not execution evidence",
            )
            adapter.require(store.digest(alias) == store.digest(media), "Join media order differs")
    else:
        adapter.require(
            stage.command
            == (
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-i",
                str(joined),
                "-i",
                str(selected),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                str(master),
            ),
            "Recorded mux command differs",
        )


def compressor_input(store: Bindings, config: render.Config, remux: dict[str, object]) -> None:
    jobs_path = config.work / "final-job.json"
    state_path = config.work / "final-job-state/final.json"
    store.inputs(remux, jobs_path, state_path)
    rows = render._list(store.read(jobs_path).get("jobs"))
    adapter.require(len(rows) == 1, "Exactly one historical Compressor job required")
    row = record(rows[0])
    output = adapter.path_value(row.get("output"))
    adapter.require(
        output.parent == config.storage / "final"
        and recorded_stem_matches(output, config.source, config.recording, historical=True),
        "Recorded Compressor output does not belong to this recording",
    )
    state = store.read(state_path)
    adapter.require(
        "inputs" in state,
        "Compressor state lacks recorded input SHA256; "
        "audit-time master hash is not submission evidence",
    )
    master = config.storage / "av-master.mov"
    store.bound(state, master)
    job = compressor_batch.Job("final", master, config.final_preset, output)
    adapter.require(
        row
        == {
            "name": job.name,
            "source": str(master),
            "preset": str(job.preset),
            "output": str(output),
        },
        "Historical Compressor job differs from render configuration",
    )
    requested = state.get("requested_output", str(job.output))
    job = compressor_batch.Job(job.name, job.source, job.preset, adapter.path_value(requested))
    adapter.require(
        compressor_batch.verify_submission(job, state, rehash=True)
        and state.get("terminalInputsVerified") is True,
        "Compressor receipt lacks historical pre-submit source/preset binding",
    )
    submit = state.get("submitResponse")
    monitor = record(state.get("monitorResponse"))
    adapter.require(isinstance(submit, str), "Missing original Compressor submit response")
    monitor_path = store.pin(adapter.path_value(monitor.get("path")), monitor.get("sha256"))
    adapter.require(
        compressor_batch.parse_submission(str(submit)) == state.get("jobID")
        and compressor_batch.parse_snapshot(monitor_path.read_text(encoding="utf-8")).get(
            str(state.get("jobID"))
        )
        == compressor_batch.DONE,
        "Compressor retained responses contradict terminal identity",
    )


def audit(
    config_path: Path,
    movie: Path,
    technical: Path,
    output: Path,
    *,
    paths: EvidencePaths | None = None,
) -> dict[str, object]:
    """Write one non-overwriting facts report, retaining independent failed checks."""
    paths = paths or EvidencePaths()
    output = adapter.path_value(str(output.absolute()))
    adapter.require(
        output.is_relative_to(Path.cwd().resolve()), "Report must remain repository-local"
    )
    if output.exists():
        raise FileExistsError(output)
    config = render.load_config(adapter.path_value(str(config_path.absolute())))
    adapter.require(
        config.viewer_root.parent == Path.cwd().resolve() / "edits",
        "Viewer root must be checkout-local",
    )
    store = Bindings(
        config.viewer_root, config.source, config.delivery_root, artifact_root=config.storage
    )
    checks: dict[str, bool] = {}
    failures: dict[str, str] = {}
    result: dict[str, object] = {
        "version": 1,
        "kind": "delivery-cut-execution",
        "accepted": False,
        "selectionPlanMatchesProtectedCuts": False,
        "recordedExecutionMatchesPlan": False,
        "selectedFrames": None,
        "sourceFrames": None,
        "removedFrames": None,
        "sourceFramesKind": "fullySupportedNominalSourceGridSlots",
        "policy": "exact-supported-source-cells/no-clamp/no-normalization-approval",
        "hashPolicy": "SHA256 content reads before and after; stat guards mutation only",
        "submissionCodeHashClaimed": False,
        "limitations": [
            "Audit-time hashes are not submission-time observations or signed attestations.",
            "No decode, encode, PCM, ASR, caption, Viewer or quality audit performed.",
            "sourceFrames counts full nominal grid cells, not decoded video frames.",
        ],
    }

    def check(name: str, action: Callable[[], object]) -> object | None:
        try:
            value = action()
            checks[name] = True
            return value
        except ERRORS as error:
            checks[name] = False
            failures[name] = str(error)
            return None

    for path in (config.path, config.cuts, config.source, movie):
        check(f"binding:{path.name}", partial(store.digest, path))
    raw_config = store.read(config.path)
    frames = check(
        "technical",
        lambda: adapter._technical(
            store, store.read(technical), render_path=config.path, render=raw_config, movie=movie
        ),
    )
    clock = check("originalSourceClock", lambda: source_clock(store, config, paths))
    if isinstance(clock, dict):
        result.update(record(clock))
    selected = check(
        "cutAccounting",
        lambda: sum(
            int((end - start) * config.rate)
            for start, end in cut_intervals(store.read(config.cuts), config.rate)
        ),
    )
    result["selectedFrames"] = selected
    result["planNormalizationVerified"] = check(
        "normalizationFlagObserved", lambda: store.read(config.cuts).get("normalizationVerified")
    )
    if type(selected) is int and type(result.get("sourceFrames")) is int:
        slots = integer(result["sourceFrames"])
        result["removedFrames"] = slots - selected
        check("protectedSpeechAndRemovedNonSpeech", lambda: protection(store, config, paths, slots))
        check(
            "sourceGridContract",
            lambda: adapter.require(
                integer(store.read(config.cuts).get("sourceFrames")) == slots
                and store.read(config.cuts).get(
                    "sourceFramesKind", "fullySupportedNominalSourceGridSlots"
                )
                == "fullySupportedNominalSourceGridSlots"
                and store.read(config.cuts).get(
                    "actualSourceFrames", result.get("decodedSourceFrames")
                )
                == result.get("decodedSourceFrames")
                and selected == frames
                and slots - selected > 0,
                "Plan/technical/source-grid accounting differs; no forced positive removal",
            ),
        )
    else:
        check(
            "protectedSpeechAndRemovedNonSpeech", lambda: _required(None, "supported source clock")
        )
    check("sourceAssembly", lambda: assembly(store, config, paths))
    planned: list[tuple[Path, int]] = []
    check("nativePartInventory", lambda: planned.extend(render.rendered_parts(config)))
    check(
        "nativeSelectionPlan",
        lambda: adapter.require(
            selection_plan(store, config) == planned, "Ordered part inventory differs"
        ),
    )
    check("completedNativeExecution", lambda: execution(store, config, planned))
    check("joinExecution", lambda: downstream(store, config, join=True))
    check("muxExecution", lambda: downstream(store, config, join=False))
    check(
        "compressorInputBinding",
        lambda: compressor_input(
            store, config, store.read(_required(paths.remux_receipt, "--remux-receipt"))
        ),
    )
    check(
        "compressorAndRemux",
        lambda: adapter._remux(
            store,
            store.read(_required(paths.remux_receipt, "--remux-receipt")),
            render_path=config.path,
            render=raw_config,
            movie=movie,
        ),
    )
    result["actualNativeReceipts"] = store.native_receipts
    result["selectionPlanMatchesProtectedCuts"] = all(
        checks.get(key) is True
        for key in (
            "originalSourceClock",
            "cutAccounting",
            "sourceGridContract",
            "protectedSpeechAndRemovedNonSpeech",
            "nativeSelectionPlan",
            "sourceAssembly",
        )
    )
    result["recordedExecutionMatchesPlan"] = all(
        checks.get(key) is True
        for key in (
            "technical",
            "nativeSelectionPlan",
            "completedNativeExecution",
            "joinExecution",
            "muxExecution",
            "compressorAndRemux",
            "compressorInputBinding",
        )
    )
    result["inputs"] = store.rows()
    check(
        "completionV2Adapter",
        lambda: adapter._cuts(
            store,
            result,
            render_path=config.path,
            render=raw_config,
            movie=movie,
            frames=integer(frames),
        ),
    )
    result["inputs"] = store.rows()
    result["inputsAfter"] = check("inputsStable", store.after)
    if not checks["inputsStable"]:
        result["selectionPlanMatchesProtectedCuts"] = False
        result["recordedExecutionMatchesPlan"] = False
    if not all(checks.values()):
        # v2._cuts consumes the flags, not this report's top-level accepted field.
        result["recordedExecutionMatchesPlan"] = False
    result.update(checks=checks, failures=failures, accepted=all(checks.values()))
    render._write_new(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "movie", "technical", "report"):
        parser.add_argument(name, type=Path)
    for name in (
        "source-clock",
        "clock-receipt",
        "assembly-receipt",
        "protection",
        "remux-receipt",
    ):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit(
            args.config,
            args.movie,
            args.technical,
            args.report,
            paths=EvidencePaths(
                args.source_clock,
                args.clock_receipt,
                args.assembly_receipt,
                args.protection,
                args.remux_receipt,
            ),
        )
    except ERRORS as error:
        print(f"CUT_AUDIT_ERROR: {error}", file=sys.stderr)
        return 1
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())
