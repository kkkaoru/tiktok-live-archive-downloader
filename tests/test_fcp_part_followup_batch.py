import fcntl
import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

import replay.fcp_part_followup_batch as batch
from replay import fcp_part_followup as canonical
from replay import part_master_publication as publications
from replay import source_caption_binding, stage_recovery, stages, workflow
from replay.artifact_json import _pin
from replay.core import ReplayError
from replay.fcp_part_assembly import FrameProbeTool
from replay.fcp_part_followup import Followup
from replay.fcp_part_followup_batch import Part, Plan, Reference, Step, run
from replay.part_master_publication import prepare_retirement, publish, verify_published


def test_canonical_batch_rejects_changed_bound_tool_without_repeated_effects(
    canonical_config: tuple[Path, list[str]], tmp_path: Path
) -> None:
    config, calls = canonical_config
    graph = batch.prepare(config, tmp_path / "tool-bound-batch")
    plan = batch.load(graph.parent / "plan.json")
    (tmp_path / "ffprobe").write_bytes(b"changed selected decoder")
    with pytest.raises(ReplayError, match="explicit fingerprints"):
        run(plan, batch.CanonicalBackend())
    assert calls == []
    assert not (tmp_path / "ssd-1.mov").exists()


class Backend:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[int, Step]] = []
        self.fail: tuple[int, Step] | None = None
        self.mutate: Path | None = None
        self.two_outputs = False
        self.protected: Path | None = None
        self.mutate_at: tuple[int, Step] | None = None
        self.collision = False
        self.ancestor = False

    def bind_inputs(self, inputs: tuple[Reference, ...]) -> None:
        assert inputs

    def outputs(self, part: Part, step: Step) -> tuple[Path, ...]:
        if self.protected is not None:
            return (self.protected,)
        if self.collision:
            return (self.root / "same-output",)
        if self.ancestor:
            return (self.root / "output",) if part.number == 1 else (self.root / "output/child",)
        first = self.root / f"{part.number}-{step}"
        return (first, self.root / f"{part.number}-{step}-second") if self.two_outputs else (first,)

    def effects(self, part: Part, step: Step) -> tuple[Path, ...]:
        return (self.root / f"{part.number}-publish.intent",) if step == Step.PUBLISH else ()

    def execute(self, part: Part, step: Step) -> None:
        self.calls.append((part.number, step))
        if self.fail == (part.number, step):
            raise RuntimeError("interrupted primitive")
        self.outputs(part, step)[0].write_bytes(b"verified primitive output")
        if self.mutate is not None and self.mutate_at in (None, (part.number, step)):
            self.mutate.write_bytes(b"changed input")

    def verify(self, part: Part, step: Step, outputs: tuple[Reference, ...]) -> None:
        if any(ref.path.read_bytes() != b"verified primitive output" for ref in outputs):
            raise ReplayError("primitive domain verification failed")
        if (
            part.retirement is not None
            and part.retirement.path.read_bytes() != b"verified retirement"
        ):
            raise ReplayError("retirement domain verification failed")


def ref(path: Path) -> Reference:
    return Reference(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def plan(tmp_path: Path) -> Plan:
    code, model = tmp_path / "code", tmp_path / "model"
    code.write_bytes(b"shared producer code")
    model.write_bytes(b"verified model")
    operation1, operation2 = tmp_path / "operation1", tmp_path / "operation2"
    operation1.write_bytes(b"one immutable operation")
    operation2.write_bytes(b"two immutable operation")
    native1, native2 = tmp_path / "native1", tmp_path / "native2"
    native1.write_bytes(b"first decoded media")
    native2.write_bytes(b"second decoded media")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    return Plan(
        (ref(code), ref(model)),
        (Part(1, ref(operation1), ref(native1)), Part(2, ref(operation2), ref(native2))),
        checkpoints,
        tmp_path / "terminal.json",
        (ref(code),),
    )


def test_batch_uses_one_fresh_common_start_and_end(plan: Plan, tmp_path: Path) -> None:
    backend = Backend(tmp_path)
    result = run(plan, backend)
    assert result.executed == 8
    assert result.adopted == 0
    assert result.hash_reads == 12
    assert len(backend.calls) == 8
    terminal = json.loads(plan.terminal.read_text())
    assert terminal["qualityApproved"] is False
    assert terminal["deliveryApproved"] is False
    assert len(terminal["inputs"]) == 6


def test_interruption_keeps_stages_and_resume_executes_only_missing(
    plan: Plan, tmp_path: Path
) -> None:
    backend = Backend(tmp_path)
    backend.fail = (2, Step.FRAMES)
    with pytest.raises(RuntimeError, match="interrupted"):
        run(plan, backend)
    assert (plan.checkpoints / "part-0001-publish.json").is_file()
    assert not plan.terminal.exists()
    backend.fail = None
    backend.calls.clear()
    result = run(replace(plan, terminal=tmp_path / "resumed.json"), backend)
    assert result.executed == 4
    assert result.adopted == 4
    assert backend.calls == [
        (2, Step.FRAMES),
        (2, Step.COMPLETE),
        (2, Step.PUBLICATION_CONFIG),
        (2, Step.PUBLISH),
    ]


@pytest.mark.parametrize("which", ["code", "native", "operation"])
def test_changed_inputs_reject_resume_without_dispatch(
    plan: Plan, tmp_path: Path, which: str
) -> None:
    run(plan, Backend(tmp_path))
    target = {
        "code": plan.common[0].path,
        "native": plan.parts[0].native.path,
        "operation": plan.parts[0].operation.path,
    }[which]
    target.write_bytes(b"changed input")
    backend = Backend(tmp_path)
    with pytest.raises(ReplayError, match="fingerprints"):
        run(replace(plan, terminal=tmp_path / "retry.json"), backend)
    assert backend.calls == []


def test_new_common_pin_cannot_relabel_old_checkpoint(plan: Plan, tmp_path: Path) -> None:
    run(plan, Backend(tmp_path))
    plan.common[0].path.write_bytes(b"new producer")
    updated = replace(
        plan,
        common=(ref(plan.common[0].path), plan.common[1]),
        backend_code=(ref(plan.common[0].path),),
        terminal=tmp_path / "retry.json",
    )
    backend = Backend(tmp_path)
    with pytest.raises(ReplayError, match="identity changed"):
        run(updated, backend)
    assert backend.calls == []


def test_uncheckpointed_complete_effect_is_verified_without_duplicate_publication(
    plan: Plan, tmp_path: Path
) -> None:
    run(plan, Backend(tmp_path))
    (plan.checkpoints / "part-0002-publish.json").unlink()
    backend = Backend(tmp_path)
    result = run(replace(plan, terminal=tmp_path / "retry.json"), backend)
    assert result.executed == 0
    assert result.adopted == 8
    assert backend.calls == []


def test_partial_side_effect_is_retained_and_never_reexecuted(plan: Plan, tmp_path: Path) -> None:
    backend = Backend(tmp_path)
    backend.two_outputs = True
    (tmp_path / "1-frames").write_bytes(b"verified primitive output")
    with pytest.raises(ReplayError, match="uncertain partial effects"):
        run(plan, backend)
    assert backend.calls == []
    assert (tmp_path / "1-frames").read_bytes() == b"verified primitive output"


@pytest.mark.parametrize("valid", [True, False])
def test_retired_staging_requires_explicit_verified_replacement(
    plan: Plan, tmp_path: Path, valid: bool
) -> None:
    run(plan, Backend(tmp_path))
    replacement = tmp_path / "published-media"
    replacement.write_bytes(plan.parts[0].native.path.read_bytes())
    plan.parts[0].native.path.unlink()
    retirement = tmp_path / "retirement"
    retirement.write_bytes(b"verified retirement" if valid else b"invalid retirement")
    adopted = replace(plan.parts[0], native=ref(replacement), retirement=ref(retirement))
    updated = replace(plan, parts=(adopted, plan.parts[1]), terminal=tmp_path / "retry.json")
    backend = Backend(tmp_path)
    if valid:
        result = run(updated, backend)
        assert result.adopted == 8
    else:
        with pytest.raises(ReplayError, match="retirement domain"):
            run(updated, backend)
    assert backend.calls == []


def test_missing_staging_without_retirement_binding_is_rejected(plan: Plan, tmp_path: Path) -> None:
    run(plan, Backend(tmp_path))
    plan.parts[0].native.path.unlink()
    with pytest.raises(FileNotFoundError):
        run(replace(plan, terminal=tmp_path / "retry.json"), Backend(tmp_path))


def test_boundary_mutation_retains_checkpoint_but_no_terminal(plan: Plan, tmp_path: Path) -> None:
    backend = Backend(tmp_path)
    backend.mutate = plan.common[0].path
    with pytest.raises(ReplayError, match="changed within"):
        run(plan, backend)
    assert (plan.checkpoints / "part-0001-frames.json").is_file()
    assert not plan.terminal.exists()


def test_checkpoint_output_mutation_rejects_adoption(plan: Plan, tmp_path: Path) -> None:
    run(plan, Backend(tmp_path))
    (tmp_path / "1-frames").write_bytes(b"different output")
    backend = Backend(tmp_path)
    with pytest.raises(ReplayError, match="output bytes changed"):
        run(replace(plan, terminal=tmp_path / "retry.json"), backend)
    assert backend.calls == []


def test_terminal_attempt_is_never_overwritten(plan: Plan, tmp_path: Path) -> None:
    run(plan, Backend(tmp_path))
    before = plan.terminal.read_bytes()
    with pytest.raises(ReplayError, match="already exists"):
        run(plan, Backend(tmp_path))
    assert plan.terminal.read_bytes() == before


@pytest.mark.parametrize(
    "mode", ["empty-common", "empty-parts", "duplicate", "conflicting", "missing-dir"]
)
def test_invalid_bounded_plan_rejects_before_execution(
    plan: Plan, tmp_path: Path, mode: str
) -> None:
    variants = {
        "empty-common": replace(plan, common=()),
        "empty-parts": replace(plan, parts=()),
        "duplicate": replace(plan, parts=(plan.parts[0], plan.parts[0])),
        "conflicting": replace(
            plan, common=(plan.common[0], Reference(plan.common[0].path, "0" * 64))
        ),
        "missing-dir": replace(plan, checkpoints=tmp_path / "missing"),
    }
    backend = Backend(tmp_path)
    with pytest.raises(ReplayError):
        run(variants[mode], backend)
    assert backend.calls == []


def test_output_namespace_cannot_overwrite_inputs(plan: Plan, tmp_path: Path) -> None:
    backend = Backend(tmp_path)
    backend.protected = plan.common[0].path
    with pytest.raises(ReplayError, match="overlaps protected"):
        run(plan, backend)
    assert backend.calls == []


@pytest.mark.parametrize("number", [0, True, 1025])
def test_invalid_partition_numbers(plan: Plan, number: int) -> None:
    with pytest.raises(ReplayError, match="partition number"):
        Part(number, plan.parts[0].operation, plan.parts[0].native)


def test_invalid_reference_digest(plan: Plan) -> None:
    with pytest.raises(ReplayError, match="SHA256"):
        Reference(plan.common[0].path, "invalid")


def test_checkpoint_link_is_rejected(plan: Plan, tmp_path: Path) -> None:
    run(plan, Backend(tmp_path))
    checkpoint = plan.checkpoints / "part-0001-frames.json"
    backup = tmp_path / "backup"
    checkpoint.replace(backup)
    checkpoint.symlink_to(backup)
    with pytest.raises(ReplayError, match="canonical"):
        run(replace(plan, terminal=tmp_path / "retry.json"), Backend(tmp_path))


@pytest.mark.parametrize("mode", ["collision", "ancestor"])
def test_global_output_collisions_are_rejected_before_effects(
    plan: Plan, tmp_path: Path, mode: str
) -> None:
    backend = Backend(tmp_path)
    backend.collision = mode == "collision"
    backend.ancestor = mode == "ancestor"
    with pytest.raises(ReplayError, match="collide across"):
        run(plan, backend)
    assert backend.calls == []


def test_publication_intent_without_outputs_is_uncertain(plan: Plan, tmp_path: Path) -> None:
    (tmp_path / "1-publish.intent").write_bytes(b"retained publication intent")
    backend = Backend(tmp_path)
    with pytest.raises(ReplayError, match="uncertain partial effects"):
        run(plan, backend)
    assert (1, Step.PUBLISH) not in backend.calls
    assert (tmp_path / "1-publish.intent").read_bytes() == b"retained publication intent"


@pytest.mark.parametrize("checkpoint", [False, True])
def test_later_part_cannot_mutate_earlier_output_or_checkpoint(
    plan: Plan, tmp_path: Path, checkpoint: bool
) -> None:
    backend = Backend(tmp_path)
    backend.mutate_at = (2, Step.FRAMES)
    backend.mutate = (
        plan.checkpoints / "part-0001-publish.json" if checkpoint else tmp_path / "1-publish"
    )
    with pytest.raises(ReplayError, match="changed within"):
        run(plan, backend)
    assert not plan.terminal.exists()


def test_missing_backend_pin_is_rejected(plan: Plan, tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="implementation pins"):
        run(replace(plan, backend_code=()), Backend(tmp_path))


def test_empty_outputs_are_rejected(
    plan: Plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = Backend(tmp_path)
    monkeypatch.setattr(backend, "outputs", lambda part, step: ())
    with pytest.raises(ReplayError, match="namespace is missing"):
        run(plan, backend)
    assert backend.calls == []


def test_excessive_side_effects_are_rejected(
    plan: Plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = Backend(tmp_path)
    monkeypatch.setattr(
        backend, "effects", lambda part, step: tuple(tmp_path / str(i) for i in range(17))
    )
    with pytest.raises(ReplayError, match="side-effect namespace"):
        run(plan, backend)
    assert backend.calls == []


def test_bounded_input_inventory_is_retained(
    plan: Plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(batch, "MAX_INPUTS", 6)
    with pytest.raises(ReplayError, match="bounded inventory"):
        run(plan, Backend(tmp_path))


def test_checkpoint_work_boundary_is_retained(plan: Plan, tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="repository work"):
        run(replace(plan, checkpoints=Path(__file__).parent), Backend(tmp_path))


def test_concurrent_checkpoint_owner_is_rejected(plan: Plan, tmp_path: Path) -> None:
    with (plan.checkpoints / ".batch.lock").open("wb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ReplayError, match="owned by another"):
            run(plan, Backend(tmp_path))


def test_interrupted_checkpoint_write_leaves_no_completed_record_and_resume_adopts(
    plan: Plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = json.dump

    def interrupted(value: object, stream: object, **kwargs: object) -> None:
        raise OSError("interrupted checkpoint write")

    backend = Backend(tmp_path)
    monkeypatch.setattr(json, "dump", interrupted)
    with pytest.raises(OSError, match="interrupted checkpoint"):
        run(plan, backend)
    assert not (plan.checkpoints / "part-0001-frames.json").exists()
    assert not plan.terminal.exists()
    assert len(tuple(plan.checkpoints.glob(".batch-record-*"))) == 1
    monkeypatch.setattr(json, "dump", original)
    backend.calls.clear()
    result = run(plan, backend)
    assert result.adopted == 1
    assert (1, Step.FRAMES) not in backend.calls


def test_interrupted_terminal_publish_keeps_all_checkpoints_for_resume(
    plan: Plan, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = batch._save_atomic

    def fail_terminal(path: Path, value: dict[str, object]) -> None:
        if path == plan.terminal:
            raise OSError("interrupted terminal publish")
        original(path, value)

    monkeypatch.setattr(batch, "_save_atomic", fail_terminal)
    with pytest.raises(OSError, match="interrupted terminal"):
        run(plan, Backend(tmp_path))
    assert not plan.terminal.exists()
    assert len(tuple(plan.checkpoints.glob("part-*.json"))) == 8
    monkeypatch.setattr(batch, "_save_atomic", original)
    backend = Backend(tmp_path)
    result = run(plan, backend)
    assert result.adopted == 8
    assert backend.calls == []


@pytest.fixture
def canonical_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    calls: list[str] = []
    executable = tmp_path / "ffprobe"
    executable.write_bytes(b"synthetic pinned ffprobe")
    executable.chmod(0o700)
    semantic = tmp_path / "semantic.fcpxml"
    semantic.write_bytes(b"synthetic semantic XML")
    cleanup = tmp_path / "cleanup.json"
    transitive = tmp_path / "transitive-repair"
    transitive.write_bytes(b"verified shared repair")
    cleanup.write_text(
        json.dumps(
            {
                "transitive": ref(transitive).json(),
                "consumer": {**ref(semantic).json(), "kind": "fcpxml"},
            }
        )
    )
    movies = [tmp_path / f"native-{i}.mov" for i in (1, 2)]
    for movie in movies:
        movie.write_bytes(b"synthetic native movie")
    staging, published = tmp_path / "staging.json", tmp_path / "published.json"
    staging.write_text(
        json.dumps(
            {
                "parts": [
                    {"movie": str(movie), "proof": str(tmp_path / f"proof-{i}.json")}
                    for i, movie in enumerate(movies, 1)
                ]
            }
        )
    )
    published.write_text(
        json.dumps(
            {
                "parts": [
                    {
                        "movie": str(tmp_path / f"ssd-{i}.mov"),
                        "proof": str(tmp_path / f"ssd-proof-{i}.json"),
                    }
                    for i in (1, 2)
                ]
            }
        )
    )
    operations: list[dict[str, object]] = []
    for i, movie in enumerate(movies, 1):
        root = tmp_path / f"operation-{i}"
        root.mkdir()
        config = root / "config.json"
        completion_config = root / "completion-config.json"
        completion_config.write_text(json.dumps({"part": i}))
        config.write_text(
            json.dumps(
                {
                    "version": 1,
                    "part": i,
                    "assembly": ref(staging).json(),
                    "publishedAssembly": ref(published).json(),
                    "sourceCleanup": ref(cleanup).json(),
                    "ffprobe": ref(executable).json(),
                    "receiptOutput": str(tmp_path / f"published-{i}.json"),
                }
            )
        )
        operation = root / "operation.json"
        inputs = tuple(
            ref(p).json()
            for p in (
                config,
                staging,
                published,
                semantic,
                cleanup,
                transitive,
                executable,
                movie,
                completion_config,
            )
        )
        operation.write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "bounded-native-part-followup",
                    "config": str(config),
                    "semantic": str(semantic),
                    "completionConfig": str(completion_config),
                    "inputs": inputs,
                }
            )
        )
        operations.append({"operation": ref(operation).json()})

    def cleanup_domain(config: Path, actual_semantic: Path) -> tuple[Path, int, bool]:
        assert actual_semantic == semantic
        _pin(json.loads(cleanup.read_text())["transitive"])
        raw = json.loads(config.read_text())
        return staging, int(raw["part"]), False

    def native_plan(
        operation: Path, actual_semantic: Path, data: dict[str, object], inputs: tuple[Path, ...]
    ) -> Followup:
        assert actual_semantic == semantic
        references = data["inputs"]
        assert isinstance(references, list)
        for value in references:
            _pin(value)
        raw = json.loads(Path(str(data["config"])).read_text())
        i = int(raw["part"])
        return Followup(
            i,
            {"assembly": staging, "publishedAssembly": published},
            movies[i - 1],
            tmp_path / f"proof-{i}.json",
            tmp_path / f"ssd-{i}.mov",
            tmp_path / f"ssd-proof-{i}.json",
            tmp_path / f"published-{i}.json",
            inputs,
            False,
            48000,
            2,
            semantic,
        )

    def frame_probe(config: Path, *, part: int, tool: FrameProbeTool) -> Path:
        assert config == staging
        assert tool.path == executable and tool.decoder_threads == 0
        tool.verify()
        calls.append(f"frames-{part}")
        proof = tmp_path / f"proof-{part}.json"
        proof.write_text(
            json.dumps({"version": 1, "part": part, "movie": ref(movies[part - 1]).json()})
        )
        return proof

    def frame_verify(config: Path, *, part: int) -> Path:
        assert config in (staging, published)
        proof = tmp_path / (f"proof-{part}.json" if config == staging else f"ssd-proof-{part}.json")
        assert json.loads(proof.read_text())["part"] == part
        return proof

    def native_complete(config: Path, output: Path, *, recover: bool) -> None:
        assert recover is False
        i = int(json.loads(config.read_text())["part"])
        calls.append(f"complete-{i}")
        proof = tmp_path / f"proof-{i}.json"
        output.write_text(
            json.dumps(
                {
                    "version": 1,
                    "kind": "verified-native-part",
                    "part": i,
                    "completionVerified": True,
                    "shareObservedIdle": True,
                    "fullAVDecodeExitCode": 0,
                    "movie": ref(movies[i - 1]).json(),
                    "frameProof": ref(proof).json(),
                    "inputs": {
                        str(config): ref(config).sha256,
                        str(movies[i - 1]): ref(movies[i - 1]).sha256,
                    },
                }
            )
        )

    def publication_domain(assembly: Path, *, part: int, publication: Path) -> None:
        proof = frame_verify(assembly, part=part)
        verify_published(publication, proof, part=part)

    monkeypatch.setattr(canonical, "CODE_NAMES", ())
    monkeypatch.setattr(canonical, "_integrated_plan", native_plan)
    monkeypatch.setattr(canonical, "validate_cleanup", cleanup_domain)
    monkeypatch.setattr(batch, "probe", frame_probe)
    monkeypatch.setattr(batch, "verify_frames", frame_verify)
    monkeypatch.setattr(batch, "complete", native_complete)
    monkeypatch.setattr(canonical, "verify_publication", publication_domain)
    monkeypatch.setattr(publications, "verify_part", frame_verify)
    monkeypatch.setattr(batch, "clean_source", lambda _: (transitive, transitive, Fraction(1)))
    config = tmp_path / "batch-config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "operations": operations,
                "maxReadyParts": 2,
                "maxNativeBytes": 4096,
            }
        )
    )
    return config, calls


def test_canonical_plan_cli_preserves_native_and_real_exclusive_publication(
    canonical_config: tuple[Path, list[str]], tmp_path: Path
) -> None:
    config, calls = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    assert json.loads(graph.read_text())["sourceCaptionProduction"]["bindings"]
    stage = json.loads(graph.read_text())["stages"][0]
    assert str(tmp_path / "cleanup.json") in stage["inputs"]
    assert str(tmp_path / "transitive-repair") in stage["inputs"]
    assert str(tmp_path / "semantic.fcpxml") in stage["inputs"]
    assert str(tmp_path / "semantic.fcpxml") in stage["command"]
    plan = batch.load(graph.parent / "plan.json")
    result = run(plan, batch.CanonicalBackend())
    assert result.executed == 8
    assert calls == ["frames-1", "complete-1", "frames-2", "complete-2"]
    assert (tmp_path / "ssd-1.mov").read_bytes() == plan.parts[0].native.path.read_bytes()
    batch.verify(plan, batch.CanonicalBackend())
    batch.main(["verify", str(graph.parent / "plan.json")])


def test_canonical_interrupted_complete_prefix_resumes_without_publication_duplicate(
    canonical_config: tuple[Path, list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, calls = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    plan = batch.load(graph.parent / "plan.json")
    backend = batch.CanonicalBackend()
    execute = backend.execute

    def interrupted(part: Part, step: Step) -> None:
        if part.number == 2 and step == Step.COMPLETE:
            raise OSError("interrupted second native completion")
        execute(part, step)

    monkeypatch.setattr(backend, "execute", interrupted)
    with pytest.raises(OSError, match="second native"):
        run(plan, backend)
    first_published = (tmp_path / "published-1.json").read_bytes()
    calls.clear()
    result = run(plan, batch.CanonicalBackend())
    assert result.adopted == 5
    assert result.executed == 3
    assert calls == ["complete-2"]
    assert (tmp_path / "published-1.json").read_bytes() == first_published


def test_workflow_explicit_recovery_resumes_same_checkpointed_controller(
    canonical_config: tuple[Path, list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, calls = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    state = tmp_path / "workflow-state"
    semantic = tmp_path / "semantic.fcpxml"
    actual_execute = batch.CanonicalBackend.execute
    interrupted = True
    commands: list[str] = []

    def execute_once(backend: batch.CanonicalBackend, part: Part, step: Step) -> None:
        if interrupted and part.number == 2 and step == Step.COMPLETE:
            raise OSError("stopped second completion")
        actual_execute(backend, part, step)

    def binding(
        path: Path, *, consumer_inputs: tuple[Path, ...], consumer_command: tuple[str, ...]
    ) -> None:
        assert path == tmp_path / "cleanup.json"
        assert path in consumer_inputs and semantic in consumer_inputs
        assert str(semantic) in consumer_command
        canonical.validate_cleanup(tmp_path / "operation-1/config.json", semantic)

    def execute(command: tuple[str, ...], log: Path, timeout: int) -> None:
        assert command[3] == "replay.fcp_part_followup_batch" and timeout > 0
        commands.append(command[4])
        log.write_text(command[4])
        try:
            batch.main(list(command[4:]))
        except (ReplayError, OSError) as error:
            raise stages.StageError("synthetic subprocess failed") from error

    def run_stage(stage: stages.Stage, *, state_directory: Path) -> str:
        return stages.run_stage(stage, state_directory=state_directory, execute=execute)

    monkeypatch.setattr(batch.CanonicalBackend, "execute", execute_once)
    monkeypatch.setattr(source_caption_binding, "validate_binding", binding)
    monkeypatch.setattr(workflow, "run_stage", run_stage)
    assert workflow.main([str(graph), "--state", str(state), "--check"]) == 0
    assert workflow.main([str(graph), "--state", str(state)]) == 1
    plan = batch.load(graph.parent / "plan.json")
    assert stage_recovery.unfinished(graph, state) == ["followup-batch"]
    first_checkpoint = plan.checkpoints / "part-0001-publish.json"
    first_publication = tmp_path / "published-1.json"
    preserved = {path: path.read_bytes() for path in (first_checkpoint, first_publication)}
    native_before = {part.native.path: part.native.path.read_bytes() for part in plan.parts}
    calls.clear()
    commands.clear()
    # A plain rerun never dispatches a mutation from a retained running receipt.
    assert workflow.main([str(graph), "--state", str(state)]) == 1
    assert commands == ["verify"] and calls == []
    interrupted = False
    # The synchronous executor has stopped; both ownership locks are now free.
    for name in (".workflow", "followup-batch"):
        with stages.execution_lock(state, name) as acquired:
            assert acquired
    archived = stage_recovery.recover(graph, state)
    assert [path.name for path in archived] == ["workflow-state.failed-0"]
    assert all(path.read_bytes() == data for path, data in preserved.items())
    assert all(path.read_bytes() == data for path, data in native_before.items())
    commands.clear()
    assert workflow.main([str(graph), "--state", str(state)]) == 0
    assert commands == ["run", "verify"]
    assert calls == ["complete-2"]
    assert all(path.read_bytes() == data for path, data in preserved.items())
    assert all(path.read_bytes() == data for path, data in native_before.items())
    assert json.loads(plan.terminal.read_text())["kind"] == "verified-common-context-native-batch"
    calls.clear()
    commands.clear()
    assert workflow.main([str(graph), "--state", str(state)]) == 0
    assert commands == ["verify"] and calls == []


def test_canonical_uncertain_publication_cutpoint_is_not_replayed(
    canonical_config: tuple[Path, list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    plan = batch.load(graph.parent / "plan.json")
    original = publish

    def cutpoint(path: Path) -> Path:
        raw = json.loads(path.read_text())
        Path(raw["receiptOutput"]).with_suffix(".intent.json").write_bytes(b"partial intent")
        raise OSError("publication interrupted after intent")

    monkeypatch.setattr(batch, "publish", cutpoint)
    with pytest.raises(OSError, match="after intent"):
        run(plan, batch.CanonicalBackend())
    monkeypatch.setattr(batch, "publish", original)
    with pytest.raises(ReplayError, match="uncertain partial effects"):
        run(plan, batch.CanonicalBackend())
    assert not (tmp_path / "ssd-1.mov").exists()


@pytest.mark.parametrize("mode", ["bytes", "count", "version", "root", "missing-number"])
def test_canonical_plan_preparation_rejects_invalid_or_unbounded_inputs(
    canonical_config: tuple[Path, list[str]], tmp_path: Path, mode: str
) -> None:
    config, calls = canonical_config
    raw = json.loads(config.read_text())
    if mode == "bytes":
        raw["maxNativeBytes"] = 1
    elif mode == "count":
        raw["maxReadyParts"] = 1
    elif mode == "version":
        raw["version"] = 2
    elif mode == "root":
        (tmp_path / "batch").mkdir()
    else:
        operation = Path(raw["operations"][0]["operation"]["path"])
        metadata = json.loads(operation.read_text())
        original = Path(metadata["config"])
        value = json.loads(original.read_text())
        value["part"] = "1"
        original.write_text(json.dumps(value))
    config.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        batch.prepare(config, tmp_path / "batch")
    assert calls == []


def test_canonical_changed_repair_after_prepare_rejects_before_any_media_primitive(
    canonical_config: tuple[Path, list[str]], tmp_path: Path
) -> None:
    config, calls = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    (tmp_path / "transitive-repair").write_bytes(b"different repair bytes")
    with pytest.raises(ReplayError, match="fingerprints"):
        run(batch.load(graph.parent / "plan.json"), batch.CanonicalBackend())
    assert calls == []


def test_canonical_completed_publication_with_partial_leftover_requires_reconciliation(
    canonical_config: tuple[Path, list[str]], tmp_path: Path
) -> None:
    config, _ = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    plan = batch.load(graph.parent / "plan.json")
    run(plan, batch.CanonicalBackend())
    (tmp_path / ".ssd-1.mov.publication-partial").write_bytes(b"retained partial")
    with pytest.raises(ReplayError, match="partial effect"):
        batch.verify(plan, batch.CanonicalBackend())


def retired_part(plan: Plan, tmp_path: Path) -> Part:
    config = tmp_path / "retirement-config.json"
    retirement = tmp_path / "retirement.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "retireStaging": True,
                "publication": ref(tmp_path / "published-1.json").json(),
                "assembly": ref(tmp_path / "published.json").json(),
                "receiptOutput": str(retirement),
            }
        )
    )
    prepare_retirement(config)
    verification = tmp_path / "retirement-verify.json"
    verification.write_text(json.dumps({"version": 1, "retirement": ref(retirement).json()}))
    original = plan.parts[0]
    original.native.path.unlink()
    evidence = ref(verification)
    return replace(
        original,
        native=ref(tmp_path / "ssd-1.mov"),
        retirement=evidence,
        adoption=batch._retirement_inputs(evidence),
    )


def test_canonical_retired_resume_preserves_original_operation_and_uses_verified_ssd(
    canonical_config: tuple[Path, list[str]], tmp_path: Path
) -> None:
    config, calls = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    plan = batch.load(graph.parent / "plan.json")
    run(plan, batch.CanonicalBackend())
    original_checkpoint = (plan.checkpoints / "part-0001-complete.json").read_bytes()
    part = retired_part(plan, tmp_path)
    updated = replace(plan, parts=(part, plan.parts[1]), terminal=tmp_path / "retired-resume.json")
    calls.clear()
    result = run(updated, batch.CanonicalBackend())
    assert result.adopted == 8
    assert result.executed == 0
    assert calls == []
    assert (plan.checkpoints / "part-0001-complete.json").read_bytes() == original_checkpoint
    batch.verify(updated, batch.CanonicalBackend())


@pytest.mark.parametrize("target", ["ssd", "retirement", "assembly", "operation"])
def test_canonical_retired_replacement_or_lineage_mutation_is_rejected(
    canonical_config: tuple[Path, list[str]], tmp_path: Path, target: str
) -> None:
    config, calls = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    plan = batch.load(graph.parent / "plan.json")
    run(plan, batch.CanonicalBackend())
    part = retired_part(plan, tmp_path)
    path = {
        "ssd": part.native.path,
        "retirement": tmp_path / "retirement.json",
        "assembly": tmp_path / "published.json",
        "operation": part.operation.path,
    }[target]
    path.write_bytes(b"mutated immutable lineage")
    calls.clear()
    with pytest.raises(ReplayError, match="fingerprints"):
        run(
            replace(plan, parts=(part, plan.parts[1]), terminal=tmp_path / "retry.json"),
            batch.CanonicalBackend(),
        )
    assert calls == []


def test_canonical_retired_missing_history_cannot_be_implicitly_filtered(
    canonical_config: tuple[Path, list[str]], tmp_path: Path
) -> None:
    config, _ = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    plan = batch.load(graph.parent / "plan.json")
    run(plan, batch.CanonicalBackend())
    part = replace(retired_part(plan, tmp_path), adoption=())
    with pytest.raises(ReplayError, match="immutable history closure"):
        run(
            replace(plan, parts=(part, plan.parts[1]), terminal=tmp_path / "retry.json"),
            batch.CanonicalBackend(),
        )


@pytest.mark.parametrize("arguments", [["prepare"], ["run", "--output"]])
def test_cli_requires_explicit_preparation_output(
    canonical_config: tuple[Path, list[str]], tmp_path: Path, arguments: list[str]
) -> None:
    config, _ = canonical_config
    if arguments == ["prepare"]:
        with pytest.raises(ReplayError, match="explicit fresh"):
            batch.main(["prepare", str(config)])
    else:
        with pytest.raises(ReplayError, match="Only preparation"):
            batch.main(["run", str(config), "--output", str(tmp_path / "batch")])


def test_cli_prepare_run_and_terminal_mutation_rejection(
    canonical_config: tuple[Path, list[str]], tmp_path: Path
) -> None:
    config, _ = canonical_config
    output = tmp_path / "batch"
    batch.main(["prepare", str(config), "--output", str(output)])
    batch.main(["run", str(output / "plan.json")])
    plan = batch.load(output / "plan.json")
    raw = json.loads(plan.terminal.read_text())
    raw["deliveryApproved"] = True
    plan.terminal.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="terminal differs"):
        batch.verify(plan, batch.CanonicalBackend())


@pytest.mark.parametrize("value", [{}, {"path": "/absent", "sha256": None}])
def test_plan_external_reference_validation(value: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        batch._read_ref(value)


def test_plan_external_inventory_requires_bounded_list() -> None:
    with pytest.raises(ReplayError, match="bounded reference"):
        batch._read_refs([])


@pytest.mark.parametrize("field", ["version", "parts", "number"])
def test_typed_plan_loader_rejects_malformed_schema(
    canonical_config: tuple[Path, list[str]], tmp_path: Path, field: str
) -> None:
    config, _ = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    path = graph.parent / "plan.json"
    raw = json.loads(path.read_text())
    if field == "number":
        raw["parts"][0]["part"] = "one"
    else:
        raw[field] = None
    path.write_text(json.dumps(raw))
    with pytest.raises(ReplayError):
        batch.load(path)


def test_canonical_manifest_cannot_omit_original_transitive_dependencies(
    canonical_config: tuple[Path, list[str]], tmp_path: Path
) -> None:
    config, calls = canonical_config
    graph = batch.prepare(config, tmp_path / "batch")
    plan = batch.load(graph.parent / "plan.json")
    omitted = replace(
        plan, common=tuple(ref for ref in plan.common if ref.path != tmp_path / "transitive-repair")
    )
    with pytest.raises(ReplayError, match="mandatory operation dependency"):
        run(omitted, batch.CanonicalBackend())
    assert calls == []
