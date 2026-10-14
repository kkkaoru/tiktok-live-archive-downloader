"""Synthetic metadata-only generation adoption; no native tools or footage."""

import hashlib
import json
from pathlib import Path

import pytest

from replay import fcp_native_queue, sequence_journal
from replay import sequence_generation as target
from replay.core import ReplayError
from replay.native_sequence import Checkpoint, Config, ContractError, run
from replay.sequence_journal import DurableJournal, Pin, decode, encode


class Host:
    def __init__(self) -> None:
        self.now = 10_000_000_000
        self.current = "new-birth"
        self.old_exists = False

    def birth(self, pid: int) -> str:
        if pid == 17:
            if self.old_exists:
                return "old-birth"
            raise ProcessLookupError("Old process absent")
        return self.current

    def now_ns(self) -> int:
        return self.now


class World:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.old, self.ledger, self.new = root / "old", root / "old-generation-ledger", root / "new"
        self.old.mkdir()
        self.ledger.mkdir()
        self.new.mkdir()
        self.host = Host()
        self.saved = self.save("saved.json", {"project": "Saved project"})
        self.progress = self.save("progress.json", {"completedParts": [4]})
        self.output = self.save("published.json", {"part": 4, "published": True})
        self.capability = self.save("capabilities.json", {"provider": "native.synthetic"})
        self.executable = self.save("executor.json", {"synthetic": True})
        self.before = self.save(
            "before.json",
            {
                "version": 1,
                "kind": "native-recovery-runtime-observation",
                "target": {"pid": 17, "birth": "old-birth"},
                "project": "Saved project",
            },
        )
        self.observation_data: dict[str, object] = {
            "version": 1,
            "kind": "native-recovery-runtime-observation",
            "target": {"pid": 23, "birth": "new-birth"},
            "project": "Saved project",
            "savedProject": ref(self.saved),
            "progress": ref(self.progress),
            "outputs": [ref(self.output)],
            "observedAtNs": 10_000_000_000,
            "maximumAgeSeconds": 30,
            "providerEvidence": [ref(self.capability)],
        }
        self.observation = self.save("observation.json", self.observation_data)
        self.planned_parts: list[dict[str, object]] = [
            {"assemblyPart": 5, "project": "Future project 5", "operations": []},
            {"assemblyPart": 8, "project": "Future project 8", "operations": []},
        ]
        self.old_queue = self.save(
            "old-queue.json",
            {"target": {"pid": 17, "birth": "old-birth"}, "parts": self.planned_parts},
        )
        self.old_pump = self.save("old-pump.json", {"queue": ref(self.old_queue)})
        self.config = Config(str(self.old_pump.path), (5, 8), self.old_pump.sha256)
        self.code = tuple(
            current(path)
            for path in (
                Path(sequence_journal.__file__).resolve(),
                Path(sequence_journal.__file__).with_name("native_sequence.py").resolve(),
                Path(sequence_journal.__file__).with_name("sequence_adapters.py").resolve(),
            )
        )
        self.journal = DurableJournal(self.old, (*self.code, self.old_pump))
        prior_config = Config(str(self.old_pump.path), (4, 5, 8), self.old_pump.sha256)
        prepared = Checkpoint(prior_config, "prepared")
        started = Checkpoint(prior_config, "started")
        published = Checkpoint(prior_config, "published", encode(ref(self.output)).decode())
        retired = Checkpoint(prior_config, "retired", encode(ref(self.output)).decode())
        released = Checkpoint(prior_config, "released", encode(ref(self.output)).decode())
        self.journal.put(4, None, prepared)
        self.journal.put(4, prepared, started)
        self.journal.put(4, started, published)
        self.journal.put(4, published, retired)
        self.journal.put(4, retired, released)
        self.journal.put(5, None, Checkpoint(self.config, "prepared"))
        self.grant = self.save(
            "grant.json",
            {
                "version": 1,
                "kind": "native-gui-idle-handoff",
                "released": True,
                "target": {"pid": 23, "birth": "new-birth"},
            },
        )
        self.queue_data: dict[str, object] = {
            "kind": "executor-native-part-queue",
            "target": {"pid": 23, "birth": "new-birth"},
            "provider": "native.synthetic",
            "executor": ref(self.executable),
            "guiGrant": ref(self.grant),
            "parts": self.planned_parts,
        }
        self.queue = self.save("new-queue.json", self.queue_data)
        self.admission = self.save(
            "admission.json",
            {
                "kind": "native-admission-preparation-request",
                "target": {"pid": 23, "birth": "new-birth"},
            },
        )
        self.runtime_data: dict[str, object] = {
            "kind": "native-delivery-runtime-evidence",
            "target": {"pid": 23, "birth": "new-birth"},
            "provider": "native.synthetic",
            "executor": ref(self.executable),
            "capabilities": ref(self.capability),
        }
        self.runtime = self.save("runtime.json", self.runtime_data)
        self.pump_data: dict[str, object] = {
            "version": 1,
            "kind": "native-delivery-pump",
            "queue": ref(self.queue),
            "admissionTemplate": ref(self.admission),
            "runtimeEvidence": ref(self.runtime),
        }
        self.pump = self.save("pump.json", self.pump_data)
        self.intent_data: dict[str, object] = {
            "kind": "finite-native-recovery-intent",
            "config": {"identity": str(self.old_pump.path), "sha256": self.old_pump.sha256},
            "pid": 17,
            "birth": "old-birth",
            "reason": "memory-pressure",
            "savedProject": ref(self.saved),
            "progress": ref(self.progress),
            "outputs": [ref(self.output)],
            "evidence": [ref(self.before)],
        }
        self.intent = self.save("intent.json", self.intent_data)
        self.quit = self.save(
            "quit.json",
            {
                "exitCode": 0,
                "stdout": json.dumps(
                    {
                        "ok": True,
                        "data": {
                            "isError": False,
                            "structuredContent": {
                                "pid": 17,
                                "dispatched": True,
                                "terminated": True,
                            },
                        },
                    }
                ),
                "stderr": "",
            },
        )
        self.continuation_data: dict[str, object] = {
            "originalConfig": self.intent_data["config"],
            "newConfig": {
                "identity": str(self.pump.path),
                "tail": [5, 8],
                "sha256": self.pump.sha256,
            },
            "observation": ref(self.observation),
        }
        self.future_data: dict[str, object] = {
            "version": 1,
            "kind": "native-sequence-continuation-data",
            "candidateOnly": True,
            "canDispatch": False,
            "continuation": self.continuation_data,
        }
        self.future = self.save("future.json", self.future_data)
        self.recovery_data: dict[str, object] = {
            "version": 1,
            "kind": "finite-native-recovery-result",
            "intent": ref(self.intent),
            "quit": ref(self.quit),
            "afterEvidence": [ref(self.observation)],
            "pid": 23,
            "birth": "new-birth",
            "improved": True,
            "futureData": ref(self.future),
            "canDispatch": False,
        }
        self.recovery = self.save("recovery.json", self.recovery_data)
        self.request_data: dict[str, object] = {
            "version": 1,
            "kind": "native-sequence-generation-request",
            "orderedParts": [4, 5, 8],
            "oldJournal": {
                "root": str(self.old),
                "generation": ref(current(self.old / "generation.json")),
                "prepared": ref(current(self.old / "part-5-00.json")),
                "predecessors": [ref(current(self.old / "part-4-04.json"))],
            },
            "recovery": ref(self.recovery),
            "ledgerRoot": str(self.ledger),
            "journalRoot": str(self.new),
        }
        self.request = self.save("request.json", self.request_data)

    def save(self, name: str, data: object) -> Pin:
        path = self.root / name
        path.write_bytes(encode(data))
        return current(path)

    def recapture(self, *, update_config: bool = True) -> target.Request:
        self.queue = self.save("new-queue.json", self.queue_data)
        self.runtime = self.save("runtime.json", self.runtime_data)
        self.pump_data.update(queue=ref(self.queue), runtimeEvidence=ref(self.runtime))
        self.pump = self.save("pump.json", self.pump_data)
        self.observation = self.save("observation.json", self.observation_data)
        self.intent = self.save("intent.json", self.intent_data)
        self.continuation_data["observation"] = ref(self.observation)
        if update_config:
            self.continuation_data["newConfig"] = {
                "identity": str(self.pump.path),
                "tail": [5, 8],
                "sha256": self.pump.sha256,
            }
        self.future = self.save("future.json", self.future_data)
        self.recovery_data.update(
            intent=ref(self.intent),
            futureData=ref(self.future),
            afterEvidence=[ref(self.observation)],
        )
        self.recovery = self.save("recovery.json", self.recovery_data)
        self.request_data["recovery"] = ref(self.recovery)
        self.request = self.save("request.json", self.request_data)
        return target.load(self.request)


def current(path: Path) -> Pin:
    return Pin(path.resolve(), hashlib.sha256(path.read_bytes()).hexdigest())


def ref(pin: Pin) -> dict[str, str]:
    return {"path": str(pin.path), "sha256": pin.sha256}


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_restart_new_generation_preserves_original_and_predecessor(world: World) -> None:
    original = (world.old / "part-5-00.json").read_bytes()
    request = target.load(world.request)
    output = target.adopt(request, host=world.host)
    opened = target.open_generation(request, host=world.host)
    state = opened.get(5)
    assert state is not None
    assert state.phase == "prepared"
    assert state.config.tail == (5, 8)
    assert state.config != world.config
    assert (world.old / "part-5-00.json").read_bytes() == original
    assert decode(output.read_bytes())["remainingParts"] == [5, 8]
    predecessors = decode(output.read_bytes())["releasedPredecessors"]
    assert isinstance(predecessors, list)
    assert len(predecessors) == 1


def test_original_journal_rejects_changed_config_but_new_one_accepts(world: World) -> None:
    from test_native_sequence import Fake

    class Changed(Fake):
        def materialize(self, tail: tuple[int, ...]) -> Config:
            return Config(str(world.pump.path), tail, world.pump.sha256)

    steps = Changed()
    with pytest.raises(ContractError, match="immutable"):
        run((5, 8), world.journal, steps)
    request = target.load(world.request)
    target.adopt(request, host=world.host)
    assert run((5, 8), target.open_generation(request, host=world.host), steps).status == "complete"
    assert world.journal.get(5) == Checkpoint(world.config, "prepared")


@pytest.mark.parametrize("phase", ["started", "uncertain", "published", "retired", "released"])
def test_any_changed_original_head_refuses(world: World, phase: str) -> None:
    old = decode((world.old / "part-5-00.json").read_bytes())
    changed = target._row(old["checkpoint"])
    changed["phase"] = phase
    old["checkpoint"] = changed
    (world.old / "part-5-00.json").write_bytes(encode(old))
    request_data = target._row(world.request_data["oldJournal"])
    request_data["prepared"] = ref(current(world.old / "part-5-00.json"))
    world.request_data["oldJournal"] = request_data
    with pytest.raises(ContractError):
        target.adopt(world.recapture(), host=world.host)
    assert not (world.new / "generation.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("improved", False),
        ("canDispatch", True),
        ("kind", "unknown"),
        ("futureData", None),
        ("pid", 17),
        ("birth", "stale-birth"),
    ],
)
def test_failed_or_foreign_recovery_refuses(world: World, field: str, value: object) -> None:
    world.recovery_data[field] = value
    request = world.recapture()
    if field == "futureData":
        world.recovery_data["futureData"] = None
        world.recovery = world.save("recovery.json", world.recovery_data)
        world.request_data["recovery"] = ref(world.recovery)
        request = target.load(world.save("request.json", world.request_data))
    with pytest.raises((ContractError, FileNotFoundError)):
        target.adopt(request, host=world.host)
    assert not (world.new / "generation.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("project", "Another project"),
        ("savedProject", {}),
        ("progress", {}),
        ("outputs", []),
        ("observedAtNs", -1),
        ("maximumAgeSeconds", 31),
        ("providerEvidence", []),
    ],
)
def test_changed_preserved_or_observed_facts_refuse(
    world: World, field: str, value: object
) -> None:
    world.observation_data[field] = value
    with pytest.raises(ContractError):
        target.adopt(world.recapture(), host=world.host)


def test_stale_observation_rejected_without_freshening(world: World) -> None:
    world.host.now = 40_000_000_001
    with pytest.raises(ContractError, match="stale"):
        target.adopt(target.load(world.request), host=world.host)
    assert decode(world.observation.path.read_bytes())["observedAtNs"] == 10_000_000_000


def test_current_old_process_blocks_adoption(world: World) -> None:
    world.host.old_exists = True
    with pytest.raises(ContractError, match="still exists"):
        target.adopt(target.load(world.request), host=world.host)


@pytest.mark.parametrize(
    "artifact",
    ["request.json", "recovery.json", "observation.json", "old-pump.json", "published.json"],
)
def test_changed_pinned_bytes_refuse(world: World, artifact: str) -> None:
    request = target.load(world.request)
    (world.root / artifact).write_bytes(b"changed bytes")
    with pytest.raises((ContractError, ValueError)):
        target.adopt(request, host=world.host)


def test_absent_original_lock_never_repaired(world: World) -> None:
    (world.old / ".lock").unlink()
    with pytest.raises(FileNotFoundError):
        target.adopt(target.load(world.request), host=world.host)
    assert not (world.old / ".lock").exists()


def test_original_missing_predecessor_refuses(world: World) -> None:
    (world.old / "part-4-04.json").unlink()
    with pytest.raises(FileNotFoundError):
        target.adopt(target.load(world.request), host=world.host)


def test_checkpoint_gap_refuses(world: World) -> None:
    (world.old / "part-4-01.json").unlink()
    with pytest.raises(ContractError, match="gap"):
        target.adopt(target.load(world.request), host=world.host)


def test_second_adoption_refuses_and_completed_open_is_allowed(world: World) -> None:
    request = target.load(world.request)
    target.adopt(request, host=world.host)
    with pytest.raises(ContractError, match="already adopted"):
        target.adopt(request, host=world.host)
    world.host.now = 90_000_000_000
    assert target.open_generation(request, host=world.host).get(5) is not None


def test_alternate_ledger_cannot_bypass_same_checkpoint_claim(world: World) -> None:
    alternative = world.root / "alternate"
    alternative.mkdir()
    world.request_data["ledgerRoot"] = str(alternative)
    with pytest.raises(ContractError, match="unique sibling"):
        target.load(world.save("request.json", world.request_data))


def test_interrupted_adoption_resumes_exact_data_only(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = target.load(world.request)
    with monkeypatch.context() as context:

        def interrupted(*_args: object) -> Path:
            raise OSError("Interrupted before new DATA")

        context.setattr(target, "_publish", interrupted)
        with pytest.raises(OSError, match="Interrupted"):
            target.adopt(request, host=world.host)
    world.host.now = 90_000_000_000
    assert target.adopt(request, host=world.host, resume=True).is_file()
    with pytest.raises(ContractError, match="already complete"):
        target.adopt(request, host=world.host, resume=True)


def test_resume_without_original_claim_refuses(world: World) -> None:
    with pytest.raises(FileNotFoundError):
        target.adopt(target.load(world.request), host=world.host, resume=True)


@pytest.mark.parametrize("scope", ["queue", "runtime"])
def test_wrong_provider_refuses(world: World, scope: str) -> None:
    fields = world.queue_data if scope == "queue" else world.runtime_data
    fields["provider"] = "wrong-provider"
    with pytest.raises(ContractError, match="provider"):
        target.adopt(world.recapture(), host=world.host)


def test_wrong_original_config_provenance_refuses(world: World) -> None:
    world.intent_data["config"] = {"identity": "different", "sha256": "invalid"}
    with pytest.raises(ContractError, match="original prepared"):
        target.adopt(world.recapture(), host=world.host)


def test_unknown_quit_cannot_be_success(world: World) -> None:
    world.quit = world.save("quit.json", {"exitCode": 1, "stdout": "{}"})
    world.recovery_data["quit"] = ref(world.quit)
    with pytest.raises(ReplayError, match="successfully"):
        target.adopt(world.recapture(), host=world.host)


def test_absent_generation_receipt_cannot_be_opened(world: World) -> None:
    request = target.load(world.request)
    target.adopt(request, host=world.host)
    (world.new / "continuation.json").unlink()
    with pytest.raises(FileNotFoundError):
        target.open_generation(request, host=world.host)


def test_changed_new_generation_receipt_refuses(world: World) -> None:
    request = target.load(world.request)
    target.adopt(request, host=world.host)
    path = world.new / "continuation.json"
    receipt = decode(path.read_bytes())
    receipt["remainingParts"] = [8]
    path.write_bytes(encode(receipt))
    with pytest.raises(ContractError, match="provenance"):
        target.open_generation(request, host=world.host)


def test_occupied_destination_is_preserved(world: World) -> None:
    (world.new / "user.json").write_bytes(b"preserve")
    with pytest.raises(ContractError, match="occupied"):
        target.adopt(target.load(world.request), host=world.host)
    assert (world.new / "user.json").read_bytes() == b"preserve"


def test_request_rejects_linked_or_escaping_root(world: World) -> None:
    alias = world.root / "alias"
    alias.symlink_to(world.new)
    world.request_data["journalRoot"] = str(alias)
    with pytest.raises(ContractError, match="canonical"):
        target.load(world.save("request.json", world.request_data))


def test_requested_predecessors_must_match_actual_released_heads(world: World) -> None:
    old_journal = target._row(world.request_data["oldJournal"])
    old_journal["predecessors"] = []
    world.request_data["oldJournal"] = old_journal
    with pytest.raises(ContractError, match="predecessor"):
        target.adopt(target.load(world.save("request.json", world.request_data)), host=world.host)


@pytest.mark.parametrize("tail", [[8, 5], [5, 5], [0], []])
def test_changed_suffix_refuses(world: World, tail: list[int]) -> None:
    world.continuation_data["newConfig"] = {
        "identity": str(world.pump.path),
        "tail": tail,
        "sha256": world.pump.sha256,
    }
    with pytest.raises(ContractError):
        target.adopt(world.recapture(update_config=False), host=world.host)


def test_same_configuration_is_not_recovery_generation(world: World) -> None:
    world.continuation_data["newConfig"] = {
        "identity": str(world.old_pump.path),
        "tail": [5, 8],
        "sha256": world.old_pump.sha256,
    }
    with pytest.raises(ContractError, match="must change"):
        target.adopt(world.recapture(update_config=False), host=world.host)


def test_unknown_observation_cannot_be_attached_to_success(world: World) -> None:
    request = world.recapture()
    world.recovery_data["afterEvidence"] = [ref(world.capability)]
    recovery = world.save("recovery.json", world.recovery_data)
    world.request_data["recovery"] = ref(recovery)
    request = target.load(world.save("request.json", world.request_data))
    with pytest.raises(ContractError, match="not pinned"):
        target.adopt(request, host=world.host)


def test_changed_actual_queue_target_refuses(world: World) -> None:
    world.queue_data["target"] = {"pid": 17, "birth": "old-birth"}
    with pytest.raises(ContractError, match="recovered process"):
        target.adopt(world.recapture(), host=world.host)


def test_foreign_quit_identity_refuses(world: World) -> None:
    world.quit = world.save(
        "quit.json",
        {
            "exitCode": 0,
            "stdout": json.dumps(
                {
                    "ok": True,
                    "data": {
                        "isError": False,
                        "structuredContent": {"pid": 99, "dispatched": True, "terminated": True},
                    },
                }
            ),
            "stderr": "",
        },
    )
    world.recovery_data["quit"] = ref(world.quit)
    with pytest.raises(ContractError, match="foreign"):
        target.adopt(world.recapture(), host=world.host)


def test_unknown_prior_project_observation_refuses(world: World) -> None:
    world.intent_data["evidence"] = [ref(world.capability)]
    with pytest.raises(ContractError, match="saved project observation"):
        target.adopt(world.recapture(), host=world.host)


def test_unobserved_provider_artifact_refuses(world: World) -> None:
    world.observation_data["providerEvidence"] = [ref(world.before)]
    with pytest.raises(ContractError, match="capability"):
        target.adopt(world.recapture(), host=world.host)


def test_no_new_handoff_is_fabricated(world: World) -> None:
    world.grant.path.write_bytes(
        encode(
            {
                "version": 1,
                "kind": "native-gui-idle-handoff",
                "released": False,
                "target": {"pid": 23, "birth": "new-birth"},
            }
        )
    )
    world.queue_data["guiGrant"] = ref(current(world.grant.path))
    with pytest.raises(ContractError, match="grant"):
        target.adopt(world.recapture(), host=world.host)


def test_original_input_mutation_during_adoption_refuses(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_publish = target._publish

    def changed(
        request: target.Request,
        prepared: Checkpoint,
        inputs: tuple[Pin, ...],
        continuation: target.Continuation,
    ) -> Path:
        result = real_publish(request, prepared, inputs, continuation)
        world.output.path.write_bytes(b"changed during adoption")
        return result

    monkeypatch.setattr(target, "_publish", changed)
    with pytest.raises(ContractError, match="metadata/code changed"):
        target.adopt(target.load(world.request), host=world.host)


def test_actual_directory_replacement_is_detected(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_publish = target._publish

    def replaced(
        request: target.Request,
        prepared: Checkpoint,
        inputs: tuple[Pin, ...],
        continuation: target.Continuation,
    ) -> Path:
        result = real_publish(request, prepared, inputs, continuation)
        world.new.rename(world.root / "retained-new")
        world.new.mkdir()
        return result

    monkeypatch.setattr(target, "_publish", replaced)
    with pytest.raises(ContractError, match="directory identity"):
        target.adopt(target.load(world.request), host=world.host)


def test_cli_adoption_and_verify_use_explicit_request(
    world: World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(fcp_native_queue, "SystemHost", lambda: world.host)
    assert target.main(["adopt", str(world.request.path), "--sha256", world.request.sha256]) == 0
    assert capsys.readouterr().out.endswith("/new/continuation.json\n")
    assert target.main(["verify", str(world.request.path), "--sha256", world.request.sha256]) == 0
    assert capsys.readouterr().out.endswith("/new/continuation.json\n")


@pytest.mark.parametrize("parts", [[4, 4], [], [False, 5], "4,5"])
def test_malformed_original_order_refuses(world: World, parts: object) -> None:
    world.request_data["orderedParts"] = parts
    with pytest.raises(ContractError):
        target.load(world.save("request.json", world.request_data))


def test_linked_evidence_is_rejected_without_reading_alias(world: World) -> None:
    saved_original = world.root / "saved-retained.json"
    world.saved.path.rename(saved_original)
    world.saved.path.symlink_to(saved_original)
    with pytest.raises(ContractError, match="canonical"):
        target.adopt(target.load(world.request), host=world.host)


def test_original_operation_changes_cannot_hide_in_new_config(world: World) -> None:
    world.queue_data["parts"] = [
        {"assemblyPart": 5, "project": "Changed project", "operations": []},
        {"assemblyPart": 8, "project": "Future project 8", "operations": []},
    ]
    with pytest.raises(ContractError, match="operation bindings"):
        target.adopt(world.recapture(), host=world.host)


def test_completed_generation_missing_lock_is_never_repaired(world: World) -> None:
    request = target.load(world.request)
    target.adopt(request, host=world.host)
    (world.new / ".lock").unlink()
    with pytest.raises(ContractError, match="lock is missing"):
        target.open_generation(request, host=world.host)
    assert not (world.new / ".lock").exists()


def test_closed_journal_hash_chain_requires_every_prior_part(world: World) -> None:
    old = target._row(world.request_data["oldJournal"])
    old["predecessors"] = []
    world.request_data["oldJournal"] = old
    (world.old / "part-4-00.json").unlink()
    (world.old / "part-4-01.json").unlink()
    (world.old / "part-4-02.json").unlink()
    (world.old / "part-4-03.json").unlink()
    (world.old / "part-4-04.json").unlink()
    with pytest.raises(ContractError, match="missing"):
        target.adopt(target.load(world.save("request.json", world.request_data)), host=world.host)


def test_future_data_cannot_dispatch(world: World) -> None:
    world.future_data["canDispatch"] = True
    with pytest.raises(ContractError, match="non-dispatchable"):
        target.adopt(world.recapture(), host=world.host)


def test_recovery_original_target_provenance_refuses(world: World) -> None:
    world.intent_data["pid"] = 99
    with pytest.raises(ContractError, match="original target"):
        target.adopt(world.recapture(), host=world.host)


def test_incomplete_completed_provenance_cannot_open(world: World) -> None:
    request = target.load(world.request)
    target.adopt(request, host=world.host)
    path = world.new / "continuation.json"
    receipt = decode(path.read_bytes())
    receipt["inputs"] = [ref(world.saved)]
    path.write_bytes(encode(receipt))
    with pytest.raises(ContractError, match="input closure"):
        target.open_generation(request, host=world.host)
