"""Exact unlink contracts on tiny synthetic files, never production media."""

import io
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from replay import boot_duplicate_retirement as target
from replay import compressed_part_retirement as kernel
from replay import fcp_native_admission as admission
from replay import part_retirement_verification_batch as batch
from replay import part_retirement_verification_readonly as readonly
from replay.core import ReplayError
from replay.native_sequence import ContractError
from replay.sequence_adapters import reference
from replay.sequence_journal import Pin, decode, encode, publish
from replay.stages import execution_lock
from replay.workflow import load_stages
from replay.workflow import main as workflow_main


class SyntheticGuards:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.fail_post = False

    def before(self, plan: target.Plan) -> None:
        if not plan.source.is_file():
            raise AssertionError("Original before-domain requires source")
        self.events.append("before")

    def after(self, plan: target.Plan) -> Pin:
        if plan.source.exists() or plan.source.is_symlink():
            raise AssertionError("Post-domain must follow actual absence")
        self.events.append("after")
        if self.fail_post:
            raise ContractError("Synthetic post-domain interruption")
        path = plan.root / "post.json"
        path.write_bytes(encode({"phase": "after-delete"}))
        return Pin(path, reference(path)["sha256"])

    def unused(self, source: Path) -> None:
        self.events.append("unused")

    def birth(self, pid: int) -> str:
        return "synthetic-birth"

    def now_ns(self) -> int:
        return 100


class RetirementTests(unittest.TestCase):
    def fresh_resume(self) -> Pin:
        fresh = self.root / "fresh-owner-disposition.json"
        raw = decode(self.disposition.read_bytes())
        raw["owner"] = {"pid": 456, "birth": "new-owner"}
        raw["observedAtNs"] = kernel.MAX_AGE_NS + 1000
        fresh.write_bytes(encode(raw))
        path = self.root / "resume-request.json"
        path.write_bytes(
            encode(
                {
                    "version": 1,
                    "kind": "exact-boot-post-only-resume",
                    "request": reference(self.request),
                    "operation": reference(self.root / "unlink-operation.json"),
                    "source": decode(self.retirement.read_bytes())["source"],
                    "survivorIdentity": list(kernel._identity(self.survivor)),
                    "archive": [reference(ref.path) for ref in self.plan.archive],
                    "code": [reference(ref.path) for ref in self.plan.code],
                    "leases": [str(path) for path in self.plan.leases],
                    "disposition": reference(fresh),
                }
            )
        )
        return Pin(path, reference(path)["sha256"])

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(dir=kernel.ROOT / "edits/work")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.mov"
        self.survivor = self.root / "survivor.mov"
        self.source.write_bytes(b"identical synthetic media")
        self.survivor.write_bytes(b"identical synthetic media")
        self.evidence = self.root / "viewer-v15.json"
        self.evidence.write_bytes(b"{}")
        self.retirement = self.root / "retirement.json"
        self.retirement.write_bytes(
            encode(
                {
                    "kind": "verified-staging-retirement",
                    "stagingRetirementAuthorized": True,
                    "deletionPerformed": False,
                    "part": 2,
                    "source": reference(self.source),
                    "movie": reference(self.survivor),
                    "sourceIdentity": list(kernel._identity(self.source)[:4]),
                    "destinationIdentity": list(kernel._identity(self.survivor)[:4]),
                    "publication": reference(self.evidence),
                    "completion": reference(self.evidence),
                    "frameProof": reference(self.evidence),
                    "intent": reference(self.evidence),
                    "config": reference(self.evidence),
                }
            )
        )
        self.verification = self.root / "verification.json"
        self.verification.write_bytes(encode({"retirement": reference(self.retirement)}))
        self.disposition = self.root / "disposition.json"
        self.disposition.write_bytes(
            encode(
                {
                    "version": 1,
                    "source": reference(self.source),
                    "owner": {"pid": 123, "birth": "synthetic-birth"},
                    "ownerReleased": True,
                    "observedAtNs": 100,
                    "references": {
                        name: {
                            "scope": "explicit owner scope",
                            "resolved": True,
                            "requiresMaterializedSource": False,
                            "evidence": [reference(self.evidence)],
                        }
                        for name in kernel.SCOPES
                    },
                }
            )
        )
        self.leases = self.root / "leases"
        self.leases.mkdir()
        self.request = self.root / "request.json"
        self.request.write_bytes(
            encode(
                {
                    "version": 1,
                    "kind": "exact-boot-duplicate-retirement",
                    "root": str(self.root),
                    "boot": str(self.root),
                    "retirement": reference(self.retirement),
                    "verification": reference(self.verification),
                    "mandatoryBefore": reference(self.evidence),
                    "disposition": reference(self.disposition),
                    "archive": [reference(self.evidence)],
                    "code": [reference(path) for path in target.code_paths()],
                    "leases": [str(self.leases)],
                }
            )
        )
        self.plan = target.load(self.request)
        self.guards = SyntheticGuards()

    def test_success_preserves_survivor_and_literal_evidence(self) -> None:
        result = target.commit(self.plan, self.guards)
        self.assertFalse(self.source.exists())
        self.assertEqual(self.survivor.read_bytes(), b"identical synthetic media")
        self.assertEqual(self.evidence.name, "viewer-v15.json")
        self.assertEqual(self.guards.events, ["before", "unused", "after"])
        self.assertTrue(decode(result.read_bytes())["sourceAbsent"])
        target.commit(self.plan, self.guards)
        self.assertEqual(self.guards.events, ["before", "unused", "after", "after"])

    def test_verifier_never_repairs_post_checkpoint(self) -> None:
        result = target.commit(self.plan, self.guards)
        before = {path.name: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
        with (
            patch.object(batch, "run") as producer,
            self.assertRaises((FileNotFoundError, ReplayError, ContractError)),
        ):
            target.verify(self.plan)
        producer.assert_not_called()
        self.assertEqual(
            before, {path.name: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
        )
        post = self.root / "post.json"
        ref = batch.Reference(post, reference(post)["sha256"])
        request = readonly.Request(ref, ref, ref, ref, ())
        part = batch.Part(
            2,
            batch.Reference(self.verification, reference(self.verification)["sha256"]),
            self.source,
        )
        with (
            patch.object(readonly, "load", return_value=request),
            patch.object(batch, "_manifest", return_value=("after-delete", (part,), ())),
            patch.object(readonly, "verify") as verifier,
            patch.object(batch, "run") as producer,
        ):
            self.assertEqual(target.verify(self.plan), result)
            verifier.assert_called_once_with(request)
            producer.assert_not_called()
        with (
            patch.object(readonly, "load", return_value=request),
            patch.object(batch, "_manifest", return_value=("before-delete", (part,), ())),
            self.assertRaises(ContractError),
        ):
            target.verify(self.plan)

    def test_post_failure_resumes_without_second_unlink(self) -> None:
        self.guards.fail_post = True
        with self.assertRaisesRegex(ContractError, "interruption"):
            target.commit(self.plan, self.guards)
        self.assertFalse(self.source.exists())
        self.assertFalse((self.root / "released.json").exists())
        self.guards.fail_post = False
        with patch.object(self.guards, "before", side_effect=AssertionError("No repeated unlink")):
            target.commit(self.plan, self.guards)
        self.assertEqual(self.guards.events, ["before", "unused", "after", "after"])

    def test_expired_dead_owner_reentry_uses_separate_fresh_observation(self) -> None:
        old_request = self.request.read_bytes()
        old_disposition = self.disposition.read_bytes()
        self.guards.fail_post = True
        with self.assertRaisesRegex(ContractError, "interruption"):
            target.commit(self.plan, self.guards)
        resume = self.fresh_resume()
        original_operation = (self.root / "unlink-operation.json").read_bytes()
        self.guards.fail_post = False

        def birth(pid: int) -> str:
            if pid == 123:
                raise ProcessLookupError("Original owner really exited")
            return "new-owner"

        with (
            patch.object(self.guards, "birth", side_effect=birth),
            patch.object(self.guards, "now_ns", return_value=kernel.MAX_AGE_NS + 1000),
        ):
            with self.assertRaises(ProcessLookupError):
                target.commit(self.plan, self.guards)
            result = target.commit(self.plan, self.guards, resume=resume, post_only=True)
            self.assertTrue(result.is_file())
            target.commit(self.plan, self.guards, resume=resume, post_only=True)
        self.assertEqual(self.request.read_bytes(), old_request)
        self.assertEqual(self.disposition.read_bytes(), old_disposition)
        self.assertEqual((self.root / "unlink-operation.json").read_bytes(), original_operation)
        self.assertEqual(self.guards.events, ["before", "unused", "after", "after", "after"])
        self.assertFalse(self.source.exists())
        self.assertEqual(self.evidence.read_bytes(), b"{}")

    def test_resume_refuses_changed_binding_and_predelete_state(self) -> None:
        self.guards.fail_post = True
        with self.assertRaises(ContractError):
            target.commit(self.plan, self.guards)
        resume = self.fresh_resume()
        raw = decode(resume.path.read_bytes())
        raw["leases"] = []
        resume.path.write_bytes(encode(raw))
        changed = Pin(resume.path, reference(resume.path)["sha256"])
        with self.assertRaisesRegex(ContractError, "immutable original"):
            target.commit(self.plan, self.guards, resume=changed)
        self.source.write_bytes(b"reintroduced file must survive")
        with self.assertRaisesRegex(ContractError, "absence"):
            target.commit(self.plan, self.guards, resume=changed)
        self.assertEqual(self.source.read_bytes(), b"reintroduced file must survive")
        self.assertEqual(self.guards.events, ["before", "unused", "after"])

    def test_resume_graph_and_cli_keep_pin_and_post_only_boundary(self) -> None:
        self.guards.fail_post = True
        with self.assertRaises(ContractError):
            target.commit(self.plan, self.guards)
        resume = self.fresh_resume()
        graph = target.prepare_graph(
            self.request,
            self.root / "resume-workflow.json",
            Path("/synthetic/python"),
            resume=resume,
        )
        stages = load_stages(graph)
        self.assertIn("--resume-observation", stages[0].command)
        self.assertIn(resume.sha256, stages[0].verify_command)
        self.assertTrue(stages[0].adopt_existing)
        with (
            patch.object(target, "commit", return_value=self.root / "released.json") as commit,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                target.main(
                    [
                        "execute",
                        str(self.request),
                        "--resume-observation",
                        str(resume.path),
                        "--resume-sha256",
                        resume.sha256,
                    ]
                ),
                0,
            )
        self.assertEqual(commit.call_args.kwargs["resume"], resume)
        with self.assertRaisesRegex(ContractError, "exact pinned"):
            target.main(["execute", str(self.request), "--resume-observation", str(resume.path)])
        (self.root / "unlink-operation.json").rename(self.root / "retained-operation.json")
        with self.assertRaisesRegex(ContractError, "original unlink"):
            target.commit(self.plan, self.guards, post_only=True)

    def test_occupied_intent_never_retries_uncertain_delete(self) -> None:
        operation = self.root / "unlink-operation.json"
        operation.write_bytes(
            encode(
                {
                    "version": 1,
                    "kind": "exact-boot-unlink-operation",
                    "request": reference(self.request),
                    "sourceIdentity": list(kernel._identity(self.source)),
                    "survivorIdentity": list(kernel._identity(self.survivor)),
                    "parentIdentity": [1, 2],
                }
            )
        )
        with self.assertRaisesRegex(ContractError, "absence"):
            target.commit(self.plan, self.guards)
        self.assertTrue(self.source.exists())
        self.assertEqual(self.guards.events, [])

    def test_changed_archive_and_active_reader_refuse_before_unlink(self) -> None:
        with (
            patch.object(self.guards, "unused", side_effect=ContractError("active FD")),
            self.assertRaisesRegex(ContractError, "active FD"),
        ):
            target.commit(self.plan, self.guards)
        self.assertTrue(self.source.exists())
        self.evidence.write_bytes(b'{"changed":true}')
        with self.assertRaises(ContractError):
            target.commit(self.plan, self.guards)
        self.assertTrue(self.source.exists())

    def test_domain_mutation_and_named_race_refuse(self) -> None:
        with (
            patch.object(
                self.guards, "before", side_effect=lambda plan: plan.source.write_bytes(b"changed")
            ),
            self.assertRaisesRegex(ContractError, "changed media"),
        ):
            target.commit(self.plan, self.guards)
        self.assertTrue(self.source.exists())

    def test_unresolved_stale_and_wrong_owner_dispositions(self) -> None:
        with (
            patch.object(self.guards, "birth", return_value="reused-pid"),
            self.assertRaisesRegex(ContractError, "identity"),
        ):
            target.disposition(self.plan, self.guards)
        with (
            patch.object(self.guards, "now_ns", return_value=kernel.MAX_AGE_NS + 101),
            self.assertRaisesRegex(ContractError, "stale"),
        ):
            target.disposition(self.plan, self.guards)
        raw = decode(self.disposition.read_bytes())
        raw["references"] = {"current": {}}
        self.disposition.write_bytes(encode(raw))
        with self.assertRaisesRegex(ContractError, "scopes"):
            target.disposition(
                self.plan, self.guards, Pin(self.disposition, reference(self.disposition)["sha256"])
            )

    def test_invalid_scope_evidence_and_authority(self) -> None:
        raw = decode(self.disposition.read_bytes())
        raw["references"] = dict.fromkeys(
            kernel.SCOPES,
            {
                "scope": "explicit",
                "resolved": True,
                "requiresMaterializedSource": False,
                "evidence": [],
            },
        )
        self.disposition.write_bytes(encode(raw))
        with self.assertRaisesRegex(ContractError, "concrete"):
            target.disposition(
                self.plan, self.guards, Pin(self.disposition, reference(self.disposition)["sha256"])
            )
        raw["references"] = dict.fromkeys(kernel.SCOPES, {})
        self.disposition.write_bytes(encode(raw))
        with self.assertRaisesRegex(ContractError, "Unresolved"):
            target.disposition(
                self.plan, self.guards, Pin(self.disposition, reference(self.disposition)["sha256"])
            )
        raw["ownerReleased"] = False
        self.disposition.write_bytes(encode(raw))
        with self.assertRaisesRegex(ContractError, "release disposition"):
            target.disposition(
                self.plan, self.guards, Pin(self.disposition, reference(self.disposition)["sha256"])
            )

    def test_request_rejects_bad_schema_and_unarchived_originals(self) -> None:
        raw = decode(self.request.read_bytes())
        raw["archive"] = [reference(self.verification)]
        self.request.write_bytes(encode(raw))
        with self.assertRaisesRegex(ContractError, "archived"):
            target.load(self.request)
        raw["leases"] = [str(self.leases), str(self.leases)]
        self.request.write_bytes(encode(raw))
        with self.assertRaisesRegex(ContractError, "ownership roots"):
            target.load(self.request)
        raw["archive"] = []
        self.request.write_bytes(encode(raw))
        with self.assertRaisesRegex(ContractError, "archival"):
            target.load(self.request)
        raw["unknown"] = 1
        self.request.write_bytes(encode(raw))
        with self.assertRaisesRegex(ContractError, "Unsupported"):
            target.load(self.request)

    def test_wrong_media_identity_and_foreign_operation_refuse(self) -> None:
        self.source.write_bytes(b"different")
        with self.assertRaisesRegex(ContractError, "identities"):
            target.commit(self.plan, self.guards)
        (self.root / "unlink-operation.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ContractError, "another request"):
            target.commit(self.plan, self.guards)
        self.assertTrue(self.source.exists())

    def test_factory_and_cli_delegate_exact_request(self) -> None:
        graph = target.prepare_graph(
            self.request, self.root / "workflow.json", Path("/synthetic/python")
        )
        self.assertEqual(decode(graph.read_bytes())["version"], 1)
        self.assertEqual(len(load_stages(graph)), 1)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                workflow_main(
                    [str(graph), "--state", str(self.root / "workflow-state"), "--check"]
                ),
                0,
            )
        with (
            patch.object(target, "commit", return_value=self.root / "released.json") as call,
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(target.main(["execute", str(self.request)]), 0)
        self.assertEqual(call.call_args.args[0], self.plan)

    def test_code_scope_omission_and_incomplete_post_refuse(self) -> None:
        raw = decode(self.request.read_bytes())
        raw["code"] = [reference(self.evidence)]
        self.request.write_bytes(encode(raw))
        with self.assertRaisesRegex(ContractError, "import closure"):
            target.load(self.request)
        raw["code"] = []
        self.request.write_bytes(encode(raw))
        with self.assertRaisesRegex(ContractError, "code closure"):
            target.load(self.request)
        with (
            patch.object(target, "load", return_value=self.plan),
            patch.object(target, "_absent"),
            self.assertRaisesRegex(ContractError, "Complete original"),
        ):
            target.main(["verify", str(self.request)])

    def test_predelete_verify_never_calls_post_and_busy_lease_preserves_source(self) -> None:
        with (
            patch.object(target, "commit") as commit,
            self.assertRaisesRegex(ContractError, "absence"),
        ):
            target.main(["verify", str(self.request)])
        commit.assert_not_called()
        with execution_lock(self.leases, ".workflow") as acquired:
            self.assertTrue(acquired)
            with self.assertRaisesRegex(ContractError, "active"):
                target.commit(self.plan, self.guards)
        self.assertTrue(self.source.exists())
        self.assertEqual(self.guards.events, [])

    def test_last_namespace_swap_preserves_new_file_and_uncertain_intent(self) -> None:
        original = publish

        def publish_then_replace(path: Path, data: bytes) -> None:
            original(path, data)
            self.source.rename(self.root / "archived-synthetic.mov")
            self.source.write_bytes(b"new file occupying former source name")

        with (
            patch.object(target, "publish", side_effect=publish_then_replace),
            self.assertRaisesRegex(ContractError, "namespace changed"),
        ):
            target.commit(self.plan, self.guards)
        self.assertEqual(self.source.read_bytes(), b"new file occupying former source name")
        self.assertTrue((self.root / "unlink-operation.json").exists())
        self.assertFalse((self.root / "released.json").exists())
        self.assertEqual(self.guards.events, ["before", "unused"])

    def test_original_adapters_keep_before_after_domains_distinct(self) -> None:
        ref = batch.Reference(self.evidence, reference(self.evidence)["sha256"])
        request = readonly.Request(ref, ref, ref, ref, ())
        part = batch.Part(
            2,
            batch.Reference(self.verification, reference(self.verification)["sha256"]),
            self.source,
        )
        guards = target.OriginalGuards()
        with (
            patch.object(readonly, "load", return_value=request),
            patch.object(batch, "_manifest", return_value=("before-delete", (part,), ())),
            patch.object(readonly, "verify") as verify,
        ):
            guards.before(self.plan)
            verify.assert_called_once_with(request)
        with (
            patch.object(readonly, "load", return_value=request),
            patch.object(batch, "_manifest", return_value=("after-delete", (part,), ())),
            self.assertRaisesRegex(ContractError, "Mandatory"),
        ):
            guards.before(self.plan)
        manifest = self.root / "after-delete-manifest.json"
        output = self.root / "after-delete"
        output.mkdir()
        (output / "operation.json").write_text("{}")
        (output / "part-0002.json").write_text("{}")
        with (
            patch.object(
                batch, "prepare", side_effect=lambda *args, **kwargs: manifest.write_bytes(b"{}")
            ),
            patch.object(batch, "_manifest", return_value=("after-delete", (part,), ())),
            patch.object(batch, "run", return_value=self.evidence),
        ):
            self.assertEqual(guards.after(self.plan).path, self.evidence)
            self.assertEqual(guards.after(self.plan).path, self.evidence)
        with (
            patch.object(batch, "_manifest", return_value=("before-delete", (part,), ())),
            self.assertRaisesRegex(ContractError, "Occupied"),
        ):
            guards.after(self.plan)
        with (
            patch.object(kernel, "_unused", side_effect=ReplayError("active")),
            self.assertRaisesRegex(ReplayError, "active"),
        ):
            guards.unused(self.source)
        with (
            patch.object(admission.SystemHost, "birth", return_value="observed"),
            patch.object(time, "time_ns", return_value=77),
        ):
            self.assertEqual(guards.birth(123), "observed")
            self.assertEqual(guards.now_ns(), 77)


if __name__ == "__main__":
    unittest.main()
