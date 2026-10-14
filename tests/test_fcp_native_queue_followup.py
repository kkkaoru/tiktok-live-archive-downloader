import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from replay import fcp_native_queue as queue
from replay import fcp_native_queue_followup as adapter
from replay import fcp_part_followup as factory
from replay.artifact_json import _pin
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest


def write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def read(path: Path) -> dict[str, object]:
    return record(json.loads(path.read_text(encoding="utf-8")))


def objects(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


class Host:
    now = 2_000_000_000_000_000_000
    birth_value = "100:200"
    available = 10**12

    def birth(self, pid: int) -> str:
        return self.birth_value

    def now_ns(self) -> int:
        return self.now

    def free(self, path: Path) -> int:
        return self.available

    def floor(self, path: Path) -> int:
        return 30 * 1024**3


@dataclass(frozen=True)
class Case:
    root: Path
    request: Path
    part: int
    movie: Path
    next_movie: Path
    original_queue: Path


@dataclass(frozen=True)
class AliasCase:
    case: Case
    alias: Path
    original: Path
    producer: Path


def xml(path: Path, *, name: str, clean: Path) -> Path:
    path.write_text(
        '<fcpxml><resources><asset hasVideo="1"><media-rep src="'
        f'{clean.as_uri()}"/></asset></resources><library><event><project name="{name}">'
        "<sequence/></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    return path


def operations(root: Path, *, project: str, source: Path, movie: Path) -> list[dict[str, object]]:
    values: dict[str, dict[str, object]] = {
        "import": {"path": str(source)},
        "windows": {},
        "project": {"project": project, "library": "Synthetic", "event": "Recorded"},
        "export": {
            "project": project,
            "directory": str(movie.parent),
            "fileName": movie.name,
            "codec": "Apple ProRes 422 LT",
            "format": "videoAndAudio",
            "allowForeground": True,
        },
        "background-open": {"menuPath": ["Window", "Background Tasks"]},
        "background-wait": {
            "window": {"title": "Background Tasks"},
            "condition": "absent",
            "element": {"role": "AXProgressIndicator"},
            "timeoutSeconds": 30,
        },
        "background": {"window": {"title": "Background Tasks"}},
        "background-close": {
            "window": {"title": "Background Tasks"},
            "element": {"subrole": "AXCloseButton"},
            "action": "press",
        },
    }
    return [
        {
            "phase": phase,
            "request": pin(
                write(
                    root / f"{movie.stem}-{phase}.json",
                    {"bundleID": "com.apple.FinalCutApp", **values[phase]},
                )
            ),
        }
        for phase in queue.PHASES
    ]


def descriptor(
    root: Path,
    *,
    events: Path,
    inventory: Path,
    parts: list[dict[str, object]],
    core: list[dict[str, str]],
) -> Path:
    target = {"pid": 123, "birth": "100:200", "bundleID": "com.apple.FinalCutApp"}
    grant = write(
        root / f"{events.name}-grant.json",
        {
            "version": 1,
            "kind": "native-gui-idle-handoff",
            "released": True,
            "eventRoot": str(events),
            "previousOwner": "Synthetic owner",
            "target": target,
        },
    )
    binary = root / "executor"
    binary.write_bytes(b"synthetic binary")
    return write(
        root / f"{events.name}-queue.json",
        {
            "version": 1,
            "kind": "executor-native-part-queue",
            "provider": "native.synthetic",
            "corePins": core,
            "guiGrant": pin(grant),
            "executor": pin(binary),
            "eventRoot": str(events),
            "bootVolume": str(root),
            "inventory": str(inventory),
            "target": target,
            "parts": parts,
            "limits": {
                "maxCalls": 50,
                "callSeconds": 60,
                "observationSeconds": 30,
                "peakAdditionalBytes": 100,
                "minimumFreeBytes": 1,
                "capacity": 1,
            },
        },
    )


@pytest.fixture(params=[(2, 3), (3, 5)])
def case(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    assert isinstance(request.param, tuple)
    part, count = request.param
    assert isinstance(part, int) and isinstance(count, int)
    original, clean, voice = (
        tmp_path / "original.mov",
        tmp_path / "clean.mov",
        tmp_path / "voice.wav",
    )
    original.write_bytes(b"original")
    clean.write_bytes(b"clean local repair")
    voice.write_bytes(b"processed voice")
    semantic = xml(tmp_path / "semantic.fcpxml", name="Full", clean=clean)
    sources = [
        xml(tmp_path / f"part-{number}.fcpxml", name=f"Project {number}", clean=clean)
        for number in range(1, count + 1)
    ]
    evidence = write(tmp_path / "repair-evidence.json", {"version": 1})
    frame_probe = tmp_path / "synthetic-frame-probe"
    frame_probe.write_bytes(b"#!/bin/sh\nexit 0\n")
    frame_probe.chmod(0o700)
    partition = write(
        tmp_path / "partition.json",
        {
            "version": 1,
            "source": pin(semantic),
            "parts": [{**pin(source), "globalStart": "0s"} for source in sources],
        },
    )
    common = {
        "version": 1,
        "partition": pin(partition),
        "voice": pin(voice),
        "voiceReceipt": pin(evidence),
    }
    staging = write(
        tmp_path / "staging.json",
        {
            **common,
            "parts": [
                {
                    "movie": str(tmp_path / f"movie-{number}.mov"),
                    "proof": str(tmp_path / f"proof-{number}.json"),
                }
                for number in range(1, count + 1)
            ],
        },
    )
    published = write(
        tmp_path / "published.json",
        {
            **common,
            "parts": [
                {
                    "movie": str(tmp_path / f"ssd-{number}.mov"),
                    "proof": str(tmp_path / f"ssd-proof-{number}.json"),
                }
                for number in range(1, count + 1)
            ],
        },
    )
    cleanup = write(
        tmp_path / "cleanup.json",
        {
            "version": 1,
            "consumer": {**pin(semantic), "kind": "fcpxml"},
            "assemblyManifest": pin(evidence),
            "assemblyReceipt": pin(evidence),
            "cleanIndex": pin(evidence),
            "voiceReceipt": pin(evidence),
        },
    )
    monkeypatch.setattr(factory, "clean_source", lambda _: (original, clean, Fraction(3)))
    monkeypatch.setattr(factory, "_cleanup_inputs", lambda _: (evidence,))
    movie, next_movie = tmp_path / f"movie-{part}.mov", tmp_path / f"movie-{part + 1}.mov"
    events = tmp_path / "events"
    events.mkdir()
    successor_events = tmp_path / "successor-events"
    successor_events.mkdir()
    capacity = write(
        tmp_path / "capacity.json",
        {
            "capturedAtNs": Host.now,
            "maximumAgeSeconds": 30,
            "activeRenders": 0,
            "stagingCount": 0,
            "stagingBytes": 0,
            "maximumStagingCount": 4,
            "maximumStagingBytes": 16000,
        },
    )
    parts = [
        {
            "assemblyPart": number,
            "libraryBasename": "Synthetic.fcpbundle",
            "project": f"Project {number}",
            "movie": str(tmp_path / f"movie-{number}.mov"),
            "operations": operations(
                tmp_path,
                project=f"Project {number}",
                source=sources[number - 1],
                movie=tmp_path / f"movie-{number}.mov",
            ),
        }
        for number in (part, part + 1)
    ]
    original_queue = descriptor(
        tmp_path,
        events=events,
        inventory=capacity,
        parts=parts,
        core=[pin(source) for source in sources],
    )
    descriptor(
        tmp_path,
        events=successor_events,
        inventory=capacity,
        parts=parts[1:],
        core=[pin(source) for source in sources],
    )
    movie.write_bytes(b"synthetic native movie bytes")
    export_request = tmp_path / f"movie-{part}-export.json"
    native_export = write(
        events / "part-0001-export.native.json",
        {
            "ok": True,
            "error": None,
            "data": {
                "isError": False,
                "structuredContent": {
                    "pid": 123,
                    "retrySafe": False,
                    "finalCut": {
                        "project": {"name": f"Project {part}"},
                        "vocabularyVerified": True,
                        "language": "en",
                        "export": {
                            "codec": "Apple ProRes 422 LT",
                            "format": "Video and Audio",
                            "action": "Save only",
                            "outputPath": str(movie),
                            "foregroundRestored": True,
                            "completionVerified": False,
                        },
                    },
                },
            },
        },
    )
    native_background = write(
        events / "part-0001-background-0002.native.json",
        {
            "ok": True,
            "data": {
                "isError": False,
                "structuredContent": {
                    "pid": 123,
                    "truncated": False,
                    "elements": [
                        {"role": "AXStaticText", "value": "Sharing"},
                        *[
                            {"identifier": "_NS:166", "value": "Idle", "path": [index]}
                            for index in range(9)
                        ],
                    ],
                },
            },
        },
    )
    ready = write(
        events / "followup-ready.json",
        {
            "queueSHA256": file_digest(original_queue),
            "queueOrdinal": 1,
            "assemblyPart": part,
            "project": f"Project {part}",
            "movie": str(movie),
            "processedParts": 1,
            "remainingParts": 1,
            "completionVerified": False,
            "qualityApproved": False,
            "exportRequest": pin(export_request),
            "exportResponse": pin(native_export),
            "background": pin(native_background),
        },
    )
    request_path = write(
        tmp_path / "adapter-request.json",
        {
            "version": 1,
            "kind": "native-queue-followup-request",
            "queue": pin(original_queue),
            "ready": pin(ready),
            "queueOrdinal": 1,
            "assemblyPart": part,
            "assembly": pin(staging),
            "publishedAssembly": pin(published),
            "sourceCleanup": pin(cleanup),
            "exportRequest": pin(export_request),
            "exportResponse": pin(native_export),
            "background": pin(native_background),
            "ffmpeg": pin(evidence),
            "ffprobe": pin(frame_probe),
            "audioSampleRate": 48000,
            "audioChannels": 2,
            "receiptOutput": str(tmp_path / "publication.json"),
        },
    )
    monkeypatch.setattr(queue, "GUI_LEASE_ROOT", tmp_path)
    return Case(tmp_path, request_path, part, movie, next_movie, original_queue)


@pytest.fixture
def alias_case(case: Case, monkeypatch: pytest.MonkeyPatch) -> AliasCase:
    original = case.root / "original.mov"
    alias = case.root / "historical-original.mov"
    alias.symlink_to(original)
    producer = write(
        case.root / "historical-producer.json",
        {"source": str(alias), "sourceSHA256": file_digest(original)},
    )
    manifest = write(
        case.root / "historical-manifest.json", {"chunks": [{"producer": pin(producer)}]}
    )
    cleanup = case.root / "cleanup.json"
    data = read(cleanup)
    data["assemblyManifest"] = pin(manifest)
    write(cleanup, data)
    request = read(case.request)
    request["sourceCleanup"] = pin(cleanup)
    request["approvedOriginal"] = pin(original)
    write(case.request, request)
    monkeypatch.setattr(
        factory, "clean_source", lambda _: (alias, case.root / "clean.mov", Fraction(3))
    )
    monkeypatch.setattr(factory, "_cleanup_inputs", lambda _: (manifest, producer))
    return AliasCase(case, alias, original, producer)


def test_approved_alias_reaches_real_factory_without_changing_history(
    alias_case: AliasCase,
) -> None:
    case = alias_case.case
    background = case.root / "events/part-0001-background-0002.native.json"
    original_time = background.stat().st_mtime_ns
    producer_bytes = alias_case.producer.read_bytes()
    request = read(case.request)
    graph = adapter.prepare(case.request, case.root / "alias-preparation")
    config = read(graph.parent.parent / "followup-config.json")
    assert config["approvedOriginal"] == request["approvedOriginal"]
    operation = read(graph.parent / "operation.json")
    paths = {record(item)["path"] for item in objects(operation["inputs"])}
    assert str(alias_case.original) in paths
    assert str(alias_case.alias) not in paths
    preparation = read(graph.parent.parent / "preparation.json")
    assert record(preparation["inputs"])[str(alias_case.original)] == file_digest(
        alias_case.original
    )
    assert preparation["qualityApproved"] is False
    assert background.stat().st_mtime_ns == original_time
    assert alias_case.producer.read_bytes() == producer_bytes


@pytest.mark.parametrize(
    "mode",
    ["omitted", "malformed", "extra-field", "wrong-sha", "linked", "wrong-target", "missing"],
)
def test_approved_alias_rejects_invalid_reference_or_domain(
    alias_case: AliasCase, mode: str
) -> None:
    case = alias_case.case
    request = read(case.request)
    if mode == "omitted":
        del request["approvedOriginal"]
    elif mode == "malformed":
        request["approvedOriginal"] = None
    elif mode == "extra-field":
        request["approvedOriginal"] = {**pin(alias_case.original), "trusted": True}
    elif mode == "wrong-sha":
        request["approvedOriginal"] = {"path": str(alias_case.original), "sha256": "0" * 64}
    elif mode == "linked":
        request["approvedOriginal"] = {
            "path": str(alias_case.alias),
            "sha256": file_digest(alias_case.original),
        }
    elif mode == "wrong-target":
        other = case.root / "same-byte-unapproved.mov"
        other.write_bytes(alias_case.original.read_bytes())
        request["approvedOriginal"] = pin(other)
    else:
        alias_case.alias.unlink()
    write(case.request, request)
    producer_bytes = alias_case.producer.read_bytes()
    with pytest.raises(ReplayError):
        adapter.prepare(case.request, case.root / "alias-refused")
    assert alias_case.producer.read_bytes() == producer_bytes
    assert not (case.root / "alias-refused/prepared/workflow.json").exists()
    assert case.movie.read_bytes() == b"synthetic native movie bytes"


def test_optional_approval_keeps_unknown_request_fields_rejected(alias_case: AliasCase) -> None:
    case = alias_case.case
    request = read(case.request)
    request["approvalOverride"] = request["approvedOriginal"]
    write(case.request, request)
    with pytest.raises(ReplayError, match="contract"):
        adapter.prepare(case.request, case.root / "unknown-refused")
    assert not (case.root / "unknown-refused").exists()


def test_approved_input_change_is_detected_by_adapter_terminal_pins(alias_case: AliasCase) -> None:
    inputs = adapter.Inputs()
    adapter._preparation(alias_case.case.request, inputs)
    alias_case.original.write_bytes(b"changed before terminal adoption")
    with pytest.raises(ReplayError, match="changed"):
        inputs.verify()


def test_shared_factory_rejects_same_byte_alias_swap_during_hash(
    alias_case: AliasCase, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = alias_case.original
    other = alias_case.case.root / "same-byte-target.mov"
    other.write_bytes(original.read_bytes())
    checked = _pin

    def swapped(value: object) -> Path:
        result = checked(value)
        if result == original:
            alias_case.alias.unlink()
            alias_case.alias.symlink_to(other)
        return result

    monkeypatch.setattr(factory, "_pin", swapped)
    with pytest.raises(ReplayError, match="during validation"):
        adapter.prepare(alias_case.case.request, alias_case.case.root / "swapped-refused")
    assert not (alias_case.case.root / "swapped-refused/prepared/workflow.json").exists()


@pytest.fixture
def completed(case: Case, monkeypatch: pytest.MonkeyPatch) -> tuple[Case, Path, list[str]]:
    graph = adapter.prepare(case.request, case.root / "adapter")
    prep = case.root / "adapter/preparation.json"
    config = read(case.root / "adapter/followup-config.json")
    stage = record(objects(read(graph)["stages"])[0])
    output_paths = [Path(str(item)) for item in objects(stage["outputs"])]
    staging_proof, completion, publication_config, ssd, proof, publication, terminal = output_paths
    ssd.write_bytes(case.movie.read_bytes())
    write(staging_proof, {"part": case.part, "movie": pin(case.movie)})
    write(
        completion,
        {
            "part": case.part,
            "completionVerified": True,
            "fullAVDecodeExitCode": 0,
            "shareObservedIdle": True,
            "qualityApproved": False,
            "movie": pin(case.movie),
            "frameProof": pin(staging_proof),
        },
    )
    intent = write(case.root / "publication.intent.json", {"completion": pin(completion)})
    write(proof, {"part": case.part, "movie": pin(ssd)})
    write(publication_config, {"completion": pin(completion)})
    write(
        publication,
        {
            "version": 1,
            "kind": "published-native-part",
            "source": pin(case.movie),
            "movie": pin(ssd),
            "completion": pin(completion),
            "frameProof": pin(proof),
            "intent": pin(intent),
            "qualityApproved": False,
        },
    )
    operation = graph.parent / "operation.json"
    write(
        terminal,
        {
            "version": 1,
            "kind": "verified-integrated-native-followup",
            "part": case.part,
            "operation": pin(operation),
            "qualityApproved": False,
            "inputs": {
                record(ref)["path"]: record(ref)["sha256"]
                for ref in objects(read(operation)["inputs"])
            },
            "outputs": [pin(path) for path in output_paths[:-1]],
        },
    )
    retirement = write(
        case.root / "retirement.json",
        {
            "version": 1,
            "kind": "verified-staging-retirement",
            "part": case.part,
            "stagingRetirementAuthorized": True,
            "deletionPerformed": False,
            "source": pin(case.movie),
            "movie": pin(ssd),
            "frameProof": pin(proof),
            "publication": pin(publication),
        },
    )
    case.movie.unlink()
    status = case.root / "postdelete.status"
    status.write_text("0\n", encoding="utf-8")
    audit = write(
        case.root / "postdelete.json",
        {
            "version": 1,
            "part": case.part,
            "stagingPath": str(case.movie),
            "stagingAbsent": True,
            "historyUnchanged": True,
            "retirementSHA256": file_digest(retirement),
            "qualityApproved": False,
            "survivingMovie": str(ssd),
            "survivingSHA256": file_digest(ssd),
            "prepareConsumerPostDeleteExitCodes": [0, 0, 0],
        },
    )
    assembly = record(config["publishedAssembly"])["path"]
    consumer = write(
        case.root / "consumer.json",
        {
            "version": 1,
            "stages": [
                {
                    "name": "assemble",
                    "inputs": [assembly, str(retirement), str(ssd), str(proof)],
                    "command": [
                        "python",
                        "-m",
                        "replay.fcp_part_assembly",
                        "assemble",
                        assembly,
                        "whole.fcpxml",
                    ],
                    "verify_command": [
                        "python",
                        "-m",
                        "replay.fcp_part_assembly",
                        "verify-assembly",
                        assembly,
                        "whole.fcpxml",
                    ],
                }
            ],
        },
    )
    release = write(
        case.root / "release.json",
        {
            "version": 1,
            "kind": "native-queue-nextpart-release-request",
            "preparation": pin(prep),
            "terminal": pin(terminal),
            "publication": pin(publication),
            "retirement": pin(retirement),
            "postDeleteStatus": pin(status),
            "retirementAudit": pin(audit),
            "history": [pin(completion), pin(proof), pin(intent)],
            "consumer": pin(consumer),
            "capacity": pin(case.root / "capacity.json"),
            "successorQueue": pin(case.root / "successor-events-queue.json"),
            "successorOrdinal": 1,
        },
    )
    calls: list[str] = []

    def published(
        path: Path, proof_path: Path, *, part: int, retirement: Path | None = None
    ) -> tuple[Path, Path]:
        assert part == case.part
        assert retirement == case.root / "retirement.json"
        assert proof_path == proof
        assert read(path)["source"] == read(completion)["movie"]
        calls.append("published")
        return completion, intent

    monkeypatch.setattr(adapter, "verify_published", published)
    return case, release, calls


def test_real_factory_connection_keeps_native_times_and_independent_part(case: Case) -> None:
    background = case.root / "events/part-0001-background-0002.native.json"
    old_time = background.stat().st_mtime_ns
    graph = adapter.prepare(case.request, case.root / "adapter")
    config = read(graph.parent.parent / "followup-config.json")
    assert config["part"] == case.part
    assert background.stat().st_mtime_ns == old_time
    assert record(objects(read(graph)["stages"])[0])["name"] == "followup"
    assert len(objects(record(objects(read(graph)["stages"])[0])["outputs"])) == 7
    assert read(graph.parent.parent / "preparation.json")["qualityApproved"] is False
    with pytest.raises(ReplayError, match="occupied"):
        adapter.prepare(case.request, case.root / "adapter")


def test_readonly_release_requires_domain_without_resubmitting(
    completed: tuple[Case, Path, list[str]],
) -> None:
    case, request, calls = completed
    before = request.read_bytes()
    result = adapter.verify_release(request, host=Host())
    assert result.eligible is True
    assert result.quality_approved is False
    assert result.assembly_part == case.part
    assert result.successor_movie == case.next_movie
    assert result.successor_project == f"Project {case.part + 1}"
    assert calls == ["published"]
    assert request.read_bytes() == before
    assert not case.movie.exists()
    assert not case.next_movie.exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("assemblyPart", 1),
        ("queueOrdinal", 2),
        ("version", 2),
        ("audioSampleRate", True),
        ("audioChannels", 65),
    ],
)
def test_bad_adapter_identity_is_rejected_before_outputs(
    case: Case, key: str, value: object
) -> None:
    raw = read(case.request)
    raw[key] = value
    write(case.request, raw)
    with pytest.raises(ReplayError):
        adapter.prepare(case.request, case.root / "adapter")
    assert not (case.root / "adapter").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("completionVerified", True),
        ("qualityApproved", True),
        ("assemblyPart", 1),
        ("remainingParts", 4),
        ("movie", "/invalid.mov"),
    ],
)
def test_false_ready_claim_does_not_generate_config(case: Case, key: str, value: object) -> None:
    ready = case.root / "events/followup-ready.json"
    raw = read(ready)
    raw[key] = value
    write(ready, raw)
    config = read(case.request)
    config["ready"] = pin(ready)
    write(case.request, config)
    with pytest.raises(ReplayError, match="Ready"):
        adapter.prepare(case.request, case.root / "adapter")


def test_native_error_is_preserved_and_not_retried(case: Case) -> None:
    native = case.root / "events/part-0001-export.native.json"
    write(native, {"ok": False, "error": "uncertain export with an actual file"})
    config, ready = read(case.request), read(case.root / "events/followup-ready.json")
    config["exportResponse"] = pin(native)
    ready["exportResponse"] = pin(native)
    write(case.root / "events/followup-ready.json", ready)
    config["ready"] = pin(case.root / "events/followup-ready.json")
    write(case.request, config)
    with pytest.raises(ReplayError, match="did not succeed"):
        adapter.prepare(case.request, case.root / "adapter")
    assert case.movie.read_bytes() == b"synthetic native movie bytes"


@pytest.mark.parametrize(
    "key,value",
    [
        ("stagingAbsent", False),
        ("historyUnchanged", False),
        ("qualityApproved", True),
        ("prepareConsumerPostDeleteExitCodes", [0, 1]),
    ],
)
def test_retirement_assertions_alone_cannot_release(
    completed: tuple[Case, Path, list[str]], key: str, value: object
) -> None:
    case, request, calls = completed
    audit = case.root / "postdelete.json"
    raw = read(audit)
    raw[key] = value
    write(audit, raw)
    release = read(request)
    release["retirementAudit"] = pin(audit)
    write(request, release)
    with pytest.raises(ReplayError, match="audit"):
        adapter.verify_release(request, host=Host())
    assert calls == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("activeRenders", 1),
        ("capturedAtNs", 1),
        ("maximumAgeSeconds", 31),
        ("stagingCount", 4),
        ("stagingBytes", 16000),
    ],
)
def test_capacity_failure_cannot_allocate(
    completed: tuple[Case, Path, list[str]], key: str, value: object
) -> None:
    case, request, calls = completed
    capacity = case.root / "capacity.json"
    raw = read(capacity)
    raw[key] = value
    write(capacity, raw)
    release = read(request)
    release["capacity"] = pin(capacity)
    write(request, release)
    with pytest.raises(ReplayError):
        adapter.verify_release(request, host=Host())
    assert calls == []


@pytest.mark.parametrize("effect", ["movie", "reservation", "owner", "gui"])
def test_reserved_successor_effects_never_replay(
    completed: tuple[Case, Path, list[str]], effect: str
) -> None:
    case, request, calls = completed
    paths = {
        "movie": case.next_movie,
        "reservation": case.next_movie.with_name(
            f".{case.next_movie.name}.native-queue-reservation.json"
        ),
        "owner": case.root / "successor-events/owner.json",
        "gui": case.root / ".fcp-native-queue-owner.json",
    }
    paths[effect].write_bytes(b"retained uncertain effect")
    with pytest.raises(ReplayError, match="reserved"):
        adapter.verify_release(request, host=Host())
    assert calls == []


def test_changed_history_is_not_rewritten(completed: tuple[Case, Path, list[str]]) -> None:
    case, request, calls = completed
    history = case.root / "publication.intent.json"
    history.write_bytes(history.read_bytes() + b" ")
    with pytest.raises((ReplayError, ValueError)):
        adapter.verify_release(request, host=Host())
    assert calls == []


def test_staging_reappearance_invalidates_retirement(
    completed: tuple[Case, Path, list[str]],
) -> None:
    case, request, calls = completed
    case.movie.write_bytes(b"changed reappearance")
    with pytest.raises(ReplayError, match="physical"):
        adapter.verify_release(request, host=Host())
    assert calls == []


def test_domain_failure_stops_release(
    completed: tuple[Case, Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    case, request, calls = completed

    def fail(path: Path, semantic: Path) -> tuple[Path, int, bool]:
        raise ReplayError("Changed cleanup producer")

    monkeypatch.setattr(factory, "validate_cleanup", fail)
    with pytest.raises(ReplayError, match="cleanup producer"):
        adapter.verify_release(request, host=Host())
    assert calls == ["published"]
    assert not case.next_movie.exists()


def test_no_symlink_or_duplicate_json_workaround(case: Case) -> None:
    linked = case.root / "linked-request.json"
    linked.symlink_to(case.request)
    with pytest.raises(ReplayError, match="canonical"):
        adapter.prepare(linked, case.root / "adapter")
    case.request.write_text('{"version":1,"version":1}', encoding="utf-8")
    with pytest.raises(ReplayError, match="Duplicate"):
        adapter.prepare(case.request, case.root / "adapter")


@pytest.mark.parametrize(
    "field,value", [("version", 2), ("part", 1), ("qualityApproved", True), ("inputs", {})]
)
def test_incomplete_full_terminal_stops_release(
    completed: tuple[Case, Path, list[str]], field: str, value: object
) -> None:
    case, request, calls = completed
    terminal = case.root / "adapter/prepared/verified-followup.json"
    raw = read(terminal)
    raw[field] = value
    write(terminal, raw)
    release = read(request)
    release["terminal"] = pin(terminal)
    write(request, release)
    with pytest.raises(ReplayError, match="terminal"):
        adapter.verify_release(request, host=Host())
    assert calls == []


@pytest.mark.parametrize(
    "field,value",
    [("inputs", []), ("verify_command", []), ("command", []), ("inputs", "not a list")],
)
def test_consumer_without_required_dependency_stops_release(
    completed: tuple[Case, Path, list[str]], field: str, value: object
) -> None:
    case, request, calls = completed
    consumer = case.root / "consumer.json"
    raw = read(consumer)
    row = record(objects(raw["stages"])[0])
    row[field] = value
    write(consumer, raw)
    release = read(request)
    release["consumer"] = pin(consumer)
    write(request, release)
    with pytest.raises(ReplayError, match="Consumer|consumer"):
        adapter.verify_release(request, host=Host())
    assert calls == []


def test_nonzero_postdelete_status_is_not_complete(completed: tuple[Case, Path, list[str]]) -> None:
    case, request, calls = completed
    status = case.root / "postdelete.status"
    status.write_text("1\n", encoding="utf-8")
    raw = read(request)
    raw["postDeleteStatus"] = pin(status)
    write(request, raw)
    with pytest.raises(ReplayError, match="Post-delete"):
        adapter.verify_release(request, host=Host())
    assert calls == []


def test_missing_or_duplicate_history_does_not_pass(
    completed: tuple[Case, Path, list[str]],
) -> None:
    case, request, calls = completed
    raw = read(request)
    raw["history"] = []
    write(request, raw)
    with pytest.raises(ReplayError, match="inventory"):
        adapter.verify_release(request, host=Host())
    raw["history"] = [
        pin(case.root / "publication.intent.json"),
        pin(case.root / "publication.intent.json"),
    ]
    write(request, raw)
    with pytest.raises(ReplayError, match="unique"):
        adapter.verify_release(request, host=Host())
    assert calls == []


def test_gui_birth_and_live_disk_are_checked(completed: tuple[Case, Path, list[str]]) -> None:
    case, request, calls = completed
    host = Host()
    host.birth_value = "restarted"
    with pytest.raises(ReplayError, match="process identity"):
        adapter.verify_release(request, host=host)
    host.birth_value = "100:200"
    host.available = 1
    with pytest.raises(ReplayError, match="capacity"):
        adapter.verify_release(request, host=host)
    assert calls == []
    assert not case.next_movie.exists()


def test_guard_rejects_cleanup_mutation_between_domain_checks(
    completed: tuple[Case, Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    case, request, calls = completed

    def changed(
        path: Path, proof_path: Path, *, part: int, retirement: Path | None = None
    ) -> tuple[Path, Path]:
        clean = case.root / "clean.mov"
        clean.write_bytes(b"mutated after initial cryptographic read")
        return case.root / "adapter/prepared/completion.json", case.root / "publication.intent.json"

    monkeypatch.setattr(adapter, "verify_published", changed)
    with pytest.raises(ReplayError, match="changed"):
        adapter.verify_release(request, host=Host())
    assert calls == []
    assert not case.next_movie.exists()


def test_pin_shape_conflict_budget_and_metadata_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = adapter.Inputs()
    valid = {"path": str(tmp_path / "one.json"), "sha256": "a" * 64}
    inputs.add(valid)
    with pytest.raises(ReplayError, match="exact"):
        inputs.add({"path": str(tmp_path / "bad.json"), "sha256": "wrong"})
    with pytest.raises(ReplayError, match="Conflicting"):
        inputs.add({"path": str(tmp_path / "one.json"), "sha256": "b" * 64})
    monkeypatch.setattr(adapter, "MAX_INPUTS", 1)
    with pytest.raises(ReplayError, match="budget"):
        inputs.add({"path": str(tmp_path / "two.json"), "sha256": "a" * 64})
    with pytest.raises(ReplayError, match="edit evidence"):
        inputs.read(adapter.Ref(adapter.ROOT / "README.md", "a" * 64))
    with pytest.raises(ReplayError, match="repository work"):
        adapter.prepare(tmp_path / "absent.json", adapter.ROOT / "new-output")


def test_missing_queue_project_and_member_are_explicit_errors() -> None:
    with pytest.raises(ReplayError, match="member"):
        adapter._member({"parts": []}, 1)
    with pytest.raises(ReplayError, match="project"):
        adapter._part_identity({"assemblyPart": 2, "movie": "/movie.mov"})


def test_staging_reappearing_during_validation_cannot_release(
    completed: tuple[Case, Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    case, request, calls = completed

    def reappeared(
        path: Path, proof_path: Path, *, part: int, retirement: Path | None = None
    ) -> tuple[Path, Path]:
        case.movie.write_bytes(b"late staging reappearance")
        return case.root / "adapter/prepared/completion.json", case.root / "publication.intent.json"

    monkeypatch.setattr(adapter, "verify_published", reappeared)
    with pytest.raises(ReplayError, match="physical"):
        adapter.verify_release(request, host=Host())
    assert calls == []
    assert not case.next_movie.exists()


@pytest.mark.parametrize("field", ["code", "queue", "ready"])
def test_prep_cannot_omit_its_independent_pins(
    completed: tuple[Case, Path, list[str]], field: str
) -> None:
    case, request, calls = completed
    preparation = case.root / "adapter/preparation.json"
    raw = read(preparation)
    paths = {
        "code": str(Path(adapter.__file__).resolve()),
        "queue": str(case.original_queue),
        "ready": str(case.root / "events/followup-ready.json"),
    }
    del record(raw["inputs"])[paths[field]]
    write(preparation, raw)
    release = read(request)
    release["preparation"] = pin(preparation)
    write(request, release)
    with pytest.raises(ReplayError, match="code pin|original queue"):
        adapter.verify_release(request, host=Host())
    assert calls == []


def test_private_and_download_references_reject_before_open() -> None:
    inputs = adapter.Inputs()
    with pytest.raises(ReplayError, match="private"):
        inputs.add({"path": str(adapter.ROOT / "private/never-opened.json"), "sha256": "a" * 64})
    with pytest.raises(ReplayError, match="download|canonical"):
        inputs.add({"path": str(adapter.ROOT / "downloads/never-opened.mov"), "sha256": "a" * 64})


def test_external_native_request_rejects_before_queue_loader(case: Case) -> None:
    raw = read(case.original_queue)
    member = record(objects(raw["parts"])[0])
    operation = record(objects(member["operations"])[0])
    operation["request"] = {"path": str(adapter.ROOT / "README.md"), "sha256": "a" * 64}
    write(case.original_queue, raw)
    config = read(case.request)
    config["queue"] = pin(case.original_queue)
    write(case.request, config)
    with pytest.raises(ReplayError, match="Native requests"):
        adapter.prepare(case.request, case.root / "adapter")
    assert not (case.root / "adapter").exists()


@pytest.fixture
def ready_cli(case: Case) -> tuple[Case, list[str]]:
    original = read(case.request)
    template = write(
        case.root / "template.json",
        {
            key: value
            for key, value in original.items()
            if key
            not in {
                "version",
                "kind",
                "queue",
                "ready",
                "queueOrdinal",
                "assemblyPart",
                "exportRequest",
                "exportResponse",
                "background",
            }
        },
    )
    pump = write(
        case.root / "pump.json",
        {
            "version": 1,
            "kind": "native-delivery-pump",
            "queue": original["queue"],
        },
    )
    checkpoint = write(
        case.root / "native-complete.json",
        {
            "version": 1,
            "kind": "native-delivery-pump-checkpoint",
            "config": pin(pump),
            "inputs": [pin(template)],
            "outputs": [original["ready"]],
            "qualityApproved": False,
        },
    )
    return case, [
        "prepare-from-ready",
        str(template),
        str(case.original_queue),
        str(case.root / "events/followup-ready.json"),
        str(case.root / "bound-request.json"),
        str(case.root / "bound-followup"),
        "--native-complete",
        str(checkpoint),
    ]


def test_ready_cli_prepares_existing_full_av_factory(ready_cli: tuple[Case, list[str]]) -> None:
    case, argv = ready_cli
    assert adapter.main(argv) == 0
    assert read(case.root / "bound-request.json")["assemblyPart"] == case.part
    assert (case.root / "bound-followup/prepared/workflow.json").is_file()
    assert adapter.main(argv) == 1


def test_ready_cli_rejects_wrong_checkpoint(ready_cli: tuple[Case, list[str]]) -> None:
    case, argv = ready_cli
    checkpoint = read(case.root / "native-complete.json")
    checkpoint["outputs"] = []
    write(case.root / "native-complete.json", checkpoint)
    assert adapter.main(argv) == 1
    assert not (case.root / "bound-request.json").exists()


def test_ready_cli_rejects_changed_clock_binding(ready_cli: tuple[Case, list[str]]) -> None:
    case, argv = ready_cli
    template = read(case.root / "template.json")
    cleanup = write(
        case.root / "wrong-cleanup.json",
        {
            **read(case.root / "cleanup.json"),
            "consumer": {**pin(case.root / "part-1.fcpxml"), "kind": "fcpxml"},
        },
    )
    template["sourceCleanup"] = pin(cleanup)
    write(case.root / "template.json", template)
    checkpoint = read(case.root / "native-complete.json")
    checkpoint["inputs"] = [pin(case.root / "template.json")]
    write(case.root / "native-complete.json", checkpoint)
    assert adapter.main(argv) == 1
    assert not (case.root / "bound-followup/prepared/workflow.json").exists()


def test_ready_cli_rejects_absent_output_parent(ready_cli: tuple[Case, list[str]]) -> None:
    case, argv = ready_cli
    argv[4] = str(case.root / "absent/request.json")
    assert adapter.main(argv) == 1


def test_ready_cli_rejects_unknown_template_field(ready_cli: tuple[Case, list[str]]) -> None:
    case, argv = ready_cli
    template = read(case.root / "template.json")
    template["unexpected"] = True
    write(case.root / "template.json", template)
    assert adapter.main(argv) == 1
