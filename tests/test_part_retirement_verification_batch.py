import fcntl
import hashlib
import json
import sys
import wave
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pytest

from replay import part_master_publication as publication
from replay import part_retirement_verification_batch as batch
from replay import whole_technical_batch as technical
from replay.core import ReplayError
from replay.fcp_partition import write_parts
from replay.input_hash_cache import InputHashCache, input_hash_context
from replay.recordings import record
from replay.stages import file_digest


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _save(path: Path, raw: dict[str, object]) -> Path:
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _identity(path: Path) -> list[int]:
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns]


@dataclass(frozen=True)
class Fixture:
    work: Path
    configs: tuple[Path, ...]
    sources: tuple[Path, ...]
    movies: tuple[Path, ...]
    voice: Path

    def manifest(self, phase: batch.Phase = "before-delete") -> Path:
        return batch.prepare(self.configs, self.work / "manifest.json", phase=phase)


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int = 2) -> Fixture:
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    monkeypatch.setattr(technical, "ROOT", tmp_path)
    work = tmp_path / "edits/work/test"
    work.mkdir(parents=True)
    code = tmp_path / "src/replay"
    code.mkdir(parents=True)
    extra_names = ("motion_caption.py", "caption_resource_inputs.py", "caption_style_defaults.py")
    for name in {*batch.CODE_NAMES, *technical.CODE_NAMES, *extra_names}:
        (code / name).write_text(f"# synthetic module {name}\n", encoding="utf-8")
    (code / "workflow.py").write_text("from replay import part_master_publication\n")
    (code / "part_master_publication.py").write_text("from replay import fcp_part_assembly\n")
    (code / "fcp_part_assembly.py").write_text("from replay import caption_font_supplier\n")
    (code / "caption_font_supplier.py").write_text(
        "from replay import motion_caption, caption_resource_inputs, caption_style_defaults\n"
    )
    semantic = work / "semantic.fcpxml"
    clips = "".join(
        f'<asset-clip offset="{index}/30s" start="{100 + index}s" duration="1/30s">'
        f'<title offset="{100 + index}s" start="3600s" duration="1/30s">'
        f"<text>cue {index}</text></title></asset-clip>"
        for index in range(count)
    )
    semantic.write_text(
        '<fcpxml version="1.11"><resources><format id="f" frameDuration="1/30s" '
        'width="1080" height="1920"/></resources><library><event><project>'
        f'<sequence format="f" duration="{count}/30s" tcStart="0s"><spine>{clips}'
        "</spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    partition = write_parts(
        semantic,
        work / "partition",
        target_duration=Fraction(1, 30),
        maximum_titles=1,
    )
    voice = work / "voice.wav"
    with wave.open(str(voice), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\0\0" * (1600 * count))
    voice_receipt = _save(
        work / "voice-receipt.json",
        {
            "outputSHA256": _sha(voice),
            "sampleRate": 48000,
            "channels": 1,
            "outputSamples": 1600 * count,
        },
    )
    tool = work / "ffprobe"
    tool.write_bytes(b"synthetic executable, never invoked")
    tool.chmod(0o700)
    assembly = work / "assembly.json"
    sources: list[Path] = []
    movies: list[Path] = []
    publication_paths: list[Path] = []
    completion_paths: list[Path] = []
    intent_paths: list[Path] = []
    proof_paths: list[Path] = []
    for number in range(1, count + 1):
        source = work / f"staging-{number}.mov"
        source.write_bytes(bytes([number]) * (1024 * (number + 1)))
        movie = work / f"published-{number}.mov"
        movie.write_bytes(source.read_bytes())
        original = _save(
            work / f"original-proof-{number}.json",
            {
                "version": 1,
                "part": number,
                "movie": _ref(source),
                "partition": _ref(partition),
                "audioIgnored": True,
                "observed": {
                    "streams": [
                        {
                            "codec_type": "video",
                            "width": 1080,
                            "height": 1920,
                            "start_pts": 0,
                            "avg_frame_rate": "30/1",
                            "duration_ts": 1,
                            "time_base": "1/30",
                            "nb_read_frames": "1",
                        }
                    ],
                    "frames": [{"best_effort_timestamp": 0}],
                },
                "probeProvenance": {"version": 1, "executable": _ref(tool), "decoderThreads": 0},
            },
        )
        completed = _save(
            work / f"completion-{number}.json",
            {
                "version": 1,
                "kind": "verified-native-part",
                "part": number,
                "completionVerified": True,
                "shareObservedIdle": True,
                "fullAVDecodeExitCode": 0,
                "movie": _ref(source),
                "inputs": {str(tool): _sha(tool), str(source): _sha(source)},
                "frameProof": _ref(original),
            },
        )
        intent = _save(
            work / f"intent-{number}.json",
            {
                "source": _ref(source),
                "destination": str(movie),
                "completion": _ref(completed),
            },
        )
        proof = _save(
            work / f"published-proof-{number}.json",
            {
                **record(json.loads(original.read_text())),
                "movie": _ref(movie),
                "sourceFrameProof": _ref(original),
                "publicationIntent": _ref(intent),
            },
        )
        published = _save(
            work / f"publication-{number}.json",
            {
                "version": 1,
                "kind": "published-native-part",
                "sourcePreserved": True,
                "source": _ref(source),
                "movie": _ref(movie),
                "completion": _ref(completed),
                "frameProof": _ref(proof),
                "intent": _ref(intent),
            },
        )
        sources.append(source)
        movies.append(movie)
        publication_paths.append(published)
        completion_paths.append(completed)
        intent_paths.append(intent)
        proof_paths.append(proof)
    _save(
        assembly,
        {
            "version": 1,
            "partition": _ref(partition),
            "voice": _ref(voice),
            "voiceReceipt": _ref(voice_receipt),
            "parts": [
                {"movie": str(movie), "proof": str(proof)}
                for movie, proof in zip(movies, proof_paths, strict=True)
            ],
        },
    )
    configs: list[Path] = []
    for index, source in enumerate(sources):
        number = index + 1
        retired = work / f"retirement-{number}.json"
        config = _save(
            work / f"retirement-config-{number}.json",
            {
                "version": 1,
                "retireStaging": True,
                "publication": _ref(publication_paths[index]),
                "assembly": _ref(assembly),
                "receiptOutput": str(retired),
            },
        )
        _save(
            retired,
            {
                "version": 1,
                "kind": "verified-staging-retirement",
                "part": number,
                "stagingRetirementAuthorized": True,
                "deletionPerformed": False,
                "assembly": _ref(assembly),
                "source": _ref(source),
                "movie": _ref(movies[index]),
                "sourceIdentity": _identity(source),
                "destinationIdentity": _identity(movies[index]),
                "publication": _ref(publication_paths[index]),
                "completion": _ref(completion_paths[index]),
                "frameProof": _ref(proof_paths[index]),
                "intent": _ref(intent_paths[index]),
                "config": _ref(config),
            },
        )
        configs.append(
            _save(work / f"verify-{number}.json", {"version": 1, "retirement": _ref(retired)})
        )
    return Fixture(work, tuple(configs), tuple(sources), tuple(movies), voice)


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    return _fixture(tmp_path, monkeypatch)


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("phase", ["before-delete", "after-delete"])
def test_real_domains_topologies_and_fresh_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    phase: batch.Phase,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, count)
    if phase == "after-delete":
        for source in fixture.sources:
            source.unlink()
    manifest = fixture.manifest(phase)
    inputs = record(json.loads(manifest.read_text()))["inputs"]
    assert isinstance(inputs, list)
    expected = {Path(str(record(ref)["path"])): record(ref)["sha256"] for ref in inputs}
    history = {path: path.read_bytes() for path in expected}
    output = fixture.work / "result"
    receipt = batch.run(manifest, _sha(manifest), output)
    raw = record(json.loads(receipt.read_text()))
    assert raw["deletionPerformed"] is False
    assert raw["releaseDecision"] is None
    assert raw["qualityApproved"] is False
    assert raw["physicalIOBytes"] is None
    assert raw["sourceAbsenceObserved"] == (phase == "after-delete")
    assert raw["hashReads"] == 2 * (len(expected) + 1)
    assert raw["bytesRead"] == 2 * (
        sum(path.stat().st_size for path in expected) + manifest.stat().st_size
    )
    assert isinstance(raw["cacheHits"], int) and raw["cacheHits"] > count
    original_receipt = receipt.read_bytes()
    assert batch.run(manifest, _sha(manifest), output, resume=True) == receipt
    assert receipt.read_bytes() == original_receipt
    assert {path: path.read_bytes() for path in expected} == history


def test_metadata_prepare_never_hashes_media(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = file_digest
    forbidden = {*fixture.movies, *fixture.sources, fixture.voice}

    def check(path: Path) -> str:
        assert path not in forbidden
        return original(path)

    monkeypatch.setattr(batch, "file_digest", check)
    fixture.manifest()


@pytest.mark.parametrize(
    "value", [None, "relative.json", "", "/tmp/private.json", "\0", "/Users/no/evidence.json"]
)
def test_reject_unauthorized_paths(value: object) -> None:
    with pytest.raises(ReplayError):
        batch._path(value)


def test_escaping_symlink_is_rejected(fixture: Fixture) -> None:
    link = fixture.work / "link"
    link.symlink_to(fixture.movies[0])
    with pytest.raises(ReplayError, match="symlinks"):
        batch._path(str(link))


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"path": "relative", "sha256": "x"},
        {"path": "/tmp/x", "sha256": "0" * 64},
        {"path": "/tmp/x", "sha256": False},
    ],
)
def test_invalid_reference(value: object) -> None:
    with pytest.raises(ReplayError):
        batch._ref(value)


@pytest.mark.parametrize("phase", [None, "deleted", True])
def test_phase_is_explicit(phase: object) -> None:
    with pytest.raises(ReplayError):
        batch._phase(phase)


def test_source_phase_requires_actual_state(fixture: Fixture) -> None:
    with pytest.raises(ReplayError, match="absence"):
        fixture.manifest("after-delete")
    fixture.sources[0].unlink()
    with pytest.raises(ReplayError, match="original staging"):
        fixture.manifest()


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate",
        "wrongpart",
        "badkind",
        "badversion",
        "badrows",
        "badref",
        "extraField",
    ],
)
def test_changed_manifest_inventory_is_refused(fixture: Fixture, change: str) -> None:
    manifest = fixture.manifest()
    raw = record(json.loads(manifest.read_text()))
    inputs = raw["inputs"]
    parts = raw["parts"]
    assert isinstance(inputs, list) and isinstance(parts, list)
    if change == "missing":
        inputs.pop()
    elif change == "extra":
        extra = fixture.work / "extra"
        extra.write_bytes(b"not consumed")
        inputs.append(_ref(extra))
    elif change == "duplicate":
        inputs.append(inputs[0])
    elif change == "wrongpart":
        record(parts[0])["part"] = 900
        parts[0] = {**record(parts[0]), "part": 900}
    elif change == "badkind":
        raw["kind"] = "other"
    elif change == "badversion":
        raw["version"] = True
    elif change == "badrows":
        raw["parts"] = []
    elif change == "badref":
        parts[0] = {"part": False, "verification": _ref(fixture.configs[0])}
    else:
        raw["unknown"] = True
    _save(manifest, raw)
    output = fixture.work / "result"
    with pytest.raises(ReplayError):
        batch.run(manifest, _sha(manifest), output)
    assert not output.exists()


def test_tampered_media_rejected_before_outputs(fixture: Fixture) -> None:
    manifest = fixture.manifest()
    fixture.movies[0].write_bytes(b"changed")
    with pytest.raises(ReplayError, match="digest"):
        batch.run(manifest, _sha(manifest), fixture.work / "result")
    assert not (fixture.work / "result").exists()


@pytest.mark.parametrize("target", ["movie", "voice", "code", "source", "operation"])
def test_mutation_between_parts_is_detected(
    fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    manifest = fixture.manifest()
    original = publication.verify_retirement
    output = fixture.work / "result"
    selected = {
        "movie": fixture.movies[0],
        "voice": fixture.voice,
        "source": fixture.sources[0],
        "code": batch.ROOT / "src/replay/core.py",
        "operation": output / "operation.json",
    }[target]

    def mutate(config: Path) -> None:
        original(config)
        selected.write_bytes(b"mutation")

    monkeypatch.setattr(publication, "verify_retirement", mutate)
    with pytest.raises((ReplayError, json.JSONDecodeError)):
        batch.run(manifest, _sha(manifest), output)
    assert not (output / "receipt.json").exists()


def test_terminal_fresh_bytes_and_checkpoint_guard(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = fixture.manifest()
    original = InputHashCache.verify
    output = fixture.work / "result"

    def mutate(cache: InputHashCache) -> dict[str, str]:
        result = original(cache)
        (output / "part-0001.json").write_bytes(b"changed checkpoint")
        return result

    monkeypatch.setattr(InputHashCache, "verify", mutate)
    with pytest.raises(ReplayError, match="checkpoint changed"):
        batch.run(manifest, _sha(manifest), output)


def test_partial_resume_rechecks_every_domain(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = fixture.manifest()
    original = publication.verify_retirement
    calls: list[Path] = []

    def fail_second(config: Path) -> None:
        calls.append(config)
        if config == fixture.configs[1]:
            raise ReplayError("controlled interrupted readonly verifier")
        original(config)

    monkeypatch.setattr(publication, "verify_retirement", fail_second)
    output = fixture.work / "result"
    with pytest.raises(ReplayError, match="interrupted"):
        batch.run(manifest, _sha(manifest), output)
    checkpoint = (output / "part-0001.json").read_bytes()

    def verify(config: Path) -> None:
        calls.append(config)
        original(config)

    monkeypatch.setattr(publication, "verify_retirement", verify)
    batch.run(manifest, _sha(manifest), output, resume=True)
    assert calls == [*fixture.configs, *fixture.configs]
    assert (output / "part-0001.json").read_bytes() == checkpoint


@pytest.mark.parametrize("target", ["occupied", "operation", "part", "receipt"])
def test_uncertain_or_changed_outputs_refuse(fixture: Fixture, target: str) -> None:
    manifest = fixture.manifest()
    output = fixture.work / "result"
    if target == "occupied":
        output.mkdir()
    else:
        batch.run(manifest, _sha(manifest), output)
        path = {"operation": "operation.json", "part": "part-0001.json", "receipt": "receipt.json"}[
            target
        ]
        _save(output / path, {"version": 1, "changed": True})
    with pytest.raises(ReplayError):
        batch.run(manifest, _sha(manifest), output, resume=True)


def test_nested_context_refuses_without_output(fixture: Fixture) -> None:
    manifest = fixture.manifest()
    cache = InputHashCache([manifest])
    cache.prime()
    with input_hash_context(cache), pytest.raises(ReplayError, match="Nested"):
        batch.run(manifest, _sha(manifest), fixture.work / "result")
    assert not (fixture.work / "result").exists()


def test_cli_is_readonly_and_never_calls_producers(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(config: Path) -> Path:
        pytest.fail(f"producer called: {config}")

    monkeypatch.setattr(publication, "publish", forbidden)
    monkeypatch.setattr(publication, "prepare_retirement", forbidden)
    manifest = fixture.work / "manifest.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch", "prepare", str(manifest), *map(str, fixture.configs), "--phase", "before-delete"],
    )
    batch.main()
    monkeypatch.setattr(
        sys, "argv", ["batch", "run", str(manifest), _sha(manifest), str(fixture.work / "result")]
    )
    batch.main()
    assert all(path.exists() for path in fixture.sources)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "empty",
        "version",
        "part",
        "publication",
        "replacement",
        "inputs",
        "partition",
        "crossRevision",
        "rows",
    ],
)
def test_metadata_discovery_rejects_unsupported_or_malformed(fixture: Fixture, case: str) -> None:
    if case in {"duplicate", "empty"}:
        configs = (fixture.configs[0], fixture.configs[0]) if case == "duplicate" else ()
        with pytest.raises(ReplayError):
            batch.prepare(configs, fixture.work / "manifest.json", phase="before-delete")
        return
    config = record(json.loads(fixture.configs[0].read_text()))
    retired_path = Path(str(record(config["retirement"])["path"]))
    retired = record(json.loads(retired_path.read_text()))
    if case == "version":
        config["version"] = True
    elif case == "part":
        retired["part"] = False
    elif case == "publication":
        retired["completion"] = retired["intent"]
    elif case == "replacement":
        retired["source"] = {**record(retired["source"]), "sha256": "0" * 64}
    elif case == "inputs":
        completed_path = Path(str(record(retired["completion"])["path"]))
        completed = record(json.loads(completed_path.read_text()))
        completed["inputs"] = {str(fixture.voice): "not a digest"}
        _save(completed_path, completed)
    else:
        proof_path = Path(str(record(retired["frameProof"])["path"]))
        proof = record(json.loads(proof_path.read_text()))
        if case == "partition":
            proof["partition"] = retired["config"]
        elif case == "crossRevision":
            proof["crossRevisionReuse"] = {}
        else:
            assembly_path = Path(str(record(retired["assembly"])["path"]))
            assembly = record(json.loads(assembly_path.read_text()))
            partition_path = Path(str(record(assembly["partition"])["path"]))
            partition = record(json.loads(partition_path.read_text()))
            partition["parts"] = []
            _save(partition_path, partition)
        _save(proof_path, proof)
    _save(retired_path, retired)
    config["retirement"] = _ref(retired_path)
    _save(fixture.configs[0], config)
    with pytest.raises(ReplayError):
        fixture.manifest()


def test_unique_keys_bounded_metadata_and_conflicting_inputs(fixture: Fixture) -> None:
    fixture.configs[0].write_text('{"version":1,"version":1}')
    with pytest.raises(ReplayError, match="duplicate keys"):
        fixture.manifest()
    inventory = batch.Inventory()
    inventory.add(_ref(fixture.voice))
    with pytest.raises(ReplayError, match="Conflicting"):
        inventory.add({"path": str(fixture.voice), "sha256": "0" * 64})
    with pytest.raises(ReplayError, match="bounded"):
        batch._read(batch.Reference(fixture.work, "0" * 64))


def test_budget_and_byte_guard(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch, "MAX_INPUTS", 1)
    with pytest.raises(ReplayError, match="budget"):
        batch.Inventory().add(_ref(fixture.voice))
    with pytest.raises(ReplayError, match="bytes changed"):
        batch._read(batch.Reference(fixture.configs[0], "0" * 64))


def test_generated_lock_is_exclusive(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = fixture.manifest()
    original = fcntl.flock

    def busy(fd: int, flags: int) -> None:
        raise BlockingIOError("owned")

    monkeypatch.setattr(fcntl, "flock", busy)
    with pytest.raises(ReplayError, match="already owned"):
        batch.run(manifest, _sha(manifest), fixture.work / "result")
    monkeypatch.setattr(fcntl, "flock", original)


def test_source_reappears_after_last_domain(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    for source in fixture.sources:
        source.unlink()
    manifest = fixture.manifest("after-delete")
    original = publication.verify_retirement

    def reappear(config: Path) -> None:
        original(config)
        fixture.sources[0].write_bytes(b"reappeared")

    monkeypatch.setattr(publication, "verify_retirement", reappear)
    with pytest.raises(ReplayError, match="absence"):
        batch.run(manifest, _sha(manifest), fixture.work / "result")


def test_overflow_stays_fresh_and_complete(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = fixture.manifest()

    def small_cache(paths: Iterable[Path], *, allow_uncached_overflow: bool) -> InputHashCache:
        return InputHashCache(paths, max_files=1, allow_uncached_overflow=allow_uncached_overflow)

    monkeypatch.setattr(batch, "InputHashCache", small_cache)
    receipt = batch.run(manifest, _sha(manifest), fixture.work / "result")
    raw = record(json.loads(receipt.read_text()))
    inputs = record(raw["inputs"])
    assert isinstance(raw["hashReads"], int) and raw["hashReads"] > len(inputs) * 2
    assert raw["initialAndTerminalBytesVerified"] is True


def test_unknown_outputs_and_changed_adopted_checkpoint_refuse(
    fixture: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = fixture.manifest()
    output = fixture.work / "result"
    batch.run(manifest, _sha(manifest), output)
    extra = output / "uncertain.partial"
    extra.write_bytes(b"preserved uncertain output")
    with pytest.raises(ReplayError, match="additional artifacts"):
        batch.run(manifest, _sha(manifest), output, resume=True)
    extra.unlink()
    original = publication.verify_retirement

    def change(config: Path) -> None:
        original(config)
        (output / "part-0001.json").write_bytes(b"changed during reuse")

    monkeypatch.setattr(publication, "verify_retirement", change)
    with pytest.raises((ReplayError, json.JSONDecodeError)):
        batch.run(manifest, _sha(manifest), output, resume=True)


def test_transitive_reader_change_rejects_old_manifest(fixture: Fixture) -> None:
    manifest = fixture.manifest()
    raw = record(json.loads(manifest.read_text()))
    inputs = raw["inputs"]
    assert isinstance(inputs, list)
    names = {Path(str(record(ref)["path"])).name for ref in inputs}
    assert {
        "motion_caption.py",
        "caption_resource_inputs.py",
        "caption_style_defaults.py",
    }.issubset(names)
    transitive = batch.ROOT / "src/replay/caption_style_defaults.py"
    transitive.write_text("# changed transitive style contract\n", encoding="utf-8")
    output = fixture.work / "result"
    with pytest.raises(ReplayError, match="exact current input closure"):
        batch.run(manifest, _sha(manifest), output)
    assert not output.exists()


def test_actual_public_closure_contains_callback_and_motion_readers() -> None:
    # Read current public code only, without metadata, recordings or media hashes.
    paths = technical.reader_code_paths()
    names = {path.name for path in paths}
    assert {
        "part_master_publication.py",
        "fcp_part_assembly.py",
        "caption_font_supplier.py",
        "motion_caption.py",
        "caption_resource_inputs.py",
        "caption_style_defaults.py",
        "delivery_render.py",
        "native_export_receipt.py",
    }.issubset(names)
    assert all(path.is_relative_to(technical.ROOT / "src/replay") for path in paths)
