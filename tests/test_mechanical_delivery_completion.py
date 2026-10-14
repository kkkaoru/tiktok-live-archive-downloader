"""Synthetic local evidence only; no media/app render or approval is performed."""

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import ModuleType

import pytest

from replay import aac_presentation_acceptance as aac
from replay import delivery_completion_v2 as standard
from replay import mechanical_delivery_completion as adapter
from replay import native_delivery_evidence as evidence
from replay import source_caption_binding
from replay.core import ReplayError
from replay.edit_completion import REQUIRED_STAGES
from replay.input_hash_snapshot import InputHashSnapshot
from replay.stages import StageError


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": digest(path)}


def row(path: Path) -> dict[str, object]:
    value = path.stat()
    return {
        **ref(path),
        "bytes": value.st_size,
        "device": value.st_dev,
        "inode": value.st_ino,
        "mtimeNs": value.st_mtime_ns,
        "ctimeNs": value.st_ctime_ns,
    }


@dataclass
class Fixture:
    root: Path
    paths: dict[str, Path] = field(default_factory=dict)
    data: dict[str, object] = field(default_factory=dict)

    def save(self, name: str, value: object) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            path.write_text(json.dumps(value), encoding="utf-8")
        self.paths[name] = path
        return path

    def document(self, name: str) -> dict[str, object]:
        return evidence.mapping(json.loads(self.paths[name].read_text()))

    @property
    def movie(self) -> Path:
        return self.paths["candidate.mp4"]

    def freeze(self) -> dict[str, object]:
        inputs = [path for name, path in self.paths.items() if name != "mechanical.json"]
        self.data["currentInputs"] = [row(path) for path in (*inputs, *adapter.code_inputs())]
        self.data["externalInputs"] = {str(path): digest(path) for path in adapter.code_inputs()}
        return ref(self.save("mechanical.json", self.data))

    def context(self) -> adapter.Context:
        self.freeze()
        return adapter._bind(self.data, self.root, self.movie)

    def assess(self, *, fresh: bool = False) -> dict[str, object]:
        return adapter.assess(
            self.freeze(),
            root=self.root,
            movie=self.movie,
            verify_content=fresh,
            content_byte_budget=10_000_000,
        )

    def producer(
        self, role: str, output: Path, inputs: list[Path] | None = None
    ) -> dict[str, object]:
        source = self.paths["source.mp4"]
        selected = inputs if inputs is not None else [source]
        stage: dict[str, object] = {
            "name": role,
            "command": ["synthetic", *map(str, selected), str(output)],
            "verify_command": ["synthetic-verify", str(output)],
            "inputs": list(map(str, selected)),
            "outputs": [str(output)],
            "timeout_seconds": 10,
        }
        definition = {
            "command": stage["command"],
            "verifyCommand": stage["verify_command"],
            "inputs": {str(path): digest(path) for path in selected},
            "outputs": stage["outputs"],
            "timeoutSeconds": 10,
            "workingDirectory": str(self.root),
        }
        fingerprint = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()
        return ref(
            self.save(
                f"producer-{role}.json",
                {
                    "role": role,
                    "workflow": ref(self.save(f"workflow-{role}.json", {"stages": [stage]})),
                    "stage": role,
                    "checkpoint": ref(
                        self.save(
                            f"checkpoint-{role}.json",
                            {
                                "status": "complete",
                                "fingerprint": fingerprint,
                                "outputs": {str(output): digest(output)},
                            },
                        )
                    ),
                    "inventory": ref(
                        self.save(
                            f"inventory-{role}.json",
                            {"inputs": {str(path): digest(path) for path in selected}},
                        )
                    ),
                    "outputs": {str(output): digest(output)},
                    "cwd": str(self.root),
                },
            )
        )


@pytest.fixture
def fixture(tmp_path: Path) -> Fixture:
    item = Fixture(tmp_path / "repo/edits/replay-7")
    source = item.save("source.mp4", b"synthetic source")
    movie = item.save("candidate.mp4", b"synthetic candidate")
    item.save("candidate.srt", b"synthetic canonical SRT")
    clean = item.save("clean.mov", b"synthetic clean")
    voice = item.save("voice.wav", b"synthetic voice")
    cuts = item.save("cuts.json", {})
    config = item.save(
        "edit-config.json",
        {
            "version": 1,
            "replay_id": "7",
            "source": str(source),
            "source_size": source.stat().st_size,
            "engine": str(item.root),
            "executor_scope": str(item.root),
            "stages": dict.fromkeys(REQUIRED_STAGES, True),
        },
    )
    render = item.save(
        "render.json",
        {
            "recording": item.root.name,
            "source": str(source),
            "storage": str(item.root),
            "work": str(item.root),
            "cuts": str(cuts),
            "audio": {"source": str(voice)},
            "rate": 30,
            "width": 1080,
            "height": 1920,
        },
    )
    outputs = {
        role: item.save(f"output-{role}.bin", role.encode())
        for role in (*REQUIRED_STAGES, "compose", "join", "encode", "technical")
    }
    outputs["bgm_removal"] = voice
    outputs["compose"] = item.save("part.mov", b"synthetic native part")
    outputs["join"] = item.save("joined.mov", b"synthetic joined")
    model = item.save("modelInput.wav", b"independent original PCM")
    weights = item.save("model/weights.bin", b"synthetic separation weights")
    model_config = item.save("model/config.json", {"model": "synthetic"})
    separation = item.save(
        "separation.json",
        {
            "version": 1,
            "source": ref(source),
            "modelInput": ref(model),
            "firstPassVoice": ref(voice),
            "producerOutput": ref(voice),
            "modelPins": {str(weights): digest(weights), str(model_config): digest(model_config)},
        },
    )
    grid = item.save(
        "grid.json",
        {
            "version": 1,
            "kind": "frame-grid-media-inventory",
            "media": ref(clean),
            "frames": 3,
            "ticks": 3,
            "timebase": 30,
            "pts": [0, 1, 2],
        },
    )
    request = item.save(
        "native-request.json",
        {
            "recipe": {
                "clips": [
                    {"sourcePath": str(clean), "startSeconds": 0.0, "durationSeconds": 2 / 30}
                ],
                "video": {},
            }
        },
    )
    native = outputs["compose"]
    tool = item.save("ffprobe", b"synthetic tool")
    part: dict[str, object] = {
        "request": ref(request),
        "response": ref(
            item.save(
                "native-response.json",
                {
                    "outputPath": str(native),
                    "frameCount": 2,
                    "expectedDurationSeconds": 2 / 30,
                    "actualDurationSeconds": 2 / 30,
                },
            )
        ),
        "movie": ref(native),
        "tool": ref(tool),
        "frames": 2,
        "proof": ref(
            item.save(
                "part-proof.json",
                {
                    "version": 1,
                    "part": 1,
                    "partition": ref(request),
                    "movie": ref(native),
                    "probeProvenance": {"version": 1, "decoderThreads": 0, "executable": ref(tool)},
                    "observed": {
                        "streams": [
                            {
                                "codec_type": "video",
                                "width": 1080,
                                "height": 1920,
                                "start_pts": 0,
                                "avg_frame_rate": "30",
                                "time_base": "1/30",
                                "nb_read_frames": "2",
                                "duration_ts": 2,
                            }
                        ],
                        "frames": [{"best_effort_timestamp": 0}, {"best_effort_timestamp": 1}],
                    },
                },
            )
        ),
        "sourceCaptionBinding": ref(
            item.save("binding.json", {"consumer": {**ref(request), "kind": "native-request"}})
        ),
        "selections": [
            {
                "media": ref(clean),
                "proof": ref(grid),
                "start": "0",
                "frames": 2,
                "nativeDuration": "1/15",
            }
        ],
    }
    composition = item.save(
        "composition.json",
        {
            "version": 1,
            "parts": [part],
            "joinFormat": "manifest",
            "joinPlan": ref(
                item.save(
                    "manifest.json",
                    {
                        "version": 1,
                        "parts": [
                            {
                                "sourcePath": str(native),
                                "durationFrames": 2,
                                "sha256": digest(native),
                            }
                        ],
                    },
                )
            ),
            "joinedMovie": ref(outputs["join"]),
        },
    )
    sync_request = item.save("sync-request.json", {})
    sync_result = item.save(
        "sync-result.json",
        {
            "request": ref(sync_request),
            "assessment": {"candidate": ref(movie), "presentationSyncAccepted": True},
        },
    )
    item.save("sync-result.json.decoded.json", {})
    item.save("sync-result.json.packets.json", {})
    sync = item.save("sync.json", {"request": ref(sync_request), "result": ref(sync_result)})
    motion = item.save("caption.moti", b"synthetic Motion XML")
    state = item.save(
        "alpha-state.json",
        {
            "status": "Successful",
            "jobID": "synthetic-alpha",
            "submissionVersion": 2,
            "terminalInputsVerified": True,
            "terminalRuntimeVerified": True,
            "inputs": [row(motion)],
        },
    )
    standard_refs: dict[str, object] = {"renderConfig": ref(render)}
    for name in standard.REFERENCES[1:]:
        inputs = [row(render), row(movie), row(source), row(cuts)]
        receipt: dict[str, object] = {"inputs": inputs, "inputsAfter": inputs}
        if name == "technical":
            receipt["expected"] = {"frames": 2, "rate": 30, "width": 1080, "height": 1920}
        if name == "captionAudit":
            receipt["measurements"] = {
                "parts": [
                    {
                        "producerState": ref(state),
                        "jobID": "synthetic-alpha",
                        "motionSourcePlanStyleVerified": True,
                    }
                ]
            }
        standard_refs[name] = ref(item.save(f"standard-{name}.json", receipt))
    standard_index = item.save(
        "standard-index.json",
        {
            "version": 2,
            "backend": adapter.STANDARD_BACKEND,
            "recording": item.root.name,
            "configurationSHA256": digest(config),
            "sourceSHA256": digest(source),
            "finalMovieSHA256": digest(movie),
            **standard_refs,
        },
    )
    actual_inputs = {
        "bgm_removal": [model, weights, model_config],
        "compose": [clean, request, item.paths["binding.json"]],
        "join": [native, item.paths["manifest.json"]],
    }
    productions = [
        item.producer(role, output, actual_inputs.get(role)) for role, output in outputs.items()
    ] + [item.producer("separation-source", model)]
    item.data = {
        "version": 1,
        "backend": adapter.BACKEND,
        "recording": item.root.name,
        "configuration": ref(config),
        "context": {
            "source": ref(source),
            "renderConfig": ref(render),
            "clean": ref(clean),
            "voice": ref(voice),
            "cuts": ref(cuts),
            "movie": ref(movie),
            "srt": ref(item.paths["candidate.srt"]),
            "frames": 2,
            "fps": 30,
            "width": 1080,
            "height": 1920,
        },
        "domains": {
            "separation": ref(separation),
            "composition": ref(composition),
            "sync": ref(sync),
        },
        "productions": productions,
        "standardIndex": ref(standard_index),
    }
    item.freeze()
    return item


def test_recorded_stages_and_current_hashes_do_not_approve_quality(fixture: Fixture) -> None:
    observed = fixture.assess(fresh=True)
    stages = evidence.mapping(observed["recordedStages"])
    assert all(evidence.mapping(value)["executionVerified"] is True for value in stages.values())
    assert observed["currentCryptoVerified"] is True
    assert observed["strictComplete"] is False
    assert observed["standardContractComplete"] is False
    assert observed["presentationSyncVerified"] is False
    assert observed["qualityApproved"] is False
    composition = evidence.mapping(evidence.mapping(observed["domains"])["composition"])
    assert composition["nativePartsAndIntendedClockBound"] is True
    assert composition["exactSelectedSourceContentVerified"] is False


def test_metadata_only_does_not_reuse_a_saved_observation(fixture: Fixture) -> None:
    fixture.data["observation"] = {"currentCryptoVerified": True, "strictComplete": True}
    report = fixture.assess()
    assert report["currentCryptoVerified"] is False
    assert report["strictComplete"] is False


@pytest.mark.parametrize(
    "backend", ["native-fcp-partitioned-whole", "motion-compressor-direct", "unknown"]
)
def test_backend_mixing_rejected(fixture: Fixture, backend: str) -> None:
    fixture.data["backend"] = backend
    with pytest.raises(ValueError, match="backend"):
        fixture.assess()


@pytest.mark.parametrize(
    "name", ["source", "movie", "srt", "renderConfig", "clean", "voice", "cuts"]
)
def test_missing_context_receipt_cannot_be_replaced_by_three_json_files(
    fixture: Fixture, name: str
) -> None:
    context = evidence.mapping(fixture.data["context"])
    del context[name]
    fixture.data["context"] = context
    with pytest.raises(ValueError, match="context"):
        fixture.assess()


def test_missing_domains_are_reported_as_pending(fixture: Fixture) -> None:
    fixture.data["domains"] = {}
    fixture.data["productions"] = []
    del fixture.data["standardIndex"]
    report = fixture.assess()
    assert set(evidence.mapping(report["strictFailures"])) == set(REQUIRED_STAGES)
    assert all(
        not evidence.mapping(value)["executionVerified"]
        for value in evidence.mapping(report["recordedStages"]).values()
    )


@pytest.mark.parametrize("field", ["status", "fingerprint", "outputs"])
def test_partial_or_tampered_original_checkpoint_fails(fixture: Fixture, field: str) -> None:
    checkpoint = fixture.document("checkpoint-bgm_removal.json")
    checkpoint[field] = "partial"
    fixture.save("checkpoint-bgm_removal.json", checkpoint)
    with pytest.raises(ValueError):
        fixture.assess()


def test_current_content_mutation_rejects_preserved_input_pin(fixture: Fixture) -> None:
    index = fixture.freeze()
    fixture.paths["clean.mov"].write_bytes(b"changed same endpoint")
    with pytest.raises(ValueError, match="identity"):
        adapter.assess(index, root=fixture.root, movie=fixture.movie)


def test_content_budget_prevents_media_hash_dispatch(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(bindings: object) -> InputHashSnapshot:
        pytest.fail("Fresh content read started outside its allowance")

    monkeypatch.setattr(InputHashSnapshot, "capture", forbidden)
    with pytest.raises(ValueError, match="byte allowance"):
        adapter.assess(
            fixture.freeze(),
            root=fixture.root,
            movie=fixture.movie,
            verify_content=True,
            content_byte_budget=1,
        )


def test_source_tail_overrun_is_not_accepted_by_correct_native_count(fixture: Fixture) -> None:
    composition = fixture.document("composition.json")
    part = evidence.rows(composition["parts"])[0]
    selection = evidence.rows(part["selections"])[0]
    selection["start"] = "1/15"
    part["selections"] = [selection]
    request = fixture.document("native-request.json")
    recipe = evidence.mapping(request["recipe"])
    clip = evidence.rows(recipe["clips"])[0]
    clip["startSeconds"] = 2 / 30
    recipe["clips"] = [clip]
    request["recipe"] = recipe
    fixture.save("native-request.json", request)
    part["request"] = ref(fixture.paths["native-request.json"])
    proof = fixture.document("part-proof.json")
    proof["partition"] = part["request"]
    part["proof"] = ref(fixture.save("part-proof.json", proof))
    binding = fixture.document("binding.json")
    binding["consumer"] = {**ref(fixture.paths["native-request.json"]), "kind": "native-request"}
    part["sourceCaptionBinding"] = ref(fixture.save("binding.json", binding))
    composition["parts"] = [part]
    context = fixture.context()
    with pytest.raises(ReplayError):
        adapter._source_selection(selection, context, evidence.MetadataReader())


def test_classic_join_plan_preserves_same_pending_content_requirement(fixture: Fixture) -> None:
    composition = fixture.document("composition.json")
    composition["joinFormat"] = "join-files"
    composition["joinPlan"] = ref(
        fixture.save(
            "join-plan.json", [{"sourcePath": str(fixture.paths["part.mov"]), "durationFrames": 2}]
        )
    )
    report = adapter._composition(
        composition,
        fixture.context(),
        evidence.MetadataReader(),
        {
            str(fixture.paths["part.mov"]): digest(fixture.paths["part.mov"]),
            str(fixture.paths["joined.mov"]): digest(fixture.paths["joined.mov"]),
        },
        adapter._productions(fixture.data, fixture.context(), evidence.MetadataReader())[2],
    )
    assert report["exactSelectedSourceContentVerified"] is False


@pytest.mark.parametrize("clamp", ["-1", "1/30", "nan", "inf", "0"])
def test_invalid_native_clamp_is_rejected(fixture: Fixture, clamp: str) -> None:
    selection = evidence.rows(
        evidence.rows(fixture.document("composition.json")["parts"])[0]["selections"]
    )[0]
    selection["nativeDuration"] = clamp
    with pytest.raises(ValueError):
        adapter._source_selection(selection, fixture.context(), evidence.MetadataReader())


def test_half_frame_trim_is_only_clock_evidence(fixture: Fixture) -> None:
    selection = evidence.rows(
        evidence.rows(fixture.document("composition.json")["parts"])[0]["selections"]
    )[0]
    selection["nativeDuration"] = "1/20"
    adapter._source_selection(selection, fixture.context(), evidence.MetadataReader())


@pytest.mark.parametrize(
    "change", [{"pts": [0, 2, 3]}, {"pts": [0]}, {"kind": "planned"}, {"frames": 0}]
)
def test_source_grid_must_be_actual_complete_clock(
    fixture: Fixture, change: dict[str, object]
) -> None:
    grid = fixture.document("grid.json")
    grid.update(change)
    fixture.save("grid.json", grid)
    selection = evidence.rows(
        evidence.rows(fixture.document("composition.json")["parts"])[0]["selections"]
    )[0]
    selection["proof"] = ref(fixture.paths["grid.json"])
    with pytest.raises(ValueError):
        adapter._source_selection(selection, fixture.context(), evidence.MetadataReader())


def test_public_standard_assembler_writes_a_true_pending_v2_shell(fixture: Fixture) -> None:
    original = fixture.paths["standard-index.json"].read_bytes()
    fixture.data["standardReferences"] = {"renderConfig": ref(fixture.paths["render.json"])}
    del fixture.data["standardIndex"]
    output = fixture.root / "new-standard.json"
    adapter.prepare_standard(
        fixture.freeze(),
        root=fixture.root,
        movie=fixture.movie,
        output=output,
        content_byte_budget=10_000_000,
    )
    document = evidence.mapping(json.loads(output.read_text()))
    assert document["backend"] == "motion-compressor-direct"
    assert document["version"] == 2
    assert fixture.paths["standard-index.json"].read_bytes() == original
    assert standard.assess(
        output, delivery=fixture.movie, root=fixture.root, source=fixture.paths["source.mp4"]
    )


def test_existing_standard_index_is_not_overwritten(fixture: Fixture) -> None:
    with pytest.raises(ValueError, match="overwrite"):
        adapter.prepare_standard(
            fixture.freeze(),
            root=fixture.root,
            movie=fixture.movie,
            output=fixture.paths["standard-index.json"],
            content_byte_budget=10_000_000,
        )


@pytest.mark.parametrize("operation", ["prepare", "observe", "prepare-standard", "assess", "check"])
def test_cli_pending_contract_is_explicit(
    fixture: Fixture, operation: str, capsys: pytest.CaptureFixture[str]
) -> None:
    index = fixture.freeze()
    args = [
        operation,
        str(fixture.root),
        str(fixture.movie),
        str(fixture.paths["mechanical.json"]),
        "--sha256",
        evidence.string(index["sha256"]),
        "--content-byte-budget",
        "10000000",
    ]
    if operation in {"prepare", "observe", "prepare-standard"}:
        args += ["--output", str(fixture.root / f"{operation}.json")]
    assert adapter.main(args) == int(operation == "check")
    assert capsys.readouterr().out


def test_cli_missing_new_output_returns_failure(
    fixture: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    index = fixture.freeze()
    assert (
        adapter.main(
            [
                "prepare",
                str(fixture.root),
                str(fixture.movie),
                str(fixture.paths["mechanical.json"]),
                "--sha256",
                evidence.string(index["sha256"]),
            ]
        )
        == 1
    )
    assert "requires new output" in capsys.readouterr().err


@pytest.mark.parametrize("value", [None, -1, float("nan"), True, "2", 0])
def test_invalid_positive_dimensions(value: object) -> None:
    with pytest.raises(ValueError):
        adapter._positive(value)


@pytest.mark.parametrize("value", [None, {}, [1, 2, 3]])
def test_finite_list_budget(value: object) -> None:
    with pytest.raises(ValueError):
        adapter._list(value, 2, "test")


def test_reader_module_without_source_is_refused() -> None:
    with pytest.raises(ValueError, match="source path"):
        adapter._code_path(ModuleType("synthetic"))


def test_empty_original_decoder_streams_are_currently_verified(fixture: Fixture) -> None:
    stream = fixture.save("decode.stderr", b"")
    assert adapter._inventory([row(stream)])[str(stream)]["bytes"] == 0
    snapshot = adapter._snapshot(
        {stream: digest(stream), fixture.movie: digest(fixture.movie)}, 1000
    )
    snapshot.finish()


def test_mutated_empty_stream_cannot_resume(fixture: Fixture) -> None:
    stream = fixture.save("decode.stdout", b"")
    snapshot = adapter._snapshot(
        {stream: digest(stream), fixture.movie: digest(fixture.movie)}, 1000
    )
    stream.write_bytes(b"unexpected decoder output")
    with pytest.raises(ValueError, match="stream changed"):
        snapshot.finish()


def test_empty_media_is_not_an_empty_process_stream(fixture: Fixture) -> None:
    movie = fixture.save("empty.mov", b"")
    with pytest.raises(ValueError, match="process streams"):
        adapter._inventory([row(movie)])


def test_standard_true_flags_without_original_motion_source_remain_pending(
    fixture: Fixture,
) -> None:
    state = fixture.document("alpha-state.json")
    state["inputs"] = [row(fixture.paths["voice.wav"])]
    fixture.save("alpha-state.json", state)
    caption = fixture.document("standard-captionAudit.json")
    measurements = evidence.mapping(caption["measurements"])
    part = evidence.rows(measurements["parts"])[0]
    part["producerState"] = ref(fixture.paths["alpha-state.json"])
    measurements["parts"] = [part]
    caption["measurements"] = measurements
    fixture.save("standard-captionAudit.json", caption)
    standard_index = fixture.document("standard-index.json")
    standard_index["captionAudit"] = ref(fixture.paths["standard-captionAudit.json"])
    fixture.data["standardIndex"] = ref(fixture.save("standard-index.json", standard_index))
    report = fixture.assess()
    assert "No actual Motion source" in json.dumps(report["strictFailures"])


def test_undeclared_nested_reader_input_stops_before_standard_verifier(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    foreign = fixture.root / "undeclared.bin"
    foreign.write_bytes(b"synthetic undeclared")
    fixture.save("extra.json", {"nested": [str(foreign)]})

    def forbidden(*args: object, **kwargs: object) -> dict[str, str]:
        pytest.fail("Unbudgeted reader input reached standard verifier")

    monkeypatch.setattr(standard, "assess", forbidden)
    with pytest.raises(ValueError, match="Nested reader input"):
        fixture.assess(fresh=True)


def test_fresh_reader_code_pin_is_mandatory(fixture: Fixture) -> None:
    index = fixture.freeze()
    path = adapter.code_inputs()[0]
    fixture.data["currentInputs"] = [
        value
        for value in adapter._list(fixture.data["currentInputs"], 4096, "inputs")
        if evidence.mapping(value)["path"] != str(path)
    ]
    external = evidence.mapping(fixture.data["externalInputs"])
    external.pop(str(path))
    fixture.data["externalInputs"] = external
    index = ref(fixture.save("mechanical.json", fixture.data))
    with pytest.raises(ValueError, match="reader code"):
        adapter.assess(
            index,
            root=fixture.root,
            movie=fixture.movie,
            verify_content=True,
            content_byte_budget=10_000_000,
        )


def test_public_sync_boolean_is_not_acceptance_without_original_processes(fixture: Fixture) -> None:
    with pytest.raises(ReplayError, match="request"):
        adapter._sync(
            fixture.document("sync.json"), fixture.context(), evidence.MetadataReader(), verify=True
        )


def test_successful_public_numeric_verifier_still_cannot_fix_source_selection(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(standard, "assess", lambda *args, **kwargs: {})
    monkeypatch.setattr(aac, "verify_presentation", lambda *args: None)
    monkeypatch.setattr(
        source_caption_binding,
        "require_cleanup",
        lambda path: source_caption_binding.Cleanup(
            path, fixture.paths["source.mp4"], fixture.paths["clean.mov"], Fraction(2, 30)
        ),
    )
    monkeypatch.setattr(
        evidence,
        "cleanup_evidence",
        lambda raw, reader: {"historicalExecutionVerified": True},
    )
    monkeypatch.setattr(
        evidence, "asr_evidence", lambda raw, reader: {"historicalExecutionVerified": True}
    )
    domains = evidence.mapping(fixture.data["domains"])
    domains["cleanup"] = ref(
        fixture.save("cleanup.json", {"source": ref(fixture.paths["source.mp4"])})
    )
    domains["asr"] = ref(fixture.save("asr.json", {"cuts": ref(fixture.paths["cuts.json"])}))
    fixture.data["domains"] = domains
    report = fixture.assess(fresh=True)
    assert report["standardContractComplete"] is True
    assert report["presentationSyncVerified"] is True
    assert report["strictComplete"] is False
    assert set(evidence.mapping(report["strictFailures"])) == {"silence_cut"}


def test_current_input_sha_tampering_is_not_stat_acceptance(fixture: Fixture) -> None:
    context = fixture.context()
    context.rows[str(fixture.paths["clean.mov"])]["sha256"] = "0" * 64
    pins = {Path(name): evidence.digest(value["sha256"]) for name, value in context.rows.items()}
    with pytest.raises(StageError, match="SHA256"):
        adapter._snapshot(pins, 10_000_000)


@pytest.mark.parametrize("omitted", ["modelInput.wav", "model/weights.bin", "model/config.json"])
def test_separation_requires_actual_producer_model_inputs(fixture: Fixture, omitted: str) -> None:
    context = fixture.context()
    _, outputs, dependencies = adapter._productions(
        fixture.data, context, evidence.MetadataReader()
    )
    dependencies[str(fixture.paths["voice.wav"])].pop(str(fixture.paths[omitted]))
    with pytest.raises(ValueError, match="separation producer"):
        adapter._separation(fixture.document("separation.json"), context, outputs, dependencies)


def test_model_input_pcm_must_come_from_original_source_producer(fixture: Fixture) -> None:
    context = fixture.context()
    _, outputs, dependencies = adapter._productions(
        fixture.data, context, evidence.MetadataReader()
    )
    dependencies[str(fixture.paths["modelInput.wav"])].clear()
    with pytest.raises(ValueError, match="original source"):
        adapter._separation(fixture.document("separation.json"), context, outputs, dependencies)


@pytest.mark.parametrize("omitted", ["clean.mov", "native-request.json", "binding.json"])
def test_compositor_cannot_substitute_unbound_request_or_raw_source(
    fixture: Fixture, omitted: str
) -> None:
    context = fixture.context()
    _, outputs, dependencies = adapter._productions(
        fixture.data, context, evidence.MetadataReader()
    )
    dependencies[str(fixture.paths["part.mov"])].pop(str(fixture.paths[omitted]))
    with pytest.raises(ValueError, match="Original compositor"):
        adapter._composition(
            fixture.document("composition.json"),
            context,
            evidence.MetadataReader(),
            outputs,
            dependencies,
        )


def test_duplicate_original_producer_or_role_is_not_resume(fixture: Fixture) -> None:
    productions = adapter._list(fixture.data["productions"], 4096, "productions")
    fixture.data["productions"] = [*productions, productions[0]]
    with pytest.raises(ValueError, match="duplicate producer"):
        fixture.assess()


def test_zero_caption_audits_cannot_be_upgraded_by_true_flags(fixture: Fixture) -> None:
    context = fixture.context()
    receipt = fixture.document("standard-captionAudit.json")
    receipt["measurements"] = {"expectedAlphaParts": 0, "auditedAlphaParts": 0, "parts": []}
    receipt["accepted"] = True
    receipt["checks"] = {"fullAlphaCoverage": True}
    fixture.save("standard-captionAudit.json", receipt)
    standard_index = fixture.document("standard-index.json")
    standard_index["captionAudit"] = ref(fixture.paths["standard-captionAudit.json"])
    index_ref = ref(fixture.save("standard-index.json", standard_index))
    fixture.data["standardIndex"] = index_ref
    context = fixture.context()
    with pytest.raises(ValueError, match="nonempty rows"):
        adapter._standard(index_ref, context, evidence.MetadataReader(), root=fixture.root)


def test_fixed_blur_is_rejected_before_native_paint_claim(fixture: Fixture) -> None:
    composition = fixture.document("composition.json")
    request = fixture.document("native-request.json")
    recipe = evidence.mapping(request["recipe"])
    recipe["video"] = {"masks": [{"blurRadius": 64}]}
    request["recipe"] = recipe
    fixture.save("native-request.json", request)
    part = evidence.rows(composition["parts"])[0]
    part["request"] = ref(fixture.paths["native-request.json"])
    proof = fixture.document("part-proof.json")
    proof["partition"] = part["request"]
    part["proof"] = ref(fixture.save("part-proof.json", proof))
    binding = fixture.document("binding.json")
    binding["consumer"] = {**ref(fixture.paths["native-request.json"]), "kind": "native-request"}
    part["sourceCaptionBinding"] = ref(fixture.save("binding.json", binding))
    composition["parts"] = [part]
    producer = fixture.producer(
        "compose",
        fixture.paths["part.mov"],
        [fixture.paths[name] for name in ("clean.mov", "native-request.json", "binding.json")],
    )
    productions = adapter._list(fixture.data["productions"], 4096, "productions")
    fixture.data["productions"] = [
        producer
        if evidence.reference(value)["path"] == str(fixture.paths["producer-compose.json"])
        else value
        for value in productions
    ]
    context = fixture.context()
    _, outputs, dependencies = adapter._productions(
        fixture.data, context, evidence.MetadataReader()
    )
    with pytest.raises(ValueError, match="Fixed blur"):
        adapter._composition(composition, context, evidence.MetadataReader(), outputs, dependencies)


def test_global_standard_provenance_failure_is_recorded_as_pending(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: object, **kwargs: object) -> dict[str, str]:
        raise ReplayError("original render configuration is incomplete")

    monkeypatch.setattr(standard, "assess", unavailable)
    report = fixture.assess(fresh=True)
    assert report["strictComplete"] is False
    assert "standard provenance" in json.dumps(report["strictFailures"])


def test_public_assembly_cannot_relabel_an_fcp_index(fixture: Fixture) -> None:
    document = fixture.document("standard-index.json")
    document["backend"] = "native-fcp-partitioned-whole"
    fixture.data["standardIndex"] = ref(fixture.save("standard-index.json", document))
    with pytest.raises(ValueError, match="relabeling"):
        adapter.prepare_standard(
            fixture.freeze(),
            root=fixture.root,
            movie=fixture.movie,
            output=fixture.root / "never-written.json",
            content_byte_budget=10_000_000,
        )
    assert not (fixture.root / "never-written.json").exists()


def test_final_native_count_must_match_mechanical_whole_clock(fixture: Fixture) -> None:
    receipt = fixture.document("standard-technical.json")
    expected = evidence.mapping(receipt["expected"])
    expected["frames"] = 3
    receipt["expected"] = expected
    fixture.save("standard-technical.json", receipt)
    document = fixture.document("standard-index.json")
    document["technical"] = ref(fixture.paths["standard-technical.json"])
    reference = ref(fixture.save("standard-index.json", document))
    fixture.data["standardIndex"] = reference
    with pytest.raises(ValueError, match="Final technical frame clock"):
        adapter._standard(
            reference, fixture.context(), evidence.MetadataReader(), root=fixture.root
        )


@pytest.mark.parametrize("field", ["version", "backend", "configurationSHA256", "finalMovieSHA256"])
def test_standard_index_backend_or_revision_cannot_be_mixed(fixture: Fixture, field: str) -> None:
    document = fixture.document("standard-index.json")
    document[field] = "wrong"
    index = ref(fixture.save("standard-index.json", document))
    fixture.data["standardIndex"] = index
    with pytest.raises(ValueError):
        adapter._standard(index, fixture.context(), evidence.MetadataReader(), root=fixture.root)


def test_prepare_refuses_linked_or_escaping_output(fixture: Fixture) -> None:
    output = fixture.root.parent.parent.parent / "outside.json"
    with pytest.raises(ValueError, match="checkout"):
        adapter.prepare(fixture.freeze(), root=fixture.root, movie=fixture.movie, output=output)
