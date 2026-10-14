import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

import pytest

from replay import delivery_render as render
from replay import delivery_render_resume as resume
from replay import native_edit_parts, native_invocation, source_caption_binding
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import _fingerprint, file_digest, run_stage
from replay.workflow import parse_stages


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def historical_workflow(config: render.Config) -> Path:
    """Retained pre-cleanup-contract workflows, independent of today's builder."""
    stages: list[dict[str, object]] = []
    for number, (request, _) in enumerate(render.rendered_parts(config)):
        stages.append(
            {
                "name": f"render-{number:06d}",
                "command": list(native_invocation.command(config.native, request, config.storage)),
                "verify_command": list(
                    native_invocation.command(config.native, request, config.storage, verify=True)
                ),
                "inputs": [
                    *map(str, native_invocation.inputs(config.native, request)),
                    str(Path(native_invocation.__file__).resolve()),
                ],
                "outputs": list(map(str, native_invocation.paths(request))),
                "timeout_seconds": render.RENDER_TIMEOUT_SECONDS,
            }
        )
    return write(config.composition / "render-0-0000.json", {"version": 1, "stages": stages})


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"candidate a")
    b.write_bytes(b"candidate b")
    native = tmp_path / "native"
    native.write_bytes(b"native executable")
    alpha = tmp_path / "alpha.mov"
    alpha.write_bytes(b"alpha unchanged")
    chunks = write(
        tmp_path / "chunks.json",
        {
            "ranges": [
                {
                    "media": str(a),
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": 1032,
                    "frames": 31,
                    "timeBase": "1/1000",
                },
                {
                    "media": str(b),
                    "sourceFirstPTS": 1032,
                    "sourceEndPTS": 3000,
                    "frames": 59,
                    "timeBase": "1/1000",
                },
            ]
        },
    )
    cuts = write(
        tmp_path / "cuts.json", {"cuts": [{"source_start_frame": 0, "source_end_frame": 90}]}
    )
    config = write(
        tmp_path / "config.json",
        {
            "version": 1,
            "recording": "replay-1",
            "captions": str(tmp_path / "captions"),
            "work": str(tmp_path / "work"),
            "storage": str(tmp_path / "storage"),
            "rate": 30,
            "width": 1080,
            "height": 1920,
            "cuts": str(cuts),
            "sourceChunks": str(chunks),
            "motion": {"template": str(alpha), "expectedText": "title", "finish": []},
            "alphaPreset": str(alpha),
            "encodePreset": str(alpha),
            "native": str(native),
            "encoding": {"codec": "hevc"},
            "audio": {"source": str(alpha), "clock": str(cuts)},
            "source": str(a),
            "deliveryRoot": str(tmp_path / "delivery"),
            "finalPreset": str(alpha),
            "calibration": str(alpha),
            "viewerRoot": str(tmp_path / "viewer"),
        },
    )
    cfg = render.load_config(config)
    source = write(
        cfg.composition / "request-0000.json",
        {
            "outputDirectory": str(tmp_path / "old-media"),
            "outputName": "video.mp4",
            "recipe": {
                "video": {"frameRate": 30, "encoding": {"codec": "hevc"}},
                "clips": [
                    {
                        "sourcePath": str(a),
                        "selection": {"startSeconds": 0, "durationSeconds": 31 / 30, "rate": 1},
                    },
                    {
                        "sourcePath": str(b),
                        "selection": {
                            "startSeconds": float(Fraction(31, 30) - Fraction(1032, 1000)),
                            "durationSeconds": 29 / 30,
                            "rate": 1,
                        },
                    },
                    {
                        "sourcePath": str(b),
                        "selection": {"startSeconds": 0.968, "durationSeconds": 1, "rate": 1},
                    },
                ],
                "additionalVideo": [],
            },
        },
    )
    parts = native_edit_parts.export(
        source, cfg.composition / "parts-0000", sha256=file_digest(source)
    )
    workflow = historical_workflow(cfg)
    # This suite models retained legacy requests, not the new producer contract.
    legacy = resume.document(workflow)
    legacy["stages"] = [
        {
            **record(row),
            "inputs": render._list(record(row)["inputs"])[:2],
            "outputs": [render._list(record(row)["outputs"])[1]],
            "verify_command": [],
            "command": [
                "/bin/bash",
                "-c",
                'exec "$1" edit-media "$2" > "$3"',
                "native-render",
                str(cfg.native),
                render._list(record(row)["inputs"])[0],
                render._list(record(row)["outputs"])[1],
            ],
        }
        for row in render._list(legacy["stages"])
    ]
    write(workflow, legacy)
    stages = parse_stages(resume.document(workflow))
    state = workflow.with_name(f"{workflow.stem}-state")
    movie = tmp_path / "complete.mp4"
    movie.write_bytes(b"complete native movie")
    write(stages[1].outputs[0], {"outputPath": str(movie), "frameCount": 29})
    complete = write(
        state / "render-000001.json",
        {
            "status": "complete",
            "fingerprint": _fingerprint(stages[1]),
            "outputs": {str(stages[1].outputs[0]): file_digest(stages[1].outputs[0])},
        },
    )
    failed = write(
        state / "render-000000.json", {"status": "running", "fingerprint": _fingerprint(stages[0])}
    )
    parent = stages[0].inputs[0].parent
    parent.rename(parent.with_name(parent.name + ".failed-0"))
    assembled = tmp_path / "assembled.mov"
    assembled.write_bytes(b"assembled packets")
    index = write(
        tmp_path / "assembly" / "single-source-index.json",
        {
            "ranges": [
                {
                    "media": str(assembled),
                    "sha256": file_digest(assembled),
                    "sourceFirstPTS": 0,
                    "sourceEndPTS": 3000,
                    "frames": 90,
                    "timeBase": "1/1000",
                }
            ]
        },
    )
    probe = write(index.parent / "output-frames.json", {"frames": []})
    receipt = write(
        index.parent / "receipt.json",
        {
            "version": 1,
            "inputs": {
                str(chunks): file_digest(chunks),
                str(a): file_digest(a),
                str(b): file_digest(b),
            },
            "output": ref(assembled),
            "indexSHA256": file_digest(index),
            "frames": 90,
            "probeEvidence": {str(probe): file_digest(probe)},
            "exactPresentationClockVerified": True,
            "compressedPacketsIdentical": True,
            "qualityApproved": False,
        },
    )
    cleanup = write(tmp_path / "cleanup.json", {"version": 1, "cleanIndex": ref(index)})
    write(config, {**resume.document(config), "sourceCleanup": ref(cleanup)})
    monkeypatch.setattr(
        source_caption_binding, "clean_source", lambda _: (a, assembled, Fraction(3))
    )
    return write(
        tmp_path / "resume-manifest.json",
        {
            "version": 1,
            "config": ref(config),
            "native": ref(native),
            "cuts": ref(cuts),
            "index": ref(index),
            "assembly": ref(receipt),
            "manifests": [ref(parts)],
            "workflows": [{"plan": ref(workflow), "receipts": [ref(complete), ref(failed)]}],
            "completed": {"1": ref(movie)},
        },
    )


def no_movie(_movie: Path, _frames: int, _rate: int, _evidence: Path) -> None:
    pass


def prep(fixture: Path, *, validator: resume.MovieValidator = no_movie) -> Path:
    return resume.prepare(
        fixture, fixture.parent / "retry", sha256=file_digest(fixture), validator=validator
    )


def test_mixed_retry_preserves_order_clocks_and_originals(fixture: Path) -> None:
    before = {str(p): file_digest(p) for p in fixture.parent.rglob("*") if p.is_file()}
    calls: list[tuple[str, int, int]] = []

    def validate(movie: Path, frames: int, rate: int, _evidence: Path) -> None:
        calls.append((movie.name, frames, rate))

    config_path = prep(fixture, validator=validate)
    cfg = render.load_config(config_path)
    assert calls == [("complete.mp4", 29, 30)]
    assert all(file_digest(Path(p)) == sha for p, sha in before.items())
    assert cfg.work == fixture.parent / "work"
    assert cfg.storage == fixture.parent / "storage"
    rows = render.rendered_parts(cfg)
    assert [frames for _, frames in rows] == [31, 29, 30]
    assert rows[1][0] == fixture.parent / "work/compose/parts-0000/part-0001/request.json"
    first, last = resume.document(rows[0][0]), resume.document(rows[2][0])
    first_recipe, last_recipe = record(first["recipe"]), record(last["recipe"])
    assert record(resume.array(first_recipe["clips"])[0])["selection"] == {
        "startSeconds": 0.0,
        "durationSeconds": 31 / 30,
        "rate": 1,
    }
    assert record(resume.array(last_recipe["clips"])[0])["selection"] == {
        "startSeconds": 2.0,
        "durationSeconds": 1.0,
        "rate": 1,
    }
    assert Path(str(first["outputDirectory"])).is_relative_to(cfg.storage)
    assert not Path(str(first["outputDirectory"])).exists()
    assert sorted(
        s.name
        for p in cfg.composition.glob("render-*-*.json")
        for s in parse_stages(resume.document(p))
    ) == ["render-000000", "render-000002"]
    assert resume.verify(cfg.composition / "resume.json")["version"] == 2
    with pytest.raises(ReplayError, match="sealed prepared"):
        render.render(cfg, workers=1)
    with pytest.raises(FileExistsError):
        prep(fixture)


def test_join_uses_mixed_order(fixture: Path) -> None:
    cfg = render.load_config(prep(fixture))
    rows = render.rendered_parts(cfg)
    write(
        rows[0][0].with_name("render.response.json"),
        {"outputPath": str(fixture.parent / "new0.mp4"), "frameCount": 31},
    )
    write(
        rows[2][0].with_name("render.response.json"),
        {"outputPath": str(fixture.parent / "new2.mp4"), "frameCount": 30},
    )
    (fixture.parent / "new0.mp4").write_bytes(b"first")
    (fixture.parent / "new2.mp4").write_bytes(b"last")
    render.join(cfg)
    assert render._json(cfg.work / "join-plan.json") == [
        {"durationFrames": 31, "sourcePath": str(fixture.parent / "new0.mp4")},
        {"durationFrames": 29, "sourcePath": str(fixture.parent / "complete.mp4")},
        {"durationFrames": 30, "sourcePath": str(fixture.parent / "new2.mp4")},
    ]


@pytest.mark.parametrize("field", ["config", "index", "assembly", "native", "cuts"])
def test_changed_input_pins_stop(fixture: Path, field: str) -> None:
    path = Path(str(record(resume.document(fixture)[field])["path"]))
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ReplayError, match="Changed input pin"):
        prep(fixture)


@pytest.mark.parametrize("change", ["missing", "changed"])
def test_completed_movie_never_becomes_retry(fixture: Path, change: str) -> None:
    movie = fixture.parent / "complete.mp4"
    if change == "missing":
        movie.unlink()
    else:
        movie.write_bytes(b"changed")
    with pytest.raises((ReplayError, OSError)):
        prep(fixture)
    assert not (fixture.parent / "retry/retry-000001/request.json").exists()


def test_unknown_manifest_fields_stop(fixture: Path) -> None:
    write(fixture, {**resume.document(fixture), "unknown": True})
    with pytest.raises(ReplayError, match="Unknown resume"):
        prep(fixture)


@pytest.mark.parametrize("conflicting", [False, True])
def test_recovery_copied_completions(fixture: Path, conflicting: bool) -> None:
    raw = resume.document(fixture)
    workflow = record(resume.array(raw["workflows"])[0])
    receipts = resume.array(workflow["receipts"])
    original = Path(str(record(receipts[0])["path"]))
    before = original.read_bytes()
    saved = resume.document(original)
    if conflicting:
        saved["fingerprint"] = "a" * 64
    duplicate = write(
        original.parent.with_name(original.parent.name + ".failed-0") / original.name, saved
    )
    workflow["receipts"] = [*receipts, ref(duplicate)]
    write(fixture, raw)
    if conflicting:
        with pytest.raises(ReplayError, match="Conflicting render"):
            prep(fixture)
    else:
        config = render.load_config(prep(fixture))
        facts = resume.document(config.composition / "resume.json")
        rows = resume.array(facts["parts"])
        assert record(rows[1])["reused"] is True
    assert original.read_bytes() == before
    assert duplicate.is_file()


def test_incomplete_receipt_inventory_stops(fixture: Path) -> None:
    raw = resume.document(fixture)
    record(resume.array(raw["workflows"])[0])["receipts"] = []
    write(fixture, raw)
    with pytest.raises(ReplayError, match="inventory"):
        prep(fixture)


def test_bad_completed_fingerprint_stops(fixture: Path) -> None:
    raw = resume.document(fixture)
    workflow = record(resume.array(raw["workflows"])[0])
    receipts = resume.array(workflow["receipts"])
    path = Path(str(record(receipts[0])["path"]))
    write(path, {**resume.document(path), "fingerprint": "0" * 64})
    receipts[0] = ref(path)
    write(fixture, raw)
    with pytest.raises(ReplayError, match="fingerprint"):
        prep(fixture)


def test_preserved_inputs_require_exact_hash_and_never_restore(tmp_path: Path) -> None:
    saved = write(tmp_path / "part.failed-0/request.json", {"old": True})
    path = tmp_path / "part/request.json"
    assert resume.preserved(path, file_digest(saved), resume.Pins()) == saved
    assert not path.exists()
    with pytest.raises(ReplayError, match="No preserved"):
        resume.preserved(path, "0" * 64, resume.Pins())
    write(path, {"changed": True})
    with pytest.raises(ReplayError, match="Changed input"):
        resume.preserved(path, file_digest(saved), resume.Pins())


def test_symlinks_and_invalid_references_stop(tmp_path: Path) -> None:
    target = write(tmp_path / "target.json", {})
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ReplayError, match="unlinked"):
        resume.checked_path(str(link))
    with pytest.raises(ReplayError, match="exact path"):
        resume.Pins().ref({"path": str(target)})
    with pytest.raises(ReplayError, match="Empty"):
        resume.array([])


def test_rebased_selection_without_support_stops(fixture: Path) -> None:
    cfg = render.load_config(fixture.parent / "config.json")
    request = resume.document(cfg.composition / "parts-0000/part-0002/request.json")
    with pytest.raises(ReplayError, match="full assembled"):
        resume.rebase(
            request,
            cfg,
            render.Chunk(fixture.parent / "assembled.mov", Fraction(0), Fraction(2)),
        )
    record(resume.array(record(request["recipe"])["clips"])[0])["sourcePath"] = str(
        fixture.parent / "unknown"
    )
    with pytest.raises(ReplayError, match="Unknown or ambiguous"):
        resume.rebase(
            request,
            cfg,
            render.Chunk(fixture.parent / "assembled.mov", Fraction(0), Fraction(3)),
        )


def test_run_rejects_reused_or_occupied_parts(fixture: Path) -> None:
    config = render.load_config(prep(fixture))
    seal = config.composition / "resume.json"
    with pytest.raises(ReplayError, match="never be rerendered"):
        resume.run(seal, 1)
    with pytest.raises(ReplayError, match="Unknown render index"):
        resume.run(seal, 99)
    (config.composition / "retry-000000/render.attempt").write_bytes(b"uncertain invocation")
    with pytest.raises(ReplayError, match="never resubmit"):
        resume.run(seal, 0)


def test_run_uses_new_request_only(fixture: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = render.load_config(prep(fixture))
    calls: list[list[str]] = []

    def execute(native: Path, request: Path, output: Path) -> None:
        calls.append([str(native), "edit-media", str(request)])
        assert output == config.storage

    monkeypatch.setattr(native_invocation, "run", execute)
    resume.run(config.composition / "resume.json", 0)
    assert calls[0][1:] == ["edit-media", str(config.composition / "retry-000000/request.json")]


def test_existing_retry_response_stops_before_native(fixture: Path) -> None:
    config = render.load_config(prep(fixture))
    request_path = config.composition / "retry-000000/request.json"
    response = request_path.with_name("render.response.json")
    response.write_bytes(b"uncertain result")
    with pytest.raises(ReplayError, match="never resubmit"):
        resume.run(config.composition / "resume.json", 0)
    assert response.read_bytes() == b"uncertain result"
    assert not Path(str(resume.document(request_path)["outputDirectory"])).exists()


@pytest.mark.parametrize("fault", ["none", "stderr", "stream", "base", "count", "clock", "decode"])
def test_movie_validator_checks_actual_probe_and_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    data: dict[str, object] = {
        "streams": [{"codec_type": "video", "time_base": "1/30"}],
        "frames": [{"pts": 0, "duration": 1}, {"pts": 1, "duration": 1}],
    }
    if fault == "stream":
        data["streams"] = [{"codec_type": "audio"}]
    if fault == "base":
        data["streams"] = [{"codec_type": "video"}]
    if fault == "count":
        data["frames"] = [{"pts": 0, "duration": 1}]
    if fault == "clock":
        data["frames"] = [{"pts": 1, "duration": 1}, {"pts": 2, "duration": 1}]

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        error = (
            b"error"
            if (fault == "stderr" or (fault == "decode" and command[0] == "ffmpeg"))
            else b""
        )
        return subprocess.CompletedProcess(command, 0, json.dumps(data).encode(), error)

    monkeypatch.setattr(subprocess, "run", execute)
    (tmp_path / "movie.mp4").write_bytes(b"synthetic mocked movie")
    if fault == "none":
        resume.validate_movie(tmp_path / "movie.mp4", 2, 30, tmp_path / "evidence")
    else:
        with pytest.raises(ReplayError):
            resume.validate_movie(tmp_path / "movie.mp4", 2, 30, tmp_path / "evidence")


@pytest.mark.parametrize("fault", ["facts", "range", "frames", "media", "probes", "candidate"])
def test_assembly_receipt_must_bind_exact_source(fixture: Path, fault: str) -> None:
    cfg = render.load_config(fixture.parent / "config.json")
    index = fixture.parent / "assembly/single-source-index.json"
    receipt = index.parent / "receipt.json"
    facts = resume.document(receipt)
    data = resume.document(index)
    row = record(resume.array(data["ranges"])[0])
    if fault == "facts":
        facts["compressedPacketsIdentical"] = False
    if fault == "range":
        row["sourceFirstPTS"] = 1
    if fault == "frames":
        row["frames"] = 89
        facts["frames"] = 89
    if fault == "media":
        row["sha256"] = "0" * 64
    if fault == "probes":
        facts["probeEvidence"] = {}
    if fault == "candidate":
        record(facts["inputs"]).pop(str(fixture.parent / "b.mp4"))
    write(index, data)
    facts["indexSHA256"] = file_digest(index)
    write(receipt, facts)
    pins = resume.Pins()
    pins.ref(ref(index))
    pins.ref(ref(receipt))
    with pytest.raises(ReplayError):
        resume.assembly(index, receipt, cfg, pins)


def test_nonzero_original_pts_rebases_without_duration_clamping(fixture: Path) -> None:
    cfg = render.load_config(fixture.parent / "config.json")
    write(
        cfg.chunks,
        {
            "ranges": [
                {
                    "media": str(fixture.parent / "a.mp4"),
                    "sourceFirstPTS": 1000,
                    "sourceEndPTS": 2032,
                    "timeBase": "1/1000",
                },
                {
                    "media": str(fixture.parent / "b.mp4"),
                    "sourceFirstPTS": 2032,
                    "sourceEndPTS": 4000,
                    "timeBase": "1/1000",
                },
            ]
        },
    )
    request = resume.document(cfg.composition / "parts-0000/part-0002/request.json")
    old = json.dumps(request)
    changed = resume.rebase(
        request, cfg, render.Chunk(fixture.parent / "assembled.mov", Fraction(1), Fraction(4))
    )
    assert record(resume.array(record(changed["recipe"])["clips"])[0])["selection"] == {
        "startSeconds": 2.0,
        "durationSeconds": 1.0,
        "rate": 1,
    }
    assert json.dumps(request) == old
    assert (
        record(changed["recipe"])["additionalVideo"] == record(request["recipe"])["additionalVideo"]
    )


def test_exact_vfr_origin_is_not_quantized_to_display_timescale(tmp_path: Path) -> None:
    chunk = render.Chunk(tmp_path / "chunk.mp4", Fraction(89999, 90000), Fraction(2))
    clip: dict[str, object] = {"selection": {"startSeconds": float(Fraction(1, 90000))}}
    assert resume.source_start(clip, chunk, 30) == Fraction(1)
    with pytest.raises(ReplayError, match="exactly encode"):
        resume.source_start({"selection": {"startSeconds": 0.00001}}, chunk, 30)
    with pytest.raises(ReplayError, match="Invalid original"):
        resume.source_start({"selection": {"startSeconds": True}}, chunk, 30)


def test_existing_downstream_target_is_stop(fixture: Path) -> None:
    write(fixture.parent / "work/join-workflow.json", {"old": True})
    with pytest.raises(ReplayError, match="namespace is occupied"):
        prep(fixture)
    assert not (fixture.parent / "retry").exists()


def test_quick_identity_rejects_change_without_rehash(fixture: Path) -> None:
    config = render.load_config(prep(fixture))
    (fixture.parent / "complete.mp4").write_bytes(b"changed after prep")
    with pytest.raises(ReplayError, match="identity changed"):
        resume.run(config.composition / "resume.json", 0)


def test_cli_paths(fixture: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resume, "prepare", lambda *args, **kwargs: fixture.parent / "config.json")
    monkeypatch.setattr(resume, "run", lambda *args: None)
    assert (
        resume.main(
            ["prepare", str(fixture), str(fixture.parent / "new"), "--sha256", file_digest(fixture)]
        )
        == 0
    )
    assert resume.main(["run", str(fixture), "0"]) == 0
    assert resume.main(["verify", str(fixture.parent / "missing")]) == 1
    seal = write(fixture.parent / "seal.json", {"version": 1, "inputs": {}})
    assert resume.main(["verify", str(seal)]) == 0
    write(seal, {"version": 2})
    assert resume.main(["verify", str(seal)]) == 1


@pytest.fixture
def rich_fixture(fixture: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = fixture.parent
    old = resume.document(fixture)
    config_raw = resume.document(root / "config.json")
    config = write(
        root / "rich-config.json",
        {
            **config_raw,
            "storage": str(root),
            "compositionDirectory": str(root / "rich-compose"),
        },
    )
    cfg = render.load_config(config)
    source = write(
        cfg.composition / "request-0000.json",
        {
            **resume.document(root / "work/compose/request-0000.json"),
            "outputDirectory": str(root / "rich-media"),
        },
    )
    manifest = native_edit_parts.export(
        source, cfg.composition / "parts-0000", sha256=file_digest(source)
    )
    workflow = historical_workflow(cfg)
    stages = parse_stages(resume.document(workflow))

    def native_call(
        argv: list[str],
        *,
        stdin: int,
        stdout: BinaryIO,
        stderr: BinaryIO,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        request = Path(argv[2])
        directory = Path(str(resume.document(request)["outputDirectory"]))
        movie = directory / "dynamic.mp4"
        movie.write_bytes(b"synthetic complete native movie")
        stdout.write(
            json.dumps(
                {
                    "outputPath": str(movie),
                    "frameCount": 29,
                    "expectedDurationSeconds": 29 / 30,
                    "actualDurationSeconds": 29 / 30,
                }
            ).encode()
        )
        return subprocess.CompletedProcess(argv, 0)

    def execute(argv: tuple[str, ...], log: Path, timeout: int) -> None:
        assert native_invocation.main(list(argv[3:])) == 0

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", native_call)
        assert (
            run_stage(
                stages[1],
                state_directory=workflow.with_name(f"{workflow.stem}-state"),
                execute=execute,
            )
            == "completed"
        )
    response = native_invocation.read(stages[1].outputs[1])
    return write(
        root / "rich-resume-manifest.json",
        {
            **old,
            "config": ref(config),
            "manifests": [ref(manifest)],
            "workflows": [
                {
                    "plan": ref(workflow),
                    "receipts": [
                        ref(workflow.with_name(f"{workflow.stem}-state") / "render-000001.json")
                    ],
                }
            ],
            "completed": {"1": ref(Path(str(response["outputPath"])))},
        },
    )


def test_richer_resume_reuses_bound_completion_and_prepares_wrapper(rich_fixture: Path) -> None:
    cfg = render.load_config(prep(rich_fixture))
    plans = sorted(cfg.composition.glob("render-*-*.json"))
    stages = [stage for plan in plans for stage in parse_stages(resume.document(plan))]
    assert [stage.name for stage in stages] == ["render-000000", "render-000002"]
    assert stages[0].command[1:4] == ("-m", "replay.native_invocation", "run")
    assert stages[0].verify_command[3] == "verify"
    assert stages[0].outputs[-1].name == "render.binding.json"
    assert (rich_fixture.parent / "assembled.mov") in stages[0].inputs
    assert (rich_fixture.parent / "assembled.mov") in stages[0].inputs
    assert resume.verify(cfg.composition / "resume.json")["version"] == 2


@pytest.mark.parametrize(
    "artifact",
    [
        "render.invocation.json",
        "render.response.json",
        "render.stderr",
        "render.binding.json",
        "render.attempt",
    ],
)
def test_richer_unknown_outcome_never_prepares_duplicate(rich_fixture: Path, artifact: str) -> None:
    request = rich_fixture.parent / "rich-compose/parts-0000/part-0000/request.json"
    request.with_name(artifact).write_bytes(b"retained incomplete outcome")
    with pytest.raises(ReplayError, match="no duplicate render"):
        prep(rich_fixture)
    assert request.with_name(artifact).read_bytes() == b"retained incomplete outcome"


def test_richer_resume_run_uses_real_wrapper_once(
    rich_fixture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = render.load_config(prep(rich_fixture))
    seal = cfg.composition / "resume.json"

    def native_call(
        argv: list[str],
        *,
        stdin: int,
        stdout: BinaryIO,
        stderr: BinaryIO,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        request = Path(argv[2])
        movie = Path(str(resume.document(request)["outputDirectory"])) / "dynamic.mp4"
        movie.write_bytes(b"synthetic retry movie")
        stdout.write(
            json.dumps(
                {
                    "outputPath": str(movie),
                    "frameCount": 31,
                    "expectedDurationSeconds": 31 / 30,
                    "actualDurationSeconds": 31 / 30,
                }
            ).encode()
        )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", native_call)
    resume.run(seal, 0)
    request = render.rendered_parts(cfg)[0][0]
    assert native_invocation.verify(cfg.native, request, cfg.storage)["status"] == "complete"
    with pytest.raises(ReplayError, match="never resubmit"):
        resume.run(seal, 0)
    raw = resume.document(seal)
    write(seal, {**raw, "version": 1})
    with pytest.raises(ReplayError, match="legacy resume seal"):
        resume.run(seal, 0)
    rows = render._list(raw["parts"])
    rows[0] = {**record(rows[0]), "invocationContract": "unknown"}
    write(seal, {**raw, "parts": rows})
    with pytest.raises(ReplayError, match="Unsupported richer"):
        resume.verify(seal)


def test_richer_resume_rejects_changed_verifier(rich_fixture: Path) -> None:
    manifest = resume.document(rich_fixture)
    workflow = rich_fixture.parent / "rich-compose/render-0-0000.json"
    plan = resume.document(workflow)
    rows = render._list(plan["stages"])
    rows[0] = {**record(rows[0]), "verify_command": []}
    write(workflow, {**plan, "stages": rows})
    workflows = render._list(manifest["workflows"])
    workflows[0] = {**record(workflows[0]), "plan": ref(workflow)}
    write(rich_fixture, {**manifest, "workflows": workflows})
    with pytest.raises(ReplayError, match="Richer native invocation contract"):
        prep(rich_fixture)


def test_failed_cleanup_cannot_prepare_or_execute(
    fixture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(_: dict[str, object]) -> tuple[Path, Path, Fraction]:
        raise ReplayError("Failed local repair")

    config = render.load_config(prep(fixture))
    monkeypatch.setattr(source_caption_binding, "clean_source", failed)
    monkeypatch.setattr(native_invocation, "run", lambda *_: pytest.fail("Must not launch"))
    with pytest.raises(ReplayError, match="Failed local repair"):
        resume.run(config.composition / "resume.json", 0)
    with pytest.raises(ReplayError, match="Failed local repair"):
        resume.prepare(
            fixture,
            fixture.parent / "another-retry",
            sha256=file_digest(fixture),
            validator=no_movie,
        )
    assert not (fixture.parent / "another-retry/resume.json").exists()


@pytest.mark.parametrize("missing", ["config", "sourceCaptionBinding", "legacy"])
def test_old_sealed_production_requires_new_cleanup_binding(
    fixture: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    config = render.load_config(prep(fixture))
    seal = config.composition / "resume.json"
    raw = resume.document(seal)
    if missing == "config":
        del raw["config"]
    else:
        row = record(resume.array(raw["parts"])[0])
        if missing == "legacy":
            row["invocationContract"] = "legacy-direct-native"
        else:
            del row["sourceCaptionBinding"]
    write(seal, raw)
    monkeypatch.setattr(native_invocation, "run", lambda *_: pytest.fail("Must not launch"))
    with pytest.raises(ReplayError):
        resume.run(seal, 0)


def test_changed_cleanup_consumer_rejected_even_with_updated_seal(
    fixture: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = render.load_config(prep(fixture))
    seal = config.composition / "resume.json"
    raw = resume.document(seal)
    row = record(resume.array(raw["parts"])[0])
    binding = Path(str(row["sourceCaptionBinding"]))
    request = Path(str(row["path"]))
    document = resume.document(request)
    record(resume.array(record(document["recipe"])["clips"])[0])["sourcePath"] = str(
        fixture.parent / "a.mp4"
    )
    write(request, document)
    write(
        binding,
        {**resume.document(binding), "consumer": {**ref(request), "kind": "native-request"}},
    )
    for path in (request, binding):
        record(raw["inputs"])[str(path)] = file_digest(path)
        record(raw["identities"])[str(path)] = resume.identity(path)
    write(seal, raw)
    monkeypatch.setattr(native_invocation, "run", lambda *_: pytest.fail("Must not launch"))
    with pytest.raises(ReplayError, match="unverified source"):
        resume.run(seal, 0)
