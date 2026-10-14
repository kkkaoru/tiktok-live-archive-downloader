"""Synthetic metadata tests; no renderer, private media, or application calls."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay import delivery_render, source_caption_binding, workflow
from replay import motion_alpha_composition_plan as planner
from replay.code_dependency_inventory import Source
from replay.core import ReplayError
from replay.recordings import record
from replay.stages import file_digest

XML = (
    '<ozml><scene><layer><scenenode factoryID="16"><parameter id="1">'
    '<parameter id="100"><parameter id="101" name="Position">'
    '<parameter id="2" value="4"/></parameter></parameter></parameter>'
    "</scenenode></layer></scene></ozml>"
)


def pin(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, delivery_render.Config]:
    clean = tmp_path / "clean.mov"
    clean.write_bytes(b"synthetic media")
    native = tmp_path / "native"
    native.write_bytes(b"synthetic executable, never run")
    alpha = tmp_path / "alpha.mov"
    alpha.write_bytes(b"synthetic alpha")
    source = tmp_path / "part.moti"
    source.write_text(XML, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "candidateOnly": True,
                "completed": {
                    "part.moti": {"sha256": file_digest(source), "offsetFrame": 15, "frames": 30}
                },
            }
        ),
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    alpha_plan = tmp_path / "alpha-plan.json"
    alpha_plan.write_text('{"frames":30,"rate":30}', encoding="utf-8")
    audit.write_text(
        json.dumps(
            {
                "kind": "delivery-caption-audit",
                "alphaOnlyAccepted": True,
                "measurements": {
                    "rate": 30,
                    "parts": [
                        {
                            "alpha": pin(alpha),
                            "source": pin(source),
                            "plan": pin(alpha_plan),
                            "producerState": pin(alpha_plan),
                            "frames": 30,
                            "historicalSubmissionBound": True,
                            "motionSourcePlanStyleVerified": True,
                            "checks": {"boundsAndGapsVerified": True, "gapsVerified": True},
                            "measurement": {
                                "fullDecodeVerified": True,
                                "decodedFrames": 30,
                                "clock": {"frames": 30, "start": "0", "end": "1", "exactCFR": True},
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    cleanup = tmp_path / "cleanup.json"
    cleanup.write_text(
        json.dumps(
            {
                "version": 1,
                "captionOverlays": [{"audit": pin(audit), "role": "motion-caption-alpha"}],
            }
        ),
        encoding="utf-8",
    )
    renderer = tmp_path / "render.json"
    renderer.write_text("{}", encoding="utf-8")
    config = delivery_render.Config(
        path=renderer,
        recording="synthetic",
        captions=renderer,
        work=tmp_path,
        storage=tmp_path / "media-new",
        rate=30,
        width=64,
        height=96,
        cuts=renderer,
        chunks=renderer,
        template=source,
        expected_text="Title",
        finish=(),
        alpha_preset=renderer,
        encode_preset=renderer,
        native=native,
        encoding={"codec": "hevc"},
        audio_source=clean,
        audio_clock=renderer,
        source=clean,
        delivery_root=tmp_path,
        final_preset=renderer,
        calibration=renderer,
        viewer_root=tmp_path,
        source_cleanup=cleanup,
    )
    bound = source_caption_binding.Cleanup(cleanup, clean, clean, Fraction(2))
    monkeypatch.setattr(delivery_render, "load_config", lambda _: config)
    monkeypatch.setattr(delivery_render, "verified_cleanup", lambda _: bound)
    monkeypatch.setattr(delivery_render, "read_cuts", lambda _: [])
    monkeypatch.setattr(delivery_render, "read_chunks", lambda _: [])
    monkeypatch.setattr(
        delivery_render, "clips", lambda *_: [delivery_render.Clip(clean, Fraction(0), 60)]
    )
    monkeypatch.setattr(
        source_caption_binding, "clean_source", lambda _: (clean, clean, Fraction(2))
    )
    monkeypatch.setattr(planner, "capture", lambda _: ())
    request = tmp_path / "config.json"
    request.write_text(
        json.dumps(
            {
                "version": 1,
                "renderConfig": pin(renderer),
                "alphas": [{"manifest": pin(manifest), "part": "part.moti", "alpha": pin(alpha)}],
            }
        ),
        encoding="utf-8",
    )
    return request, config


def test_roundtrip_ordinary_native_requests_and_guards(
    ready: tuple[Path, delivery_render.Config],
) -> None:
    path, _ = ready
    output = path.parent / "planned"
    result = planner.prepare(path, output)
    assert planner.prepare(path, output, verify=True) == result
    data = json.loads(result.read_text(encoding="utf-8"))
    assert len(data["parts"]) == 3
    assert data["nativeBBoxVerified"] is False
    assert data["renderedColourVerified"] is False
    assert data["deliveryComplete"] is False
    middle = json.loads((output / "part-000001/request.json").read_text(encoding="utf-8"))
    assert middle["recipe"]["clips"][0]["selection"] == {
        "startSeconds": 0.5,
        "durationSeconds": 1.0,
        "rate": 1,
    }
    assert middle["recipe"]["additionalVideo"][0]["selection"] == {
        "startSeconds": 0.0,
        "durationSeconds": 1.0,
        "rate": 1,
    }
    assert middle["recipe"]["muteOriginalAudio"] is True
    assert (
        workflow.main(
            [str(output / "part-000001/workflow.json"), "--state", str(output / "state"), "--check"]
        )
        == 0
    )
    with pytest.raises(FileExistsError):
        planner.prepare(path, output)


@pytest.mark.parametrize(
    "leaf", ["part-000001/request.json", "part-000001/workflow.json", "plan.json"]
)
def test_tampered_generated_records_rejected(
    ready: tuple[Path, delivery_render.Config], leaf: str
) -> None:
    path, _ = ready
    output = path.parent / "planned"
    planner.prepare(path, output)
    (output / leaf).write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="changed"):
        planner.prepare(path, output, verify=True)


@pytest.mark.parametrize(
    "body",
    [
        "<ozml/>",
        XML.replace('id="2" value="4"', 'id="3" value="4"'),
        XML.replace('value="4"', 'value="nan"'),
        XML.replace('value="4"/>', 'value="4"><curve/></parameter>'),
    ],
)
def test_motion_unknown_or_varying_y_rejected(body: str) -> None:
    with pytest.raises(ReplayError):
        planner.verify_motion_origins(body)


def test_static_motion_does_not_claim_native_bbox() -> None:
    assert planner.verify_motion_origins(XML) == 1


@pytest.mark.parametrize("value", [None, "relative", "/a/../b"])
def test_bad_canonical_path(value: object) -> None:
    with pytest.raises(ReplayError):
        planner._path(value)


@pytest.mark.parametrize("value", [True, -1, 0.2])
def test_bad_frame(value: object) -> None:
    with pytest.raises(ReplayError):
        planner._integer(value)


def test_symlink_path_rejected(tmp_path: Path) -> None:
    (tmp_path / "real").write_text("x", encoding="utf-8")
    (tmp_path / "alias").symlink_to(tmp_path / "real")
    with pytest.raises(ReplayError):
        planner._path(str(tmp_path / "alias"))


@pytest.mark.parametrize(
    "change", ["schema", "part", "manifest", "duplicate", "clock", "source", "missing"]
)
def test_bad_alpha_contracts(
    ready: tuple[Path, delivery_render.Config], change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, config = ready
    data = json.loads(path.read_text(encoding="utf-8"))
    if change == "schema":
        data["alphas"][0]["unknown"] = True
    elif change == "part":
        data["alphas"][0]["part"] = "../part.moti"
    elif change == "duplicate":
        data["alphas"].append(data["alphas"][0])
    elif change == "clock":
        monkeypatch.setattr(
            planner, "_approved", lambda *_: {path.parent / "alpha.mov": Fraction(2)}
        )
    elif change == "source":
        (path.parent / "part.moti").write_text(XML + " ", encoding="utf-8")
    else:
        manifest = path.parent / "manifest.json"
        content = json.loads(manifest.read_text(encoding="utf-8"))
        if change == "manifest":
            content["version"] = 2
        else:
            content["completed"]["unselected.moti"] = {}
        manifest.write_text(json.dumps(content), encoding="utf-8")
        data["alphas"][0]["manifest"] = pin(manifest)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReplayError):
        planner.prepare(path, config.work / "output")


def test_pins_bounds_duplicates_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError):
        planner._json(path)
    with pytest.raises(ReplayError, match="exact"):
        planner._pin({"path": str(path)})
    with pytest.raises(ReplayError, match="changed"):
        planner._pin({"path": str(path), "sha256": "0" * 64})


def test_manifest_outside_timeline(ready: tuple[Path, delivery_render.Config]) -> None:
    _, config = ready
    alpha = planner.Alpha(config.source, config.template, config.path, 60, 2)
    with pytest.raises(ReplayError, match="timeline"):
        planner.requests(config, (alpha,))


def test_native_part_limit_and_clock_preservation(
    ready: tuple[Path, delivery_render.Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config = ready
    monkeypatch.setattr(
        delivery_render,
        "clips",
        lambda *_: [delivery_render.Clip(config.source, Fraction(2), 18030)],
    )
    alpha = planner.Alpha(config.source, config.template, config.path, 0, 18030)
    parts = planner.requests(config, (alpha,))
    assert len(parts) == 2
    clips = record(parts[1]["recipe"])["clips"]
    assert isinstance(clips, list)
    assert record(clips[0])["selection"] == {
        "startSeconds": 602.0,
        "durationSeconds": 1.0,
        "rate": 1,
    }
    monkeypatch.setattr(planner, "MAX_REQUESTS", 1)
    with pytest.raises(ReplayError, match="request count"):
        planner.requests(config, (alpha,))


def test_cli_and_input_budget(
    ready: tuple[Path, delivery_render.Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = ready
    assert planner.main([str(path), str(path.parent / "out")]) == 0
    assert planner.main([str(path), str(path.parent / "out"), "--verify"]) == 0
    monkeypatch.setattr(planner, "MAX_STAGE_INPUTS", 1)
    assert planner.main([str(path), str(path.parent / "small")]) == 1


def test_cleanup_failure_not_bypassed(
    ready: tuple[Path, delivery_render.Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = ready

    def reject(_: delivery_render.Config) -> source_caption_binding.Cleanup:
        raise ReplayError("Genuine cleanup failed")

    monkeypatch.setattr(delivery_render, "verified_cleanup", reject)
    with pytest.raises(ReplayError, match="Genuine cleanup failed"):
        planner.prepare(path, path.parent / "out")


def test_existing_media_namespace_refused(ready: tuple[Path, delivery_render.Config]) -> None:
    path, config = ready
    (config.storage / "motion-alpha-000000").mkdir(parents=True)
    with pytest.raises(ReplayError, match="namespace already"):
        planner.prepare(path, path.parent / "planned")


def test_real_alpha_validator_rejects_failed_decode(
    ready: tuple[Path, delivery_render.Config],
) -> None:
    path, _ = ready
    audit = path.parent / "audit.json"
    value = json.loads(audit.read_text(encoding="utf-8"))
    value["measurements"]["parts"][0]["measurement"]["fullDecodeVerified"] = False
    audit.write_text(json.dumps(value), encoding="utf-8")
    cleanup = path.parent / "cleanup.json"
    value = json.loads(cleanup.read_text(encoding="utf-8"))
    value["captionOverlays"][0]["audit"] = pin(audit)
    cleanup.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReplayError, match="incomplete"):
        planner.prepare(path, path.parent / "planned")


def test_current_code_change_rejected_at_terminal(
    ready: tuple[Path, delivery_render.Config], monkeypatch: pytest.MonkeyPatch
) -> None:
    path, _ = ready
    calls = 0

    def capture(_: tuple[str, ...]) -> tuple[Source, ...]:
        nonlocal calls
        calls += 1
        return () if calls == 1 else (Source(path, "a" * 64, (1, 1, 1, 1, 1)),)

    monkeypatch.setattr(planner, "capture", capture)
    with pytest.raises(ReplayError, match="Code changed"):
        planner.prepare(path, path.parent / "planned")
    assert not (path.parent / "planned/plan.json").exists()


def test_bounded_metadata_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "large.json"
    path.write_text('{"example": 4}', encoding="utf-8")
    monkeypatch.setattr(planner, "MAX_METADATA_BYTES", 4)
    with pytest.raises(ValueError):
        planner._json(path)


def test_unknown_ancestor_motion_animation_rejected() -> None:
    body = XML.replace(
        "<layer>",
        '<layer><parameter id="100"><parameter id="101">'
        '<parameter id="2" value="4"><curve/></parameter></parameter></parameter>',
    )
    with pytest.raises(ReplayError, match="Animated"):
        planner.verify_motion_origins(body)
