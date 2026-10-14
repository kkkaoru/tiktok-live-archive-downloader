"""Production consumers cannot replace a failed local repair with raw footage."""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from replay import source_caption_binding as binding
from replay.core import ReplayError
from replay.fcp_part_assembly import _reference
from replay.fcp_project import GAUSSIAN_UID


@pytest.fixture
def clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    movie = tmp_path / "clean.mov"
    movie.write_bytes(b"synthetic clean media")
    monkeypatch.setattr(
        binding, "clean_source", lambda config: (tmp_path / "raw.mov", movie, Fraction(2))
    )
    return movie


def recipe(clean: Path) -> dict[str, object]:
    return {"clips": [{"sourcePath": str(clean)}], "video": {}}


def invoke(
    tmp_path: Path, clean: Path, kind: str, document: object, *, module: str = "external"
) -> None:
    artifact = tmp_path / "consumer.json"
    artifact.write_text(document if isinstance(document, str) else json.dumps(document))
    config = tmp_path / "binding.json"
    config.write_text(
        json.dumps({"version": 1, "consumer": {**_reference(artifact), "kind": kind}})
    )
    binding.validate_binding(
        config,
        consumer_inputs=(config, clean, artifact),
        consumer_command=("python", "-m", module, str(artifact)),
    )


@pytest.mark.parametrize("kind", ["native-request", "part-plan"])
@pytest.mark.parametrize("reuse", [False, True])
def test_new_and_reused_clean_artifacts_use_the_same_validation(
    tmp_path: Path, clean: Path, kind: str, reuse: bool
) -> None:
    document: object = {"recipe": recipe(clean)}
    if kind == "part-plan":
        document = [{"request": document}]
    invoke(tmp_path, clean, kind, document)
    if reuse:
        config = tmp_path / "binding.json"
        artifact = tmp_path / "consumer.json"
        binding.validate_binding(
            config, consumer_inputs=(config, clean, artifact), consumer_command=(str(artifact),)
        )


def test_failed_repair_stops_before_reading_consumer(
    tmp_path: Path, clean: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failed(config: dict[str, object]) -> tuple[Path, Path, Fraction]:
        raise ReplayError("Repair producer missing or stale")

    monkeypatch.setattr(binding, "clean_source", failed)
    with pytest.raises(ReplayError, match="producer missing"):
        invoke(tmp_path, clean, "native-request", {"recipe": recipe(clean)})


@pytest.mark.parametrize(
    "document,match",
    [
        ({"recipe": {"clips": [], "video": {}}}, "no source clips"),
        ({"recipe": {"clips": [{"sourcePath": "/raw.mov"}], "video": {}}}, "unverified source"),
    ],
)
def test_raw_or_missing_clip_cannot_hide_behind_declared_clean_input(
    tmp_path: Path, clean: Path, document: object, match: str
) -> None:
    with pytest.raises(ReplayError, match=match):
        invoke(tmp_path, clean, "native-request", document)


@pytest.mark.parametrize("field", ["masks", "additionalVideo"])
def test_no_overlay_fallback(tmp_path: Path, clean: Path, field: str) -> None:
    value = recipe(clean)
    if field == "masks":
        value["video"] = {"masks": [{"blurRadius": 64}]}
    else:
        value[field] = [{"sourcePath": "/raw.mov"}]
    with pytest.raises(ReplayError):
        invoke(tmp_path, clean, "native-request", {"recipe": value})


@pytest.mark.parametrize(
    "kind,document",
    [
        ("unknown", {}),
        ("part-plan", []),
        ("transformed-source-config", {}),
        ("delivery-render-config", {}),
    ],
)
def test_invalid_contracts_fail(tmp_path: Path, clean: Path, kind: str, document: object) -> None:
    with pytest.raises((ReplayError, ValueError)):
        invoke(tmp_path, clean, kind, document)


def test_binding_kind_must_match_actual_command(tmp_path: Path, clean: Path) -> None:
    with pytest.raises(ReplayError, match="kind does not match"):
        invoke(
            tmp_path,
            clean,
            "native-request",
            {"recipe": recipe(clean)},
            module="replay.part_render",
        )


@pytest.mark.parametrize("misplaced", [False, True])
def test_native_command_uses_bound_artifact_as_input(
    tmp_path: Path, clean: Path, misplaced: bool
) -> None:
    artifact = tmp_path / "consumer.json"
    artifact.write_text(json.dumps({"recipe": recipe(clean)}))
    config = tmp_path / "binding.json"
    config.write_text(
        json.dumps({"version": 1, "consumer": {**_reference(artifact), "kind": "native-request"}})
    )
    command = ("python", "-m", "replay.native_invocation", "run", "/native")
    arguments = ("/raw-request.json", str(artifact)) if misplaced else (str(artifact), "/output")
    if misplaced:
        with pytest.raises(ReplayError, match="positional input"):
            binding.validate_binding(
                config,
                consumer_inputs=(config, clean, artifact),
                consumer_command=command + arguments,
            )
    else:
        binding.validate_binding(
            config, consumer_inputs=(config, clean, artifact), consumer_command=command + arguments
        )


@pytest.mark.parametrize("mode", ["version", "missing-input", "wrong-command", "changed-artifact"])
def test_binding_checks_identity_and_actual_command(tmp_path: Path, clean: Path, mode: str) -> None:
    artifact = tmp_path / "consumer.json"
    artifact.write_text(json.dumps({"recipe": recipe(clean)}))
    config = tmp_path / "binding.json"
    config.write_text(
        json.dumps(
            {
                "version": 0 if mode == "version" else 1,
                "consumer": {**_reference(artifact), "kind": "native-request"},
            }
        )
    )
    if mode == "changed-artifact":
        artifact.write_text("{}")
    with pytest.raises(ReplayError):
        binding.validate_binding(
            config,
            consumer_inputs=(config, artifact)
            if mode == "missing-input"
            else (config, artifact, clean),
            consumer_command=("wrong",) if mode == "wrong-command" else (str(artifact),),
        )


@pytest.mark.parametrize("valid", [True, False])
def test_xml_binds_actual_video_resource(tmp_path: Path, clean: Path, valid: bool) -> None:
    source = clean if valid else tmp_path / "raw.mov"
    xml = (
        '<fcpxml><resources><asset hasVideo="1">'
        f'<media-rep src="{source.as_uri()}"/></asset></resources></fcpxml>'
    )
    if valid:
        invoke(tmp_path, clean, "fcpxml", xml)
    else:
        with pytest.raises(ReplayError, match="unverified source"):
            invoke(tmp_path, clean, "fcpxml", xml)


def test_xml_rejects_return_of_fixed_blur(tmp_path: Path, clean: Path) -> None:
    xml = (
        '<fcpxml><resources><asset hasVideo="1">'
        f'<media-rep src="{clean.as_uri()}"/></asset>'
        f'<effect id="g1" uid="{GAUSSIAN_UID}"/></resources>'
        '<filter-video ref="g1"/></fcpxml>'
    )
    with pytest.raises(ReplayError, match="Gaussian"):
        invoke(tmp_path, clean, "fcpxml", xml)


@pytest.mark.parametrize("kind", ["transformed-source-config", "delivery-render-config"])
def test_existing_verified_clean_contract_is_reused(tmp_path: Path, clean: Path, kind: str) -> None:
    index = tmp_path / "index.json"
    index.write_text("{}")
    fields: dict[str, object] = {
        key: _reference(index) for key in ("assemblyManifest", "assemblyReceipt", "cleanIndex")
    }
    consumer = tmp_path / "consumer.json"
    document = (
        {"version": 1, "removeSourceMasks": True, **fields}
        if kind == "transformed-source-config"
        else {"sourceChunks": str(index)}
    )
    consumer.write_text(json.dumps(document))
    config = tmp_path / "binding.json"
    config.write_text(
        json.dumps({"version": 1, **fields, "consumer": {**_reference(consumer), "kind": kind}})
    )
    binding.validate_binding(
        config, consumer_inputs=(config, consumer, clean), consumer_command=(str(consumer),)
    )
