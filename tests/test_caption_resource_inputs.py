"""Synthetic data-only evidence; no installed resources or native execution."""

import json
import plistlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from replay import caption_resource_inputs as resources
from replay import (
    compressor_batch,
    motion_dependencies,
    motion_font_lines,
    motion_resource_inventory,
    stages,
)
from replay.core import ReplayError
from replay.recordings import record


def write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": stages.file_digest(path)}


@dataclass
class Chain:
    config: Path
    plan: Path
    source: Path
    state: dict[str, object]
    font: Path
    root: Path

    def validate(self) -> resources.ResourceInputs:
        return resources.validate(
            config=self.config, plan=self.plan, source=self.source, state=self.state
        )


@pytest.fixture
def chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Chain:
    root = tmp_path / "repo"
    root.mkdir()
    font_root = tmp_path / "fonts"
    font_root.mkdir()
    font = font_root / "NotoSansJP-VariableFont_wght.ttf"
    font.write_bytes(b"\x00\x01\x00\x00synthetic-font")
    app = tmp_path / "Motion.app"
    (app / "Contents/MacOS").mkdir(parents=True)
    (app / "Contents/Resources").mkdir()
    executable = app / "Contents/MacOS/Motion"
    executable.write_bytes(b"edition")
    info = app / "Contents/Info.plist"
    info.write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": "com.apple.motionapp",
                "CFBundleExecutable": "Motion",
                "CFBundleShortVersionString": "1.0",
                "CFBundleVersion": "1",
            }
        )
    )
    helper = root / "consumer.py"
    helper.write_bytes(b"synthetic consumer pin")
    inventory_code = root / "inventory.py"
    inventory_code.write_bytes(b"synthetic inventory pin")
    monkeypatch.setattr(resources, "REPOSITORY", root)
    monkeypatch.setattr(resources, "__file__", str(helper))
    monkeypatch.setattr(motion_resource_inventory, "__file__", str(inventory_code))
    monkeypatch.setattr(resources, "font_roots", lambda: (font_root,))
    monkeypatch.setattr(resources, "application_root", lambda _: app)
    native = root / "font-identity-probe"
    native.write_bytes(b"mock tool, never invoked")
    source_paths = [
        root / "native/apple-pro-apps" / name
        for name in (
            "Sources/FontIdentityProbe/FontIdentity.swift",
            "Sources/FontIdentityProbe/Main.swift",
            "Package.swift",
            "Package.resolved",
        )
    ]
    [write(path, "synthetic native source") for path in source_paths]
    identity = {
        "ascent": 10,
        "descent": 2,
        "face": "Bold",
        "family": "Noto Sans JP",
        "fontURL": font.as_uri(),
        "pointSize": 12,
        "postScriptName": "NotoSansJP-Thin_Bold",
        "variations": [{"identifier": 2003265652, "value": 700}],
    }
    request = write(
        root / "request.json",
        {"fontName": "NotoSansJP-Thin_Bold", "pointSize": 12, "texts": ["abc"]},
    )
    response = write(
        root / "response.json",
        {
            "font": identity,
            "requestedTexts": ["abc"],
            "fallbackStatus": "measured-coretext-runs",
            "motionRenderVerified": False,
            "textRuns": [
                {
                    "text": "abc",
                    "runs": [
                        {"font": identity, "glyphCount": 3, "utf16Location": 0, "utf16Length": 3}
                    ],
                }
            ],
        },
    )
    pins = [ref(path) for path in [native, *source_paths]]
    before = write(
        root / "before.json", {"stdin": ref(request), "command": [str(native)], "inputs": pins}
    )
    after = write(
        root / "after.json",
        {"response": ref(response), "returncode": 0, "unchanged": True, "inputs": pins},
    )
    font_receipt = write(
        root / "font.json",
        {
            "version": 1,
            "nativeRequest": ref(request),
            "nativeResponse": ref(response),
            "executionBefore": ref(before),
            "executionAfter": ref(after),
            "fonts": [ref(font)],
            "primary": identity,
            "fallbackStatus": "measured-coretext-runs",
            "uniqueRunPostScriptNames": ["NotoSansJP-Thin_Bold"],
            "motionRenderVerified": False,
        },
    )
    resource = app / "Contents/Resources/synthetic.bin"
    resource.write_bytes(b"synthetic application data")
    template = root / "template.moti"
    template.write_text(
        f'<ozml version="5.13"><scene><file>{resource}</file></scene></ozml>', encoding="utf-8"
    )
    inventory = write(
        root / "inventory.json",
        motion_resource_inventory.inventory(template, allowed_roots=(root, app)).report(),
    )
    app_receipt = write(
        root / "app.json",
        {
            "version": 1,
            "kind": "motion-application-resources",
            "template": ref(template),
            "inventory": ref(inventory),
            "resources": [ref(resource)],
            "application": {
                "info": ref(info),
                "executable": ref(executable),
                "bundleIdentifier": "com.apple.motionapp",
                "shortVersion": "1.0",
                "buildVersion": "1",
            },
        },
    )
    workflow_path = root / "prepare.json"
    state_directory = root / "prepare-state"
    state_directory.mkdir()
    evidence = {
        "version": 1,
        "fontIdentity": ref(font_receipt),
        "applicationResources": ref(app_receipt),
        "preparation": {"workflow": str(workflow_path), "stateDirectory": str(state_directory)},
    }
    preset = root / "alpha.setting"
    preset.write_bytes(b"preset")
    motion: dict[str, object] = {
        "template": str(template),
        "finish": ["--font-name", "NotoSansJP-Thin_Bold", "--font-size", "12"],
        "resolvedAssets": {"fonts": [ref(font)], "resources": [ref(resource)]},
        "resourceEvidence": evidence,
    }
    config = write(root / "config.json", {"motion": motion, "alphaPreset": str(preset)})
    plan = write(root / "plan.json", {"placements": [{"text": "abc"}]})
    collected = resources.ResourceInputs()
    collected.read(config)
    collected.read(plan)
    resources._font_evidence(collected, motion, evidence, {"abc"})
    resources._application_evidence(collected, motion, evidence)
    collected.bind(helper)
    prepared, source = root / "prepared.moti", root / "finished.moti"
    prepared.write_bytes(b"synthetic prepared")
    source.write_bytes(b"synthetic finished")
    calibration = write(root / "calibration.json", {"synthetic": True})
    inputs = tuple(
        dict.fromkeys(
            (
                *collected.pins,
                *motion_dependencies.preparation_inputs(
                    config=config, template=template, plan=plan, calibration=calibration
                ),
            )
        )
    )
    first = stages.Stage(
        name="prepare",
        command=("python", "-m", "replay.motion_stages"),
        inputs=inputs,
        outputs=(prepared,),
        timeout_seconds=30,
    )
    finish = stages.Stage(
        name="finish",
        command=("python", "-m", "replay.motion_finish"),
        inputs=(*inputs, prepared),
        outputs=(source,),
        timeout_seconds=30,
    )
    write(
        workflow_path,
        {
            "version": 1,
            "stages": [
                {
                    "name": stage.name,
                    "command": stage.command,
                    "inputs": list(map(str, stage.inputs)),
                    "outputs": list(map(str, stage.outputs)),
                    "timeout_seconds": stage.timeout_seconds,
                }
                for stage in (first, finish)
            ],
        },
    )
    receipts = [
        write(
            state_directory / f"{stage.name}.json",
            {
                "status": "complete",
                "fingerprint": stages._fingerprint(stage),
                "outputs": {str(path): stages.file_digest(path) for path in stage.outputs},
            },
        )
        for stage in (first, finish)
    ]
    compressor = root / "Compressor"
    compressor.write_bytes(b"mock compressor")
    compressor.chmod(0o700)
    output = root / "output.mov"
    job = compressor_batch.Job(
        "caption", source, preset, output, (*inputs, workflow_path, *receipts)
    )

    def cli(command: Sequence[str]) -> str:
        if "-monitor" in command:
            return '[{"jobStatus":{"jobID":"synthetic","status":"Successful"}}]'
        output.write_bytes(b"synthetic, not media")
        return '{"batch":{"jobs":[{"jobID":"synthetic"}]}}'

    worker = compressor_batch.Batch(root / "batch", compressor, "mock", cli, lambda _: None)
    assert compressor_batch.run([job], worker, in_flight=1) == 1
    state = record(json.loads((root / "batch/caption.json").read_text()))
    return Chain(config, plan, source, state, font, root)


def test_preparation_does_not_require_future_workflow_or_receipts(chain: Chain) -> None:
    (chain.root / "prepare.json").unlink()
    (chain.root / "prepare-state/prepare.json").unlink()
    result = resources.preparation_inputs(config=chain.config, plan=chain.plan)
    assert chain.font in result.pins
    assert chain.root / "before.json" in result.pins
    assert chain.root / "prepare.json" not in result.pins
    with pytest.raises(FileNotFoundError):
        resources.completed_preparation(config=chain.config, plan=chain.plan, source=chain.source)


@pytest.mark.parametrize("scope", [motion_font_lines.SCOPE, "unknown", None])
def test_line_context_is_explicit(
    chain: Chain, monkeypatch: pytest.MonkeyPatch, scope: str | None
) -> None:
    line_code = write(chain.root / "lines.py", "synthetic line contract")
    monkeypatch.setattr(motion_font_lines, "__file__", str(line_code))
    layout = write(chain.root / "src/replay/caption_native_layout.py", "synthetic layout")
    measurement = write(
        chain.root / "native/apple-pro-apps/Sources/ProAppsCore/CaptionTextMeasurement.swift",
        "synthetic measurement",
    )
    path = chain.root / "font.json"
    receipt = record(json.loads(path.read_text()))
    if scope is not None:
        receipt["textScope"] = scope
    receipt["contextSources"] = [ref(line_code), ref(layout), ref(measurement)]
    write(path, receipt)
    raw = record(json.loads(chain.config.read_text()))
    motion = record(raw["motion"])
    evidence = record(motion["resourceEvidence"])
    evidence["fontIdentity"] = ref(path)
    if scope == motion_font_lines.SCOPE:
        resources._font_evidence(resources.ResourceInputs(), motion, evidence, {"abc\nabc"})
    else:
        with pytest.raises(ReplayError):
            resources._font_evidence(resources.ResourceInputs(), motion, evidence, {"abc\nabc"})


def test_exact_font_chain(chain: Chain) -> None:
    result = chain.validate()
    assert len(result.data) == 2
    assert chain.font in result.data
    assert chain.font in result.pins
    assert not hasattr(result, "motionRenderVerified")


def test_selected_run_rejects_other_weight_even_with_same_file(chain: Chain) -> None:
    response = record(json.loads((chain.root / "response.json").read_text(encoding="utf-8")))
    identity = record(response["font"])
    identity["variations"] = [{"identifier": 2003265652, "value": 400}]
    with pytest.raises(ReplayError, match="variable file or weight"):
        resources._run_identity(
            {
                "text": "abc",
                "runs": [{"font": identity, "glyphCount": 3, "utf16Location": 0, "utf16Length": 3}],
            },
            {chain.font},
            selected=True,
        )


@pytest.mark.parametrize(
    "filename",
    [
        "font.json",
        "request.json",
        "response.json",
        "before.json",
        "after.json",
        "app.json",
        "inventory.json",
        "consumer.py",
    ],
)
def test_changed_proof_is_rejected(chain: Chain, filename: str) -> None:
    (chain.root / filename).write_bytes(b"changed")
    with pytest.raises((ReplayError, ValueError)):
        chain.validate()


@pytest.mark.parametrize("change", ["sha", "executable", "symlink", "secret"])
def test_font_is_data_not_external_exception(chain: Chain, change: str) -> None:
    if change == "sha":
        chain.font.write_bytes(b"\x00\x01\x00\x00changed")
    elif change == "executable":
        chain.font.chmod(0o700)
    elif change == "symlink":
        chain.font.rename(chain.font.with_suffix(".original"))
        chain.font.symlink_to(chain.font.with_suffix(".original"))
    else:
        chain.font.write_bytes(b"synthetic secret, not a font")
    with pytest.raises(ReplayError):
        chain.validate()


@pytest.mark.parametrize(
    "key,value",
    [
        ("submissionVersion", 1),
        ("dependencies", []),
        ("terminalInputsVerified", False),
        ("status", "Running"),
    ],
)
def test_old_or_partial_state_cannot_be_upgraded(chain: Chain, key: str, value: object) -> None:
    chain.state[key] = value
    with pytest.raises(ReplayError):
        chain.validate()


@pytest.mark.parametrize("key,value", [("status", "running"), ("fingerprint", "0" * 64)])
def test_prepare_receipt_required(chain: Chain, key: str, value: object) -> None:
    path = chain.root / "prepare-state/prepare.json"
    data = record(json.loads(path.read_text()))
    data[key] = value
    write(path, data)
    with pytest.raises(ReplayError, match="receipt"):
        chain.validate()


def test_no_arbitrary_font_roots(chain: Chain, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "font_roots", lambda: (chain.root,))
    with pytest.raises(ReplayError, match="Foreign"):
        chain.validate()


def test_missing_supplier_contract(chain: Chain) -> None:
    raw = record(json.loads(chain.config.read_text()))
    record(raw["motion"]).pop("resourceEvidence")
    write(chain.config, raw)
    with pytest.raises(ReplayError):
        chain.validate()


def test_os_roots_require_canonical_existing_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / "Library/Fonts").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    assert home / "Library/Fonts" in resources.font_roots()
    (home / "Library/Fonts").rmdir()
    (home / "Library/Fonts").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ReplayError):
        resources.font_roots()


@pytest.mark.parametrize("change", ["variation", "strings", "range", "fallback", "source"])
def test_native_semantics_not_just_file_pins(chain: Chain, change: str) -> None:
    response_path = chain.root / "response.json"
    response = record(json.loads(response_path.read_text()))
    if change == "variation":
        record(response["font"])["variations"] = [{"identifier": 1, "value": 400}]
    elif change == "strings":
        response["requestedTexts"] = ["different"]
    elif change == "range":
        response["textRuns"] = [
            {
                "text": "abc",
                "runs": [
                    {
                        "font": response["font"],
                        "glyphCount": 2,
                        "utf16Location": 1,
                        "utf16Length": 2,
                    }
                ],
            }
        ]
    elif change == "fallback":
        response["fallbackStatus"] = "unknown"
    else:
        (chain.root / "native/apple-pro-apps/Sources/FontIdentityProbe/Main.swift").write_bytes(
            b"changed"
        )
    write(response_path, response)
    after_path = chain.root / "after.json"
    after = record(json.loads(after_path.read_text()))
    after["response"] = ref(response_path)
    write(after_path, after)
    receipt_path = chain.root / "font.json"
    receipt = record(json.loads(receipt_path.read_text()))
    receipt.update(nativeResponse=ref(response_path), executionAfter=ref(after_path))
    write(receipt_path, receipt)
    raw = record(json.loads(chain.config.read_text()))
    motion = record(raw["motion"])
    evidence = record(motion["resourceEvidence"])
    evidence["fontIdentity"] = ref(receipt_path)
    with pytest.raises(ReplayError):
        resources._font_evidence(resources.ResourceInputs(), motion, evidence, {"abc"})


@pytest.mark.parametrize(
    "identifier,expected",
    [
        ("com.apple.motionapp", "/Applications/Motion.app"),
        ("com.apple.motionappApp", "/Applications/Motion Creator Studio.app"),
    ],
)
def test_application_root_is_edition_specific(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, identifier: str, expected: str
) -> None:
    seen: list[object] = []

    def canonical(value: object) -> Path:
        seen.append(value)
        return tmp_path

    monkeypatch.setattr(resources, "canonical", canonical)
    assert resources.application_root(identifier) == tmp_path
    assert seen == [expected]


@pytest.mark.parametrize("identifier", ["foreign.app", None, "/Applications/Other.app"])
def test_no_arbitrary_application_edition(identifier: object) -> None:
    with pytest.raises(ReplayError, match="edition"):
        resources.application_root(identifier)
