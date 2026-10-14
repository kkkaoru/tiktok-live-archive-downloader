"""Caption-only external data evidence; never Motion raster/font approval.

Local receipts are trusted provenance, not authenticated native attestations.
No directory allowance is exported to other completion domains.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import plistlib
import stat
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlsplit

from replay import compressor_batch, motion_font_lines, motion_resource_inventory, stages, workflow
from replay.caption_font_identity import require_selected_identity
from replay.caption_measured_job import _pairs
from replay.core import ReplayError
from replay.recordings import record

REPOSITORY = Path(__file__).resolve().parents[2]
MAX_FILE = 256 * 1024 * 1024
MAX_JSON = 8 * 1024 * 1024
FONT_SIGNATURES = {b"\x00\x01\x00\x00", b"OTTO", b"ttcf"}


def require(value: bool, message: str) -> None:
    if not value:
        raise ReplayError(message)


def canonical(value: object) -> Path:
    require(isinstance(value, str), "Resource path must be absolute text")
    path = Path(str(value))
    require(
        path.is_absolute()
        and path == path.resolve(strict=True)
        and not any(parent.is_symlink() for parent in (path, *path.parents))
        and "private" not in path.parts,
        "Resource path must be canonical, nonlinked and nonprivate",
    )
    return path


def rows(value: object, maximum: int = 256) -> list[dict[str, object]]:
    require(isinstance(value, list) and len(value) <= maximum, "Invalid resource evidence array")
    if not isinstance(value, list):
        raise ReplayError("Expected array")
    return [record(item) for item in value]


def font_roots() -> tuple[Path, ...]:
    candidates = (
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path.home() / "Library/Fonts",
    )
    return tuple(canonical(str(path)) for path in candidates if path.is_dir())


def application_root(identifier: object) -> Path:
    editions = {
        "com.apple.motionapp": "/Applications/Motion.app",
        "com.apple.motionappApp": "/Applications/Motion Creator Studio.app",
    }
    require(isinstance(identifier, str) and identifier in editions, "Unknown Motion edition")
    root = canonical(editions[str(identifier)])
    require(root.is_dir(), "Installed Motion edition is missing")
    return root


@dataclass
class ResourceInputs:
    pins: dict[Path, str] = field(default_factory=dict)
    data: set[Path] = field(default_factory=set)

    def bind(self, path: Path, digest: object = None, *, external: bool = False) -> Path:
        path = canonical(str(path))
        require(
            external or path.is_relative_to(REPOSITORY), "Resource evidence must stay in checkout"
        )
        before = path.stat()
        require(
            stat.S_ISREG(before.st_mode)
            and before.st_nlink == 1
            and 0 < before.st_size <= MAX_FILE,
            "Resource must be a bounded nonempty single-link regular file",
        )
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        after = path.stat()
        fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        require(
            all(getattr(before, key) == getattr(after, key) for key in fields),
            "Resource changed during reading",
        )
        require(digest is None or digest == actual, "Resource SHA256 changed")
        require(path not in self.pins or self.pins[path] == actual, "Conflicting resource pins")
        self.pins[path] = actual
        return path

    def reference(self, value: object, *, external: bool = False) -> Path:
        ref = record(value)
        require(set(ref) == {"path", "sha256"}, "Resource reference requires exact path/SHA256")
        digest = ref["sha256"]
        require(isinstance(digest, str) and len(digest) == 64, "Resource reference needs SHA256")
        return self.bind(canonical(ref["path"]), digest, external=external)

    def read(self, path: Path) -> dict[str, object]:
        self.bind(path)
        require(path.stat().st_size <= MAX_JSON, "Resource JSON budget exceeded")
        return record(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs))

    def document(self, value: object) -> dict[str, object]:
        return self.read(self.reference(value))

    def asset(self, value: object, *, roots: tuple[Path, ...], font: bool) -> Path:
        ref = record(value)
        path = canonical(ref.get("path"))
        require(any(path.is_relative_to(root) for root in roots), "Foreign resource root")
        require(not os.access(path, os.X_OK), "Executable cannot masquerade as resource data")
        self.reference(value, external=True)
        if font:
            with path.open("rb") as stream:
                require(stream.read(4) in FONT_SIGNATURES, "Not font data")
        self.data.add(path)
        return path


def _identity(value: object, fonts: set[Path]) -> dict[str, object]:
    identity = record(value)
    require(
        set(identity)
        == {
            "ascent",
            "descent",
            "face",
            "family",
            "fontURL",
            "pointSize",
            "postScriptName",
            "variations",
        },
        "Incomplete native font identity",
    )
    url = urlsplit(str(identity["fontURL"]))
    require(
        url.scheme == "file" and not url.netloc and not url.query and not url.fragment,
        "Invalid native font URL",
    )
    require(canonical(unquote(url.path)) in fonts, "Undeclared native font")
    require(
        all(
            isinstance(identity[key], str) and bool(identity[key])
            for key in ("face", "family", "postScriptName")
        ),
        "Missing font names",
    )
    require(
        all(
            type(identity[key]) in (int, float)
            and math.isfinite(float(str(identity[key])))
            and float(str(identity[key])) > 0
            for key in ("ascent", "descent", "pointSize")
        ),
        "Invalid font metrics",
    )
    variations = rows(identity["variations"], 64)
    require(
        all(
            set(row) == {"identifier", "value"}
            and type(row["identifier"]) is int
            and type(row["value"]) in (int, float)
            and math.isfinite(float(str(row["value"])))
            for row in variations
        ),
        "Invalid font variations",
    )
    require(
        len({row["identifier"] for row in variations}) == len(variations), "Duplicate variation"
    )
    return identity


def _font_evidence(
    store: ResourceInputs, motion: dict[str, object], evidence: dict[str, object], texts: set[str]
) -> None:
    receipt = store.document(evidence["fontIdentity"])
    require(
        receipt.get("version") == 1 and receipt.get("motionRenderVerified") is False,
        "Font receipt is not unapproved native selection evidence",
    )
    fonts = {
        store.asset(ref, roots=font_roots(), font=True) for ref in rows(receipt.get("fonts"), 64)
    }
    require(bool(fonts), "Missing actual font data")
    assets = record(motion.get("resolvedAssets"))
    require(
        rows(assets.get("fonts"), 64) == rows(receipt.get("fonts"), 64), "Declared fonts differ"
    )
    request = store.document(receipt.get("nativeRequest"))
    response = store.document(receipt.get("nativeResponse"))
    before = store.document(receipt.get("executionBefore"))
    after = store.document(receipt.get("executionAfter"))
    require(
        before.get("stdin") == receipt.get("nativeRequest")
        and after.get("response") == receipt.get("nativeResponse")
        and after.get("returncode") == 0
        and after.get("unchanged") is True
        and before.get("inputs") == after.get("inputs"),
        "Native execution binding differs",
    )
    sources = {
        store.reference(ref, external=canonical(ref.get("path")) in fonts)
        for ref in rows(before.get("inputs"))
    }
    command = before.get("command")
    require(isinstance(command, list) and len(command) == 1, "Invalid native font command")
    if not isinstance(command, list):
        raise ReplayError("Missing command")
    tool = canonical(command[0])
    require(tool in sources and tool.name == "font-identity-probe", "Missing native font tool pin")
    required = {
        REPOSITORY / "native/apple-pro-apps" / name
        for name in (
            "Sources/FontIdentityProbe/FontIdentity.swift",
            "Sources/FontIdentityProbe/Main.swift",
            "Package.swift",
            "Package.resolved",
        )
    }
    require(required <= sources, "Missing native font source pins")
    primary = _identity(response.get("font"), fonts)
    require(
        receipt.get("primary") == primary
        and request.get("fontName") == primary["postScriptName"]
        and request.get("pointSize") == primary["pointSize"],
        "Selected font/variation mismatch",
    )
    finish = motion.get("finish")
    require(isinstance(finish, list), "Missing Motion finish font")
    if not isinstance(finish, list):
        raise ReplayError("Missing finish")
    require(
        finish.count("--font-name") == finish.count("--font-size") == 1,
        "Ambiguous Motion font selection",
    )
    name_index, size_index = finish.index("--font-name") + 1, finish.index("--font-size") + 1
    require(max(name_index, size_index) < len(finish), "Incomplete finish font")
    require(
        finish[name_index] == primary["postScriptName"]
        and float(str(finish[size_index])) == primary["pointSize"],
        "Motion font differs",
    )
    selected_font = "--top-srgb" not in finish and "--bottom-srgb" not in finish
    if selected_font:
        require_selected_identity(primary)
    requested = request.get("texts")
    require(
        isinstance(requested, list)
        and 0 < len(requested) <= 128
        and all(isinstance(text, str) and 0 < len(text) <= 32768 for text in requested),
        "Invalid requested font strings",
    )
    if not isinstance(requested, list):
        raise ReplayError("Missing texts")
    if receipt.get("textScope") == motion_font_lines.SCOPE:
        texts = set(motion_font_lines.placement_lines(sorted(texts)))
        context_sources = {store.reference(ref) for ref in rows(receipt.get("contextSources"))}
        require(
            {
                Path(motion_font_lines.__file__).resolve(),
                REPOSITORY / "src/replay/caption_native_layout.py",
                REPOSITORY
                / "native/apple-pro-apps/Sources/ProAppsCore/CaptionTextMeasurement.swift",
            }
            <= context_sources,
            "Missing per-line measurement context sources",
        )
    else:
        require("textScope" not in receipt, "Unknown font text scope")
    require(
        response.get("requestedTexts") == requested and texts <= set(requested),
        "Actual placement strings lack native font evidence",
    )
    require(
        response.get("fallbackStatus") == receipt.get("fallbackStatus") == "measured-coretext-runs"
        and response.get("motionRenderVerified") is False,
        "Unmeasured font runs",
    )
    measured = rows(response.get("textRuns"), 128)
    require(
        [row.get("text") for row in measured] == requested, "Native text/run correspondence differs"
    )
    names = {_run_identity(row, fonts, selected=selected_font) for row in measured}
    observed = set().union(*names)
    require(
        sorted(observed) == receipt.get("uniqueRunPostScriptNames"), "Run identity summary differs"
    )


def _run_identity(
    row: dict[str, object], fonts: set[Path], *, selected: bool = False
) -> frozenset[str]:
    text = row.get("text")
    require(isinstance(text, str), "Invalid run text")
    length = len(str(text).encode("utf-16-le")) // 2
    runs = rows(row.get("runs"), 32768)
    require(bool(runs), "Missing actual native runs")
    intervals: list[tuple[int, int]] = []
    names: set[str] = set()
    for run in runs:
        start, count, glyphs = (
            run.get("utf16Location"),
            run.get("utf16Length"),
            run.get("glyphCount"),
        )
        if type(start) is not int or type(count) is not int or type(glyphs) is not int:
            raise ReplayError("Invalid native run range")
        require(
            start >= 0 and count > 0 and glyphs > 0 and start + count <= length,
            "Invalid native run extent",
        )
        intervals.append((start, start + count))
        identity = _identity(run.get("font"), fonts)
        if selected:
            require_selected_identity(identity)
        names.add(str(identity["postScriptName"]))
    ordered = sorted(intervals)
    require(
        ordered[0][0] == 0
        and ordered[-1][1] == length
        and all(left[1] == right[0] for left, right in zip(ordered, ordered[1:], strict=False)),
        "Native runs omit or overlap requested text",
    )
    return frozenset(names)


def _application_evidence(
    store: ResourceInputs, motion: dict[str, object], evidence: dict[str, object]
) -> None:
    receipt = store.document(evidence["applicationResources"])
    require(
        receipt.get("version") == 1 and receipt.get("kind") == "motion-application-resources",
        "Unknown application resource receipt",
    )
    template = store.reference(receipt.get("template"))
    require(str(template) == motion.get("template"), "Resource template differs")
    edition = record(receipt.get("application"))
    root = application_root(edition.get("bundleIdentifier"))
    info_ref, exe_ref = record(edition.get("info")), record(edition.get("executable"))
    require(
        canonical(info_ref.get("path")) == root / "Contents/Info.plist"
        and canonical(exe_ref.get("path")) == root / "Contents/MacOS/Motion",
        "Not the installed Motion edition",
    )
    info = store.reference(info_ref, external=True)
    store.reference(exe_ref, external=True)
    require(info.stat().st_size <= MAX_JSON, "Application info budget exceeded")
    raw: object = plistlib.loads(info.read_bytes())
    plist = record(raw)
    require(
        plist.get("CFBundleExecutable") == "Motion"
        and edition.get("bundleIdentifier") == plist.get("CFBundleIdentifier")
        and edition.get("shortVersion") == plist.get("CFBundleShortVersionString")
        and edition.get("buildVersion") == plist.get("CFBundleVersion")
        and isinstance(edition.get("shortVersion"), str)
        and isinstance(edition.get("buildVersion"), str),
        "Installed application edition changed",
    )
    resources = {
        store.asset(ref, roots=(root / "Contents/Resources",), font=False)
        for ref in rows(receipt.get("resources"), 64)
    }
    require(
        rows(record(motion.get("resolvedAssets")).get("resources"), 64)
        == rows(receipt.get("resources"), 64),
        "Declared application resources differ",
    )
    inventory = store.document(receipt.get("inventory"))
    store.bind(Path(motion_resource_inventory.__file__).resolve())
    actual = motion_resource_inventory.inventory(
        template, allowed_roots=(REPOSITORY, root)
    ).report()
    require(inventory == json.loads(json.dumps(actual)), "Resource classification changed")
    fields = rows(inventory.get("fields"), 100_000)
    require(all(row.get("kind") != "unknown" for row in fields), "Unknown template dependency")
    external = [row for row in fields if row.get("kind") == "external-file"]
    require(
        all(row.get("status") == "resolved" for row in external), "Unresolved template dependency"
    )
    resolved = {canonical(row.get("path")) for row in external}
    require(
        resources == {path for path in resolved if not path.is_relative_to(REPOSITORY)},
        "Undeclared application dependency",
    )
    for path in resolved:
        store.bind(path, external=path in resources)


def preparation_inputs(*, config: Path, plan: Path) -> ResourceInputs:
    """Collect verified evidence before preparation; never read future stage receipts."""
    store = ResourceInputs()
    raw = store.read(config)
    motion = record(raw.get("motion"))
    evidence = record(motion.get("resourceEvidence"))
    require(
        set(evidence) == {"version", "fontIdentity", "applicationResources", "preparation"}
        and evidence["version"] == 1,
        "Missing explicit resource evidence",
    )
    placements = rows(store.read(plan).get("placements"), 32)
    texts = {str(row.get("text")) for row in placements}
    require(bool(texts), "Missing prepared strings")
    _font_evidence(store, motion, evidence, texts)
    _application_evidence(store, motion, evidence)
    store.bind(Path(__file__).resolve())
    preparation = record(evidence["preparation"])
    require(set(preparation) == {"workflow", "stateDirectory"}, "Invalid preparation binding")
    return store


def completed_preparation(*, config: Path, plan: Path, source: Path) -> ResourceInputs:
    """Require exact completed producers, retaining evidence for alpha submission."""
    store = preparation_inputs(config=config, plan=plan)
    required = set(store.pins)
    motion = record(store.read(config).get("motion"))
    evidence = record(motion.get("resourceEvidence"))
    preparation = record(evidence["preparation"])
    require(set(preparation) == {"workflow", "stateDirectory"}, "Invalid preparation binding")
    manifest = store.bind(canonical(preparation["workflow"]))
    directory = canonical(preparation["stateDirectory"])
    require(directory.is_relative_to(REPOSITORY) and directory.is_dir(), "Foreign prepare state")
    definitions = workflow.load_stages(manifest)
    finishes = [stage for stage in definitions if source in stage.outputs]
    require(len(finishes) == 1, "Missing exact finish producer")
    finish = finishes[0]
    prepares = [stage for stage in definitions if set(stage.outputs) & set(finish.inputs)]
    require(len(prepares) == 1, "Missing exact prepare producer")
    for stage, module in ((prepares[0], "replay.motion_stages"), (finish, "replay.motion_finish")):
        require(
            not stage.adopt_existing
            and stage.command[1:3] == ("-m", module)
            and required <= set(stage.inputs),
            "Prepare omits resource proof inputs",
        )
        saved = store.read(directory / f"{stage.name}.json")
        require(
            saved
            == {
                "status": "complete",
                "fingerprint": stages._fingerprint(stage),
                "outputs": {str(path): stages.file_digest(path) for path in stage.outputs},
            },
            "Preparation receipt changed or incomplete",
        )
    return store


def validate(*, config: Path, plan: Path, source: Path, state: dict[str, object]) -> ResourceInputs:
    """Verify data and fresh prepare/submit pins; return exact caption-only file scope."""
    store = completed_preparation(config=config, plan=plan, source=source)
    raw = store.read(config)
    pins = {
        canonical(row.get("path")): row.get("sha256") for row in rows(state.get("dependencies"))
    }
    require(
        state.get("submissionVersion") == 2
        and state.get("terminalInputsVerified") is True
        and state.get("terminalRuntimeVerified") is True
        and state.get("status") == "Successful"
        and all(pins.get(path) == digest for path, digest in store.pins.items()),
        "Submission omits fresh resource/prepare pins",
    )
    job_config = record(state.get("jobConfig"))
    require(
        job_config.get("source") == str(source)
        and job_config.get("preset") == raw.get("alphaPreset"),
        "Resource submission source/preset differs",
    )
    job = compressor_batch.Job(
        str(job_config.get("name")),
        source,
        canonical(job_config.get("preset")),
        canonical(job_config.get("output")),
        tuple(pins),
    )
    require(
        compressor_batch.verify_submission(job, state, rehash=True), "Unbound resource submission"
    )
    return store
