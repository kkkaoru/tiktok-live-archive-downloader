"""Bounded FCP title text to fresh CoreText evidence; never Motion raster approval.

All title text in the supplied FCPXML is measured, including prefix contexts.
Oversize inputs fail explicitly; this CLI does not silently sample or truncate.
Application resources and renderer consumption require independent evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from xml.parsers import expat

from replay import (
    caption_font_identity,
    caption_native_layout,
    caption_resource_inputs,
    caption_style_defaults,
    fcp_tracking_caption,
    motion_caption,
    motion_entrance,
    motion_font_lines,
    recordings,
    stages,
)
from replay.caption_resource_inputs import ResourceInputs, _font_evidence, canonical, font_roots
from replay.core import ReplayError
from replay.motion_caption import MAX_XML_CHARACTERS, _parse
from replay.recordings import record
from replay.stages import file_digest

ROOT = Path(__file__).resolve().parents[2]
MAX_RESPONSE = 1024 * 1024
MAX_FCPXML_BYTES = 256 * 1024 * 1024
MAX_UNIQUE_LINES = 65_536
MAX_TITLES = 200_000
NATIVE_SOURCES = (
    "Sources/FontIdentityProbe/FontIdentity.swift",
    "Sources/FontIdentityProbe/Main.swift",
    "Package.swift",
    "Package.resolved",
)


def _save(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=1)


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_digest(path)}


def _local(path: Path) -> Path:
    path = canonical(str(path.absolute()))
    if not path.is_relative_to(ROOT):
        raise ReplayError("Font supplier inputs must stay inside the checkout")
    return path


class _TitleTextCollector:
    """Streaming title text only; discard tracking keyframes and media attributes."""

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.effects: dict[str, str] = {}
        self.refs: set[str] = set()
        self.lines: dict[str, None] = {}
        self.title_count = 0
        self.body_count = 0
        self.in_title = False
        self.body: str | None = None

    def start(self, name: str, attributes: dict[str, str]) -> None:
        if not self.stack and name != "fcpxml":
            raise ReplayError("Expected FCPXML root")
        if len(self.stack) > 64 or self.body is not None:
            raise ReplayError("Unsupported nested XML or title text overrides")
        if name == "effect" and self.stack == ["fcpxml", "resources"]:
            key = attributes.get("id", "")
            if key in self.effects:
                raise ReplayError("Duplicate FCP effect identifier")
            self.effects[key] = attributes.get("src", "")
        if name == "title":
            if self.in_title or self.title_count >= MAX_TITLES:
                raise ReplayError("Unsupported nested or excessive titles")
            self.in_title = True
            self.body_count = 0
            self.title_count += 1
            self.refs.add(attributes.get("ref", ""))
        if self.in_title and name == "text-style-def":
            raise ReplayError("Title style overrides are unsupported")
        if self.in_title and name == "text":
            if self.stack[-1] != "title":
                raise ReplayError("Expected direct title text")
            self.body = ""
        self.stack.append(name)

    def data(self, text: str) -> None:
        if self.body is not None:
            self.body += text
            if len(self.body) > caption_style_defaults.MAX_CAPTION_LINES * 121 - 1:
                raise ReplayError("Title text exceeds native line bounds")

    def end(self, name: str) -> None:
        if name == "text" and self.body is not None:
            self.lines.update(dict.fromkeys(motion_font_lines.placement_lines((self.body,))))
            if len(self.lines) > MAX_UNIQUE_LINES:
                raise ReplayError("FCP title unique line budget exceeded")
            self.body = None
            self.body_count += 1
        if name == "title":
            if not self.body_count:
                raise ReplayError("Every title requires plain text")
            self.in_title = False
        self.stack.pop()


def _doctype(name: str, system: str | None, public: str | None, internal: int) -> None:
    if name != "fcpxml" or system is not None or public is not None or internal:
        raise ReplayError("Custom or external XML declarations are unsupported")


def _title_lines(fcpxml: Path, template: Path) -> tuple[str, ...]:
    collector = _TitleTextCollector()
    parser = expat.ParserCreate()
    parser.StartElementHandler = collector.start
    parser.EndElementHandler = collector.end
    parser.CharacterDataHandler = collector.data
    parser.StartDoctypeDeclHandler = _doctype
    try:
        with fcpxml.open("rb") as stream:
            parser.ParseFile(stream)
    except expat.ExpatError as error:
        raise ReplayError("Invalid FCPXML") from error
    refs = {key for key, value in collector.effects.items() if value == template.as_uri()}
    if not collector.title_count or len(refs) != 1 or collector.refs != refs:
        raise ReplayError("All titles must reference the supplied Motion template")
    return tuple(collector.lines)


def title_request(fcpxml: Path, template: Path) -> dict[str, object]:
    """Read exact plain text titles referencing one selected template, without media."""
    if fcpxml.stat().st_size > MAX_FCPXML_BYTES or template.stat().st_size > MAX_XML_CHARACTERS:
        raise ReplayError("Font supplier XML exceeds its bounded input size")
    lines = _title_lines(fcpxml, template)
    motion = _parse(template.read_text(encoding="utf-8"))
    styles = [
        style for style in motion.findall(".//style") if style.find("parameter/font") is not None
    ]
    if not styles:
        raise ReplayError("Motion template has no font styles")
    sizes: set[float] = set()
    for style in styles:
        caption_font_identity.require_selected_name(style.findtext("parameter/font") or "")
        size = style.find("parameter[@id='3']")
        try:
            value = float(size.get("value", "") if size is not None else "")
        except ValueError as error:
            raise ReplayError("Motion font size is missing or invalid") from error
        if not math.isfinite(value) or not 1 <= value <= 1024:
            raise ReplayError("Motion font size exceeds native bounds")
        sizes.add(value)
    if len(sizes) != 1:
        raise ReplayError("One native request requires a single Motion font size")
    return {
        "fontName": caption_style_defaults.FONT_POSTSCRIPT_NAME,
        "pointSize": sizes.pop(),
        "texts": list(lines),
    }


def _requests(request: dict[str, object]) -> Iterator[dict[str, object]]:
    texts = request["texts"]
    if not isinstance(texts, list) or not texts:
        raise ReplayError("Expected native font request texts")
    batch: list[str] = []
    units = 0
    for text in texts:
        if not isinstance(text, str):
            raise ReplayError("Expected text in native request")
        count = len(text.encode("utf-16-le")) // 2
        if batch and (len(batch) == 128 or units + count > 4096):
            yield {**request, "texts": batch}
            batch, units = [], 0
        batch.append(text)
        units += count
    yield {**request, "texts": batch}


def supply(*, fcpxml: Path, template: Path, font: Path, tool: Path, output: Path) -> Path:
    """Execute a bounded read-only native probe and validate fresh receipt semantics."""
    fcpxml, template, tool = map(_local, (fcpxml, template, tool))
    output = output.absolute()
    parent = _local(output.parent)
    if not parent.is_relative_to(ROOT / "edits/work") or output.exists() or output.is_symlink():
        raise ReplayError("Use a new output directory under checkout edits/work")
    store = ResourceInputs()
    font_ref = _ref(canonical(str(font.absolute())))
    store.asset(font_ref, roots=font_roots(), font=True)
    contexts = (
        Path(motion_font_lines.__file__).resolve(),
        Path(caption_native_layout.__file__).resolve(),
        ROOT / "native/apple-pro-apps/Sources/ProAppsCore/CaptionTextMeasurement.swift",
    )
    sources = (
        tool,
        fcpxml,
        template,
        Path(__file__).resolve(),
        Path(caption_style_defaults.__file__).resolve(),
        Path(caption_font_identity.__file__).resolve(),
        Path(caption_resource_inputs.__file__).resolve(),
        Path(fcp_tracking_caption.__file__).resolve(),
        Path(motion_caption.__file__).resolve(),
        Path(motion_entrance.__file__).resolve(),
        Path(recordings.__file__).resolve(),
        Path(stages.__file__).resolve(),
        *contexts,
        *(ROOT / "native/apple-pro-apps" / name for name in NATIVE_SOURCES),
    )
    inputs = [_ref(store.bind(path)) for path in sources] + [font_ref]
    request = title_request(fcpxml, template)
    batches = tuple(_requests(request))
    if len(batches) == 1:
        return _supply_request(
            request=batches[0],
            output=output,
            inputs=inputs,
            font_ref=font_ref,
            contexts=contexts,
            fcpxml=fcpxml,
            template=template,
            tool=tool,
        )
    output.mkdir(mode=0o700)
    receipts: list[dict[str, str]] = []
    for index, batch in enumerate(batches):
        path = _supply_request(
            request=batch,
            output=output / f"batch-{index:06d}",
            inputs=inputs,
            font_ref=font_ref,
            contexts=contexts,
            fcpxml=fcpxml,
            template=template,
            tool=tool,
        )
        receipts.append(_ref(path))
    aggregate = {
        "version": 2,
        "fcpxml": _ref(fcpxml),
        "template": _ref(template),
        "fonts": [font_ref],
        "batchCount": len(batches),
        "lineCount": len(request["texts"]) if isinstance(request["texts"], list) else 0,
        "batches": receipts,
        "motionRenderVerified": False,
    }
    candidate, receipt = output / "aggregate-candidate.json", output / "receipt.json"
    _save(candidate, aggregate)
    verify(candidate)
    _save(receipt, aggregate)
    return receipt


def _supply_request(
    *,
    request: dict[str, object],
    output: Path,
    inputs: list[dict[str, str]],
    font_ref: dict[str, str],
    contexts: tuple[Path, ...],
    fcpxml: Path,
    template: Path,
    tool: Path,
) -> Path:
    store = ResourceInputs()
    output.mkdir(mode=0o700)
    request_path, response_path = output / "request.json", output / "response.json"
    before, after, receipt_path = (
        output / "execution-before.json",
        output / "execution-after.json",
        output / "receipt.json",
    )
    _save(request_path, request)
    request_ref = _ref(request_path)
    _save(
        before,
        {
            "command": [str(tool)],
            "stdin": request_ref,
            "inputs": inputs,
            "timeoutSeconds": 30,
            "textScope": motion_font_lines.SCOPE,
        },
    )
    environment = dict(os.environ)
    environment.update(
        TMPDIR=str(output),
        TMP=str(output),
        TEMP=str(output),
        LLVM_PROFILE_FILE=str(output / "font-%p-%m.profraw"),
    )
    with (
        request_path.open("rb") as source,
        response_path.open("xb") as response,
        (output / "stderr.log").open("xb") as error_log,
    ):
        result = subprocess.run(
            [str(tool)],
            stdin=source,
            stdout=response,
            stderr=error_log,
            timeout=30,
            check=False,
            env=environment,
        )
    unchanged = all(_ref(Path(pin["path"])) == pin for pin in inputs)
    unchanged = unchanged and _ref(request_path) == request_ref
    _save(
        after,
        {
            "inputs": inputs,
            "returncode": result.returncode,
            "unchanged": unchanged,
            "response": _ref(response_path) if response_path.stat().st_size else None,
        },
    )
    if result.returncode or not unchanged or not 0 < response_path.stat().st_size <= MAX_RESPONSE:
        raise ReplayError("Native font probe failed, changed inputs or exceeded response bounds")
    response_data = store.read(response_path)
    receipt = {
        "version": 1,
        "fcpxml": _ref(fcpxml),
        "template": _ref(template),
        "nativeRequest": request_ref,
        "nativeResponse": _ref(response_path),
        "executionBefore": _ref(before),
        "executionAfter": _ref(after),
        "fonts": [font_ref],
        "primary": record(response_data.get("font")),
        "fallbackStatus": "measured-coretext-runs",
        "uniqueRunPostScriptNames": [caption_style_defaults.FONT_POSTSCRIPT_NAME],
        "motionRenderVerified": False,
        "textScope": motion_font_lines.SCOPE,
        "contextSources": [_ref(path) for path in contexts],
    }
    _save(output / "candidate.json", receipt)
    motion: dict[str, object] = {
        "resolvedAssets": {"fonts": [font_ref]},
        "finish": ["--font-name", request["fontName"], "--font-size", str(request["pointSize"])],
    }
    requested = request["texts"]
    if not isinstance(requested, list):
        raise ReplayError("Expected native font request texts")
    _font_evidence(
        store,
        motion,
        {"fontIdentity": _ref(output / "candidate.json")},
        {str(text) for text in requested},
    )
    _save(receipt_path, receipt)
    return receipt_path


def verify(receipt_path: Path) -> None:
    """Recheck current text/template/code/font pins without executing the native tool."""
    store = ResourceInputs()
    receipt_path = _local(receipt_path)
    receipt = store.read(receipt_path)
    fcpxml = store.reference(receipt.get("fcpxml"))
    template = store.reference(receipt.get("template"))
    request = title_request(fcpxml, template)
    if receipt.get("version") == 2:
        texts = request["texts"]
        if not isinstance(texts, list):
            raise ReplayError("Expected aggregate font request texts")
        batches = tuple(_requests(request))
        refs = receipt.get("batches")
        if (
            not isinstance(refs, list)
            or len(refs) != len(batches)
            or receipt.get("batchCount") != len(batches)
            or receipt.get("lineCount") != len(texts)
            or receipt.get("motionRenderVerified") is not False
        ):
            raise ReplayError("Aggregate font batch coverage differs")
        previous: dict[str, object] | None = None
        paths: set[Path] = set()
        for ref, expected in zip(refs, batches, strict=True):
            path = store.reference(ref)
            if path in paths:
                raise ReplayError("Duplicate font batch receipt")
            paths.add(path)
            child = store.read(path)
            if any(child.get(key) != receipt.get(key) for key in ("fcpxml", "template", "fonts")):
                raise ReplayError("Aggregate font source or font identity differs")
            before = store.document(child.get("executionBefore"))
            identity = {"inputs": before.get("inputs"), "primary": child.get("primary")}
            if previous is not None and identity != previous:
                raise ReplayError("Font batches bind different source or font identities")
            previous = identity
            _verify_request(store, path, child, expected)
        return
    _verify_request(store, receipt_path, receipt, request)


def _verify_request(
    store: ResourceInputs,
    receipt_path: Path,
    receipt: dict[str, object],
    request: dict[str, object],
) -> None:
    if store.document(receipt.get("nativeRequest")) != request:
        raise ReplayError("Current FCP text or Motion style differs from the native request")
    motion: dict[str, object] = {
        "resolvedAssets": {"fonts": receipt.get("fonts")},
        "finish": ["--font-name", request["fontName"], "--font-size", str(request["pointSize"])],
    }
    requested = request["texts"]
    if not isinstance(requested, list):
        raise ReplayError("Expected native font request texts")
    _font_evidence(
        store, motion, {"fontIdentity": _ref(receipt_path)}, {str(text) for text in requested}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--fcpxml", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--font-file", type=Path)
    parser.add_argument("--tool", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    supplied = (args.fcpxml, args.template, args.font_file, args.tool, args.output)
    if args.verify is not None:
        if any(value is not None for value in supplied):
            parser.error("--verify cannot be combined with supplier inputs")
        verify(args.verify)
        print("Verified current CoreText input/font evidence; Motion rendering remains unverified")
        return
    if any(value is None for value in supplied):
        parser.error("Supply --fcpxml, --template, --font-file, --tool and --output")
    print(
        supply(
            fcpxml=args.fcpxml,
            template=args.template,
            font=args.font_file,
            tool=args.tool,
            output=args.output,
        )
    )


if __name__ == "__main__":
    main()
