"""Synthetic probe transport and evidence; no native execution or installed font reads."""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import pytest

from replay import (
    caption_font_identity,
    caption_native_layout,
    caption_style_defaults,
    fcp_tracking_caption,
    motion_caption,
    motion_entrance,
    motion_font_lines,
    recordings,
    stages,
)
from replay import caption_font_supplier as supplier
from replay import caption_resource_inputs as resources
from replay.core import ReplayError
from replay.recordings import record


@dataclass
class Fixture:
    fcpxml: Path
    template: Path
    font: Path
    tool: Path
    output: Path

    def run(self) -> Path:
        return supplier.supply(
            fcpxml=self.fcpxml,
            template=self.template,
            font=self.font,
            tool=self.tool,
            output=self.output,
        )


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.setattr(supplier, "ROOT", tmp_path)
    monkeypatch.setattr(resources, "REPOSITORY", tmp_path)
    modules = (
        supplier,
        motion_font_lines,
        caption_native_layout,
        caption_style_defaults,
        caption_font_identity,
        resources,
        fcp_tracking_caption,
        motion_caption,
        motion_entrance,
        recordings,
        stages,
    )
    paths = [
        tmp_path / "src/replay" / (module.__name__.rsplit(".", 1)[-1] + ".py") for module in modules
    ]
    [write(path, "synthetic source") for path in paths]
    [
        monkeypatch.setattr(module, "__file__", str(path))
        for module, path in zip(modules, paths, strict=True)
    ]
    [
        write(tmp_path / "native/apple-pro-apps" / name, "synthetic native source")
        for name in (*supplier.NATIVE_SOURCES, "Sources/ProAppsCore/CaptionTextMeasurement.swift")
    ]
    template = tmp_path / "template.moti"
    write(
        template,
        '<ozml><style><parameter id="83"><font>NotoSansJP-Thin_Bold</font>'
        '</parameter><parameter id="3" value="12"/></style></ozml>',
    )
    fcpxml = tmp_path / "title.fcpxml"
    write(
        fcpxml,
        '<fcpxml><resources><effect id="r3" src="' + template.as_uri() + '"/>'
        '</resources><title ref="r3"><text>abc\ndef\nghi</text></title></fcpxml>',
    )
    font = tmp_path / "fonts/NotoSansJP-VariableFont_wght.ttf"
    font.parent.mkdir()
    font.write_bytes(b"\x00\x01\x00\x00synthetic-font")
    monkeypatch.setattr(supplier, "font_roots", lambda: (font.parent,))
    monkeypatch.setattr(resources, "font_roots", lambda: (font.parent,))
    tool = tmp_path / "font-identity-probe"
    tool.write_bytes(b"synthetic tool never executed")
    output = tmp_path / "edits/work/proof"
    output.parent.mkdir(parents=True)

    def execute(
        command: list[str],
        *,
        stdin: BinaryIO,
        stdout: BinaryIO,
        stderr: BinaryIO,
        timeout: int,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        request = record(json.load(stdin))
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
        texts = request["texts"]
        assert isinstance(texts, list)
        response = {
            "font": identity,
            "requestedTexts": texts,
            "fallbackStatus": "measured-coretext-runs",
            "motionRenderVerified": False,
            "textRuns": [
                {
                    "text": text,
                    "runs": [
                        {
                            "font": identity,
                            "glyphCount": len(str(text)),
                            "utf16Location": 0,
                            "utf16Length": len(str(text).encode("utf-16-le")) // 2,
                        }
                    ],
                }
                for text in texts
            ],
        }
        stdout.write(json.dumps(response).encode())
        stderr.write(b"synthetic stderr")
        assert timeout == 30
        assert check is False
        assert "/edits/work/proof" in env["TMPDIR"]
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", execute)
    return Fixture(fcpxml, template, font, tool, output)


def test_fresh_actual_contexts_are_validated(fixture: Fixture) -> None:
    path = fixture.run()
    supplier.verify(path)
    receipt = record(json.loads(path.read_text()))
    assert receipt["motionRenderVerified"] is False
    assert receipt["uniqueRunPostScriptNames"] == ["NotoSansJP-Thin_Bold"]
    assert json.loads((fixture.output / "request.json").read_text()) == {
        "fontName": "NotoSansJP-Thin_Bold",
        "pointSize": 12,
        "texts": ["abc", "def", "ghi"],
    }
    assert json.loads((fixture.output / "execution-after.json").read_text())["unchanged"] is True
    with pytest.raises(ReplayError, match="new output"):
        fixture.run()


def test_readonly_verifier_detects_changed_bound_inputs(fixture: Fixture) -> None:
    path = fixture.run()
    fixture.template.write_text("changed")
    with pytest.raises(ReplayError, match="SHA256 changed"):
        supplier.verify(path)


def test_entrance_implementation_is_bound_to_new_font_proof(fixture: Fixture) -> None:
    path = fixture.run()
    Path(motion_entrance.__file__).write_text("changed entrance", encoding="utf-8")
    with pytest.raises(ReplayError, match="SHA256 changed"):
        supplier.verify(path)


def test_readonly_verifier_reconstructs_request(fixture: Fixture) -> None:
    path = fixture.run()
    request = fixture.output / "request.json"
    request.write_text("{}")
    receipt = record(json.loads(path.read_text()))
    receipt["nativeRequest"] = supplier._ref(request)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ReplayError, match="differs from the native request"):
        supplier.verify(path)


@pytest.mark.parametrize(
    "replacement",
    [
        '<title ref="other"><text>abc</text></title>',
        '<title ref="r3"/>',
        '<title ref="r3"><text/></title>',
        '<title ref="r3"><text><text-style>abc</text-style></text></title>',
        '<title ref="r3"><text>abc</text><text-style-def/></title>',
    ],
)
def test_title_binding_rejects_missing_or_overridden_text(
    fixture: Fixture, replacement: str
) -> None:
    text = fixture.fcpxml.read_text().replace(
        '<title ref="r3"><text>abc\ndef\nghi</text></title>', replacement
    )
    fixture.fcpxml.write_text(text)
    with pytest.raises(ReplayError):
        fixture.run()


@pytest.mark.parametrize(
    "template",
    [
        "<ozml/>",
        "<ozml><style><parameter><font>Other</font></parameter></style></ozml>",
        "<ozml><style><parameter><font>NotoSansJP-Thin_Bold</font></parameter></style></ozml>",
        "<ozml><style><parameter><font>NotoSansJP-Thin_Bold</font></parameter>"
        '<parameter id="3" value="nan"/></style></ozml>',
        "<ozml><style><parameter><font>NotoSansJP-Thin_Bold</font></parameter>"
        '<parameter id="3" value="12"/></style><style><parameter>'
        "<font>NotoSansJP-Thin_Bold</font></parameter>"
        '<parameter id="3" value="13"/></style></ozml>',
    ],
)
def test_template_font_contract(fixture: Fixture, template: str) -> None:
    fixture.template.write_text(template)
    with pytest.raises(ReplayError):
        fixture.run()


def test_oversized_xml_rejected(fixture: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supplier, "MAX_XML_CHARACTERS", 1)
    with pytest.raises(ReplayError, match="bounded input"):
        fixture.run()


def test_external_input_rejected(fixture: Fixture) -> None:
    with pytest.raises(ReplayError, match="inside the checkout"):
        supplier._local(Path(__file__))


@pytest.mark.parametrize("mode", ["nonzero", "empty", "changed", "oversize", "malformed"])
def test_failed_native_result_retains_evidence(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    def execute(
        command: list[str],
        *,
        stdin: BinaryIO,
        stdout: BinaryIO,
        stderr: BinaryIO,
        timeout: int,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        if mode == "changed":
            fixture.template.write_text("changed during execution")
        if mode != "empty":
            stdout.write(b"{}")
        return subprocess.CompletedProcess(command, 1 if mode == "nonzero" else 0)

    monkeypatch.setattr(subprocess, "run", execute)
    if mode == "oversize":
        monkeypatch.setattr(supplier, "MAX_RESPONSE", 1)
    with pytest.raises(ReplayError):
        fixture.run()
    assert (fixture.output / "execution-before.json").exists()
    assert (fixture.output / "execution-after.json").exists()
    assert not (fixture.output / "receipt.json").exists()


def test_cli(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "supplier",
            "--fcpxml",
            str(fixture.fcpxml),
            "--template",
            str(fixture.template),
            "--font-file",
            str(fixture.font),
            "--tool",
            str(fixture.tool),
            "--output",
            str(fixture.output),
        ],
    )
    supplier.main()
    assert capsys.readouterr().out.endswith("/edits/work/proof/receipt.json\n")


def test_verify_cli(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = fixture.run()
    monkeypatch.setattr(sys, "argv", ["supplier", "--verify", str(path)])
    supplier.main()
    assert capsys.readouterr().out == (
        "Verified current CoreText input/font evidence; Motion rendering remains unverified\n"
    )


@pytest.mark.parametrize("arguments", [[], ["--verify", "receipt", "--tool", "tool"]])
def test_cli_rejects_ambiguous_modes(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["supplier", *arguments])
    with pytest.raises(SystemExit):
        supplier.main()


def many_titles(fixture: Fixture, count: int) -> None:
    fixture.fcpxml.write_text(
        '<fcpxml><resources><effect id="r3" src="' + fixture.template.as_uri() + '"/>'
        "</resources>"
        + "".join(f'<title ref="r3"><text>word{i:05d}</text></title>' for i in range(count))
        + "</fcpxml>",
        encoding="utf-8",
    )


def test_full_replay_scale_covers_19000_unique_lines(fixture: Fixture) -> None:
    many_titles(fixture, 19_000)
    path = fixture.run()
    receipt = record(json.loads(path.read_text()))
    assert receipt["version"] == 2
    assert receipt["lineCount"] == 19_000
    assert receipt["batchCount"] == 149
    last = record(json.loads((fixture.output / "batch-000148/request.json").read_text()))["texts"]
    assert isinstance(last, list)
    assert len(last) == 56
    supplier.verify(path)


def test_batches_respect_utf16_limit() -> None:
    request = {"fontName": "NotoSansJP-Thin_Bold", "pointSize": 12, "texts": ["😀" * 60] * 35}
    batches = tuple(supplier._requests(request))
    assert len(batches) == 2
    first, second = batches[0]["texts"], batches[1]["texts"]
    assert isinstance(first, list)
    assert isinstance(second, list)
    assert len(first) == 34
    assert len(second) == 1


def test_later_native_failure_never_publishes_aggregate(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    many_titles(fixture, 129)
    successful = subprocess.run

    def fail_second(
        command: list[str],
        *,
        stdin: BinaryIO,
        stdout: BinaryIO,
        stderr: BinaryIO,
        timeout: int,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[bytes]:
        if env["TMPDIR"].endswith("batch-000001"):
            return subprocess.CompletedProcess(command, 1)
        return successful(
            command,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            check=check,
            env=env,
        )

    monkeypatch.setattr(subprocess, "run", fail_second)
    with pytest.raises(ReplayError, match="probe failed"):
        fixture.run()
    assert (fixture.output / "batch-000000/receipt.json").exists()
    assert (fixture.output / "batch-000001/execution-after.json").exists()
    assert not (fixture.output / "receipt.json").exists()


@pytest.mark.parametrize("mode", ["omit", "reorder", "duplicate", "count", "unverified", "changed"])
def test_aggregate_rejects_missing_reordered_or_changed_batches(
    fixture: Fixture, mode: str
) -> None:
    many_titles(fixture, 129)
    path = fixture.run()
    receipt = record(json.loads(path.read_text()))
    refs = receipt["batches"]
    assert isinstance(refs, list)
    if mode == "omit":
        receipt["batches"] = refs[:1]
    elif mode == "reorder":
        receipt["batches"] = refs[::-1]
    elif mode == "duplicate":
        receipt["batches"] = [refs[0], refs[0]]
    elif mode == "count":
        receipt["lineCount"] = 1
    elif mode == "unverified":
        receipt["motionRenderVerified"] = True
    else:
        (fixture.output / "batch-000001/response.json").write_text("{}")
    path.write_text(json.dumps(receipt))
    with pytest.raises(ReplayError):
        supplier.verify(path)


@pytest.mark.parametrize("key", ["fonts", "template", "primary", "executionBefore"])
def test_aggregate_requires_identical_sources_and_fonts(fixture: Fixture, key: str) -> None:
    many_titles(fixture, 129)
    path = fixture.run()
    child_path = fixture.output / "batch-000001/receipt.json"
    child = record(json.loads(child_path.read_text()))
    if key == "executionBefore":
        before = fixture.output / "batch-000001/execution-before.json"
        document = record(json.loads(before.read_text()))
        document["inputs"] = []
        before.write_text(json.dumps(document))
        child[key] = supplier._ref(before)
    else:
        child[key] = {}
    child_path.write_text(json.dumps(child))
    receipt = record(json.loads(path.read_text()))
    refs = receipt["batches"]
    assert isinstance(refs, list)
    refs[1] = supplier._ref(child_path)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ReplayError):
        supplier.verify(path)


@pytest.mark.parametrize(
    "xml",
    [
        '<!DOCTYPE fcpxml [<!ENTITY x "expanded">]><fcpxml/>',
        '<!DOCTYPE fcpxml SYSTEM "file:///unused"><fcpxml/>',
        "<unexpected/>",
        "<fcpxml>",
        '<fcpxml><resources><effect id="x"/><effect id="x"/></resources></fcpxml>',
        "<fcpxml><title><title/></title></fcpxml>",
        "<fcpxml><title><param><text>abc</text></param></title></fcpxml>",
        "<fcpxml><title><text>" + "a" * 365 + "</text></title></fcpxml>",
    ],
)
def test_streaming_parser_rejects_unsafe_or_invalid_xml(fixture: Fixture, xml: str) -> None:
    fixture.fcpxml.write_text(xml)
    with pytest.raises(ReplayError):
        fixture.run()


def test_streaming_parser_accepts_empty_doctype(fixture: Fixture) -> None:
    fixture.fcpxml.write_text("<!DOCTYPE fcpxml>" + fixture.fcpxml.read_text())
    assert supplier.title_request(fixture.fcpxml, fixture.template)["texts"] == [
        "abc",
        "def",
        "ghi",
    ]


@pytest.mark.parametrize("bound", ["MAX_UNIQUE_LINES", "MAX_TITLES", "MAX_FCPXML_BYTES"])
def test_streaming_limits_fail_without_sampling(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch, bound: str
) -> None:
    monkeypatch.setattr(supplier, bound, 0)
    with pytest.raises(ReplayError):
        fixture.run()


@pytest.mark.parametrize("texts", [[], None, [1]])
def test_batches_reject_invalid_requests(texts: object) -> None:
    with pytest.raises(ReplayError):
        tuple(supplier._requests({"texts": texts}))
