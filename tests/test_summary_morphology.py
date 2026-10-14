import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Never

import pytest
from vibrato import Vibrato

from replay import caption_linguistics as linguistics
from replay import summary_caption_inputs as canonical
from replay import summary_morphology as morphology
from replay.core import ReplayError
from replay.stages import file_digest
from replay.workflow import load_stages


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(morphology, "ROOT", tmp_path)
    monkeypatch.setattr(canonical, "ROOT", tmp_path)
    monkeypatch.setattr(morphology, "CODE_FILES", ())
    monkeypatch.setattr(canonical, "CODE_FILES", ())
    (tmp_path / "edits" / "work").mkdir(parents=True)
    (tmp_path / ".python-version").write_text("3.12.13\n")
    (tmp_path / "uv.lock").write_text("synthetic pinned package lock\n")
    (tmp_path / "pyproject.toml").write_text("synthetic package settings\n")
    dictionary = tmp_path / "synthetic.dic"
    dictionary.write_bytes(b"synthetic dictionary")
    monkeypatch.setattr(linguistics, "DICTIONARY_PATH", dictionary)
    monkeypatch.setattr(linguistics, "DICTIONARY_SHA256", file_digest(dictionary))
    tokenizer = Vibrato.from_textdict(
        "今日,0,0,0,名詞,一般\nは,0,0,0,助詞,係助詞\n空,0,0,0,名詞,一般\n",
        "1 1\n0 0 0\n",
        "DEFAULT 0 1 0\nSPACE 0 1 0\n0x0020 SPACE\n",
        "DEFAULT,0,0,10000,名詞,一般\nSPACE,0,0,10000,記号,空白\n",
        ignore_space=False,
        max_grouping_len=1,
    )

    def load_tokenizer(*, max_grouping_len: int) -> Vibrato:
        assert max_grouping_len == 1
        return tokenizer

    monkeypatch.setattr(linguistics, "_load_tokenizer", load_tokenizer)
    captions = tmp_path / "captions"
    captions.mkdir()
    tokens = captions / "tokens-0000.json"
    tokens.write_text(
        '{"version":1,"clock":"output","tokens":[{"text":"今日は空","start":0,"end":1,"utterance":"speech"}]}'
    )
    index = captions / "segments.json"
    index.write_text(
        '{"version":1,"rate":30,"segments":[{"segment":0,"offsetFrame":0,"frames":30,"jobs":[0]}]}'
    )
    reflow = tmp_path / "reflow.json"
    reflow.write_text(
        json.dumps({"inputs": {str(index): file_digest(index), str(tokens): file_digest(tokens)}})
    )
    xml = tmp_path / "display.fcpxml"
    xml.write_text(
        '<fcpxml><library><event><project><sequence duration="1s"><spine>'
        '<asset-clip offset="0s" start="0s" duration="1s">'
        '<title offset="0s" duration="1s"><text>今日は空</text></title>'
        "</asset-clip></spine></sequence></project></event></library></fcpxml>"
    )
    srt = tmp_path / "display.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n今日は空\n")
    media = tmp_path / "actual.mov"
    media.write_bytes(b"regular identity only")
    source_config = tmp_path / "canonical-config.json"
    source_config.write_text(
        json.dumps(
            {
                "version": 1,
                "captions": str(captions),
                "segmentIndex": {"path": str(index), "sha256": file_digest(index)},
                "reflow": {"path": str(reflow), "sha256": file_digest(reflow)},
                "displayXML": {"path": str(xml), "sha256": file_digest(xml)},
                "displaySRT": {"path": str(srt), "sha256": file_digest(srt)},
                "media": str(media),
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(canonical.load(source_config).manifest))
    preview = tmp_path / "preview.json"
    preview.write_bytes(
        '{"sections":[{"endSeconds":600,"startSeconds":0,"text":"今日は空"}]}\n'.encode()
    )
    executable = tmp_path / "frozen-consumer"
    executable.write_bytes(b"synthetic frozen preview consumer")
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "preview": {"path": str(preview), "sha256": file_digest(preview)},
                "canonicalManifest": {"path": str(manifest), "sha256": file_digest(manifest)},
                "canonicalConfig": {
                    "path": str(source_config),
                    "sha256": file_digest(source_config),
                },
                "dictionary": {"path": str(dictionary), "sha256": file_digest(dictionary)},
                "previewExecutable": {"path": str(executable), "sha256": file_digest(executable)},
            }
        )
    )
    return path


def test_real_vibrato_pos_retains_surfaces_and_exact_character_ranges(config: Path) -> None:
    assert morphology.tokens("今日は空") == [
        {
            "text": "今日",
            "lowerCharacter": 0,
            "upperCharacter": 2,
            "partOfSpeech": "名詞",
            "features": ["名詞", "一般"],
        },
        {
            "text": "は",
            "lowerCharacter": 2,
            "upperCharacter": 3,
            "partOfSpeech": "助詞",
            "features": ["助詞", "係助詞"],
        },
        {
            "text": "空",
            "lowerCharacter": 3,
            "upperCharacter": 4,
            "partOfSpeech": "名詞",
            "features": ["名詞", "一般"],
        },
    ]


def test_whitespace_and_newlines_remain_in_complete_coverage(config: Path) -> None:
    tokens = morphology.tokens("今日 \n空")
    assert "".join(str(token["text"]) for token in tokens) == "今日 \n空"
    assert tokens[0]["lowerCharacter"] == 0
    assert tokens[-1]["upperCharacter"] == 5


@pytest.mark.parametrize("text", ["か\u3099", "👩\u200d💻", "🇯🇵"])
def test_native_codepoint_boundary_inside_grapheme_is_rejected(
    config: Path, text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = Vibrato.from_textdict(
        f"{text[0]},0,0,0,名詞,一般\n",
        "1 1\n0 0 0\n",
        "DEFAULT 0 1 0\n",
        "DEFAULT,0,0,10000,名詞,一般\n",
        max_grouping_len=1,
    )
    monkeypatch.setattr(linguistics, "_load_tokenizer", lambda *, max_grouping_len: tokenizer)
    with pytest.raises(ReplayError, match="grapheme"):
        morphology.tokens(text)


def test_empty_source_is_rejected(config: Path) -> None:
    with pytest.raises(ReplayError, match="omitted"):
        morphology.tokens("")


def test_token_budget_is_bounded(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(morphology, "MAX_SECTION_TOKENS", 1)
    with pytest.raises(ReplayError, match="coverage"):
        morphology.tokens("今日は空")


def test_load_preserves_full_canonical_token_pins(config: Path) -> None:
    loaded = morphology.load(config)
    assert loaded.sections == ({"endSeconds": 600, "startSeconds": 0, "text": "今日は空"},)
    assert str(config.parent / "captions" / "tokens-0000.json") in loaded.pins
    assert loaded.preview.endswith(b"\n")


@pytest.mark.parametrize("selection", [[], [1], [600, 600], [True], "all", [-600]])
def test_invalid_section_selection_rejected(config: Path, selection: object) -> None:
    with pytest.raises(ReplayError, match="section selection"):
        morphology._manifest_matches({"version": 1, "sectionStarts": selection}, {"version": 1})


def test_explicit_section_selection_keeps_other_bindings() -> None:
    morphology._manifest_matches({"version": 1, "sectionStarts": [1800]}, {"version": 1})
    with pytest.raises(ReplayError, match="differs"):
        morphology._manifest_matches({"version": 2, "sectionStarts": [1800]}, {"version": 1})


def test_manifest_version_jobs_flags_and_explicit_selection_are_strict() -> None:
    with pytest.raises(ReplayError, match="integer version"):
        morphology._manifest_matches({"version": True}, {"version": 1})
    with pytest.raises(ReplayError, match="integer membership"):
        morphology._manifest_matches({"jobs": [{"job": True}]}, {"jobs": [{"job": 1}]})
    with pytest.raises(ReplayError, match="booleans"):
        morphology._manifest_matches({"candidateOnly": 1}, {"candidateOnly": True})
    morphology._manifest_matches({"version": 1}, {"version": 1, "candidateOnly": True})
    with pytest.raises(ReplayError, match="section selection"):
        morphology._manifest_matches({"sectionStarts": None}, {})


def test_unlocked_tokenizer_version_is_rejected(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(morphology, "version", lambda name: "unknown")
    with pytest.raises(ReplayError, match="locked Vibrato"):
        morphology.load(config)


@pytest.mark.parametrize(
    "data", [b"{}\n", b'{"sections":[]}\n', b'{"sections":null}\n', b'{"sections":[]}']
)
def test_invalid_preview_shape_or_terminal_lf(data: bytes) -> None:
    with pytest.raises(ReplayError):
        morphology._sections(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("startSeconds", True),
        ("startSeconds", -1),
        ("endSeconds", 0),
        ("endSeconds", "600"),
        ("text", ""),
        ("text", "a\0b"),
        ("text", 1),
        ("text", "\ud800"),
    ],
)
def test_invalid_preview_section_values(field: str, value: object) -> None:
    section: dict[str, object] = {"startSeconds": 0, "endSeconds": 600, "text": "今日"}
    section[field] = value
    with pytest.raises(ReplayError):
        morphology._sections((json.dumps({"sections": [section]}) + "\n").encode())


def test_preview_order_fields_and_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ReplayError):
        morphology._sections(
            b'{"sections":[{"startSeconds":0,"endSeconds":600,"text":"a","extra":1}]}\n'
        )
    with pytest.raises(ReplayError):
        morphology._sections(
            b'{"sections":[{"startSeconds":0,"endSeconds":600,"text":"a"},{"startSeconds":0,"endSeconds":600,"text":"b"}]}\n'
        )
    with pytest.raises(ReplayError, match="terminal"):
        morphology._sections(b'{"sections":[]}\n\n')
    monkeypatch.setattr(morphology, "MAX_SECTION_CODEPOINTS", 1)
    with pytest.raises(ReplayError):
        morphology._sections(b'{"sections":[{"startSeconds":0,"endSeconds":600,"text":"ab"}]}\n')


def test_preview_utf8_and_huge_clock_are_bounded() -> None:
    with pytest.raises(ReplayError, match="UTF8"):
        morphology._sections(b"\xff\n")
    with pytest.raises(ReplayError, match="finite"):
        morphology._clock(10**1000)


def test_empty_features_invalid_pos_and_complete_whitespace_are_measured(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = Vibrato.from_textdict(
        "今日,0,0,0,Unknown\n",
        "1 1\n0 0 0\n",
        "DEFAULT 0 1 0\n",
        "DEFAULT,0,0,10000,名詞,一般\n",
        max_grouping_len=1,
    )
    monkeypatch.setattr(linguistics, "_load_tokenizer", lambda *, max_grouping_len: tokenizer)
    with pytest.raises(ReplayError, match="coverage"):
        morphology.tokens("今日")


def test_pinned_audit_namespace_cannot_be_a_file(config: Path) -> None:
    output = config.parent / "edits" / "work" / "morphology.json"
    (output.parent / "morphology-audits").write_bytes(b"occupied")
    with pytest.raises(ReplayError, match="audit scratch"):
        morphology.produce(config, output)
    with pytest.raises(ReplayError, match="repository work"):
        morphology.produce(config, config.parent / "unsafe.json")


@pytest.mark.parametrize("value", [None, "", "relative", "nul\0path"])
def test_unsafe_path_rejected_before_read(value: object) -> None:
    with pytest.raises(ReplayError):
        morphology._path(value)


@pytest.mark.parametrize("directory", ["private", "downloads"])
def test_sensitive_path_rejected_before_content(config: Path, directory: str) -> None:
    path = config.parent / directory
    path.mkdir()
    file = path / "user.json"
    file.write_bytes(b"user data")
    with pytest.raises(ReplayError, match="canonical checkout"):
        morphology._pin({"path": str(file), "sha256": "0" * 64}, {})


def test_link_and_external_path_rejected(config: Path) -> None:
    link = config.parent / "link.json"
    link.symlink_to(config)
    with pytest.raises(ReplayError):
        morphology._path(str(link))
    with pytest.raises(ReplayError):
        morphology._path(str(Path(__file__).resolve()))


@pytest.mark.parametrize("pin", [{}, {"path": "relative", "sha256": "invalid"}])
def test_bad_pin_shapes(pin: dict[str, object]) -> None:
    with pytest.raises(ReplayError):
        morphology._pin(pin, {})


def test_bad_digest_and_changed_input(config: Path) -> None:
    with pytest.raises(ReplayError, match="SHA256"):
        morphology._pin({"path": str(config), "sha256": "bad"}, {})
    with pytest.raises(ReplayError, match="changed"):
        morphology._pin({"path": str(config), "sha256": "0" * 64}, {})


def test_changed_dictionary_and_invalid_config_version(config: Path) -> None:
    raw = json.loads(config.read_text())
    wrong = config.parent / "wrong.dic"
    wrong.write_bytes(b"wrong dictionary")
    raw["dictionary"] = {"path": str(wrong), "sha256": file_digest(wrong)}
    config.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="unchanged pinned"):
        morphology.load(config)
    raw["version"] = True
    config.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="version1"):
        morphology.load(config)


def test_missing_source_scope_is_checked_before_canonical_reader(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = json.loads(config.read_text())
    raw["preview"]["path"] = "outside"
    config.write_text(json.dumps(raw))

    def reject_reader(_path: Path) -> Never:
        raise AssertionError("canonical reader must not run after unsafe source path")

    monkeypatch.setattr(canonical, "load", reject_reader)
    with pytest.raises(ReplayError, match="canonical checkout"):
        morphology.load(config)


def test_total_token_bytes_guard(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(morphology, "MAX_TOKEN_BYTES", 1)
    with pytest.raises(ReplayError, match="token bytes"):
        morphology.load(config)


def test_exact_fingerprint_pos_clocks_and_repeatable_verification(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = config.parent / "edits" / "work" / "morphology.json"

    def preview(inputs: morphology.Inputs, directory: Path) -> None:
        (directory / "actual-preview.json").write_bytes(inputs.preview)

    monkeypatch.setattr(morphology, "_verify_preview", preview)
    assert morphology.produce(config, output) == output
    raw = json.loads(output.read_text())
    assert (
        raw["transcriptFingerprint"]
        == hashlib.sha256((config.parent / "preview.json").read_bytes()[:-1]).hexdigest()
    )
    assert raw["sections"][0]["startSeconds"] == 0
    assert raw["sections"][0]["endSeconds"] == 600
    assert raw["sections"][0]["tokens"][1]["partOfSpeech"] == "助詞"
    assert raw["recognitionClocksChanged"] is False
    assert raw["summaryQualityVerified"] is False
    morphology.produce(config, output, existing=True)
    morphology.produce(config, output, existing=True)
    with pytest.raises(ReplayError, match="new canonical"):
        morphology.produce(config, output)
    raw["sections"][0]["tokens"][0]["text"] = "改変"
    output.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="Saved morphology"):
        morphology.produce(config, output, existing=True)


def test_bounded_workflow_inventory_is_verified_transitively(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = morphology.plan(config, config.parent / "edits" / "work" / "planned")
    stages = load_stages(graph)
    assert len(stages) == 1
    assert len(stages[0].inputs) <= 256
    assert stages[0].verify_command[-1] == "--verify"
    inventory = graph.parent / "input-inventory.json"
    monkeypatch.setattr(morphology, "_verify_preview", lambda inputs, directory: None)
    morphology.produce(config, graph.parent / "morphology.json", inventory=inventory)
    raw = json.loads(inventory.read_text())
    raw["inputs"].pop(str(config.parent / "captions" / "tokens-0000.json"))
    inventory.write_text(json.dumps(raw))
    with pytest.raises(ReplayError, match="full transitive"):
        morphology.produce(
            config, graph.parent / "morphology.json", existing=True, inventory=inventory
        )


def test_terminal_source_mutation_prevents_publication(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(_inputs: morphology.Inputs, _directory: Path) -> None:
        (config.parent / "captions" / "tokens-0000.json").write_text("mutated")

    monkeypatch.setattr(morphology, "_verify_preview", mutate)
    output = config.parent / "edits" / "work" / "morphology.json"
    with pytest.raises(ValueError, match="changed"):
        morphology.produce(config, output)
    assert not output.exists()


def test_frozen_preview_child_receives_real_repository_scratch(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = morphology.load(config)
    directory = config.parent / "edits" / "work" / "child"
    directory.mkdir()

    def fake_run(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        assert command[1:] == ["preview", "--caption-tokens", str(config.parent / "manifest.json")]
        assert options["cwd"] == config.parent
        environment = options["env"]
        assert isinstance(environment, dict)
        assert environment["TMPDIR"] == str(directory)
        assert str(environment["LLVM_PROFILE_FILE"]).startswith(str(directory))
        (directory / "preview-check.log").write_bytes(inputs.preview)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    morphology._verify_preview(inputs, directory)


def test_preview_mismatch_is_rejected(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = morphology.load(config)
    directory = config.parent / "edits" / "work" / "mismatch"
    directory.mkdir()

    def fake_run(command: list[str], **_options: object) -> subprocess.CompletedProcess[bytes]:
        (directory / "preview-check.log").write_bytes(b"different")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ReplayError, match="exact canonical"):
        morphology._verify_preview(inputs, directory)


def test_cli_plan_and_produce(
    config: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = config.parent / "edits" / "work" / "cli"
    monkeypatch.setattr(sys, "argv", ["summary-morphology", "plan", str(config), str(directory)])
    morphology.main()
    assert "workflow.json" in capsys.readouterr().out
    monkeypatch.setattr(morphology, "_verify_preview", lambda inputs, directory: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summary-morphology", "produce", str(config), str(directory / "morphology.json")],
    )
    morphology.main()
    assert "morphology.json" in capsys.readouterr().out
