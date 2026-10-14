"""Synthetic ownership audit inputs; never private recordings."""

import json
from pathlib import Path

import pytest
from test_caption_linguistics import synthetic_dictionary as synthetic_dictionary

from replay import caption_continuity_audit as audit
from replay.stages import file_digest


@pytest.fixture
def tokens(tmp_path: Path) -> Path:
    path = tmp_path / "tokens-0000.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "clock": "output",
                "tokens": [
                    {"text": "フ", "start": 0, "end": 0.2, "utterance": "000000000000:0:cut-1"},
                    {
                        "text": "ォロワー",
                        "start": 0.2,
                        "end": 0.5,
                        "utterance": "000000000000:1:cut-1",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_units_and_cli(
    tokens: Path, capsys: pytest.CaptureFixture[str], synthetic_dictionary: object
) -> None:
    assert audit.main([str(tokens), "--include-units"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["measuredLayoutVerified"] is False
    assert result["audit"]["assembledGroupTokens"] == 2
    assert result["audit"]["normalizedOwnerTokens"] == 2
    assert result["audit"]["continuityOwners"] == 1
    assert result["audit"]["lexicalUnits"] == [
        {"text": "フォロワー", "start": 0, "end": 0.5, "utterance": "assembled:cut-1"}
    ]


def test_pinned_reflow_only_adopted_tokens(
    tokens: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "reflow.json"
    path.write_text(
        json.dumps(
            {
                "inputs": {
                    str(tokens): file_digest(tokens),
                    **dict.fromkeys(map(str, range(129)), "ignored"),
                }
            }
        ),
        encoding="utf-8",
    )
    assert audit.main([str(path), "--reflow"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["audit"]["files"]) == 1
    assert "lexicalUnits" not in result["audit"]["files"][0]


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 2, "clock": "output", "tokens": []},
        {"version": 1, "clock": "source", "tokens": []},
        {"version": 1, "clock": "output", "tokens": []},
    ],
)
def test_invalid_input(tmp_path: Path, payload: dict[str, object]) -> None:
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        audit.audit_tokens(path)


def test_changed_token(tokens: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digests = iter(["first", "second"])
    monkeypatch.setattr(audit, "file_digest", lambda path: next(digests))
    with pytest.raises(ValueError, match="input changed"):
        audit.audit_tokens(tokens)


def test_bad_reflow_pin(tokens: Path, tmp_path: Path) -> None:
    path = tmp_path / "reflow.json"
    path.write_text(json.dumps({"inputs": {str(tokens): "bad"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="pin mismatch"):
        audit.audit_reflow(path)


def test_missing_reflow_tokens(tmp_path: Path) -> None:
    path = tmp_path / "reflow.json"
    path.write_text('{"inputs": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="count exceeds"):
        audit.audit_reflow(path)


def test_changed_reflow(tokens: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "reflow.json"
    path.write_text(json.dumps({"inputs": {str(tokens): file_digest(tokens)}}), encoding="utf-8")
    digests = iter(["first", file_digest(tokens), file_digest(tokens), "second"])
    monkeypatch.setattr(audit, "file_digest", lambda path: next(digests))
    with pytest.raises(ValueError, match="input changed"):
        audit.audit_reflow(path)


def test_invalid_cli_combination(tokens: Path) -> None:
    with pytest.raises(SystemExit):
        audit.main([str(tokens), "--reflow", "--include-units"])
