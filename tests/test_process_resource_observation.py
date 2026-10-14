"""Synthetic snapshots; no real process operations, signals, media or models."""

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from replay import process_resource_observation as observe
from replay.core import ReplayError
from replay.process_family_observation import SystemHost


@pytest.fixture
def request_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(observe, "REPOSITORY", tmp_path)
    work = tmp_path / "edits/work"
    work.mkdir(parents=True)
    path = work / "request.json"
    path.write_text(
        '{"version":1,"kind":"process-resource-observation-request",'
        '"sampleCount":3,"intervalSeconds":0,"families":[{"pid":10,"birth":"123:456"}]}',
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", True),
        ("kind", "unknown"),
        ("sampleCount", 2),
        ("intervalSeconds", 2.01),
        ("intervalSeconds", -1),
        ("intervalSeconds", True),
        ("intervalSeconds", "1"),
        ("intervalSeconds", float("nan")),
        ("families", []),
        ("families", [{"pid": True, "birth": "123:456"}]),
        ("families", [{"pid": 0, "birth": "123:456"}]),
        ("families", [{"pid": 10, "birth": "bad"}]),
        ("families", [{"pid": 10, "birth": "123:456", "extra": 1}]),
        ("families", [{"pid": 10, "birth": "123:456"}] * 2),
        ("extra", 1),
    ],
)
def test_external_schema(request_path: Path, field: str, value: object) -> None:
    raw = json.loads(request_path.read_bytes())
    raw[field] = value
    request_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayError):
        observe.load(request_path)


@pytest.mark.parametrize("mode", ["duplicate", "oversize", "relative", "outside", "linked"])
def test_bad_input_paths(request_path: Path, mode: str) -> None:
    path = request_path
    if mode == "duplicate":
        path.write_text('{"version":1,"version":1}', encoding="utf-8")
    if mode == "oversize":
        path.write_bytes(b"x" * 65537)
    if mode == "relative":
        path = Path("request.json")
    if mode == "outside":
        path = request_path.parents[2] / "outside.json"
    if mode == "linked":
        path = request_path.with_name("link.json")
        path.symlink_to(request_path)
    with pytest.raises(ReplayError):
        observe.load(path)


@pytest.mark.parametrize("mode", ["ok", "membership", "reuse", "exit", "missing", "badbytes"])
def test_stable_physical_and_rss(
    request_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    host = MagicMock(spec=SystemHost)
    host.birth.return_value = "changed" if mode == "reuse" else "123:456"
    sample = {
        "observedAtNs": 123,
        "members": [{"pid": 10, "birth": "123:456"}],
        "exitedDuringObservation": ["bad"] if mode == "exit" else [],
        "physicalBytes": True if mode == "badbytes" else 200,
        "rssBytes": 500,
    }
    changed = dict(sample)
    changed["members"] = [{"pid": 11, "birth": "123:456"}]
    sampler = MagicMock(side_effect=ReplayError("missing") if mode == "missing" else None)
    sampler.side_effect = (
        ReplayError("missing")
        if mode == "missing"
        else [sample, changed if mode == "membership" else sample, sample]
    )
    monkeypatch.setattr(observe, "sample_family", sampler)
    monkeypatch.setattr(time, "sleep", MagicMock())
    if mode != "ok":
        with pytest.raises(ReplayError):
            observe.observe(observe.load(request_path), host)
        return
    result = observe.observe(observe.load(request_path), host)
    assert result[0]["physicalBytes"] == {"maxBytes": 200, "meanBytes": 200.0, "variationBytes": 0}
    assert result[0]["rssBytes"] == {"maxBytes": 500, "meanBytes": 500.0, "variationBytes": 0}
    assert result[0]["observerIncluded"] is False
    assert sampler.call_count == 3


@pytest.mark.parametrize("mode", ["empty", "duplicate", "overlap", "self"])
def test_members_and_self(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    host = MagicMock(spec=SystemHost)
    host.birth.return_value = "123:456"
    monkeypatch.setattr(os, "getpid", lambda: 10)
    members = [] if mode == "empty" else [{"pid": 10, "birth": "123:456"}]
    if mode == "duplicate":
        members += [{"pid": 10, "birth": "123:456"}]
    monkeypatch.setattr(
        observe,
        "sample_family",
        MagicMock(
            return_value={
                "members": members,
                "exitedDuringObservation": [],
                "physicalBytes": 1,
                "rssBytes": 2,
            }
        ),
    )
    families: tuple[observe.Family, ...] = (observe.Family(10, "123:456"),)
    if mode == "overlap":
        families += (observe.Family(11, "123:456"),)
    if mode != "self":
        with pytest.raises(ReplayError):
            observe.observe(observe.Request(families, 0), host)
        return
    assert observe.observe(observe.Request(families, 0), host)[0]["observerIncluded"] is True


@pytest.mark.parametrize("mode", ["ok", "occupied", "source", "request"])
def test_run_and_cli(request_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    output = request_path.with_name("report.json")
    host = MagicMock(spec=SystemHost)
    host.birth.return_value = "123:456"
    monkeypatch.setattr(observe, "SystemHost", MagicMock(return_value=host))
    monkeypatch.setattr(observe, "observe", MagicMock(return_value=[]))
    monkeypatch.setattr(observe, "SOURCE_PATHS", (request_path,))
    if mode == "occupied":
        output.write_text("preserved", encoding="utf-8")
    if mode in {"source", "request"}:
        monkeypatch.setattr(
            observe,
            "pin",
            MagicMock(
                side_effect=[{}, {}, {"changed": 1} if mode == "request" else {}, {"changed": 1}]
            ),
        )
    result = observe.main([str(request_path), str(output)])
    assert result == (0 if mode == "ok" else 1)
    if mode == "ok":
        report = json.loads(output.read_bytes())
        assert report["kind"] == "process-resource-observation"
        assert report["peakGuarantee"] is False
        assert report["sampleCount"] == 3
        assert output.stat().st_mode & 0o777 == 0o600
    if mode == "occupied":
        assert output.read_text(encoding="utf-8") == "preserved"


def test_request_changes_before_sampling(
    request_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = request_path.with_name("changed-request-report.json")
    sampler = MagicMock()
    monkeypatch.setattr(observe, "observe", sampler)
    monkeypatch.setattr(
        observe,
        "load",
        MagicMock(
            side_effect=[
                observe.Request((observe.Family(10, "123:456"),), 0),
                observe.Request((observe.Family(10, "123:456"),), 1),
            ]
        ),
    )
    with pytest.raises(ReplayError, match="Request changed before observation"):
        observe.run(request_path, output)
    assert sampler.call_count == 0
    assert output.exists() is False
