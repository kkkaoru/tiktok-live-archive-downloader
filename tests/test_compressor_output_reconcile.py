import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import compressor_batch
from replay import compressor_output_reconcile as reconcile
from replay.core import ReplayError
from replay.recordings import record


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path)}


def _save(path: Path, data: dict[str, object]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@dataclass
class Fixture:
    work: Path
    request: Path
    receipt: Path
    source: Path
    preset: Path
    movie: Path
    legacy: Path
    owner: Path
    manifest: Path
    plan: Path
    plan_manifest: Path
    response: dict[str, object]
    calls: list[list[str]]
    mutate: Callable[[], object] | None = None
    writer: str = ""
    decode_failure: bool = False

    def request_update(self, key: str, value: object) -> None:
        row = record(json.loads(self.request.read_text()))
        row[key] = value
        _save(self.request, row)

    def refresh(self, key: str, path: Path) -> None:
        self.request_update(key, _ref(path))


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, alternate: bool = False
) -> Fixture:
    monkeypatch.setattr(reconcile, "ROOT", tmp_path)
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    work = tmp_path / "edits/work/synthetic"
    work.mkdir(parents=True)
    code = tmp_path / "src/replay"
    code.mkdir(parents=True)
    (code / "reader.py").write_text("# synthetic transitive reader\n")
    for name in (*reconcile.CODE_SEEDS, "compressor_output_reconcile.py"):
        (code / name).write_text("# synthetic bounded reader\n")
    closure = [code / name for name in (*reconcile.CODE_SEEDS, "reader.py")]
    monkeypatch.setattr(reconcile, "reader_code_paths", lambda: tuple(closure))
    frames, rate = (7, 24) if alternate else (3, 30)
    width, height = (1920, 1080) if alternate else (1080, 1920)
    source = work / "source.moti"
    source.write_text(
        "<ozml><scene><sceneSettings>"
        f"<width>{width}</width><height>{height}</height><duration>{frames}</duration>"
        f"<frameRate>{rate}</frameRate><NTSC>0</NTSC></sceneSettings></scene>"
        '<timeRange duration="76876800 7680000 1 0"/>'
        '<playRange duration="76876800 7680000 1 0"/></ozml>'
    )
    preset, movie, dependency = work / "preset.setting", work / "alpha.mov", work / "resource.png"
    preset.write_bytes(b"synthetic preset")
    movie.write_bytes(b"synthetic encoded output, mocked tools only")
    dependency.write_bytes(b"synthetic declared dependency")
    manifest = _save(
        work / "jobs.json",
        {
            "jobs": [
                {
                    "name": "alpha-a",
                    "source": str(source),
                    "preset": str(preset),
                    "output": str(movie),
                    "dependencies": [str(dependency)],
                }
            ]
        },
    )
    legacy = _save(work / "legacy.json", {"status": "submitted", "jobID": "legacy-job-a"})
    plan = _save(
        work / "plan.json", {"frames": frames, "rate": rate, "width": width, "height": height}
    )
    plan_manifest = _save(
        work / "plans.json",
        {
            "version": 1,
            "rate": rate,
            "parts": [{**_ref(plan), "frames": frames, "offsetFrame": 12}],
        },
    )
    status = work / "wrapper.status"
    status.write_text("1\n")
    owner = _save(
        work / "owner.json",
        {
            "version": 1,
            "knownOwnersTerminal": True,
            "activeOwners": [],
            "observedAtEpochSeconds": 1000,
            "terminalRecords": [_ref(status)],
        },
    )
    probe, decoder = work / "ffprobe", work / "ffmpeg"
    for tool in (probe, decoder):
        tool.write_bytes(b"synthetic executable, never actually invoked")
        tool.chmod(0o700)
    request = _save(
        work / "request.json",
        {
            "version": 1,
            "jobName": "alpha-a",
            "jobID": "legacy-job-a",
            "manifest": _ref(manifest),
            "legacyReceipt": _ref(legacy),
            "plan": _ref(plan),
            "planManifest": _ref(plan_manifest),
            "output": _ref(movie),
            "inputs": [_ref(source), _ref(preset), _ref(dependency)],
            "ffprobe": _ref(probe),
            "ffmpeg": _ref(decoder),
            "ownerEvidence": _ref(owner),
            "expectedEncoding": {
                "codec": "prores",
                "profile": "4444",
                "pixelFormats": ["yuva444p12le"],
            },
        },
    )
    response: dict[str, object] = {
        "streams": [
            {
                "codec_name": "prores",
                "profile": "4444",
                "pix_fmt": "yuva444p12le",
                "width": width,
                "height": height,
                "start_pts": 0,
                "avg_frame_rate": f"{rate}/1",
                "duration_ts": frames * 100,
                "time_base": f"1/{rate * 100}",
                "nb_read_frames": str(frames),
            }
        ],
        "frames": [{"best_effort_timestamp": index * 100} for index in range(frames)],
    }
    item = Fixture(
        work,
        request,
        work / "disposition.json",
        source,
        preset,
        movie,
        legacy,
        owner,
        manifest,
        plan,
        plan_manifest,
        response,
        [],
    )

    def run(command: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        assert timeout in {30, 1800}
        item.calls.append(command)
        if command[0] == "/usr/sbin/lsof":
            if item.writer:
                return subprocess.CompletedProcess(command, 0, item.writer, "")
            return subprocess.CompletedProcess(command, 1, "", "")
        if command[0] == str(probe):
            return subprocess.CompletedProcess(command, 0, json.dumps(item.response), "")
        assert command[0] == str(decoder)
        if item.mutate is not None:
            item.mutate()
        return subprocess.CompletedProcess(
            command, int(item.decode_failure), "", "decode failed" if item.decode_failure else ""
        )

    monkeypatch.setattr(reconcile, "_run", run)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Independent verification must not operate Compressor jobs")

    monkeypatch.setattr(compressor_batch.Batch, "finish", forbidden)
    monkeypatch.setattr(compressor_batch.Batch, "submit", forbidden)
    return item


@pytest.mark.parametrize("alternate", [False, True])
def test_success_preserves_history_and_readonly_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alternate: bool
) -> None:
    f = _fixture(tmp_path, monkeypatch, alternate=alternate)
    before = f.legacy.read_bytes()
    assert reconcile.reconcile(f.request, f.receipt) == f.receipt
    document = record(json.loads(f.receipt.read_text()))
    assert document["officialTerminalVerified"] is False
    assert document["historicalInputsVerified"] is False
    assert document["terminalVerified"] is False
    assert document["candidateOnly"] is True
    assert document["qualityApproved"] is False
    assert document["currentOutputVerified"] is True
    assert document["currentDeclaredInputsVerified"] is True
    assert document["motionRangesUsedForExpectedClock"] is False
    assert document["motionRanges"] == [
        {
            "element": name,
            "location": name,
            "duration": "76876800 7680000 1 0",
            "observedDuration": "1001/100",
            "clockComparison": "conflicts-with-project-duration",
        }
        for name in ("timeRange", "playRange")
    ]
    receipt_before = f.receipt.read_bytes()
    with pytest.raises(ReplayError, match="Occupied"):
        reconcile.reconcile(f.request, f.receipt)
    f.calls.clear()
    reconcile.reconcile(f.request, f.receipt, resume=True)
    assert len(f.calls) == 4
    assert f.receipt.read_bytes() == receipt_before
    assert f.legacy.read_bytes() == before
    assert any("-count_frames" in call and "-show_frames" in call for call in f.calls)
    assert any("0:a?" in call and "-xerror" in call for call in f.calls)


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", 99),
        ("height", 99),
        ("codec_name", "h264"),
        ("profile", "422"),
        ("pix_fmt", "yuv444p12le"),
        ("nb_read_frames", "2"),
        ("start_pts", 100),
        ("avg_frame_rate", "29/1"),
        ("duration_ts", 299),
        ("time_base", "0"),
        ("time_base", "invalid"),
        ("duration_ts", "300"),
    ],
)
def test_wrong_output_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    f = _fixture(tmp_path, monkeypatch)
    streams = f.response["streams"]
    assert isinstance(streams, list)
    record(streams[0])[field] = value
    with pytest.raises(ReplayError):
        reconcile.reconcile(f.request, f.receipt)
    assert not f.receipt.exists()
    assert record(json.loads(f.legacy.read_text()))["status"] == "submitted"


@pytest.mark.parametrize(
    "field,value",
    [
        ("frames", []),
        ("streams", []),
        ("frames", [{"best_effort_timestamp": 7}] * 3),
        ("frames", [{"best_effort_timestamp": "0"}] * 3),
    ],
)
def test_incomplete_pts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    f = _fixture(tmp_path, monkeypatch)
    f.response[field] = value
    with pytest.raises(ReplayError):
        reconcile.reconcile(f.request, f.receipt)


@pytest.mark.parametrize("role", ["source", "preset", "movie"])
def test_mutation_during_decode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str) -> None:
    f = _fixture(tmp_path, monkeypatch)
    path = getattr(f, role)
    assert isinstance(path, Path)
    f.mutate = lambda: path.write_bytes(b"changed current bytes")
    with pytest.raises(ReplayError, match="changed during"):
        reconcile.reconcile(f.request, f.receipt)
    assert not f.receipt.exists()


def test_changed_transitive_reader_and_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = _fixture(tmp_path, monkeypatch)
    code = tmp_path / "src/replay/reader.py"
    f.mutate = lambda: code.write_text("# changed transitive reader\n")
    with pytest.raises(ReplayError, match="changed during"):
        reconcile.reconcile(f.request, f.receipt)
    new_reader = code.parent / "new_reader.py"
    closure = [code.parent / name for name in (*reconcile.CODE_SEEDS, "reader.py")]
    monkeypatch.setattr(reconcile, "reader_code_paths", lambda: tuple(closure))

    def change_closure() -> None:
        new_reader.write_text("# added reachable reader\n")
        closure.append(new_reader)

    f.mutate = change_closure
    with pytest.raises(ReplayError, match="Reader inventory"):
        reconcile.reconcile(f.request, f.receipt)


def test_unrelated_reader_does_not_freeze_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = _fixture(tmp_path, monkeypatch)
    unrelated = tmp_path / "src/replay/unrelated.py"
    f.mutate = lambda: unrelated.write_text("# unrelated module was added\n")
    reconcile.reconcile(f.request, f.receipt)
    receipt = record(json.loads(f.receipt.read_text()))
    assert str(unrelated) not in record(receipt["inputs"])


def test_missing_shared_seed_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        reconcile, "reader_code_paths", lambda: (tmp_path / "src/replay/reader.py",)
    )
    with pytest.raises(ReplayError, match="code seed"):
        reconcile.reconcile(f.request, f.receipt)
    assert not f.calls


@pytest.mark.parametrize("changed", ["stale", "future", "active", "not-terminal", "status"])
def test_owner_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str) -> None:
    f = _fixture(tmp_path, monkeypatch)
    owner = record(json.loads(f.owner.read_text()))
    if changed == "stale":
        owner["observedAtEpochSeconds"] = 800
    elif changed == "future":
        owner["observedAtEpochSeconds"] = 1001
    elif changed == "active":
        owner["activeOwners"] = [123]
    elif changed == "not-terminal":
        owner["knownOwnersTerminal"] = False
    else:
        status = f.work / "wrapper.status"
        status.write_text("running")
        owner["terminalRecords"] = [_ref(status)]
    _save(f.owner, owner)
    f.refresh("ownerEvidence", f.owner)
    with pytest.raises(ReplayError):
        reconcile.reconcile(f.request, f.receipt)
    assert not f.calls


@pytest.mark.parametrize("writer", ["p123\nf5\naw\n", "p123\nf5\nau\n", "p123\nf5\n"])
def test_writers_and_uncertain_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, writer: str
) -> None:
    f = _fixture(tmp_path, monkeypatch)
    f.writer = writer
    with pytest.raises(ReplayError, match="writer|uncertain"):
        reconcile.reconcile(f.request, f.receipt)
    assert len(f.calls) == 1


def test_readers_allowed_decode_failure_and_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = _fixture(tmp_path, monkeypatch)
    f.writer = "p123\nf5\nar\n"
    f.decode_failure = True
    with pytest.raises(ReplayError, match="full AV"):
        reconcile.reconcile(f.request, f.receipt)
    assert not f.receipt.exists()
    f.movie.unlink()
    with pytest.raises(ReplayError, match="output"):
        reconcile.reconcile(f.request, f.receipt)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_linked_output_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    f = _fixture(tmp_path, monkeypatch)
    target = f.work / "other.mov"
    if kind == "symlink":
        f.movie.rename(target)
        f.movie.symlink_to(target)
    else:
        os.link(f.movie, target)
    with pytest.raises(ReplayError, match="link"):
        reconcile.reconcile(f.request, f.receipt)
    assert not f.calls


@pytest.mark.parametrize(
    "key,value",
    [
        ("jobID", "wrong"),
        ("jobName", "unknown"),
        ("version", True),
        ("inputs", []),
        ("expectedEncoding", {"codec": "h264"}),
    ],
)
def test_wrong_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str, value: object
) -> None:
    f = _fixture(tmp_path, monkeypatch)
    f.request_update(key, value)
    with pytest.raises(ReplayError):
        reconcile.reconcile(f.request, f.receipt)
    assert not f.calls


def test_missing_pin_wrong_hash_and_uncertain_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = _fixture(tmp_path, monkeypatch)
    f.request_update("output", {"path": str(f.movie), "sha256": "0" * 64})
    with pytest.raises(ReplayError, match="SHA256"):
        reconcile.reconcile(f.request, f.receipt)
    f.refresh("output", f.movie)
    _save(f.receipt, {"currentOutputVerified": True})
    with pytest.raises(ReplayError, match="uncertain"):
        reconcile.reconcile(f.request, f.receipt, resume=True)
    assert not f.calls


def test_plan_and_motion_disagreement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = _fixture(tmp_path, monkeypatch)
    f.source.write_text(f.source.read_text().replace("<duration>3", "<duration>4"))
    row = record(json.loads(f.request.read_text()))
    values = row["inputs"]
    assert isinstance(values, list)
    row["inputs"] = [_ref(f.source), *values[1:]]
    _save(f.request, row)
    with pytest.raises(ReplayError, match="Motion duration"):
        reconcile.reconcile(f.request, f.receipt)
    f.refresh("plan", f.plan)
    catalog = record(json.loads(f.plan_manifest.read_text()))
    catalog["parts"] = []
    _save(f.plan_manifest, catalog)
    f.refresh("planManifest", f.plan_manifest)
    with pytest.raises(ReplayError, match="bounded validated"):
        reconcile.reconcile(f.request, f.receipt)


@pytest.mark.parametrize("hierarchy", ["scene", "both", "absent", "object"])
def test_project_range_hierarchies_do_not_change_expected_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hierarchy: str
) -> None:
    f = _fixture(tmp_path, monkeypatch)
    root = ET.fromstring(f.source.read_text())
    scene = root.find("scene")
    assert scene is not None
    for name in ("timeRange", "playRange"):
        node = root.find(name)
        assert node is not None
        if hierarchy == "scene":
            root.remove(node)
            scene.append(node)
        elif hierarchy == "both":
            scene.append(ET.fromstring(ET.tostring(node, encoding="unicode")))
        elif hierarchy == "absent":
            root.remove(node)
        else:
            root.remove(node)
            objects = ET.SubElement(scene, "objects")
            ET.SubElement(objects, "object").append(node)
    f.source.write_text(ET.tostring(root, encoding="unicode"))
    row = record(json.loads(f.request.read_text()))
    values = row["inputs"]
    assert isinstance(values, list)
    row["inputs"] = [_ref(f.source), *values[1:]]
    _save(f.request, row)
    reconcile.reconcile(f.request, f.receipt)
    result = record(json.loads(f.receipt.read_text()))
    assert record(result["expected"])["frames"] == 3
    assert record(result["expected"])["duration"] == "1/10"
    assert result["motionRangesUsedForExpectedClock"] is False
    ranges = result["motionRanges"]
    assert isinstance(ranges, list)
    expected_count = 4 if hierarchy == "both" else 2 if hierarchy == "scene" else 0
    assert len(ranges) == expected_count
    if hierarchy == "scene":
        assert {record(item)["location"] for item in ranges} == {
            "scene/timeRange",
            "scene/playRange",
        }
    assert all(
        record(item)["clockComparison"] == "conflicts-with-project-duration" for item in ranges
    )
    assert result["officialTerminalVerified"] is False
    assert result["historicalInputsVerified"] is False


@pytest.mark.parametrize("value", ["relative", "/outside/private/file", "", "/a\nfile"])
def test_paths_refuse_escape(value: str) -> None:
    with pytest.raises(ReplayError):
        reconcile._path(value)


def test_parser_budgets_and_duplicate_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ReplayError):
        reconcile._ref({"path": str(f.movie), "sha256": "bad"})
    with pytest.raises(ReplayError, match="unique"):
        reconcile._unique([("x", 1), ("x", 2)])
    with pytest.raises(ReplayError):
        reconcile._positive(True, 5)
    with pytest.raises(ReplayError):
        reconcile._fraction(3)
    monkeypatch.setattr(reconcile, "MAX_BYTES", 4)
    with pytest.raises(ReplayError, match="bounded"):
        reconcile._bytes(reconcile.Reference(f.source, _sha(f.source)))


def test_main_and_real_tool_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["reconcile", str(f.request), str(f.receipt)])
    reconcile.main()
    assert f.receipt.exists()
    monkeypatch.undo()
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, "ok", "")
    )
    assert reconcile._run(["synthetic"], timeout=30).stdout == "ok"
    monkeypatch.setattr(reconcile, "MAX_BYTES", 1)
    with pytest.raises(ReplayError, match="budget"):
        reconcile._run(["synthetic"], timeout=30)
