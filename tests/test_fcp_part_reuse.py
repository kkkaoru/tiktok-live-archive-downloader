import json
import os
import subprocess
import sys
import wave
from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from replay import fcp_part_assembly as assembly
from replay import fcp_part_reuse as reuse
from replay import part_master_publication as publication
from replay.core import ReplayError
from replay.fcp_partition import write_parts
from replay.stages import file_digest


@pytest.fixture
def plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    ffprobe = tmp_path / "ffprobe"
    ffprobe.write_bytes(b"synthetic pinned executable")
    ffprobe.chmod(0o700)
    source = tmp_path / "before.fcpxml"
    source.write_text(
        '<fcpxml version="1.11"><resources><format id="f" frameDuration="1/30s" '
        'width="1080" height="1920"/></resources><library><event><project name="Before">'
        '<sequence format="f" duration="1/15s"><spine>'
        '<asset-clip offset="0s" start="100s" duration="1/30s">'
        '<title offset="100s" start="3600s" duration="1/30s"><text>first</text>'
        '<adjust-transform position="0 1"/></title></asset-clip>'
        '<asset-clip offset="1/30s" start="200s" duration="1/30s">'
        '<title offset="200s" start="3600s" duration="1/30s"><text>second</text>'
        '<adjust-transform position="0 1"/></title></asset-clip>'
        "</spine></sequence></project></event></library></fcpxml>",
        encoding="utf-8",
    )
    old_manifest = write_parts(
        source, tmp_path / "old-parts", target_duration=Fraction(1, 30), maximum_titles=1
    )
    voice = tmp_path / "voice.wav"
    with wave.open(str(voice), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(48000)
        handle.writeframes(b"\0\0" * 3200)
    staging, ssd = tmp_path / "staging", tmp_path / "ssd"
    staging.mkdir()
    ssd.mkdir()
    (staging / "part-0001-master.mov").write_bytes(b"verified synthetic native bytes")
    old_staging = tmp_path / "old-staging.json"
    assembly.prepare(old_manifest, staging, voice, old_staging)
    observation = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1080,
                "height": 1920,
                "start_pts": 0,
                "avg_frame_rate": "30/1",
                "duration_ts": 1,
                "time_base": "1/30",
                "nb_read_frames": "1",
            }
        ],
        "frames": [{"best_effort_timestamp": 0}],
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, json.dumps(observation)),
    )
    monkeypatch.setattr(
        os,
        "statvfs",
        lambda path: os.statvfs_result(
            (4096, 4096, 100_000_000, 90_000_000, 90_000_000, 1, 1, 1, 0, 255)
        ),
    )
    old_proof = assembly.probe(
        old_staging,
        part=1,
        tool=assembly.FrameProbeTool(ffprobe, file_digest(ffprobe), decoder_threads=0),
    )
    completion = tmp_path / "completion.json"
    completion.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "verified-native-part",
                "part": 1,
                "completionVerified": True,
                "shareObservedIdle": True,
                "fullAVDecodeExitCode": 0,
                "movie": assembly._reference(staging / "part-0001-master.mov"),
                "frameProof": assembly._reference(old_proof),
                "inputs": {str(old_staging): assembly._reference(old_staging)["sha256"]},
            }
        ),
        encoding="utf-8",
    )
    publish_config = tmp_path / "publish.json"
    publish_config.write_text(
        json.dumps(
            {
                "version": 1,
                "completion": assembly._reference(completion),
                "destination": str(ssd / "part-0001-master.mov"),
                "receiptOutput": str(tmp_path / "published.json"),
                "frameProofOutput": str(tmp_path / "published-proof.json"),
            }
        ),
        encoding="utf-8",
    )
    published = publication.publish(publish_config)
    old_config = tmp_path / "old-assembly.json"
    raw = json.loads(old_staging.read_text(encoding="utf-8"))
    raw["parts"][0] = {
        "movie": str(ssd / "part-0001-master.mov"),
        "proof": str(tmp_path / "published-proof.json"),
    }
    old_config.write_text(json.dumps(raw), encoding="utf-8")
    retirement_config = tmp_path / "retirement-config.json"
    retirement_config.write_text(
        json.dumps(
            {
                "version": 1,
                "retireStaging": True,
                "publication": assembly._reference(published),
                "assembly": assembly._reference(old_config),
                "receiptOutput": str(tmp_path / "retirement.json"),
            }
        ),
        encoding="utf-8",
    )
    retired = publication.prepare_retirement(retirement_config)
    after = tmp_path / "after.fcpxml"
    root = ET.parse(source).getroot()
    project = root.find("./library/event/project")
    transform = root.findall(".//adjust-transform")[1]
    assert project is not None
    project.set("name", "After")
    transform.set("position", "0 2")
    ET.ElementTree(root).write(after)
    new_manifest = write_parts(
        after, tmp_path / "new-parts", target_duration=Fraction(1, 30), maximum_titles=1
    )
    new_config = tmp_path / "new-assembly.json"
    assembly.prepare(new_manifest, ssd, voice, new_config)
    new = json.loads(new_config.read_text(encoding="utf-8"))
    new["parts"][0]["proof"] = str(tmp_path / "reuse-proof.json")
    new_config.write_text(json.dumps(new), encoding="utf-8")
    config = tmp_path / "reuse.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "part": 1,
                "priorAssembly": assembly._reference(old_config),
                "targetAssembly": assembly._reference(new_config),
                "publication": assembly._reference(published),
                "retirement": assembly._reference(retired),
            }
        ),
        encoding="utf-8",
    )
    return config


def test_reuse_retired_staging_and_fresh_other_part_keep_historical_proofs(plan: Path) -> None:
    before = (plan.parent / "published-proof.json").read_bytes()
    old_manifest = (plan.parent / "old-parts/manifest.json").read_bytes()
    (plan.parent / "staging/part-0001-master.mov").unlink()
    proof = reuse.prepare(plan)
    target = plan.parent / "new-assembly.json"
    assert assembly.verify(target, part=1) == proof
    raw = json.loads(proof.read_text(encoding="utf-8"))
    assert raw["crossRevisionReuse"]["qualityApproved"] is False
    assert raw["crossRevisionReuse"]["partXMLIdentical"] is True
    assert raw["audioIgnored"] is True
    (plan.parent / "ssd/part-0002-master.mov").write_bytes(b"fresh changed native bytes")
    ffprobe = plan.parent / "ffprobe"
    assembly.probe(
        target,
        part=2,
        tool=assembly.FrameProbeTool(ffprobe, file_digest(ffprobe), decoder_threads=0),
    )
    output = plan.parent / "whole.fcpxml"
    assembly.assemble(target, output, name="whole")
    assembly.verify_assembly(target, output)
    assert (plan.parent / "published-proof.json").read_bytes() == before
    assert (plan.parent / "old-parts/manifest.json").read_bytes() == old_manifest
    assert (
        plan.parent / "ssd/part-0001-master.mov"
    ).read_bytes() == b"verified synthetic native bytes"


def test_missing_staging_requires_explicit_retirement(plan: Path) -> None:
    raw = json.loads(plan.read_text(encoding="utf-8"))
    raw.pop("retirement")
    plan.write_text(json.dumps(raw), encoding="utf-8")
    reuse.prepare(plan)
    (plan.parent / "staging/part-0001-master.mov").unlink()
    with pytest.raises((ReplayError, FileNotFoundError)):
        assembly.verify(plan.parent / "new-assembly.json", part=1)


@pytest.mark.parametrize("field", ["version", "part", "priorAssembly", "targetAssembly"])
def test_invalid_configuration_rejected(plan: Path, field: str) -> None:
    raw = json.loads(plan.read_text(encoding="utf-8"))
    raw[field] = 0
    plan.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        reuse.prepare(plan)


@pytest.mark.parametrize("field", ["movie", "partition", "part", "observed", "crossRevisionReuse"])
def test_reuse_proof_cannot_be_forged(plan: Path, field: str) -> None:
    proof = reuse.prepare(plan)
    raw = json.loads(proof.read_text(encoding="utf-8"))
    raw[field] = None
    proof.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((ReplayError, ValueError)):
        assembly.verify(plan.parent / "new-assembly.json", part=1)


@pytest.mark.parametrize("change", ["y", "text", "scale", "template", "source-clock"])
def test_changed_part_cannot_be_reused(plan: Path, change: str) -> None:
    source = plan.parent / "after.fcpxml"
    root = ET.parse(source).getroot()
    transform = root.findall(".//adjust-transform")[0]
    if change == "y":
        transform.set("position", "0 3")
    elif change == "scale":
        transform.set("scale", "0.5 0.5")
    elif change == "text":
        root.findall(".//text")[0].text = "different text"
    elif change == "template":
        root.findall(".//title")[0].set("ref", "changed-template")
    else:
        root.findall(".//asset-clip")[0].set("start", "101s")
        root.findall(".//title")[0].set("offset", "101s")
    revised = plan.parent / "changed.fcpxml"
    ET.ElementTree(root).write(revised)
    manifest = write_parts(
        revised, plan.parent / "changed-parts", target_duration=Fraction(1, 30), maximum_titles=1
    )
    target = json.loads((plan.parent / "new-assembly.json").read_text(encoding="utf-8"))
    target["partition"] = assembly._reference(manifest)
    target_path = plan.parent / "changed-assembly.json"
    target_path.write_text(json.dumps(target), encoding="utf-8")
    config = json.loads(plan.read_text(encoding="utf-8"))
    config["targetAssembly"] = assembly._reference(target_path)
    plan.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ReplayError, match="Reuse"):
        reuse.prepare(plan)


@pytest.mark.parametrize("change", ["static-x", "scientific-x", "animated-x", "resampled-y"])
def test_unselected_part_keeps_x_and_allows_only_y_sampling(plan: Path, change: str) -> None:
    root = ET.parse(plan.parent / "after.fcpxml").getroot()
    transform = root.findall(".//adjust-transform")[1]
    if change in {"static-x", "scientific-x"}:
        transform.set("position", "1e-12 2" if change == "scientific-x" else "1 2")
    else:
        transform.attrib.pop("position")
        param = ET.SubElement(transform, "param", name="position")
        animation = ET.SubElement(param, "keyframeAnimation")
        for time, y in (("3600s", 2), ("216001/60s", 3), ("108001/30s", 4)):
            ET.SubElement(
                animation,
                "keyframe",
                time=time,
                value=f"{1 if change == 'animated-x' else 0} {y}",
                curve="linear",
            )
    source = plan.parent / "x-audit.fcpxml"
    ET.ElementTree(root).write(source)
    manifest = write_parts(
        source, plan.parent / "x-audit-parts", target_duration=Fraction(1, 30), maximum_titles=1
    )
    target = json.loads((plan.parent / "new-assembly.json").read_text(encoding="utf-8"))
    target["partition"] = assembly._reference(manifest)
    target_path = plan.parent / "x-audit-assembly.json"
    target_path.write_text(json.dumps(target), encoding="utf-8")
    config = json.loads(plan.read_text(encoding="utf-8"))
    config["targetAssembly"] = assembly._reference(target_path)
    plan.write_text(json.dumps(config), encoding="utf-8")
    if change == "resampled-y":
        proof = reuse.prepare(plan)
        assert assembly.verify(target_path, part=1) == proof
    else:
        with pytest.raises(ReplayError, match="non-Y"):
            reuse.prepare(plan)


def test_reuse_cli_and_existing_output(plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["reuse", str(plan)])
    reuse.main()
    with pytest.raises(ReplayError, match="absent"):
        reuse.prepare(plan)


def test_producer_change_invalidates_reuse(plan: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reuse.prepare(plan)
    original = file_digest

    def changed(path: Path) -> str:
        return "0" * 64 if path.name == "fcp_part_reuse.py" else original(path)

    monkeypatch.setattr(reuse, "file_digest", changed)
    with pytest.raises(ReplayError, match="producer inputs changed"):
        assembly.verify(plan.parent / "new-assembly.json", part=1)


def test_changed_target_movie_rejected(plan: Path) -> None:
    target_path = plan.parent / "new-assembly.json"
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target["parts"][0]["movie"] = str(plan.parent / "different.mov")
    target_path.write_text(json.dumps(target), encoding="utf-8")
    config = json.loads(plan.read_text(encoding="utf-8"))
    config["targetAssembly"] = assembly._reference(target_path)
    plan.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ReplayError, match="target movie"):
        reuse.prepare(plan)


def test_changed_partition_boundary_rejected(plan: Path) -> None:
    manifest = write_parts(
        plan.parent / "after.fcpxml",
        plan.parent / "combined-parts",
        target_duration=Fraction(1, 15),
        maximum_titles=2,
    )
    target_path = plan.parent / "combined-assembly.json"
    assembly.prepare(manifest, plan.parent / "ssd", plan.parent / "voice.wav", target_path)
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target["parts"][0]["proof"] = str(plan.parent / "combined-reuse-proof.json")
    target_path.write_text(json.dumps(target), encoding="utf-8")
    config = json.loads(plan.read_text(encoding="utf-8"))
    config["targetAssembly"] = assembly._reference(target_path)
    plan.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ReplayError, match="ordered parts"):
        reuse.prepare(plan)
