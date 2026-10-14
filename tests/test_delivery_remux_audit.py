"""Synthetic compressed-stream identity and retained failure evidence."""

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

from replay import delivery_remux_audit as audit
from replay import delivery_verification as verifier
from replay.core import ReplayError
from replay.delivery_render import CONFIG_KEYS
from replay.recordings import record


@pytest.fixture
def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    storage = tmp_path / "storage"
    (storage / "final").mkdir(parents=True)
    source = storage / "final" / "2024-01-02_03-04-05_JST_replay-123-review.m4v"
    source.write_bytes(b"source")
    (tmp_path / "final-job-state").mkdir()
    (tmp_path / "final-job-state" / "final.json").write_text(
        json.dumps(
            {
                "status": "Successful",
                "jobID": "native-1",
                "output": str(source),
                "bytes": 6,
            }
        )
    )
    (tmp_path / "final-job.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "name": "final",
                        "source": str(storage / "av-master.mov"),
                        "preset": str(tmp_path / "unused"),
                        "output": str(source),
                    }
                ]
            }
        )
    )
    raw: dict[str, object] = dict.fromkeys(CONFIG_KEYS, str(tmp_path / "unused"))
    raw.update(
        version=1,
        recording="replay-123",
        source=str(tmp_path / "2024-01-02_03-04-05_JST_replay-123.mp4"),
        rate=30,
        width=64,
        height=64,
        work=str(tmp_path),
        storage=str(storage),
        deliveryRoot=str(tmp_path),
        encoding={"codec": "hevc"},
        motion={"template": str(tmp_path / "template"), "expectedText": "Title", "finish": []},
        audio={"source": str(tmp_path / "voice"), "clock": str(tmp_path / "clock")},
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw))
    (tmp_path / "unused").write_text(
        '<setting><encoder name="H264iPod"><file-extension>m4v</file-extension>'
        "</encoder><filter-set/></setting>",
        encoding="utf-8",
    )
    movie = tmp_path / "candidate.mp4"
    movie.write_bytes(b"candidate")
    return config, source, movie


@pytest.fixture
def probe_data() -> dict[str, object]:
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "hevc",
                "codec_tag_string": "hvc1",
                "time_base": "1/30",
                "extradata_hash": "SHA256:" + "a" * 64,
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "codec_tag_string": "mp4a",
                "time_base": "1/48000",
                "extradata_hash": "SHA256:" + "b" * 64,
            },
        ],
        "packets": [
            {
                "stream_index": 0,
                "pts": 0,
                "dts": -1,
                "duration": 1,
                "size": "23",
                "data_hash": "SHA256:" + "c" * 64,
                "flags": "K_",
            },
            {
                "stream_index": 1,
                "pts": -1024,
                "dts": -1024,
                "duration": 1024,
                "size": "12",
                "data_hash": "SHA256:" + "d" * 64,
                "flags": "KD",
            },
        ],
    }


@pytest.fixture
def tool(tmp_path: Path, probe_data: dict[str, object]) -> Path:
    path = tmp_path / "probe"
    path.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "assert os.environ['TMPDIR'] == os.getcwd()\n"
        "assert '-show_frames' not in sys.argv\n"
        "assert '-show_data_hash' in sys.argv\n"
        f"print({json.dumps(probe_data)!r})\n"
    )
    path.chmod(0o700)
    return path


@pytest.mark.parametrize("inside", [True, False])
def test_configured_external_delivery_root(
    inputs: tuple[Path, Path, Path], tool: Path, monkeypatch: pytest.MonkeyPatch, inside: bool
) -> None:
    config, _, original = inputs
    monkeypatch.setattr(verifier, "REPOSITORY", config.parent)
    external = config.parent.with_name(config.parent.name + "-external")
    external.mkdir()
    configured = external / "recording"
    configured.mkdir()
    movie = (configured if inside else external) / "candidate.mp4"
    movie.write_bytes(original.read_bytes())
    data = verifier._json(config)
    data["deliveryRoot"] = str(configured)
    config.write_text(json.dumps(data))
    evidence = config.parent / "external-evidence"
    assert audit.audit(config, movie, evidence, ffprobe=tool) is inside
    receipt = verifier._json(evidence / "receipt.json")
    assert receipt["remuxIdentityAccepted"] is inside
    if not inside:
        assert receipt["error"] == "Candidate is outside configured delivery root"


def test_success(inputs: tuple[Path, Path, Path], tool: Path) -> None:
    config, _, movie = inputs
    evidence = config.parent / "evidence"
    assert audit.audit(config, movie, evidence, ffprobe=tool)
    receipt = verifier._json(evidence / "receipt.json")
    assert receipt["remuxIdentityAccepted"] is True
    assert receipt["qualityApproval"] is False
    assert receipt["syncApproval"] is False
    assert receipt["fullDecode"] is False
    assert receipt["nativeJobID"] == "native-1"
    assert len(verifier._rows(receipt, "inputs")) == 6
    assert receipt["inputs"] == receipt["inputsAfter"]
    assert verifier._rows(receipt, "streams")[1]["packets"] == 1
    with pytest.raises(FileExistsError):
        audit.audit(config, movie, evidence, ffprobe=tool)


@pytest.mark.parametrize("pin_changed", [False, True])
def test_legacy_history_requires_explicit_exact_job_state_pins(
    inputs: tuple[Path, Path, Path], tool: Path, pin_changed: bool
) -> None:
    config, source, movie = inputs
    old = source.with_name("final-encode.m4v")
    source.rename(old)
    jobs = config.parent / "final-job.json"
    state = config.parent / "final-job-state/final.json"
    rows = verifier._json(jobs)
    verifier._rows(rows, "jobs")[0]["output"] = str(old)
    jobs.write_text(json.dumps(rows))
    status = verifier._json(state)
    status["output"] = str(old)
    state.write_text(json.dumps(status))
    assert not audit.audit(config, movie, config.parent / "new-production", ffprobe=tool)
    descriptor = config.parent / "historical.json"
    descriptor.write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "historical-compressor-output",
                "jobs": {"path": str(jobs), "sha256": verifier._fingerprint(jobs)["sha256"]},
                "state": {"path": str(state), "sha256": verifier._fingerprint(state)["sha256"]},
            }
        )
    )
    if pin_changed:
        state.write_text(state.read_text() + " ")
    result = audit.audit(
        config,
        movie,
        config.parent / "historical-audit",
        ffprobe=tool,
        historical_bindings=descriptor,
    )
    assert result is (not pin_changed)
    evidence = verifier._json(config.parent / "historical-audit/receipt.json")
    assert evidence["historicalNaming"] is True
    assert old.read_bytes() == b"source"
    if pin_changed:
        assert evidence["error"] == "Historical job/state binding changed"


def test_history_unknown_schema_is_rejected(inputs: tuple[Path, Path, Path], tool: Path) -> None:
    config, _, movie = inputs
    descriptor = config.parent / "historical.json"
    descriptor.write_text('{"version":2}')
    assert not audit.audit(
        config,
        movie,
        config.parent / "historical-audit",
        ffprobe=tool,
        historical_bindings=descriptor,
    )
    assert verifier._json(config.parent / "historical-audit/receipt.json")["error"] == (
        "Unknown historical Compressor binding schema"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "Failed"),
        ("jobID", "bad id"),
        ("jobID", None),
        ("output", "/no/such/output.m4v"),
        ("output", "relative.m4v"),
        ("bytes", 7),
        ("bytes", True),
        ("retry_count", 1),
    ],
)
def test_receipt_rejected(
    inputs: tuple[Path, Path, Path], tool: Path, field: str, value: object
) -> None:
    config, _, movie = inputs
    state = config.parent / "final-job-state" / "final.json"
    raw = verifier._json(state)
    raw[field] = value
    state.write_text(json.dumps(raw))
    evidence = config.parent / "evidence"
    assert not audit.audit(config, movie, evidence, ffprobe=tool)
    assert verifier._json(evidence / "receipt.json")["remuxIdentityAccepted"] is False
    assert not (evidence / "source.stdout").exists()


@pytest.mark.parametrize("mode", ["ambiguous", "missing", "jobs", "outside", "same", "symlink"])
def test_mapping_rejected(inputs: tuple[Path, Path, Path], tool: Path, mode: str) -> None:
    config, source, movie = inputs
    if mode == "ambiguous":
        source.with_suffix(".mov").write_bytes(b"other")
    elif mode == "missing":
        source.unlink()
    elif mode == "jobs":
        (config.parent / "final-job.json").write_text('{"jobs":[{}]}')
    elif mode == "outside":
        raw = verifier._json(config)
        raw["deliveryRoot"] = str(config.parent / "different")
        config.write_text(json.dumps(raw))
    elif mode == "same":
        movie = source
    else:
        link = config.parent / "linked.mp4"
        link.symlink_to(movie)
        movie = link
    assert not audit.audit(config, movie, config.parent / "evidence", ffprobe=tool)


@pytest.mark.parametrize(
    "target",
    [
        "config.json",
        "final-job.json",
        "final-job-state/final.json",
        "candidate.mp4",
        "storage/final/2024-01-02_03-04-05_JST_replay-123-review.m4v",
        "unused",
    ],
)
def test_mutation(inputs: tuple[Path, Path, Path], tool: Path, target: str) -> None:
    config, _, movie = inputs
    tool.write_text(
        tool.read_text() + f"from pathlib import Path\np=Path({str(config.parent / target)!r})\n"
        "p.write_bytes(p.read_bytes()+b' ')\n"
    )
    evidence = config.parent / "evidence"
    assert not audit.audit(config, movie, evidence, ffprobe=tool)
    assert (
        verifier._json(evidence / "receipt.json")["bindingError"]
        == "Bound input changed during remux audit"
    )


@pytest.mark.parametrize("script", ["print('bad json')", "import sys; sys.exit(4)", "print('{}')"])
def test_probe_error(inputs: tuple[Path, Path, Path], tool: Path, script: str) -> None:
    config, _, movie = inputs
    tool.write_text(f"#!{sys.executable}\n{script}\n")
    assert not audit.audit(config, movie, config.parent / "evidence", ffprobe=tool)


@pytest.mark.parametrize(
    "field,value",
    [
        ("data_hash", "SHA256:" + "e" * 64),
        ("pts", 1),
        ("dts", 0),
        ("duration", 2),
        ("size", "24"),
        ("side_data_list", [{"skip_samples": 1}]),
        ("stream_index", 9),
    ],
)
@pytest.mark.parametrize("stream", [0, 1])
def test_packet_mismatch(
    probe_data: dict[str, object],
    field: str,
    value: object,
    stream: int,
) -> None:
    changed = copy.deepcopy(probe_data)
    verifier._rows(changed, "packets")[stream][field] = value
    with pytest.raises(ReplayError):
        audit.compare(probe_data, changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("codec_name", "h264"),
        ("width", 32),
        ("extradata_hash", "SHA256:" + "e" * 64),
        ("codec_tag_string", "hev1"),
    ],
)
def test_stream_mismatch(probe_data: dict[str, object], field: str, value: object) -> None:
    changed = copy.deepcopy(probe_data)
    verifier._rows(changed, "streams")[0][field] = value
    with pytest.raises(ReplayError):
        audit.compare(probe_data, changed)


def test_rational_rescale_and_tag(probe_data: dict[str, object]) -> None:
    changed = copy.deepcopy(probe_data)
    verifier._rows(probe_data, "streams")[0]["codec_tag_string"] = "hev1"
    verifier._rows(changed, "streams")[0]["time_base"] = "1/60"
    packet = verifier._rows(changed, "packets")[0]
    packet.update(dts=-2, duration=2)
    assert audit.compare(probe_data, changed, allow_hvc1=True)[0] == {
        "index": 0,
        "type": "video",
        "packets": 1,
        "sourceTimeBase": "1/30",
        "candidateTimeBase": "1/60",
        "hvc1TagChanged": True,
        "orderedPayloadsEqual": True,
        "rationalClocksEqual": True,
    }
    with pytest.raises(ReplayError, match="tag"):
        audit.compare(probe_data, changed)


@pytest.mark.parametrize(
    "mode", ["extra", "audio", "hash", "codec", "extradata", "duration", "count", "empty", "index"]
)
def test_malformed(probe_data: dict[str, object], mode: str) -> None:
    streams = verifier._rows(probe_data, "streams")
    packets = verifier._rows(probe_data, "packets")
    if mode == "extra":
        probe_data["streams"] = streams + [streams[0]]
    elif mode == "audio":
        streams[1]["codec_type"] = "video"
    elif mode == "hash":
        packets[0].pop("data_hash")
    elif mode == "codec":
        streams[0].pop("codec_name")
    elif mode == "extradata":
        streams[0].pop("extradata_hash")
    elif mode == "duration":
        packets[0]["duration"] = 0
    elif mode == "index":
        streams[1]["index"] = 0
    else:
        probe_data["packets"] = packets[:1] if mode == "count" else []
    with pytest.raises(ReplayError):
        audit.compare(probe_data, probe_data)


def test_order(probe_data: dict[str, object]) -> None:
    packets = verifier._rows(probe_data, "packets")
    probe_data["packets"] = packets + [dict(packets[0], data_hash="SHA256:" + "e" * 64)]
    changed = copy.deepcopy(probe_data)
    changed["packets"] = list(reversed(verifier._rows(changed, "packets")))
    with pytest.raises(ReplayError, match="payload"):
        audit.compare(probe_data, changed)


def test_cli(inputs: tuple[Path, Path, Path], tool: Path) -> None:
    config, _, movie = inputs
    args = [str(config), str(movie), str(config.parent / "evidence"), "--ffprobe", str(tool)]
    assert audit.main(args) == 0
    assert audit.main(args) == 1
    args[2] = str(config.parent / "timeout")
    assert audit.main(args + ["--timeout", "0"]) == 1
    args[2] = str(config.parent / "missing-tool")
    args[4] = "missing"
    assert audit.main(args) == 1
    with pytest.raises(ReplayError, match="repository-local"):
        audit.audit(config, movie, verifier.REPOSITORY.parent / "never-create", ffprobe=tool)


@pytest.mark.parametrize("field,value", [("data_hash", "SHA256:" + "e" * 64), ("pts", 1)])
def test_mismatch_returns_false(
    inputs: tuple[Path, Path, Path],
    tool: Path,
    probe_data: dict[str, object],
    field: str,
    value: object,
) -> None:
    config, _, movie = inputs
    changed = copy.deepcopy(probe_data)
    verifier._rows(changed, "packets")[0][field] = value
    tool.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"print({json.dumps(changed)!r} if sys.argv[-1].endswith('.mp4') "
        f"else {json.dumps(probe_data)!r})\n"
    )
    assert not audit.audit(config, movie, config.parent / "evidence", ffprobe=tool)


def test_extra_validation() -> None:
    with pytest.raises(ReplayError, match="Missing"):
        audit._extra({})
    with pytest.raises(ReplayError, match="Malformed"):
        audit._extra({"extradata": "00000001: 0000"})
    with pytest.raises(ReplayError, match="hash mismatch"):
        audit._extra({"extradata": "00000000: 0000", "extradata_size": 2})


@pytest.mark.parametrize(
    "data,message",
    [
        (b"short", "header"),
        (b"\x01" + b"\x00" * 21 + b"\x01", "array"),
        (b"\x01" + b"\x00" * 21 + b"\x01\x20\x00\x01", "NAL length"),
        (b"\x01" + b"\x00" * 21 + b"\x01\x20\x00\x01\x00\x10", "NAL payload"),
        (b"\x01" + b"\x00" * 23, "Trailing"),
    ],
)
def test_bad_hvcc(data: bytes, message: str) -> None:
    text = "\n".join(
        f"{offset:08x}: {data[offset : offset + 16].hex(' ', 2):39}  ignored"
        for offset in range(0, len(data), 16)
    )
    stream: dict[str, object] = {
        "extradata": text,
        "extradata_size": len(data),
        "extradata_hash": "SHA256:" + hashlib.sha256(data).hexdigest(),
    }
    with pytest.raises(ReplayError, match=message):
        audit._hvc1_extra(stream, stream)


@pytest.mark.parametrize(
    "script,timeout", [("import time; time.sleep(3)", 1), ("print('x'*20000)", 10)]
)
def test_process_limits(
    inputs: tuple[Path, Path, Path],
    tool: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    timeout: int,
) -> None:
    config, _, movie = inputs
    monkeypatch.setattr(verifier, "MAX_BYTES", 10000)
    tool.write_text(f"#!{sys.executable}\n{script}\n")
    evidence = config.parent / "evidence"
    assert not audit.audit(config, movie, evidence, ffprobe=tool, timeout=timeout)
    assert verifier._json(evidence / "receipt.json")["remuxIdentityAccepted"] is False
    assert verifier._json(evidence / "source.process.json")["returncode"] is not None


def test_hvcc_payload_mismatch() -> None:
    data = b"\x01" + b"\x00" * 21 + b"\x01\x20\x00\x01\x00\x01\x42"
    changed = data[:23] + b"\xa0" + data[24:-1] + b"\x43"
    source: dict[str, object] = {
        "extradata": "\n".join(
            f"{offset:08x}: {data[offset : offset + 16].hex(' ', 2):39}  ignored"
            for offset in range(0, len(data), 16)
        ),
        "extradata_size": len(data),
        "extradata_hash": "SHA256:" + hashlib.sha256(data).hexdigest(),
    }
    candidate: dict[str, object] = {
        "extradata": "\n".join(
            f"{offset:08x}: {changed[offset : offset + 16].hex(' ', 2):39}  ignored"
            for offset in range(0, len(changed), 16)
        ),
        "extradata_size": len(changed),
        "extradata_hash": "SHA256:" + hashlib.sha256(changed).hexdigest(),
    }
    assert not audit._hvc1_extra(source, candidate)


def test_real_hevc_aac(inputs: tuple[Path, Path, Path]) -> None:
    config, source, movie = inputs
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    assert ffmpeg is not None and ffprobe is not None
    source.unlink()
    movie.unlink()
    verifier._run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-f",
            "lavfi",
            "-i",
            "color=s=64x64:r=30:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.2",
            "-c:v",
            "libx265",
            "-x265-params",
            "log-level=error:pools=1",
            "-tag:v",
            "hev1",
            "-c:a",
            "aac",
            "-f",
            "mp4",
            str(source),
        ],
        directory=config.parent,
        name="fixture",
        timeout=30,
    )
    verifier._run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            str(source),
            "-map",
            "0",
            "-c",
            "copy",
            "-tag:v",
            "hvc1",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        directory=config.parent,
        name="remux",
        timeout=30,
    )
    state = config.parent / "final-job-state" / "final.json"
    raw = verifier._json(state)
    raw["bytes"] = source.stat().st_size
    state.write_text(json.dumps(raw))
    evidence = config.parent / "evidence"
    assert audit.audit(config, movie, evidence, ffprobe=Path(ffprobe), allow_hvc1=True)
    receipt = verifier._json(evidence / "receipt.json")
    assert verifier._rows(receipt, "streams")[0]["packets"] == 6
    bindings = verifier._rows(receipt, "inputs")
    assert record(bindings[4])["sha256"] != record(bindings[5])["sha256"]
    assert receipt["fullDecode"] is False
