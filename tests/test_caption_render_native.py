"""The native caption-render backend with a fake renderer, Compressor and parity frames."""

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from replay import caption_gradient_audit, caption_parity
from replay import caption_render_native as native
from replay.core import ReplayError


def binary(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "caption-alpha-render"
    path.write_bytes(b"synthetic renderer")
    return path, hashlib.sha256(b"synthetic renderer").hexdigest()


def jobs(tmp_path: Path, names: Sequence[str] = ("w000-part-0000", "w001-part-0000")) -> Path:
    rows = []
    for name in names:
        source = tmp_path / "moti" / f"{name}.moti"
        source.parent.mkdir(exist_ok=True)
        source.write_text(f"<ozml>{name}</ozml>", encoding="utf-8")
        output = tmp_path / "alpha" / name / "compressor-batch-v1" / "a.mov"
        rows.append({"name": name, "source": str(source), "preset": "/p", "output": str(output)})
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": rows}), encoding="utf-8")
    return path


class Renderer:
    """Writes the requested output like `caption-alpha-render --moti` and records the calls."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> str:
        self.commands.append(list(command))
        if command[1] in ("--moti", "--inspect") and "glow" in Path(command[2]).read_text("utf-8"):
            raise native.Refused("Unsupported Motion caption part: title 0 differs (glow)")
        if command[1] == "--moti":
            Path(command[3]).write_bytes(b"alpha")
        return json.dumps({"frames": 3})


def test_every_job_is_rendered_once_and_recorded_like_a_compressor_state(tmp_path: Path) -> None:
    path, digest = binary(tmp_path)
    run = Renderer()
    outcome = native.render(jobs(tmp_path), tmp_path / "state", binary=path, binary_sha256=digest,
                            workers=2, run=run)  # fmt: skip
    assert outcome == {"rendered": 2}
    state = json.loads((tmp_path / "state" / "w000-part-0000.json").read_text(encoding="utf-8"))
    assert (state["status"], state["renderer"], state["bytes"], state["report"]) == (
        "Successful", "native-motion-document", 5, {"frames": 3},
    )  # fmt: skip
    assert state["output"] == str(tmp_path / "alpha/w000-part-0000/compressor-batch-v1/a.mov")
    again = native.render(jobs(tmp_path), tmp_path / "state", binary=path, binary_sha256=digest,
                          workers=1, run=run)  # fmt: skip
    assert (again, len(run.commands)) == ({"reused": 2}, 2)


def test_an_unrecorded_output_is_never_overwritten(tmp_path: Path) -> None:
    path, digest = binary(tmp_path)
    listed = jobs(tmp_path, ["w000-part-0000"])
    stale = tmp_path / "alpha/w000-part-0000/compressor-batch-v1/a.mov"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"left over")
    with pytest.raises(ReplayError, match="inspect it"):
        native.render(listed, tmp_path / "state", binary=path, binary_sha256=digest, workers=1,
                      run=Renderer())  # fmt: skip


def test_a_document_changed_while_rendering_is_refused(tmp_path: Path) -> None:
    path, digest = binary(tmp_path)
    listed = jobs(tmp_path, ["w000-part-0000"])

    def changing(command: Sequence[str]) -> str:
        Path(command[2]).write_text("<ozml>changed</ozml>", encoding="utf-8")
        Path(command[3]).write_bytes(b"alpha")
        return "{}"

    with pytest.raises(ReplayError, match="changed while rendering"):
        native.render(listed, tmp_path / "state", binary=path, binary_sha256=digest, workers=1,
                      run=changing)  # fmt: skip


def test_only_the_pinned_renderer_runs(tmp_path: Path) -> None:
    path, _ = binary(tmp_path)
    with pytest.raises(ReplayError, match="pinned native caption renderer"):
        native.render(jobs(tmp_path), tmp_path / "state", binary=path, binary_sha256="0" * 64,
                      workers=1, run=Renderer())  # fmt: skip


def test_the_worker_count_is_bounded(tmp_path: Path) -> None:
    path, digest = binary(tmp_path)
    with pytest.raises(ReplayError, match="1 to 8"):
        native.render(jobs(tmp_path), tmp_path / "state", binary=path, binary_sha256=digest,
                      workers=9, run=Renderer())  # fmt: skip


def test_malformed_job_lists_are_refused(tmp_path: Path) -> None:
    listed = tmp_path / "jobs.json"
    listed.write_text(json.dumps({"jobs": []}), encoding="utf-8")
    with pytest.raises(ReplayError, match="lists no caption jobs"):
        native.jobs_of(listed)
    listed.write_text(json.dumps({"jobs": [{"name": "a", "source": "/s"}]}), encoding="utf-8")
    with pytest.raises(ReplayError, match="name, a source and an output"):
        native.jobs_of(listed)
    row = {"name": "a", "source": "/s", "output": "/o"}
    listed.write_text(json.dumps({"jobs": [row, row]}), encoding="utf-8")
    with pytest.raises(ReplayError, match="unique"):
        native.jobs_of(listed)


def test_a_recorded_state_is_never_replaced(tmp_path: Path) -> None:
    state = tmp_path / "w.json"
    native.write_state(state, {"status": "Successful"})
    with pytest.raises(ReplayError, match="already recorded"):
        native.write_state(state, {"status": "Successful"})


def parts(tmp_path: Path) -> Path:
    path = tmp_path / "parts.json"
    rows = [
        {"name": "w000-part-0000", "stages": 3, "frames": 90},
        {"name": "w001-part-0000", "stages": 9, "frames": 30},
        {"name": "w002-part-0000", "stages": 9, "frames": 60},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_ranking_prefers_more_stages_then_longer_parts(tmp_path: Path) -> None:
    assert native.ranked(parts(tmp_path)) == ["w002-part-0000", "w001-part-0000", "w000-part-0000"]


def test_sampling_skips_parts_the_reader_refuses(tmp_path: Path) -> None:
    path, _ = binary(tmp_path)
    listed = native.jobs_of(jobs(tmp_path, ["w000-part-0000", "w001-part-0000", "w002-part-0000"]))
    (tmp_path / "moti/w002-part-0000.moti").write_text("<ozml>glow</ozml>", encoding="utf-8")
    by_name = {job["name"]: job for job in listed}
    chosen = native.sampled(parts(tmp_path), 2, by_name, binary=path, run=Renderer())
    assert chosen == ["w001-part-0000", "w000-part-0000"]


def caption(shift: int) -> NDArray[np.uint8]:
    frame = np.zeros((80, 120, 4), dtype=np.uint8)
    frame[10:60, 10 + shift : 100 + shift] = (0, 0, 0, 255)
    frame[18:52, 18 + shift : 92 + shift] = (47, 176, 238, 255)
    for gap in range(27, 92, 9):
        frame[18:52, gap + shift : gap + 2 + shift] = (0, 0, 0, 255)
    return frame


def gate_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native_shift: int
) -> tuple[Path, str, Path, Path, Renderer]:
    path, digest = binary(tmp_path)
    listed = jobs(tmp_path, ["w000-part-0000", "w001-part-0000", "w002-part-0000"])
    floor = tmp_path / "floor.json"
    floor.write_text(
        json.dumps({"floor": {name: 0.0 for name in caption_parity.MINIMUM_ERROR}}),
        encoding="utf-8",
    )

    def frames(movie: str, *, step: int) -> Iterator[NDArray[np.uint8]]:
        yield caption(native_shift if "/native/" in movie else 0)

    monkeypatch.setattr(caption_gradient_audit, "frames", frames)
    return path, digest, listed, floor, Renderer()


def test_the_gate_passes_identical_renders_and_records_its_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest, listed, floor, run = gate_setup(tmp_path, monkeypatch, 0)
    receipt = native.gate(parts(tmp_path), listed, tmp_path / "gate", binary=path,
                          binary_sha256=digest, floor_json=floor, sample=2, run=run)  # fmt: skip
    assert (receipt["pass"], receipt["parts"]) == (True, ["w002-part-0000", "w001-part-0000"])
    assert [command[1] for command in run.commands[:2]] == ["--inspect", "--inspect"]
    assert run.commands[2][3:5] == [
        "replay.compressor_batch",
        str(tmp_path / "gate/motion-jobs.json"),
    ]
    assert [command[1] for command in run.commands[3:]] == ["--moti", "--moti"]
    assert json.loads((tmp_path / "gate/gate.json").read_text(encoding="utf-8"))["pass"] is True


def test_the_gate_fails_a_displaced_native_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest, listed, floor, run = gate_setup(tmp_path, monkeypatch, 3)
    with pytest.raises(ReplayError, match="failed the parity gate"):
        native.gate(parts(tmp_path), listed, tmp_path / "gate", binary=path, binary_sha256=digest,
                    floor_json=floor, sample=1, run=run)  # fmt: skip
    receipt = json.loads((tmp_path / "gate/gate.json").read_text(encoding="utf-8"))
    assert "glyphX" in receipt["results"]["w002-part-0000"]["failed"]


def test_the_gate_refuses_missing_documents_and_bad_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest, listed, floor, run = gate_setup(tmp_path, monkeypatch, 0)
    with pytest.raises(ReplayError, match="1 to 8"):
        native.gate(parts(tmp_path), listed, tmp_path / "gate", binary=path, binary_sha256=digest,
                    floor_json=floor, sample=0, run=run)  # fmt: skip
    (tmp_path / "moti/w002-part-0000.moti").unlink()
    with pytest.raises(ReplayError, match="Motion document of w002-part-0000 is missing"):
        native.gate(parts(tmp_path), listed, tmp_path / "gate", binary=path, binary_sha256=digest,
                    floor_json=floor, sample=1, run=run)  # fmt: skip


def test_the_gate_fails_when_no_part_is_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest, listed, floor, run = gate_setup(tmp_path, monkeypatch, 0)
    for name in ("w000-part-0000", "w001-part-0000", "w002-part-0000"):
        (tmp_path / f"moti/{name}.moti").write_text("<ozml>glow</ozml>", encoding="utf-8")
    with pytest.raises(ReplayError, match="failed the parity gate"):
        native.gate(parts(tmp_path), listed, tmp_path / "gate", binary=path, binary_sha256=digest,
                    floor_json=floor, sample=1, run=run)  # fmt: skip
    receipt = json.loads((tmp_path / "gate/gate.json").read_text(encoding="utf-8"))
    assert receipt == {"pass": False, "parts": [], "reason": "no part was readable"}


def run_caption_render(tmp_path: Path, run: Renderer, path: Path, digest: str, floor: Path,
                       listed: Path) -> dict[str, object]:  # fmt: skip
    return native.caption_render(
        parts(tmp_path), listed, tmp_path / "state", tmp_path / "work", binary=path,
        binary_sha256=digest, floor_json=floor, sample=1, workers=1, in_flight=2,
        submit_workers=1, run=run,
    )  # fmt: skip


def test_a_refused_part_alone_falls_back_to_compressor_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest, listed, floor, run = gate_setup(tmp_path, monkeypatch, 0)
    (tmp_path / "moti/w000-part-0000.moti").write_text("<ozml>glow</ozml>", encoding="utf-8")
    receipt = run_caption_render(tmp_path, run, path, digest, floor, listed)
    assert (receipt["renderer"], receipt["native"]) == ("mixed", {"rendered": 2})
    assert receipt["fallback"] == {
        "w000-part-0000": "Unsupported Motion caption part: title 0 differs (glow)"
    }
    fallback = run.commands[-1]
    assert fallback[3:5] == ["replay.compressor_batch", str(tmp_path / "work/compressor-jobs.json")]
    assert fallback[fallback.index("--state") + 1] == str(tmp_path / "state")
    queued = json.loads((tmp_path / "work/compressor-jobs.json").read_text(encoding="utf-8"))
    assert [job["name"] for job in queued["jobs"]] == ["w000-part-0000"]
    written = json.loads((tmp_path / "work/receipt.json").read_text(encoding="utf-8"))
    assert written["renderer"] == "mixed"


def test_every_part_is_native_when_all_are_readable_and_the_gate_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest, listed, floor, run = gate_setup(tmp_path, monkeypatch, 0)
    receipt = run_caption_render(tmp_path, run, path, digest, floor, listed)
    assert (receipt["renderer"], receipt["native"], receipt["fallback"]) == (
        "native", {"rendered": 3}, {},
    )  # fmt: skip
    assert not (tmp_path / "work/compressor-jobs.json").exists()


def test_a_failed_gate_renders_every_part_with_compressor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest, listed, floor, run = gate_setup(tmp_path, monkeypatch, 3)
    receipt = run_caption_render(tmp_path, run, path, digest, floor, listed)
    assert (receipt["renderer"], receipt["native"]) == ("compressor", {})
    assert sorted(record_names(receipt["fallback"])) == [
        "w000-part-0000", "w001-part-0000", "w002-part-0000",
    ]  # fmt: skip
    assert str(receipt["fallback"]).count("parity gate failed") == 3
    assert not (tmp_path / "state").exists()


def record_names(value: object) -> list[str]:
    assert isinstance(value, dict)
    return [str(name) for name in value]


def test_the_renderer_refusal_is_told_apart_from_a_failure(tmp_path: Path) -> None:
    refusing = ["/bin/sh", "-c", "echo 'Unsupported Motion caption part: x' >&2; exit 2"]
    with pytest.raises(native.Refused, match="Unsupported Motion caption part: x"):
        native.run_checked(refusing)
    with pytest.raises(ReplayError, match=r"failed \(3\): broken") as failure:
        native.run_checked(["/bin/sh", "-c", "echo broken >&2; exit 3"])
    assert not isinstance(failure.value, native.Refused)
    assert native.run_checked(["/bin/sh", "-c", "echo ok"]) == "ok\n"


def test_main_runs_both_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path, digest, listed, floor, run = gate_setup(tmp_path, monkeypatch, 0)
    monkeypatch.setattr(native, "run_checked", run)
    common = ["--binary", str(path), "--binary-sha256", digest]
    assert native.main(["gate", str(parts(tmp_path)), str(listed), *common,
                        "--work", str(tmp_path / "gate"), "--floor-json", str(floor),
                        "--sample", "1"]) == 0  # fmt: skip
    assert json.loads(capsys.readouterr().out) == {"pass": True, "parts": ["w002-part-0000"]}
    assert native.main(["render", str(listed), *common, "--state", str(tmp_path / "s")]) == 0
    assert json.loads(capsys.readouterr().out) == {"rendered": 3}
    (tmp_path / "again").mkdir()
    again = jobs(tmp_path / "again", ["w000-part-0000", "w001-part-0000", "w002-part-0000"])
    assert native.main(["run", str(parts(tmp_path)), str(again), *common,
                        "--work", str(tmp_path / "w"), "--floor-json", str(floor),
                        "--state", str(tmp_path / "t"), "--sample", "1"]) == 0  # fmt: skip
    assert json.loads(capsys.readouterr().out) == {"renderer": "native", "fallback": {}}
