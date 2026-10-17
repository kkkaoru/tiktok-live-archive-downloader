import hashlib
import json
from pathlib import Path

import pytest

from replay import receipt_repin, work_roots
from replay.core import ReplayError

OLD = "e2f53c85c4cee47bacde57bb00646a295f8e7989d887443db318deab214cae99"
NEW = "4e27c20c2cf66c3311b8783f024fd19a647be02bbecb9e10c8c168d338263556"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_pins_finds_nested_pin_objects() -> None:
    value = {
        "inputs": [{"path": "/a", "sha256": "0" * 64, "bytes": 1, "stat": {}}],
        "nested": {"code": {"path": "/b", "sha256": "1" * 64, "bytes": 2}},
        "other": {"path": "/c", "sha256": "short", "bytes": 3},
    }
    assert [pin["path"] for pin in receipt_repin.pins(value)] == ["/a", "/b"]


def test_hasher_reads_a_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "f"
    source.write_bytes(b"abc")
    hasher = receipt_repin.Hasher()
    assert (
        hasher.digest(source) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    monkeypatch.setattr(Path, "open", lambda *_: pytest.fail("read twice"))
    assert (
        hasher.digest(source) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_check_verifies_a_relocated_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    moved = tmp_path / "new" / "x.json"
    moved.parent.mkdir()
    moved.write_bytes(b"data")
    monkeypatch.setattr(work_roots, "resolve", lambda _: moved)
    result = receipt_repin.check(
        {"path": "/old/x.json", "sha256": digest(b"data"), "bytes": 4}, receipt_repin.Hasher()
    )
    assert (result.recorded, result.current, result.status) == (
        "/old/x.json",
        str(moved),
        "verified",
    )
    assert result.identity is not None
    assert result.identity["bytes"] == 4


def test_check_resolves_relative_paths_from_the_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Path] = []

    def resolve(path: Path) -> Path:
        seen.append(path)
        return Path("/missing/nowhere")

    monkeypatch.setattr(work_roots, "resolve", resolve)
    result = receipt_repin.check(
        {"path": "edits/x", "sha256": "0" * 64, "bytes": 1}, receipt_repin.Hasher()
    )
    assert seen == [work_roots.CHECKOUT / "edits/x"]
    assert result.status == "missing"


def test_check_reports_a_changed_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = tmp_path / "x"
    current.write_bytes(b"new")
    monkeypatch.setattr(work_roots, "resolve", lambda _: current)
    result = receipt_repin.check(
        {"path": str(current), "sha256": digest(b"old"), "bytes": 3}, receipt_repin.Hasher()
    )
    assert result.status == "changed"


def test_check_rejects_a_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "t"
    target.write_bytes(b"x")
    link = tmp_path / "l"
    link.symlink_to(target)
    monkeypatch.setattr(work_roots, "resolve", lambda _: link)
    assert (
        receipt_repin.check(
            {"path": str(link), "sha256": digest(b"x"), "bytes": 1}, receipt_repin.Hasher()
        ).status
        == "missing"
    )


def test_check_accepts_only_the_approved_code_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = tmp_path / "src" / "replay" / "native_invocation.py"
    code.parent.mkdir(parents=True)
    code.write_bytes(b"code")
    monkeypatch.setattr(work_roots, "resolve", lambda _: code)
    monkeypatch.setattr(receipt_repin.Hasher, "digest", lambda _self, _path: NEW)
    approved = receipt_repin.check(
        {"path": str(code), "sha256": OLD, "bytes": 4}, receipt_repin.Hasher()
    )
    other = receipt_repin.check(
        {"path": str(code), "sha256": "2" * 64, "bytes": 4}, receipt_repin.Hasher()
    )
    assert (approved.status, other.status) == ("approved-code-change", "changed")


def test_repin_writes_a_sidecar_and_leaves_the_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(work_roots, "resolve", lambda path: path)
    source = tmp_path / "in.bin"
    source.write_bytes(b"payload")
    receipt = tmp_path / "report.json"
    text = json.dumps(
        {
            "inputs": [
                {
                    "path": str(source),
                    "sha256": digest(b"payload"),
                    "bytes": 7,
                    "stat": {"device": 1},
                }
            ]
        }
    )
    receipt.write_text(text, encoding="utf-8")
    outcome = receipt_repin.repin(receipt, receipt_repin.Hasher())
    sidecar = json.loads((tmp_path / "report.json.repin.json").read_text(encoding="utf-8"))
    assert outcome == {"receipt": str(receipt), "pins": 1, "status": "repinned"}
    assert receipt.read_text(encoding="utf-8") == text
    assert sidecar["receiptSha256"] == digest(text.encode())
    assert sidecar["inputs"][0]["status"] == "verified"
    assert sidecar["inputs"][0]["currentPath"] == str(source)


def test_repin_reports_unresolved_inputs_without_a_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(work_roots, "resolve", lambda path: path)
    receipt = tmp_path / "report.json"
    receipt.write_text(
        json.dumps({"inputs": [{"path": "/gone/x", "sha256": "0" * 64, "bytes": 1}]}),
        encoding="utf-8",
    )
    outcome = receipt_repin.repin(receipt, receipt_repin.Hasher())
    assert outcome == {
        "receipt": str(receipt),
        "pins": 1,
        "status": "unresolved",
        "unresolved": [{"path": "/gone/x", "status": "missing"}],
    }
    assert not (tmp_path / "report.json.repin.json").exists()


def test_repin_rejects_non_json(tmp_path: Path) -> None:
    receipt = tmp_path / "bad.json"
    receipt.write_text("{", encoding="utf-8")
    with pytest.raises(ReplayError, match="is not JSON"):
        receipt_repin.repin(receipt, receipt_repin.Hasher())


def test_repin_rejects_an_oversized_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receipt_repin, "MAX_RECEIPT_BYTES", 1)
    receipt = tmp_path / "big.json"
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="exceeds the receipt size bound"):
        receipt_repin.repin(receipt, receipt_repin.Hasher())


def test_run_turns_errors_into_outcomes(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    assert receipt_repin.run([missing], workers=1)[0]["status"] == "error"


def test_summary_counts_statuses() -> None:
    outcomes = [{"status": "repinned"}, {"status": "error"}, {"status": "repinned"}]
    assert receipt_repin.summary(outcomes) == {"error": 1, "repinned": 2}


def test_main_writes_a_report_and_fails_on_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(work_roots, "resolve", lambda path: path)
    good = tmp_path / "good.json"
    good.write_text("{}", encoding="utf-8")
    listing = tmp_path / "list.txt"
    listing.write_text(f"{good}\n\n", encoding="utf-8")
    assert (
        receipt_repin.main([str(listing), "--report", str(tmp_path / "r.json"), "--workers", "1"])
        == 0
    )
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    listing.write_text(f"{bad}\n", encoding="utf-8")
    assert receipt_repin.main([str(listing), "--report", str(tmp_path / "r.json")]) == 1
    assert capsys.readouterr().out == '{"repinned": 1}\n{"error": 1}\n'
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["summary"] == {"error": 1}
