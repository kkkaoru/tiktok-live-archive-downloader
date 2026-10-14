"""Pinned pre-recovery generation rules use synthetic dicts and tiny local files only."""

import ast
import hashlib
from pathlib import Path

import pytest

from replay import caption_io, cleanup_lane_successor
from replay import code_generation_compat as compat


def test_cleanup_uses_only_ocr_parser_from_caption_io() -> None:
    tree = ast.parse(Path(cleanup_lane_successor.__file__).read_text(encoding="utf-8"))
    names = {
        name.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "replay.caption_io"
        for name in node.names
    }
    assert names == {"native_ocr_frames"}
    assert "SpokenToken" not in caption_io.native_ocr_frames.__code__.co_names
    assert caption_io.native_ocr_frames([]) == ()


@pytest.mark.parametrize("name", tuple(compat.CAPTION_IMPORT_DIGESTS))
def test_reviewed_transitive_import_requires_exact_recorded_digest(name: str) -> None:
    path = f"/repo/src/replay/{name}"
    recorded = {path: compat.CAPTION_IMPORT_DIGESTS[name], "/repo/src/replay/data.py": "a" * 64}
    current = {path: "c" * 64, "/repo/src/replay/data.py": "a" * 64}
    assert compat.historical_generation(recorded, current)
    assert not compat.historical_generation({**recorded, path: "f" * 64}, current)
    assert not compat.historical_generation(
        {**recorded, "/repo/src/replay/data.py": "b" * 64}, current
    )


def current(extra: dict[str, str] | None = None) -> dict[str, str]:
    base = {
        "/repo/src/replay/alpha.py": "a" * 64,
        "/repo/src/replay/subtitle_session_cleanup.py": "1" * 64,
        "/repo/src/replay/subtitle_session_cleanup_plan.py": "2" * 64,
        "/repo/src/replay/subtitle_session_cleanup_wave.py": "3" * 64,
    }
    if extra:
        base.update(extra)
    return base


def without_compat(code: dict[str, str]) -> dict[str, str]:
    return {
        path: digest for path, digest in code.items() if not path.endswith(compat.COMPAT_BASENAME)
    }


def test_current_generation_accepted() -> None:
    code = current()
    assert compat.historical_generation(code, code) is True


def test_pre_recovery_generation_accepted_without_compat_key() -> None:
    code = current()
    recorded = {
        path: compat.PRE_RECOVERY_DIGESTS[path.rsplit("/", 1)[-1]]
        for path in without_compat(code)
        if path.rsplit("/", 1)[-1] in compat.RECOVERY_FILES
    }
    recorded.update(
        {
            path: digest
            for path, digest in without_compat(code).items()
            if path.rsplit("/", 1)[-1] not in compat.RECOVERY_FILES
        }
    )
    assert compat.historical_generation(recorded, code) is True


def test_v4_style_generation_accepted_without_compat_key() -> None:
    code = current()
    assert compat.historical_generation(without_compat(code), code) is True


def test_recovery_wave_generation_accepted() -> None:
    code = current()
    recorded = {
        path: compat.RECOVERY_WAVE_DIGESTS[path.rsplit("/", 1)[-1]]
        for path in without_compat(code)
        if path.rsplit("/", 1)[-1] in compat.RECOVERY_FILES
    }
    recorded.update(
        {
            path: digest
            for path, digest in without_compat(code).items()
            if path.rsplit("/", 1)[-1] not in compat.RECOVERY_FILES
        }
    )
    assert compat.historical_generation(recorded, code) is True


@pytest.mark.parametrize(
    "mutation", ["fourth-file", "wrong-old-digest", "extra-key", "missing-key", "non-string"]
)
def test_unknown_generations_rejected(mutation: str) -> None:
    code = current()
    recorded: dict[str, object] = dict(without_compat(code))
    if mutation == "fourth-file":
        recorded["/repo/src/replay/alpha.py"] = "f" * 64
    elif mutation == "wrong-old-digest":
        recorded["/repo/src/replay/subtitle_session_cleanup.py"] = "e" * 64
    elif mutation == "extra-key":
        recorded["/repo/src/replay/unknown.py"] = "e" * 64
    elif mutation == "missing-key":
        del recorded["/repo/src/replay/alpha.py"]
    else:
        recorded["/repo/src/replay/alpha.py"] = 42
    assert compat.historical_generation(recorded, code) is False


@pytest.mark.parametrize("recorded", [None, ["x"], 42])
def test_non_mapping_rejected(recorded: object) -> None:
    assert compat.historical_generation(recorded, current()) is False
    assert compat.historical_generation(current(), recorded) is False
    with pytest.raises(ValueError, match="explicit text"):
        compat.verify_recorded_pins(recorded, current())


def test_missing_compat_key_only_tolerated() -> None:
    code = current({"/repo/src/replay/code_generation_compat.py": "c" * 64})
    recorded = {path: digest for path, digest in code.items() if not path.endswith("alpha.py")}
    assert compat.historical_generation(recorded, code) is False


def test_verify_recorded_pins(tmp_path: Path) -> None:
    current_file = tmp_path / "current.py"
    current_file.write_bytes(b"current code")
    current_digest = hashlib.sha256(b"current code").hexdigest()
    recorded = {str(current_file): current_digest}
    current = {str(current_file): current_digest}
    compat.verify_recorded_pins(recorded, current)
    old_path = tmp_path / "subtitle_session_cleanup.py"
    old_path.write_bytes(b"new code")
    recorded_old = {str(old_path): compat.PRE_RECOVERY_DIGESTS["subtitle_session_cleanup.py"]}
    current_new = {str(old_path): hashlib.sha256(b"new code").hexdigest()}
    compat.verify_recorded_pins(recorded_old, current_new)
    old_path.write_bytes(b"tampered code")
    with pytest.raises(ValueError, match="Running recovery code"):
        compat.verify_recorded_pins(recorded_old, current_new)
    with pytest.raises(ValueError, match="Recorded code changed"):
        compat.verify_recorded_pins({str(current_file): "0" * 64}, current)
    with pytest.raises((OSError, ValueError)):
        compat.verify_recorded_pins({str(tmp_path / "absent.py"): "0" * 64}, current)
    bad: dict[str, object] = {str(current_file): 42}
    with pytest.raises(ValueError, match="explicit text"):
        compat.verify_recorded_pins(bad, current)


def test_verify_symlink_and_oversize_rejected(tmp_path: Path) -> None:
    target = tmp_path / "real.py"
    target.write_bytes(b"code")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        compat.verify_recorded_pins({str(link): "0" * 64}, {str(link): "0" * 64})
    big = tmp_path / "big.py"
    big.write_bytes(b"x" * (compat.MAX_PIN_BYTES + 1))
    with pytest.raises(ValueError, match="bounded pin size"):
        compat.verify_recorded_pins({str(big): "0" * 64}, {str(big): "0" * 64})


def test_historical_stage_input_sets() -> None:
    assert compat.historical_stage_input_sets({"/x/other.py": "a" * 64}) == []
    assert compat.historical_stage_input_sets({}) == []
    rewritten = compat.historical_stage_input_sets(
        {
            "/repo/subtitle_session_cleanup.py": "9" * 64,
            "/repo/other.py": "a" * 64,
        }
    )
    assert rewritten == [
        {
            "/repo/subtitle_session_cleanup.py": compat.PRE_RECOVERY_DIGESTS[
                "subtitle_session_cleanup.py"
            ],
            "/repo/other.py": "a" * 64,
        },
        {
            "/repo/subtitle_session_cleanup.py": compat.RECOVERY_WAVE_DIGESTS[
                "subtitle_session_cleanup.py"
            ],
            "/repo/other.py": "a" * 64,
        },
    ]


def test_pins_are_full_hex_digests() -> None:
    assert set(compat.PRE_RECOVERY_DIGESTS) == compat.RECOVERY_FILES
    assert set(compat.RECOVERY_WAVE_DIGESTS) == compat.RECOVERY_FILES
    for digests in (compat.PRE_RECOVERY_DIGESTS, compat.RECOVERY_WAVE_DIGESTS):
        for digest in digests.values():
            assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
    assert compat.PRE_RECOVERY_DIGESTS != compat.RECOVERY_WAVE_DIGESTS


def test_accepted_input_digest_only_for_reviewed_code() -> None:
    current = "c" * 64
    assert (
        compat.accepted_input_digest(
            "/repo/subtitle_session_cleanup.py",
            compat.PRE_RECOVERY_DIGESTS["subtitle_session_cleanup.py"],
            current,
        )
        is True
    )
    assert (
        compat.accepted_input_digest(
            "/repo/subtitle_session_cleanup_wave.py",
            compat.RECOVERY_WAVE_DIGESTS["subtitle_session_cleanup_wave.py"],
            current,
        )
        is True
    )
    assert compat.accepted_input_digest("/repo/subtitle_session_cleanup.py", current, current)
    assert not compat.accepted_input_digest("/repo/subtitle_session_cleanup.py", "e" * 64, current)
    assert not compat.accepted_input_digest("/repo/other.py", "e" * 64, current)
    assert not compat.accepted_input_digest("/repo/other.py", current, current)
    assert not compat.accepted_input_digest(None, current, current)
    assert not compat.accepted_input_digest("/repo/subtitle_session_cleanup.py", None, current)
