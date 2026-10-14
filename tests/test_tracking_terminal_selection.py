import json
import sys
from pathlib import Path

import pytest

from replay import tracking_terminal_selection as terminal
from replay.core import ReplayError
from replay.stages import file_digest
from replay.tracking_observations import load_job


@pytest.fixture
def case(tmp_path: Path) -> tuple[Path, Path]:
    source, native = tmp_path / "source", tmp_path / "native"
    source.write_bytes(b"synthetic only")
    native.write_text(
        f"#!{sys.executable}\nimport json,sys\n"
        "if sys.argv[1]=='inspect-media':\n"
        " print(json.dumps(dict(durationSeconds=.09,width=16,height=16,firstFrameDecoded=True)))\n"
        "else:\n"
        " print(json.dumps([dict(presentationValue=6,presentationTimescale=100,"
        "width=16,height=16,method='missing',detectionAttempted=True,faces=[])]))\n",
        encoding="utf-8",
    )
    native.chmod(0o700)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "version": 1,
                "source": str(source),
                "sourceSHA256": file_digest(source),
                "nativeExecutable": str(native),
                "nativeSHA256": file_digest(native),
                "clock": {"pts": [6], "timeBase": "1/100", "lastDuration": 4},
                "width": 16,
                "height": 16,
            }
        ),
        encoding="utf-8",
    )
    return config, tmp_path / "output"


def test_observe_verify_preserves_original_clock(case: tuple[Path, Path]) -> None:
    config, output = case
    terminal.observe(config, output)
    terminal.verify(config, output)
    data = json.loads((output / "receipt.json").read_text())
    assert data["originalClock"] == {"pts": [6], "timeBase": "1/100", "lastDuration": 4}
    assert data["frames"] == 1
    assert data["captionPlacementApproved"] is False
    assert json.loads((output / "request.json").read_text())["plan"]["durationSeconds"] == 0.03
    with pytest.raises(FileExistsError):
        terminal.observe(config, output)


@pytest.mark.parametrize("duration", [True, -1, float("nan"), 0.06, 0.1, 0.08 + 0.00001])
def test_reject_extent(case: tuple[Path, Path], duration: object) -> None:
    with pytest.raises(ValueError):
        terminal.selection_request(
            load_job(case[0]),
            {
                "durationSeconds": duration,
                "width": 16,
                "height": 16,
                "firstFrameDecoded": True,
            },
        )


@pytest.mark.parametrize(
    "field,value", [("width", 17), ("height", 17), ("firstFrameDecoded", False)]
)
def test_geometry_guard(case: tuple[Path, Path], field: str, value: object) -> None:
    inspection: dict[str, object] = {
        "durationSeconds": 0.09,
        "width": 16,
        "height": 16,
        "firstFrameDecoded": True,
    }
    inspection[field] = value
    with pytest.raises(ValueError, match="observation"):
        terminal.selection_request(load_job(case[0]), inspection)


@pytest.mark.parametrize(
    "name,text",
    [
        ("request.json", "{}"),
        ("receipt.json", "{}"),
        ("stderr.log", "denied"),
        ("response.json", "[]"),
    ],
)
def test_changed_outputs(case: tuple[Path, Path], name: str, text: str) -> None:
    config, output = case
    terminal.observe(config, output)
    (output / name).write_text(text, encoding="utf-8")
    with pytest.raises((ValueError, ReplayError)):
        terminal.verify(config, output)


def test_source_change_rejected(case: tuple[Path, Path]) -> None:
    config, output = case
    terminal.observe(config, output)
    load_job(config).source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        terminal.verify(config, output)


def test_cli(case: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    config, output = case
    monkeypatch.setattr(sys, "argv", ["terminal", "observe", str(config), str(output)])
    terminal.main()
    monkeypatch.setattr(sys, "argv", ["terminal", "verify", str(config), str(output)])
    terminal.main()
