"""Strict thresholds use raw counters, not rounded llvm-cov percentages."""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check-swift-coverage.sh"


@pytest.mark.parametrize(
    "row",
    [
        "Sources/A.swift 20 1 95% 20 1 95% 20 1 95% 20 1 95%",
        "Sources/A.swift 20 1 95% 20 1 95% 20 1 95% 0 0 -",
    ],
)
def test_swift_coverage_accepts_exact_threshold(tmp_path: Path, row: str) -> None:
    report = tmp_path / "coverage.txt"
    report.write_text(row + "\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(SCRIPT), str(report)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "PASS per-file coverage >=95%" in result.stdout


@pytest.mark.parametrize(
    "row",
    [
        "Sources/A.swift 20 1 95% 20 2 90% 20 1 95% 0 0 -",
        "Sources/A.swift 20 1 95% 20 1 95% 10001 501 95% 0 0 -",
        "Sources/A.swift 20 1 95% 20 1 95% 20 1 95% 20 2 90%",
        "Sources/A.swift malformed",
        "Sources/A.swift 20 1 95% 20 -1 95% 20 1 95% 0 0 -",
        "Sources/A.swift 20 1 95% 20 nope 95% 20 1 95% 0 0 -",
        "Sources/A.swift 20 0 100% 20 0 100% 20 0 100% 0 0 -\n"
        "Sources/B.swift 20 0 100% 20 2 90% 20 0 100% 0 0 -",
        "TOTAL 20 0 100%",
    ],
)
def test_swift_coverage_rejects_invalid_or_low_file(tmp_path: Path, row: str) -> None:
    report = tmp_path / "coverage.txt"
    report.write_text(row + "\n", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(SCRIPT), str(report)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout
