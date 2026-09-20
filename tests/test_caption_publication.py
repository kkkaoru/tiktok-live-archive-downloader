import shutil
import subprocess
from pathlib import Path


def test_edit_workspaces_are_ignored_including_nonmedia_evidence(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(root / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, timeout=10)
    paths = (
        "edits/example/transcript.json\n"
        "edits/example/models/upstream.py\n"
        "edits/example/evidence/README.md\n"
    )
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        input=paths,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    assert result.stdout == paths
