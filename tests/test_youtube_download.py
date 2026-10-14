import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "scripts/download-youtube.sh",
        root / "scripts/download-youtube.sh",
    )
    external = tmp_path / "external"
    external.mkdir(mode=0o700)
    binary = tmp_path / "bin"
    binary.mkdir()
    stat = binary / "stat"
    stat.write_text(
        '#!/bin/bash\nif [[ "$1" == -f && "$2" == "%d" && "$3" == *external* ]]; '
        'then echo 999; else /usr/bin/stat "$@"; fi\n',
        encoding="utf-8",
    )
    stat.chmod(0o755)
    node = binary / "node"
    node.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    node.chmod(0o755)
    ffmpeg_home = tmp_path / "ffmpeg-full"
    (ffmpeg_home / "bin").mkdir(parents=True)
    ffmpeg = ffmpeg_home / "bin/ffmpeg"
    ffmpeg.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    ffmpeg.chmod(0o755)
    brew = binary / "brew"
    brew.write_text('#!/bin/sh\nprintf "%s\\n" "$FFMPEG_HOME"\n', encoding="utf-8")
    brew.chmod(0o755)
    uv = binary / "uv"
    uv.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$TRACE"\n', encoding="utf-8")
    uv.chmod(0o755)
    return (
        root / "scripts/download-youtube.sh",
        external,
        {
            "PATH": f"{binary}:/usr/bin:/bin",
            "TRACE": str(tmp_path / "trace"),
            "FFMPEG_HOME": str(ffmpeg_home),
            "YOUTUBE_DOWNLOAD_DIR": str(external / "youtube"),
        },
    )


def test_best_streams_go_to_external_regular_files(
    setup: tuple[Path, Path, dict[str, str]],
) -> None:
    script, external, environ = setup
    result = subprocess.run(
        ["bash", str(script), "https://www.youtube.com/watch?v=lL4opb4iB3w"],
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert (script.parent.parent / "youtube-originals").is_symlink()
    assert (script.parent.parent / "youtube-originals").resolve() == external / "youtube"
    assert (external / "youtube").stat().st_mode & 0o777 == 0o700
    assert Path(environ["TRACE"]).read_text(encoding="utf-8") == (
        "run\n--locked\n--extra\nyoutube\npython\n-m\nreplay.youtube_delivery\n"
        "https://www.youtube.com/watch?v=lL4opb4iB3w\n"
        f"{external}/youtube\n{environ['FFMPEG_HOME']}/bin\n"
    )


def test_existing_data_is_not_replaced(setup: tuple[Path, Path, dict[str, str]]) -> None:
    script, _, environ = setup
    directory = script.parent.parent / "youtube-originals"
    directory.mkdir()
    (directory / "video.mp4").write_bytes(b"keep")
    result = subprocess.run(
        ["bash", str(script), "https://www.youtube.com/watch?v=lL4opb4iB3w"],
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert (directory / "video.mp4").read_bytes() == b"keep"
    assert not Path(environ["TRACE"]).exists()


def test_non_youtube_url_is_rejected(setup: tuple[Path, Path, dict[str, str]]) -> None:
    script, _, environ = setup
    result = subprocess.run(
        ["bash", str(script), "https://example.com/watch?v=lL4opb4iB3w"],
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert not (script.parent.parent / "youtube-originals").exists()
    assert not Path(environ["TRACE"]).exists()


def test_internal_or_unmounted_destination_is_rejected(
    setup: tuple[Path, Path, dict[str, str]], tmp_path: Path
) -> None:
    script, _, environ = setup
    environ = environ | {"YOUTUBE_DOWNLOAD_DIR": str(tmp_path / "internal")}
    result = subprocess.run(
        ["bash", str(script), "https://www.youtube.com/watch?v=lL4opb4iB3w"],
        env=environ,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert not Path(environ["TRACE"]).exists()
