from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest

from replay import cli
from replay.acquire import DownloadSummary
from replay.core import Candidate, save_candidate
from replay.session_refresh import Account, RefreshResult
from replay.tiktok_api import Credentials


def test_refresh_command_does_not_print_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TIKTOK_SESSIONID", "a" * 32)
    operation = Mock(
        return_value=RefreshResult(Account("42", Credentials("b" * 32), 1735693200), True, False)
    )
    monkeypatch.setattr(cli, "refresh_session", operation)
    monkeypatch.setattr("sys.argv", ["replay", "refresh-session"])
    assert cli.main() == 0
    text = capsys.readouterr().out
    assert "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" not in text
    assert '"status": "rotated"' in text
    assert operation.call_args.kwargs["web_login"] is False


def test_refresh_login_without_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TIKTOK_SESSIONID", raising=False)
    operation = Mock(
        return_value=RefreshResult(
            Account("42", Credentials("a" * 32), 1735693200), False, None, True
        )
    )
    monkeypatch.setattr(cli, "refresh_session", operation)
    monkeypatch.setattr("sys.argv", ["replay", "refresh-session", "--login"])
    assert cli.main() == 0
    assert operation.call_args.kwargs["credentials"] is None
    assert operation.call_args.kwargs["web_login"] is True


def test_recordings_uses_refreshed_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIKTOK_SESSIONID", "a" * 32)
    operation = Mock(
        return_value=RefreshResult(Account("42", Credentials("b" * 32), 1735693200), True, None)
    )
    rows = Mock(return_value=[])
    monkeypatch.setattr(cli, "refresh_session", operation)
    monkeypatch.setattr(cli, "recording_rows", rows)
    monkeypatch.setattr(
        "sys.argv", ["replay", "recordings", "--anchor-id", "42", "--refresh-session"]
    )
    assert cli.main() == 0
    assert rows.call_args.args[0].credentials.session_id == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def test_recordings_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TIKTOK_SESSIONID", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    rows = Mock(return_value=[])
    monkeypatch.setattr("replay.cli.recording_rows", rows)
    monkeypatch.setattr("sys.argv", ["replay", "recordings", "--anchor-id", "2"])
    assert cli.main() == 0
    assert capsys.readouterr().out == "[]\n"
    assert rows.call_args.kwargs["anchor_id"] == "2"


def test_download_all_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TIKTOK_SESSIONID", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    acquire = Mock(return_value=DownloadSummary(saved=1, skipped=5))
    monkeypatch.setattr("replay.cli.acquire_all", acquire)
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "download-all",
            "downloads",
            "--anchor-id",
            "2",
            "--workers",
            "3",
            "--host",
            "extra.test",
        ],
    )
    assert cli.main() == 0
    assert capsys.readouterr().out == "Saved: 1; complete/busy: 5; unavailable: 0\n"
    assert acquire.call_args.kwargs["options"].workers == 3
    assert acquire.call_args.kwargs["options"].hosts == (
        "v16m.tiktokcdn.com",
        "sf16-videoarch-live-tos-sign.tiktokcdn.com",
        "extra.test",
    )


def test_missing_session(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("TIKTOK_SESSIONID", raising=False)
    monkeypatch.setattr("sys.argv", ["replay", "recordings", "--anchor-id", "2"])
    assert cli.main() == 1
    assert "TIKTOK_SESSIONID" in capsys.readouterr().out


def test_web_login(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    authenticate = Mock()
    monkeypatch.setattr("replay.cli.login", authenticate)
    monkeypatch.setattr("sys.argv", ["replay", "web-login", "--timeout", "120"])
    assert cli.main() == 0
    authenticate.assert_called_once_with(session=Path("private/web-session.json"), timeout=120)
    assert "saved privately" in capsys.readouterr().out


def test_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    save_candidate(tmp_path, Candidate("https://cdn.test/a?secret=1", {}, "video"))
    monkeypatch.setattr("sys.argv", ["replay", "--store", str(tmp_path), "list"])
    assert cli.main() == 0
    assert "secret" not in capsys.readouterr().out


def test_import_android_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "capture.log"
    log.write_text(
        '{"payload":{"type":"url","url":"https://webcast.tiktokv.com/webcast/room/replay/info/?room_ids=123"}}\n'
        '{"payload":{"type":"url","url":"https://v16m.tiktokcdn.com/a.m3u8?secret=1"}}\n',
        encoding="utf-8",
    )
    store = tmp_path / "candidates"
    monkeypatch.setattr(
        "sys.argv", ["replay", "--store", str(store), "import-android-log", str(log)]
    )
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert output.startswith("123 ")
    assert "secret" not in output
    assert len(list(store.glob("*.json"))) == 1


def test_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    har = tmp_path / "empty.har"
    har.write_text('{"log":{"entries":[]}}', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["replay", "import-har", str(har), "--host", "cdn.test"])
    assert cli.main() == 0
    assert capsys.readouterr().out == "Imported 0 new media candidates.\n"


def test_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    candidate = Candidate("https://cdn.test/a", {}, "video")
    save_candidate(tmp_path, candidate)
    download = Mock()
    monkeypatch.setattr("replay.cli.Downloader.download", download)
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--store",
            str(tmp_path),
            "download",
            candidate.id,
            str(tmp_path / "out"),
            "--host",
            "cdn.test",
        ],
    )
    assert cli.main() == 0
    download.assert_called_once_with(tmp_path / "out")
    assert "Saved:" in capsys.readouterr().out


def test_replay_id_deduplication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "candidates"
    candidate = Candidate("https://cdn.test/a", {}, "video")
    save_candidate(store, candidate)
    output = tmp_path / "out.mp4"
    download = Mock(side_effect=lambda path: path.write_bytes(b"video"))
    monkeypatch.setattr("replay.cli.Downloader.download", download)
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--store",
            str(store),
            "download",
            candidate.id,
            str(output),
            "--host",
            "cdn.test",
            "--replay-id",
            "123",
        ],
    )
    assert cli.main() == 0
    assert cli.main() == 0
    assert download.call_count == 1
    assert "Skipped replay 123: complete or already running." in capsys.readouterr().out


@pytest.mark.parametrize("candidate", ["../../x", "aaaaaaaaaaaaaaaa"])
def test_bad_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate: str) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--store",
            str(tmp_path),
            "download",
            candidate,
            str(tmp_path / "out"),
            "--host",
            "cdn.test",
        ],
    )
    assert cli.main() == 1


@pytest.mark.parametrize(
    ("failure", "code", "message"),
    [
        (httpx.ConnectError("SENSITIVE_TOKEN"), 1, "Network error"),
        (ValueError("secret"), 1, "Local file"),
        (KeyboardInterrupt(), 130, "Stopped"),
    ],
)
def test_main_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    code: int,
    message: str,
) -> None:
    monkeypatch.setattr("sys.argv", ["replay", "list"])
    monkeypatch.setattr(cli, "run", Mock(side_effect=failure))
    assert cli.main() == code
    output = capsys.readouterr().out
    assert message in output
    assert "SENSITIVE_TOKEN" not in output


@pytest.mark.parametrize("hosts", [[], ["cdn.test"]])
def test_capture_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hosts: list[str]) -> None:
    call = Mock(return_value=0)
    monkeypatch.setattr("replay.cli.shutil.which", lambda name: "/bin/mitmdump")
    monkeypatch.setattr("replay.cli.subprocess.call", call)
    argv = ["replay", "--store", str(tmp_path / "store"), "capture"]
    if hosts:
        argv.extend(["--host", hosts[0]])
    monkeypatch.setattr("sys.argv", argv)
    assert cli.main() == 0
    command = call.call_args.args[0]
    assert command[:2] == ["/bin/mitmdump", "-q"]
    assert ("--allow-hosts" in command) == bool(hosts)
    assert ("--ignore-hosts" in command) != bool(hosts)
    assert "mitm" in command[-1]
    assert (tmp_path / "mitmproxy").stat().st_mode & 0o777 == 0o700


def test_missing_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("replay.cli.shutil.which", lambda name: None)
    monkeypatch.setattr("sys.argv", ["replay", "capture"])
    assert cli.main() == 1
