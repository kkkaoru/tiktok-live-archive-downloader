"""Local command-line interface. No remote UI or automatic credential sharing."""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import httpx

from replay.acquire import DEFAULT_MEDIA_HOSTS, DownloadOptions, acquire_all, recording_rows
from replay.android_log import replay_candidates
from replay.core import (
    ReplayError,
    Scope,
    import_har,
    load_candidate,
    private_directory,
    save_candidate,
)
from replay.download import Downloader
from replay.jobs import download_once
from replay.session_refresh import refresh_session
from replay.targets import target_id
from replay.tiktok_api import Credentials, TikTokAPI
from replay.web_auth import login


class Arguments(argparse.Namespace):
    command: str
    store: Path
    host: list[str]
    har: Path
    candidate: str
    output: Path
    max_gib: float
    listen: str
    port: int
    api_host: list[str]
    replay_id: str | None
    workers: int
    log: Path
    session: Path
    login_timeout: int
    anchor_id: str | None
    target_user: str | None
    refresh_session: bool
    token_output: Path
    web_login: bool


def add_target_arguments(command: argparse.ArgumentParser) -> None:
    target = command.add_mutually_exclusive_group()
    target.add_argument("--anchor-id", help="Verified numeric creator ID")
    target.add_argument(
        "--user", dest="target_user", help="Username, @handle, or HTTPS TikTok profile URL"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Download only authorized, observed replay media.")
    result.add_argument("--store", type=Path, default=Path("private/candidates"))
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List candidates without displaying signed URLs")
    recordings = commands.add_parser(
        "recordings", help="Fetch a fresh recording list using TIKTOK_SESSIONID"
    )
    add_target_arguments(recordings)
    recordings.add_argument("--refresh-session", action="store_true")
    batch = commands.add_parser(
        "download-all", help="Download available recordings without Android"
    )
    batch.add_argument("output", type=Path)
    add_target_arguments(batch)
    batch.add_argument("--host", action="append", default=[])
    batch.add_argument("--workers", type=int, default=6)
    batch.add_argument("--max-gib", type=float, default=20)
    batch.add_argument("--refresh-session", action="store_true")
    refresh = commands.add_parser(
        "refresh-session", help="Reissue and verify a session on Mac; no browser by default"
    )
    refresh.add_argument(
        "--output", dest="token_output", type=Path, default=Path("private/tiktok-sessionid.secret")
    )
    refresh.add_argument("--login", dest="web_login", action="store_true")
    refresh.add_argument("--timeout", dest="login_timeout", type=int, default=600)
    web_login = commands.add_parser("web-login", help="Log in using an isolated Mac Chrome window")
    web_login.add_argument("--session", type=Path, default=Path("private/web-session.json"))
    web_login.add_argument("--timeout", type=int, default=600, dest="login_timeout")
    android = commands.add_parser(
        "import-android-log", help="Import observed room IDs and HLS URLs"
    )
    android.add_argument("log", type=Path)
    importer = commands.add_parser("import-har", help="Import successful media requests from HAR")
    importer.add_argument("har", type=Path)
    importer.add_argument("--host", action="append", required=True)
    download = commands.add_parser("download", help="Download a selected candidate")
    download.add_argument("candidate", help="16-character candidate ID")
    download.add_argument("output", type=Path)
    download.add_argument("--host", action="append", required=True)
    download.add_argument("--max-gib", type=float, default=20)
    download.add_argument("--replay-id", help="Stable room ID: skip completed or running replays")
    download.add_argument("--workers", type=int, default=6, help="Concurrent HLS downloads (1-16)")
    capture = commands.add_parser(
        "capture", help="Capture scoped requests or discover CONNECT hosts"
    )
    capture.add_argument("--host", action="append", default=[])
    capture.add_argument(
        "--api-host", action="append", default=[], help="Opt-in replay API URL inspection"
    )
    capture.add_argument("--listen", default="127.0.0.1")
    capture.add_argument("--port", type=int, default=8080)
    return result


def run_capture(args: Arguments) -> int:
    executable = shutil.which("mitmdump")
    if executable is None:
        raise ReplayError("Install capture support: uv sync --extra capture")
    private_directory(args.store)
    confdir = args.store.resolve().parent / "mitmproxy"
    private_directory(confdir)
    command = [
        executable,
        "-q",
        "-s",
        str(Path(__file__).with_name("addon.py")),
        "--listen-host",
        args.listen,
        "--listen-port",
        str(args.port),
        "--set",
        f"confdir={confdir}",
        "--set",
        f"replay_store={args.store.resolve()}",
        "--set",
        f"replay_hosts={','.join(args.host)}",
        "--set",
        f"replay_api_hosts={','.join(args.api_host)}",
        "--set",
        "connection_strategy=lazy",
    ]
    if args.host or args.api_host:
        pattern = (
            "^(?:"
            + "|".join(re.escape(host) for host in args.host + args.api_host)
            + r")(?::443)?$|^mitm\.it(?::80)?$"
        )
        command.extend(["--allow-hosts", pattern])
    else:
        command.extend(["--ignore-hosts", r"^(?!mitm\.it(?::80)?$).*"])
        print("Discovery only: CONNECT hostnames, no TLS decryption. Ctrl-C to stop.", flush=True)
    print("Use a trusted Wi-Fi network only. Disable the phone proxy after use.", flush=True)
    return subprocess.call(command)


def run_recordings(args: Arguments) -> int:
    credentials = Credentials.from_environment(os.environ)
    anchor_id = target_id(anchor_id=args.anchor_id, user=args.target_user, environ=os.environ)
    if args.refresh_session:
        credentials = refresh_session(
            output=args.store.parent / "tiktok-sessionid.secret", credentials=credentials
        ).account.credentials
    with httpx.Client(timeout=httpx.Timeout(30, connect=15), trust_env=False) as client:
        api = TikTokAPI(client=client, credentials=credentials)
        if args.command == "recordings":
            rows = recording_rows(api, anchor_id=anchor_id, jobs=args.store.parent / "jobs")
            print(json.dumps(rows, ensure_ascii=False, indent=2))
            return 0
        options = DownloadOptions(
            store=args.store,
            output=args.output,
            hosts=tuple(dict.fromkeys((*DEFAULT_MEDIA_HOSTS, *args.host))),
            workers=args.workers,
            max_bytes=int(args.max_gib * 1024**3),
        )
        summary = acquire_all(api, anchor_id=anchor_id, options=options)
        print(
            f"Saved: {summary.saved}; complete/busy: {summary.skipped}; "
            f"unavailable: {summary.unavailable}"
        )
        return 0


def run(args: Arguments) -> int:
    if args.command in {"recordings", "download-all"}:
        return run_recordings(args)
    if args.command == "refresh-session":
        credentials = (
            Credentials.from_environment(os.environ) if "TIKTOK_SESSIONID" in os.environ else None
        )
        result = refresh_session(
            output=args.token_output,
            credentials=credentials,
            web_login=args.web_login,
            timeout=args.login_timeout,
        )
        print(json.dumps(result.report(), indent=2))
        return 0
    if args.command == "web-login":
        login(session=args.session, timeout=args.login_timeout)
        print("Web session saved privately. Replay API access still requires verification.")
        return 0
    if args.command == "list":
        for path in sorted(args.store.glob("*.json")):
            print(load_candidate(path).label)
        return 0
    if args.command == "capture":
        return run_capture(args)
    if args.command == "import-android-log":
        with args.log.open(encoding="utf-8") as stream:
            records = replay_candidates(stream)
        for replay_id, candidate in records.items():
            save_candidate(args.store, candidate)
            print(f"{replay_id} {candidate.id}")
        return 0
    scope = Scope(tuple(host.lower().strip() for host in args.host))
    if args.command == "import-har":
        count = import_har(args.har, root=args.store, scope=scope)
        print(f"Imported {count} new media candidates.")
        return 0
    if re.fullmatch(r"[0-9a-f]{16}", args.candidate) is None:
        raise ReplayError("Invalid candidate ID; use replay list.")
    candidate = load_candidate(args.store / f"{args.candidate}.json")

    def fetch() -> None:
        with httpx.Client(timeout=httpx.Timeout(30, connect=15), trust_env=False) as client:
            Downloader(
                client=client,
                scope=scope,
                candidate=candidate,
                max_bytes=int(args.max_gib * 1024**3),
                workers=args.workers,
            ).download(args.output)

    if args.replay_id is not None:
        saved = download_once(
            root=args.store.parent / "jobs",
            replay_id=args.replay_id,
            output=args.output,
            download=fetch,
        )
        if not saved:
            print(f"Skipped replay {args.replay_id}: complete or already running.")
            return 0
    else:
        fetch()
    print(f"Saved: {args.output}")
    return 0


def main() -> int:
    try:
        return run(parser().parse_args(namespace=Arguments()))
    except ReplayError as error:
        print(f"Error: {error}")
    except httpx.HTTPError:
        print(
            "Network error. Check connectivity and recapture expired URLs; secrets were not logged."
        )
    except (OSError, ValueError, json.JSONDecodeError):
        print("Local file, configuration or data error. Check paths, permissions and input format.")
    except KeyboardInterrupt:
        print("Stopped. Disable the phone proxy if capture was active.")
        return 130
    return 1
