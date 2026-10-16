"""Call one tool of the apple-pro-apps MCP server over stdio, without an Executor session.

The server speaks JSON-RPC lines on stdin/stdout. This client starts it, performs the MCP
handshake, sends one `tools/call` whose arguments come from a JSON file, and writes the whole
response to a new file. The server binary is pinned by sha256, the request is read once, an
existing output is never overwritten, and a tool error (`isError`) is a non-zero exit. A tool
that renders media runs for as long as the render takes, so the timeout is an argument.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import IO

from replay.core import ReplayError

PROTOCOL = "2024-11-05"
CLIENT = {"name": "replay-tab2", "version": "1"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_line(stream: IO[str]) -> dict[str, object]:
    line = stream.readline()
    if not line:
        raise ReplayError("The MCP server closed its output")
    loaded: object = json.loads(line)
    if not isinstance(loaded, dict):
        raise ReplayError("The MCP server sent a message that is not an object")
    return loaded


def call(
    server: Path, tool: str, arguments: dict[str, object], *, timeout: int
) -> dict[str, object]:
    """Start the server, hand-shake, call `tool` once and return the JSON-RPC result."""
    with subprocess.Popen(
        [str(server), "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ) as process:
        if process.stdin is None or process.stdout is None:
            raise ReplayError("The MCP server gave no pipes")
        stdin, stdout = process.stdin, process.stdout

        def send(message: dict[str, object]) -> None:
            stdin.write(json.dumps(message) + "\n")
            stdin.flush()

        try:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": PROTOCOL,
                        "capabilities": {},
                        "clientInfo": CLIENT,
                    },
                }
            )
            if "result" not in read_line(stdout):
                raise ReplayError("The MCP server refused to initialize")
            send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": tool, "arguments": arguments},
                }
            )
            response = read_line(stdout)
        finally:
            process.terminate()
            process.wait(timeout=timeout)
    if response.get("id") != 2:
        raise ReplayError("The MCP server answered a different request")
    return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool")
    parser.add_argument("arguments", type=Path, help="JSON file with the tool arguments")
    parser.add_argument("output", type=Path, help="new file for the JSON-RPC response")
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--server-sha256", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ReplayError("Refusing to overwrite the output")
    if not args.server.is_file() or sha256(args.server) != args.server_sha256:
        raise ReplayError("The MCP server binary differs from its pinned sha256")
    loaded: object = json.loads(args.arguments.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ReplayError("The tool arguments must be a JSON object")
    response = call(args.server, args.tool, loaded, timeout=args.timeout)
    with args.output.open("x", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(response, stream, ensure_ascii=False, indent=1)
    result = response.get("result")
    failed = "error" in response or (isinstance(result, dict) and result.get("isError") is True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
