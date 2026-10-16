"""The stdio MCP client: handshake, one call, error mapping, pinned server, no overwrite."""

import json
import stat
import sys
from pathlib import Path

import pytest

from replay import apple_pro_apps_call as client
from replay.core import ReplayError

SERVER = """#!{python}
import json, sys
MODE = {mode!r}


def reply(identifier, key, value):
    print(json.dumps({{"jsonrpc": "2.0", "id": identifier, key: value}}), flush=True)


for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        reply(message["id"], "result", {{"capabilities": {{}}}})
    elif method == "tools/call":
        if MODE == "wrong-id":
            reply(99, "result", {{}})
        elif MODE == "tool-error":
            reply(message["id"], "result", {{"isError": True, "content": []}})
        elif MODE == "rpc-error":
            reply(message["id"], "error", {{"code": -1, "message": "no"}})
        else:
            text = json.dumps(message["params"])
            reply(message["id"], "result", {{"content": [{{"type": "text", "text": text}}]}})
"""


def make_server(tmp_path: Path, mode: str = "ok") -> Path:
    path = tmp_path / f"server-{mode}"
    path.write_text(SERVER.format(python=sys.executable, mode=mode), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def arguments_file(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "arguments.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_a_call_hands_the_tool_name_and_arguments_to_the_server(tmp_path: Path) -> None:
    server = make_server(tmp_path)
    response = client.call(server, "media_edit", {"a": 1}, timeout=30)
    result = response["result"]
    assert isinstance(result, dict)
    echoed = json.loads(result["content"][0]["text"])
    assert echoed == {"name": "media_edit", "arguments": {"a": 1}}


def test_main_writes_the_response_and_returns_zero(tmp_path: Path) -> None:
    server = make_server(tmp_path)
    output = tmp_path / "response.json"
    code = client.main(
        ["media_edit_plan", str(arguments_file(tmp_path, {"b": 2})), str(output),
         "--server", str(server), "--server-sha256", client.sha256(server), "--timeout", "30"]
    )  # fmt: skip
    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["id"] == 2


def test_a_tool_error_or_a_rpc_error_is_a_non_zero_exit(tmp_path: Path) -> None:
    for mode in ("tool-error", "rpc-error"):
        server = make_server(tmp_path, mode)
        output = tmp_path / f"{mode}.json"
        code = client.main(
            ["t", str(arguments_file(tmp_path, {})), str(output),
             "--server", str(server), "--server-sha256", client.sha256(server)]
        )  # fmt: skip
        assert code == 1


def test_an_answer_to_another_request_is_refused(tmp_path: Path) -> None:
    server = make_server(tmp_path, "wrong-id")
    with pytest.raises(ReplayError, match="different request"):
        client.call(server, "t", {}, timeout=30)


def test_an_unpinned_server_an_existing_output_and_bad_arguments_are_refused(
    tmp_path: Path,
) -> None:
    server = make_server(tmp_path)
    arguments = arguments_file(tmp_path, {})
    with pytest.raises(ReplayError, match="differs from its pinned sha256"):
        client.main(["t", str(arguments), str(tmp_path / "a.json"), "--server", str(server),
                     "--server-sha256", "0" * 64])  # fmt: skip
    existing = tmp_path / "taken.json"
    existing.write_text("x", encoding="utf-8")
    with pytest.raises(ReplayError, match="Refusing to overwrite"):
        client.main(["t", str(arguments), str(existing), "--server", str(server),
                     "--server-sha256", client.sha256(server)])  # fmt: skip
    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    with pytest.raises(ReplayError, match="must be a JSON object"):
        client.main(["t", str(listed), str(tmp_path / "b.json"), "--server", str(server),
                     "--server-sha256", client.sha256(server)])  # fmt: skip


def test_a_server_that_closes_its_output_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "silent"
    path.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(0)\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    with pytest.raises(ReplayError, match="closed its output"):
        client.call(path, "t", {}, timeout=30)
