"""mitmproxy addon. Stream bodies and store only scoped media request metadata."""

from pathlib import Path
from urllib.parse import urlsplit

from mitmproxy import ctx, http, tls
from mitmproxy.addonmanager import Loader

from replay.api import api_stream
from replay.core import Scope, candidate_from_request, save_candidate


class ReplayCapture:
    def load(self, loader: Loader) -> None:
        loader.add_option("replay_hosts", str, "", "Comma-separated exact media host allowlist")
        loader.add_option("replay_store", str, "private/candidates", "Private candidate directory")
        loader.add_option("replay_api_hosts", str, "", "Exact hosts for replay API URL inspection")

    def http_connect(self, flow: http.HTTPFlow) -> None:
        if not str(ctx.options.replay_hosts) and not str(ctx.options.replay_api_hosts):
            print(f"CONNECT host: {flow.request.host}", flush=True)

    def requestheaders(self, flow: http.HTTPFlow) -> None:
        if flow.request.host in str(ctx.options.replay_api_hosts).split(","):
            path = urlsplit(flow.request.pretty_url).path.lower()
            if any(word in path for word in ("replay", "record", "notice")):
                flow.request.headers["accept-encoding"] = "identity"

    def tls_failed_client(self, data: tls.TlsData) -> None:
        print(f"Client TLS failed: {data.conn.sni}; check certificate trust.", flush=True)

    def responseheaders(self, flow: http.HTTPFlow) -> None:
        response = flow.response
        if response is None:
            return
        response.stream = True
        api_hosts = str(ctx.options.replay_api_hosts).split(",")
        if flow.request.host in api_hosts:
            path = urlsplit(flow.request.pretty_url).path
            print(f"API endpoint: {path}", flush=True)
            if (
                response.status_code == 200
                and any(word in path.lower() for word in ("replay", "record", "notice"))
                and "json" in response.headers.get("content-type", "")
            ):
                response.stream = api_stream(
                    root=Path(str(ctx.options.replay_store)),
                    encoding=response.headers.get("content-encoding", ""),
                    report=report,
                )
        hosts = str(ctx.options.replay_hosts)
        scope = Scope(tuple(host.strip().lower() for host in hosts.split(",") if host.strip()))
        candidate = candidate_from_request(
            url=flow.request.pretty_url,
            headers=dict(flow.request.headers),
            content_type=response.headers.get("content-type", ""),
            status=response.status_code,
            method=flow.request.method,
            scope=scope,
        )
        if candidate is not None and save_candidate(Path(str(ctx.options.replay_store)), candidate):
            print(f"Replay candidate: {candidate.label}", flush=True)


def report(message: str) -> None:
    print(message, flush=True)


addons = [ReplayCapture()]
