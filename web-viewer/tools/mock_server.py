"""Local UI check: serves public/index.html for client-routed paths and /api from data/."""
import http.server, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Handler(http.server.BaseHTTPRequestHandler):
    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path.startswith("/streamers"):
            return self.send(200, open(f"{ROOT}/public/index.html", "rb").read(), "text/html; charset=utf-8")
        if path == "/api/catalog":
            return self.send(200, open(f"{ROOT}/data/catalog.json", "rb").read(), "application/json")
        if path == "/api/me":
            return self.send(200, b'{"username":"kkk4oru","admin":true}', "application/json")
        m = re.match(r"/api/videos/([\w.-]+)\.json$", path)
        if m and os.path.exists(f"{ROOT}/data/{m.group(1)}.json"):
            return self.send(200, open(f"{ROOT}/data/{m.group(1)}.json", "rb").read(), "application/json")
        self.send(404, b"not found", "text/plain")

    def log_message(self, *args):
        pass


http.server.ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), Handler).serve_forever()
