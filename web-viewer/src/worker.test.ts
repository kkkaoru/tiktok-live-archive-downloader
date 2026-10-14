import { webcrypto } from "node:crypto";
import { beforeAll, expect, it, vi } from "vitest";
import worker from "./worker.js";

const SECRET = "local-test-secret-not-the-real-one";
const TOKEN = "d".repeat(40);
const encoder = new TextEncoder();
const base64url = (bytes: ArrayBuffer | Uint8Array) =>
  Buffer.from(new Uint8Array(bytes)).toString("base64url");

async function sessionCookie(payload: object): Promise<string> {
  const key = await webcrypto.subtle.importKey(
    "raw",
    encoder.encode(SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const body = base64url(encoder.encode(JSON.stringify(payload)));
  const mac = base64url(await webcrypto.subtle.sign("HMAC", key, encoder.encode(body)));
  return `tv_session=${body}.${mac}`;
}

const environment = {
  X_CLIENT_ID: "id",
  SESSION_SECRET: SECRET,
  X_ADMIN_USERNAMES: "tester",
  DEBUG_TOKEN: TOKEN,
  DEBUG_TOKEN_EXPIRES: String(Math.floor(Date.now() / 1000) + 600),
  ASSETS: {
    fetch: async (request: Request) => new Response(`INDEX:${new URL(request.url).pathname}`),
  },
  ALLOW_KV: { get: async () => ({ users: [] }), put: async () => undefined },
  MEDIA: { get: async () => null },
};

function call(path: string, init: RequestInit) {
  return worker.fetch(
    new Request(`https://example.test${path}`, { redirect: "manual", ...init }),
    environment,
  );
}

let cookie = "";

beforeAll(async () => {
  vi.stubGlobal("caches", {
    default: { match: async () => undefined, put: async () => undefined },
  });
  cookie = await sessionCookie({ uid: "1", u: "tester", exp: Math.floor(Date.now() / 1000) + 600 });
});

it("sends visitors without a session to the login page", async () => {
  const response = await call("/", { headers: { Accept: "text/html" } });
  expect([response.status, response.headers.get("Location")]).toStrictEqual([302, "/login"]);
});

it("redirects the top page to the streamer list", async () => {
  const response = await call("/", { headers: { Cookie: cookie } });
  expect([response.status, response.headers.get("Location")]).toStrictEqual([302, "/streamers"]);
});

it("serves the app for streamer, list and video paths", async () => {
  const paths = ["/streamers", "/streamers/x", "/streamers/x/videos/y"];
  const bodies = await Promise.all(
    paths.map(async (path) => (await call(path, { headers: { Cookie: cookie } })).text()),
  );
  expect(bodies).toStrictEqual(["INDEX:/", "INDEX:/", "INDEX:/"]);
});

it("redirects the old stremers spelling", async () => {
  const response = await call("/stremers/x?a=1", { headers: { Cookie: cookie } });
  expect(response.headers.get("Location")).toBe("/streamers/x?a=1");
});

it("returns the session user on /api/me", async () => {
  const response = await call("/api/me", { headers: { Cookie: cookie } });
  expect(await response.json()).toStrictEqual({ username: "tester", admin: true });
});

it("hides the debug route for a wrong token", async () => {
  const response = await call("/auth/debug", {
    method: "POST",
    headers: { "X-Debug-Token": "x".repeat(40) },
  });
  expect(response.status).toBe(404);
});

it("mints a read-only debug session for the right token", async () => {
  const minted = await call("/auth/debug", { method: "POST", headers: { "X-Debug-Token": TOKEN } });
  const debugCookie = (minted.headers.get("Set-Cookie") ?? "").split(";")[0] ?? "";
  const read = await call("/api/me", { headers: { Cookie: debugCookie } });
  const write = await call("/api/admin/users", {
    method: "POST",
    headers: {
      Cookie: debugCookie,
      Origin: "https://example.test",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ username: "x" }),
  });
  expect([minted.status, read.status, write.status]).toStrictEqual([204, 200, 403]);
});

it("disables the debug route once the token expires", async () => {
  const expired = {
    ...environment,
    DEBUG_TOKEN_EXPIRES: String(Math.floor(Date.now() / 1000) - 1),
  };
  const response = await worker.fetch(
    new Request("https://example.test/auth/debug", {
      method: "POST",
      headers: { "X-Debug-Token": TOKEN },
    }),
    expired,
  );
  expect(response.status).toBe(404);
});
