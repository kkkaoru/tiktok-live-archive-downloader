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

it("asks browsers to revalidate the app page on every load", async () => {
  const response = await call("/streamers/x", { headers: { Cookie: cookie } });
  expect(response.headers.get("Cache-Control")).toBe("no-cache");
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

// In-memory R2 with the subset the share routes use.
function memoryBucket(seed: Record<string, string>) {
  const store = new Map(Object.entries(seed));
  const object = (key: string) => {
    const body = store.get(key);
    return body === undefined
      ? null
      : {
          size: body.length,
          httpEtag: '"e"',
          body,
          json: async () => JSON.parse(body),
        };
  };
  return {
    store,
    get: async (key: string) => object(key),
    head: async (key: string) => object(key),
    put: async (key: string, body: string) => {
      store.set(key, body);
    },
    delete: async (key: string) => {
      store.delete(key);
    },
    list: async ({ prefix }: { prefix: string }) => ({
      objects: [...store.keys()].filter((key) => key.startsWith(prefix)).map((key) => ({ key })),
    }),
  };
}

const VIDEO = {
  id: "v1",
  streamer: "s",
  title: "t",
  broadcastStartJST: "2026-10-14 23:01:14 JST",
  duration: 10,
  movie: "v1.mp4",
  original: { movie: "v1-orig.mp4", localPath: "/Users/secret/v1.mp4" },
  localPath: "/Users/secret/v1-edit.mp4",
};

function shareEnvironment() {
  const MEDIA = memoryBucket({
    "catalog.json": JSON.stringify({ videos: [VIDEO, { ...VIDEO, id: "v2", movie: null }] }),
    "data/v1.json": JSON.stringify({ id: "v1", cues: [], summary: null }),
    "data/v2.json": JSON.stringify({ id: "v2", cues: [], summary: null }),
    "media/v1.mp4": "EDITED",
    "media-original/v1-orig.mp4": "ORIGINAL",
  });
  return {
    ...environment,
    MEDIA,
    TURNSTILE_SITE_KEY: "1x00000000000000000000AA",
    TURNSTILE_SECRET_KEY: "1x0000000000000000000000000000000AA",
  };
}

type ShareEnvironment = ReturnType<typeof shareEnvironment>;

function shareCall(env: ShareEnvironment, path: string, init: RequestInit = {}) {
  return worker.fetch(
    new Request(`https://example.test${path}`, { redirect: "manual", ...init }),
    env,
  );
}

function adminWrite(env: ShareEnvironment, method: string, body: object, as = cookie) {
  return shareCall(env, "/api/admin/shares", {
    method,
    headers: { Cookie: as, Origin: "https://example.test", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

function stubSiteverify(result: object) {
  vi.stubGlobal("fetch", async () => Response.json(result));
}

function solve(env: object, token: string, response = "ok", origin = "https://example.test") {
  return worker.fetch(
    new Request(`https://example.test/s/${token}/verify`, {
      method: "POST",
      headers: { Origin: origin },
      body: new URLSearchParams({ "cf-turnstile-response": response }),
    }),
    env,
  );
}

async function passChallenge(env: ShareEnvironment, token: string): Promise<string> {
  stubSiteverify({ success: true, hostname: "example.test" });
  const response = await solve(env, token);
  return (response.headers.get("Set-Cookie") ?? "").split(";")[0] ?? "";
}

async function createShare(env: ShareEnvironment, body: object): Promise<string> {
  const response = await adminWrite(env, "POST", { videoId: "v1", ...body });
  return (await response.json()).share.token;
}

it("issues a random 7-day share that serves only the shared video without login", async () => {
  const env = shareEnvironment();
  const token = await createShare(env, { edited: true });
  const headers = { Cookie: await passChallenge(env, token) };
  const catalog = await (await shareCall(env, `/s/${token}/api/catalog`, { headers })).json();
  const page = await shareCall(env, `/s/${token}`, { headers });
  const statuses = await Promise.all(
    [
      `/s/${token}/api/videos/v1.json`,
      `/s/${token}/media/v1.mp4`,
      `/s/${token}/media-original/v1-orig.mp4`,
      `/s/${token}/api/videos/v2.json`,
      "/api/catalog",
    ].map(async (path) => (await shareCall(env, path, { headers })).status),
  );
  expect([
    token.length,
    catalog.share.expiresAt - Math.floor(Date.now() / 1000),
    JSON.stringify(catalog).includes("localPath"),
    catalog.videos[0].original,
    page.headers.get("Referrer-Policy"),
    await page.text(),
    statuses,
  ]).toStrictEqual([
    43,
    7 * 86400,
    false,
    null,
    "no-referrer",
    "INDEX:/",
    [200, 200, 404, 404, 401],
  ]);
});

it("honors the chosen days, both kinds, and adds a download header", async () => {
  const env = shareEnvironment();
  const token = await createShare(env, { edited: true, original: true, days: 2 });
  const headers = { Cookie: await passChallenge(env, token) };
  const catalog = await (await shareCall(env, `/s/${token}/api/catalog`, { headers })).json();
  const download = await shareCall(env, `/s/${token}/media-original/v1-orig.mp4?download=1`, {
    headers,
  });
  expect([
    catalog.share.expiresAt - Math.floor(Date.now() / 1000),
    download.headers.get("Content-Disposition"),
    await download.text(),
  ]).toStrictEqual([2 * 86400, 'attachment; filename="v1-orig.mp4"', "ORIGINAL"]);
});

it("stops serving a share once it is turned off or expired", async () => {
  const env = shareEnvironment();
  const token = await createShare(env, { edited: true });
  const listed = await (
    await shareCall(env, "/api/admin/shares?video=v1", { headers: { Cookie: cookie } })
  ).json();
  await adminWrite(env, "DELETE", { token });
  const revoked = await shareCall(env, `/s/${token}`);
  const expiredToken = await createShare(env, { edited: true });
  const record = JSON.parse(env.MEDIA.store.get(`shares/${expiredToken}.json`) ?? "{}");
  env.MEDIA.store.set(`shares/${expiredToken}.json`, JSON.stringify({ ...record, expiresAt: 1 }));
  const expired = await shareCall(env, `/s/${expiredToken}/media/v1.mp4`);
  expect([
    listed.shares.length,
    revoked.status,
    expired.status,
    (await shareCall(env, "/s/nope")).status,
  ]).toStrictEqual([1, 404, 404, 404]);
});

it("rejects invalid share requests and non-admin or debug writers", async () => {
  const env = shareEnvironment();
  const member = await sessionCookie({
    uid: "2",
    u: "member",
    exp: Math.floor(Date.now() / 1000) + 600,
  });
  const minted = await shareCall(env, "/auth/debug", {
    method: "POST",
    headers: { "X-Debug-Token": TOKEN },
  });
  const debugCookie = (minted.headers.get("Set-Cookie") ?? "").split(";")[0] ?? "";
  const statuses = await Promise.all(
    [
      adminWrite(env, "POST", { videoId: "v1" }),
      adminWrite(env, "POST", { videoId: "v1", edited: true, days: 0 }),
      adminWrite(env, "POST", { videoId: "v1", edited: true, days: 1.5 }),
      adminWrite(env, "POST", { videoId: "v2", edited: true }),
      adminWrite(env, "POST", { videoId: "zz", edited: true }),
      adminWrite(env, "DELETE", { token: "../catalog" }),
      adminWrite(env, "PUT", {}),
      adminWrite(env, "POST", { videoId: "v1", edited: true }, debugCookie),
      shareCall(env, "/api/admin/shares", { headers: { Cookie: member } }),
    ].map(async (response) => (await response).status),
  );
  expect(statuses).toStrictEqual([400, 400, 400, 400, 400, 400, 405, 403, 401]);
});

it("revalidates transcripts every time and answers an unchanged one with 304", async () => {
  const env = shareEnvironment();
  const first = await shareCall(env, "/api/videos/v1.json", { headers: { Cookie: cookie } });
  const again = await shareCall(env, "/api/videos/v1.json", {
    headers: { Cookie: cookie, "If-None-Match": '"e"' },
  });
  expect([
    first.status,
    first.headers.get("Cache-Control"),
    first.headers.get("ETag"),
    again.status,
  ]).toStrictEqual([200, "private, no-cache", '"e"', 304]);
});

it("asks share visitors to pass Turnstile before serving anything", async () => {
  const env = shareEnvironment();
  const token = await createShare(env, { edited: true });
  const other = await createShare(env, { edited: true });
  const page = await shareCall(env, `/s/${token}`);
  const media = await shareCall(env, `/s/${token}/media/v1.mp4`);
  const passed = await solve(env, token);
  const foreign = await shareCall(env, `/s/${token}/media/v1.mp4`, {
    headers: { Cookie: await passChallenge(env, other) },
  });
  const html = await page.text();
  expect([
    page.status,
    html.includes('data-sitekey="1x00000000000000000000AA"'),
    page.headers.get("Referrer-Policy"),
    media.status,
    passed.status,
    passed.headers.get("Location"),
    (passed.headers.get("Set-Cookie") ?? "").includes(`Path=/s/${token};`),
    foreign.status,
  ]).toStrictEqual([200, true, "origin", 403, 303, `/s/${token}`, true, 403]);
});

it("keeps the challenge when Turnstile rejects or the request is not a form post", async () => {
  const env = shareEnvironment();
  const token = await createShare(env, { edited: true });
  stubSiteverify({ success: false });
  const rejected = await solve(env, token);
  stubSiteverify({ success: true, hostname: "elsewhere.test" });
  const otherHost = await solve(env, token);
  const empty = await solve(env, token, "");
  const crossSite = await solve(env, token, "ok", "https://evil.test");
  const get = await shareCall(env, `/s/${token}/verify`);
  vi.stubGlobal("fetch", async () => {
    throw new Error("network");
  });
  const offline = await solve(env, token);
  expect([
    rejected.status,
    (await rejected.text()).includes("確認できませんでした"),
    otherHost.status,
    empty.status,
    crossSite.status,
    get.status,
    offline.status,
  ]).toStrictEqual([403, true, 403, 403, 400, 400, 403]);
});

it("keeps shares closed while Turnstile keys are missing", async () => {
  const env = { ...shareEnvironment(), TURNSTILE_SECRET_KEY: "" };
  const token = await createShare(env, { edited: true });
  const page = await shareCall(env, `/s/${token}`);
  const verify = await solve(env, token);
  expect([
    page.status,
    (await page.text()).includes("現在利用できません"),
    verify.status,
  ]).toStrictEqual([503, true, 400]);
});

it("serves robots.txt that disallows every crawler without login", async () => {
  const response = await call("/robots.txt", {});
  expect([response.status, await response.text()]).toStrictEqual([
    200,
    "User-agent: *\nDisallow: /\n",
  ]);
});

it("gives crawlers preview tags and a public preview image, nothing else", async () => {
  const env = shareEnvironment();
  env.MEDIA.store.set("og/v1.jpg", "JPEG");
  const token = await createShare(env, { edited: true });
  const page = await shareCall(env, `/s/${token}`);
  const html = await page.text();
  const image = await shareCall(env, `/s/${token}/og.jpg`);
  expect([
    html.includes('<meta property="og:title" content="s の配信 2026-10-14 23:01">'),
    html.includes(`<meta property="og:image" content="https://example.test/s/${token}/og.jpg">`),
    html.includes('<meta name="twitter:card" content="summary_large_image">'),
    html.includes("JSTの配信アーカイブ（0分）"),
    image.status,
    image.headers.get("Content-Type"),
    await image.text(),
  ]).toStrictEqual([true, true, true, true, 200, "image/jpeg", "JPEG"]);
});

it("falls back to a text card and escapes names when no preview image exists", async () => {
  const env = shareEnvironment();
  const catalog = JSON.parse(env.MEDIA.store.get("catalog.json") ?? "{}");
  catalog.videos[0].streamer = '<b>"x"</b>';
  catalog.videos[0].duration = 3720;
  env.MEDIA.store.set("catalog.json", JSON.stringify(catalog));
  const token = await createShare(env, { edited: true });
  const html = await (await shareCall(env, `/s/${token}`)).text();
  const image = await shareCall(env, `/s/${token}/og.jpg`);
  expect([
    html.includes("&lt;b&gt;&quot;x&quot;&lt;/b&gt; の配信"),
    html.includes("（1時間2分）"),
    html.includes('<meta name="twitter:card" content="summary">'),
    image.status,
  ]).toStrictEqual([true, true, true, 404]);
});
