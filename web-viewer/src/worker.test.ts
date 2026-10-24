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
  const object = (key: string, range?: { offset: number; length: number }) => {
    const body = store.get(key);
    return body === undefined
      ? null
      : {
          size: body.length,
          httpEtag: '"e"',
          body,
          json: async () => JSON.parse(body),
          arrayBuffer: async () =>
            new TextEncoder().encode(
              range ? body.slice(range.offset, range.offset + range.length) : body,
            ).buffer,
        };
  };
  return {
    store,
    get: async (key: string, options?: { range?: { offset: number; length: number } }) =>
      object(key, options?.range),
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

function shareCall(env: object, path: string, init: RequestInit = {}) {
  return worker.fetch(
    new Request(`https://example.test${path}`, { redirect: "manual", ...init }),
    env,
  );
}

function adminWrite(env: object, method: string, body: object, as = cookie) {
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

async function passChallenge(env: object, token: string): Promise<string> {
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

const CLIPS_TOKEN = "c".repeat(40);
const CLIP_RECORD = {
  version: 1,
  videoId: "v1",
  movieKey: "a".repeat(32),
  inputHash: "h",
  models: { score: "s", summary: "m" },
  createdAt: "t",
  clips: [],
};

function clipEnvironment() {
  const sent: unknown[] = [];
  const env = {
    ...shareEnvironment(),
    CLIPS_API_TOKEN: CLIPS_TOKEN,
    CLIPS_QUEUE: { send: async (body: unknown) => sent.push(body) },
  };
  return { env, sent };
}

it("never queues clips that are missing and reads them once stored", async () => {
  const { env, sent } = clipEnvironment();
  const get = () => shareCall(env, "/api/clips/v1.json", { headers: { Cookie: cookie } });
  const first = await (await get()).json();
  env.MEDIA.store.set("clips/videos/v1.json", JSON.stringify({ movieKey: "a".repeat(32) }));
  env.MEDIA.store.set(`clips/${"a".repeat(32)}.json`, JSON.stringify(CLIP_RECORD));
  const second = await (await get()).json();
  const unknown = await shareCall(env, "/api/clips/v9.json", { headers: { Cookie: cookie } });
  expect([first, sent, second.status, unknown.status]).toStrictEqual([
    { status: "missing" },
    [],
    "ready",
    404,
  ]);
});

it("stores a record from the bearer under the published movie's key", async () => {
  const { env, sent } = clipEnvironment();
  env.MEDIA.store.set("media/v1.mp4", "abc");
  const put = (body: unknown, headers: Record<string, string>) =>
    shareCall(env, "/api/clips/v1.json", { method: "PUT", headers, body: JSON.stringify(body) });
  const bearer = { Authorization: `Bearer ${CLIPS_TOKEN}` };
  const stored = await put({ ...CLIP_RECORD, movieKey: "local" }, bearer);
  const invalid = await put({ videoId: "v1" }, bearer);
  const cookieUser = await put(CLIP_RECORD, { Cookie: cookie, Origin: "https://example.test" });
  expect([
    await stored.json(),
    invalid.status,
    cookieUser.status,
    JSON.parse(String(env.MEDIA.store.get("clips/2e8e504c5e6b886af70f8c03cbfc31ac.json"))).movieKey,
    env.MEDIA.store.get("clips/videos/v1.json"),
    sent,
  ]).toStrictEqual([
    { outcome: "written" },
    400,
    405,
    "2e8e504c5e6b886af70f8c03cbfc31ac",
    '{"movieKey":"2e8e504c5e6b886af70f8c03cbfc31ac"}',
    [],
  ]);
});

it("reloads clips only for same-origin admins or the bearer token", async () => {
  const { env, sent } = clipEnvironment();
  const reload = (headers: Record<string, string>) =>
    shareCall(env, "/api/clips/v1/reload", { method: "POST", headers });
  const crossSite = await reload({ Cookie: cookie });
  const admin = await reload({ Cookie: cookie, Origin: "https://example.test" });
  const bearer = await reload({ Authorization: `Bearer ${CLIPS_TOKEN}` });
  const wrongBearer = await reload({ Authorization: "Bearer wrong" });
  const get = await shareCall(env, "/api/clips/v1/reload", { headers: { Cookie: cookie } });
  expect([
    crossSite.status,
    admin.status,
    bearer.status,
    wrongBearer.status,
    get.status,
    sent,
  ]).toStrictEqual([400, 200, 200, 401, 405, [{ videoId: "v1", force: true }]]);
});

it("lets the bearer read clips with its own movie key", async () => {
  const { env } = clipEnvironment();
  env.MEDIA.store.set(`clips/${"a".repeat(32)}.json`, JSON.stringify(CLIP_RECORD));
  const headers = { Authorization: `Bearer ${CLIPS_TOKEN}` };
  const ready = await shareCall(env, `/api/clips/v1.json?key=${"a".repeat(32)}`, { headers });
  const invalid = await shareCall(env, "/api/clips/v1.json?key=x", { headers });
  expect([(await ready.json()).status, invalid.status]).toStrictEqual(["ready", 400]);
});

it("serves a shared video's clips without queueing work", async () => {
  const { env, sent } = clipEnvironment();
  const token = await createShare(env, { edited: true });
  const pass = await passChallenge(env, token);
  const response = await shareCall(env, `/s/${token}/api/clips/v1.json`, {
    headers: { Cookie: pass },
  });
  expect([await response.json(), sent]).toStrictEqual([{ status: "missing" }, []]);
});

it("acks processed queue messages and retries failures", async () => {
  const { env } = clipEnvironment();
  env.MEDIA.store.set("data/bad.json", "{}");
  const log: string[] = [];
  const message = (body: unknown) => ({
    body,
    ack: () => log.push("ack"),
    retry: () => log.push("retry"),
  });
  vi.spyOn(console, "log").mockImplementation(() => undefined);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  await worker.queue(
    {
      messages: [
        message({ object: { key: "data/none.json" } }),
        message({ videoId: "bad" }),
        message("x"),
      ],
    },
    env,
  );
  expect(log).toStrictEqual(["ack", "retry", "ack"]);
});

const KASUMI = "UCEmlNV5-edGhNWYDMeGDgWw";
const MILK = "UCCaEb7R2MKNapKQ6WWN80Ig";
const CHANNELS_KEY = "config/youtube-channels.json";
const CACHE_URL = "https://channels.internal/v1";
let clock = 2_000_000_000_000;

const record = (revision: number, label: string) =>
  JSON.stringify({ version: 1, revision, channels: { [KASUMI]: { label } } });

// Fresh KV and Cache API fakes, with the clock moved past every isolate-memory entry.
async function channelTiers() {
  clock += 3_600_000;
  vi.spyOn(Date, "now").mockImplementation(() => clock);
  const edge = new Map<string, string>();
  vi.stubGlobal("caches", {
    default: {
      match: async (key: string) => {
        const body = edge.get(key);
        return body === undefined ? undefined : new Response(body);
      },
      put: async (key: string, response: Response) => {
        edge.set(key, await response.text());
      },
    },
  });
  const kv = new Map<string, string>();
  const reads: string[] = [];
  const ALLOW_KV = {
    get: async (key: string): Promise<unknown> => {
      reads.push(`kv ${key}`);
      const body = kv.get(key);
      return body === undefined ? null : JSON.parse(body);
    },
    put: async (key: string, body: string) => {
      kv.set(key, body);
    },
  };
  const session = await sessionCookie({ uid: "1", u: "tester", exp: clock / 1000 + 600 });
  const base = clipEnvironment().env;
  const MEDIA = {
    ...base.MEDIA,
    get: async (key: string) => {
      if (key === CHANNELS_KEY) reads.push("r2");
      return base.MEDIA.get(key);
    },
  };
  const env = { ...base, ALLOW_KV, MEDIA };
  const read = async () => {
    const response = await shareCall(env, "/api/channels.json", { headers: { Cookie: session } });
    return { etag: response.headers.get("ETag"), body: await response.json() };
  };
  return { env, edge, kv, reads, read, session, store: base.MEDIA.store };
}

function channelsCall(env: object, method: string, body: object, auth: string) {
  return worker.fetch(
    new Request("https://example.test/api/channels.json", {
      method,
      headers: { Authorization: auth, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
    env,
  );
}

it("fills isolate memory, the Cache API and KV from R2 on a cold read", async () => {
  const tiers = await channelTiers();
  tiers.store.set(CHANNELS_KEY, record(3, "熊童子かすみ"));
  const first = await tiers.read();
  const second = await tiers.read();
  expect([first, second.etag, tiers.reads]).toStrictEqual([
    {
      etag: '"r3"',
      body: { version: 1, revision: 3, channels: { [KASUMI]: { label: "熊童子かすみ" } } },
    },
    '"r3"',
    ["kv youtube-channels", "r2"],
  ]);
  expect([tiers.kv.get("youtube-channels"), tiers.edge.get(CACHE_URL)]).toStrictEqual([
    record(3, "熊童子かすみ"),
    record(3, "熊童子かすみ"),
  ]);
});

it("reads the Cache API, then KV, once isolate memory expires, refilling the faster tiers", async () => {
  const tiers = await channelTiers();
  tiers.edge.set(CACHE_URL, record(5, "edge"));
  tiers.kv.set("youtube-channels", record(4, "kv"));
  tiers.store.set(CHANNELS_KEY, record(4, "r2"));
  const edge = await tiers.read();
  clock += 29_000;
  const memory = await tiers.read();
  clock += 2_000;
  tiers.edge.delete(CACHE_URL);
  const kv = await tiers.read();
  expect([edge.etag, memory.etag, kv.etag, tiers.reads]).toStrictEqual([
    '"r5"',
    '"r5"',
    '"r4"',
    ["kv youtube-channels"],
  ]);
  expect(tiers.edge.get(CACHE_URL)).toBe(record(4, "kv"));
});

it("treats a malformed tier as a miss and an absent R2 record as no labels", async () => {
  const tiers = await channelTiers();
  tiers.edge.set(CACHE_URL, '{"users":[]}');
  tiers.kv.set("youtube-channels", '{"users":[]}');
  const empty = await tiers.read();
  expect([empty.body, tiers.reads]).toStrictEqual([
    { version: 1, revision: 0, channels: {} },
    ["kv youtube-channels", "r2"],
  ]);
});

it("writes R2, KV, the Cache API and isolate memory, starting from R2 not a stale tier", async () => {
  const tiers = await channelTiers();
  tiers.store.set(CHANNELS_KEY, record(7, "熊童子かすみ"));
  tiers.edge.set(CACHE_URL, record(1, "stale"));
  await tiers.read();
  const bearer = `Bearer ${CLIPS_TOKEN}`;
  const set = await channelsCall(
    tiers.env,
    "PUT",
    { channelId: MILK, label: " かしすみるく " },
    bearer,
  );
  const expected = JSON.stringify({
    version: 1,
    revision: 8,
    channels: { [KASUMI]: { label: "熊童子かすみ" }, [MILK]: { label: "かしすみるく" } },
  });
  const afterSet = await tiers.read();
  const removed = await channelsCall(tiers.env, "DELETE", { channelId: KASUMI }, bearer);
  const afterRemove = await tiers.read();
  expect([set.headers.get("ETag"), afterSet.etag, tiers.store.get(CHANNELS_KEY)]).toStrictEqual([
    '"r8"',
    '"r8"',
    '{"version":1,"revision":9,"channels":{"UCCaEb7R2MKNapKQ6WWN80Ig":{"label":"かしすみるく"}}}',
  ]);
  expect([tiers.kv.get("youtube-channels"), tiers.edge.get(CACHE_URL)]).toStrictEqual([
    tiers.store.get(CHANNELS_KEY),
    tiers.store.get(CHANNELS_KEY),
  ]);
  expect([
    JSON.stringify(afterSet.body),
    (await removed.json()).revision,
    afterRemove.body,
  ]).toStrictEqual([
    expected,
    9,
    { version: 1, revision: 9, channels: { [MILK]: { label: "かしすみるく" } } },
  ]);
});

it("rejects channel writes without the bearer and invalid channel IDs or labels", async () => {
  const tiers = await channelTiers();
  const bearer = `Bearer ${CLIPS_TOKEN}`;
  const viewer = await shareCall(tiers.env, "/api/channels.json", {
    method: "PUT",
    headers: { Cookie: tiers.session, "Content-Type": "application/json" },
    body: JSON.stringify({ channelId: KASUMI, label: "x" }),
  });
  const anonymous = await shareCall(tiers.env, "/api/channels.json", {});
  const badId = await channelsCall(tiers.env, "PUT", { channelId: "@handle", label: "x" }, bearer);
  const empty = await channelsCall(tiers.env, "PUT", { channelId: KASUMI, label: " " }, bearer);
  const long = await channelsCall(
    tiers.env,
    "PUT",
    { channelId: KASUMI, label: "x".repeat(65) },
    bearer,
  );
  const control = await channelsCall(
    tiers.env,
    "PUT",
    { channelId: KASUMI, label: "a\nb" },
    bearer,
  );
  const post = await channelsCall(tiers.env, "POST", { channelId: KASUMI }, bearer);
  expect([
    viewer.status,
    anonymous.status,
    badId.status,
    empty.status,
    long.status,
    control.status,
    post.status,
    tiers.store.has(CHANNELS_KEY),
  ]).toStrictEqual([403, 401, 400, 400, 400, 400, 405, false]);
});

it("gives a share only its own video's channel label", async () => {
  const tiers = await channelTiers();
  const catalog = { videos: [{ ...VIDEO, channelId: KASUMI }] };
  tiers.store.set("catalog.json", JSON.stringify(catalog));
  tiers.store.set(
    CHANNELS_KEY,
    JSON.stringify({
      version: 1,
      revision: 2,
      channels: { [KASUMI]: { label: "熊童子かすみ" }, [MILK]: { label: "別" } },
    }),
  );
  const { env } = tiers;
  const created = await adminWrite(env, "POST", { videoId: "v1", edited: true }, tiers.session);
  const token = (await created.json()).share.token;
  const pass = await passChallenge(env, token);
  const response = await shareCall(env, `/s/${token}/api/channels.json`, {
    headers: { Cookie: pass },
  });
  expect(await response.json()).toStrictEqual({
    version: 1,
    channels: { [KASUMI]: { label: "熊童子かすみ" } },
  });
});

// R2's ranged get: offset/length, offset only or suffix; it answers with the whole size.
function rangeBucket(content: string, refuses: boolean) {
  const heads: string[] = [];
  const object = (body: string) => ({ size: content.length, httpEtag: '"m"', body });
  type R2Range = { offset?: number; length?: number; suffix?: number };
  return {
    heads,
    head: async (key: string) => {
      heads.push(key);
      return key === "media/v.mp4" ? object(content) : null;
    },
    get: async (key: string, options?: { range?: R2Range }) => {
      if (key !== "media/v.mp4") return null;
      const range = options?.range;
      if (range === undefined) return object(content);
      const start =
        range.suffix === undefined ? (range.offset ?? 0) : content.length - range.suffix;
      if (refuses && start >= content.length) throw new Error("InvalidRange");
      const end = range.length === undefined ? content.length : start + range.length;
      return object(content.slice(Math.max(start, 0), end));
    },
  };
}

async function media(
  range: string | null,
  options: { refuses?: boolean; path?: string; method?: string } = {},
) {
  const MEDIA = rangeBucket("0123456789", options.refuses ?? true);
  // Minted now: an earlier test moves Date.now() hours ahead.
  const fresh = await sessionCookie({
    uid: "1",
    u: "tester",
    exp: Math.floor(Date.now() / 1000) + 600,
  });
  const headers: Record<string, string> = { Cookie: fresh };
  if (range !== null) headers.Range = range;
  const response = await worker.fetch(
    new Request(`https://example.test${options.path ?? "/media/v.mp4"}`, {
      method: options.method ?? "GET",
      headers,
    }),
    { ...environment, MEDIA },
  );
  return [
    response.status,
    response.headers.get("Content-Range"),
    response.headers.get("Content-Length"),
    response.headers.get("ETag"),
    await response.text(),
    MEDIA.heads.length,
  ];
}

it("serves media ranges straight from R2 and asks head() only without a usable range", async () => {
  expect([
    await media(null),
    await media("bytes=4-"),
    await media("bytes=2-5"),
    await media("bytes=8-20"),
    await media("bytes=-3"),
    await media("bytes=-30"),
    await media("bytes=12-"),
    await media("bytes=10-", { refuses: false }),
    await media("bytes=5-2"),
    await media("bytes=a-"),
    await media("bytes=-0"),
    await media(null, { method: "HEAD" }),
    await media("bytes=0-1", { path: "/media/missing.mp4" }),
    await media(null, { path: "/media/missing.mp4" }),
  ]).toStrictEqual([
    [200, null, "10", '"m"', "0123456789", 1],
    [206, "bytes 4-9/10", "6", '"m"', "456789", 0],
    [206, "bytes 2-5/10", "4", '"m"', "2345", 0],
    [206, "bytes 8-9/10", "2", '"m"', "89", 0],
    [206, "bytes 7-9/10", "3", '"m"', "789", 0],
    [206, "bytes 0-9/10", "10", '"m"', "0123456789", 0],
    [416, "bytes */10", null, null, "Invalid range", 1],
    [416, "bytes */10", null, null, "Invalid range", 0],
    [416, "bytes */10", null, null, "Invalid range", 1],
    [416, "bytes */10", null, null, "Invalid range", 1],
    [416, "bytes */10", null, null, "Invalid range", 1],
    [200, null, "10", '"m"', "", 1],
    [404, null, null, null, "Not found", 0],
    [404, null, null, null, "Not found", 1],
  ]);
});

it("keeps the download header on a ranged media answer", async () => {
  const MEDIA = rangeBucket("0123456789", true);
  const response = await worker.fetch(
    new Request("https://example.test/media/v.mp4?download=1", {
      headers: {
        Cookie: await sessionCookie({
          uid: "1",
          u: "tester",
          exp: Math.floor(Date.now() / 1000) + 600,
        }),
        Range: "bytes=0-0",
      },
    }),
    { ...environment, MEDIA },
  );
  expect([
    response.status,
    response.headers.get("Content-Disposition"),
    await response.text(),
  ]).toStrictEqual([206, 'attachment; filename="v.mp4"', "0"]);
});
