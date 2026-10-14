// TranscriptViewer Web: authenticated SPA + catalog + R2 media with Range.
// Every route except /login and /auth/* requires a signed session cookie issued
// after X (OAuth 2.0 + PKCE) sign-in of an admin (env X_ADMIN_USERNAMES) or an
// account the admin added on /admin (stored in R2 config/allowed-users.json), because the
// bucket holds personal recordings; never add a public route here.
// Until X_CLIENT_ID is configured, HTTP Basic (BASIC_USER / BASIC_PASS) stays
// as the only gate so a half-configured deploy never opens the site.
const SESSION_COOKIE = "tv_session";
const OAUTH_COOKIE = "tv_oauth";
const SESSION_SECONDS = 60 * 60 * 24 * 30;
const OAUTH_SECONDS = 600;
const DEBUG_SESSION_SECONDS = 3600;
const ALLOW_KEY = "config/allowed-users.json";
const ALLOW_CACHE_MS = 10_000;
let allowCache = { at: 0, users: [] };

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (xConfigured(env)) {
      if (url.pathname === "/login") return loginPage(url);
      if (url.pathname === "/auth/login") return startLogin(url, env);
      if (url.pathname === "/auth/callback") return finishLogin(request, url, env);
      if (url.pathname === "/auth/logout") return logout(url);
      if (url.pathname === "/auth/debug") return debugLogin(request, env);
      const session = await currentUser(request, env);
      if (!session) return denyUnauthenticated(request);
      // Debug sessions (minted by /auth/debug) are read-only.
      if (session.dbg && request.method !== "GET" && request.method !== "HEAD") {
        return new Response("Forbidden", { status: 403, headers: { "Cache-Control": "no-store" } });
      }
      if (url.pathname === "/api/me") return json({ username: session.u, admin: isAdmin(session.u, env) });
      if (url.pathname === "/admin" || url.pathname.startsWith("/api/admin/")) {
        if (!isAdmin(session.u, env)) return new Response("Forbidden", { status: 403, headers: { "Cache-Control": "no-store" } });
        if (url.pathname === "/admin") return adminPage();
        return adminApi(request, url, env);
      }
    } else if (!basicAuthorized(request, env)) {
      return new Response("Authentication required", {
        status: 401,
        headers: { "WWW-Authenticate": 'Basic realm="TranscriptViewer"' },
      });
    }
    if (url.pathname === "/") return redirect("/streamers");
    if (url.pathname === "/stremers" || url.pathname.startsWith("/stremers/")) {
      return redirect("/streamers" + url.pathname.slice("/stremers".length) + url.search);
    }
    // /streamers, /streamers/<name> and /streamers/<name>/videos/<id> are client-routed pages.
    if (url.pathname === "/streamers" || url.pathname.startsWith("/streamers/")) {
      return env.ASSETS.fetch(new Request(new URL("/", url), request));
    }
    if (url.pathname === "/api/catalog") {
      return r2Json(env, "catalog.json");
    }
    const dataMatch = url.pathname.match(/^\/api\/videos\/([A-Za-z0-9._-]+)\.json$/);
    if (dataMatch) {
      return r2Json(env, `data/${dataMatch[1]}.json`);
    }
    const mediaMatch = url.pathname.match(/^\/media\/([A-Za-z0-9._-]+\.mp4)$/);
    if (mediaMatch) {
      return serveMedia(request, env, `media/${mediaMatch[1]}`);
    }
    const originalMatch = url.pathname.match(/^\/media-original\/([A-Za-z0-9._-]+\.mp4)$/);
    if (originalMatch) {
      return serveMedia(request, env, `media-original/${originalMatch[1]}`);
    }
    return env.ASSETS.fetch(request);
  },
};

function redirect(location) {
  return new Response(null, { status: 302, headers: { Location: location, "Cache-Control": "no-store" } });
}

function xConfigured(env) {
  return Boolean(env.X_CLIENT_ID && env.SESSION_SECRET);
}

function basicAuthorized(request, env) {
  const expected = "Basic " + btoa(`${env.BASIC_USER}:${env.BASIC_PASS}`);
  const got = request.headers.get("Authorization") || "";
  if (got.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < got.length; i++) diff |= got.charCodeAt(i) ^ expected.charCodeAt(i);
  return diff === 0;
}

const enc = new TextEncoder();
const b64url = (bytes) =>
  btoa(String.fromCharCode(...new Uint8Array(bytes))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const unb64url = (text) => Uint8Array.from(atob(text.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0));
const randomToken = (n = 32) => b64url(crypto.getRandomValues(new Uint8Array(n)));

function hmacKey(env, usage) {
  return crypto.subtle.importKey("raw", enc.encode(env.SESSION_SECRET), { name: "HMAC", hash: "SHA-256" }, false, [usage]);
}

async function sign(env, payload) {
  const body = b64url(enc.encode(JSON.stringify(payload)));
  const mac = await crypto.subtle.sign("HMAC", await hmacKey(env, "sign"), enc.encode(body));
  return `${body}.${b64url(mac)}`;
}

async function verify(env, token) {
  const [body, mac] = (token || "").split(".");
  if (!body || !mac) return null;
  try {
    const ok = await crypto.subtle.verify("HMAC", await hmacKey(env, "verify"), unb64url(mac), enc.encode(body));
    if (!ok) return null;
    const payload = JSON.parse(new TextDecoder().decode(unb64url(body)));
    return payload.exp > Date.now() / 1000 ? payload : null;
  } catch {
    return null;
  }
}

function cookie(request, name) {
  for (const part of (request.headers.get("Cookie") || "").split(/;\s*/)) {
    const i = part.indexOf("=");
    if (i > 0 && part.slice(0, i) === name) return part.slice(i + 1);
  }
  return "";
}

const setCookie = (name, value, maxAge) =>
  `${name}=${value}; Path=/; Max-Age=${maxAge}; HttpOnly; Secure; SameSite=Lax`;

const normalizeName = (v) => String(v || "").trim().replace(/^@/, "").toLowerCase();
const isAdmin = (name, env) => (env.X_ADMIN_USERNAMES || "").split(",").map(normalizeName).filter(Boolean).includes(normalizeName(name));

// R2 config/allowed-users.json is the source of truth. KV (global replica) and
// the Cache API (per-colo) are read-through layers refreshed on every admin change.
// Cache API entries only exist on custom domains; on workers.dev it is a no-op and
// reads fall through to KV, then R2.
const ALLOW_CACHE_URL = "https://allowlist.internal/v1";
const ALLOW_KV_KEY = "allowed-users";
const ALLOW_CACHE_SECONDS = 60;

async function readAllowlist(env) {
  const cached = await caches.default.match(ALLOW_CACHE_URL);
  if (cached) return (await cached.json()).users;
  let users = (await env.ALLOW_KV.get(ALLOW_KV_KEY, { type: "json", cacheTtl: 60 }))?.users;
  if (!users) {
    const obj = await env.MEDIA.get(ALLOW_KEY);
    users = obj ? (await obj.json()).users || [] : [];
    await env.ALLOW_KV.put(ALLOW_KV_KEY, JSON.stringify({ users }));
  }
  await cacheAllowlist(users);
  return users;
}

function cacheAllowlist(users) {
  return caches.default.put(
    ALLOW_CACHE_URL,
    new Response(JSON.stringify({ users }), { headers: { "Cache-Control": `max-age=${ALLOW_CACHE_SECONDS}` } }),
  );
}

async function writeAllowlist(env, users) {
  const body = JSON.stringify({ users });
  await env.MEDIA.put(ALLOW_KEY, body, { httpMetadata: { contentType: "application/json" } });
  await env.ALLOW_KV.put(ALLOW_KV_KEY, body);
  await cacheAllowlist(users);
  allowCache = { at: Date.now(), users };
}

async function allowedUsers(env) {
  if (Date.now() - allowCache.at < ALLOW_CACHE_MS) return allowCache.users;
  const users = await readAllowlist(env);
  allowCache = { at: Date.now(), users };
  return users;
}

async function isAllowed(name, env) {
  return isAdmin(name, env) || (await allowedUsers(env)).includes(normalizeName(name));
}

// Re-checks the allow-list on every request so removing a user takes effect
// within about a minute even while their cookie is still valid.
async function currentUser(request, env) {
  const payload = await verify(env, cookie(request, SESSION_COOKIE));
  return payload && (await isAllowed(payload.u, env)) ? payload : null;
}

const json = (value, status = 200) =>
  new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" },
  });

async function adminApi(request, url, env) {
  if (url.pathname !== "/api/admin/users") return new Response("Not found", { status: 404 });
  if (request.method === "GET") return json({ users: await allowedUsers(env) });
  const sameOrigin = request.headers.get("Origin") === url.origin;
  const isJson = (request.headers.get("Content-Type") || "").startsWith("application/json");
  if (!sameOrigin || !isJson) return new Response("Bad request", { status: 400 });
  const name = normalizeName((await request.json().catch(() => ({}))).username);
  if (!/^[a-z0-9_]{1,15}$/.test(name)) return json({ error: "invalid username" }, 400);
  const current = await allowedUsers(env);
  let next;
  if (request.method === "POST") next = [...new Set([...current, name])].sort();
  else if (request.method === "DELETE") next = current.filter((u) => u !== name);
  else return new Response("Method not allowed", { status: 405 });
  await writeAllowlist(env, next);
  return json({ users: next });
}

function adminPage() {
  const html = `<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>閲覧許可の管理 | TranscriptViewer Web</title><meta name="robots" content="noindex"><link rel="icon" type="image/svg+xml" href="${FAVICON}"><style>
:root{color-scheme:dark}body{margin:0;font:15px/2 system-ui,-apple-system,"Hiragino Sans",sans-serif;background:#0f1115;color:#e8eaed}
main{max-width:560px;margin:0 auto;padding:24px 16px}h1{font-size:20px}a{color:#6ea8ff}p,.note{color:#9aa3b2;font-size:13px}
form{display:flex;gap:8px;margin:16px 0}input{flex:1;font:inherit;padding:9px 12px;border:1px solid #2a2f3a;border-radius:8px;background:#1d212a;color:inherit}
button{font:inherit;padding:9px 14px;border:1px solid #2a2f3a;border-radius:8px;background:#1f3a63;color:#fff;cursor:pointer}
li{display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border:1px solid #2a2f3a;border-radius:8px;margin:6px 0;background:#171a21;list-style:none}
ul{padding:0}li button{background:#2a2f3a}input:focus-visible,button:focus-visible{outline:2px solid #6ea8ff;outline-offset:2px}.err{color:#ff8a8a}</style>
<main><p><a href="/">← 動画一覧へ</a></p><h1>閲覧許可の管理</h1>
<p>ここに追加したXアカウントは、ログイン後に閲覧できます。管理者は環境変数 X_ADMIN_USERNAMES で指定され、常に許可されます。</p>
<form id="f"><input id="u" placeholder="Xのユーザー名（@なし）" autocomplete="off" required><button>追加</button></form>
<p id="msg" class="err" role="alert"></p><ul id="list"></ul><p class="note" id="empty" hidden>追加されたアカウントはありません。</p></main>
<script>
const list=document.getElementById("list"),msg=document.getElementById("msg"),empty=document.getElementById("empty");
function render(users){list.replaceChildren(...users.map(n=>{const li=document.createElement("li"),b=document.createElement("button");li.append("@"+n);b.textContent="削除";b.onclick=()=>call("DELETE",n);li.append(b);return li;}));empty.hidden=users.length>0;}
async function call(method,username){msg.textContent="";const r=await fetch("/api/admin/users",{method,headers:{"Content-Type":"application/json"},body:JSON.stringify({username})});const d=await r.json().catch(()=>({}));if(!r.ok){msg.textContent=d.error||"失敗しました";return;}render(d.users);}
document.getElementById("f").onsubmit=e=>{e.preventDefault();const v=document.getElementById("u");call("POST",v.value);v.value="";};
fetch("/api/admin/users").then(r=>r.json()).then(d=>render(d.users));
</script></html>`;
  return new Response(html, { headers: PAGE_HEADERS });
}

function denyUnauthenticated(request) {
  const wantsPage = (request.headers.get("Accept") || "").includes("text/html");
  if (wantsPage && request.method === "GET") {
    return new Response(null, { status: 302, headers: { Location: "/login", "Cache-Control": "no-store" } });
  }
  return new Response("Authentication required", { status: 401, headers: { "Cache-Control": "no-store" } });
}

const FAVICON = "data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%2064%2064%22%3E%3Ccircle%20cx=%2232%22%20cy=%2232%22%20r=%2231%22%20fill=%22%23bfe6fb%22/%3E%3Cellipse%20cx=%2214%22%20cy=%2246%22%20rx=%227%22%20ry=%224%22%20fill=%22%23cfd8e0%22%20transform=%22rotate%28-25%2014%2046%29%22/%3E%3Cellipse%20cx=%2250%22%20cy=%2246%22%20rx=%227%22%20ry=%224%22%20fill=%22%23cfd8e0%22%20transform=%22rotate%2825%2050%2046%29%22/%3E%3Cellipse%20cx=%2232%22%20cy=%2238%22%20rx=%2222%22%20ry=%2219%22%20fill=%22%23eef2f6%22/%3E%3Ccircle%20cx=%2223%22%20cy=%2233%22%20r=%223.2%22%20fill=%22%231d2430%22/%3E%3Ccircle%20cx=%2241%22%20cy=%2233%22%20r=%223.2%22%20fill=%22%231d2430%22/%3E%3Ccircle%20cx=%2224%22%20cy=%2232%22%20r=%221%22%20fill=%22%23fff%22/%3E%3Ccircle%20cx=%2242%22%20cy=%2232%22%20r=%221%22%20fill=%22%23fff%22/%3E%3Cellipse%20cx=%2232%22%20cy=%2240%22%20rx=%224.5%22%20ry=%223.2%22%20fill=%22%231d2430%22/%3E%3Cellipse%20cx=%2218%22%20cy=%2241%22%20rx=%223%22%20ry=%222%22%20fill=%22%23f7b6c4%22/%3E%3Cellipse%20cx=%2246%22%20cy=%2241%22%20rx=%223%22%20ry=%222%22%20fill=%22%23f7b6c4%22/%3E%3Cg%20stroke=%22%238b97a6%22%20stroke-width=%221%22%20stroke-linecap=%22round%22%3E%3Cpath%20d=%22M24%2043%20L14%2041%22/%3E%3Cpath%20d=%22M24%2045%20L14%2046%22/%3E%3Cpath%20d=%22M40%2043%20L50%2041%22/%3E%3Cpath%20d=%22M40%2045%20L50%2046%22/%3E%3C/g%3E%3Cpath%20d=%22M28%2044%20Q32%2048%2036%2044%22%20stroke=%22%231d2430%22%20stroke-width=%221.4%22%20fill=%22none%22%20stroke-linecap=%22round%22/%3E%3C/svg%3E";

const PAGE_HEADERS = {
  "Content-Type": "text/html; charset=utf-8",
  "Cache-Control": "no-store",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
};

function loginPage(url) {
  const failed = url.searchParams.get("error");
  const message = failed
    ? `<p class="err" role="alert">${failed === "denied" ? "このXアカウントには閲覧権限がありません。" : "ログインに失敗しました。もう一度お試しください。"}</p>`
    : "";
  const html = `<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ログイン | TranscriptViewer Web</title><meta name="robots" content="noindex"><link rel="icon" type="image/svg+xml" href="${FAVICON}"><style>
:root{color-scheme:dark}body{margin:0;min-height:100dvh;display:grid;place-items:center;font-family:system-ui,-apple-system,"Hiragino Sans",sans-serif;background:#0f1115;color:#e8eaed}
main{width:min(380px,calc(100vw - 32px));padding:28px;border:1px solid #2a2f3a;border-radius:16px;background:#171a21;text-align:center}
h1{font-size:20px;margin:0 0 8px}p{color:#9aa3b2;font-size:14px;line-height:2;margin:0 0 20px}
a.btn{display:block;padding:13px;border-radius:999px;background:#fff;color:#000;font-weight:700;text-decoration:none}a.btn:focus-visible{outline:3px solid #6ea8ff;outline-offset:2px}
.err{color:#ff8a8a}</style><main><h1>TranscriptViewer Web</h1><p>個人の配信アーカイブです。許可されたXアカウントでログインしてください。</p>${message}<a class="btn" href="/auth/login">Xでログイン</a></main></html>`;
  return new Response(html, { headers: PAGE_HEADERS });
}

async function startLogin(url, env) {
  const state = randomToken(16);
  const verifier = randomToken(48);
  const challenge = b64url(await crypto.subtle.digest("SHA-256", enc.encode(verifier)));
  const auth = new URL("https://x.com/i/oauth2/authorize");
  auth.search = new URLSearchParams({
    response_type: "code",
    client_id: env.X_CLIENT_ID,
    redirect_uri: `${url.origin}/auth/callback`,
    scope: "users.read tweet.read",
    state,
    code_challenge: challenge,
    code_challenge_method: "S256",
  }).toString();
  const pending = await sign(env, { state, verifier, exp: Math.floor(Date.now() / 1000) + OAUTH_SECONDS });
  return new Response(null, {
    status: 302,
    headers: { Location: auth.toString(), "Set-Cookie": setCookie(OAUTH_COOKIE, pending, OAUTH_SECONDS), "Cache-Control": "no-store" },
  });
}

async function finishLogin(request, url, env) {
  const fail = (reason) =>
    new Response(null, {
      status: 302,
      headers: { Location: `/login?error=${reason}`, "Set-Cookie": setCookie(OAUTH_COOKIE, "", 0), "Cache-Control": "no-store" },
    });
  const pending = await verify(env, cookie(request, OAUTH_COOKIE));
  const code = url.searchParams.get("code");
  if (!pending || !code || url.searchParams.get("state") !== pending.state) return fail("failed");
  const headers = { "Content-Type": "application/x-www-form-urlencoded" };
  if (env.X_CLIENT_SECRET) headers.Authorization = "Basic " + btoa(`${env.X_CLIENT_ID}:${env.X_CLIENT_SECRET}`);
  const tokenResponse = await fetch("https://api.x.com/2/oauth2/token", {
    method: "POST",
    headers,
    body: new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: `${url.origin}/auth/callback`,
      code_verifier: pending.verifier,
      client_id: env.X_CLIENT_ID,
    }),
  });
  if (!tokenResponse.ok) return fail("failed");
  const { access_token: accessToken } = await tokenResponse.json();
  const meResponse = await fetch("https://api.x.com/2/users/me", { headers: { Authorization: `Bearer ${accessToken}` } });
  if (!meResponse.ok) return fail("failed");
  const user = (await meResponse.json()).data;
  if (!user || !(await isAllowed(user.username, env))) return fail("denied");
  const session = await sign(env, { uid: user.id, u: normalizeName(user.username), exp: Math.floor(Date.now() / 1000) + SESSION_SECONDS });
  const out = new Headers({ Location: "/streamers", "Cache-Control": "no-store" });
  out.append("Set-Cookie", setCookie(SESSION_COOKIE, session, SESSION_SECONDS));
  out.append("Set-Cookie", setCookie(OAUTH_COOKIE, "", 0));
  return new Response(null, { status: 302, headers: out });
}

// Debug access for development: POST /auth/debug with header X-Debug-Token mints a
// one-hour, read-only admin session. It exists only while DEBUG_TOKEN (32+ chars) is set
// and DEBUG_TOKEN_EXPIRES (unix seconds) is in the future; otherwise the route is a 404.
function debugEnabled(env) {
  return Boolean(env.DEBUG_TOKEN && env.DEBUG_TOKEN.length >= 32 && Number(env.DEBUG_TOKEN_EXPIRES) > Date.now() / 1000);
}

async function sameSecret(given, expected) {
  const [a, b] = await Promise.all([given, expected].map((v) => crypto.subtle.digest("SHA-256", enc.encode(v))));
  const x = new Uint8Array(a), y = new Uint8Array(b);
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff |= x[i] ^ y[i];
  return diff === 0;
}

async function debugLogin(request, env) {
  const notFound = new Response("Not found", { status: 404, headers: { "Cache-Control": "no-store" } });
  if (request.method !== "POST" || !debugEnabled(env)) return notFound;
  if (!(await sameSecret(request.headers.get("X-Debug-Token") || "", env.DEBUG_TOKEN))) return notFound;
  const admin = normalizeName((env.X_ADMIN_USERNAMES || "").split(",")[0]);
  if (!admin) return notFound;
  const session = await sign(env, { uid: "debug", u: admin, dbg: 1, exp: Math.floor(Date.now() / 1000) + DEBUG_SESSION_SECONDS });
  return new Response(null, {
    status: 204,
    headers: { "Set-Cookie": setCookie(SESSION_COOKIE, session, DEBUG_SESSION_SECONDS), "Cache-Control": "no-store" },
  });
}

function logout() {
  return new Response(null, {
    status: 302,
    headers: { Location: "/login", "Set-Cookie": setCookie(SESSION_COOKIE, "", 0), "Cache-Control": "no-store" },
  });
}

async function r2Json(env, key) {
  const obj = await env.MEDIA.get(key);
  if (!obj) return new Response("Not found", { status: 404 });
  return new Response(obj.body, {
    headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "private, max-age=300, stale-while-revalidate=86400" },
  });
}

// R2 honors HTTP range reads; without this a <video> element cannot seek.
async function serveMedia(request, env, key) {
  const head = await env.MEDIA.head(key);
  if (!head) return new Response("Not found", { status: 404 });
  const size = head.size;
  const base = {
    "Content-Type": "video/mp4",
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, max-age=3600",
    ETag: head.httpEtag,
  };
  if (request.method === "HEAD") {
    return new Response(null, { headers: { ...base, "Content-Length": String(size) } });
  }
  const range = parseRange(request.headers.get("Range"), size);
  if (range === "invalid") {
    return new Response("Invalid range", { status: 416, headers: { "Content-Range": `bytes */${size}` } });
  }
  if (!range) {
    const obj = await env.MEDIA.get(key);
    return new Response(obj.body, { headers: { ...base, "Content-Length": String(size) } });
  }
  const obj = await env.MEDIA.get(key, { range: { offset: range.start, length: range.end - range.start + 1 } });
  return new Response(obj.body, {
    status: 206,
    headers: {
      ...base,
      "Content-Length": String(range.end - range.start + 1),
      "Content-Range": `bytes ${range.start}-${range.end}/${size}`,
    },
  });
}

function parseRange(header, size) {
  if (!header) return null;
  const m = header.match(/^bytes=(\d*)-(\d*)$/);
  if (!m || (m[1] === "" && m[2] === "")) return "invalid";
  let start, end;
  if (m[1] === "") {
    const suffix = Number(m[2]);
    if (suffix <= 0) return "invalid";
    start = Math.max(size - suffix, 0);
    end = size - 1;
  } else {
    start = Number(m[1]);
    end = m[2] === "" ? size - 1 : Math.min(Number(m[2]), size - 1);
  }
  return start > end || start >= size ? "invalid" : { start, end };
}
