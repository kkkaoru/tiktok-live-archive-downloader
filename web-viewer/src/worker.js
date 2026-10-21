// TranscriptViewer Web: authenticated SPA + catalog + R2 media with Range.
// Every route except /login and /auth/* requires a signed session cookie issued
// after X (OAuth 2.0 + PKCE) sign-in of an admin (env X_ADMIN_USERNAMES) or an
// account the admin added on /admin (stored in R2 config/allowed-users.json), because the
// bucket holds personal recordings; never add a public route here. The one exception is
// /s/<token>/…: an admin-issued share (R2 shares/<token>.json, random 256-bit token,
// expiry, revocable) that exposes only the page and the single video it names, and only
// after a Cloudflare Turnstile check (no Turnstile keys configured: shares stay closed).
// Until X_CLIENT_ID is configured, HTTP Basic (BASIC_USER / BASIC_PASS) stays
// as the only gate so a half-configured deploy never opens the site.
import { enqueue, lookup, MOVIE_KEY, refresh, requestOf } from "./clips.ts";

const SESSION_COOKIE = "tv_session";
const OAUTH_COOKIE = "tv_oauth";
const SESSION_SECONDS = 60 * 60 * 24 * 30;
const OAUTH_SECONDS = 600;
const DEBUG_SESSION_SECONDS = 3600;
const ALLOW_KEY = "config/allowed-users.json";
const ALLOW_CACHE_MS = 10_000;
const SHARE_DEFAULT_DAYS = 7;
const SHARE_MAX_DAYS = 365;
const SHARE_PASS_COOKIE = "tv_share";
const SHARE_PASS_SECONDS = 60 * 60 * 12;
const TURNSTILE_VERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify";
const CLIPS_PATH = /^\/api\/clips\/([A-Za-z0-9._-]+)(\.json|\/reload)$/;
let allowCache = { at: 0, users: [] };

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith("/s/")) return shareRoute(request, url, env);
    // The Mac viewer has no X login: it reads and reloads clips with the CLIPS_API_TOKEN bearer.
    if (CLIPS_PATH.test(url.pathname) && (await clipsBearer(request, env))) return clipsRoute(request, url, env, { admin: true, bearer: true, queue: true });
    if (url.pathname === "/robots.txt") {
      return new Response("User-agent: *\nDisallow: /\n", { headers: { "Content-Type": "text/plain; charset=utf-8" } });
    }
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
      if (CLIPS_PATH.test(url.pathname)) return clipsRoute(request, url, env, { admin: isAdmin(session.u, env), bearer: false, queue: !session.dbg });
      if (url.pathname === "/admin" || url.pathname.startsWith("/api/admin/")) {
        if (!isAdmin(session.u, env)) return new Response("Forbidden", { status: 403, headers: { "Cache-Control": "no-store" } });
        if (url.pathname === "/admin") return adminPage();
        return adminApi(request, url, env, session);
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
      // Revalidate every load (cheap 304) so a deploy is never hidden behind a cached old page.
      const page = await env.ASSETS.fetch(new Request(new URL("/", url), request));
      const headers = new Headers(page.headers);
      headers.set("Cache-Control", "no-cache");
      return new Response(page.body, { status: page.status, headers });
    }
    if (url.pathname === "/api/catalog") {
      return r2Json(request, env, "catalog.json");
    }
    if (CLIPS_PATH.test(url.pathname)) return clipsRoute(request, url, env, { admin: true, bearer: false, queue: true });
    const dataMatch = url.pathname.match(/^\/api\/videos\/([A-Za-z0-9._-]+)\.json$/);
    if (dataMatch) {
      return r2Json(request, env, `data/${dataMatch[1]}.json`);
    }
    const mediaMatch = url.pathname.match(/^\/media\/([A-Za-z0-9._-]+\.mp4)$/);
    if (mediaMatch) {
      return serveMedia(request, url, env, `media/${mediaMatch[1]}`);
    }
    const originalMatch = url.pathname.match(/^\/media-original\/([A-Za-z0-9._-]+\.mp4)$/);
    if (originalMatch) {
      return serveMedia(request, url, env, `media-original/${originalMatch[1]}`);
    }
    return env.ASSETS.fetch(request);
  },

  // R2 event notifications for data/<id>.json and the viewers' own requests: one video per message.
  async queue(batch, env) {
    for (const message of batch.messages) {
      const request = requestOf(message.body);
      try {
        if (request) console.log(JSON.stringify({ clips: request.videoId, outcome: await refresh(env, request, new Date()) }));
        message.ack();
      } catch (error) {
        console.error(JSON.stringify({ clips: request?.videoId, error: String(error) }));
        message.retry();
      }
    }
  },
};

async function clipsBearer(request, env) {
  const header = request.headers.get("Authorization") || "";
  const token = env.CLIPS_API_TOKEN || "";
  return token.length >= 32 && header.startsWith("Bearer ") && (await sameSecret(header.slice(7), token));
}

// GET returns the stored clips (a missing record is queued and reads as pending); POST …/reload
// recomputes them (admins, the bearer, or Basic mode). Pages poll while the status is pending.
async function clipsRoute(request, url, env, { admin, bearer, queue }) {
  const [, videoId, kind] = url.pathname.match(CLIPS_PATH);
  const expected = kind === "/reload" ? "POST" : "GET";
  if (request.method !== expected) return new Response("Method not allowed", { status: 405 });
  if (kind === "/reload") {
    if (!admin) return new Response("Forbidden", { status: 403, headers: { "Cache-Control": "no-store" } });
    // Cookie sessions only from this site's own pages, like the admin API.
    if (!bearer && request.headers.get("Origin") !== url.origin) return new Response("Bad request", { status: 400 });
    await enqueue(env, { videoId, force: true }, new Date());
    return json({ status: "pending", force: true });
  }
  const key = url.searchParams.get("key");
  if (key !== null && !MOVIE_KEY.test(key)) return json({ error: "invalid key" }, 400);
  const found = await lookup(env, { videoId, key, now: new Date() });
  // Read-only debug sessions never queue work.
  if (found.status !== "missing" || !queue) return json(found);
  if (!(await env.MEDIA.head(`data/${videoId}.json`))) return json(found, 404);
  await enqueue(env, { videoId, force: false }, new Date());
  return json({ status: "pending", force: false });
}

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

async function adminApi(request, url, env, session) {
  const shares = url.pathname === "/api/admin/shares";
  if (!shares && url.pathname !== "/api/admin/users") return new Response("Not found", { status: 404 });
  if (request.method === "GET") {
    return shares ? json({ shares: await listShares(env, url.searchParams.get("video")) }) : json({ users: await allowedUsers(env) });
  }
  const sameOrigin = request.headers.get("Origin") === url.origin;
  const isJson = (request.headers.get("Content-Type") || "").startsWith("application/json");
  if (!sameOrigin || !isJson) return new Response("Bad request", { status: 400 });
  if (shares) return shareWrite(request, env, session);
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
<p id="msg" class="err" role="alert"></p><ul id="list"></ul><p class="note" id="empty" hidden>追加されたアカウントはありません。</p>
<h1>共有リンク</h1><p>動画ページの「共有」で発行したリンクです。オフにするとすぐに無効になります。</p><ul id="shares"></ul><p class="note" id="noShares" hidden>共有リンクはありません。</p></main>
<script>
const list=document.getElementById("list"),msg=document.getElementById("msg"),empty=document.getElementById("empty");
function render(users){list.replaceChildren(...users.map(n=>{const li=document.createElement("li"),b=document.createElement("button");li.append("@"+n);b.textContent="削除";b.onclick=()=>call("DELETE",n);li.append(b);return li;}));empty.hidden=users.length>0;}
async function call(method,username){msg.textContent="";const r=await fetch("/api/admin/users",{method,headers:{"Content-Type":"application/json"},body:JSON.stringify({username})});const d=await r.json().catch(()=>({}));if(!r.ok){msg.textContent=d.error||"失敗しました";return;}render(d.users);}
document.getElementById("f").onsubmit=e=>{e.preventDefault();const v=document.getElementById("u");call("POST",v.value);v.value="";};
fetch("/api/admin/users").then(r=>r.json()).then(d=>render(d.users));
function renderShares(shares){const now=Date.now()/1000;document.getElementById("noShares").hidden=shares.length>0;document.getElementById("shares").replaceChildren(...shares.map(s=>{const li=document.createElement("li"),a=document.createElement("a"),b=document.createElement("button"),kinds=[s.video.movie&&"編集済み",s.video.original&&"元動画"].filter(Boolean).join("・");a.href="/s/"+s.token;a.target="_blank";a.textContent=(s.video.broadcastStartJST||s.video.id)+" "+kinds;li.append(a," ",s.expiresAt>now?"〜"+new Date(s.expiresAt*1000).toLocaleString("ja-JP"):"期限切れ");b.textContent="オフにする";b.onclick=async()=>{const r=await fetch("/api/admin/shares",{method:"DELETE",headers:{"Content-Type":"application/json"},body:JSON.stringify({token:s.token})});if(r.ok)loadShares();else msg.textContent="失敗しました";};li.append(b);return li;}));}
function loadShares(){fetch("/api/admin/shares").then(r=>r.json()).then(d=>renderShares(d.shares));}
loadShares();
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

// Catalog and transcripts change when a sync uploads them: browsers revalidate every time and
// an unchanged object costs only a 304 (a long stale-while-revalidate kept renamed speakers for a day).
async function r2Json(request, env, key) {
  const headers = { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "private, no-cache" };
  const head = await env.MEDIA.head(key);
  if (!head) return new Response("Not found", { status: 404 });
  if (request.headers.get("If-None-Match") === head.httpEtag) {
    return new Response(null, { status: 304, headers: { ...headers, ETag: head.httpEtag } });
  }
  const obj = await env.MEDIA.get(key);
  if (!obj) return new Response("Not found", { status: 404 });
  return new Response(obj.body, { headers: { ...headers, ETag: obj.httpEtag } });
}

// R2 honors HTTP range reads; without this a <video> element cannot seek.
// ?download=1 adds Content-Disposition; Range stays on so large downloads can resume.
async function serveMedia(request, url, env, key) {
  const head = await env.MEDIA.head(key);
  if (!head) return new Response("Not found", { status: 404 });
  const size = head.size;
  const base = {
    "Content-Type": "video/mp4",
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, max-age=3600",
    ETag: head.httpEtag,
  };
  if (url.searchParams.get("download") === "1") {
    base["Content-Disposition"] = `attachment; filename="${key.split("/").pop()}"`;
  }
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

// Shares: R2 shares/<token>.json holds the video snapshot and the allowed kinds, so the
// public route never reads the catalog (whose entries carry local filesystem paths).
const SHARE_TOKEN = /^[A-Za-z0-9_-]{43}$/;
const shareKey = (token) => `shares/${token}.json`;
const nowSeconds = () => Math.floor(Date.now() / 1000);

async function readShare(env, token) {
  if (!SHARE_TOKEN.test(token)) return null;
  const obj = await env.MEDIA.get(shareKey(token));
  return obj ? obj.json() : null;
}

// shortcut: one list page (1000 shares), paginate if shares ever exceed that.
async function listShares(env, videoId) {
  const listed = await env.MEDIA.list({ prefix: "shares/" });
  const shares = await Promise.all(listed.objects.map(async (o) => (await env.MEDIA.get(o.key))?.json()));
  return shares
    .filter((s) => s && (!videoId || s.video.id === videoId))
    .sort((a, b) => b.createdAt - a.createdAt);
}

async function shareWrite(request, env, session) {
  const body = await request.json().catch(() => ({}));
  if (request.method === "DELETE") {
    if (!SHARE_TOKEN.test(String(body.token))) return json({ error: "invalid token" }, 400);
    await env.MEDIA.delete(shareKey(body.token));
    return json({ ok: true });
  }
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
  const days = body.days ?? SHARE_DEFAULT_DAYS;
  if (!Number.isInteger(days) || days < 1 || days > SHARE_MAX_DAYS) return json({ error: `days must be 1-${SHARE_MAX_DAYS}` }, 400);
  if (!body.edited && !body.original) return json({ error: "choose edited and/or original" }, 400);
  const catalog = await env.MEDIA.get("catalog.json");
  const entry = catalog ? (await catalog.json()).videos.find((v) => v.id === body.videoId) : undefined;
  if (!entry) return json({ error: "unknown video" }, 400);
  if ((body.edited && !entry.movie) || (body.original && !entry.original)) return json({ error: "video kind not available" }, 400);
  const createdAt = nowSeconds();
  const share = {
    token: randomToken(32),
    createdAt,
    expiresAt: createdAt + days * 86400,
    createdBy: session.u,
    // Named fields only: catalog entries also hold local paths that must stay private.
    video: {
      id: entry.id,
      streamer: entry.streamer,
      title: entry.title,
      broadcastStartJST: entry.broadcastStartJST,
      duration: entry.duration,
      movie: body.edited ? entry.movie : null,
      original: body.original ? { movie: entry.original.movie } : null,
    },
  };
  await env.MEDIA.put(shareKey(share.token), JSON.stringify(share), { httpMetadata: { contentType: "application/json" } });
  return json({ share });
}

function shareGone(url) {
  if (url.pathname.split("/").length > 3) return new Response("Not found", { status: 404, headers: { "Cache-Control": "no-store" } });
  const html = `<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>共有リンク | TranscriptViewer Web</title><meta name="robots" content="noindex"><link rel="icon" type="image/svg+xml" href="${FAVICON}"><style>
:root{color-scheme:dark}body{margin:0;min-height:100dvh;display:grid;place-items:center;font-family:system-ui,-apple-system,"Hiragino Sans",sans-serif;background:#0f1115;color:#e8eaed}p{color:#9aa3b2}</style>
<main><p>この共有リンクは無効か、有効期限が切れています。</p></main></html>`;
  return new Response(html, { status: 404, headers: { ...PAGE_HEADERS, "X-Robots-Tag": "noindex" } });
}

// Missing, malformed, expired and revoked (deleted) shares all look the same.
async function shareRoute(request, url, env) {
  const [, , token, ...rest] = url.pathname.split("/");
  const share = await readShare(env, token);
  if (!share || share.expiresAt <= nowSeconds()) return shareGone(url);
  const path = "/" + rest.join("/");
  // The link-preview image is the one public piece: crawlers fetching OGP cannot pass Turnstile.
  if (path === "/og.jpg") return shareImage(env, share.video);
  if (path === "/verify") return verifyHuman(request, url, env, share);
  if (!(await passedChallenge(request, env, token))) {
    return path === "/" ? challengePage(env, share, url) : new Response("Forbidden", { status: 403, headers: { "Cache-Control": "no-store" } });
  }
  const { video } = share;
  if (path === "/") {
    // No conditional headers from the visitor: always the current page, never a 304.
    const page = await env.ASSETS.fetch(new Request(new URL("/", url)));
    // The token is in the URL: keep it out of Referer (Google Fonts) and search indexes.
    const headers = new Headers(page.headers);
    headers.set("Referrer-Policy", "no-referrer");
    headers.set("X-Robots-Tag", "noindex");
    headers.set("Cache-Control", "no-store");
    return new Response(page.body, { status: page.status, headers });
  }
  if (path === "/api/catalog") return json({ videos: [video], share: { expiresAt: share.expiresAt } });
  if (path === `/api/clips/${video.id}.json`) return json(await lookup(env, { videoId: video.id, key: null, now: new Date() }));
  if (path === `/api/videos/${video.id}.json`) {
    const obj = await env.MEDIA.get(`data/${video.id}.json`);
    return obj ? new Response(obj.body, { headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" } }) : shareGone(url);
  }
  if (video.movie && path === `/media/${video.movie}`) return serveMedia(request, url, env, `media/${video.movie}`);
  if (video.original && path === `/media-original/${video.original.movie}`) {
    return serveMedia(request, url, env, `media-original/${video.original.movie}`);
  }
  return shareGone(url);
}

// Turnstile gate for shares: a visitor proves to be human once, then a signed cookie scoped
// to this share's path (12 h) lets the page, data and media requests through.
const turnstileConfigured = (env) => Boolean(env.TURNSTILE_SITE_KEY && env.TURNSTILE_SECRET_KEY);

async function passedChallenge(request, env, token) {
  if (!turnstileConfigured(env)) return false;
  const pass = await verify(env, cookie(request, SHARE_PASS_COOKIE));
  return Boolean(pass && pass.share === token);
}

async function shareImage(env, video) {
  const obj = await env.MEDIA.get(`og/${video.id}.jpg`);
  if (!obj) return new Response("Not found", { status: 404, headers: { "Cache-Control": "no-store" } });
  return new Response(obj.body, { headers: { "Content-Type": "image/jpeg", "Cache-Control": "public, max-age=600" } });
}

const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// Link-preview tags for chat apps and social crawlers: who, when, how long, until when.
// Nothing from the transcript or summary is exposed here.
async function previewTags(env, share, url) {
  const v = share.video;
  const when = (v.broadcastStartJST || "").slice(0, 16);
  const minutes = Math.round((v.duration || 0) / 60);
  const length = minutes >= 60 ? `${Math.floor(minutes / 60)}時間${minutes % 60}分` : `${minutes}分`;
  const until = new Date(share.expiresAt * 1000).toLocaleString("ja-JP", { timeZone: "Asia/Tokyo", month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
  const title = escapeHtml(`${v.streamer} の配信 ${when}`);
  const description = escapeHtml(`${when} JSTの配信アーカイブ（${length}）。${until}まで視聴できます。`);
  const page = `${url.origin}/s/${share.token}`;
  const image = (await env.MEDIA.head(`og/${v.id}.jpg`))
    ? `<meta property="og:image" content="${page}/og.jpg"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630"><meta property="og:image:alt" content="${title}"><meta name="twitter:image" content="${page}/og.jpg"><meta name="twitter:card" content="summary_large_image">`
    : '<meta name="twitter:card" content="summary">';
  return {
    title,
    tags: `<meta name="description" content="${description}"><meta property="og:type" content="video.other"><meta property="og:site_name" content="TranscriptViewer Web"><meta property="og:title" content="${title}"><meta property="og:description" content="${description}"><meta property="og:url" content="${page}"><meta property="og:locale" content="ja_JP"><meta name="twitter:title" content="${title}"><meta name="twitter:description" content="${description}">${image}`,
  };
}

async function challengePage(env, share, url, failed = false) {
  const token = share.token;
  const preview = await previewTags(env, share, url);
  const ready = turnstileConfigured(env);
  const body = ready
    ? `<p>ボットではないことを確認しています。</p>${failed ? '<p class="err" role="alert">確認できませんでした。もう一度お試しください。</p>' : ""}
<form id="f" method="post" action="/s/${token}/verify"><div class="cf-turnstile" data-sitekey="${env.TURNSTILE_SITE_KEY}" data-callback="passed" data-language="ja"></div><noscript><p>JavaScriptを有効にしてください。</p></noscript></form>
<script>function passed(){document.getElementById("f").submit();}</script>
<script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>`
    : "<p>共有リンクは現在利用できません。</p>";
  const html = `<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${preview.title}</title><meta name="robots" content="noindex">${preview.tags}<link rel="icon" type="image/svg+xml" href="${FAVICON}"><style>
:root{color-scheme:dark}body{margin:0;min-height:100dvh;display:grid;place-items:center;font-family:system-ui,-apple-system,"Hiragino Sans",sans-serif;background:#0f1115;color:#e8eaed}main{text-align:center;padding:16px}p{color:#9aa3b2}.err{color:#ff8a8a}</style>
<main>${body}</main></html>`;
  // "origin" (not no-referrer): Turnstile checks the embedding origin, and the token path stays private.
  return new Response(html, {
    status: ready ? (failed ? 403 : 200) : 503,
    headers: { ...PAGE_HEADERS, "Referrer-Policy": "origin", "X-Robots-Tag": "noindex" },
  });
}

async function verifyHuman(request, url, env, share) {
  const token = share.token;
  if (request.method !== "POST" || request.headers.get("Origin") !== url.origin || !turnstileConfigured(env)) {
    return new Response("Bad request", { status: 400, headers: { "Cache-Control": "no-store" } });
  }
  const form = await request.formData().catch(() => null);
  const response = form?.get("cf-turnstile-response");
  if (typeof response !== "string" || !response) return challengePage(env, share, url, true);
  const body = new FormData();
  body.append("secret", env.TURNSTILE_SECRET_KEY);
  body.append("response", response);
  const ip = request.headers.get("CF-Connecting-IP");
  if (ip) body.append("remoteip", ip);
  const result = await fetch(TURNSTILE_VERIFY, { method: "POST", body })
    .then((r) => r.json())
    .catch(() => ({ success: false }));
  if (!result.success || result.hostname !== url.hostname) return challengePage(env, share, url, true);
  const pass = await sign(env, { share: token, exp: nowSeconds() + SHARE_PASS_SECONDS });
  return new Response(null, {
    status: 303,
    headers: {
      Location: `/s/${token}`,
      "Cache-Control": "no-store",
      "Set-Cookie": `${SHARE_PASS_COOKIE}=${pass}; Path=/s/${token}; Max-Age=${SHARE_PASS_SECONDS}; HttpOnly; Secure; SameSite=Lax`,
    },
  });
}
