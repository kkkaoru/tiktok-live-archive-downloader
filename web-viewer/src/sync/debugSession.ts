// Runs with bun. Reads post-login pages as a read-only admin debug session: mints a one-hour
// session cookie with the DEBUG_TOKEN from secrets.json, then fetches a path with it.
import { z } from "zod";
import type { FileSystem, HttpClient } from "./ports.ts";

const MINT_PATH = "/auth/debug";
const MINT_OK = 204;
const SESSION_COOKIE = "tv_session";
const secretsSchema = z.object({ DEBUG_TOKEN: z.string().min(32) });

export interface DebugRequest {
  fs: FileSystem;
  http: HttpClient;
  projectDirectory: string;
  baseUrl: string;
  path: string;
}

export interface DebugFetch {
  mintStatus: number;
  status: number;
  body: unknown;
}

export function cookieFrom(setCookie: string | null): string | null {
  if (setCookie === null) {
    return null;
  }
  const first = setCookie.split(";")[0] ?? "";
  return first.startsWith(`${SESSION_COOKIE}=`) ? first : null;
}

export async function debugFetch(request: DebugRequest): Promise<DebugFetch> {
  const secrets = secretsSchema.parse(
    JSON.parse(await request.fs.readText(`${request.projectDirectory}/secrets.json`)),
  );
  const minted = await request.http.request({
    method: "POST",
    url: `${request.baseUrl}${MINT_PATH}`,
    headers: { "X-Debug-Token": secrets.DEBUG_TOKEN, "User-Agent": "Mozilla/5.0" },
    body: null,
  });
  const cookie = cookieFrom(minted.setCookie);
  if (minted.status !== MINT_OK || cookie === null) {
    return { mintStatus: minted.status, status: 0, body: null };
  }
  const page = await request.http.request({
    method: "GET",
    url: `${request.baseUrl}${request.path}`,
    headers: { Cookie: cookie, "User-Agent": "Mozilla/5.0" },
    body: null,
  });
  return { mintStatus: minted.status, status: page.status, body: page.body };
}
