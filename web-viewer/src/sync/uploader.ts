// Runs with bun. A temporary, token-gated uploader Worker for media larger than the 300 MB
// limit of `cf r2 objects put`: deployed with a fresh random token, always deleted afterwards.
import { createHash } from "node:crypto";
import { basename, join } from "node:path";
import { z } from "zod";
import { findWorkersUrl } from "./cf.ts";
import type { FileSystem, HttpClient, ProcessRunner } from "./ports.ts";

export const CHUNK_BYTES = 64 * 1024 * 1024;
export const PARALLEL_PARTS = 4;
// A failed part aborts the whole multipart upload, so a part rides out a few minutes of outage:
// the retry delay doubles from READY_DELAY_MS (3 + 6 + ... + 96 s, about 3 min in all).
export const PART_ATTEMPTS = 7;
export const READY_ATTEMPTS = 30;
export const READY_DELAY_MS = 3000;
const READY_STATUS = 200;
const readySchema = z.object({ ready: z.literal(true) });
const OK_LIMIT = 300;
// Cloudflare blocks the default HTTP client signature at the edge.
const USER_AGENT = "curl/8.7.1";
const uploadIdSchema = z.object({ uploadId: z.string().min(1) });
const partSchema = z.object({ partNumber: z.number().int(), etag: z.string().min(1) });
const MKV_COPY_CODEC = "h264";

interface Services {
  fs: FileSystem;
  runner: ProcessRunner;
  http: HttpClient;
}

export interface UploaderProject {
  directory: string;
  projectDirectory: string;
  name: string;
  bucket: string;
}

interface DeployRequest {
  services: Pick<Services, "fs" | "runner">;
  project: UploaderProject;
  token: string;
}

interface ReadyRequest {
  http: HttpClient;
  sleep: (milliseconds: number) => Promise<void>;
  url: string;
  token: string;
}

interface UploadRequest {
  services: Pick<Services, "fs" | "http">;
  sleep: (milliseconds: number) => Promise<void>;
  url: string;
  token: string;
  key: string;
  path: string;
  size: number;
}

interface PartRequest extends UploadRequest {
  uploadId: string;
  partNumber: number;
  chunk: Uint8Array;
}

// What R2 holds after an upload: the whole file's SHA-256 and the multipart ETag, which R2
// derives from the parts' MD5s (each checked against the ETag R2 returned for it).
export interface Uploaded {
  parts: number;
  sha256: string;
  etag: string;
}

interface ProxyRequest {
  services: Pick<Services, "fs" | "runner">;
  source: string;
  destination: string;
  cwd: string;
}

export function uploaderConfig(args: { name: string; bucket: string }): string {
  return [
    'import { defineConfig, bindings } from "cf/config";',
    "",
    "export default defineConfig({",
    "  worker: {",
    `    name: ${JSON.stringify(args.name)},`,
    '    entrypoint: "worker.js",',
    '    compatibilityDate: "2025-06-01",',
    "    workersDev: true,",
    `    env: { MEDIA: bindings.r2({ name: ${JSON.stringify(args.bucket)} }) },`,
    "  },",
    "});",
    "",
  ].join("\n");
}

function uploaderPackage(name: string): string {
  const manifest = {
    name,
    private: true,
    type: "module",
    devDependencies: { cf: "1.0.0-beta.16", wrangler: "4.150.0" },
  };
  return `${JSON.stringify(manifest)}\n`;
}

export async function deployUploader({ services, project, token }: DeployRequest): Promise<string> {
  const { fs, runner } = services;
  await fs.mkdir(project.directory);
  await fs.copy({
    from: join(project.projectDirectory, "src", "upload-worker.js"),
    to: join(project.directory, "worker.js"),
  });
  await fs.writeText({
    path: join(project.directory, "cloudflare.config.ts"),
    text: uploaderConfig({ name: project.name, bucket: project.bucket }),
    mode: 0o644,
  });
  await fs.writeText({
    path: join(project.directory, "package.json"),
    text: uploaderPackage(project.name),
    mode: 0o644,
  });
  await fs.writeText({
    path: join(project.directory, "secrets.json"),
    text: JSON.stringify({ UPLOAD_TOKEN: token }),
    mode: 0o600,
  });
  // `cf deploy` needs wrangler installed in the project directory itself.
  await fs.link({
    target: join(project.projectDirectory, "node_modules"),
    path: join(project.directory, "node_modules"),
  });
  const result = await runner.run({
    command: "cf",
    args: ["deploy", "--secrets-file", "secrets.json"],
    cwd: project.directory,
  });
  const url = findWorkersUrl(`${result.stdout}\n${result.stderr}`);
  if (result.code !== 0 || url === null) {
    throw new Error(`Uploader deploy failed: ${result.stderr.slice(-400)}`);
  }
  return url;
}

function authHeaders(token: string): Record<string, string> {
  return { "X-Upload-Token": token, "User-Agent": USER_AGENT };
}

export async function waitUntilReady(request: ReadyRequest): Promise<void> {
  const attempt = async (remaining: number): Promise<void> => {
    if (remaining === 0) {
      throw new Error("Uploader Worker did not become ready");
    }
    const response = await request.http
      .request({
        method: "GET",
        url: `${request.url}/ready`,
        headers: authHeaders(request.token),
        body: null,
      })
      .catch(() => ({ status: 0, body: null, setCookie: null }));
    // Only the Worker's own answer counts: the edge returns generic 404s before it is live.
    if (response.status === READY_STATUS && readySchema.safeParse(response.body).success) {
      return;
    }
    await request.sleep(READY_DELAY_MS);
    return attempt(remaining - 1);
  };
  return attempt(READY_ATTEMPTS);
}

async function postJson(args: {
  http: HttpClient;
  url: string;
  token: string;
  body: unknown;
}): Promise<unknown> {
  const response = await args.http.request({
    method: "POST",
    url: args.url,
    headers: { ...authHeaders(args.token), "Content-Type": "application/json" },
    body: JSON.stringify(args.body),
  });
  if (response.status >= OK_LIMIT) {
    throw new Error(`Uploader request failed with HTTP ${response.status}`);
  }
  return response.body;
}

async function sendPart(
  request: PartRequest,
  remaining: number,
): Promise<z.infer<typeof partSchema>> {
  const { chunk } = request;
  const md5 = createHash("md5").update(chunk).digest("hex");
  const query = `key=${encodeURIComponent(request.key)}&uploadId=${encodeURIComponent(request.uploadId)}&n=${request.partNumber}`;
  const response = await request.services.http
    .request({
      method: "PUT",
      url: `${request.url}/part?${query}`,
      headers: authHeaders(request.token),
      body: chunk,
    })
    .catch(() => ({ status: 0, body: null }));
  const parsed = partSchema.safeParse(response.body);
  const sent = response.status > 0 && response.status < OK_LIMIT && parsed.success;
  if (sent && parsed.data.etag.replaceAll('"', "") === md5) {
    return parsed.data;
  }
  if (remaining <= 1) {
    throw new Error(`Part ${request.partNumber} of ${request.key} failed or did not match its MD5`);
  }
  await request.sleep(READY_DELAY_MS * 2 ** (PART_ATTEMPTS - remaining));
  return sendPart(request, remaining - 1);
}

// PARALLEL_PARTS workers each take the next part as soon as theirs is done: fixed batches of four
// left three connections idle behind the slowest part of every batch.
async function inPool<T, R>(items: T[], run: (item: T) => Promise<R>): Promise<R[]> {
  const results: R[] = [];
  const pending = [...items.entries()];
  const worker = async (): Promise<void> => {
    for (let next = pending.shift(); next !== undefined; next = pending.shift()) {
      results[next[0]] = await run(next[1]);
    }
  };
  const workers = Array.from({ length: Math.min(PARALLEL_PARTS, items.length) }, worker);
  try {
    await Promise.all(workers);
  } catch (error) {
    pending.length = 0; // the upload is aborted: start no further part
    throw error;
  }
  return results;
}

export async function uploadFile(request: UploadRequest): Promise<Uploaded> {
  const { http } = request.services;
  const base = { http, token: request.token };
  const created = uploadIdSchema.parse(
    await postJson({ ...base, url: `${request.url}/create`, body: { key: request.key } }),
  );
  const count = Math.max(1, Math.ceil(request.size / CHUNK_BYTES));
  const numbers = Array.from({ length: count }, (_, index) => index + 1);
  // The pool takes parts in order, so chaining the reads hashes the file in order too.
  const whole = createHash("sha256");
  let reading: Promise<unknown> = Promise.resolve();
  const read = (partNumber: number): Promise<Uint8Array> => {
    const start = (partNumber - 1) * CHUNK_BYTES;
    const length = Math.min(CHUNK_BYTES, request.size - start);
    const chunk = reading.then(() =>
      request.services.fs.readChunk({ path: request.path, start, length }),
    );
    // A failed read aborts the upload through its part; the chain only orders the hashing.
    reading = chunk.then(
      (bytes) => whole.update(bytes),
      () => undefined,
    );
    return chunk;
  };
  try {
    const parts = await inPool(numbers, async (partNumber) =>
      sendPart(
        { ...request, uploadId: created.uploadId, partNumber, chunk: await read(partNumber) },
        PART_ATTEMPTS,
      ),
    );
    await postJson({
      ...base,
      url: `${request.url}/complete`,
      body: { key: request.key, uploadId: created.uploadId, parts },
    });
    await reading;
    const md5s = Buffer.concat(
      parts.map((part) => Buffer.from(part.etag.replaceAll('"', ""), "hex")),
    );
    const etag = `${createHash("md5").update(md5s).digest("hex")}-${count}`;
    return { parts: count, sha256: whole.digest("hex"), etag };
  } catch (error) {
    await postJson({
      ...base,
      url: `${request.url}/abort`,
      body: { key: request.key, uploadId: created.uploadId },
    }).catch(() => null);
    throw error;
  }
}

export async function ensureProxy({
  services,
  source,
  destination,
  cwd,
}: ProxyRequest): Promise<void> {
  const existing = await services.fs.stat(destination);
  if (existing !== null && existing.size > 0) {
    return;
  }
  const probe = await services.runner.run({
    command: "ffprobe",
    args: [
      "-v",
      "error",
      "-select_streams",
      "v:0",
      "-show_entries",
      "stream=codec_name",
      "-of",
      "csv=p=0",
      source,
    ],
    cwd,
  });
  const video =
    probe.stdout.trim() === MKV_COPY_CODEC
      ? ["-c:v", "copy"]
      : ["-c:v", "h264_videotoolbox", "-b:v", "8M"];
  const partial = join(cwd, "proxies", `${basename(destination)}.part.mp4`);
  await services.fs.mkdir(join(cwd, "proxies"));
  const result = await services.runner.run({
    command: "ffmpeg",
    args: [
      "-y",
      "-v",
      "error",
      "-i",
      source,
      ...video,
      "-c:a",
      "aac",
      "-b:a",
      "160k",
      "-movflags",
      "+faststart",
      partial,
    ],
    cwd,
  });
  if (result.code !== 0) {
    throw new Error(`ffmpeg proxy failed: ${result.stderr.slice(0, 300)}`);
  }
  await services.fs.rename({ from: partial, to: destination });
}

export async function deleteUploader(args: {
  services: Pick<Services, "fs" | "runner">;
  project: UploaderProject;
}): Promise<void> {
  await args.services.runner.run({
    command: "cf",
    args: ["workers", "delete", args.project.name, "--force"],
    cwd: args.project.projectDirectory,
  });
  await args.services.fs.removeTree(args.project.directory);
}
