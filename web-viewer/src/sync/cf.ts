// Runs with bun. Thin wrappers around the authenticated `cf` CLI (R2 listing and small puts).
import { z } from "zod";
import type { ProcessRunner } from "./ports.ts";

const LIST_PAGE_SIZE = "1000";
const MEDIA_PREFIXES: readonly string[] = ["media/", "media-original/"];
const objectListSchema = z.array(z.object({ key: z.string(), size: z.number() }));
const WORKERS_DEV_URL = /https:\/\/[\w.-]+\.workers\.dev/;

interface CfContext {
  runner: ProcessRunner;
  cwd: string;
}

interface ListRequest extends CfContext {
  bucket: string;
}

interface PutRequest extends ListRequest {
  key: string;
  path: string;
}

function parseJsonArray(output: string): unknown {
  const start = output.indexOf("[");
  const end = output.lastIndexOf("]");
  if (start < 0 || end < start) {
    throw new Error("cf returned no JSON list");
  }
  return JSON.parse(output.slice(start, end + 1));
}

async function listPrefix(
  request: ListRequest & { prefix: string },
): Promise<Record<string, number>> {
  const result = await request.runner.run({
    command: "cf",
    args: [
      "r2",
      "objects",
      "list",
      "--bucket-name",
      request.bucket,
      "--prefix",
      request.prefix,
      "--per-page",
      LIST_PAGE_SIZE,
    ],
    cwd: request.cwd,
  });
  if (result.code !== 0) {
    throw new Error(`cf r2 objects list failed: ${result.stderr.slice(0, 300)}`);
  }
  const objects = objectListSchema.parse(parseJsonArray(result.stdout));
  return Object.fromEntries(objects.map((object) => [object.key, object.size]));
}

export async function listRemoteSizes(request: ListRequest): Promise<Record<string, number>> {
  const parts = await Promise.all(
    MEDIA_PREFIXES.map((prefix) => listPrefix({ ...request, prefix })),
  );
  return Object.assign({}, ...parts);
}

export async function putJson(request: PutRequest): Promise<void> {
  const result = await request.runner.run({
    command: "cf",
    args: [
      "r2",
      "objects",
      "put",
      request.key,
      "--bucket-name",
      request.bucket,
      "--file",
      request.path,
      "--content-type",
      "application/json",
    ],
    cwd: request.cwd,
  });
  if (result.code !== 0) {
    throw new Error(`cf r2 objects put failed for ${request.key}: ${result.stderr.slice(0, 300)}`);
  }
}

export function findWorkersUrl(output: string): string | null {
  const match = WORKERS_DEV_URL.exec(output);
  return match === null ? null : match[0];
}
