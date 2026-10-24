// Runs with bun. Entry point: `bun run clips:mac -- <videoId|data.json>...` computes clip points on
// the Mac (the default) for web-viewer/data/<id>.json (or a copy of it) and stores them through the
// viewer's bearer PUT.
// It reuses a model server already on PORT or starts `python -m replay.clip_models` in an isolated
// uv environment (the shared .venv is left alone) and stops it afterwards. No Cloudflare AI runs.
import { type ChildProcess, spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { z } from "zod";
import { videoDataSchema } from "../clips.ts";
import { localAi, runMac, viewerOf } from "../clips-mac.ts";
import { parseEnvFile } from "../sync/envFile.ts";

const PORT: number = 8791;
const BASE: string = `http://127.0.0.1:${PORT}`;
const READY_TIMEOUT_MS: number = 20 * 60_000;
const SERVER: string[] = [
  "run",
  "--isolated",
  "--locked",
  "--extra",
  "clips",
  "python",
  "-B",
  "-m",
  "replay.clip_models",
  "--port",
  String(PORT),
];
const projectDirectory = resolve(import.meta.dirname, "..", "..");
const checkout = resolve(projectDirectory, "..");
const secretsSchema = z.object({ CLIPS_API_TOKEN: z.string().min(1) });

/** Our server answers a body it cannot read with 400; nothing listening throws. */
async function serving(): Promise<boolean> {
  const answer = await fetch(`${BASE}/clef`, { method: "POST", body: "{}" }).catch(() => null);
  return answer?.status === 400;
}

function started(): Promise<ChildProcess> {
  const child = spawn("uv", SERVER, { cwd: checkout, stdio: ["ignore", "pipe", "inherit"] });
  return new Promise((done, failed) => {
    const timer = setTimeout(
      () => failed(new Error("the local model server was not ready in time")),
      READY_TIMEOUT_MS,
    );
    child.stdout?.on("data", (chunk: Buffer) => {
      if (!String(chunk).includes(`ready ${PORT}`)) return;
      clearTimeout(timer);
      done(child);
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      failed(new Error(`the local model server exited (${code})`));
    });
  });
}

const ids = process.argv.slice(2);
if (ids.length === 0) {
  console.error("usage: bun run clips:mac -- <videoId|data.json>...");
  process.exit(2);
}
const host = parseEnvFile(readFileSync(`${projectDirectory}/deploy.env`, "utf8"))
  .VIEWER_DOMAIN?.split(",")[0]
  ?.trim();
const { CLIPS_API_TOKEN } = secretsSchema.parse(
  JSON.parse(readFileSync(`${projectDirectory}/secrets.json`, "utf8")),
);
if (host === undefined || host === "") throw new Error("VIEWER_DOMAIN must be set in deploy.env");
const videos = ids.map((id) =>
  videoDataSchema.parse(
    JSON.parse(
      readFileSync(id.endsWith(".json") ? id : `${projectDirectory}/data/${id}.json`, "utf8"),
    ),
  ),
);
const server = (await serving()) ? null : await started();
try {
  const results = await runMac({
    videos,
    ai: localAi(fetch, BASE),
    viewer: viewerOf(fetch, `https://${host}`, CLIPS_API_TOKEN),
    now: new Date(),
  });
  console.log(results.map((r) => JSON.stringify(r)).join("\n"));
} finally {
  server?.kill("SIGTERM");
}
