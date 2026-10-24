// Runs with bun. Entry point: `bun run sync [--dry-run] [--data-only]`.
import { randomBytes } from "node:crypto";
import { resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { createNodeFs } from "../sync/nodeFs.ts";
import { createNodeHttp } from "../sync/nodeHttp.ts";
import { createNodeRunner } from "../sync/nodeProcess.ts";
import { parseOptions } from "../sync/options.ts";
import { runSync } from "../sync/run.ts";

const TOKEN_BYTES = 36;
const projectDirectory = resolve(import.meta.dirname, "..", "..");

await runSync({
  options: parseOptions(process.argv.slice(2)),
  services: {
    fs: createNodeFs(),
    runner: createNodeRunner(process.env),
    http: createNodeHttp(),
    logger: { info: (message) => console.log(message) },
    sleep: (milliseconds) => delay(milliseconds),
    newToken: () => randomBytes(TOKEN_BYTES).toString("base64url"),
    projectDirectory,
    clock: () => performance.now(),
  },
});
