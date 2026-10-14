// Runs with bun. Entry point: `bun run deploy` deploys the viewer Worker. VIEWER_DOMAIN (custom
// domain, comma separated) comes from deploy.env; secrets come from secrets.json (both untracked).
import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { parseEnvFile } from "../sync/envFile.ts";
import { createNodeFs } from "../sync/nodeFs.ts";

const projectDirectory = resolve(import.meta.dirname, "..", "..");
const environment = parseEnvFile(await createNodeFs().readText(`${projectDirectory}/deploy.env`));
const child = spawn("cf", ["deploy", "--secrets-file", "secrets.json"], {
  cwd: projectDirectory,
  env: { ...process.env, ...environment },
  stdio: "inherit",
});
child.on("close", (code) => process.exit(code ?? 1));
