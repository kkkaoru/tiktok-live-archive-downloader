// Runs with bun. Entry point: `bun run debug /api/me` prints a post-login page or API response
// fetched through a read-only admin debug session. The token and cookie are never printed.
import { resolve } from "node:path";
import { debugFetch } from "../sync/debugSession.ts";
import { parseEnvFile } from "../sync/envFile.ts";
import { createNodeFs } from "../sync/nodeFs.ts";
import { createNodeHttp } from "../sync/nodeHttp.ts";

const projectDirectory = resolve(import.meta.dirname, "..", "..");
const fs = createNodeFs();
const environment = parseEnvFile(await fs.readText(`${projectDirectory}/deploy.env`));
const path = process.argv[2];
const host = environment.VIEWER_DOMAIN?.split(",")[0];
if (path === undefined || !path.startsWith("/") || host === undefined) {
  console.error("usage: bun run debug /path (VIEWER_DOMAIN must be set in deploy.env)");
  process.exit(2);
}
const result = await debugFetch({
  fs,
  http: createNodeHttp(),
  projectDirectory,
  baseUrl: `https://${host}`,
  path,
});
console.log(JSON.stringify(result, null, 1));
process.exit(result.status >= 200 && result.status < 300 ? 0 : 1);
