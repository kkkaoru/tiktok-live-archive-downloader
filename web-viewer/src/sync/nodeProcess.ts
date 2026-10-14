// Runs with bun. Real process adapter for the ProcessRunner port; output is captured, never echoed.
import { spawn } from "node:child_process";
import type { ProcessRunner } from "./ports.ts";

export function createNodeRunner(env: NodeJS.ProcessEnv): ProcessRunner {
  return {
    run: ({ command, args, cwd }) =>
      new Promise((resolveRun, reject) => {
        const child = spawn(command, args, { cwd, env, stdio: ["ignore", "pipe", "pipe"] });
        const out: Buffer[] = [];
        const err: Buffer[] = [];
        child.stdout.on("data", (block: Buffer) => out.push(block));
        child.stderr.on("data", (block: Buffer) => err.push(block));
        child.on("error", reject);
        child.on("close", (code) =>
          resolveRun({
            code: code ?? 1,
            stdout: Buffer.concat(out).toString("utf8"),
            stderr: Buffer.concat(err).toString("utf8"),
          }),
        );
      }),
  };
}
