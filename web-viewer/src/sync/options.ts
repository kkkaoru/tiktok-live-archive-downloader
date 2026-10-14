// Runs with bun. Command-line options of the sync command.
import { parseArgs } from "node:util";

export interface SyncOptions {
  dryRun: boolean;
  dataOnly: boolean;
}

export function parseOptions(argv: string[]): SyncOptions {
  const { values } = parseArgs({
    args: argv,
    options: {
      "dry-run": { type: "boolean" },
      "data-only": { type: "boolean" },
    },
    strict: true,
  });
  return { dryRun: values["dry-run"] === true, dataOnly: values["data-only"] === true };
}
