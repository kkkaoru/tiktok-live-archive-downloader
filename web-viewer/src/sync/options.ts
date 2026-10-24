// Runs with bun. Command-line options of the sync command.
import { parseArgs } from "node:util";

export interface SyncOptions {
  dryRun: boolean;
  dataOnly: boolean;
  // Upload only the originals (media-original/); publish no data or catalog.
  originalsOnly?: boolean;
  // Upload the movies of this catalog video ID only; data and the catalog stay global.
  only?: string;
}

export function parseOptions(argv: string[]): SyncOptions {
  const { values } = parseArgs({
    args: argv,
    options: {
      "dry-run": { type: "boolean" },
      "data-only": { type: "boolean" },
      "originals-only": { type: "boolean" },
      only: { type: "string" },
    },
    strict: true,
  });
  return {
    dryRun: values["dry-run"] === true,
    dataOnly: values["data-only"] === true,
    originalsOnly: values["originals-only"] === true,
    ...(values.only === undefined ? {} : { only: values.only }),
  };
}
