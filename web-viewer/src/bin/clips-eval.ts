// Runs with bun. Entry point: `bun run clips:eval -- --label v5 --sets SETS.json --out DIR
// --cache DIR [--records DIR] [--previous DIR] [--calibrate]` evaluates clip points offline.
// Without --records it runs the shipping generate() on web-viewer/data/<id>.json; every model
// call goes through `cf ai run` one at a time and is cached, and nothing is written to R2.
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { z } from "zod";
import {
  type Ai,
  type ClipRecord,
  clipRecordSchema,
  generate,
  utterances,
  videoDataSchema,
} from "../clips.ts";
import {
  cachedAi,
  calibrate,
  calibration,
  evaluate,
  type Meter,
  markdown,
  type Store,
  type Subset,
  type SubsetSummary,
  summarize,
  type VideoEval,
} from "../clips-eval.ts";

const ATTEMPTS: number = 5;
const RETRY_MS: number = 15_000;
const UNKNOWN_KEY: string = "0".repeat(32);
const SUBSETS: Subset[] = ["dev", "heldOut"];
const setsSchema = z.object({ dev: z.array(z.string()), heldOut: z.array(z.string()) });
const runSchema = z.object({ label: z.string(), subsets: z.record(z.string(), z.unknown()) });

const projectDirectory = resolve(import.meta.dirname, "..", "..");
const argsSchema = z.object({
  label: z.string().regex(/^[A-Za-z0-9_.]+$/),
  sets: z.string(),
  out: z.string(),
  cache: z.string(),
  records: z.string().optional(),
  previous: z.string().optional(),
  calibrate: z.boolean().optional(),
});
const { values } = parseArgs({
  options: {
    label: { type: "string" },
    sets: { type: "string" },
    out: { type: "string" },
    cache: { type: "string" },
    records: { type: "string" },
    previous: { type: "string" },
    calibrate: { type: "boolean" },
  },
});
const parsed = argsSchema.safeParse(values);
if (!parsed.success) {
  console.error("usage: bun run clips:eval -- --label L --sets SETS.json --out DIR --cache DIR");
  process.exit(2);
}
const options = parsed.data;
const { label, sets, out, cache } = options;
mkdirSync(out, { recursive: true });
mkdirSync(cache, { recursive: true });

const readJson = (path: string): unknown => JSON.parse(readFileSync(path, "utf8"));
const store: Store = {
  read: async (key) => {
    const path = join(cache, `${key}.json`);
    return existsSync(path) ? readFileSync(path, "utf8") : null;
  },
  write: async (key, text) => writeFileSync(join(cache, `${key}.json`), text),
};
const wait = (ms: number): Promise<void> => new Promise((done) => setTimeout(done, ms));

/** One `cf ai run` call; the CLI holds the credentials, so none pass through here. */
async function cf(model: string, input: unknown, left: number): Promise<unknown> {
  // One body file per process: concurrent harness runs share the cache directory.
  const body = join(cache, `body-${process.pid}.json`);
  writeFileSync(body, JSON.stringify(input));
  const result = spawnSync(
    join(projectDirectory, "node_modules", ".bin", "cf"),
    ["ai", "run", model, "--body", `@${body}`],
    { cwd: projectDirectory, encoding: "utf8", maxBuffer: 1 << 26 },
  );
  if (result.status === 0) return JSON.parse(result.stdout.slice(result.stdout.indexOf("{")));
  if (left <= 1) throw new Error(`cf ai run ${model} failed: ${result.stderr.slice(-300)}`);
  console.error(`retrying ${model}: ${result.stderr.slice(-120).trim()}`);
  await wait(RETRY_MS);
  return cf(model, input, left - 1);
}
const remote: Ai = { run: (model, input) => cf(model, input, ATTEMPTS) };

async function recordOf(id: string, meter: Meter): Promise<ClipRecord> {
  const data = videoDataSchema.parse(readJson(join(projectDirectory, "data", `${id}.json`)));
  const saved = options.previous ? join(options.previous, `${id}.json`) : "";
  return options.records
    ? clipRecordSchema.parse(readJson(join(options.records, `${id}.json`)))
    : generate({
        place: "cloudflare",
        ai: cachedAi(remote, store, meter),
        data,
        movieKey: UNKNOWN_KEY,
        previous: saved && existsSync(saved) ? clipRecordSchema.parse(readJson(saved)) : null,
        now: new Date(),
      });
}

async function videoOf(id: string, subset: Subset): Promise<VideoEval> {
  const meter: Meter = { calls: 0, fresh: 0 };
  const started = Date.now();
  const record = await recordOf(id, meter);
  writeFileSync(join(out, `${label}-${id}.record.json`), JSON.stringify(record, null, 1));
  const data = videoDataSchema.parse(readJson(join(projectDirectory, "data", `${id}.json`)));
  const judgeMeter: Meter = { calls: 0, fresh: 0 };
  const result = await evaluate({
    ai: cachedAi(remote, store, judgeMeter),
    record,
    said: utterances(data.cues),
    subset,
    calls: meter.calls,
  });
  console.error(
    `${subset} ${id.slice(0, 40)} clips ${record.clips.length} calls ${meter.calls} fresh ${meter.fresh} ${Math.round((Date.now() - started) / 1000)} s`,
  );
  return result;
}

const chosen = setsSchema.parse(readJson(sets));
const videos: VideoEval[] = [];
for (const subset of SUBSETS) {
  for (const id of chosen[subset]) videos.push(await videoOf(id, subset));
}
const subsets: Record<Subset, SubsetSummary> = {
  dev: summarize(videos.filter((v) => v.subset === "dev")),
  heldOut: summarize(videos.filter((v) => v.subset === "heldOut")),
};
writeFileSync(join(out, `${label}.json`), JSON.stringify({ label, subsets, videos }, null, 1));

if (options.calibrate) {
  const pairs = [];
  for (const id of chosen.dev) {
    const data = videoDataSchema.parse(readJson(join(projectDirectory, "data", `${id}.json`)));
    const record = clipRecordSchema.parse(readJson(join(out, `${label}-${id}.record.json`)));
    const meter: Meter = { calls: 0, fresh: 0 };
    pairs.push(
      ...(await calibrate({
        ai: cachedAi(remote, store, meter),
        clips: record.clips,
        said: utterances(data.cues),
      })),
    );
  }
  writeFileSync(join(out, `calibration-${label}.json`), JSON.stringify(pairs, null, 1));
  console.log(JSON.stringify(calibration(pairs)));
}

const runs = readdirSync(out)
  .filter((f) => /^[^-]+\.json$/.test(f) && !f.startsWith("calibration"))
  .sort()
  .map((f) => runSchema.parse(readJson(join(out, f))))
  .map((r) => ({
    label: r.label,
    subsets: { dev: summaryOf(r.subsets.dev), heldOut: summaryOf(r.subsets.heldOut) },
  }));
console.log(markdown(runs));

function summaryOf(value: unknown): SubsetSummary {
  return z
    .object({
      videos: z.number(),
      clips: z.number(),
      clipsPerVideo: z.number(),
      problems: z.number(),
      cleanStartRate: z.number(),
      cleanEndRate: z.number(),
      firstCleanRate: z.number(),
      means: z.object({
        topic: z.number(),
        standalone: z.number(),
        hook: z.number(),
        ending: z.number(),
        flow: z.number(),
      }),
      goodRate: z.number(),
      goodPerVideo: z.number(),
      unjudged: z.number(),
      callsPerVideo: z.number(),
    })
    .parse(value);
}
