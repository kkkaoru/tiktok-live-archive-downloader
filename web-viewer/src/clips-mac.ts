// Runs with bun. Clip points (切り抜き) computed on the Mac, the default: generate() with clef and
// Ternary-Bonsai served by `python -m replay.clip_models` on 127.0.0.1, stored through the viewer's
// bearer PUT. Cloudflare AI is not used here; it runs only when a user asks for a reload.
import { z } from "zod";
import {
  type Ai,
  type ClipRecord,
  clipRecordSchema,
  GATE_MODEL,
  generate,
  inputHash,
  READ_MODEL,
  REGION_MODEL,
  SUMMARY_MODEL,
  type VideoData,
} from "./clips.ts";

export interface Viewer {
  /** The record viewers see now, or null when there is none for the published movie. */
  current(videoId: string): Promise<ClipRecord | null>;
  store(videoId: string, record: ClipRecord): Promise<void>;
}
export interface MacRun {
  videos: readonly VideoData[];
  ai: Ai;
  viewer: Viewer;
  now: Date;
}
export interface MacResult {
  videoId: string;
  outcome: "written" | "current";
  clips: number;
}
type Fetcher = (
  url: string,
  init: { method: string; headers: Record<string, string>; body?: string },
) => Promise<{
  ok: boolean;
  status: number;
  json(): Promise<unknown>;
}>;

/** The local server's route for each model the stages ask for; any other model is not on the Mac. */
const ROUTES: Record<string, string> = {
  [REGION_MODEL]: "clef",
  [READ_MODEL]: "clef",
  [GATE_MODEL]: "clef",
  [SUMMARY_MODEL]: "bonsai",
};
/** Replaced by the published movie's key when the Worker stores the record. */
const MAC_MOVIE_KEY: string = "mac";
const lookupSchema: z.ZodType<{ status: string; record?: unknown }> = z.object({
  status: z.string(),
  record: z.unknown().optional(),
});

/** Answers every stage from the local server; a model it does not serve fails instead of reaching Cloudflare. */
export function localAi(fetcher: Fetcher, base: string): Ai {
  return {
    run: async (model, input) => {
      const route = ROUTES[model];
      if (route === undefined) throw new Error(`${model} is not served on the Mac`);
      const response = await fetcher(`${base}/${route}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      });
      if (!response.ok) throw new Error(`local ${route} answered ${response.status}`);
      return response.json();
    },
  };
}

/** The viewer's clips API with the CLIPS_API_TOKEN bearer. */
export function viewerOf(fetcher: Fetcher, origin: string, token: string): Viewer {
  const headers = { Authorization: `Bearer ${token}`, "User-Agent": "clips-mac/1" };
  return {
    current: async (videoId) => {
      const response = await fetcher(`${origin}/api/clips/${videoId}.json`, {
        method: "GET",
        headers,
      });
      const found = lookupSchema.safeParse(await response.json());
      const record = found.success && found.data.status === "ready" ? found.data.record : null;
      const parsed = clipRecordSchema.safeParse(record);
      return parsed.success ? parsed.data : null;
    },
    store: async (videoId, record) => {
      const response = await fetcher(`${origin}/api/clips/${videoId}.json`, {
        method: "PUT",
        headers: { ...headers, "Content-Type": "application/json" },
        body: JSON.stringify(record),
      });
      if (!response.ok) throw new Error(`storing clips for ${videoId} answered ${response.status}`);
    },
  };
}

/**
 * Whether a record was made from this data, on either place: then the Mac leaves it, so records
 * Cloudflare made are not redone just for the move.
 */
export async function matches(record: ClipRecord, data: VideoData): Promise<boolean> {
  const hashes = [await inputHash(data, "mac"), await inputHash(data, "cloudflare")];
  return hashes.includes(record.inputHash);
}

/** Computes and stores each video without a matching record, in turn; unchanged summaries are reused. */
export async function runMac(run: MacRun): Promise<MacResult[]> {
  return run.videos.reduce<Promise<MacResult[]>>(async (done, data) => {
    const results = await done;
    const previous = await run.viewer.current(data.id);
    if (previous !== null && (await matches(previous, data))) {
      results.push({ videoId: data.id, outcome: "current", clips: previous.clips.length });
      return results;
    }
    const record = await generate({
      ai: run.ai,
      place: "mac",
      data,
      movieKey: MAC_MOVIE_KEY,
      previous,
      now: run.now,
    });
    await run.viewer.store(data.id, record);
    results.push({ videoId: data.id, outcome: "written", clips: record.clips.length });
    return results;
  }, Promise.resolve([]));
}
