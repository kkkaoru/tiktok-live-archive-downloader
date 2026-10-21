// Runs with bun (tests) and in the Cloudflare Worker. Clip points (切り抜き) of one video, each
// built as one of two stories from ranges of the same stream, played back to back:
//  - climax-first: the strongest moment opens the clip, then the lead-up that led to it;
//  - hook-to-climax: a question or curious opening first, rising to the strongest moment last.
// Candidate units come from the excitement peaks and the bins just before them (the lead-up),
// cut at utterance boundaries. @cf/cloudflare/clef scores each unit (interest, question, topic
// link to the previous unit) and every possible opening's first 3 seconds; a clip is kept only
// when its opening 3 s is a strong hook, and its first range starts at the strongest opening.
// Every clip lasts 10-30 s or 30-90 s in total. @cf/zai-org/glm-5.3-flash titles and summarizes
// it. Records are cached in R2 per movie content key (size + first/last MiB, the same key as
// replay.excitement.movie_key and the Mac viewer), so an unchanged video is never recomputed.
import { z } from "zod";

export type ClipKind = "short" | "long";
export type Pattern = "climax-first" | "hook-to-climax";
export type Outcome = "written" | "unchanged" | "no-movie" | "no-data";

export interface Range {
  start: number;
  end: number;
}
/** One utterance inside a unit. */
export interface Said {
  start: number;
  text: string;
}
export interface Segment extends Range {
  text: string;
  excitement: number;
  /** The excitement peak (bin index) whose lead-up this unit belongs to. */
  peak: number;
  parts: Said[];
}
export interface Scored extends Segment {
  quality: number;
  question: number;
  linked: boolean;
  /** Strength of the best opening's first 3 s (0-1), and where that opening starts. */
  hook: number;
  hookStart: number;
}
export interface Draft {
  kind: ClipKind;
  pattern: Pattern;
  duration: number;
  score: number;
  ranges: Range[];
}
export interface Clip extends Draft {
  id: string;
  title: string;
  summary: string;
}
export interface Ai {
  run(model: string, input: unknown): Promise<unknown>;
}
interface RangeRead {
  range: { offset: number; length: number };
}
interface StoredObject {
  json(): Promise<unknown>;
  arrayBuffer(): Promise<ArrayBuffer>;
}
export interface Bucket {
  head(key: string): Promise<{ size: number } | null>;
  get(key: string, options?: RangeRead): Promise<StoredObject | null>;
  put(key: string, value: string, options: unknown): Promise<unknown>;
  delete(key: string): Promise<unknown>;
}
export interface ClipEnv {
  AI: Ai;
  MEDIA: Bucket;
  CLIPS_QUEUE: { send(body: unknown): Promise<unknown> };
}
export interface ClipRequest {
  videoId: string;
  force: boolean;
}
export interface Generation {
  ai: Ai;
  data: VideoData;
  movieKey: string;
  previous: ClipRecord | null;
  now: Date;
}
interface RunSearch {
  said: readonly Cue[];
  from: number;
  to: number;
  taken: readonly Range[];
}
export interface LookupRequest {
  videoId: string;
  key: string | null;
  now: Date;
}
interface Run extends Range {
  text: string;
  size: number;
  parts: Said[];
}
interface Story {
  pattern: Pattern;
  /** The opening unit, started at its strongest opening, then the rest in playing order. */
  parts: Scored[];
}
interface Question {
  id: string;
  body: unknown;
}

export const SCORE_MODEL: string = "@cf/cloudflare/clef";
export const SUMMARY_MODEL: string = "@cf/zai-org/glm-5.3-flash";
/** Bump when the algorithm or prompts change; records recompute on the next upload or reload. */
export const ALGORITHM: number = 3;
export const BANDS: Record<ClipKind, readonly [number, number]> = {
  short: [10, 30],
  long: [30, 90],
};
export const MOVIE_KEY: RegExp = /^[0-9a-f]{32}$/;
const SEGMENT_MIN: number = 5;
const SEGMENT_MAX: number = 15;
/** Excitement peaks, each with LEAD_BINS bins of lead-up before it. */
const PEAKS: number = 10;
const LEAD_BINS: number = 3;
/** The first 3 s of a clip are what must hook the viewer. */
const HOOK_SECONDS: number = 3;
/** Openings tried per unit: its first utterance and the next ones. */
const OPENINGS: number = 3;
/** The shortest first range an opening may leave. */
const OPENING_MIN: number = 3;
const HOOK_MIN: number = 0.6;
/** clef accepts at most 64 questions per request. */
const QUESTIONS_PER_REQUEST: number = 64;
const LINKED: number = 0.5;
const SHORT_CLIPS: number = 6;
const LONG_CLIPS: number = 4;
/** At most this many clips tell the story of one peak. */
const PER_PEAK: number = 2;
const INTEREST_WEIGHT: number = 0.7;
const HOOK_WEIGHT: number = 0.4;
const CLIMAX_WEIGHT: number = 0.3;
const KEY_SAMPLE_BYTES: number = 1 << 20;
const KEY_HEX: number = 32;
const MERGE_GAP: number = 1.5;
/** More utterances than ever fit in SEGMENT_MAX seconds; bounds the work per bin. */
const RUN_LOOKAHEAD: number = 64;
const PENDING_SECONDS: number = 600;
const RATE_LIMITED: string = "3021";
const RATE_RETRIES: number = 4;
const RATE_WAIT_MS: number = 20_000;
const SUMMARY_TOKENS: number = 2000;
const LEVELS: string[] = ["退屈", "やや退屈", "普通", "面白い", "とても面白い"];
const HOOK_LEVELS: string[] = ["引きがない", "弱い", "普通", "強い", "非常に強い"];
const VIDEO_ID: RegExp = /^[A-Za-z0-9._-]+$/;
const DATA_KEY: RegExp = /^data\/([A-Za-z0-9._-]+)\.json$/;
const JSON_PUT: unknown = { httpMetadata: { contentType: "application/json" } };
const STORY: Record<Pattern, string> = {
  "climax-first":
    "最初の区間がこの話の山場（一番の盛り上がり）で、続く区間はそこに至るまでの経緯を時系列で見せる構成",
  "hook-to-climax":
    "最初の区間で疑問・気になる起点を見せ、区間が進むにつれて盛り上がり、最後の区間が山場になる構成",
};
const SYSTEM: string =
  'あなたはライブ配信の切り抜き動画の編集者です。与えられた区間（複数ある場合は順に続けて再生される）の文字起こしと盛り上がり（0〜1）と構成から、視聴者が見たくなる日本語のタイトルと要約を作ります。タイトルは最初の3秒の引きを生かし、続きが気になるものにすること。文字起こしにない出来事を作らないこと。JSONだけを返す: {"title": "25字以内", "summary": "120字以内"}';

const cueSchema = z.object({
  start: z.number(),
  end: z.number(),
  text: z.string(),
  speaker: z.string().optional(),
});
const seriesSchema = z.object({ score: z.array(z.number()) });
export const videoDataSchema = z.object({
  id: z.string(),
  cues: z.array(cueSchema),
  excitement: z
    .object({
      binSeconds: z.number().positive(),
      speech: seriesSchema,
      combined: seriesSchema.nullable().optional(),
    })
    .nullable()
    .optional(),
});
const rangeSchema = z.object({ start: z.number(), end: z.number() });
export const clipRecordSchema = z.object({
  version: z.number(),
  videoId: z.string(),
  movieKey: z.string(),
  inputHash: z.string(),
  models: z.object({ score: z.string(), summary: z.string() }),
  createdAt: z.string(),
  /** False when the video had no excitement scores yet, so no candidates could be chosen. */
  excitement: z.boolean().optional(),
  clips: z.array(
    z.object({
      id: z.string(),
      kind: z.enum(["short", "long"]),
      pattern: z.enum(["climax-first", "hook-to-climax"]).optional(),
      duration: z.number(),
      score: z.number(),
      ranges: z.array(rangeSchema).min(1),
      title: z.string(),
      summary: z.string(),
    }),
  ),
});
const clefSchema = z.object({
  answers: z.record(
    z.string(),
    z.union([z.object({ score: z.number() }), z.object({ noul: z.number() })]),
  ),
});
const chatSchema = z.object({
  choices: z.array(z.object({ message: z.object({ content: z.string().nullable() }) })).min(1),
});
const summarySchema = z.object({ title: z.string().min(1), summary: z.string().min(1) });
const ownRequestSchema = z.object({ videoId: z.string(), force: z.boolean().optional() });
const eventSchema = z.object({ object: z.object({ key: z.string() }) });
const pendingSchema = z.object({ at: z.number(), force: z.boolean() });
const pointerSchema = z.object({ movieKey: z.string().regex(MOVIE_KEY) });
const catalogSchema = z.object({
  videos: z.array(
    z.object({ id: z.string(), original: z.object({ movie: z.string() }).nullable().optional() }),
  ),
});

export type VideoData = z.infer<typeof videoDataSchema>;
export type ClipRecord = z.infer<typeof clipRecordSchema>;
type Cue = z.infer<typeof cueSchema>;
type Answers = z.infer<typeof clefSchema>["answers"];
export type Lookup =
  | { status: "ready"; record: ClipRecord }
  | { status: "pending"; force: boolean }
  | { status: "mismatch" }
  | { status: "missing" };

const round = (value: number): number => Math.round(value * 1000) / 1000;
const length = (ranges: readonly Range[]): number =>
  round(ranges.reduce((total, r) => total + r.end - r.start, 0));
const pad = (value: number): string => String(value).padStart(2, "0");
const clock = (seconds: number): string =>
  `${Math.floor(seconds / 3600)}:${pad(Math.floor((seconds % 3600) / 60))}:${pad(Math.floor(seconds % 60))}`;
const byStart = (a: Range, b: Range): number => a.start - b.start;
const byScore = (a: Draft, b: Draft): number => b.score - a.score;
const byFirstRange = (a: Draft, b: Draft): number =>
  (a.ranges[0]?.start ?? 0) - (b.ranges[0]?.start ?? 0);
const byExcitementThenLater = (a: Scored, b: Scored): number =>
  b.excitement - a.excitement || b.start - a.start;
const recordKey = (movieKey: string): string => `clips/${movieKey}.json`;
const pointerKey = (videoId: string): string => `clips/videos/${videoId}.json`;
const pendingKey = (videoId: string): string => `clips/pending/${videoId}.json`;
const hex = (bytes: ArrayBuffer): string =>
  [...new Uint8Array(bytes)].map((b) => b.toString(16).padStart(2, "0")).join("");
const scoreOf = (answers: Answers, id: string, levels: readonly string[]): number => {
  const answer = answers[id];
  return answer && "score" in answer ? answer.score / (levels.length - 1) : 0;
};
const noulOf = (answers: Answers, id: string): number => {
  const answer = answers[id];
  return answer && "noul" in answer ? answer.noul : 0;
};

/** Same merge as the viewers: cumulative captions collapse to their final text. */
export function utterances(cues: readonly Cue[]): Cue[] {
  return cues.reduce<Cue[]>((out, cue) => {
    const last = out.at(-1);
    const grows =
      last !== undefined &&
      cue.start <= last.end + MERGE_GAP &&
      last.text !== "" &&
      cue.text.startsWith(last.text) &&
      cue.speaker === last.speaker;
    if (grows) out.splice(-1, 1, { ...last, end: Math.max(last.end, cue.end), text: cue.text });
    else out.push({ ...cue });
    return out;
  }, []);
}

export function bandOf(duration: number): ClipKind | null {
  if (duration >= BANDS.short[0] && duration <= BANDS.short[1]) return "short";
  return duration > BANDS.long[0] && duration <= BANDS.long[1] ? "long" : null;
}

/** Runs of whole utterances from `first` on that stay within SEGMENT_MAX seconds. */
function runsFrom(said: readonly Cue[], first: number): Run[] {
  const head = said[first];
  if (!head) return [];
  const tail = said
    .slice(first, first + RUN_LOOKAHEAD)
    .filter((u) => u.end - head.start <= SEGMENT_MAX);
  return tail.map((last, index) => {
    const parts = tail.slice(0, index + 1).map((u) => ({ start: u.start, text: u.text }));
    const text = parts.map((p) => p.text);
    return {
      start: head.start,
      end: last.end,
      text: text.join(" "),
      size: text.join("").length,
      parts,
    };
  });
}

/** The longest-text run that starts in the bin, lasts SEGMENT_MIN+ s and overlaps nothing taken. */
function bestRun({ said, from, to, taken }: RunSearch): Run | null {
  const free = (run: Range): boolean =>
    taken.every((r) => run.end <= r.start || run.start >= r.end);
  const runs = said
    .flatMap((u, index) => (u.start >= from && u.start < to ? runsFrom(said, index) : []))
    .filter((run) => run.end - run.start >= SEGMENT_MIN && free(run));
  return runs.reduce<Run | null>((best, run) => (best && best.size >= run.size ? best : run), null);
}

/** The strongest bins, at least LEAD_BINS + 1 apart so their lead-ups do not overlap. */
export function peaksOf(score: readonly number[]): number[] {
  const order = [...score.keys()].sort((a, b) => (score[b] ?? 0) - (score[a] ?? 0) || a - b);
  return order.reduce<number[]>((chosen, bin) => {
    const apart = chosen.every((other) => Math.abs(other - bin) > LEAD_BINS);
    if (chosen.length < PEAKS && apart && (score[bin] ?? 0) > 0) chosen.push(bin);
    return chosen;
  }, []);
}

/** Units for every peak: one per bin from LEAD_BINS before the peak up to the peak, in time order. */
export function candidates(data: VideoData): Segment[] {
  const excitement = data.excitement;
  if (!excitement) return [];
  const score = excitement.combined?.score ?? excitement.speech.score;
  const width = excitement.binSeconds;
  const said = utterances(data.cues).filter((u) => u.text.trim() !== "" && u.end > u.start);
  const bins = peaksOf(score).flatMap((peak) =>
    Array.from({ length: LEAD_BINS + 1 }, (_, back) => ({ peak, bin: peak - back })).filter(
      ({ bin }) => bin >= 0,
    ),
  );
  const chosen = bins.reduce<Segment[]>((taken, { peak, bin }) => {
    const run = bestRun({ said, from: bin * width, to: (bin + 1) * width, taken });
    if (run) {
      taken.push({
        start: round(run.start),
        end: round(run.end),
        text: run.text,
        excitement: score[bin] ?? 0,
        peak,
        parts: run.parts,
      });
    }
    return taken;
  }, []);
  return chosen.sort(byStart);
}

/** The openings a unit may start at: its first utterances, each leaving OPENING_MIN+ s. */
export function openings(segment: Segment): Said[] {
  return segment.parts
    .slice(0, OPENINGS)
    .filter((p) => segment.end - p.start >= OPENING_MIN)
    .map((p) => ({
      start: p.start,
      text: segment.parts
        .filter((q) => q.start >= p.start && q.start < p.start + HOOK_SECONDS)
        .map((q) => q.text)
        .join(" "),
    }));
}

/** Every question about the units: interest, question, topic link and each opening's hook. */
export function questions(segments: readonly Segment[]): Question[] {
  return segments.flatMap((s, i) => {
    const previous = segments[i - 1];
    const own: Question[] = [
      {
        id: `clip.${i}`,
        body: {
          type: "score",
          instructions: `区間${i}は、ライブ配信の切り抜き動画の一部として面白いか（会話の山場・笑い・驚き・感情の動きがあるか）`,
          criteria: LEVELS,
        },
      },
      {
        id: `question.${i}`,
        body: {
          type: "noul",
          instructions: `区間${i}は、疑問・謎・気になる問いかけ・意外な前振りなど、続きを知りたくなる話の起点になっているか`,
        },
      },
      ...openings(s).map((opening, k) => ({
        id: `hook.${i}.${k}`,
        body: {
          type: "score",
          instructions: `区間${i}をここから見せたとき、最初の${HOOK_SECONDS}秒「${opening.text}」は、視聴者が思わず続きを見たくなる強い引き（疑問・驚き・意外な一言・感情の爆発）か。前置き・あいさつ・相づちだけなら弱い`,
          criteria: HOOK_LEVELS,
        },
      })),
    ];
    const link: Question[] =
      previous?.peak === s.peak
        ? [
            {
              id: `link.${i}`,
              body: {
                type: "noul",
                instructions: `区間${i}は区間${i - 1}と同じ話題・出来事の流れの中にあり、続けて見ると話がつながるか`,
              },
            },
          ]
        : [];
    return [...own, ...link];
  });
}

/** clef requests of at most QUESTIONS_PER_REQUEST questions over the same units. */
export function clefRequests(segments: readonly Segment[]): unknown[] {
  const all = questions(segments);
  const state = {
    description:
      "ライブ配信の文字起こしから選んだ候補区間。excitementは配信内での盛り上がり（0〜1）。同じpeakの区間は同じ山場とその前の流れ",
    segments: segments.map((s, i) => ({
      id: i,
      peak: s.peak,
      time: `${clock(s.start)}〜${clock(s.end)}`,
      excitement: Math.round(s.excitement * 100) / 100,
      text: s.text,
    })),
  };
  return Array.from({ length: Math.ceil(all.length / QUESTIONS_PER_REQUEST) }, (_, n) => ({
    model: "clef",
    state,
    questions: Object.fromEntries(
      all
        .slice(n * QUESTIONS_PER_REQUEST, (n + 1) * QUESTIONS_PER_REQUEST)
        .map((q) => [q.id, q.body]),
    ),
  }));
}

export function scored(segments: readonly Segment[], responses: readonly unknown[]): Scored[] {
  const answers = Object.assign({}, ...responses.map((r) => clefSchema.parse(r).answers));
  return segments.map((s, i) => {
    const hooks = openings(s).map((opening, k) => ({
      start: opening.start,
      hook: scoreOf(answers, `hook.${i}.${k}`, HOOK_LEVELS),
    }));
    const best = hooks.reduce((b, h) => (h.hook > b.hook ? h : b), { start: s.start, hook: 0 });
    return {
      ...s,
      quality: round(
        INTEREST_WEIGHT * scoreOf(answers, `clip.${i}`, LEVELS) +
          (1 - INTEREST_WEIGHT) * s.excitement,
      ),
      question: round(noulOf(answers, `question.${i}`)),
      linked: noulOf(answers, `link.${i}`) >= LINKED,
      hook: round(best.hook),
      hookStart: best.start,
    };
  });
}

/** Consecutive units of the same peak that clef linked into one flow. */
function chainsOf(segments: readonly Scored[]): Scored[][] {
  return segments.reduce<Scored[][]>((chains, s) => {
    const chain = chains.at(-1);
    if (chain && s.linked && chain[0]?.peak === s.peak) chain.push(s);
    else chains.push([s]);
    return chains;
  }, []);
}

/** The first range starts at the opening with the strongest hook. */
function rangesOf(story: Story): Range[] {
  return story.parts.map((p, i) => ({ start: i === 0 ? p.hookStart : p.start, end: p.end }));
}

/** Adds `extra` units (most wanted first) while the story stays within `cap` seconds. */
function fill(fixed: readonly Scored[], extra: readonly Scored[], cap: number): Scored[] {
  return extra.reduce<Scored[]>((kept, unit) => {
    if (length(fixed) + length(kept) + unit.end - unit.start <= cap) kept.push(unit);
    return kept;
  }, []);
}

/** Climax-first: the climax opens (it must hook), then its lead-up in time order. */
function climaxFirst(chain: readonly Scored[], climax: Scored, cap: number): Story[] {
  const before = chain.filter((u) => u.start < climax.start).reverse();
  const lead = fill([climax], before, cap).sort(byStart);
  return climax.hook >= HOOK_MIN && lead.length > 0
    ? [{ pattern: "climax-first", parts: [climax, ...lead] }]
    : [];
}

/** Hook-to-climax: the most curious hooking unit before the climax opens, the climax ends. */
function hookToClimax(chain: readonly Scored[], climax: Scored, cap: number): Story[] {
  const openers = chain
    .filter((u) => u.start < climax.start && u.hook >= HOOK_MIN)
    .sort((a, b) => b.question + b.hook - (a.question + a.hook));
  const opener = openers[0];
  if (!opener) return [];
  const middle = chain.filter((u) => u.start > opener.start && u.start < climax.start).reverse();
  const kept = fill([opener, climax], middle, cap).sort(byStart);
  return [{ pattern: "hook-to-climax", parts: [opener, ...kept, climax] }];
}

function draft(story: Story): Draft[] {
  const ranges = rangesOf(story);
  const duration = length(ranges);
  const kind = bandOf(duration);
  const first = story.parts[0];
  const climax = story.parts.reduce((c, p) => (p.excitement > c.excitement ? p : c));
  const quality = story.parts.reduce((t, p) => t + p.quality, 0) / story.parts.length;
  const score = round(
    HOOK_WEIGHT * (first?.hook ?? 0) +
      (1 - HOOK_WEIGHT - CLIMAX_WEIGHT) * quality +
      CLIMAX_WEIGHT * climax.excitement,
  );
  return kind ? [{ kind, pattern: story.pattern, duration, score, ranges }] : [];
}

/** Both stories of every linked chain, each as a short and a long cut where they fit. */
function stories(chain: readonly Scored[]): Draft[] {
  const climax = [...chain].sort(byExcitementThenLater)[0];
  if (!climax || chain.length < 2) return [];
  return [BANDS.short[1], BANDS.long[1]].flatMap((cap) =>
    [...climaxFirst(chain, climax, cap), ...hookToClimax(chain, climax, cap)].flatMap(draft),
  );
}

/**
 * Every clip is one of the two stories over a linked chain of one peak's units; the best are
 * kept per band, at most PER_PEAK per peak so one moment does not fill the list.
 */
export function compose(segments: readonly Scored[]): Draft[] {
  const all = chainsOf(segments).flatMap((chain) =>
    stories(chain).map((d) => ({ draft: d, peak: chain[0]?.peak ?? 0 })),
  );
  const unique = [...new Map(all.map((c) => [JSON.stringify(c.draft.ranges), c])).values()];
  const pick = (kind: ClipKind, count: number): Draft[] =>
    unique
      .filter((c) => c.draft.kind === kind)
      .sort((a, b) => byScore(a.draft, b.draft))
      .reduce<{ draft: Draft; peak: number }[]>((kept, c) => {
        const same = kept.filter((k) => k.peak === c.peak).length;
        if (kept.length < count && same < PER_PEAK) kept.push(c);
        return kept;
      }, [])
      .map((c) => c.draft);
  return [...pick("long", LONG_CLIPS), ...pick("short", SHORT_CLIPS)].sort(byFirstRange);
}

/** The words spoken in a range: its unit's utterances from the range start on. */
function spoken(range: Range, segments: readonly Segment[]): Segment | undefined {
  return segments.find((s) => s.start <= range.start && range.end <= s.end);
}

export function summaryRequest(clip: Draft, segments: readonly Segment[]): unknown {
  const parts = clip.ranges.map((r) => {
    const s = spoken(r, segments);
    return {
      time: `${clock(r.start)}〜${clock(r.end)}`,
      excitement: s?.excitement ?? 0,
      text: (s?.parts ?? [])
        .filter((p) => p.start >= r.start)
        .map((p) => p.text)
        .join(" "),
    };
  });
  return {
    messages: [
      { role: "system", content: SYSTEM },
      {
        role: "user",
        content: JSON.stringify({
          story: STORY[clip.pattern],
          seconds: clip.duration,
          ranges: parts,
        }),
      },
    ],
    reasoning_effort: "low",
    max_completion_tokens: SUMMARY_TOKENS,
    response_format: { type: "json_object" },
  };
}

export function parseSummary(response: unknown): { title: string; summary: string } {
  const content = chatSchema.parse(response).choices[0]?.message.content ?? "";
  return summarySchema.parse(
    JSON.parse(content.slice(content.indexOf("{"), content.lastIndexOf("}") + 1)),
  );
}

const sha256 = async (bytes: Uint8Array<ArrayBuffer> | ArrayBuffer): Promise<string> =>
  hex(await crypto.subtle.digest("SHA-256", bytes));

/** Everything the result depends on: the algorithm, both models, the cues and the scores. */
export const inputHash = (data: VideoData): Promise<string> =>
  sha256(
    new TextEncoder().encode(
      JSON.stringify({
        algorithm: ALGORITHM,
        models: [SCORE_MODEL, SUMMARY_MODEL],
        cues: data.cues.map((c) => [c.start, c.end, c.text, c.speaker ?? null]),
        excitement: data.excitement ?? null,
      }),
    ),
  );

/** Runs the requests one after another: parallel calls hit Workers AI's per-minute limit. */
async function inTurn(ai: Ai, model: string, inputs: readonly unknown[]): Promise<unknown[]> {
  return inputs.reduce<Promise<unknown[]>>(async (done, input) => {
    const list = await done;
    list.push(await ai.run(model, input));
    return list;
  }, Promise.resolve([]));
}

/**
 * Builds the record; summaries of clips whose story and ranges are unchanged since `previous`
 * are reused, so a reload only pays for the summaries that actually changed.
 */
export async function generate({
  ai,
  data,
  movieKey,
  previous,
  now,
}: Generation): Promise<ClipRecord> {
  const segments = candidates(data);
  const drafts =
    segments.length === 0
      ? []
      : compose(scored(segments, await inTurn(ai, SCORE_MODEL, clefRequests(segments))));
  const known = new Map(
    (previous?.clips ?? []).map((c) => [JSON.stringify([c.pattern, c.ranges]), c]),
  );
  const summaries = await drafts.reduce<Promise<{ title: string; summary: string }[]>>(
    async (done, d) => {
      const list = await done;
      list.push(
        known.get(JSON.stringify([d.pattern, d.ranges])) ??
          parseSummary(await ai.run(SUMMARY_MODEL, summaryRequest(d, segments))),
      );
      return list;
    },
    Promise.resolve([]),
  );
  const clips = drafts.map((d, i) => ({
    id: `c${i + 1}`,
    ...d,
    title: summaries[i]?.title ?? "",
    summary: summaries[i]?.summary ?? "",
  }));
  return {
    version: ALGORITHM,
    videoId: data.id,
    movieKey,
    inputHash: await inputHash(data),
    models: { score: SCORE_MODEL, summary: SUMMARY_MODEL },
    createdAt: now.toISOString(),
    excitement: Boolean(data.excitement),
    clips,
  };
}

/** Retries a Workers AI call that hit the per-minute rate limit (error 3021). */
export function patient(ai: Ai, wait: (ms: number) => Promise<unknown>): Ai {
  const attempt = async (model: string, input: unknown, left: number): Promise<unknown> => {
    try {
      return await ai.run(model, input);
    } catch (error) {
      if (left === 0 || !String(error).includes(RATE_LIMITED)) throw error;
      await wait(RATE_WAIT_MS);
      return attempt(model, input, left - 1);
    }
  };
  return { run: (model, input) => attempt(model, input, RATE_RETRIES) };
}

const sleep = (ms: number): Promise<unknown> => new Promise((done) => setTimeout(done, ms));

/** The content key of an R2 movie without downloading it: two 1 MiB range reads. */
export async function movieKey(bucket: Bucket, key: string): Promise<string | null> {
  const head = await bucket.head(key);
  if (!head) return null;
  const read = async (offset: number): Promise<ArrayBuffer> => {
    const object = await bucket.get(key, {
      range: { offset, length: Math.min(KEY_SAMPLE_BYTES, head.size - offset) },
    });
    return head.size > offset && object ? object.arrayBuffer() : new ArrayBuffer(0);
  };
  const parts = [
    String(head.size),
    await read(0),
    await read(Math.max(head.size - KEY_SAMPLE_BYTES, 0)),
  ];
  return (await sha256(await new Blob(parts).arrayBuffer())).slice(0, KEY_HEX);
}

async function readJson(bucket: Bucket, key: string): Promise<unknown> {
  const object = await bucket.get(key);
  return object ? object.json() : null;
}

const putJson = (bucket: Bucket, key: string, value: unknown): Promise<unknown> =>
  bucket.put(key, JSON.stringify(value), JSON_PUT);

async function readRecord(bucket: Bucket, movie: string): Promise<ClipRecord | null> {
  const parsed = clipRecordSchema.safeParse(await readJson(bucket, recordKey(movie)));
  return parsed.success ? parsed.data : null;
}

/** The movie the cues follow: the edited one, or the original of a recording not edited yet. */
async function movieOf(bucket: Bucket, videoId: string): Promise<string | null> {
  const edited = `media/${videoId}.mp4`;
  if (await bucket.head(edited)) return edited;
  const catalog = catalogSchema.safeParse(await readJson(bucket, "catalog.json"));
  const original = catalog.success
    ? catalog.data.videos.find((v) => v.id === videoId)?.original
    : null;
  return original ? `media-original/${original.movie}` : null;
}

/** Queue step for one video; idempotent, so a duplicate message costs a few reads. */
export async function refresh(env: ClipEnv, request: ClipRequest, now: Date): Promise<Outcome> {
  const raw = VIDEO_ID.test(request.videoId)
    ? await readJson(env.MEDIA, `data/${request.videoId}.json`)
    : null;
  if (!raw) return "no-data";
  const data = videoDataSchema.parse(raw);
  const movie = await movieOf(env.MEDIA, request.videoId);
  const key = movie ? await movieKey(env.MEDIA, movie) : null;
  if (!key) return "no-movie";
  const previous = await readRecord(env.MEDIA, key);
  const stale = request.force || previous?.inputHash !== (await inputHash(data));
  const record = stale
    ? await generate({ ai: patient(env.AI, sleep), data, movieKey: key, previous, now })
    : null;
  await Promise.all([
    record ? putJson(env.MEDIA, recordKey(key), record) : null,
    putJson(env.MEDIA, pointerKey(request.videoId), { movieKey: key }),
  ]);
  await env.MEDIA.delete(pendingKey(request.videoId));
  return record ? "written" : "unchanged";
}

/** A queue message: our own request, or an R2 event for an uploaded data/<id>.json. */
export function requestOf(body: unknown): ClipRequest | null {
  const own = ownRequestSchema.safeParse(body);
  if (own.success) return { videoId: own.data.videoId, force: own.data.force === true };
  const event = eventSchema.safeParse(body);
  const id = event.success ? event.data.object.key.match(DATA_KEY)?.[1] : undefined;
  return id ? { videoId: id, force: false } : null;
}

/** Queues a computation unless an equal or stronger one was queued in the last PENDING_SECONDS. */
export async function enqueue(env: ClipEnv, request: ClipRequest, now: Date): Promise<void> {
  if (!VIDEO_ID.test(request.videoId)) return;
  const at = Math.floor(now.getTime() / 1000);
  const pending = pendingSchema.safeParse(await readJson(env.MEDIA, pendingKey(request.videoId)));
  const queued =
    pending.success &&
    at - pending.data.at < PENDING_SECONDS &&
    (pending.data.force || !request.force);
  if (queued) return;
  await putJson(env.MEDIA, pendingKey(request.videoId), { at, force: request.force });
  await env.CLIPS_QUEUE.send(request);
}

/**
 * Stored clips of a video. `key` (a viewer's own movie key) reads that content's record directly;
 * without it the key last computed for the video is used. A forced recomputation reads as pending,
 * and a key other than the published movie's reads as a mismatch.
 */
export async function lookup(env: ClipEnv, request: LookupRequest): Promise<Lookup> {
  const { videoId, key, now } = request;
  if (!VIDEO_ID.test(videoId)) return { status: "missing" };
  const [queued, pointer] = await Promise.all([
    readJson(env.MEDIA, pendingKey(videoId)).then((v) => pendingSchema.safeParse(v)),
    readJson(env.MEDIA, pointerKey(videoId)).then((v) => pointerSchema.safeParse(v)),
  ]);
  // A marker older than PENDING_SECONDS belongs to a computation that gave up: queue again.
  const pending =
    queued.success && now.getTime() / 1000 - queued.data.at < PENDING_SECONDS ? queued : null;
  if (pending?.data.force) return { status: "pending", force: true };
  const movie = key ?? (pointer.success ? pointer.data.movieKey : null);
  const record = movie ? await readRecord(env.MEDIA, movie) : null;
  if (record) return { status: "ready", record };
  if (pending) return { status: "pending", force: false };
  // The viewer's movie is not the one published: computing again would not produce its key.
  return pointer.success && key !== null ? { status: "mismatch" } : { status: "missing" };
}
