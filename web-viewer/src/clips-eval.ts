// Runs with bun. Offline evaluation harness for clip points (切り抜き). It runs the shipping
// `generate()` (or reads saved records), checks every record against the hard rules, measures
// whether ranges start and end on sentence boundaries, and asks a judge model from another
// family than the pipeline's to grade each clip with a fixed rubric. Model responses are cached
// by model + input, so re-running an unchanged algorithm costs nothing. Calibration grades
// synthetic negatives of real clips (an unrelated range spliced in, edges cut mid-sentence)
// that the judge must score lower before its grades are trusted.
import { createHash } from "node:crypto";
import { z } from "zod";
import { type Ai, BANDS, type ClipRecord, type Range, type Said, safeJson } from "./clips.ts";

export type Rule = "band" | "overlap" | "replay" | "pattern";
export type Negative = "spliced" | "cut";
export type Subset = "dev" | "heldOut";
type Rubric = "topic" | "standalone" | "hook" | "ending" | "flow";
type EvalClip = ClipRecord["clips"][number];

export interface Problem {
  clip: string;
  rule: Rule;
  detail: string;
}
/** How a clip's ranges meet the stream's sentences. */
export interface Shape {
  ranges: number;
  cleanStarts: number;
  cleanEnds: number;
  /** Whether the first range, the one the viewer hears first, starts cleanly. */
  firstClean: boolean;
}
export interface Grade {
  topic: number;
  standalone: number;
  hook: number;
  ending: number;
  flow: number;
  /** Numbers of the played utterances the judge found off the clip's topic. */
  offTopic: number[];
  reason: string;
}
export interface ClipEval {
  id: string;
  kind: EvalClip["kind"];
  duration: number;
  shape: Shape;
  grade: Grade | null;
}
export interface VideoEval {
  id: string;
  subset: Subset;
  clips: ClipEval[];
  problems: Problem[];
  /** Model calls the record cost, cached or not. */
  calls: number;
}
export interface SubsetSummary {
  videos: number;
  clips: number;
  clipsPerVideo: number;
  problems: number;
  cleanStartRate: number;
  cleanEndRate: number;
  firstCleanRate: number;
  means: Record<Rubric, number>;
  /** Share of graded clips that hold one topic, stand alone, end and flow acceptably. */
  goodRate: number;
  goodPerVideo: number;
  unjudged: number;
  callsPerVideo: number;
}
export interface Store {
  read(key: string): Promise<string | null>;
  write(key: string, text: string): Promise<void>;
}
export interface Meter {
  calls: number;
  fresh: number;
}
export interface Pair {
  clip: string;
  negative: Negative;
  original: Grade | null;
  altered: Grade | null;
}
export interface Calibration {
  pairs: number;
  /** Pairs the judge ranked correctly, per negative kind. */
  correct: Record<Negative, number>;
  total: Record<Negative, number>;
}
export interface Played {
  number: number;
  range: number;
  text: string;
}

export const JUDGE_MODEL: string = "@cf/deepseek-ai/deepseek-v4.1-flash-1017";
const JUDGE_TOKENS: number = 6000;
/** Within this many seconds an edge and an utterance boundary are the same moment. */
const EDGE_SLACK: number = 0.05;
/** A silence this long ends a sentence even when the next utterance is not a clean start. */
const SENTENCE_PAUSE: number = 3;
const GOOD_TOPIC: number = 4;
const GOOD_REST: number = 3;
const SPLICE_SECONDS: number = 10;
const SPLICE_DISTANCE: number = 600;
const NEGATIVES: Negative[] = ["spliced", "cut"];
const STORIES: Record<NonNullable<EvalClip["pattern"]>, string> = {
  "climax-first": "山場→経緯",
  "hook-to-climax": "引き→山場",
};
const JUDGE_SYSTEM: string = [
  "あなたはライブ配信の切り抜き動画を審査する編集長です。切り抜きは配信のいくつかの区間を、示した順に続けて再生したものです（区間の間は編集でつながります）。",
  "文字起こしだけを根拠に、次の5項目を1〜5の整数で採点します。音声認識の誤字は減点しません。",
  "- topic: 全体が一つの話題・出来事だけを扱っているか。5=最初から最後まで同じ話題、3=主題はあるが無関係な発話が少し混じる、1=無関係な二つ以上の話題が混在する。",
  "- standalone: 配信を見ていない人がこの切り抜きだけで何の話か理解できるか。5=前提がすべて含まれる、3=少し分かりにくい、1=何の話か分からない。",
  "- hook: 最初の1〜2発話（再生開始3秒）が続きを見たくさせるか。5=強い疑問・驚き・感情の爆発、3=普通、1=あいさつ・相づち・前置きだけ。",
  "- ending: 最後が話のオチ・結論・反応で区切れているか。5=きれいにオチる、3=一応区切れる、1=文や話の途中で切れる。",
  "- flow: 発話のつながりが自然で話が飛ばないか。5=自然、3=少し飛ぶ、1=支離滅裂。区間が一つでも、文の途中から始まる・途中で終わるなら下げる。",
  "構成が示されたら、その構成どおりの区間の順番は減点しない。山場→経緯は、最初の区間が山場で、続く区間がそこに至る経緯を時系列で見せ、ループ再生で山場に戻る構成なので、endingは経緯が山場の直前まで自然につながっているかで判断する。引き→山場は、疑問や気になる起点から始まり最後の区間が山場になる構成。",
  "offTopicには主題と無関係な発話の番号を入れる（なければ空配列）。reasonは40字以内の日本語。",
  'JSONだけを返す: {"topic": 1-5, "standalone": 1-5, "hook": 1-5, "ending": 1-5, "flow": 1-5, "offTopic": [番号], "reason": "..."}',
].join("\n");

const score = z.number().int().min(1).max(5);
const gradeSchema = z.object({
  topic: score,
  standalone: score,
  hook: score,
  ending: score,
  flow: score,
  offTopic: z.array(z.number().int()),
  reason: z.string(),
});
const chatSchema = z.object({
  choices: z.array(z.object({ message: z.object({ content: z.string().nullable() }) })).min(1),
});

const round = (value: number): number => Math.round(value * 1000) / 1000;
const share = (part: number, whole: number): number => (whole === 0 ? 0 : round(part / whole));
const mean = (values: readonly number[]): number =>
  share(
    values.reduce((t, v) => t + v, 0),
    values.length,
  );
const clock = (seconds: number): string =>
  `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
const touches = (a: Range, b: Range): boolean => !(a.end < b.start || b.end < a.start);
const replays = (a: Range, b: Range): boolean =>
  a.end > b.start + EDGE_SLACK && b.end > a.start + EDGE_SLACK;
const problem = (clip: string, rule: Rule, detail: string): Problem => ({ clip, rule, detail });
const isGood = (g: Grade): boolean =>
  g.topic >= GOOD_TOPIC &&
  g.standalone >= GOOD_REST &&
  g.ending >= GOOD_REST &&
  g.flow >= GOOD_REST;
const heard = (r: Range, said: readonly Said[]): Said[] =>
  said.filter((u) => u.start >= r.start - EDGE_SLACK && u.start < r.end - EDGE_SLACK);

/** Rules every record must keep: the bands, the two stories, no shared moment between clips. */
export function invariants(record: ClipRecord): Problem[] {
  const ranges = record.clips.flatMap((c) => c.ranges.map((r) => ({ clip: c.id, ...r })));
  // A clip's own ranges may meet (a climax-first lead-up ends where the climax starts) but
  // never overlap, or that moment plays twice.
  const overlaps = ranges.flatMap((a, i) =>
    ranges
      .slice(i + 1)
      .filter((b) => (b.clip === a.clip ? replays(a, b) : touches(a, b)))
      .map((b) =>
        problem(
          a.clip,
          b.clip === a.clip ? "replay" : "overlap",
          `${a.clip} ${clock(a.start)}-${clock(a.end)} touches ${b.clip} ${clock(b.start)}-${clock(b.end)}`,
        ),
      ),
  );
  const shaped = record.clips.flatMap((c) => {
    const total = c.ranges.reduce((t, r) => t + r.end - r.start, 0);
    const fits =
      (total >= BANDS.short[0] && total <= BANDS.short[1]) ||
      (total > BANDS.long[0] && total <= BANDS.long[1]);
    return [
      ...(fits ? [] : [problem(c.id, "band", `${round(total)} s`)]),
      ...(c.pattern ? [] : [problem(c.id, "pattern", "no story pattern")]),
    ];
  });
  return [...shaped, ...overlaps];
}

/** A range starts cleanly on an utterance that opens a sentence or follows a silence. */
export function cleanStart(r: Range, said: readonly Said[]): boolean {
  const first = said.find((u) => u.end > r.start + EDGE_SLACK);
  const before = said.filter((u) => u.end <= r.start + EDGE_SLACK).at(-1);
  return (
    first === undefined ||
    (first.start >= r.start - EDGE_SLACK &&
      (first.opens || before === undefined || first.start - before.end >= SENTENCE_PAUSE))
  );
}

/** A range ends cleanly when its last utterance is whole and the next one starts a sentence. */
export function cleanEnd(r: Range, said: readonly Said[]): boolean {
  const last = heard(r, said).at(-1);
  const next = said.find((u) => u.start >= r.end - EDGE_SLACK);
  const whole = last === undefined || last.end <= r.end + EDGE_SLACK;
  return (
    whole &&
    (next === undefined || next.opens || next.start - (last?.end ?? r.end) >= SENTENCE_PAUSE)
  );
}

export function shapeOf(ranges: readonly Range[], said: readonly Said[]): Shape {
  const first = ranges[0];
  return {
    ranges: ranges.length,
    cleanStarts: ranges.filter((r) => cleanStart(r, said)).length,
    cleanEnds: ranges.filter((r) => cleanEnd(r, said)).length,
    firstClean: first !== undefined && cleanStart(first, said),
  };
}

/** The utterances a viewer hears, in playing order, numbered from 1. */
export function played(ranges: readonly Range[], said: readonly Said[]): Played[] {
  return ranges
    .flatMap((r, k) =>
      heard(r, said).map((u) => ({ range: k + 1, text: u.text.replace(/\n/g, "") })),
    )
    .map((p, i) => ({ number: i + 1, ...p }));
}

export function judgeRequest(
  ranges: readonly Range[],
  said: readonly Said[],
  pattern: EvalClip["pattern"],
): unknown {
  const lines = played(ranges, said);
  const story = pattern ? [`構成: ${STORIES[pattern]}`] : [];
  const transcript = ranges.map((r, k) =>
    [
      `[区間${k + 1} ${clock(r.start)}〜${clock(r.end)}]`,
      ...lines.filter((p) => p.range === k + 1).map((p) => `${p.number}: ${p.text}`),
    ].join("\n"),
  );
  return {
    messages: [
      { role: "system", content: JUDGE_SYSTEM },
      { role: "user", content: [...story, ...transcript].join("\n") },
    ],
    reasoning_effort: "low",
    max_completion_tokens: JUDGE_TOKENS,
    response_format: { type: "json_object" },
  };
}

/** The judge's grade, or null when it did not answer in the rubric's form. */
export function parseGrade(response: unknown): Grade | null {
  const chat = chatSchema.safeParse(response);
  const content = chat.success ? (chat.data.choices[0]?.message.content ?? "") : "";
  const json = content.slice(content.indexOf("{"), content.lastIndexOf("}") + 1);
  const parsed = gradeSchema.safeParse(json === "" ? null : safeJson(json));
  return parsed.success ? parsed.data : null;
}

/** An Ai that answers repeated requests from `store` and counts every call in `meter`. */
export function cachedAi(ai: Ai, store: Store, meter: Meter): Ai {
  return {
    async run(model: string, input: unknown): Promise<unknown> {
      const key = createHash("sha256")
        .update(`${model}\n${JSON.stringify(input)}`)
        .digest("hex");
      meter.calls += 1;
      const known = await store.read(key);
      if (known !== null) return JSON.parse(known);
      meter.fresh += 1;
      const answer = await ai.run(model, input);
      await store.write(key, JSON.stringify(answer));
      return answer;
    },
  };
}

/** Grades every clip of one record and checks it; `calls` is what the record cost. */
export async function evaluate(input: {
  ai: Ai;
  record: ClipRecord;
  said: readonly Said[];
  subset: Subset;
  calls: number;
}): Promise<VideoEval> {
  const clips = await input.record.clips.reduce<Promise<ClipEval[]>>(async (done, c) => {
    const list = await done;
    const response = await input.ai.run(JUDGE_MODEL, judgeRequest(c.ranges, input.said, c.pattern));
    list.push({
      id: c.id,
      kind: c.kind,
      duration: c.duration,
      shape: shapeOf(c.ranges, input.said),
      grade: parseGrade(response),
    });
    return list;
  }, Promise.resolve([]));
  return {
    id: input.record.videoId,
    subset: input.subset,
    clips,
    problems: invariants(input.record),
    calls: input.calls,
  };
}

export function summarize(videos: readonly VideoEval[]): SubsetSummary {
  const clips = videos.flatMap((v) => v.clips);
  const grades = clips.flatMap((c) => (c.grade ? [c.grade] : []));
  const ranges = clips.reduce((t, c) => t + c.shape.ranges, 0);
  const good = grades.filter(isGood).length;
  return {
    videos: videos.length,
    clips: clips.length,
    clipsPerVideo: share(clips.length, videos.length),
    problems: videos.reduce((t, v) => t + v.problems.length, 0),
    cleanStartRate: share(
      clips.reduce((t, c) => t + c.shape.cleanStarts, 0),
      ranges,
    ),
    cleanEndRate: share(
      clips.reduce((t, c) => t + c.shape.cleanEnds, 0),
      ranges,
    ),
    firstCleanRate: share(clips.filter((c) => c.shape.firstClean).length, clips.length),
    means: {
      topic: mean(grades.map((g) => g.topic)),
      standalone: mean(grades.map((g) => g.standalone)),
      hook: mean(grades.map((g) => g.hook)),
      ending: mean(grades.map((g) => g.ending)),
      flow: mean(grades.map((g) => g.flow)),
    },
    goodRate: share(good, grades.length),
    goodPerVideo: share(good, videos.length),
    unjudged: clips.length - grades.length,
    callsPerVideo: share(
      videos.reduce((t, v) => t + v.calls, 0),
      videos.length,
    ),
  };
}

/** The clip with a 10 s stretch from at least 10 minutes away inserted after its first range. */
export function spliced(ranges: readonly Range[], said: readonly Said[]): Range[] {
  const first = ranges[0];
  const far = said.find(
    (u) =>
      u.opens &&
      first !== undefined &&
      Math.abs(u.start - first.start) >= SPLICE_DISTANCE &&
      ranges.every((r) => u.start + SPLICE_SECONDS < r.start || u.start > r.end),
  );
  return first && far
    ? [first, { start: far.start, end: far.start + SPLICE_SECONDS }, ...ranges.slice(1)]
    : [];
}

/** The clip without each range's first and last utterance: it now starts and ends mid-story. */
export function cut(ranges: readonly Range[], said: readonly Said[]): Range[] {
  const inner = ranges.flatMap((r) => {
    const lines = heard(r, said);
    const second = lines[1];
    const penultimate = lines.at(-2);
    return second && penultimate && second.start < penultimate.end
      ? [{ start: second.start, end: penultimate.end }]
      : [];
  });
  return inner.length === ranges.length ? inner : [];
}

const ALTER: Record<Negative, (ranges: readonly Range[], said: readonly Said[]) => Range[]> = {
  spliced,
  cut,
};
/** What a correct judge ranks lower for each negative kind. */
const WORSE: Record<Negative, (original: Grade, altered: Grade) => boolean> = {
  spliced: (o, a) => a.topic < o.topic,
  cut: (o, a) => a.hook + a.ending + a.flow < o.hook + o.ending + o.flow,
};

/** Every clip graded as is and with each negative applied. */
export async function calibrate(input: {
  ai: Ai;
  clips: readonly EvalClip[];
  said: readonly Said[];
}): Promise<Pair[]> {
  const grade = async (c: EvalClip, ranges: readonly Range[]): Promise<Grade | null> =>
    parseGrade(await input.ai.run(JUDGE_MODEL, judgeRequest(ranges, input.said, c.pattern)));
  return input.clips.reduce<Promise<Pair[]>>(async (done, c) => {
    const list = await done;
    const original = await grade(c, c.ranges);
    const variants = NEGATIVES.map((negative) => ({
      negative,
      ranges: ALTER[negative](c.ranges, input.said),
    })).filter((v) => v.ranges.length > 0);
    const graded = await variants.reduce<Promise<Pair[]>>(async (made, v) => {
      const pairs = await made;
      pairs.push({
        clip: c.id,
        negative: v.negative,
        original,
        altered: await grade(c, v.ranges),
      });
      return pairs;
    }, Promise.resolve([]));
    list.push(...graded);
    return list;
  }, Promise.resolve([]));
}

export function calibration(pairs: readonly Pair[]): Calibration {
  const count = (keep: (p: Pair) => boolean): Record<Negative, number> => ({
    spliced: pairs.filter((p) => p.negative === "spliced" && keep(p)).length,
    cut: pairs.filter((p) => p.negative === "cut" && keep(p)).length,
  });
  return {
    pairs: pairs.length,
    correct: count(
      (p) => p.original !== null && p.altered !== null && WORSE[p.negative](p.original, p.altered),
    ),
    total: count((p) => p.original !== null && p.altered !== null),
  };
}

/** The comparison table of labelled runs, one row per measure and one column per run. */
export function markdown(
  runs: readonly { label: string; subsets: Record<Subset, SubsetSummary> }[],
): string {
  const rows: [string, (s: SubsetSummary) => number][] = [
    ["videos", (s) => s.videos],
    ["clips / video", (s) => s.clipsPerVideo],
    ["good clips / video", (s) => s.goodPerVideo],
    ["good rate", (s) => s.goodRate],
    ["topic", (s) => s.means.topic],
    ["standalone", (s) => s.means.standalone],
    ["hook", (s) => s.means.hook],
    ["ending", (s) => s.means.ending],
    ["flow", (s) => s.means.flow],
    ["clean starts", (s) => s.cleanStartRate],
    ["clean ends", (s) => s.cleanEndRate],
    ["first range clean", (s) => s.firstCleanRate],
    ["rule problems", (s) => s.problems],
    ["unjudged", (s) => s.unjudged],
    ["model calls / video", (s) => s.callsPerVideo],
  ];
  const subsets: Subset[] = ["dev", "heldOut"];
  const header = subsets.flatMap((k) => runs.map((r) => `${r.label} ${k}`));
  return [
    `| measure | ${header.join(" | ")} |`,
    `| --- |${header.map(() => " ---: |").join("")}`,
    ...rows.map(
      ([name, value]) =>
        `| ${name} | ${subsets.flatMap((k) => runs.map((r) => String(value(r.subsets[k])))).join(" | ")} |`,
    ),
  ].join("\n");
}
