// Runs with bun. Pure planning: compares the current files with what was last uploaded and
// lists exactly which objects need publishing, with the reason for each.
import { isMediaLabel, sameFingerprint } from "./fingerprint.ts";
import type { CatalogVideo, Fingerprint, Sources, SyncState } from "./schemas.ts";

export interface MediaChange {
  kind: "media";
  key: string;
  path: string;
  reason: string;
  videoId: string;
  label: string;
  proxy: boolean;
}

export interface ObjectChange {
  kind: "data" | "catalog";
  key: string;
  path: string;
  reason: string;
}

export type Change = MediaChange | ObjectChange;

export function isMediaChange(change: Change): change is MediaChange {
  return change.kind === "media";
}

export interface PlanInput {
  videos: CatalogVideo[];
  current: Record<string, Sources>;
  dataSha: Record<string, string>;
  catalogSha: string;
  dataDirectory: string;
  state: SyncState;
  remoteSizes: Record<string, number>;
  proxySizes: Record<string, number>;
}

export interface Plan {
  changes: Change[];
  settledMedia: string[];
}

interface MediaCandidate {
  video: CatalogVideo;
  label: string;
  key: string;
  path: string;
  proxy: boolean;
}

interface MediaDecision {
  settled: boolean;
  change: MediaChange | null;
}

interface MediaRequest {
  candidate: MediaCandidate;
  input: PlanInput;
}

const MKV_SUFFIX = ".mkv";

export function settledKey(videoId: string, label: string): string {
  return `${videoId}:${label}`;
}

function mediaCandidates(video: CatalogVideo): MediaCandidate[] {
  const movie: MediaCandidate = {
    video,
    label: "movie",
    key: `media/${video.movie}`,
    path: video.localPath,
    proxy: false,
  };
  if (video.original === null) {
    return [movie];
  }
  const original: MediaCandidate = {
    video,
    label: "original",
    key: `media-original/${video.original.movie}`,
    path: video.original.localPath,
    proxy: video.original.localPath.toLowerCase().endsWith(MKV_SUFFIX),
  };
  return [movie, original];
}

function mediaReason({ previous, remote, current }: MediaReasonRequest): string {
  if (previous !== undefined) {
    return "file changed since the last upload";
  }
  if (remote === undefined) {
    return "not in the bucket";
  }
  return `size differs (bucket ${remote} B, local ${current.size} B)`;
}

interface MediaReasonRequest {
  previous: Fingerprint | undefined;
  remote: number | undefined;
  current: Fingerprint;
}

function decideMedia({ candidate, input }: MediaRequest): MediaDecision {
  const { video, label, key, path, proxy } = candidate;
  const current = input.current[video.id]?.[label];
  if (current === undefined) {
    return { settled: true, change: null };
  }
  const previous = input.state.sources[video.id]?.[label];
  const remote = input.remoteSizes[key];
  // A proxied original is stored in the bucket as an H.264 proxy, so its size is compared
  // with the local proxy file, never with the MKV source.
  const comparable = proxy ? input.proxySizes[key] : current.size;
  const unchanged = previous !== undefined && sameFingerprint(previous, current);
  const adopted = previous === undefined && comparable !== undefined && remote === comparable;
  if (unchanged || adopted) {
    return { settled: true, change: null };
  }
  const reason = mediaReason({ previous, remote, current });
  const change: MediaChange = { kind: "media", key, path, reason, videoId: video.id, label, proxy };
  return { settled: false, change };
}

export function changedLabels(args: { previous: Sources | undefined; current: Sources }): string[] {
  const before = args.previous ?? {};
  const nonMedia = (label: string) => !isMediaLabel(label);
  const labels = [...new Set([...Object.keys(before), ...Object.keys(args.current)])].filter(
    nonMedia,
  );
  return labels.filter((label) => {
    const left = before[label];
    const right = args.current[label];
    return left === undefined || right === undefined || !sameFingerprint(left, right);
  });
}

function dataChange(video: CatalogVideo, input: PlanInput): ObjectChange | null {
  const key = `data/${video.id}.json`;
  const previousSha = input.state.data[key];
  const labels = changedLabels({
    previous: input.state.sources[video.id],
    current: input.current[video.id] ?? {},
  });
  const sha = input.dataSha[video.id];
  const reason = dataReason({ previousSha, sha, labels });
  if (reason === null) {
    return null;
  }
  const path = `${input.dataDirectory}/${video.id}.json`;
  return { kind: "data", key, path, reason };
}

interface DataReasonRequest {
  previousSha: string | undefined;
  sha: string | undefined;
  labels: string[];
}

function dataReason({ previousSha, sha, labels }: DataReasonRequest): string | null {
  if (previousSha === undefined) {
    return "not uploaded yet";
  }
  if (labels.length > 0) {
    return `source files changed: ${labels.join(", ")}`;
  }
  return previousSha === sha ? null : "generated data changed";
}

export function planChanges(input: PlanInput): Plan {
  const decisions = input.videos.flatMap((video) =>
    mediaCandidates(video).map((candidate) => ({
      candidate,
      decision: decideMedia({ candidate, input }),
    })),
  );
  const media = decisions.flatMap((item) =>
    item.decision.change === null ? [] : [item.decision.change],
  );
  const settledMedia = decisions
    .filter((item) => item.decision.settled)
    .map((item) => settledKey(item.candidate.video.id, item.candidate.label));
  const data = input.videos.flatMap((video) => {
    const change = dataChange(video, input);
    return change === null ? [] : [change];
  });
  const listed = [...media, ...data];
  const catalogChanged = listed.length > 0 || input.state.catalogSha256 !== input.catalogSha;
  const catalog: Change[] = catalogChanged
    ? [
        {
          kind: "catalog",
          key: "catalog.json",
          path: `${input.dataDirectory}/catalog.json`,
          reason: listed.length > 0 ? "videos changed" : "catalog changed",
        },
      ]
    : [];
  return { changes: [...media, ...data, ...catalog], settledMedia };
}
