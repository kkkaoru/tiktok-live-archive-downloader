// Runs with bun. Fingerprints every file that feeds a video's published data: the movie, the
// original, the transcript, the saved summary and the files stored next to the transcript.
import { basename, dirname, extname, join } from "node:path";
import type { FileSystem } from "./ports.ts";
import type { CatalogVideo, Fingerprint, Sources } from "./schemas.ts";

export const HASH_LIMIT_BYTES = 64 * 1024 * 1024;
export const MEDIA_LABELS: readonly string[] = ["movie", "original"];
const SUMMARY_LABEL = "summary";
const SUMMARY_DIRECTORY_SUFFIX = ".summaries";
const SUMMARY_FILE = "current.json";

interface SourceEntry {
  label: string;
  path: string;
  required: boolean;
}

interface FingerprintRequest {
  fs: FileSystem;
  path: string;
}

interface SourcesRequest {
  fs: FileSystem;
  video: CatalogVideo;
}

async function fingerprintIfFile({ fs, path }: FingerprintRequest): Promise<Fingerprint | null> {
  const stat = await fs.stat(path);
  if (stat === null || !stat.isFile) {
    return null;
  }
  const sha256 = stat.size <= HASH_LIMIT_BYTES ? await fs.sha256(path) : null;
  return { size: stat.size, mtimeNs: stat.mtimeNs.toString(), sha256 };
}

export function isMediaLabel(label: string): boolean {
  return MEDIA_LABELS.includes(label);
}

export function sameFingerprint(left: Fingerprint, right: Fingerprint): boolean {
  return left.size === right.size && left.mtimeNs === right.mtimeNs && left.sha256 === right.sha256;
}

async function sourceEntries({ fs, video }: SourcesRequest): Promise<SourceEntry[]> {
  const directory = dirname(video.transcriptPath);
  const transcriptName = basename(video.transcriptPath);
  const stem = basename(transcriptName, extname(transcriptName));
  const names = await fs.readdir(directory);
  const siblings = names
    .filter((name) => name.startsWith(`${stem}.`) && name !== transcriptName)
    .map((name) => ({ label: `sibling:${name}`, path: join(directory, name), required: false }));
  const original = video.original === null ? [] : [video.original];
  return [
    { label: "movie", path: video.localPath, required: true },
    { label: "transcript", path: video.transcriptPath, required: true },
    ...original.map((item) => ({ label: "original", path: item.localPath, required: true })),
    {
      label: SUMMARY_LABEL,
      path: join(directory, `${transcriptName}${SUMMARY_DIRECTORY_SUFFIX}`, SUMMARY_FILE),
      required: false,
    },
    ...siblings,
  ];
}

export async function collectSources({ fs, video }: SourcesRequest): Promise<Sources> {
  const entries = await sourceEntries({ fs, video });
  const found = await Promise.all(
    entries.map(async (entry) => ({
      entry,
      fingerprint: await fingerprintIfFile({ fs, path: entry.path }),
    })),
  );
  const missing = found.filter((item) => item.entry.required && item.fingerprint === null);
  if (missing.length > 0) {
    throw new Error(
      `Missing source file for ${video.id}: ${missing.map((m) => m.entry.label).join(", ")}`,
    );
  }
  const present = found.flatMap(
    (item): Array<[string, Fingerprint]> =>
      item.fingerprint === null ? [] : [[item.entry.label, item.fingerprint]],
  );
  return Object.fromEntries(present);
}

export async function fingerprintPath({ fs, path }: FingerprintRequest): Promise<Fingerprint> {
  const fingerprint = await fingerprintIfFile({ fs, path });
  if (fingerprint === null) {
    throw new Error(`Missing file: ${path}`);
  }
  return fingerprint;
}
