// Runs with bun. Validated shapes for the sync configuration, the catalog built from the
// delivery libraries, and the state of what was last uploaded.
import { z } from "zod";

export const configSchema = z.object({
  streamer: z.string().min(1),
  roots: z.array(z.string().min(1)).min(1),
  youtubeOriginals: z.string().min(1).nullable(),
  bucket: z.string().min(1),
  uploaderWorker: z.string().min(1),
});

const originalSchema = z.object({ movie: z.string().min(1), localPath: z.string().min(1) });

export const catalogSchema = z.object({
  videos: z.array(
    z.object({
      id: z.string().regex(/^[A-Za-z0-9._-]+$/),
      movie: z.string().min(1),
      localPath: z.string().min(1),
      transcriptPath: z.string().min(1),
      original: originalSchema.nullable(),
    }),
  ),
});

export const fingerprintSchema = z.object({
  size: z.number().int().nonnegative(),
  mtimeNs: z.string().regex(/^[0-9]+$/),
  sha256: z.string().nullable(),
});

export const stateSchema = z.object({
  sources: z.record(z.string(), z.record(z.string(), fingerprintSchema)),
  data: z.record(z.string(), z.string()),
  catalogSha256: z.string().nullable(),
});

export type SyncConfig = z.infer<typeof configSchema>;
export type Catalog = z.infer<typeof catalogSchema>;
export type CatalogVideo = Catalog["videos"][number];
export type Fingerprint = z.infer<typeof fingerprintSchema>;
export type SyncState = z.infer<typeof stateSchema>;
export type Sources = Record<string, Fingerprint>;

export const EMPTY_STATE: SyncState = { sources: {}, data: {}, catalogSha256: null };
