// Runs with bun. Persistent record of what was last published, plus the pure update rule.
import { isMediaLabel } from "./fingerprint.ts";
import { settledKey } from "./plan.ts";
import type { FileSystem } from "./ports.ts";
import { EMPTY_STATE, type Sources, type SyncState, stateSchema } from "./schemas.ts";

export interface NextStateRequest {
  previous: SyncState;
  current: Record<string, Sources>;
  dataSha: Record<string, string>;
  catalogSha: string;
  settledMedia: ReadonlySet<string>;
  dataPublished: boolean;
  catalogPublished: boolean;
}

interface StatePath {
  fs: FileSystem;
  path: string;
}

interface SaveRequest extends StatePath {
  state: SyncState;
}

export async function loadState({ fs, path }: StatePath): Promise<SyncState> {
  const stat = await fs.stat(path);
  if (stat === null) {
    return EMPTY_STATE;
  }
  return stateSchema.parse(JSON.parse(await fs.readText(path)));
}

export async function saveState({ fs, path, state }: SaveRequest): Promise<void> {
  const temporary = `${path}.tmp`;
  await fs.writeText({ path: temporary, text: `${JSON.stringify(state, null, 1)}\n`, mode: 0o600 });
  await fs.rename({ from: temporary, to: path });
}

function videoSources(args: { id: string; current: Sources; request: NextStateRequest }): Sources {
  const { id, current, request } = args;
  const previous = request.previous.sources[id] ?? {};
  const entries = Object.entries(current);
  const media = entries.filter(
    ([label]) => isMediaLabel(label) && request.settledMedia.has(settledKey(id, label)),
  );
  const nonMedia = request.dataPublished ? entries.filter(([label]) => !isMediaLabel(label)) : [];
  const keptMedia = Object.entries(previous).filter(
    ([label]) => isMediaLabel(label) && !media.some(([settled]) => settled === label),
  );
  const keptNonMedia = request.dataPublished
    ? []
    : Object.entries(previous).filter(([label]) => !isMediaLabel(label));
  return Object.fromEntries([...keptMedia, ...keptNonMedia, ...media, ...nonMedia]);
}

export function nextState(request: NextStateRequest): SyncState {
  const sources = Object.fromEntries(
    Object.entries(request.current).map(([id, current]) => [
      id,
      videoSources({ id, current, request }),
    ]),
  );
  const publishedData = Object.fromEntries(
    Object.entries(request.dataSha).map(([id, sha]) => [`data/${id}.json`, sha]),
  );
  return {
    sources,
    data: request.dataPublished ? publishedData : request.previous.data,
    catalogSha256: request.catalogPublished ? request.catalogSha : request.previous.catalogSha256,
  };
}
