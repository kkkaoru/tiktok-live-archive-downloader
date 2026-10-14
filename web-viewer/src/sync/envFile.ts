// Runs with bun. Parses a simple KEY=VALUE file (comments and blank lines ignored).
const KEY = /^[A-Za-z_][A-Za-z0-9_]*$/;
const COMMENT = /^\s*#/;
const QUOTED = /^(".*"|'.*')$/;

function parseLine(line: string): Array<[string, string]> {
  const index = line.indexOf("=");
  if (index <= 0 || COMMENT.test(line)) {
    return [];
  }
  const key = line.slice(0, index).trim();
  if (!KEY.test(key)) {
    return [];
  }
  const raw = line.slice(index + 1).trim();
  return [[key, QUOTED.test(raw) ? raw.slice(1, -1) : raw]];
}

export function parseEnvFile(text: string): Record<string, string> {
  return Object.fromEntries(text.split("\n").flatMap(parseLine));
}
