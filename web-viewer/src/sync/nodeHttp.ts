// Runs with bun. Real HTTP adapter for the HttpClient port (global fetch).
import type { HttpClient } from "./ports.ts";

export function createNodeHttp(): HttpClient {
  return {
    request: async ({ method, url, headers, body }) => {
      const response = await fetch(url, { method, headers, body });
      const parsed = await response.json().catch(() => null);
      return {
        status: response.status,
        body: parsed,
        setCookie: response.headers.get("set-cookie"),
      };
    },
  };
}
