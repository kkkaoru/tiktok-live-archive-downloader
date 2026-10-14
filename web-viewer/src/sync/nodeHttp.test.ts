import { afterEach, expect, it, vi } from "vitest";
import { createNodeHttp } from "./nodeHttp.ts";

afterEach(() => {
  vi.unstubAllGlobals();
});

it("returns the status and the parsed JSON body", async () => {
  const fetchMock = vi.fn().mockResolvedValue({
    status: 200,
    json: async () => ({ ok: true }),
    headers: new Headers({ "set-cookie": "a=b; Path=/" }),
  });
  vi.stubGlobal("fetch", fetchMock);
  const response = await createNodeHttp().request({
    method: "POST",
    url: "https://u/x",
    headers: { A: "b" },
    body: "{}",
  });
  expect(response).toStrictEqual({ status: 200, body: { ok: true }, setCookie: "a=b; Path=/" });
  expect(fetchMock).toHaveBeenCalledWith("https://u/x", {
    method: "POST",
    headers: { A: "b" },
    body: "{}",
  });
});

it("returns a null body when the response is not JSON", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      status: 404,
      headers: new Headers(),
      json: async () => {
        throw new Error("not json");
      },
    }),
  );
  const response = await createNodeHttp().request({
    method: "GET",
    url: "https://u",
    headers: {},
    body: null,
  });
  expect(response).toStrictEqual({ status: 404, body: null, setCookie: null });
});
