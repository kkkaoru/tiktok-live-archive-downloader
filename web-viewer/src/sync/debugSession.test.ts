import { expect, it } from "vitest";
import { cookieFrom, debugFetch } from "./debugSession.ts";
import { createFakeFs, createFakeHttp } from "./fakes.fake.ts";

const secrets = JSON.stringify({ DEBUG_TOKEN: "t".repeat(40) });

it("extracts only the session cookie", () => {
  expect(cookieFrom("tv_session=abc; Path=/; HttpOnly")).toBe("tv_session=abc");
  expect(cookieFrom("other=1; Path=/")).toBeNull();
  expect(cookieFrom(null)).toBeNull();
});

it("mints a session and fetches the page with it", async () => {
  const { fs } = createFakeFs({ "/p/secrets.json": { text: secrets } });
  const { http, calls } = createFakeHttp((call) =>
    call.method === "POST"
      ? { status: 204, body: null, setCookie: "tv_session=abc; Path=/" }
      : { status: 200, body: { ok: true } },
  );
  const result = await debugFetch({
    fs,
    http,
    projectDirectory: "/p",
    baseUrl: "https://v",
    path: "/api/me",
  });
  expect(result).toStrictEqual({ mintStatus: 204, status: 200, body: { ok: true } });
  expect(calls[0]?.headers["X-Debug-Token"]).toBe("t".repeat(40));
  expect(calls[1]?.headers.Cookie).toBe("tv_session=abc");
});

it("stops when the Worker refuses to mint a session", async () => {
  const { fs } = createFakeFs({ "/p/secrets.json": { text: secrets } });
  const { http, calls } = createFakeHttp(() => ({ status: 404, body: null }));
  const result = await debugFetch({
    fs,
    http,
    projectDirectory: "/p",
    baseUrl: "https://v",
    path: "/x",
  });
  expect(result).toStrictEqual({ mintStatus: 404, status: 0, body: null });
  expect(calls.length).toBe(1);
});

it("rejects secrets without a long debug token", async () => {
  const { fs } = createFakeFs({
    "/p/secrets.json": { text: JSON.stringify({ DEBUG_TOKEN: "short" }) },
  });
  const { http } = createFakeHttp(() => ({ status: 204, body: null }));
  await expect(
    debugFetch({ fs, http, projectDirectory: "/p", baseUrl: "https://v", path: "/x" }),
  ).rejects.toThrow();
});
