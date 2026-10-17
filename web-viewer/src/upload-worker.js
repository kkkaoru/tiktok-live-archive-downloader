// Temporary bulk-upload helper (delete this Worker after the upload finishes).
// R2 multipart via binding, gated by a dedicated UPLOAD_TOKEN secret because
// `cf r2 objects put` caps at 300 MB and no S3 token can be minted here.
// `replay.r2_worker` deploys the same file to offload and restore render intermediates.
export default {
  async fetch(request, env) {
    if (request.headers.get("X-Upload-Token") !== env.UPLOAD_TOKEN) {
      return new Response("Forbidden", { status: 403 });
    }
    const url = new URL(request.url);
    try {
      if (url.pathname === "/ready") {
        return Response.json({ ready: true });
      }
      if (url.pathname === "/head") {
        const head = await env.MEDIA.head(url.searchParams.get("key"));
        return head
          ? Response.json({
              size: head.size,
              uploaded: head.uploaded.toISOString(),
              metadata: head.customMetadata || {},
            })
          : new Response("Not found", { status: 404 });
      }
      if (url.pathname === "/get") { // streamed download for restoring offloaded files
        const object = await env.MEDIA.get(url.searchParams.get("key"));
        return object
          ? new Response(object.body, { headers: { "Content-Length": String(object.size) } })
          : new Response("Not found", { status: 404 });
      }
      if (url.pathname === "/put") { // small objects only (catalog / per-video JSON)
        const body = await request.arrayBuffer();
        if (body.byteLength > 90 * 1024 * 1024) return new Response("Too large", { status: 413 });
        await env.MEDIA.put(url.searchParams.get("key"), body, {
          httpMetadata: { contentType: url.searchParams.get("type") || "application/octet-stream" },
        });
        return Response.json({ ok: true, size: body.byteLength });
      }
      if (url.pathname === "/create") {
        const { key, type, metadata } = await request.json();
        const up = await env.MEDIA.createMultipartUpload(key, {
          httpMetadata: { contentType: type || "video/mp4" },
          customMetadata: metadata || {},
        });
        return Response.json({ uploadId: up.uploadId });
      }
      if (url.pathname === "/part") {
        const up = env.MEDIA.resumeMultipartUpload(url.searchParams.get("key"), url.searchParams.get("uploadId"));
        const part = await up.uploadPart(Number(url.searchParams.get("n")), request.body);
        return Response.json({ partNumber: part.partNumber, etag: part.etag });
      }
      if (url.pathname === "/complete") {
        const { key, uploadId, parts } = await request.json();
        const up = env.MEDIA.resumeMultipartUpload(key, uploadId);
        const obj = await up.complete(parts);
        return Response.json({ ok: true, size: obj.size });
      }
      if (url.pathname === "/abort") {
        const { key, uploadId } = await request.json();
        await env.MEDIA.resumeMultipartUpload(key, uploadId).abort();
        return Response.json({ ok: true });
      }
      return new Response("Not found", { status: 404 });
    } catch (e) {
      return new Response(String(e && e.message || e), { status: 500 });
    }
  },
};
