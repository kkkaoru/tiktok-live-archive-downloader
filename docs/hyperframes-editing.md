# Local HyperFrames editing companion

`replay.hyperframes_edit` creates a **separate, private** HyperFrames project from an explicitly supplied local recording, source-time keep ranges and optional SRT. It never changes the source, jobs, existing revisions or delivery selections. Use it alongside the existing processing pipeline, not as a replacement for it.

Create a JSON configuration outside Git, under `edits/` (all paths absolute):

```json
{
  "version": 1,
  "source": "/absolute/path/to/authorized/source.mp4",
  "project": "/absolute/path/to/checkout/edits/my-new-hyperframes-project",
  "clips": [[0, 10], [20, 30]],
  "srt": "/absolute/path/to/reviewed/captions.srt",
  "gsap": "/absolute/path/to/pinned/gsap.min.js"
}
```

Use `null` for `srt` if no reviewed caption sidecar exists. `clips` are half-open **source seconds**, not output times; the specified ranges are concatenated in order. SRT times likewise refer to the source, and overlapping captions are clipped and translated to output time. When the local source is a separately prepared excerpt of a longer recording, set optional `caption_offset` to the excerpt's original source start in seconds; this subtracts that offset from SRT clocks. The default is zero. Include the source only with authorization. Obtain the pinned GSAP 3.14.2 browser bundle locally (for example `bun add gsap@3.14.2` in an ignored working directory) and pass its `dist/gsap.min.js`. The generator copies only that script; media is linked rather than duplicated. Keep the resulting project local: its `assets/source.*` symlink points at user footage and must not be staged, published or uploaded. Do not use HyperFrames cloud commands on it.

```sh
uv run --locked python -m replay.hyperframes_edit edits/config.json
HYPERFRAMES_NO_TELEMETRY=1 bunx hyperframes check edits/my-new-hyperframes-project --json
# After inspecting a short preview and confirming the intended result:
HYPERFRAMES_NO_TELEMETRY=1 bunx hyperframes render edits/my-new-hyperframes-project --quality looks
```

For a long HEVC source, HyperFrames `check` may first create a browser-compatible H.264 proxy of **the entire source**, even for a short clip. Prepare a separately verified, bounded local excerpt for preview, adjust `caption_offset` to its exact original start, and keep the original untouched. This excerpt is only a feasibility/visual review, not full-length delivery evidence. Preserve failed attempts for inspection; never treat their partial proxy as complete.

HyperFrames handles local cut placement, playback, sound placement and **visible** captions in this path. It does **not** remove burned-in source subtitles, run Japanese ASR, separate BGM, produce selectable MP4 subtitle streams, preserve compressed video bitstreams through rendering or certify native frame/audio clocks. Use existing local processors for those operations and their independent full-decode/storage gates. Full-length renders of long recordings can be slow and much larger than a stream-copy delivery; verify clocks, audio, decode and size before considering any result a delivery. The generated project is deliberately non-overwriting, with a bound on visible caption elements. Failed attempts remain under `edits/` for inspection, never reused silently.
