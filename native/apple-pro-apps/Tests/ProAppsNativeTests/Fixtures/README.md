# Synthetic media fixture

`black.mp4` was generated locally with Apple's AVAssetWriter, not copied from a
user download or third-party video. It contains one opaque black 16×16 BGRA frame,
no audio, and a one-second presentation session. AVOutputSettingsAssistant's
640×480 preset was used with width/height changed to 16; its nominal frame rate
is 15 fps. The frame was filled with a cropped solid-black CIImage via CIContext.
No raw uninitialized pixel memory was encoded.

The test checks decoded dimensions, duration, nominal rate, absence of audio,
first-frame decoding, serialization and that the fixture bytes remain unchanged.
This intentionally does not prove full-stream validation or any Apple editor's
import/export behavior. ffmpeg is not a test or runtime dependency.

## Continuous 30-frame quadrants

`quadrants-30.mp4` is a separate, locally generated synthetic 16×16 fixture:
red/green upper quadrants and blue/yellow lower quadrants, 30 decoded frames,
30 fps, exactly one second, and no audio. It contains no user media.

The old runtime fixture performed two AVAssetExportSession exports before the
editor test. Under Address Sanitizer it produced only 29 source frames despite
reporting one second at 30 fps. The source prerequisite correctly failed; the
editor's 45-frame output assertion was not reached. A golden fixture removes
that independent, nondeterministic export from setup. The 30-source-frame and
45-edited-frame assertions remain unchanged, and every copied fixture is fully
decoded through the native reader before use. No assertion or sanitizer is waived.

One-time generation (not executed during tests):

```sh
ffmpeg -nostdin -n -f lavfi -i 'color=red:s=16x16:r=30:d=1,drawbox=x=8:y=0:w=8:h=8:color=lime:t=fill,drawbox=x=0:y=8:w=8:h=8:color=blue:t=fill,drawbox=x=8:y=8:w=8:h=8:color=yellow:t=fill' -an -c:v libx264 -profile:v high -crf 18 -bf 0 -pix_fmt yuv420p -frames:v 30 -video_track_timescale 30000 quadrants-30.mp4
```

Independent ffprobe frame counting verifies the stored source separately from
the native reader and editor under test. A lossless H.264 first attempt passed
FFmpeg decoding but Apple's reader rejected it; it was retained privately as
failure evidence, not adopted as the fixture. Native ASan verification of the
High-profile replacement and the full release gates remain required.

## Tracking object

`tracking-object-30.mp4` is a locally generated, static 256×256 synthetic image:
the existing colored quadrants, a centered white frame and asymmetric markers.
It has 30 decoded frames at 30 fps, one second and no audio or user media. SHA-256:
`59a8c52598ea0589f9b5c3e981f747bd157725e09f649398fd103ac24f0b772f`.

The tracking test previously invoked the editor to add a text title. Its fixture
setup could block in SwiftUI appearance lookup or FCPXML application inventory
before reaching tracking. This golden texture isolates tracking from those
unrelated GUI integrations. Real native decode/tracking, 30 frames, 256×256,
periodic redetection and confidence >0.3 remain required; no assertion is removed.
The test first verifies the copied fixture with the native full decoder.

One-time generation, not a test/runtime dependency:

```sh
ffmpeg -nostdin -n -i quadrants-30.mp4 -vf 'scale=256:256,drawbox=x=64:y=64:w=128:h=128:color=white:t=6,drawbox=x=76:y=76:w=28:h=28:color=black:t=fill,drawbox=x=150:y=116:w=24:h=58:color=white:t=fill,drawbox=x=92:y=145:w=36:h=18:color=magenta:t=fill' -an -frames:v 30 -c:v libx264 -profile:v high -crf 18 -bf 0 -pix_fmt yuv420p -video_track_timescale 30000 tracking-object-30.mp4
```

Independent FFprobe frame counting and complete FFmpeg decode passed before
fixture use. This does not certify editor integration or any private recording.
An earlier periodic checkerboard did not sustain native tracking and failed the
unchanged tracking-method assertion. It was retained as private failure evidence,
not adopted as this fixture. No confidence or frame assertions were relaxed.

## Motion caption part

`motion-caption-part.moti` is a finished two-title caption part built from synthetic text
("テスト\nです" revealed by synthetic speech tokens, then "はい") with the repository's
caption tools exactly as `replay.caption_parts` runs them: `replay.motion_reveal_stages`
on the production caption template, then `replay.motion_finish` with the production style-20
arguments. It holds no user media or transcript. `MotionCaptionDocumentTests` read it and
mutate copies in memory to check every refusal of the native document reader.
