# Avatar-face / source-caption YOLO detector

One small detector (Ultralytics YOLO11n, two classes `avatar_face` and `source_caption`) replaces
the slow per-frame stages of source-caption removal and of fixed-Y caption placement. It runs on
MPS at about 3.3 ms per frame (640-pixel inference, batched). Every quality gate of the existing
pipeline stays in force; the detector only supplies inputs that those gates then judge.

## Production model

| Item | Value |
| --- | --- |
| Weights | `edits/models/yolo/avatar-face-caption-v2.pt` (sha256 `8c6e45b4b81cc4cc550ff988cbd4ab800c8cbd4db82dcfff5bfa3a46dda6b488`, `.sha256` beside it) |
| Inference size | 640 (better than 960 on unseen recordings and faster) |
| Training | `yolo11n.pt` start, 6 recordings (10-04, 10-06, 10-08, 10-10, 10-11, 10-14b), 7,109 frames, 960 px, early stop, best epoch 14 of 22 |
| Validation (epoch selection) | 10-14a, never trained: mAP50 0.969, caption mAP50 0.995 |
| Frozen test | 10-07 and 10-12, never used for training or selection |
| Previous model | `avatar-face-caption-v1.pt` (10-04 only); kept for reproducing older evidence |

Results on recordings never used in training (`replay.yolo_joint_eval`, Vision-labelled frames):

| Recording | Caption glyph coverage v1 → v2 | Face detection v1 → v2 | Face top p50 v1 → v2 |
| --- | --- | --- | --- |
| 10-07 | 99.06% → 99.83% | 99.89% → 100% | 16.8 → 13.0 px |
| 10-12 | 98.86% → 99.91% | 99.86% → 100% | 17.1 → 12.2 px |
| 10-14a | 99.10% → 99.94% | 99.81% → 100% | 29.1 → 29.2 px |

The face-top p90 against Vision is 31–45 px on 10-07/10-12 and about 131 px on 10-14a. On
10-14a Vision's *tracked* boxes drift and inflate onto the hair (images in
`edits/work/yolo-final-v1/worst-face-14a-e14.jpg`) while the detector boxes the face, so that
number measures the reference, not the detector. Placement quality is therefore judged by the
caption-embedding benchmark below, not by this p90.

### Why the earlier attempts did not generalize

- Caption labels came from OCR, which missed the pink English translation lines and boxed lines
  inconsistently between recordings, so real captions were taught as background. Labels now come
  from the caption pixels (pink fill, or white glyphs outlined in pink) joined along a text line
  (`replay.yolo_joint_dataset.caption_lines`).
- Training used few recordings and only frames Vision could label. The dataset is now built from
  every Vision-tracked recording by one manifest.
- `yolo_track` derived time as `frame / rate`. 10-07 is variable-frame-rate (a 44-second
  timestamp jump), so YOLO samples landed up to 45 s away from their real time. Tracks are now
  protocol v2 with each frame's presentation timestamp; v1 tracks are refused by placement.
- The removal chunk decode for YOLO first took "N frames after a time seek" while the renderer
  selects frames by exact PTS; across the 6–7 s timestamp gaps of some chunks the boxes landed on
  other frames and the audit escalated. The decode now uses the renderer's PTS selection.
- Training died with the agent session. Launch through `scripts/tmux-job.sh` and continue an
  interrupted run with `replay.yolo_train ... --resume`.

## Using it in production

All commands run from the checkout with `PYTHONPATH=src` and the pinned Ultralytics settings in
`edits/models/yolo/ultralytics-config` (every external service off, directories inside the
checkout). The detector description used below is:

```json
{"kind": "yolo",
 "weights": "<checkout>/edits/models/yolo/avatar-face-caption-v2.pt",
 "sha256": "8c6e45b4b81cc4cc550ff988cbd4ab800c8cbd4db82dcfff5bfa3a46dda6b488",
 "configDirectory": "<checkout>/edits/models/yolo/ultralytics-config",
 "work": "<checkout>/edits",
 "imgsz": 640, "masks": true, "admission": "yolo"}
```

### Source-caption removal

Add the object above as `regionDetector` to the machine cleanup defaults (or to one recording
description). `replay.source_clean_run describe/configs/run/assemble` then passes it to every
chunk job. Per chunk, `replay.subtitle_sampled_job`:

1. decodes the chunk once at 640 px with the renderer's exact PTS selection, detects caption boxes
   in every frame and grows each box sideways along its own line's caption glyphs (long lines can
   reach the frame edges);
2. `admission: "yolo"` builds the repair admission from those boxes
   (`replay.subtitle_yolo_admission`, same schema and rules as the OCR admission), instead of
   native OCR on sampled frames; `"ocr"` keeps the sampled-OCR admission;
3. `masks: true` turns the same boxes into the repair masks (`BoxMaskDetector`) instead of
   running the PP-OCR detector on every frame inside the renderer;
4. the unchanged OCR audit and colour-residual gate judge the result. A failed sampled attempt
   escalates to the full-OCR plan (stride 1) exactly as before.

The receipt pins the detector (`regionDetector: yolo:<sha256>:640:masks:admission`) and records
`sampling.kind = "yolo-caption-boxes"`.

### Caption embedding (fixed-Y placement)

```sh
python -m replay.yolo_track SOURCE.mp4 TRACK.json --weights edits/models/yolo/avatar-face-caption-v2.pt \
  --weights-sha256 8c6e45b4… --config-directory edits/models/yolo/ultralytics-config --work edits \
  --end FRAMES --step 2
python -m replay.caption_fixed_y_job PAGING - CUTS.json CALIBRATION.json OUT.json --yolo-track TRACK.json \
  --start-seconds 0 --end-seconds 100000 --region-top 120 --region-bottom 1800 --clearance 60 --maximum-drift 150
```

`-` in place of the Vision window index plans from YOLO alone and skips the Vision head-tracking
stage (60–80 minutes per recording). Passing a window index instead keeps Vision as the primary
source and lets YOLO fill only the frames Vision did not observe.

### Review video

`python -m replay.yolo_overlay SOURCE.mp4 OUT.mp4 --weights … --start-seconds S --seconds N` draws
the face box with its top line (green) and each caption line (yellow) with confidences and the
measured detection speed. It is a visual check, not evidence that a stage passed.

## Future recordings

Every recording that gets Vision head tracking can be added without code changes.

1. **Acceptance check before relying on YOLO for a new avatar or scene**: build a test-only
   dataset and evaluate it.

   ```json
   {"output": "<checkout>/edits/work/yolo-check-<rec>/dataset",
    "recordings": [{"name": "rec", "source": "/…/rec.mp4",
                    "windowIndex": "edits/work/future-head-factory/<rec>-v1/finalize-v1/window-index.json",
                    "step": 150, "role": "test"}]}
   ```

   `python -m replay.yolo_joint_dataset MANIFEST.json --jobs 1` then
   `python -m replay.yolo_joint_eval WEIGHTS DATASET OUT.json --recording rec --config-directory … --work edits`.
   Caption glyph coverage of at least 0.98 and face detection of at least 0.95 are the gates
   (module constants). Add `"faceMethods": ["detection"]` to grade faces against Vision's direct
   detections only.
2. **Retraining when a recording fails the check or a new avatar appears**: add the recording to
   the training manifest (`role: train`), rebuild the dataset (`--jobs 5` decodes recordings in
   parallel), train with `replay.yolo_train` (validation on a held-out recording, `--patience 8`),
   and evaluate on recordings that were never trained. Promote new weights only when they match or
   beat the pinned weights on that frozen set, then pin the new sha256 here and in the README.

## Benchmarks

Measured on 2026-10-17 on this Mac with the pinned v2 weights; inputs and scripts are kept in
`edits/work/yolo-final-v1/{removal-bench,addition-bench}`.

### Source-caption removal (10-07, five 30-second chunks, run one after another)

| Chunk | PP-OCR regions + PP-OCR masks + sampled OCR (before) | YOLO admission + regions + masks |
| --- | --- | --- |
| 10 | 92.8 s | 42.4 s |
| 24 | 95.5 s | 48.1 s |
| 58 (densest captions, timestamp gaps) | 140.6 s | 52.3 s |
| 106 | 112.6 s (colour residual max 6) | 59.1 s (colour residual max 0) |
| 140 | 98.9 s | 44.2 s |
| Total for 150 s of video | 540.5 s | 246.2 s (2.2× faster) |

Every chunk of both variants passed the unchanged OCR audit and colour-residual gate on the
first attempt (no escalation, no residual line). Per chunk the stages were, before → after:
planning 24.7 → 3.9 s (no planning OCR), render 45.4 → 16–19 s (no per-frame PP-OCR), OCR
audit about 20 s (unchanged gate), colour gate about 5 s. The OCR audit is now the largest stage.
A side-by-side video of chunk 10 is `edits/work/yolo-final-v1/demo/removal-compare-oct07-chunk10.mp4`.

### Caption embedding (10-07, 7,710 pages, same paging, cut plan and style-20 calibration)

| Person-top source | Tracking time | Planning time | Placed / failed runs | Pages overlapping a Vision-detected face |
| --- | --- | --- | --- | --- |
| Vision head tracking (before) | about 60–80 min | about 220 s | 7,710 / 4 | 0 (by construction) |
| YOLO v2, every 6th frame | 124 s | 23 s | 7,716 / 2 | 77 of 4,968 judged (7 over 30 px, max 39 px) |
| YOLO v2, every 2nd frame | 325 s | 23 s | 7,716 / 2 | 62 of 4,968 judged (4 over 30 px, max 36 px) |

"Overlapping" compares each page's bottom edge with the face top of every Vision *detection*
frame shown during the page. The worst cases only reach the top of Vision's face box
(forehead/hair line); they never cover the eyes (`addition-bench/worst-overlaps.jpg`). Use
`--step 2` for production placement.

## Known limits

- The OCR audit (about 20 s per chunk) remains as the independent removal gate.
- Vision's face box is larger than the detector's on some avatars; placement keeps a 60 px
  clearance above the detected face top.
- `edits/work/deliveries-v2/defaults.json` (now under the external work root) still names
  `edits/work/requested-deliveries-v1/selected-style20-calibration-v1/calibration.json`, which no
  longer exists. The embedding benchmark used a reconstruction that reproduces the surviving 10-07
  candidate to 5e-12 px (`addition-bench/calibration-style20-v1.json`); it is not a delivery
  calibration.
