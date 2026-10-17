#!/usr/bin/env python3
"""Build the web catalog from the same libraries TranscriptViewer scans.

Rules mirror TranscriptCatalog/CandidateSelection (subset, fail-closed):
 1. evidence/candidate-selection.json (validated: size + transcript SHA-256)
    selects the movie/transcript for that recording.
 2. Otherwise a canonical `<stamp>_JST_<folder>-review.mp4` under a `delivery`
    directory with a same-stem .srt/.csv beside it; superseded deliveries
    (evidence/superseded/*.json path+size match) are excluded.
Outputs data/catalog.json plus data/<id>.json (cues + saved summary).
Stdlib only. Never writes outside the given output dir.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

STAMP = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})_JST_")


def stamp_label(name):
    m = STAMP.match(name)
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d} {h}:{mi}:{s} JST"


def parse_srt(text):
    cues = []
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    for block in blocks:
        lines = [row for row in block.strip("\n").split("\n") if row.strip()]
        if len(lines) < 3 or not lines[0].strip().isdigit() or "-->" not in lines[1]:
            if block.strip():
                raise ValueError(f"malformed SRT block: {block[:40]!r}")
            continue
        start_raw, end_raw = lines[1].split("-->", 1)
        start_label, end_label = start_raw.strip(), end_raw.strip().split()[0]

        def sec(v):
            clock, ms = v.split(",")
            h, m, s = clock.split(":")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

        cues.append(
            {
                "id": len(cues),
                "start": sec(start_label),
                "end": sec(end_label),
                "startLabel": start_label,
                "endLabel": end_label,
                "text": "\n".join(lines[2:]).strip(),
            }
        )
    return cues


def parse_csv(text):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or rows[0] != ["start_seconds", "end_seconds", "start", "end", "text"]:
        raise ValueError("CSV header mismatch")
    return [
        {
            "id": i,
            "start": float(r[0]),
            "end": float(r[1]),
            "startLabel": r[2],
            "endLabel": r[3],
            "text": r[4],
        }
        for i, r in enumerate(rows[1:])
        if r
    ]


def load_cues(path):
    text = Path(path).read_text(encoding="utf-8")
    return parse_srt(text) if path.lower().endswith(".srt") else parse_csv(text)


def load_summary(transcript_path):
    p = transcript_path + ".summaries/current.json"
    if not os.path.isfile(p):
        return None
    try:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
        if rec.get("version") != 1 or not rec.get("sections") or not rec.get("overview"):
            return None
        return {
            "overview": rec["overview"],
            "sections": [
                {"start": s["startSeconds"], "end": s["endSeconds"], "text": s["text"]}
                for s in rec["sections"]
            ],
        }
    except (ValueError, KeyError, TypeError):
        return None


def superseded(folder):
    out, d = {}, os.path.join(folder, "evidence", "superseded")
    if os.path.isdir(d):
        for name in sorted(os.listdir(d)):
            if name.endswith(".json"):
                try:
                    rec = json.loads(Path(d, name).read_text(encoding="utf-8"))
                    out[rec["delivery"]] = rec["size"]
                except (ValueError, KeyError, TypeError):
                    pass
    return out


def candidate_selection(folder, library):
    path = os.path.join(folder, "evidence", "candidate-selection.json")
    if not os.path.isfile(path):
        return None
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if rec.get("version") != 1 or rec.get("status") != "unverified-candidate":
        raise ValueError("candidate-selection: bad version/status")
    movie = os.path.join(folder, rec["movie"])
    transcript = os.path.join(folder, rec["transcript"])
    target = os.path.realpath(movie)
    lib_real = os.path.realpath(library)
    if not target.startswith(lib_real + os.sep) or os.path.getsize(target) != rec["movieSize"]:
        raise ValueError("candidate-selection: movie target/size mismatch")
    digest = hashlib.sha256(Path(transcript).read_bytes()).hexdigest()
    if digest != rec["transcriptSha256"]:
        raise ValueError("candidate-selection: transcript hash mismatch")
    return target, transcript, True


def canonical_deliveries(folder, name):
    sup = superseded(folder)
    found = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in ("work", "audio", "evidence", "transcripts", "proxy", "models")
        ]
        rel_parts = os.path.relpath(dirpath, folder).split(os.sep)
        if "delivery" not in rel_parts:
            continue
        for f in filenames:
            if not f.endswith(".mp4") or not STAMP.match(f):
                continue
            stem = f[:-4]
            if not stem.endswith(f"_JST_{name}-review"):
                continue
            movie = os.path.join(dirpath, f)
            rel = os.path.relpath(movie, folder)
            if sup.get(rel) == os.path.getsize(movie):
                continue
            tr = next(
                (
                    os.path.join(dirpath, stem + e)
                    for e in (".srt", ".csv")
                    if os.path.isfile(os.path.join(dirpath, stem + e))
                ),
                None,
            )
            if tr:
                found.append((movie, tr))
    return found


def find_original(folder, name, youtube_originals, roots=()):
    """Pre-edit source: edit-config.json `source` (size-validated), else the
    YouTube originals layout `<stamp>_youtube-<id>/<same>.{mp4,mkv}`.
    Candidate folders in a second library find their edit-config in another root."""
    for base in [folder] + [os.path.join(r, name) for r in roots]:
        cfg = os.path.join(base, "edit-config.json")
        if not os.path.isfile(cfg):
            continue
        try:
            rec = json.loads(Path(cfg).read_text(encoding="utf-8"))
            src = rec.get("source")
            if src and os.path.isfile(src) and os.path.getsize(src) == rec.get("source_size"):
                return src
        except (ValueError, KeyError, TypeError):
            pass
    if name.startswith("youtube-") and youtube_originals:
        import glob as g

        hits = g.glob(os.path.join(youtube_originals, f"*_youtube-{name[8:]}/*"))
        media = [h for h in hits if h.lower().endswith((".mp4", ".mkv"))]
        if len(media) == 1:
            return media[0]
    return None


def duration(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return round(float(out.stdout.strip()), 3)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True, help="library root (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-duration", action="store_true")
    ap.add_argument("--streamer", required=True, help="streamer tag for every video in these roots")
    ap.add_argument("--youtube-originals", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    videos, issues = [], []
    seen_ids = set()
    for root in a.root:
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder) or not name.startswith(("replay-", "youtube-")):
                continue
            try:
                sel = candidate_selection(folder, root)
                if sel:
                    choices, unverified = [sel[:2]], True
                else:
                    choices, unverified = canonical_deliveries(folder, name), False
                if not choices:
                    continue
                if len(choices) > 1:  # prefer the one carrying saved summaries
                    with_sum = [c for c in choices if load_summary(c[1])]
                    if len(with_sum) == 1:
                        choices = with_sum
                    else:
                        issues.append(f"ambiguous: {name} ({len(choices)} deliveries)")
                        continue
                movie, transcript = choices[0]
                base = os.path.basename(movie)
                vid = base[:-4]
                if vid in seen_ids:
                    continue
                seen_ids.add(vid)
                cues = load_cues(transcript)
                data = {"id": vid, "cues": cues, "summary": load_summary(transcript)}
                with open(os.path.join(a.out, vid + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
                original = find_original(folder, name, a.youtube_originals, a.root)
                videos.append(
                    {
                        "id": vid,
                        "streamer": a.streamer,
                        "title": name + (" [未検証候補]" if unverified else ""),
                        "broadcastStartJST": stamp_label(base),
                        "movie": base,
                        "size": os.path.getsize(movie),
                        # MKV originals are served as a generated H.264 MP4 proxy
                        # (same basename, .mp4); the uploader creates it from localPath.
                        "original": (
                            {
                                "movie": os.path.basename(original)[:-4] + ".mp4"
                                if original.lower().endswith(".mkv")
                                else os.path.basename(original),
                                "size": os.path.getsize(original),
                                "localPath": os.path.abspath(original),
                            }
                            if original
                            else None
                        ),
                        "duration": None if a.no_duration else duration(movie),
                        "transcriptPath": os.path.abspath(transcript),
                        "cueCount": len(cues),
                        "hasSummary": data["summary"] is not None,
                        "localPath": os.path.abspath(movie),
                    }
                )
            except Exception as e:  # fail-closed per recording, like the native scan
                issues.append(f"unreadable: {name} ({e})")
    videos.sort(key=lambda v: (v["broadcastStartJST"] or "", v["id"]), reverse=True)
    catalog = {"version": 1, "videos": videos, "issues": issues}
    with open(os.path.join(a.out, "catalog.json"), "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, ensure_ascii=False, indent=1)
    total = sum(v["size"] for v in videos)
    print(f"videos={len(videos)} total={total / 2**30:.1f} GiB issues={len(issues)}")
    for i in issues:
        print("  " + i)


if __name__ == "__main__":
    sys.exit(main())
