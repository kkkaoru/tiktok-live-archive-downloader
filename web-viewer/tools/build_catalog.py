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
import shutil
import subprocess
import sys
from pathlib import Path

RAW = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_JST_replay-(\d+)\.mp4$")
STAMP = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})_JST_")


def stamp_label(name):
    m = STAMP.match(name)
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d} {h}:{mi}:{s} JST"


SPEAKER = re.compile(r"\[([A-Za-z0-9_-]{1,40})\]\s*")


def load_speakers(transcript_path):
    """Display names and colours of a multi-speaker transcript (<transcript>.speakers.json)."""
    p = transcript_path + ".speakers.json"
    if not os.path.isfile(p):
        return None
    rec = json.loads(Path(p).read_text(encoding="utf-8"))
    return {
        str(k): {"label": str(v["label"]), "color": str(v["color"])}
        for k, v in rec.get("speakers", {}).items()
        if re.fullmatch(r"#[0-9A-Fa-f]{6}", str(v.get("color", "")))
    } or None


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

        text = "\n".join(lines[2:]).strip()
        speaker = SPEAKER.match(text)  # "[name] text": a multi-speaker transcript names each cue
        cues.append(
            {
                "id": len(cues),
                "start": sec(start_label),
                "end": sec(end_label),
                "startLabel": start_label,
                "endLabel": end_label,
                **({"speaker": speaker.group(1)} if speaker else {}),
                "text": text[speaker.end() :] if speaker else text,
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


def load_excitement(transcript_path):
    """Per-bin excitement scores written by `python -m replay.excitement`."""
    p = transcript_path + ".excitement.json"
    if not os.path.isfile(p):
        return None
    try:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
    except ValueError:
        return None
    if rec.get("version") != 1:
        return None
    rec.pop("inputs", None)  # local fingerprints only
    return rec


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


def canonical_deliveries(folder, name, roots=()):
    """A delivery movie may live at the same relative path below another root
    (an offloaded movie beside a mirrored transcript)."""
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
        stems = {f[:-4] for f in filenames if f.endswith((".mp4", ".srt", ".csv"))}
        for stem in sorted(stems):
            if not STAMP.match(stem) or not stem.endswith(f"_JST_{name}-review"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, stem + ".mp4"), folder)
            movie = next(
                (
                    m
                    for m in [os.path.join(folder, rel)]
                    + [os.path.join(r, name, rel) for r in roots]
                    if os.path.isfile(m)
                ),
                None,
            )
            if movie is not None and sup.get(rel) == os.path.getsize(movie):
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
    ap.add_argument("--raw", default=None, help="downloads whose recording has no delivery yet")
    ap.add_argument("--keep", default=None, help="published catalog dir: keep entries not found")
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
                    choices, unverified = canonical_deliveries(folder, name, a.root), False
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
                base = (
                    os.path.basename(movie) if movie else os.path.basename(transcript)[:-4] + ".mp4"
                )
                vid = base[:-4]
                if vid in seen_ids:
                    continue
                seen_ids.add(vid)
                cues = load_cues(transcript)
                data = {"id": vid, "cues": cues, "summary": load_summary(transcript)}
                excite = load_excitement(transcript)
                if excite:
                    data["excitement"] = excite
                speakers = load_speakers(transcript)
                if speakers:
                    data["speakers"] = speakers
                with open(os.path.join(a.out, vid + ".json"), "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
                original = find_original(folder, name, a.youtube_originals, a.root)
                if movie is None and not original:
                    continue  # a transcript alone has nothing to play
                videos.append(
                    {
                        "id": vid,
                        "streamer": a.streamer,
                        "title": name
                        + (" [未検証候補]" if unverified else "")
                        + ("" if movie else " [文字起こし先行・元動画]"),
                        "broadcastStartJST": stamp_label(base),
                        "movie": base if movie else None,
                        "size": os.path.getsize(movie or original),
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
                        "duration": None if a.no_duration else duration(movie or original),
                        "transcriptPath": os.path.abspath(transcript),
                        "cueCount": len(cues),
                        "hasSummary": data["summary"] is not None,
                        "localPath": os.path.abspath(movie) if movie else None,
                    }
                )
            except Exception as e:  # fail-closed per recording, like the native scan
                issues.append(f"unreadable: {name} ({e})")
    delivered = {v["title"].split(" ")[0] for v in videos}
    if a.keep and os.path.isfile(os.path.join(a.keep, "catalog.json")):
        # A published recording whose local files were retired stays published as it is.
        published = json.loads(Path(a.keep, "catalog.json").read_text(encoding="utf-8"))
        for video in published["videos"]:
            folder = video["title"].split(" ")[0]
            data_path = os.path.join(a.keep, video["id"] + ".json")
            raw = video["movie"] is None  # rebuilt from the downloads below
            if (
                raw
                or folder in delivered
                or video["id"] in seen_ids
                or not os.path.isfile(data_path)
            ):
                continue
            shutil.copyfile(data_path, os.path.join(a.out, video["id"] + ".json"))
            videos.append({**video, "kept": True})
            delivered.add(folder)
            issues.append(f"kept published: {folder} (local delivery not found)")
    if a.raw:
        for base in sorted(os.listdir(a.raw)):
            match = RAW.match(base)
            if not match or f"replay-{match.group(1)}" in delivered:
                continue
            movie, vid = os.path.join(a.raw, base), base[:-4]
            data_path = os.path.join(a.out, vid + ".json")
            with open(data_path, "w", encoding="utf-8") as fh:
                json.dump({"id": vid, "cues": [], "summary": None}, fh, separators=(",", ":"))
            # Only the original is published, at the key its edited delivery will reuse.
            videos.append(
                {
                    "id": vid,
                    "streamer": a.streamer,
                    "title": f"replay-{match.group(1)} [未編集・字幕なし]",
                    "broadcastStartJST": stamp_label(base),
                    "movie": None,
                    "size": os.path.getsize(movie),
                    "original": {
                        "movie": base,
                        "size": os.path.getsize(movie),
                        "localPath": os.path.abspath(movie),
                    },
                    "duration": None if a.no_duration else duration(movie),
                    "transcriptPath": os.path.abspath(data_path),
                    "cueCount": 0,
                    "hasSummary": False,
                    "localPath": None,
                }
            )
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
