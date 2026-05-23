"""
One-off backfill (2026-05-23): populate recent_topics.json from the channel's
published YouTube videos.

Problem this solves: recent_topics.json was reset around 2026-05-21 — only
4 entries exist, but the channel has 30+ published videos. The dedup logic
in _pick_next_arc_topic() (script_generator.py:260) does exact-string match
against recent_topics.json, so historical topics can be re-rendered.

Approach:
  1. Pull all published video titles from the channel via the YouTube API.
  2. Score each title against the corpus of known topic strings:
       - assets/character_arcs.json arcs (7 arcs × 7 topics = 49)
       - STORY_TOPICS legacy pool from script_generator.py
  3. Match by character-name overlap + incident-keyword overlap.
  4. For each video, record the best-matching topic into recent_topics.json
     (skipping already-present entries by topic string).

Usage:
  python backfill_recent_topics.py             # dry-run, show proposed matches
  python backfill_recent_topics.py --apply     # write recent_topics.json

Safe to delete after the backfill is run successfully.
"""
import argparse
import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone

from googleapiclient.discovery import build

sys.path.insert(0, ".")
# STORY_TOPICS is the legacy random-pool fallback used when arcs are exhausted.
# Importing it gives us the second pool of canonical topic strings.
from pipeline.script_generator import STORY_TOPICS
# Character / incident banks + signature helpers are shared with the runtime
# pre-check in pipeline/topic_signatures.py (single source of truth).
from pipeline.topic_signatures import characters_in as _characters_in
from pipeline.topic_signatures import incidents_in as _incidents_in

RECENT_TOPICS_PATH  = "recent_topics.json"
CHARACTER_ARCS_PATH = "assets/character_arcs.json"
SERIES              = "mahabharata"


def _score(title_chars: set, title_incidents: set, topic_chars: set, topic_incidents: set) -> float:
    """Score a (title, topic) pair. Higher = better match.
    - +2.0 per shared character (must overlap or no match)
    - +1.0 per shared incident keyword
    - 0 if zero character overlap (no match — return -1)
    """
    char_overlap = title_chars & topic_chars
    if not char_overlap:
        return -1.0
    incident_overlap = title_incidents & topic_incidents
    return 2.0 * len(char_overlap) + 1.0 * len(incident_overlap)


def _yt_client():
    with open("token.pickle", "rb") as f:
        creds = pickle.load(f)
    return build("youtube", "v3", credentials=creds)


def _all_my_videos(yt):
    ch = yt.channels().list(part="contentDetails", mine=True).execute()
    uploads_pl = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    ids = []
    page = None
    while True:
        r = yt.playlistItems().list(
            part="contentDetails",
            playlistId=uploads_pl,
            maxResults=50,
            pageToken=page,
        ).execute()
        ids += [it["contentDetails"]["videoId"] for it in r["items"]]
        page = r.get("nextPageToken")
        if not page:
            break

    titles = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        r = yt.videos().list(part="snippet,contentDetails", id=",".join(chunk)).execute()
        for it in r["items"]:
            title = it["snippet"]["title"]
            # PT#M#S → seconds
            iso = it["contentDetails"]["duration"].replace("PT", "")
            secs = 0
            n = ""
            for ch in iso:
                if ch.isdigit():
                    n += ch
                else:
                    v = int(n or 0)
                    if ch == "H": secs += v * 3600
                    elif ch == "M": secs += v * 60
                    elif ch == "S": secs += v
                    n = ""
            titles[it["id"]] = {"title": title, "duration_s": secs}
    return [(vid, titles[vid]) for vid in ids if vid in titles]


def _load_canonical_topics() -> list:
    """Return the unified corpus: all character_arcs.json arc topics
    + the STORY_TOPICS legacy pool. Each entry is the exact topic string
    used in the dedup."""
    corpus = []
    try:
        with open(CHARACTER_ARCS_PATH, encoding="utf-8") as f:
            for arc in json.load(f).get("arcs", []):
                for t in arc.get("topics", []):
                    corpus.append(t)
    except Exception as e:
        print(f"[warn] could not load {CHARACTER_ARCS_PATH}: {e}")
    corpus.extend(STORY_TOPICS)
    # Dedupe within the corpus itself
    seen = set()
    out = []
    for t in corpus:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _load_existing_used() -> set:
    if not os.path.exists(RECENT_TOPICS_PATH):
        return set()
    try:
        with open(RECENT_TOPICS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return {e["topic"] for e in data.get(SERIES, []) if e.get("topic")}
    except Exception:
        return set()


def _save_recent(new_entries: list) -> None:
    """Merge new_entries into recent_topics.json (mahabharata bucket).
    Preserves existing entries; appends new ones; sorts by ts; caps at 120."""
    data = {}
    if os.path.exists(RECENT_TOPICS_PATH):
        try:
            with open(RECENT_TOPICS_PATH, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    bucket = data.setdefault(SERIES, [])
    bucket.extend(new_entries)
    # Sort by ts (string sort works for ISO-ish timestamps)
    bucket.sort(key=lambda e: e.get("ts", ""))
    # Cap at 120 (matches _RECENT_TOPICS_KEEP)
    data[SERIES] = bucket[-120:]
    with open(RECENT_TOPICS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="Write to recent_topics.json (default: dry-run)")
    ap.add_argument("--min-score", type=float, default=2.0,
                    help="Minimum match score required (default 2.0 = "
                         "at least one character overlap)")
    args = ap.parse_args()

    print("Loading canonical topic corpus...")
    corpus = _load_canonical_topics()
    print(f"  {len(corpus)} canonical topics (arcs + legacy pool)")

    print("Pulling channel videos...")
    yt = _yt_client()
    videos = _all_my_videos(yt)
    print(f"  {len(videos)} videos on the channel")

    existing_used = _load_existing_used()
    print(f"  {len(existing_used)} topics already in recent_topics.json")

    # Pre-compute character / incident sets per topic
    topic_sigs = []
    for t in corpus:
        topic_sigs.append((t, _characters_in(t), _incidents_in(t)))

    proposed = []
    skipped_already_used = 0
    skipped_no_match = 0
    print()
    print("=" * 90)
    print("Proposed matches:")
    print("=" * 90)
    for vid, info in videos:
        title = info["title"]
        title_chars = _characters_in(title)
        title_incidents = _incidents_in(title)
        if not title_chars:
            print(f"\n  {vid}  NO CHARACTER MATCH")
            print(f"    title: {title!r}")
            skipped_no_match += 1
            continue

        best_topic, best_score = None, -1.0
        for topic, tc, ti in topic_sigs:
            s = _score(title_chars, title_incidents, tc, ti)
            if s > best_score:
                best_score = s
                best_topic = topic

        if best_score < args.min_score:
            print(f"\n  {vid}  LOW-CONFIDENCE  best_score={best_score:.1f}")
            print(f"    title: {title!r}")
            print(f"    best:  {(best_topic or '?')[:100]!r}")
            skipped_no_match += 1
            continue

        if best_topic in existing_used:
            skipped_already_used += 1
            continue

        print(f"\n  {vid}  match (score={best_score:.1f})  chars={title_chars}  incidents={title_incidents}")
        print(f"    title: {title!r}")
        print(f"    topic: {best_topic[:100]!r}")
        proposed.append({"video_id": vid, "title": title, "topic": best_topic, "score": best_score})

    print()
    print("=" * 90)
    print(f"Summary:")
    print(f"  Videos on channel:           {len(videos)}")
    print(f"  Already in recent_topics:    {skipped_already_used}")
    print(f"  Low-confidence / no match:   {skipped_no_match}")
    print(f"  Proposed new entries:        {len(proposed)}")
    print("=" * 90)

    if not args.apply:
        print()
        print("(dry-run) re-run with --apply to write recent_topics.json")
        return

    if not proposed:
        print("Nothing to write.")
        return

    # Sort proposed by score desc so the highest-confidence go first when
    # the 120-cap trims oldest
    entries = []
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    # Use a back-dated timestamp prefix so backfilled entries sort before
    # any genuine post-backfill entries (so a future cap trim drops
    # backfilled entries first if needed). Format keeps string sort.
    for i, p in enumerate(proposed):
        entries.append({
            "topic":      p["topic"],
            "ts":         f"2026-05-21 00:00:{i:02d}",  # back-dated stamp
            "source":     "backfill-2026-05-23",
            "video_id":   p["video_id"],
            "video_title":p["title"],
        })

    _save_recent(entries)
    print()
    print(f"[OK] wrote {len(entries)} entries to {RECENT_TOPICS_PATH}")


if __name__ == "__main__":
    main()
