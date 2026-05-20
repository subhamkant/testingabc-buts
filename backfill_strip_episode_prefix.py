"""
One-off backfill (2026-05-21): strip `महाभारत #N:` / `Mahabharata #N:` / etc.
from existing video titles on the Vyasa AI channel.

Per assets/channel_analysis_2026-05-20.md, the 6 worst-performing recent
uploads all carry the `#N:` prefix and are sitting at 0–31 views. Phase 1
removed the prefix from the pipeline going forward; this script gives the
already-published videos a second algorithm pass by retro-fixing their titles.

Usage:
    python backfill_strip_episode_prefix.py             # dry-run — prints what would change
    python backfill_strip_episode_prefix.py --limit 6   # dry-run, bottom-6 by view count
    python backfill_strip_episode_prefix.py --limit 6 --apply

Safe to delete after the backfill is run successfully.
"""
import argparse
import pickle
import sys

from googleapiclient.discovery import build

# Reuse the production sanitizer so a backfilled title matches what new
# uploads will look like going forward (single source of truth).
sys.path.insert(0, ".")
from pipeline.youtube_uploader import _sanitize_title, _BANNED_TITLE_PATTERN


def _yt_client():
    with open("token.pickle", "rb") as f:
        creds = pickle.load(f)
    return build("youtube", "v3", credentials=creds)


def _all_my_videos(yt):
    """Return every video on the authenticated channel as (id, full_snippet) tuples."""
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

    snippets = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        r = yt.videos().list(part="snippet,status", id=",".join(chunk)).execute()
        for it in r["items"]:
            snippets[it["id"]] = it
    return [(vid, snippets[vid]) for vid in ids if vid in snippets]


def _video_view_counts(yt, ids):
    """Return {video_id: view_count}."""
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        r = yt.videos().list(part="statistics", id=",".join(chunk)).execute()
        for it in r["items"]:
            out[it["id"]] = int(it.get("statistics", {}).get("viewCount", 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually PATCH titles (default: dry-run)")
    ap.add_argument("--limit", type=int, default=None,
                    help="limit to bottom-N candidates by view count (default: all)")
    args = ap.parse_args()

    yt = _yt_client()
    all_vids = _all_my_videos(yt)
    print(f"Total videos on channel: {len(all_vids)}")

    candidates = []
    for vid, item in all_vids:
        old_title = item["snippet"]["title"]
        if not _BANNED_TITLE_PATTERN.search(old_title):
            continue
        new_title = _sanitize_title(old_title)[:100]  # YouTube title cap = 100
        if new_title == old_title or not new_title:
            continue
        candidates.append((vid, item, old_title, new_title))

    print(f"Videos with bannable prefix: {len(candidates)}")

    # Sort by view count ascending so --limit picks the worst performers.
    views = _video_view_counts(yt, [c[0] for c in candidates])
    candidates.sort(key=lambda c: views.get(c[0], 0))

    if args.limit is not None:
        candidates = candidates[: args.limit]
        print(f"Limited to bottom-{args.limit} by view count.")

    print("=" * 80)
    for vid, _, old, new in candidates:
        print(f"  {vid}  ({views.get(vid, 0)} views)")
        print(f"    OLD: {old}")
        print(f"    NEW: {new}")
        print()

    if not candidates:
        print("Nothing to do.")
        return

    if not args.apply:
        print()
        print("(dry-run) re-run with --apply to PATCH these titles.")
        return

    print()
    print("Applying changes...")
    print("=" * 80)
    success, failure = 0, 0
    for vid, item, old, new in candidates:
        # videos.update requires the FULL snippet body — categoryId is required.
        # Read the existing snippet, swap title only, send back.
        snippet = dict(item["snippet"])
        snippet["title"] = new
        # The API rejects snippet without categoryId; pull it from the existing item.
        # All channel uploads are categoryId=27 (Education) per CATEGORY_ID, but
        # use the value already on the video to avoid accidental reclassification.
        try:
            yt.videos().update(
                part="snippet",
                body={"id": vid, "snippet": snippet},
            ).execute()
            print(f"  [OK]   {vid}  =>  {new[:60]}")
            success += 1
        except Exception as e:
            print(f"  [FAIL] {vid}  =>  {str(e)[:120]}")
            failure += 1

    print()
    print(f"Done. {success} updated, {failure} failed.")


if __name__ == "__main__":
    main()
