"""
One-off: fetch the channel's recent YouTube uploads + auto-fill the
YOUTUBE_URLS dict in upload_episodes_1_2_3.py.

Reuses the existing OAuth token (token.pickle) — no new auth needed.

Logic:
  1. Auth via pipeline.youtube_uploader.get_youtube_service()
  2. Fetch the channel's uploads playlist (most recent 20)
  3. Match titles to episodes 1/2/3 by keyword:
       • #1: title contains "Bhishma" AND ("Vow" OR "Pratigya" OR "प्रतिज्ञा")
       • #2: title contains "Bhishma" AND ("Raising" OR "दुविधा" OR "Pandavas")
       • #3: title contains "Bhishma" AND ("Silence" OR "Shocking" OR "अपमान")
  4. Print matches + auto-update upload_episodes_1_2_3.py's YOUTUBE_URLS dict

Usage:
    python fetch_youtube_urls.py
"""

import os
import re
import sys
import io
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from pipeline.youtube_uploader import get_youtube_service


# Title-matching patterns per episode. Both English and Hindi keywords so
# we match whether the title leads with Hindi or English half.
_MATCHERS = {
    1: {
        "label": "Bhishma vow of celibacy",
        # exclude "Painful Vow" or anything that's clearly #2's mistitle
        "must_have": ["bhishma", "vow"],
        "must_not_have": ["raising", "painful", "dilemma", "दादा"],
        "hindi_alt": ["प्रतिज्ञा"],
    },
    2: {
        "label": "Bhishma raising the Pandavas + Kauravas",
        "must_have": ["bhishma", "raising"],
        "must_not_have": [],
        "hindi_alt": ["दुविधा", "पांडवों"],
    },
    3: {
        "label": "Bhishma silence during vastraharan",
        "must_have": ["bhishma", "silence"],
        "must_not_have": [],
        "hindi_alt": ["अपमान", "वस्त्रहरण"],
    },
}


def _matches(title: str, ep_id: int) -> bool:
    title_lower = title.lower()
    m = _MATCHERS[ep_id]
    # First, exclusion check
    for bad in m["must_not_have"]:
        if bad.lower() in title_lower:
            return False
    # Match if BOTH must_have words present (case-insensitive)
    if all(w.lower() in title_lower for w in m["must_have"]):
        return True
    # Or if any hindi_alt keyword present alongside one must_have
    for alt in m["hindi_alt"]:
        if alt in title and any(w.lower() in title_lower for w in m["must_have"][:1]):
            return True
    return False


def fetch_recent_videos(youtube, max_results: int = 25) -> list[dict]:
    """Fetch recent uploads. Returns list of {id, title, published_at}."""
    # First get uploads playlist ID from the channel
    ch = youtube.channels().list(part="contentDetails", mine=True).execute()
    items = ch.get("items", [])
    if not items:
        sys.exit("[ERROR] no channel found for this token")
    uploads_pl = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # Then list playlistItems
    pli = youtube.playlistItems().list(
        part="snippet,contentDetails",
        playlistId=uploads_pl,
        maxResults=max_results,
    ).execute()
    videos = []
    for it in pli.get("items", []):
        sn = it.get("snippet", {})
        cd = it.get("contentDetails", {})
        videos.append({
            "id": cd.get("videoId", ""),
            "title": sn.get("title", ""),
            "published_at": cd.get("videoPublishedAt", ""),
        })
    return videos


def update_youtube_urls_in_helper(found: dict[int, str]) -> int:
    """Patch the YOUTUBE_URLS dict in upload_episodes_1_2_3.py. Returns
    number of lines updated."""
    path = "upload_episodes_1_2_3.py"
    if not os.path.exists(path):
        print(f"[!] {path} not found — skipping auto-patch")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    updated = 0
    for ep_id, url in found.items():
        # Match either an empty value OR an existing value for this ep number
        pattern = re.compile(
            rf'(^\s*{ep_id}:\s*)"[^"]*"(\s*,?\s*(?:#[^\n]*)?)$',
            re.MULTILINE,
        )
        new_src, n = pattern.subn(
            rf'\g<1>"{url}"\g<2>',
            src,
            count=1,
        )
        if n > 0:
            src = new_src
            updated += 1
            print(f"    [patched] YOUTUBE_URLS[{ep_id}] = \"{url}\"")
        else:
            print(f"    [!] could not find YOUTUBE_URLS[{ep_id}] line in {path}")

    if updated:
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
    return updated


def main():
    print("Authenticating with YouTube...")
    youtube = get_youtube_service()

    print("Fetching recent uploads from your channel...\n")
    videos = fetch_recent_videos(youtube, max_results=25)
    if not videos:
        sys.exit("[ERROR] no videos found in uploads playlist")

    print(f"Found {len(videos)} recent upload(s):\n")
    for v in videos:
        print(f"  [{v['published_at'][:10]}] {v['id']}  {v['title'][:80]}")
    print()

    # Match
    found = {}
    for ep_id in (1, 2, 3):
        matches = [v for v in videos if _matches(v["title"], ep_id)]
        if not matches:
            print(f"  [!] No match for #{ep_id} ({_MATCHERS[ep_id]['label']})")
            continue
        # Prefer the MOST RECENT match (in case of duplicate uploads)
        chosen = sorted(matches, key=lambda v: v["published_at"], reverse=True)[0]
        url = f"https://youtube.com/watch?v={chosen['id']}"
        found[ep_id] = url
        if len(matches) > 1:
            print(f"  [info] #{ep_id} has {len(matches)} match(es), using newest:")
        print(f"  ✓ #{ep_id} -> {url}")
        print(f"      ({chosen['title'][:80]})")

    if not found:
        sys.exit("\n[ERROR] no matches — check title keywords in _MATCHERS")

    print(f"\nPatching upload_episodes_1_2_3.py YOUTUBE_URLS dict...")
    updated = update_youtube_urls_in_helper(found)
    print(f"\nDone — {updated} URL(s) updated in upload_episodes_1_2_3.py")
    if updated < 3:
        print("(Some URLs may still need manual entry — check the file.)")


if __name__ == "__main__":
    main()
