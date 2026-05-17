"""
One-off helper: manually upload Bhishma arc episodes #1, #2, #3 to YouTube
using the local Tier 1 / Tier 1.5 / Tier 2 videos respectively.

For each episode it:
  • Loads the SEO metadata (description, tags) from the matching cache's
    script.json — these prompts have been tuned with hashtag blocks,
    long-tail tags, and high-volume keywords for max discovery
  • Overrides the title where the cache emitted a mismatched one
  • Calls pipeline.youtube_uploader.upload_to_youtube() — which already
    plumbs the series tag pack, auto-appends #Shorts to description,
    sets language metadata, category, public privacy, and adds to playlist
  • Cross-posts to Instagram Reels (non-fatal) via pipeline.instagram_uploader

Run with `--dry-run` first to print the metadata that WILL be uploaded
without actually pushing to YouTube. Verify titles/descriptions, then
re-run without --dry-run.

Usage:
    python upload_episodes_1_2_3.py --dry-run        # preview
    python upload_episodes_1_2_3.py                  # YT + IG upload all
    python upload_episodes_1_2_3.py --only 2         # YT + IG upload only #2
    python upload_episodes_1_2_3.py --ig-backfill    # IG only (uses YOUTUBE_URLS dict)
    python upload_episodes_1_2_3.py --ig-backfill --only 1   # IG-backfill only #1
"""

import os
import sys
import json
import argparse
import io

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

# Force public privacy explicitly for these manual re-uploads (yesterday's
# default was already public; setting explicitly so YT_PRIVACY in .env
# doesn't accidentally make them unlisted).
os.environ["YT_PRIVACY"] = "public"

from pipeline.youtube_uploader import upload_to_youtube
from pipeline.instagram_uploader import upload_to_instagram


# For --ig-backfill mode: paste the YouTube URLs for the 3 already-live
# videos so IG captions can include the cross-promo backlink. Get URLs
# from YouTube Studio's Content listing.
YOUTUBE_URLS = {
    1: "https://youtube.com/watch?v=o5L2QNw1L4M",                                            # paste #1 (Bhishma vow) URL
    2: "https://youtube.com/watch?v=WVAYsbm8rvc",    # #2 — known (re-uploaded 2026-05-17)
    3: "https://youtube.com/watch?v=L1ZPCZJLDe0",                                            # paste #3 (Bhishma silence) URL
}


# Each episode: (episode_num, video_path, cache_dir_for_metadata,
#                title_override_or_None, thumbnail_path_or_None)
EPISODES = [
    {
        "n": 1,
        "video": "output/video_hi_20260516_002630.mp4",
        "cache": "cache/mahabharata_hi_20260515_18",  # yesterday's cron, same topic
        "title_override": None,  # cache 18 already has correct "महाभारत #1: ..." title
        "thumbnail": None,  # YouTube auto-generates from first frame; can replace manually in Studio
        "label": "Bhishma's vow of celibacy",
    },
    {
        "n": 2,
        # Re-rendered 2026-05-17 with Pixabay music (stereo_color-indian-
        # cinematic-484488.mp3) after the original used mahabharat_sad_theme
        # which hit Content ID. Original blocked + unlisted on YouTube.
        "video": "output/video_hi_20260516_232139_REROLL_PIXABAY_MUSICADDED.mp4",
        "cache": "cache/mahabharata_hi_20260516_09",
        # Cache 09 LLM emitted wrong title ("Bhishma's Painful Vow") AND a
        # description for the wrong topic ("vow" instead of "raising both
        # factions"). Override BOTH for topic-aligned SEO.
        "title_override": "महाभारत #2: भीष्म दादा की दुविधा | Bhishma Raising Pandavas & Kauravas",
        "description_override": (
            "भीष्म पितामह ने दोनों पक्षों — पांडवों और कौरवों — को एक साथ पाला... "
            "अपनी ही रक्त की दो शाखाओं को विरोधी बनाते देखा। यही उनकी सबसे गहरी पीड़ा थी।\n\n"
            "#Shorts #Mahabharata #महाभारत #Bhishma #भीष्म #HinduMythology\n\n"
            "जब भीष्म ने हस्तिनापुर के सिंहासन का त्याग किया, तब उन्होंने अपने भाई "
            "विचित्रवीर्य के पुत्रों को पाला — पांडु और धृतराष्ट्र। फिर उनके पोते-पोतियों — "
            "पांडवों और कौरवों — को भी अपने ही हाथों से बड़ा किया। दादा होते हुए भी, "
            "कुरुक्षेत्र के युद्ध में वे एक तरफ खड़े हुए। एक तरफ युधिष्ठिर, अर्जुन, भीम — "
            "जिन्हें उन्होंने धर्म और न्याय सिखाया था। दूसरी तरफ दुर्योधन — जिसके अधर्म को "
            "वे रोक नहीं सके। वो रिश्ता जो खून का था, वो प्रतिज्ञा जो वचन का था, और वो "
            "धर्म जो टूट गया था। क्या भीष्म पितामह सच में चुप रह सकते थे? Comment में बताओ।\n\n"
            "#Shorts #Mahabharata #महाभारत #Bhishma #भीष्म #BhishmaPitamah #भीष्मपितामह "
            "#Pandavas #पांडव #Kauravas #कौरव #Duryodhana #दुर्योधन #Yudhishthira #युधिष्ठिर "
            "#Arjuna #अर्जुन #Bhima #भीम #Hastinapura #हस्तिनापुर #Kurukshetra #कुरुक्षेत्र "
            "#Krishna #कृष्ण #BhagavadGita #भगवद_गीता #Dharma #धर्म #HinduMythology "
            "#IndianMythology #AncientIndia #EpicStory #MythologyShorts #VedicWisdom "
            "#HinduDharma #IndianHistory #SpiritualShorts #PauranikKathayein "
            "#भारतीयइतिहास #SanatanDharma #सनातनधर्म #HindiShorts #trending"
        ),
        "tags_override": [
            # Topic-specific long-tail (highest SEO leverage)
            "Bhishma raising Pandavas Kauravas", "भीष्म पितामह की कहानी",
            "भीष्म दादा", "Bhishma grandfather Pandavas",
            "why Bhishma trained both sides", "भीष्म ने दोनों पक्षों को क्यों पाला",
            "Bhishma Kuru dynasty", "Bhishma dharma dilemma",
            "Pandavas Kauravas childhood", "Hastinapura royal family",
            # Named characters
            "Bhishma", "भीष्म", "Bhishma Pitamah", "भीष्म पितामह",
            "Pandavas", "पांडव", "Kauravas", "कौरव",
            "Duryodhana", "दुर्योधन", "Yudhishthira", "युधिष्ठिर",
            "Arjuna", "अर्जुन", "Bhima", "भीम",
            # General fallbacks
            "Mahabharata", "महाभारत", "Shorts", "Hindu mythology", "Krishna",
        ],
        "thumbnail": None,
        "label": "Bhishma raising the Pandavas + Kauravas",
    },
    {
        "n": 3,
        "video": "output/video_hi_20260516_211919_REROLL.mp4",
        "cache": "cache/mahabharata_hi_20260516_14",
        # Cache 14 title says "#13" because forced-topic fell back to
        # `len(used)+1` numbering. Override to "#3".
        "title_override": "महाभारत #3: द्रौपदी का अपमान | Bhishma's Shocking Silence",
        "thumbnail": "output/thumbnail.jpg",  # current thumbnail.jpg is for Tier 2 vastraharan
        "label": "Bhishma's silence during Draupadi's vastraharan (Tier 2)",
    },
]


def load_script(cache_dir: str) -> dict:
    path = os.path.join(cache_dir, "script.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def adjust_description_episode_number(desc: str, new_n: int) -> str:
    """
    Some Tier 2 descriptions include "▶️ अगला भाग: <next topic>" prepended.
    This stays correct as-is. But if description body mentions "#13" or
    similar bad-numbering text, normalize.
    """
    # Not strictly necessary — descriptions usually don't reference episode #.
    return desc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print metadata + exit, do not upload")
    parser.add_argument("--only", type=int, default=None,
                        help="Upload only the given episode # (1, 2, or 3)")
    parser.add_argument("--ig-backfill", action="store_true",
                        help="IG-only mode: cross-post to Instagram using "
                             "the YOUTUBE_URLS dict at the top of this file. "
                             "Skips the YouTube upload step entirely. Use "
                             "for backfilling IG for videos already on YT.")
    args = parser.parse_args()

    targets = [e for e in EPISODES if args.only is None or e["n"] == args.only]
    if not targets:
        sys.exit(f"No matching episode for --only={args.only}")

    mode_label = "[DRY RUN — no actual upload]" if args.dry_run else (
        "[IG-BACKFILL — IG only, no YT]" if args.ig_backfill else "[YT + IG combined]"
    )
    print(f"{'=' * 70}")
    print(f"  Manual upload — {len(targets)} episode(s) {mode_label}")
    print(f"{'=' * 70}\n")

    for ep in targets:
        n = ep["n"]
        video = ep["video"]
        cache = ep["cache"]
        title_override = ep["title_override"]
        thumb = ep["thumbnail"]
        label = ep["label"]

        if not os.path.exists(video):
            print(f"[FAIL] #{n} ({label}): video missing at {video}")
            continue
        if not os.path.exists(os.path.join(cache, "script.json")):
            print(f"[FAIL] #{n} ({label}): script.json missing at {cache}")
            continue

        script = load_script(cache)
        original_title = script.get("title", "")
        if title_override:
            script["title"] = title_override

        # Description override (when cache had wrong-topic body)
        desc_override = ep.get("description_override")
        if desc_override:
            script["description"] = desc_override
        else:
            script["description"] = adjust_description_episode_number(
                script.get("description", ""), n
            )

        # Tags override (topic-aligned long-tail keywords)
        tags_override = ep.get("tags_override")
        if tags_override:
            script["tags"] = tags_override

        size_mb = os.path.getsize(video) / (1024 * 1024)
        print(f"--- #{n}: {label} ---")
        print(f"    video       : {video} ({size_mb:.1f} MB)")
        print(f"    cache       : {cache}")
        print(f"    title (orig): {original_title[:80]}")
        if title_override:
            print(f"    title (NEW) : {title_override[:80]}")
        print(f"    thumbnail   : {thumb or '(YouTube auto-generate)'}")
        n_tags = len(script.get("tags", []))
        desc_len = len(script.get("description", ""))
        print(f"    tags        : {n_tags} entries")
        print(f"    desc length : {desc_len} chars")
        print(f"    desc (first line): {(script.get('description') or '').splitlines()[0][:100]}")
        print()

        if args.dry_run:
            print(f"    [dry-run] skipped\n")
            continue

        # ── IG-backfill mode: skip YT, only push to Instagram ────────
        if args.ig_backfill:
            yt_url = YOUTUBE_URLS.get(n, "").strip()
            if not yt_url:
                print(f"    [FAIL] #{n}: YOUTUBE_URLS[{n}] is empty — "
                      f"paste the YT URL into the dict at top of file first.\n")
                continue
            print(f"    YT URL (for caption backlink): {yt_url}")
            print(f"    Posting to Instagram...")
            try:
                ig_media_id = upload_to_instagram(
                    video_path=video,
                    script_data=script,
                    youtube_url=yt_url,
                )
                if ig_media_id:
                    print(f"    [OK] #{n} IG -> media_id={ig_media_id}\n")
                else:
                    print(f"    [FAIL] #{n} IG returned no media_id\n")
            except Exception as e:
                print(f"    [FAIL] #{n} IG upload error: {e}\n")
            continue

        # ── Combined mode: YT first, then IG cross-post ──────────────
        try:
            print(f"    Uploading to YouTube...")
            video_id = upload_to_youtube(
                video_path=video,
                script_data=script,
                language="hi",
                thumbnail_path=thumb or "",
                series="mahabharata",
            )
            yt_url = f"https://youtube.com/watch?v={video_id}"
            print(f"    [OK] #{n} YT -> {yt_url}")
        except Exception as e:
            print(f"    [FAIL] #{n} YT upload error: {e}\n")
            continue

        # IG cross-post (non-fatal — same pattern as main.py)
        try:
            ig_media_id = upload_to_instagram(
                video_path=video,
                script_data=script,
                youtube_url=yt_url,
            )
            if ig_media_id:
                print(f"    [OK] #{n} IG -> media_id={ig_media_id}\n")
            else:
                print(f"    [skip] #{n} IG returned no media_id (env not configured?)\n")
        except Exception as ig_err:
            print(f"    Instagram upload failed (non-fatal): {ig_err}\n")

    if args.dry_run:
        print("\n=== DRY RUN COMPLETE — no uploads made ===")
        print("Re-run without --dry-run to actually upload.\n")


if __name__ == "__main__":
    main()
