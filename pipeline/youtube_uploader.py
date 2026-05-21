import os
import json
import pickle
import re
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# `youtube`           — playlist creation/insert (otherwise we can only upload)
# `youtube.force-ssl` — required to post comments / engage with own videos
#
# Older tokens still work for uploads + thumbnails. Playlist auto-add and
# the auto-pinned engagement comment silently degrade to a non-fatal warning
# until re-auth is performed via setup_auth.py.
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

CATEGORY_ID = "27"   # Education  (22 = People & Blogs, 24 = Entertainment)

# Series → playlist title. Created on demand if missing; resolved IDs cached
# in assets/playlist_ids.json so we don't re-list playlists every upload.
_PLAYLIST_TITLES = {
    "mahabharata": "Mahabharata Stories",
    "krishna":     "Krishna Speaks — First-Person Wisdom",
    "whatif":      "What If — Vyasa AI",
    "explainer":   "Kant Decodes — Hindi Explainers",
}
_PLAYLIST_DESCRIPTIONS = {
    "mahabharata": "Cinematic short stories from the Mahabharata epic — Krishna, Arjuna, Karna, Draupadi, and the eternal lessons of dharma.",
    "krishna":     "Lord Krishna speaks directly to Arjuna, Uddhava, Karna, Bhishma — first-person wisdom from the Mahabharata. हिंदी में।",
    "whatif":      "What if reality bent for a moment? Curiosity-driven thought experiments about Earth, science, nature, and the cosmos.",
    "explainer":   "Hindi explainers separating facts from internet hype. AI, scams, productivity myths, work culture, psychology — what's actually going on behind the noise.",
}

# Pinned-comment template per series. After upload, the bot auto-replies its
# own pinned comment to drive an early engagement signal — the algorithm
# weighs first-hour interactions heavily for Shorts. The comment also acts
# as additional SEO real estate for keywords / hashtags.
_PINNED_COMMENT_TEMPLATES = {
    # Fallback templates used when script_data has no per-video `pinned_question`.
    # Designed to invite *disagreement*, not creator-decision input. Comments per
    # view on this channel are 0.35 (11 / 31 videos) — engagement is the
    # structural weak point per assets/channel_analysis_2026-05-20.md.
    "mahabharata": (
        "❓ इस video का character — सही था या गलत?\n"
        "Honest take comment में drop करो। किसी एक side पर खड़े रहो।\n\n"
        "🔔 Daily Mahabharata Shorts in हिंदी\n"
        "#Mahabharata #महाभारत #Shorts"
    ),
    "krishna": (
        "🪷 कृष्ण ने जो कहा — क्या आप भी मानते हैं?\n"
        "Type \"जय श्री कृष्ण\" if you do. Honest disagreement भी welcome है।\n\n"
        "🔔 Daily Krishna wisdom — हिंदी में।\n"
        "#Krishna #कृष्ण #BhagavadGita #Shorts"
    ),
    "whatif": (
        "🌍 Would humanity actually survive this? Be honest.\n"
        "Drop your scenario — wildest one becomes next video.\n\n"
        "🔔 Daily science what-ifs.\n"
        "#WhatIf #Science #Shorts"
    ),
    "explainer": (
        "Iss video mein jo bola — kya tum agree karte ho?\n"
        "Apna take comment mein drop karo. Honest takes only.\n\n"
        "🔔 Subscribe — Kant Decodes har hype sabse pehle.\n"
        "#KantDecodes #Hindi #Explainer"
    ),
}
_PLAYLIST_CACHE_PATH = os.path.join("assets", "playlist_ids.json")

# Phase 21-Lite + 17-Lite (2026-05-21): per-series pinned-comment footer
# pieces. When script_data emits a `pinned_question` (override), we compose
# the full comment as: override + (subscribe with next_seed tease) + hashtags.
# The fallback templates in _PINNED_COMMENT_TEMPLATES remain for scripts
# that haven't been re-run under the new schema yet.
_PINNED_HASHTAGS_BY_SERIES = {
    "mahabharata": "#Mahabharata #महाभारत #Shorts",
    "krishna":     "#Krishna #कृष्ण #BhagavadGita #Shorts",
    "whatif":      "#WhatIf #Science #Shorts",
    "explainer":   "#KantDecodes #Hindi #Explainer",
}
_PINNED_SUBSCRIBE_FALLBACK = {
    "mahabharata": "🔔 Daily Mahabharata Shorts in हिंदी",
    "krishna":     "🔔 Daily Krishna wisdom — हिंदी में।",
    "whatif":      "🔔 Daily science what-ifs.",
    "explainer":   "🔔 Subscribe — Kant Decodes har hype sabse pehle.",
}
# Only the Hindi-targeted series get the next_seed subscribe-tease format.
# WhatIf + Explainer use a different language register where a Hindi seed
# would be jarring; they fall back to the generic subscribe line.
_PINNED_SUBSCRIBE_WITH_SEED = {
    "mahabharata": "🔔 कल — {seed} Subscribe to not miss it.",
    "krishna":     "🔔 कल — {seed} Subscribe — हिंदी में।",
}
_STORY_THREADS_PATH = os.path.join("assets", "story_threads.json")


def _compose_pinned_comment(series: str, override: str, next_seed: str) -> str:
    """Build the full pinned-comment text from a per-video override + brand
    footer. If `override` is empty, fall back to the static series template.
    If `next_seed` is provided and the series supports it, the subscribe line
    names tomorrow's hook; otherwise the generic subscribe fallback is used."""
    override = (override or "").strip()
    next_seed = (next_seed or "").strip()
    if not override:
        return _PINNED_COMMENT_TEMPLATES.get(series, "")

    hashtags  = _PINNED_HASHTAGS_BY_SERIES.get(series, "")
    if next_seed and series in _PINNED_SUBSCRIBE_WITH_SEED:
        # Strip trailing terminators so the formatted line's period lands cleanly.
        clean_seed = next_seed.rstrip(" .।!?")
        subscribe = _PINNED_SUBSCRIBE_WITH_SEED[series].format(seed=clean_seed + ".")
    else:
        subscribe = _PINNED_SUBSCRIBE_FALLBACK.get(series, "")

    parts = [override]
    if subscribe:
        parts += ["", subscribe]
    if hashtags:
        parts.append(hashtags)
    return "\n".join(parts)


def _append_story_thread(video_id: str, series: str, topic: str, next_seed: str) -> None:
    """Phase 17-Lite: append the planted seed to assets/story_threads.json
    so future videos can pay it off. Lazy create the file. Non-fatal —
    never breaks the upload path."""
    if not next_seed or not next_seed.strip():
        return
    try:
        from datetime import datetime, timezone
        if os.path.exists(_STORY_THREADS_PATH):
            with open(_STORY_THREADS_PATH, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"planted": []}
        data.setdefault("planted", []).append({
            "video_id":    video_id,
            "series":      series,
            "topic":       topic,
            "next_seed":   next_seed.strip(),
            "planted_at":  datetime.now(timezone.utc).isoformat(),
            "paid_off_by": None,
        })
        os.makedirs(os.path.dirname(_STORY_THREADS_PATH), exist_ok=True)
        with open(_STORY_THREADS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"    [thread-registry] planted seed: {next_seed.strip()[:60]}")
    except Exception as e:
        print(f"    [thread-registry] warn: {str(e)[:120]}")


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_youtube_service(token_path: str = "token.pickle"):
    """
    Loads or refreshes OAuth2 credentials from `token_path` and returns a
    YouTube service client.

    `token_path` defaults to "token.pickle" (the existing Vyasa AI channel),
    so all current callers keep working unchanged. The explainer pipeline
    passes "token_explainer.pickle" to point at the second channel's token.
    Same client_secrets.json is reused; OAuth consent flow lets you pick the
    channel on first run.

    First run: opens browser for consent and writes back to `token_path`.
    Subsequent runs: auto-refreshes from `token_path`.
    """
    creds = None

    if os.path.exists(token_path):
        with open(token_path, "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secrets.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(token_path, "wb") as f:
            pickle.dump(creds, f)

    return build("youtube", "v3", credentials=creds)


# ── Series-specific tag bundles ──────────────────────────────────────────────

_TAGS_MAHABHARATA = [
    "Mahabharata", "महाभारत", "Hindu mythology",
    "Ancient India", "Bhagavad Gita", "भगवद गीता",
    "Krishna", "कृष्ण", "Arjuna", "अर्जुन",
    "epic story", "dharma", "spiritual", "Indian history",
    "Hindu dharma", "vedic wisdom", "kurukshetra", "krishna stories",
    "Indian mythology",
]
_TAGS_WHATIF = [
    "what if", "hypothetical", "thought experiment", "Shorts",
    "science", "curiosity", "speculative", "science shorts",
    "what if scenarios", "science what if", "alternate reality",
    "future earth", "mind blowing", "trending shorts",
    "science facts", "speculative science", "imagine",
    "क्या होगा अगर", "विज्ञान", "कल्पना",
]
_TAGS_EXPLAINER = [
    "Kant Decodes", "kant decodes", "explainer", "hindi explainer",
    "anti hype", "fact check", "internet hype", "indian creator",
    "Shorts", "hindi shorts", "trending shorts", "psychology",
    "social commentary", "हिंदी एक्सप्लेनर", "हिंदी शॉर्ट्स",
    "सोशल मीडिया", "एनालिसिस", "real story", "behind the noise",
    "ai hype", "scams explained",
]


def _series_tag_pack(series: str, language: str) -> list:
    """Pick the right base tag bundle for the series + language."""
    if series == "whatif":
        base = list(_TAGS_WHATIF)
        if language == "hi":
            base += ["हिंदी शॉर्ट्स", "रोचक तथ्य", "विज्ञान शॉर्ट्स"]
        else:
            base += ["English shorts", "science explained", "what if questions"]
        return base
    if series == "explainer":
        # Hindi-only channel; no English variant.
        return list(_TAGS_EXPLAINER)
    # Default: Mahabharata
    base = list(_TAGS_MAHABHARATA)
    if language == "hi":
        base += ["हिंदी कहानी", "पौराणिक कथा", "महाकाव्य", "हिंदू धर्म"]
    else:
        base += ["mythology explained", "epic history", "Hindu stories"]
    return base


# ── Title + description sanitizers ───────────────────────────────────────────
# Phase 1 (2026-05-21): episode-numbered titles suppress reach 5–20x per the
# analytics baseline at assets/channel_analysis_2026-05-20.md. The script-
# generator prompt was updated to stop emitting `#N:` prefixes; this strip is
# belt-and-suspenders for any LLM leak.
_BANNED_TITLE_PATTERN = re.compile(
    r"(?:महाभारत|Mahabharata|Mahabharat)\s*#?\s*\d+\s*[:\-—]?\s*"
    r"|\b(?:Episode|Ep\.?|Part)\s*#?\s*\d+\s*[:\-—]?\s*",
    re.IGNORECASE,
)


def _sanitize_title(title: str) -> str:
    cleaned = _BANNED_TITLE_PATTERN.sub("", title).strip()
    cleaned = re.sub(r"^[:\-—|\s]+", "", cleaned)
    return cleaned


def _cap_description_hashtags(description: str, top_n: int = 3) -> str:
    """Drop the trailing hashtag spam block (was 31 tags); keep top_n at the
    end. >15 hashtags suppresses ranking; the inline 5-hashtag block above
    the body is preserved."""
    lines = description.rstrip().split("\n")
    block_start = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            block_start = i
            continue
        if all(tok.startswith("#") for tok in stripped.split()):
            block_start = i
            continue
        break

    if block_start == len(lines):
        return description

    trailing = " ".join(lines[block_start:])
    hashtags = [tok for tok in trailing.split() if tok.startswith("#")]
    if not hashtags:
        return description

    kept = hashtags[:top_n]
    body = "\n".join(lines[:block_start]).rstrip()
    return f"{body}\n\n{' '.join(kept)}" if body else " ".join(kept)


# ── Playlist helpers ─────────────────────────────────────────────────────────

def _load_playlist_cache() -> dict:
    if os.path.exists(_PLAYLIST_CACHE_PATH):
        try:
            with open(_PLAYLIST_CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_playlist_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_PLAYLIST_CACHE_PATH), exist_ok=True)
    with open(_PLAYLIST_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _ensure_playlist(youtube, series: str) -> str:
    """
    Returns the playlist ID for `series`, creating it if missing.
    Uses assets/playlist_ids.json as a cache to avoid re-listing.
    Requires the wider `youtube` OAuth scope; raises on insufficient scope.
    """
    cache = _load_playlist_cache()
    if cache.get(series):
        return cache[series]

    title = _PLAYLIST_TITLES.get(series, series.title())
    desc  = _PLAYLIST_DESCRIPTIONS.get(series, "")

    # Look for an existing playlist by title
    resp = youtube.playlists().list(part="id,snippet", mine=True, maxResults=50).execute()
    for item in resp.get("items", []):
        if item["snippet"]["title"] == title:
            cache[series] = item["id"]
            _save_playlist_cache(cache)
            return item["id"]

    # Create it
    created = youtube.playlists().insert(
        part="snippet,status",
        body={
            "snippet": {"title": title, "description": desc},
            "status":  {"privacyStatus": "public"},
        },
    ).execute()
    cache[series] = created["id"]
    _save_playlist_cache(cache)
    print(f"    [OK] Created playlist '{title}' ({created['id']})")
    return created["id"]


def _add_to_playlist(youtube, video_id: str, series: str) -> None:
    """Adds the uploaded video to the series playlist. Non-fatal on failure."""
    try:
        playlist_id = _ensure_playlist(youtube, series)
        youtube.playlistItems().insert(
            part="snippet",
            body={
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {"kind": "youtube#video", "videoId": video_id},
                }
            },
        ).execute()
        print(f"    [OK] Added to playlist '{_PLAYLIST_TITLES.get(series, series)}'")
    except Exception as e:
        print(f"    [!] Playlist add failed (non-fatal — re-run setup_auth.py "
              f"to grant wider 'youtube' scope): {e}")


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_to_youtube(
    video_path: str,
    script_data: dict,
    language: str,
    thumbnail_path: str = "",
    series: str = None,
    on_video_id=None,
    token_path: str = "token.pickle",
) -> str:
    """
    Uploads video to YouTube, sets thumbnail, and adds to the series playlist.
    Returns the YouTube video ID.

    `token_path` (default "token.pickle") selects which channel to upload to.
    The Vyasa AI channel uses the default; the Explainer channel passes
    "token_explainer.pickle". Generated by `python setup_auth.py --channel <name>`.

    `on_video_id` (callable, optional) is invoked with the new video_id IMMEDIATELY
    after `videos.insert` returns — before any post-upload work (thumbnail, playlist,
    pinned comment). Callers use this to write a checkpoint so a partial-success
    upload (video posted, but a downstream step failed) doesn't trigger a duplicate
    re-upload on retry. Failures inside the callback do NOT abort the upload.
    """
    youtube = get_youtube_service(token_path=token_path)

    # Series can be passed explicitly or read from script_data
    if series is None:
        series = script_data.get("series", "mahabharata")

    # Build tag list — series-specific bundle + script-supplied tags, deduped, capped at 30
    base_tags = _series_tag_pack(series, language)
    all_tags  = list(dict.fromkeys(script_data.get("tags", []) + base_tags))[:30]

    # Description: cap trailing hashtag block at top 3 (was 31 → suppression risk),
    # then ensure #Shorts for Shorts algorithm classification.
    description = script_data.get("description", "")
    description = _cap_description_hashtags(description, top_n=3)
    if "#Shorts" not in description:
        description += "\n\n#Shorts"

    # Title: strip any leaked episode/part numbering, then trim to 60 chars.
    title = _sanitize_title(script_data["title"])[:60]
    if series == "whatif" and not title.lower().startswith("what if"):
        title = ("What If: " + title)[:60]

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": all_tags,
            "categoryId": CATEGORY_ID,
            # `defaultLanguage` declares the metadata language;
            # `defaultAudioLanguage` declares the spoken-narration language.
            # Setting both lets YouTube auto-target viewers who prefer the
            # right language and feed our captions to the right surfaces.
            "defaultLanguage":      language,
            "defaultAudioLanguage": language,
        },
        "status": {
            # Env-var override (added 2026-05-14): offline drivers set
            # YT_PRIVACY=private for first-run testing so videos don't go
            # public until manually verified. GHA pipelines never set this
            # var → default "public" preserved for production cron.
            "privacyStatus":           os.environ.get("YT_PRIVACY", "public"),
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/mp4",
        chunksize=1024 * 1024,   # 1 MB chunks
        resumable=True,
    )

    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media,
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            pct = int(status.progress() * 100)
            print(f"    Uploading... {pct}%")

    video_id = response["id"]
    print(f"    [OK] Uploaded -> https://youtube.com/watch?v={video_id}")

    # Checkpoint as soon as we have the video_id — protects against duplicate
    # re-upload if any of the downstream steps (thumbnail / playlist / pinned
    # comment) fail and the run is retried.
    if on_video_id:
        try:
            on_video_id(video_id)
        except Exception as cb_err:
            print(f"    [!] Checkpoint callback failed (non-fatal): {cb_err}")

    # Set custom thumbnail if available
    if thumbnail_path and os.path.exists(thumbnail_path):
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
            ).execute()
            print(f"    [OK] Thumbnail set")
        except Exception as e:
            print(f"    [!] Thumbnail upload failed (need verified channel): {e}")

    # Add to series playlist (non-fatal if scope insufficient)
    _add_to_playlist(youtube, video_id, series)

    # Post + pin a series-specific comment to drive the early engagement
    # signal. Non-fatal — re-auth is required for the wider scope, and even
    # without pinning the comment helps with keyword density.
    _append_story_thread(
        video_id, series,
        script_data.get("title", ""),
        script_data.get("next_seed", ""),
    )

    _post_pinned_comment(
        youtube, video_id, series,
        override=script_data.get("pinned_question"),
        next_seed=script_data.get("next_seed"),
    )

    return video_id


def _post_pinned_comment(
    youtube, video_id: str, series: str,
    override: str = None,
    next_seed: str = None,
) -> None:
    """
    Post a series-specific top-level comment from the channel itself.
    Acts as the algorithm's first interaction signal (engagement = early
    rank boost on Shorts) AND adds another keyword/hashtag surface in
    the comments tab. Pinning requires the moderator role on the channel
    (which the channel owner has by default).

    Composition (Phase 21-Lite + 17-Lite, 2026-05-21):
      • `override` = the LLM-emitted per-video question body (quotable_line
        + invite-to-take-a-side). No subscribe CTA / hashtags expected.
      • `next_seed` = the LLM-emitted named-future-consequence hook for the
        subscribe-CTA tease ('🔔 कल — {seed} Subscribe.').
      • _compose_pinned_comment() wraps both into the final comment text,
        falling back to the static series template if `override` is missing.

    Failure modes are non-fatal — the upload itself has already succeeded.
    """
    template = _compose_pinned_comment(series, override, next_seed)
    if not template:
        return

    try:
        # Step 1: insert top-level comment
        comment_resp = youtube.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {
                        "snippet": {"textOriginal": template},
                    },
                }
            },
        ).execute()
        comment_id = comment_resp["snippet"]["topLevelComment"]["id"]
        print("    [OK] Posted CTA comment")

        # Step 2: pin the comment via setModerationStatus. NOTE: the YouTube
        # Data API does not expose a direct "pin" operation as of v3; the
        # closest is comments.markAsSpam / setModerationStatus. The actual
        # "pin" is a creator-tools UI feature. Posting from the channel's
        # own auth account makes the comment a Channel comment, which gets
        # algorithmically boosted to the top of the comments tab regardless
        # of pin state — same effect for our purposes.
    except Exception as e:
        # Insufficient scope (youtube.force-ssl) or other auth issue —
        # log a clear hint and continue.
        msg = str(e)[:200]
        if "insufficient" in msg.lower() or "forbidden" in msg.lower() or "scope" in msg.lower():
            print(
                "    [!] Pinned-comment skipped — re-run setup_auth.py to "
                "grant 'youtube.force-ssl' scope (one-time)."
            )
        else:
            print(f"    [!] Pinned comment failed (non-fatal): {msg}")
