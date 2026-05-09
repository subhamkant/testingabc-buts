import os
import json
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# `youtube` scope is needed for playlist creation/insert. Older tokens issued
# with only upload+readonly will still work for uploads — playlist auto-add
# silently degrades to a non-fatal warning until re-auth is performed.
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube",
]

CATEGORY_ID = "27"   # Education  (22 = People & Blogs, 24 = Entertainment)

# Series → playlist title. Created on demand if missing; resolved IDs cached
# in assets/playlist_ids.json so we don't re-list playlists every upload.
_PLAYLIST_TITLES = {
    "mahabharata": "Mahabharata Stories",
    "whatif":      "What If — Vyasa AI",
}
_PLAYLIST_DESCRIPTIONS = {
    "mahabharata": "Cinematic short stories from the Mahabharata epic — Krishna, Arjuna, Karna, Draupadi, and the eternal lessons of dharma.",
    "whatif":      "What if reality bent for a moment? Curiosity-driven thought experiments about Earth, science, nature, and the cosmos.",
}
_PLAYLIST_CACHE_PATH = os.path.join("assets", "playlist_ids.json")


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_youtube_service():
    """
    Loads or refreshes OAuth2 credentials.
    First run: opens browser for consent.
    Subsequent runs: auto-refreshes from token.pickle.
    """
    creds = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as f:
            creds = pickle.load(f)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "client_secrets.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as f:
            pickle.dump(creds, f)

    return build("youtube", "v3", credentials=creds)


# ── Series-specific tag bundles ──────────────────────────────────────────────

_TAGS_MAHABHARATA = [
    "Mahabharata", "महाभारत", "Shorts", "Hindu mythology",
    "Ancient India", "Bhagavad Gita", "भगवद गीता",
    "Krishna", "कृष्ण", "Arjuna", "अर्जुन",
    "epic story", "dharma", "spiritual", "Indian history",
    "Mahabharata shorts", "mythology shorts", "trending shorts",
    "Hindu dharma", "vedic wisdom", "kurukshetra", "krishna stories",
    "Indian mythology", "spiritual shorts",
]
_TAGS_WHATIF = [
    "what if", "hypothetical", "thought experiment", "Shorts",
    "science", "curiosity", "speculative", "science shorts",
    "what if scenarios", "science what if", "alternate reality",
    "future earth", "mind blowing", "trending shorts",
    "science facts", "speculative science", "imagine",
    "क्या होगा अगर", "विज्ञान", "कल्पना",
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
    # Default: Mahabharata
    base = list(_TAGS_MAHABHARATA)
    if language == "hi":
        base += ["हिंदी शॉर्ट्स", "हिंदी कहानी", "पौराणिक कथा", "महाकाव्य", "हिंदू धर्म"]
    else:
        base += ["English shorts", "mythology explained", "epic history", "Hindu stories"]
    return base


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
) -> str:
    """
    Uploads video to YouTube, sets thumbnail, and adds to the series playlist.
    Returns the YouTube video ID.
    """
    youtube = get_youtube_service()

    # Series can be passed explicitly or read from script_data
    if series is None:
        series = script_data.get("series", "mahabharata")

    # Build tag list — series-specific bundle + script-supplied tags, deduped, capped at 30
    base_tags = _series_tag_pack(series, language)
    all_tags  = list(dict.fromkeys(script_data.get("tags", []) + base_tags))[:30]

    # Ensure #Shorts is in description — required for Shorts algorithm classification
    description = script_data.get("description", "")
    if "#Shorts" not in description:
        description += "\n\n#Shorts"

    # Trim title to 60 chars (Shorts display limit). Ensure What If prefix.
    title = script_data["title"][:60]
    if series == "whatif" and not title.lower().startswith("what if"):
        title = ("What If: " + title)[:60]

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": all_tags,
            "categoryId": CATEGORY_ID,
            "defaultLanguage": language,
        },
        "status": {
            "privacyStatus": "public",
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

    return video_id
