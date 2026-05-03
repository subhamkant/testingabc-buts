import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

CATEGORY_ID = "27"   # Education  (22 = People & Blogs, 24 = Entertainment)


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


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_to_youtube(
    video_path: str,
    script_data: dict,
    language: str,
    thumbnail_path: str = "",
) -> str:
    """
    Uploads video to YouTube and optionally sets thumbnail.
    Returns the YouTube video ID.
    """
    youtube = get_youtube_service()

    # Build extra tags
    base_tags = [
        "Mahabharata", "महाभारत", "Shorts", "Hindu mythology",
        "Ancient India", "Bhagavad Gita", "भगवद गीता",
        "Krishna", "कृष्ण", "Arjuna", "अर्जुन",
        "epic story", "dharma", "spiritual", "Indian history",
        "Mahabharata shorts", "mythology shorts", "trending shorts",
        "Hindu dharma", "vedic wisdom", "kurukshetra", "krishna stories",
        "Indian mythology", "spiritual shorts",
    ]
    if language == "hi":
        base_tags += ["हिंदी शॉर्ट्स", "हिंदी कहानी", "पौराणिक कथा", "महाकाव्य", "हिंदू धर्म"]
    else:
        base_tags += ["English shorts", "mythology explained", "epic history", "Hindu stories"]

    all_tags = list(dict.fromkeys(script_data.get("tags", []) + base_tags))[:30]

    # Ensure #Shorts is in description — required for Shorts algorithm classification
    description = script_data.get("description", "")
    if "#Shorts" not in description:
        description += "\n\n#Shorts"

    # Trim title to 60 chars (Shorts display limit)
    title = script_data["title"][:60]

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

    return video_id
