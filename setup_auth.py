"""
🔑  One-Time YouTube OAuth Setup
=================================
Run this ONCE on your local machine before deploying to GitHub Actions.
It will open a browser window for you to log in with your YouTube account.

After login, it creates:
  - token.pickle          (used by the bot)
  - token_base64.txt      (paste this into GitHub Secrets as YOUTUBE_TOKEN_B64)

Steps:
  1. pip install -r requirements.txt
  2. Place your client_secrets.json in this folder
  3. python setup_auth.py
  4. Copy the printed base64 string into GitHub Secret: YOUTUBE_TOKEN_B64
"""

import os
import pickle
import base64
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    # `youtube` scope is needed for playlist creation/insert (used to auto-add
    # uploads to per-series playlists). Re-run this script and update the
    # YOUTUBE_TOKEN_B64 secret to grant it.
    "https://www.googleapis.com/auth/youtube",
    # `yt-analytics.readonly` enables YouTube Analytics API queries
    # (watch time, retention, traffic sources, demographics, geography).
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def main():
    if not os.path.exists("client_secrets.json"):
        print("[ERROR] client_secrets.json not found!")
        print()
        print("To get it:")
        print("  1. Go to https://console.cloud.google.com/")
        print("  2. Create a project → Enable 'YouTube Data API v3'")
        print("  3. OAuth consent screen → Add your Gmail as test user")
        print("  4. Credentials → Create OAuth 2.0 Client ID (Desktop App)")
        print("  5. Download JSON → rename to client_secrets.json")
        return

    print("[*] Opening browser for YouTube login...")
    flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
    creds = flow.run_local_server(port=0)

    # Save token.pickle
    with open("token.pickle", "wb") as f:
        pickle.dump(creds, f)
    print("[OK] token.pickle saved")

    # Test the connection
    youtube = build("youtube", "v3", credentials=creds)
    channels = youtube.channels().list(part="snippet", mine=True).execute()
    items = channels.get("items", [])
    if items:
        channel_name = items[0]["snippet"]["title"]
        print(f"[OK] Connected to YouTube channel: '{channel_name}'")
    else:
        print("[!] Login succeeded but no YouTube channel found on this account.")
        print("    Create a YouTube channel first, then re-run this script.")

    # Encode for GitHub Secrets
    with open("token.pickle", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    with open("token_base64.txt", "w") as f:
        f.write(b64)

    print()
    print("=" * 60)
    print("COPY THIS INTO GITHUB SECRET 'YOUTUBE_TOKEN_B64':")
    print("=" * 60)
    print(b64[:80] + "...")
    print("(Full value saved in token_base64.txt)")
    print()
    print("Also add these GitHub Secrets:")
    print("  GEMINI_API_KEY  ->  your Gemini API key from ai.google.dev")


if __name__ == "__main__":
    main()
