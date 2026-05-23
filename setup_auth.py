"""
🔑  One-Time YouTube OAuth Setup
=================================
Run this ONCE on your local machine before deploying to GitHub Actions.
It will open a browser window for you to log in with your YouTube account.

This codebase manages TWO channels via separate OAuth tokens:
  • Vyasa AI (Mahabharata / Krishna / WhatIf) → token.pickle
  • Anti-Hype Analyzer (Hindi explainer)      → token_explainer.pickle

Both share the same client_secrets.json. During the consent flow you pick
which YouTube channel (Brand Account) the token is for — Google will show
a channel picker after Google login.

Usage:
  python setup_auth.py                       # writes token.pickle      (Vyasa)
  python setup_auth.py --channel vyasa       # same as above
  python setup_auth.py --channel explainer   # writes token_explainer.pickle

After login each command creates:
  - <token_path>             (used by the bot)
  - <token_path>_base64.txt  (paste into the matching GitHub Secret)

GitHub Secret names per channel:
  vyasa     → YOUTUBE_TOKEN_B64
  explainer → YOUTUBE_TOKEN_EXPLAINER_B64

Setup steps:
  1. pip install -r requirements.txt
  2. Place your client_secrets.json in this folder (one file, same for both)
  3. python setup_auth.py --channel <name>
  4. Copy the printed base64 string into the matching GitHub Secret
"""

import argparse
import base64
import os
import pickle

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
    # `youtube.force-ssl` is required to POST comments via commentThreads.insert
    # (the auto long-form-backlink-comment feature). Added 2026-05-14 after
    # Stage 2 Shorts run hit "Insufficient authentication scopes" 403s when
    # trying to post a CTA comment with the long-form link.
    # Note: YouTube API has NO endpoint to PIN a comment — only post it.
    # After this scope is granted, the bot can post the backlink comment on
    # each Short; you pin it manually in YouTube Studio (one click per video).
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


_CHANNELS = {
    # short name → (token file, GitHub Secret name)
    "vyasa":     ("token.pickle",           "YOUTUBE_TOKEN_B64"),
    "explainer": ("token_explainer.pickle", "YOUTUBE_TOKEN_EXPLAINER_B64"),
}


def _run_for(channel: str) -> None:
    if channel not in _CHANNELS:
        raise SystemExit(f"unknown --channel '{channel}'. Valid: {list(_CHANNELS)}")
    token_path, secret_name = _CHANNELS[channel]

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

    print(f"[*] Channel: {channel}  →  token file: {token_path}")
    print(f"[*] Opening browser for YouTube login...")
    print(f"    When the browser asks 'which channel', pick the {channel.upper()} channel.")
    flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
    creds = flow.run_local_server(port=0)

    with open(token_path, "wb") as f:
        pickle.dump(creds, f)
    print(f"[OK] {token_path} saved")

    # Test the connection — print which channel actually got authorized
    youtube = build("youtube", "v3", credentials=creds)
    channels = youtube.channels().list(part="snippet", mine=True).execute()
    items = channels.get("items", [])
    if items:
        channel_name = items[0]["snippet"]["title"]
        print(f"[OK] Connected to YouTube channel: '{channel_name}'")
        if channel == "explainer" and "vyasa" in channel_name.lower():
            print("[!] WARNING: token says you're connected to a Vyasa-ish channel.")
            print("    Did you pick the right channel during OAuth? Re-run if not.")
    else:
        print("[!] Login succeeded but no YouTube channel found on this account.")
        print("    Create a YouTube channel first, then re-run this script.")

    # Encode for GitHub Secrets
    with open(token_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    b64_path = f"{token_path}_base64.txt"
    with open(b64_path, "w") as f:
        f.write(b64)

    print()
    print("=" * 60)
    print(f"COPY THIS INTO GITHUB SECRET '{secret_name}':")
    print("=" * 60)
    print(b64[:80] + "...")
    print(f"(Full value saved in {b64_path})")
    print()
    if channel == "explainer":
        print("Other GitHub Secrets the explainer workflow needs:")
        print("  CLOUDFLARE_ACCOUNT_ID  +  CLOUDFLARE_API_TOKEN   (already shared with image pipeline)")
        print("  YOUTUBE_CLIENT_SECRETS (already shared with vyasa workflow)")
    else:
        print("Also add these GitHub Secrets:")
        print("  GEMINI_API_KEY  ->  your Gemini API key from ai.google.dev")


def main():
    ap = argparse.ArgumentParser(description="One-time YouTube OAuth setup")
    ap.add_argument(
        "--channel", choices=tuple(_CHANNELS.keys()), default="vyasa",
        help="Which channel's token to generate (default: vyasa)",
    )
    args = ap.parse_args()
    _run_for(args.channel)


if __name__ == "__main__":
    main()
