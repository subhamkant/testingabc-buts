"""
Download free background music for Mahabharata Bot
====================================================
Downloads a royalty-free epic/ambient track and saves it as assets/bgmusic.mp3.
Then updates .env to activate it.

Run once:
    python download_music.py

Music source: Free Music Archive / incompetech.com (CC BY 4.0 — Kevin MacLeod)
These tracks are free for commercial use with attribution.
"""

import os
import requests

MUSIC_DIR  = "assets"
MUSIC_PATH = os.path.join(MUSIC_DIR, "bgmusic.mp3")

# Kevin MacLeod — "Angkor Wat" — epic cinematic ambient, CC BY 4.0
# Perfect for Mahabharata narration backdrop
TRACKS = [
    {
        "name": "Angkor Wat (Kevin MacLeod)",
        "url":  "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Angkor%20Wat.mp3",
    },
    {
        "name": "Dreaming of Earth (Kevin MacLeod)",
        "url":  "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Dreaming%20of%20Earth.mp3",
    },
    {
        "name": "Himalaya (Kevin MacLeod)",
        "url":  "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Himalaya.mp3",
    },
]


def download_track(url: str, dest: str) -> bool:
    try:
        print(f"  Downloading: {url[:60]}...")
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code == 200 and len(resp.content) > 10_000:
            with open(dest, "wb") as f:
                f.write(resp.content)
            size_kb = os.path.getsize(dest) // 1024
            print(f"  Saved {size_kb} KB -> {dest}")
            return True
        print(f"  HTTP {resp.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
    return False


def activate_in_env(path: str):
    env_path = ".env"
    abs_path = path.replace("\\", "/")
    updated = False

    if os.path.exists(env_path):
        lines = open(env_path, encoding="utf-8").readlines()
        new_lines = []
        for line in lines:
            if line.strip().startswith("BACKGROUND_MUSIC_PATH"):
                new_lines.append(f"BACKGROUND_MUSIC_PATH={abs_path}\n")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"\nBACKGROUND_MUSIC_PATH={abs_path}\n")
        open(env_path, "w", encoding="utf-8").writelines(new_lines)
    else:
        with open(env_path, "a", encoding="utf-8") as f:
            f.write(f"\nBACKGROUND_MUSIC_PATH={abs_path}\n")
    print(f"  .env updated: BACKGROUND_MUSIC_PATH={abs_path}")


def main():
    os.makedirs(MUSIC_DIR, exist_ok=True)

    if os.path.exists(MUSIC_PATH) and os.path.getsize(MUSIC_PATH) > 10_000:
        print(f"[OK] Music already exists at {MUSIC_PATH}")
        activate_in_env(MUSIC_PATH)
        return

    print("Downloading background music...")
    for track in TRACKS:
        print(f"\nTrying: {track['name']}")
        if download_track(track["url"], MUSIC_PATH):
            print(f"\n[OK] Background music ready: {MUSIC_PATH}")
            activate_in_env(MUSIC_PATH)
            print("\nAttribution (required for CC BY): Music by Kevin MacLeod (incompetech.com)")
            print("Licensed under Creative Commons: By Attribution 4.0 License")
            return

    print("\n[!] Auto-download failed. Manual steps:")
    print("  1. Go to https://pixabay.com/music/search/epic%20indian/")
    print("  2. Download any free epic/ambient track as MP3")
    print("  3. Save it as: assets/bgmusic.mp3")
    print("  4. Add to .env: BACKGROUND_MUSIC_PATH=assets/bgmusic.mp3")


if __name__ == "__main__":
    main()
