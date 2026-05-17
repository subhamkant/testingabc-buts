"""
Instagram Reels publisher — mirrors YouTube uploads to Instagram via the
Graph API. Designed to run alongside `upload_to_youtube()` so every video
that goes to the channel also goes to the Reels feed.

Architecture:
    1. Upload the local mp4 to a single "instagram-staging" GitHub Release
       as a clobber-able asset → returns a public download URL.
    2. POST that URL to the IG Graph API's `/media` endpoint with
       `media_type=REELS` → returns a container ID.
    3. Poll the container's status_code until FINISHED (≤ 120s).
    4. POST `/media_publish` with the container's creation_id → returns
       the final IG media ID.

All failures are CAUGHT and returned as None — IG hiccups must never
break the YouTube path or the Fix 2.8 state-commit guard.

Env vars consumed:
    IG_ACCESS_TOKEN          — long-lived (~60 day) Graph API token
    IG_BUSINESS_ACCOUNT_ID   — numeric IG business account ID
    GITHUB_TOKEN             — for GH release upload (auto-set in GHA;
                                falls back to `gh auth token` locally)
    GITHUB_REPOSITORY        — "owner/repo" (auto-set in GHA;
                                falls back to git remote parsing)
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import subprocess
from typing import Optional

import requests


# Instagram Graph API base. Pin to v19.0 — newer versions sometimes
# deprecate endpoints we depend on; pinning avoids surprise breakage.
_GRAPH_API = "https://graph.facebook.com/v19.0"

# Container polling: IG container processing usually completes in 30-60s
# for ~60s mp4s but can take up to 2 min on busy days.
_CONTAINER_POLL_TIMEOUT_S = 120
_CONTAINER_POLL_INTERVAL_S = 5

# GitHub Release used as transient public hosting for IG to fetch the mp4.
# Reused across uploads via --clobber semantics.
_RELEASE_TAG = "instagram-staging"
_RELEASE_TITLE = "Instagram staging (transient mp4 host)"
_RELEASE_BODY = (
    "Auto-managed by pipeline/instagram_uploader.py. The single asset "
    "below is the most-recent mp4 that needed to be served to the "
    "Instagram Graph API. Safe to delete this release if you want — it "
    "will be re-created on the next IG upload."
)

# IG caption hard limit is 2200 chars. We cap conservatively at 2150 to
# leave room for the appended YouTube backlink.
_IG_CAPTION_MAX = 2200

# IG silently rejects captions with more than 30 hashtags (caption lands
# EMPTY when the limit is exceeded — this is what bit the 2026-05-17
# backfill, where script descriptions had 39 hashtags). Capping at 28
# leaves a 2-tag safety margin in case the body has inline hashtags we
# missed counting.
_IG_HASHTAG_LIMIT = 28


# ───────────────────────────────────────────────────────────────────────
# GitHub release hosting (transient mp4 hosting for IG to fetch)
# ───────────────────────────────────────────────────────────────────────

def _get_github_token() -> str:
    """
    Return a GitHub token suitable for repo-write operations (release
    create/upload). In GHA, GITHUB_TOKEN is auto-set. Locally, falls back
    to the `gh` CLI's stored token via `gh auth token`.
    """
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok
    try:
        r = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    raise RuntimeError(
        "no GitHub token found; set GITHUB_TOKEN env or run `gh auth login`"
    )


def _get_github_repo() -> str:
    """Return 'owner/repo'. In GHA this is GITHUB_REPOSITORY. Locally we
    parse `git remote get-url origin`."""
    env_repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if env_repo:
        return env_repo
    try:
        r = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5,
        )
        url = r.stdout.strip()
        # Match both git@github.com:owner/repo.git and https URLs
        m = re.search(r"github\.com[:/]([^/]+/[^/.]+)(?:\.git)?$", url)
        if m:
            return m.group(1)
    except Exception:
        pass
    raise RuntimeError("could not determine GitHub repo (set GITHUB_REPOSITORY)")


def _ensure_release(token: str, repo: str) -> dict:
    """Get-or-create the instagram-staging release. Returns the release
    JSON object (with id, upload_url, etc.)."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    # Try to fetch existing release by tag
    r = requests.get(
        f"https://api.github.com/repos/{repo}/releases/tags/{_RELEASE_TAG}",
        headers=headers, timeout=15,
    )
    if r.status_code == 200:
        return r.json()
    # Create new release
    body = {
        "tag_name": _RELEASE_TAG,
        "name": _RELEASE_TITLE,
        "body": _RELEASE_BODY,
        "draft": False,
        "prerelease": False,
    }
    r = requests.post(
        f"https://api.github.com/repos/{repo}/releases",
        headers=headers, json=body, timeout=15,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"failed to create release: {r.status_code} {r.text[:200]}")
    return r.json()


def _delete_existing_asset(token: str, repo: str, release_id: int, asset_name: str) -> None:
    """If an asset with this name exists on the release, delete it (so
    we can re-upload with the same name = clobber semantics)."""
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
    }
    r = requests.get(
        f"https://api.github.com/repos/{repo}/releases/{release_id}/assets",
        headers=headers, timeout=15,
    )
    if r.status_code != 200:
        return
    for asset in r.json():
        if asset.get("name") == asset_name:
            requests.delete(
                f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}",
                headers=headers, timeout=15,
            )


def _upload_asset_to_github_release(asset_path: str, content_type: str) -> str:
    """Upload an asset (mp4 or jpg) to the instagram-staging release with
    clobber semantics. Returns the public browser_download_url."""
    token = _get_github_token()
    repo = _get_github_repo()
    release = _ensure_release(token, repo)
    release_id = release["id"]
    # Strip the upload_url's template params: ".../assets{?name,label}" → ".../assets"
    upload_url = release["upload_url"].split("{")[0]
    asset_name = os.path.basename(asset_path)

    # Clobber any previous asset with the same name
    _delete_existing_asset(token, repo, release_id, asset_name)

    kb = os.path.getsize(asset_path) // 1024
    print(f"    [ig] uploading {asset_name} ({kb} KB) to GH release '{_RELEASE_TAG}'...")
    with open(asset_path, "rb") as f:
        data = f.read()
    headers = {
        "Authorization": f"token {token}",
        "Content-Type": content_type,
    }
    r = requests.post(
        f"{upload_url}?name={asset_name}",
        headers=headers, data=data, timeout=600,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"release asset upload failed: {r.status_code} {r.text[:200]}")
    public_url = r.json().get("browser_download_url", "")
    if not public_url:
        raise RuntimeError(f"upload OK but no browser_download_url: {r.text[:200]}")
    return public_url


def _upload_to_github_release(video_path: str) -> str:
    """Upload mp4 to the instagram-staging release. Backwards-compat
    wrapper around _upload_asset_to_github_release."""
    url = _upload_asset_to_github_release(video_path, "video/mp4")
    print(f"    [ig] mp4 public URL: {url}")
    return url


# ───────────────────────────────────────────────────────────────────────
# Cover frame extraction (avoids black-thumbnail problem on Reels)
# ───────────────────────────────────────────────────────────────────────

# Seek to this timestamp (seconds) for the cover. 3.5s lands past the
# 0.3s opening fade-in + ~2-3s into scene 1's narration = a frame with
# the lead character clearly visible. Tunable per video via env var.
_DEFAULT_COVER_OFFSET_S = 3.5


def _extract_cover_frame(video_path: str, timestamp_s: float = None) -> str:
    """Extract a single jpg frame from the video at `timestamp_s` (default
    3.5s — past the fade-in into scene 1's body). Returns the path to the
    saved jpg, written next to the source mp4 with suffix `_cover.jpg`.

    Without this, IG defaults to the very first frame which is in the
    middle of our 0.3s fade-in animation — black cover in the Reels feed.
    """
    if timestamp_s is None:
        try:
            timestamp_s = float(os.environ.get("IG_COVER_OFFSET_S", _DEFAULT_COVER_OFFSET_S))
        except ValueError:
            timestamp_s = _DEFAULT_COVER_OFFSET_S

    output_jpg = video_path.replace(".mp4", "_cover.jpg")
    # -ss BEFORE -i = fast seek (decoder skips ahead at keyframes)
    # -frames:v 1 = extract exactly 1 frame
    # -q:v 2 = high-quality jpg (2 = best, 31 = worst on the jpg scale)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp_s),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_jpg,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(output_jpg):
        err = r.stderr.decode("utf-8", errors="replace")[-300:] if r.stderr else ""
        raise RuntimeError(f"cover-frame extraction failed: {err}")
    return output_jpg


def _upload_cover_to_github_release(jpg_path: str) -> str:
    """Upload cover jpg to the instagram-staging release."""
    url = _upload_asset_to_github_release(jpg_path, "image/jpeg")
    print(f"    [ig] cover public URL: {url}")
    return url


# ───────────────────────────────────────────────────────────────────────
# Caption builder
# ───────────────────────────────────────────────────────────────────────

def _cap_hashtags(text: str, limit: int = _IG_HASHTAG_LIMIT) -> str:
    """
    Trim trailing hashtags so the total count is ≤ `limit`.

    IG enforces a 30-hashtag-per-caption limit silently — over the limit,
    the WHOLE caption gets rejected to empty (no error returned). Cut at
    the position where the (limit+1)-th hashtag starts so the body stays
    intact and we only drop excess tags from the trailing block.
    """
    hashtags = list(re.finditer(r"#\S+", text))
    if len(hashtags) <= limit:
        return text
    cutoff = hashtags[limit].start()
    return text[:cutoff].rstrip()


def _build_instagram_caption(script_data: dict, youtube_url: str) -> str:
    """
    Build the IG caption from the script's existing description field.
    Mirrors the user's "no platform fragmentation" rule — same content,
    only mechanical transforms:

      1. Prepend the episode title line ("महाभारत #N: …") so the series
         number is visible at the top of the IG caption (users scrolling
         Reels don't see the post title separately the way YT viewers do).
      2. Strip the leading "▶️ अगला भाग: <next title>" line (YouTube-
         specific Tier 2 cliffhanger header — IG users can't easily click
         description links to jump to the next episode).
      3. Cap hashtags to 28 (IG silently rejects >30-hashtag captions).
      4. Append "📺 Full video: {youtube_url}" cross-promo at the bottom.
      5. Cap to 2200 chars (IG limit), trimming from the trailing
         hashtag block rather than the body if needed.
    """
    desc = (script_data.get("description") or "").strip()
    title = (script_data.get("title") or "").strip()

    # Strip the "▶️ अगला भाग:" prefix line (Fix 2.0 YT-specific header)
    lines = desc.split("\n")
    if lines and lines[0].lstrip().startswith("▶️ अगला भाग"):
        # Drop the line and any immediately following blank line(s)
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    desc = "\n".join(lines).strip()

    # Prepend the title (e.g. "महाभारत #2: भीष्म दादा की दुविधा | Bhishma…")
    # so IG viewers see which episode they're watching. Only prepend if
    # the title isn't already echoed by the first line of the description.
    if title and not desc.lower().startswith(title.lower()[:20]):
        desc = f"{title}\n\n{desc}"

    # Trim excess hashtags BEFORE adding backlink, so we don't accidentally
    # trim the YouTube URL.
    desc = _cap_hashtags(desc)

    # Build the backlink suffix
    backlink = ""
    if youtube_url:
        backlink = f"\n\n📺 Full video: {youtube_url}"

    full = desc + backlink

    # Fit to 2200 chars. If too long, trim the trailing hashtag block.
    if len(full) <= _IG_CAPTION_MAX:
        return full

    # Over limit. Strategy: find the LAST "#" before the trim point and
    # trim cleanly at a hashtag boundary so we don't leave a broken
    # half-hashtag like "#भार".
    overflow = len(full) - _IG_CAPTION_MAX
    # Trim from the body BEFORE the backlink (preserve the backlink).
    body, sep, link = (
        (desc, "\n\n📺 Full video: ", youtube_url) if backlink else (full, "", "")
    )
    trimmed = body[: len(body) - overflow - 20]   # 20-char safety margin
    # Find the last whitespace boundary to avoid mid-word cut
    last_space = trimmed.rfind("\n")
    if last_space > 100:
        trimmed = trimmed[:last_space]
    trimmed = trimmed.rstrip() + "\n…" + (sep + link if link else "")
    return trimmed[:_IG_CAPTION_MAX]


# ───────────────────────────────────────────────────────────────────────
# Instagram Graph API calls
# ───────────────────────────────────────────────────────────────────────

def _create_reels_container(
    ig_user_id: str, access_token: str, video_url: str, caption: str,
    cover_url: str = "",
) -> str:
    """POST /media → returns container ID.

    Sends content params (caption, video_url, cover_url, etc.) as
    **multipart/form-data** instead of URL-encoded form body — both URL
    params AND application/x-www-form-urlencoded silently dropped the
    caption on Devanagari + emoji + 1000+ char strings during the
    2026-05-17 backfill (3 Reels landed with empty captions). Multipart
    is the only encoding that reliably preserves long Unicode through
    Meta's Graph API for IG Reels.

    `cover_url` is optional; when provided, IG uses that jpg as the Reel
    cover instead of grabbing the first frame (which is mid-fade-in =
    black on our videos). Pass an empty string to fall back to first-frame.
    """
    # `files=` triggers multipart/form-data encoding in `requests`. Each
    # tuple is (filename, value, content_type). filename=None for non-file
    # fields, content-type explicit text/plain; charset=utf-8 to force
    # UTF-8 encoding on Devanagari/emoji caption bytes.
    files = {
        "media_type":    (None, "REELS"),
        "video_url":     (None, video_url),
        "caption":       (None, caption.encode("utf-8"), "text/plain; charset=utf-8"),
        "share_to_feed": (None, "true"),
    }
    if cover_url:
        files["cover_url"] = (None, cover_url)
    r = requests.post(
        f"{_GRAPH_API}/{ig_user_id}/media",
        params={"access_token": access_token},
        files=files,
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"container create failed: {r.status_code} {r.text[:300]}")
    container_id = r.json().get("id")
    if not container_id:
        raise RuntimeError(f"container create returned no id: {r.text[:200]}")
    return container_id


def _poll_container_until_ready(container_id: str, access_token: str) -> None:
    """Poll status_code until FINISHED or timeout. Raises if status_code
    is ERROR/EXPIRED. Returns when ready (no value)."""
    deadline = time.time() + _CONTAINER_POLL_TIMEOUT_S
    while time.time() < deadline:
        r = requests.get(
            f"{_GRAPH_API}/{container_id}",
            params={
                "access_token": access_token,
                "fields": "status_code,status",
            },
            timeout=15,
        )
        if r.status_code != 200:
            raise RuntimeError(f"poll failed: {r.status_code} {r.text[:200]}")
        body = r.json()
        status_code = body.get("status_code", "")
        if status_code == "FINISHED":
            return
        if status_code in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"container failed: {body}")
        # Still IN_PROGRESS or PUBLISHED — wait and retry
        time.sleep(_CONTAINER_POLL_INTERVAL_S)
    raise TimeoutError(f"container not ready within {_CONTAINER_POLL_TIMEOUT_S}s")


def _publish_container(
    ig_user_id: str, access_token: str, container_id: str
) -> str:
    """POST /media_publish → returns final IG media ID."""
    r = requests.post(
        f"{_GRAPH_API}/{ig_user_id}/media_publish",
        params={
            "access_token": access_token,
            "creation_id": container_id,
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"publish failed: {r.status_code} {r.text[:300]}")
    media_id = r.json().get("id")
    if not media_id:
        raise RuntimeError(f"publish returned no id: {r.text[:200]}")
    return media_id


# ───────────────────────────────────────────────────────────────────────
# Public entrypoint
# ───────────────────────────────────────────────────────────────────────

def upload_to_instagram(
    video_path: str,
    script_data: dict,
    youtube_url: str = "",
) -> Optional[str]:
    """
    Upload a vertical mp4 to Instagram Reels via Graph API.

    Returns the IG media ID on success, or None on any failure. Callers
    should treat IG failures as non-fatal — never let an IG hiccup break
    the YouTube path or the Fix 2.8 state-commit guard.

    Args:
      video_path:   Path to the local 9:16 mp4 file.
      script_data:  The same dict passed to upload_to_youtube — uses
                    description for caption derivation.
      youtube_url:  Optional. The just-uploaded YT URL for the cross-
                    promo backlink in the IG caption.
    """
    if not os.path.exists(video_path):
        print(f"    [ig] FAIL: video missing: {video_path}")
        return None

    access_token = os.environ.get("IG_ACCESS_TOKEN", "").strip()
    ig_user_id = os.environ.get("IG_BUSINESS_ACCOUNT_ID", "").strip()
    if not access_token or not ig_user_id:
        print("    [ig] SKIP: IG_ACCESS_TOKEN or IG_BUSINESS_ACCOUNT_ID not set")
        return None

    try:
        # Step 1: host the mp4 publicly on GH release
        public_url = _upload_to_github_release(video_path)

        # Step 1.5: extract + upload a cover frame (avoids fade-in black thumb)
        cover_url = ""
        try:
            cover_jpg = _extract_cover_frame(video_path)
            cover_url = _upload_cover_to_github_release(cover_jpg)
        except Exception as cover_err:
            print(f"    [ig] WARN: cover extraction failed (non-fatal): {cover_err}")
            # Fall through — IG will use first-frame fallback (likely black)

        # Step 2: build the caption (same description, mechanical transform)
        caption = _build_instagram_caption(script_data, youtube_url)
        print(f"    [ig] caption: {len(caption)} chars (limit 2200)")

        # Step 3: create the Reels container
        print(f"    [ig] creating Reels container...")
        container_id = _create_reels_container(
            ig_user_id, access_token, public_url, caption, cover_url=cover_url,
        )
        print(f"    [ig] container_id={container_id}")

        # Step 4: poll until container is FINISHED
        print(f"    [ig] polling container status (timeout {_CONTAINER_POLL_TIMEOUT_S}s)...")
        _poll_container_until_ready(container_id, access_token)
        print(f"    [ig] container ready")

        # Step 5: publish
        media_id = _publish_container(ig_user_id, access_token, container_id)
        print(f"    [ig] [OK] published → media_id={media_id}")
        return media_id

    except Exception as e:
        print(f"    [ig] FAIL: {type(e).__name__}: {str(e)[:300]}")
        return None
