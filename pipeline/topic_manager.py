"""
Topic Manager — scheduled topic queue + video activity log.

scheduled_topics_<series>.txt (per-series queue files)
    One topic per line. Lines starting with # are comments.
    The pipeline picks the first non-comment line, uses it, then removes it.
    If the file is empty the pipeline falls back to a random built-in topic.

    Series-aware files:
      - scheduled_topics_mahabharata.txt  (Mahabharata stories)
      - scheduled_topics_whatif.txt        (What If thought experiments)

    Backwards compat: if scheduled_topics_mahabharata.txt is missing but the
    older scheduled_topics.txt exists, the Mahabharata series falls back to it.

video_log_001.txt / video_log_002.txt …
    Every completed video is appended here with timestamp, language, series,
    and metadata. A new file is created automatically when the current one
    exceeds 5 MB.
"""

import os
from datetime import datetime

LEGACY_SCHEDULED_FILE = "scheduled_topics.txt"   # legacy, Mahabharata-only fallback
LOG_PREFIX     = "video_log_"
LOG_MAX_BYTES  = 5 * 1024 * 1024   # 5 MB per log file


def _scheduled_file_for(series: str) -> str:
    return f"scheduled_topics_{series}.txt"


# ── Topic queue ───────────────────────────────────────────────────────────────

def get_next_topic(series: str = "mahabharata") -> str | None:
    """
    Returns the first queued topic for the given series and removes it.
    Returns None if the file is empty or missing — caller falls back to a
    random built-in topic.
    """
    primary = _scheduled_file_for(series)
    if os.path.exists(primary):
        path = primary
    elif series == "mahabharata" and os.path.exists(LEGACY_SCHEDULED_FILE):
        # Backwards compat: pre-WhatIf installs use scheduled_topics.txt
        path = LEGACY_SCHEDULED_FILE
    else:
        return None

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    topic     = None
    remaining = []
    found     = False

    for line in lines:
        stripped = line.strip()
        if not found and stripped and not stripped.startswith("#"):
            topic = stripped
            found = True
        else:
            remaining.append(line)

    if topic:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(remaining)
        print(f"    Using scheduled {series} topic: {topic}")

    return topic


# ── Video log ─────────────────────────────────────────────────────────────────

def _current_log_path() -> str:
    """Returns the active log file path, rolling over when it exceeds 5 MB."""
    i = 1
    while True:
        path = f"{LOG_PREFIX}{i:03d}.txt"
        if not os.path.exists(path):
            return path
        if os.path.getsize(path) < LOG_MAX_BYTES:
            return path
        i += 1


def log_video(video_path: str, script_data: dict, language: str) -> None:
    """Appends a completed-video entry to the rolling log file."""
    log_path  = _current_log_path()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scenes    = script_data.get("scenes", [])
    series    = script_data.get("series", "mahabharata")

    entry = (
        f"[{timestamp}]\n"
        f"  Series   : {series}\n"
        f"  Language : {language}\n"
        f"  Title    : {script_data.get('title', 'N/A')}\n"
        f"  Topic    : {script_data.get('topic', 'N/A')}\n"
        f"  Type     : {script_data.get('content_type', 'N/A')}\n"
        f"  Scenes   : {len(scenes)}\n"
        f"  File     : {video_path}\n"
        f"{'-' * 60}\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"    Logged -> {log_path}")
