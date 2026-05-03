"""
Topic Manager — scheduled topic queue + video activity log.

scheduled_topics.txt
    One topic per line. Lines starting with # are comments.
    The pipeline picks the first non-comment line, uses it, then removes it.
    If the file is empty the pipeline falls back to a random built-in topic.

video_log_001.txt / video_log_002.txt …
    Every completed video is appended here with timestamp and metadata.
    A new file is created automatically when the current one exceeds 5 MB.
"""

import os
from datetime import datetime

SCHEDULED_FILE = "scheduled_topics.txt"
LOG_PREFIX     = "video_log_"
LOG_MAX_BYTES  = 5 * 1024 * 1024   # 5 MB per log file


# ── Topic queue ───────────────────────────────────────────────────────────────

def get_next_topic() -> str | None:
    """
    Returns the first queued topic from scheduled_topics.txt and removes it.
    Returns None if the file is empty or missing — pipeline uses random topic.
    """
    if not os.path.exists(SCHEDULED_FILE):
        return None

    with open(SCHEDULED_FILE, "r", encoding="utf-8") as f:
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
        with open(SCHEDULED_FILE, "w", encoding="utf-8") as f:
            f.writelines(remaining)
        print(f"    Using scheduled topic: {topic}")

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

    entry = (
        f"[{timestamp}]\n"
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
