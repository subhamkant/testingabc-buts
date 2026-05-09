"""
Topic Manager — scheduled topic queue + video activity log + LLM auto-replenish.

scheduled_topics_<series>.txt (per-series queue files)
    One topic per line. Lines starting with # are comments.
    The pipeline picks the first non-comment line, uses it, then removes it.
    If the file is empty the pipeline falls back to a random built-in topic.

    Series-aware files:
      - scheduled_topics_mahabharata.txt  (Mahabharata stories)
      - scheduled_topics_whatif.txt        (What If thought experiments)

    Backwards compat: if scheduled_topics_mahabharata.txt is missing but the
    older scheduled_topics.txt exists, the Mahabharata series falls back to it.

    Auto-replenish (WhatIf only): when the queue drops below
    _WHATIF_REPLENISH_THRESHOLD, an LLM call generates fresh topics and
    appends them. Recently-uploaded topics (from video_log_*.txt) are passed
    in as an avoidance list so the model doesn't re-suggest stale ideas.
    The workflow's "Commit updated state" step persists the modified queue
    back to git after each run.

video_log_001.txt / video_log_002.txt …
    Every completed video is appended here with timestamp, language, series,
    and metadata. A new file is created automatically when the current one
    exceeds 5 MB.
"""

import glob
import json
import os
import random
import re
import time
from datetime import datetime

import requests

LEGACY_SCHEDULED_FILE = "scheduled_topics.txt"   # legacy, Mahabharata-only fallback
LOG_PREFIX     = "video_log_"
LOG_MAX_BYTES  = 5 * 1024 * 1024   # 5 MB per log file

# Auto-replenish config — when scheduled_topics_whatif.txt has fewer than
# this many remaining lines (excluding comments), generate this many fresh
# topics in one LLM call. One call -> ~10 days of new content; cheap.
_WHATIF_REPLENISH_THRESHOLD = 3
_WHATIF_REPLENISH_COUNT     = 10
# Avoidance window — ~2 months of WhatIf uploads (1/day). The LLM is told
# never to reproduce anything in this window, and the static-topic fallback
# is filtered the same way. Same topic only re-surfaces after this many
# uploads have rolled past.
_WHATIF_RECENT_HISTORY      = 60


def _scheduled_file_for(series: str) -> str:
    return f"scheduled_topics_{series}.txt"


# ── Recent-topic history (for LLM avoidance) ─────────────────────────────────

# Compact tracked file recording recently-used topics per series. Designed to
# be small (~few KB), committed back by the workflow's state-persistence step
# so avoidance survives across ephemeral GHA runners. video_log_*.txt files
# are gitignored and thus useless for cross-run avoidance — this file is the
# durable source of truth.
_RECENT_TOPICS_PATH = "recent_topics.json"
_RECENT_TOPICS_KEEP = 120   # cap entries per series — keeps file small


def _load_recent_topics_db() -> dict:
    if not os.path.exists(_RECENT_TOPICS_PATH):
        return {}
    try:
        with open(_RECENT_TOPICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_recent_topics_db(db: dict) -> None:
    try:
        with open(_RECENT_TOPICS_PATH, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"    [recent-topics] save failed: {e}")


def record_used_topic(series: str, topic: str) -> None:
    """
    Append `topic` to the recent-topics database for `series`. Called from
    log_video on every successful upload so the avoidance list grows.
    Capped at _RECENT_TOPICS_KEEP entries per series (oldest dropped).
    """
    if not topic or not series:
        return
    db = _load_recent_topics_db()
    bucket = db.setdefault(series, [])
    bucket.append({
        "topic": topic,
        "ts":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    # Trim — keep newest _RECENT_TOPICS_KEEP entries
    if len(bucket) > _RECENT_TOPICS_KEEP:
        db[series] = bucket[-_RECENT_TOPICS_KEEP:]
    _save_recent_topics_db(db)


def _read_recent_topics(series: str, limit: int = 60) -> list:
    """
    Return the most recent `limit` topic strings used for `series` from the
    tracked recent_topics.json. Newest first. Backed by the durable JSON
    file (committed across runs) — NOT the gitignored video_log_*.txt.
    """
    db = _load_recent_topics_db()
    bucket = db.get(series, [])
    # Newest first
    topics = [entry["topic"] for entry in reversed(bucket) if entry.get("topic")]
    return topics[:limit]


# ── Trending headlines (for trend-aware WhatIf topic generation) ────────────

# Cache trending headlines for ~6h so we don't hammer Reddit on every run.
# This file is gitignored — purely runtime cache, no point committing.
_TRENDING_CACHE_PATH = "temp/trending_cache.json"
_TRENDING_CACHE_TTL_S = 6 * 3600

# Subreddits chosen for: global signal, science alignment, future-speculation
# fit. r/worldnews + r/science are the news anchors; r/Futurology and r/space
# bias toward speculative/scientific framings; r/todayilearned surfaces
# evergreen factoids that often spark good "what if" reframes.
_TRENDING_SUBREDDITS = (
    "worldnews",
    "science",
    "Futurology",
    "space",
    "todayilearned",
)


def _load_trending_cache() -> list:
    if not os.path.exists(_TRENDING_CACHE_PATH):
        return []
    try:
        with open(_TRENDING_CACHE_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        if time.time() - payload.get("ts", 0) > _TRENDING_CACHE_TTL_S:
            return []
        return payload.get("headlines", [])
    except Exception:
        return []


def _save_trending_cache(headlines: list) -> None:
    os.makedirs(os.path.dirname(_TRENDING_CACHE_PATH), exist_ok=True)
    try:
        with open(_TRENDING_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "headlines": headlines}, f, ensure_ascii=False)
    except Exception:
        pass


def _fetch_trending_headlines(limit: int = 15) -> list:
    """
    Pull top-of-the-day post titles from a curated set of public Reddit JSON
    endpoints. No auth required (User-Agent header is enough). Cached for ~6h.

    Returns list of {"title": str, "source": str} dicts. Soft-fails to empty
    list on any network/parse error — trending context is a *bonus* signal,
    NEVER a blocker for topic generation.
    """
    cached = _load_trending_cache()
    if cached:
        return cached[:limit]

    headlines = []
    seen_titles = set()
    for sub in _TRENDING_SUBREDDITS:
        try:
            url = f"https://www.reddit.com/r/{sub}/top.json?limit=8&t=day"
            resp = requests.get(
                url,
                headers={"User-Agent": "vyasa-ai-trending-fetch/1.0"},
                timeout=10,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for child in data.get("data", {}).get("children", []):
                title = (child.get("data") or {}).get("title", "").strip()
                if not title or len(title) < 20 or len(title) > 220:
                    continue
                key = title.lower()[:80]
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                headlines.append({"title": title, "source": f"r/{sub}"})
        except Exception as e:
            print(f"    [trending] r/{sub} fetch failed (non-fatal): {str(e)[:80]}")
            continue

    headlines = headlines[:limit]
    if headlines:
        _save_trending_cache(headlines)
        print(f"    [trending] fetched {len(headlines)} headlines from {len(_TRENDING_SUBREDDITS)} subs")
    return headlines


# ── LLM topic generator ──────────────────────────────────────────────────────

# Diverse seed categories the LLM samples from when generating WhatIf topics.
# Picking a few each call prevents one-domain spam (e.g. all "what if planets
# moved" topics) and forces variety across runs.
_WHATIF_SEED_CATEGORIES = [
    "astronomy / cosmology",
    "Earth geology / tectonics",
    "global climate / weather",
    "biology / evolution / extinction",
    "physics / fundamental forces",
    "human consciousness / cognition",
    "civilization / human history",
    "future technology / AI",
    "ocean / hydrosphere",
    "atmosphere / breathable air",
    "geography / continents",
    "speed of time / relativity",
    "human anatomy / biology limits",
    "agriculture / food chain",
]


def _generate_whatif_topics_via_llm(count: int, avoid: list) -> list:
    """
    Use the existing Groq+Gemini LLM cascade (lazy-imported from
    script_generator to avoid import cycles) to generate `count` fresh
    WhatIf topics, avoiding any string-similarity duplicates of `avoid`.

    Pulls live trending headlines and asks the LLM to reframe ~30% of
    its output as trend-piggyback WhatIfs (rides search waves the
    audience is already on) while keeping ~70% as evergreen science
    speculation. Returns a list of topic strings, possibly empty if
    the LLM fails.
    """
    try:
        from pipeline.script_generator import _call_llm
    except Exception as e:
        print(f"    [topic-gen] _call_llm unavailable: {e}")
        return []

    seeds = random.sample(
        _WHATIF_SEED_CATEGORIES,
        min(5, len(_WHATIF_SEED_CATEGORIES)),
    )
    avoid_block = "\n".join(f"- {t}" for t in avoid[:_WHATIF_RECENT_HISTORY]) or "(none)"

    # Trending context is a bonus signal — fully optional.
    trending = _fetch_trending_headlines(limit=15)
    if trending:
        trending_block = (
            "TRENDING NOW (top-of-day titles from global news + science + "
            "futurology subreddits — viewers are actively searching these):\n"
            + "\n".join(f"- {h['title']}  ({h['source']})" for h in trending)
            + "\n\nTREND-PIGGYBACK INSTRUCTION:\n"
            "Aim for roughly 30% of your topics (≈ "
            f"{max(1, count // 3)} of {count}) to be inspired by these "
            "trends — reframe a trending event as a SCIENCE-grounded WhatIf "
            "speculation. The remaining ~70% should be evergreen science "
            "topics not tied to news.\n\n"
            "TREND→WHATIF transformation examples:\n"
            "- Trending: 'Iran-Israel tensions escalate'\n"
            "  WhatIf: 'What if a country the size of Iran suddenly ceased "
            "to exist — what would happen to global oil prices, Middle East "
            "geography, and the 80M people who'd need somewhere to go?'\n"
            "- Trending: 'Asteroid 2024-XX flies past Earth at 1.2 lunar distances'\n"
            "  WhatIf: 'What if asteroid 2024-XX had hit the Pacific Ocean "
            "instead — how big would the tsunami be, and which coastlines vanish?'\n"
            "- Trending: 'New Marvel/Christopher Nolan film announced'\n"
            "  WhatIf: 'What if real-world physics matched the universe of "
            "<film> — how long could a human survive in that environment?'\n"
            "- Trending: 'AI passes new benchmark / new chip released'\n"
            "  WhatIf: 'What if AI got 1000× faster overnight — which jobs "
            "vanish in the first 24 hours, and what new ones appear?'\n\n"
            "SAFETY: SKIP any trending headline that involves:\n"
            "- Active war casualties / atrocities (the geographic counterfactual "
            "  is OK — see Iran example above; specific living-people violence "
            "  is not)\n"
            "- Direct named-political-figure attacks\n"
            "- Medical advice claims\n"
            "- Religious / ethnic flashpoints\n"
            "If a trend is risky, just skip it and use other trends or fully "
            "evergreen topics. Do not flag the skip — silently move on.\n"
        )
    else:
        trending_block = ""

    prompt = f"""
You are generating WhatIf topics for a YouTube Shorts channel. Each topic
becomes a 60-second curiosity-driven thought experiment grounded in real
science (NOT fantasy / mythology / magic).

Generate EXACTLY {count} topics that are:
- Genuinely thought-provoking — the viewer should want to know the answer
- Scientifically plausible — speculation grounded in physics, biology,
  astronomy, geology, climate, anatomy, or human civilization. NOT
  supernatural, not "what if magic existed".
- Specific and visualizable — not abstract philosophy
- Distinct from each other (no two near-duplicates in this batch)
- Distinct from the AVOID LIST below — that list is roughly the last
  2 months of uploads. Do not regenerate ANY topic in the same theme as
  anything in that list, even with different phrasing. If the avoid list
  has "What if Earth's gravity halved", do not produce "What if gravity
  weakened on Earth" — pick a totally different domain.

CATEGORIES TO DRAW FROM (sample broadly across these — don't all come
from the same one):
{chr(10).join(f"- {c}" for c in seeds)}

EXAMPLE GOOD TOPICS (use this style — concrete, single-clause, ends with "?"):
- What if Earth had two moons of equal size?
- What if oxygen levels doubled overnight?
- What if humans only needed 2 hours of sleep?
- What if the Pacific Ocean evaporated in one day?
- What if every plant produced light at night?

EXAMPLE BAD TOPICS (avoid these patterns):
- "What if magic was real?"  (not science)
- "What if you could fly?"   (too generic, not specific)
- "What if life had meaning?" (abstract philosophy, no visual)

AVOID LIST — do NOT generate near-duplicates of any of these:
{avoid_block}

{trending_block}
OUTPUT FORMAT — return ONLY a JSON array of strings, no markdown fences,
no preamble, no commentary:
[
  "What if ...?",
  "What if ...?",
  ...
]
"""

    try:
        raw = _call_llm(prompt)
    except Exception as e:
        print(f"    [topic-gen] LLM call failed: {e}")
        return []

    # Extract JSON array from response (LLM may wrap in markdown fences)
    start = raw.find("[")
    end   = raw.rfind("]")
    if start == -1 or end == -1:
        print("    [topic-gen] no JSON array in response")
        return []
    try:
        topics = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as e:
        print(f"    [topic-gen] JSON parse failed: {e}")
        return []

    # Sanity: each must be a non-empty string starting with What if/What If
    cleaned = []
    avoid_lc = {a.lower().strip() for a in avoid}
    for t in topics:
        if not isinstance(t, str):
            continue
        t = t.strip().rstrip(",")
        if not t or len(t) > 200:
            continue
        if not t.lower().startswith("what if"):
            continue
        if t.lower() in avoid_lc:
            continue
        cleaned.append(t)

    print(f"    [topic-gen] LLM produced {len(cleaned)}/{count} usable topics")
    return cleaned[:count]


def _maybe_replenish_whatif_queue(queue_path: str, remaining_count: int) -> None:
    """
    If the WhatIf queue drops below _WHATIF_REPLENISH_THRESHOLD, generate a
    fresh batch via LLM and append to the queue file. Failure is non-fatal —
    the caller falls back to STORY_TOPICS_WHATIF random pick.
    """
    if remaining_count >= _WHATIF_REPLENISH_THRESHOLD:
        return

    print(f"    [topic-gen] WhatIf queue low ({remaining_count} remaining) — auto-replenishing...")
    avoid = _read_recent_topics("whatif", limit=_WHATIF_RECENT_HISTORY)
    new_topics = _generate_whatif_topics_via_llm(_WHATIF_REPLENISH_COUNT, avoid)
    if not new_topics:
        print("    [topic-gen] no new topics produced — queue not replenished")
        return

    # Append to queue file (creates if missing)
    header_needed = not os.path.exists(queue_path) or os.path.getsize(queue_path) == 0
    with open(queue_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write(
                "# What If — scheduled topic queue (auto-generated entries below)\n"
                "# One topic per line. Pipeline pops first non-comment line per run.\n\n"
            )
        # Mark when this batch was generated for audit trail
        f.write(f"\n# auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC\n")
        for t in new_topics:
            f.write(f"{t}\n")

    print(f"    [topic-gen] Appended {len(new_topics)} fresh topics to {queue_path}")


# ── Topic queue ───────────────────────────────────────────────────────────────

def get_next_topic(series: str = "mahabharata") -> str | None:
    """
    Returns the first queued topic for the given series and removes it.
    Returns None if the file is empty or missing — caller falls back to a
    random built-in topic.

    For series=="whatif", the queue is auto-replenished via an LLM call when
    fewer than _WHATIF_REPLENISH_THRESHOLD entries remain. This gives the
    channel an effectively unlimited supply of fresh thought experiments
    while still respecting any manually-queued topics that take priority.
    """
    primary = _scheduled_file_for(series)
    if os.path.exists(primary):
        path = primary
    elif series == "mahabharata" and os.path.exists(LEGACY_SCHEDULED_FILE):
        # Backwards compat: pre-WhatIf installs use scheduled_topics.txt
        path = LEGACY_SCHEDULED_FILE
    else:
        # No queue file exists. WhatIf can synthesize one from scratch.
        if series == "whatif":
            path = primary  # _maybe_replenish_whatif_queue creates it
            _maybe_replenish_whatif_queue(path, remaining_count=0)
            if not os.path.exists(path):
                return None
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

    # After consuming a topic, see if the WhatIf queue needs auto-replenish
    # for future runs. Doing it AFTER the pop means today's run already has
    # its topic; the appended batch is for upcoming runs.
    if series == "whatif":
        non_comment_remaining = sum(
            1 for ln in remaining
            if ln.strip() and not ln.strip().startswith("#")
        )
        _maybe_replenish_whatif_queue(path, non_comment_remaining)

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
    """Appends a completed-video entry to the rolling log file AND records
    the topic into the tracked recent_topics.json so cross-run avoidance
    works (rolling log is gitignored and thus discarded between runs)."""
    log_path  = _current_log_path()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scenes    = script_data.get("scenes", [])
    series    = script_data.get("series", "mahabharata")
    topic     = script_data.get("topic", "")

    entry = (
        f"[{timestamp}]\n"
        f"  Series   : {series}\n"
        f"  Language : {language}\n"
        f"  Title    : {script_data.get('title', 'N/A')}\n"
        f"  Topic    : {topic or 'N/A'}\n"
        f"  Type     : {script_data.get('content_type', 'N/A')}\n"
        f"  Scenes   : {len(scenes)}\n"
        f"  File     : {video_path}\n"
        f"{'-' * 60}\n"
    )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(entry)

    # Persist topic into the cross-run avoidance database (small JSON,
    # tracked in git, committed back by the workflow). For dual-language
    # WhatIf runs this is called twice — once per language — but we only
    # want one entry per generation. log_video is called once per video
    # upload though, so for dual-language we'd get 2 entries with the same
    # topic. That's harmless (the avoidance set dedups via lowercase).
    record_used_topic(series, topic)

    print(f"    Logged -> {log_path}")
