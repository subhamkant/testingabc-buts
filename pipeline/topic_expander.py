"""
Phase 21 (2026-06-25) — Mahabharata arc auto-expansion.

Why this exists
===============
The Phoenix audit (2026-06-17) + post-Phase-20 forensic (2026-06-25) confirmed
that `assets/character_arcs.json` has only 49 hand-curated topics (7 chars × 7
topics each) while the channel has 59 published videos. The pipeline has been
running on `STORY_TOPICS` legacy fallback for several uploads. Without auto-
expansion, the daily cron hits a hard "no fresh story" wall once dedup catches
up.

The user's explicit constraint: "new story daily, don't repeat any story."
This module guarantees that constraint by automatically generating fresh
DNA-passing topics whenever an arc's eligible pool drops below threshold.

Design contract
===============
1. **Sidecar provenance (plan-review decision 2026-06-25):** the existing
   `arc["topics"]` array is `list[str]` and is read by 3 consumers
   (script_generator, backfill, signature dedup) that iterate raw strings.
   Migrating to `list[dict]` would be a breaking schema change. Instead,
   auto-generated topics go into a NEW per-arc key `_auto_added` shaped as
   `list[dict]` with provenance metadata. The picker unions both pools.
   Existing consumers see no change.

2. **Trigger threshold:** expander fires when any character's eligible pool
   (unused AND dedup-passing) drops to ≤ 2 OR the global eligible pool drops
   below 10.

3. **Hard cap:** 30 topics per arc (counting `topics` + `_auto_added`).
   Prevents runaway expansion. Once cap hit, expander is a no-op for that arc.

4. **LLM quality tier:** Gemini Flash (`quality="fast"`) — per memory
   `gemini_pro_free_tier_removed`, Pro is limit:0 on the channel's free tier.
   Flash is sufficient for topic strings (4-part DNA is structurally simple;
   not full narrative prose).

5. **Atomic write:** uses `tempfile.NamedTemporaryFile` + `os.replace` so a
   crash mid-write leaves the original `character_arcs.json` intact.

6. **Fail-safe:** any LLM error or validation failure → log warning, skip
   expansion this run. The picker still has whatever topics existed before.
   Renders are never blocked.

Public API
==========
- `expand_arcs_if_needed(arcs, used, published_signatures)` — called at the
   top of `_pick_next_arc_topic`. No-op if no arc is starving.
- `arc_topic_pool(arc)` — returns the unioned pool (topics + _auto_added) for
   a single arc. Used by callers that need to iterate the full eligible set.
"""

import json
import os
import re
import tempfile
from datetime import datetime, timezone


# ── Constants ────────────────────────────────────────────────────────────────

ARCS_PATH = os.path.join("assets", "character_arcs.json")

# Trigger thresholds
PER_ARC_STARVATION_THRESHOLD = 2     # if eligible <= 2 for a character, expand
GLOBAL_STARVATION_THRESHOLD = 10     # OR if total eligible < 10, expand
HARD_CAP_PER_ARC = 30                # never grow past 30 topics per arc
TOPICS_PER_EXPANSION = 5             # LLM emits 5 per call, validate, filter

_SOURCE_TAG = "auto_expansion_v21"


# ── Public helpers ───────────────────────────────────────────────────────────

def arc_topic_pool(arc: dict) -> list[str]:
    """Return the unioned topic pool for an arc:
      - human-curated strings from `arc["topics"]`
      - auto-expander entries from `arc["_auto_added"]` (their `.topic` field)

    The picker treats both as equivalent strings at draw time; provenance
    is preserved on disk only. Backwards-compatible — arcs without
    `_auto_added` (pre-Phase-21 state) return the same list as before.
    """
    pool = list(arc.get("topics", []))
    pool.extend(
        entry["topic"]
        for entry in arc.get("_auto_added", [])
        if isinstance(entry, dict) and "topic" in entry
    )
    return pool


def _count_eligible(arc: dict, used: set, published_signatures: list) -> int:
    """How many topics in this arc's pool are unused AND dedup-passing?"""
    # Lazy import to avoid circular dep
    from pipeline.topic_signatures import topic_overlaps_published

    pool = arc_topic_pool(arc)
    count = 0
    for topic in pool:
        if topic in used:
            continue
        if published_signatures and topic_overlaps_published(topic, published_signatures):
            continue
        count += 1
    return count


def _atomic_write_json(path: str, data: dict) -> None:
    """Write JSON atomically via tempfile + os.replace. Survives crash."""
    dir_ = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=dir_, suffix=".tmp",
        delete=False, newline="\n",
    ) as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tmp_path = tf.name
    os.replace(tmp_path, path)


# ── Expansion prompt template ────────────────────────────────────────────────

_EXPANSION_PROMPT_TEMPLATE = """\
You are generating new topics for a Mahabharata YouTube Shorts channel that \
ships 1 unique story per day with NO REPEATS. Generate exactly {n} fresh \
story topics for the {character_name} arc.

CHARACTER EMOTIONAL FINGERPRINT: {emotional_fingerprint}
(every topic's emotional center must land on this fingerprint, not a generic \
dramatic register).

EXISTING TOPICS for this character (do NOT duplicate themes or DNA):
{existing_topics_list}

ALREADY PUBLISHED on the channel (do NOT duplicate any of these incidents):
{published_titles_list}

EACH TOPIC MUST FOLLOW THE 4-PART DNA from Phase 15 doctrine:
1. Name-first hook (character name first or second word)
2. Subversion adjective: असली / बड़ी / छिपा / अनकहा / गुप्त / \
real / hidden / untold / secret / greatest / final / fatal
3. Charged moral noun: पाप / गलती / घमंड / सच / धोखा / बलिदान / \
sin / mistake / pride / truth / betrayal / sacrifice / curse / regret / \
vow / loyalty / jealousy / fear
4. Trait inversion (the character's celebrated virtue reframed as a flaw)

Each topic should be 80-150 chars, English, narrative-rich enough that the \
script generator can build a 90-100 word voiceover from it.

OUTPUT exactly {n} topics, one per line, no preamble, no numbering, no \
markdown fences, no commentary. Just the topic strings.\
"""


# ── Core expansion function ──────────────────────────────────────────────────

def _generate_topics_for_character(
    character_name: str,
    emotional_fingerprint: str,
    existing_topics: list[str],
    published_titles: list[str],
    n: int = TOPICS_PER_EXPANSION,
) -> list[str]:
    """Call Gemini Flash to generate N raw topic strings. Returns the parsed
    lines (un-validated). Caller is responsible for semantic-gate + dedup
    validation before appending."""
    # Lazy import to avoid circular dep with script_generator
    from pipeline.script_generator import _call_llm

    prompt = _EXPANSION_PROMPT_TEMPLATE.format(
        n=n,
        character_name=character_name,
        emotional_fingerprint=emotional_fingerprint or "(not specified)",
        existing_topics_list="\n".join(f"  - {t}" for t in existing_topics) or "  (none)",
        published_titles_list="\n".join(f"  - {t}" for t in published_titles[:30]) or "  (none)",
    )

    try:
        raw = _call_llm(prompt, quality="fast")
    except Exception as e:
        print(f"    [topic-expander] LLM call failed for {character_name}: {str(e)[:120]}")
        return []

    # Parse: strip markdown fences if any, split on newlines, filter empties
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:\w+)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    lines = [line.strip().lstrip("-*•0123456789. ").strip() for line in text.split("\n")]
    lines = [l for l in lines if l and len(l) >= 30]   # filter blank + too-short lines
    return lines[:n]


def _validate_new_topic(
    topic: str,
    existing_pool: list[str],
    published_signatures: list,
) -> tuple[bool, str]:
    """Validate a candidate topic for inclusion in _auto_added.

    Must:
    1. Pass the Phase 15 semantic gate (charged noun + subversion adjective)
    2. Not appear in existing_pool (case-insensitive substring match for
       near-duplicates)
    3. Not overlap any published-title signature (Stage 2 dedup)
    """
    from pipeline.script_generator import _topic_passes_semantic_gate
    from pipeline.topic_signatures import topic_overlaps_published

    ok, why = _topic_passes_semantic_gate(topic)
    if not ok:
        return False, f"semantic_gate: {why}"

    # Near-duplicate check vs existing pool (lowercased, first 50 chars match)
    topic_key = topic.lower()[:50]
    for existing in existing_pool:
        if existing.lower()[:50] == topic_key:
            return False, f"near-duplicate of existing topic"

    if published_signatures and topic_overlaps_published(topic, published_signatures):
        return False, "stage2_dedup: overlaps a published title signature"

    return True, "passed"


def expand_arcs_if_needed(
    arcs: list[dict],
    used: set,
    published_signatures: list,
) -> int:
    """Phase 21 entry point. Called from `_pick_next_arc_topic` before the
    weighted draw. Checks each arc's eligible pool; for any arc with eligible
    count <= PER_ARC_STARVATION_THRESHOLD (and total <= HARD_CAP_PER_ARC),
    generates new topics via Gemini Flash and appends to `arc["_auto_added"]`.

    Returns the number of topics added across all arcs this call (0 if no
    expansion happened).

    Fail-safe: any error logs a warning and returns 0; the picker continues
    with whatever topics existed before.
    """
    # Quick global check first — if every arc is healthy, skip the LLM cost
    total_eligible = sum(
        _count_eligible(arc, used, published_signatures)
        for arc in arcs
    )
    starving_arcs = [
        arc for arc in arcs
        if _count_eligible(arc, used, published_signatures) <= PER_ARC_STARVATION_THRESHOLD
    ]

    if not starving_arcs and total_eligible >= GLOBAL_STARVATION_THRESHOLD:
        return 0   # all arcs healthy, no expansion needed

    print(f"    [topic-expander] starvation detected: total_eligible={total_eligible}, "
          f"starving_arcs={[a.get('character', '?') for a in starving_arcs]}")

    # Pull published titles from the runtime signature cache. We need raw
    # title strings for the LLM prompt (not signatures); piggyback on the
    # signature_of() data the picker already fetched.
    # In practice the picker doesn't expose the raw titles, so we re-fetch
    # them here (cached process-wide by topic_signatures).
    published_titles = _fetch_published_titles_safe()

    added_count = 0
    arcs_dirty = False
    for arc in starving_arcs:
        character_name = arc.get("name", "Unknown arc")
        character_dev = arc.get("character", "")
        fingerprint = arc.get("emotional_fingerprint", "")
        pool = arc_topic_pool(arc)

        if len(pool) >= HARD_CAP_PER_ARC:
            print(f"    [topic-expander] {character_name}: hard-cap "
                  f"({HARD_CAP_PER_ARC}) reached — skipping expansion")
            continue

        print(f"    [topic-expander] {character_name}: requesting "
              f"{TOPICS_PER_EXPANSION} new topics from Gemini Flash...")
        candidates = _generate_topics_for_character(
            character_name=character_name,
            emotional_fingerprint=fingerprint,
            existing_topics=pool,
            published_titles=published_titles,
        )

        accepted = []
        for cand in candidates:
            ok, why = _validate_new_topic(cand, pool + [e["topic"] for e in accepted], published_signatures)
            if not ok:
                print(f"    [topic-expander] {character_name}: rejected '{cand[:60]}...' — {why}")
                continue
            accepted.append({
                "topic": cand,
                "source": _SOURCE_TAG,
                "added_at": datetime.now(timezone.utc).isoformat(),
                "validated_passes": ["semantic_gate", "stage2_dedup"],
                "llm_quality_tier": "fast",
            })

        if accepted:
            arc.setdefault("_auto_added", []).extend(accepted)
            added_count += len(accepted)
            arcs_dirty = True
            total_pool = len(arc.get("topics", [])) + len(arc.get("_auto_added", []))
            print(f"    [topic-expander] {character_dev or character_name}: "
                  f"added {len(accepted)} new topics "
                  f"(_auto_added.len={len(arc['_auto_added'])}, "
                  f"total_pool={total_pool}/{HARD_CAP_PER_ARC})")

    # Persist if anything changed
    if arcs_dirty:
        _persist_arcs(arcs)

    return added_count


def _fetch_published_titles_safe() -> list[str]:
    """Pull recent published video titles for LLM grounding. Fail-safe:
    returns [] on any error (network, missing token, etc.)."""
    try:
        from googleapiclient.discovery import build
        import pickle

        token_path = "token.pickle"
        if not os.path.exists(token_path):
            return []
        with open(token_path, "rb") as f:
            creds = pickle.load(f)
        yt = build("youtube", "v3", credentials=creds)

        ch = yt.channels().list(part="contentDetails", mine=True).execute()
        uploads_pl = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

        ids = []
        page = None
        while len(ids) < 60:
            r = yt.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_pl,
                maxResults=min(50, 60 - len(ids)),
                pageToken=page,
            ).execute()
            ids += [it["contentDetails"]["videoId"] for it in r["items"]]
            page = r.get("nextPageToken")
            if not page:
                break

        titles = []
        for i in range(0, len(ids), 50):
            chunk = ids[i:i + 50]
            r = yt.videos().list(part="snippet", id=",".join(chunk)).execute()
            for it in r["items"]:
                titles.append(it["snippet"]["title"])
        return titles
    except Exception as e:
        print(f"    [topic-expander] could not fetch published titles ({str(e)[:80]}); "
              f"LLM will generate without that grounding")
        return []


def _persist_arcs(arcs: list[dict]) -> None:
    """Atomically rewrite character_arcs.json preserving the wrapper structure."""
    if not os.path.exists(ARCS_PATH):
        print(f"    [topic-expander] WARNING: {ARCS_PATH} not found, "
              "cannot persist expansion. New topics will be in-memory only.")
        return

    try:
        with open(ARCS_PATH, encoding="utf-8") as f:
            wrapper = json.load(f)
    except Exception as e:
        print(f"    [topic-expander] WARNING: could not read {ARCS_PATH} "
              f"to update ({str(e)[:80]}); skipping persist")
        return

    if isinstance(wrapper, dict) and "arcs" in wrapper:
        wrapper["arcs"] = arcs
        wrapper["_phase21_auto_expansion"] = (
            "Phase 21 (2026-06-25) — `_auto_added` sidecar keys per arc hold "
            "auto-expander provenance objects {topic, source, added_at, "
            "validated_passes, llm_quality_tier}. The picker unions arc['topics'] "
            "and arc['_auto_added'][].topic as a single eligibility pool. "
            "See pipeline/topic_expander.py."
        )
        try:
            _atomic_write_json(ARCS_PATH, wrapper)
            print(f"    [topic-expander] persisted updated arcs to {ARCS_PATH}")
        except Exception as e:
            print(f"    [topic-expander] WARNING: atomic write failed "
                  f"({str(e)[:80]}); changes are in-memory only")
    else:
        print(f"    [topic-expander] WARNING: unexpected wrapper shape in "
              f"{ARCS_PATH}; skipping persist (in-memory changes only)")
