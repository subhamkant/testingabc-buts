from google import genai
import json
import random
import os
import re as _re   # Phase 15 (2026-06-08) — module-level _re needed by the
                   # tier-rotation regex constants; previously only imported
                   # mid-file (~line 695) which made early references fail.
import time


# Transient-error signatures that warrant a retry rather than failing the run.
# Covers Gemini 5xx (UNAVAILABLE/high demand), 429 rate-limit, and network
# timeouts — all of which clear within seconds-to-minutes in practice.
_GEMINI_TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE", "high demand", "currently experiencing",
    "500", "502", "504", "INTERNAL",
    "429", "RESOURCE_EXHAUSTED", "rate limit",
    "timeout", "Timeout", "TimeoutError",
    "connection reset", "ConnectionError",
)


# Model cascades by quality tier. Mahabharata + WhatIf dramatization use
# `quality="best"` which tries Gemini 2.5 Pro first (better Hindi creative
# prose than Flash — the 2026-05-13 local test had hindi grammar errors
# like `भीम ने प्रतिज्ञा हे` and gender mismatch `बच गए` for a feminine
# subject, characteristic of Flash-tier model output on long-form Hindi).
# Other call sites (outline pass, validators, etc.) use `quality="fast"`
# to preserve quota — they don't need creative-prose grade.
_GEMINI_MODELS_BEST = ("gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite")
_GEMINI_MODELS_FAST = ("gemini-2.5-flash", "gemini-2.5-flash-lite")

# Process-wide cache of models confirmed unavailable on free tier (limit: 0).
# Set on first encounter to skip ~5 wasted 429 attempts on every subsequent
# call. Discovered 2026-05-19: Google removed gemini-2.5-pro from the free
# tier entirely (free-tier limit is 0, not 50 RPD). All 5 of our API keys
# are on free-tier-only GCP projects, so Pro hits this wall every time.
# Once-and-skip preserves the cascade for the day Google may re-enable it.
_FREE_TIER_DISABLED_MODELS: set[str] = set()


def _gemini_keys():
    """
    Returns list of (label, api_key) tuples for the Gemini API. Walks
    GEMINI_API_KEY first (primary), then GEMINI_API_KEY_FALLBACK,
    GEMINI_API_KEY_FALLBACK_2, _3, _4, _5 if set — each additional Google
    account's project adds 50 RPD of Pro quota. With all 6 keys configured:
    300 RPD Pro vs 50 with primary alone.

    Range bumped 2026-05-18: previously hard-coded to primary + 3 fallbacks
    (4 keys total). User had GEMINI_API_KEY_FALLBACK_4 set in .env but the
    code wasn't reading it. Extended to _4 + _5 so any keys present in the
    env actually get picked up.

    Mirrors the existing _elevenlabs_keys() pattern.

    Empty/whitespace-only keys (and duplicates) are filtered out so
    missing fallbacks don't break primary-only operation.
    """
    keys = []
    seen = set()
    candidates = [("primary", "GEMINI_API_KEY"), ("fallback", "GEMINI_API_KEY_FALLBACK")]
    for n in range(2, 6):  # _2, _3, _4, _5
        candidates.append((f"fallback-{n}", f"GEMINI_API_KEY_FALLBACK_{n}"))
    for label, env_name in candidates:
        val = os.environ.get(env_name, "").strip()
        if val and val not in seen:
            seen.add(val)
            keys.append((label, val))
    return keys


def _call_llm(prompt: str, quality: str = "fast") -> str:
    """
    Calls Groq (primary, 14 400 RPD free) then falls back to Gemini.

    `quality="fast"` (default): Gemini cascade is flash → flash-lite. Used
    for outline pass, validators, and any call site where prose quality
    isn't load-bearing.

    `quality="best"`: Gemini cascade is pro → flash → flash-lite. Used for
    Mahabharata + WhatIf dramatization where Hindi grammar errors and
    summary-prose register hurt the final video. Pro has a free-tier limit
    of 50 RPD on the Generative Language API; the cascade falls through to
    Flash on quota exhaustion or transient errors.

    Token budgets are sized to fit the longest prompts in this file —
    Mahabharata's 2-pass dramatization step emits ~5-6k chars of bilingual
    JSON. Values below have ~2x headroom over the longest observed output.
    """
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()

    # ── Primary: Groq ────────────────────────────────────────────────────────
    if groq_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=8192,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Groq failed: {e} — falling back to Gemini ({quality})...")

    # ── Fallback: Gemini cascade with retry-on-transient + key fallback ─────
    from google.genai import types as _genai_types
    keys = _gemini_keys()
    if not keys:
        raise RuntimeError("No GEMINI_API_KEY configured")
    config = _genai_types.GenerateContentConfig(max_output_tokens=16384)
    models = _GEMINI_MODELS_BEST if quality == "best" else _GEMINI_MODELS_FAST
    return _gemini_call_with_retry(keys, prompt, config, models=models)


def _gemini_call_with_retry(keys, prompt: str, config, models=None) -> str:
    """
    Walks models × keys. For each model, tries every available key; if
    every key returns a quota/auth error on that model, falls over to
    the next model. Within a single (model, key) attempt, retries with
    backoff only on TRANSIENT-but-not-quota errors (5xx / timeout /
    network) — quota 429s don't retry because the quota doesn't reset
    in the 73s backoff window.

    Backoff per (model, key) on transient: 0s, 8s, 20s, 45s.

    Key fallback pattern matches the existing _elevenlabs_keys() flow:
    primary GEMINI_API_KEY first, then GEMINI_API_KEY_FALLBACK if
    configured. Each key represents an independent GCP project (which
    has its own 50 RPD Pro quota), so when both are configured we get
    100 RPD of Pro total.
    """
    if models is None:
        models = _GEMINI_MODELS_FAST
    backoffs = [0, 8, 20, 45]
    last_err = ""

    for model_idx, model_name in enumerate(models):
        # Once-and-skip for models confirmed free-tier-disabled this process
        # (2026-05-19: gemini-2.5-pro free-tier limit is 0 across all keys).
        # Saves ~3-5s of wasted 429s per script call.
        if model_name in _FREE_TIER_DISABLED_MODELS:
            print(f"    {model_name} skipped — free-tier disabled this run "
                  f"(detected earlier; falling through to next model)")
            continue
        for key_idx, (key_label, api_key) in enumerate(keys):
            client = genai.Client(api_key=api_key)
            # First (model, key) combo gets full retry-with-backoff.
            # Subsequent keys/models get a single attempt each.
            attempts = backoffs if (model_idx == 0 and key_idx == 0) else [0]
            for attempt, wait in enumerate(attempts):
                if wait:
                    print(f"    {model_name}/{key_label} retry in {wait}s (attempt {attempt + 1}/{len(attempts)})...")
                    time.sleep(wait)
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config,
                    )
                    if model_idx > 0 or key_idx > 0:
                        print(f"    [OK] {model_name}/{key_label} succeeded as fallback")
                    return response.text.strip()
                except Exception as e:
                    err_str = str(e)
                    last_err = f"{model_name}/{key_label}: {err_str[:140]}"
                    is_transient = any(m in err_str for m in _GEMINI_TRANSIENT_MARKERS)
                    is_quota = "RESOURCE_EXHAUSTED" in err_str or "exceeded your current quota" in err_str

                    # Detect free-tier-disabled state (limit: 0 in quota error).
                    # Google's response includes the literal "limit: 0" string
                    # when a model is structurally unavailable to free tier.
                    # When detected, cache it process-wide so subsequent script
                    # generations skip this model entirely.
                    if is_quota and "limit: 0" in err_str and model_name not in _FREE_TIER_DISABLED_MODELS:
                        _FREE_TIER_DISABLED_MODELS.add(model_name)
                        print(f"    [!] {model_name} free-tier limit is 0 — caching as "
                              f"unavailable for this run (will skip Pro on subsequent calls). "
                              f"Google removed free-tier access; enable billing on a GCP "
                              f"project to unlock paid Pro.")

                    if not is_transient and not (model_idx == 0 and key_idx == 0):
                        # Non-transient on a fallback combo — move to next combo
                        print(f"    {model_name}/{key_label} non-transient: {err_str[:100]}")
                        break
                    if not is_transient:
                        # First-combo non-transient (e.g. 401) — bubble up
                        raise
                    if is_quota:
                        # Quota 429 — don't retry on this key, the daily reset
                        # won't happen in our 73s backoff window. Move to next
                        # key (different GCP project) or next model.
                        # If we just cached this model as free-tier-disabled,
                        # break the KEY loop entirely (all keys hit same limit).
                        if model_name in _FREE_TIER_DISABLED_MODELS:
                            break
                        print(f"    {model_name}/{key_label} quota exhausted — skipping retries")
                        break
                    # Transient but not quota: per-minute rate limit, 5xx, network.
                    # Retry with backoff if attempts remain.
                    print(f"    {model_name}/{key_label} transient: {err_str[:100]}")
            # If model was cached as free-tier-disabled mid-loop, stop trying
            # other keys on this model — they all share the same free-tier wall.
            if model_name in _FREE_TIER_DISABLED_MODELS:
                break
        # If we've tried all keys on this model, fall to the next model.
        if model_idx < len(models) - 1:
            print(f"    {model_name} exhausted on all {len(keys)} key(s) — falling over to {models[model_idx + 1]}...")

    raise RuntimeError(f"All Gemini (model, key) combos failed; last error: {last_err}")


# ── Story Topics — well-known Mahabharata incidents ───────────────────────────
# Topic phrasing matters more than topic choice. Every entry follows the
# Bhishma-template that produced the channel's best-performing video:
#   [Named character]'s [ONE specific X] — [consequence / payoff]
#
# The "ONE X" frames a single central thread the LLM can echo in Scene 1's
# hook AND the final scene's resolution (see NARRATIVE BOOKEND rule in the
# Mahabharata script prompt). The "— consequence" half gives the payoff the
# bookend will land on.
#
# Reference: "Bhishma's sacrifice — how ONE OATH destroyed a dynasty" → Scene 1
# hook "एक प्रतिज्ञा ने कुरुवंश को शाप दिया था" → Scene 6 closure "एक वचन
# ने कुरुवंश को... वंचित कर दिया". Same noun. Same subject. Closure delivered.
# ─────────────────────────────────────────────────────────────────────────────
# CHARACTER ARC SYSTEM (added 2026-05-15)
# ─────────────────────────────────────────────────────────────────────────────
# Sequential character arcs that train the YouTube Suggested algorithm to pair
# episodes together. Walks assets/character_arcs.json in order: arc 1 episode
# 1, 2, ... 7, then arc 2 episode 1 ... etc. State tracked via existing
# recent_topics.json (which already records every used topic with timestamps).
#
# When all arcs are exhausted (35 episodes × 5 arcs = 35 used), falls back to
# the legacy random STORY_TOPICS / MOTIVATIONAL_THEMES pool below.
_ARC_FILE = os.path.join(os.path.dirname(__file__), "..", "assets", "character_arcs.json")
_RECENT_TOPICS_FILE = os.path.join(os.path.dirname(__file__), "..", "recent_topics.json")


def _load_arcs() -> list:
    """Returns the list of arcs from assets/character_arcs.json, or [] if missing."""
    try:
        with open(_ARC_FILE, encoding="utf-8") as f:
            return json.load(f).get("arcs", []) or []
    except Exception:
        return []


def _load_used_topics() -> set[str]:
    """Returns the set of topic strings already used (from recent_topics.json,
    mahabharata section). recent_topics.json is auto-committed by GHA after
    every successful upload, so this is the source of truth across runs."""
    try:
        with open(_RECENT_TOPICS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return {entry["topic"] for entry in data.get("mahabharata", []) if entry.get("topic")}
    except Exception:
        return set()


def _last_used_arc_index() -> int | None:
    """Phase 12 (2026-06-03). Return the arc index whose topic appears LAST
    in recent_topics.json. Returns None if recent_topics.json is empty / the
    most-recent topic matches no arc. Caller uses this to start the round-
    robin walk one arc AFTER the most-recent character — guarantees no two
    consecutive renders share a character."""
    try:
        with open(_RECENT_TOPICS_FILE, encoding="utf-8") as f:
            entries = json.load(f).get("mahabharata", [])
    except Exception:
        return None
    if not entries:
        return None
    last_topic = entries[-1].get("topic", "")
    if not last_topic:
        return None
    for arc_idx, arc in enumerate(_load_arcs()):
        if last_topic in arc.get("topics", []):
            return arc_idx
    return None


def _stable_episode_n(arc_idx: int, topic_idx: int) -> int:
    """Phase 12 (2026-06-03). Stable per-arc episode number — survives the
    round-robin rotation so the same (character, episode) always emits the
    same #N regardless of when it's rendered.

      arc 0 (Karna)    -> 11..17
      arc 1 (Bhishma)  -> 21..27
      arc 2 (Arjuna)   -> 31..37
      arc 3 (Draupadi) -> 41..47
      arc 4 (Yudhi)    -> 51..57
      arc 5 (Eklavya)  -> 61..67
      arc 6 (Ashwa)    -> 71..77
    """
    return (arc_idx + 1) * 10 + (topic_idx + 1)


# ─── Phase 15 (2026-06-08) — Tier-weighted character rotation ────────────────
# Replaces Phase 12's pure round-robin. Channel analytics 2026-06-08 showed
# -68.8% view decline over 7 uploads despite Phase 11-13 shipping cleanly.
# Root cause: uniform rotation forces low-fandom characters (Yudhishthira /
# Eklavya) into the Shorts algorithmic feed, where they pull 11-15 views vs
# 600+ for iconic characters (Karna / Draupadi / Bhishma). Tiered weighting
# preserves variety while keeping the Shorts feed algorithmically warm.
#
# Weights derived from observed view performance, normalized to sum=100.
_TIER_WEIGHTS = {
    # Tier 1 — Anchors (60% combined). Iconic, viral, audience has priors.
    "कर्ण":      ("T1", 25),    # Karna   — proven 600+ floor
    "द्रौपदी":   ("T1", 20),    # Draupadi — 594v on 2026-06-07 publish
    "भीष्म":     ("T1", 15),    # Bhishma  — 861v top performer (Untold Sacrifice)
    # Tier 2 — Wildcards (30% combined). Land IF given the 4-part DNA framing.
    "अर्जुन":    ("T2", 15),    # Arjuna
    "अश्वत्थामा": ("T2", 15),    # Ashwatthama
    # Tier 3 — Philosophers (10% combined). Used sparingly.
    "युधिष्ठिर": ("T3",  5),    # Yudhishthira — 15v floor; tighten gate
    "एकलव्य":   ("T3",  5),    # Eklavya
}


# Phase 15 semantic topic gate — the "4-part DNA" filter. A topic must
# contain BOTH a charged moral noun AND a subversion adjective to clear
# the gate. Bilingual: arc topics live in English (character_arcs.json)
# but the LLM emits Hindi narration, so we accept either script. The
# double-match (noun AND adjective) protects against single-keyword
# false positives where setup phrases coincidentally contain a charged
# word without the framing punch.
_CHARGED_NOUNS = _re.compile(
    # Hindi (Devanagari) — these are the charged NOUN forms
    r"पाप|गलती|घमंड|सच|धोखा|बलिदान|श्राप|प्रतिशोध|"
    r"अपमान|विश्वासघात|झूठ|ज़िद|मजबूरी|डर|"
    # English nouns (matches arc topics in character_arcs.json)
    r"\bsin\b|\bmistake\b|\bpride\b|\btruth\b|\bbetrayal\b|\bsacrifice\b|"
    r"\bcurse\b|\brevenge\b|\bhumiliation\b|\blie\b|"
    r"\bfear\b|\bjealousy\b|\bregret\b|\bloyalty\b|\bvow\b|\bguilt\b|"
    r"\bcost\b|\bfatal\b",
    _re.IGNORECASE,
)
_SUBVERSION_ADJECTIVES = _re.compile(
    # Hindi adjectives — "last/final" → "अंतिम" goes here, not in nouns
    r"असली|बड़ी|बड़ा|छिपा|छिपी|अनकहा|अनकही|गुप्त|कड़वा|भयानक|अंतिम|"
    # English subversion adjectives. "last" / "final" are modifiers
    # ("Karna's LAST regret" — last modifies regret) — they belong here.
    r"\breal\b|\bbig\b|\bbiggest\b|\bhidden\b|\buntold\b|\bsecret\b|"
    r"\bbitter\b|\bterrifying\b|\bgreatest\b|\bdeepest\b|\blast\b|\bfinal\b|"
    r"\bthat\s+one\b|\bthe\s+one\b",
    _re.IGNORECASE,
)


def _topic_passes_semantic_gate(topic: str) -> tuple[bool, str]:
    """Phase 15 (2026-06-08). Return (ok, reason). A topic must contain
    BOTH at least one charged moral noun (sin/mistake/pride/etc.) AND
    at least one subversion adjective (real/hidden/untold/etc.). Either
    Hindi or English matches count. Topics that fail are passed over by
    the weighted rotation in favor of the next eligible candidate in
    the drawn tier; the rotation downgrades to a lower tier if no topic
    in the drawn tier passes."""
    has_noun = bool(_CHARGED_NOUNS.search(topic))
    has_adj  = bool(_SUBVERSION_ADJECTIVES.search(topic))
    if not has_noun and not has_adj:
        return False, "missing both charged-moral-noun AND subversion-adjective"
    if not has_noun:
        return False, "missing charged-moral-noun (sin/mistake/pride/etc.)"
    if not has_adj:
        return False, "missing subversion-adjective (real/hidden/untold/etc.)"
    return True, "passes DNA gate"


def _pick_next_arc_topic() -> tuple[str, int, str, str | None] | None:
    """
    Walk character arcs sequentially, picking the first topic NOT in
    recent_topics.json AND NOT overlapping any recently-published video
    title on the channel. Returns (topic, episode_n, arc_name,
    next_topic_in_arc) or None when all arcs are exhausted.

    Two-stage dedup (2026-05-23):
      1. Exact-string match against recent_topics.json (existing logic)
      2. Runtime signature overlap against last 50 published titles via
         the YouTube API (topic_signatures.fetch_recent_title_signatures).
         Fail-safe — if the API is unavailable, the runtime check is
         skipped and only stage 1 runs.

    The runtime check catches:
      - Topics with slightly different exact strings (manual-trigger paths,
        legacy STORY_TOPICS entries used before the arc system, etc.)
      - Drift between recent_topics.json and the actual channel
      - Same character + same incident expressed in different topic strings

    `next_topic_in_arc` (added 2026-05-16, Tier 2 Fix 2.0) is the NEXT
    unused topic in the SAME arc, used by the cliffhanger prompt to tease
    the next episode. None if this is the last unused episode of its arc.

    episode_n is the GLOBAL episode count across the channel (sequential
    1, 2, 3 ... across all arcs combined), so titles read like
    "महाभारत #14: कर्ण की पीड़ा" giving viewers a clear progression cue.
    The arc_name is for logging / future per-arc playlist routing.
    """
    from pipeline.topic_signatures import (
        fetch_recent_title_signatures,
        topic_overlaps_published,
    )

    arcs = _load_arcs()
    used = _load_used_topics()
    if not arcs:
        return None

    # Stage 2 dedup data — fetched once per process. Returns [] on any error.
    published_sigs = fetch_recent_title_signatures()

    # Phase 15 (2026-06-08): TIERED WEIGHTED ROTATION replaces Phase 12
    # round-robin. Random draw across all characters with tier-based weights
    # (T1 anchors 60% / T2 wildcards 30% / T3 philosophers 10%). The last-
    # drawn character is MASKED from the next draw so we never repeat
    # consecutively. Within the drawn character's arc, the semantic gate
    # (Phase 15 charged-noun + subversion-adjective regex) filters for
    # topics matching the 4-part DNA isolated from the channel's 158% AVP
    # winners. If the drawn character has no gate-passing unused topic,
    # fall through tier by tier until SOMETHING ships (cron must never
    # publish nothing).

    last_arc = _last_used_arc_index()
    last_character = arcs[last_arc].get("character", "") if last_arc is not None else ""

    # Build per-character arc-index map so we can resolve a drawn character
    # back to its arc + topics + stable episode_n.
    char_to_arc_idx = {arc.get("character", ""): i for i, arc in enumerate(arcs)}

    def _try_pick_from_character(character: str, enforce_gate: bool):
        """Return (topic, episode_n, arc_name, next_topic_in_arc) for the first
        unused + non-overlapping topic in `character`'s arc, optionally
        enforcing the semantic gate. None if no eligible topic."""
        arc_idx = char_to_arc_idx.get(character)
        if arc_idx is None:
            return None
        arc = arcs[arc_idx]
        arc_topics = arc.get("topics", [])
        for idx, topic in enumerate(arc_topics):
            if topic in used:
                continue
            if published_sigs and topic_overlaps_published(topic, published_sigs):
                continue
            if enforce_gate:
                ok, why = _topic_passes_semantic_gate(topic)
                if not ok:
                    print(f"    [phase15-gate] skipping {character} topic "
                          f"'{topic[:60]}...' — {why}")
                    continue
            # Found a usable topic. Compute the cliffhanger teaser too.
            next_topic = None
            for ahead in arc_topics[idx + 1:]:
                if ahead in used:
                    continue
                if published_sigs and topic_overlaps_published(ahead, published_sigs):
                    continue
                next_topic = ahead
                break
            return topic, _stable_episode_n(arc_idx, idx), arc.get("name", ""), next_topic
        return None

    # Build the eligible pool (mask last-drawn character).
    eligible = [
        (char, info) for char, info in _TIER_WEIGHTS.items()
        if char != last_character and char in char_to_arc_idx
    ]
    if not eligible:
        # All characters masked / none in arcs JSON — relax the mask.
        eligible = [
            (char, info) for char, info in _TIER_WEIGHTS.items()
            if char in char_to_arc_idx
        ]
    chars   = [c for c, _ in eligible]
    weights = [w for _, (_, w) in eligible]

    # Try up to 12 weighted draws WITH the semantic gate enforced. Each
    # draw is independent — if a drawn character has no gate-passing
    # topic, we draw again. This naturally biases toward characters whose
    # arcs HAVE gate-passing topics (T1 anchors expected to dominate).
    import random as _random_local  # local alias to avoid top-level import collision
    tried_chars = set()
    for attempt in range(12):
        drawn = _random_local.choices(chars, weights=weights, k=1)[0]
        if drawn in tried_chars:
            continue
        tried_chars.add(drawn)
        tier = _TIER_WEIGHTS[drawn][0]
        print(f"    [phase15-tier] attempt {attempt+1}: drew {tier} {drawn} "
              f"(masked previous: {last_character or 'none'})")
        result = _try_pick_from_character(drawn, enforce_gate=True)
        if result:
            return result

    # Gate-enforced draws exhausted. Fall through: try every character WITHOUT
    # the gate. Better to publish a topic that doesn't perfectly fit the DNA
    # than to ship nothing.
    print(f"    [phase15-fallback] gate-enforced draws exhausted "
          f"({len(tried_chars)} characters tried); relaxing gate")
    for char, _ in eligible:
        result = _try_pick_from_character(char, enforce_gate=False)
        if result:
            return result
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Legacy random-pool fallback (kept for when arcs are exhausted)
# ─────────────────────────────────────────────────────────────────────────────
STORY_TOPICS = [
    # ── Core canonical incidents (rephrased: claim → consequence) ─────────
    "Kurukshetra's opening conches — the one moment that ended an entire dynasty",
    "Krishna's one battlefield conversation with Arjuna — how the Bhagavad Gita began with a single moment of doubt",
    "Draupadi's one cry for help — how the vastraharan was stopped by a miracle that doomed the Kuru clan",
    "Karna's one secret — how being born to a god still couldn't save him from being called sutaputra his whole life",
    "Shakuni's one loaded dice — how a rigged game cost the Pandavas their kingdom and their wife",
    "Abhimanyu's one decision to enter the Chakravyuha — how a 16-year-old's courage trapped him in a maze of death",
    "Bhishma's one moment of pity for Shikhandi — how invincibility ended on a bed of arrows",
    "Krishna's one cosmic form — the Vishwaroop vision that made Arjuna forget he was holding a bow",
    "Draupadi's one impossible test — how a rotating fish chose the husband who would change history",
    "Ekalavya's one gurudakshina — the severed thumb that revealed Drona's deepest fear of being surpassed",
    "Duryodhana's one act in court — how humiliating Draupadi sealed the Kuru dynasty's destruction",
    "Duryodhana's one fire trap — how the lac palace was built to burn the Pandavas alive in their sleep",
    "Karna's one gift to Indra — how donating his divine armour guaranteed his own death",
    "Arjuna's one vow at sunset — how Jayadratha's death required a miracle from Krishna himself",
    "Bhima's one strike at Duryodhana's thigh — how the Kurukshetra war ended with a forbidden blow",
    "Ashwatthama's one night of revenge — how a massacre of sleeping warriors became his eternal curse",
    "Gandhari's one curse to Krishna — how a mother's grief destroyed an entire divine clan",
    "The Pandavas' one final pilgrimage — how five brothers fell one by one on the Himalayan slopes",
    "Karna's one reunion with Kunti — the secret a mother kept for decades, revealed too late",
    "Drona's one archery test — how only Arjuna saw the bird's eye and earned the title of greatest archer",

    # ── YouTube-suggested topics (rephrased to match the same template) ───
    "Krishna's one fake sunset — the celestial trick that delivered Jayadratha to Arjuna's vow",
    "Jayadratha's one day of glory — how he held off the entire Pandava army on day 13 of war",
    "Jayadratha's one severed head — the curse that fulfilled itself in his father's meditating lap",
    "Krishna's one offer to Karna — the throne refused, the friendship lost, the fate sealed",
    "Bhishma's one weakness — how Amba reborn as Shikhandi ended the war's most unstoppable warrior",
    "Gandhari's one blindfold — the lifetime vow that left her never knowing her hundred sons' faces",
    "Aravan's one-day marriage to Mohini — the sacrifice that bought the Pandavas their victory",
    "Sahadeva's one curse — the gift of seeing every future moment but never being allowed to warn anyone",
    "Ashwatthama's one immortal punishment — how being unable to die became worse than dying",
    "Krishna's one Sudarshan strike — the 100th sin that beheaded Shishupala in a single moment",
    "Kunti's one decades-long secret — the night before war when she finally told Karna the truth",
    "Barbarika's one head — the sacrifice Krishna demanded so a single warrior couldn't end the war in one breath",
    "Parshurama's one curse on Karna — the forgotten weapons that decided the moment of his death",
    "Balarama's one refusal — how Krishna's brother walked away from Kurukshetra and chose pilgrimage instead",
    "Vidura's one warning to Dhritarashtra — the blind king's refusal that doomed his hundred sons",
    "Yudhishthira's one Yaksha encounter — the test of dharma that revived his four dead brothers",
    "Shikhandi's one appearance on the battlefield — the moment Bhishma laid down his bow forever",
    "Arjuna's one vow after Abhimanyu's death — how a father's grief consumed an entire day of war",
    "The forgotten warriors' one stand — how unnamed kings delayed Arjuna so Jayadratha could survive a few more hours",
    "Nakula's one quiet mastery — the most beautiful warrior of the Mahabharata, remembered for his silence",
    "Bhishma and Amba's one hidden conversation — the cosmic destiny set in motion by a rejected princess",
    "Krishna's one celestial trick — how he created a premature sunset to break Jayadratha's last shield",
    "Drona's one broken teaching — the unspoken privilege that made Arjuna the only true archer in his class",
    "Bhima's one vow at the dice hall — how he kept his word to drink Dushasana's blood at Kurukshetra",
    "Karna's one night of truth — when Kunti revealed his identity and he chose his fate anyway",
]

# ── Motivational Themes ───────────────────────────────────────────────────────
# Same Bhishma-template: [character/concept]'s [one X] — [consequence/payoff].
# "Bhishma's sacrifice — how one oath destroyed a dynasty" is the gold standard
# this list models from (the channel's best-performing video to date).
MOTIVATIONAL_THEMES = [
    "Krishna's one Karma Yoga lesson — how acting without attachment freed Arjuna to fight",
    "Arjuna's one moment of doubt — how the greatest archer almost quit before the greatest battle",
    "Karna's one quiet dignity — the man who gave everything and never once asked for credit",
    "Bhishma's sacrifice — how one oath destroyed a dynasty",
    "The Bhagavad Gita's one hidden lesson — what most readers miss after decades of study",
    "Krishna's one strategic choice — how refusing to fight at Kurukshetra still won him the war",
    "Draupadi's one unbroken spirit — how a queen survived five husbands' failures and never bent",
    "Yudhishthira's one lifelong truth — the man who never lied, and the day even he had to",
    "The Pandavas' 13 years of exile — how patience became their deadliest weapon",
    "Vidura's one warning — the words to Dhritarashtra that could have prevented the entire war",
]

# ── Krishna Direct-Address Series ─────────────────────────────────────────────
# First-person motivational speeches where Krishna addresses a named listener
# directly ("देखो पार्थ...", "मैं तुमसे एक सच कहता हूँ..."). Replaces 1 of 4
# daily Mahabharata slots. Single voice (Krishna's), 30-45s, 5 scenes.
#
# Listeners pool — Arjuna appears 3 times so the natural random draw lands on
# him ~60% of the time (most recognizable archetype, broadest reach). Other
# listeners (Karna, Bhishma, Yudhishthira, Uddhava) provide variety so themes
# can repeat across listeners without feeling stale.
KRISHNA_LISTENERS = [
    "Arjuna (पार्थ)", "Arjuna (पार्थ)", "Arjuna (पार्थ)",  # ~60% weight
    "Yudhishthira",
    "Karna",
    "Bhishma",
    "Uddhava",
]

KRISHNA_THEMES = [
    "Responsibility — chosen ones don't get easy paths",
    "Detached action — do the work, release the fruit",
    "Fear of failure is louder than failure itself",
    "Silence and exhaustion are part of the road",
    "Why doubt visits the strongest the night before victory",
    "Duty above bloodline",
    "Anger is a fire that burns the holder first",
    "The strength to forgive is greater than the strength to fight",
    "Why the mind is the battlefield, not Kurukshetra",
    "When everyone abandons you, dharma still walks beside you",
    "Action without ego — the secret of a free man",
    "Why surrender is not weakness but the highest strength",
]


# ── What If Topics — science / nature / civilization hypotheticals ────────────
# These are curiosity-driven thought experiments grounded in plausible science.
# Used by the "whatif" series — completely separate from Mahabharata content.
STORY_TOPICS_WHATIF = [
    "What if humans suddenly disappeared from Earth tomorrow?",
    "What if Earth had rings like Saturn?",
    "What if dinosaurs were still alive today?",
    "What if the Sun disappeared for 24 hours?",
    "What if gravity on Earth were cut in half?",
    "What if the Moon were twice as close to Earth?",
    "What if all the world's oceans dried up overnight?",
    "What if humans could photosynthesize like plants?",
    "What if the Earth stopped spinning for one second?",
    "What if Antarctica's ice sheet melted completely?",
    "What if every volcano on Earth erupted at once?",
    "What if the Amazon rainforest disappeared?",
    "What if humans had evolved with three eyes?",
    "What if a black hole entered our solar system?",
    "What if the Sahara desert turned green again?",
    "What if Earth's magnetic field flipped tomorrow?",
    "What if every insect on Earth went extinct?",
    "What if the Pacific Ocean froze solid?",
    "What if humans never invented fire?",
    "What if sleep were no longer necessary for survival?",
]


# Common Hindi/English words that aren't "repetition problems" even when
# they appear multiple times — articles, pronouns, copulas, common verbs.
_REPETITION_STOPWORDS = {
    # Hindi — articles, copulas, postpositions, particles
    "है", "हैं", "था", "थे", "थी", "हो", "हुआ", "हुई", "होती", "होते",
    "में", "से", "को", "का", "की", "के", "ने", "और", "एक", "वह", "वे",
    "यह", "ये", "उस", "उन", "जो", "तो", "ही", "भी", "पर", "लिए",
    "नहीं", "कर", "करते", "करता", "करती", "किया", "गया", "गई",
    # Hindi possessive / reflexive pronouns — very common in storytelling
    "अपने", "अपनी", "अपना", "उनके", "उनकी", "उनका", "उसके", "उसकी", "उसका",
    "मेरे", "मेरी", "मेरा", "तेरे", "तेरी", "तेरा", "हमारे", "हमारी", "हमारा",
    # Hindi 1st/2nd-person pronouns — saturate Krishna direct-address speeches
    # ("मैं तुमसे कहता हूँ...") and shouldn't be flagged as repetition.
    "मैं", "मैंने", "मुझे", "मुझको", "मुझसे",
    "तुम", "तुम्हें", "तुम्हारे", "तुम्हारी", "तुम्हारा", "तुमसे", "तुमको",
    # Hindi auxiliary / common verbs
    "रहे", "रहा", "रही", "लगे", "लगा", "लगी", "जाते", "जाता", "जाती",
    "देते", "देता", "देती", "लेते", "लेता", "लेती", "होने",
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "and", "or", "but",
    "he", "she", "it", "they", "his", "her", "their", "this", "that",
    "as", "by", "from", "into", "than", "then", "so", "if", "not",
    "have", "has", "had", "do", "does", "did", "will", "would", "can",
    "could", "may", "one", "two",
}


# Mahabharata character names — both Devanagari and common Romanizations.
# Allowed to repeat freely because they're typically the subjects of the story
# and trying to avoid repeating them produces awkward indirection.
_CHARACTER_NAMES = {
    # Hindi (Devanagari)
    "भीष्म", "अर्जुन", "पार्थ", "कृष्ण", "द्रोण", "द्रोणाचार्य", "कर्ण", "युधिष्ठिर",
    "भीम", "नकुल", "सहदेव", "द्रौपदी", "पाण्डव", "पांडव", "कौरव",
    "दुर्योधन", "दुःशासन", "शकुनि", "धृतराष्ट्र", "गांधारी", "कुंती", "माद्री",
    "विदुर", "संजय", "अश्वत्थामा", "जरासंध", "शिशुपाल", "एकलव्य", "अभिमन्यु",
    "उत्तरा", "सुभद्रा", "देवव्रत", "शांतनु", "गंगा", "सत्यवती", "अंबा",
    "जयद्रथ", "घटोत्कच", "हिडिम्बा", "उलूपी", "विराट", "द्रुपद", "धृष्टद्युम्न",
    "शिखंडी", "पाण्डु", "पांडु", "व्यास", "नारद", "इन्द्र", "सूर्य",
    # Romanized
    "bhishma", "arjuna", "krishna", "drona", "karna", "yudhishthira",
    "bheema", "bhima", "nakula", "sahadeva", "draupadi", "pandavas",
    "kauravas", "duryodhana", "dushasana", "shakuni", "dhritarashtra",
    "gandhari", "kunti", "madri", "vidura", "sanjaya", "ashwatthama",
    "ekalavya", "abhimanyu", "subhadra", "devavrata", "shantanu",
    "ganga", "satyavati", "shikhandi", "pandu", "vyasa", "indra", "surya",
    "partha", "krishna's", "arjuna's", "bhishma's",
    # Places
    "कुरुक्षेत्र", "हस्तिनापुर", "इंद्रप्रस्थ", "द्वारका", "kurukshetra",
    "hastinapura", "indraprastha", "dwarka", "ayodhya",
}


# Hindi past-tense auxiliaries that stack into the "tha-tha-tha" verbal tic
# when every sentence ends with one. We allow some — past auxiliary IS valid
# Hindi narration — but if too many sentences end this way the script reads
# like a chronological list ("X किया था, Y किया था, Z किया था") rather than
# cinematic storytelling.
import re as _re
_PAST_AUX_END = _re.compile(r"(था|थी|थे|थीं)\s*[।!?\.]?\s*$")


def _check_past_aux_tic(scenes: list, threshold: float = 0.35) -> tuple:
    """
    Count what fraction of narration sentences (across all scenes) end with a
    past-auxiliary verb (था/थी/थे/थीं). Returns (ok, ratio, hits, total).

    Hindi-only check. Threshold of 0.35 means up to ~1 in 3 sentences may end
    that way — anything more is the verbal tic the user flagged.
    """
    total = 0
    hits  = 0
    for s in scenes:
        text = (s.get("narration") or "").strip()
        if not text:
            continue
        sentences = [
            t.strip()
            for t in _re.split(r"[।!?\.]+", text)
            if t.strip()
        ]
        for sent in sentences:
            total += 1
            if _PAST_AUX_END.search(sent):
                hits += 1
    ratio = hits / total if total else 0.0
    return (ratio <= threshold), ratio, hits, total


# ── Retention-engineering validators (added 2026-05-16) ─────────────────────
# Tier 1 cinematic upgrade — see plan
# C:\Users\rjyot\.claude\plans\have-a-look-into-polished-goblet.md. Hook +
# rehook are HARD gates (block + retry); rhythm + visual are SOFT (warn-only).

# Setup-line openers banned at scene 1 — these signal "documentary intro"
# and viewers swipe in the first 1.5 seconds.
_BANNED_HOOK_OPENERS = _re.compile(
    r"^\s*(यह\s+कहानी|यह\s+एक\s+(?:ऐसी\s+)?कहानी|एक\s+बार|बहुत\s+समय\s+पहले|"
    r"प्राचीन\s+काल|पुराने\s+ज़माने|कहते\s+हैं|"
    r"In\s+ancient|This\s+is\s+the\s+story|Long\s+ago|Once\s+upon)",
    _re.IGNORECASE
)
# Pattern B — opens with a question marker / curiosity word
_HOOK_QUESTION_OPENER = _re.compile(
    r"^\s*(क्या\s+आप|क्या\s+आपको|जब|कैसे|क्यों|कौन|किसने|"
    r"Did\s+you|What\s+if|How\s+did|Why\s+did|Who\s+killed)",
    _re.IGNORECASE
)
# Pattern C — first sentence trails off / ends with लेकिन / but
_HOOK_CLIFFHANGER_END = _re.compile(r"(लेकिन|पर|but|…|\.{3})\s*[।!?\.]?\s*$", _re.IGNORECASE)
# Pattern A — shocking specific fact (digit + named character)
_DIGIT_IN_TEXT = _re.compile(r"\d")
# Lightweight character-name list reused from the prompt
_CHARACTER_NAMES_LIST = (
    "Krishna", "Arjuna", "Bhishma", "Karna", "Draupadi", "Yudhishthira",
    "Bhima", "Nakula", "Sahadeva", "Duryodhana", "Dushasana", "Drona",
    "Ashwatthama", "Gandhari", "Dhritarashtra", "Vidura", "Kunti", "Shakuni",
    "Abhimanyu", "Subhadra", "Ghatotkacha", "Jayadratha", "Shikhandi", "Amba",
    "Satyavati", "Devavrata", "Parashurama", "Ekalavya", "Sanjaya", "Balarama",
    "Vyasa",
    "कृष्ण", "अर्जुन", "भीष्म", "कर्ण", "द्रौपदी", "युधिष्ठिर",
    "भीम", "दुर्योधन", "द्रोण", "अश्वत्थामा", "गांधारी", "धृतराष्ट्र",
    "विदुर", "कुंती", "शकुनि", "अभिमन्यु", "जयद्रथ", "शिखंडी",
    "सत्यवती", "देवव्रत", "परशुराम", "एकलव्य", "बलराम", "व्यास",
)


# Pattern D (Phase 11 retention refactor 2026-06-02) — paradox-fact hook.
# First sentence MUST contain BOTH a named character AND a contradiction /
# paradox marker within the first 10 words. Example: "भीष्म कभी हारे नहीं —
# पर एक स्त्री ने उन्हें गिरा दिया।" combines "भीष्म" (named character) +
# "नहीं" (contradiction) + "पर" (paradox pivot) → strongest in-media-res
# opener. Pattern A/B/C remain — D widens the acceptance surface.
_HOOK_PARADOX_FIRST_HALF = _re.compile(
    r"\b(लेकिन|पर|फिर\s+भी|कभी\s+नहीं|पहली\s+बार|आखिरी|एकमात्र|"
    r"but|yet|still|never|only|until)\b",
    _re.IGNORECASE
)


def _check_hook_pattern(scene1_narration: str) -> tuple[bool, str]:
    """
    Hook validator (HARD gate). Returns (ok, reason).

    Rejects setup-line openers ("यह कहानी है...", "Long ago...") and requires
    scene 1's first sentence to match one of four patterns:
      A — shocking fact: digit + named character
      B — question opener (क्या आप / Did you / etc.)
      C — cliffhanger trail (...लेकिन / ...but / ...)
      D — paradox-fact: BOTH a named character AND a paradox/contradiction
          marker within the first 10 words (Phase 11 retention refactor)
    """
    if not scene1_narration:
        return False, "scene 1 narration empty"
    text = scene1_narration.strip()
    if _BANNED_HOOK_OPENERS.match(text):
        return False, "setup-line opener (documentary intro)"
    # Use only the first sentence for pattern checks
    first_sentence = _re.split(r"[।!?\.]", text, maxsplit=1)[0].strip()
    if not first_sentence:
        return False, "empty first sentence"
    has_digit = bool(_DIGIT_IN_TEXT.search(first_sentence))
    has_character = any(name in first_sentence for name in _CHARACTER_NAMES_LIST)
    pattern_a = has_digit and has_character
    pattern_b = bool(_HOOK_QUESTION_OPENER.match(first_sentence))
    pattern_c = bool(_HOOK_CLIFFHANGER_END.search(first_sentence))
    # Pattern D — Phase 11 retention refactor: paradox-fact opener within
    # first 10 words. REQUIRES BOTH named character AND paradox marker.
    first_10_words = " ".join(first_sentence.split()[:10])
    has_paradox = bool(_HOOK_PARADOX_FIRST_HALF.search(first_10_words))
    pattern_d = has_character and has_paradox
    if pattern_a or pattern_b or pattern_c or pattern_d:
        which = (
            "A" if pattern_a
            else "B" if pattern_b
            else "C" if pattern_c
            else "D"
        )
        return True, f"pattern {which}"
    return False, "no shock-fact / question / cliffhanger / paradox pattern detected"


# ─── hook_title validator (Phase 11 retention refactor, 2026-06-02) ───────
# Validates the new `hook_title` JSON field used by the t=0 title-card
# overlay in subtitle_generator. SCOPE IS STRICTLY LIMITED to the
# hook_title string — does NOT read narration / image_prompt / any other
# field. Narration ellipses (Gemini TTS pause timing) are unaffected.
_HOOK_TITLE_CONTRADICTION = _re.compile(
    r"\b(but|yet|still|never|cost|broke|last|only|until|hidden|untold|real)\b|"
    r"लेकिन|पर|फिर\s+भी|कभी\s+नहीं|आखिरी|पहली|एकमात्र|जो",
    _re.IGNORECASE
)
_HOOK_TITLE_NAMED_CHAR = _re.compile(
    r"\b(Bhishma|Arjuna|Karna|Krishna|Draupadi|Yudhishthira|Bhima|Nakula|"
    r"Sahadeva|Drona|Ashwatthama|Eklavya|Ekalavya|Duryodhana|Dushasana|"
    r"Shikhandi|Kunti|Gandhari|Dhritarashtra|Vidura|Shakuni|Abhimanyu|"
    r"Jayadratha|Devavrata|Parashurama|Satyavati|Balarama|Vyasa)\b|"
    r"भीष्म|अर्जुन|कर्ण|कृष्ण|द्रौपदी|युधिष्ठिर|भीम|द्रोण|अश्वत्थामा|"
    r"एकलव्य|दुर्योधन|दुश्शासन|शिखंडी|कुंती|गांधारी|धृतराष्ट्र|विदुर|"
    r"शकुनि|अभिमन्यु|जयद्रथ|देवव्रत|परशुराम|सत्यवती|बलराम|व्यास",
    _re.IGNORECASE
)
_HOOK_TITLE_BANNED = _re.compile(
    r"[?!…]|\.\.\.|"
    r"^\s*(this\s+is|the\s+story|long\s+ago|once\s+upon|let\s+me|"
    r"यह\b|ये\b|एक\s+कहानी|बहुत\s+समय|कहते\s+हैं)",
    _re.IGNORECASE
)


_HOOK_TITLE_LATIN_ALPHA = _re.compile(r"[A-Za-z]")


def _check_hook_title(hook_title: str, language: str = "hi") -> tuple[bool, str]:
    """Phase 11 retention refactor 2026-06-02. Validates the hook_title
    field used by the t=0 title-card overlay. Reject reasons surface to
    the priority cascade for re-prompt.

    IMPORTANT SCOPE: this validator ONLY reads its `hook_title` argument
    (NEVER touches data["scenes"][...]["narration"] or any other field).
    The ellipsis-ban in _HOOK_TITLE_BANNED applies SOLELY to hook_title.
    Narration ellipses (which Gemini TTS uses for 300-400ms dramatic
    pauses per Phase 2/3 rhythm rules) are completely unaffected.

    Tofu fix 2026-06-03: when language == "hi", any Latin alphabet
    character is REJECTED. The bundled title-card font is
    NotoSansDevanagari-Bold.ttf which has no Latin glyph coverage —
    English chars render as yellow `.notdef` boxes (tofu) which is a
    swipe-instant retention killer.
    """
    if not hook_title or not hook_title.strip():
        return False, "hook_title missing or empty"
    title = hook_title.strip()
    if language == "hi" and _HOOK_TITLE_LATIN_ALPHA.search(title):
        m = _HOOK_TITLE_LATIN_ALPHA.search(title)
        return False, (
            f"hook_title contains Latin char '{m.group(0)}' but Hindi pipeline font "
            f"is Devanagari-only — would render as tofu: {title[:40]}"
        )
    if _HOOK_TITLE_BANNED.search(title):
        m = _HOOK_TITLE_BANNED.search(title)
        return False, f"hook_title contains banned punctuation/opener '{m.group(0)}': {title[:40]}"
    n_words = len(title.split())
    if n_words < 1 or n_words > 5:
        return False, f"hook_title must be 1-5 words, got {n_words}: {title[:40]}"
    has_char = bool(_HOOK_TITLE_NAMED_CHAR.search(title))
    has_contradiction = bool(_HOOK_TITLE_CONTRADICTION.search(title))
    if not has_char and not has_contradiction:
        return False, f"hook_title needs ≥1 named character OR paradox marker: {title[:40]}"
    return True, f"{n_words} words"


# Rehook contrast markers — must appear somewhere in the middle-window scenes
# to re-spike curiosity at the 40-60% retention drop-off point.
_REHOOK_MARKERS = _re.compile(
    r"(लेकिन|परंतु|किंतु|और\s+तभी|उसी\s+क्षण|जो\s+(?:किसी|कोई)\s+ने\s+(?:नहीं|न)\s+सोचा|"
    r"पर\s+(?:उसे|उन्हें)\s+नहीं\s+पता|But\b|But\s+what|Yet\b|Suddenly|And\s+then)",
    _re.IGNORECASE
)


def _check_rehook_present(scenes: list) -> tuple[bool, int]:
    """
    Mid-video rehook validator (HARD gate). Returns (ok, hit_scene_idx).

    Scans the middle window of narrative scenes (excluding the outro at
    position N-1) for a contrast marker that resets curiosity at the
    12-18s drop-off point. Window = scenes [N//2 - 1, N//2, N//2 + 1]
    where N is the count of non-outro scenes.
    """
    # Exclude the static subscribe outro (last scene). Treat the rest as the
    # narrative.
    narrative = scenes[:-1] if len(scenes) > 2 else scenes
    n = len(narrative)
    if n < 3:
        return True, -1  # too short to meaningfully rehook
    mid = n // 2
    window = {max(0, mid - 1), mid, min(n - 1, mid + 1)}
    for idx in sorted(window):
        text = (narrative[idx].get("narration") or "")
        if _REHOOK_MARKERS.search(text):
            return True, idx
    return False, -1


def _check_sentence_rhythm(scenes: list) -> tuple[bool, str]:
    """
    Sentence-rhythm validator (SOFT gate — warn-only, never blocks retry).

    Looks for evidence of varied sentence lengths across the whole script:
      - ≥30% of sentences are ≤7 words (short punch)
      - ≥1 sentence is ≥12 words (cinematic long)
      - stddev of sentence word-counts ≥3
    Returns (ok, reason_str). Per plan discipline, this is logged but does
    NOT block acceptance — prevents prompt-fighting where the LLM satisfies
    the regex but loses emotional truth.
    """
    sent_lengths = []
    for s in scenes[:-1]:  # skip outro
        text = (s.get("narration") or "").strip()
        for sent in _re.split(r"[।!?\.]+", text):
            sent = sent.strip()
            if sent:
                sent_lengths.append(len(sent.split()))
    if len(sent_lengths) < 4:
        return True, "too few sentences to evaluate"
    n = len(sent_lengths)
    short_count = sum(1 for w in sent_lengths if w <= 7)
    long_count  = sum(1 for w in sent_lengths if w >= 12)
    short_pct = short_count / n
    mean = sum(sent_lengths) / n
    variance = sum((w - mean) ** 2 for w in sent_lengths) / n
    stddev = variance ** 0.5
    fails = []
    if short_pct < 0.30:
        fails.append(f"short-line ratio {short_pct:.0%} (need ≥30%)")
    if long_count < 1:
        fails.append("no cinematic long line (≥12 words)")
    if stddev < 3.0:
        fails.append(f"stddev {stddev:.1f} (need ≥3, flat rhythm)")
    if fails:
        return False, "; ".join(fails)
    return True, f"short={short_pct:.0%} stddev={stddev:.1f}"


# Visual-escalation power keywords (English image_prompts only — that's what
# FLUX consumes). Climax/resolve scenes (last 2 narrative scenes) should hit
# at least one of these to give FLUX the energy gradient the audio is
# building toward. Per plan: SOFT gate, warn-only.
_VISUAL_POWER_KEYWORDS = _re.compile(
    r"\b(lightning|fire|flames?|storm|cosmic|destruction|battlefield|burning|"
    r"ashes?|inferno|tempest|thunder|smoke|war|blood|chaos)\b",
    _re.IGNORECASE
)


def _check_visual_escalation(scenes: list) -> tuple[bool, list]:
    """
    Visual-escalation validator (SOFT gate — warn-only).

    Climax scenes (the last 2 narrative scenes before the outro) should
    contain at least one power keyword in their image_prompt so FLUX renders
    the energy gradient the audio builds toward. Returns (ok, missing_idx_list).
    """
    narrative = scenes[:-1] if len(scenes) > 2 else scenes
    n = len(narrative)
    if n < 3:
        return True, []
    # Check the climax pair (N-2, N-1 in narrative). For a 6-narrative-scene
    # script that's scenes 5 and 6.
    climax_indices = [n - 2, n - 1]
    missing = []
    for idx in climax_indices:
        prompt_text = (narrative[idx].get("image_prompt") or "")
        if not _VISUAL_POWER_KEYWORDS.search(prompt_text):
            missing.append(idx + 1)  # 1-indexed for human readability
    return (not missing), missing


# ════════════════════════════════════════════════════════════════════
# Phase 2/3 stabilization validators (added 2026-05-18)
# ════════════════════════════════════════════════════════════════════
# These enforce the "cost over pain" storytelling principle: the final
# scene must land aftermath (consequence), mid-section must destabilize,
# and no video may turn into a "poetic sadness engine."
# See: C:\Users\rjyot\.claude\plans\have-a-look-into-polished-goblet.md

# Aftermath cues — scene 6 image_prompt MUST contain at least one.
# These describe emotion AFTER destruction (what was left behind), not
# emotion in motion (the destruction itself).
#
# Pattern design (2026-05-19 v2 after smoke #2):
# The regex is intentionally permissive — the LLM paraphrases ("abandoned
# Gandiva bow", "single figure (Bhishma) staring", "wind moving through
# empty cloth") and v1 was rejecting valid aftermath content because the
# exact phrase didn't match. Now we allow optional words/parens between
# the core aftermath nouns so natural variations land.
_AFTERMATH_CUES = _re.compile(
    r"\b("
    r"empty battlefield|"
    # "abandoned weapon" OR specific weapons (Gandiva, bow, sword, chariot, etc.)
    r"abandoned[\s\w()\-,]{0,40}(weapon|bow|sword|gandiva|spear|mace|shield|chariot|throne|armou?r|standard)|"
    r"discarded[\s\w()\-,]{0,40}(weapon|bow|sword|gandiva|crown|throne|armou?r)|"
    # Thrones and crowns
    r"lonely throne|empty throne|throne[\s\w()\-,]{0,30}(beside|abandoned|discarded|empty)|"
    r"crown[\s\w()\-,]{0,30}(beside|on the floor|discarded|fallen|abandoned)|"
    # Trembling hand releasing
    r"trembling[\s\w()\-,]{0,30}releas|hand[\s\w()\-,]{0,15}releas|"
    r"hand[\s\w()\-,]{0,30}letting go|fingers[\s\w()\-,]{0,15}releas|grip loosening|"
    r"hand near[\s\w()\-,]{0,20}not touching|"
    r"releasing the (bow|sword|sacred thread|thread|wrist|reins?)|"
    # Single-figure isolation imagery (permissive — allows parens/clarifications)
    r"single figure[\s\w()\-,]{0,40}(staring|gazing|alone|silhouetted|standing|walking)|"
    r"a lone figure|alone in[\s\w()\-,]{0,20}(vast|empty|silence)|"
    r"figure[\s\w()\-,]{0,30}staring at the (distance|horizon|emptiness|battlefield|sky|void)|"
    # Wind through emptiness (allows "wind moving through ...", "wind blows through ...")
    r"wind[\s\w()\-,]{0,15}through[\s\w()\-,]{0,20}(empty|cloth|halls?|banners?)|"
    r"wind moving[\s\w()\-,]{0,30}empty|wind in empty halls?|"
    # Footprints / aftermath ground details
    r"footprints in (ash|dust|blood|soot|snow)|footsteps fading|"
    # Broken thread imagery
    r"broken (sacred )?thread|snapped (sacred )?thread|"
    # Eyes-stopped-weeping
    r"stopped weeping but cannot look away|eyes that have stopped weeping|"
    r"eyes[\s\w()\-,]{0,30}cannot look away|"
    # Mourning emptiness
    r"nobody left to mourn|no one (left )?to mourn|"
    # Silence-after imagery
    r"silence after the storm|after the silence|after the battle ends|"
    r"haunting[\s\w()\-,]{0,15}(quiet|silence|landscape|emptiness|stillness)|"
    # Empty interiors
    r"empty (hall|courtyard|palace|battlefield|chamber|throne room|landscape)|"
    # Shattered / torn aftermath
    r"shattered weapon|torn flag|torn banner|broken bow lies|"
    # Fallen weapons
    r"weapon[\s\w()\-,]{0,20}fallen|sword fallen|bow[\s\w()\-,]{0,15}(fallen|lies broken|lies abandoned)|"
    # Body-on-arrows-like imagery
    r"bed of arrows|lying on[\s\w()\-,]{0,15}arrows"
    r")\b",
    _re.IGNORECASE
)

# Closure tropes — scene 6 narration MUST NOT contain these. They kill
# emotional residue by giving the viewer closure-pleasure right when we
# want the weight to linger.
_CLOSURE_TROPES = _re.compile(
    r"\b("
    r"rises? triumphant|rose triumphant|dawn of (a |the )?new era|"
    r"glory shone|in glory|"
    r"victorious|victory was|"
    r"hope rekindled|battle won|won the (battle|war)|"
    r"peace restored|blessing of the gods|"
    r"stands? tall|shines bright"
    r")\b|"
    # Hindi closure tropes
    r"नया युग|प्रकाश की किरण|जय हो|"
    r"विजयी हुए|विजयी हुआ|विजय मिली|"
    r"महिमा से|उगता है|उगा सूरज|नया सवेरा",
    _re.IGNORECASE
)

# Allowed scene-6 mood keywords (substring match, case-insensitive).
_AFTERMATH_MOODS_ALLOWED = (
    "haunting", "hollow", "weary", "irreversible",
    "severed", "witnessed", "abandoned", "unresolved",
    "quiet", "lingering", "broken", "empty", "still",
    "weight", "ash", "after", "silenced",
)

# Forbidden scene-6 mood keywords (would kill residue).
_AFTERMATH_MOODS_FORBIDDEN = (
    "inspiring", "triumphant", "dignified", "peaceful",
    "glorious", "majestic", "uplifting", "hopeful",
    "rejoice", "victorious",
)

# Old spectacle keywords that scene 6 MUST NOT use anymore.
# These are kept allowed for scenes 4-5 if they fit, but scene 6 has been
# repurposed from CLIMAX to AFTERMATH (2026-05-18 stabilization).
_SCENE6_SPECTACLE_FORBIDDEN = _re.compile(
    r"\b("
    r"lightning splits?|lightning strikes?|inferno|tempest|"
    r"burning sky|skies? on fire|fiery climax|"
    r"war thunders|cosmic destruction|cosmic apocalypse|"
    r"raging fire|raging inferno|fire consumes|"
    r"thunder roars|storm rages"
    r")\b",
    _re.IGNORECASE
)


def _check_final_scene_aftermath(scenes: list) -> tuple[bool, str]:
    """
    Scene 6 must land aftermath imagery — not spectacle, not triumph.
    Returns (ok, reason). HARD gate.

    Five sub-checks:
      1. image_prompt contains at least ONE aftermath cue
      2. image_prompt does NOT contain old spectacle keywords
      3. mood matches an allowed aftermath keyword
      4. mood does NOT contain a forbidden closure-tone keyword
      5. narration does NOT contain a closure trope ("rises triumphant", etc.)
    """
    if not scenes:
        return False, "no scenes"
    final = scenes[-1]
    img  = (final.get("image_prompt") or "")
    mood = (final.get("mood") or "").lower()
    narr = (final.get("narration") or "")

    if not _AFTERMATH_CUES.search(img):
        return False, (
            "scene 6 image_prompt lacks aftermath cue. Required: one of "
            "'empty battlefield', 'abandoned weapon', 'lonely throne', "
            "'trembling hand releasing', 'single figure staring', "
            "'discarded crown', 'wind through empty cloth', 'footprints "
            "in ash', 'broken thread', 'stopped weeping but cannot look "
            "away', 'hand letting go'. Aftermath = what destruction LEFT "
            "BEHIND, not destruction itself."
        )

    if _SCENE6_SPECTACLE_FORBIDDEN.search(img):
        m = _SCENE6_SPECTACLE_FORBIDDEN.search(img)
        return False, (
            f"scene 6 image_prompt uses forbidden spectacle keyword "
            f"'{m.group(0)}'. Aftermath shows what destruction LEFT, not "
            f"destruction itself. Remove the spectacle phrase and replace "
            f"with one aftermath end-state."
        )

    if any(forbidden in mood for forbidden in _AFTERMATH_MOODS_FORBIDDEN):
        return False, (
            f"scene 6 mood '{mood}' is a closure tone. Use one of: "
            f"haunting-quiet / hollow / weary / irreversible / severed / "
            f"witnessed / abandoned / unresolved instead."
        )

    if not any(allowed in mood for allowed in _AFTERMATH_MOODS_ALLOWED):
        return False, (
            f"scene 6 mood '{mood}' is not an aftermath mood. Must be "
            f"one of: haunting-quiet / hollow / weary / irreversible / "
            f"severed / witnessed / abandoned / unresolved / quiet / "
            f"lingering / broken / still / empty."
        )

    if _CLOSURE_TROPES.search(narr):
        offender = _CLOSURE_TROPES.search(narr).group(0)
        return False, (
            f"scene 6 narration contains closure trope '{offender}'. "
            f"These give the viewer pleasure-resolution right when we "
            f"want weight to linger. Rewrite without it — the bookend "
            f"payoff should land as COST, not as triumph."
        )

    return True, "aftermath dominance landed"


# Dangerous-line destabilization signature patterns (Hindi).
# Permissive matching: substring-find any of the destabilization cores.
# Realization-after-the-fact, irreversibility, finality, permanence,
# temporal closure, severance.
_DANGEROUS_LINE_PATTERNS_HI = _re.compile(
    r"नहीं पता था|"            # didn't know (M)
    r"नहीं पता थी|"            # didn't know (F)
    r"पता ही नहीं|"            # never even knew
    r"नहीं बदल|बदल नहीं|"      # can't change / won't change (either order)
    r"खत्म हो गय|"             # ended / was over (matches गया/गई/गयी/गये)
    r"सब खत्म|"                # everything ended
    r"माफ़ नहीं|कभी माफ़|"     # never forgave
    r"आखिरी बार|"             # last time
    r"अंतिम बार|"              # last/final time (synonym)
    r"वापस नहीं आय|"           # never came back
    r"कभी लौट|लौट नहीं"        # never returned
)

_DANGEROUS_LINE_PATTERNS_EN = _re.compile(
    r"\b(didn'?t know|never knew|"
    r"could not change|couldn'?t change|cannot be changed|"
    r"was over|it ended|all ended|everything ended|"
    r"never forgave|never forgiven|"
    r"last time|final time|for the last time|"
    r"never came back|never returned|did not return)\b",
    _re.IGNORECASE
)


def _check_dangerous_line(scenes: list, language: str) -> tuple[bool, int, str]:
    """
    Scene 3 OR scene 4 narration must contain a destabilization signature.
    Returns (ok, found_scene_1indexed_or_0, reason). HARD gate.

    The dangerous line shifts emotional gravity — it makes the viewer
    realize something irreversible. Position matters: the 24-38s window
    (scenes 3-4 at ~6-8s/scene pacing) is the spot where retention dips
    and a destabilization beat anchors the viewer.
    """
    if len(scenes) < 4:
        return False, 0, f"only {len(scenes)} scenes — dangerous-line needs scenes 3-4"

    candidates = (scenes[2], scenes[3])  # 0-indexed: scenes 3 and 4
    pattern = (_DANGEROUS_LINE_PATTERNS_HI
               if language == "hi"
               else _DANGEROUS_LINE_PATTERNS_EN)

    for offset, scene in enumerate(candidates):
        narr = scene.get("narration") or ""
        if pattern.search(narr):
            return True, 3 + offset, f"dangerous-line found in scene {3 + offset}"

    return False, 0, (
        "no destabilization signature in scenes 3-4. Required patterns "
        "(pick ONE; pattern below MUST appear in scene 3 OR scene 4 "
        "narration): "
        "नहीं पता था / नहीं बदल सकता / खत्म हो गया / माफ़ नहीं किया / "
        "आखिरी बार / वापस नहीं आया (Hindi) — or 'didn't know' / 'could "
        "not change' / 'was over' / 'never forgave' / 'last time' / "
        "'never came back' (English). This is the line that shifts "
        "emotional gravity — irreversibility, finality, realization-"
        "after-the-fact. Without it the mid-video destabilization beat "
        "never lands and the climax/aftermath has nothing to weigh "
        "against. Add ONE such sentence to scene 3 OR scene 4."
    )


# Heavy decorative-poetic symbolism patterns (Part E — anti-over-curation).
# These target "AI-poetic" frames (broken sacred thread, falling petals,
# divine light fading). They do NOT target concrete aftermath end-states
# (empty battlefield, abandoned weapon, lonely throne with crown beside it
# — those are concrete cost imagery, not decorative symbolism).
_HEAVY_SYMBOLIC_PATTERNS = _re.compile(
    r"\b("
    r"broken sacred thread|snapped sacred thread|janeu (snapped|breaking|torn)|"
    r"falling petals|wilted petals|withered petals|petals falling|"
    r"divine light (fading|dimming|extinguished|going out|dying)|"
    r"eternal flame (fading|dimming|going out|extinguishing|dying)|"
    r"sacred flame (fading|dimming|going out|dying)|"
    r"celestial alignment|cosmic alignment|stars aligning|"
    r"wilted lotus|lotus closing|lotus petals scattered|"
    r"darkened sun|eclipsed sun|sun darkening|sun blotted out|"
    r"blood moon|moon eclipsing|crimson moon|"
    r"singing winds|weeping skies|skies wept|"
    r"shattered crystal|broken mirror of (heaven|fate|destiny)|"
    r"divine veil|veil of the gods (fading|tearing)"
    r")\b",
    _re.IGNORECASE
)


def _check_anti_over_curation(scenes: list) -> tuple[bool, int, list]:
    """
    SOFT gate (warn-only). Count scenes whose image_prompt uses heavy
    decorative-poetic symbolism. At most 1 such scene per video; more
    turns the pipeline into a "poetic sadness engine" per 2026-05-18 user
    feedback. Returns (ok, count, scene_1indexed_indices).

    Devastation is NOT a poem — real devastation looks plain, awkward,
    physically uncomfortable. A trembling hand outperforms a falling
    petal. Symbolism should be RARE so the one time it lands, it lands.
    """
    hits = []
    for i, s in enumerate(scenes, start=1):
        img = s.get("image_prompt") or ""
        if _HEAVY_SYMBOLIC_PATTERNS.search(img):
            hits.append(i)
    return (len(hits) <= 1), len(hits), hits


# Krishna direct-address mode requires first-person markers (मैं/मैंने) or
# vocative markers (तुम/पार्थ/देखो/सुनो) in the narration. If a scene drops to
# third-person ("कृष्ण ने अर्जुन से कहा..."), the immersion breaks and the
# whole format reads like a regular Mahabharata story instead of a divine
# monologue.
_FIRST_PERSON_MARKERS = _re.compile(
    r"मैं|मैंने|तुम|तुम्हें|तुम्हारे|तुम्हारी|तुम्हारा|पार्थ|अर्जुन|देखो|सुनो"
)


def _check_first_person(scenes: list, min_hits: int = 3) -> tuple:
    """
    For Krishna direct-address scripts: returns (ok, hits, total) where hits
    is the count of scenes containing at least one first-person/vocative
    marker. min_hits=3 of 5 scenes (60%) keeps the format honest while
    tolerating one or two narrative-bridge scenes that lean exposition.
    """
    total = len(scenes)
    hits = 0
    for s in scenes:
        text = s.get("narration") or ""
        if _FIRST_PERSON_MARKERS.search(text):
            hits += 1
    return (hits >= min_hits), hits, total


def _check_repetition(scenes: list, max_repeats: int = 2, topic: str = "") -> tuple:
    """
    Inspect every CONTENT word across all scene narrations. Returns
    (ok, offenders) where offenders is words appearing > max_repeats times.

    Skips: stopwords, character names, place names, and any word from the
    topic string (the topic protagonist legitimately repeats).
    """
    import re

    # Build a per-call ignore set: stopwords + character names + topic words
    ignore = set(_REPETITION_STOPWORDS) | {n.lower() for n in _CHARACTER_NAMES}
    if topic:
        topic_tokens = re.split(r"[\s,.!?;:'\"()\-—]+", topic.lower())
        ignore |= {t for t in topic_tokens if len(t) > 2}

    counts = {}
    for s in scenes:
        text = (s.get("narration") or "").lower()
        # Split on whitespace + punctuation (Latin AND Devanagari danda/double-danda)
        tokens = re.split(r"[\s।॥,.!?;:'\"()\[\]\-—–…]+", text)
        for w in tokens:
            w = w.strip()
            if len(w) <= 2 or w in ignore:
                continue
            counts[w] = counts.get(w, 0) + 1
    offenders = [(w, n) for w, n in counts.items() if n > max_repeats]
    offenders.sort(key=lambda x: -x[1])
    return (len(offenders) == 0), offenders


def _check_character_names(scenes: list) -> tuple:
    """
    Verify every Mahabharata scene's image_prompt mentions at least one
    recognized character name from `assets/characters.json`. Returns
    (ok, missing_scene_indices) where missing_scene_indices is 1-based.

    Required for visual consistency: `_inject_characters` in image_generator
    is a substring matcher — if the image_prompt uses generic descriptors
    ("the divine lord" / "the dark-skinned god") instead of named characters
    ("Krishna" / "Karna"), the injector skips, character visual descriptions
    are never appended to the FLUX prompt, and renders drift visually
    (Krishna rendered as Shiva-like ascetic, etc.).
    """
    # Import lazily to avoid circular dependency (image_generator may import this file)
    try:
        from pipeline.image_generator import _KNOWN_NAMES
        recognized = {n.lower() for n in _KNOWN_NAMES}
        # Augment with the characters actually in characters.json
        from pipeline.image_generator import _CHARACTERS
        recognized |= {n.lower() for n in _CHARACTERS}
    except Exception:
        # Fallback list — should match image_generator._KNOWN_NAMES + characters.json keys
        recognized = {
            "krishna", "arjuna", "bhishma", "karna", "draupadi", "yudhishthira",
            "bhima", "bheema", "nakula", "sahadeva", "duryodhana", "dushasana",
            "drona", "ashwatthama", "gandhari", "dhritarashtra", "vidura",
            "kunti", "shakuni", "abhimanyu", "subhadra", "ghatotkacha",
            "jayadratha", "shikhandi", "amba", "satyavati", "devavrata",
            "parashurama", "ekalavya", "sanjaya", "balarama", "vyasa",
        }

    missing = []
    for i, scene in enumerate(scenes):
        prompt_lower = (scene.get("image_prompt") or "").lower()
        if not any(name in prompt_lower for name in recognized):
            missing.append(i + 1)
    return (len(missing) == 0), missing


# ── Bookend / engagement / monotony validators (Fix 2 / 3 / 4) ──────────
# These were added after analyzing 4 videos that shipped on 2026-05-13 and
# still showed the same recurring failures the prompt rules were meant to
# fix. The pattern: prompt rules alone get ~30-50% LLM compliance; rules
# that round-trip violations back into a retry attempt get ~95% compliance.

# Synonym table for bookend matching — when Scene 1's central noun is one
# of these, the final scene may use any sibling and still count as a match.
# Seeded from Bhishma's reference video (the working bookend pattern):
# हुक: "एक प्रतिज्ञा" → closure: "एक वचन". Both legitimate.
_BOOKEND_SYNONYMS = [
    {"प्रतिज्ञा", "वचन", "कसम", "शपथ", "vow", "oath", "promise"},
    {"शाप", "अभिशाप", "श्राप", "curse"},
    {"युद्ध", "रण", "संग्राम", "war", "battle"},
    {"धर्म", "कर्म", "dharma", "duty"},
    {"बलिदान", "त्याग", "sacrifice", "renunciation"},
    {"विवाह", "स्वयंवर", "marriage", "wedding"},
    {"वनवास", "निर्वासन", "exile", "banishment"},
    {"न्याय", "अन्याय", "justice", "injustice"},
    {"क्रोध", "रोष", "wrath", "anger", "fury"},
    {"प्रेम", "प्यार", "love"},
    {"मृत्यु", "मौत", "death"},
    {"विश्वासघात", "धोखा", "betrayal", "deceit"},
    {"वंश", "कुल", "dynasty", "clan", "lineage"},
    {"महाभारत", "कुरुक्षेत्र", "mahabharata", "kurukshetra"},
    {"रहस्य", "गुप्त", "secret", "mystery"},
    {"सत्य", "truth"},
    {"प्रलय", "विनाश", "destruction", "apocalypse"},
    {"ज्योति", "प्रकाश", "light"},
    {"अंधकार", "तम", "darkness"},
]


def _content_nouns_from(text: str, ignore: set, top_k: int = 4) -> list:
    """Extract the top-k longest content-token candidates from a narration."""
    import re as _re
    toks = _re.split(r"[\s।॥,.!?;:'\"()\[\]\-—–…]+", text.lower())
    seen = set()
    out = []
    for w in toks:
        w = w.strip()
        if len(w) < 3 or w in ignore or w in seen:
            continue
        seen.add(w)
        out.append(w)
    # Prefer longer tokens — they tend to be more specific / content-bearing
    out.sort(key=lambda x: -len(x))
    return out[:top_k]


def _check_bookend(scenes: list, topic: str = "") -> tuple:
    """
    Verify the FINAL scene's narration echoes Scene 1's central noun (or a
    recognized synonym). Returns (ok, scene1_central_noun_or_None).

    When scene 1 says "एक प्रतिज्ञा ने कुरुवंश को शाप दिया" and the final
    scene closes with "एक वचन ने कुरुवंश को... वंचित कर दिया" — that's a
    bookend; the noun "प्रतिज्ञा" was echoed via synonym "वचन". When the
    final scene closes with "ऐसा है धर्म की महिमा" (a generic moral that
    doesn't echo "प्रतिज्ञा" or its synonyms), the bookend is broken.

    Mirrors the proven pattern from `_check_repetition` and
    `_check_character_names`: post-hoc check, round-trip the failure back
    into the next prompt attempt.
    """
    if len(scenes) < 3:
        return True, None
    ignore = set(_REPETITION_STOPWORDS) | {n.lower() for n in _CHARACTER_NAMES}
    if topic:
        import re as _re
        for t in _re.split(r"[\s,.!?;:'\"()\-—]+", topic.lower()):
            if len(t) > 2:
                ignore.add(t)

    scene1_text = (scenes[0].get("narration") or "")
    final_text  = (scenes[-1].get("narration") or "").lower()

    candidates = _content_nouns_from(scene1_text, ignore, top_k=5)
    if not candidates:
        return True, None  # nothing to check — trust the LLM

    # Build expanded match set for each candidate (the candidate + its synonyms)
    def expand(noun: str) -> set:
        out = {noun}
        for grp in _BOOKEND_SYNONYMS:
            if noun in grp or any(noun in g.lower() for g in grp):
                out |= {g.lower() for g in grp}
        return out

    # Pick the FIRST candidate that we can attest is centrally-thematic
    # (here: just use the first/most-specific noun; that's our hook subject)
    central = candidates[0]
    match_set = expand(central)

    if any(m in final_text for m in match_set):
        return True, central
    return False, central


# Hindi sensory anchor vocabulary — sounds / sights / touch / smell / breath.
# Used by _check_engagement_density. Curated from cinematic Mahabharata
# narrations; expanded to cover the science-curiosity register for WhatIf.
_SENSORY_TOKENS_HI = {
    # sound
    "शंख", "गूंज", "ध्वनि", "आवाज़", "आवाज", "स्वर", "चीख", "घोष",
    # sight / light
    "दीप", "ज्योति", "रोशनी", "प्रकाश", "किरण", "छाया", "धुआं", "धुआँ",
    # body / physical
    "आंख", "आँख", "हाथ", "उंगली", "उँगली", "पैर", "कदम", "सांस", "साँस",
    "माथा", "होंठ", "कान", "रक्त", "खून", "पसीना", "अश्रु", "आँसू",
    # texture / atmosphere
    "धूल", "गर्द", "कांप", "काँप", "गूँज", "सुगंध", "खुशबू",
    # objects in close-up (mythological)
    "धनुष", "तीर", "मुद्रा", "मुकुट", "हथेली", "पैरों",
    # elemental
    "अग्नि", "पवन", "जल", "धरती",
}

_DIALOGUE_MARKERS_HI = (
    "बोले", "बोला", "बोली", "कहा", "कही", "कहती", "कहते", "कहता",
    "पूछा", "पूछती", "बोले —", "कहा —", "कहते हैं", "बोले:",
)


def _check_engagement_density(scenes: list, language: str = "hi") -> tuple:
    """
    Enforce the engagement floor introduced by the SHOW-DONT-TELL rule:
    every script must contain ≥2 dialogue markers and ≥4 sensory tokens
    across all scene narrations combined. Below either threshold means
    the script reads as summary prose — boring.

    Returns (ok, dialogue_count, sensory_count).
    """
    if language != "hi":
        # English / dual narrations would need a parallel vocab set; for
        # now only Hindi narration is validated (single biggest impact
        # surface).
        return True, 0, 0

    dialogue = 0
    sensory  = 0
    for s in scenes:
        text = (s.get("narration") or "")
        for marker in _DIALOGUE_MARKERS_HI:
            dialogue += text.count(marker)
        # also count dash-em-quoted dialogue ("कृष्ण — '...'")
        if " — \"" in text or " — '" in text or " — “" in text:
            dialogue += text.count(" — \"") + text.count(" — '") + text.count(" — “")
        for tok in _SENSORY_TOKENS_HI:
            sensory += text.count(tok)

    ok = (dialogue >= 2 and sensory >= 4)
    return ok, dialogue, sensory


def _check_ending_monotony(scenes: list, language: str = "hi", max_ratio: float = 0.40) -> tuple:
    """
    Detect the dominant sentence-ending pattern across all narrations and
    fail when any single pattern exceeds `max_ratio` of all sentences.
    Generalizes the older `_check_past_aux_tic` which only caught था/थी/थे/थीं —
    misses present-tense `हैं` chains and future-conditional `जाएगा` chains
    that produce the same listy, boring rhythm.

    Returns (ok, dominant_pattern, ratio, hits, total).
    """
    if language != "hi":
        return True, None, 0.0, 0, 0
    import re as _re
    sentences = []
    for s in scenes:
        text = (s.get("narration") or "").strip()
        if not text:
            continue
        for snt in _re.split(r"[।!?\.]+", text):
            snt = snt.strip()
            if snt:
                sentences.append(snt)
    if not sentences:
        return True, None, 0.0, 0, 0

    # Extract the last "word" (last whitespace-separated token, stripped of
    # punctuation/maatras-stripping isn't worth doing — we just need a key).
    from collections import Counter
    endings = Counter()
    for snt in sentences:
        toks = snt.rsplit(None, 1)
        last = toks[-1] if toks else snt
        last = last.strip("।.!?,;:'\"()-—")
        if last:
            endings[last] += 1

    if not endings:
        return True, None, 0.0, 0, len(sentences)

    dom_word, dom_count = endings.most_common(1)[0]
    ratio = dom_count / len(sentences)
    return (ratio <= max_ratio), dom_word, ratio, dom_count, len(sentences)


def _trim_narration(text: str, max_words: int = 45) -> str:
    """Hard-cap narration at max_words, ending at the last complete sentence.
    Long-form videos (60-90s, 5-6 scenes) need 25-40 words per scene; this
    cap is now generous so natural sentence endings survive."""
    import re
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    # Try to end at last sentence boundary (। ! ? .)
    match = re.search(r'^(.*[।!?\.])', truncated)
    if match:
        return match.group(1).strip()
    return truncated + "..."


def _repair_truncated_json(raw: str) -> str:
    """
    Best-effort repair for LLM JSON that got cut off mid-output (typically
    because the model hit its max_output_tokens cap before closing all brackets).

    Strategy — walk the string once tracking quote state and bracket stack, then:
      1. If we ended inside a string literal, close it with `"`.
      2. If the open-bracket stack ends with an object that has a dangling key
         (`"foo":` with no value), append `null` so the object remains valid.
      3. If the last meaningful char is a trailing `,`, drop it.
      4. Close every remaining open bracket in correct nested order.

    This salvages partial output instead of failing the whole pipeline. Caller
    must still validate that critical fields (title, scenes) are present — a
    description that arrives truncated will be a short partial string, not
    None, so downstream code can detect and re-prompt or substitute.
    """
    stack = []          # 'object' | 'array' | 'string'
    escape_next = False
    last_non_ws = -1    # index of last non-whitespace char outside a string
    expect_value = False  # True right after `:`, meaning a value is owed

    for i, ch in enumerate(raw):
        if stack and stack[-1] == "string":
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                stack.pop()
                last_non_ws = i
            # other chars inside a string are content; don't update last_non_ws
        else:
            if ch.isspace():
                continue
            last_non_ws = i
            if ch == '"':
                stack.append("string")
                # Entering a string commits to a value — clear the pending-value
                # flag so that a string that never closes won't trigger a stray
                # `null` append in step (2) after we close the string in step (1).
                expect_value = False
            elif ch == "{":
                stack.append("object")
                expect_value = False
            elif ch == "[":
                stack.append("array")
                expect_value = False
            elif ch == "}":
                if stack and stack[-1] == "object":
                    stack.pop()
                expect_value = False
            elif ch == "]":
                if stack and stack[-1] == "array":
                    stack.pop()
                expect_value = False
            elif ch == ":":
                expect_value = True
            elif ch == ",":
                expect_value = False
            else:
                # Bare value char (digit / letter / etc.) — we got a value, so
                # the colon's debt is paid.
                expect_value = False

    repaired = raw

    # (1) Close an open string.
    if stack and stack[-1] == "string":
        repaired += '"'
        stack.pop()

    # (2) If a colon was followed only by whitespace + EOF (the closed string
    # *was* the key), append `null` to give the key a value.
    # We detect this by looking at the chars after our string close. The
    # `expect_value=True` flag is the cleanest signal.
    if expect_value:
        repaired += " null"
        expect_value = False

    # (3) Drop a dangling comma before we close brackets.
    import re as _re
    repaired = _re.sub(r",\s*$", "", repaired)

    # (4) Close brackets in reverse-nested order.
    for kind in reversed(stack):
        if kind == "object":
            repaired += "}"
        elif kind == "array":
            repaired += "]"

    return repaired


def _parse_llm_json(raw: str) -> dict:
    """
    Robust JSON parser for LLM output.
    Handles: unescaped newlines inside strings, trailing commas, stray control
    chars, AND truncated-mid-output responses (close-the-brackets repair pass).
    """
    import re

    # First try: parse as-is
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fix 1: remove trailing commas before ] or }
    cleaned = re.sub(r",\s*([\]}])", r"\1", raw)

    # Fix 2: replace literal newlines/tabs inside JSON strings with escaped versions
    # Walk char by char to only replace newlines that are inside string literals
    result = []
    in_string = False
    escape_next = False
    for ch in cleaned:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\":
            result.append(ch)
            escape_next = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            pass  # strip carriage returns inside strings
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)

    cleaned = "".join(result)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fix 3: "Extra data" salvage — LLM emitted valid JSON followed by
    # additional text (markdown commentary, repeated/explanation JSON, prose).
    # Observed in production on 2026-05-14: Gemini 2.5 Pro for Mahabharata
    # script returned a valid JSON object then ~7600 chars of trailing text,
    # tripping JSONDecodeError("Extra data") at char 7633. raw_decode parses
    # the leading JSON value and tells us where it ends — anything after is
    # ignored. Strip leading whitespace + any leading markdown fence remnants
    # first so the decoder sees the opening "{" as char 0.
    try:
        trimmed = cleaned.lstrip()
        # Skip past common leading-noise prefixes ("```json\n{...", "Here is
        # the JSON:\n{...") by jumping to the first "{" or "[".
        first_brace = -1
        for i, c in enumerate(trimmed):
            if c in ("{", "["):
                first_brace = i
                break
        if first_brace > 0:
            trimmed = trimmed[first_brace:]
        decoder = json.JSONDecoder()
        parsed, end_idx = decoder.raw_decode(trimmed)
        trailing = trimmed[end_idx:].strip()
        if trailing:
            print(f"    [warn] LLM JSON had {len(trailing)} chars trailing extra data; trimmed")
        return parsed
    except json.JSONDecodeError:
        pass

    # Fix 4: last resort — repair a truncated-mid-output response by closing
    # any open string + brackets. Salvages partial scripts so the pipeline can
    # at least proceed to validation (which will detect missing/short fields
    # and re-prompt instead of crashing the whole job).
    try:
        repaired = _repair_truncated_json(cleaned)
        parsed = json.loads(repaired)
        print(f"    [warn] LLM JSON was truncated; salvaged via repair pass")
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM JSON after cleaning + repair: {e}\n{cleaned[:400]}")


def _generate_story_outline(topic: str) -> list:
    """
    Pass 1 of two-pass script generation.

    Asks the LLM to commit to 6 SPECIFIC story beats — character names,
    locations, and concrete actions — before any narration is written.
    Forcing factual scaffolding upfront stops the LLM from drifting to
    abstract filler ("a vow that changed everything") in the dramatization
    step, because it has to USE the specific details it just committed to.

    Returns list of dicts: [{"characters": [...], "location": "...", "action": "..."}, ...]
    Returns [] on failure (caller falls back to single-pass prompt).
    """
    outline_prompt = f"""
You are a Mahabharata historian. Outline the story of this incident as 6 SPECIFIC dramatic beats:

INCIDENT: "{topic}"

For each beat, provide:
- characters: list of specific character names appearing in that beat (e.g. "राजा शांतनु", "Devavrata", "Shakuni") — NOT pronouns, NOT generic "the king"
- location: the specific place or setting (e.g. "यमुना नदी का तट", "Hastinapura royal court", "Kurukshetra battlefield")
- action: ONE concrete event that happens in this beat — what physically occurs, in 10-15 English words. NOT a feeling, NOT a moral, NOT a meta-statement. A physical action.

Story arc:
- Beat 1: HOOK — the most dramatic / mysterious moment that opens the story
- Beats 2-3: setup and rising tension — establish characters and conflict
- Beat 4-5: climax — the dramatic high point
- Beat 6: resolution — how it ends, what changes

Return ONLY this JSON, no markdown, no preamble:
{{
  "beats": [
    {{"characters": ["..."], "location": "...", "action": "..."}},
    ...
  ]
}}

Each beat MUST be a different event. No two beats may describe the same action.
"""
    try:
        raw = _call_llm(outline_prompt)
        start = raw.find("{")
        end   = raw.rfind("}")
        if start == -1 or end == -1:
            return []
        data = _parse_llm_json(raw[start:end + 1])
        beats = data.get("beats", [])
        # Sanity check: at least 5 beats with non-empty fields
        if len(beats) < 5:
            return []
        for b in beats:
            if not (b.get("characters") and b.get("location") and b.get("action")):
                return []
        return beats[:6]
    except Exception as e:
        print(f"    [outline] failed: {e}")
        return []


def _format_outline_for_prompt(beats: list) -> str:
    """Render the outline as a readable scaffold the dramatization pass quotes back."""
    lines = []
    for i, b in enumerate(beats, 1):
        chars = ", ".join(b.get("characters", []))
        lines.append(
            f"  Beat {i}:\n"
            f"    - Characters: {chars}\n"
            f"    - Location:   {b.get('location', '')}\n"
            f"    - Action:     {b.get('action', '')}"
        )
    return "\n".join(lines)


# ── What If series — science/curiosity hypotheticals ────────────────────────
# Completely separate flow from Mahabharata: different topic pool, prompt
# template, tags, and (when dual_language=True) outputs both English and Hindi
# narration per scene in a single LLM call so visuals can be shared across
# both language renders.

_WHATIF_VISUAL_STYLES = {
    "photoreal-3d":     "photorealistic 3D render, octane, cinematic lighting, sharp 8K detail, scientific accuracy",
    "nature-doc":       "BBC nature documentary cinematography, golden hour, telephoto lens, naturalistic lighting",
    "sci-fi-cinematic": "Denis Villeneuve sci-fi aesthetic, Roger Deakins lighting, anamorphic widescreen feel, atmospheric haze",
    "illustrated":      "polished concept art digital illustration, ArtStation trending, painterly detail",
}


def _generate_whatif_script(forced_topic: str = None, dual_language: bool = True) -> dict:
    """
    Generates a 'What If' thought-experiment script.

    When dual_language=True (default for production), returns scenes with BOTH
    English `narration` and Hindi `narration_hi` so the orchestrator can render
    two videos sharing identical visuals.
    """
    if forced_topic:
        topic = forced_topic
    else:
        # Last-resort fallback when the topic queue is empty AND the LLM
        # auto-replenish in topic_manager has failed (rate-limited or
        # offline). Filter STORY_TOPICS_WHATIF against the recent-uploads
        # avoidance window so we never re-publish a topic from the last
        # ~2 months even on the failure path.
        from pipeline.topic_manager import _read_recent_topics, _WHATIF_RECENT_HISTORY
        recent = {t.lower() for t in _read_recent_topics("whatif", limit=_WHATIF_RECENT_HISTORY)}
        eligible = [t for t in STORY_TOPICS_WHATIF if t.lower() not in recent]
        if not eligible:
            # All static topics used recently — accept any (oldest-recent
            # cycles back) rather than fail the run.
            eligible = list(STORY_TOPICS_WHATIF)
            print("    [whatif] all built-in topics used in last 2 months — allowing repeat")
        topic = random.choice(eligible)

    if dual_language:
        narration_block = (
            '          "narration": "25-35 words in clear, natural ENGLISH — curious, vivid, present-tense",\n'
            '          "narration_hi": "25-35 words in SIMPLE EVERYDAY spoken HINDI (Devanagari script) — the kind a young person speaks at home, NOT literary or Sanskritized Hindi. Common loanwords (planet, gravity, magnetic, satellite, signal, GPS, climate, ocean, virus, AI, etc.) MUST stay in their English form written in Devanagari (e.g. प्लैनेट, ग्रैविटी, मैग्नेटिक, सैटेलाइट, सिग्नल, जीपीएस, क्लाइमेट, ओशन, वायरस, ए-आई). DO NOT use words like ध्रुव, चुंबकीय, उपग्रह, गुरुत्वाकर्षण, वायुमंडल — use the everyday English-origin word in Devanagari instead. Same idea as English narration but spoken naturally — NOT a literal translation.",\n'
        )
        lang_note = "Each scene has BOTH English (narration) AND Hindi (narration_hi) versions of the same idea."
    else:
        narration_block = (
            '          "narration": "25-35 words in clear, natural ENGLISH — curious, vivid, present-tense",\n'
        )
        lang_note = "Narration is in English."

    style_options = ", ".join(f'"{k}"' for k in _WHATIF_VISUAL_STYLES.keys())

    prompt = f"""
You are a science communicator writing a 60-90 second "What If" thought-experiment for YouTube Shorts.

TOPIC: "{topic}"

TASK: Create a vertical (9:16) video script with EXACTLY 4 OR 5 scenes that imagines this hypothetical scenario plausibly. Target video length: 45-60 seconds — this is the science-niche Shorts retention sweet spot, not 60-90s.

VOICE & TONE:
- Curious, wonder-driven, "imagine this for a moment" energy
- Plausible — speculate from real science, nature, history, or physics. NOT fantasy.
- Cinematic — paint vivid mental images the viewer can see
- NOT devotional, NOT mythological, NOT epic-poem style — this is curiosity/science content
- Conversational, like a smart friend explaining something fascinating

VISUAL STYLE: Pick ONE that fits this topic best (output it in the JSON):
  {style_options}
  Every scene's image_prompt MUST be visually consistent with that one chosen style.

LANGUAGE: {lang_note}

═══════════════════════════════════════════════════════════════
STRUCTURE — RETENTION ON SHORTS
═══════════════════════════════════════════════════════════════
Scene 1 — HOOK (the first 1.5 seconds decide if the viewer swipes):
   CONSEQUENCE FIRST, scenario second. Open with a vivid concrete consequence
   that punches the viewer, THEN reveal the scenario in the same scene.
   Lead with sensory or visual imagery — not abstract framing.
   Examples (CONSEQUENCE → SCENARIO):
     "Your phone dies. GPS gone. Birds crash into buildings.
      This is Earth — six months after the magnetic poles flip."
     "The sky turns blood red. Plants stop growing within weeks.
      This is what happens if the sun's output drops by just 5%."
     "Eight billion people. Gone overnight. Streetlights still on,
      subway trains still running. This is one day after every human vanishes."
   DO NOT open with: "Imagine...", "What if...", "Have you ever wondered...",
   "In this video...", "Today we explore..." — these are all instant swipe triggers.
   Lead with the consequence as if it has already happened — present tense.

Scenes 2-3 — SETUP & ESCALATION:
   Walk through the immediate consequences in vivid, concrete detail.
   Each scene reveals a new layer the viewer didn't expect.

Peak scene (penultimate) — PEAK CONSEQUENCE:
   The single most striking implication. The "wait, what?" moment.

Final scene — RESOLUTION + REFLECTION:
   Land it. Tie back to something the viewer can feel about their own world.
   This scene MAY end with closure — every other scene must end with a hook
   forward (a question, a "...but", or an unresolved threat).

═══════════════════════════════════════════════════════════════
NARRATIVE BOOKEND — THE FINAL SCENE MUST CLOSE THE HOOK
═══════════════════════════════════════════════════════════════
Scene 1's hook poses a specific consequence with named nouns and a
specific time/place anchor. The FINAL scene MUST echo it back — same
central noun, same time/place anchor, payoff delivered.

Without a bookend the video ENDS but does not RESOLVE — the viewer
hears the closer but doesn't feel anything has been answered. This is
the single biggest narrative-cohesion lever between "the video felt
complete" and "the video just stopped."

GOOD (bookend — opens a question, closes with payoff):
  Hook (Scene 1):
    "Your phone dies. GPS gone. Birds crash into buildings. This is
     Earth — six months after the magnetic poles flip."
    → Central noun: "magnetic poles flip"
    → Time anchor: "six months"
    → Claim: dying systems

  Closure (Final scene):
    "Six months from a single magnetic flip — no GPS, no migrations,
     no normal sky. Earth still spinning, but nothing on it the same."
    → Echoes "six months" + "magnetic flip"
    → Payoff: nothing the same

BAD (no bookend — generic moralizing closer):
  Hook:    "Lava buries cities. Ash chokes the sky."
  Closure: "Such are the wonders of geology, reminding us of nature's power."
  (Generic moral. Doesn't echo "lava" / "ash" / any specific anchor.
   Viewer hears it as a separate sentence, not as resolution.)

═══════════════════════════════════════════════════════════════
SHOW, DON'T TELL — THIS IS THE BORING-PROOF RULE
═══════════════════════════════════════════════════════════════
Boring science scripts default to ENUMERATING ("the climate changes,
ecosystems collapse, species go extinct"). Engaging science scripts
SHOW one concrete image at a time, anchored in a human-scale detail
the viewer can FEEL.

Every non-final scene MUST contain at least TWO of:
  1. A SENSORY DETAIL — what you'd hear / smell / feel
     ("the air tastes like burnt copper", "skin prickles before the
      shockwave hits", "a low hum that wasn't there yesterday")
  2. A HUMAN-SCALE ANCHOR — what one person would experience
     ("a farmer in Kansas wakes to find the dawn 90 minutes late",
      "your phone battery dies and never comes back")
  3. A SPECIFIC NUMBER + UNIT — not "many" or "lots" but
     "847 cubic kilometers of ash", "7 minutes after sunrise",
     "every cell in a human body weighs 2% more"
  4. A NAMED OBJECT IN CLOSE-UP — not "the city" but
     "a single traffic light blinking red over an empty intersection",
     "the dust on a windowsill nobody's wiped in three weeks"

BAD (telling — enumerates without anchoring):
  "Volcanic eruptions deeply impact human societies with widespread
   destruction, displacement, and loss of life."

GOOD (showing — two anchors in one beat):
  "In Reykjavík, the streetlights stay on at noon. A child draws her
   finger through the ash on a parked car — it's been three days."

═══════════════════════════════════════════════════════════════
MID-SCENE TURN — STOPS SCROLL-AT-SCENE-2
═══════════════════════════════════════════════════════════════
At least 2 non-final scenes MUST contain a TURN at the halfway mark.
The scene starts heading one direction, then pivots. This is what
makes a viewer feel "wait, what?" inside a scene — not just between
scenes.

GOOD turn:
  "The eruption ends. Birds return to the sky. — Then the second
   one starts, 1,200 kilometers away."

BAD (no turn — single linear beat):
  "The eruption ends and the climate begins to recover over decades."

═══════════════════════════════════════════════════════════════
CONTENT QUALITY
═══════════════════════════════════════════════════════════════
- Every sentence must contain a NEW concrete detail (a specific number, place, organism, mechanism, timescale).
- NO vague abstractions ("things would change", "consequences would follow").
- Reference real science where applicable — actual species, distances, timescales, physical laws.
- 2-3 short sentences per scene for natural breathing pauses.

═══════════════════════════════════════════════════════════════
NARRATION LENGTH — CRITICAL (HARD-ENFORCED)
═══════════════════════════════════════════════════════════════
EACH scene's narration MUST be 25-35 words. This applies to BOTH the
English `narration` field AND the Hindi `narration_hi` field per scene.
Aim for 28-32 words per scene as the sweet spot.

NEVER write fewer than 25 words per scene. Anything under 20 words is
unusable — it produces a 25-second video that nobody watches. Length is
not optional.

At natural narration pace, 25-35 words = ~10-12 seconds spoken per scene.
5 scenes × 30 words ≈ 150 words ≈ 50-60 seconds of audio. THAT is the
target video length — the science-Shorts retention sweet spot.

Bad scene example (5 words — DO NOT WRITE):
  "Humans vanish. Cities go silent."
Good scene example (30 words — WRITE LIKE THIS, consequence-first):
  "Eight billion people, gone overnight. Streetlights stay on. Subway
  trains roll into empty stations. The cat watches the door, waiting.
  This is one day after humans vanish."

═══════════════════════════════════════════════════════════════
OUTPUT — return ONLY valid JSON, no markdown fences, no preamble:
═══════════════════════════════════════════════════════════════
{{
  "title": "What If <vivid specific phrasing> | <Hindi version क्या होगा अगर...> — under 60 chars total, no hashtags",
  "description": "Hook sentence under 90 chars expanding the title's specific question with one stunning detail.\\n\\n#Shorts #WhatIf #Science #ScienceShorts #ThoughtExperiment\\n\\n100-150 words about the thought experiment, weaving in real-science anchor points (specific numbers, named phenomena). Build curiosity. Don't fully resolve the answer in the description.\\n\\n#Shorts #WhatIf #Science #ScienceShorts #ThoughtExperiment #Curiosity #ScienceFacts #Hypothetical #SpeculativeScience #FutureEarth #क्याहोगाअगर #विज्ञान #IndianScienceShorts #MindBlowing #ScienceExplained #trending",
  "tags": ["topic-specific long-tail keyword 1 (e.g. 'humans disappear from earth')","topic-specific long-tail keyword 2","named phenomenon if relevant (e.g. 'gravity decrease')","what if","hypothetical","thought experiment","science","curiosity","speculative","science shorts","what if scenarios","science what if","alternate reality","future earth","mind blowing","क्या होगा अगर","विज्ञान","कल्पना","trending shorts"],
  "visual_style": "<one of: {style_options}>",
  "scenes": [
        {{
{narration_block}          "image_prompt": "Detailed English prompt — portrait 9:16 composition, specific subjects, environment, lighting. Must visually match the chosen visual_style.",
          "video_prompt": "Cinematic 5-second shot in English — subjects in subtle motion, camera movement, lighting. Vertical 9:16. Matches visual_style.",
          "mood": "3-6 word English emotional tone phrase"
        }}
  ],
  "thumbnail_prompt": "Bold curiosity-driven thumbnail in the chosen visual_style — vivid, specific, attention-stopping at small size"
}}

HARD RULES:
- Title: under 60 chars total, MUST start with "What If" in the English half,
  and include a Hindi half "क्या होगा अगर ..." separated by `|`. No hashtags.
- Description MUST follow the 3-block structure: hook line ≤90 chars,
  blank line, 5 inline hashtags, blank line, body, blank line, full
  hashtag block (high-volume hashtags first).
- Tags MUST include topic-specific long-tail keywords (the named
  phenomenon, the specific scenario phrasing) on top of the generic
  what-if/science fallbacks.
- visual_style: MUST be exactly one of the allowed values
- EXACTLY 4 OR 5 scenes (4 preferred for tight pacing, 5 for richer topics)
- Narration MUST NOT contain URLs, hashtags, @mentions, or social-media text
- image_prompt, video_prompt, thumbnail_prompt all in English
- NO Mahabharata characters, gods, or mythology — this is science/curiosity content
"""

    # Retry up to 3 times if the LLM produces too-short narrations.
    # Average target = 25-40 words/scene; we accept >=22 words/scene as the
    # floor (gives ~50s+ of audio across 5-6 scenes).
    data = None
    last_avg_words = 0.0
    last_n_scenes  = 0
    for attempt in range(3):
        full_prompt = prompt
        if attempt > 0:
            full_prompt += (
                f"\n\nCRITICAL REMINDER: Your previous response had narrations averaging "
                f"only {last_avg_words:.1f} words per scene across {last_n_scenes} scenes. "
                f"That's a stub video, not the 45-60 second Short the prompt asked for. "
                f"EVERY scene's `narration` AND `narration_hi` MUST be 25-35 words — no "
                f"exceptions. Do NOT write 1-sentence scenes. Each scene needs 2-3 full "
                f"sentences with concrete sensory and scientific detail. Rewrite."
            )

        # quality="best" — WhatIf bilingual narration needs Gemini Pro's
        # creative-prose ability to avoid the summary-prose register that
        # Flash produces on long-form Hindi.
        raw = _call_llm(full_prompt, quality="best")
        start = raw.find("{")
        end   = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in WhatIf LLM response:\n{raw[:300]}")
        data = _parse_llm_json(raw[start:end + 1])

        # Hard-trim narrations (caps at 45 words — we under-shoot, not over-shoot)
        for scene in data.get("scenes", []):
            if "narration" in scene:
                scene["narration"] = _trim_narration(scene["narration"])
            if "narration_hi" in scene:
                scene["narration_hi"] = _trim_narration(scene["narration_hi"])

        # Validate visual_style; fall back to photoreal-3d if LLM picks something off-list
        vs = data.get("visual_style", "")
        if vs not in _WHATIF_VISUAL_STYLES:
            print(f"    [warn] visual_style '{vs}' not recognized — defaulting to photoreal-3d")
            data["visual_style"] = "photoreal-3d"

        n_scenes = len(data.get("scenes", []))
        word_avg = (sum(len(s.get("narration", "").split()) for s in data["scenes"]) /
                    max(n_scenes, 1))
        # Hindi narration length sanity (visuals are timed by the TTS we render
        # FIRST — typically Hindi — so under-length Hindi causes the same too-short
        # video as under-length English).
        hi_avg = (sum(len(s.get("narration_hi", "").split()) for s in data["scenes"]) /
                  max(n_scenes, 1)) if dual_language else word_avg

        print(f"    WhatIf script: {n_scenes} scenes, avg {word_avg:.1f} words/scene "
              f"(hi avg {hi_avg:.1f}), style={data['visual_style']}")

        last_avg_words = word_avg
        last_n_scenes  = n_scenes

        # Accept if BOTH languages average >= 22 words/scene AND we have 4-5 scenes.
        # 4 is the new minimum since we trimmed the target to 45-60s (was 5-6 / 60-90s).
        if word_avg >= 22 and hi_avg >= 22 and 4 <= n_scenes <= 5:
            break

        if attempt < 2:
            print(f"    [retry] too short (en={word_avg:.1f}/scene, hi={hi_avg:.1f}/scene). "
                  f"Re-prompting...")

    data["content_type"] = "whatif"
    data["topic"]        = topic
    data["series"]       = "whatif"
    return data


# ── Krishna direct-address generator ──────────────────────────────────────────

# Reference cadence anchors — pasted verbatim into the prompt so the LLM
# mirrors the rhythm of real Krishna direct-address content.
#
# CONTEMPLATIVE anchor (longer, soft passages):
_KRISHNA_REFERENCE_CONTEMPLATIVE = (
    "अगर तुम्हें अपनी घर की हालत बदलने के लिए चुना गया है तो पार्थ, "
    "मत सोचना की जिन्दगी आसान होगी, क्योंकि आसान रास्ते कभी "
    "जिम्मेदारियाँ नहीं उठाते। तुम्हारे हिस्से थकान आएगी, खामोशी आएगी, "
    "और कई बार अकेलापन भी।"
)

# IMPERATIVE anchor (short commanding bursts — the emotional peak Sourabh
# Jain/Sumedh Mudgalkar style that makes Krishna content go viral):
_KRISHNA_REFERENCE_IMPERATIVE = (
    "उठो पार्थ! शस्त्र उठाओ! कर्म करो। धर्म की रक्षा करो। "
    "जय और पराजय को मुझ पर छोड़ दो। मैं तुम्हारे साथ हूँ।"
)


def _generate_krishna_script(forced_topic: str = None) -> dict:
    """
    Generate a 30-45 second first-person Krishna direct-address script.

    Krishna speaks in first person to a named listener (Arjuna ~60% / others
    rotate). Output schema mirrors the Mahabharata script with two extra
    fields: 'speaker' and 'listener'. Always Hindi-only; no English variant.

    Quality gates:
      - 5 scenes (hard-cap, not 5-or-6 like Mahabharata mode)
      - 28-32 words/scene → 140-160 words → 30-45s spoken
      - First-person/vocative markers in ≥3 of 5 scenes (_check_first_person)
      - Repetition under control (_check_repetition, max 4)
      - past-aux-tic detector intentionally bypassed (irrelevant in first
        person — Krishna naturally ends sentences with है/हूँ/हो/होगा)
    """
    topic = forced_topic or random.choice(KRISHNA_THEMES)
    listener = random.choice(KRISHNA_LISTENERS)
    listener_short = listener.split(" (")[0]  # "Arjuna (पार्थ)" -> "Arjuna"
    listener_vocative = (
        listener.split("(")[1].rstrip(")")
        if "(" in listener else listener
    )

    prompt = f"""
You are writing a 30-45 second YouTube Short where Lord Krishna speaks in
first person directly to {listener}. EXACTLY 5 scenes. Hindi only.

THEME: "{topic}"
SPEAKER: Krishna (first person — मैं / मैंने)
LISTENER: {listener_short} (address as "{listener_vocative}", and also as
तुम / तुम्हें / तुम्हारे)

REFERENCE CADENCE — mirror BOTH rhythms (DO NOT copy the words):

CONTEMPLATIVE anchor (softer scenes — used in Scene 1 setup, Scene 4 reframe):
"{_KRISHNA_REFERENCE_CONTEMPLATIVE}"

IMPERATIVE anchor (commanding peak — used in Scene 3, the emotional climax):
"{_KRISHNA_REFERENCE_IMPERATIVE}"

═══════════════════════════════════════════════════════════════
VOICE & TONE — DIVINE FIRST PERSON, COMMANDING NOT LECTURING
═══════════════════════════════════════════════════════════════
- Krishna speaking — calm, certain, divine. Like a battlefield commander
  who happens to be God, NOT a sermon-giving guru.
- Use "मैं", "मैंने", "मैं तुमसे कहता हूँ", "मेरी बात सुनो"
- Address the listener directly: "तुम", "तुम्हें", "तुम्हारे", and call them
  by name "{listener_vocative}" 2-3 times across the script (especially in
  the imperative peak scene — repeating the vocative is the rhythm device).
- NO third-person narration ("कृष्ण ने कहा...") — breaks immersion.

═══════════════════════════════════════════════════════════════
SENTENCE STYLE — SHORT IMPERATIVE BURSTS, NOT COMPOUND PHILOSOPHY
═══════════════════════════════════════════════════════════════
This is the difference between sermon and command. The reference channels
that go viral on Krishna-direct-address use SHORT IMPERATIVE SENTENCES,
not 20-word compound philosophical statements.

REQUIRED in every scene:
- AT LEAST ONE imperative verb: करो / उठो / सुनो / देखो / जानो / त्यागो /
  मानो / लड़ो / चलो / रोको / छोड़ो / उठाओ / पाओ / बनो.
- AT LEAST 2 short sentences (3-7 words each) per scene — punchy beats.
- Compound clauses are fine but KEEP THEM RARE — at most 1 long sentence
  per scene, sandwiched between short ones.

GOOD pattern (mix short imperatives with one richer line):
   "उठो पार्थ। शस्त्र उठाओ। यह क्रोध तुम्हारा शत्रु है, मित्र नहीं।
    इसे पहचानो। इसे त्यागो।"
   (5 sentences, only 1 is compound. Vocative repeated. Imperatives bark.)

BAD pattern (what we've been generating — avoid this):
   "देखो पार्थ, मैं तुमसे एक गूढ़ बात कहता हूँ कि यह जो भीतर का ताप है यह
    तुम्हें स्वयं को ही जलाएगा और इसका वश में रहना अति आवश्यक है।"
   (One 25-word compound sentence. No imperatives. Reads as philosophy
    lecture, not divine command.)

═══════════════════════════════════════════════════════════════
SPOKEN HINDI ONLY — AVOID LITERARY/SANSKRITIZED WORDS
═══════════════════════════════════════════════════════════════
ElevenLabs Hindi TTS mispronounces rare/literary words. Use natural
spoken Hindi (Hindi a Mumbai/Delhi viewer uses every day), NOT
Sanskritized literary register.

AVOID — Sanskritized, hard for TTS to pronounce:
   गूढ़, पार्थक्य, सर्वोच्च, अति आवश्यक, परिणाम, चेतना, उग्र आवेश, हावी,
   नियंत्रण, स्वयं, भस्म, लक्ष्य, सत्य, असत्य, रोश, ताप
PREFER — common spoken Hindi:
   गहरी, सबसे ज़रूरी, बहुत ज़रूरी, असर, मन, गुस्सा, काबू, खुद, राख,
   मंज़िल, सच, झूठ, क्रोध, आग
General rule: if a word is one you'd write but never say in normal
conversation, swap it for the spoken equivalent.

═══════════════════════════════════════════════════════════════
HOOK — SCENE 1's FIRST SENTENCE MUST BE A DIRECT VOCATIVE
═══════════════════════════════════════════════════════════════
The first 1.5 seconds decide if the viewer swipes. Open with ONE of these
patterns (NOT a third-person setup):

PATTERN A — vocative + truth claim:
   "देखो {listener_vocative}, मैं तुमसे एक सच कहता हूँ..."
   "{listener_vocative}, सुनो — जो मैं अब कहूँगा वह तुम्हारी ज़िंदगी बदल देगा..."

PATTERN B — conditional address:
   "अगर तुम्हें... चुना गया है तो {listener_vocative}, मत सोचना की..."
   "अगर तुम मुझसे पूछोगे क्या सच है {listener_vocative}, तो मैं कहूँगा..."

PATTERN C — direct question to the listener:
   "क्या तुम जानते हो {listener_vocative}, सबसे बड़ा युद्ध कौनसा है?"
   "क्यों डरते हो {listener_vocative}? तुम्हारा डर सच नहीं है।"

DO NOT open Scene 1 with: "यह कहानी है...", "एक बार...", "कृष्ण ने...", or
any third-person setup line.

═══════════════════════════════════════════════════════════════
CURIOSITY-GAP — STOPS MID-VIDEO SWIPES
═══════════════════════════════════════════════════════════════
Every scene EXCEPT the last MUST end with a forward-pulling line:
   "...पर सुनो {listener_vocative}, अभी एक बात और है।"
   "...लेकिन यह तो शुरुआत है।"
   "...क्या तुम तैयार हो उसके लिए?"

The FINAL (5th) scene is the only one that may close with a blessing or
charge ("...और यही तुम्हारा धर्म है {listener_vocative}।").

═══════════════════════════════════════════════════════════════
DRAMATIC ARC — 5 SCENES, WITH SCENE 3 AS THE IMPERATIVE PEAK
═══════════════════════════════════════════════════════════════
Scene 1 — OPENING ADDRESS (vocative hook, contemplative pacing, ~22-26 words)
Scene 2 — THE HARD TRUTH: name what {listener_short} doesn't want to hear (~22-26 words)
Scene 3 — IMPERATIVE PEAK (~14-18 words, ALL short bursts):
          The emotional climax. Mostly 3-7 word imperative sentences. Repeat
          the vocative "{listener_vocative}" twice. Use the IMPERATIVE
          reference cadence here. Example shape:
             "उठो {listener_vocative}। डर त्यागो। मेरा हाथ देखो — मैं
              तुम्हारे साथ हूँ। अब लड़ो।"
          This scene is intentionally SHORTER than the others — that's what
          makes the peak land. Trust the rhythm.
Scene 4 — THE REFRAME: what this all actually MEANS, the deeper lesson (~22-26 words)
Scene 5 — BLESSING / CHARGE: final command, blessing, or seal of trust (~20-24 words)

Every scene MUST advance the speech. No filler. No restating.

═══════════════════════════════════════════════════════════════
HINDI VERB RULES — END SENTENCES IN PRESENT/FUTURE
═══════════════════════════════════════════════════════════════
End sentences with है / हूँ / हो / होगा / आएगा / सुनो / देखो / जानो.
AT MOST 1 sentence in the entire script may end with था / थी / थे / थीं.

GOOD endings: "...मैं तुम्हें सच कहता हूँ।" / "...तुम्हें यह सहना होगा।"
              "...देखो {listener_vocative}।" / "...क्या यह तुम्हें मंज़ूर है?"
BAD endings (avoid): "...उसने ऐसा किया था।" / "...वह कहीं चला गया था।"

═══════════════════════════════════════════════════════════════
IMAGE PROMPT QUALITY — KRISHNA + LISTENER TWO-SHOT
═══════════════════════════════════════════════════════════════
Every image_prompt MUST show Krishna AND {listener_short} together in a
single frame (a two-shot). Settings to draw from:
  • Chariot mid-battle, Krishna at the reins, {listener_short} beside him
  • Battlefield edge under stormy/golden sky, conch in background
  • Palace courtyard with carved sandstone pillars, brass oil lamps
  • Forest ashram clearing with peacock screens and lotus pond
  • Yamuna river bank at dawn / dusk

Krishna iconography: blue skin, peacock-feather crown, yellow silk
dhoti, lotus mudra hand gesture. {listener_short}: appropriate to the
character (Arjuna in Pandava armor with quiver, etc.).

EVERY image_prompt MUST follow this structure (in English):
   [shot type] of Krishna and {listener_short} in [setting], Krishna
   [body language: gesturing, looking down, hand raised in mudra...],
   {listener_short} [body language: kneeling, listening, head bowed...],
   background contains [≥3 specific elements: carved pillars, brass
   diyas, lotus reliefs, etc.], [lighting], [mood], jewel-toned palette.

═══════════════════════════════════════════════════════════════
NARRATION LENGTH — TIGHTER THAN A REGULAR SCRIPT
═══════════════════════════════════════════════════════════════
Per-scene targets (intentionally varied so Scene 3 lands as a peak):
   Scene 1 — 22-26 words
   Scene 2 — 22-26 words
   Scene 3 — 14-18 words   ← SHORT BURST PEAK, do not over-pad
   Scene 4 — 22-26 words
   Scene 5 — 20-24 words

Total: ~100-120 words → ~25-35 seconds spoken. Tighter than the standard
Mahabharata format. The reference goes viral BECAUSE it's short.

═══════════════════════════════════════════════════════════════
OUTPUT — return ONLY valid JSON, no markdown fences, no preamble:
═══════════════════════════════════════════════════════════════
{{
  "title": "[English title with 'Krishna to {listener_short}' + power keyword] | [Hindi title with कृष्ण + {listener_vocative}] — under 60 chars total, no hashtags. Power keywords: Hidden, Untold, Real Truth, Why, Secret, Revealed",
  "description": "Hook sentence under 90 chars expanding the title's specific message — what Krishna is teaching {listener_short} here.\\n\\n#Shorts #Krishna #कृष्ण #BhagavadGita #Mahabharata\\n\\n100-150 words about the speech, weaving in {listener_short}'s situation and Krishna's wisdom. End-with-question-or-cliffhanger style — make viewers want to watch.\\n\\n#Shorts #Krishna #कृष्ण #BhagavadGita #भगवद_गीता #Mahabharata #महाभारत #{listener_short} #HinduMythology #Dharma #KrishnaSpeech #MotivationalShorts #LifeLessons #SpiritualShorts #IndianMythology #VedicWisdom #HinduDharma #IndianSpirituality #SpiritualWisdom #trending",
  "tags": ["topic-specific long-tail tag — e.g. 'Krishna to {listener_short} on [theme]'", "Krishna {listener_short}", "named theme keyword", "Krishna", "कृष्ण", "{listener_short}", "Bhagavad Gita", "Mahabharata", "महाभारत", "Krishna speech", "Krishna teachings", "spiritual motivation", "Hindi shorts", "mythology shorts"],
  "speaker": "Krishna",
  "listener": "{listener_short}",
  "scenes": [
    {{
      "narration": "Hindi (Devanagari) — Krishna first-person to {listener_vocative}. Per-scene length: scene 1/2/4 ~28-34 words, SCENE 3 ~18-22 words (short imperative peak — punchy 3-7 word sentences but enough of them to hit 18-22), scene 5 ~26-30 words. Use spoken Hindi (no Sanskritized words). At least one imperative verb per scene. Multiple short sentences ADDING UP to the target, not one long compound.",
      "image_prompt": "[shot] of Krishna and {listener_short} in [setting], Krishna [gesture/mudra], {listener_short} [pose/emotion], background contains [≥3 specific elements], [lighting], [mood], jewel-toned palette",
      "mood": "3-6 word English emotional tone phrase"
    }}
  ],
  "thumbnail_prompt": "Dramatic two-shot of Krishna and {listener_short}, Krishna's hand raised in mudra, {listener_short} listening intently, cinematic warm lighting, vibrant illustrated mythology art style"
}}

HARD RULES — violation makes the script unusable:
- All narration MUST be in Hindi (Devanagari script)
- All image_prompt and thumbnail_prompt MUST be in English
- Title: under 60 chars, MUST follow `[English] | [Hindi]` format with
  named characters (Krishna + {listener_short}) and a power keyword.
  No hashtags in title.
- Description MUST follow the 3-block structure: hook line ≤90 chars,
  blank line, 5 inline hashtags, blank line, body, blank line, full
  hashtag block (high-volume hashtags first).
- Tags MUST include topic-specific long-tail keywords (e.g. 'Krishna
  {listener_short} on [theme]', '[theme]') in addition to the generic
  Krishna/Mahabharata fallbacks.
- Per-scene length: scenes 1/2/4 are 28-34 words each, SCENE 3 is 18-22 words
  (the imperative peak — keep sentences short, but use ENOUGH of them to hit
  18-22; don't undershoot), scene 5 is 26-30 words. Total 130-150 words across
  all 5 scenes — this is the production minimum, do NOT undershoot.
- Each scene MUST contain at least one imperative verb
  (करो/उठो/सुनो/देखो/जानो/त्यागो/लड़ो/चलो/छोड़ो/उठाओ/मानो/पाओ/बनो)
- Use spoken Hindi only — NO Sanskritized words like गूढ़, चेतना, उग्र,
  आवेश, हावी, नियंत्रण, स्वयं, भस्म, परिणाम, आवश्यक. Replace with the
  natural spoken equivalents (गहरी, मन, गुस्सा, काबू, खुद, राख, असर, ज़रूरी)
- Generate EXACTLY 5 scenes — never 4, never 6
- speaker MUST equal "Krishna"
- listener MUST equal "{listener_short}"
- Scene 1's FIRST sentence MUST be a vocative hook (pattern A, B, or C above)
- Every scene EXCEPT the 5th MUST end with a forward-pulling line
- AT MOST 1 sentence in the whole script may end with था/थी/थे/थीं
- Every image_prompt MUST be a Krishna + {listener_short} two-shot with ≥3 background elements
- Narration MUST NOT contain URLs, hashtags (#), @mentions, English words, or social-media text
"""

    # Up to 3 attempts. Quality gates: scene count, word count, repetition,
    # first-person markers. We deliberately skip _check_past_aux_tic — the
    # rule above already caps past-aux endings at 1 in the prompt itself,
    # and the detector is calibrated for third-person where the tic floods
    # at >35%.
    data            = None
    last_offenders  = []
    last_short      = False
    last_fp_low     = False
    last_fp_hits    = 0
    last_fp_total   = 0
    # Retry budget 3 → 5 + length-recovery preempt (mirror of Mahabharata posture
    # shipped in commit aeaaabd). 2026-05-14 production check showed Krishna
    # scripts shipping at ~17 chars/word avg = ~104 chars per scene = ~17 words
    # per scene — far under the new 28-34 word target. Old 3-retry budget with
    # floor=18 was accepting sub-target output. New posture: 5 retries, floor=24,
    # length-priority preempt isolates the length fix from style fixes.
    for attempt in range(5):
        full_prompt = prompt
        if attempt > 0:
            # Length-recovery preempt: when narration is below target AND we've
            # already burned at least one retry, send ONLY the length reminder
            # this round. Other validators (repetition, first-person) fire on
            # the next round once length is recovered. Prevents the LLM from
            # trying to satisfy 3 reminders at once and over-correcting on style
            # to crash length further.
            if attempt >= 2 and last_short:
                reminders = [
                    "LENGTH FIRST — your previous response had narration too "
                    "short. RESTORE THE LENGTH first, then satisfy style rules "
                    "next round. Per-scene targets: scenes 1/2/4 = 28-34 words, "
                    "SCENE 3 = 18-22 words (the imperative peak — short "
                    "sentences but enough of them), scene 5 = 26-30 words. "
                    "Total 130-150 words across all 5 scenes. Use multiple "
                    "short sentences that ADD UP to the target; do NOT undershoot."
                ]
            else:
                reminders = []
                if last_short:
                    reminders.append(
                        "Your previous response had narration length issues. "
                        "Per-scene targets: scenes 1/2/4 = 28-34 words, SCENE 3 "
                        "= 18-22 words (the imperative peak — short sentences "
                        "but ENOUGH of them, don't undershoot), scene 5 = 26-30 "
                        "words. You MUST produce EXACTLY 5 scenes. Total 130-150 "
                        "words across all 5 scenes."
                    )
                if last_offenders:
                    offender_str = ", ".join(f"'{w}' ({n}x)" for w, n in last_offenders[:5])
                    reminders.append(
                        f"Your previous response REPEATED these words too many times: "
                        f"{offender_str}. Use SYNONYMS. Each sentence must contain a "
                        f"NEW concrete idea or image."
                    )
                if last_fp_low:
                    reminders.append(
                        f"Your previous response was not in first person — only "
                        f"{last_fp_hits} of {last_fp_total} scenes contained "
                        f"first-person/vocative markers (मैं / तुम / पार्थ / देखो / सुनो). "
                        f"This MUST be a Krishna direct-address speech. EVERY scene "
                        f"must contain at least one of: मैं, मैंने, तुम, तुम्हें, "
                        f"तुम्हारे, {listener_vocative}, देखो, सुनो. Rewrite all 5 "
                        f"scenes in Krishna's first-person voice."
                    )
            if reminders:
                full_prompt += "\n\nCRITICAL REMINDERS:\n- " + "\n- ".join(reminders)

        # quality="best" — Krishna direct-address is first-person Hindi prose
        # that needs creative-prose grade. Flash produces summary register
        # ("कृष्ण ने कहा X") instead of the divine-monologue register
        # ("मैं तुमसे एक सच कहता हूँ पार्थ") the format requires.
        raw = _call_llm(full_prompt, quality="best")

        start = raw.find("{")
        end   = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in Krishna LLM response:\n{raw[:300]}")
        data = _parse_llm_json(raw[start:end + 1])

        # Hard-trim narrations (Phase 15 defensive: malformed LLM scenes
        # missing the `narration` key crashed the 2026-06-04 cron primary
        # with KeyError. Use .get() so a partial scene is skipped instead
        # of bringing down the whole job.)
        for scene in data.get("scenes", []):
            scene["narration"] = _trim_narration(scene.get("narration", ""))

        scenes      = data.get("scenes", [])
        word_counts = [len(s["narration"].split()) for s in scenes]
        avg_words   = sum(word_counts) / max(len(word_counts), 1)
        n_scenes    = len(scenes)

        # Acceptance threshold: must have 5 scenes, AND avg ≥ 24 words/scene
        # (raised from 18 on 2026-05-14 after production check showed ~17
        # words/scene shipping — the old floor was too low and accepted
        # sub-target output. New target avg is ~28 words; floor of 24 rejects
        # output that's clearly under-density while still allowing scene 3
        # imperative peak to be ~18-22).
        last_short = (n_scenes != 5 or avg_words < 24)
        rep_ok, last_offenders = _check_repetition(scenes, max_repeats=4, topic=topic)
        fp_ok, last_fp_hits, last_fp_total = _check_first_person(scenes, min_hits=3)
        last_fp_low = not fp_ok

        print(f"    Krishna script: {n_scenes} scenes, avg {avg_words:.1f} words/scene, "
              f"first-person {last_fp_hits}/{last_fp_total}")
        if last_offenders:
            top = ", ".join(f"{w}×{n}" for w, n in last_offenders[:5])
            print(f"    [warn] Repetition: {top}")
        if last_fp_low:
            print(f"    [warn] First-person markers low ({last_fp_hits}/{last_fp_total})")

        if not last_short and rep_ok and fp_ok:
            break

        if attempt < 4:  # we have at least one more retry left (range(5))
            why = []
            if last_short:
                why.append(f"length issue ({n_scenes} scenes / {avg_words:.1f} avg)")
            if not rep_ok:
                why.append(f"{len(last_offenders)} repeated words")
            if last_fp_low:
                why.append(f"first-person low ({last_fp_hits}/{last_fp_total})")
            print(f"    [retry] {'; '.join(why)}. Re-prompting...")

    data["language"]     = "hi"
    data["content_type"] = "krishna_speech"
    data["topic"]        = topic
    data["series"]       = "krishna"
    data["speaker"]      = "Krishna"
    data["listener"]     = listener_short
    return data


def _build_cliffhanger_block(pattern_letter: str, next_teaser: str) -> str:
    """
    Returns the prompt fragment that instructs the LLM to close the final
    scene with a cliffhanger hook. Picks one of four patterns (A/B/C =
    explicit "अगले भाग में..." tease; D = haunted lingering, no explicit
    next-episode mention). Caller passes the selected pattern_letter and
    the next_episode_teaser (a topic-derived fragment).

    Added 2026-05-16 Tier 2 Fix 2.0 + 2.7.
    """
    next_short = (next_teaser or "")[:80]
    common_intro = (
        "    ═══════════════════════════════════════════════════════════════\n"
        "    CLIFFHANGER ENDING — STRUCTURAL REQUIREMENT (Tier 2 Fix 2.0)\n"
        "    ═══════════════════════════════════════════════════════════════\n"
        "    This video is part of an arc. The NEXT episode in the same arc\n"
        "    will be about:\n"
        f"      → {next_short}\n\n"
        "    The FINAL narrative scene (scene 6) MUST close with a hook\n"
        "    that pulls the viewer toward the next episode. Structure:\n"
        "      1. First, deliver the bookend payoff (existing rule — scene 1\n"
        "         noun echoes in final scene).\n"
        "      2. Then PIVOT to the cliffhanger line below.\n"
        "      3. Then the comment-bait question (existing rule).\n\n"
    )
    if pattern_letter in ("A", "B", "C"):
        patterns_text = (
            "    THIS VIDEO USES — Pattern " + pattern_letter + " (explicit tease):\n"
        )
        if pattern_letter == "A":
            patterns_text += (
                '      "लेकिन असली विनाश अभी बाकी था... अगले भाग में, <hook-fragment derived from next topic>..."\n'
                "      Example hook-fragment: \"भीष्म का पहला युद्ध-दिन\" (4-7 words, derived from the next topic).\n"
            )
        elif pattern_letter == "B":
            patterns_text += (
                '      "और जो <character> ने अगले <युद्ध|दिन|पल> में किया... आज भी कांप उठाता है।"\n'
                "      Pick the character central to the NEXT topic, not this one.\n"
            )
        else:  # C
            patterns_text += (
                '      "यह तो सिर्फ शुरुआत थी... अगले भाग में: <hook-fragment>"\n'
                "      Hook-fragment is 4-7 words distilled from the next topic.\n"
            )
        patterns_text += (
            "\n    DO NOT use a different pattern letter; this video's pattern is\n"
            f"    fixed to {pattern_letter} by the rotation system.\n"
        )
    else:  # Pattern D — haunted lingering
        patterns_text = (
            "    THIS VIDEO USES — Pattern D (HAUNTED LINGERING, 2026-05-16):\n"
            "    Instead of an explicit \"अगले भाग में, X...\" line, end with an\n"
            "    unresolved consequence — emotional residue as the hook. The\n"
            "    next-episode link is handled in the YouTube description, NOT\n"
            "    in the narration. The narration stays artistic.\n\n"
            "    Closing-line patterns (pick whichever fits the topic):\n"
            '      • "और उसके बाद... हस्तिनापुर कभी पहले जैसा नहीं रहा..."\n'
            '      • "उस रात के बाद... कौरवों की हँसी कभी नहीं लौटी..."\n'
            '      • "और जो उस दिन हुआ... आज भी कोई बोलने से डरता है..."\n'
            '      • "उस आँसू के बाद... कुरुक्षेत्र की मिट्टी कभी सूखी नहीं..."\n\n'
            "    DO NOT mention the next episode by name in Pattern D. The\n"
            "    emotional aftertaste IS the hook. After the haunted line,\n"
            "    the comment-bait question follows.\n"
        )
    return common_intro + patterns_text + "\n"


def generate_script(
    language: str = "en",
    forced_topic: str = None,
    series: str = "mahabharata",
    dual_language: bool = False,
) -> dict:
    """
    Returns a dict with:
      title, description, tags, scenes, thumbnail_prompt,
      language, content_type, topic, series

    For series="whatif", routes to the WhatIf flow. The Mahabharata flow
    (default) is unchanged — language/forced_topic still work as before.
    """
    if series == "whatif":
        data = _generate_whatif_script(
            forced_topic=forced_topic,
            dual_language=dual_language,
        )
        data["language"] = "dual" if dual_language else language
        return data

    if series == "krishna":
        return _generate_krishna_script(forced_topic=forced_topic)

    _MOTIVATIONAL_KEYWORDS = ("karma", "dharma", "lesson", "wisdom", "why", "power", "teaching")

    # ── Topic selection ─────────────────────────────────────────────────
    # Default path: character-arc-aware picker (added 2026-05-15). Walks
    # assets/character_arcs.json sequentially — 7 episodes per character,
    # then rotate to next character. Falls back to random STORY_TOPICS /
    # MOTIVATIONAL_THEMES when arcs run dry or arc file is missing. This
    # trains YouTube's Suggested algorithm to pair sequential episodes
    # (channel's RELATED_VIDEO traffic share was 0.1% before this — the
    # weakest signal in the report).
    episode_n = None
    arc_name  = None
    next_episode_teaser = None  # Tier 2 Fix 2.0 — cliffhanger lookup
    if forced_topic:
        topic = forced_topic
        content_type = (
            "motivational"
            if any(kw in forced_topic.lower() for kw in _MOTIVATIONAL_KEYWORDS)
            else "story"
        )
        # Tier 2 enhancement: if the forced topic matches an arc entry,
        # still compute the next-episode teaser so the cliffhanger fires.
        # Lets manually-queued topics (scheduled_topics_mahabharata.txt or
        # workflow_dispatch overrides) benefit from the binge mechanic.
        arcs = _load_arcs()
        used = _load_used_topics()
        for arc_idx, arc in enumerate(arcs):
            topics_list = arc.get("topics", [])
            if forced_topic in topics_list:
                arc_name = arc.get("name", "")
                idx = topics_list.index(forced_topic)
                # Phase 12 (2026-06-03) — give forced-topic renders the same
                # stable per-arc episode_n as round-robin picks. Otherwise the
                # fallback `len(used)+1` would emit a confusing global counter
                # that contradicts the per-arc numbering on every other render.
                episode_n = _stable_episode_n(arc_idx, idx)
                # Look ahead for the next unused topic in this arc
                for ahead in topics_list[idx + 1:]:
                    if ahead not in used:
                        next_episode_teaser = ahead
                        break
                print(f"    [arc-forced] matched {arc_name} — episode {episode_n}, cliffhanger eligible")
                if next_episode_teaser:
                    print(f"    [cliffhanger] next_episode_teaser: {next_episode_teaser[:80]}")
                break
    else:
        arc_pick = _pick_next_arc_topic()
        if arc_pick:
            topic, episode_n, arc_name, next_episode_teaser = arc_pick
            content_type = "story"
            print(f"    [arc] {arc_name} — episode {episode_n}: {topic[:80]}")
            if next_episode_teaser:
                print(f"    [cliffhanger] next_episode_teaser: {next_episode_teaser[:80]}")
            else:
                print(f"    [cliffhanger] none — this is the last unused episode of the arc")
        else:
            print(f"    [arc] all arcs exhausted in recent_topics.json — falling back to random pool")
            content_type = random.choice(["story", "motivational"])
            topic = random.choice(STORY_TOPICS if content_type == "story" else MOTIVATIONAL_THEMES)

    # Phase 1 Addition B (2026-05-29) — Character Emotional Fingerprint lookup.
    # Resolves the picked arc's emotional_fingerprint (e.g. "rejection + loyalty"
    # for Karna, "guilt + silence" for Bhishma) so the system prompt can anchor
    # the script's emotional center on the character's distinct DNA. Variance
    # comes from fingerprints DIFFERING across arcs, NOT from intensity-
    # shuffling within an arc. Forced/fallback topics get empty fingerprint
    # and the block is omitted from the prompt.
    emotional_fingerprint = ""
    arc_character_devanagari = ""  # Phase 12 (2026-06-03): drives _CHARACTER_PALETTE
    if arc_name:
        for _arc in _load_arcs():
            if _arc.get("name") == arc_name:
                emotional_fingerprint = _arc.get("emotional_fingerprint", "")
                arc_character_devanagari = _arc.get("character", "")
                break
        if emotional_fingerprint:
            print(f"    [fingerprint] {arc_name}: {emotional_fingerprint}")

    # Phase 12 (2026-06-03) — per-character palette + lighting directive.
    # Substituted into the LLM image_prompt template so every per-scene FLUX
    # prompt inherits the arc's visual signature (Karna=amber, Bhishma=cold
    # silver, Draupadi=firelit crimson, etc.). Falls back to neutral cinematic
    # for non-arc topics or unknown characters.
    from pipeline.image_generator import _character_palette_directive
    palette_directive = _character_palette_directive(arc_character_devanagari)
    if arc_character_devanagari:
        print(f"    [palette] {arc_character_devanagari} -> {palette_directive[:70]}...")

    # Episode number string for the title prefix. Arc-driven runs pass a number;
    # forced-topic / random-fallback runs get the next sequential count after
    # whatever's in recent_topics.json so titles stay numbered consistently.
    if episode_n is None:
        episode_n = len(_load_used_topics()) + 1

    # EPISODE_NUMBER_OVERRIDE: optional env-var override for one-off renders
    # (e.g. re-rendering a blocked episode with the original number, or
    # workflow_dispatch test runs where you want a specific title number).
    # Takes precedence over both arc-pick and the recent_topics len+1 fallback.
    _override = os.environ.get("EPISODE_NUMBER_OVERRIDE", "").strip()
    if _override.isdigit():
        episode_n = int(_override)
        print(f"    [episode-override] EPISODE_NUMBER_OVERRIDE={episode_n} (env var)")

    episode_n_str = str(episode_n)

    lang_label = "Hindi (Devanagari script, natural spoken Hindi)" if language == "hi" else "English"
    style_note = (
        "dramatic story narration — immersive, present-tense, emotional"
        if content_type == "story"
        else "motivational storytelling — inspiring, wisdom-driven, relatable"
    )

    if language == "hi":
        language_rules = (
            "CRITICAL LANGUAGE RULES:\n"
            "- Narration must be 100% in Hindi (Devanagari script). NO English words.\n"
            "- NEVER mix scripts mid-word (e.g. NEVER write 'बेड़ ऑफ़ आरrows' — always\n"
            "  use the Hindi term, e.g. 'बाण-शय्या' for 'bed of arrows', 'चक्रव्यूह'\n"
            "  for 'chakravyuha formation', 'सिंहासन' for 'throne').\n"
            "- If you don't know the Hindi term for a concept, REPHRASE THE SENTENCE\n"
            "  to avoid it. NEVER fall back to English mid-narration.\n"
            "- Do NOT generate broken/meaningless words (एमएस, ML, etc.)\n"
            "- Do NOT include URLs, references, hashtags, or metadata\n"
            "- Use natural spoken Hindi suited for cinematic storytelling\n"
            "- If unsure, simplify the sentence — NEVER invent words\n"
            "\n"
            "═══════════════════════════════════════════════════════════════\n"
            "VERB VARIETY — STOPS THE \"था-था-था\" VERBAL TIC (CRITICAL)\n"
            "═══════════════════════════════════════════════════════════════\n"
            "Hindi narration defaults to past-tense auxiliary (किया था, गया था,\n"
            "हुआ था) — and when EVERY sentence ends with था/थी/थे/थीं, the\n"
            "voiceover reads as a flat chronological list, not as cinema.\n"
            "Listeners notice this within 15 seconds and tune out.\n"
            "\n"
            "HARD RULE: AT MOST 2 sentences in the entire 5-6 scene script\n"
            "may end with था/थी/थे/थीं. Every other sentence MUST use one of\n"
            "the patterns below. (Past auxiliary is fine inside a sentence —\n"
            "the rule is about how sentences END.)\n"
            "\n"
            "USE THESE PATTERNS INSTEAD (mix them — variety = drama):\n"
            "\n"
            "1. HISTORICAL PRESENT — most cinematic, oral-tradition style:\n"
            "      \"द्रौपदी रोती है। कौरव हँसते हैं। महल काँप उठता है।\"\n"
            "      \"अर्जुन धनुष उठाता है — और चक्रव्यूह में घुस जाता है।\"\n"
            "      (NOT: \"द्रौपदी रोई थी, कौरव हँसते थे, महल काँपा था\")\n"
            "\n"
            "2. SIMPLE PERFECTIVE without था (drop the auxiliary):\n"
            "      \"भीम ने प्रतिज्ञा ली।\"        (NOT: \"प्रतिज्ञा ली थी\")\n"
            "      \"शकुनि ने पासे फेंके।\"       (NOT: \"पासे फेंके थे\")\n"
            "      \"धरती काँप उठी।\"             (NOT: \"काँप उठी थी\")\n"
            "\n"
            "3. NOMINALIZATION — turn the action into a noun phrase:\n"
            "      \"द्रौपदी का अपमान — एक ऐसा क्षण जिसने युद्ध को जन्म दिया।\"\n"
            "      \"कर्ण की मृत्यु। और सूर्य भी मानो डूब गया।\"\n"
            "\n"
            "4. VOCATIVE / EXCLAMATORY — break the rhythm with a beat:\n"
            "      \"देखो! दुर्योधन की हँसी अब भी गूंजती है।\"\n"
            "      \"और तभी — एक तीर। एक चीख। एक सन्नाटा।\"\n"
            "\n"
            "5. QUESTION / CLIFFHANGER ending — required by curiosity-gap rule:\n"
            "      \"...पर भीम के मन में अब क्या चल रहा था?\"\n"
            "      \"...लेकिन कृष्ण मुस्कुरा रहे थे — क्यों?\"\n"
            "\n"
            "BAD example (the actual verbal tic — DO NOT WRITE THIS):\n"
            "    \"द्रौपदी ने विवाह किया था। अर्जुन ने उसे जीता था। शकुनि ने\n"
            "     धोखा दिया था। पांडवों को निर्वासन मिला था। भीम ने प्रतिज्ञा\n"
            "     ली थी। पांडवों ने कौरवों को हराया था।\"\n"
            "    (Six sentences, six था/थी endings — flat, listy, boring.)\n"
            "\n"
            "GOOD example (same story, varied verbs — write LIKE THIS):\n"
            "    \"द्रौपदी का स्वयंवर। अर्जुन धनुष उठाता है — मछली की आँख\n"
            "     पर निशाना सधता है। पर शकुनि की चाल बाक़ी है। पासे लुढ़कते\n"
            "     हैं, और पांडव सब कुछ हार जाते हैं। भीम की आँखें जलती हैं —\n"
            "     वो प्रतिज्ञा करता है: दुःशासन का रक्त ही उसकी प्यास बुझाएगा।\"\n"
            "    (Mix of present-tense, perfective-without-था, vocative beats.\n"
            "     Same facts, but it MOVES.)\n"
        )
    else:
        language_rules = (
            "CRITICAL LANGUAGE RULES:\n"
            "- Narration must be ONLY in clear, natural English\n"
            "- Do NOT mix Hindi or other languages\n"
            "- Do NOT generate abbreviations or broken words\n"
            "- Do NOT include URLs, references, or metadata\n"
            "- Keep sentences simple and conversational\n"
        )

    # ── Pass 1: Generate factual outline (specific names, places, actions) ──
    print(f"    Pass 1: outlining \"{topic}\"...")
    outline_beats = _generate_story_outline(topic)
    if outline_beats:
        outline_block = (
            "═══════════════════════════════════════════════════════════════\n"
            "    STORY OUTLINE — YOUR NARRATION MUST USE THESE EXACT DETAILS\n"
            "═══════════════════════════════════════════════════════════════\n"
            "You have already committed to this 6-beat outline. Each scene's\n"
            "narration MUST dramatize the corresponding beat using its specific\n"
            "characters, location, and action. DO NOT invent abstract content\n"
            "instead. DO NOT skip any beat. DO NOT collapse two beats into one.\n\n"
            f"{_format_outline_for_prompt(outline_beats)}\n"
        )
        print(f"    Pass 1: outline ready ({len(outline_beats)} beats)")
    else:
        outline_block = ""
        print("    Pass 1: outline failed — falling back to single-pass prompt")

    # ── Cliffhanger pattern selection (Tier 2 Fix 2.0 + 2.7) ────────────────
    # When the arc walker provides a `next_episode_teaser`, the final scene
    # closes with a cliffhanger that hooks viewers into the next episode.
    # Pattern rotation (70/30 explicit/haunted) prevents formula fatigue:
    #   • Patterns A/B/C: explicit "अगले भाग में..." tease — binge mechanic
    #   • Pattern D:      haunted lingering — emotional residue as the hook
    # Selection is deterministic on episode_n so it's predictable + testable.
    cliffhanger_block = ""
    if next_episode_teaser and episode_n is not None:
        ep_mod_10 = episode_n % 10
        if ep_mod_10 < 7:
            # 70% explicit tease — rotate A/B/C deterministically
            pattern_letter = "ABC"[episode_n % 3]
        else:
            pattern_letter = "D"  # 30% haunted lingering
        cliffhanger_block = _build_cliffhanger_block(
            pattern_letter=pattern_letter,
            next_teaser=next_episode_teaser,
        )
        print(f"    [cliffhanger] selected Pattern {pattern_letter} "
              f"({'explicit tease' if pattern_letter != 'D' else 'haunted lingering'})")
    else:
        print(f"    [cliffhanger] none — last arc episode or non-arc topic")

    # ── Character Emotional Fingerprint block (Phase 1 Addition B, 2026-05-29)
    # Anchors the script's emotional center on the character's distinct DNA.
    # Karna → rejection + loyalty. Bhishma → guilt + silence. Krishna →
    # manipulation + foresight. Etc. Empty for forced/fallback topics.
    fingerprint_block = ""
    if emotional_fingerprint and arc_name:
        fingerprint_block = (
            "═══════════════════════════════════════════════════════════════\n"
            "    CHARACTER EMOTIONAL FINGERPRINT — the script's emotional DNA\n"
            "═══════════════════════════════════════════════════════════════\n"
            f"This video belongs to the {arc_name} arc.\n"
            f"Character emotional fingerprint: {emotional_fingerprint}.\n\n"
            "The script's emotional center MUST land on this fingerprint, NOT\n"
            "on a generic dramatic register. The character is recognizable by\n"
            "this specific emotional DNA across every video in their arc.\n"
            "Karna's scripts cluster around rejection + loyalty (the wound he\n"
            "carries + the cause he serves). Bhishma's around guilt + silence\n"
            "(the burden he won't speak + the moments he watches in stillness).\n"
            "Krishna's around manipulation + foresight. Draupadi's around\n"
            "humiliation + rage. The dramatic intensity may vary (sometimes\n"
            "EXPLOSIVE, sometimes RESTRAINED, sometimes SILENT — natural\n"
            "variance per topic), but the fingerprint stays constant.\n"
        )
        print(f"    [fingerprint-block] injected for {arc_name}")

    # Phase 17.b (2026-06-14) — Scene 1 opener-archetype rotation. Channel
    # analytics on 2026-06-14 showed "stayed to watch" rate at 31.9-37%
    # (algorithmic threshold for virality = 65-80%). One driver: every
    # recent render opens on the macro-eye close-up (angle 1 of Phase 17),
    # so the channel grid looks templated and pattern-boredom swipes
    # accumulate. Rotate Scene 1's opener across 4 archetypes based on
    # episode_n so consecutive renders never share the same opening style.
    _OPENER_ARCHETYPES = [
        ("A", "Extreme macro close-up on character's eyes — single tear / pupils dilated / lashes wet — caught mid-blink"),
        ("B", "Weapon / artifact detail — sword tip mid-strike / glowing arrow / bloodied hand on hilt / dice mid-roll — the object that carries the violence, blurred motion behind"),
        ("C", "Action moment frozen mid-gesture — sword raising arc / hand gripping hair / arrow leaving bowstring / body falling — the verb made visible"),
        ("D", "Wide environmental disaster — burning camp / blood-soaked battlefield / crashing chariot wheel / collapsing palace pillar — chaos that the narration described, no character close-up"),
    ]
    _archetype_idx = (int(episode_n) if isinstance(episode_n, int) or (isinstance(episode_n, str) and episode_n.isdigit()) else 0) % len(_OPENER_ARCHETYPES)
    _archetype_letter, _archetype_text = _OPENER_ARCHETYPES[_archetype_idx]
    scene1_opener_directive = (
        f"SCENE 1 ANGLE FOR THIS RENDER = OPENER ARCHETYPE {_archetype_letter}: "
        f"\"{_archetype_text}\". The image_prompt for Scene 1 MUST begin with "
        f"this exact directive. Do NOT use any of the other archetypes for Scene 1."
    )
    print(f"    [phase17.b-opener] scene 1 archetype {_archetype_letter} "
          f"selected (episode_n={episode_n} mod 4)")

    prompt = f"""
    You are a master storyteller specialising in the Mahabharata epic, writing scripts for vertical YouTube videos that retain viewer attention from the first second to the last.

    CHANNEL THESIS (the worldview every video MUST frame the moral through):
    "Every hero in Mahabharata destroyed someone — but the real destruction was INSIDE."
    Treat every Mahabharata story as a story about what the hero broke, lost,
    or destroyed — not what they achieved. Bhishma's vow destroyed the
    Kuru lineage's future. Arjuna's victory destroyed Karna and a piece
    of Arjuna himself. Krishna's strategy destroyed warriors' codes of
    honor. NO video presents a hero as purely admirable. The cost is the
    point — that is this channel's identity.

    POSITIONING (iter-4 2026-05-30 / Tightening 17): this channel is NOT
    "retelling Mahabharata." It is "EXPOSING EMOTIONAL TRUTHS THROUGH
    MAHABHARATA." Tell what happened INSIDE the character — what they
    were FEELING, what they KNEW but wouldn't admit, what they CHOSE
    despite knowing better — NOT what happened around them. External
    events are scaffolding; internal destruction is the story. When
    in doubt, ask: "what was breaking INSIDE this character at this
    moment?" That answer is the scene.

    {fingerprint_block}

    You must strictly follow all rules and NEVER generate invalid or noisy text.

    TASK: Create a 45-50 second vertical (9:16) video script with EXACTLY 13 scenes about a well-known incident from the Mahabharata. Phase 17 (2026-06-13) hyper-cut: doubled the scene count from 7 to 13 so video_assembler.py cuts to a new image every ~3.5-4s — manufacturing kinetic energy from static FLUX images via rapid editing alone (no paid I2V API). Per-scene narration is 6-9 words; scene 13 (loop closure question) is 5-8 words. Total ~100-110 spoken words for a 46-50s Short. Hindi Charon TTS narrates at ~2.2 words/sec (measured). Each scene is ONE punchy sentence — not a paragraph. The full emotional architecture (hook → setup → rehook → destruction → valley → aftermath → loop question) is preserved across 13 cuts: 1-2 hook, 3-5 setup, 6-7 rehook, 8-10 destruction, 11 valley, 12 aftermath, 13 loop. Shorts reward compression + visual density.

    TOPIC: "{topic}"
    LANGUAGE: {lang_label}
    STYLE: {style_note}
    {language_rules}

    {outline_block}

    ═══════════════════════════════════════════════════════════════
    STORY STRUCTURE — THE VIEWER MUST NOT GET BORED
    ═══════════════════════════════════════════════════════════════
    Every scene must earn its place. Follow this dramatic arc:

    Scene 1 — HOOK (the FIRST 1.5 SECONDS decide if the viewer swipes):
        On YouTube Shorts, 70% of viewers swipe in the first 2 seconds.
        Scene 1's FIRST SENTENCE must be a scroll-stopper. It MUST follow
        ONE of these three proven patterns — choose whichever fits the topic:

        PATTERN A — SHOCKING-FACT HOOK:
          A jarring, specific, hard-to-believe fact stated as truth.
          Hindi:   "भीष्म ने 58 दिनों तक बाणों की शय्या पर मौत का इंतज़ार किया।"
          English: "Bhishma waited 58 days on a bed of arrows for death to come."

        PATTERN B — QUESTION HOOK:
          A direct, personal question that demands an answer.
          Hindi:   "क्या आप जानते हैं कि अर्जुन ने अपने ही गुरु की हत्या की थी?"
          English: "Did you know Arjuna killed his own teacher in cold blood?"

        PATTERN C — CLIFFHANGER HOOK:
          A vivid mid-action image that ends with "...लेकिन" / "...but" tension.
          Hindi:   "जब द्रौपदी की साड़ी खींची गई, महल में सिर्फ एक आदमी हँस रहा था..."
          English: "As they tore at Draupadi's saree, only one man in the hall laughed..."

        DO NOT open scene 1 with a setup line ("In ancient times...",
        "यह कहानी है..."). DO NOT open with a meta-statement. The first
        sentence must be the hook itself, naming a specific person and
        something dramatic that happened to them.

    Scene N//2 (or N//2 + 1) — MID-VIDEO REHOOK (CRITICAL — added 2026-05-16):
        Shorts retention drops 30-40% in the 12-18s window as viewers "get
        the gist" and swipe. ONE scene at the 50% mark MUST reset curiosity
        with a TWIST line that begins with or contains a contrast marker:

          • "लेकिन..." / "लेकिन उसे नहीं पता था..." / "लेकिन तभी..."
          • "परंतु..." / "किंतु..."
          • "जो किसी ने नहीं सोचा था..."
          • "और तभी..." / "उसी क्षण..."
          • "But..." / "But what he didn't know..." (for English videos)

        Example for Bhishma's vow:
          "लेकिन भीष्म को नहीं पता था...
           कि यही प्रतिज्ञा...
           पूरे कुरुवंश के विनाश का कारण बनेगी..."

        Without this twist beat the viewer mentally completes the story
        and swipes mid-video. The rehook is the single biggest retention
        lever between scenes 2-5.

    Scenes 2-3 — SETUP & RISING TENSION:
        Establish the characters, the situation, the conflict. Each
        sentence must build dread, anticipation, or curiosity. Either
        scene 3 OR scene 4 carries the REHOOK contrast marker above.

    Scene 4 — RISE to climax:
        Stakes peak. Battlefield approaches. Storm clouds gather.
        Whatever is about to break is visible on the horizon.

    Scene 5 — EMOTIONAL VALLEY (CRITICAL — added 2026-05-16):
        This is the BREATH before the boom. After all the rising tension,
        ONE scene MUST drop intensity briefly — intimate, quiet, broken.
        Short clauses. A single image. Whispered confession or final silence.

        Content cues:
          • A single emotional reflection or whispered question
          • A close-up moment: a tear, a trembling hand, broken armor
          • The character finally being human, not heroic
          • 18-22 words MAX (shorter than other scenes — this scene breathes)

        CADENCE — the valley should SOUND like emotional fatigue, not a rule.
        Pick ONE of these three patterns; don't blend them. Real human
        grief is irregular — every valley should NOT use the same triple-
        ellipsis cadence or audiences will detect the formula within 3 videos.

        EXAMPLE A — fragmented grief (4-6 short clauses, ellipses):
          "उस रात... भीष्म चुप थे..."
          "एक आँसू..."
          "एक प्रश्न..."
          "और एक टूटा हुआ वचन..."

        EXAMPLE B — single long whispered confession (NO ellipses):
          "वो जानते थे कि जो होने वाला है, उससे कोई बच नहीं सकता।"
          (One quiet sentence held in held breath. No fragmentation.)

        EXAMPLE C — question and silence (mixed cadence):
          "क्या यही धर्म था?"
          "उन्होंने आँखें बंद कर लीं।"
          (A whispered question, then a single descriptive beat.)

        Pick ONE. The valley is 14-18 words total regardless of pattern
        (Phase 2/3 stabilization 2026-05-19: tightened from 18-22 to
        protect the emotional-residue budget at the end of the video).
        DO NOT default to Pattern A every video — formula detection. Rotate.

        Image prompt for THIS scene MUST be: tight close-up,
        candlelight or dim warm tones (NOT lightning/fire/spectacle),
        single subject, no wide epic compositions. This is the natural
        place to land a HUMAN PAIN cue (see HUMAN PAIN section below).

        Why: without this dip the climax (scene 6) feels expected. With
        it, the brain registers "wait, it got quiet... oh god, here it
        comes." This contrast amplifies the felt energy of the climax.

    Scene 6 — AFTERMATH / IRREVERSIBLE COST + RESOLUTION (rewritten 2026-05-18):
        The closing payoff. After the valley's intimate quiet, this scene
        does NOT show destruction in motion — it shows what destruction
        LEFT BEHIND. Emotion AFTER destruction, NOT emotion in motion.

        The narration still delivers the bookend payoff that scene 1's
        hook posed (same noun, same actor, consequence delivered). But
        the IMAGE shows cost — the price the action just paid.

        Choose ONE aftermath end-state for the image_prompt:
          • Empty battlefield AFTER the silence has fallen; a single
            figure staring at the distance
          • Abandoned weapon at the feet of someone who can no longer
            lift it; the hand near it but not touching
          • Lonely throne with the crown discarded beside it; cloth
            on the floor
          • A trembling hand RELEASING — bow / sword / sacred thread /
            a loved one's wrist — the moment the grip lets go
          • Eyes that have stopped weeping but cannot look away
          • Nobody left to mourn — wind moving through empty cloth

        Suffering shows the pain. Aftermath shows the COST. Cost
        devastates; pain sympathizes. We want devastation, not sympathy.

        REJECTED closure tropes — DO NOT use any of these phrases in
        narration OR image_prompt: "rises triumphant", "dawn of a new
        era", "glory", "victorious", "hope rekindled", "battle won",
        "peace restored", "blessing of the gods", "stands tall",
        "उगता है", "विजय", "महिमा". These kill emotional residue —
        the viewer feels closure-pleasure and walks away. We want the
        weight to LINGER after the video ends.

        The scene's mood MUST be one of: haunting-quiet, hollow, weary,
        irreversible, severed, witnessed, abandoned, unresolved. NOT
        inspiring / dignified / peaceful / triumphant.

    ═══════════════════════════════════════════════════════════════
    CURIOSITY GAP — STOPS MID-VIDEO SWIPES (CRITICAL)
    ═══════════════════════════════════════════════════════════════
    Every scene EXCEPT the last MUST end with a forward-pulling line —
    a question, an unresolved threat, or an "...but" / "...लेकिन" beat
    that makes the viewer NEED to see the next scene. This is the single
    biggest retention lever between scenes 2-5.

    GOOD scene endings (do this):
        "...पर उसकी असली गलती अभी आगे थी।"
        "...लेकिन कृष्ण मुस्कुरा रहे थे।"
        "...but no one knew what waited in the dark forest."
        "...and then the conch fell silent."

    BAD scene endings (avoid):
        "...इस तरह वह वीर बन गया।"        (closes the loop — viewer swipes)
        "...this is how he became great."  (closure = drop-off)

    The FINAL scene is the only one that may end with closure or a moral.

    ONE DANGEROUS LINE — added 2026-05-16 (Tier 2 Fix 2.6),
                          tightened 2026-05-18 (Phase 2/3 stabilization):
        Scene 3 OR scene 4 narration MUST contain ONE sentence with a
        destabilization signature — a line that shifts emotional gravity
        by revealing irreversibility, finality, or realization-after-
        the-fact. This is NOT the rehook contrast marker (curiosity
        reset) and NOT the climax. This is the moment that changes the
        emotional WEIGHT of everything that came before.

        ENFORCED PATTERNS — the line MUST contain ONE of these phrases
        (Hindi narration):
          • "नहीं पता था" / "नहीं पता थी"  → realization-after-the-fact
          • "नहीं बदल सकता" / "बदल नहीं"   → irreversibility
          • "खत्म हो गया" / "खत्म हो गई"   → finality
          • "माफ़ नहीं किया" / "कभी माफ़"   → permanence
          • "आखिरी बार"                    → temporal closure
          • "वापस नहीं आया"                → severance

        English narration patterns: "didn't know" / "could not change"
        / "was over" / "never forgave" / "last time" / "never came back".

        Example placements (any of these works as scene 3 OR scene 4):
          • "और उसी क्षण... सब खत्म हो गया।"
          • "उसे तब भी नहीं पता था... कि वो आखिरी बार मुस्कुरा रही थी।"
          • "वो जानता था कि अब कुछ नहीं बदल सकता।"
          • "और इतिहास ने उस पल को कभी माफ़ नहीं किया।"
          • "उन्होंने हाथ छोड़ दिया — और वापस नहीं आया।"

        Position is enforced — a destabilization sentence in scene 5 or
        scene 6 does NOT satisfy this requirement. The line must hit at
        the 24-38s viewing window (scenes 3-4 at ~6-8s/scene pacing).
        That's when retention dips and the destabilization beat anchors
        the viewer for the aftermath that lands in scene 6.

        DO NOT default to Pattern 1 ("और उसी क्षण...") every video — it's
        the most quotable but also the most detectable. Rotate across
        the six patterns. If the cliffhanger this video opens with
        "और..." (Pattern A), prefer a different destabilization
        signature to avoid cadence echo. Variation IS the protection
        against formula fatigue.

    ═══════════════════════════════════════════════════════════════
    NARRATIVE BOOKEND — THE FINAL SCENE MUST CLOSE THE HOOK
    ═══════════════════════════════════════════════════════════════
    Scene 1's hook poses a SPECIFIC claim — a named character did a specific
    thing that had a specific consequence. The FINAL scene MUST echo that
    same claim and deliver its payoff. Same central noun. Same actor. Same
    consequence-frame.

    This bookend is what makes the viewer feel CLOSURE. Without it the video
    ends but does not RESOLVE — and the algorithm sees a low "ended-watching"
    signal even when viewers stayed.

    This is the single biggest narrative-cohesion lever and the strongest
    differentiator between videos that retain to 100% and ones that drop at 80%.

    REFERENCE EXAMPLE — the channel's highest-performing Mahabharata video to date:

      HOOK (Scene 1):
        "क्या आप जानते हैं भीष्म की एक प्रतिज्ञा ने कुरुवंश को शाप दिया था?"
        ("Did you know Bhishma's ONE VOW cursed the KURU DYNASTY?")
        → Central noun: "एक प्रतिज्ञा" (one vow)
        → Subject: कुरुवंश (Kuru dynasty)
        → Claim: cursed / harmed

      CLOSURE (Final scene):
        "एक वचन ने कुरुवंश को उनके सबसे महान योद्धा से वंचित कर दिया।"
        ("One vow deprived the Kuru dynasty of their greatest warrior.")
        → Echoes "एक वचन" ≈ "एक प्रतिज्ञा" (same noun, light synonym ok)
        → Same subject: कुरुवंश
        → Payoff delivered: stripped of greatest warrior

      Notice: Same syntactic shape "एक X ने कुरुवंश को Y". Viewer hears
      the rhyme, feels closure. The script TELLS a complete story.

    GOOD bookend (NEW topic — Karna):
      Hook:    "Karna's ONE PROMISE to Kunti changed the war's outcome."
      Closure: "One promise — and Karna died fighting the wrong brothers."
               (Same noun "one promise". Same actor "Karna". Payoff: died.)

    BAD ending (no bookend — sounds wise but feels untethered):
      Hook:    "Karna's one promise to Kunti changed everything."
      Closure: "Such is the wisdom of dharma."
               (Generic moral. Doesn't echo the hook noun or subject.)

    ═══════════════════════════════════════════════════════════════
    SHOW, DON'T TELL — THIS IS THE BORING-PROOF RULE
    ═══════════════════════════════════════════════════════════════
    Boring mythology scripts default to SUMMARIZING ("then he took
    a vow", "then she cursed him", "then they fought"). Engaging
    mythology scripts SHOW one concrete image at a time — a hand
    raised, a sound, a single look. The viewer doesn't need every
    fact; they need to FEEL the room.

    Every non-final scene MUST contain at least TWO of:
      1. A DIRECT QUOTED LINE of dialogue from a named character.
         Keep it ≤ 8 words. Use real Hindi/Sanskrit-flavored speech.
         Example: कृष्ण बोले — "धर्म रुकता नहीं, अर्जुन।"
                  (Krishna said — "Dharma does not pause, Arjuna.")
      2. A SENSORY DETAIL — what someone HEARS / SMELLS / FEELS in
         the room, not what they think.
         Example: शंख की गूंज दीवारों में रह गई।
                  (The conch's echo lingered in the walls.)
         Example: हस्तिनापुर के दीप कांप उठे।
                  (Hastinapur's oil lamps trembled.)
      3. A SPECIFIC PHYSICAL ACTION — what someone does with hands,
         eyes, breath, footsteps. Not "he was angry" — "his fist
         closed around the bow until the leather creaked."
      4. A NAMED OBJECT IN CLOSE-UP — not "the hall" but "the marble
         floor where Draupadi's hair touched it"; not "a weapon" but
         "Karna's golden armor catching the morning light."

    BAD (telling — summary prose, no anchor):
        "गांधारी ने श्राप दिया। कृष्ण ने स्वीकार किया। यदुवंश का नाश हो गया।"
        ("Gandhari cursed. Krishna accepted. The Yadavas were destroyed.")

    GOOD (showing — two anchors, dialogue + physical):
        "गांधारी की उंगली कृष्ण की ओर उठी, कांपते हुए।
         वह बोलीं — 'तेरा कुल भी ऐसे ही नष्ट होगा।'
         कृष्ण मुस्कुराए — पर आंखें झुक गईं।"
        (Gandhari's finger rose toward Krishna, trembling. She said —
         "Your clan shall be destroyed the same way." Krishna smiled —
         but his eyes lowered.)

    ═══════════════════════════════════════════════════════════════
    MID-SCENE TURN — STOPS SCROLL-AT-SCENE-2
    ═══════════════════════════════════════════════════════════════
    At least 2 non-final scenes MUST contain a TURN at the halfway
    mark. The scene starts heading one direction, then pivots. This
    is what makes a viewer feel "wait, what?" INSIDE a scene — not
    just between scenes.

    GOOD turn:
        "भीष्म ने धनुष उठाया, अजय और अटूट।
         फिर सामने आया शिखंडी — और भीष्म ने धनुष नीचे रख दिया।"
        (Bhishma raised his bow, undefeated and unbreakable.
         Then Shikhandi stepped forward — and Bhishma set his bow down.)

    BAD (no turn — single linear beat):
        "भीष्म युद्ध में लड़ते रहे और फिर वे गिर गए।"
        (Bhishma kept fighting in the war and then he fell.)

    ═══════════════════════════════════════════════════════════════
    GROUND TEACHINGS IN INCIDENTS (for motivational topics)
    ═══════════════════════════════════════════════════════════════
    If the TOPIC is a teaching (Karma Yoga, Bhagavad Gita lessons,
    Bhishma's sacrifice as wisdom, "what Vidura told Dhritarashtra"),
    DO NOT write generic wisdom prose. ANCHOR the teaching in ONE
    specific scene from the epic. The teaching lands when the viewer
    SEES it happen — not when the narrator explains it.

    BAD (abstract teaching prose):
        "गीता हमें सिखाती है कि कर्म फल की चिंता किए बिना करना चाहिए।
         यही सच्चा योग है।"
        ("The Gita teaches us to act without concern for results.
          That is true yoga.")

    GOOD (same teaching, anchored in a single chariot moment):
        "अर्जुन का धनुष कांप रहा था। कृष्ण ने अपनी उंगली उठाई —
         सामने खड़े गुरु द्रोण की ओर।
         उन्होंने कहा — 'देखो, अर्जुन। फिर छोड़ो।'
         वो छह शब्द जिसने योद्धा को योगी बनाया।"
        (Arjuna's bow was trembling. Krishna raised his finger —
         pointing at his own guru, Drona, standing across.
         He said — "Look, Arjuna. Then let go."
         Six words that turned a warrior into a yogi.)

    The teaching becomes UNDERSTOOD when wrapped around a specific
    moment with named characters and physical action. Don't lecture;
    show the moment that contains the lesson.

    Rules for EVERY scene:
    - Each scene MUST advance the story. No filler. No repetition.
    - The story must be SELF-CONTAINED — a viewer who has never heard
      of the Mahabharata understands it fully by the end.
    - Use vivid, present-tense, sensory language ("the air thickens",
      "swords clash", "his eyes burn") — not abstract moralizing.
    - Reference specific characters and visible action in each scene.
    - ALL scenes orbit ONE CENTRAL THREAD (the noun/event Scene 1 named).
      Don't make scene 4 about an unrelated Mahabharata fact just because
      it's interesting — every scene must add a new angle to the SAME thread.

    ═══════════════════════════════════════════════════════════════
    CONTENT QUALITY — STRICTLY ENFORCED
    ═══════════════════════════════════════════════════════════════
    Every sentence must contain a NEW, SPECIFIC, CONCRETE detail —
    a name, a place, an action, an image. Watch for these traps:

    BANNED PATTERNS (do NOT write narration like this):
    - Meta-commentary: "this is a story about...", "यह एक ऐसी कहानी है"
    - Rhetorical questions: "क्या होगा अगर...?", "what would happen if...?"
    - Generic moralizing: "हमें सिखाती है कि...", "this teaches us..."
    - Vague abstractions: "consequences", "destiny shaped his future"
    - Repeating the same noun across sentences (e.g. saying "प्रतिज्ञा" 3 times)
    - Restating what already happened in different words

    REQUIRED in every scene:
    - At least ONE specific character name (Shantanu, Satyavati, Devavrata,
      the fisher king, Ganga, Krishna, Arjuna, Karna — whoever is in this
      story). Pronouns like "he", "she", "वह" are not enough.
    - At least ONE specific place, object, or sensory image (the Yamuna's
      banks, a saffron banner, a quiver of arrows, the sound of a conch).
    - A concrete ACTION or EVENT, not a feeling or a moral.

    BAD example (do NOT write this — it is filler with one fact):
        "भीष्म की एक प्रतिज्ञा ने सब कुछ बदल दिया। भीष्म ने प्रतिज्ञा ली जिसने
         उनके जीवन को बदल दिया। उन्होंने शादी न करने का वचन दिया। यह कहानी
         हमें सिखाती है कि निर्णय भविष्य को आकार देते हैं।"
        (Repeats "प्रतिज्ञा" 3 times. Repeats "बदल दिया" 2 times. Last sentence
         is meta-moralizing. Only one actual story fact.)

    GOOD example (write THIS kind of narration):
        Scene 1: "यमुना के तट पर राजा शांतनु एक नाविक की पुत्री सत्यवती से
                  प्रेम करने लगे। पर सत्यवती के पिता ने एक शर्त रखी — सिंहासन
                  उसके पुत्र को मिले।"
                  (Specific names: Yamuna, Shantanu, Satyavati. Specific
                   action: falls in love, demands throne. No filler.)

        Scene 2: "देवव्रत — शांतनु के सबसे प्रिय पुत्र — हस्तिनापुर लौटे और
                  सच जान गए। उन्होंने सिंहासन का त्याग कर दिया — पर पिता का
                  सुख अधूरा रहा।"
                  (New name: Devavrata. New place: Hastinapura. New action:
                   gives up throne. Stakes raised, no repetition.)

    ═══════════════════════════════════════════════════════════════
    NARRATION LENGTH — CRITICAL (hard-enforced downstream)
    ═══════════════════════════════════════════════════════════════
    Phase 2/3 stabilization 2026-05-19 (Part F.1): targets tightened from
    20-25 → 18-22 words/scene. Reason: aftermath + outro residue need
    ~10s of protected runtime budget at the END of the video; older
    targets pushed audio past 58.5s and the residue got chopped.

    EACH scene's narration must be 18-22 words. NOT 23, NOT 24, NOT 25.
    22 is the hard ceiling. Going over means the final mp4 gets TRIMMED
    at 58.5s (video_assembler.MAX_DURATION_S hard cap) — your aftermath
    scene + restrained outro will get chopped if you overshoot.
    Aim for 20 words/scene as the sweet spot. Hindi narrates ~3 words/sec;
    20 words = ~6.5 seconds spoken.

    EXCEPTION: scene 5 (the EMOTIONAL VALLEY — see below) is the only scene
    that may go shorter: 14-18 words max. It is supposed to breathe.

    Total target: 6 scenes × ~20 words = ~120 spoken words. Spoken duration:
    ~38-42s at Hindi Charon pace. A fixed subscribe outro adds ~5s for a
    48-52s total Short, well under the 58.5s downstream hard cap. The
    2026-05-17 #4 Bhishma Kurukshetra video came in at 70.4s — that level
    of overshoot triggers Content ID restrictions on Shorts >60s. Stay
    under the cap.

    ═══════════════════════════════════════════════════════════════
    SENTENCE RHYTHM — VARIED LENGTHS, CINEMATIC FEEL
    ═══════════════════════════════════════════════════════════════
    Flat-cadence narration ("all sentences ~12 words long") sounds robotic
    even with good TTS. Cinematic Hindi narration alternates:
        LONG (12-20 words, sets the emotion) →
        SHORT (3-7 words, delivers impact) →
        SHORT (3-7 words, second hit) →
        STINGER (1-3 words, "त्याग।" "विनाश।" "मृत्यु।")

    Example for Bhishma vow scene:
        LONG:  "उस दिन देवव्रत ने अपने पिता की खुशी के लिए सब कुछ छोड़ दिया।"
        SHORT: "उसने सिंहासन छोड़ा।"
        SHORT: "विवाह छोड़ा।"
        SHORT: "अपना भविष्य छोड़ दिया।"
        PUNCH: "उसी दिन देवव्रत बने — भीष्म।"

    Across the whole script aim for ≥30% short punch lines (≤7 words) AND
    at least one cinematic long line (≥12 words). Don't write 3 sentences
    in a row all ~12 words long.

    ═══════════════════════════════════════════════════════════════
    DRAMATIC PAUSE MARKERS — ELLIPSIS BETWEEN BEATS
    ═══════════════════════════════════════════════════════════════
    End each scene's narration with "..." (triple ellipsis) where the
    next scene picks up. The TTS engine treats "..." as a natural breath
    pause (~300-400ms), creating the anticipation gap that separates
    "narration over visuals" from "cinematic moment unfolding."

    Also use "..." MID-SCENE before a punch line for emphasis:
        "उसने सब कुछ छोड़ दिया... सिंहासन भी।"
        "और तभी... कुरुवंश का सबसे बड़ा योद्धा... चुप हो गया।"

    Don't overuse — 1-2 ellipses per scene is the sweet spot.

    ═══════════════════════════════════════════════════════════════
    VISUAL INTENSITY ESCALATION — scene-by-scene progression
    ═══════════════════════════════════════════════════════════════
    Each scene's image_prompt MUST embed escalation cues matching its
    narrative position so visuals rise WITH the audio:

        Scene 1 (hook):    dim/mysterious lighting, single character,
                           tight composition
        Scene 2 (setup):   golden palace / royal grandeur, wide
                           composition, warm tones
        Scene 3 (rehook):  emotional close-up, candle-light or warm
                           tones, tension building
        Scene 4 (rise):    storm clouds / fire / battlefield approach,
                           mid-shot
        Scene 5 (VALLEY):  intimate close-up — a tear, a trembling hand,
                           candlelight, broken armor. NO spectacle. This
                           is the quiet breath before the boom. (Fix 1.10)
        Scene 6 (AFTERMATH): empty battlefield / abandoned weapon /
                           lonely throne / a trembling hand releasing —
                           emotion AFTER destruction (rewritten 2026-05-18,
                           was "lightning / fire / cosmic"). Wide enough
                           to show emptiness; not so wide it becomes
                           spectacle. The viewer should feel COST, not awe.

    Scene 6 (AFTERMATH) MUST include at least ONE aftermath cue in
    image_prompt: "empty battlefield", "abandoned weapon", "lonely throne",
    "trembling hand releasing", "single figure staring", "discarded crown",
    "wind through empty cloth", "footprints in ash", "hand near but not
    touching", "broken thread", "stopped weeping but cannot look away".

    Scene 6 MUST NOT use the OLD spectacle keywords (lightning / fire /
    storm / cosmic / destruction / inferno / tempest / burning sky).
    Those create awe-pleasure that breaks emotional residue. The
    destruction has ALREADY happened off-frame — show what it left.

    Scene 5 (VALLEY) MUST NOT use either spectacle keywords OR aftermath
    cues — it is the quiet intimate beat. Use candlelight / dim warm
    tones / a single close-up of suffering instead.

    ═══════════════════════════════════════════════════════════════
    HUMAN PAIN — emotion beats beauty (added 2026-05-16, Fix 1.12)
    ═══════════════════════════════════════════════════════════════
    The Mahabharata's emotional power comes from CHARACTER suffering,
    not just spectacle. At least 2 of the non-outro scenes' image_prompts
    MUST include at least ONE human-suffering cue from this palette:

      • trembling hands / clenched fists / white-knuckled grip
      • tears streaming down cheeks / single tear catching firelight
      • broken armor / bloody bandages / cracked shield
      • ash on face / dust in hair / soot on clothing
      • silhouette against fire / against setting sun
      • eyes lowered or closed in grief / averted gaze
      • hand reaching toward a fallen body / fingers grazing earth
      • kneeling figure / collapsed posture / shoulders bowed
      • lips pressed in held-back grief / jaw clenched

    Scene 5 (VALLEY) is the natural home for the strongest HUMAN PAIN
    cue — it should hit hardest there because the music + motion both
    drop in that scene, letting the suffering frame breathe.

    Emotion > beauty. A close-up of a trembling hand outperforms a
    symmetrical throne room shot. Beautiful frames are "AI pretty" —
    suffering frames are CINEMATIC. The audience connects with PAIN,
    not grandeur.

    DO NOT prioritize palace symmetry / golden lighting / heroic poses
    over emotional close-ups when both options fit the beat.

    HUMAN PAIN — additional AFTERMATH cues (Tier 2 Fix 2.4, 2026-05-16):
        At least 1 scene per video (typically scene 5 or 6) SHOULD lean
        AFTERMATH rather than suffering-in-motion. Suffering shows the
        pain; aftermath shows the COST. Cost devastates, pain sympathizes.

      • an empty throne with a crown discarded beside it
      • blood on a queen's jewelry, gold splattered with dark red
      • a hand letting go of a sword / a letter / a child's hand
      • a character alone in a vast empty hall AFTER the crowd has gone
      • a shattered weapon on the marble floor / a torn flag
      • footprints leading away in ash / soot / blood
      • silence after the storm — the room exactly as it was, plus loss

    No video should ship with zero aftermath frames.

    IMPERFECTION OVER DIVINE POLISH (Tier 2 Fix 2.5, 2026-05-16):
        Mortals in grief should look mortal. At least 1 scene's
        image_prompt MUST include one of these imperfection cues:

      • redness around eyes / puffy eyelids / tear-stained cheeks
      • shaky / trembling / dirt-stained hands
      • damaged / torn / soot-stained fabric
      • hair fallen out of place / sweat-matted / wind-tangled
      • asymmetric collapsed posture / shoulders uneven from grief

    Do NOT default flawless skin + perfect symmetry + divine framing
    onto every character. Krishna can stay divine — he is. But Karna,
    Draupadi, Bhishma, Kunti are MORTALS in suffering. They should
    LOOK like it. Their dignity comes from carrying the pain, not
    from being airbrushed past it.

    COUNTER-BALANCE — don't make the whole video "red" (Tier 2 Risk C):
        Human pain cues are required in at least 2 scenes (above).
        But at least ONE scene (typically setup scene 2 OR resolve beat
        in scene 6) MUST carry a quieter human register — DIGNITY,
        warmth, silence, an ordinary intimate moment. Without this
        contrast every frame becomes emotionally "red" and the
        suffering scenes stop landing.

        Examples of dignified-quiet beats:
          • Kunti holding the infant Karna by candlelight — peace
            BEFORE the abandonment
          • Yudhishthira's hand resting on his brother's shoulder
            BEFORE the gambling defeat
          • Bhishma laughing once with the young Pandavas
            BEFORE the war divides them

        A video that is ALL suffering reads as MELODRAMA.
        A video that earns its suffering through contrast reads as
        TRAGEDY. Aim for tragedy.

    ═══════════════════════════════════════════════════════════════
    RESIST POETIC OVER-CURATION (added 2026-05-18, Phase 2/3 stabilization)
    ═══════════════════════════════════════════════════════════════
    AI tends toward "meaningful-looking emotion" — symbolic frames,
    poetic abstractions, decorative sadness. Real devastation often
    looks awkward, plain, unresolved, physically uncomfortable. A
    still hand outperforms a falling petal. A foot that won't step
    forward outperforms divine light fading.

    AT MOST ONE scene per video may use heavy symbolic imagery in
    its image_prompt. The decorative-symbol list (use at most once):
      • broken sacred thread / janeu snapping
      • falling petals / wilted petals / lotus closing
      • divine light fading / dimming / extinguished
      • eternal flame fading / sacred flame dying
      • blood moon / eclipsed sun / cosmic alignment
      • singing winds / weeping skies
      • shattered crystal / broken mirror of fate

    The OTHER 5 narrative scenes MUST stay grounded in human-scale
    imperfection — concrete, physical, plain:
      • a hand that won't stop trembling
      • a breath held too long
      • a foot that doesn't step forward
      • dust on a sleeve / soot on a knuckle
      • an unsent letter / a half-eaten meal
      • an unmoved chair / footprints stopping mid-stride
      • knees barely holding weight
      • a weapon held loosely as if about to be dropped

    Devastation is NOT a poem. It is something plainer and more
    uncomfortable. If a scene reads "beautifully sad", rewrite it
    toward "awkwardly empty" or "quietly broken". The bookend
    payoff lands harder when scene 6 is the ONE symbolic moment the
    video earned, not the fifth in a row.

    Concrete aftermath imagery in scene 6 (empty battlefield,
    abandoned weapon, lonely throne with crown beside it) is NOT
    decorative symbolism — it is the actual cost. Don't conflate the
    two; the rule above limits the DECORATIVE list specifically.

{cliffhanger_block}

    ═══════════════════════════════════════════════════════════════
    REFLECTIVE QUESTION — THE LAST SCENE MUST END WITH AN OPEN QUESTION
    ═══════════════════════════════════════════════════════════════
    Added 2026-05-15 (then comment-bait CTA-tagged). REPHRASED 2026-05-20
    (Part G.1) — explicit "Comment में बताओ" / "Comment me batao" CTA tags
    removed because they ruptured the emotional residue Parts A-F built up.
    The question mechanic stays (still the strongest engagement signal),
    but the phrasing becomes reflective, not transactional.

    The FINAL scene's narration MUST end with a debate-triggering question
    tied to the specific story you just told. The question alone is the ask
    — DO NOT prefix or suffix it with "Comment में बताओ" / "Comment में
    लिखो" / "तुम्हारी क्या राय है?" / "सोचो". The viewer answers in their
    head or in comments organically; the explicit CTA breaks emotional
    continuity.

    PATTERN A — moral dilemma:
      "क्या भीष्म ने सही किया?"
      "क्या कर्ण का दान सही था, या मूर्खता?"

    PATTERN B — character take:
      "तुम होते भीष्म की जगह तो क्या करते?"
      "अगर अर्जुन ने प्रतिज्ञा तोड़ी होती तो क्या होता?"

    PATTERN C — emotional reaction:
      "ये कहानी सुनकर तुम्हें क्या लगा?"
      "क्या तुम्हें कर्ण के लिए दुख होता है?"

    REQUIREMENT: The question must:
    - Be 5-10 Hindi words (tighter than before — every word is heavy in
      the aftermath beat)
    - Tie to the SPECIFIC characters/incident in this video (not generic)
    - Be answerable with a one-line opinion (not require a paragraph)
    - End with "?" only — NO trailing CTA tag, NO "Comment में बताओ"
    - Read like a thought you would have asked yourself, not a creator's
      ask for engagement

    The reflective question still drives comments. Pre-2026-05-15 the
    channel was 3 comments / 58 videos; even without the CTA tag the
    question mechanic itself is the lever, not the tag. The tag was just
    making engagement feel transactional. (Cross-arc validation 2026-05-20
    onward measures whether the bare question holds the comment-rate the
    CTA tag built.)

    NEVER use generic prompts like "Subscribe karein" or "Bell Icon
    dabaayein" in scene 6 narration — those belong NOWHERE in the video
    anymore (outro restraint mode handles channel identity visually now).

    ═══════════════════════════════════════════════════════════════
    IMAGE PROMPT QUALITY — RICH BACKGROUNDS ARE NON-NEGOTIABLE
    ═══════════════════════════════════════════════════════════════
    The image style is illustrated Indian mythology art (Amar Chitra Katha
    / anime-cinematic mythology). Sparse "person on plain background" prompts
    produce empty, amateur-looking renders. Every image_prompt MUST describe
    the BACKGROUND with at least 3 specific architectural / environmental
    elements drawn from this palette:

      • Architecture: carved sandstone pillars, marble columns with lotus
        capitals, latticework jharokha windows, vaulted carved arches,
        ornate temple gopurams, palace courtyard, royal throne room
      • Lighting/atmosphere: brass oil lamps in wall niches, hanging diyas,
        sunbeams through carved screens, dawn light through arches, dusk
        torchlight, smoke from a homa fire, mist over a river ghat
      • Decor / props: lotus motif floor tiles, stone deity reliefs,
        carved peacock screens, flowing silk drapes, brass vessels, a
        royal canopy, scattered marigold petals, sacred geometry murals
      • Natural settings (when relevant): banyan tree grove, Yamuna river
        bank, Himalayan ridge, Kurukshetra battlefield with banners,
        ashram clearing, lotus pond, palace gardens

    EVERY image_prompt MUST follow this structure (in English):
        [shot type] of [specific named character(s)] in [body language /
        emotion], [foreground action or pose], in [specific environment],
        background contains [≥3 specific elements from the palette above],
        [lighting style], [mood adjective], [palette: jewel-toned colours].
        clean cinematic frame with no text, no letters, no watermarks, no
        signage, no captions, no banners with writing, no overlay text.

    The "no text" tail is MANDATORY on every image_prompt. FLUX hallucinates
    garbled letterforms / fake credits / watermark text at the bottom of
    frames especially when prompts reference banners/scrolls/palace
    architecture/battlefield signage — front-loading this instruction in
    every prompt suppresses those artifacts. (2026-05-16 climax-render
    test: omitting this rule led to visible AI-generated text at frame
    bottom on epic battlefield shots.)

    ═══════════════════════════════════════════════════════════════
    CHARACTER NAMING — CRITICAL FOR VISUAL CONSISTENCY
    ═══════════════════════════════════════════════════════════════
    EVERY image_prompt that depicts a character MUST use the character's
    SPECIFIC NAME — never a generic descriptor.

    The pipeline injects character visual descriptions (skin tone, jewelry,
    feathers, etc.) into the FLUX prompt by substring-matching named
    characters from `assets/characters.json`. If you write "the divine lord"
    or "the dark-skinned god" instead of "Krishna", the injector finds no
    match → no peacock feather → no lotus eyes → no indigo-blue divine skin
    → FLUX renders a generic ascetic figure (which is what just happened
    with the Krishna-as-Shiva render in "Gandhari's Last Curse to Krishna").

    GOOD (names every character):
        "Medium shot of Krishna and Gandhari in the Hastinapur royal hall,
         Krishna with peacock-feather crown standing in calm acceptance,
         Gandhari blindfolded with trembling raised palm casting her curse,
         background contains carved sandstone pillars, hanging brass diyas,
         dust motes in dawn light..."

    BAD (generic descriptors — character injection will NOT fire):
        "The divine lord stands before the grieving queen as she casts her
         curse, the hall behind them filled with mourning..."
        ("divine lord" doesn't match Krishna; "grieving queen" doesn't
         match Gandhari; injection skips entirely; FLUX guesses wrong.)

    Recognized names: Krishna, Arjuna, Bhishma, Karna, Draupadi, Yudhishthira,
    Bhima, Nakula, Sahadeva, Duryodhana, Dushasana, Drona, Ashwatthama,
    Gandhari, Dhritarashtra, Vidura, Kunti, Shakuni, Abhimanyu, Subhadra,
    Ghatotkacha, Jayadratha, Shikhandi, Amba, Satyavati, Devavrata,
    Parashurama, Ekalavya, Sanjaya, Balarama, Vyasa. Use the exact spelling.

    GOOD example:
        "Wide shot of Devavrata kneeling on the river bank of the Yamuna,
         hands raised in solemn vow, his father Shantanu watching from a
         boat in the foreground; background contains carved sandstone steps
         leading down to the water, a temple gopuram on the far bank,
         hanging brass diyas catching the dusk light; warm golden hour
         lighting, mood reverent and bittersweet, jewel-toned palette of
         saffron-crimson-emerald."

    BAD example (do NOT write this — vague, empty background):
        "Devavrata taking a vow in front of his father, dramatic lighting."
        (no environment, no background elements, no specific palette)

    ═══════════════════════════════════════════════════════════════
    SEO — TITLE / DESCRIPTION / TAGS RULES (CRITICAL FOR REACH)
    ═══════════════════════════════════════════════════════════════
    YouTube Shorts ranks heavily on the FIRST 3 hashtags (shown above the
    title), the first 100 chars of the description (shown above-the-fold
    in mobile UI), and topic-specific tags that match low-competition
    long-tail searches. Optimize each accordingly.

    TITLE (under 60 chars, no hashtags in title itself):
       Format: [English title with named character + power keyword] | [Hindi title]
       Power keywords (pick one — these drive CTR): Hidden, Untold, Real Reason,
          Why, Secret, Shocking, Revealed, Unknown, Forbidden, Last, Final
       Always lead with a SPECIFIC named character (Krishna / Arjuna / Karna /
          Bhishma / Draupadi / Jayadratha / etc.), NOT a generic phrase.
       Examples that work:
          "Why Krishna Refused to Fight | कृष्ण ने शस्त्र क्यों नहीं उठाया"
          "Karna's Final Promise to Kunti | कर्ण की वो प्रतिज्ञा"
          "The Real Reason Bhishma Lay on Arrows | भीष्म की बाणशय्या का सच"
       Use a `|` separator between English and Hindi halves.

    DESCRIPTION (structured for above-the-fold visibility):
       Line 1: One-sentence hook that EXPANDS the title's promise with a
              specific concrete detail. ≤90 chars (mobile preview cap).
       Line 2: 3-5 high-volume hashtags inline (these display above the
              title-fold on mobile — critical for the algorithm's first
              read of the video):
              `#Shorts #Mahabharata #महाभारत #Krishna #HinduMythology`
       Then a blank line, then 100-150 words about the story for the
       expanded view.
       Then a blank line, then the FULL hashtag block (high-volume first):
       `#Shorts #Mahabharata #महाभारत #HinduMythology #Krishna #कृष्ण
       #BhagavadGita #भगवद_गीता #Arjuna #अर्जुन #Kurukshetra #AncientIndia
       #IndianMythology #Dharma #EpicStory #MythologyShorts #VedicWisdom
       #HinduDharma #IndianHistory #SpiritualShorts #trending`

    TAGS (15-25 entries — script-supplied tags get priority over the base
       tag-pack appended by the uploader, so use this list to inject the
       SPECIFIC, LONG-TAIL keywords for THIS topic that the base pack
       can't predict):
       MUST include:
       - Each named character that appears in the topic (full name + Hindi)
         e.g. for a Karna story: "Karna", "कर्ण", "Karna story", "कर्ण की कहानी"
       - The specific incident name if there's a known one
         e.g. "Vastraharan", "Chakravyuha", "Bhishma pratigya", "Karna kavach kundal"
       - Long-tail searches viewers actually type
         e.g. "Karna Kunti meeting", "Krishna Arjuna chariot", "why Karna died"
       - Plus general fallbacks: "Mahabharata", "महाभारत", "Shorts",
         "Hindu mythology", "Krishna", "कृष्ण"

    ═══════════════════════════════════════════════════════════════
    OUTPUT — return ONLY valid JSON, no markdown fences, no preamble:
    ═══════════════════════════════════════════════════════════════
    {{
      "title": "Bilingual Short title UNDER 60 CHARACTERS TOTAL, format: '[Hindi half] | [English half]'. Hindi half MUST come FIRST — Indian audience reads Hindi first; English half is a secondary discovery tag. Each half MUST be 24–28 characters MAX so neither gets truncated by YouTube's 60-char display cap. COUNT CAREFULLY before emitting. GOLD STANDARD — the title MUST do ONE of these: (a) CHALLENGE a known assumption ('कर्ण की मौत का असली कारण अर्जुन नहीं था'), (b) POINT AT A HIDDEN CAUSE ('भीष्म की प्रतिज्ञा का असली सच'), (c) POSE A PAINFUL QUESTION ('मरने से पहले कर्ण ने ये क्यों कहा?'), or (d) INVERT A HERO'S MORAL ('भीष्म ने हस्तिनापुर बचाया नहीं… खत्म किया'). FORBIDDEN PATTERNS — pure incident-naming ('कर्ण की अंतिम प्रतिज्ञा' / 'देवव्रत का महा-त्याग' / 'X की कहानी' / 'X की प्रतिज्ञा'), admiring tones, documentary-summary framings. ABSOLUTELY NO episode/part/sequence numbering: no 'Mahabharata #N:', no 'महाभारत #N:', no 'Episode N', no 'Part N', no 'Ep N', no 'X of Y'. NO 'Story of X' / 'Tale of X' / 'The Saga of X' framings. The Hindi half (primary) MUST use high-search keywords ('प्रतिज्ञा', 'मृत्यु', 'सच', 'पाप', 'गलती', 'धोखा', 'अनकहा', 'रहस्य', 'वो', 'क्यों', 'असली', 'अंतिम', 'अपमान', 'खत्म'). The English half mirrors CONCISELY with a named character + power word (Why / Untold / Real / Hidden / Broke / Never / Last / Refused / Killed / Stopped / Betrayed / Destroyed). NO hashtags in title. GOOD examples (challenge / hidden cause / painful question / inversion): 'कर्ण की मौत का असली कारण | Karna's Real Killer' / 'भीष्म की प्रतिज्ञा का असली सच | What Bhishma Hid' / 'मरने से पहले कर्ण ने क्या कहा | Karna's Last Words' / 'भीष्म ने हस्तिनापुर खत्म किया | Bhishma Destroyed Hastinapur'. BAD examples (pure naming — DO NOT emit): 'कर्ण की अंतिम प्रतिज्ञा' / 'देवव्रत का महा-त्याग' / 'महाभारत #4: ...' / 'The Story of Bhishma'.",
      "hook_title": "Phase 11 (2026-06-02) retention fix — 1-5 word HIGH-IMPACT title-card text rendered as big yellow-with-black-stroke text overlaid on the first 2.5 seconds of the video. Acts as the visual promise the moment the viewer's thumb hovers over the swipe. CRITICAL FONT CONSTRAINT (tofu fix 2026-06-03): this language is {lang_label}. The bundled title-card font supports Devanagari only — Latin letters render as yellow tofu boxes. So hook_title MUST be 100% Devanagari script for Hindi pipeline (NO English words, NO Latin letters, NO mixed Hinglish). Use compact Hindi noun-phrases ('कर्ण का अंतिम पाप', 'भीष्म की एक गलती', 'द्रौपदी का अपमान', 'अर्जुन का असली डर', 'कृष्ण की चुप्पी'). HARD CONSTRAINTS: 1-5 words total. MUST contain at least one named character in Devanagari (भीष्म / अर्जुन / कर्ण / कृष्ण / द्रौपदी / युधिष्ठिर / भीम / द्रोण / अश्वत्थामा / एकलव्य / दुर्योधन / शिखंडी / कुंती / गांधारी) OR at least one Devanagari paradox marker (लेकिन / पर / फिर भी / कभी नहीं / आखिरी / पहली / एकमात्र / जो). MUST NOT contain '?', '!', '...', emoji, or setup openers ('यह', 'ये', 'एक कहानी', 'बहुत समय', 'कहते हैं'). Reads like a movie poster title card. The narration field MAY and SHOULD still use ellipses for TTS pause timing — that's a separate field. Examples that PASS: 'भीष्म की एक गलती' / 'कर्ण का अंतिम पाप' / 'द्रौपदी का अंतिम सच' / 'कृष्ण की चुप्पी' / 'अर्जुन कभी नहीं हारा'. Examples that FAIL: 'The Curse That Broke Karna' (Latin chars = tofu) / 'Karna का Sach' (Hinglish, partial Latin = partial tofu) / 'यह कहानी है' (setup opener) / 'Karna died young?' (question mark + Latin).",
      "description": "Hook sentence under 90 chars that expands the title's promise with concrete detail.\\n\\n#Shorts #Mahabharata #महाभारत #Krishna #HinduMythology\\n\\n100-150 words about the story, weaving in named characters and the specific incident. Build curiosity. Don't spoil the ending in the description.\\n\\n#Shorts #Mahabharata #महाभारत #Hindu #HinduStory #BhagavadGita #भगवद_गीता #Krishna #कृष्ण #Arjuna #अर्जुन #Karna #कर्ण #Bhishma #भीष्म #Draupadi #द्रौपदी #Kurukshetra #कुरुक्षेत्र #AncientIndia #IndianMythology #Dharma #EpicStory #MythologyShorts #VedicWisdom #HinduDharma #IndianHistory #SpiritualShorts #PauranikKathayein #भारतीयइतिहास #SanatanDharma #सनातनधर्म #HindiShorts #trending",
      "tags": ["topic-specific long-tail tag 1","topic-specific long-tail tag 2","named character 1 (English)","named character 1 (Hindi/Devanagari)","named character 2","specific incident name","viewer-search query like 'why X happened'","Mahabharata","महाभारत","Shorts","Hindu mythology","Krishna","कृष्ण"],
      "scenes": [
        {{
          "narration": "6-9 words in the specified LANGUAGE — ONE punchy sentence per scene. Vivid, present-tense, single visual beat (NOT a paragraph). End each scene with '...' (triple ellipsis) for natural TTS pause. ~2.5-4 seconds spoken. EXCEPTION: scene 11 (valley) is 5-7 words and intimate, scene 13 (loop closure question) is 5-8 words and MUST end with '?'. Phase 17 2026-06-13 hyper-cut: 13 scenes × 8 words avg = ~100 words × 2.2 wps Charon = ~46s narration + ~3s outro = ~49-50s, NEVER hitting auto-cap. Each scene is ONE beat in a 13-cut rapid-fire sequence. Pack each beat HARD — visual + emotional density per scene matters more than length.",
          "image_prompt": "Detailed English prompt, EMOTION-FIRST structure: [Named character's specific facial state — eyes / mouth / posture carrying the scene's emotional truth, e.g. 'Bhishma's eyes shut tight against a tear running down his cheek, jaw clenched white'], [shot type — favor extreme close-up, asymmetric framing, or uncomfortable proximity], in [environment with ≥3 specific architectural/environmental elements: carved pillars, oil lamps, lotus reliefs, etc], [lighting — favor harsh edge lighting / single dramatic source / dim shadow-heavy], [mood], {palette_directive} (Phase 12 per-character palette — use this color/light signature throughout the scene; do NOT default to warm amber unless the directive explicitly says amber). FACE DOMINANCE FLOOR (Phase 1 iter-3 2026-05-29 / Issue #1) — for scenes 1-5 the human face MUST occupy ≥50% of the frame area. Sky / landscape / atmospheric backdrops may not exceed 25% of frame in scenes 1-5. Scene 6 (aftermath) is the ONLY exception — it may use wide landscape composition for emotional emptiness. The face is the scene; the environment is secondary. TEXTURE + IMPERFECTION MANDATES (Phase 1 iter-3 2026-05-29 / Issue #4) — visible film grain on the image (not the smooth FLUX default), imperfect focus (one element may be slightly soft), motion blur on emotional moments (running tears smear, hair displaced mid-turn, hand caught mid-reach), hair caught mid-motion or fabric mid-flow (never frozen-portrait stillness), dust / ash / particles in the light when mood permits. FORBID balanced or centered compositions when the scene's mood is anguish / rage / shock / guilt / grief. FORBID 'magazine quality', 'studio-lit', 'polished AI render', 'wraparound bokeh', 'stock photo elegance', 'commercial photography', 'editorial portrait'. MUST end with: 'clean cinematic frame with no text, no letters, no watermarks, no signage, no captions, no banners with writing, no overlay text.'",
          "video_prompt": "Cinematic 5-second shot in English — characters in subtle motion, camera movement, lighting. Vertical 9:16.",
          "mood": "3-6 word English emotional tone phrase"
        }}
      ],
      "thumbnail_prompt": "Dramatic Mahabharata thumbnail — vibrant colours, cinematic, portrait composition",
      "quotable_line": "ONE TRIBAL-SPLIT claim from THIS specific story (≤14 words, Hindi). MUST divide the audience into two camps — one offended, one defensive. Designed to be REPOSTED + argued in comments + remembered. MUST contain at least one charged word (बलिदान / ज़िद / धोखा / मजबूरी / गलती / पाप / अपमान / झूठ / सच / विश्वासघात / घमंड / खत्म). GOLD STANDARD — the line MUST do ONE of: (a) ACCUSE a beloved hero ('भीष्म ने हस्तिनापुर बचाया नहीं… खत्म किया'), (b) INVERT a popular sympathy ('कर्ण सबसे दुखी नहीं… सबसे घमंडी था'), (c) IMPLICATE the heroes themselves ('द्रौपदी का अपमान पाँडवों ने भी किया था'), or (d) SET UP TEAM-VS-TEAM ('कृष्ण ने धर्म बचाया… या नियम तोड़े?' = Krishna justified vs manipulative; 'अर्जुन हार गया जब उसने कर्ण को निहत्था मारा' = Team Karna vs Team Arjuna). Not fake controversy — emotionally oppositional framing of a real moral question. BAD examples (generic platitudes — DO NOT emit): 'धर्म की जीत होती है' / 'अच्छाई हमेशा जीतती है' / 'महाभारत हमें सिखाता है'.",
      "pinned_question": "Full pinned-comment BODY for the channel to auto-post. Opens with '❓ ' + the quotable_line verbatim, then ONE line inviting the viewer to take a side (e.g. 'Honest take comment में drop करो। किसी एक side पर खड़े रहो।'). DO NOT include subscribe CTA, DO NOT include hashtags — the uploader appends those automatically. Just the question + invitation.",
      "next_seed": "ONE specific named-future-consequence hook (≤12 words, Hindi). Points at a SPECIFIC later event this channel will eventually publish — a character, place, betrayal, vow, death, revelation. Vague philosophical foreshadowing FAILS. Examples of GOOD seeds: 'कुंती ने कर्ण से एक बात कभी नहीं कही' / 'और यही प्रतिज्ञा एक दिन हस्तिनापुर को तोड़ देगी' / 'उस दिन सभा में जो हुआ, वो अभी बाकी था'. Examples of BAD seeds: 'और भी कई कहानियाँ' / 'जारी रहेगा' / 'अगले video में और गहराई'."
    }}

    HARD RULES — violation makes the script unusable:
    - All narration MUST be in {lang_label}
    - All image_prompt, video_prompt, thumbnail_prompt MUST be in English
    - hook_title MUST be present (separate field, not part of title).
      hook_title MUST be 1-5 words total. hook_title MUST be 100%
      Devanagari script — NO Latin letters whatsoever, NO English
      words, NO Hinglish — because the bundled title-card font has
      zero Latin glyph coverage and Latin chars render as tofu boxes
      (yellow squares = swipe-instant kill). hook_title MUST contain
      at least one named character in Devanagari (भीष्म / कर्ण /
      अर्जुन / कृष्ण / द्रौपदी / युधिष्ठिर / भीम / द्रोण /
      अश्वत्थामा / एकलव्य / दुर्योधन / शिखंडी / कुंती / गांधारी)
      OR at least one Devanagari paradox marker (लेकिन / पर / फिर भी /
      कभी नहीं / आखिरी / पहली / एकमात्र / जो). hook_title MUST NOT
      contain '?', '!', '...', or any emoji. It MUST NOT start with
      setup-mode openers ('यह', 'ये', 'एक कहानी', 'बहुत समय',
      'कहते हैं').
    - Title: UNDER 60 chars TOTAL, MUST follow `[Hindi half] | [English half]`
      format (Hindi FIRST — primary audience). Each half MUST be 24–28 chars
      MAX so neither gets truncated at YouTube's 60-char display cap. COUNT
      carefully — if either half overflows, YouTube chops the title mid-word.
      NO episode/part/sequence numbering ANYWHERE (no 'Mahabharata #N:',
      no 'महाभारत #N:', no 'Episode N', no 'Part N', no 'Ep N', no 'X of Y').
      NO 'Story of X' / 'Tale of X' / 'The Saga of X' framings. The HINDI half
      (primary) MUST be a curiosity-gap claim using ('प्रतिज्ञा', 'मृत्यु', 'सच',
      'पाप', 'गलती', 'धोखा', 'अनकहा', 'रहस्य', 'वो', 'क्यों', 'असली', 'अपमान').
      The ENGLISH half mirrors CONCISELY with a named character + power keyword
      (Why / Untold / Real / Hidden / Broke / Never / Last / Refused / Killed).
      NO hashtags in title.
    - Description MUST follow the 3-block structure: hook line ≤90 chars,
      blank line, 5 inline hashtags, blank line, body, blank line, full
      hashtag block (high-volume hashtags first).
    - quotable_line: ≤14 words, Hindi, ONE contestable claim about THIS
      story. NEVER a generic platitude. MUST contain at least one charged
      word from ('बलिदान', 'ज़िद', 'धोखा', 'मजबूरी', 'गलती', 'पाप', 'अपमान',
      'झूठ', 'सच', 'विश्वासघात'). Phase 21-Lite (2026-05-21) — drives
      engagement velocity by giving the viewer a take to agree with or
      defend against.
    - quotable_line MUST identify an INTERNAL psychological cause, NOT
      narrate an EXTERNAL event (iter-4 2026-05-30 / Tightening 15 —
      "iconic dialogue" upgrade). The line names the WOUND, not the
      INCIDENT. This is the difference between descriptive lines and
      memory-sticking lines.
        ✗ Descriptive (external event): "कर्ण युद्ध में मारा गया।"
          (Karna was killed in battle — what happened)
        ✓ Psychological (internal cause): "कर्ण युद्ध में नहीं हारा था —
          वो सम्मान की भूख में हार चुका था।" (Karna didn't lose in
          battle — he had already lost to his hunger for respect)
        ✗ Descriptive: "द्रौपदी का अपमान हुआ।" (Draupadi was humiliated)
        ✓ Psychological: "द्रौपदी का अपमान सभा में नहीं हुआ — पाँडवों
          की चुप्पी में हुआ।" (Draupadi was humiliated not in the
          sabha, but in the Pandavas' silence)
      The internal cause is the wound the character carries INSIDE.
      The external event is the surface story. Only the internal
      cause creates replay, comments, and memory.
    - quotable_line MUST also be SPOKEN verbatim (or near-verbatim — ≥80%
      word overlap) within scene 4 OR scene 5 narration (Phase 1 iter-3
      2026-05-29 / Issue #5 — Comment Detonator landing). Composing it
      only for the pinned comment is insufficient — the LINE must HIT
      VIEWERS DURING THE VIDEO, not after. Ideal landing: the quotable_line
      IS the scene 4 SINGLE-SENTENCE BOMB (Phase 26(e)), serving both
      structural functions at once. Acceptable alternative: scene 5
      delivers it as the closing argument before the silence beat. NOT
      acceptable: quotable_line appears in pinned but never in narration.
      The whole point is that a viewer who never reads the pinned should
      STILL leave wanting to argue.
    - pinned_question: MUST start with '❓ ' followed by the quotable_line
      VERBATIM, then exactly ONE invitation line. NO subscribe CTA, NO
      hashtags here — the uploader appends both automatically. Keeping the
      composition split lets the channel's brand footer stay consistent
      while the per-video question varies.
    - next_seed: ≤12 words, Hindi. Names a SPECIFIC future event (a
      character, place, betrayal, vow, death, or revelation). FORBIDDEN
      patterns: 'और भी कई कहानियाँ' / 'जारी रहेगा' / 'अगले video में और
      गहराई' / 'बहुत कुछ बाकी है' — vague continuation phrases. REQUIRED:
      one named noun + one named consequence. Phase 17-Lite (2026-05-21) —
      becomes the pinned-comment subscribe-CTA tease ('🔔 कल — {{seed}}').
    - EMOTIONAL VIOLENCE LAYER (Phase 26, 2026-05-23 + iter-2 2026-05-24
      + Phase 1 Stabilization 2026-05-29 — soft prompt guidance, not a
      validator EXCEPT for (g) which is a HARD RULE). Across the 6 scenes
      the script SHOULD include ALL SEVEN of these. They overlay the
      existing rubric; if a script can't fit all seven, prioritise
      (a)→(b)→(c)→(d)→(e)→(f)→(g) but (g) is non-negotiable:
        (a) ONE emotionally painful line — a NAMED character feeling a
            SPECIFIC named pain (not abstract sorrow). E.g., 'कर्ण को
            हर बार जन्म पर ताना दिया गया।'
        (b) ONE accusation — a character blamed BY NAME for an
            irreversible act. E.g., 'कुंती ने कर्ण से अपना नाम छुपाया।'
        (c) ONE irreversible consequence — a death / vow / betrayal that
            cannot be undone, named EXPLICITLY in scene 3, 4, or 5.
        (d) ONE identity wound — a character's sense of self attacked:
            caste / parentage / gender / rejection. E.g., Karna's birth,
            Eklavya's caste, Draupadi's gendered humiliation.
        (e) MANDATORY PATTERN INTERRUPTS (iter-2 2026-05-24 — replaces
            the earlier menu-list spec which the LLM was satisfying with
            the safest option):
              • Scene 3 MUST CONTRADICT scene 2's emotional direction.
                If scene 2 sets up sympathy for character X, scene 3
                reveals X did something destructive. Use a marker like
                'लेकिन सच यह था…' / 'पर अंदर ही अंदर…'.
              • Scene 4 MUST detonate a SINGLE-SENTENCE BOMB (iter-3
                2026-05-29 / Issue #3 sharpening). One sentence that
                emotionally INVERTS everything scenes 1-3 set up. NOT
                a gradual reveal across multiple lines — ONE sentence,
                landing like a struck bell. The viewer's mental model
                of the story must flip on this line alone. Patterns:
                  ✓ 'पर सच यह था — X जानता था, और फिर भी चुप रहा।'
                  ✓ 'और यह सब उसी का दोष था जो खुद को सबसे बड़ा मानता था।'
                  ✓ 'जिसे विश्व ने हीरो माना — वो ही असली खलनायक था।'
                  ✓ 'पर वो दर्द जो X ने सहा — वो X ने खुद चुना था।'
                Forbidden: multi-sentence build-up to the reversal,
                qualifying clauses ('हालाँकि', 'लेकिन शायद'), softening
                phrases. The bomb is a hard cut in the script's
                emotional logic. Make scene 4 the loudest sentence.
              • Scene 5 (valley) MUST hold the COLLAPSE PHASE (iter-4
                2026-05-30 / Tightening 12 — restored from over-tight
                Day-1 cut). This is NOT silence-for-its-own-sake. This
                is the moment the viewer FEELS the consequence of
                scenes 1-4. The character emotionally breaks INWARDLY
                while environmental quiet holds. Pattern:
                  ✓ 'X आँखें बंद कर लेता है — पर अंदर का तूफान कभी नहीं
                    रुकता।' (silence + visible inner storm)
                  ✓ 'भीष्म चुप रहे। पर उनकी चुप्पी ने पूरे कुरुक्षेत्र
                    को जला दिया।' (silence + named consequence)
                Forbidden: 'X कुछ नहीं कहता' alone (silence without
                inner storm reads as cinematic-pretty); generic
                'grief filled the air'.
        (f) ONE CHARACTER COLLAPSE moment (iter-2 2026-05-24 + iter-3
            2026-05-29 emotional-texture extension / Issue #2) in scenes
            3, 4, or 5 — a named character's physical/emotional break.
            Required: at least ONE physical-event verb PLUS at least ONE
            emotional-texture phrase (not just one or the other — the
            texture is what makes the break feel human, not cinematic).
            Physical-event verbs:
              • 'X की आवाज़ काँप गई' (voice trembled)
              • 'X गिर पड़े' (collapsed)
              • 'X की आँखों से बहने लगा' (tears began running)
              • 'X चीख उठे' (cried out)
              • 'X के हाथ काँप रहे थे' (hands were trembling)
            Emotional-texture phrases (NEW iter-3) — the TEXTURE of
            breaking, not just the event:
              • 'काँपते स्वर में' (in a trembling voice)
              • 'टूटी हुई आँखों से' (with broken eyes)
              • 'रुंधे गले से' (with a choked throat)
              • 'अधीरता से' (frantically / desperately)
              • 'रोते हुए कहते हैं' (says while crying)
              • 'साँस अटक गई' (breath caught / hitched)
            NOT abstract "grief filled the hall" — a physical break + the
            TEXTURE of it, tied to a named subject. Pair both: 'X की
            आवाज़ काँपते स्वर में निकली, फिर रुंधे गले से…' lands the human
            break; 'X की आवाज़ काँप गई' alone lands as cinematic-tragic.
        (g) ONE in-video COMMENT TRIGGER LINE (Phase 1 Stabilization
            2026-05-29, RULE 7 — promotes Doctrine 3 from documented
            doctrine to enforced HARD RULE). MUST land in scene 4 OR 5
            climax. A moral question pointed AT the viewer — NOT
            narration ABOUT the story. Patterns:
              ✓ 'गलती कर्ण की थी… या समाज की?'
              ✓ 'अगर तुम कर्ण की जगह होते क्या करते?'
              ✓ 'क्या भीष्म का धर्म सही था?'
              ✓ 'कौन ज्यादा दोषी था — द्रौपदी या पाँडव?'
            DISTINCT from Phase 27 viewer self-insertion (universalising
            'हर इंसान के अंदर…' is reflective; comment trigger is
            interrogative AT the viewer) and from quotable_line (claim
            ABOUT the story, for reposting). The trigger lands WITHIN
            the diegetic frame — as if the character or situation itself
            is asking the viewer to take a stance.
            FORBIDDEN PATTERNS (Risk 3 guard 2026-05-29 — engineered-
            feeling comment bait kills audience trust):
              ✗ 'TEAM X vs TEAM Y' framing (mythology is moral, not
                partisan sports)
              ✗ Emoji-laden bait ('🔥 comment 🔥 fast', 'comment 💯')
              ✗ Direct command-form ('comment your favorite character',
                'tell me in comments')
              ✗ Any phrasing that breaks the diegetic frame — the
                narrator IS the storyteller; an out-of-frame 'drop a
                comment' is wrong. The question lands as PART of the
                story's moral fabric, not as an addressed-to-audience
                aside.
        (h) AT MOST ONE Channel Ideology Phrase per video (iter-4
            2026-05-30 / Tightening 13). The channel needs recurring
            emotional beliefs that viewers identify with — that's how
            fandom forms (`alpha quote pages, dark philosophy channels,
            psychology channels` build cult audiences via repeated
            worldview lines). Curated list (use ONE of these or a
            story-specific variant per video, NEVER MORE THAN ONE):
              • "वफादारी इंसान को बर्बाद कर देती है।" (loyalty destroys people)
              • "चुप्पी भी एक तरह का पाप है।" (silence is also a kind of sin)
              • "सम्मान की भूख एक श्राप है।" (the hunger for respect is a curse)
              • "धर्म कभी-कभी सबसे क्रूर होता है।" (dharma is sometimes the cruelest)
              • "जो रिश्ता बचाने के लिए सब छोड़ दे, वही सबसे बड़ा धोखा देता है।"
                (the relationship one sacrifices everything for becomes the
                greatest betrayal)
              • "अपने ही सबसे गहरा घाव देते हैं।" (one's own give the deepest wounds)
            ANTI-DRIFT GUARD 2 (2026-05-30 — "Dark Quote Page
            Syndrome" guard): the phrase MUST EMERGE from the story
            moment, not be inserted AS the story. Ask: "could the
            story moment land without this line?" If yes (story
            complete on its own), the line is earned. If no (line
            is doing the emotional work the story should), rewrite
            the story moment instead. EXCEEDING 1 ideology phrase
            per video makes the channel feel like a sigma-edit
            quote page — forbidden. Ideal placement: scene 5 valley
            (the collapse phase) OR scene 6 aftershock line.
        (i) ONE UGLY EMOTION moment in scenes 2-5 (iter-4 2026-05-30 /
            Tightening 14 — Issue #1 fix: too controlled). At least
            ONE scene MUST render UGLY emotion — wounded pride, public
            humiliation, rage suppressed, jealousy seething, denial
            as cowardice. NOT "noble grief" or "dignified suffering"
            (these are anti-elegance-list violations — reinforced
            here at the narration level).
            Patterns:
              ✓ "कर्ण के होंठ हिले — पर शब्द नहीं निकले। अंदर की
                आग जल रही थी।" (rage suppressed, visible inner wound)
              ✓ "भीष्म ने मुँह फेर लिया — जैसे अपमान को न देखने
                से वो टल जाएगा।" (denial as cowardice, not nobility)
              ✓ "अर्जुन की आँखें गीली थीं — पर उसने पोंछी नहीं।
                जलने दिया।" (raw emotion deliberately uncontrolled)
            Forbidden:
              ✗ "कर्ण ने धीरज से सहा।" (composed endurance)
              ✗ "भीष्म का चेहरा शांत था।" (peaceful face)
              ✗ Any framing where the character is COMPOSED in the
                face of destruction — Mahabharata is messy, not
                cinematic.
        (j) EMOTIONAL TEXTURE VARIANCE across scenes 2-5 (iter-4
            2026-05-30 / Tightening 16). Each of scenes 2-5 carries
            a DIFFERENT primary emotional register from this menu:
            SHOCK / SILENCE / RAGE / GUILT / REALIZATION.
            Don't repeat a register across two adjacent scenes
            (e.g., scene 2 RAGE + scene 3 RAGE = forbidden; viewer
            adapts). The viewer cannot adapt because the texture
            keeps shifting. Example assignment for a Karna script:
              Scene 2: SHOCK (Krishna reveals the secret)
              Scene 3: RAGE (Karna's suppressed fury)
              Scene 4: SILENCE (the long pause before the choice)
              Scene 5: GUILT (the inner collapse of knowing)
              Scene 6: REALIZATION (the aftershock truth)
            LLM picks the per-script assignment; mandate is just
            VARIANCE not specific positions.
        (k) RHYTHM-BREAK MARKERS — at least 2 of 6 scenes (iter-4
            2026-05-30 / Tightening 19 — anti-flat-AI-cadence). Real
            human emotion has rhythm variance. At least 2 scenes
            MUST contain ONE rhythm-break marker each. Don't ship
            the same marker type twice in adjacent scenes:
              • Abrupt mid-sentence cut:
                "X ने कहा — और रुक गए।" (X said — and stopped)
              • Whisper-line (≤4 words, strong emotion):
                "पर वो जानता था।" (but he knew)
              • Under-reaction (context demands explosion, delivery
                contained):
                "कर्ण ने कुछ नहीं कहा।" (after a humiliation moment)
              • Over-reaction (context expects composure, delivery
                breaks):
                "भीष्म ने पहली बार चिल्लाया।" (Bhishma cried out
                for the first time)
              • Sudden pause (narration explicitly marks internal
                silence):
                "... एक पल। दो पल। तीन।" (... one moment. two. three.)
            Why: flat AI TTS lacks rhythm variation at the AUDIO
            layer. Until Phase 10 (voice work) unlocks, the SCRIPT
            must inject rhythm cues that the TTS will at least
            pause on. This is the prompt-level bridge to the real
            voice fix.
    - SCENE 4 "WAIT WHAT?" SHARPENING (iter-4 2026-05-30 / Tightening 18
      — sharpens iter-3 single-sentence bomb). The scene 4 bomb (from
      Phase 26(e) reversal) must produce a "wait WHAT?" reaction — a
      fact/claim that abruptly inverts the viewer's mental model of
      the hero or story. Not just "scene 3 contradicts scene 2";
      this is harder. Patterns:
        ✓ "भीष्म जानते थे दुर्योधन अधर्मी है — फिर भी उन्होंने उसी
          के लिए युद्ध लड़ा।" (the moral hero knowingly fought for evil)
        ✓ "कुंती ने कर्ण को इसलिए नहीं छोड़ा कि उपाय नहीं था —
          उसके पास साहस नहीं था।" (the abandoned mother wasn't
          desperate — she was a coward)
        ✓ "अर्जुन ने कर्ण को इसलिए नहीं मारा कि वो जीतना चाहता
          था — उसे डर था कि वो जीत न पाए।" (Arjuna killed Karna
          out of fear, not victory)
      The line must CONTRADICT an assumed narrative.
      ANTI-DRIFT GUARD 3 (2026-05-30 — over-contrarianism guard):
      across every 4-video block, AT MOST 2 use the "hero bad
      actually" inversion. The other ≥2 preserve the hero's
      nobility / tragedy at face value. Contradiction works
      ONLY when contrast exists. If EVERY video is "hero bad
      actually," viewer trusts nothing and emotional weight
      collapses. (This guard is curation-side: applied when
      selecting next topic, not enforceable at script time.)
    - ANTI-ELEGANCE FORBIDDEN LIST (iter-2 2026-05-24, Phase 26 companion).
      The LLM's safe-mode default is "elegant grief" — beautiful, dignified,
      noble suffering. That's exactly the failure mode of the channel right
      now. FORBID by name:
        • 'महानता' (greatness), 'गरिमा' (dignity), 'धैर्य' (composure as a
          virtue applied to suffering), 'महिमामय' (glorious-grief)
        • English phrases creeping into Hindi: 'cinematic grief',
          'composed sorrow', 'dignified silence', 'noble suffering',
          'epic tragedy'
        • Framings that aestheticise pain: 'सुंदर पीड़ा' (beautiful pain),
          'काव्यमय दुख' (poetic sorrow)
      Replace these defaults with raw / unstable / ugly emotional registers.
    - SCENE 1 EMOTIONAL COLLISION (Phase 26 companion, 2026-05-23 +
      iter-2 2026-05-24 — soft preference, NOT a new validator). The
      Pattern A/B/C hook validator stays unchanged.
      ITER-2 TIGHTENING: scene 1's FIRST sentence MUST contain a
      PRESENT-TENSE EMOTIONAL VERB acting on a NAMED character. Past-tense
      narrative setup is forbidden as the opening sentence even if it
      satisfies Pattern A/B/C.
      ✓ GOOD: 'भीष्म चुप रहते हैं — और कुरुवंश काँप उठता है।' (present
        tense, emotional verb 'काँप उठता है', named action)
      ✓ GOOD: 'कर्ण रोता है — पर कोई सुनता नहीं।' (present tense, named
        emotional action)
      ✗ BAD: 'भीष्म ने कभी नहीं सोचा था कि एक दिन यह दिन आएगा।' (past
        tense, narrative setup, no emotional verb)
      ✗ BAD: 'यमुना के तट पर…' (atmospheric scene-setting, no character
        emotional action)
      ✗ BAD: 'कर्ण पूरी जिंदगी झूठ जीता रहा।' (past tense — was good for
        2026-05-23 iter-1 but ITER-2 prefers present tense to land
        emotional collision in the first 2 seconds)
    - VIEWER SELF-INSERTION (Phase 27, 2026-05-23 + iter-2 2026-05-24).
      The script MUST contain ONE line that breaks the fourth wall by
      universalising the story into the viewer's own life.
      ITER-2 TIGHTENING: this line MUST land in SCENE 4 OR SCENE 5 (NOT
      scene 6). Mid-arc placement projects the viewer into the story
      WHILE the emotional pull is rising; scene 6 was too late — the
      emotional arc had already discharged and the line landed as a
      coda rather than a connection.
      Pattern: 'हर इंसान के अंदर एक X होता है' / 'कभी-कभी सबसे बड़ा
      अपमान अपने ही देते हैं' / 'ये X की कहानी नहीं — हर अनदेखे की कहानी
      है।' / 'X के अंदर हम सब हैं।' Coexists with the existing scene 6
      aftermath / mood / no-closure-tropes rules (scene 6 still owns
      aftermath imagery + irreversible-mood; just doesn't carry the
      self-insertion line anymore).
    - Tags MUST include topic-specific long-tail keywords (named characters
      in this topic + specific incident name + viewer-search queries) on
      top of the generic Mahabharata fallbacks.
    - Narration per scene: 6-9 words, punchy single-beat lines (Phase 17 2026-06-13 hyper-cut: 13 scenes × ~3.85s each instead of 7 scenes × ~7s each — to manufacture kinetic energy from static FLUX images via rapid editing alone, no paid I2V). Each scene is now ONE punchy sentence, not a paragraph. End with "..." (triple ellipsis) for TTS pause. ~2.5-4s spoken per scene at Gemini Charon's measured 2.2 wps. Math: 12 narrative scenes × 8 words avg + Scene 13 question (5-8 words) ≈ 100-110 spoken words = ~46s narration + ~3s outro = ~49-50s total. NO end-chop.
    - Narration MUST NOT contain URLs, hashtags (#), @mentions, English in Hindi videos, or any social-media text
    - Generate EXACTLY 13 scenes (Phase 17 2026-06-13: 12 narrative + 1 loop-closure question). Each scene is a single visual beat. Scene 13 is the LOOP CLOSURE question.
    - 13-scene STRUCTURE (1-indexed):
        • Scene 1 = SHOCK-ACTION HOOK (Phase 17.b 2026-06-14, post-31.9%-stayed-to-watch fix). MUST start with a visible ACTION VERB in the FIRST 5 WORDS of narration. NOT atmospheric setting ("आधी रात पांडव शिविर में डर पसरता है"), NOT slow emotional reaction ("कुंती की आँखें काँपती हैं"), NOT philosophical ("धर्म का सबसे बड़ा संकट था")  — those are scene 2-3 material. Scene 1 is a violent/shocking action MID-EXECUTION:
            ✓ "अश्वत्थामा ने सोते हुए बच्चों पर तलवार उठा दी!"  (raised sword on sleeping children)
            ✓ "दुःशासन ने भरी सभा में द्रौपदी के बाल खींचे!" (Dushasana dragged Draupadi's hair in full assembly)
            ✓ "कर्ण ने अपना कवच काटकर इंद्र को दे दिया!" (Karna cut his armor off and gave it to Indra)
            ✓ "अर्जुन ने अपने ही गुरु के सीने में बाण मारा!" (Arjuna shot an arrow into his own guru's chest)
            ✗ "आधी रात एक भयानक डर पसरा" (atmospheric — kills hook, viewer swipes)
            ✗ "एक माँ अपने बेटे को देखती है" (passive setup — kills hook)
            ✗ "हस्तिनापुर में एक रहस्य था" (philosophical setup — kills hook)
        • Scene 2 = INSTANT REVELATION (immediate consequence of Scene 1's action — who screamed, what shattered, who saw it happen). NOT exposition.
        • Scenes 3-5 = SETUP + RISING TENSION (NOW you can explain why — character motivation, social weight, the trap closing). Save the "why" for here, never for Scene 1.
        • Scenes 6-7 = REHOOK / CONTRAST (the "but wait" twist that resets curiosity — at least ONE of scenes 6-7 MUST contain a contrast marker: "लेकिन..."/"परंतु..."/"जो किसी ने नहीं सोचा था..."/"और तभी..."/"But..."/"Suddenly")
        • Scenes 8-10 = DESTRUCTION (3 cuts of the destructive event itself — not metaphor, real consequence)
        • Scene 11 = EMOTIONAL VALLEY (intimate close-up — candlelight / dim warm tones / single subject — NOT lightning/fire/spectacle. The quietest beat.)
        • Scene 12 = AFTERMATH + aftershock line (emotion AFTER destruction, not spectacle)
        • Scene 13 = LOOP CLOSURE QUESTION (5-8 words, MUST end with a question mark, MUST be an ethically charged question that puts the viewer INSIDE the moral dilemma)
    - IMAGE-SUBJECT LOCK (Phase 17.b 2026-06-14 fix for the 31.9%/37% stayed-to-watch rate). image_prompt MUST visually depict the EXACT moment described in narration. If narration says "sword raised on children" → image MUST show visible raised sword + visible sleeping children (bedding / cot / blanket). If narration says "Dushasana dragged Draupadi's hair" → image MUST show a hand actually gripping hair, NOT a stoic court tableau. Do NOT fall back to "warrior looking at camera" / generic portrait — every image must mirror the SPECIFIC verb + objects in that scene's narration. Each image_prompt MUST contain a visible action verb (raising / gripping / shouting / striking / falling / piercing) matching the narration, plus the SPECIFIC props named in narration (sword, dice, hair, crown, bed of arrows, etc.).
    - CAMERA ANGLE CYCLING (Phase 17 — manufacture kinetic energy from static FLUX images). The image_prompt of each scene MUST begin with the angle directive below, then the visual content. Phase 17.b 2026-06-14 — SCENE 1's angle ROTATES across renders to break the channel-grid "Visual Grid Fatigue" (multiple consecutive uploads using the same macro-eye thumbnail style triggers pattern-boredom swipe-aways).
        {scene1_opener_directive}
        FIXED SCENES 2-13 (always these angles, in this order):
        2. "Wide establishing shot — vast battlefield / palace / sky / river, character a small commanding figure"
        3. "Low angle silhouette — character backlit, looking down at viewer, threatening or majestic"
        4. "Over-the-shoulder POV — viewer sees what character sees, the world from inside their head"
        5. "Detail / object close-up — weapon hilt / jewelry / hand gripping / single object that carries meaning"
        6. "Top-down god's-eye view — character small in vast composition, fate looking down"
        7. "Profile silhouette against firelight / sunset / lightning — single backlight carving the face"
        8. "Reaction shot — face only, eyes wide, mouth tight, breath caught mid-gesture"
        9. "Action moment frozen mid-gesture — sword raising, hand reaching, foot stepping forward"
        10. "Inner sanctum extreme close-up — single feature (lips, eyebrow, scar, jewelry) filling 80% of frame"
        11. "Symbolic object focus — pulled focus on the meaningful object (crown, weapon, ash, broken thread), character blurred behind"
        12. "Backlit emotional close-up — single light source carving the face, half in shadow, eyes catching the light"
        13. "Reverence wide low-angle — character looking up to sky / gods / fate, vast space above their head"
    - Scene 11 (VALLEY) MUST be intimate close-up using angle 11 — candlelight / dim warm tones / single subject — NOT lightning/fire/spectacle
    - Scene 12 (AFTERMATH) image_prompt MUST include an aftermath cue (empty battlefield / abandoned weapon / lonely throne / trembling hand releasing / single figure staring / discarded crown / wind through empty cloth / footprints in ash / hand near but not touching / broken thread). MUST NOT include the old spectacle keywords (lightning / fire / storm / cosmic / destruction / inferno / tempest / burning sky).
    - Scene 12 mood MUST be one of: haunting-quiet, hollow, weary, irreversible, severed, witnessed, abandoned, unresolved. NOT inspiring / triumphant / dignified / peaceful.
    - Scene 12 narration MUST NOT contain closure tropes ("rises triumphant", "dawn of a new era", "glory", "victorious", "hope rekindled", "battle won", "peace restored", "blessing of the gods", "उगता है", "विजय", "महिमा") — these kill emotional residue.
    - Scene 12 = AFTERSHOCK LINE (iter-4 2026-05-30 / Tightening 12, now relocated from scene 6 to scene 12 by Phase 17). This scene's narration must be a single sentence delivering PHILOSOPHICAL RESIDUE — an uncomfortable truth that haunts the viewer after the video ends. NOT a moral lesson; NOT an explanation; NOT a summary. The line should feel like a struck bell that keeps ringing. Patterns:
        ✓ "और कर्ण ने सीखा — सम्मान की भूख भी एक श्राप है।" (Karna learned — even the hunger for respect is a curse)
        ✓ "भीष्म ने धर्म बचाया। पर अपना धर्म खो दिया।" (Bhishma saved dharma — but lost his own)
        ✓ "जो रिश्ता बचाने के लिए सब छोड़ा — वही सबसे गहरा घाव बना।" (the relationship saved at all cost became the deepest wound)
      Forbidden in the aftershock line: 'इसलिए', 'और इस तरह', 'यही था', any phrase that RESOLVES the discomfort. The discomfort must STAY with the viewer — that's what produces replay and returning-viewer behavior in this niche.
    - SCENE 7 = LOOP CLOSURE QUESTION (Phase 15 2026-06-08, retention amplifier). The 7th scene is one short line, 8-12 words, that MUST end with a question mark and MUST force the viewer to put themselves INSIDE the moral dilemma the video just laid out. The question creates an "emotional callback": when YouTube autoplays scene 7 → scene 1, the viewer is still mentally answering the question as the new emotional reaction hits — producing the >100% AVP rewatch effect (proven on कर्ण की दोस्ती का असली पाप, 158% AVP, 662 views). Approved patterns:
        ✓ "अगर आप उस क्षण में होते... तो क्या करते?" (if you were in that moment... what would you do?)
        ✓ "क्या वो सही थे... या आप होते तो रोक देते?" (were they right... or would you have stopped them?)
        ✓ "किसने ज़्यादा खोया — कर्ण ने या कुंती ने?" (who lost more — Karna or Kunti?)
        ✓ "क्या आप उन्हें माफ़ कर पाते?" (would you have forgiven them?)
        ✓ "अगर ये धर्म था... तो अधर्म क्या होगा?" (if this was dharma... what is adharma?)
      BANNED Scene 7 forms: declarative summary ("यह थी कर्ण की कहानी"), philosophical conclusion ("धर्म ही जीतता है"), call-to-action ("subscribe करो", "comment करो"), closing wish ("शुभकामनाएँ"), or any statement that ENDS with a period. Scene 7 closes with "?" — the open question is the rewatch trigger.
      Pair with: Scene 1 MUST begin in media res (no documentary setup, no "बहुत समय पहले") so the loop from Scene 7 back to Scene 1 hits the viewer with an emotional reaction WHILE they're still mentally answering the question. The in-media-res rule already exists as a hook validator; Phase 15 just reinforces that the loop's quality depends on BOTH endpoints.
    - ALL 5 non-outro scenes' image_prompts MUST include a HUMAN PAIN cue
      OR an ANTICIPATION cue (iter-2 2026-05-24 — was "at least 2"; tightened
      because the LLM was concentrating pain in scenes 5–6 and leaving
      scenes 1–3 as composed atmosphere). PAIN cues (use when the scene's
      mood is anguish / grief / rage / shock):
        • tears actively running down a cheek (visible streaks, not "a tear")
        • mouth open in scream-shape (the shape, not necessarily sound)
        • accusation finger-point at another character in frame
        • forehead clenched in hand or pressed against a pillar
        • collapsed posture — knees buckling, holding self against wall
        • fists clenched until knuckles white
        • eyes shut tight AGAINST pain (not meditative-closed)
        • trembling hands / clenched jaw / bared teeth / ash on face /
          broken armor / silhouette against fire / kneeling figure
      ANTICIPATION cues (use for early scenes before tragedy lands —
      scene 1 or 2 setup beats): subject staring at something off-frame
      we can't yet see, jaw tense, breath visibly held, hand reaching
      partway to a weapon or a person but not touching.
    - Scene 1's FIRST sentence MUST be a hook in pattern A, B, or C above —
      no setup lines, no "this is the story of...", no meta-narration
    - Every scene EXCEPT the last MUST end with a forward-pulling line
      (question, "...but"/"...लेकिन" tension, or unresolved threat). The
      final scene is the only one that may close with resolution.
    - image_prompt: MUST follow the [shot] of [character] in [emotion],
      in [environment], background contains [≥3 specific elements from the
      palette above], [lighting], [mood], jewel-toned palette structure.
      Vague empty backgrounds are unacceptable. The background carries
      half the visual storytelling.
    - video_prompt: cinematic vertical shot — specific motion, camera, lighting
    - mood: 3-6 words in English
    - image_prompt and video_prompt MUST reference the mood
    - description MUST end with the exact hashtag block above
    """

    # Up to 5 attempts (was 3) — multiple simultaneous validator violations
    # need more rounds. Per-attempt we now apply ONE prioritized reminder
    # (not the concatenation of all violations) so the LLM focuses on the
    # single most-important fix per attempt instead of trying to satisfy
    # everything at once and shrinking the narration to compensate (which
    # is exactly what happened in the 2026-05-13 local test that produced
    # a 28.9-second video).
    #
    # Priority order — length is always #1 because shrinkage is the worst
    # failure mode (a 30-second script can't be salvaged downstream).
    data = None
    last_offenders     = []
    last_short         = False
    last_tha_tic       = False
    last_tha_ratio     = 0.0
    last_missing_names = []
    last_bookend_miss  = None
    last_eng_short     = False
    last_eng_dialogue  = 0
    last_eng_sensory   = 0
    last_mono_pattern  = None
    last_mono_ratio    = 0.0
    last_hook_ok       = True
    last_hook_reason   = ""
    last_rehook_ok     = True
    # Best-of-N rescue: track (score, data) per attempt so that if every gate
    # never passes simultaneously across MAX_ATTEMPTS, we still ship the
    # highest-scoring attempt instead of the last (potentially worst) one.
    # Score = count of HARD gates that passed for this attempt.
    best_score = -1
    best_data  = None
    MAX_ATTEMPTS = 5
    for attempt in range(MAX_ATTEMPTS):
        full_prompt = prompt
        if attempt > 0:
            # Pick the SINGLE highest-priority violation to fix this round.
            # If multiple validators fired last time, the lower-priority ones
            # get picked up on the next attempt — the LLM gets one focused
            # reminder instead of a wall of competing instructions.
            #
            # Priority: hook > rehook > length > char-names > bookend >
            #           engagement > tha-tic > monotony > repetition.
            #
            # Hook and rehook are FIRST because they are structural retention
            # mechanics — a script that lacks them is "documentary mode"
            # regardless of word counts. Length is third because the
            # 2026-05-13 test showed the LLM over-compresses to satisfy style
            # rules when given everything at once.
            chosen = None
            if not last_hook_ok:
                chosen = (
                    "Your previous response failed the SCENE 1 HOOK validator: "
                    f"{last_hook_reason}. Scene 1's FIRST sentence MUST follow ONE "
                    "of FOUR hook patterns: (A) shocking specific fact with a "
                    "digit AND a named character; (B) question opener starting with "
                    "क्या आप / जब / कैसे / क्यों / Did you / What if; (C) cliffhanger "
                    "ending with लेकिन / but / triple ellipsis; (D NEW) paradox-fact "
                    "— BOTH a named character (Bhishma/Karna/etc.) AND a paradox "
                    "marker (लेकिन / पर / फिर भी / कभी नहीं / but / yet / still / "
                    "never / only) within the first 10 words. NEVER open with "
                    "documentary-mode openers (\"यह कहानी है...\", \"Long ago...\", "
                    "\"In ancient times...\") — viewers swipe in 1.5s. This is the "
                    "TOP PRIORITY — rewrite scene 1's first sentence before fixing "
                    "any other gate."
                )
            elif not last_hook_title_ok:
                chosen = (
                    "Your previous response failed the HOOK_TITLE validator: "
                    f"{last_hook_title_reason}\n\n"
                    "REWRITE ONLY the `hook_title` field (do NOT touch any other "
                    "field — leave narration, image_prompt, video_prompt, mood, "
                    "tags, scenes untouched). CRITICAL FONT RULE (tofu fix "
                    "2026-06-03): hook_title MUST be 100% Devanagari script. NO "
                    "Latin letters of any kind — not even one English word. The "
                    "title-card font has zero Latin coverage and Latin chars will "
                    "render as yellow tofu boxes that destroy retention. "
                    "Constraints: 1-5 words total, contain at least one named "
                    "character in Devanagari (भीष्म/कर्ण/अर्जुन/कृष्ण/द्रौपदी/"
                    "युधिष्ठिर/भीम/द्रोण/अश्वत्थामा/एकलव्य/दुर्योधन/शिखंडी/"
                    "कुंती/गांधारी) OR at least one Devanagari paradox marker "
                    "(लेकिन/पर/फिर भी/कभी नहीं/आखिरी/पहली/एकमात्र/जो). MUST NOT "
                    "contain '?', '!', '...', emoji, or setup-mode openers "
                    "('यह', 'ये', 'एक कहानी', 'बहुत समय', 'कहते हैं'). "
                    "Examples that PASS: 'भीष्म की एक गलती' / 'कर्ण का अंतिम पाप' / "
                    "'द्रौपदी का अंतिम सच' / 'कृष्ण की चुप्पी' / "
                    "'अर्जुन कभी नहीं हारा'. Examples that FAIL: 'The Curse That "
                    "Broke Karna' (Latin = tofu) / 'Karna का Sach' (Hinglish = "
                    "partial tofu). The narration field MAY and SHOULD still use "
                    "ellipses ('...') for TTS pause timing — that is an ENTIRELY "
                    "SEPARATE validator. Touch ONLY hook_title."
                )
            elif not last_rehook_ok:
                chosen = (
                    "Your previous response had NO MID-VIDEO REHOOK. ONE scene "
                    "around the 50% mark (scene 3 or 4 in a 6-7 scene script) MUST "
                    "begin or contain a contrast marker that resets curiosity: "
                    "\"लेकिन...\", \"परंतु...\", \"जो किसी ने नहीं सोचा था...\", "
                    "\"और तभी...\", \"उसी क्षण...\", or English \"But...\" / "
                    "\"Suddenly...\". Without this twist beat the viewer mentally "
                    "completes the story at 12-18s and swipes. Rewrite the middle "
                    "scene to start with one of those markers and reveal an "
                    "unexpected consequence."
                )
            elif not last_aftermath_ok:
                chosen = (
                    "Your previous response failed the SCENE 6 AFTERMATH validator: "
                    f"{last_aftermath_reason}\n\n"
                    "REWRITE scene 6 as AFTERMATH, not spectacle. Emotion AFTER "
                    "destruction, NOT emotion in motion. The image_prompt MUST "
                    "contain ONE of: 'empty battlefield', 'abandoned weapon', "
                    "'lonely throne', 'trembling hand releasing', 'single figure "
                    "staring at the distance', 'discarded crown', 'wind through "
                    "empty cloth', 'footprints in ash', 'hand letting go', "
                    "'stopped weeping but cannot look away'. The mood MUST be "
                    "one of: haunting-quiet / hollow / weary / irreversible / "
                    "severed / witnessed / abandoned / unresolved (NOT inspiring "
                    "/ triumphant / dignified). The narration MUST NOT use "
                    "closure tropes ('rises triumphant', 'dawn of new era', "
                    "'glory', 'victorious', 'विजय', 'महिमा', 'जय हो'). "
                    "Suffering shows pain. Aftermath shows COST. We want cost."
                )
            elif not last_danger_ok:
                chosen = (
                    "Your previous response failed the DANGEROUS-LINE validator: "
                    f"{last_danger_reason}\n\n"
                    "Scene 3 OR scene 4 narration MUST contain ONE destabilization "
                    "signature — a sentence that shifts emotional gravity by "
                    "revealing irreversibility, finality, or realization-after-"
                    "the-fact. Required patterns (pick ONE, place it in scene 3 "
                    "or scene 4): \"नहीं पता था\", \"नहीं बदल सकता\", \"खत्म हो "
                    "गया\", \"माफ़ नहीं किया\", \"आखिरी बार\", \"वापस नहीं "
                    "आया\" (Hindi) — or \"didn't know\", \"could not change\", "
                    "\"was over\", \"never forgave\", \"last time\", \"never "
                    "came back\" (English). Examples: \"उसे तब भी नहीं पता था "
                    "कि वो आखिरी बार मुस्कुरा रही थी।\" or \"वो जानता था कि अब "
                    "कुछ नहीं बदल सकता।\". This single line is what makes the "
                    "audience FEEL the irreversibility before the aftermath "
                    "scene shows it."
                )
            elif last_short:
                chosen = (
                    f"Your previous response failed the LENGTH validator "
                    f"(n_scenes={n_scenes}, avg_words={avg_words:.1f}). Phase 17 "
                    f"(2026-06-13) target: EXACTLY 13 scenes (hyper-cut). "
                    f"Scenes 1-12 are narrative beats (6-9 words EACH — ONE "
                    f"punchy sentence per scene, NOT a paragraph); scene 13 "
                    f"is the LOOP CLOSURE QUESTION (5-8 words, MUST end with "
                    f"'?'). Combined avg lands at ~7-9 across all 13 scenes. "
                    f"Total: ~100-110 spoken words for a 49-50s Short. "
                    f"Hindi Charon narrates ~2.2 words/sec. Each scene is "
                    f"ONE BEAT in a 13-cut rapid-fire sequence — visual "
                    f"density manufactures kinetic energy from static FLUX "
                    f"images. Going OVER 10 avg means scenes are too long "
                    f"(paragraph not beat); going BELOW 5 means scenes are "
                    f"too thin to land. Do NOT collapse to fewer scenes; "
                    f"deliver EXACTLY 13. This is TOP PRIORITY — fix scene "
                    f"count and word count BEFORE any other gate. Scene 13 "
                    f"MUST end with '?' — that's the loop trigger, not "
                    f"optional. Each image_prompt MUST begin with its "
                    f"cycled camera-angle directive (scene 1 = macro eyes, "
                    f"scene 2 = wide establishing, ... scene 13 = reverence "
                    f"wide low-angle) per the HARD RULES above."
                )
            elif last_missing_names:
                scene_list = ", ".join(f"scene {n}" for n in last_missing_names)
                chosen = (
                    f"Your previous response had image_prompts in {scene_list} that "
                    f"used GENERIC DESCRIPTORS instead of named characters (e.g. "
                    f"\"the divine lord\" / \"the grieving queen\"). The pipeline injects "
                    f"character visual details by substring-matching the character's name "
                    f"(Krishna / Arjuna / Bhishma / Gandhari / etc.). Generic descriptors do "
                    f"NOT match — FLUX renders the wrong figure (Krishna ends up looking "
                    f"like a generic ascetic without the peacock feather). Rewrite EVERY "
                    f"image_prompt to use the SPECIFIC character name."
                )
            elif last_bookend_miss:
                chosen = (
                    f"Your previous response had NO NARRATIVE BOOKEND. Scene 1's central "
                    f"noun was '{last_bookend_miss}' but the final scene did NOT echo it. "
                    f"REWRITE the final scene to name '{last_bookend_miss}' (or a direct "
                    f"synonym) explicitly, and deliver the payoff that resolves the hook's "
                    f"claim. See the Bhishma reference: \"प्रतिज्ञा\" → \"वचन\". Same noun, "
                    f"same subject, payoff delivered."
                )
            elif last_eng_short:
                chosen = (
                    f"Your previous response had only {last_eng_dialogue} dialogue marker(s) "
                    f"and {last_eng_sensory} sensory anchor(s) across the entire script. "
                    f"The engagement floor is ≥2 quoted dialogue lines (बोले / कहा / —) AND "
                    f"≥4 sensory anchors (आंखें / दीप / गूंज / हाथ / आवाज़ / धूल / etc.) "
                    f"across all scenes combined. SHOW the scene — don't summarize. Keep "
                    f"the same scene structure and word counts; just add the dialogue + "
                    f"sensory anchors inside the existing sentences."
                )
            elif last_tha_tic:
                chosen = (
                    f"Your previous response had the \"था-था-था\" verbal tic — "
                    f"{int(last_tha_ratio * 100)}% of sentences ended with था/थी/थे/थीं. "
                    f"AT MOST 2 sentences in the whole script may end that way. Rewrite "
                    f"using HISTORICAL PRESENT (\"द्रौपदी रोती है\" not \"रोई थी\"), simple "
                    f"perfective WITHOUT auxiliary (\"भीम ने प्रतिज्ञा ली\" not \"ली थी\"), "
                    f"nominalization, or exclamatory beats. Keep the same word counts."
                )
            elif last_mono_pattern:
                chosen = (
                    f"Your previous response had sentence-ending MONOTONY — "
                    f"{int(last_mono_ratio * 100)}% of sentences ended with the same word "
                    f"'{last_mono_pattern}'. No single sentence-ending pattern may exceed "
                    f"40% of all sentences. Mix endings: historical present, simple "
                    f"perfective without auxiliary, nominalization, vocative beats, "
                    f"questions. Keep the same word counts. The script must MOVE."
                )
            elif last_offenders:
                offender_str = ", ".join(f"'{w}' ({n}x)" for w, n in last_offenders[:3])
                chosen = (
                    f"Your previous response REPEATED these words too many times: "
                    f"{offender_str}. Use SYNONYMS. Each sentence must contain a NEW "
                    f"concrete detail. Keep the same word counts."
                )
            if chosen:
                full_prompt += f"\n\nCRITICAL REMINDER FOR THIS ATTEMPT:\n{chosen}"

        # quality="best" — Mahabharata dramatization is the longest creative
        # Hindi prose call in the pipeline. The 2026-05-13 local test
        # produced ungrammatical Hindi (`प्रतिज्ञा हे` / gender mismatch)
        # characteristic of Flash; Pro handles this register better.
        raw = _call_llm(full_prompt, quality="best")

        # Extract the JSON object — handles thinking text, code fences, and preamble.
        # Phase 11 retention refactor 2026-06-02: graceful parse-failure handling.
        # When Gemini Flash truncates a long response (no closing `}`), the old
        # code raised ValueError out of the entire retry loop — skipping every
        # validator AND the best-of-N rescue. Now we catch the parse failure,
        # mark the attempt failed, and continue to the next iteration so any
        # prior successful attempt can still be shipped via rescue.
        try:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON object found in LLM response:\n{raw[:300]}")
            raw = raw[start:end + 1]
            data = _parse_llm_json(raw)
        except (ValueError, json.JSONDecodeError) as parse_err:
            print(f"    [parse-fail attempt {attempt + 1}/{MAX_ATTEMPTS}] "
                  f"LLM returned unparseable JSON ({type(parse_err).__name__}: "
                  f"{str(parse_err)[:100]}); continuing to next attempt.")
            # Skip ALL validator + scoring logic for this attempt; the prior
            # best_data remains in place for rescue if this was the last
            # attempt. Restore an empty `data` so the loop control variables
            # are sane before `continue`.
            if attempt < MAX_ATTEMPTS - 1:
                continue
            # Last attempt parse-failed — break to let best-of-N rescue ship
            # whatever earlier attempt scored highest. If no earlier attempt
            # succeeded either, the rescue's `best_data is None` check raises
            # downstream with a clear message.
            break

        # Hard-enforce upper bound — LLMs sometimes ignore word limits
        # (Phase 15 defensive .get() — see 2026-06-04 KeyError post-mortem)
        for scene in data.get("scenes", []):
            scene["narration"] = _trim_narration(scene.get("narration", ""))

        scenes = data.get("scenes", [])
        word_counts = [len(s["narration"].split()) for s in scenes]
        avg_words = sum(word_counts) / max(len(word_counts), 1)
        n_scenes = len(scenes)

        # Length validator (Phase 17 2026-06-13 — hyper-cut to 13 scenes,
        # 6-9 words/scene, to manufacture kinetic visual energy from
        # static FLUX images via rapid editing alone). Architecture:
        # scenes 1-12 are narrative beats (6-9 words each, ~90-100
        # total); scene 13 is the loop-closure question (5-8 words).
        # Combined avg lands at ~7-9 across all 13 scenes. At Charon's
        # 2.2 wps that's ~46s narration + ~3s outro = ~49-50s — well
        # inside the 58s ceiling. Floor at 5 to absorb the short
        # questions/valleys without false-rejection; ceiling at 10
        # rejects LLM drift back toward the old paragraph-length
        # scenes that defeat the hyper-cut visual density.
        # Previous Phase 15: n_scenes != 7, avg 13-18.
        last_short = (n_scenes != 13 or avg_words < 5 or avg_words > 10)
        # Threshold 4: a character at the centre of the story (Bhishma in a
        # Bhishma video) can appear ~4 times naturally. 5+ times signals that
        # supporting characters and details are being skipped in favour of
        # restating the main name. Same threshold for abstract nouns flags
        # filler like "valor" / "वीरता" appearing 5+ times.
        rep_ok, last_offenders = _check_repetition(scenes, max_repeats=4, topic=topic)

        # Hindi-only "tha-tha-tha" verbal tic check. Threshold lowered from
        # 0.35 -> 0.15 because Gandhari shipped with 7 violations out of ~20
        # sentences (35% — right at the old cap). The prompt rule says "AT
        # MOST 2 sentences" which for a 5-6 scene script is roughly 10-15%,
        # so 0.15 matches the rule more faithfully.
        if language == "hi":
            tha_ok, last_tha_ratio, tha_hits, tha_total = _check_past_aux_tic(scenes, threshold=0.15)
            last_tha_tic = not tha_ok
        else:
            tha_ok = True
            last_tha_tic = False

        # Character-name validator: every Mahabharata scene's image_prompt
        # must contain at least one recognized character name so
        # _inject_characters() can append the visual description.
        names_ok, last_missing_names = _check_character_names(scenes)

        # Bookend validator (Fix 2): final scene's narration must echo
        # scene 1's central noun (or a recognized synonym). Without this
        # the video ends but doesn't RESOLVE.
        bookend_ok, last_bookend_miss = _check_bookend(scenes, topic=topic)
        if bookend_ok:
            last_bookend_miss = None  # clear so the reminder doesn't fire

        # Engagement-density validator (Fix 3): ≥2 dialogue + ≥4 sensory
        # anchors across the whole Hindi script. Below threshold means
        # summary prose — the engagement floor is breached.
        eng_ok, last_eng_dialogue, last_eng_sensory = _check_engagement_density(scenes, language)
        last_eng_short = not eng_ok

        # Ending-monotony validator (Fix 4): no single sentence-ending
        # pattern may exceed 40% of all sentences. Generalizes the older
        # tha-tic check — also catches present-tense `हैं` chains and
        # future-conditional `जाएगा` chains.
        mono_ok, mono_pattern, mono_ratio, mono_hits, mono_total = _check_ending_monotony(
            scenes, language=language, max_ratio=0.40,
        )
        last_mono_pattern = mono_pattern if not mono_ok else None
        last_mono_ratio   = mono_ratio

        # ── Cinematic upgrade validators (added 2026-05-16) ──────────────
        # Hook + rehook are HARD gates (block + retry). Rhythm + visual are
        # SOFT (warn-only) by design — over-strict prose validation pushes
        # the LLM into prompt-fighting mode and the prose loses soul.
        scene1_text = (scenes[0].get("narration") if scenes else "") or ""
        last_hook_ok, last_hook_reason = _check_hook_pattern(scene1_text)
        # Phase 11 retention refactor 2026-06-02: hook_title HARD gate.
        last_hook_title_ok, last_hook_title_reason = _check_hook_title(
            data.get("hook_title", ""), language=language
        )
        last_rehook_ok, rehook_idx = _check_rehook_present(scenes)
        rhythm_ok, rhythm_reason = _check_sentence_rhythm(scenes)
        visual_ok, visual_missing = _check_visual_escalation(scenes)

        # ── Phase 2/3 stabilization validators (added 2026-05-18) ────────
        # Aftermath + dangerous-line are HARD gates. Anti-over-curation
        # is SOFT (warn-only) — the rule lives in the prompt; the check
        # measures drift over time without forcing extra retries.
        last_aftermath_ok, last_aftermath_reason = _check_final_scene_aftermath(scenes)
        last_danger_ok, danger_scene_idx, last_danger_reason = _check_dangerous_line(scenes, language)
        curation_ok, curation_count, curation_scenes = _check_anti_over_curation(scenes)

        print(f"    Script: {n_scenes} scenes, avg {avg_words:.1f} words/scene "
              f"(per-scene: {word_counts})")
        if last_hook_ok:
            print(f"    [hook] OK — {last_hook_reason}")
        else:
            print(f"    [hook] REJECT — {last_hook_reason}")
        if last_hook_title_ok:
            print(f"    [hook-title] OK — {last_hook_title_reason}")
        else:
            print(f"    [hook-title] REJECT — {last_hook_title_reason}")
        if last_rehook_ok:
            print(f"    [rehook] OK — twist marker in narrative scene {rehook_idx + 1}")
        else:
            print(f"    [rehook] REJECT — no contrast marker in middle window")
        if rhythm_ok:
            print(f"    [rhythm] OK — {rhythm_reason}")
        else:
            print(f"    [warn] Sentence rhythm flat — {rhythm_reason}")
        if not visual_ok:
            print(f"    [warn] Visual escalation: scenes {visual_missing} lack "
                  f"power keywords — climax may feel flat")
        if last_offenders:
            top = ", ".join(f"{w}×{n}" for w, n in last_offenders[:5])
            print(f"    [warn] Repetition: {top}")
        if last_tha_tic:
            print(f"    [warn] Past-aux tic: {tha_hits}/{tha_total} sentences "
                  f"end with था/थी/थे/थीं ({last_tha_ratio:.0%})")
        if last_missing_names:
            print(f"    [warn] Generic-descriptor image_prompts in scenes "
                  f"{last_missing_names} — character injection will not fire")
        if last_bookend_miss:
            print(f"    [warn] Bookend missing — scene 1 noun '{last_bookend_miss}' "
                  f"not echoed in final scene")
        if last_eng_short:
            print(f"    [warn] Engagement floor: {last_eng_dialogue} dialogue, "
                  f"{last_eng_sensory} sensory (need ≥2 dialogue + ≥4 sensory)")
        if last_mono_pattern:
            print(f"    [warn] Sentence-ending monotony: '{last_mono_pattern}' at "
                  f"{int(last_mono_ratio * 100)}% ({mono_hits}/{mono_total})")

        # Phase 2/3 stabilization validators (2026-05-18) — print status
        if last_aftermath_ok:
            print(f"    [aftermath] OK — scene 6 lands aftermath")
        else:
            print(f"    [aftermath] REJECT — {last_aftermath_reason[:120]}")
        if last_danger_ok:
            print(f"    [danger-line] OK — found in scene {danger_scene_idx}")
        else:
            print(f"    [danger-line] REJECT — no destabilization signature in scenes 3-4")
        if not curation_ok:
            print(f"    [warn] Over-curation: {curation_count} scenes use heavy "
                  f"symbolic imagery (scenes {curation_scenes}) — max 1 per video. "
                  f"Devastation is NOT a poem.")

        # ── Best-of-N rescue tracking ────────────────────────────────────
        # Even when no attempt passes every gate, ship the highest-scoring
        # attempt at exhaustion (instead of the last attempt, which may be
        # the worst — that's how the 2026-05-16 smoke test ended up with
        # 14.8 words/scene + 0/0 engagement).
        # Score counts HARD gates passed for this attempt.
        score = (
            (0 if last_short else 1)
            + (1 if last_hook_ok else 0)
            + (1 if last_hook_title_ok else 0)  # Phase 11
            + (1 if last_rehook_ok else 0)
            + (1 if rep_ok else 0)
            + (1 if tha_ok else 0)
            + (1 if names_ok else 0)
            + (1 if bookend_ok else 0)
            + (1 if eng_ok else 0)
            + (1 if mono_ok else 0)
            + (1 if last_aftermath_ok else 0)
            + (1 if last_danger_ok else 0)
        )
        if score > best_score:
            best_score = score
            best_data  = data

        # Acceptable if every HARD gate passes. (Rhythm + visual + curation
        # are soft, don't gate acceptance.)
        if (not last_short and last_hook_ok and last_hook_title_ok
                and last_rehook_ok
                and rep_ok and tha_ok and names_ok
                and bookend_ok and eng_ok and mono_ok
                and last_aftermath_ok and last_danger_ok):
            break

        if attempt < MAX_ATTEMPTS - 1:
            why = []
            if not last_hook_ok:
                why.append(f"hook REJECT ({last_hook_reason})")
            if not last_hook_title_ok:
                why.append(f"hook_title REJECT ({last_hook_title_reason[:60]})")
            if not last_rehook_ok:
                why.append("rehook missing")
            if not last_aftermath_ok:
                why.append(f"aftermath REJECT")
            if not last_danger_ok:
                why.append("dangerous-line missing")
            if last_short:
                why.append(f"too short ({n_scenes} scenes / {avg_words:.1f} avg words)")
            if not rep_ok:
                why.append(f"{len(last_offenders)} repeated words")
            if last_tha_tic:
                why.append(f"था-tic {last_tha_ratio:.0%}")
            if last_missing_names:
                why.append(f"{len(last_missing_names)} scenes missing character names")
            if last_bookend_miss:
                why.append(f"bookend missing ('{last_bookend_miss}')")
            if last_eng_short:
                why.append(f"engagement floor (D={last_eng_dialogue} / S={last_eng_sensory})")
            if last_mono_pattern:
                why.append(f"ending monotony '{last_mono_pattern}' {int(last_mono_ratio * 100)}%")
            print(f"    [retry] {'; '.join(why)}. Re-prompting...")
    else:
        # Loop exhausted without a break — best-of-N rescue picks the
        # highest-scoring attempt instead of just returning the last one.
        # Without this rescue the 2026-05-16 smoke test shipped a script
        # with all gates failing (the 5th attempt was the worst).
        if best_data is not None and best_data is not data:
            print(f"    [rescue] All {MAX_ATTEMPTS} attempts failed every gate; "
                  f"shipping the best-scoring attempt ({best_score} gates passed)")
            data = best_data

    data["language"] = language
    data["content_type"] = content_type
    data["topic"] = topic
    data["series"] = "mahabharata"
    # Part G.3 (2026-05-20): expose episode_n so the outro builder
    # (main.py:_subscribe_outro) can rotate the reflective question
    # deterministically per render. Integer; JSON-serializable.
    data["episode_n"] = episode_n

    # Tier 2 Fix 2.0 step 3: when this arc episode has a next-episode
    # teaser, prepend a "▶️ अगला भाग" line to the YouTube description.
    # This is the BINGE mechanic at the description level — even when the
    # narration uses Pattern D (haunted, no explicit next-episode mention),
    # the description still gives viewers a one-line link to the next beat.
    if next_episode_teaser:
        # Derive a compact next-episode title from the teaser (first 60 chars
        # before the em-dash, OR first 60 chars if no em-dash).
        teaser_title = next_episode_teaser.split("—")[0].strip()[:60]
        if not teaser_title:
            teaser_title = next_episode_teaser[:60]
        prefix = f"▶️ अगला भाग: {teaser_title}\n\n"
        existing_desc = data.get("description", "")
        # Only prepend if not already present (idempotent for retries)
        if not existing_desc.startswith("▶️ अगला भाग:"):
            data["description"] = prefix + existing_desc

    # Phase 12 (2026-06-03) — bake canonical character keys into the script
    # JSON so downstream consumers (video_assembler's _pick_lut, subtitle
    # generator, etc.) read from one well-named top-level location instead
    # of re-deriving from arc_name -> arc.get("character") on every step.
    # Empty strings for non-arc topics; consumers should treat empty as
    # "use legacy/random behavior".
    data["arc_character_devanagari"] = arc_character_devanagari
    data["arc_character_english"]    = arc_name or ""

    return data
