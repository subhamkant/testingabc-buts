"""Devotional-awe register pool (2026-07-09, user-approved thesis shift).

Competitive forensic across the niche showed every winner's currency is
BHAKTI — devotional awe, divine protection, grace — while our entire
catalog was dark (धोखा/श्राप/हार). User approved a 50/50 emotional mix:
odd IST days ship a devotional-awe story from this pool; even days keep
the established dark-hook register. Two weeks of retention + view data
then decides the final mix.

Every topic is ONE canonical incident (no-repeats doctrine), royal/divine
casting, and dramatically arced (crisis -> divine intervention -> wonder)
so retention mechanics survive the register change.

Env: DEVOTIONAL_MIX=off disables the devotional-day override entirely.
"""
from datetime import datetime, timedelta, timezone
import os

_IST = timezone(timedelta(hours=5, minutes=30))

DEVOTIONAL_TOPICS = [
    "Krishna's endless cloth in the Kuru sabha — the moment Draupadi let go "
    "of her sari, raised both arms and called 'Govind', and the fabric "
    "became infinite, the surrender that summoned divine protection",

    "Krishna's Vishwaroop in the Kuru court — when peace talks failed, the "
    "cosmic form blazing like a thousand suns that made the blind king's "
    "court fall to its knees in awe",

    "The day Arjuna's chariot turned to ash — Krishna stepping down after "
    "the war, Hanuman leaving the banner, and the chariot burning to reveal "
    "who had silently absorbed every celestial weapon for eighteen days",

    "Krishna choosing Vidura's humble saag over Duryodhana's royal feast — "
    "the night God walked past a palace banquet to eat greens served with "
    "love, and taught Hastinapur what devotion tastes like",

    "The single grain that fed ten thousand — Durvasa's curse looming, and "
    "Krishna tasting the last speck in Draupadi's Akshaya Patra until every "
    "sage rose from the river feeling full",

    "Bhishma's fifty-eight nights on the bed of arrows — the grandsire "
    "holding death itself at bay for Uttarayana, and Krishna granting his "
    "final darshan as he chose his moment to leave",

    "The moment Arjuna dropped his bow between two armies — and Krishna "
    "began the song that still guides the world, the Bhagavad Gita born on "
    "a battlefield at sunrise",

    "Yudhishthira refusing heaven without his dog — the final test at the "
    "gates of Swarga, where the stray who walked the Himalayas beside him "
    "was Dharma himself in disguise",

    "Hanuman roaring on Arjuna's banner — the invisible protector whose "
    "war-cry alone scattered Kaurava hearts for eighteen days, the promise "
    "Rama's devotee kept to Krishna",

    "Karna the Daanveer at sunrise — the giver who never once said no, "
    "whose charity even Indra came disguised to test, and whom even Krishna "
    "saluted as the greatest giver who ever lived",
]


def is_devotional_day() -> bool:
    """Odd IST day-of-year = devotional register. Deterministic per day so
    same-day retries and checkpoint resumes agree."""
    mode = os.environ.get("DEVOTIONAL_MIX", "on").strip().lower()
    if mode in ("off", "false", "0"):
        return False
    if mode == "force":          # testing knob: devotional regardless of day
        return True
    return datetime.now(_IST).timetuple().tm_yday % 2 == 1


def devotional_topic_for_today(used_topics: list | None = None) -> str | None:
    """First unused devotional topic if today is a devotional day."""
    if not is_devotional_day():
        return None
    used = {t.lower() for t in (used_topics or [])}
    for t in DEVOTIONAL_TOPICS:
        if t.lower() not in used:
            return t
    return None
