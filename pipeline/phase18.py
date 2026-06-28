"""
Phase 18 (2026-06-16) — Decoupled Voiceover + Anchored B-Roll.

ARCHITECTURE PIVOT vs Phase 17.b:

Phase 17.b: 1 scene = 1 sentence = 1 audio chunk = 1 image. Forced 13 scenes
× 6–9 words produced fragmented narration ("Arjuna hides bow. Loses identity.
One year.") that, even though TTS runs as ONE continuous Charon call, sounds
like a telegram because of the per-scene `। ` join delimiter. Images suffered
"subject bleed" — Arjuna was baked into every image_prompt and FLUX defaulted
to a generic Mahabharata warrior even when narration was about Draupadi.

Phase 18: 1 flowing voiceover + N anchored B-roll images.
  - LLM emits ONE 80–110 word continuous Hindi paragraph (the "voiceover").
  - LLM also emits 8–10 B-roll entries, each with a literal-physical
    image_prompt + a verbatim anchor_phrase from the voiceover.
  - Video assembler computes each image's start_t via proportional
    char-position mapping: idx = voiceover.find(anchor_phrase); t = (idx /
    len(voiceover)) * audio_duration. Clamped to audio_duration - MIN_DUR.
  - SFX hits the visual cuts (film-editing convention), not narration
    boundaries.

The legacy `scenes` path remains intact. This module is opt-in via
PHASE18_DECOUPLED=true. The detector at all consumer sites is the presence
of `voiceover` (TTS path) / `broll` (image + assembly path) keys in the
script dict — caches are self-describing, env flag controls only generation.
"""

from __future__ import annotations

import json
import os
import re
import time

# Reuse the heavy machinery from script_generator — model cascade, hook-title
# checker, character-name lookup, palette directive, opener archetypes, etc.
# Importing at module-import time creates a circular import (script_generator
# may import phase18 from inside generate_script). Lazy-import inside
# generate_phase18_script() instead. Constants we DO duplicate (lightweight,
# small) live below.


# ─── Subject-lock map (Phase 18) ──────────────────────────────────────────
# Devanagari nouns → English physical-form equivalents the LLM uses in
# image_prompts. Used by _check_subject_lock to verify each broll entry's
# image_prompt physically contains the anchor's primary subject. Closes
# the "subject bleed" gap where the narration says Draupadi but the image
# shows Arjuna.
_SUBJECT_LOCK_MAP = {
    # Characters — STRICT match (only the actual name passes; generic
    # "warrior"/"queen" would let subject-bleed slip through). The whole
    # POINT of subject-lock is forcing the LLM to name the character that
    # the anchor calls out, not fall back to a generic Mahabharata stock
    # figure that lets FLUX render whichever face it pleases.
    "अर्जुन":     ("arjuna",),
    "द्रौपदी":    ("draupadi",),
    "कर्ण":       ("karna",),
    "भीष्म":      ("bhishma",),
    "कृष्ण":      ("krishna",),
    "अश्वत्थामा": ("ashwatthama",),
    "युधिष्ठिर":  ("yudhishthira",),
    "एकलव्य":    ("eklavya",),
    "भीम":        ("bhima",),
    "द्रोण":      ("drona",),
    "दुर्योधन":   ("duryodhana",),
    "कुंती":      ("kunti",),
    "गांधारी":    ("gandhari",),
    "अभिमन्यु":   ("abhimanyu",),
    "जयद्रथ":    ("jayadratha",),
    "शिखंडी":    ("shikhandi",),
    # Props / body / damage
    "तलवार":  ("sword", "blade"),
    "धनुष":   ("bow",),
    "बाण":    ("arrow",),
    "रथ":     ("chariot",),
    "आंसू":   ("tear", "wet", "crying"),
    "आँसू":   ("tear", "wet", "crying"),
    "रक्त":   ("blood", "bloodied"),
    "बच्चे":  ("child", "children", "sleeping"),
    "साड़ी":  ("sari",),
    "हाथ":    ("hand", "fingers", "grip"),
    "आंखें":  ("eye", "eyes", "gaze"),
    "आँखें":  ("eye", "eyes", "gaze"),
}


# ─── Intensity adjectives (Phase 18, 2026-06-16) ──────────────────────────
# FLUX defaults to clean, shiny, museum-quality assets unless explicitly told
# otherwise. Every broll image_prompt MUST include ≥1 intensity adjective so
# the rendered image matches the narration's emotional charge. List is
# intentionally short and CONCRETE — abstract terms like "dramatic" or
# "intense" slip past FLUX and produce generic output. These specific
# physical adjectives FLUX actually parses into visible visual differences.
_INTENSITY_ADJECTIVES = (
    # Damage / decay
    "bloodied", "blood-soaked", "blood-streaked", "rusted", "shattered",
    "cracked", "broken", "torn", "tattered", "ash-covered", "scorched",
    "burned", "wounded", "scarred", "bruised",
    # Light / motion intensity
    "smoldering", "glowing", "flickering", "dim", "harsh", "blinding",
    "fading", "burning", "smoking",
    # Body / face under strain
    "trembling", "clenched", "tear-streaked", "white-knuckled",
    "sweat-drenched", "wide-eyed", "hollow-eyed", "kohl-rimmed",
    # Environmental tension
    "wind-whipped", "rain-soaked", "dust-choked", "shadow-heavy",
    "claustrophobic", "suffocating",
)


# ─── Phase 22 (2026-06-25) — Verb-per-frame + Anti-merge + Title DNA ──────
# Verb-per-frame: forensic on 4 channel winners (EksW2W5aP5A 655v,
# IcSK2Sl7-p8 644v, f3oZzuAiN3Q 500v, garplHH-k3A 484v) showed every frame
# was a VERB (someone doing something to someone), not a NOUN (a crowned
# warrior existing in a costume). Winners 6/8 verb-frames; recents 1-2/8.
_VERB_BLACKLIST = {
    "standing", "looking", "gazing", "contemplating", "praying",
    "holding", "wearing", "sitting", "watching", "observing",
    "thinking", "meditating", "posing", "facing", "smiling",
}

_VERB_WHITELIST = {
    "drawing", "drawing back", "drawing the string", "drawing the bow",
    "sinking", "severing", "kneeling", "kneeling beneath",
    "looming", "looming over", "towering", "towering over",
    "leaning into", "leaning over", "dragging", "dragging away from",
    "charging", "falling", "fallen", "gripping", "snarling",
    "recoiling", "fleeing", "walking away from", "walking from",
    "hurled", "hurling", "slashing", "parrying", "weeping over",
    "kicking", "kicking aside", "raised in halt", "palm raised",
    "drawing arrow", "drawing sword", "striking", "piercing",
    "collapsing", "lunging", "advancing", "marching", "wielding",
    "drawing back from", "leaning in",
}

_VERB_BLACKLIST_RE = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in _VERB_BLACKLIST) + r")\b",
    re.IGNORECASE,
)

# 5 power-imbalance composition templates derived from the winners.
# Each is a tagged prefix the LLM literally emits as the LEADING tokens
# of a two-character image_prompt — easy to validate via substring
# search, explicit signal to FLUX about scene geometry to defeat the
# multi-subject merge problem (3 arms / fused faces / shared torso).
_COMPOSITION_TEMPLATES = {
    "OVER-SHOULDER": (
        "Over-shoulder shot: secondary character's shoulder and head "
        "(back/side, no face) bokeh in left foreground at 1/3 frame, "
        "named character mid-ground facing camera in focus at "
        "2/3 frame height. "
    ),
    "POWER-LOOM": (
        "Power-imbalance two-shot: named character standing tall at "
        "2/3 frame height in left foreground, secondary character "
        "kneeling at lower 1/3 frame, head bowed, identity partly "
        "obscured by shadow. Vertical scale separation. "
    ),
    "CONFRONTATION-WIDE": (
        "Wide two-shot confrontation: named character in left foreground "
        "side-profile under warm rim-light, secondary character in right "
        "mid-ground torso-up under cool shadow zone, diagonal "
        "light/shadow boundary cutting the frame. "
    ),
    "WITNESS-FROM-BEHIND": (
        "Witness composition: named character back-of-head and shoulders "
        "silhouette foreground (no face visible), secondary character "
        "mid-ground facing camera in focus, named character watching "
        "the secondary's reaction. "
    ),
    "THE-FALLEN": (
        "Aftermath two-shot: named character standing at upper third of "
        "frame, secondary character prone in dust at named character's "
        "feet, only legs/torso/weapon visible (face withheld), low-angle "
        "ground-level camera. "
    ),
}

# Title DNA — Phase 22 hard gate on title wording. Forensic showed
# winners used tribal-relational nouns (loyalty / pride / sacrifice /
# mistake), recents drifted into mood-nouns (regret / wound / ego) and
# YT-India-demotable identity nouns (caste / religion).
_TITLE_ADJ_WHITELIST = {
    "real", "hidden", "untold", "secret", "greatest", "final", "fatal",
    "big", "biggest", "last",
    "असली", "छुपा", "छुपी", "गुप्त", "सबसे बड़ा", "सबसे बड़ी",
    "आखिरी", "अंतिम", "घातक",
}

_TITLE_ADJ_BLACKLIST = {
    "silent", "internal", "intellectual", "abstract", "quiet",
    "subtle", "deep", "inner", "spiritual", "philosophical",
    "चुप", "मौन", "आंतरिक", "गहरा", "गहरी", "सूक्ष्म",
}

_TITLE_NOUN_WHITELIST = {
    "sacrifice", "vow", "friendship", "pride", "loss", "fear",
    "loyalty", "mistake", "betrayal", "curse", "promise", "doubt",
    "oath", "debt", "shame",
    "बलिदान", "प्रतिज्ञा", "वचन", "दोस्ती", "मित्रता", "अभिमान",
    "घमंड", "हार", "डर", "वफ़ादारी", "गलती", "धोखा", "विश्वासघात",
    "श्राप", "शर्म",
}

_TITLE_NOUN_BLACKLIST = {
    # Mood / introspection (recents drifted into these)
    "regret", "wound", "ego", "self", "war within", "rage",
    "पछतावा", "दर्द", "घाव", "अहंकार", "स्वयं", "स्वार्थ",
    # YT-India identity / caste / religion (algorithmic demotion risk)
    "caste", "identity", "religion", "hindu",
    "जाति", "वर्ण", "पहचान", "धर्म",
    # Vague abstractions
    "sin", "truth", "lie", "fate", "destiny",
    "पाप", "सच", "झूठ", "भाग्य", "नियति", "काला सच",
}

# Per-character canonical-virtue inversion sets — does the title assert
# the OPPOSITE of the hero's most celebrated virtue? Karna's loyalty
# → mistake (sharp inversion). Mild non-inversion ("Karna's Silent
# Wound") fails this gate.
_CHARACTER_VIRTUE_INVERSION = {
    "कर्ण":      ("loyalty",   {"mistake", "betrayal", "curse", "गलती", "धोखा", "श्राप"}),
    "अर्जुन":    ("skill",     {"doubt", "fear", "mistake", "डर", "गलती"}),
    "भीष्म":    ("vow",       {"betrayal", "mistake", "promise", "धोखा", "गलती", "वचन"}),
    "द्रौपदी":   ("dignity",   {"shame", "betrayal", "loss", "शर्म", "धोखा", "हार"}),
    "युधिष्ठिर":  ("dharma",   {"mistake", "shame", "loss", "गलती", "शर्म", "हार"}),
    "एकलव्य":   ("devotion",  {"sacrifice", "betrayal", "loss", "बलिदान", "धोखा", "हार"}),
    "अश्वत्थामा": ("vengeance", {"curse", "shame", "mistake", "श्राप", "शर्म", "गलती"}),
}


# Hindi rehook markers — used by middle-window subversion check
_REHOOK_MARKERS = re.compile(
    r"लेकिन|परंतु|पर\s|फिर\s+भी|और\s+तभी|उसी\s+क्षण|"
    r"\bbut\b|\bsuddenly\b|\byet\b",
    re.IGNORECASE,
)

# Hindi sensory anchors — used by engagement-density check
_SENSORY_TOKENS_HI = (
    "शंख", "दीप", "आंख", "आँख", "हाथ", "रक्त", "धुआं", "धुआँ",
    "आंसू", "आँसू", "पसीना", "होंठ", "साँस", "सांस", "त्वचा",
    "वस्त्र", "धूल", "राख", "बाल", "कांप", "काँप",
)

# Hindi dialogue beats — quoted speech markers or direct second-person address
_DIALOGUE_MARKERS_HI = re.compile(
    r"[\"“”‘’]|"   # straight + curly quotes
    r"बोल|कह|पूछ|चीख|फुसफुसा|"           # speech verbs
    r"क्या\s+तुम|क्या\s+आप|तुम्हें|आपको",
)

# Banned documentary openers
_BANNED_OPENERS_HI = re.compile(
    r"^\s*(यह\s+कहानी|ये\s+कहानी|एक\s+कहानी|बहुत\s+समय|कहते\s+हैं|"
    r"long\s+ago|once\s+upon|in\s+ancient|this\s+is\s+the)",
    re.IGNORECASE,
)

# Action verbs (Devanagari present-tense markers that indicate shock-action)
_ACTION_VERBS_HI = re.compile(
    r"\b\w+ता\s|\b\w+ती\s|\b\w+ते\s|"      # present-tense ता/ती/ते
    r"उठा|फेंक|काट|मार|गिर|टूट|जल|खींच|"     # explicit action verbs
    r"घसीट|छीन|पकड़|बहा|दौड़|भाग|छेद|वार"
)


# Phase 17.b opener archetype rotation. Defined here (not imported from
# script_generator) because the legacy version lives INSIDE generate_script()
# as a local. Keeping a copy here is the cheapest decoupling — same content,
# same A/B/C/D semantics; if either drifts the channel grid loses its planned
# variance, so this is a known dual-maintenance cost worth flagging.
_OPENER_ARCHETYPES_PHASE18 = [
    ("A", "Extreme macro close-up on character's eyes — single tear / pupils dilated / lashes wet — caught mid-blink"),
    ("B", "Weapon / artifact detail — sword tip mid-strike / glowing arrow / bloodied hand on hilt / dice mid-roll — the object that carries the violence, blurred motion behind"),
    ("C", "Action moment frozen mid-gesture — sword raising arc / hand gripping hair / arrow leaving bowstring / body falling — the verb made visible"),
    ("D", "Wide environmental disaster — burning camp / blood-soaked battlefield / crashing chariot wheel / collapsing palace pillar — chaos that the narration described, no character close-up"),
]


def phase18_enabled() -> bool:
    """Env flag gate. Default OFF until smoke-validated and a live render
    confirms retention lift. Flip to true via workflow_dispatch input or
    by editing schedule.yml `PHASE18_DECOUPLED: "true"` default."""
    return os.environ.get("PHASE18_DECOUPLED", "false").strip().lower() == "true"


# ─── Validators ───────────────────────────────────────────────────────────

def _check_image_intensity(image_prompt: str) -> bool:
    """Every broll image_prompt must contain ≥1 intensity adjective.
    Case-insensitive substring match. Prevents FLUX from defaulting to
    museum-clean output when narration is depicting violence/grief/decay."""
    image_lower = image_prompt.lower()
    return any(adj in image_lower for adj in _INTENSITY_ADJECTIVES)


def _check_verb_per_frame_single(image_prompt: str) -> tuple[bool, str]:
    """Phase 22 (2026-06-25). Single-frame check. The LEADING 80 chars
    must contain a whitelist verb (single or multi-word) AND zero
    blacklist verbs (whole-word). FLUX weights early tokens heaviest —
    that's where the action must land."""
    if not image_prompt:
        return False, "empty image_prompt"
    leading = image_prompt[:80].lower()

    bl_hit = _VERB_BLACKLIST_RE.search(leading)
    if bl_hit:
        return False, f"leading-80 contains blacklisted noun-pose verb '{bl_hit.group(1)}'"

    if not any(verb in leading for verb in _VERB_WHITELIST):
        return False, "leading-80 lacks any action-encoding verb from whitelist"

    return True, ""


def _check_verb_per_frame(broll: list) -> tuple[bool, str]:
    """Phase 22 (2026-06-25). Aggregate: ≥5/8 broll entries must pass
    the single-frame verb check, AND the final broll[-1].image_prompt
    must contain an aftermath-silhouette token. Winners had 6/8 verb-
    frames (75%); 5/8 (62.5%) threshold leaves room for one REACTION
    beat. Closer-token requirement matches the consequence-as-closer
    pattern in all 4 winners."""
    if not broll:
        return False, "broll is empty"

    passing = sum(1 for b in broll
                  if _check_verb_per_frame_single(b.get("image_prompt", ""))[0])
    threshold = max(5, int(len(broll) * 0.625 + 0.999))
    if passing < threshold:
        return False, (
            f"verb-per-frame: {passing}/{len(broll)} broll entries lead with "
            f"an action verb (need ≥{threshold}). Winners average 6/8 verb-"
            f"frames; recents average 1-2/8 — this is the SINGLE most "
            f"predictive feature in the 500-660 view dataset. Replace "
            f"'standing/looking/holding' poses with 'drawing/sinking/"
            f"looming over/walking away from' actions."
        )

    last_prompt = (broll[-1].get("image_prompt", "") or "").lower()
    aftermath_tokens = ("silhouette", "abandoned", "walking away",
                        "walking from", "withheld", "face withheld",
                        "fading dusk", "lone diya", "prone body")
    if not any(t in last_prompt for t in aftermath_tokens):
        return False, (
            "broll[-1] is not an aftermath-silhouette beat. Winners close "
            "on consequence (silhouette walking from abandoned weapons / "
            "victor with face withheld / lone diya beside palm of refusal). "
            "Final broll image_prompt must contain at least one of: "
            "silhouette, abandoned, walking away, withheld, lone diya, "
            "prone body, fading dusk."
        )

    return True, ""


def _check_anti_merge_composition(broll: list) -> tuple[bool, str]:
    """Phase 22 (2026-06-25). When an image_prompt names ≥2 characters
    from _CHARACTER_NAMES_LIST, it MUST start with a [COMPOSITION-TAG]
    from _COMPOSITION_TEMPLATES. This forces spatial / scale separation
    that defeats FLUX's multi-subject merge. Single-character + macro-
    on-prop + environmental scenes are exempt (verb-per-frame covers
    them)."""
    from pipeline.script_generator import _CHARACTER_NAMES_LIST

    for i, entry in enumerate(broll):
        prompt = entry.get("image_prompt", "") or ""
        prompt_lower = prompt.lower()
        names_hit = [name for name in _CHARACTER_NAMES_LIST
                     if name.lower() in prompt_lower]
        if len(names_hit) < 2:
            continue

        if not any(prompt.startswith(f"[{tag}]") or tag in prompt[:60]
                   for tag in _COMPOSITION_TEMPLATES):
            return False, (
                f"broll[{i}] names ≥2 characters ({names_hit[:2]}) but does "
                f"NOT lead with an anti-merge composition tag. Required: "
                f"start with one of {sorted(_COMPOSITION_TEMPLATES.keys())} "
                f"to force scale/spatial separation. Without this, FLUX "
                f"merges the two characters (3 arms, fused faces, "
                f"shared torso)."
            )
    return True, ""


def _check_aftermath_closer(broll: list) -> tuple[bool, str]:
    """Phase 22 (2026-06-25). The final broll entry MUST declare
    wardrobe_context: AFTERMATH — forces the silhouette / face-withheld
    / abandoned-weapon closing beat that ALL 4 channel winners used.
    Recents close on portrait beats (returning to character glamour)
    which fails the title→thumbnail→payoff loop."""
    if not broll:
        return False, "broll is empty"
    last_ctx = (broll[-1].get("wardrobe_context", "") or "").strip().upper()
    if last_ctx != "AFTERMATH":
        return False, (
            f"broll[-1].wardrobe_context = '{last_ctx}' — must be "
            f"'AFTERMATH'. The closing beat is the consequence frame "
            f"(silhouette, abandoned weapon, face withheld). The 4 "
            f"channel winners ALL closed on consequence, not triumph."
        )
    return True, ""


def _check_title_dna_gate(
    title: str,
    character_devanagari: str = "",
) -> tuple[bool, str, int]:
    """Phase 22 (2026-06-25). 4-step title-DNA cascade. Returns
    (ok, violation_msg, inversion_score 0-3).
      Step 1: noun blacklist (mood / caste / vague)
      Step 2: adjective blacklist (soft / introspective)
      Step 3: require ≥1 tribal-relational whitelisted noun
      Step 4: trait-inversion score ≥2 vs the character's canonical
              virtue (gracefully degrades to 3-step when character is
              empty — legacy / WhatIf / forced-topic edge cases).
    """
    if not title:
        return False, "title is empty", 0

    title_lower = title.lower()

    for noun in _TITLE_NOUN_BLACKLIST:
        if noun.lower() in title_lower:
            return False, f"title contains blacklisted noun '{noun}' (mood/caste/vague)", 0

    for adj in _TITLE_ADJ_BLACKLIST:
        if adj.lower() in title_lower:
            return False, f"title contains blacklisted soft-adjective '{adj}'", 0

    if not any(n.lower() in title_lower for n in _TITLE_NOUN_WHITELIST):
        return False, (
            "title lacks any tribal-relational noun from whitelist. "
            "Need one of: Sacrifice/Vow/Friendship/Pride/Loyalty/Mistake/"
            "Betrayal/Curse/Promise/Doubt/Loss/Fear (or Devanagari "
            "equivalents)."
        ), 0

    # Step 4 — trait-inversion score (1-3) — only if character is known
    char_key = (character_devanagari or "").strip()
    if not char_key:
        # No character context — degrade to 3-step cascade. Steps 1-3
        # already passed (no blacklist, has whitelist noun) which is
        # enough for legacy paths.
        return True, "", 1

    score = 1
    virtue_entry = _CHARACTER_VIRTUE_INVERSION.get(char_key)
    if virtue_entry:
        _virtue, inversion_nouns = virtue_entry
        if any(n.lower() in title_lower for n in inversion_nouns):
            score = 2
        if any(a.lower() in title_lower for a in _TITLE_ADJ_WHITELIST):
            score = max(score, 2)
            if any(n.lower() in title_lower for n in inversion_nouns):
                score = 3

    if score < 2:
        return False, (
            f"title inversion-score={score}/3 (need ≥2). Title does not "
            f"assert the OPPOSITE of {char_key}'s canonical virtue. "
            f"Karna's loyalty → mistake (sharp). Mild/no-inversion "
            f"(e.g. 'Karna's Silent Wound') = reject."
        ), score

    return True, "", score


def _check_subject_lock(anchor_phrase: str, image_prompt: str) -> bool:
    """For each Devanagari noun in the anchor that has a mapping, the
    image_prompt must contain at least one of its English physical-form
    equivalents. Pass-through when the anchor contains no mapped noun
    (warn only, don't block — vocabulary mapping is conservative)."""
    image_lower = image_prompt.lower()
    for hi_noun, en_forms in _SUBJECT_LOCK_MAP.items():
        if hi_noun in anchor_phrase:
            if not any(form in image_lower for form in en_forms):
                return False
    return True


def _validate_anchors(voiceover: str, broll: list) -> tuple[bool, str]:
    """Every broll[i].anchor_phrase MUST appear in voiceover string,
    in the same order as the broll array (monotonic char positions).
    Single-word anchors rejected (ambiguous in 100-word voiceover).
    Final anchor MUST land in last 30% of voiceover to prevent static
    tail. Returns (ok, violation_msg)."""
    last_idx = -1
    for i, entry in enumerate(broll):
        phrase = entry.get("anchor_phrase", "").strip()
        if not phrase:
            return False, f"broll[{i}] has empty anchor_phrase"
        if len(phrase.split()) < 2:
            return False, f"broll[{i}] anchor '{phrase}' is single-word (ambiguous)"
        idx = voiceover.find(phrase)
        if idx < 0:
            return False, f"broll[{i}] anchor '{phrase[:30]}' not found in voiceover"
        if idx <= last_idx:
            return False, (
                f"broll[{i}] anchor '{phrase[:30]}' at char {idx} "
                f"is at or before previous anchor at char {last_idx}"
            )
        last_idx = idx
    # Final-anchor 70% guard
    if broll and last_idx < len(voiceover) * 0.70:
        return False, (
            f"final anchor at char {last_idx} is before 70%-mark of "
            f"voiceover ({int(len(voiceover) * 0.70)})"
        )
    return True, ""


def _check_hook_pattern_text(first_10_words: str) -> bool:
    """Phase 18 (2026-06-16) — voiceover-mode hook validator. First 10
    words of voiceover must NOT be a documentary opener AND must contain
    EITHER a named character OR a paradox marker (or both)."""
    if _BANNED_OPENERS_HI.match(first_10_words):
        return False
    # Lazy-import to avoid circular dep
    from pipeline.script_generator import _CHARACTER_NAMES_LIST, _HOOK_PARADOX_FIRST_HALF
    has_char = any(name in first_10_words for name in _CHARACTER_NAMES_LIST)
    has_paradox = bool(_HOOK_PARADOX_FIRST_HALF.search(first_10_words))
    return has_char or has_paradox


def _check_shock_action(first_words: list) -> bool:
    """Phase 17.b shock-action requirement adapted for voiceover. First
    8 words must contain at least one action-verb marker (present-tense
    Devanagari verb OR explicit action verb). No documentary setup."""
    text = " ".join(first_words)
    if _BANNED_OPENERS_HI.match(text):
        return False
    return bool(_ACTION_VERBS_HI.search(text))


def _check_engagement_density_text(vo: str) -> bool:
    """≥4 sensory anchors AND ≥2 dialogue beats across the full voiceover."""
    sensory_count = sum(vo.count(tok) for tok in _SENSORY_TOKENS_HI)
    dialogue_count = len(_DIALOGUE_MARKERS_HI.findall(vo))
    return sensory_count >= 4 and dialogue_count >= 2


def _check_past_aux_tic_text(vo: str) -> bool:
    """Hindi past-aux tic — था/थी/थे/थीं ending ≤15% of sentences."""
    sentences = [s for s in re.split(r"[।!?]", vo) if s.strip()]
    if not sentences:
        return True
    tic = sum(
        1 for s in sentences
        if re.search(r"(था|थी|थे|थीं)\s*$", s.strip())
    )
    return (tic / len(sentences)) <= 0.15


def _check_repetition_text(vo: str, max_repeats: int = 4) -> bool:
    """No single word repeated >max_repeats times. Topic-keyword leniency
    handled in best-of-N rescue (this is the strict version)."""
    from collections import Counter
    words = [w for w in re.findall(r"[ऀ-ॿ]+|\w+", vo) if len(w) >= 3]
    counts = Counter(words)
    if not counts:
        return True
    return counts.most_common(1)[0][1] <= max_repeats


def _check_ending_monotony_text(vo: str, max_ratio: float = 0.40) -> bool:
    """No single sentence-ending suffix may exceed max_ratio of sentences."""
    sentences = [s.strip() for s in re.split(r"[।!?]", vo) if s.strip()]
    if len(sentences) < 4:
        return True
    from collections import Counter
    # Bucket each sentence by its last 3 Devanagari chars (rough suffix)
    suffixes = []
    for s in sentences:
        deva = re.findall(r"[ऀ-ॿ]+", s)
        if deva:
            suffixes.append(deva[-1][-3:] if len(deva[-1]) >= 3 else deva[-1])
    if not suffixes:
        return True
    counts = Counter(suffixes)
    top_ratio = counts.most_common(1)[0][1] / len(suffixes)
    return top_ratio <= max_ratio


def _check_bookend_text(vo: str) -> bool:
    """Voiceover's first 10 words and last 10 words must share at least
    one charged Devanagari noun (3+ chars). Adapted from _check_bookend."""
    words = vo.split()
    if len(words) < 20:
        return True  # too short to enforce
    first = " ".join(words[:10])
    last = " ".join(words[-10:])
    first_nouns = set(w for w in re.findall(r"[ऀ-ॿ]{3,}", first))
    last_nouns = set(w for w in re.findall(r"[ऀ-ॿ]{3,}", last))
    return bool(first_nouns & last_nouns)


def _check_character_names_single(image_prompt: str) -> bool:
    """Every broll image_prompt must contain at least one recognized
    character name (English form) so _inject_characters can append the
    visual descriptor block. Lazy-imports the canonical name list."""
    from pipeline.script_generator import _CHARACTER_NAMES_LIST
    text = image_prompt.lower()
    # _CHARACTER_NAMES_LIST has Devanagari + English; we want any match
    return any(name.lower() in text for name in _CHARACTER_NAMES_LIST)


# ─── Phase 19 (2026-06-16) wardrobe + story entity ────────────────────────

_VALID_WARDROBE_CONTEXTS = {
    "WAR", "PALACE", "DIVINE", "FOREST", "JOURNEY",
    # Phase 22 (2026-06-25) — AFTERMATH required by _check_aftermath_closer
    # for broll[-1]. Forces silhouette / face-withheld / abandoned-weapon
    # closing beat that all 4 channel winners used.
    "AFTERMATH",
}

# Devanagari noun → list of English equivalents the LLM should use in
# image_prompts. When a voiceover names a story-critical entity, at least
# ONE broll image_prompt must contain a matching English form in its
# foreground. Without this rule, the LLM emits "Yudhishthira walking with
# his dog" in the voiceover but no broll image_prompt actually contains
# "dog" — and FLUX renders him alone (verified failure mode on the
# 2026-06-16 Heaven's Gate render where the Dharma dog vanished entirely).
_STORY_ENTITY_MAP = {
    "कुत्ता":      ["dog", "stray dog"],
    "धनुष":        ["bow", "gandiva"],
    "गाण्डीव":    ["gandiva", "bow"],
    "बाण-शय्या":  ["bed of arrows", "arrow bed"],
    "बाण":         ["arrow", "arrows"],
    "मोम-महल":    ["wax palace", "lacquered palace", "lakshagriha"],
    "लाक्षागृह":   ["wax palace", "lacquered palace", "lakshagriha"],
    "अक्षय-पात्र": ["begging bowl", "akshaya patra"],
    "सुदर्शन":    ["sudarshana", "discus", "chakra"],
    "चक्र":        ["chakra", "discus"],
    "कुंडल":       ["kundal", "earring", "kundala"],
    "कवच":         ["kavach", "armor", "kavacha"],
    "गदा":         ["mace", "gada"],
    "रथ":          ["chariot"],
    "बच्चे":       ["children", "sleeping children"],
}


def _check_wardrobe_context_set(broll: list) -> tuple[bool, str]:
    """Phase 19 (2026-06-16). Every broll entry must declare a
    `wardrobe_context` ∈ {WAR, PALACE, DIVINE, FOREST, JOURNEY} so
    image_generator.py can inject the right clothing+lighting prefix.
    Empty / invalid → reject."""
    for i, entry in enumerate(broll):
        ctx = (entry.get("wardrobe_context", "") or "").strip().upper()
        if not ctx:
            return False, f"broll[{i}] missing wardrobe_context"
        if ctx not in _VALID_WARDROBE_CONTEXTS:
            return False, (
                f"broll[{i}] wardrobe_context='{ctx}' invalid "
                f"(must be one of {sorted(_VALID_WARDROBE_CONTEXTS)})"
            )
    return True, ""


def _check_story_entity_present(voiceover: str, broll: list) -> tuple[bool, str]:
    """Phase 19 (2026-06-16). When the voiceover names a known story-
    critical entity (कुत्ता / धनुष / सुदर्शन-चक्र / etc.), at least ONE
    broll image_prompt must contain a matching English form. Loosened
    from ≥2 to ≥1 per plan Adjustment #2 — one clean establishing shot
    is enough; demanding 2 occupies 25%+ of an 8-image video and shoehorns
    the entity into bad framings.
    Default-pass-closed for topics where no mapped entity appears."""
    needed = []
    for hi_noun, en_forms in _STORY_ENTITY_MAP.items():
        if hi_noun in voiceover:
            needed.append((hi_noun, en_forms))
    if not needed:
        return True, ""  # no mapped entities in voiceover → nothing to check

    combined_prompts = " ".join(
        b.get("image_prompt", "").lower() for b in broll
    )
    missing = []
    for hi_noun, en_forms in needed:
        if not any(form in combined_prompts for form in en_forms):
            missing.append((hi_noun, en_forms))
    if missing:
        hi, en = missing[0]
        return False, (
            f"voiceover names '{hi}' but no broll image_prompt contains "
            f"its English equivalent ({'/'.join(en[:2])})"
        )
    return True, ""


# ─── Phase 20 (2026-06-17) ────────────────────────────────────────────────
# Two new validators:
#   1. _check_verb_action_lock — Audio-Visual Verb Lock. When the anchor
#      phrase contains an action verb (sinking / raising / striking / etc.),
#      the image_prompt MUST contain the English equivalent so FLUX renders
#      the ACTION not a static portrait holding the noun.
#   2. _check_subject_diversity — film-editor cycle. Broll is not 8
#      portraits; it cycles ACTION / REACTION / PROP / ENVIRONMENT shots.
#
# Both addressed in the Phase 20 plan; both ship with infinite-loop
# protection (default-pass-closed for unmapped vocab + AMBIGUOUS escape
# hatch in the classifier + existing best-of-N rescue path).

# Devanagari ACTION-verb CONJUGATED forms → English equivalents.
# Phase 20 plan-review fix (2026-06-17): NEVER use 2-letter verb stems
# (रो / जल / सुन / मार / etc.) as keys — they substring-match common
# unrelated words:
#   रो substring-matches रोशनी (light), रोग (disease), भरोसा (trust)
#   सुन substring-matches सुनहरा (golden — extremely common in Hindu prompts)
#   जल substring-matches जल (water — abundant in Mahabharata vocabulary)
#   मार substring-matches हमारा / तुम्हारा (ours / yours)
# Instead use explicit conjugated forms (3-5 chars). Combined with
# .startswith() on word-split tokens in _check_verb_action_lock below,
# this eliminates both internal-substring AND prefix false positives.
_VERB_ACTION_MAP = {
    # raising / lifting (root उठा)
    "उठाता":   ["raising", "lifting", "rises", "lifts"],
    "उठाती":   ["raising", "lifting", "rises", "lifts"],
    "उठाते":   ["raising", "lifting"],
    "उठाई":    ["raised", "lifted"],
    "उठाया":  ["raised", "lifted"],
    "उठाकर":  ["raised", "lifted"],
    "उठाने":  ["raising", "to raise"],

    # falling / sinking (root गिर)
    "गिरता":   ["falling", "sinking", "fell"],
    "गिरती":   ["falling", "sinking"],
    "गिरते":   ["falling"],
    "गिरा":    ["fell", "fallen", "collapsed"],
    "गिरी":    ["fell", "fallen"],
    "गिराकर":  ["fallen", "dropped"],

    # cutting / severing (root काट)
    "काटता":   ["cutting", "severing"],
    "काटती":   ["cutting", "severing"],
    "काटी":    ["cut", "severed", "sliced"],
    "काटा":    ["cut", "severed", "sliced"],
    "काटकर":   ["cut", "severed"],

    # striking / killing (root मार — NEVER bare; मार substring-matches हमारा/तुम्हारा)
    "मारता":   ["striking", "killing", "slaying"],
    "मारती":   ["striking", "killing"],
    "मारी":    ["struck", "slain", "killed"],
    "मारा":    ["struck", "slain", "killed"],
    "मारकर":   ["struck", "slain"],
    "मारने":   ["striking", "to strike"],

    # hiding / concealing (root छिप)
    "छिपा":    ["hidden", "concealed", "covered"],
    "छिपी":    ["hidden", "concealed"],
    "छिपाता":  ["hiding", "concealing"],
    "छिपाती":  ["hiding", "concealing"],
    "छिपाकर":  ["hidden", "concealed"],
    "छिपाने":  ["hiding", "to hide"],

    # flowing / shedding (root बहा)
    "बहाता":   ["flowing", "shedding", "streaming"],
    "बहाती":   ["flowing", "shedding", "weeping"],
    "बहाया":   ["shed", "flowed"],
    "बहाई":    ["shed", "flowed"],

    # crying / weeping (root रो — NEVER bare; रो substring-matches रोशनी/रोग/भरोसा)
    "रोता":    ["crying", "weeping", "sobbing"],
    "रोती":    ["crying", "weeping", "sobbing"],
    "रोते":    ["crying", "weeping"],
    "रोया":    ["cried", "wept", "sobbed"],
    "रोई":     ["cried", "wept", "sobbed"],
    "रोकर":    ["crying", "weeping"],

    # throwing / hurling (root फेंक)
    "फेंकता":  ["throwing", "hurling"],
    "फेंकती":  ["throwing", "hurling"],
    "फेंका":   ["threw", "hurled", "flung"],
    "फेंकी":   ["threw", "hurled"],

    # pulling / drawing back (root खींच)
    "खींचता":  ["pulling", "drawing back", "tugging"],
    "खींचती":  ["pulling", "drawing back"],
    "खींची":   ["pulled", "drew back"],
    "खींचा":   ["pulled", "drew back"],

    # breaking / shattering (root तोड़)
    "तोड़ता":  ["breaking", "shattering"],
    "तोड़ती":  ["breaking", "shattering"],
    "तोड़ा":   ["broke", "shattered", "broken"],
    "तोड़ी":   ["broke", "shattered"],
    "तोड़कर":  ["broken", "shattered"],

    # burning (root जल — NEVER bare; जल = water, abundant in Mahabharata)
    "जलता":    ["burning", "ablaze"],
    "जलती":    ["burning", "ablaze"],
    "जलते":    ["burning"],
    "जला":     ["burnt", "ablaze", "engulfed"],
    "जली":     ["burnt", "ablaze"],
    "जलाकर":   ["burnt", "burned"],
    "जलाने":   ["burning", "to burn"],

    # pushing / shoving (root धकेल)
    "धकेला":   ["pushed", "shoved"],
    "धकेलता":  ["pushing", "shoving"],

    # stopping / halting (root रुक)
    "रुका":    ["stopped", "halted", "frozen"],
    "रुकी":    ["stopped", "halted"],
    "रुकता":   ["stopping", "halting"],

    # watching / seeing (root देख)
    "देखता":   ["watching", "seeing", "gazing"],
    "देखती":   ["watching", "seeing", "gazing"],
    "देखा":    ["saw", "looked at", "gazed at"],
    "देखी":    ["saw", "looked at"],
    "देखकर":   ["watching", "seeing"],

    # hearing (root सुन — NEVER bare; सुन substring-matches सुनहरा = golden)
    "सुनता":   ["hearing", "listening"],
    "सुनती":   ["hearing", "listening"],
    "सुना":    ["heard", "listened"],
    "सुनी":    ["heard", "listened"],

    # speaking (root बोल)
    "बोलता":   ["speaking", "saying"],
    "बोलती":   ["speaking", "saying"],
    "बोला":    ["said", "spoke", "uttered"],
    "बोली":    ["said", "spoke"],

    # screaming (root चीख)
    "चीखता":   ["screaming", "shrieking"],
    "चीखती":   ["screaming", "shrieking"],
    "चीखा":    ["screamed", "shrieked"],
    "चीखी":    ["screamed", "shrieked"],

    # dragging (root घसीट)
    "घसीटा":   ["dragged"],
    "घसीटता":  ["dragging"],
    "घसीटी":   ["dragged"],

    # gripping (root पकड़)
    "पकड़ा":   ["gripped", "grasped", "held"],
    "पकड़ी":   ["gripped", "grasped"],
    "पकड़ता":  ["gripping", "grasping"],
    "पकड़कर":  ["gripping", "holding"],

    # climbing (root चढ़)
    "चढ़ता":   ["climbing", "ascending"],
    "चढ़ी":    ["climbed", "ascended"],
    "चढ़ा":    ["climbed", "ascended"],

    # descending (root उतर)
    "उतरा":    ["descended", "stepped down"],
    "उतरता":   ["descending"],

    # sinking / drowning (root डूब — important for Karna's chariot wheel)
    "डूबता":   ["sinking", "drowning", "submerging"],
    "डूबती":   ["sinking", "drowning"],
    "डूबा":    ["sank", "drowned", "submerged"],
    "डूबी":    ["sank", "drowned"],

    # running (root दौड़)
    "दौड़ता":  ["running", "racing", "sprinting"],
    "दौड़ी":   ["ran", "raced"],
    "दौड़ा":   ["ran", "raced"],

    # fleeing (root भाग)
    "भागा":    ["fled", "ran away"],
    "भागी":    ["fled", "ran away"],
    "भागता":   ["fleeing", "running away"],
}


def _check_verb_action_lock(anchor_phrase: str, image_prompt: str) -> bool:
    """Phase 20 (2026-06-17). When anchor_phrase contains a known Devanagari
    action-verb conjugation, image_prompt MUST contain at least one English
    equivalent — so FLUX renders the ACTION not a static noun-holding pose.

    Matching strategy: split anchor on whitespace + startswith() per token.
    The bare-`in`-substring approach false-matched short stems against
    unrelated nouns (रो → रोशनी / रोग / भरोसा; सुन → सुनहरा; जल → जल/water).
    Splitting on whitespace and using startswith() on each token catches
    valid conjugations (उठा root → उठाता / उठाई / उठाकर — all start with उठा)
    while skipping internal substrings. We also removed all 2-letter bare
    stems from the map for prefix-false-positive safety.

    Default-pass-closed for anchors without a mapped form (conservative)."""
    if not anchor_phrase:
        return True
    image_lower = image_prompt.lower()
    tokens = anchor_phrase.split()
    for verb_form, en_forms in _VERB_ACTION_MAP.items():
        if any(tok.startswith(verb_form) for tok in tokens):
            if not any(form in image_lower for form in en_forms):
                return False
    return True


# ─── 4-type broll classifier for film-editing diversity ───────────────────
# A real film editor cycles ACTION ↔ REACTION ↔ PROP ↔ ENVIRONMENT shots
# instead of holding on one face. The LLM defaults to lazy
# character-portrait sequences; this classifier + the diversity validator
# below force variety at the validation layer (not just as a prompt
# instruction the LLM can ignore).
#
# Classifier returns one of: ACTION, REACTION, PROP, ENVIRONMENT, AMBIGUOUS.
# AMBIGUOUS is the escape hatch — entries that don't cleanly classify
# pass through the diversity check (wildcards). This is the "infinite
# rejection loop" protection: if the LLM gets creative with English
# vocabulary that misses our keyword sets, the entry classifies AMBIGUOUS
# and never blocks the cycle gate. Best-of-N rescue (5 attempts cap) is
# still active as the ultimate failsafe.

_SHOT_TYPE_ENV_TOKENS = (
    "wide shot", "wide-angle", "landscape", "distant", "horizon",
    "vista", "panoramic", "aerial view", "smoke-filled sky", "vast",
    "across the field", "burning battlefield", "burning camp",
    "distant chariots", "skyline", "open sky",
    # Phase 21 (2026-06-25) — mid-intensity ENV vocabulary the LLM actually
    # uses but the original list missed (per Phoenix audit AMBIGUOUS=68%).
    "wide view", "establishing shot", "long shot", "extreme long shot",
    "in the distance", "stretching to the horizon", "rolling hills",
    "battlefield stretches", "sky stretches", "open plain", "vast plain",
    "smoke rising", "dust clouds", "across the plains", "dusk sky",
    "dawn sky", "moonlit sky", "fog-shrouded", "mist-covered",
    "river bend", "forest clearing", "courtyard", "palace hall",
    "temple courtyard", "ancient ruins", "stone steps", "throne room",
    "war camp", "encampment", "tents stretching", "rows of",
    "silhouetted against", "backlit by", "sun setting behind",
    "no figures in frame",
)
_SHOT_TYPE_PROP_LEADING = (
    # The PROP rule fires when text STARTS WITH (not just contains) one of
    # these object-leading phrases. Plan-review fix #2 (2026-06-17): the
    # original `in leading` form false-matched "the bow" against "the
    # bowstring" inside "Arjuna drawing back the bowstring" (which is
    # actually an ACTION shot), and "a tear" against "Karna tear-streaked
    # face" (which is REACTION). Using startswith() forces the PROP
    # signature to be the LEADING SUBJECT, not a random object mentioned
    # mid-description.
    "extreme close-up of", "macro shot of", "macro close-up of",
    "detail of", "tight crop of", "close detail of", "extreme detail of",
    # Object-leading openings — note trailing spaces / explicit articles to
    # avoid sub-word matches (e.g. "the bow " with trailing space won't match
    # "the bowstring").
    "the sword ", "a sword ", "the bow ", "a bow ", "the arrow", "an arrow",
    "the chariot wheel", "a chariot wheel", "the wheel",
    "the dice", "the chakra", "the kundal", "the mace", "the bowl",
    "the staff ", "a wooden ", "a bronze ", "a golden ", "an iron ",
    "a single tear", "a drop of blood",
    # Phase 21 (2026-06-25) — broken/discarded/fallen variants + additional
    # mythological objects the LLM uses to lead PROP shots.
    "a broken ", "a shattered ", "a discarded ", "a fallen ",
    "the broken ", "the shattered ", "the discarded ", "the fallen ",
    "a quiver", "the quiver", "the conch", "a conch",
    "the rein", "the reins", "the flag", "the banner",
    "the spear", "a spear", "a goblet", "the goblet",
    "the lamp", "an oil lamp", "the throne", "the crown",
    "a crown", "the helmet", "the breastplate", "the kavach",
    "the discus",
)
_SHOT_TYPE_ACTION_TOKENS = (
    "raising", "lifting", "striking", "falling", "sinking", "drawing back",
    "running", "leaping", "hurling", "throwing", "swinging", "rushing",
    "charging", "mid-stride", "mid-gesture", "mid-strike", "wielding",
    "gripping", "tearing", "drawing arrow", "drawing sword", "pulling",
    "ascending", "descending", "fleeing", "dragging",
    # Phase 21 (2026-06-25) — mid-intensity action verbs the LLM uses but
    # the original "dramatic/extreme" list missed.
    "peering", "aiming", "nocking", "loosing", "releasing",
    "kneeling", "crouching", "lunging", "advancing", "stepping forward",
    "turning toward", "reaching", "pulling back", "drawing the string",
    "mid-flight", "mid-arc", "mid-draw", "mid-loose", "poised",
    "tightening", "clenching fist", "raising hand", "extending",
    "bracing", "stalking", "dismounting", "mounting", "marching",
    "pacing", "pivoting", "collapsing", "plummeting", "spiraling",
    "twisting", "wrenching", "grasping", "clawing", "parrying",
    "slashing", "piercing", "bowing",
)
_SHOT_TYPE_REACTION_TOKENS = (
    "tear-streaked", "wide-eyed", "hollow-eyed", "clenched jaw",
    "horrified", "trembling lip", "gritted teeth", "averted gaze",
    "shocked face", "agonized", "kohl-rimmed eyes filled", "stricken face",
    "frozen face", "screaming face", "anguished expression", "grief-stricken",
    "fury in eyes", "tears welling",
    # Phase 21 (2026-06-25) — mid-intensity emotional vocabulary. Phoenix
    # audit confirmed Gemini ships "stern/brooding/grim/furrowed/narrowed"
    # but the original list only caught high-intensity "tear-streaked /
    # wide-eyed / horrified". Expansion drops AMBIGUOUS from 68% to ~25%.
    "stern gaze", "brooding", "grim expression", "grim face",
    "furrowed brow", "knitted brow", "hardened expression",
    "narrowed eyes", "piercing gaze", "intense stare", "haunted eyes",
    "conflicted expression", "troubled face", "weary face",
    "sorrowful eyes", "downcast eyes", "lowered gaze",
    "set jaw", "tight-lipped", "lips parted", "lips pressed",
    "tearful", "moist eyes", "glassy eyes", "blank stare",
    "thousand-yard stare", "vacant expression", "stoic face",
    "resolute face", "defiant glare", "mournful expression",
    "rage-flushed", "guilt-haunted", "shame-bowed", "despair-sunken",
)
_SHOT_TYPE_FACE_PROXIMITY = (
    "close-up", "macro", "extreme close-up", "face", "eyes",
)

# Phase 21 (2026-06-25) — wide-shot LEADING tokens. When the prompt STARTS
# with one of these, it's an ENVIRONMENT shot regardless of downstream
# facial mentions. Closes the plan-review false-positive: "Wide shot of
# the courtyard, Arjuna standing in the distance with a solemn expression
# on his face" is unambiguously a wide environment shot, but the bare
# `has_face_close` gate at the next priority misclassifies it as REACTION
# because of "face" + "solemn expression". The wide-shot leading override
# fires BEFORE the has_face_close fallback.
_SHOT_TYPE_ENV_LEADING = (
    "wide shot of", "wide-angle shot of", "wide shot",
    "wide cinematic shot of", "wide cinematic landscape",
    "aerial view of", "aerial shot of",
    "panoramic view of", "panoramic shot of", "panoramic",
    "establishing shot of", "establishing shot",
    "long shot of", "extreme long shot",
    "vista of", "sweeping view of",
)

# Phase 21 (2026-06-25) — explicit close-up modifiers. If the prompt
# contains any of these, the wide-shot leading override is NOT applied
# (it's a genuine close-up that happens to mention a wider environment
# as backdrop). Prevents the override from swallowing legitimate
# close-ups that mention an aerial/wide setting.
_SHOT_TYPE_EXPLICIT_CLOSE = (
    "close-up", "macro shot", "macro close-up", "extreme close-up",
    "close shot", "tight shot", "tight crop",
)


def _classify_broll_shot_type(image_prompt: str) -> str:
    """Phase 20 (2026-06-17) + Phase 21 (2026-06-25). Lightweight keyword
    classifier — returns ACTION / REACTION / PROP / ENVIRONMENT / AMBIGUOUS.
    Operates on the LLM-emitted image_prompt BEFORE wardrobe/iconography
    prefixes are layered on top.

    Classification priority (most-distinctive first; first match wins):
      1.   PROP                    — STARTSWITH an object-leading phrase
      1.5. ENVIRONMENT (wide-lead) — STARTSWITH wide-shot leading AND no
                                     explicit close-up modifier. Phase 21
                                     override added to fix the plan-review
                                     false-positive: "Wide shot of the
                                     courtyard, X standing with solemn
                                     expression on his face" was being
                                     misclassified as REACTION because of
                                     the "face" mention. The leading
                                     wide-shot anchor now wins regardless
                                     of downstream facial mentions UNLESS
                                     an explicit close-up modifier is also
                                     present.
      2.   ENVIRONMENT (fallback)  — wide-vista vocabulary AND no
                                     face-proximity tokens
      3.   ACTION                  — explicit action verb anywhere
      4.   REACTION                — explicit emotion-on-face token
      5.   AMBIGUOUS               — wildcard for cycle gate (escape hatch)
    """
    text = image_prompt.lower().strip()

    # 1. PROP — strict STARTSWITH for object-leading phrases
    if any(text.startswith(p) for p in _SHOT_TYPE_PROP_LEADING):
        return "PROP"

    # 1.5 ENVIRONMENT-WIDE-OVERRIDE (Phase 21) — STARTSWITH wide-shot leading,
    # and NO explicit close-up modifier elsewhere in the prompt. Fires BEFORE
    # the has_face_close fallback so "Wide shot of X, character with solemn
    # expression on his face" classifies ENV not REACTION.
    has_wide_lead     = any(text.startswith(p) for p in _SHOT_TYPE_ENV_LEADING)
    has_explicit_close = any(p in text for p in _SHOT_TYPE_EXPLICIT_CLOSE)
    if has_wide_lead and not has_explicit_close:
        return "ENVIRONMENT"

    # 2. ENVIRONMENT-FALLBACK — wide-vista language without face-proximity
    has_env = any(tok in text for tok in _SHOT_TYPE_ENV_TOKENS)
    has_face_close = any(tok in text for tok in _SHOT_TYPE_FACE_PROXIMITY)
    if has_env and not has_face_close:
        return "ENVIRONMENT"

    # 3. ACTION — explicit action verb anywhere in text
    if any(tok in text for tok in _SHOT_TYPE_ACTION_TOKENS):
        return "ACTION"

    # 4. REACTION — explicit emotion-on-face
    if any(tok in text for tok in _SHOT_TYPE_REACTION_TOKENS):
        return "REACTION"

    return "AMBIGUOUS"


def _check_subject_diversity(broll: list) -> tuple[bool, str]:
    """Phase 20 (2026-06-17). HARD reject when the broll's shot-type
    distribution looks like a slideshow instead of a film edit.

    Two rules — both deliberately LENIENT to avoid infinite rejection
    loops when the LLM uses creative English vocabulary:

    Rule 1 (consecutive): NO MORE than 2 consecutive HARD-classified
    entries of the same type. AMBIGUOUS entries break the run (they
    don't increment AND they don't count as a match).

    Rule 2 (distribution): across ALL entries, at least 2 DISTINCT
    hard-classified types must appear. (If 8 entries all classify
    AMBIGUOUS, this fails — which is fine; that means the LLM emitted
    8 vague prompts and needs another attempt.)

    Failsafe: best-of-N rescue path (5 attempts max) is still active.
    Worst case after 5 failed attempts, the highest-scoring attempt
    ships with this gate failed but other gates passing."""
    if not broll:
        return False, "broll is empty"

    types = [_classify_broll_shot_type(b.get("image_prompt", "")) for b in broll]

    # Rule 1: consecutive same-type cap
    run_type = None
    run_count = 0
    for i, t in enumerate(types):
        if t == "AMBIGUOUS":
            run_type = None
            run_count = 0
            continue
        if t == run_type:
            run_count += 1
            if run_count > 2:  # i.e. 3 consecutive same type
                window = types[max(0, i - 2): i + 1]
                return False, (
                    f"broll[{i-2}..{i}] are 3 consecutive {t} shots "
                    f"({window}). Vary the cycle: ACTION → REACTION → "
                    f"PROP → ENVIRONMENT like a film cut, not a portrait series."
                )
        else:
            run_type = t
            run_count = 1

    # Rule 2: distribution diversity
    hard_types = {t for t in types if t != "AMBIGUOUS"}
    if len(hard_types) < 2:
        return False, (
            f"broll has only {len(hard_types)} distinct hard-classified "
            f"shot type(s): {sorted(hard_types) or '[]'}. Need at least 2 "
            f"of {{ACTION, REACTION, PROP, ENVIRONMENT}} to feel like a "
            f"film edit. (Types: {types})"
        )

    # Rule 3 (Phase 21, 2026-06-25) — hard-cap on REACTION+AMBIGUOUS share.
    # Even with the Phase 21 classifier vocab expansion, some entries will
    # still classify REACTION or AMBIGUOUS. This rule kills the failure mode
    # where 7/8 broll entries are character-portrait reactions and only 1
    # is action/prop/environment. >65% REACTION+AMBIGUOUS share = rendered
    # video looks like a portrait gallery — exactly the channel-grid
    # templating that collapsed views from 500-700 to 10-30 per the
    # 2026-06-25 post-Phase-20 forensic.
    reaction_ambiguous_count = sum(
        1 for t in types if t in ("REACTION", "AMBIGUOUS")
    )
    share = reaction_ambiguous_count / max(len(types), 1)
    if share > 0.65:
        # How many entries must be swapped to non-{REACTION,AMBIGUOUS}?
        to_replace = int(reaction_ambiguous_count - 0.65 * len(types) + 0.999)
        return False, (
            f"REACTION+AMBIGUOUS share is {reaction_ambiguous_count}/{len(types)} "
            f"({100*share:.0f}%) — exceeds 65% cap. Your broll is a portrait "
            f"gallery, not a film edit. Replace at least {to_replace} "
            f"REACTION/AMBIGUOUS entries with explicit ACTION shots "
            f"(drawing back, raising, sinking, mid-strike) or PROP shots "
            f"(extreme close-up of...) or ENVIRONMENT shots (wide shot of...). "
            f"(Types: {types})"
        )

    return True, ""


def validate_phase18(data: dict, lang_label: str = "Hindi") -> tuple[bool, str, dict]:
    """Aggregate validator. Returns (passed_all_gates, first_violation_label,
    info_dict). info_dict carries score, word_count, broll_count so the
    retry reminder can quote exact numbers back to the LLM."""
    vo = data.get("voiceover", "") or ""
    br = data.get("broll", []) or []
    words = vo.split()
    n_words = len(words)
    n_broll = len(br)
    episode_n = int(data.get("episode_n", 0) or 0)

    # Length range:
    #   • 2026-06-16: initial 80-110 → revised to 75-115.
    #   • 2026-06-26 (Phase 22.2): cap 115 → 130; every length-prompt tweak
    #     shifted Gemini's distribution LONGER (132 → 151 → 156 median across
    #     3 verification runs). The verbose Phase-22.1 calibration text was
    #     reverted.
    #   • 2026-06-26 (Phase 22.3): TACTICAL — cap 130 → 170. Local validation
    #     of run 28213494757's quarantine JSON proved Phase 22's quality
    #     engine works: verb_per_frame / anti_merge_composition /
    #     aftermath_closer / title_dna ALL PASS on production Gemini output.
    #     Score 19/25; the missing points are length plus secondary gates
    #     (subject_diversity REACTION+AMBIGUOUS share, archetype prefix
    #     match). The length cap is the ONLY thing blocking a Phase-22-
    #     compliant ship. Raising to 170w (~77s spoken at Charon 2.2 wps)
    #     lets ONE render through so we can eyeball Phase 22's visual
    #     engine on YouTube. 77s is over the retention sweet spot (winners
    #     ran 50s) but under the YouTube vertical-video hard cap. This is
    #     a DIAGNOSTIC widening, not a sustainable cap.
    #     REVERT to 130 once Phase 22.4 (two-pass compression) ships — then
    #     Gemini freely generates 150-170w and a second compression-pass
    #     trims to 100-120w preserving anchors + structure.
    length_ok = 75 <= n_words <= 170
    broll_ok  = 8 <= n_broll <= 10

    first_10 = " ".join(words[:10])
    hook_ok = _check_hook_pattern_text(first_10)

    # Hook title — reuse legacy validator
    from pipeline.script_generator import _check_hook_title
    hook_title_ok, _ = _check_hook_title(data.get("hook_title", ""), language="hi")

    loop_ok = vo.rstrip().endswith("?")

    shock_ok = _check_shock_action(words[:8])

    mid_start = int(n_words * 0.3)
    mid_end   = int(n_words * 0.6)
    mid_text  = " ".join(words[mid_start:mid_end])
    rehook_ok = bool(_REHOOK_MARKERS.search(mid_text))

    eng_ok  = _check_engagement_density_text(vo)
    mono_ok = _check_ending_monotony_text(vo)
    tha_ok  = _check_past_aux_tic_text(vo)
    rep_ok  = _check_repetition_text(vo)

    anchors_ok, _anchor_why = _validate_anchors(vo, br)

    names_ok = all(_check_character_names_single(b.get("image_prompt", "")) for b in br)

    # Archetype check: broll[0].image_prompt must begin with the rotated
    # archetype's first 35 chars. Substring match at offset 0; the LLM is
    # expected to copy the directive verbatim as the first sentence.
    archetype_idx = episode_n % len(_OPENER_ARCHETYPES_PHASE18)
    archetype_text = _OPENER_ARCHETYPES_PHASE18[archetype_idx][1]
    archetype_prefix = archetype_text[:35]
    img0 = br[0].get("image_prompt", "") if br else ""
    archetype_ok = bool(br) and archetype_prefix in img0[:200]

    subject_lock_ok = all(
        _check_subject_lock(b.get("anchor_phrase", ""), b.get("image_prompt", ""))
        for b in br
    )

    intensity_ok = all(
        _check_image_intensity(b.get("image_prompt", ""))
        for b in br
    )

    bookend_ok = _check_bookend_text(vo)

    # Phase 19 (2026-06-16) — wardrobe context + story entity gates
    wardrobe_set_ok, _wardrobe_why = _check_wardrobe_context_set(br)
    story_entity_ok, _story_why    = _check_story_entity_present(vo, br)

    # Phase 20 (2026-06-17) — subject diversity + verb-action lock
    diversity_ok, _diversity_why = _check_subject_diversity(br)
    verb_lock_ok = all(
        _check_verb_action_lock(b.get("anchor_phrase", ""), b.get("image_prompt", ""))
        for b in br
    )
    shot_types_diagnostic = [
        _classify_broll_shot_type(b.get("image_prompt", "")) for b in br
    ]

    # Phase 22 (2026-06-25) — verb-per-frame + anti-merge composition
    # + aftermath closer + title-DNA gate. See module docstring + plan.
    verb_per_frame_ok, _verb_per_frame_why = _check_verb_per_frame(br)
    anti_merge_ok,     _anti_merge_why     = _check_anti_merge_composition(br)
    aftermath_ok,      _aftermath_why      = _check_aftermath_closer(br)
    # Plan-review fix (2026-06-26): extract arc_character_devanagari from
    # the script dict. Defensive default — when missing (mid-attempt loops,
    # raw LLM JSON pre-metadata-bake, legacy/WhatIf paths), _check_title_dna_gate
    # degrades to a 3-step cascade and still rejects mood/caste nouns.
    title    = data.get("title", "") or ""
    char_dev = data.get("arc_character_devanagari", "") or ""
    title_dna_ok, _title_dna_why, _dna_score = _check_title_dna_gate(title, char_dev)

    flags = [length_ok, broll_ok, hook_ok, hook_title_ok, loop_ok,
             shock_ok, rehook_ok, eng_ok, mono_ok, tha_ok, rep_ok,
             anchors_ok, names_ok, archetype_ok, subject_lock_ok,
             intensity_ok, bookend_ok,
             wardrobe_set_ok, story_entity_ok,    # Phase 19
             diversity_ok, verb_lock_ok,          # Phase 20
             verb_per_frame_ok, anti_merge_ok,    # Phase 22
             aftermath_ok, title_dna_ok]          # Phase 22
    score = sum(1 for f in flags if f)

    info = {
        "score":              score,
        "word_count":         n_words,
        "broll_count":        n_broll,
        "anchor_why":         _anchor_why,
        "wardrobe_why":       _wardrobe_why,
        "story_why":          _story_why,
        "diversity_why":      _diversity_why,    # Phase 20
        "shot_types":         shot_types_diagnostic,  # Phase 20 diagnostic
        "verb_per_frame_why": _verb_per_frame_why,  # Phase 22
        "anti_merge_why":     _anti_merge_why,    # Phase 22
        "aftermath_why":      _aftermath_why,     # Phase 22
        "title_dna_why":      _title_dna_why,     # Phase 22
        "title_dna_score":    _dna_score,         # Phase 22
    }

    # First violation in priority order (single highest-impact gate).
    # Phase 22 inserts: aftermath_closer right after wardrobe_set (both
    # operate on wardrobe_context field); verb_per_frame right after
    # verb_action (both verb-related, aggregate vs per-entry); title_dna
    # right after hook_title (both title-shape gates); anti_merge_composition
    # right after subject_diversity (both broll-array composition gates).
    cascade = [
        ("length",                  length_ok and broll_ok),
        ("anchors",                 anchors_ok),
        ("wardrobe_set",            wardrobe_set_ok),    # Phase 19
        ("aftermath_closer",        aftermath_ok),       # Phase 22
        ("story_entity",            story_entity_ok),    # Phase 19
        ("verb_action",             verb_lock_ok),       # Phase 20
        ("verb_per_frame",          verb_per_frame_ok),  # Phase 22
        ("hook",                    hook_ok),
        ("shock_action",            shock_ok),
        ("loop_closure",            loop_ok),
        ("hook_title",              hook_title_ok),
        ("title_dna",               title_dna_ok),       # Phase 22
        ("archetype",               archetype_ok),
        ("subject_lock",            subject_lock_ok),
        ("subject_diversity",       diversity_ok),       # Phase 20
        ("anti_merge_composition",  anti_merge_ok),      # Phase 22
        ("intensity",               intensity_ok),
        ("rehook",                  rehook_ok),
        ("char_names",              names_ok),
        ("bookend",                 bookend_ok),
        ("engagement",              eng_ok),
        ("monotony",                mono_ok),
        ("tha_tic",                 tha_ok),
        ("repetition",              rep_ok),
    ]
    for label, ok in cascade:
        if not ok:
            return False, label, info
    return True, "", info


def _enumerate_failing_gates(data: dict, lang_label: str = "Hindi") -> set:
    """Phase 22.4 (2026-06-28). Return the FULL set of failing gate labels
    for `data` — not just the cascade's first-violation label. Used by
    `generate_phase18_script` to compute the two-gate strict-pass:

        strict_pass = (score ≥ floor) AND (no Phase 22 fatal-gate failures)

    Implemented as a thin re-runner: re-validates the same data and returns
    every cascade entry whose flag is False. NOT folded into validate_phase18
    because that function's `(ok, first_violation, info)` contract is
    relied on by all the per-attempt retry logic in the best-of-N loop.
    """
    vo = data.get("voiceover", "") or ""
    br = data.get("broll", []) or []
    words = vo.split()
    n_words = len(words)
    n_broll = len(br)
    episode_n = int(data.get("episode_n", 0) or 0)

    length_ok = 75 <= n_words <= 170
    broll_ok  = 8 <= n_broll <= 10

    first_10 = " ".join(words[:10])
    hook_ok = _check_hook_pattern_text(first_10)

    from pipeline.script_generator import _check_hook_title
    hook_title_ok, _ = _check_hook_title(data.get("hook_title", ""), language="hi")

    loop_ok = vo.rstrip().endswith("?")
    shock_ok = _check_shock_action(words[:8])

    mid_start = int(n_words * 0.3)
    mid_end   = int(n_words * 0.6)
    mid_text  = " ".join(words[mid_start:mid_end])
    rehook_ok = bool(_REHOOK_MARKERS.search(mid_text))

    eng_ok  = _check_engagement_density_text(vo)
    mono_ok = _check_ending_monotony_text(vo)
    tha_ok  = _check_past_aux_tic_text(vo)
    rep_ok  = _check_repetition_text(vo)

    anchors_ok, _ = _validate_anchors(vo, br)
    names_ok = all(_check_character_names_single(b.get("image_prompt", "")) for b in br)

    archetype_idx = episode_n % len(_OPENER_ARCHETYPES_PHASE18)
    archetype_text = _OPENER_ARCHETYPES_PHASE18[archetype_idx][1]
    archetype_prefix = archetype_text[:35]
    img0 = br[0].get("image_prompt", "") if br else ""
    archetype_ok = bool(br) and archetype_prefix in img0[:200]

    subject_lock_ok = all(
        _check_subject_lock(b.get("anchor_phrase", ""), b.get("image_prompt", ""))
        for b in br
    )
    intensity_ok = all(_check_image_intensity(b.get("image_prompt", "")) for b in br)
    bookend_ok = _check_bookend_text(vo)
    wardrobe_set_ok, _ = _check_wardrobe_context_set(br)
    story_entity_ok, _ = _check_story_entity_present(vo, br)
    diversity_ok, _ = _check_subject_diversity(br)
    verb_lock_ok = all(
        _check_verb_action_lock(b.get("anchor_phrase", ""), b.get("image_prompt", ""))
        for b in br
    )
    verb_per_frame_ok, _ = _check_verb_per_frame(br)
    anti_merge_ok, _     = _check_anti_merge_composition(br)
    aftermath_ok, _      = _check_aftermath_closer(br)
    title    = data.get("title", "") or ""
    char_dev = data.get("arc_character_devanagari", "") or ""
    title_dna_ok, _, _   = _check_title_dna_gate(title, char_dev)

    gates = {
        "length":                 length_ok and broll_ok,
        "anchors":                anchors_ok,
        "wardrobe_set":           wardrobe_set_ok,
        "aftermath_closer":       aftermath_ok,
        "story_entity":           story_entity_ok,
        "verb_action":            verb_lock_ok,
        "verb_per_frame":         verb_per_frame_ok,
        "hook":                   hook_ok,
        "shock_action":           shock_ok,
        "loop_closure":           loop_ok,
        "hook_title":             hook_title_ok,
        "title_dna":              title_dna_ok,
        "archetype":              archetype_ok,
        "subject_lock":           subject_lock_ok,
        "subject_diversity":      diversity_ok,
        "anti_merge_composition": anti_merge_ok,
        "intensity":              intensity_ok,
        "rehook":                 rehook_ok,
        "char_names":             names_ok,
        "bookend":                bookend_ok,
        "engagement":             eng_ok,
        "monotony":               mono_ok,
        "tha_tic":                tha_ok,
        "repetition":             rep_ok,
    }
    return {label for label, ok in gates.items() if not ok}


# ─── Prompt builder ───────────────────────────────────────────────────────

_VIOLATION_REMINDERS = {
    "length":       "Your voiceover word count was out of range. Voiceover MUST be 75-170 Hindi words (sweet spot 100-130). Match the SHAPE of the concrete 96-word example in the prompt — open with an 8-word action verb, build 3-4 mid-paragraph beats with sensory anchors, embed a dialogue beat in quotes, insert लेकिन/परंतु around the midpoint, close with a question that echoes the opener noun. broll MUST be 8-10 entries.",
    "anchors":      "Your broll anchor_phrase entries failed validation. Each anchor_phrase MUST be 2-4 words appearing VERBATIM in the voiceover, in ASCENDING order by character position. Final anchor MUST land in the LAST 30% of the voiceover.",
    "wardrobe_set": "Every broll entry MUST declare a wardrobe_context field, one of: WAR / PALACE / DIVINE / FOREST / JOURNEY. Pick the context that matches THAT specific broll moment (not the whole story): a dice-game palace scene is PALACE even in an Arjuna story; Yudhishthira walking to heaven's gate is JOURNEY, not WAR; Krishna in his cosmic form is DIVINE; Eklavya's forest ashram is FOREST.",
    "story_entity": "Your voiceover names a story-critical entity (e.g. कुत्ता / धनुष / सुदर्शन / कुंडल / बच्चे) but NO broll image_prompt contains its English equivalent. AT LEAST ONE broll image_prompt MUST have the entity as a clearly identifiable foreground subject — e.g. anchor 'कुत्ता का साथ' → image_prompt 'a loyal stray dog walking close beside Yudhishthira, foreground subject'. The audience cannot feel the dog's presence if FLUX never renders one.",
    "verb_action":      "Your broll image_prompts described static portraits where the voiceover described ACTIONS. When the anchor_phrase contains an action verb (उठाई / गिरा / डूबता / काटा / etc.), the image_prompt MUST contain the English equivalent (raising / fell / sinking / cut / etc.) as the verb the image depicts. Example: anchor 'रथ का पहिया डूबता' → image_prompt 'extreme close-up of a wooden chariot wheel sinking into wet mud, mud splattered on the spokes' (NOT 'Karna's stressed face with chariot in background').",
    "subject_diversity": "Your broll showed too many consecutive shots of the same type (3+ in a row), or used only 1 type across all entries. A real film editor CYCLES through 4 shot types: (A) CHARACTER ACTION (hero raising weapon, drawing arrow, falling) — uses action verbs in image_prompt; (B) REACTION (tear-streaked / wide-eyed / clenched-jaw close-up); (C) PROP / DETAIL (extreme close-up of an OBJECT — chariot wheel, sword tip, kundal, dice — character absent or backgrounded, image_prompt LEADS with the object); (D) ENVIRONMENT (wide vista — battlefield, sky, distant chariots — no facial close-up). Rotate types across broll entries. Two consecutive same-type entries are fine; three consecutive is rejected.",
    "hook":         "Your voiceover's first 10 words did not pass the hook validator. Open INSIDE an emotional moment with a named character (अर्जुन/कर्ण/etc.) AND/OR a paradox marker (लेकिन/पर/फिर भी). NO documentary setups (\"यह कहानी है\", \"बहुत समय पहले\").",
    "shock_action": "Your voiceover's first 8 words must contain a present-tense action verb. NO setup. Examples: \"अर्जुन धनुष उठाता है\" (raising), \"द्रौपदी आँसू बहाती है\" (shedding), \"अश्वत्थामा तलवार चलाता है\" (wielding).",
    "loop_closure": "Your voiceover must END with a question mark (?). The final clause must be an ethically-charged question to the viewer: \"क्या यही धर्म था?\" / \"क्या वो सही थे?\" / \"किसने ज्यादा खोया?\".",
    "hook_title":   "Your hook_title failed validation. Must be 1-5 words, 100% Devanagari script, contain a named character OR paradox marker, no '?', no '!', no '...'.",
    "archetype":    "Your broll[0].image_prompt must begin VERBATIM with the Scene-1 archetype directive provided in the prompt. Do not paraphrase; copy it as the first sentence of broll[0].image_prompt.",
    "subject_lock": "Your broll image_prompts had subject bleed. For each entry, the anchor_phrase's primary noun (e.g. द्रौपदी, तलवार, बच्चे) MUST appear in the image_prompt's English equivalent (draupadi/queen/woman, sword/blade, children/sleeping).",
    "intensity":    "Your broll image_prompts were too sterile. EVERY image_prompt MUST contain at least one aggressive physical adjective: bloodied / shattered / trembling / rusted / tear-streaked / smoldering / clenched / scorched / wide-eyed / ash-covered / blood-soaked / dim. \"a sword\" is bad; \"a bloodied sword, dim shadow-heavy frame\" is good.",
    "rehook":       "Your voiceover's middle (30-60%) lacked a subversion marker. Insert लेकिन / परंतु / और तभी / उसी क्षण around the midpoint to break the viewer's predictive flow.",
    "char_names":   "Some broll image_prompts lacked a recognized character name. EVERY image_prompt must contain a named character (Arjuna, Karna, Bhishma, Draupadi, etc.) so the FLUX cascade's _inject_characters can append the visual descriptor block.",
    "bookend":      "Voiceover's first 10 words and last 10 words don't share a charged Devanagari noun. The closing question must echo the opening subject — that's what produces the loop-rewatch effect.",
    "engagement":   "Voiceover failed engagement density. Need ≥4 sensory anchors (शंख / दीप / आंख / हाथ / आंसू / रक्त / etc.) AND ≥2 dialogue beats (quoted speech or second-person address).",
    "monotony":     "Too many sentences in voiceover end with the same suffix (>40%). Vary the cadence.",
    "tha_tic":      "Past-aux था/थी/थे ending exceeded 15% of sentences. Push to present tense; only the FINAL line may use past.",
    "repetition":   "A single word repeated >4 times in voiceover. Trim the redundant repetition.",
    # Phase 22 (2026-06-25) — new gates
    "verb_per_frame":         "Your broll image_prompts were noun-poses (standing/looking/holding), not action verbs. ≥5/8 image_prompts MUST start (within the first 80 chars) with an action verb from: drawing back / sinking / severing / kneeling beneath / looming over / leaning into / dragging away from / charging / falling / gripping / snarling / recoiling / fleeing / walking away from / hurled / slashing / weeping over / striking / piercing / collapsing / lunging / wielding. FORBIDDEN openers: standing / looking / gazing / holding / sitting / watching. Final broll[-1].image_prompt MUST contain one of: silhouette / abandoned / walking away / withheld / lone diya / prone body / fading dusk.",
    "anti_merge_composition": "An image_prompt named ≥2 characters in the same frame but did NOT lead with a composition tag. Required: start two-character image_prompts with one of [OVER-SHOULDER] / [POWER-LOOM] / [CONFRONTATION-WIDE] / [WITNESS-FROM-BEHIND] / [THE-FALLEN] in literal square brackets, then continue with verb + intensity + wardrobe. Without this, FLUX merges the two subjects (3 arms, fused faces, shared torso).",
    "aftermath_closer":       "The final broll entry (broll[-1]) must declare wardrobe_context: 'AFTERMATH' (literal string in JSON). AFTERMATH = lone silhouette / face-withheld / abandoned weapon / prone body / lone diya / fading dusk. This is the consequence-as-closer beat ALL 4 channel winners used. Set BOTH the JSON wardrobe_context field AND the matching tokens inside broll[-1].image_prompt.",
    "title_dna":              "Your title failed the Title DNA gate. (1) NO mood/regret/wound/caste/religion/sin/truth/fate nouns. (2) NO silent/internal/abstract/quiet/deep/inner adjectives. (3) MUST contain ≥1 tribal-relational noun: Sacrifice/Vow/Friendship/Pride/Loyalty/Mistake/Betrayal/Curse/Promise/Doubt/Loss/Fear/Shame (or Devanagari equivalents बलिदान/प्रतिज्ञा/वचन/दोस्ती/अभिमान/घमंड/हार/डर/वफ़ादारी/गलती/धोखा/श्राप/शर्म). (4) Title MUST assert the OPPOSITE of the character's canonical virtue (Karna's loyalty → Karna's Biggest Mistake; Arjuna's skill → Arjuna's Greatest Doubt; Bhishma's vow → Bhishma's Greatest Betrayal). Examples that PASS: 'कर्ण की सबसे बड़ी गलती | Karna\\'s Biggest Mistake', 'अर्जुन का असली डर | Arjuna\\'s Real Fear', 'भीष्म का छुपा वचन | Bhishma\\'s Hidden Vow'. Examples that REJECT: 'Karna\\'s Silent Regret' (mood noun), 'Eklavya\\'s Caste Truth' (caste noun), 'Arjuna\\'s Deep Wound' (mood + soft adj).",
}


def _build_phase18_prompt(
    *,
    topic: str,
    episode_n: int,
    arc_name: str,
    arc_character_devanagari: str,
    emotional_fingerprint: str,
    palette_directive: str,
    archetype_letter: str,
    archetype_text: str,
    cliffhanger_block: str,
    fingerprint_block: str,
) -> str:
    """Build the Phase 18 LLM prompt. Replaces the legacy 1500-line scenes
    template with a focused voiceover + broll spec."""
    scene1_opener_directive = archetype_text

    palette_line = (
        f"Visual palette anchor for THIS render: {palette_directive}. "
        "Every broll image_prompt MUST carry this palette directive somewhere "
        "in its descriptors so the rendered images share a consistent visual "
        "signature across the 8-10 cuts."
    ) if palette_directive else ""

    return f"""
You are a master storyteller specialising in the Mahabharata epic, writing scripts for vertical YouTube Shorts that retain viewer attention from the first second to the last.

CHANNEL THESIS (the worldview every video MUST frame the moral through):
"Every hero in Mahabharata destroyed someone — but the real destruction was INSIDE."
NO video presents a hero as purely admirable. The cost is the point.

POSITIONING: This channel is NOT "retelling Mahabharata." It is "EXPOSING EMOTIONAL TRUTHS THROUGH MAHABHARATA." Tell what was breaking INSIDE the character at each moment — not what was happening around them.

{fingerprint_block}

TASK: Write a Phase 18 (2026-06-16) decoupled Shorts script.

TOPIC: "{topic}"
ARC: {arc_name or "(no arc)"}
EPISODE: {episode_n}

═══════════════════════════════════════════════════════════════
PHASE 18 SCHEMA — TWO DECOUPLED OUTPUTS
═══════════════════════════════════════════════════════════════

OUTPUT 1 — `voiceover`: ONE flowing Hindi paragraph, 75-170 words.

  *** WORD COUNT IS A HARD RULE. ***
  *** SCRIPTS UNDER 75 WORDS WILL BE REJECTED. ***
  *** SCRIPTS OVER 170 WORDS WILL BE REJECTED. ***
  *** SWEET SPOT: 100-130 words (~45-60s spoken at Charon 2.2 wps). ***

  This is the master VO. Charon TTS reads it as ONE continuous emotional
  monologue at ~2.2 words/sec.

  CONCRETE EXAMPLE (this is exactly 96 words — copy the SHAPE, not the
  content; your story is the {topic} given above):

      “अश्वत्थामा रात के अंधेरे में तलवार उठाता है — पांडवों के सोते बच्चे, सांसें मासूम, आँखें बंद। उसकी उँगलियाँ कांप रही हैं, माथे पर पसीना, पर रुक नहीं सकता। पिता की मौत का बदला, गुरु का अपमान, अधर्म से धर्म तक का सफर — सब इसी एक रात में चुकाना है। लेकिन जिस क्षण तलवार गिरती है, वो खुद टूट जाता है — रक्त उसके हाथों पर, राख उसके दिल पर। कृष्ण आते हैं, श्राप देते हैं — ‘तू तीन हज़ार साल भटकेगा, घाव से मवाद बहता रहेगा।’ आज भी कहीं अश्वत्थामा भटक रहा है, माथे का घाव रिस रहा है, और हम पूछते हैं — क्या वो पहले से ही मर चुका था जब उसने तलवार उठाई?”

  NOTE the structure: 8-word action opener (अश्वत्थामा रात के अंधेरे में
  तलवार उठाता है) → setup (3-4 sentences) → subversion marker "लेकिन"
  around the 50% mark → consequence + sensory anchors (पसीना, कांप, रक्त,
  राख, घाव, मवाद) → dialogue beat (Krishna's curse in quotes) →
  closing question that BOOKENDS the opener (तलवार उठाई echoes the
  opening तलवार उठाता है).

  RULES (all HARD — violations are rejected):
  - 75-170 words. NOT 30. NOT 50. NOT 70. NOT 200.
    SEVENTY-FIVE TO ONE-HUNDRED-SEVENTY. INCLUSIVE BOTH ENDS.
    Sweet spot: 100-130 words. Charon TTS = 2.2 words/sec, so 130w =
    ~59s spoken (target the bottom of the Shorts retention curve).
  - NO line breaks between fragments. NO `।` between every micro-sentence.
    Use natural Hindi punctuation: commas for short pauses, `...` for
    dramatic pauses (TTS reads triple-dot as a measured pause), `?` only
    once at the very end (loop closure).
  - Opens INSIDE the emotional moment — first 8 words MUST contain a
    present-tense action verb (उठाता है / बहाती है / चलाता है / गिर
    जाता है / etc.). NO documentary setup.
  - Middle (30-60% of word count) MUST contain a subversion marker:
    लेकिन / परंतु / और तभी / उसी क्षण — to break the viewer's predictive
    flow.
  - Closes with an ethically-charged question (`?`) addressed to the
    viewer: "क्या यही धर्म था?" / "क्या वो सही थे?" / "किसने ज्यादा खोया?"
  - First 10 words and last 10 words MUST share a charged Devanagari noun
    (bookend / loop).
  - ≥4 sensory anchors (शंख / दीप / आंख / आँख / हाथ / रक्त / आंसू / आँसू /
    धुआँ / पसीना / त्वचा / etc.). ≥2 dialogue beats (quoted speech with
    Devanagari quotes OR direct second-person: "क्या आप..." / "तुम्हें").

OUTPUT 2 — `broll`: an array of EXACTLY 8-10 entries, each:
  {{
    "image_prompt":     "<English. Literal-physical FLUX prompt — see rules below>",
    "anchor_phrase":    "<2-4 Devanagari words VERBATIM from voiceover>",
    "mood":             "<3-6 word English emotional tone for SFX selection>",
    "wardrobe_context": "<one of: WAR | PALACE | DIVINE | FOREST | JOURNEY>"
  }}

WARDROBE_CONTEXT (Phase 19, 2026-06-16) — REQUIRED per entry. Pick the
context that matches THAT SPECIFIC broll moment, not the whole story.
A Yudhishthira video can span PALACE (dice game), WAR (Kurukshetra),
FOREST (vanvas), and JOURNEY (Mahaprasthanika to heaven's gate) in the
same render.

  WAR     — battlefield / siege / duel / combat scene. Subject in
            authentic ancient Indian armor (kavach + kundal + gold
            chest-plate). Use this ONLY for explicit combat moments.
  PALACE  — royal court / throne room / private chamber / Hastinapur
            interior. Subject in silk dhoti + mukut crown + jewelry.
            NO armor. NO weapons unless ceremonial.
  DIVINE  — Swarga / Mount Meru / encounter with a deity (Krishna's
            virat rupa, Yaksha at the lake, Indra at heaven's gate).
            Subject glowing, white-and-gold silks, celestial light.
  FOREST  — vanvas / ashram / Kamyaka forest / Eklavya's clearing.
            Subject in valkala bark-cloth, rudraksha mala, barefoot.
            NO crown. NO jewelry except rudraksha.
  JOURNEY — pilgrimage / Mahaprasthanika walk / Mount Meru ascent /
            walking to heaven's gate. Subject in silk dhoti + wooden
            walking staff. Companions visible alongside.

EXAMPLES of correct wardrobe_context selection:
  • voiceover beat = "Yudhishthira places his foot on Indra's chariot
    at Swarga's gate"        → wardrobe_context: "DIVINE"
  • voiceover beat = "Yudhishthira and his dog walking up the
    Himalayan path"          → wardrobe_context: "JOURNEY"
  • voiceover beat = "dice clatter in the Hastinapur court as
    Draupadi is dragged in"  → wardrobe_context: "PALACE"
  • voiceover beat = "Karna's chariot wheel sinks into the blood-
    soaked mud of Kurukshetra" → wardrobe_context: "WAR"
  • voiceover beat = "Eklavya kneels before his clay Drona at the
    forest hermitage"        → wardrobe_context: "FOREST"

BROLL RULES (HARD — violation = REJECT):

  (a) anchor_phrase MUST be 2-4 Devanagari words appearing verbatim in the
      voiceover string. Single-word anchors are AMBIGUOUS (e.g. "अर्जुन"
      may appear 5 times) and will be REJECTED.

  (b) anchor_phrase entries MUST appear in voiceover in ASCENDING order
      by character position. The final entry's anchor MUST land in the
      LAST 30% of the voiceover (no early-tail static).

  (c) image_prompt is a LITERAL-PHYSICAL FLUX prompt (English).
      ❌ ABSTRACT: "Arjuna feels betrayed by dharma"
      ✅ LITERAL:  "Arjuna warrior, tear-streaked face, white-knuckled grip
                    on bow, dim shadow-heavy frame, golden palette, no hands
                    in frame"
      FLUX renders nouns and adjectives. Verbs and abstractions render as
      generic. Build the prompt from: [character + identity descriptors] +
      [intensity adjective] + [environment] + [palette] + [composition].

  (d) SUBJECT LOCK — for each entry, the anchor_phrase's primary subject
      MUST appear in the image_prompt's English equivalent:
        anchor "द्रौपदी आंसू बहाती"      → image MUST contain "draupadi" (or queen/woman) + "tear"
        anchor "अश्वत्थामा तलवार उठाता" → image MUST contain "ashwatthama" (or warrior) + "sword"
        anchor "बच्चे सो रहे"             → image MUST contain "children" + "sleeping"
      No subject bleed (Arjuna image when narration is about Draupadi).

  (d.2) STORY-CRITICAL ENTITY (Phase 19, 2026-06-16) — when the voiceover
        names a non-human story element (animal companion / divine entity
        / named weapon / named object), AT LEAST ONE broll image_prompt
        MUST contain the English equivalent as a clearly identifiable
        foreground subject. Prefer placing it on the broll entry whose
        anchor_phrase names the entity directly.
          voiceover names "कुत्ता"  → ≥1 image_prompt contains "a loyal
                                     stray dog ... foreground subject"
          voiceover names "धनुष"   → ≥1 image_prompt contains "Gandiva
                                     bow / Arjuna's bow ... foreground"
          voiceover names "कुंडल"  → ≥1 image_prompt contains "golden
                                     kundal earrings ... close-up"
          voiceover names "बच्चे"  → ≥1 image_prompt contains "sleeping
                                     children ... foreground"
          voiceover names "सुदर्शन" → ≥1 image_prompt contains "sudarshana
                                     chakra / discus ... divine glow"
        The Dharma dog vanishing from the 2026-06-16 Heaven's Gate render
        — the entire emotional pivot of that story — is exactly the
        failure mode this rule prevents.

  (e) INTENSITY ADJECTIVE (Phase 18, FLUX-quality floor) — EVERY image_prompt
      MUST contain at least ONE aggressive physical adjective from this list:
      bloodied / blood-soaked / blood-streaked / rusted / shattered / cracked
      / broken / torn / ash-covered / scorched / wounded / scarred / smoldering
      / glowing / flickering / dim / harsh / fading / burning / trembling /
      clenched / tear-streaked / white-knuckled / sweat-drenched / wide-eyed /
      hollow-eyed / kohl-rimmed / wind-whipped / rain-soaked / dust-choked /
      shadow-heavy / claustrophobic / suffocating.
      FLUX defaults to STUDIO-CLEAN museum-quality output when given polite
      nouns. The intensity adjective forces it to render the scene's emotional
      damage.
      ❌ BAD:  "arjuna warrior with bow"
      ✅ GOOD: "arjuna warrior, tear-streaked face, white-knuckled grip on
                bow, blood-streaked armor, dim shadow-heavy frame"

  (f) CHARACTER NAME — every image_prompt MUST contain at least one named
      character in English (Arjuna / Karna / Bhishma / Draupadi /
      Yudhishthira / Bhima / Drona / Ashwatthama / Eklavya / Krishna / etc.)
      so the FLUX cascade's _inject_characters can append visual descriptors.

  (g) PALETTE — every image_prompt MUST embed the palette directive.
      {palette_line}

  (h) PHASE 17.b OPENER ARCHETYPE for THIS render (episode {episode_n} %% 4 = {episode_n % 4}):
      broll[0].image_prompt MUST BEGIN VERBATIM with this exact directive
      as its FIRST sentence — do not paraphrase, copy as-is:
      "{scene1_opener_directive}"
      Then continue with the rest of broll[0]'s literal-physical prompt
      (the subject from the voiceover's opening 8 words, intensity adjective,
      palette, etc.).

  (i) NO HANDS IN FRAME — FLUX-schnell mangles fingers. End each image_prompt
      with "no hands in frame, no text, no letters, no watermarks, no signage,
      no captions, no banners with writing, no overlay text" (the existing
      anti-text guard).

  (j) SUBJECT DIVERSITY (Phase 20, 2026-06-17) — broll is NOT 8 portraits
      of the same character. It is a FILM EDIT. Cycle these 4 shot types
      across your 8-10 entries. Two consecutive same-type entries are OK;
      three consecutive will be REJECTED. At least 2 of the 4 types must
      appear across your entries.

      (A) CHARACTER ACTION — hero mid-verb. Include action verb in
          image_prompt: "raising", "lifting", "drawing back", "falling",
          "sinking", "striking", "running". Example: "Arjuna drawing
          back the bowstring, fingers white-knuckled, mid-gesture, arrow
          tip glowing"

      (B) REACTION — tight close-up of EMOTION on face. Example:
          "Yudhishthira's tear-streaked face wide-eyed in horror,
          kohl-rimmed eyes, anguished expression, gold mukut tilted"

      (C) PROP / DETAIL — OBJECT in foreground, character absent or
          backgrounded. image_prompt MUST LEAD with the object, not
          the character name. Examples: "extreme close-up of a wooden
          chariot wheel sinking into wet mud, blood-streaked spokes",
          "macro shot of the Gandiva bow, gold-and-pearl inlay glowing,
          string drawn taut", "detail of golden kundal earrings, intricate
          filigree, lit by oil-lamp glow"

      (D) ENVIRONMENT / WIDE — vista, no facial close-up. Examples:
          "wide shot of burning Kurukshetra battlefield, smoke-filled
          sky, distant chariots silhouetted against the dusk", "aerial
          view of the Pandava forest hermitage, dappled sunlight through
          banyan canopy"

      broll[0] is already constrained by the Phase 17.b opener archetype
      directive in rule (h). Pick the type that matches that archetype:
      archetype A → ACTION (verb in image), B → PROP, C → ACTION,
      D → ENVIRONMENT.

  (k) AUDIO-VISUAL VERB LOCK (Phase 20, 2026-06-17) — the image_prompt
      MUST physically depict the EXACT verb in the anchor_phrase, not
      just the noun. If anchor is "रथ का पहिया डूबता" (chariot wheel
      sinking), image_prompt MUST contain "sinking" (or "drowning",
      "submerging") — NOT just "chariot wheel in background". Same for
      "बाण छोड़ा" (arrow released) → "arrow leaving the bowstring,
      mid-flight"; "तलवार उठाई" (sword raised) → "sword raised
      overhead, mid-strike arc". Subject_lock already enforces the noun
      appears; verb_action_lock now enforces the action is depicted.
      Render the VERB the audience HEARS, not a static portrait.

  (l) VERB-PER-FRAME (Phase 22, 2026-06-25) — ≥5 of your 8-10 broll
      image_prompts MUST start (within the first 80 chars) with an
      ACTION VERB from this list:
      drawing / drawing back / sinking / severing / kneeling beneath /
      looming over / leaning into / dragging away from / charging /
      falling / gripping / snarling / recoiling / fleeing / walking
      away from / hurled / slashing / weeping over / towering / kicking
      aside / raised in halt / palm raised / drawing arrow / drawing
      sword / striking / piercing / collapsing / lunging / advancing /
      marching / wielding.

      Forbidden noun-pose verbs (auto-reject): standing / looking /
      gazing / contemplating / praying / holding / wearing / sitting /
      watching / observing / thinking / posing / facing / smiling.

      FINAL broll[-1] entry MUST be an AFTERMATH beat. TWO requirements
      both must hold for broll[-1] (the LAST entry in the broll array):

        (1) You MUST set the "wardrobe_context" JSON field to "AFTERMATH"
            for this entry (NOT "WAR", NOT "PALACE" — the literal string
            "AFTERMATH" in the JSON). This is checked by the
            _check_aftermath_closer validator BEFORE the image_prompt
            token check fires.

        (2) AND the image_prompt for broll[-1] must contain at least one
            of: silhouette / abandoned / walking away / withheld /
            lone diya / prone body / fading dusk.

      Both gates must pass independently. Setting one without the other
      hard-fails the render. Without requirement (1), the LLM tends to
      write beautiful aftermath image_prompts but mislabel the
      wardrobe_context as "WAR", triggering an infinite quarantine loop.

      WHY: forensic on the channel's 500-655 view winners showed every
      frame was a VERB (someone doing something to someone), not a
      NOUN (a crowned warrior existing in a costume). Verb-density is
      the single most predictive feature of view performance. AFTERMATH
      wardrobe + image_prompt together produce the "consequence as
      closer" beat that ALL 4 winners used.

  (m) ANTI-MERGE COMPOSITION (Phase 22, 2026-06-25) — when image_prompt
      names ≥2 characters in the same frame, it MUST start with one of
      these tagged composition templates (literal string, in brackets):

      [OVER-SHOULDER] — secondary's shoulder/head (back, no face) bokeh
        in foreground, named character mid-ground in focus
      [POWER-LOOM] — named character standing tall, secondary kneeling
        below; vertical scale separation
      [CONFRONTATION-WIDE] — named character left-foreground side-profile,
        secondary right mid-ground torso-up; diagonal light/shadow boundary
      [WITNESS-FROM-BEHIND] — named character back-of-head foreground,
        secondary mid-ground facing camera
      [THE-FALLEN] — named character standing upper-third, secondary
        prone in dust below (face withheld)

      Then continue with the rest of the image_prompt (verb, intensity,
      wardrobe, palette).

      WHY: FLUX merges multi-character scenes (3 arms, fused faces,
      shared torso) when both subjects are at the same scale with full
      iconographic descriptors. Power-imbalance composition geometrically
      separates the subjects so FLUX can't merge them. The 4 channel
      winners ALL used this technique.

      For single-character scenes (1 named character + 0/N secondaries
      described purely structurally as "a kneeling brahmin disciple" /
      "a bound captive") AND macro-on-prop scenes (no characters): no
      composition tag required, but you may still use [POWER-LOOM] /
      [THE-FALLEN] if it fits.

  (n) ARCHITECTURAL PROMPT FORMAT (Phase 23.2, 2026-06-28) — every
      image_prompt MUST follow this exact 3-part structure:

      [CANVAS]  →  [ACTION]  →  [CHARACTER SIGNATURE]

      • CANVAS = wide environmental opener. Describes the SCENE / setting
        the camera is in. Examples:
          "Wide cinematic shot, sun-baked battlefield of Kurukshetra,
           broken chariot wheels half-buried in mud,"
          "Sweeping panoramic shot, ornate Nagara-style palace court
           with carved stone columns and marigold petals on marble floor,"
          "Extreme macro close-up of dice mid-roll on polished marble,"

      • ACTION = what happens IN that canvas. Verb-led. Examples:
          "...Karna draws back his bow with white-knuckled fingers..."
          "...Draupadi recoils, her hair untied, eyes blazing..."
          "...the dice clatter to a stop, three sixes face-up..."

      • CHARACTER SIGNATURE = the canonical Vedic descriptors from the
        signature library below. Use these EXACT phrases — they will
        land in FLUX cross-attention as iconographic anchors.

      DO NOT start image_prompts with a character's name. Start with
      the canvas. FLUX weights early tokens heaviest — if the first
      word is "Karna," FLUX commits all its rendering budget to drawing
      Karna and the environment becomes a bokeh wall.

      WRONG (Phase 22 X-LWlg1DW5s failure mode):
        "Karna draws back his bow at sunset, golden-bronze skin..."
      RIGHT (Phase 23.2 architectural format):
        "Wide cinematic shot, sun-baked Kurukshetra battlefield with
         broken chariot wheels and distant marching armies, Karna draws
         back the Gandiva bow with white-knuckled fingers, radiant
         golden-bronze Indian skin, luminous divine golden kavacha
         armor fused to bare chest, intricate gold kundal earrings,
         battle-stained dark red silk dhoti, NO leather pauldrons"

      MAHABHARATA WARDROBE DOCTRINE (Phase 23.4, 2026-06-28) — Vedic
      kshatriya warriors fight BARE-CHESTED with only yajnopavita
      thread + gold armbands + gold necklaces ON BARE SKIN. The
      "kavacha" for Karna/Duryodhana is a divine GLOW emanating from
      WITHIN the bare chest skin, NOT a Western plate-mail breastplate
      worn over a shirt. FLUX defaults to Roman/medieval plate armor
      when given the word "warrior" without an explicit "bare chest"
      anchor — so EVERY warrior image_prompt MUST include the literal
      phrase "bare chest visible" and append the negative tail
      "NO breastplate, NO chest plate, NO cuirass, NO plate armor,
      NO Western armor, NO leather pauldrons". Reference images from
      the channel owner: Karna with sun-medallion kavacha glowing
      FROM bare chest skin (no armor over it); Arjuna bare-chested
      with gold yajnopavita + ornate gold arm bands + ornate gold-
      edged quiver across bare back (NO chest plate); Bhima bare-
      chested with thick gold necklace on bare chest. PALACE / DIVINE
      contexts are the only exceptions (court silk robes / celestial
      diaphanous shawls cover the chest legitimately).

      CANONICAL CHARACTER SIGNATURES (use these EXACT phrases):

      Krishna: "vivid cobalt-blue divine Indian skin, BARE CHEST
        visible, gold yajnopavita thread across bare chest, layered
        gold-jewel chain necklaces on bare chest, dark wavy hair with
        twin peacock feathers NOT a full crown, yellow pitambara silk
        dhoti at waist only, holding bansuri flute, youthful build,
        NO breastplate"

      Karna: "radiant golden-bronze Indian skin, broad-shouldered
        warrior, BARE-CHESTED with luminous divine sun-kavacha glow
        emanating FROM WITHIN bare chest skin (kavacha embedded INTO
        chest skin, NOT armor worn over it, NOT a breastplate),
        intricate gold kundal earrings ALWAYS visible, battle-stained
        dark red silk dhoti at waist, NO chest armor"

      Arjuna: "wheatish golden-bronze Indian skin, dark wavy hair tied
        back, BARE-CHESTED warrior (NO breastplate, NO kavacha, NO
        plate, NO shirt on torso), sacred gold yajnopavita thread
        crossing bare chest diagonally, ornate gold armbands on bare
        upper biceps, ornate gold-edged leather quiver slung across
        bare back, drawing colossal Gandiva bow, cream silk dhoti at
        waist, NO chest armor"

      Bhishma: "weathered late-50s Indian face, long majestic silver-
        white beard flowing to mid-chest NO mustache, BARE-CHESTED
        elder warrior (NO breastplate, NO leather chest plate, NO
        plate, NO shirt), sacred gold yajnopavita thread crossing
        bare chest, simple gold armlets on bare upper arms, austere
        off-white silk dhoti at waist, NO chest armor"

      Draupadi: "luminous dark Indian skin, large expressive lotus
        eyes, long untied black hair blowing in the wind, flowing
        battle-stained crimson red silk sari, fierce proud posture"

      Eklavya: "dark mud-streaked tribal Indian skin, bare-chested
        with natural bark-cloth lower garment, severed right thumb
        ALWAYS visible (sometimes dripping blood), rudraksha bead
        necklace, intense devoted eyes"

      Ashwatthama: "gaunt Indian face, haunted feral eyes, glowing
        ruby-like gem embedded into the center of his forehead casting
        a faint red light down his face, prematurely curse-white hair,
        ash-smeared blood-streaked skin, tattered dark cloth dhoti"

      Bhima: "dark golden-bronze Indian skin, colossal heavily-muscled
        warrior, BARE-CHESTED (NO breastplate, NO kavacha, NO plate,
        NO shirt on torso), thick chunky gold necklace on bare massive
        chest, ornate gold armbands on bare bulging biceps, rugged
        leather belt over thick red silk dhoti at waist, gripping
        massive battle-dented iron Gada mace, fierce dark moustache"

      Yudhishthira: "warm golden-bronze Indian skin, salt-and-pepper
        short beard, deep sorrowful eyes, BARE-CHESTED (NO breastplate,
        NO kavacha, NO plate, NO shirt on torso), sacred gold
        yajnopavita thread crossing bare chest, soft red-gold
        angavastram drape over ONE shoulder only (leaving other side
        of chest bare), modest white-and-gold silk dhoti at waist,
        NO chest armor, NO crown by default"

      Duryodhana: "warm golden-bronze Indian skin, massive arrogant
        prince, muscular broad-chested build, rich purple and gold
        silk dhoti, intricate golden kavacha, ornate royal mukut crown,
        gripping a heavy golden Gada mace on shoulder"

      Shakuni: "sharp angular Indian features, grey-streaked beard,
        calculating gleaming eyes, slightly hunched posture, dark
        green and black silk dhoti, leaning heavily on a carved wooden
        cane, tossing two bone dice with a sly wicked smile"

      Drona / Dronacharya: "weathered elderly Indian face, white hair
        tied in a sage's topknot, long white beard, simple white cotton
        dhoti, white angavastram, sacred yajnopavita thread across
        bare chest, pointing an intricate wooden bow"

      Kunti: "weary but graceful older Indian woman, pale golden Indian
        skin, sorrowful dignified eyes, silver-streaked dark hair,
        simple unadorned white silk sari, holding a simple earthen pot"

  (o) GROUP-SCENE MANDATE (Phase 23.1) — AT LEAST 3 of your 8-10
      image_prompts must include 2+ named characters in the SAME canvas.
      Single-character image_prompts cap at 7 of 10. This is the
      narrative-density rule. The 500+ view winners ALL had multi-
      character layered compositions (Bhima + Draupadi + Shakuni at
      the dice game; Eklavya kneeling + Arjuna watching from behind
      a tree). When you write a 2-character canvas, lead with a
      composition tag from rule (m):
        "[OVER-SHOULDER] Sweeping shot of ornate palace court,
         Eklavya kneels before his clay idol of Drona while Arjuna
         watches from behind a banyan tree..."

  (p) NON-HUMAN FOCUS RULE (Phase 23.2, 2026-06-28) — if the story beat
      involves an animal (dog, horse, snake, eagle) OR a named object
      (severed thumb, dice, chariot wheel, kavacha, Gandiva bow,
      sudarshana chakra, akshaya patra, lone diya), the image_prompt
      MUST NOT start with a character's name. It MUST lead with the
      object/animal as the foreground subject. FLUX weights early
      tokens heaviest — if a character name comes first, FLUX spends
      100% of its rendering budget on the character and IGNORES the
      object entirely.

      WRONG: "Eklavya looking angry after shooting the dog"
        → FLUX renders an angry warrior, the dog is absent.

      RIGHT: "Extreme macro close-up of a stray dog's mouth, seven
              finely crafted arrows piercing the air around its muzzle
              without injury, drops of blood, dense forest backdrop
              with dappled sunlight, Eklavya's tribal silhouette
              mid-ground in soft focus"

      RIGHT: "Close-up shot of an iron Gandiva bowstring drawn taut,
              knuckles white in foreground, distant battlefield with
              marching armies behind"

      RIGHT: "Macro shot of two bone dice mid-roll on polished marble
              palace floor, lone diya flickering nearby, Shakuni's
              calculating hand in shallow focus"

      Use this object-led pattern for AT LEAST 1 of your 8-10
      image_prompts whenever the voiceover names a non-human story
      element (dog / weapon / object / divine entity).

{cliffhanger_block}

═══════════════════════════════════════════════════════════════
OUTPUT — return ONLY valid JSON, no markdown fences, no preamble:
═══════════════════════════════════════════════════════════════
{{
  "voiceover": "<single flowing Hindi paragraph, 80-110 words, ends with '?'>",
  "broll": [
    {{
      "image_prompt": "<literal-physical English prompt with character + intensity + palette + 'no hands in frame, no text...'>",
      "anchor_phrase": "<2-4 Devanagari words verbatim from voiceover>",
      "mood": "<3-6 word English mood>",
      "wardrobe_context": "<WAR | PALACE | DIVINE | FOREST | JOURNEY | AFTERMATH (AFTERMATH only on broll[-1])>"
    }}
  ],
  "title": "<Bilingual <60 chars, format: '[Hindi half] | [English half]'. Hindi FIRST. Each half 24-28 chars max. PHASE 22 HARD GATE (2026-06-25): TITLE DNA = [character] + [subversion adjective from whitelist: असली/छुपा/सबसे बड़ा/आखिरी/घातक or real/hidden/biggest/final/fatal] + [tribal-relational noun from whitelist: गलती/धोखा/श्राप/प्रतिज्ञा/अभिमान/बलिदान/वचन/हार/डर/शर्म or mistake/betrayal/curse/promise/pride/sacrifice/vow/loss/fear/shame]. MUST assert the OPPOSITE of the hero's canonical virtue (Karna's loyalty → 'Karna\\'s Biggest Mistake' = SHARP; Arjuna's skill → 'Arjuna\\'s Real Fear' = SHARP; Bhishma's vow → 'Bhishma\\'s Hidden Betrayal' = SHARP). PASS examples: 'कर्ण की सबसे बड़ी गलती | Karna\\'s Biggest Mistake' / 'अर्जुन का असली डर | Arjuna\\'s Real Fear' / 'भीष्म का छुपा वचन | Bhishma\\'s Hidden Vow'. REJECT (hard-fail): mood/regret/wound/sin/truth/fate/ego/'war within'/'काला सच'/'inner X'; identity nouns caste/religion/Hindu/जाति/धर्म; soft adjectives silent/quiet/inner/deep/spiritual/चुप/मौन/गहरा; pure incident-naming ('कर्ण की प्रतिज्ञा'); 'Story of X'; episode numbering. The title is the OPENING PROMISE the thumbnail pays off — make it sharp and binary, not pensive.>",
  "hook_title": "<1-5 words, 100% Devanagari script (NO Latin). MUST contain ≥1 named character (भीष्म/अर्जुन/कर्ण/कृष्ण/द्रौपदी/...) OR ≥1 paradox marker (लेकिन/पर/फिर भी/कभी नहीं/आखिरी/पहली/एकमात्र/जो). MUST NOT contain '?', '!', '...', emoji, or setup openers ('यह'/'ये'/'एक कहानी'/'बहुत समय'/'कहते हैं'). Examples that PASS: 'अर्जुन का अंतिम पाप', 'कर्ण की एक गलती', 'द्रौपदी का अंतिम सच'.>",
  "description": "<Hook line under 90 chars expanding the title.\\n\\n#Shorts #Mahabharata #महाभारत #Krishna #HinduMythology\\n\\n100-150 words about the story.\\n\\n#Shorts #Mahabharata #महाभारत #Hindu #BhagavadGita #भगवद_गीता #Krishna #कृष्ण #Arjuna #अर्जुन #Karna #कर्ण #Bhishma #भीष्म #Draupadi #द्रौपदी #Kurukshetra #कुरुक्षेत्र #AncientIndia #IndianMythology #Dharma #MythologyShorts #VedicWisdom #HinduDharma #IndianHistory #SpiritualShorts #PauranikKathayein #SanatanDharma #HindiShorts #trending>",
  "tags": ["topic-specific tag 1","topic-specific tag 2","named char 1","named char 1 Devanagari","Mahabharata","महाभारत","Shorts","Hindu mythology"],
  "thumbnail_prompt": "<Dramatic Mahabharata thumbnail — vibrant colours, cinematic, portrait composition>",
  "quotable_line": "<≤14 words Hindi, tribal-split moral claim. MUST contain at least one charged word (बलिदान/ज़िद/धोखा/मजबूरी/गलती/पाप/अपमान/झूठ/सच/विश्वासघात/घमंड/खत्म).>",
  "pinned_question": "<❓ {{quotable_line}} + one-line invitation to take a side in Hindi.>",
  "next_seed": "<≤12 words Hindi, named-future-consequence hook (e.g. 'और यही प्रतिज्ञा एक दिन हस्तिनापुर को तोड़ देगी').>"
}}
""".strip()


# ─── Main entry point ─────────────────────────────────────────────────────

def generate_phase18_script(
    language: str,
    forced_topic: str | None,
    series: str,
) -> dict:
    """Phase 18 self-contained script generation. Returns a dict with:
      voiceover (str), broll (list), title, hook_title, description, tags,
      thumbnail_prompt, quotable_line, pinned_question, next_seed,
      episode_n, arc_character_devanagari, arc_character_english, language,
      series, topic, content_type.

    NO `scenes` key — that's the legacy Phase 17.b shape."""
    # Lazy-import everything from script_generator to avoid circular import
    from pipeline.script_generator import (
        _pick_next_arc_topic,
        _load_arcs,
        _load_used_topics,
        _stable_episode_n,
        _build_cliffhanger_block,
        _call_llm,
        STORY_TOPICS,
    )
    from pipeline.image_generator import _character_palette_directive

    # ── Topic selection (mirrors legacy generate_script lines 2664-2740) ──
    episode_n = 0
    arc_name = ""
    arc_character_devanagari = ""
    emotional_fingerprint = ""
    next_episode_teaser: str | None = None
    content_type = "story"

    if forced_topic:
        topic = forced_topic
        arcs = _load_arcs()
        used = _load_used_topics()
        for arc_idx, arc in enumerate(arcs):
            topics_list = arc.get("topics", [])
            if forced_topic in topics_list:
                arc_name = arc.get("name", "")
                idx = topics_list.index(forced_topic)
                episode_n = _stable_episode_n(arc_idx, idx)
                emotional_fingerprint = arc.get("emotional_fingerprint", "")
                arc_character_devanagari = arc.get("character", "")
                for ahead in topics_list[idx + 1:]:
                    if ahead not in used:
                        next_episode_teaser = ahead
                        break
                print(f"    [arc-forced] matched {arc_name} — episode {episode_n}")
                break
    else:
        arc_pick = _pick_next_arc_topic()
        if arc_pick:
            topic, episode_n, arc_name, next_episode_teaser = arc_pick
            for _arc in _load_arcs():
                if _arc.get("name") == arc_name:
                    emotional_fingerprint = _arc.get("emotional_fingerprint", "")
                    arc_character_devanagari = _arc.get("character", "")
                    break
            print(f"    [phase18-arc] {arc_name} — episode {episode_n}: {topic[:80]}")
        else:
            import random
            topic = random.choice(STORY_TOPICS)
            print(f"    [phase18-fallback] random topic (all arcs exhausted)")

    palette_directive = _character_palette_directive(arc_character_devanagari)
    if arc_character_devanagari:
        print(f"    [phase18-palette] {arc_character_devanagari} -> {palette_directive[:60]}...")

    # ── Phase 17.b opener archetype rotation ──
    archetype_idx = episode_n % len(_OPENER_ARCHETYPES_PHASE18)
    archetype_letter, archetype_text = _OPENER_ARCHETYPES_PHASE18[archetype_idx]
    print(f"    [phase18-opener] archetype {archetype_letter} "
          f"(episode_n={episode_n} mod {len(_OPENER_ARCHETYPES_PHASE18)})")

    # ── Cliffhanger pattern selection ──
    cliffhanger_block = ""
    if next_episode_teaser and episode_n:
        ep_mod_10 = episode_n % 10
        pattern_letter = "ABC"[episode_n % 3] if ep_mod_10 < 7 else "D"
        cliffhanger_block = _build_cliffhanger_block(
            pattern_letter=pattern_letter,
            next_teaser=next_episode_teaser,
        )

    # ── Fingerprint block ──
    fingerprint_block = ""
    if emotional_fingerprint and arc_name:
        fingerprint_block = (
            "═══════════════════════════════════════════════════════════════\n"
            "CHARACTER EMOTIONAL FINGERPRINT — the script's emotional DNA\n"
            "═══════════════════════════════════════════════════════════════\n"
            f"This video belongs to the {arc_name} arc.\n"
            f"Character emotional fingerprint: {emotional_fingerprint}.\n"
            "The voiceover's emotional center MUST land on this fingerprint."
        )

    base_prompt = _build_phase18_prompt(
        topic=topic,
        episode_n=episode_n,
        arc_name=arc_name,
        arc_character_devanagari=arc_character_devanagari,
        emotional_fingerprint=emotional_fingerprint,
        palette_directive=palette_directive,
        archetype_letter=archetype_letter,
        archetype_text=archetype_text,
        cliffhanger_block=cliffhanger_block,
        fingerprint_block=fingerprint_block,
    )

    # ── Best-of-N loop ──
    # Phase 22.7 (2026-06-28):
    #   • MAX_ATTEMPTS 5 → 8 (Improvement A). Gemini Flash variance is the
    #     real blocker — 7 verification runs showed best-of-N=5 producing
    #     fatal-clean 19+/25 in only 2/7 renders (~28% ship rate). At 8
    #     attempts, expected ship rate climbs above ~70% even under the
    #     same per-attempt distribution.
    #   • Selection prefers fatal-clean over higher-but-fatal-failed
    #     (Improvement B). Old logic picked max(score) regardless. So a
    #     20/25 with `anti_merge_composition` fatal-failed beat an 18/25
    #     fatal-clean — but the 20/25 will NEVER ship (fatal-blocked),
    #     while the 18/25 was one cosmetic away from publish. Picking
    #     fatal-clean attempts when they exist is the correct selection
    #     under the Phase 22.4 two-gate strict-pass.
    _PHASE22_FATAL_GATES_FOR_SELECT = {
        "verb_per_frame",
        "anti_merge_composition",
        "aftermath_closer",
        "title_dna",
    }
    MAX_ATTEMPTS = 8
    attempts: list = []   # each: {data, score, is_fatal_clean, violation, info, fatal_failures}
    last_violation = ""
    last_info: dict = {}

    for attempt in range(MAX_ATTEMPTS):
        full_prompt = base_prompt
        if attempt > 0 and last_violation:
            base_reminder = _VIOLATION_REMINDERS.get(
                last_violation,
                f"Your previous response failed gate: {last_violation}. Fix this specific gate before any other."
            )
            # Dynamic length reminder — quote the actual count back to the LLM.
            # Gemini chronically under-emits when given an abstract "80-110"
            # range; quoting "you wrote 32 words" + "we need 75-115" anchors
            # the next attempt around the right magnitude.
            dynamic_prefix = ""
            if last_violation == "length":
                w = last_info.get("word_count", 0)
                b = last_info.get("broll_count", 0)
                # Phase 21 (2026-06-25): symmetric overshoot branch. The
                # original undershoot-only framing ("THREE times what you
                # produced") was triggering overcorrection — the post-Phase-20
                # forensic showed LLM swinging from 31-38w (undershoot) to
                # 119-153w (overshoot) because the retry prompt only ever
                # said "write MORE" with no anti-overshoot guard.
                if w < 75:
                    dynamic_prefix = (
                        f"YOUR LAST VOICEOVER WAS ONLY {w} WORDS (broll={b}). "
                        f"You MUST write 75-170 words — aim for ~120. "
                        f"At Charon's 2.2 wps that's ~55s narration — anything "
                        f"shorter than 75 words = <34s = a quarter-finished Short. "
                    )
                elif w > 170:
                    # Phase 22.3 (2026-06-26): cap raised 130→170 tactically
                    # to ship a Phase-22-compliant render to YouTube.
                    overshoot = w - 130
                    dynamic_prefix = (
                        f"YOUR LAST VOICEOVER WAS {w} WORDS — {overshoot} OVER "
                        f"the 130-word sweet-spot target (cap is 170, broll={b}). "
                        f"You MUST trim to 75-170 words. Cut the WEAKEST "
                        f"sensory beat. Remove one adjective per remaining "
                        f"sentence. DO NOT add new ideas. AIM FOR 120. "
                    )
                else:
                    # Length is in-range but broll count failed (8-10 entries)
                    dynamic_prefix = (
                        f"Length OK ({w}w) but broll count={b} is out of "
                        f"range — you MUST produce 8-10 broll entries. "
                    )
            elif last_violation == "anchors":
                why = last_info.get("anchor_why", "")
                dynamic_prefix = f"Anchor validation failed: {why[:200]}. "
            elif last_violation == "wardrobe_set":
                why = last_info.get("wardrobe_why", "")
                dynamic_prefix = f"Wardrobe-context validation failed: {why[:200]}. "
            elif last_violation == "story_entity":
                why = last_info.get("story_why", "")
                dynamic_prefix = f"Story-entity validation failed: {why[:200]}. "
            elif last_violation == "subject_diversity":
                why = last_info.get("diversity_why", "")
                types = last_info.get("shot_types", [])
                dynamic_prefix = f"Subject-diversity validation failed: {why[:240]}. (Your shot_types were: {types}.) "
            elif last_violation == "verb_action":
                types = last_info.get("shot_types", [])
                dynamic_prefix = f"Verb-action lock failed — at least one broll image_prompt described a STATIC pose where the anchor's Devanagari verb required the depicted action. (Your shot_types: {types}.) "
            elif last_violation == "verb_per_frame":
                why = last_info.get("verb_per_frame_why", "")
                dynamic_prefix = f"Verb-per-frame failed: {why[:280]}. "
            elif last_violation == "anti_merge_composition":
                why = last_info.get("anti_merge_why", "")
                dynamic_prefix = f"Anti-merge composition failed: {why[:280]}. "
            elif last_violation == "aftermath_closer":
                why = last_info.get("aftermath_why", "")
                dynamic_prefix = f"Aftermath closer failed: {why[:280]}. "
            elif last_violation == "title_dna":
                why = last_info.get("title_dna_why", "")
                dna_score = last_info.get("title_dna_score", 0)
                dynamic_prefix = f"Title DNA gate failed (score={dna_score}/3): {why[:280]}. "
            full_prompt = base_prompt + (
                f"\n\n── RETRY REMINDER (attempt {attempt+1}/{MAX_ATTEMPTS}) ──\n"
                f"{dynamic_prefix}{base_reminder}"
            )

        print(f"    [phase18-gen] attempt {attempt + 1}/{MAX_ATTEMPTS}")
        try:
            raw = _call_llm(full_prompt, quality="best")
        except Exception as e:
            print(f"    [phase18-gen] LLM call failed: {str(e)[:120]}")
            continue

        # Strip markdown fences if the LLM emits them despite the rule
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)

        # Defensive: strip ASCII control characters that occasionally appear
        # in long Hindi LLM output (the 2026-06-15 retry-1 attempt 2 failed
        # with "Invalid control character at line 47 column 128"). \n \r \t
        # are legal in JSON whitespace position; we only nuke C0 control
        # chars OUTSIDE of strings — but a blanket strip of bytes < 0x20
        # except \n \r \t is the simpler safe move here.
        text = "".join(c for c in text if c >= " " or c in "\n\r\t")

        try:
            data = json.loads(text)
        except Exception as e:
            print(f"    [phase18-gen] JSON parse failed: {str(e)[:120]}")
            continue

        ok, violation, info = validate_phase18(data)
        score = info.get("score", 0)
        vo_chars = len(data.get("voiceover", ""))
        vo_words = info.get("word_count", 0)
        br_n = info.get("broll_count", 0)
        types = info.get("shot_types", [])

        # Phase 22.7 (2026-06-28): compute fatal-clean per attempt so the
        # post-loop selection can prefer fatal-clean attempts over
        # higher-but-fatal-failed ones. Cheap (~25 small predicate calls).
        failing_gates  = _enumerate_failing_gates(data)
        fatal_failures = failing_gates & _PHASE22_FATAL_GATES_FOR_SELECT
        is_fatal_clean = (len(fatal_failures) == 0)

        print(f"    [phase18-validate] score={score}/25 ok={ok} "
              f"fatal_clean={is_fatal_clean} "
              f"voiceover={vo_words}w/{vo_chars}c broll={br_n} "
              f"types={types} "
              f"violation={violation or 'none'}")

        attempts.append({
            "data":            data,
            "score":           score,
            "is_fatal_clean":  is_fatal_clean,
            "fatal_failures":  fatal_failures,
            "violation":       violation,
            "info":            info,
        })

        # Strict 25/25 strict-pass — no need to keep trying.
        if ok:
            break

        last_violation = violation
        last_info = info

    if not attempts:
        raise RuntimeError(
            f"Phase 18 script generation failed all {MAX_ATTEMPTS} attempts "
            "(no valid JSON across the entire best-of-N). "
            "Check the [phase18-gen] log for LLM/parse errors."
        )

    # Phase 22.7 (2026-06-28): fatal-clean-preferring selection.
    # If ANY attempt was fatal-clean, pick the highest-scoring among
    # those (publishable under the Phase 22.4 two-gate strict-pass).
    # Otherwise fall back to the highest score overall — that attempt
    # won't ship either way (fatal-blocked), but it gives the cleanest
    # forensic signal in the quarantine dump.
    fatal_clean_attempts = [a for a in attempts if a["is_fatal_clean"]]
    if fatal_clean_attempts:
        chosen = max(fatal_clean_attempts, key=lambda a: a["score"])
        print(f"    [phase18-select] FATAL-CLEAN candidate selected: "
              f"score={chosen['score']}/25 from {len(fatal_clean_attempts)}/"
              f"{len(attempts)} fatal-clean attempts.")
    else:
        chosen = max(attempts, key=lambda a: a["score"])
        print(f"    [phase18-select] no fatal-clean attempts in {len(attempts)} "
              f"tries — falling back to highest-score for forensic review: "
              f"score={chosen['score']}/25 fatal_failures={sorted(chosen['fatal_failures'])}.")

    best_data      = chosen["data"]
    best_score     = chosen["score"]
    last_violation = chosen["violation"]
    last_info      = chosen["info"]

    # Phase 22 (2026-06-25): cascade bumped /21 → /25.
    # Phase 22.4 (2026-06-28): TWO-GATE strict-pass.
    #   • SCORE FLOOR (22/25) — up to 3 cosmetic / structural misses
    #     among the 21 OLD Phase 18-20 gates (length / anchors / archetype
    #     / monotony / etc.) are forgiven. Gemini Flash misses these
    #     intermittently; previous "25/25 absolute" blocked 4 verification
    #     renders for 4 days with 0 publishes.
    #   • FATAL GATES — the 4 NEW Phase 22 visual gates that DEFINE the
    #     channel's visual thesis (verb-led frames, anti-merge composition,
    #     aftermath closer, title-DNA inversion). ANY failure here =
    #     quarantine, regardless of total score. A script scoring 24/25
    #     with verb_per_frame=False would otherwise ship as the templated-
    #     portrait slideshow Phase 22 was built to ELIMINATE.
    #   • COMBINED: strict_pass = (score ≥ 22) AND (all fatal gates clean)
    _PHASE22_SCORE_MAX   = 25
    # Phase 22.5 (2026-06-28): floor 22 → 20.
    # Phase 22.6 (2026-06-28): floor 20 → 19. Run 28317529922 (Bhishma's
    # Hidden Sin topic) produced a FATAL-CLEAN script scoring 19/25 —
    # all 4 Phase 22 visual gates passed, just 6 cosmetic-gate misses
    # (archetype, char_names, engagement, intensity, subject_diversity,
    # tha_tic). Floor 20 quarantined it by 1 point. Gemini's actual
    # output ceiling across 6 verification runs is 19-20/25; floor 19
    # is the smallest calibration that lets shipping-quality scripts
    # actually publish while the 4 fatal Phase 22 gates remain absolute.
    # First Phase-22.4 verification render (28317323989, Draupadi
    # humiliation) scored 11-17/25 across 5 attempts — well below floor.
    # The two-gate guard correctly caught the fatal Phase 22 visual
    # failures (anti_merge +
    # verb_per_frame) but NO attempts cleared the score floor either.
    # Historical best across ALL verification renders is 20/25 (Eklavya +
    # Abhimanyu, both fatal-clean for Eklavya). Floor 20 = "allow up to
    # 5 cosmetic misses if Phase 22 visual gates are CLEAN". This trades
    # some surface polish (length, archetype, monotony, tha_tic) for a
    # shippable channel, while the 4 fatal gates still ABSOLUTELY enforce
    # the Phase 22 visual thesis.
    _PHASE22_SCORE_FLOOR = 19
    _PHASE22_FATAL_GATES = {
        "verb_per_frame",
        "anti_merge_composition",
        "aftermath_closer",
        "title_dna",
    }

    _all_failing_gates = _enumerate_failing_gates(best_data)
    _fatal_failures   = _all_failing_gates & _PHASE22_FATAL_GATES
    _score_floor_ok   = (best_score >= _PHASE22_SCORE_FLOOR)
    _fatal_clean      = (len(_fatal_failures) == 0)
    _strict_pass      = _score_floor_ok and _fatal_clean

    if not _strict_pass:
        if _fatal_failures:
            print(f"    [phase18-quarantine] FATAL Phase 22 visual gates "
                  f"failed: {sorted(_fatal_failures)} — non-negotiable. "
                  f"Score {best_score}/{_PHASE22_SCORE_MAX}.")
        else:
            print(f"    [phase18-quarantine] score {best_score}/{_PHASE22_SCORE_MAX} "
                  f"below floor {_PHASE22_SCORE_FLOOR} — too many cosmetic-gate "
                  f"misses. Failing: {sorted(_all_failing_gates)}.")
    elif best_score < _PHASE22_SCORE_MAX:
        print(f"    [phase18-rescue] SHIPPING at score={best_score}/"
              f"{_PHASE22_SCORE_MAX} (≥{_PHASE22_SCORE_FLOOR} floor, "
              f"Phase 22 fatal gates CLEAN). "
              f"Acceptable cosmetic misses: {sorted(_all_failing_gates)}.")

    # ── Bake metadata + return ──
    data = best_data
    data["language"] = language
    data["content_type"] = content_type
    data["topic"] = topic
    data["series"] = series
    data["episode_n"] = episode_n
    data["arc_character_devanagari"] = arc_character_devanagari
    data["arc_character_english"] = arc_name or ""

    data["_phase18_strict_pass"]    = _strict_pass
    data["_phase18_score"]          = best_score
    data["_phase18_score_max"]      = _PHASE22_SCORE_MAX
    data["_phase18_score_floor"]    = _PHASE22_SCORE_FLOOR
    data["_phase18_violation"]      = last_violation if not _strict_pass else ""
    data["_phase18_failing_gates"]  = sorted(_all_failing_gates)
    data["_phase18_fatal_failures"] = sorted(_fatal_failures)

    # Phase 23 (2026-06-28): stamp shot_type onto each broll entry. The
    # downstream image_generator routes width/height per shot_type
    # (ENVIRONMENT/ACTION/PROP → 1344x768 wide; REACTION/AMBIGUOUS →
    # 768x1344 vertical). Computing once here vs re-classifying at render
    # time keeps the classifier output deterministic across the
    # script-gen → image-gen → assembly pipeline.
    for _br in data.get("broll", []):
        _br["shot_type"] = _classify_broll_shot_type(_br.get("image_prompt", ""))

    # Cliffhanger prepend to description (mirrors legacy line 4687)
    if next_episode_teaser:
        teaser_title = next_episode_teaser.split("—")[0].strip()[:60] or next_episode_teaser[:60]
        prefix = f"▶️ अगला भाग: {teaser_title}\n\n"
        existing_desc = data.get("description", "")
        if not existing_desc.startswith("▶️ अगला भाग:"):
            data["description"] = prefix + existing_desc

    return data
