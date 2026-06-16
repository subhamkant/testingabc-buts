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

    # Length range tuned 2026-06-16 after first GHA run: original 80-110
    # bound saw 4 of 5 attempts come in at 31-38 words (Gemini under-emits
    # without a concrete length anchor). Floor lowered to 75 because the
    # LLM clusters around 70 when asked for "80" — and 75 still gives a
    # ~34s narration which is the bottom of the Shorts attention window.
    length_ok = 75 <= n_words <= 115
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

    flags = [length_ok, broll_ok, hook_ok, hook_title_ok, loop_ok,
             shock_ok, rehook_ok, eng_ok, mono_ok, tha_ok, rep_ok,
             anchors_ok, names_ok, archetype_ok, subject_lock_ok,
             intensity_ok, bookend_ok]
    score = sum(1 for f in flags if f)

    info = {
        "score":       score,
        "word_count":  n_words,
        "broll_count": n_broll,
        "anchor_why":  _anchor_why,
    }

    # First violation in priority order (single highest-impact gate)
    cascade = [
        ("length",       length_ok and broll_ok),
        ("anchors",      anchors_ok),
        ("hook",         hook_ok),
        ("shock_action", shock_ok),
        ("loop_closure", loop_ok),
        ("hook_title",   hook_title_ok),
        ("archetype",    archetype_ok),
        ("subject_lock", subject_lock_ok),
        ("intensity",    intensity_ok),
        ("rehook",       rehook_ok),
        ("char_names",   names_ok),
        ("bookend",      bookend_ok),
        ("engagement",   eng_ok),
        ("monotony",     mono_ok),
        ("tha_tic",      tha_ok),
        ("repetition",   rep_ok),
    ]
    for label, ok in cascade:
        if not ok:
            return False, label, info
    return True, "", info


# ─── Prompt builder ───────────────────────────────────────────────────────

_VIOLATION_REMINDERS = {
    "length":       "Your voiceover word count was out of range. Voiceover MUST be 75-115 Hindi words (AIM FOR ~95). Match the SHAPE of the concrete 96-word example in the prompt — open with an 8-word action verb, build 3-4 mid-paragraph beats with sensory anchors, embed a dialogue beat in quotes, insert लेकिन/परंतु around the midpoint, close with a question that echoes the opener noun. broll MUST be 8-10 entries.",
    "anchors":      "Your broll anchor_phrase entries failed validation. Each anchor_phrase MUST be 2-4 words appearing VERBATIM in the voiceover, in ASCENDING order by character position. Final anchor MUST land in the LAST 30% of the voiceover.",
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

OUTPUT 1 — `voiceover`: ONE flowing Hindi paragraph, 75-115 words (AIM FOR ~95).

  *** WORD COUNT IS THE #1 HARD RULE. ***
  *** SCRIPTS UNDER 75 WORDS WILL BE REJECTED. ***
  *** AIM FOR ~95 WORDS. AT CHARON 2.2 wps THAT'S ~43s SPOKEN — THE BOTTOM
      OF THE YOUTUBE SHORTS ATTENTION WINDOW. ***

  This is the master VO. Charon TTS reads it as ONE continuous emotional
  monologue at ~2.2 words/sec → 34-52 seconds spoken.

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
  - 75-115 words. NOT 30. NOT 50. NOT 70. SEVENTY-FIVE TO ONE-HUNDRED-FIFTEEN.
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
    "image_prompt":    "<English. Literal-physical FLUX prompt — see rules below>",
    "anchor_phrase":   "<2-4 Devanagari words VERBATIM from voiceover>",
    "mood":            "<3-6 word English emotional tone for SFX selection>"
  }}

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
      "mood": "<3-6 word English mood>"
    }}
  ],
  "title": "<Bilingual <60 chars, format: '[Hindi half] | [English half]'. Hindi FIRST. Each half 24-28 chars max. MUST do ONE of: challenge a known assumption / point at a hidden cause / pose a painful question / invert a hero's moral. FORBIDDEN: pure incident-naming ('कर्ण की प्रतिज्ञा'), Story of X, episode numbering.>",
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
    MAX_ATTEMPTS = 5
    best_score = -1
    best_data: dict | None = None
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
                dynamic_prefix = (
                    f"YOUR LAST VOICEOVER WAS ONLY {w} WORDS (broll={b}). "
                    f"You MUST write 75-115 words — aim for ~95. "
                    f"That is THREE times what you just produced. "
                    f"At Charon's 2.2 wps that's ~43s narration — anything "
                    f"shorter than 75 words = <34s = a quarter-finished Short. "
                )
            elif last_violation == "anchors":
                why = last_info.get("anchor_why", "")
                dynamic_prefix = f"Anchor validation failed: {why[:200]}. "
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
        print(f"    [phase18-validate] score={score}/17 ok={ok} "
              f"voiceover={vo_words}w/{vo_chars}c broll={br_n} "
              f"violation={violation or 'none'}")

        if ok:
            best_data = data
            best_score = score
            break

        if score > best_score:
            best_score = score
            best_data = data
        last_violation = violation
        last_info = info

    if best_data is None:
        raise RuntimeError(
            f"Phase 18 script generation failed all {MAX_ATTEMPTS} attempts "
            "(no valid JSON across the entire best-of-N). "
            "Check the [phase18-gen] log for LLM/parse errors."
        )

    if best_score < 17:
        print(f"    [phase18-rescue] shipping best-of-N (score={best_score}/17, "
              f"last violation={last_violation})")

    # ── Bake metadata + return ──
    data = best_data
    data["language"] = language
    data["content_type"] = content_type
    data["topic"] = topic
    data["series"] = series
    data["episode_n"] = episode_n
    data["arc_character_devanagari"] = arc_character_devanagari
    data["arc_character_english"] = arc_name or ""

    # Cliffhanger prepend to description (mirrors legacy line 4687)
    if next_episode_teaser:
        teaser_title = next_episode_teaser.split("—")[0].strip()[:60] or next_episode_teaser[:60]
        prefix = f"▶️ अगला भाग: {teaser_title}\n\n"
        existing_desc = data.get("description", "")
        if not existing_desc.startswith("▶️ अगला भाग:"):
            data["description"] = prefix + existing_desc

    return data
