from google import genai
import json
import random
import os


def _call_llm(prompt: str) -> str:
    """
    Calls Groq (primary, 14 400 RPD free) then falls back to Gemini 2.5 Flash.
    Returns the raw text response.
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
                max_tokens=4096,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"    Groq failed: {e} — falling back to Gemini...")

    # ── Fallback: Gemini 2.5 Flash ───────────────────────────────────────────
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()

# ── Story Topics — well-known Mahabharata incidents ───────────────────────────
STORY_TOPICS = [
    "The Kurukshetra war begins — conches blow and armies clash",
    "Krishna reveals the Bhagavad Gita to the trembling Arjuna",
    "Draupadi's vastraharan — her divine cry saved by Krishna",
    "Karna's tragic birth, abandonment and lifelong struggle for respect",
    "The dice game — Shakuni cheats and the Pandavas lose everything",
    "Abhimanyu enters the Chakravyuha alone and fights to his last breath",
    "Bhishma falls on a bed of arrows at Kurukshetra",
    "Krishna shows Arjuna his Vishwaroop — the terrifying cosmic form",
    "Draupadi's swayamvara — only Arjuna pierces the rotating fish",
    "Ekalavya cuts off his thumb as Gurudakshina for Drona",
    "Duryodhana humiliates Draupadi in the Kuru royal court",
    "The lac palace trap — Duryodhana tries to burn the Pandavas alive",
    "Karna donates his armour to Indra and faces certain death",
    "The death of Jayadratha — Arjuna's vow and Krishna's miracle",
    "Bhima kills Duryodhana — the war of Kurukshetra ends",
    "Ashwatthama's revenge — the massacre of sleeping warriors",
    "Gandhari's curse destroys Krishna's entire Yadava clan",
    "The Pandavas' final journey — one by one they fall in the Himalayas",
    "Karnas's reunion with Kunti — the secret she kept for decades",
    "Drona teaches archery — Arjuna sees only the eye of the bird",
]

# ── Motivational Themes ───────────────────────────────────────────────────────
MOTIVATIONAL_THEMES = [
    "Karma Yoga — act without attachment, Krishna's greatest lesson",
    "Why Arjuna almost quit before the greatest battle of his life",
    "Karna's dignity — the man who gave everything and asked nothing",
    "Bhishma's sacrifice — how one oath destroyed a dynasty",
    "The real lesson of the Bhagavad Gita most people never learn",
    "Why Krishna did not fight at Kurukshetra but still won the war",
    "Draupadi's resilience — how she never broke despite everything",
    "Yudhishthira's truth — the one man who never lied in the Mahabharata",
    "The Pandavas' 13 years of exile — patience as the greatest weapon",
    "What Vidura told Dhritarashtra that could have prevented the war",
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
    "भीष्म", "अर्जुन", "कृष्ण", "द्रोण", "द्रोणाचार्य", "कर्ण", "युधिष्ठिर",
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
    "krishna's", "arjuna's", "bhishma's",
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


def _parse_llm_json(raw: str) -> dict:
    """
    Robust JSON parser for LLM output.
    Handles: unescaped newlines inside strings, trailing commas, stray control chars.
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
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse LLM JSON after cleaning: {e}\n{cleaned[:400]}")


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
    topic = forced_topic or random.choice(STORY_TOPICS_WHATIF)

    if dual_language:
        narration_block = (
            '          "narration": "25-40 words in clear, natural ENGLISH — curious, vivid, present-tense",\n'
            '          "narration_hi": "25-40 words in natural spoken HINDI (Devanagari script). Same content as English narration but read naturally — NOT a literal word-for-word translation.",\n'
        )
        lang_note = "Each scene has BOTH English (narration) AND Hindi (narration_hi) versions of the same idea."
    else:
        narration_block = (
            '          "narration": "25-40 words in clear, natural ENGLISH — curious, vivid, present-tense",\n'
        )
        lang_note = "Narration is in English."

    style_options = ", ".join(f'"{k}"' for k in _WHATIF_VISUAL_STYLES.keys())

    prompt = f"""
You are a science communicator writing a 60-90 second "What If" thought-experiment for YouTube Shorts.

TOPIC: "{topic}"

TASK: Create a vertical (9:16) video script with EXACTLY 5 OR 6 scenes that imagines this hypothetical scenario plausibly.

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
   Open with the question itself, framed dramatically. Examples:
     "Imagine waking up tomorrow — and every human is gone."
     "What if Earth had rings like Saturn? You wouldn't sleep tonight."
   DO NOT open with "In this video..." or "Today we explore..." — get straight into the scenario.

Scenes 2-3 — SETUP & ESCALATION:
   Walk through the immediate consequences in vivid, concrete detail.
   Each scene reveals a new layer the viewer didn't expect.

Scene 4 (and 5 if 6-scene) — PEAK CONSEQUENCE:
   The single most striking implication. The "wait, what?" moment.

Final scene — RESOLUTION + REFLECTION:
   Land it. Tie back to something the viewer can feel about their own world.
   This scene MAY end with closure — every other scene must end with a hook
   forward (a question, a "...but", or an unresolved threat).

═══════════════════════════════════════════════════════════════
CONTENT QUALITY
═══════════════════════════════════════════════════════════════
- Every sentence must contain a NEW concrete detail (a specific number, place, organism, mechanism, timescale).
- NO vague abstractions ("things would change", "consequences would follow").
- Reference real science where applicable — actual species, distances, timescales, physical laws.
- 2-3 short sentences per scene for natural breathing pauses.
- 25-40 words per scene narration (target ~30-35).

═══════════════════════════════════════════════════════════════
OUTPUT — return ONLY valid JSON, no markdown fences, no preamble:
═══════════════════════════════════════════════════════════════
{{
  "title": "What If <topic phrasing>? (under 60 characters)",
  "description": "Hook sentence in first line. 100-150 words about the thought experiment. End with: \\n\\n#Shorts #WhatIf #ScienceShorts #ThoughtExperiment #Curiosity #Hypothetical #ScienceFacts #FutureEarth #SpeculativeScience #VyasaAI",
  "tags": ["what if","hypothetical","thought experiment","science","curiosity","speculative","science shorts","what if scenarios","science what if","alternate reality","future earth","mind blowing","क्या होगा अगर","विज्ञान","कल्पना","trending shorts"],
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
- Title: under 60 characters, MUST start with "What If"
- visual_style: MUST be exactly one of the allowed values
- EXACTLY 5 OR 6 scenes
- Narration MUST NOT contain URLs, hashtags, @mentions, or social-media text
- image_prompt, video_prompt, thumbnail_prompt all in English
- Description ends with the exact hashtag block above
- NO Mahabharata characters, gods, or mythology — this is science/curiosity content
"""

    raw = _call_llm(prompt)
    start = raw.find("{")
    end   = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in WhatIf LLM response:\n{raw[:300]}")
    data = _parse_llm_json(raw[start:end + 1])

    # Hard-trim narrations
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

    n_scenes  = len(data.get("scenes", []))
    word_avg  = (sum(len(s.get("narration", "").split()) for s in data["scenes"]) /
                 max(n_scenes, 1))
    print(f"    WhatIf script: {n_scenes} scenes, avg {word_avg:.1f} words/scene, "
          f"style={data['visual_style']}")

    data["content_type"] = "whatif"
    data["topic"]        = topic
    data["series"]       = "whatif"
    return data


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

    _MOTIVATIONAL_KEYWORDS = ("karma", "dharma", "lesson", "wisdom", "why", "power", "teaching")

    if forced_topic:
        topic = forced_topic
        content_type = (
            "motivational"
            if any(kw in forced_topic.lower() for kw in _MOTIVATIONAL_KEYWORDS)
            else "story"
        )
    else:
        content_type = random.choice(["story", "motivational"])
        topic = random.choice(STORY_TOPICS if content_type == "story" else MOTIVATIONAL_THEMES)

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

    prompt = f"""
    You are a master storyteller specialising in the Mahabharata epic, writing scripts for vertical YouTube videos that retain viewer attention from the first second to the last.

    You must strictly follow all rules and NEVER generate invalid or noisy text.

    TASK: Create a 60-90 second vertical (9:16) video script with EXACTLY 5 OR 6 scenes about a well-known incident from the Mahabharata.

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

    Scenes 2-3 — SETUP & RISING TENSION:
        Establish the characters, the situation, the conflict. Each
        sentence must build dread, anticipation, or curiosity.

    Scene 4 (and 5 if 6-scene script) — CLIMAX or REVELATION:
        The dramatic high point. The viewer should feel something —
        awe, shock, sorrow, vindication. Vivid sensory detail.

    Final scene — RESOLUTION + LESSON:
        Tie it off cleanly. Leave the viewer with a takeaway, a moral,
        or an emotional landing that makes the video feel complete.

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

    Rules for EVERY scene:
    - Each scene MUST advance the story. No filler. No repetition.
    - The story must be SELF-CONTAINED — a viewer who has never heard
      of the Mahabharata understands it fully by the end.
    - Use vivid, present-tense, sensory language ("the air thickens",
      "swords clash", "his eyes burn") — not abstract moralizing.
    - Reference specific characters and visible action in each scene.

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
    NARRATION LENGTH — CRITICAL
    ═══════════════════════════════════════════════════════════════
    EACH scene's narration must be 25-40 words.
    NEVER write fewer than 25 words per scene — this produces a too-short video.
    Aim for 30-35 words per scene as the sweet spot.
    At natural Hindi/English narration pace this gives ~10-13 seconds per scene.
    Use 2-3 short sentences per scene for natural breathing pauses.

    Total target: 5 scenes × ~33 words = 165 words OR 6 scenes × ~28 words = 168 words.
    Spoken duration: ~55-75 seconds (a fixed subscribe outro adds ~6s for 60-80s total video).

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
        [lighting style], [mood adjective], [palette: jewel-toned colours]

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
    OUTPUT — return ONLY valid JSON, no markdown fences, no preamble:
    ═══════════════════════════════════════════════════════════════
    {{
      "title": "Captivating title under 60 characters — no hashtags",
      "description": "Hook sentence that grabs attention in the first 2 lines. Then 100-150 words about the story. End with this exact hashtag block:\\n\\n#Shorts #Mahabharata #महाभारत #HinduMythology #Krishna #कृष्ण #BhagavadGita #भगवद_गीता #AncientIndia #EpicStory #Dharma #SpiritualWisdom #IndianMythology #HinduDharma #Arjuna #VedicWisdom #IndianHistory #MythologyShorts #KrishnaStories #trending",
      "tags": ["Mahabharata","महाभारत","Shorts","Hindu mythology","Krishna","कृष्ण","Arjuna","अर्जुन","Bhagavad Gita","भगवद गीता","Ancient India","dharma","spiritual","epic story","Indian history","Mahabharata shorts","mythology shorts","trending shorts","Hindu dharma","vedic wisdom","Indian mythology","spiritual shorts","krishna stories","kurukshetra"],
      "scenes": [
        {{
          "narration": "25-40 words in the specified LANGUAGE — vivid, present-tense, dramatic. 2-3 short sentences. ~10-13 seconds spoken.",
          "image_prompt": "Detailed English prompt following the [shot type] of [character(s)] in [emotion/action], in [environment], background contains [≥3 specific architectural/environmental elements: carved pillars, oil lamps, lotus reliefs, etc], [lighting], [mood], jewel-toned palette",
          "video_prompt": "Cinematic 5-second shot in English — characters in subtle motion, camera movement, lighting. Vertical 9:16.",
          "mood": "3-6 word English emotional tone phrase"
        }}
      ],
      "thumbnail_prompt": "Dramatic Mahabharata thumbnail — vibrant colours, cinematic, portrait composition"
    }}

    HARD RULES — violation makes the script unusable:
    - All narration MUST be in {lang_label}
    - All image_prompt, video_prompt, thumbnail_prompt MUST be in English
    - Title: under 60 characters, no hashtags in title
    - Narration per scene: 25-40 words, 2-3 sentences, ~10-13 seconds spoken
    - Narration MUST NOT contain URLs, hashtags (#), @mentions, English in Hindi videos, or any social-media text
    - Generate EXACTLY 5 OR 6 scenes — never fewer, never more
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

    # Try up to 3 times — if a response fails any quality gate (too short,
    # too few scenes, too repetitive, or "tha-tha-tha" verb tic), re-prompt
    # with a targeted reminder appended that names the specific failure.
    data = None
    last_offenders = []
    last_short     = False
    last_tha_tic   = False
    last_tha_ratio = 0.0
    for attempt in range(3):
        full_prompt = prompt
        if attempt > 0:
            reminders = []
            if last_short:
                reminders.append(
                    "Your previous response had narrations that were too short. "
                    "Each scene MUST be 25-40 words. Below 25 words is unacceptable."
                )
            if last_offenders:
                offender_str = ", ".join(f"'{w}' ({n}x)" for w, n in last_offenders[:5])
                reminders.append(
                    f"Your previous response REPEATED these words too many times: "
                    f"{offender_str}. Use SYNONYMS. Each sentence must contain a "
                    f"NEW concrete detail (a different name, place, or action). "
                    f"DO NOT restate the same fact twice in different words."
                )
            if last_tha_tic:
                reminders.append(
                    f"Your previous response had the \"था-था-था\" verbal tic — "
                    f"{int(last_tha_ratio * 100)}% of sentences ended with "
                    f"था/थी/थे/थीं. AT MOST 2 sentences in the whole script "
                    f"may end that way. Rewrite using HISTORICAL PRESENT "
                    f"(\"द्रौपदी रोती है\" not \"रोई थी\"), simple perfective "
                    f"WITHOUT auxiliary (\"भीम ने प्रतिज्ञा ली\" not \"ली थी\"), "
                    f"nominalization (\"द्रौपदी का अपमान — एक क्षण...\"), or "
                    f"exclamatory beats. Mix the patterns. The script must MOVE, "
                    f"not list events chronologically."
                )
            if reminders:
                full_prompt += "\n\nCRITICAL REMINDERS:\n- " + "\n- ".join(reminders)

        raw = _call_llm(full_prompt)

        # Extract the JSON object — handles thinking text, code fences, and preamble
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in LLM response:\n{raw[:300]}")
        raw = raw[start:end + 1]

        data = _parse_llm_json(raw)

        # Hard-enforce upper bound — LLMs sometimes ignore word limits
        for scene in data.get("scenes", []):
            scene["narration"] = _trim_narration(scene["narration"])

        scenes = data.get("scenes", [])
        word_counts = [len(s["narration"].split()) for s in scenes]
        avg_words = sum(word_counts) / max(len(word_counts), 1)
        n_scenes = len(scenes)

        last_short = (n_scenes < 5 or avg_words < 22)
        # Threshold 4: a character at the centre of the story (Bhishma in a
        # Bhishma video) can appear ~4 times naturally. 5+ times signals that
        # supporting characters and details are being skipped in favour of
        # restating the main name. Same threshold for abstract nouns flags
        # filler like "valor" / "वीरता" appearing 5+ times.
        rep_ok, last_offenders = _check_repetition(scenes, max_repeats=4, topic=topic)

        # Hindi-only "tha-tha-tha" verbal tic check. Threshold 0.35 = at most
        # ~1 in 3 sentences may end with past auxiliary; otherwise the
        # narration reads as a chronological list rather than cinema.
        if language == "hi":
            tha_ok, last_tha_ratio, tha_hits, tha_total = _check_past_aux_tic(scenes, threshold=0.35)
            last_tha_tic = not tha_ok
        else:
            tha_ok = True
            last_tha_tic = False

        print(f"    Script: {n_scenes} scenes, avg {avg_words:.1f} words/scene "
              f"(per-scene: {word_counts})")
        if last_offenders:
            top = ", ".join(f"{w}×{n}" for w, n in last_offenders[:5])
            print(f"    [warn] Repetition: {top}")
        if last_tha_tic:
            print(f"    [warn] Past-aux tic: {tha_hits}/{tha_total} sentences "
                  f"end with था/थी/थे/थीं ({last_tha_ratio:.0%})")

        # Acceptable if length OK AND repetition AND verb-variety all pass
        if not last_short and rep_ok and tha_ok:
            break

        if attempt < 2:
            why = []
            if last_short:
                why.append(f"too short ({n_scenes} scenes / {avg_words:.1f} avg words)")
            if not rep_ok:
                why.append(f"{len(last_offenders)} repeated words")
            if last_tha_tic:
                why.append(f"था-tic {last_tha_ratio:.0%}")
            print(f"    [retry] {'; '.join(why)}. Re-prompting...")

    data["language"] = language
    data["content_type"] = content_type
    data["topic"] = topic
    data["series"] = "mahabharata"
    return data
