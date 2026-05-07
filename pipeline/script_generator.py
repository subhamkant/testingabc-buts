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


def generate_script(language: str = "en", forced_topic: str = None) -> dict:
    """
    Returns a dict with:
      title, description, tags, scenes, thumbnail_prompt,
      language, content_type, topic
    """
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
            "- Narration must be ONLY in Hindi (Devanagari script)\n"
            "- Do NOT use English words or abbreviations\n"
            "- Do NOT generate meaningless or broken words (like एमएस, ML, etc.)\n"
            "- Do NOT include URLs, references, or metadata\n"
            "- Use simple, natural spoken Hindi\n"
            "- If unsure, generate simpler Hindi — NEVER invent words\n"
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

    prompt = f"""
    You are a master storyteller specialising in the Mahabharata epic, writing scripts for vertical YouTube videos that retain viewer attention from the first second to the last.

    You must strictly follow all rules and NEVER generate invalid or noisy text.

    TASK: Create a 60-90 second vertical (9:16) video script with EXACTLY 5 OR 6 scenes about a well-known incident from the Mahabharata.

    TOPIC: "{topic}"
    LANGUAGE: {lang_label}
    STYLE: {style_note}
    {language_rules}

    ═══════════════════════════════════════════════════════════════
    STORY STRUCTURE — THE VIEWER MUST NOT GET BORED
    ═══════════════════════════════════════════════════════════════
    Every scene must earn its place. Follow this dramatic arc:

    Scene 1 — HOOK (the most critical 10 seconds of the video):
        Open with the most shocking, mysterious, or emotionally charged
        moment of the entire story. Set the stakes immediately. Make the
        viewer NEED to keep watching to find out what happens.

    Scenes 2-3 — SETUP & RISING TENSION:
        Establish the characters, the situation, the conflict. Each
        sentence must build dread, anticipation, or curiosity.

    Scene 4 (and 5 if 6-scene script) — CLIMAX or REVELATION:
        The dramatic high point. The viewer should feel something —
        awe, shock, sorrow, vindication. Vivid sensory detail.

    Final scene — RESOLUTION + LESSON:
        Tie it off cleanly. Leave the viewer with a takeaway, a moral,
        or an emotional landing that makes the video feel complete.

    Rules for EVERY scene:
    - Each scene MUST advance the story. No filler. No repetition.
    - The story must be SELF-CONTAINED — a viewer who has never heard
      of the Mahabharata understands it fully by the end.
    - Use vivid, present-tense, sensory language ("the air thickens",
      "swords clash", "his eyes burn") — not abstract moralizing.
    - Reference specific characters and visible action in each scene.

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
    OUTPUT — return ONLY valid JSON, no markdown fences, no preamble:
    ═══════════════════════════════════════════════════════════════
    {{
      "title": "Captivating title under 60 characters — no hashtags",
      "description": "Hook sentence that grabs attention in the first 2 lines. Then 100-150 words about the story. End with this exact hashtag block:\\n\\n#Shorts #Mahabharata #महाभारत #HinduMythology #Krishna #कृष्ण #BhagavadGita #भगवद_गीता #AncientIndia #EpicStory #Dharma #SpiritualWisdom #IndianMythology #HinduDharma #Arjuna #VedicWisdom #IndianHistory #MythologyShorts #KrishnaStories #trending",
      "tags": ["Mahabharata","महाभारत","Shorts","Hindu mythology","Krishna","कृष्ण","Arjuna","अर्जुन","Bhagavad Gita","भगवद गीता","Ancient India","dharma","spiritual","epic story","Indian history","Mahabharata shorts","mythology shorts","trending shorts","Hindu dharma","vedic wisdom","Indian mythology","spiritual shorts","krishna stories","kurukshetra"],
      "scenes": [
        {{
          "narration": "25-40 words in the specified LANGUAGE — vivid, present-tense, dramatic. 2-3 short sentences. ~10-13 seconds spoken.",
          "image_prompt": "Detailed English prompt — portrait composition, specific characters with body language, environment, colour palette, mood",
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
    - image_prompt: detailed portrait scene with characters, body language, environment, palette
    - video_prompt: cinematic vertical shot — specific motion, camera, lighting
    - mood: 3-6 words in English
    - image_prompt and video_prompt MUST reference the mood
    - description MUST end with the exact hashtag block above
    """

    # Try up to 2 times — if first attempt gives too-short narrations,
    # re-prompt with a strict reminder appended.
    data = None
    for attempt in range(2):
        full_prompt = prompt
        if attempt > 0:
            full_prompt += (
                "\n\nIMPORTANT REMINDER: your previous response had narrations that were too short. "
                "Each scene's narration MUST be 25-40 words. Below 25 words is unacceptable. "
                "Aim for 30-35 words per scene."
            )

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

        print(f"    Script: {n_scenes} scenes, avg {avg_words:.1f} words/scene "
              f"(per-scene: {word_counts})")

        # Acceptable if at least 5 scenes AND avg >= 22 words
        if n_scenes >= 5 and avg_words >= 22:
            break
        if attempt == 0:
            print(f"    [warn] Script too short (need 5+ scenes & 22+ avg words). Re-prompting...")

    data["language"] = language
    data["content_type"] = content_type
    data["topic"] = topic
    return data
