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

# ── Story Topics ──────────────────────────────────────────────────────────────
STORY_TOPICS = [
    "Arjuna's dilemma on the Kurukshetra battlefield",
    "Krishna reveals the Bhagavad Gita to Arjuna",
    "Draupadi's swayamvara and the fish-eye challenge",
    "Bhishma's unbreakable oath of celibacy",
    "Karna's birth and his lifelong battle for recognition",
    "The cursed dice game that exiled the Pandavas",
    "Abhimanyu's heroic last stand inside the Chakravyuha",
    "Draupadi's humiliation in the Kuru court",
    "Krishna reveals his Vishwaroop - the universal cosmic form",
    "Ekalavya's devotion and his ultimate sacrifice as Gurudakshina",
    "Shakuni's cunning plan to destroy the Pandavas",
    "Duryodhana's jealousy ignites the great war",
    "Dronacharya and the bird's eye lesson in focus",
    "The birth of Pandavas and their divine origins",
    "Barbareek - the mightiest warrior who watched the war",
    "Vidura's wisdom versus Dhritarashtra's blind love",
    "The death of Ghatotkacha and Karna's divine weapon",
    "Yudhishthira's final journey to heaven",
    "The friendship of Krishna and Arjuna - a timeless bond",
    "Hanuman on Arjuna's chariot - the hidden blessing",
]

# ── Motivational Themes ───────────────────────────────────────────────────────
MOTIVATIONAL_THEMES = [
    "Karma Yoga - acting without attachment to results",
    "Why dharma must be chosen over personal comfort",
    "The power of perseverance - Arjuna's lifelong training",
    "Truth always triumphs - Yudhishthira's path",
    "Overcoming fear through knowledge - Gita's wisdom",
    "The real meaning of victory according to Krishna",
    "Forgiveness - the greatest weapon from Mahabharata",
    "Controlling anger - lessons from the Pandavas",
    "Self-belief and courage - Arjuna's transformation",
    "Why attachment is the root of all suffering - Bhagavad Gita",
]


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

    prompt = f"""
You are a master storyteller specialising in the Mahabharata epic.

Create a YouTube Shorts script (30–60 seconds, 4–6 scenes) about:
TOPIC: "{topic}"
LANGUAGE: {lang_label}
STYLE: {style_note}

Return ONLY valid JSON — no markdown fences, no extra text:
{{
  "title": "Captivating Shorts title under 60 characters — no hashtags in title",
  "description": "Engaging hook sentence that grabs attention in first 2 lines. Then 100-150 words about the story. End with this exact block:\n\n#Shorts #Mahabharata #महाभारत #HinduMythology #Krishna #कृष्ण #BhagavadGita #भगवद_गीता #AncientIndia #EpicStory #Dharma #SpiritualWisdom #IndianMythology #HinduDharma #Arjuna #VedicWisdom #IndianHistory #MythologyShorts #KrishnaStories #trending",
  "tags": ["Mahabharata","महाभारत","Shorts","Hindu mythology","Krishna","कृष्ण","Arjuna","अर्जुन","Bhagavad Gita","भगवद गीता","Ancient India","dharma","spiritual","epic story","Indian history","Mahabharata shorts","mythology shorts","trending shorts","Hindu dharma","vedic wisdom","Indian mythology","spiritual shorts","krishna stories","kurukshetra"],
  "scenes": [
    {{
      "narration": "Spoken narration text in {lang_label} — 2-3 sentences written for a master storyteller's voice. Use vivid imagery, natural dramatic pauses (use '...' for pauses), and emotionally charged language that grabs the listener instantly. Write as if narrating to a spellbound audience around a fire.",
      "image_prompt": "Detailed English image prompt — portrait orientation composition, specific characters, body language, environment, colour palette, and mood",
      "video_prompt": "Cinematic 5-second shot in English — characters in motion, camera movement, environment, lighting, mood. Vertical portrait composition.",
      "mood": "One evocative phrase describing the emotional tone, e.g. 'tense and apocalyptic', 'serene golden dawn', 'grief-stricken and desolate'"
    }}
  ],
  "thumbnail_prompt": "Dramatic thumbnail — epic Mahabharata scene, vibrant colours, cinematic, portrait composition"
}}

Rules:
- All narration must be in {lang_label}
- All image_prompt, video_prompt, thumbnail_prompt must ALWAYS be in English
- Title: under 60 characters, no hashtags
- Narration: 2-3 emotionally gripping sentences per scene
- image_prompt: detailed portrait-oriented scene with characters, body language, environment, colour palette
- video_prompt: cinematic vertical shot description — specific motion, camera, lighting
- mood must be 3-6 words in English
- image_prompt and video_prompt must reference the mood
- description must end with the exact hashtags listed above
"""

    raw = _call_llm(prompt)

    # Extract the JSON object — handles thinking text, code fences, and preamble
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response:\n{raw[:300]}")
    raw = raw[start:end + 1]

    data = _parse_llm_json(raw)
    data["language"] = language
    data["content_type"] = content_type
    data["topic"] = topic
    return data
