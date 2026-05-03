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

Create a YouTube video script (2–2.5 minutes, 6–8 scenes) about:
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
      "narration": "Spoken narration in {lang_label} — STRICTLY 2-3 short sentences, maximum 35 words total. Must take 10-15 seconds to speak aloud. Use vivid imagery and dramatic pauses (use '...'). No long sentences.",
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
- Narration: STRICTLY 2-3 short sentences, MAX 35 words, 10-15 seconds when spoken aloud
- Narration MUST NOT contain any URLs, hashtags (#), @mentions, or social media text
- Generate 6-8 scenes — total video must be 2 to 2.5 minutes
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
