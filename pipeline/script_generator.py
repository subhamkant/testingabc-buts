from google import genai
import json
import random
import os
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


def _call_llm(prompt: str) -> str:
    """
    Calls Groq (primary, 14 400 RPD free) then falls back to Gemini 2.5 Flash.
    Returns the raw text response.

    Token budgets are sized to fit the longest prompts in this file —
    Mahabharata's 2-pass dramatization step emits ~5-6k chars of bilingual
    JSON (title + 100-150-word description + 17 tags + 5 scenes with both
    English + Hindi narration + image/video prompts + character dialogue +
    thumbnail prompt). Earlier defaults (Groq 4096, Gemini library default
    of ~8192 output tokens) truncated mid-description on Mahabharata runs
    and produced unparseable JSON. Values below have ~2x headroom over the
    longest observed valid output.

    Gemini fallback is hardened against the failure mode observed on
    2026-05-11: Groq daily quota exhausted -> Gemini returned `503 UNAVAILABLE
    "This model is currently experiencing high demand"`. The naive single-call
    fallback crashed the whole workflow attempt. We now retry the same model
    with exponential backoff on transient errors (5xx / 429 / timeout), then
    fall over to gemini-2.5-flash-lite (lower-demand sibling) as a last resort
    before raising.
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
            print(f"    Groq failed: {e} — falling back to Gemini...")

    # ── Fallback: Gemini 2.5 Flash with retry-on-transient ───────────────────
    # Explicit max_output_tokens=16384 (was library default ~8192). Gemini 2.5
    # Flash supports up to 65k output tokens; 16k is enough headroom for our
    # longest bilingual script JSON without paying for tokens we don't use.
    from google.genai import types as _genai_types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    config = _genai_types.GenerateContentConfig(max_output_tokens=16384)

    return _gemini_call_with_retry(client, prompt, config)


def _gemini_call_with_retry(client, prompt: str, config) -> str:
    """
    Call gemini-2.5-flash with retry-on-transient (503/429/5xx/timeout),
    then fall over to gemini-2.5-flash-lite as a last-resort sibling model.

    Backoff schedule: 0s, 8s, 20s, 45s — total worst-case wait ~73s before
    we even try the fallback model. That's tolerable because the next layer
    up (the workflow's chained retry job) would otherwise spin up a fresh
    runner from scratch (~90s of setup) only to crash on the same call.
    """
    backoffs = [0, 8, 20, 45]
    last_err = ""

    # Tier 1 — gemini-2.5-flash with retry
    for attempt, wait in enumerate(backoffs):
        if wait:
            print(f"    Gemini 2.5 Flash retry in {wait}s (attempt {attempt + 1}/{len(backoffs)})...")
            time.sleep(wait)
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=config,
            )
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            last_err = err_str[:200]
            is_transient = any(m in err_str for m in _GEMINI_TRANSIENT_MARKERS)
            if not is_transient:
                # Non-transient error — bubble up immediately, don't waste retry budget
                raise
            print(f"    Gemini 2.5 Flash transient error: {err_str[:120]}")
            # Loop continues to next backoff slot

    # Tier 2 — gemini-2.5-flash-lite fallback (lighter sibling, less demand)
    print(f"    Gemini 2.5 Flash exhausted retries — trying gemini-2.5-flash-lite as last resort...")
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config=config,
        )
        print(f"    [OK] gemini-2.5-flash-lite succeeded as fallback")
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(
            f"Both Gemini models failed. "
            f"flash (after {len(backoffs)} attempts): {last_err}; "
            f"flash-lite: {str(e)[:200]}"
        )


# ── Story Topics — well-known Mahabharata incidents ───────────────────────────
STORY_TOPICS = [
    # ── Core canonical incidents (original set) ─────────────────────────────
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
    "Karna's reunion with Kunti — the secret she kept for decades",
    "Drona teaches archery — Arjuna sees only the eye of the bird",

    # ── YouTube-suggested topics (expanded reach into lesser-known incidents)
    # Curated from algorithm-recommended Mahabharata search terms — these are
    # what viewers are actively searching for but few channels cover well.
    "Krishna creates a fake sunset to trap Jayadratha — Arjuna's vengeance fulfilled",
    "Jayadratha's lone stand against the entire Pandava army on day 13",
    "Jayadratha's severed head lands in his father's meditative lap — a curse fulfilled",
    "Krishna offers Karna the throne before the war — and Karna refuses, sealing his fate",
    "Bhishma faces Amba reborn as Shikhandi — the curse that ends an unstoppable warrior",
    "Gandhari blindfolds herself for Dhritarashtra — and never sees her hundred sons",
    "Aravan offers his life so the Pandavas can win — a one-day marriage to Mohini",
    "Sahadeva sees every future moment — but is cursed to never warn anyone",
    "Ashwatthama's immortality becomes his eternal punishment after the war",
    "Krishna beheads Shishupala with the Sudarshan — exactly 100 sins counted",
    "Kunti hides Karna's identity from her own sons — until the night before war",
    "Barbarika's three arrows could end the war in one breath — Krishna asks for his head",
    "Parshurama curses Karna — the moment Karna forgets his weapons in battle",
    "Balarama refuses to fight at Kurukshetra — chooses a pilgrimage instead",
    "Vidura warns Dhritarashtra — and the blind king refuses to listen, again and again",
    "Yudhishthira answers the Yaksha's questions — the test that revives his brothers",
    "Shikhandi appears on the battlefield — and Bhishma lays down his bow",
    "Arjuna's silent grief after Abhimanyu's death — and the vow that consumed him",
    "The forgotten warriors who delayed Arjuna so Jayadratha could survive",
    "Nakula's silent mastery — the most beautiful warrior of the Mahabharata",
    "The hidden conversation between Bhishma and Amba — a tragic cosmic destiny",
    "Krishna's celestial mechanics — orchestrating the premature sunset",
    "Why Drona broke his own teaching for one student — Arjuna's unspoken privilege",
    "Bhima vows to drink Dushasana's blood — and keeps his word at Kurukshetra",
    "The night before the war — Karna learns the truth from Kunti and accepts his fate",
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

    # Fix 3: last resort — repair a truncated-mid-output response by closing
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

        raw = _call_llm(full_prompt)
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
      "narration": "Hindi (Devanagari) — Krishna first-person to {listener_vocative}. Per-scene length: scene 1/2/4 ~22-26 words, SCENE 3 ~14-18 words (short imperative peak, mostly 3-7 word sentences), scene 5 ~20-24 words. Use spoken Hindi (no Sanskritized words). At least one imperative verb per scene.",
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
- Per-scene length: scenes 1/2/4 are 22-26 words, SCENE 3 is 14-18 words
  (the imperative peak — short bursts, do not over-pad), scene 5 is 20-24 words
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
    for attempt in range(3):
        full_prompt = prompt
        if attempt > 0:
            reminders = []
            if last_short:
                reminders.append(
                    "Your previous response had narration length issues. "
                    "Per-scene targets: scenes 1/2/4 = 22-26 words, SCENE 3 "
                    "= 14-18 words (the imperative peak — KEEP IT SHORT, don't "
                    "pad), scene 5 = 20-24 words. You MUST produce EXACTLY 5 "
                    "scenes. Total ~100-120 words across all 5 scenes."
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

        raw = _call_llm(full_prompt)

        start = raw.find("{")
        end   = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON object found in Krishna LLM response:\n{raw[:300]}")
        data = _parse_llm_json(raw[start:end + 1])

        # Hard-trim narrations
        for scene in data.get("scenes", []):
            scene["narration"] = _trim_narration(scene["narration"])

        scenes      = data.get("scenes", [])
        word_counts = [len(s["narration"].split()) for s in scenes]
        avg_words   = sum(word_counts) / max(len(word_counts), 1)
        n_scenes    = len(scenes)

        # Acceptance threshold: must have 5 scenes, AND avg ≥ 18 words/scene
        # (the new format averages ~22 words; below 18 means the LLM lost the
        # plot and produced 1-2 word stub scenes). The prompt itself enforces
        # the per-scene shape; the avg is just a floor.
        last_short = (n_scenes != 5 or avg_words < 18)
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

        if attempt < 2:
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
      "title": "[English title with named character + power keyword] | [Hindi title with character name] — under 60 chars total, no hashtags",
      "description": "Hook sentence under 90 chars that expands the title's promise with concrete detail.\\n\\n#Shorts #Mahabharata #महाभारत #Krishna #HinduMythology\\n\\n100-150 words about the story, weaving in named characters and the specific incident. Build curiosity. Don't spoil the ending in the description.\\n\\n#Shorts #Mahabharata #महाभारत #HinduMythology #Krishna #कृष्ण #BhagavadGita #भगवद_गीता #Arjuna #अर्जुन #Kurukshetra #AncientIndia #IndianMythology #Dharma #EpicStory #MythologyShorts #VedicWisdom #HinduDharma #IndianHistory #SpiritualShorts #trending",
      "tags": ["topic-specific long-tail tag 1","topic-specific long-tail tag 2","named character 1 (English)","named character 1 (Hindi/Devanagari)","named character 2","specific incident name","viewer-search query like 'why X happened'","Mahabharata","महाभारत","Shorts","Hindu mythology","Krishna","कृष्ण"],
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
    - Title: under 60 chars, MUST follow `[English] | [Hindi]` format with a
      named character + power keyword in the English half. NO hashtags in title.
    - Description MUST follow the 3-block structure: hook line ≤90 chars,
      blank line, 5 inline hashtags, blank line, body, blank line, full
      hashtag block (high-volume hashtags first).
    - Tags MUST include topic-specific long-tail keywords (named characters
      in this topic + specific incident name + viewer-search queries) on
      top of the generic Mahabharata fallbacks.
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

    ═══════════════════════════════════════════════════════════════
    PRIORITY ORDER — IF YOU CAN'T SATISFY EVERY RULE
    ═══════════════════════════════════════════════════════════════
    Narration storytelling quality is #1. SEO metadata is #5.
    If the SEO formatting and the narration quality ever pull against
    each other, ALWAYS keep the narration punchy and concrete and let
    SEO take the hit. A boring title on a gripping story still earns
    views; a perfectly-optimized title on a flat narration does not.

    Priority order, top is most important:
      1. Scene 1 hook is a real scroll-stopper (pattern A/B/C, named character)
      2. Every non-final scene ends with a forward-pull (curiosity gap)
      3. Hindi: verb variety — at most 2 sentences end in था/थी/थे/थीं
      4. Each scene 25-40 words, present-tense, vivid sensory detail
      5. SEO title/description/tag formatting

    The two reference videos this channel is optimizing toward — "Bhishma's
    Untold Sacrifice" and "What If Humans Vanish?" — both prioritized 1-4
    above 5. Do the same.
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
