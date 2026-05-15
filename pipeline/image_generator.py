import requests
import os
import re
import time
import json
import base64
import io
import subprocess
from urllib.parse import quote

# Mahabharata style suffix — photorealistic cinematic period film aesthetic
# tuned for NATURAL color and skin tones. Earlier iterations went too far in
# either direction:
#   - "Amar Chitra Katha cel-shaded" → cartoonish (Anger's Fire-style)
#   - heavy "warm Bhansali" + saturated jewel tones → magenta/pink wash on
#     everything (the "भीम और दुर्योधन" upload — purple skin, pink sky)
# This sweet spot leans on the Star Bharat Mahabharat live-action / Baahubali
# reference: photoreal, ornate, jewel-toned BUT with accurate skin and
# balanced color. Explicitly bans color casts in the negative cues.
#
# WhatIf series uses a different per-style suffix from _WHATIF_STYLE_SUFFIXES
# based on the script's visual_style — this Mahabharata suffix does not apply.
# MORTAL variant — golden-bronze skin anchor for mortal Mahabharata warriors
# (Karna, Arjuna, Bhima, etc.). The "luminous golden glowing complexion" line
# was added 2026-05-14 evening to push back against FLUX silhouette-dark
# tendencies under "oil-lamp lighting" cues.
STYLE_SUFFIX_MORTAL = (
    # Photoreal anchor — kept short for distilled FLUX variants (Pollinations,
    # Cloudflare) which honor long prompts less reliably than HF FLUX-schnell.
    # WALKED BACK from earlier version: "physically-based skin with visible
    # pores and fine facial hair" + "ultra-sharp 8K detail throughout" were
    # producing splotchy/scarred face artifacts and broken eye anatomy on
    # women's close-ups in the 2026-05-13 local test (Draupadi face had
    # mottled "wet/grimy" texture, eyelid anatomy was broken in another shot).
    # Distilled FLUX-schnell over-interprets "visible pores" as "lots of
    # skin texture" and produces splotches. Softer anchors below.
    "photorealistic cinematic film still shot on Arri Alexa LF, "
    "Kodak Vision3 5219 film stock, "
    "natural skin texture, realistic facial features, "
    # Eye-specific anchor — added 2026-05-14 after the Karna-arc local test
    # shipped 7/10 frames with dead-eye / black-void pupils. Distilled FLUX
    # at 4 steps loses eye micro-detail without an explicit eye cue. This
    # is eye-only — does not re-introduce "pores"/"8K" that caused splotchy
    # skin on the prior iteration.
    "detailed expressive eyes with clearly defined iris and pupils, "
    "natural catch-light reflections in the eyes, "
    # Face-exposure anchor — added 2026-05-14 after the v3_images smoke test
    # shipped Karna with near-black skin from FLUX over-interpreting "oil-lamp"
    # / "subtle glow" lighting cues. Canonical Karna is golden-bronze (Surya's
    # son), not silhouette-dark. This anchor pushes back without overruling
    # mood / shadow direction from the scene prompt.
    "warm golden-bronze skin tone for Indian characters, "
    "luminous golden glowing complexion, "
    "well-lit faces with key-light on the face, even facial exposure, "
    "ancient India Mahabharat live-action / Baahubali period-film aesthetic, "
    "carved sandstone temple architecture in sharp focus, oil-lamp lighting, "
    "balanced natural color grading, neutral whites, true skin tones, "
    "sharp focus on subject, clear facial features, "
    "no global color wash, no orange filter, no magenta or pink cast, "
    "no CGI plastic look, no airbrushed skin"
)

# DIVINE variant — same as MORTAL minus the golden-bronze skin anchor that
# conflicts with canonical divine skin tones (Krishna indigo-blue, Hanuman
# red-gold). Added 2026-05-14 after the "Krishna to Karna" upload showed
# Krishna rendered as dark teal/muddied-blue because the global golden anchor
# was fighting his `characters.json` "dark indigo-blue divine skin" descriptor.
# All other anchors (eye detail, well-lit face, photoreal cinema, anti-color-
# wash negatives) preserved — only the skin-tone enforcement changes.
STYLE_SUFFIX_DIVINE = (
    "photorealistic cinematic film still shot on Arri Alexa LF, "
    "Kodak Vision3 5219 film stock, "
    "natural skin texture, realistic facial features, "
    "detailed expressive eyes with clearly defined iris and pupils, "
    "natural catch-light reflections in the eyes, "
    # Skin-tone routing for divine + mortal in the same frame: explicitly name
    # BOTH so FLUX doesn't average them into a single muddy tone. The character
    # descriptor injected by _inject_characters already says "indigo-blue divine
    # skin" for Krishna and "tan-brown body + vanara face with reddish-brown fur"
    # for Hanuman — this anchor reinforces Krishna's blue AND keeps mortal
    # warriors bronze when they share a frame with him. Hanuman's per-character
    # descriptor is detailed enough on its own; don't fight it from the suffix.
    "Krishna with brilliant indigo-blue divine skin (when present in scene), "
    "mortal warriors with warm golden-bronze skin tone, "
    "well-lit faces with key-light on the face, even facial exposure, "
    "ancient India Mahabharat live-action / Baahubali period-film aesthetic, "
    "carved sandstone temple architecture in sharp focus, oil-lamp lighting, "
    "balanced natural color grading, neutral whites, "
    "sharp focus on subject, clear facial features, "
    "no global color wash, no orange filter, no magenta or pink cast, "
    "no CGI plastic look, no airbrushed skin"
)

# Backwards-compat alias — kept so existing `style_suffix: str = STYLE_SUFFIX`
# default-arg signatures continue to work. Mortal is the safe default since
# the divine override fires only when a divine character is detected in the
# prompt (see generate_image_bytes).
STYLE_SUFFIX = STYLE_SUFFIX_MORTAL

# WhatIf series style suffixes — picked by `script["visual_style"]`. Mahabharata
# style does NOT apply to WhatIf scripts; the LLM picks the most suitable style
# per topic (e.g. dinosaurs -> nature-doc, black hole -> sci-fi-cinematic).
_WHATIF_STYLE_SUFFIXES = {
    "photoreal-3d": (
        "photorealistic CGI render, octane / unreal-engine-5 quality, "
        "physically-based shading, accurate scale and proportions, "
        "raytraced reflections, soft global illumination, volumetric atmosphere, "
        "scientific-illustration accuracy, true-to-life materials and textures, "
        "ultra-sharp 8K detail, professional cinematic lighting, "
        "high dynamic range, neutral color grading, no fantasy distortion"
    ),
    "nature-doc": (
        "BBC Planet Earth documentary cinematography, shot on Arri Alexa Mini LF, "
        "telephoto 600mm lens compression, naturalistic golden-hour lighting, "
        "true-to-life animal anatomy, accurate ecosystem detail, "
        "shallow depth of field with creamy bokeh, "
        "ultra-sharp 8K wildlife photography detail, professional color grading, "
        "no fantasy elements, no cartoon styling"
    ),
    "sci-fi-cinematic": (
        "Denis Villeneuve sci-fi cinematography, Roger Deakins lighting, "
        "shot on 35mm anamorphic Kodak Vision3 5219, "
        "monumental scale composition with tiny human silhouettes for scale, "
        "atmospheric haze and god-rays, muted desaturated palette with one accent color, "
        "deep ultra-wide compositions, cinematic 2.39:1 framing feel, "
        "ultra-sharp 8K detail, no neon kitsch, no pulp space-opera tropes"
    ),
    "illustrated": (
        "high-end editorial illustration, National Geographic feature art style, "
        "polished concept art with painterly detail, ArtStation Featured quality, "
        "dramatic composition with strong focal hierarchy, "
        "vivid but believable color palette, sharp readable subjects, "
        "no anime, no cel-shading, no flat cartoon styling"
    ),
}


def _resolve_style_suffix(series: str, visual_style: str) -> str:
    """Return the right style suffix for the series + visual_style."""
    if series == "whatif":
        return _WHATIF_STYLE_SUFFIXES.get(visual_style, _WHATIF_STYLE_SUFFIXES["photoreal-3d"])
    return STYLE_SUFFIX

# Negative prompt — distilled FLUX variants (Pollinations, Cloudflare)
# down-weight late tokens, so the highest-priority failure modes are
# front-loaded. The order below reflects what the Volcanoes + Gandhari
# analysis flagged as the most-recurring visible quality regressions
# (heavy color wash, plastic skin, CGI-game-character vibe).
_NEGATIVE = (
    # ── Eye-detail failure mode (TOP priority — 2026-05-14 Karna-arc local
    # test shipped 7/10 frames with dead-eye / black-void pupils. Distilled
    # FLUX-schnell loses eye micro-detail; front-loaded negatives push it
    # to actually render iris/pupil structure rather than dark voids). ──
    "dead eyes,glassy eyes,vacant stare,blank eyes,soulless eyes,"
    "missing pupils,missing iris,black void eyes,recessed eye sockets,"
    "asymmetric eyes,one eye closed,wonky eyes,smeared eyes,blurred eyes,"
    "eyes without detail,unfocused eyes,"
    # ── Face-underexposure failure mode (2026-05-14 v3_images test: Karna
    # rendered with near-black skin from FLUX over-reading "oil-lamp" cues
    # as silhouette lighting; canonical Karna is golden-bronze). ──
    "underexposed face,blackened skin,overly dark face,silhouette face,"
    "face in deep shadow,unlit face,muddy skin tone,washed-out dark skin,"
    "ashen face,grey skin,"
    # ── Face-distortion failure mode (observed in 2026-05-13 test where
    # women's faces had splotchy/scarred patterns + broken eye anatomy.
    # Distilled FLUX-schnell variants downweight late negatives). ──
    "splotchy skin,mottled face,scarred face,skin condition,acne,"
    "exaggerated pore detail,over-textured skin,patchy skin,"
    "deformed eyes,broken facial anatomy,extra eye,"
    "missing eye,crooked eye,droopy eyelid,merged eyebrows,"
    # ── Color-wash failure mode (most visible color regression) ──
    "orange cast,magenta cast,pink cast,purple cast,color wash,"
    "warm filter,heavy filter,sepia overlay,monochrome filter,"
    "over-saturated,over-graded,"
    # ── Plastic / CGI-character failure mode ──
    "cgi plastic skin,doll-like face,waxy skin,smooth airbrushed skin,"
    "3d render plastic,video game character,pixar style,"
    "unreal engine character,over-rendered,"
    # ── Cartoon / illustration bans ──
    "cartoon,anime,cel shaded,illustration,drawing,comic book,"
    # ── Embedded-text failure mode (2026-05-14 production check: FLUX-schnell
    # garbled "Vyasa AI" as VyssA / Virtasy / Vilysaria / Viysas across uploaded
    # outro frames. Distilled FLUX cannot reliably spell brand names — keep
    # text out of the prompts AND front-load explicit negatives so the model
    # avoids letterforms even when prompts accidentally suggest them). ──
    "text,letters,letterforms,typography,calligraphy,handwriting,"
    "channel name,subscribe text,logo text,bold text,glowing text,"
    "scribbled letters,garbled text,misspelled text,fake text,"
    "watermark,signature,caption,subtitle text in image,"
    # ── Standard quality / anatomy fixes (preserved from prior version) ──
    "blurry,blur,out of focus,low quality,pixelated,distorted,"
    "ugly,bad anatomy,logo,duplicate,deformed,"
    "extra fingers,six fingers,seven fingers,too many fingers,"
    "mutated hands,malformed hands,fused fingers,missing fingers,"
    "extra limbs,extra arms,malformed limbs,disfigured,"
    "cross-eyed,bad proportions"
)

# 3 compositional angles per scene — gives genuine visual variety.
# "dramatic close-up" was walked back to "medium close-up" on 2026-05-14
# after the Karna-arc local test shipped 7/10 frames with dead-eye / black-
# void pupils. FLUX-schnell at 4 steps cannot resolve eye micro-detail when
# the eye fills > ~15% of the frame — medium close-up keeps the emotional
# punch but gives the model enough pixel budget to actually render iris/
# pupil structure.
_SHOT_ANGLES = [
    "WIDE SHOT, ",                  # establishing — env + character together
    "MEDIUM SHOT, ",                # mid-range character focus
    "MEDIUM CLOSE-UP, ",            # head-and-shoulders, eyes still readable
]

# Divine non-golden-skin characters — when one of these is referenced anywhere
# in an image prompt, swap from STYLE_SUFFIX_MORTAL to STYLE_SUFFIX_DIVINE so
# the global golden-bronze anchor doesn't override their canonical color.
#   Krishna  — dark indigo-blue divine skin
#   Hanuman  — red-gold / red-orange divine form
# Barbarik is NOT in this set — he renders fine on the warrior-golden path.
# Substring scan (case-insensitive) catches both primary and secondary roles,
# e.g. "Karna and Krishna two-shot" should still trigger divine mode because
# Krishna's blue is more sensitive to override than Karna's bronze.
_DIVINE_NON_GOLDEN_CHARACTERS = {"Krishna", "Hanuman"}


# Load character reference descriptions once at import time
_CHAR_FILE = os.path.join(os.path.dirname(__file__), "..", "assets", "characters.json")
try:
    with open(_CHAR_FILE, encoding="utf-8") as _f:
        _CHARACTERS: dict = {
            k: v for k, v in json.load(_f).items() if not k.startswith("_")
        }
except Exception:
    _CHARACTERS = {}


def _primary_character(prompt: str) -> str:
    """
    Returns the FIRST known character name found in the prompt (case-insensitive),
    or empty string if none found. Used to derive a stable per-character seed
    so the same character renders with a similar face across all scenes of a
    video — eliminates the worst face-drift problem (Karna looking like four
    different actors across six scenes). The "first" character takes priority
    because image prompts typically lead with the scene's hero ("Karna sits...",
    "Indra appears..."), and downstream characters are usually secondary.

    Searches both _CHARACTERS (15 heroes with full visual descriptions) AND
    _KNOWN_NAMES (the broader Mahabharat character list including Indra,
    Surya, etc.). Seed-stability needs only the *name*, not a visual
    description, so secondary characters still get consistent faces across
    scenes.
    """
    prompt_lower = prompt.lower()
    earliest_pos = len(prompt_lower) + 1
    earliest_name = ""
    # Combined candidate pool — dedupe via set, preserve original casing
    candidates = set(_CHARACTERS.keys()) | set(_KNOWN_NAMES)
    for name in candidates:
        pos = prompt_lower.find(name.lower())
        if 0 <= pos < earliest_pos:
            earliest_pos = pos
            earliest_name = name
    return earliest_name


def _char_stable_seed(character: str, shot_index: int) -> int:
    """
    Deterministic seed derived from character name + shot index. Same character
    → same seed across all scenes → FLUX-schnell renders a visually similar
    face. The shot_index offset gives each angle a *related-but-not-identical*
    seed so the wide / medium / close-up shots don't look like the same crop
    of one image. Empty character falls back to scene-position seeding.
    """
    if not character:
        return 0
    h = 0
    for ch in character:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h + shot_index * 17) % 99991  # prime modulus for spread


def _inject_characters(prompt: str) -> str:
    """
    Scans the prompt for known Mahabharata character names and appends
    their detailed visual description so every image is visually consistent.
    """
    if not _CHARACTERS:
        return prompt
    injected = []
    prompt_lower = prompt.lower()
    for name, data in _CHARACTERS.items():
        if name.lower() in prompt_lower:
            # 250 char cap — enough for skin/eyes/clothing/jewelry/posture
            # without drowning the scene-specific prompt. The earlier 120 cap
            # produced one-line descriptors that the strong style suffix
            # routinely overpowered.
            visual = data.get("visual", "")[:250]
            injected.append(visual)
    if injected:
        return prompt + ". CHARACTER DETAILS — " + "; ".join(injected)
    return prompt


# Known Mahabharata characters to watch for in scripts
_KNOWN_NAMES = [
    "Ashwatthama", "Nakula", "Sahadeva", "Bhima", "Bheema",
    "Jayadratha", "Gandhari", "Madri", "Subhadra", "Uttara",
    "Ghatotkacha", "Hidimba", "Jarasandha", "Shishupala",
    "Sanjaya", "Kripa", "Kritavarma", "Vikarna", "Dushasana",
    "Satyavati", "Parashurama", "Narada", "Panchali",
    "Yuyutsu", "Virata", "Drupada", "Dhrishtadyumna",
    "Shikhandi", "Amba", "Ambika", "Ambalika", "Pandu",
    "Chitrangada", "Urvashi", "Menaka", "Indra", "Surya",
]


def update_characters(script_data: dict) -> list:
    """
    Scans the generated script for Mahabharata characters not yet in
    characters.json, generates visual descriptions via Gemini, and
    saves them back to the file. Returns list of newly added names.
    """
    if not script_data or "scenes" not in script_data:
        return []

    all_text = " ".join(
        scene.get("image_prompt", "") + " " + scene.get("narration", "")
        for scene in script_data["scenes"]
    ).lower()

    existing_lower = {k.lower() for k in _CHARACTERS}
    new_names = [
        name for name in _KNOWN_NAMES
        if name.lower() in all_text and name.lower() not in existing_lower
    ]

    if not new_names:
        return []

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return []

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = (
            f"Generate visual descriptions for these Mahabharata characters "
            f"for AI image generation: {new_names}\n\n"
            "For each character return a JSON object:\n"
            '{"CharacterName": {"visual": "specific physical appearance under 120 chars", '
            '"colors": "primary color palette"}}\n'
            "Be specific: clothing, weapons, jewelry, skin tone, hair style. "
            "Return valid JSON only, no markdown."
        )

        resp = model.generate_content(prompt)
        text = re.sub(r"^```(?:json)?\s*", "", resp.text.strip())
        text = re.sub(r"\s*```$", "", text)
        new_data = json.loads(text)

        with open(_CHAR_FILE, encoding="utf-8") as f:
            existing = json.load(f)

        added = []
        for name, data in new_data.items():
            if name not in existing:
                existing[name] = data
                _CHARACTERS[name] = data
                added.append(name)

        if added:
            with open(_CHAR_FILE, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            print(f"    [OK] New characters added to file: {added}")

        return added

    except Exception as e:
        print(f"    [!] Character auto-update skipped: {e}")
        return []


def _build_full_prompt(prompt: str, mood: str = "", style_suffix: str = STYLE_SUFFIX) -> str:
    mood_prefix = f"{mood}, " if mood else ""
    return f"{mood_prefix}{prompt}, {style_suffix}"


# ── Provider cascade ─────────────────────────────────────────────────────────
# Order: Hugging Face FLUX-schnell -> Cloudflare FLUX-schnell -> Pollinations.
# Each provider returns raw image bytes or raises. First success wins.
# Free-tier only — no paid keys involved.

_HF_MODEL = "black-forest-labs/FLUX.1-schnell"
# HF migrated from api-inference.huggingface.co to the inference router.
_HF_URL   = f"https://router.huggingface.co/hf-inference/models/{_HF_MODEL}"
_MIN_BYTES = 5000   # below this, treat response as a corrupted/empty image


def _ensure_dims(img_bytes: bytes, width: int, height: int) -> bytes:
    """If the returned image isn't the requested size, resize via Pillow."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        if img.size == (width, height):
            return img_bytes
        # Cover-fit: fill target then center-crop to avoid letterboxing
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        new_w, new_h = int(src_w * scale + 0.5), int(src_h * scale + 0.5)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - width)  // 2
        top  = (new_h - height) // 2
        img = img.crop((left, top, left + width, top + height))
        if img.mode != "RGB":
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=92)
        return out.getvalue()
    except Exception:
        return img_bytes


def _gen_hf(prompt: str, seed: int, width: int, height: int) -> bytes:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/jpeg",
        "x-wait-for-model": "true",
    }
    body = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
            "num_inference_steps": 4,
            "seed": seed,
            "negative_prompt": _NEGATIVE,
        },
    }
    resp = requests.post(_HF_URL, headers=headers, json=body, timeout=60)
    if resp.status_code != 200 or len(resp.content) < _MIN_BYTES:
        raise RuntimeError(f"hf status={resp.status_code} bytes={len(resp.content)}")
    ctype = resp.headers.get("content-type", "")
    if not ctype.startswith("image/"):
        raise RuntimeError(f"hf non-image response: {ctype}")
    return _ensure_dims(resp.content, width, height)


def _cloudflare_accounts() -> list[tuple[str, str, str]]:
    """
    Return all configured Cloudflare (label, account_id, api_token) triples
    in cascade order. Primary first, then numbered fallbacks _2, _3, ... .
    Empty / unset entries are skipped. Same shape as _gemini_keys() in
    script_generator.py — when the primary account exhausts its 10k-neuron/day
    free quota with a 429 (or similar quota error), the cascade walks to the
    next account immediately so production never falls to Pollinations
    (which produces visibly muddier output) while quota is available on
    another account.
    """
    out: list[tuple[str, str, str]] = []
    pid = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    ptk = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if pid and ptk:
        out.append(("cf-primary", pid, ptk))
    for n in range(2, 6):
        aid = os.environ.get(f"CLOUDFLARE_ACCOUNT_ID_{n}", "").strip()
        atk = os.environ.get(f"CLOUDFLARE_API_TOKEN_{n}", "").strip()
        if aid and atk and (aid, atk) not in [(a, t) for _, a, t in out]:
            out.append((f"cf-acct-{n}", aid, atk))
    return out


def _cf_is_quota_error(status: int, body_text: str) -> bool:
    """
    Distinguish quota exhaustion (try next account) from transient errors
    (try same account on next pipeline retry). Cloudflare returns 429 for
    rate-limit AND for daily-neuron-cap; both warrant fast-fail to next
    account. 401/403 = bad token (try next account). 5xx = transient (still
    try next account since the alternative is Pollinations).
    """
    if status in (429, 401, 403):
        return True
    low = body_text.lower()
    if "neuron" in low and ("limit" in low or "quota" in low or "exceeded" in low):
        return True
    if "rate" in low and "limit" in low:
        return True
    return False


def _gen_cloudflare(prompt: str, seed: int, width: int, height: int) -> bytes:
    """
    Multi-account FLUX-schnell call. Walks every configured CF account in
    order; on quota/auth errors fast-fails to next account, on the LAST
    account also re-raises any 5xx so the outer cascade can fall to
    Pollinations. steps=8 is the free-tier max for flux-schnell — 4 is
    faster but produces more anatomy errors.
    """
    accounts = _cloudflare_accounts()
    if not accounts:
        raise RuntimeError("no CLOUDFLARE_ACCOUNT_ID/API_TOKEN pairs configured")

    body = {"prompt": prompt, "steps": 8, "seed": seed, "negative_prompt": _NEGATIVE}
    last_err = "no accounts attempted"
    for label, account_id, token in accounts:
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/ai/run/@cf/black-forest-labs/flux-1-schnell"
        )
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
        except Exception as net_err:
            last_err = f"{label} network: {net_err}"
            continue

        if resp.status_code != 200:
            last_err = f"{label} status={resp.status_code} body={resp.text[:120]}"
            if _cf_is_quota_error(resp.status_code, resp.text):
                # Hit quota / auth — move to next account immediately.
                continue
            # Non-quota non-200 (rare). Still try next account — better than
            # giving up on Cloudflare entirely and falling to Pollinations.
            continue

        try:
            data = resp.json()
        except Exception as je:
            last_err = f"{label} bad json: {je}"
            continue
        if not data.get("success"):
            errs = data.get("errors") or []
            err_str = str(errs).lower()
            last_err = f"{label} success=false: {errs}"
            if any(k in err_str for k in ("quota", "neuron", "rate", "limit", "exceeded")):
                continue
            continue
        img_b64 = data.get("result", {}).get("image", "")
        if not img_b64:
            last_err = f"{label} missing result.image"
            continue
        raw = base64.b64decode(img_b64)
        if len(raw) < _MIN_BYTES:
            last_err = f"{label} bytes={len(raw)}"
            continue
        return _ensure_dims(raw, width, height)

    raise RuntimeError(f"cloudflare cascade exhausted: {last_err}")


def _gen_pollinations(prompt: str, seed: int, width: int, height: int) -> bytes:
    encoded  = quote(prompt)
    negative = quote(_NEGATIVE)
    # model=flux: bare FLUX-schnell, no postprocessing filter applied.
    # Was model=flux-realism — that variant adds a "realism" LoRA + heavy
    # warm/saturated filter on top of FLUX-schnell, which is what gave the
    # earlier Pollinations-rendered scenes the muddy plastic-CGI look the
    # user wanted to move away from (Anger's Fire register). Bare flux
    # produces output closer to HF FLUX-schnell — sharper, cleaner, no
    # forced color wash. Switch is reversible via env or one-line revert.
    pollinations_model = os.environ.get("POLLINATIONS_MODEL", "flux").strip() or "flux"
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&seed={seed}"
        f"&model={pollinations_model}&nologo=true&enhance=true&negative={negative}"
    )
    resp = requests.get(url, timeout=45)
    if resp.status_code != 200 or len(resp.content) < _MIN_BYTES:
        raise RuntimeError(f"pollinations status={resp.status_code} bytes={len(resp.content)}")
    return resp.content


def _correct_warm_cast(img_bytes: bytes, provider: str) -> bytes:
    """
    Neutralize the magenta/orange warm wash that Cloudflare FLUX-schnell and
    Pollinations consistently produce on this pipeline. Prompt-based anti-
    cast anchors (`no magenta cast`, `no warm filter` in _NEGATIVE) have
    repeatedly failed across multiple test runs — the bias is at the pixel
    level on these distilled FLUX variants, not at the prompt level.

    Three-filter ffmpeg pipeline applied via stdin/stdout pipes:
      - colortemperature mix=0.5 toward 5200K — cools the warm wash
      - eq saturation=0.88 gamma=1.02 — desat slightly + lift midtones
      - unsharp 5:5:0.4 — compensate for blurry backgrounds

    HF FLUX-schnell output is bypass-skipped — when HF quota becomes
    available again (or with HF Pro), its renders don't have the cast and
    we don't want to double-correct.

    Intensity is gated by IMAGE_COLOR_CORRECT_INTENSITY env var
    (default 1.0; set 0.0 to disable entirely without code change).

    Any ffmpeg failure returns the original bytes unchanged — never crashes
    the pipeline.
    """
    if "hf-flux" in provider.lower():
        return img_bytes

    try:
        intensity = float(os.environ.get("IMAGE_COLOR_CORRECT_INTENSITY", "1.0"))
    except ValueError:
        intensity = 1.0
    if intensity <= 0:
        return img_bytes

    # Scale the three filter strengths by intensity. At intensity=1.0 these
    # match the values that produced visible improvement in local A/B testing.
    temp_mix = max(0.0, min(1.0, 0.5 * intensity))
    sat      = 1.0 - (0.12 * intensity)
    gamma    = 1.0 + (0.02 * intensity)
    sharpen  = 0.4 * intensity

    vf = (
        f"colortemperature=temperature=5200:mix={temp_mix:.2f},"
        f"eq=saturation={sat:.2f}:gamma={gamma:.2f},"
        f"unsharp=5:5:{sharpen:.2f}"
    )

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", "pipe:0",
                "-vf", vf,
                "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2",
                "pipe:1",
            ],
            input=img_bytes,
            capture_output=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout and len(result.stdout) > _MIN_BYTES:
            return result.stdout
        # ffmpeg failed (non-zero exit or tiny output) — fall through, keep original
        return img_bytes
    except Exception:
        # subprocess timeout / OS error / etc. — never crash, just skip correction
        return img_bytes


def generate_image_bytes(prompt: str, seed: int, width: int, height: int, mood: str = "", style_suffix: str = STYLE_SUFFIX) -> tuple[bytes, str]:
    """
    Tries Cloudflare (multi-account cascade) -> HF -> Pollinations until one
    returns a usable image. Returns (image_bytes, provider_name).

    Order rationale (2026-05-14):
      1. Cloudflare first — produces the best visible quality on our prompts
         (the v3/v4 smoke-test images that the user approved were all CF).
         With 2-3 configured accounts, the daily 10k-neuron quota stretches
         to 4-6 clean videos/day.
      2. HF FLUX-schnell — backup. Frequently 429s on free tier; FLUX.1-dev
         was deprecated entirely 2026-05-14. Kept in the cascade for any
         account that still has HF Pro / unused quota.
      3. Pollinations — last resort only. Free-unlimited but produces visibly
         muddier output (the v5_pollinations smoke test the user rejected
         was Pollinations-only).

    Post-processing pipeline applied to every successful render:
      - _correct_warm_cast — neutralizes magenta/orange wash (Cloudflare/
        Pollinations only; HF FLUX-schnell bypassed since its output has
        no cast).

    Divine-character override (added 2026-05-14):
      When the caller passed STYLE_SUFFIX_MORTAL (the default for Mahabharata
      + Krishna series) AND the prompt mentions a divine non-golden-skin
      character (Krishna, Hanuman), swap to STYLE_SUFFIX_DIVINE. This stops
      the global golden-bronze skin anchor from fighting Krishna's canonical
      indigo-blue / Hanuman's red-gold. WhatIf suffixes are left alone.
    """
    if style_suffix is STYLE_SUFFIX_MORTAL:
        prompt_lower = prompt.lower()
        if any(name.lower() in prompt_lower for name in _DIVINE_NON_GOLDEN_CHARACTERS):
            style_suffix = STYLE_SUFFIX_DIVINE
    full_prompt = _build_full_prompt(prompt, mood, style_suffix=style_suffix)
    providers = [
        ("cloudflare-flux-schnell", lambda: _gen_cloudflare(full_prompt, seed, width, height)),
        ("hf-flux-schnell",         lambda: _gen_hf(full_prompt, seed, width, height)),
        ("pollinations-flux-realism", lambda: _gen_pollinations(full_prompt, seed, width, height)),
    ]
    last_err = None
    for name, fn in providers:
        try:
            data = fn()
            data = _correct_warm_cast(data, name)
            return data, name
        except Exception as e:
            last_err = f"{name}: {e}"
            continue
    raise RuntimeError(f"all image providers failed; last={last_err}")


def generate_images(scenes: list, single_shot: bool = False, series: str = "mahabharata", visual_style: str = "", ck=None) -> list:
    """
    Generates portrait (768x1344) images per scene.

    Default: 3 shots per scene (wide / medium / closeup) for the static-image
    Ken Burns pipeline.

    With single_shot=True: only the wide-establishing shot is generated. Used
    when the AI-video pipeline is active — I2V models only need a single
    first-frame image, so generating 3 wastes time and Pollinations quota.

    series + visual_style select the style suffix and skip Mahabharata-specific
    character injection when generating WhatIf imagery.

    PARTIAL RESUME — if `ck` (CheckpointStore) is provided, this function:
      1. Reads `visuals_partial.json` from the cache on entry. Any scene
         already listed there has its cached shot paths loaded and the
         generation step is skipped for that scene — the run resumes
         mid-batch instead of regenerating everything from scratch.
      2. After each new scene's shots complete (success OR placeholder
         fallback), the shots are immediately saved to the cache and
         `visuals_partial.json` is updated atomically. A cancellation
         mid-batch (e.g. 29-min GHA cap) preserves all work up to the
         last completed scene.

    Returns list[list[str]] — outer index = scene, inner index = shot.
    When `ck` is provided returned paths point into the cache directory;
    otherwise they point into temp/images.
    """
    os.makedirs("temp/images", exist_ok=True)
    scene_groups = []

    angles = _SHOT_ANGLES[:1] if single_shot else _SHOT_ANGLES
    style_suffix = _resolve_style_suffix(series, visual_style)

    # ── Partial-resume bootstrap ──────────────────────────────────────
    # visuals_partial.json shape: {"<scene_idx>": ["<abs_cache_path_shot0>", ...]}
    # Only honored when the cached files still exist on disk (defensive — a
    # botched artifact restore could leave a manifest pointing at missing
    # files; in that case we regenerate from scratch for safety).
    partial_key = "visuals_partial.json"
    partial: dict = {}
    if ck is not None and ck.has(partial_key):
        try:
            raw = ck.load_json(partial_key)
            for k, v in (raw or {}).items():
                if not isinstance(v, list):
                    continue
                if all(isinstance(p, str) and os.path.exists(p) for p in v) and v:
                    partial[int(k)] = v
            if partial:
                print(f"    [resume] Partial visuals manifest: {len(partial)} of "
                      f"{len(scenes)} scenes already cached — will skip those")
        except Exception as _e:
            print(f"    [resume] Could not load partial manifest: {_e} — regenerating from scratch")
            partial = {}

    for i, scene in enumerate(scenes):
        # ── Resume path ───────────────────────────────────────────────
        if i in partial:
            shot_paths = list(partial[i])
            scene_groups.append(shot_paths)
            print(f"    [resume] Scene {i+1}/{len(scenes)} loaded from partial checkpoint — {len(shot_paths)} shots")
            continue

        # ── Static-asset path (2026-05-14) ────────────────────────────
        # Subscribe-outro scenes carry an `image_path` field pointing at a
        # hand-picked asset in assets/outro/. Skip FLUX entirely and copy
        # the asset to the per-scene output path. Guarantees the channel
        # name overlay / outro composition is exactly what was approved —
        # no FLUX gambling on text rendering ("Vyasa AI" -> "VyssA" garble
        # observed in production on 2026-05-14).
        static_path = scene.get("image_path", "")
        if static_path and os.path.exists(static_path):
            output_path = f"temp/images/scene_{i:02d}_shot_00.jpg"
            try:
                import shutil
                shutil.copy2(static_path, output_path)
                shot_paths = [output_path]
                if ck is not None:
                    try:
                        cached = ck.save_file(f"visuals/scene_{i:02d}_shot_00.jpg", output_path)
                        shot_paths = [cached]
                        current = ck.load_json(partial_key) if ck.has(partial_key) else {}
                        current[str(i)] = shot_paths
                        ck.save_json(partial_key, current)
                    except Exception as _e:
                        print(f"    [warn] Could not checkpoint static outro asset: {_e}")
                scene_groups.append(shot_paths)
                print(f"    [static] Scene {i+1}/{len(scenes)} — using outro asset {static_path}")
                continue
            except Exception as _e:
                # Fall through to regular generation as a defensive backup
                print(f"    [warn] Static asset copy failed ({_e}) — falling through to FLUX")

        shot_paths = []
        mood = scene.get("mood", "")

        # Hook scene gets a dramatic close-up first frame in single-shot mode.
        # On Shorts the first frame is what stops the swipe; a face mid-emotion
        # outperforms a wide establishing shot for first-frame retention.
        scene_angles = angles
        if single_shot and i == 0:
            # Hook frame — medium close-up keeps emotional punch but gives
            # FLUX-schnell room to render eyes (extreme close-ups produced
            # dead-eye pupils in the 2026-05-14 Karna-arc test).
            scene_angles = ["MEDIUM CLOSE-UP on face mid-emotion (head and shoulders visible), "]

        for j, angle_prefix in enumerate(scene_angles):
            output_path = f"temp/images/scene_{i:02d}_shot_{j:02d}.jpg"
            # Character injection is Mahabharata-specific (Krishna/Arjuna/etc.
            # visual descriptors); skip it for WhatIf science content.
            raw_prompt = scene["image_prompt"]
            base_prompt = raw_prompt if series == "whatif" else _inject_characters(raw_prompt)
            prompt = f"{angle_prefix}{base_prompt}"
            # Stable per-character seed: same hero across scenes → similar
            # face. Falls back to scene-position seed when no known character
            # is mentioned (WhatIf scenes, environment-only shots).
            hero = "" if series == "whatif" else _primary_character(raw_prompt)
            seed = _char_stable_seed(hero, j) if hero else (i * 137 + j * 31)

            success = False
            for attempt in range(3):
                try:
                    img_bytes, provider = generate_image_bytes(
                        prompt, seed=seed, width=768, height=1344, mood=mood,
                        style_suffix=style_suffix,
                    )
                    with open(output_path, "wb") as f:
                        f.write(img_bytes)
                    shot_paths.append(output_path)
                    print(f"    [OK] Scene {i+1} shot {j+1}/{len(scene_angles)} via {provider}")
                    success = True
                    break
                except Exception as e:
                    print(f"    [!] Scene {i+1} shot {j+1} attempt {attempt+1}: {e}")

                wait = (attempt + 1) * 3
                print(f"    Waiting {wait}s...")
                time.sleep(wait)

            if not success:
                _create_placeholder(output_path, i * 3 + j, series=series)
                shot_paths.append(output_path)
                print(f"    [~] Placeholder (outro-asset fallback) for scene {i+1} shot {j+1}")

            time.sleep(1)

        # ── Per-scene checkpoint ──────────────────────────────────────
        # Save the just-finished scene to the cache + update partial manifest
        # so the next workflow attempt resumes from here on cancellation.
        if ck is not None:
            cached_paths = []
            for j_idx, temp_path in enumerate(shot_paths):
                ext = os.path.splitext(temp_path)[1] or ".jpg"
                cache_name = f"visuals/scene_{i:02d}_shot_{j_idx:02d}{ext}"
                try:
                    cached_paths.append(ck.save_file(cache_name, temp_path))
                except Exception as _e:
                    print(f"    [warn] Could not checkpoint scene {i+1} shot {j_idx+1}: {_e}")
                    cached_paths.append(temp_path)
            # Replace returned paths with cache paths so downstream uses the cache copy
            shot_paths = cached_paths
            try:
                # Atomic load → update → save (save_json itself is atomic)
                current = ck.load_json(partial_key) if ck.has(partial_key) else {}
                current[str(i)] = cached_paths
                ck.save_json(partial_key, current)
            except Exception as _e:
                print(f"    [warn] Could not update partial manifest for scene {i+1}: {_e}")

        scene_groups.append(shot_paths)
        print(f"    [OK] Scene {i+1}/{len(scenes)} complete — {len(shot_paths)} shots")

    return scene_groups


def _overlay_thumbnail_text(thumb_path: str, overlay_text: str) -> bool:
    """
    Composite a bold Hindi text card onto an existing FLUX-rendered thumbnail.
    Used to add the search-stopping "shock phrase" overlay (e.g. "कर्ण का सच",
    "भीष्म प्रतिज्ञा") that successful Hindi mythology Shorts channels use to
    drive CTR on browse/search/suggested surfaces.

    Added 2026-05-15 after channel analytics showed thumbnails were
    text-free FLUX art — competing channels overlay 2-3 Hindi words for the
    eye-catch. Renders via pipeline.text_renderer (HarfBuzz for Devanagari
    shaping; FLUX can't spell so it's done as a separate raster pass).

    Position: bottom 1/3 of the 1280×720 thumbnail, centered horizontally,
    yellow fill + black outline + soft shadow (the canonical "thumbnail
    text" look). Returns True on success; on any failure the original
    thumbnail is left untouched (defensive — never break the pipeline).
    """
    if not overlay_text or not overlay_text.strip():
        return False
    try:
        from PIL import Image
        from pipeline.text_renderer import render_text_card
        from pipeline.subtitle_generator import FONT_PATH
    except Exception as e:
        print(f"    [!] Thumbnail text overlay deps missing: {e}")
        return False
    if not os.path.exists(thumb_path):
        return False
    try:
        # Auto-size text to fit ~85% of thumbnail width. 1280px wide, target
        # ~1080px text width. For a 4-word Hindi phrase that lands at ~70-90
        # pixels per glyph, font_size ~110-130 works. Start at 130, downscale
        # if too wide.
        text = overlay_text.strip()
        base = Image.open(thumb_path).convert("RGBA")
        W, H = base.size  # 1280, 720
        max_text_w = int(W * 0.88)

        font_size = 130
        for _ in range(5):
            card = render_text_card(
                text, FONT_PATH, font_size=font_size,
                fill=(255, 230, 0, 255),
                outline=(0, 0, 0, 255),
                outline_px=7,
                shadow=(0, 0, 0, 180),
                shadow_offset=(4, 4),
            )
            if card.width <= max_text_w:
                break
            # Too wide; shrink
            font_size = int(font_size * max_text_w / max(card.width, 1))
        if card.width > max_text_w:
            # Last resort: brute-shrink to fit
            ratio = max_text_w / card.width
            card = card.resize(
                (int(card.width * ratio), int(card.height * ratio)),
                Image.LANCZOS,
            )

        # Place at bottom 1/3: y_center = H * 0.72
        x = (W - card.width) // 2
        y = int(H * 0.72) - card.height // 2
        base.paste(card, (x, y), card)
        base.convert("RGB").save(thumb_path, "JPEG", quality=92)
        print(f"    [OK] Thumbnail text overlay: {text!r}")
        return True
    except Exception as e:
        print(f"    [!] Thumbnail text overlay failed: {e}")
        return False


def generate_thumbnail(
    thumbnail_prompt: str,
    output_path: str = "output/thumbnail.jpg",
    series: str = "mahabharata",
    visual_style: str = "",
    overlay_text: str = "",
) -> str:
    """
    Generates a 1280x720 thumbnail (YouTube native size — landscape).

    For WhatIf the thumbnail is the single most clicked-on asset and Pollinations
    output at small sizes tends to be muddy with weak focal hierarchy. We append
    a thumbnail-specific composition rider (centered subject, single high-contrast
    focal point, dark backdrop, no small text) and vary the seed each retry so a
    bad first composition isn't simply repeated four times.

    `overlay_text` (added 2026-05-15): when provided, a bold Hindi text card
    is composited onto the bottom 1/3 of the thumbnail after FLUX renders.
    Used to add the search-stopping shock phrase ("कर्ण का सच", "भीष्म प्रतिज्ञा")
    competing mythology channels use to drive CTR. Empty = no overlay.
    """
    os.makedirs("output", exist_ok=True)
    style_suffix = _resolve_style_suffix(series, visual_style)

    # Thumbnail composition rider — appended to the style suffix only for the
    # landscape thumbnail call, not the scene images. Pushes the model toward
    # phone-feed-readable framing.
    if series == "whatif":
        style_suffix = (
            style_suffix
            + ", thumbnail composition — single centered hero subject filling 60% "
              "of the frame, dramatic single light source, dark moody backdrop "
              "with strong contrast, no small text, no UI elements, no logos, "
              "phone-feed legible at thumbnail size"
        )

    # Vary seeds across attempts so a poor first composition doesn't repeat.
    seeds = [9999, 4242, 8137, 1729]
    for attempt in range(len(seeds)):
        try:
            img_bytes, provider = generate_image_bytes(
                thumbnail_prompt, seed=seeds[attempt], width=1280, height=720,
                style_suffix=style_suffix,
            )
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            print(f"    [OK] Thumbnail generated via {provider} (seed {seeds[attempt]})")
            # Post-render text overlay (Hindi shock phrase via HarfBuzz).
            # Defensive — never crashes the thumbnail step on overlay failure.
            if overlay_text:
                _overlay_thumbnail_text(output_path, overlay_text)
            return output_path
        except Exception as e:
            print(f"    [!] Thumbnail attempt {attempt+1}: {e}")

        wait = (attempt + 1) * 3
        time.sleep(wait)

    print(f"    [ERROR] Thumbnail generation failed after {len(seeds)} attempts")
    return ""


_OUTRO_FALLBACK_BY_SERIES = {
    "mahabharata": "assets/outro/mahabharata.jpg",
    "krishna":     "assets/outro/krishna.jpg",
    "whatif":      "assets/outro/whatif.jpg",
}


def _create_placeholder(output_path: str, index: int, series: str = "mahabharata"):
    """
    Fallback image used when ALL providers (HF + 3 CF accounts + Pollinations)
    fail for a given scene. Reuses the series' hand-picked outro asset so
    the viewer sees ACTUAL imagery instead of a solid color tile.

    2026-05-14 production check (Krishna "Uddhava" video) shipped 5 scenes
    of near-black solid-color tiles when Pollinations 429-stormed during a
    retry — the "video" was effectively just audio over a black screen.
    Reusing the outro asset means worst case = an "outro tableau loop"
    that's visibly mythological even if it's the same image repeated.

    Falls back to the old solid-color tile if the asset is missing on disk.
    """
    import subprocess
    import shutil

    asset = _OUTRO_FALLBACK_BY_SERIES.get(series, "")
    if asset and os.path.exists(asset):
        try:
            shutil.copy2(asset, output_path)
            return
        except Exception:
            pass  # fall through to ffmpeg solid color

    colors = ["#1a0a2e", "#0d1b2a", "#1b2838", "#2d1b69", "#0f0c29"]
    color = colors[index % len(colors)]
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color={color}:size=768x1344:duration=1",
            "-vframes", "1", output_path,
        ],
        capture_output=True,
    )
