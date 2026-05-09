import requests
import os
import re
import time
import json
import base64
import io
from urllib.parse import quote

# Mahabharata style suffix — illustrated mythology art (Amar Chitra Katha /
# modern anime-cinematic mythology), NOT photoreal. The illustrated path wins
# on three axes for short-form mythology content: (1) stylistic consistency
# tolerates AI inconsistencies that photoreal exposes as uncanny actor faces,
# (2) viewers associate this look with mythology storytelling (ACK shipped
# 100M+ copies in this exact style), (3) FLUX renders ornate Indian temple
# architecture far more reliably as illustrated art than as photoreal sets.
#
# WhatIf series uses a different per-style suffix from _WHATIF_STYLE_SUFFIXES
# based on the script's visual_style — this Mahabharata suffix does not apply.
STYLE_SUFFIX = (
    "highly detailed traditional Indian mythology illustration, "
    "Amar Chitra Katha style mixed with modern anime-cinematic mythology art, "
    "hand-painted cel-shaded look with crisp linework and soft painterly shading, "
    "ornate Hindu temple-palace architecture in the background — carved sandstone "
    "pillars, hanging brass oil lamps, lotus reliefs, stone deity carvings, "
    "patterned floor tiles, multiple planes of architectural depth, "
    "dramatic warm directional sunlight streaming through carved arches, "
    "jewel-toned saturated palette of gold, crimson, deep emerald, lapis blue, "
    "ornate gold jewelry and rich silk garments with visible embroidery, "
    "cinematic storybook composition, rule of thirds, hero character in sharp focus, "
    "expressive facial emotion, ancient India epic Mahabharata atmosphere, "
    "consistent character design, sharp focus throughout, no photographic look, "
    "no 3D render look, no plastic skin, no realistic photo"
)

# WhatIf series style suffixes — picked by `script["visual_style"]`. Mahabharata
# style does NOT apply to WhatIf scripts; the LLM picks the most suitable style
# per topic (e.g. dinosaurs -> nature-doc, black hole -> sci-fi-cinematic).
_WHATIF_STYLE_SUFFIXES = {
    "photoreal-3d": (
        "photorealistic 3D render, octane render quality, cinematic lighting, "
        "sharp 8K detail, scientific illustration accuracy, "
        "studio depth of field, volumetric atmosphere"
    ),
    "nature-doc": (
        "BBC nature documentary cinematography, golden hour lighting, "
        "telephoto lens compression, naturalistic colour palette, "
        "shallow depth of field, ultra-sharp 8K detail"
    ),
    "sci-fi-cinematic": (
        "Denis Villeneuve sci-fi aesthetic, Roger Deakins lighting, "
        "anamorphic widescreen feel, atmospheric haze, dramatic silhouettes, "
        "muted tonal palette, cinematic 8K detail"
    ),
    "illustrated": (
        "polished concept art digital illustration, ArtStation trending, "
        "painterly brush detail, vivid colour palette, dramatic composition, "
        "sharp readable subjects"
    ),
}


def _resolve_style_suffix(series: str, visual_style: str) -> str:
    """Return the right style suffix for the series + visual_style."""
    if series == "whatif":
        return _WHATIF_STYLE_SUFFIXES.get(visual_style, _WHATIF_STYLE_SUFFIXES["photoreal-3d"])
    return STYLE_SUFFIX

# Negative prompt — suppresses blurry/low-quality outputs and FLUX-schnell's
# known anatomy weaknesses (especially hands and fingers).
_NEGATIVE = (
    "blurry,blur,out of focus,low quality,pixelated,distorted,"
    "ugly,bad anatomy,watermark,text,logo,duplicate,deformed,"
    "extra fingers,six fingers,seven fingers,too many fingers,"
    "mutated hands,malformed hands,fused fingers,missing fingers,"
    "extra limbs,extra arms,malformed limbs,disfigured,"
    "asymmetric eyes,cross-eyed,bad proportions"
)

# 3 compositional angles per scene — gives genuine visual variety
_SHOT_ANGLES = [
    "",                     # base prompt — wide establishing shot
    "medium shot, ",        # mid-range character focus
    "dramatic close-up, ",  # emotional detail / facial expression
]

# Load character reference descriptions once at import time
_CHAR_FILE = os.path.join(os.path.dirname(__file__), "..", "assets", "characters.json")
try:
    with open(_CHAR_FILE, encoding="utf-8") as _f:
        _CHARACTERS: dict = {
            k: v for k, v in json.load(_f).items() if not k.startswith("_")
        }
except Exception:
    _CHARACTERS = {}


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
            visual = data.get("visual", "")[:120]
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


def _gen_cloudflare(prompt: str, seed: int, width: int, height: int) -> bytes:
    token = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if not (token and account_id):
        raise RuntimeError("CLOUDFLARE_API_TOKEN/ACCOUNT_ID not set")
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/@cf/black-forest-labs/flux-1-schnell"
    )
    headers = {"Authorization": f"Bearer {token}"}
    # steps=8 is the free-tier max for flux-schnell; 4 is faster but produces
    # more anatomy errors (extra fingers, malformed hands). 8 is the quality cap.
    body = {"prompt": prompt, "steps": 8, "seed": seed, "negative_prompt": _NEGATIVE}
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"cloudflare status={resp.status_code}")
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"cloudflare success=false: {data.get('errors')}")
    img_b64 = data.get("result", {}).get("image", "")
    if not img_b64:
        raise RuntimeError("cloudflare missing result.image")
    raw = base64.b64decode(img_b64)
    if len(raw) < _MIN_BYTES:
        raise RuntimeError(f"cloudflare bytes={len(raw)}")
    return _ensure_dims(raw, width, height)


def _gen_pollinations(prompt: str, seed: int, width: int, height: int) -> bytes:
    encoded  = quote(prompt)
    negative = quote(_NEGATIVE)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&seed={seed}"
        f"&model=flux-realism&nologo=true&enhance=true&negative={negative}"
    )
    resp = requests.get(url, timeout=45)
    if resp.status_code != 200 or len(resp.content) < _MIN_BYTES:
        raise RuntimeError(f"pollinations status={resp.status_code} bytes={len(resp.content)}")
    return resp.content


def generate_image_bytes(prompt: str, seed: int, width: int, height: int, mood: str = "", style_suffix: str = STYLE_SUFFIX) -> tuple[bytes, str]:
    """
    Tries HF -> Cloudflare -> Pollinations until one returns a usable image.
    Returns (image_bytes, provider_name). Raises only if all three fail.
    """
    full_prompt = _build_full_prompt(prompt, mood, style_suffix=style_suffix)
    providers = [
        ("hf-flux-schnell",         lambda: _gen_hf(full_prompt, seed, width, height)),
        ("cloudflare-flux-schnell", lambda: _gen_cloudflare(full_prompt, seed, width, height)),
        ("pollinations-flux-realism", lambda: _gen_pollinations(full_prompt, seed, width, height)),
    ]
    last_err = None
    for name, fn in providers:
        try:
            data = fn()
            return data, name
        except Exception as e:
            last_err = f"{name}: {e}"
            continue
    raise RuntimeError(f"all image providers failed; last={last_err}")


def generate_images(scenes: list, single_shot: bool = False, series: str = "mahabharata", visual_style: str = "") -> list:
    """
    Generates portrait (768x1344) images per scene.

    Default: 3 shots per scene (wide / medium / closeup) for the static-image
    Ken Burns pipeline.

    With single_shot=True: only the wide-establishing shot is generated. Used
    when the AI-video pipeline is active — I2V models only need a single
    first-frame image, so generating 3 wastes time and Pollinations quota.

    series + visual_style select the style suffix and skip Mahabharata-specific
    character injection when generating WhatIf imagery.

    Returns list[list[str]] — outer index = scene, inner index = shot.
    """
    os.makedirs("temp/images", exist_ok=True)
    scene_groups = []

    angles = _SHOT_ANGLES[:1] if single_shot else _SHOT_ANGLES
    style_suffix = _resolve_style_suffix(series, visual_style)

    for i, scene in enumerate(scenes):
        shot_paths = []
        mood = scene.get("mood", "")

        # Hook scene gets a dramatic close-up first frame in single-shot mode.
        # On Shorts the first frame is what stops the swipe; a face mid-emotion
        # outperforms a wide establishing shot for first-frame retention.
        scene_angles = angles
        if single_shot and i == 0:
            scene_angles = ["dramatic close-up on face mid-emotion, "]

        for j, angle_prefix in enumerate(scene_angles):
            output_path = f"temp/images/scene_{i:02d}_shot_{j:02d}.jpg"
            # Character injection is Mahabharata-specific (Krishna/Arjuna/etc.
            # visual descriptors); skip it for WhatIf science content.
            base_prompt = scene["image_prompt"] if series == "whatif" else _inject_characters(scene["image_prompt"])
            prompt = f"{angle_prefix}{base_prompt}"
            seed = i * 137 + j * 31

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
                _create_placeholder(output_path, i * 3 + j)
                shot_paths.append(output_path)
                print(f"    [~] Placeholder for scene {i+1} shot {j+1}")

            time.sleep(1)

        scene_groups.append(shot_paths)
        print(f"    [OK] Scene {i+1}/{len(scenes)} complete — {len(shot_paths)} shots")

    return scene_groups


def generate_thumbnail(thumbnail_prompt: str, output_path: str = "output/thumbnail.jpg", series: str = "mahabharata", visual_style: str = "") -> str:
    """Generates a 1280x720 thumbnail (YouTube native size — landscape)."""
    os.makedirs("output", exist_ok=True)
    style_suffix = _resolve_style_suffix(series, visual_style)

    for attempt in range(3):
        try:
            img_bytes, provider = generate_image_bytes(
                thumbnail_prompt, seed=9999, width=1280, height=720,
                style_suffix=style_suffix,
            )
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            print(f"    [OK] Thumbnail generated via {provider}")
            return output_path
        except Exception as e:
            print(f"    [!] Thumbnail attempt {attempt+1}: {e}")

        wait = (attempt + 1) * 3
        time.sleep(wait)

    print(f"    [ERROR] Thumbnail generation failed after 4 attempts")
    return ""


def _create_placeholder(output_path: str, index: int):
    """Creates a solid-colour 768x1344 placeholder image via FFmpeg."""
    import subprocess
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
