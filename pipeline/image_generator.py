import requests
import os
import re
import time
import json
import base64
import io
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
STYLE_SUFFIX = (
    "photorealistic cinematic film still, "
    # ── Photoreal cinema-camera anchors (pushes FLUX away from the "Anger's
    # Fire" plastic-CGI look toward Bhishma-style real cinematography) ──
    "shot on Arri Alexa LF with 50mm anamorphic lens, "
    "Kodak Vision3 5219 film stock, professional cinema-grade color grading, "
    "physically-based skin shader with visible pores and fine facial detail, "
    "natural sub-surface scattering on skin, realistic beard and hair texture, "
    # ── Period / aesthetic register (unchanged from prior iteration) ──
    "ancient India epic Mahabharata, "
    "Star Bharat Mahabharat live-action / Baahubali period-film aesthetic, "
    "real human faces with natural skin tones and accurate skin texture, "
    "balanced cinematic color grading — neutral whites, true skin colors, "
    "ultra-sharp 8K resolution, "
    "ornate Hindu temple-palace architecture in the background — carved sandstone "
    "pillars, hanging brass oil lamps, lotus reliefs, stone deity carvings, "
    "patterned floor tiles, multiple planes of architectural depth, "
    "directional cinematic lighting through carved arches, soft volumetric haze, "
    "rich jewel-toned palette of gold, crimson, deep emerald, lapis blue — "
    "but applied as accent colors over natural background tones, NOT as a "
    "global color wash across the whole frame, "
    "intricate gold jewelry detail and rich silk garments with visible embroidery, "
    "shallow depth of field, hero character in sharp focus, expressive facial emotion, "
    "cinematic composition, rule of thirds, inspired by Raja Ravi Varma paintings "
    "rendered as live-action film, consistent character design, sharp focus, "
    # ── Negative anchors — bans cartoonish + the Anger's Fire CGI-plastic look ──
    "no cartoon, no anime, no cel shading, no comic book illustration, "
    "no CGI plastic skin, no 3D-render plastic look, no video-game character render, "
    "no smooth airbrushed skin, no Pixar-style stylization, no over-rendered, "
    "no magenta cast, no pink cast, no purple skin, no over-saturated wash"
)

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

# Negative prompt — suppresses blurry/low-quality outputs, FLUX-schnell's
# known anatomy weaknesses (hands/fingers), the heavy magenta/pink wash that
# older STYLE_SUFFIX iterations produced, AND the CGI-plastic / video-game-
# character look that made the Anger's Fire video look cartoonish.
_NEGATIVE = (
    "blurry,blur,out of focus,low quality,pixelated,distorted,"
    "ugly,bad anatomy,watermark,text,logo,duplicate,deformed,"
    "extra fingers,six fingers,seven fingers,too many fingers,"
    "mutated hands,malformed hands,fused fingers,missing fingers,"
    "extra limbs,extra arms,malformed limbs,disfigured,"
    "asymmetric eyes,cross-eyed,bad proportions,"
    "magenta cast,pink cast,purple skin,over-saturated,"
    "color wash,monochrome filter,sepia overlay,"
    "cartoon,anime,cel shaded,illustration,drawing,"
    "cgi plastic skin,3d render plastic,video game character,"
    "smooth airbrushed skin,pixar style,unreal engine character,"
    "doll-like face,waxy skin,over-rendered"
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


def generate_thumbnail(thumbnail_prompt: str, output_path: str = "output/thumbnail.jpg", series: str = "mahabharata", visual_style: str = "") -> str:
    """
    Generates a 1280x720 thumbnail (YouTube native size — landscape).

    For WhatIf the thumbnail is the single most clicked-on asset and Pollinations
    output at small sizes tends to be muddy with weak focal hierarchy. We append
    a thumbnail-specific composition rider (centered subject, single high-contrast
    focal point, dark backdrop, no small text) and vary the seed each retry so a
    bad first composition isn't simply repeated four times.
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
            return output_path
        except Exception as e:
            print(f"    [!] Thumbnail attempt {attempt+1}: {e}")

        wait = (attempt + 1) * 3
        time.sleep(wait)

    print(f"    [ERROR] Thumbnail generation failed after {len(seeds)} attempts")
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
