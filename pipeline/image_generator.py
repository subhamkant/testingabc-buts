import requests
import os
import re
import time
import json
from urllib.parse import quote

# Style suffix — photorealistic FLUX renders produce sharp crisp details
STYLE_SUFFIX = (
    "hyper-detailed photorealistic digital art, ancient Indian epic Mahabharata, "
    "ultra-sharp 8K resolution, crystal-clear facial features, "
    "intricate gold jewelry textures clearly visible, rich silk fabric details, "
    "dramatic cinematic lighting with golden volumetric rays, "
    "jewel-toned palette of gold crimson lapis and emerald, "
    "inspired by Raja Ravi Varma paintings but photorealistic, "
    "sharp focus throughout, no blur, no motion blur"
)

# Negative prompt — suppresses blurry/low-quality outputs
_NEGATIVE = (
    "blurry,blur,out of focus,low quality,pixelated,distorted,"
    "ugly,bad anatomy,watermark,text,logo,duplicate,deformed"
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


def _build_url(prompt: str, seed: int, width: int = 768, height: int = 1344, mood: str = "") -> str:
    """
    Default 768×1344 — optimal 9:16 size for FLUX.
    Thumbnail overrides to 1280×720. Ken Burns upscales to 1080×1920.
    """
    mood_prefix = f"{mood}, " if mood else ""
    full_prompt = f"{mood_prefix}{prompt}, {STYLE_SUFFIX}"
    encoded  = quote(full_prompt)
    negative = quote(_NEGATIVE)
    return (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&seed={seed}"
        f"&model=flux-realism&nologo=true&enhance=true&negative={negative}"
    )


def generate_images(scenes: list) -> list:
    """
    Generates 3 portrait (1080x1920) images per scene:
      shot 0 — wide establishing
      shot 1 — medium
      shot 2 — dramatic close-up

    Returns list[list[str]] — outer index = scene, inner index = shot.
    """
    os.makedirs("temp/images", exist_ok=True)
    scene_groups = []

    for i, scene in enumerate(scenes):
        shot_paths = []
        mood = scene.get("mood", "")

        for j, angle_prefix in enumerate(_SHOT_ANGLES):
            output_path = f"temp/images/scene_{i:02d}_shot_{j:02d}.jpg"
            base_prompt = _inject_characters(scene["image_prompt"])
            prompt = f"{angle_prefix}{base_prompt}"
            url = _build_url(prompt, seed=i * 137 + j * 31, mood=mood)

            success = False
            for attempt in range(3):
                try:
                    resp = requests.get(url, timeout=45)
                    if resp.status_code == 200 and len(resp.content) > 5000:
                        with open(output_path, "wb") as f:
                            f.write(resp.content)
                        shot_paths.append(output_path)
                        print(f"    [OK] Scene {i+1} shot {j+1}/3")
                        success = True
                        break
                    else:
                        print(f"    [!] Scene {i+1} shot {j+1} attempt {attempt+1}: status {resp.status_code}")
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


def generate_thumbnail(thumbnail_prompt: str, output_path: str = "output/thumbnail.jpg") -> str:
    """Generates a 1280x720 thumbnail (YouTube native size — landscape)."""
    os.makedirs("output", exist_ok=True)
    url = _build_url(thumbnail_prompt, seed=9999, width=1280, height=720)

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=45)
            if resp.status_code == 200 and len(resp.content) > 5000:
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                print(f"    [OK] Thumbnail generated")
                return output_path
            else:
                print(f"    [!] Thumbnail attempt {attempt+1}: status {resp.status_code}")
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
