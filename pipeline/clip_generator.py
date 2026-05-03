"""
Clip Generator — LTX-2 Video via HuggingFace Spaces (Imosu/image_audio_to_video_NSFW)
=======================================================================================
Generates real video clips from scene images using LTX-2 Distilled I2V model.

.ENV KEYS:
    HF_TOKEN    = your free HuggingFace token
    HF_SPACE    = Imosu/image_audio_to_video_NSFW
    HF_API_NAME = /generate_video

FALLBACK: If all clips fail, raises RuntimeError → main.py falls back to static images.
"""

import os
import asyncio
import random
import shutil

from pipeline.image_generator import generate_images


HF_SPACE    = os.environ.get("HF_SPACE",    "").strip()
HF_TOKEN    = os.environ.get("HF_TOKEN",    "").strip() or None
HF_API_NAME = os.environ.get("HF_API_NAME", "/generate_video").strip()

MAX_RETRIES  = 2
RETRY_WAIT_S = 20

# Map Ken Burns motions to LTX-2 camera LoRA options
_CAMERA_LORA = {
    "zoom_in":       "Zoom In",
    "zoom_out":      "Zoom Out",
    "pan_left":      "Slide Left",
    "pan_right":     "Slide Right",
    "pan_up":        "Slide Up",
    "pan_down":      "Slide Down",
    "zoom_in_left":  "Zoom In",
    "zoom_in_right": "Zoom In",
}
_CAMERA_MOTIONS = list(_CAMERA_LORA.keys())


def _build_video_prompt(scene: dict) -> str:
    base = (
        scene.get("video_prompt")
        or scene.get("image_prompt", "")[:150]
        or "dramatic cinematic scene"
    )
    return (
        f"{base}, cinematic motion, smooth animation, "
        "epic lighting, shallow depth of field, high quality"
    )


async def _generate_one_clip(
    image_path: str,
    prompt: str,
    duration: float,
    clip_index: int,
    motion: str = None,
) -> str | None:
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print("    gradio_client not installed — run: pip install gradio_client")
        return None

    camera_lora = _CAMERA_LORA.get(motion, "No LoRA") if motion else random.choice(
        ["Zoom In", "Zoom Out", "Slide Left", "Slide Right"]
    )
    clip_duration = min(max(duration, 2.0), 3.0)  # stay under ZeroGPU 180s limit

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"        [{clip_index}] Attempt {attempt}/{MAX_RETRIES} — camera: {camera_lora}")

            client = await asyncio.to_thread(Client, HF_SPACE, token=HF_TOKEN)

            result = await asyncio.to_thread(
                client.predict,
                first_frame=handle_file(image_path),
                end_frame=None,
                prompt=prompt,
                duration=clip_duration,
                input_video=None,
                generation_mode="Image-to-Video",
                enhance_prompt=False,
                seed=random.randint(0, 9999),
                randomize_seed=True,
                height=512,   # keep small — ZeroGPU has 180s limit per request
                width=288,    # 9:16 portrait at low res; assembler upscales to 1080x1920
                camera_lora=camera_lora,
                custom_lora="None",
                audio_path=None,
                api_name=HF_API_NAME,
            )

            # result is a file path string or dict
            video_path = None
            if isinstance(result, str) and os.path.exists(result):
                video_path = result
            elif isinstance(result, dict):
                for key in ("video", "output", "file", "path"):
                    c = result.get(key)
                    if c and os.path.exists(str(c)):
                        video_path = str(c)
                        break
            elif isinstance(result, (list, tuple)) and result:
                c = result[0]
                if os.path.exists(str(c)):
                    video_path = str(c)

            if video_path:
                print(f"        [{clip_index}] Clip generated OK")
                return video_path
            else:
                print(f"        [{clip_index}] Unexpected result format: {str(result)[:100]}")

        except Exception as e:
            err = str(e)
            print(f"        [{clip_index}] Attempt {attempt} failed: {err[:120]}")
            if attempt < MAX_RETRIES:
                print(f"        Retrying in {RETRY_WAIT_S}s...")
                await asyncio.sleep(RETRY_WAIT_S)

    return None


async def generate_video_clips(scenes: list) -> list:
    """
    Generates one video clip per scene using LTX-2 I2V on HuggingFace Spaces.
    Returns list of clip paths. Raises RuntimeError on full failure → triggers image fallback.
    """
    print(f"    HF Space : {HF_SPACE}")
    print(f"    Endpoint : {HF_API_NAME}")
    print(f"    Generating {len(scenes)} clip(s)...")

    # Generate Pollinations images as I2V reference frames
    print("\n    Step 3a — Generating reference images...")
    image_file_groups = generate_images(scenes)
    # Use the wide-shot (index 0) from each scene group as the I2V input
    image_files = [group[0] if isinstance(group, list) else group
                   for group in image_file_groups]

    os.makedirs("temp/clips", exist_ok=True)

    clip_paths = []
    failed     = 0
    motions    = random.sample(_CAMERA_MOTIONS, min(len(scenes), len(_CAMERA_MOTIONS)))

    for i, (scene, image_path) in enumerate(zip(scenes, image_files)):
        duration = float(scene.get("duration", 5.0))
        prompt   = _build_video_prompt(scene)
        motion   = motions[i % len(motions)]

        print(f"\n    Clip {i+1}/{len(scenes)}  ({duration:.1f}s, {motion})")
        print(f"        Prompt: {prompt[:80]}...")

        raw_path = await _generate_one_clip(image_path, prompt, duration, i + 1, motion)

        if raw_path:
            dest = f"temp/clips/raw_{i:02d}.mp4"
            shutil.copy2(raw_path, dest)
            clip_paths.append(dest)
        else:
            failed += 1
            print(f"        Clip {i+1} failed — will fall back to images")

    if failed == len(scenes):
        raise RuntimeError(
            "All HF Space clip generations failed. Falling back to static images."
        )
    if failed > 0:
        raise RuntimeError(
            f"{failed}/{len(scenes)} clips failed. Falling back to image pipeline."
        )

    print(f"\n    All {len(scenes)} clips generated via HF Spaces")
    return clip_paths


if __name__ == "__main__":
    print(f"Discovering API for Space: {HF_SPACE}\n")
    try:
        from gradio_client import Client
        Client(HF_SPACE, token=HF_TOKEN).view_api()
    except ImportError:
        print("Install: pip install gradio_client")
    except Exception as e:
        print(f"Error: {e}")
