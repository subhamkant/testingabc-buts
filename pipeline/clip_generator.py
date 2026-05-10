"""
Clip Generator — Parallel AI video clip generation with provider cascade
=========================================================================
Generates one short portrait video clip per scene by trying providers in
this order, per scene, until one succeeds:

    1. fal.ai          (env: FAL_KEY,                FAL_MODEL)
    2. Replicate       (env: REPLICATE_API_TOKEN,    REPLICATE_MODEL)
    3. HuggingFace     (env: HF_SPACE, HF_TOKEN,     HF_API_NAME)

All scenes are processed CONCURRENTLY via asyncio.gather (gated by a
semaphore) so total wall time ≈ slowest single clip, not sum of clips.

If any scene cannot produce a clip on any provider, RuntimeError is raised —
main.py catches this and falls back to the static-image Ken Burns pipeline,
so we never ship a half-AI / half-static video.

ENV defaults aim at portrait 9:16 outputs (1080x1920 after upscale in
video_assembler.py). Free-tier providers vary in argument schemas; this file
keeps the per-provider mapping isolated so adding a fourth is simple.
"""

import os
import asyncio
import random
import shutil
import requests

from pipeline.image_generator import generate_images


# ── Config ────────────────────────────────────────────────────────────────────

FAL_KEY          = os.environ.get("FAL_KEY",          "").strip()
FAL_MODEL        = os.environ.get("FAL_MODEL",        "fal-ai/wan-i2v").strip()

REPLICATE_TOKEN  = os.environ.get("REPLICATE_API_TOKEN", "").strip()
REPLICATE_MODEL  = os.environ.get("REPLICATE_MODEL",     "wan-video/wan-2.5-i2v").strip()

# Per-model input-argument mapping for Replicate. Each model uses a different
# parameter name for the reference image — this dict tells us which.
_REPLICATE_IMAGE_ARG = {
    "kwaivgi/kling-v2.0":            "start_image",
    "kwaivgi/kling-v2.1":            "start_image",
    "kwaivgi/kling-v1.6":            "start_image",
    "wan-video/wan-2.5-i2v":         "image",
    "wan-video/wan-2.1-i2v-720p":    "image",
    "wan-video/wan-2.1-i2v-480p":    "image",
    "minimax/video-01-live":         "first_frame_image",
    "minimax/hailuo-02":             "first_frame_image",
}

HF_SPACE         = os.environ.get("HF_SPACE",         "").strip()
HF_TOKEN         = os.environ.get("HF_TOKEN",         "").strip() or None
HF_API_NAME      = os.environ.get("HF_API_NAME",      "/generate_video").strip()

MAX_CONCURRENT   = int(os.environ.get("CLIP_CONCURRENCY", "4"))
PER_PROVIDER_TIMEOUT_S = int(os.environ.get("CLIP_PROVIDER_TIMEOUT", "180"))


# Map our Ken Burns motions to the camera_lora option used by the LTX-2 HF Space.
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_video_prompt(scene: dict) -> str:
    base = (
        scene.get("video_prompt")
        or scene.get("image_prompt", "")[:150]
        or "dramatic cinematic scene"
    )
    return (
        f"{base}, cinematic motion, smooth animation, "
        "epic lighting, shallow depth of field, high quality, vertical 9:16"
    )


def _download_url(url: str, dest: str, timeout: int = 90) -> str | None:
    """Download URL → local path. Returns dest on success, None on failure."""
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            print(f"        download HTTP {r.status_code}")
            return None
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)
        if os.path.getsize(dest) < 50_000:
            print(f"        download too small ({os.path.getsize(dest)} bytes)")
            return None
        return dest
    except Exception as e:
        print(f"        download error: {e}")
        return None


def _extract_video_url(result) -> str | None:
    """Best-effort extraction of a video URL from heterogeneous provider responses."""
    if result is None:
        return None
    if isinstance(result, str) and result.startswith("http"):
        return result
    if isinstance(result, dict):
        # fal — {"video": {"url": "..."}}
        v = result.get("video")
        if isinstance(v, dict) and isinstance(v.get("url"), str):
            return v["url"]
        # generic — {"url": "..."}
        if isinstance(result.get("url"), str):
            return result["url"]
        for key in ("output", "file", "path"):
            c = result.get(key)
            if isinstance(c, str) and c.startswith("http"):
                return c
    if isinstance(result, (list, tuple)) and result:
        return _extract_video_url(result[0])
    # replicate FileOutput object
    if hasattr(result, "url"):
        try:
            return str(result.url)
        except Exception:
            pass
    return None


# ── Provider 1: fal.ai ────────────────────────────────────────────────────────

def _image_to_data_uri(image_path: str) -> str:
    """Encode a JPEG/PNG into a base64 data URI. Avoids fal storage upload
    which requires billing setup on some free accounts."""
    import base64, mimetypes
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def _generate_via_fal(scene: dict, image_path: str, idx: int) -> str | None:
    if not FAL_KEY:
        return None
    try:
        import fal_client
    except ImportError:
        print(f"        [scene {idx}] fal-client not installed")
        return None

    try:
        # Pass the image inline as a base64 data URI. This bypasses fal's
        # storage endpoint which 403s on accounts without billing setup.
        print(f"        [scene {idx}] fal: encoding image as data URI...")
        image_url = await asyncio.to_thread(_image_to_data_uri, image_path)

        prompt = _build_video_prompt(scene)
        print(f"        [scene {idx}] fal: calling {FAL_MODEL}...")

        result = await asyncio.wait_for(
            asyncio.to_thread(
                fal_client.subscribe,
                FAL_MODEL,
                arguments={
                    "prompt":       prompt,
                    "image_url":    image_url,
                    "duration":     "5",
                    "aspect_ratio": "9:16",
                },
                with_logs=False,
            ),
            timeout=PER_PROVIDER_TIMEOUT_S,
        )

        video_url = _extract_video_url(result)
        if not video_url:
            print(f"        [scene {idx}] fal: no video URL in response")
            return None

        dest = f"temp/clips/raw_{idx:02d}.mp4"
        return await asyncio.to_thread(_download_url, video_url, dest)

    except asyncio.TimeoutError:
        print(f"        [scene {idx}] fal: timeout after {PER_PROVIDER_TIMEOUT_S}s")
        return None
    except Exception as e:
        print(f"        [scene {idx}] fal failed: {str(e)[:160]}")
        return None


# ── Provider 2: Replicate ─────────────────────────────────────────────────────

async def _generate_via_replicate(scene: dict, image_path: str, idx: int) -> str | None:
    if not REPLICATE_TOKEN:
        return None
    try:
        import replicate
    except ImportError:
        print(f"        [scene {idx}] replicate not installed")
        return None

    try:
        prompt = _build_video_prompt(scene)
        image_arg = _REPLICATE_IMAGE_ARG.get(REPLICATE_MODEL, "image")
        print(f"        [scene {idx}] replicate: calling {REPLICATE_MODEL} (image_arg={image_arg})...")

        def _call():
            with open(image_path, "rb") as f:
                payload = {
                    "prompt":   prompt,
                    image_arg:  f,
                    "duration": 5,
                }
                # Aspect ratio: only Kling accepts it; others use resolution flags
                if REPLICATE_MODEL.startswith("kwaivgi/"):
                    payload["aspect_ratio"] = "9:16"
                # Wan resolution flag — keep at 480p for cheaper generations
                if REPLICATE_MODEL.startswith("wan-video/"):
                    payload["resolution"] = os.environ.get("WAN_RESOLUTION", "480p")
                return replicate.run(REPLICATE_MODEL, input=payload)

        result = await asyncio.wait_for(
            asyncio.to_thread(_call),
            timeout=PER_PROVIDER_TIMEOUT_S,
        )

        video_url = _extract_video_url(result)
        if not video_url:
            print(f"        [scene {idx}] replicate: no video URL in response")
            return None

        dest = f"temp/clips/raw_{idx:02d}.mp4"
        return await asyncio.to_thread(_download_url, video_url, dest)

    except asyncio.TimeoutError:
        print(f"        [scene {idx}] replicate: timeout after {PER_PROVIDER_TIMEOUT_S}s")
        return None
    except Exception as e:
        print(f"        [scene {idx}] replicate failed: {str(e)[:160]}")
        return None


# ── Provider 3: HuggingFace Spaces (fallback) ─────────────────────────────────

def _hf_extract_video_path(result) -> str | None:
    """Find a local video file path in heterogeneous gradio response shapes."""
    if isinstance(result, str) and os.path.exists(result):
        return result
    if isinstance(result, dict):
        # multimodalart/wan2-1-fast returns {"video": "/tmp/xxx.mp4", "subtitles": None}
        v = result.get("video")
        if isinstance(v, str) and os.path.exists(v):
            return v
        if isinstance(v, dict):
            p = v.get("path") or v.get("name")
            if p and os.path.exists(p):
                return p
        for key in ("output", "file", "path", "name"):
            c = result.get(key)
            if isinstance(c, str) and os.path.exists(c):
                return c
    if isinstance(result, (list, tuple)) and result:
        for item in result:
            p = _hf_extract_video_path(item)
            if p:
                return p
    return None


async def _generate_via_hf(scene: dict, image_path: str, idx: int, motion: str) -> str | None:
    if not HF_SPACE:
        return None
    try:
        from gradio_client import Client, handle_file
    except ImportError:
        print(f"        [scene {idx}] gradio_client not installed")
        return None

    prompt = _build_video_prompt(scene)
    space_lower = HF_SPACE.lower()

    # Dispatch by Space slug — each Space has a different API signature.
    if "wan2-1-fast" in space_lower or "wan-2-1-fast" in space_lower:
        # multimodalart/wan2-1-fast schema
        def _call():
            client = Client(HF_SPACE, token=HF_TOKEN)
            return client.predict(
                input_image=handle_file(image_path),
                prompt=prompt,
                height=864,
                width=480,                 # 9:16 portrait
                negative_prompt=(
                    "blurry, low quality, distorted, watermark, text, "
                    "static picture, ugly, deformed, multiple people walking backwards"
                ),
                duration_seconds=5,
                guidance_scale=1.0,
                steps=4,
                seed=random.randint(0, 99999),
                randomize_seed=True,
                api_name="/generate_video",
            )
        provider_label = "wan2-1-fast"
    else:
        # Legacy Imosu/image_audio_to_video_NSFW signature
        camera_lora = _CAMERA_LORA.get(motion, "Zoom In")
        def _call():
            client = Client(HF_SPACE, token=HF_TOKEN)
            return client.predict(
                first_frame=handle_file(image_path),
                end_frame=None,
                prompt=prompt,
                duration=3.0,
                input_video=None,
                generation_mode="Image-to-Video",
                enhance_prompt=False,
                seed=random.randint(0, 9999),
                randomize_seed=True,
                height=512,
                width=288,
                camera_lora=camera_lora,
                custom_lora="None",
                audio_path=None,
                api_name=HF_API_NAME,
            )
        provider_label = f"imosu-style ({camera_lora})"

    try:
        print(f"        [scene {idx}] HF: calling {HF_SPACE} as {provider_label}...")
        result = await asyncio.wait_for(
            asyncio.to_thread(_call),
            timeout=PER_PROVIDER_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        print(f"        [scene {idx}] HF: timeout after {PER_PROVIDER_TIMEOUT_S}s")
        return None
    except Exception as e:
        print(f"        [scene {idx}] HF failed: {str(e)[:200]}")
        return None

    src = _hf_extract_video_path(result)
    if not src:
        print(f"        [scene {idx}] HF: unrecognized result format: {str(result)[:120]}")
        return None

    dest = f"temp/clips/raw_{idx:02d}.mp4"
    try:
        shutil.copy2(src, dest)
        return dest
    except Exception as e:
        print(f"        [scene {idx}] HF copy failed: {e}")
        return None


# ── Per-scene cascade ─────────────────────────────────────────────────────────

async def _generate_one_scene(
    scene: dict,
    image_path: str,
    idx: int,
    motion: str,
    sem: asyncio.Semaphore,
) -> str | None:
    async with sem:
        print(f"\n    [scene {idx}] cascade start (motion={motion})")

        for provider_name, runner in (
            ("fal",       lambda: _generate_via_fal(scene, image_path, idx)),
            ("replicate", lambda: _generate_via_replicate(scene, image_path, idx)),
            ("hf",        lambda: _generate_via_hf(scene, image_path, idx, motion)),
        ):
            result = await runner()
            if result and os.path.exists(result):
                print(f"    [scene {idx}] OK via {provider_name} -> {result}")
                return result

        print(f"    [scene {idx}] all providers failed")
        return None


# ── Public API ────────────────────────────────────────────────────────────────

async def generate_video_clips(scenes: list) -> tuple[list, list]:
    """
    Returns (clip_paths, image_files):
      clip_paths   list[str | None]  one per scene; None where every provider
                                     failed for that scene
      image_files  list[str]         the I2V reference image per scene, also
                                     usable as Ken Burns fallback by the
                                     assembler when clip_paths[i] is None

    Per-scene fallback is the caller's job: a None in clip_paths signals
    "render this scene from image_files[i] via Ken Burns". This function
    NEVER raises on partial failure — the orchestrator decides whether the
    failure rate warrants a full static-image pipeline (e.g. when zero clips
    succeeded). The earlier all-or-nothing behaviour threw away every
    successful clip the moment one scene failed; per-scene fallback keeps
    the cinematic look on the scenes that did succeed.
    """
    print(f"    Providers configured: "
          f"fal={'yes' if FAL_KEY else 'no'}  "
          f"replicate={'yes' if REPLICATE_TOKEN else 'no'}  "
          f"hf={'yes' if HF_SPACE else 'no'}")
    print(f"    Concurrency limit: {MAX_CONCURRENT}")

    # Step 3a — single wide-shot reference image per scene (no medium/closeup)
    print("\n    Step 3a — Generating reference images (1 wide shot per scene)...")
    image_groups = generate_images(scenes, single_shot=True)
    image_files = [
        g[0] if isinstance(g, list) and g else g
        for g in image_groups
    ]

    os.makedirs("temp/clips", exist_ok=True)

    # Step 3b — parallel cascade
    sem     = asyncio.Semaphore(MAX_CONCURRENT)
    motions = random.sample(_CAMERA_MOTIONS, min(len(scenes), len(_CAMERA_MOTIONS)))
    while len(motions) < len(scenes):
        motions.append(random.choice(_CAMERA_MOTIONS))

    print(f"\n    Step 3b — Generating {len(scenes)} clips in parallel...")
    tasks = [
        _generate_one_scene(scene, img_path, i + 1, motions[i], sem)
        for i, (scene, img_path) in enumerate(zip(scenes, image_files))
    ]
    results = await asyncio.gather(*tasks)

    n_ok     = sum(1 for r in results if r)
    failed   = [i + 1 for i, r in enumerate(results) if not r]
    if failed:
        print(f"\n    {n_ok}/{len(scenes)} clips succeeded; "
              f"scenes {failed} will render from static images")
    else:
        print(f"\n    All {len(scenes)} clips generated successfully")
    return list(results), image_files


# ── CLI: API discovery for whichever provider is configured ──────────────────

if __name__ == "__main__":
    if HF_SPACE:
        print(f"Discovering API for HF Space: {HF_SPACE}\n")
        try:
            from gradio_client import Client
            Client(HF_SPACE, token=HF_TOKEN).view_api()
        except Exception as e:
            print(f"Error: {e}")
    elif FAL_KEY:
        print(f"fal.ai configured. Default model: {FAL_MODEL}")
        print("Override via FAL_MODEL env. Browse models: https://fal.ai/models")
    elif REPLICATE_TOKEN:
        print(f"Replicate configured. Default model: {REPLICATE_MODEL}")
        print("Override via REPLICATE_MODEL env. Browse: https://replicate.com/collections/image-to-video")
    else:
        print("No provider configured. Set one of: FAL_KEY, REPLICATE_API_TOKEN, HF_SPACE")
