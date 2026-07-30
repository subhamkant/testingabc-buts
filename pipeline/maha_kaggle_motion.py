"""Mahabharata Kaggle LTX I2V motion — free-T4 image-to-video for hero scenes.

Phase 30 (2026-07-31). The fal/Replicate/HF cascade in clip_generator.py is
either paid (fal/Replicate) or unreliable (the free HF ZeroGPU space returns
server-side RuntimeError). Kaggle's free T4 running the PROVEN LTX I2V kernel
(kaggle_notebooks/maha-i2v/run_ltx_phase.py — 512x768 portrait, 65 frames /
guidance 2.5, ~140s/clip) is free AND reliable, at the cost of latency
(async push→poll→download, ~10-15 min warm / ~30 min cold + queue).

This module is a thin Mahabharata-specific orchestrator that reuses the wuxia
LTX-only Kaggle client (push_wuxia_kernel) and the generic poll/download
helpers. It animates ONLY the 2-3 hero scenes selected upstream; every other
scene, and any clip that fails, returns None so the assembler Ken-Burns-falls-
back. It never raises for a motion failure — motion is an enhancement layer.

Isolation: separate module + separate kernel slug (MAHA_KAGGLE_KERNEL_REF,
default subhamkant11/maha-i2v) so it never collides with the curiosity
(KAGGLE_KERNEL_REF, FLUX+IP-Adapter) or wuxia kernel pools.
"""

import os
import re
import base64
import shutil
import hashlib
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from pipeline.clip_generator import _build_video_prompt
from pipeline.wuxia_kaggle import push_wuxia_kernel, KaggleClientError
from pipeline import kaggle_client

_KERNEL_DIR = Path(__file__).resolve().parent.parent / "kaggle_notebooks" / "maha-i2v"


# LTX conditions at 512x768 portrait, so seeding at that size is lossless.
_SEED_W = int(os.environ.get("MAHA_SEED_W", "512"))
_SEED_H = int(os.environ.get("MAHA_SEED_H", "768"))


def _seed_b64(path: str) -> str:
    """Downscale the still to the LTX conditioning size and JPEG-encode before
    base64. A full-res seed (1792x3200) bloats the inlined notebook past
    Kaggle's SaveKernel payload limit → 400 Bad Request (learned 2026-07-31).
    ~50-120 KB/seed keeps the push small. Mirrors wuxia_motion._seed_b64."""
    img = Image.open(path).convert("RGB").resize((_SEED_W, _SEED_H), Image.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _stable_seed(run_id: str) -> int:
    # hashlib (not hash()) for cross-process determinism.
    return int(hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8], 16) % 100000


async def generate_kaggle_motion_clips(
    scenes: list,
    scene_indices: list,
    ref_image_paths: list,
    run_id: str,
    max_p100_retries: int = 3,
) -> list:
    """Animate the hero `scene_indices` on Kaggle's free T4 LTX I2V kernel.

    Returns clip_files aligned to `scenes`: an mp4 path at each hero index a
    clip was produced for, else None (→ Ken Burns fallback in the assembler).
    Never raises for a motion failure.
    """
    kernel_ref = os.environ.get(
        "MAHA_KAGGLE_KERNEL_REF", "subhamkant11/maha-i2v").strip()
    n = len(scenes)
    clip_files = [None] * n

    # Build hero_motion from the already-generated reference stills.
    hero = []
    for idx in scene_indices:
        if idx < 0 or idx >= n:
            continue
        img = ref_image_paths[idx] if idx < len(ref_image_paths) else None
        if not img or not os.path.exists(img):
            print(f"    [maha-kaggle] scene {idx}: no ref still — skipping motion")
            continue
        hero.append({
            "idx": idx,
            "prompt": _build_video_prompt(scenes[idx]),
            "image_b64": _seed_b64(img),
        })
    if not hero:
        print("    [maha-kaggle] no hero stills to animate — all Ken Burns")
        return clip_files

    if not (_KERNEL_DIR / "kernel-metadata.json").exists():
        print(f"    [maha-kaggle] kernel folder missing ({_KERNEL_DIR}) — all Ken Burns")
        return clip_files

    run_config = {
        "hero_motion":    hero,
        "master_seed":    _stable_seed(run_id),
        "run_id":         run_id,
        "ltx_num_frames": int(os.environ.get("MAHA_LTX_FRAMES", "65")),
        "ltx_num_steps":  int(os.environ.get("MAHA_LTX_STEPS", "40")),
        "ltx_guidance":   float(os.environ.get("MAHA_LTX_GUIDANCE", "2.5")),
    }
    poll_interval = int(os.environ.get("KAGGLE_POLL_INTERVAL_S", "60"))
    timeout_s = int(os.environ.get("KAGGLE_TIMEOUT_S", "2700"))

    tmp = Path(tempfile.mkdtemp(prefix="maha_i2v_"))
    try:
        completed = False
        for attempt in range(max_p100_retries + 1):
            print(f"    [maha-kaggle] pushing {kernel_ref} "
                  f"({len(hero)} clip(s)) — attempt {attempt + 1}...")
            try:
                await push_wuxia_kernel(_KERNEL_DIR, run_config)
            except KaggleClientError as e:
                print(f"    [maha-kaggle] push failed: {str(e)[:200]} — all Ken Burns")
                return clip_files
            try:
                status = await kaggle_client.poll_kernel(
                    kernel_ref, poll_interval_s=poll_interval, timeout_s=timeout_s)
            except KaggleClientError as e:
                print(f"    [maha-kaggle] poll failed: {str(e)[:200]} — all Ken Burns")
                return clip_files
            st = (status or {}).get("status", "")
            print(f"    [maha-kaggle] kernel status: {st or 'unknown'}")
            if st == "complete":
                completed = True
                break
            # Kaggle randomly allocates a T4 or an incompatible P100 — a cheap
            # re-push often draws a T4 next time.
            if attempt < max_p100_retries:
                try:
                    if await kaggle_client.is_p100_failure(kernel_ref):
                        print("    [maha-kaggle] P100 draw — re-pushing for a T4...")
                        continue
                except Exception:
                    pass
            print("    [maha-kaggle] kernel did not complete — all Ken Burns")
            return clip_files

        if not completed:
            return clip_files

        try:
            files = await kaggle_client.download_output(kernel_ref, tmp)
        except KaggleClientError as e:
            print(f"    [maha-kaggle] download failed: {str(e)[:200]} — all Ken Burns")
            return clip_files

        os.makedirs("temp/clips", exist_ok=True)
        got = 0
        for p in files:
            m = re.search(r"hero_(\d+)\.mp4$", str(p))
            if not m:
                continue
            idx = int(m.group(1))
            if 0 <= idx < n:
                dest = os.path.abspath(f"temp/clips/maha_kaggle_{idx:02d}.mp4")
                try:
                    shutil.copy2(str(p), dest)
                    clip_files[idx] = dest
                    got += 1
                except OSError as e:
                    print(f"    [maha-kaggle] copy hero_{idx:02d} failed: {e}")
        print(f"    [maha-kaggle] {got}/{len(hero)} motion clip(s) landed "
              f"(the rest → Ken Burns)")
        return clip_files
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
