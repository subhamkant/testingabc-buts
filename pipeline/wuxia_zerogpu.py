"""
Wuxia ZeroGPU 14B motion — pipeline/wuxia_zerogpu.py
====================================================

Free Wan2.2-A14B (27B MoE) image-to-video via the HuggingFace ZeroGPU Space
`zerogpu-aoti/wan2-2-fp8da-aoti-faster` (gradio_client + HF_TOKEN). This is the
HERO-shot engine in the hybrid pipeline: reliably ~11-14 motion with real 3D
camera movement, vs the free-T4 Wan-5B's ~3-4 (see [[project_wuxia_channel_ltx]]).

Constraints (measured 2026-07-21):
  * ~5-6 clips/day/HF-account quota (resets ~daily). Caller budgets accordingly.
  * Transient "AcceleratorError" (bad H200 alloc) — retry 2-3x w/ backoff.
  * Also "down" (config-fetch fail) + "exceeded quota, try in Nh" states.
  * Output ~832x480 @16fps 81f (5s). Caller ESRGANs/masters downstream.

Sync API (kernel round-trip on a worker thread so it is safe to call from
inside an already-running asyncio loop, like wuxia_main.render()).
"""
from __future__ import annotations

import os
import shutil
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

_SPACE = os.environ.get("WUXIA_ZEROGPU_SPACE",
                        "zerogpu-aoti/wan2-2-fp8da-aoti-faster")


def _bar_crop_b64_file(still_path: str, dst: str) -> str:
    """Strip baked-in letterbox bars before sending to the space (the 14B
    otherwise inherits them into every frame). Writes a JPEG, returns path."""
    img = Image.open(still_path).convert("RGB")
    a = np.asarray(img.convert("L"), dtype=np.float32)
    rows = a.mean(axis=1)
    top = 0
    while top < len(rows) // 3 and rows[top] < 14:
        top += 1
    bot = len(rows)
    while bot > 2 * len(rows) // 3 and rows[bot - 1] < 14:
        bot -= 1
    img.crop((0, top, img.width, bot)).save(dst, quality=95)
    return dst


def zerogpu_available() -> bool:
    """Cheap liveness probe — True if the space config fetches (not down)."""
    try:
        from gradio_client import Client
        Client(_SPACE, token=os.environ.get("HF_TOKEN"))
        return True
    except Exception:
        return False


def generate_14b(still_path: str, prompt: str, out_path: str, *,
                 seed: int = 42, steps: int = 6, duration_s: int = 5,
                 max_retries: int = 3) -> bool:
    """Generate ONE 14B clip. Returns True on success (out_path written).
    Retries transient AcceleratorError; gives up (False) on quota-exhausted
    or persistent failure so the caller can fall back to the 5B/Ken Burns."""
    from gradio_client import Client, handle_file
    tmp_crop = out_path.replace(".mp4", "_seed.jpg")
    _bar_crop_b64_file(still_path, tmp_crop)
    try:
        for attempt in range(max_retries):
            try:
                c = Client(_SPACE, token=os.environ.get("HF_TOKEN"))
                video, _seed = c.predict(
                    input_image=handle_file(tmp_crop), prompt=prompt,
                    steps=steps, duration_seconds=duration_s, seed=seed,
                    randomize_seed=False, api_name="/generate_video")
                shutil.copy(video, out_path)
                return os.path.exists(out_path)
            except Exception as e:
                msg = str(e)
                if "quota" in msg.lower() or "exceeded" in msg.lower():
                    print(f"    [zerogpu] quota exhausted — {msg[:90]}", flush=True)
                    return False
                print(f"    [zerogpu] attempt {attempt+1}/{max_retries} "
                      f"failed ({msg[:80]}) — retrying", flush=True)
                time.sleep(45)
        return False
    finally:
        try:
            os.remove(tmp_crop)
        except OSError:
            pass


def generate_14b_batch(jobs: list[dict], out_dir: str, *,
                       budget: int | None = None) -> dict[int, str]:
    """jobs = [{idx, still_path, prompt, seed}]. Runs sequentially (ZeroGPU
    serializes per-account anyway), stops at `budget` clips or first quota
    hit. Returns {idx: clip_path} for successes. Safe from inside an async
    loop (no asyncio here — pure blocking gradio_client calls)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    got: dict[int, str] = {}
    limit = budget if budget is not None else len(jobs)
    for j in jobs[:limit]:
        out = os.path.join(out_dir, f"hero14b_{j['idx']:02d}.mp4")
        t0 = time.time()
        ok = generate_14b(j["still_path"], j["prompt"], out,
                          seed=int(j.get("seed", 42)))
        if ok:
            got[j["idx"]] = out
            print(f"    [zerogpu] idx {j['idx']} OK ({time.time()-t0:.0f}s)",
                  flush=True)
        else:
            print(f"    [zerogpu] idx {j['idx']} FAILED — falls back to 5B/KB",
                  flush=True)
            if not got and j is jobs[0]:
                # first job failed hard (likely space down/quota) — bail early
                # so we don't burn the whole loop retrying a dead space.
                break
    return got
