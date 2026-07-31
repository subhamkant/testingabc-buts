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
import asyncio
import base64
import shutil
import hashlib
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from pipeline.clip_generator import _build_video_prompt
from pipeline.wuxia_kaggle import (
    push_wuxia_kernel, poll_kernel_as, download_output_as, KaggleClientError,
)
from pipeline import kaggle_client

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NB_ROOT = _REPO_ROOT / "kaggle_notebooks"
_KERNEL_DIR = _NB_ROOT / "maha-i2v"


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


# ──────────────────────────────────────────────────────────────────────────
# Phase 31 — resumable 6-slot multi-account pool (all-scene motion)
# ──────────────────────────────────────────────────────────────────────────
# Mirrors pipeline/wuxia_motion.py's pool + _kernel_task, adapted to the
# Mahabharata (clip_files) contract and the hero_<idx>.mp4 output naming.
# Takes a CheckpointStore `ck` so a job killed mid-poll (GHA 29-min cap)
# resumes by polling the already-pushed kernels — NEVER re-pushing a running
# one. That lets the existing single-run retry chain wait out a cold
# ~10-30 min Kaggle run without wasting GPU quota.

def _build_maha_pool() -> list:
    """[(slug, kernel_dir, creds_json)] across up to 3 accounts x 2 kernels.
    MAHA_KERNEL_POOL (csv of main,sk9,vyasa; default all) selects accounts.
    Skips an account whose creds file is missing or a kernel whose folder /
    metadata is absent, so a partial setup degrades gracefully."""
    acct_defs = [
        ("main", None, [
            ("subhamkant11/maha-i2v",   _NB_ROOT / "maha-i2v"),
            ("subhamkant11/maha-i2v-2", _NB_ROOT / "maha-i2v-2"),
        ]),
        ("sk9", str(_REPO_ROOT / "sk9_kaggle.json"), [
            ("subhamkant9/maha-i2v",    _NB_ROOT / "maha-i2v-sk9"),
            ("subhamkant9/maha-i2v-2",  _NB_ROOT / "maha-i2v-sk9-2"),
        ]),
        ("vyasa", str(_REPO_ROOT / "vyasa_ai_kaggle.json"), [
            ("vyasaai/maha-i2v",        _NB_ROOT / "maha-i2v-vyasa"),
            ("vyasaai/maha-i2v-2",      _NB_ROOT / "maha-i2v-vyasa-2"),
        ]),
    ]
    want = [w.strip() for w in
            os.environ.get("MAHA_KERNEL_POOL", "main,sk9,vyasa").split(",")
            if w.strip()]
    pool = []
    for name, creds, kernels in acct_defs:
        if name not in want:
            continue
        if creds and not os.path.exists(creds):
            print(f"    [maha-motion] account '{name}' creds missing ({creds}) - skipped")
            continue
        for slug, kdir in kernels:
            if not (kdir / "kernel-metadata.json").exists():
                continue
            pool.append((slug, kdir, creds))
    return pool


def _split(items: list, n: int) -> list:
    groups = [[] for _ in range(n)]
    for k, e in enumerate(items):
        groups[k % n].append(e)
    return groups


async def _kernel_task(ck, i, slug, kdir, creds, run_config, target,
                       max_attempts, stagger_s=0):
    """Push-if-new-else-resume-poll one kernel. Downloads its hero_<idx>.mp4
    into the shared `target`. Resumable + bounded-retry; best-effort (a failed
    kernel's scenes fall back to Ken Burns in the positional merge)."""
    done_key = f"maha_kernel_{i}.done"
    if ck.has(done_key):
        print(f"    [maha-motion] kernel {i} ({slug}) already done - skip")
        return
    state_key = f"maha_kernel_{i}.json"
    if not ck.has(state_key):
        if stagger_s:
            await asyncio.sleep(stagger_s)
        version = await push_wuxia_kernel(Path(kdir), run_config, creds_json=creds)
        ck.save_json(state_key, {"slug": slug, "version": version, "attempt": 1})
        print(f"    [maha-motion] kernel {i} submitted {slug} v{version} "
              f"({len(run_config['hero_motion'])} clip(s))")
    else:
        meta = ck.load_json(state_key)
        print(f"    [maha-motion] kernel {i} RESUME poll {slug} "
              f"v{meta.get('version')} (no re-push)")

    poll_interval = int(os.environ.get("KAGGLE_POLL_INTERVAL_S", "60"))
    poll_timeout = int(os.environ.get("MAHA_KAGGLE_TIMEOUT_S", "2700"))
    while True:
        try:
            res = await poll_kernel_as(slug, creds, poll_interval_s=poll_interval,
                                       timeout_s=poll_timeout)
        except KaggleClientError as e:
            # Timeout (kernel still running) or a network blip. Do NOT re-push
            # a live kernel - return WITHOUT marking done so a later attempt
            # (GHA retry sibling) resumes polling this same version.
            print(f"    [maha-motion] kernel {i} ({slug}) poll interrupted "
                  f"({str(e)[:100]}) - will resume next attempt")
            return
        if res["status"] == "complete":
            await download_output_as(slug, target, creds)
            ck.mark_done(done_key)
            print(f"    [maha-motion] kernel {i} ({slug}) complete + downloaded")
            return
        # Terminal FAILURE (error/failed/cancelled) - bounded re-push for a
        # fresh GPU draw (covers P100 draws + transient kernel crashes).
        meta = ck.load_json(state_key)
        attempt = int(meta.get("attempt", 1))
        if attempt < max_attempts:
            version = await push_wuxia_kernel(Path(kdir), run_config, creds_json=creds)
            ck.save_json(state_key, {"slug": slug, "version": version,
                                     "attempt": attempt + 1})
            print(f"    [maha-motion] kernel {i} ({slug}) status={res['status']} "
                  f"- re-pushed v{version} (attempt {attempt + 1}/{max_attempts})")
            continue
        # Exhausted -> give up so we never re-push again (bounded GPU spend);
        # this kernel's scenes stay Ken Burns.
        ck.mark_done(done_key)
        print(f"    [maha-motion] kernel {i} ({slug}) FAILED after {max_attempts} "
              f"attempt(s) - its scenes fall back to Ken Burns")
        return


async def generate_motion_clips_pool(ck, scenes, scene_indices,
                                     ref_image_paths, run_id):
    """Animate `scene_indices` in parallel across the multi-account pool.
    Returns clip_files aligned to `scenes` (mp4 path where a clip landed, else
    None -> Ken Burns). Resumable across invocations via `ck`."""
    n = len(scenes)
    clip_files = [None] * n

    entries = []
    for idx in scene_indices:
        if not (0 <= idx < n):
            continue
        img = ref_image_paths[idx] if idx < len(ref_image_paths) else None
        if not img or not os.path.exists(img):
            print(f"    [maha-motion] scene {idx}: no ref still - skipping motion")
            continue
        entries.append({
            "idx": idx,
            "prompt": _build_video_prompt(scenes[idx]),
            "image_b64": _seed_b64(img),
        })
    if not entries:
        print("    [maha-motion] no stills to animate - all Ken Burns")
        return clip_files

    pool = _build_maha_pool()
    if not pool:
        print("    [maha-motion] no Kaggle kernels available - all Ken Burns")
        return clip_files

    n_k = max(1, min(len(pool), len(entries)))
    groups = _split(entries, n_k)
    target = Path(ck.path("maha_motion_clips"))
    target.mkdir(parents=True, exist_ok=True)
    max_attempts = int(os.environ.get("MAHA_MOTION_MAX_ATTEMPTS", "4"))
    master_seed = _stable_seed(run_id)
    frames = int(os.environ.get("MAHA_LTX_FRAMES", "65"))
    steps = int(os.environ.get("MAHA_LTX_STEPS", "40"))
    guidance = float(os.environ.get("MAHA_LTX_GUIDANCE", "2.5"))

    print(f"    [maha-motion] {len(entries)} clip(s) across {n_k} kernel(s) "
          f"in pool of {len(pool)}: groups={[len(g) for g in groups]}")

    tasks = []
    for i in range(n_k):
        if not groups[i]:
            continue
        slug, kdir, creds = pool[i]
        rc = {
            "hero_motion":    groups[i],
            "master_seed":    master_seed,
            "run_id":         run_id,
            "ltx_num_frames": frames,
            "ltx_num_steps":  steps,
            "ltx_guidance":   guidance,
        }
        # Stagger first-pushes ~40s apart so 6 concurrent pushes don't hit the
        # Kaggle API simultaneously (resume polls are not delayed).
        tasks.append(_kernel_task(ck, i, slug, kdir, creds, rc, target,
                                  max_attempts, stagger_s=i * 40))

    await asyncio.gather(*tasks, return_exceptions=True)

    for p in target.glob("hero_*.mp4"):
        m = re.search(r"hero_(\d+)\.mp4$", p.name)
        if m:
            idx = int(m.group(1))
            if 0 <= idx < n:
                clip_files[idx] = str(p.resolve())
    got = sum(1 for c in clip_files if c)
    print(f"    [maha-motion] {got}/{len(entries)} motion clip(s) landed "
          f"(the rest -> Ken Burns)")
    return clip_files
