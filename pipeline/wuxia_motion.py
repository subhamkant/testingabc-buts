"""
Wuxia motion — pipeline/wuxia_motion.py
=======================================

Orchestrates Kaggle LTX motion-clip generation for motion-flagged shots across
TWO parallel kernels (Kaggle's free-tier concurrent limit), with per-kernel
CROSS-JOB checkpoint resume.

Flow:
  plan_motion(scenes, still_groups)  -> deterministic [{idx,scene_idx,shot_idx,prompt,still_path}]
  run_motion(ck, run_id, ...)        -> splits clips across wuxia-i2v / wuxia-i2v-2,
                                         pushes+polls+downloads BOTH concurrently,
                                         merges clips over stills -> final list-of-lists

Resume invariants (protect the 30h/wk Kaggle quota):
  * Per kernel i: `motion_kernel_<i>.json` is saved BEFORE polling => presence
    means "already submitted, RESUME polling, do NOT re-push".
  * `motion_kernel_<i>.done` marks a kernel whose clips are downloaded => skipped
    on resume (no re-poll).
  * A new push happens ONLY on first submit or a confirmed P100 draw.
  * A genuine error/timeout RAISES so the GHA retry job restores the checkpoint
    and re-enters; already-done kernels are skipped, in-flight kernels resume.

Parallelism note: 2 concurrent kernels ~halve wall-clock but burn the weekly GPU
quota ~2x faster (same total capacity). Each kernel batches ~half the clips
(model loads once per kernel).

Motion is an enhancement: motion shots that fail to produce an mp4 fall back to
their CF still (Ken Burns) in the assembler.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
from io import BytesIO
from pathlib import Path

from PIL import Image

from pipeline.image_generator import _stable_master_seed
from pipeline.wuxia_kaggle import (
    KaggleClientError,
    download_output_as,
    is_p100_failure,
    poll_kernel_as,
    push_wuxia_kernel,
)

_NB_ROOT = Path(__file__).resolve().parent.parent / "kaggle_notebooks"

# ── 6-slot kernel pool: 3 accounts x 2 concurrent kernels (Kaggle's per-account
# cap). Each entry = (slug, kernel_dir, creds_json|None); creds None = ambient
# default account (repo-root kaggle.json = subhamkant11). Fresh accounts need
# PHONE VERIFICATION on kaggle.com before their kernels get GPU/internet.
# WUXIA_KERNEL_POOL selects accounts (csv of main,sk9,vyasa; default all);
# accounts whose creds file is missing are skipped. WUXIA_MAX_KERNELS caps slots.
_REPO_ROOT = _NB_ROOT.parent


def _build_pool() -> list:
    acct_defs = [
        ("main", None, [
            (os.environ.get("WUXIA_KAGGLE_KERNEL_REF", "subhamkant11/wuxia-i2v"), _NB_ROOT / "wuxia-i2v"),
            (os.environ.get("WUXIA_KAGGLE_KERNEL_REF_2", "subhamkant11/wuxia-i2v-2"), _NB_ROOT / "wuxia-i2v-2"),
        ]),
        ("sk9", str(_REPO_ROOT / "sk9_kaggle.json"), [
            ("subhamkant9/wuxia-i2v", _NB_ROOT / "wuxia-i2v-sk9"),
            ("subhamkant9/wuxia-i2v-2", _NB_ROOT / "wuxia-i2v-sk9-2"),
        ]),
        ("vyasa", str(_REPO_ROOT / "vyasa_ai_kaggle.json"), [
            ("vyasaai/wuxia-i2v", _NB_ROOT / "wuxia-i2v-vyasa"),
            ("vyasaai/wuxia-i2v-2", _NB_ROOT / "wuxia-i2v-vyasa-2"),
        ]),
    ]
    want = [a.strip() for a in os.environ.get(
        "WUXIA_KERNEL_POOL", "main,sk9,vyasa").split(",") if a.strip()]
    canonical = _NB_ROOT / "wuxia-i2v" / "run_ltx_phase.py"
    pool = []
    for name, creds, kernels in acct_defs:
        if name not in want:
            continue
        if creds and not os.path.exists(creds):
            continue  # account not configured on this machine
        for slug, kdir in kernels:
            if not kdir.exists():
                continue
            # Anti-drift: every pool dir runs the CANONICAL kernel script.
            try:
                script = canonical.read_text(encoding="utf-8")
                target = kdir / "run_ltx_phase.py"
                if not target.exists() or target.read_text(encoding="utf-8") != script:
                    target.write_text(script, encoding="utf-8")
            except OSError:
                pass
            pool.append((slug, kdir, creds))
    return pool


_KERNELS = _build_pool()

_LTX_W, _LTX_H = 1152, 640

# Seeds are base64-INLINED into the pushed notebook; Kaggle's SaveKernel API
# rejects oversized source (~1MB => 400). LTX only uses the seed as a
# composition guide (resizes internally; in-kernel ESRGAN reconstructs detail),
# so a small 640x360 q78 seed (~50-70KB b64) keeps ~12 clips/kernel under the cap.
_SEED_W, _SEED_H = 640, 360

_LOCK = asyncio.Lock()
_CLIP_RE = re.compile(r"scene_(\d+)_shot_(\d+)\.mp4$", re.IGNORECASE)


def plan_motion(scenes: list, still_groups: list) -> list:
    """One LTX clip per shot flagged `requires_motion` (that has a still)."""
    plan: list[dict] = []
    idx = 0
    for i, scene in enumerate(scenes):
        for j, shot in enumerate(scene.get("visual_track", []) or []):
            if not shot.get("requires_motion"):
                continue
            still = None
            if i < len(still_groups) and j < len(still_groups[i]):
                still = still_groups[i][j]
            if not still or not os.path.exists(still):
                continue
            plan.append({
                "idx": idx, "scene_idx": i, "shot_idx": j,
                "prompt": shot.get("prompt", "") or "", "still_path": still,
            })
            idx += 1
    return plan


def _split(plan: list, n: int) -> list:
    """Round-robin split so both kernels get a balanced clip count."""
    groups: list[list] = [[] for _ in range(n)]
    for k, e in enumerate(plan):
        groups[k % n].append(e)
    return groups


def _seed_b64(still_path: str) -> str:
    img = Image.open(still_path).convert("RGB").resize((_SEED_W, _SEED_H), Image.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# RESTING-LIMB + ENVIRONMENTAL-KINETICS motion prompts. LTX-2B on a free T4 CANNOT
# compute articulated limb motion (arm/broom swings) — it loses spatial tracking
# and the face/body MELT into a smeared blur by ~2s. So the LTX prompt DELIBERATELY
# ignores the scene's action verb and freezes the body while animating only the
# environment. Validated on the 2026-07-15 sk9 motion lab (5 styles, zero smear)
# and locked by the user as the "C + D hybrid rotation":
#   CINEMATIC (Style C, default ~70%): slow camera orbit + drifting mist/parallax —
#     makes stills read as 3D-CGI renders in dialogue/scenic beats.
#   IMPACT (Style D, ~30% high-impact beats): wind + embers + golden aura + dramatic
#     push-in — for power-ups, standoffs, techniques, climaxes.
# Per-scene override via a "motion_prompt" field; WUXIA_MOTION_PROMPT overrides ALL.
_NB_FREEZE = (
    "IMPORTANT: the character's body, arms, hands and pose stay COMPLETELY STILL "
    "and frozen — no limb movement, no re-posing, no swinging, face perfectly stable."
)
_MOTION_CINEMATIC = os.environ.get("WUXIA_MOTION_CINEMATIC") or (
    f"Cinematic 3D donghua. The character stands motionless like a statue. {_NB_FREEZE} "
    "The camera slowly orbits around him with gentle parallax, atmospheric mist and "
    "haze drift across the scene, soft shifting sunlight rays, dust motes float. ONLY "
    "the camera and atmosphere move. stable geometry, crisp, highly detailed face, "
    "no warping, no morphing."
)
_MOTION_IMPACT = os.environ.get("WUXIA_MOTION_IMPACT") or (
    f"Cinematic 3D donghua. The character is frozen in a still powerful stance. {_NB_FREEZE} "
    "Black robes and hair whip in the wind, swirling dust and glowing embers drift, a "
    "faint golden energy aura shimmers around him, slow dramatic camera push-in. "
    "stable geometry, crisp, highly detailed face, no warping, no morphing."
)

# High-impact classifier: if the shot's IMAGE prompt (which describes the beat)
# carries energy/combat/supernatural cues, use IMPACT; else CINEMATIC.
_IMPACT_RE = re.compile(
    r"\b(aura|energy|qi\b|glow|lightning|flame|fire|ember|explod|erupt|surge|"
    r"skelet|spirit|vortex|power|battle|fight|duel|strike|attack|clash|punch|"
    r"kick|technique|breakthrough|rage|furious|storm)\w*",
    re.IGNORECASE,
)

# GROUP-fight beats (a horde attacking): animate the GROUP as one kinetic force
# advancing while the hero's body stays locked — never individual choreography
# for 3+ characters (that's what melts).
_GROUP_RE = re.compile(
    r"\b(group of|horde|mob|crowd of|surround(ed|ing)|dozens|several (disciples|"
    r"opponents|attackers))\b", re.IGNORECASE)
_MOTION_HORDE = os.environ.get("WUXIA_MOTION_HORDE") or (
    f"Cinematic 3D donghua. The hero holds a completely FROZEN defensive stance — "
    f"body, arms and face perfectly still and stable. The surrounding group of "
    f"disciples surges forward aggressively toward him AS ONE MASS, robes billowing "
    f"violently, heavy dust clouds erupting from the ground, weapon blades glinting; "
    f"slow cinematic camera pan. High visual tension, fluid donghua physics, stable "
    f"geometry, no warping, no morphing, distinct crowd figures."
)


def _pick_motion_prompt(entry: dict) -> str:
    override = os.environ.get("WUXIA_MOTION_PROMPT", "").strip()
    if override:
        return override
    if entry.get("motion_prompt"):  # explicit per-scene override
        return entry["motion_prompt"]
    text = entry.get("prompt", "") or ""
    if _GROUP_RE.search(text):
        return _MOTION_HORDE
    return _MOTION_IMPACT if _IMPACT_RE.search(text) else _MOTION_CINEMATIC


def _build_run_config(run_id: str, group: list) -> dict:
    hero = [{
        "idx": e["idx"], "scene_idx": e["scene_idx"], "shot_idx": e["shot_idx"],
        # C+D hybrid environmental-kinetics prompt (NOT the scene's action verb)
        # to avoid the articulated-limb smear/collapse on the free T4.
        "prompt": _pick_motion_prompt(e),
        "image_b64": _seed_b64(e["still_path"]),
    } for e in group]
    return {
        "skip_flux": True,
        "hero_motion": hero,
        "master_seed": _stable_master_seed(run_id),
        "run_id": run_id,
        "ltx_num_frames": int(os.environ.get("WUXIA_LTX_FRAMES", "65")),
        # 36 steps (was 30) = more refinement; guidance 3.5 (was 3.0) forces
        # sharper cel-shaded line-work for the 2D-anime look.
        "ltx_num_steps": int(os.environ.get("WUXIA_LTX_STEPS", "36")),
        "ltx_guidance": float(os.environ.get("WUXIA_LTX_GUIDANCE", "3.5")),
        "ltx_width": _LTX_W, "ltx_height": _LTX_H,
        "esrgan": os.environ.get("WUXIA_ESRGAN", "true").lower() != "false",
        "out_w": 1920, "out_h": 1080,
        "ltx_timeout_s": int(os.environ.get("WUXIA_LTX_TIMEOUT_S", "5400")),
        "clip_timeout_s": int(os.environ.get("WUXIA_CLIP_TIMEOUT_S", "540")),  # kernel watchdog
    }


def _merge_clips_into_stills(still_groups: list, ck) -> list:
    """Positional merge: motion mp4 where present, else the CF still."""
    clip_by_pos: dict[tuple[int, int], str] = {}
    clips_dir = Path(ck.path("motion_clips"))
    if clips_dir.exists():
        for p in clips_dir.glob("*.mp4"):
            m = _CLIP_RE.search(p.name)
            if m:
                clip_by_pos[(int(m.group(1)) - 1, int(m.group(2)) - 1)] = str(p)
    merged: list[list[str]] = []
    for i, row in enumerate(still_groups):
        merged.append([clip_by_pos.get((i, j), still) for j, still in enumerate(row)])
    return merged


async def _kernel_task(ck, i: int, slug: str, kernel_dir: Path, run_config: dict,
                       max_p100: int, creds_json: str | None = None,
                       stagger_s: int = 0) -> None:
    """Push (if needed) -> poll -> download for ONE kernel. Own checkpoint keys.
    creds_json selects the Kaggle account (None = ambient default). stagger_s
    delays the FIRST push only (not resume) so multi-account pushes don't land
    on Kaggle's API at the same instant."""
    done_key = f"motion_kernel_{i}.done"
    if ck.has(done_key):
        print(f"    [wuxia-motion] kernel {i} ({slug}) already done — skip", flush=True)
        return

    state_key = f"motion_kernel_{i}.json"
    if not ck.has(state_key):
        if stagger_s:
            await asyncio.sleep(stagger_s)
        version = await push_wuxia_kernel(Path(kernel_dir), run_config, creds_json=creds_json)
        ck.save_json(state_key, {"slug": slug, "version": version, "attempt": 1})
        print(f"    [wuxia-motion] kernel {i} submitted {slug} v{version} "
              f"({len(run_config['hero_motion'])} clips)", flush=True)
    else:
        meta = ck.load_json(state_key)
        print(f"    [wuxia-motion] kernel {i} RESUME poll {slug} v{meta.get('version')} "
              f"(no re-push)", flush=True)

    poll_interval = int(os.environ.get("KAGGLE_POLL_INTERVAL_S", "60"))
    # Own timeout (NOT the shared KAGGLE_TIMEOUT_S, which Mahabharata tunes short):
    # a ~7-clip batch on a T4 is ~50 min, so give 2h headroom. In GHA the 50-min
    # job timeout kills the poll first and the retry chain resumes anyway.
    poll_timeout = int(os.environ.get("WUXIA_KAGGLE_TIMEOUT_S", "7200"))
    target = Path(ck.path("motion_clips"))
    target.mkdir(parents=True, exist_ok=True)

    while True:
        res = await poll_kernel_as(slug, creds_json,
                                   poll_interval_s=poll_interval, timeout_s=poll_timeout)
        if res["status"] == "complete":
            await download_output_as(slug, target, creds_json)
            ck.mark_done(done_key)
            print(f"    [wuxia-motion] kernel {i} ({slug}) complete + downloaded", flush=True)
            return
        meta = ck.load_json(state_key)
        attempt = int(meta.get("attempt", 1))
        # P100 log-check uses ambient creds — only meaningful for the main account.
        if creds_json is None and await is_p100_failure(slug) and attempt < max_p100:
            version = await push_wuxia_kernel(Path(kernel_dir), run_config, creds_json=creds_json)
            attempt += 1
            ck.save_json(state_key, {"slug": slug, "version": version, "attempt": attempt})
            print(f"    [wuxia-motion] kernel {i} P100 draw — re-pushed (attempt {attempt})",
                  flush=True)
            continue
        raise KaggleClientError(
            f"wuxia kernel {i} ({slug}) ended status={res['status']} (attempt {attempt}); "
            f"GHA retry job will resume."
        )


async def run_motion(ck, run_id: str, scenes: list, still_groups: list,
                     max_p100_retries: int = 5) -> list:
    """Generate motion clips across up to 2 parallel Kaggle kernels (resumable),
    then return the merged list-of-lists (motion mp4 where available, else still)."""
    async with _LOCK:
        if ck.has("motion_done"):
            return _merge_clips_into_stills(still_groups, ck)

        if ck.has("motion_plan.json"):
            plan = ck.load_json("motion_plan.json")
        else:
            plan = plan_motion(scenes, still_groups)
            ck.save_json("motion_plan.json", plan)

        if not plan:
            ck.mark_done("motion_done")
            return _merge_clips_into_stills(still_groups, ck)

        # WUXIA_MAX_KERNELS caps concurrency. Default = all (2) for speed; set to
        # 1 to route every clip through the single WARM kernel (reliable — the
        # 2nd slug cold-starts slowly and can die past ltx_timeout with 0 clips).
        max_k = int(os.environ.get("WUXIA_MAX_KERNELS", str(len(_KERNELS))))
        n_kernels = max(1, min(len(_KERNELS), len(plan), max_k))
        groups = _split(plan, n_kernels)
        print(f"    [wuxia-motion] {len(plan)} clips across {n_kernels} parallel kernel(s): "
              f"{[len(g) for g in groups]}", flush=True)

        tasks = []
        for i in range(n_kernels):
            if not groups[i]:
                continue
            slug, kdir, creds = _KERNELS[i]
            rc = _build_run_config(run_id, groups[i])
            # Stagger first-pushes ~40s apart so multi-account pushes don't hit
            # Kaggle's API simultaneously (resume polls are not delayed).
            tasks.append(_kernel_task(ck, i, slug, kdir, rc, max_p100_retries,
                                      creds_json=creds, stagger_s=i * 40))

        await asyncio.gather(*tasks)  # raises on any kernel failure -> retry resumes

        target = Path(ck.path("motion_clips"))
        mp4s = [str(p) for p in target.glob("*.mp4")] if target.exists() else []
        ck.save_json("motion_manifest.json", {"clips": mp4s})
        ck.mark_done("motion_done")
        print(f"    [wuxia-motion] all kernels complete — {len(mp4s)} clip(s) total", flush=True)
        return _merge_clips_into_stills(still_groups, ck)
