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
    # NEW PLATFORM (2026-07-19): the canonical bulk engine is Wan2.2-5B
    # (motion-lab-wan script: golden envelope, per-entry flow_shift, phased
    # ESRGAN, swap-cell notebook — the full July hardening), replacing LTX.
    # WUXIA_ENGINE=ltx reverts to the old script.
    engine_dir = ("wuxia-i2v" if os.environ.get("WUXIA_ENGINE") == "ltx"
                  else "motion-lab-wan")
    canonical = _NB_ROOT / engine_dir / "run_ltx_phase.py"
    canonical_nb = _NB_ROOT / engine_dir / "notebook.ipynb"
    pool = []
    for name, creds, kernels in acct_defs:
        if name not in want:
            continue
        if creds and not os.path.exists(creds):
            continue  # account not configured on this machine
        for slug, kdir in kernels:
            if not kdir.exists():
                continue
            # Anti-drift: every pool dir runs the CANONICAL kernel script
            # AND notebook (the notebook carries the 8GB-swap cell that
            # absorbs the mp4-writer RAM spike — the historical -9s).
            try:
                script = canonical.read_text(encoding="utf-8")
                target = kdir / "run_ltx_phase.py"
                if not target.exists() or target.read_text(encoding="utf-8") != script:
                    target.write_text(script, encoding="utf-8")
                nb = canonical_nb.read_text(encoding="utf-8")
                nb_target = kdir / "notebook.ipynb"
                if not nb_target.exists() or nb_target.read_text(encoding="utf-8") != nb:
                    nb_target.write_text(nb, encoding="utf-8")
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
# 2026-07-20: seed dims env-configurable so a PORTRAIT caller (Mahabharata
# Shorts) can seed 360x640 without distorting the still. Default = landscape
# (wuxia), zero behavior change.
_SEED_W = int(os.environ.get("WUXIA_SEED_W", "640"))
_SEED_H = int(os.environ.get("WUXIA_SEED_H", "360"))

_LOCK = asyncio.Lock()
_CLIP_RE = re.compile(r"scene_(\d+)_shot_(\d+)\.mp4$", re.IGNORECASE)


def plan_motion(scenes: list, still_groups: list) -> list:
    """One motion clip per shot flagged `requires_motion` (that has a still).

    NEW PLATFORM (2026-07-19): user-locked coverage = 60-70% of SCENES carry
    real motion (rest are Ken Burns stills). If the script's own flags fall
    short of WUXIA_MOTION_COVERAGE (default 0.65), promote additional scenes'
    first shot — highest-impact scenes first (the IMPACT regex), then in
    script order — until the target is met.
    """
    plan: list[dict] = []
    covered: set[int] = set()
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
            covered.add(i)
            idx += 1

    target = float(os.environ.get("WUXIA_MOTION_COVERAGE", "0.65"))
    want = int(round(target * len(scenes)))
    if len(covered) < want:
        candidates = []
        for i, scene in enumerate(scenes):
            if i in covered:
                continue
            track = scene.get("visual_track", []) or []
            if not track:
                continue
            still = (still_groups[i][0]
                     if i < len(still_groups) and still_groups[i] else None)
            if not still or not os.path.exists(still):
                continue
            prompt = track[0].get("prompt", "") or ""
            impact = 1 if _IMPACT_RE.search(prompt) else 0
            candidates.append((-impact, i, prompt, still))
        candidates.sort()
        for _neg_impact, i, prompt, still in candidates[:want - len(covered)]:
            plan.append({
                "idx": idx, "scene_idx": i, "shot_idx": 0,
                "prompt": prompt, "still_path": still,
            })
            covered.add(i)
            idx += 1
        print(f"[motion] coverage promoted to {len(covered)}/{len(scenes)} "
              f"scenes (target {target:.0%})", flush=True)
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


# WAN-5B motion prompts (2026-07-20): Wan2.2-5B does NOT melt on body motion
# the way LTX-2B did, so the hard-freeze language (_NB_FREEZE) that crushed it
# to motion 1-3 is REPLACED with "subtle natural motion" — confirmed to lift
# the same still from 2 -> 15 motion while staying coherent + identity-locked.
# The LTX melt-guard prompts are kept for WUXIA_ENGINE=ltx only.
_WAN_MOVE = ("subtle natural body sway and breathing, head stable and face "
             "perfectly sharp, no warping, no morphing, identity unchanged")
_WAN_CINEMATIC = os.environ.get("WUXIA_WAN_CINEMATIC") or (
    f"Cinematic 3D donghua. Slow cinematic camera push-in with gentle parallax. "
    f"His robes and hair ripple in a soft wind, atmospheric mist and dust motes "
    f"drift through shifting light. {_WAN_MOVE}.")
_WAN_IMPACT = os.environ.get("WUXIA_WAN_IMPACT") or (
    f"Cinematic 3D donghua. Dramatic slow camera push-in. His robes and hair "
    f"whip in the wind, swirling dust and glowing embers stream past, a golden "
    f"energy aura flares and pulses around him, high kinetic energy. {_WAN_MOVE}.")
_WAN_HORDE = os.environ.get("WUXIA_WAN_HORDE") or (
    f"Cinematic 3D donghua. The hero holds his stance, face sharp and stable, "
    f"as the surrounding group surges forward as one mass, robes billowing, dust "
    f"erupting, weapons glinting; slow cinematic pan. {_WAN_MOVE}, distinct "
    f"crowd figures, no fused bodies.")


def _pick_motion_prompt(entry: dict) -> str:
    override = os.environ.get("WUXIA_MOTION_PROMPT", "").strip()
    if override:
        return override
    if entry.get("motion_prompt"):  # explicit per-scene override
        return entry["motion_prompt"]
    text = entry.get("prompt", "") or ""
    ltx = os.environ.get("WUXIA_ENGINE") == "ltx"
    if _GROUP_RE.search(text):
        return _MOTION_HORDE if ltx else _WAN_HORDE
    impact = bool(_IMPACT_RE.search(text))
    if ltx:
        return _MOTION_IMPACT if impact else _MOTION_CINEMATIC
    return _WAN_IMPACT if impact else _WAN_CINEMATIC


# Motion negative: the July lab list — includes the Wan Chinese-calligraphy
# hallucination ban (sk9 scene3, 2026-07-19) and the anti-limb/fire set.
_MOTION_NEG = (
    "motion blur, blur streaks, smeared movement, ghosting, blurry, low quality, "
    "deformed, melting, warping, extra limbs, mutated, face distortion, changing "
    "face, fire, flames, burning, walking, stepping, punching, kicking, "
    "body rotation, turning around, chinese text, calligraphy, watermark, "
    "text overlay, subtitles")


def _build_run_config(run_id: str, group: list) -> dict:
    hero = [{
        "idx": e["idx"], "scene_idx": e["scene_idx"], "shot_idx": e["shot_idx"],
        # C+D hybrid environmental-kinetics prompt (NOT the scene's action verb)
        # to avoid the articulated-limb smear/collapse on the free T4.
        "prompt": _pick_motion_prompt(e),
        "negative": _MOTION_NEG,
        "image_b64": _seed_b64(e["still_path"]),
    } for e in group]
    if os.environ.get("WUXIA_ENGINE") == "ltx":
        return {
            "skip_flux": True,
            "hero_motion": hero,
            "master_seed": _stable_master_seed(run_id),
            "run_id": run_id,
            "ltx_num_frames": int(os.environ.get("WUXIA_LTX_FRAMES", "65")),
            "ltx_num_steps": int(os.environ.get("WUXIA_LTX_STEPS", "36")),
            "ltx_guidance": float(os.environ.get("WUXIA_LTX_GUIDANCE", "3.5")),
            "ltx_width": _LTX_W, "ltx_height": _LTX_H,
            "esrgan": os.environ.get("WUXIA_ESRGAN", "true").lower() != "false",
            "out_w": 1920, "out_h": 1080,
            "ltx_timeout_s": int(os.environ.get("WUXIA_LTX_TIMEOUT_S", "5400")),
            "clip_timeout_s": int(os.environ.get("WUXIA_CLIP_TIMEOUT_S", "540")),
        }
    # Wan2.2-5B GOLDEN ENVELOPE (do not change without a lab run): fp16,
    # 33 frames, 35 steps @ 832x480 = 4/4 reliability on free T4 (bf16 OOMs,
    # 49f@40steps host-RAM SIGKILLs). Phased ESRGAN upscales 2x in-kernel.
    return {
        "skip_flux": True,
        "hero_motion": hero,
        "master_seed": _stable_master_seed(run_id),
        "run_id": run_id,
        "wan_dtype": "fp16",
        "wan_frames": int(os.environ.get("WUXIA_WAN_FRAMES", "33")),
        "wan_steps": int(os.environ.get("WUXIA_WAN_STEPS", "35")),
        # flow_shift = Wan's motion-amplitude knob. Default 12 (mapped 2026-07-19:
        # coherent strong motion; 8 detonates on explosive stills). WITHOUT this
        # the pipeline ran conservative-default motion (clips came out ~1-3);
        # 12 + loosened prompts lifted the same still to motion 15 (confirmed).
        "flow_shift": float(os.environ.get("WUXIA_WAN_FLOW_SHIFT", "12")),
        # 2026-07-20: gen dims env-configurable (kernel default is 832x480
        # landscape). PORTRAIT caller sets 480x832 -> ESRGAN 2x -> 960x1664,
        # which the portrait assembler crops to 1080x1920. Default preserves
        # the wuxia golden envelope (832x480).
        "wan_width": int(os.environ.get("WUXIA_WAN_WIDTH", "832")),
        "wan_height": int(os.environ.get("WUXIA_WAN_HEIGHT", "480")),
        "esrgan": os.environ.get("WUXIA_ESRGAN", "true").lower() != "false",
        # generous ceilings: ~10 clips/kernel x (gen ~5min + esrgan ~1.5min)
        "ltx_timeout_s": int(os.environ.get("WUXIA_LTX_TIMEOUT_S", "10800")),
        "clip_timeout_s": int(os.environ.get("WUXIA_CLIP_TIMEOUT_S", "900")),
    }


def _clip_motion(path: str) -> float:
    """Mean inter-frame luma delta — cheap motion-energy metric (matches the
    audit tooling). Returns 0.0 on any failure (treated as static)."""
    import subprocess
    import tempfile

    import numpy as np
    d = tempfile.mkdtemp(prefix="mchk_")
    try:
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", path,
             "-vf", "fps=8,scale=320:180", f"{d}/f%03d.png"],
            capture_output=True)
        if r.returncode != 0:
            return 0.0
        frames = sorted(Path(d).glob("f*.png"))
        prev = None
        deltas = []
        for f in frames:
            g = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
            if prev is not None:
                deltas.append(float(np.abs(g - prev).mean()))
            prev = g
        return float(np.mean(deltas)) if deltas else 0.0
    except Exception:
        return 0.0
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def _merge_clips_into_stills(still_groups: list, ck) -> list:
    """Positional merge: motion mp4 where present AND actually moving, else the
    CF still (which the assembler animates with a Ken Burns push).

    MOTION FLOOR (2026-07-20): a clip below WUXIA_MOTION_FLOOR (default 4.0)
    barely moves — it would read as a dead freeze. Dropping it to its still
    means the assembler gives it a Ken Burns camera push instead, which at
    least breathes. Result cached so the (slow) measurement runs once."""
    floor = float(os.environ.get("WUXIA_MOTION_FLOOR", "5.0"))
    scored = ck.load_json("motion_scores.json") if ck.has("motion_scores.json") else {}
    clip_by_pos: dict[tuple[int, int], str] = {}
    dropped = 0
    clips_dir = Path(ck.path("motion_clips"))
    if clips_dir.exists():
        for p in clips_dir.glob("*.mp4"):
            m = _CLIP_RE.search(p.name)
            if not m:
                continue
            key = p.name
            if key not in scored:
                scored[key] = _clip_motion(str(p))
            if scored[key] < floor:
                dropped += 1
                continue
            clip_by_pos[(int(m.group(1)) - 1, int(m.group(2)) - 1)] = str(p)
    ck.save_json("motion_scores.json", scored)
    if dropped:
        print(f"    [wuxia-motion] {dropped} clip(s) below motion floor {floor} "
              f"-> Ken Burns (no dead freezes)", flush=True)
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
        # BEST-EFFORT (default, local runs): a single kernel failing must NOT
        # abort the whole episode — its shots fall back to Ken Burns stills in
        # the positional merge. Set WUXIA_MOTION_BEST_EFFORT=0 for the GHA
        # retry-chain behavior (raise so the retry job resumes this kernel).
        if os.environ.get("WUXIA_MOTION_BEST_EFFORT", "1") == "0":
            raise KaggleClientError(
                f"wuxia kernel {i} ({slug}) ended status={res['status']} "
                f"(attempt {attempt}); GHA retry job will resume.")
        print(f"    [wuxia-motion] kernel {i} ({slug}) FAILED status={res['status']} "
              f"— those shots fall back to Ken Burns (best-effort)", flush=True)
        return


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

        # best-effort: gather without letting one kernel's failure abort the
        # rest. In GHA mode (_BEST_EFFORT=0) _kernel_task still raises and the
        # exception surfaces here to fail the job for the retry chain.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errs = [r for r in results if isinstance(r, Exception)]
        if errs and os.environ.get("WUXIA_MOTION_BEST_EFFORT", "1") == "0":
            raise errs[0]

        target = Path(ck.path("motion_clips"))
        mp4s = [str(p) for p in target.glob("*.mp4")] if target.exists() else []
        if not mp4s:
            # total motion wipeout — don't checkpoint 'done' (so a re-run can
            # retry motion once the -9/quota cause is fixed); degrade to stills.
            print("    [wuxia-motion] NO clips produced — episode = all Ken Burns "
                  "this pass (motion NOT marked done; re-run to retry)", flush=True)
            return _merge_clips_into_stills(still_groups, ck)
        ck.save_json("motion_manifest.json", {"clips": mp4s})
        ck.mark_done("motion_done")
        print(f"    [wuxia-motion] {len(mp4s)} clip(s) landed "
              f"({len(errs)} kernel(s) failed -> those shots = Ken Burns)", flush=True)
        return _merge_clips_into_stills(still_groups, ck)
