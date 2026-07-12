"""
Wuxia stills — pipeline/wuxia_images.py
=======================================

Generate one Cloudflare FLUX-schnell still per shot at 1920x1080 (landscape).
Reuses image_generator's proven CF cascade (`_gen_cloudflare`, 5-account
rotation, prompt clamp) and longform_assembler's landscape prompt rewriter.

Every shot gets a still — static shots use it for Ken Burns, motion shots use
it as the LTX conditioning seed (pipeline/wuxia_motion.py). Checkpoint-resumable:
already-generated stills are skipped. Deterministic seeds via _stable_master_seed
so a re-run reproduces the same frames.

Isolation: new module; does not modify image_generator. Imports its helpers.
"""
from __future__ import annotations

import os
import tempfile

from pipeline.image_generator import _gen_cloudflare, _stable_master_seed
from pipeline.longform_assembler import rewrite_prompt_to_landscape

_W, _H = 1920, 1080

_WUXIA_STYLE = (
    "cinematic donghua, Chinese wuxia animation style, dramatic volumetric "
    "lighting, intricate detail, sharp focus, epic 16:9 landscape composition, "
    "moody atmospheric"
)
_WUXIA_NEG = (
    "blurry, low quality, low resolution, soft focus, distorted face, "
    "deformed hands, extra fingers, mutated limbs, jpeg artifacts, "
    "oversaturated, washed out, watermark, text, portrait, vertical crop"
)


def _build_prompt(shot: dict, style_anchor: str | None) -> str:
    base = rewrite_prompt_to_landscape(shot.get("prompt", "") or "")
    anchor = (style_anchor or _WUXIA_STYLE).strip()
    return f"{base}, {anchor}" if base else anchor


def generate_wuxia_stills(scenes: list, ck, run_id: str, style_anchor: str | None = None) -> list:
    """Generate a 1920x1080 CF still per shot. Returns a positional list-of-lists
    (one inner list per scene, one entry per shot) of cache paths.

    Resumable: `ck.has()` skips shots already saved. Raises if the CF cascade is
    fully exhausted for a shot (retry chain re-runs; done shots persist)."""
    master = _stable_master_seed(run_id)
    manifest: dict[str, str] = {}
    scene_groups: list[list[str]] = []

    for i, scene in enumerate(scenes):
        shots = scene.get("visual_track", []) or []
        row: list[str] = []
        for j, shot in enumerate(shots):
            name = f"stills/scene_{i+1:02d}_shot_{j+1:02d}.jpg"
            if ck.has(name):
                row.append(ck.path(name))
                manifest[f"{i}_{j}"] = name
                continue

            seed = (master + i * 100 + j) % (2**31 - 1)
            prompt = _build_prompt(shot, style_anchor)
            raw = _gen_cloudflare(prompt, seed, _W, _H, negative=_WUXIA_NEG)

            # _gen_cloudflare returns raw image bytes; write to a temp then
            # atomic-copy into the checkpoint (reuses ck.save_file's tmp+rename).
            tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            try:
                tf.write(raw)
                tf.close()
                ck.save_file(name, tf.name)
            finally:
                try:
                    os.unlink(tf.name)
                except OSError:
                    pass

            row.append(ck.path(name))
            manifest[f"{i}_{j}"] = name
            print(f"    [wuxia-still] scene {i+1} shot {j+1} -> {name}", flush=True)

        scene_groups.append(row)

    ck.save_json("stills_manifest.json", manifest)
    return scene_groups
