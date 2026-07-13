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
import re
import tempfile

from pipeline.image_generator import _gen_cloudflare, _stable_master_seed
from pipeline.longform_assembler import rewrite_prompt_to_landscape

_W, _H = 1920, 1080

# LOCKED 2D-anime look, PREPENDED to every prompt (FLUX weights early tokens
# most). One uniform style across all frames kills the 2D/3D/chibi/age
# shape-shifting that a free CF-schnell flow (no IP-Adapter/FaceID) otherwise
# produces. "consistent character design" nudges frame-to-frame identity; true
# face-lock still needs IP-Adapter (Kaggle path) — this is the ₹0 best-effort.
_STYLE_LOCK = (
    "2D anime cel-shaded art style, flat cel shading, bold clean black ink "
    "outlines, vibrant flat colors, modern donghua TV anime key art, "
    "consistent character design, coherent art direction"
)

# Kept for back-compat / fallback only; _build_prompt no longer trusts the
# script's style_anchor (it fought the 2D lock with a "3D cinematic" cue).
_WUXIA_STYLE = _STYLE_LOCK

# Negatives ONLY (FLUX renders bans placed in the positive prompt). The
# corner-text terms kill FLUX's habit of hallucinating fake Chinese studio
# logos / calligraphy watermarks scraped from donghua training frames.
_WUXIA_NEG = (
    "watermark, text, signature, chinese characters, chinese text, japanese "
    "text, logo, calligraphy, username, stamp, subtitles, caption, letters, "
    "3d render, cgi, photorealistic, realistic, chibi, deformed, distorted "
    "face, deformed hands, extra fingers, mutated limbs, blurry, low quality, "
    "low resolution, soft focus, jpeg artifacts, oversaturated, washed out, "
    "portrait, vertical crop"
)


def _build_prompt(shot: dict, style_anchor: str | None = None) -> str:
    # PREPEND the hard 2D lock; the script's style_anchor is intentionally
    # ignored (it carried a conflicting "cinematic donghua 3D" cue).
    base = rewrite_prompt_to_landscape(shot.get("prompt", "") or "")
    return f"{_STYLE_LOCK}. {base}" if base else _STYLE_LOCK


# Cloudflare FLUX's NSFW classifier is trigger-happy on combat/gore/occult words
# and will 400 an entire cascade if the last account with quota flags them. Map
# such words to safe, visually-equivalent phrasings so a spicy scene degrades
# gracefully instead of crashing the whole render (critical for the daily cron).
_NSFW_MAP = [
    (r"\bblood(y|ed)?\b", "glowing red light"),
    (r"\bbleed(ing)?\b", "glowing"),
    (r"\bskelet(on|ons|al)\b", "radiant golden spirit"),
    (r"\bbones?\b", "golden light"),
    (r"\bbrutal(ly)?\b", "powerful"),
    (r"\bagoniz(ing|ed)\b|\bagony\b", "intense"),
    (r"\bviolent(ly)?\b|\bviolence\b", "dynamic"),
    (r"\bgore\b|\bgory\b", "energy"),
    (r"\bcorpse(s)?\b|\bdead\b|\bdeath\b", "fallen"),
    (r"\bkill(ing|ed)?\b", "defeat"),
    (r"\bwound(ed|s)?\b", "mark"),
    (r"\bdark energy\b|\bblack energy\b", "glowing energy"),
    (r"\bmenacing(ly)?\b", "powerful"),
    (r"\btortur\w*\b", "strain"),
    (r"\bpain(ful)?\b", "strain"),
]


def _sanitize(prompt: str) -> str:
    out = prompt
    for pat, repl in _NSFW_MAP:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    return out


def _gen_still_resilient(shot: dict, seed: int) -> bytes:
    """CF still with graceful NSFW degradation: original prompt -> sanitized ->
    generic safe fallback. Raises only if even the safe fallback exhausts (i.e.
    genuine quota death, not a content flag)."""
    prompt = _build_prompt(shot)
    try:
        return _gen_cloudflare(prompt, seed, _W, _H, negative=_WUXIA_NEG)
    except RuntimeError as e:
        if "NSFW" not in str(e):
            raise  # genuine quota/exhaustion — let it bubble to the retry chain
        safe = _sanitize(prompt)
        print(f"    [wuxia-still] NSFW flag -> retrying sanitized", flush=True)
        try:
            return _gen_cloudflare(safe, seed, _W, _H, negative=_WUXIA_NEG)
        except RuntimeError as e2:
            if "NSFW" not in str(e2):
                raise
            fallback = (f"{_STYLE_LOCK}. A dramatic cinematic wuxia moment, a young "
                        f"martial artist, glowing golden energy, epic atmosphere, "
                        f"wide landscape.")
            print(f"    [wuxia-still] still flagged -> generic safe fallback", flush=True)
            return _gen_cloudflare(fallback, seed, _W, _H, negative=_WUXIA_NEG)


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
            raw = _gen_still_resilient(shot, seed)

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
