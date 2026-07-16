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

import json
import os
import re
import tempfile
from io import BytesIO

from PIL import Image

from pipeline.image_generator import _gen_cloudflare, _stable_master_seed
from pipeline.longform_assembler import rewrite_prompt_to_landscape

_W, _H = 1920, 1080

# House style + character identity come from assets/character_registry.json —
# one source of truth across episodes. Style = 3D DONGHUA CGI REALISM
# (Wu Dong Qian Kun register), NOT flat 2D: 2D looked less mature AND starved
# LTX-Video of the depth/gradient cues it animates from (flat art smears in
# motion). Character master_tokens are PREPENDED (FLUX weights early tokens most)
# to lock age/hair/robes/build wherever a character is named — the ₹0 stand-in
# for IP-Adapter (reduces drift; not pixel-perfect face-lock).
_REG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "character_registry.json",
)
try:
    _REG = json.loads(open(_REG_PATH, encoding="utf-8").read())
except Exception as _e:  # pragma: no cover
    print(f"[wuxia-images] character_registry.json not loaded ({_e}) — style fallback")
    _REG = {}

_STYLE_LOCK = _REG.get("_style_lock") or (
    "cinematic 3D Chinese donghua CGI, Wu Dong Qian Kun style, realistic "
    "atmospheric lighting, richly textured robes, mature serious tone, "
    "intricate detail, sharp focus, epic 16:9 landscape, masterpiece"
)
_WUXIA_STYLE = _STYLE_LOCK  # back-compat

# Negatives ONLY (FLUX RENDERS bans placed in the positive prompt). Bans flat-2D
# so the 3D look holds, and kills FLUX's hallucinated Chinese studio logos /
# corner calligraphy / on-rock engraved text.
_WUXIA_NEG = _REG.get("_global_negative") or (
    "2D, flat cartoon, anime cel-shaded, chibi, watermark, text, chinese "
    "characters, logo, calligraphy, signature, blurry, low quality, deformed "
    "hands, extra fingers, portrait, vertical crop"
)

# Precompiled character matchers: (alias_regex, master_token, compact_token, negative_token)
_CHARS = []
for _key, _c in (_REG.get("characters") or {}).items():
    _aliases = [_key.replace("_", " ")] + list(_c.get("aliases") or [])
    _pat = re.compile(r"\b(" + "|".join(re.escape(a) for a in _aliases) + r")\b", re.IGNORECASE)
    _CHARS.append((_pat, _c.get("master_token", ""),
                   _c.get("compact_token") or _c.get("master_token", ""),
                   _c.get("negative_token", "")))

# Multi-character shots: prevent the classic two-character failure modes —
# identity bleed (both get the same face) and body fusion. Injected ONLY when
# 2+ registry characters are matched in one shot.
_MULTI_SCAFFOLD = (
    "wide two-shot composition, the characters clearly separated with visible "
    "space between them, each with a DISTINCT face and outfit"
)
_MULTI_NEG = (
    "fused characters, merged bodies, conjoined bodies, merged clothing, shared "
    "face, identical faces, face blending, prompt bleeding, overlapping features, "
    "extra limbs"
)


def _match_characters(text: str):
    """(identity_tokens, negative_tokens, count) for registry characters named in
    `text`. With 2+ matches, COMPACT tokens are used instead of full master_tokens
    (two full tokens + style lock would blow FLUX's ~1000-char window) and the
    caller adds the spatial scaffold + anti-fusion negatives."""
    hits, negs = [], []
    for pat, master, compact, neg in _CHARS:
        if pat.search(text or ""):
            hits.append((master, compact))
            if neg and neg not in negs:
                negs.append(neg)
    if len(hits) >= 2:
        tokens = [c for _m, c in hits]
    else:
        tokens = [m for m, _c in hits]
    return tokens, negs, len(hits)


def _build_prompt(shot: dict, style_anchor: str | None = None):
    """Returns (positive, negative). PREPENDS: 3D style lock + character identity
    tokens; multi-character shots additionally get a spatial two-shot scaffold
    (left/right separation) + anti-fusion negatives."""
    base = rewrite_prompt_to_landscape(shot.get("prompt", "") or "")
    tokens, negs, n_chars = _match_characters(shot.get("prompt", "") or "")
    head = _STYLE_LOCK
    if n_chars >= 2:
        head += ". " + _MULTI_SCAFFOLD
    if tokens:
        head += ". " + ". ".join(tokens)
    pos = f"{head}. {base}" if base else head
    neg = _WUXIA_NEG + ((", " + ", ".join(negs)) if negs else "")
    if n_chars >= 2:
        neg += ", " + _MULTI_NEG
    return pos, neg


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


def _strip_watermark(raw: bytes) -> bytes:
    """Crop off FLUX's hallucinated corner seals/calligraphy, then scale back to
    full size. FLUX-schnell IGNORES negative prompts (guidance-distilled), and
    3D-donghua-style prompts strongly associate with real donghua frames carrying
    studio watermarks — so some stills render a fake corner seal no matter what.
    The seals live in the top ~15% corners; cropping top 16% / sides 8% / bottom
    2% removes them (validated on the 2026-07-15 motion lab: cropped seeds came
    out clean in stills AND the motion clips conditioned on them). Costs a slight
    punch-in, which reads as cinematic framing. Gate: WUXIA_STRIP_WATERMARK=0."""
    if os.environ.get("WUXIA_STRIP_WATERMARK", "1").strip() == "0":
        return raw
    img = Image.open(BytesIO(raw)).convert("RGB")
    w, h = img.size
    img = img.crop((int(w * 0.08), int(h * 0.16), int(w * 0.92), int(h * 0.98)))
    img = img.resize((w, h), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _gen_still_resilient(shot: dict, seed: int) -> bytes:
    """CF still with graceful NSFW degradation: original prompt -> sanitized ->
    generic safe fallback. Raises only if even the safe fallback exhausts (i.e.
    genuine quota death, not a content flag)."""
    pos, neg = _build_prompt(shot)
    try:
        return _gen_cloudflare(pos, seed, _W, _H, negative=neg)
    except RuntimeError as e:
        if "NSFW" not in str(e):
            raise  # genuine quota/exhaustion — let it bubble to the retry chain
        print("    [wuxia-still] NSFW flag -> retrying sanitized", flush=True)
        try:
            return _gen_cloudflare(_sanitize(pos), seed, _W, _H, negative=neg)
        except RuntimeError as e2:
            if "NSFW" not in str(e2):
                raise
            fallback = (f"{_STYLE_LOCK}. A dramatic cinematic wuxia moment, a young "
                        f"martial artist, glowing golden energy, epic atmosphere, "
                        f"wide landscape.")
            print("    [wuxia-still] still flagged -> generic safe fallback", flush=True)
            return _gen_cloudflare(fallback, seed, _W, _H, negative=neg)


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
            raw = _strip_watermark(_gen_still_resilient(shot, seed))

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
