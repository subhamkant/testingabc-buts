import requests
import os
import re
import time
import json
import base64
import io
import subprocess
from urllib.parse import quote


# Per-pipeline temp root. Mahabharata leaves env unset → "temp". The explainer
# driver sets PIPELINE_TEMP_ROOT="temp/fe" before importing this module so its
# generated images land under temp/fe/images/ (and its cached copies under
# cache/<run_id>/visuals/ — which is unchanged because cache is already
# per-run-id-namespaced).
_TEMP_ROOT = os.environ.get("PIPELINE_TEMP_ROOT", "temp")

# Mahabharata style suffix — photorealistic cinematic period film aesthetic
# tuned for NATURAL color and skin tones. Earlier iterations went too far in
# either direction:
#   - "Amar Chitra Katha cel-shaded" → cartoonish (Anger's Fire-style)
#   - heavy "warm Bhansali" + saturated jewel tones → magenta/pink wash on
#     everything (the "भीम और दुर्योधन" upload — purple skin, pink sky)
# This sweet spot leans on the Star Bharat Mahabharat live-action / Baahubali
# reference: photoreal, ornate, jewel-toned BUT with accurate skin and
# balanced color. Explicitly bans color casts in the negative cues.
#
# WhatIf series uses a different per-style suffix from _WHATIF_STYLE_SUFFIXES
# based on the script's visual_style — this Mahabharata suffix does not apply.
# MORTAL variant — golden-bronze skin anchor for mortal Mahabharata warriors
# (Karna, Arjuna, Bhima, etc.). The "luminous golden glowing complexion" line
# was added 2026-05-14 evening to push back against FLUX silhouette-dark
# tendencies under "oil-lamp lighting" cues.
STYLE_SUFFIX_MORTAL = (
    # Photoreal anchor — kept short for distilled FLUX variants (Pollinations,
    # Cloudflare) which honor long prompts less reliably than HF FLUX-schnell.
    # WALKED BACK from earlier version: "physically-based skin with visible
    # pores and fine facial hair" + "ultra-sharp 8K detail throughout" were
    # producing splotchy/scarred face artifacts and broken eye anatomy on
    # women's close-ups in the 2026-05-13 local test (Draupadi face had
    # mottled "wet/grimy" texture, eyelid anatomy was broken in another shot).
    # Distilled FLUX-schnell over-interprets "visible pores" as "lots of
    # skin texture" and produces splotches. Softer anchors below.
    "photorealistic cinematic film still shot on Arri Alexa LF, "
    "Kodak Vision3 5219 film stock, "
    "natural skin texture, realistic facial features, "
    # Eye-specific anchor — added 2026-05-14 after the Karna-arc local test
    # shipped 7/10 frames with dead-eye / black-void pupils. Distilled FLUX
    # at 4 steps loses eye micro-detail without an explicit eye cue. This
    # is eye-only — does not re-introduce "pores"/"8K" that caused splotchy
    # skin on the prior iteration.
    "detailed expressive eyes with clearly defined iris and pupils, "
    "natural catch-light reflections in the eyes, "
    # Face-exposure anchor — added 2026-05-14 after the v3_images smoke test
    # shipped Karna with near-black skin from FLUX over-interpreting "oil-lamp"
    # / "subtle glow" lighting cues. Canonical Karna is golden-bronze (Surya's
    # son), not silhouette-dark. This anchor pushes back without overruling
    # mood / shadow direction from the scene prompt.
    "warm golden-bronze skin tone for Indian characters, "
    "luminous golden glowing complexion, "
    "well-lit faces with key-light on the face, even facial exposure, "
    "ancient India Mahabharat live-action / Baahubali period-film aesthetic, "
    "carved sandstone temple architecture in sharp focus, oil-lamp lighting, "
    "balanced natural color grading, neutral whites, true skin tones, "
    "sharp focus on subject, clear facial features, "
    "no global color wash, no orange filter, no magenta or pink cast, "
    "no CGI plastic look, no airbrushed skin"
)

# DIVINE variant — same as MORTAL minus the golden-bronze skin anchor that
# conflicts with canonical divine skin tones (Krishna indigo-blue, Hanuman
# red-gold). Added 2026-05-14 after the "Krishna to Karna" upload showed
# Krishna rendered as dark teal/muddied-blue because the global golden anchor
# was fighting his `characters.json` "dark indigo-blue divine skin" descriptor.
# All other anchors (eye detail, well-lit face, photoreal cinema, anti-color-
# wash negatives) preserved — only the skin-tone enforcement changes.
STYLE_SUFFIX_DIVINE = (
    "photorealistic cinematic film still shot on Arri Alexa LF, "
    "Kodak Vision3 5219 film stock, "
    "natural skin texture, realistic facial features, "
    "detailed expressive eyes with clearly defined iris and pupils, "
    "natural catch-light reflections in the eyes, "
    # Skin-tone routing for divine + mortal in the same frame: explicitly name
    # BOTH so FLUX doesn't average them into a single muddy tone. The character
    # descriptor injected by _inject_characters already says "indigo-blue divine
    # skin" for Krishna and "tan-brown body + vanara face with reddish-brown fur"
    # for Hanuman — this anchor reinforces Krishna's blue AND keeps mortal
    # warriors bronze when they share a frame with him. Hanuman's per-character
    # descriptor is detailed enough on its own; don't fight it from the suffix.
    "Krishna with brilliant indigo-blue divine skin (when present in scene), "
    "mortal warriors with warm golden-bronze skin tone, "
    "well-lit faces with key-light on the face, even facial exposure, "
    "ancient India Mahabharat live-action / Baahubali period-film aesthetic, "
    "carved sandstone temple architecture in sharp focus, oil-lamp lighting, "
    "balanced natural color grading, neutral whites, "
    "sharp focus on subject, clear facial features, "
    "no global color wash, no orange filter, no magenta or pink cast, "
    "no CGI plastic look, no airbrushed skin"
)

# Backwards-compat alias — kept so existing `style_suffix: str = STYLE_SUFFIX`
# default-arg signatures continue to work. Mortal is the safe default since
# the divine override fires only when a divine character is detected in the
# prompt (see generate_image_bytes).
STYLE_SUFFIX = STYLE_SUFFIX_MORTAL

# WhatIf series style suffixes — picked by `script["visual_style"]`. Mahabharata
# style does NOT apply to WhatIf scripts; the LLM picks the most suitable style
# per topic (e.g. dinosaurs -> nature-doc, black hole -> sci-fi-cinematic).
_WHATIF_STYLE_SUFFIXES = {
    "photoreal-3d": (
        "photorealistic CGI render, octane / unreal-engine-5 quality, "
        "physically-based shading, accurate scale and proportions, "
        "raytraced reflections, soft global illumination, volumetric atmosphere, "
        "scientific-illustration accuracy, true-to-life materials and textures, "
        "ultra-sharp 8K detail, professional cinematic lighting, "
        "high dynamic range, neutral color grading, no fantasy distortion"
    ),
    "nature-doc": (
        "BBC Planet Earth documentary cinematography, shot on Arri Alexa Mini LF, "
        "telephoto 600mm lens compression, naturalistic golden-hour lighting, "
        "true-to-life animal anatomy, accurate ecosystem detail, "
        "shallow depth of field with creamy bokeh, "
        "ultra-sharp 8K wildlife photography detail, professional color grading, "
        "no fantasy elements, no cartoon styling"
    ),
    "sci-fi-cinematic": (
        "Denis Villeneuve sci-fi cinematography, Roger Deakins lighting, "
        "shot on 35mm anamorphic Kodak Vision3 5219, "
        "monumental scale composition with tiny human silhouettes for scale, "
        "atmospheric haze and god-rays, muted desaturated palette with one accent color, "
        "deep ultra-wide compositions, cinematic 2.39:1 framing feel, "
        "ultra-sharp 8K detail, no neon kitsch, no pulp space-opera tropes"
    ),
    "illustrated": (
        "high-end editorial illustration, National Geographic feature art style, "
        "polished concept art with painterly detail, ArtStation Featured quality, "
        "dramatic composition with strong focal hierarchy, "
        "vivid but believable color palette, sharp readable subjects, "
        "no anime, no cel-shading, no flat cartoon styling"
    ),
}


def _resolve_style_suffix(series: str, visual_style: str) -> str:
    """Return the right style suffix for the series + visual_style."""
    if series == "whatif":
        return _WHATIF_STYLE_SUFFIXES.get(visual_style, _WHATIF_STYLE_SUFFIXES["photoreal-3d"])
    return STYLE_SUFFIX

# Negative prompt — distilled FLUX variants (Pollinations, Cloudflare)
# down-weight late tokens, so the highest-priority failure modes are
# front-loaded. The order below reflects what the Volcanoes + Gandhari
# analysis flagged as the most-recurring visible quality regressions
# (heavy color wash, plastic skin, CGI-game-character vibe).
_NEGATIVE_DEFAULT = (
    # ── Eye-detail failure mode (TOP priority — 2026-05-14 Karna-arc local
    # test shipped 7/10 frames with dead-eye / black-void pupils. Distilled
    # FLUX-schnell loses eye micro-detail; front-loaded negatives push it
    # to actually render iris/pupil structure rather than dark voids). ──
    "dead eyes,glassy eyes,vacant stare,blank eyes,soulless eyes,"
    "missing pupils,missing iris,black void eyes,recessed eye sockets,"
    "asymmetric eyes,one eye closed,wonky eyes,smeared eyes,blurred eyes,"
    "eyes without detail,unfocused eyes,"
    # ── Face-underexposure failure mode (2026-05-14 v3_images test: Karna
    # rendered with near-black skin from FLUX over-reading "oil-lamp" cues
    # as silhouette lighting; canonical Karna is golden-bronze). ──
    "underexposed face,blackened skin,overly dark face,silhouette face,"
    "face in deep shadow,unlit face,muddy skin tone,washed-out dark skin,"
    "ashen face,grey skin,"
    # ── Face-distortion failure mode (observed in 2026-05-13 test where
    # women's faces had splotchy/scarred patterns + broken eye anatomy.
    # Distilled FLUX-schnell variants downweight late negatives). ──
    "splotchy skin,mottled face,scarred face,skin condition,acne,"
    "exaggerated pore detail,over-textured skin,patchy skin,"
    "deformed eyes,broken facial anatomy,extra eye,"
    "missing eye,crooked eye,droopy eyelid,merged eyebrows,"
    # ── Color-wash failure mode (most visible color regression) ──
    "orange cast,magenta cast,pink cast,purple cast,color wash,"
    "warm filter,heavy filter,sepia overlay,monochrome filter,"
    "over-saturated,over-graded,"
    # ── Plastic / CGI-character failure mode ──
    "cgi plastic skin,doll-like face,waxy skin,smooth airbrushed skin,"
    "3d render plastic,video game character,pixar style,"
    "unreal engine character,over-rendered,"
    # ── Cartoon / illustration bans ──
    "cartoon,anime,cel shaded,illustration,drawing,comic book,"
    # ── Embedded-text failure mode (2026-05-14 production check: FLUX-schnell
    # garbled "Vyasa AI" as VyssA / Virtasy / Vilysaria / Viysas across uploaded
    # outro frames. Distilled FLUX cannot reliably spell brand names — keep
    # text out of the prompts AND front-load explicit negatives so the model
    # avoids letterforms even when prompts accidentally suggest them). ──
    "text,letters,letterforms,typography,calligraphy,handwriting,"
    "channel name,subscribe text,logo text,bold text,glowing text,"
    "scribbled letters,garbled text,misspelled text,fake text,"
    "watermark,signature,caption,subtitle text in image,"
    # ── Standard quality / anatomy fixes (preserved from prior version) ──
    "blurry,blur,out of focus,low quality,pixelated,distorted,"
    "ugly,bad anatomy,logo,duplicate,deformed,"
    "extra fingers,six fingers,seven fingers,too many fingers,"
    "mutated hands,malformed hands,fused fingers,missing fingers,"
    "extra limbs,extra arms,malformed limbs,disfigured,"
    "cross-eyed,bad proportions"
)

# Restraint-mode negative — used when an imperfection cue is active for the
# scene (mood matches grief/aftermath/witnessed/etc.). 2026-05-18 Phase 2/3
# stabilization. Strips the skin-condition + face-underexposure rejections
# because those FIGHT the weathered / dust-streaked / red-rimmed / unevenly-
# lit anchors that imperfection routing requests. Keeps every real-bug
# rejection (dead eyes, deformed anatomy, color wash, plastic skin, text,
# finger/limb).
#
# Risk note: distilled FLUX-schnell can produce uncanny skin when the
# splotchy/scarred negatives are absent. Mitigation — this variant only
# applies on scenes whose mood matches restraint keywords. Standard scenes
# (hook, setup, rising tension) keep the full DEFAULT rejection list.
_NEGATIVE_RESTRAINT = (
    # Eye-detail — still a real-bug rejection; keep
    "dead eyes,glassy eyes,vacant stare,blank eyes,soulless eyes,"
    "missing pupils,missing iris,black void eyes,recessed eye sockets,"
    "asymmetric eyes,one eye closed,wonky eyes,smeared eyes,blurred eyes,"
    "eyes without detail,unfocused eyes,"
    # Face-distortion — keep ONLY structural-anatomy bugs. The skin-texture
    # bans (splotchy/mottled/scarred/acne/pore-detail/etc.) have been dropped
    # so weathered / dust-streaked / red-rimmed cues can land.
    "deformed eyes,broken facial anatomy,extra eye,"
    "missing eye,crooked eye,droopy eyelid,merged eyebrows,"
    # Color-wash — still want neutral grade
    "orange cast,magenta cast,pink cast,purple cast,color wash,"
    "warm filter,heavy filter,sepia overlay,monochrome filter,"
    "over-saturated,over-graded,"
    # Plastic / CGI — still want photoreal
    "cgi plastic skin,doll-like face,waxy skin,smooth airbrushed skin,"
    "3d render plastic,video game character,pixar style,"
    "unreal engine character,over-rendered,"
    # Cartoon / illustration bans
    "cartoon,anime,cel shaded,illustration,drawing,comic book,"
    # Embedded-text — still want clean frames
    "text,letters,letterforms,typography,calligraphy,handwriting,"
    "channel name,subscribe text,logo text,bold text,glowing text,"
    "scribbled letters,garbled text,misspelled text,fake text,"
    "watermark,signature,caption,subtitle text in image,"
    # Anatomy
    "blurry,blur,out of focus,low quality,pixelated,distorted,"
    "ugly,bad anatomy,logo,duplicate,deformed,"
    "extra fingers,six fingers,seven fingers,too many fingers,"
    "mutated hands,malformed hands,fused fingers,missing fingers,"
    "extra limbs,extra arms,malformed limbs,disfigured,"
    "cross-eyed,bad proportions"
)

# Backwards-compat alias — _NEGATIVE was the single global pre-2026-05-18.
_NEGATIVE = _NEGATIVE_DEFAULT


# ─── Imperfection cue routing (Phase 2/3 stabilization, 2026-05-18) ──────
# Mood-routed cue table analogous to _HOOK_VISUALS. When scene.mood contains
# any of these keywords, the corresponding cue gets APPENDED to the scene's
# image_prompt (additive, not a replacement) AND _NEGATIVE_RESTRAINT is used
# for that scene's provider calls in place of _NEGATIVE_DEFAULT.
#
# Three cue families (earlier slots win first-match):
#   grief/loss/mourning  — pain-in-motion: red-rimmed eyes, trembling hand
#   aftermath/haunting/  — cost: weathered skin, dust on fingers, weapon
#     hollow/irreversible/  held loosely as if about to be dropped, knees
#     weary/severed         barely supporting weight
#   witnessed/abandoned/ — frozen vulnerability: posture of one who has
#     unresolved            seen too much, eyes that look past the viewer,
#                           frozen mid-motion as if forgotten how to step
#
# The cues encode PHYSICAL VULNERABILITY (inability, hesitation, weakness)
# not just "weathered" surface — per 2026-05-18 user feedback distinguishing
# "warrior staring sadly at battlefield" (composed grief) from "warrior
# unable to lift weapon anymore" (cost embodied).
_IMPERFECTION_GRIEF = (
    "dust-streaked face, eyes red-rimmed but not crying, "
    "hair displaced by wind, garment torn at one edge, "
    "asymmetric posture, hand trembling slightly, "
    "shoulders slumped under invisible weight"
)
_IMPERFECTION_AFTERMATH = (
    "weathered skin with visible age lines, ash on shoulders, "
    "dust on bowstring fingers, weapon held loosely as if about to be "
    "dropped, knees barely supporting weight, the weariness of years "
    "visible in stance, unevenly lit by dying light"
)
_IMPERFECTION_WITNESSED = (
    "single figure in vast empty space, posture of one who has seen too "
    "much, eyes that look past the viewer, imperfect symmetry, frozen "
    "mid-motion as if forgotten how to step forward"
)

_IMPERFECTION_CUES: list[tuple[str, str]] = [
    ("grief",        _IMPERFECTION_GRIEF),
    ("loss",         _IMPERFECTION_GRIEF),
    ("mourning",     _IMPERFECTION_GRIEF),
    ("aftermath",    _IMPERFECTION_AFTERMATH),
    ("haunting",     _IMPERFECTION_AFTERMATH),
    ("hollow",       _IMPERFECTION_AFTERMATH),
    ("irreversible", _IMPERFECTION_AFTERMATH),
    ("weary",        _IMPERFECTION_AFTERMATH),
    ("severed",      _IMPERFECTION_AFTERMATH),
    ("witnessed",    _IMPERFECTION_WITNESSED),
    ("abandoned",    _IMPERFECTION_WITNESSED),
    ("unresolved",   _IMPERFECTION_WITNESSED),
]


def _lookup_imperfection_cue(mood: str) -> str:
    """
    Substring-match scene.mood against _IMPERFECTION_CUES. Returns the cue
    string on first match; empty string when no match (graceful fallback —
    standard composition only, no cue append, no negative split).
    """
    if not mood:
        return ""
    mood_lower = mood.lower()
    for keyword, cue in _IMPERFECTION_CUES:
        if keyword in mood_lower:
            return cue
    return ""

# 3 compositional angles per scene — gives genuine visual variety.
# "dramatic close-up" was walked back to "medium close-up" on 2026-05-14
# after the Karna-arc local test shipped 7/10 frames with dead-eye / black-
# void pupils. FLUX-schnell at 4 steps cannot resolve eye micro-detail when
# the eye fills > ~15% of the frame — medium close-up keeps the emotional
# punch but gives the model enough pixel budget to actually render iris/
# pupil structure.
# ─── Shot composition templates (2026-05-17, Phase 1 cinematic upgrade) ──
# Each tuple is (angle_label, composition_directive). composition_directive
# injects framing/composition guidance BEFORE the scene prompt, so the
# LLM's subject content gets framed differently per shot instead of all
# three landing as centered emotional portraits.
#
# Replaces the prior _SHOT_ANGLES (which was just angle prefixes with no
# real composition variety — all 3 shots ended up portraits because the
# LLM-generated scene prompt always centered on a named character).
#
# Anti-portrait language is intentionally aggressive: diffusion models
# default hard to faces on named mythological subjects, so weak directives
# get overridden. Every prohibition is paired with a positive replacement
# (AVOID X / PREFER Y) to keep prompts stable — diffusion stacks become
# unpredictable under negation-only constraint stacks.
_SHOT_COMPOSITIONS = [
    (
        "ENVIRONMENT WIDE SHOT, ",
        # Shot 1: ESTABLISHING — environment + scale + READABLE subject.
        # The single biggest fix for the "80% close-up portrait" problem.
        # Critical: subject must remain readable on a phone screen — a
        # tiny indistinct dot reads as landscape photography, not myth.
        "wide cinematic landscape. camera pulled FAR BACK. wide focal length. "
        "atmospheric depth and scale. landscape dominant. "
        "if a character is present: FULL BODY visible, READABLE SILHOUETTE "
        "on a phone screen (recognizable shape — armor, posture, weapon — "
        "NOT a tiny indistinct dot), occupying roughly 1/4 to 1/3 of frame "
        "height. subject grounded in the environment. ENVIRONMENT DOMINANT — "
        "show the SETTING explicitly (battlefield, palace court, forest, "
        "sky, terrain). subject may be partially obscured by terrain, "
        "foliage, banners, smoke, or architecture. "
        "AVOID: facial closeup, head-and-shoulders portrait, eye-level "
        "camera. PREFER: wide vista, environmental storytelling, scale. ",
    ),
    (
        "DYNAMIC SHOT, ",
        # Shot 2: ACTION / ASYMMETRY — low-angle, off-center, foreground depth.
        "low-angle dramatic perspective. extended foreground with layered "
        "depth. FULL BODY or three-quarter body visible (waist up at "
        "minimum, NOT head-and-shoulders). subject placed in the LEFT "
        "THIRD or RIGHT THIRD of frame, NEVER centered. strong foreground "
        "element partially occluding view (weapon, hand, banner, smoke, "
        "chariot wheel, horse mane, courtier silhouette). asymmetric "
        "composition. strong sense of motion or tension. "
        "AVOID: centered portrait, head-and-shoulders framing, eye-level "
        "camera. PREFER: low-angle dynamic, off-center placement, "
        "foreground-mid-background layering. ",
    ),
    (
        "EMOTIONAL CLOSE-UP, ",
        # Shot 3: EMOTIONAL — the existing strength, kept as-is.
        # This is the one shot per scene where face-dominance is welcome.
        "medium close-up, head-and-shoulders, eyes readable, dramatic "
        "side lighting, emotional weight on face. ",
    ),
]


# ─── Explainer visual_track categories (v2) ──────────────────────────────
# When a scene has a `visual_track` list (explainer series only), the LLM
# supplies one prompt per shot with a category tag. Each category prepends
# its own composition directive so FLUX gets a consistent framing per type.
# This is what shifts the channel from 80% portrait montage to investigative
# systems-thinking visuals.
_CATEGORY_PREFIX = {
    "human":    "close-up portrait, single human subject, dramatic single-source lighting, shallow depth of field, intense expression, ",
    "system":   "wide architectural photograph of large-scale infrastructure, no humans visible, no text, no logos, dark moody atmosphere, ",
    "symbolic": "conceptual editorial photograph, single iconic object as metaphor, no humans, no text, dramatic isolation lighting, ",
    "ui":       "stylized vertical screen UI mock, dark interface, looks like a captured frame from an investigative documentary or news terminal, minimal blurred body text, no logos, no readable proper-noun strings, ",
}


# ─── Hook visual override (Phase 2, 2026-05-17) ──────────────────────────
# Scene-0-shot-0 gets a mood-routed "scroll-stopping" cinematic moment
# instead of the generic environment wide.
#
# Architecture (refactored 2026-05-17 hot patch after first test render):
#   _HOOK_PROMPTS    — one composition prompt per INTENSITY TIER
#   _HOOK_KEYWORDS   — many substring triggers → tier mapping (deduped)
#   _lookup_hook_visual(mood, image_prompt) — checks BOTH fields
#
# Why two sources? The LLM produces emotion-tone moods ("Foreboding,
# solemn, conflicted") more often than event-type moods ("battle, oath").
# The literal event lives in scene[0]['image_prompt'] ("taking a vow with
# raised hand"). Checking both fields raises hit rate substantially.
#
# Intensity tiers prevent tonal dissonance — grief narration with an
# explosive battlefield hook breaks immersion. Subject placement biases
# toward the UPPER-MIDDLE THIRD of frame so the hook's visual punch
# survives YouTube Shorts UI overlays (bottom 25% captions/subscribe;
# top ~10% title).

_HOOK_PROMPTS: dict[str, str] = {
    "explosive": (
        "burning battlefield silhouette at golden-hour dawn. arrows mid-flight "
        "in the UPPER MIDDLE of the frame. war banners against dark smoke. "
        "cinematic high-contrast lighting. distant army silhouettes positioned "
        "in the upper-middle third (above the Shorts UI safe zone). no "
        "central character. PREFER: scale, scope, motion. AVOID: portrait, face. "
    ),
    "solemn-quiet": (
        "low-angle dramatic figure raising hand in solemn oath, raised arm "
        "and face positioned in the UPPER MIDDLE of the frame (above Shorts "
        "UI safe zone). rim-lit by torchlight in cavernous palace court. "
        "courtiers as silhouettes in the foreground. stillness, weight, "
        "ceremonial gravity. no swords drawn. "
    ),
    "haunting-quiet": (
        "single readable figure silhouetted in vast empty space, positioned in "
        "the UPPER MIDDLE of the frame (above Shorts UI safe zone), recognizable "
        "as a person on a phone screen but small relative to the vastness "
        "around. wind-blown ash or rain falling slowly. dramatic god-rays through "
        "fog. NO action. NO weapons. NO movement implied. emotional stillness, "
        "haunting silence. "
    ),
    "tense-action": (
        "extreme close-up of weapon being drawn in slow motion, weapon body "
        "anchored across the UPPER MIDDLE of the frame (above Shorts UI safe "
        "zone). sparks, fire reflection on steel. tight focus on metal. no face. "
    ),
    "awe-scale": (
        "celestial vista with god-rays piercing dark clouds, the iconography "
        "placed in the UPPER MIDDLE of the frame (above Shorts UI safe zone). "
        "distant mythological iconography (chakra, conch, lotus). vast scale. "
        "no human figure. silent awe. "
    ),
    "charged-still": (
        "two figures in tense stillness, both heads positioned in the UPPER "
        "MIDDLE of the frame (above Shorts UI safe zone) — one in foreground "
        "looking away, one in background lit by torch. no movement implied. "
        "charged silence between them. dramatic side lighting. NO weapons drawn. "
    ),
}

# Substring keyword → intensity tier. First-match-wins; ordered so the
# most-impactful tiers (explosive scale) match before generic tones. The
# expanded vocabulary catches the round-1 test-run gap where moods like
# "Foreboding, solemn, conflicted" missed every keyword and fell back to
# standard wide. Now those route to solemn-quiet (oath-style hook).
_HOOK_KEYWORDS: list[tuple[str, str]] = [
    # === EXPLOSIVE (battle / war / large-scale action) ===
    ("battle", "explosive"),
    ("war", "explosive"),
    ("kurukshetra", "explosive"),
    ("chaotic", "explosive"),         # NEW — LLM often uses this for war scenes

    # === SOLEMN-QUIET (oath / ceremony / weight without action) ===
    ("oath", "solemn-quiet"),
    ("vow", "solemn-quiet"),
    ("promise", "solemn-quiet"),
    ("foreboding", "solemn-quiet"),   # NEW — caught the #4 vow-scene miss
    ("solemn", "solemn-quiet"),       # NEW
    ("fated", "solemn-quiet"),        # NEW
    ("resolute", "solemn-quiet"),     # NEW

    # === HAUNTING-QUIET (grief / loss / death / mourning / suffering) ===
    # NO weapons / NO motion / NO action — emotional stillness only.
    ("grief", "haunting-quiet"),
    ("loss", "haunting-quiet"),
    ("death", "haunting-quiet"),
    ("mourning", "haunting-quiet"),
    ("sorrow", "haunting-quiet"),     # also matches "sorrowful" via substring
    ("suffering", "haunting-quiet"),  # NEW
    ("painful", "haunting-quiet"),    # NEW
    ("resignation", "haunting-quiet"),# NEW

    # === TENSE-ACTION (rage / fury / fierce drawn weapon) ===
    ("rage", "tense-action"),
    ("fury", "tense-action"),
    ("fierce", "tense-action"),       # NEW

    # === AWE-SCALE (divine / revelation / cosmic / epic) ===
    ("divine", "awe-scale"),
    ("revelation", "awe-scale"),
    ("cosmic", "awe-scale"),
    ("epic", "awe-scale"),            # NEW
    ("awe", "awe-scale"),             # NEW — also matches "awe-struck"
    ("grand", "awe-scale"),           # NEW
    ("inspiring", "awe-scale"),       # NEW

    # === CHARGED-STILL (betrayal / deception / dilemma / conflicted) ===
    ("betrayal", "charged-still"),
    ("deception", "charged-still"),
    ("dilemma", "charged-still"),
    ("conflicted", "charged-still"),  # NEW
    # NOTE: "tense" deliberately NOT included — too generic, would override
    # more specific matches in mixed moods like "Resolute, tense, grand".
]


def _lookup_hook_visual(mood: str, image_prompt: str = "") -> tuple[str, str] | None:
    """
    Match scene[0]['mood'] substring against _HOOK_KEYWORDS. Falls back
    to scene[0]['image_prompt'] when no mood match — the LLM frequently
    generates emotional-tone moods that miss the keyword list while the
    literal event keyword lives in the image_prompt. Both sources raise
    hit rate substantially (round-1 test had 0% hit; round-2 covers vow/
    battle/grief scenes via either source).

    Returns (intensity_tier, hook_prompt) on match, None on no match.
    Caller falls through to standard ENVIRONMENT WIDE SHOT when None.
    """
    for source in (mood, image_prompt):
        if not source:
            continue
        source_lower = source.lower()
        for keyword, intensity in _HOOK_KEYWORDS:
            if keyword in source_lower:
                return intensity, _HOOK_PROMPTS[intensity]
    return None


# Legacy alias kept for any external import that referenced the old name.
# Refactored 2026-05-17 hot patch: actual logic lives in _HOOK_PROMPTS +
# _HOOK_KEYWORDS + _lookup_hook_visual() above. This empty list is just
# a "do not crash if anyone imports the old symbol" stub.
_HOOK_VISUALS: list[tuple[str, str, str]] = []

# Divine non-golden-skin characters — when one of these is referenced anywhere
# in an image prompt, swap from STYLE_SUFFIX_MORTAL to STYLE_SUFFIX_DIVINE so
# the global golden-bronze anchor doesn't override their canonical color.
#   Krishna  — dark indigo-blue divine skin
#   Hanuman  — red-gold / red-orange divine form
# Barbarik is NOT in this set — he renders fine on the warrior-golden path.
# Substring scan (case-insensitive) catches both primary and secondary roles,
# e.g. "Karna and Krishna two-shot" should still trigger divine mode because
# Krishna's blue is more sensitive to override than Karna's bronze.
_DIVINE_NON_GOLDEN_CHARACTERS = {"Krishna", "Hanuman"}


# Load character reference descriptions once at import time
_CHAR_FILE = os.path.join(os.path.dirname(__file__), "..", "assets", "characters.json")
try:
    with open(_CHAR_FILE, encoding="utf-8") as _f:
        _CHARACTERS: dict = {
            k: v for k, v in json.load(_f).items() if not k.startswith("_")
        }
except Exception:
    _CHARACTERS = {}


def _primary_character(prompt: str) -> str:
    """
    Returns the FIRST known character name found in the prompt (case-insensitive),
    or empty string if none found. Used to derive a stable per-character seed
    so the same character renders with a similar face across all scenes of a
    video — eliminates the worst face-drift problem (Karna looking like four
    different actors across six scenes). The "first" character takes priority
    because image prompts typically lead with the scene's hero ("Karna sits...",
    "Indra appears..."), and downstream characters are usually secondary.

    Searches both _CHARACTERS (15 heroes with full visual descriptions) AND
    _KNOWN_NAMES (the broader Mahabharat character list including Indra,
    Surya, etc.). Seed-stability needs only the *name*, not a visual
    description, so secondary characters still get consistent faces across
    scenes.
    """
    prompt_lower = prompt.lower()
    earliest_pos = len(prompt_lower) + 1
    earliest_name = ""
    # Combined candidate pool — dedupe via set, preserve original casing
    candidates = set(_CHARACTERS.keys()) | set(_KNOWN_NAMES)
    for name in candidates:
        pos = prompt_lower.find(name.lower())
        if 0 <= pos < earliest_pos:
            earliest_pos = pos
            earliest_name = name
    return earliest_name


def _char_stable_seed(character: str, shot_index: int) -> int:
    """
    Deterministic seed derived from character name + shot index. Same character
    → same seed across all scenes → FLUX-schnell renders a visually similar
    face. The shot_index offset gives each angle a *related-but-not-identical*
    seed so the wide / medium / close-up shots don't look like the same crop
    of one image. Empty character falls back to scene-position seeding.
    """
    if not character:
        return 0
    h = 0
    for ch in character:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (h + shot_index * 17) % 99991  # prime modulus for spread


def _inject_characters(prompt: str) -> str:
    """
    Scans the prompt for known Mahabharata character names and appends
    their detailed visual description so every image is visually consistent.
    """
    if not _CHARACTERS:
        return prompt
    injected = []
    prompt_lower = prompt.lower()
    for name, data in _CHARACTERS.items():
        if name.lower() in prompt_lower:
            # 250 char cap — enough for skin/eyes/clothing/jewelry/posture
            # without drowning the scene-specific prompt. The earlier 120 cap
            # produced one-line descriptors that the strong style suffix
            # routinely overpowered.
            visual = data.get("visual", "")[:250]
            injected.append(visual)
    if injected:
        return prompt + ". CHARACTER DETAILS — " + "; ".join(injected)
    return prompt


# Known Mahabharata characters to watch for in scripts
_KNOWN_NAMES = [
    "Ashwatthama", "Nakula", "Sahadeva", "Bhima", "Bheema",
    "Jayadratha", "Gandhari", "Madri", "Subhadra", "Uttara",
    "Ghatotkacha", "Hidimba", "Jarasandha", "Shishupala",
    "Sanjaya", "Kripa", "Kritavarma", "Vikarna", "Dushasana",
    "Satyavati", "Parashurama", "Narada", "Panchali",
    "Yuyutsu", "Virata", "Drupada", "Dhrishtadyumna",
    "Shikhandi", "Amba", "Ambika", "Ambalika", "Pandu",
    "Chitrangada", "Urvashi", "Menaka", "Indra", "Surya",
]


def update_characters(script_data: dict) -> list:
    """
    Scans the generated script for Mahabharata characters not yet in
    characters.json, generates visual descriptions via Gemini, and
    saves them back to the file. Returns list of newly added names.
    """
    if not script_data or "scenes" not in script_data:
        return []

    all_text = " ".join(
        scene.get("image_prompt", "") + " " + scene.get("narration", "")
        for scene in script_data["scenes"]
    ).lower()

    existing_lower = {k.lower() for k in _CHARACTERS}
    new_names = [
        name for name in _KNOWN_NAMES
        if name.lower() in all_text and name.lower() not in existing_lower
    ]

    if not new_names:
        return []

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return []

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = (
            f"Generate visual descriptions for these Mahabharata characters "
            f"for AI image generation: {new_names}\n\n"
            "For each character return a JSON object:\n"
            '{"CharacterName": {"visual": "specific physical appearance under 120 chars", '
            '"colors": "primary color palette"}}\n'
            "Be specific: clothing, weapons, jewelry, skin tone, hair style. "
            "Return valid JSON only, no markdown."
        )

        resp = model.generate_content(prompt)
        text = re.sub(r"^```(?:json)?\s*", "", resp.text.strip())
        text = re.sub(r"\s*```$", "", text)
        new_data = json.loads(text)

        with open(_CHAR_FILE, encoding="utf-8") as f:
            existing = json.load(f)

        added = []
        for name, data in new_data.items():
            if name not in existing:
                existing[name] = data
                _CHARACTERS[name] = data
                added.append(name)

        if added:
            with open(_CHAR_FILE, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)
            print(f"    [OK] New characters added to file: {added}")

        return added

    except Exception as e:
        print(f"    [!] Character auto-update skipped: {e}")
        return []


# CF FLUX-schnell hard limit: `/prompt` must be ≤ 2048 chars. We cap at
# 2000 to leave headroom for the ", " separator between mood prefix and
# the rest. Pollinations + HF have higher limits but capping at 2000 is
# safe everywhere.
#
# 2026-05-18 fix: scene 2+ renders were getting 400 "Length of '/prompt'
# must be <= 2048" because Phase 1 composition directives + Phase 4
# character anchors + style suffix pushed combined prompt to 2147-2731
# chars. Truncation strategy preserves the HIGH-VALUE anchors (mood,
# style_suffix) in full and trims only from the END of the variable
# `prompt` portion (composition directive + scene content + character
# injection) at the nearest clean sentence boundary.
_CF_PROMPT_MAX_CHARS = 2000


def _build_full_prompt(prompt: str, mood: str = "", style_suffix: str = STYLE_SUFFIX) -> str:
    mood_prefix = f"{mood}, " if mood else ""
    # Fixed components that always survive: mood_prefix + style_suffix.
    fixed_cost = len(mood_prefix) + len(", ") + len(style_suffix)
    available_for_prompt = _CF_PROMPT_MAX_CHARS - fixed_cost - 5  # 5-char safety

    if len(prompt) > available_for_prompt:
        # Truncate the variable prompt portion at the last clean sentence/
        # clause boundary (". " or ", ") within the available budget, so we
        # don't cut mid-word. Keep at least 60% to preserve scene content.
        cap = prompt[:available_for_prompt]
        for sep in (". ", ", "):
            idx = cap.rfind(sep)
            if idx > available_for_prompt * 0.6:
                cap = cap[:idx]
                break
        prompt = cap.rstrip(' ,.')

    return f"{mood_prefix}{prompt}, {style_suffix}"


# ── Provider cascade ─────────────────────────────────────────────────────────
# Order: Hugging Face FLUX-schnell -> Cloudflare FLUX-schnell -> Pollinations.
# Each provider returns raw image bytes or raises. First success wins.
# Free-tier only — no paid keys involved.

_HF_MODEL = "black-forest-labs/FLUX.1-schnell"
# HF migrated from api-inference.huggingface.co to the inference router.
_HF_URL   = f"https://router.huggingface.co/hf-inference/models/{_HF_MODEL}"
_MIN_BYTES = 5000   # below this, treat response as a corrupted/empty image


def _ensure_dims(img_bytes: bytes, width: int, height: int) -> bytes:
    """If the returned image isn't the requested size, resize via Pillow."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        if img.size == (width, height):
            return img_bytes
        # Cover-fit: fill target then center-crop to avoid letterboxing
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        new_w, new_h = int(src_w * scale + 0.5), int(src_h * scale + 0.5)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - width)  // 2
        top  = (new_h - height) // 2
        img = img.crop((left, top, left + width, top + height))
        if img.mode != "RGB":
            img = img.convert("RGB")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=92)
        return out.getvalue()
    except Exception:
        return img_bytes


def _gen_hf(prompt: str, seed: int, width: int, height: int,
            negative: str = _NEGATIVE_DEFAULT) -> bytes:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "image/jpeg",
        "x-wait-for-model": "true",
    }
    body = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
            "num_inference_steps": 4,
            "seed": seed,
            "negative_prompt": negative,
        },
    }
    resp = requests.post(_HF_URL, headers=headers, json=body, timeout=60)
    if resp.status_code != 200 or len(resp.content) < _MIN_BYTES:
        raise RuntimeError(f"hf status={resp.status_code} bytes={len(resp.content)}")
    ctype = resp.headers.get("content-type", "")
    if not ctype.startswith("image/"):
        raise RuntimeError(f"hf non-image response: {ctype}")
    return _ensure_dims(resp.content, width, height)


def _cloudflare_accounts() -> list[tuple[str, str, str]]:
    """
    Return all configured Cloudflare (label, account_id, api_token) triples
    in cascade order. Primary first, then numbered fallbacks _2, _3, ... .
    Empty / unset entries are skipped. Same shape as _gemini_keys() in
    script_generator.py — when the primary account exhausts its 10k-neuron/day
    free quota with a 429 (or similar quota error), the cascade walks to the
    next account immediately so production never falls to Pollinations
    (which produces visibly muddier output) while quota is available on
    another account.
    """
    out: list[tuple[str, str, str]] = []
    pid = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    ptk = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if pid and ptk:
        out.append(("cf-primary", pid, ptk))
    for n in range(2, 6):
        aid = os.environ.get(f"CLOUDFLARE_ACCOUNT_ID_{n}", "").strip()
        atk = os.environ.get(f"CLOUDFLARE_API_TOKEN_{n}", "").strip()
        if aid and atk and (aid, atk) not in [(a, t) for _, a, t in out]:
            out.append((f"cf-acct-{n}", aid, atk))
    return out


def _cf_is_quota_error(status: int, body_text: str) -> bool:
    """
    Distinguish quota exhaustion (try next account) from transient errors
    (try same account on next pipeline retry). Cloudflare returns 429 for
    rate-limit AND for daily-neuron-cap; both warrant fast-fail to next
    account. 401/403 = bad token (try next account). 5xx = transient (still
    try next account since the alternative is Pollinations).
    """
    if status in (429, 401, 403):
        return True
    low = body_text.lower()
    if "neuron" in low and ("limit" in low or "quota" in low or "exceeded" in low):
        return True
    if "rate" in low and "limit" in low:
        return True
    return False


def _gen_cloudflare(prompt: str, seed: int, width: int, height: int,
                    negative: str = _NEGATIVE_DEFAULT) -> bytes:
    """
    Multi-account FLUX-schnell call — branding-style flow (2026-05-18 rewrite).

    Strategy: ONE attempt per account, walk to next account on ANY failure.
    No within-account retry loops. This mirrors generate_branding.py which
    has been calling the SAME CF endpoint reliably while this pipeline kept
    tripping CF's per-IP burst rate limit.

    Why the rewrite:
      The previous retry logic (2-3 attempts per account with backoff sleeps)
      was the threat-model-mismatch case. CF's modern per-IP burst detection
      treats rapid retries as abuse — even spaced 5s apart. The retry loops
      were creating the very problem they were meant to mitigate.

      Multiple identical-environment runs (workflow_dispatch 25995643033,
      25996494447, and local re-renders) reliably failed after the first
      successful CF call. Single isolated probes succeeded in 2-3s during
      those same windows — confirming CF service was fine, the request
      pattern was the issue.

    Per-account flow now:
      • ONE POST attempt (120s timeout — let CF be slow if it wants)
      • Quota error  → log EXHAUSTED, try next account
      • Any other failure → log status/error, try next account
      • Success → return immediately (deterministic by seed, no retry needed)

    Body includes width + height so CF generates at native target dimensions
    (768x1344) instead of default 1024x1024 + post-crop. Marginally sharper.

    steps=8 is the free-tier max for flux-schnell. CF_TIMEOUT_S env-tunable.
    """
    accounts = _cloudflare_accounts()
    if not accounts:
        raise RuntimeError("no CLOUDFLARE_ACCOUNT_ID/API_TOKEN pairs configured")

    body = {
        "prompt":          prompt,
        "negative_prompt": negative,
        "steps":           8,
        "seed":            seed,
        "width":           width,
        "height":          height,
    }
    try:
        timeout_s = float(os.environ.get("CF_TIMEOUT_S", "120"))
    except ValueError:
        timeout_s = 120.0

    last_err = "no accounts attempted"
    for label, account_id, token in accounts:
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/ai/run/@cf/black-forest-labs/flux-1-schnell"
        )
        headers = {"Authorization": f"Bearer {token}"}
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=timeout_s)
        except Exception as net_err:
            last_err = f"{label} network: {net_err}"
            print(f"    [{label}] network error — trying next account: {str(net_err)[:80]}")
            continue

        if resp.status_code != 200:
            last_err = f"{label} status={resp.status_code} body={resp.text[:120]}"
            if _cf_is_quota_error(resp.status_code, resp.text):
                print(f"    [{label}] EXHAUSTED (status={resp.status_code}) — trying next account")
            else:
                print(f"    [{label}] status={resp.status_code} — trying next account")
            continue

        try:
            data = resp.json()
        except Exception as je:
            last_err = f"{label} bad json: {je}"
            print(f"    [{label}] bad JSON — trying next account")
            continue

        if not data.get("success"):
            errs = data.get("errors") or []
            err_str = str(errs).lower()
            last_err = f"{label} success=false: {errs}"
            if any(k in err_str for k in ("quota", "neuron", "rate", "limit", "exceeded", "allocation")):
                print(f"    [{label}] EXHAUSTED (success=false) — trying next account")
            else:
                print(f"    [{label}] success=false — trying next account: {str(errs)[:80]}")
            continue

        img_b64 = data.get("result", {}).get("image", "")
        if not img_b64:
            last_err = f"{label} missing result.image"
            print(f"    [{label}] no image in result — trying next account")
            continue

        raw = base64.b64decode(img_b64)
        if len(raw) < _MIN_BYTES:
            last_err = f"{label} image too small ({len(raw)} bytes)"
            print(f"    [{label}] image too small — trying next account")
            continue

        return _ensure_dims(raw, width, height)

    print(f"    [cf-cascade] ALL {len(accounts)} ACCOUNT(S) EXHAUSTED — falling through to HF/Pollinations")
    raise RuntimeError(f"cloudflare cascade exhausted: {last_err}")


def _gen_pollinations(prompt: str, seed: int, width: int, height: int,
                      negative: str = _NEGATIVE_DEFAULT) -> bytes:
    encoded  = quote(prompt)
    negative = quote(negative)
    # model=flux: bare FLUX-schnell, no postprocessing filter applied.
    # Was model=flux-realism — that variant adds a "realism" LoRA + heavy
    # warm/saturated filter on top of FLUX-schnell, which is what gave the
    # earlier Pollinations-rendered scenes the muddy plastic-CGI look the
    # user wanted to move away from (Anger's Fire register). Bare flux
    # produces output closer to HF FLUX-schnell — sharper, cleaner, no
    # forced color wash. Switch is reversible via env or one-line revert.
    pollinations_model = os.environ.get("POLLINATIONS_MODEL", "flux").strip() or "flux"
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&seed={seed}"
        f"&model={pollinations_model}&nologo=true&enhance=true&negative={negative}"
    )
    resp = requests.get(url, timeout=45)
    if resp.status_code != 200 or len(resp.content) < _MIN_BYTES:
        raise RuntimeError(f"pollinations status={resp.status_code} bytes={len(resp.content)}")
    return resp.content


def _correct_warm_cast(img_bytes: bytes, provider: str) -> bytes:
    """
    Neutralize the magenta/orange warm wash that Cloudflare FLUX-schnell and
    Pollinations consistently produce on this pipeline. Prompt-based anti-
    cast anchors (`no magenta cast`, `no warm filter` in _NEGATIVE) have
    repeatedly failed across multiple test runs — the bias is at the pixel
    level on these distilled FLUX variants, not at the prompt level.

    Three-filter ffmpeg pipeline applied via stdin/stdout pipes:
      - colortemperature mix=0.5 toward 5200K — cools the warm wash
      - eq saturation=0.88 gamma=1.02 — desat slightly + lift midtones
      - unsharp 5:5:0.4 — compensate for blurry backgrounds

    HF FLUX-schnell output is bypass-skipped — when HF quota becomes
    available again (or with HF Pro), its renders don't have the cast and
    we don't want to double-correct.

    Intensity is gated by IMAGE_COLOR_CORRECT_INTENSITY env var
    (default 1.0; set 0.0 to disable entirely without code change).

    Any ffmpeg failure returns the original bytes unchanged — never crashes
    the pipeline.
    """
    if "hf-flux" in provider.lower():
        return img_bytes

    try:
        intensity = float(os.environ.get("IMAGE_COLOR_CORRECT_INTENSITY", "1.0"))
    except ValueError:
        intensity = 1.0
    if intensity <= 0:
        return img_bytes

    # Scale the three filter strengths by intensity. At intensity=1.0 these
    # match the values that produced visible improvement in local A/B testing.
    temp_mix = max(0.0, min(1.0, 0.5 * intensity))
    sat      = 1.0 - (0.12 * intensity)
    gamma    = 1.0 + (0.02 * intensity)
    sharpen  = 0.4 * intensity

    vf = (
        f"colortemperature=temperature=5200:mix={temp_mix:.2f},"
        f"eq=saturation={sat:.2f}:gamma={gamma:.2f},"
        f"unsharp=5:5:{sharpen:.2f}"
    )

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", "pipe:0",
                "-vf", vf,
                "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2",
                "pipe:1",
            ],
            input=img_bytes,
            capture_output=True,
            timeout=15,
        )
        if result.returncode == 0 and result.stdout and len(result.stdout) > _MIN_BYTES:
            return result.stdout
        # ffmpeg failed (non-zero exit or tiny output) — fall through, keep original
        return img_bytes
    except Exception:
        # subprocess timeout / OS error / etc. — never crash, just skip correction
        return img_bytes


# Per-run image-provider tally — incremented on every successful return
# from generate_image_bytes(). main.py calls reset_provider_tally() at the
# start of each video and prints the tally at the end via [image-summary].
# Tracks how many scenes fell to Pollinations so silent CF-quota degradation
# (the #4 Bhishma Kurukshetra symptom) becomes immediately visible.
_PROVIDER_TALLY: dict[str, int] = {
    "cloudflare-flux-schnell":    0,
    "hf-flux-schnell":            0,
    "pollinations-flux-realism":  0,
}


def reset_provider_tally() -> None:
    """Reset the per-run provider tally. Call at start of each video."""
    for k in _PROVIDER_TALLY:
        _PROVIDER_TALLY[k] = 0


def get_provider_tally() -> dict[str, int]:
    """Return the current per-run provider tally."""
    return dict(_PROVIDER_TALLY)


def generate_image_bytes(prompt: str, seed: int, width: int, height: int,
                         mood: str = "",
                         style_suffix: str = STYLE_SUFFIX,
                         negative_prompt: str = _NEGATIVE_DEFAULT) -> tuple[bytes, str]:
    """
    Tries Cloudflare (multi-account cascade) -> HF -> Pollinations until one
    returns a usable image. Returns (image_bytes, provider_name).

    Order rationale (2026-05-14):
      1. Cloudflare first — produces the best visible quality on our prompts
         (the v3/v4 smoke-test images that the user approved were all CF).
         With 2-3 configured accounts, the daily 10k-neuron quota stretches
         to 4-6 clean videos/day.
      2. HF FLUX-schnell — backup. Frequently 429s on free tier; FLUX.1-dev
         was deprecated entirely 2026-05-14. Kept in the cascade for any
         account that still has HF Pro / unused quota.
      3. Pollinations — last resort only. Free-unlimited but produces visibly
         muddier output (the v5_pollinations smoke test the user rejected
         was Pollinations-only).

    Post-processing pipeline applied to every successful render:
      - _correct_warm_cast — neutralizes magenta/orange wash (Cloudflare/
        Pollinations only; HF FLUX-schnell bypassed since its output has
        no cast).

    Divine-character override (added 2026-05-14):
      When the caller passed STYLE_SUFFIX_MORTAL (the default for Mahabharata
      + Krishna series) AND the prompt mentions a divine non-golden-skin
      character (Krishna, Hanuman), swap to STYLE_SUFFIX_DIVINE. This stops
      the global golden-bronze skin anchor from fighting Krishna's canonical
      indigo-blue / Hanuman's red-gold. WhatIf suffixes are left alone.
    """
    if style_suffix is STYLE_SUFFIX_MORTAL:
        prompt_lower = prompt.lower()
        if any(name.lower() in prompt_lower for name in _DIVINE_NON_GOLDEN_CHARACTERS):
            style_suffix = STYLE_SUFFIX_DIVINE
    full_prompt = _build_full_prompt(prompt, mood, style_suffix=style_suffix)
    providers = [
        ("cloudflare-flux-schnell", lambda: _gen_cloudflare(full_prompt, seed, width, height, negative=negative_prompt)),
        ("hf-flux-schnell",         lambda: _gen_hf(full_prompt, seed, width, height, negative=negative_prompt)),
        ("pollinations-flux-realism", lambda: _gen_pollinations(full_prompt, seed, width, height, negative=negative_prompt)),
    ]
    last_err = None
    for name, fn in providers:
        try:
            data = fn()
            data = _correct_warm_cast(data, name)
            _PROVIDER_TALLY[name] = _PROVIDER_TALLY.get(name, 0) + 1
            # Loud warning whenever we didn't use Cloudflare — makes silent
            # quality degradation grep-able in cron logs.
            if name != "cloudflare-flux-schnell":
                print(f"    [FALLBACK ⚠] image generated via {name} (cloudflare unavailable)")
            return data, name
        except Exception as e:
            last_err = f"{name}: {e}"
            continue
    raise RuntimeError(f"all image providers failed; last={last_err}")


def generate_images(scenes: list, single_shot: bool = False, series: str = "mahabharata", visual_style: str = "", ck=None) -> list:
    """
    Generates portrait (768x1344) images per scene.

    Default: 3 shots per scene (wide / medium / closeup) for the static-image
    Ken Burns pipeline.

    With single_shot=True: only the wide-establishing shot is generated. Used
    when the AI-video pipeline is active — I2V models only need a single
    first-frame image, so generating 3 wastes time and Pollinations quota.

    series + visual_style select the style suffix and skip Mahabharata-specific
    character injection when generating WhatIf imagery.

    PARTIAL RESUME — if `ck` (CheckpointStore) is provided, this function:
      1. Reads `visuals_partial.json` from the cache on entry. Any scene
         already listed there has its cached shot paths loaded and the
         generation step is skipped for that scene — the run resumes
         mid-batch instead of regenerating everything from scratch.
      2. After each new scene's shots complete (success OR placeholder
         fallback), the shots are immediately saved to the cache and
         `visuals_partial.json` is updated atomically. A cancellation
         mid-batch (e.g. 29-min GHA cap) preserves all work up to the
         last completed scene.

    Returns list[list[str]] — outer index = scene, inner index = shot.
    When `ck` is provided returned paths point into the cache directory;
    otherwise they point into temp/images.
    """
    os.makedirs(f"{_TEMP_ROOT}/images", exist_ok=True)
    scene_groups = []

    # _SHOT_COMPOSITIONS replaced the prior _SHOT_ANGLES (2026-05-17 Phase 1).
    # Each entry is (angle_label, composition_directive). In single_shot
    # mode (I2V path) only the first composition is generated.
    compositions = _SHOT_COMPOSITIONS[:1] if single_shot else _SHOT_COMPOSITIONS
    style_suffix = _resolve_style_suffix(series, visual_style)

    # ── Partial-resume bootstrap ──────────────────────────────────────
    # visuals_partial.json shape: {"<scene_idx>": ["<abs_cache_path_shot0>", ...]}
    # Only honored when the cached files still exist on disk (defensive — a
    # botched artifact restore could leave a manifest pointing at missing
    # files; in that case we regenerate from scratch for safety).
    partial_key = "visuals_partial.json"
    partial: dict = {}
    if ck is not None and ck.has(partial_key):
        try:
            raw = ck.load_json(partial_key)
            for k, v in (raw or {}).items():
                if not isinstance(v, list):
                    continue
                if all(isinstance(p, str) and os.path.exists(p) for p in v) and v:
                    partial[int(k)] = v
            if partial:
                print(f"    [resume] Partial visuals manifest: {len(partial)} of "
                      f"{len(scenes)} scenes already cached — will skip those")
        except Exception as _e:
            print(f"    [resume] Could not load partial manifest: {_e} — regenerating from scratch")
            partial = {}

    for i, scene in enumerate(scenes):
        # ── Resume path ───────────────────────────────────────────────
        if i in partial:
            shot_paths = list(partial[i])
            scene_groups.append(shot_paths)
            print(f"    [resume] Scene {i+1}/{len(scenes)} loaded from partial checkpoint — {len(shot_paths)} shots")
            continue

        # ── Static-asset path (2026-05-14) ────────────────────────────
        # Subscribe-outro scenes carry an `image_path` field pointing at a
        # hand-picked asset in assets/outro/. Skip FLUX entirely and copy
        # the asset to the per-scene output path. Guarantees the channel
        # name overlay / outro composition is exactly what was approved —
        # no FLUX gambling on text rendering ("Vyasa AI" -> "VyssA" garble
        # observed in production on 2026-05-14).
        static_path = scene.get("image_path", "")
        if static_path and os.path.exists(static_path):
            output_path = f"{_TEMP_ROOT}/images/scene_{i:02d}_shot_00.jpg"
            try:
                import shutil
                shutil.copy2(static_path, output_path)
                shot_paths = [output_path]
                if ck is not None:
                    try:
                        cached = ck.save_file(f"visuals/scene_{i:02d}_shot_00.jpg", output_path)
                        shot_paths = [cached]
                        current = ck.load_json(partial_key) if ck.has(partial_key) else {}
                        current[str(i)] = shot_paths
                        ck.save_json(partial_key, current)
                    except Exception as _e:
                        print(f"    [warn] Could not checkpoint static outro asset: {_e}")
                scene_groups.append(shot_paths)
                print(f"    [static] Scene {i+1}/{len(scenes)} — using outro asset {static_path}")
                continue
            except Exception as _e:
                # Fall through to regular generation as a defensive backup
                print(f"    [warn] Static asset copy failed ({_e}) — falling through to FLUX")

        shot_paths = []
        mood = scene.get("mood", "")

        # ── v2 EXPLAINER FAST PATH ───────────────────────────────────────
        # When the script supplies a `visual_track` (explainer series), the LLM
        # has already authored 3 distinct prompts with category tags. Use those
        # directly with category-specific framing instead of the generic
        # _SHOT_COMPOSITIONS triplet. This delivers the visual diversity that
        # the wide/dynamic/closeup pattern can't (every shot was a riff on the
        # same human-centric image_prompt).
        if series == "explainer" and isinstance(scene.get("visual_track"), list) and scene["visual_track"]:
            track = scene["visual_track"]
            for shot_idx, shot in enumerate(track):
                if not isinstance(shot, dict):
                    continue
                cat = shot.get("category", "system")
                subject = (shot.get("prompt") or "").strip()
                if not subject:
                    continue
                framing = _CATEGORY_PREFIX.get(cat, _CATEGORY_PREFIX["system"])
                full_prompt = framing + subject
                # Seed: stable per (scene, shot, category) so re-runs reproduce
                seed = i * 211 + shot_idx * 37 + (hash(cat) & 0xFFF)
                output_path = f"{_TEMP_ROOT}/images/scene_{i:02d}_shot_{shot_idx:02d}.jpg"
                success = False
                for attempt in range(3):
                    try:
                        img_bytes, provider = generate_image_bytes(
                            full_prompt, seed=seed, width=768, height=1344,
                            mood="", style_suffix="",  # explainer has its own LUT downstream
                        )
                        with open(output_path, "wb") as f:
                            f.write(img_bytes)
                        shot_paths.append(output_path)
                        print(f"    [OK] Scene {i+1} shot {shot_idx+1}/3 ({cat}) via {provider}")
                        success = True
                        break
                    except Exception as e:
                        print(f"    [!] Scene {i+1} shot {shot_idx+1} ({cat}) attempt {attempt+1}: {e}")
                    time.sleep((attempt + 1) * 3)
                if not success:
                    _create_placeholder(output_path, i * 3 + shot_idx, series=series)
                    shot_paths.append(output_path)
                    print(f"    [~] Placeholder for scene {i+1} shot {shot_idx+1} ({cat})")
                try:
                    inter_shot_s = float(os.environ.get("INTER_SHOT_COOLDOWN_S", "5.0"))
                except ValueError:
                    inter_shot_s = 5.0
                time.sleep(inter_shot_s)

            # Per-scene checkpoint + skip the legacy loop below
            if ck is not None:
                cached_paths = []
                for j_idx, temp_path in enumerate(shot_paths):
                    ext = os.path.splitext(temp_path)[1] or ".jpg"
                    cache_name = f"visuals/scene_{i:02d}_shot_{j_idx:02d}{ext}"
                    try:
                        cached_paths.append(ck.save_file(cache_name, temp_path))
                    except Exception as _e:
                        print(f"    [warn] Could not checkpoint scene {i+1} shot {j_idx+1}: {_e}")
                        cached_paths.append(temp_path)
                shot_paths = cached_paths
                try:
                    current = ck.load_json(partial_key) if ck.has(partial_key) else {}
                    current[str(i)] = cached_paths
                    ck.save_json(partial_key, current)
                except Exception as _e:
                    print(f"    [warn] Could not update partial manifest for scene {i+1}: {_e}")

            scene_groups.append(shot_paths)
            print(f"    [OK] Scene {i+1}/{len(scenes)} complete via visual_track — {len(shot_paths)} shots")
            continue

        # Phase 2/3 stabilization (2026-05-18): mood-routed imperfection cue.
        # When mood matches a restraint keyword (grief/aftermath/witnessed/etc.)
        # the cue gets APPENDED to the scene's image_prompt and the negative
        # prompt switches to _NEGATIVE_RESTRAINT (skin-condition rejections
        # stripped) so weathered / dust-streaked / red-rimmed anchors can land
        # without getting fought by the negative. Mahabharata only — WhatIf
        # science content does not carry mythology grief moods.
        if series == "mahabharata":
            imperfection_cue = _lookup_imperfection_cue(mood)
        else:
            imperfection_cue = ""
        scene_negative = _NEGATIVE_RESTRAINT if imperfection_cue else _NEGATIVE_DEFAULT
        if imperfection_cue:
            print(f"    [imperfection] scene {i+1} mood='{mood[:40]}' → "
                  f"physical-vulnerability cue + restraint negative")

        # Scene-0 hook override (Phase 2, 2026-05-17): for the FIRST shot of
        # scene 0, swap the standard ENVIRONMENT WIDE composition for a
        # mood-routed "scroll-stopping" hook visual. Subsequent shots of
        # scene 0 + all shots of scenes 1+ use the standard composition list.
        # Only the FIRST shot is hooked — shots 2 and 3 of scene 0 still
        # provide composition variety via the standard table.
        #
        # Falls through to standard ENVIRONMENT WIDE if no mood substring
        # matches (safe fallback). In single_shot/I2V mode the existing
        # behavior is preserved to avoid disturbing the I2V path.
        scene_compositions: list[tuple[str, str]] = list(compositions)
        if single_shot and i == 0:
            # I2V hook — unchanged from prior behavior. The face-mid-emotion
            # cue is designed for I2V's first-frame retention, which is a
            # different optimisation than Ken Burns Phase 2's hook punch.
            scene_compositions = [(
                "MEDIUM CLOSE-UP on face mid-emotion (head and shoulders visible), ",
                "",
            )]
        elif (not single_shot) and i == 0:
            # Pass image_prompt as fallback signal — LLM-generated moods
            # often miss event-type keywords ("Foreboding, solemn, conflicted")
            # while the literal event lives in image_prompt ("taking a vow
            # with raised hand"). Two-source matching dramatically raises
            # hook-hit rate (round-1 test had 0% hit, post-fix targets 80%+).
            hook = _lookup_hook_visual(mood, scene.get("image_prompt", ""))
            if hook is not None:
                intensity, hook_prompt = hook
                # Override ONLY shot 0; keep shots 1 + 2 of scene 0 as the
                # standard dynamic + emotional-closeup compositions so the
                # hook scene still has shot variety internally.
                scene_compositions = [
                    ("HOOK FRAME, ", hook_prompt),
                ] + list(compositions[1:])
                print(f"    [hook] scene 0 shot 0 → {intensity} (mood='{mood[:40]}')")
            # else: scene_compositions stays as the full standard list

        for j, (angle_label, composition_directive) in enumerate(scene_compositions):
            output_path = f"{_TEMP_ROOT}/images/scene_{i:02d}_shot_{j:02d}.jpg"
            # Character injection is Mahabharata-specific (Krishna/Arjuna/etc.
            # visual descriptors); skip it for WhatIf science content.
            raw_prompt = scene["image_prompt"]
            base_prompt = raw_prompt if series == "whatif" else _inject_characters(raw_prompt)
            # Append the imperfection cue AFTER character injection so it
            # rides on top of the character's existing descriptors instead
            # of getting buried by them. Additive — does not replace any
            # part of the scene prompt. Empty cue = no-op.
            if imperfection_cue:
                base_prompt = f"{base_prompt}. {imperfection_cue}"
            prompt = f"{angle_label}{composition_directive}{base_prompt}"
            # Stable per-character seed: same hero across scenes → similar
            # face. Falls back to scene-position seed when no known character
            # is mentioned (WhatIf scenes, environment-only shots).
            hero = "" if series == "whatif" else _primary_character(raw_prompt)
            seed = _char_stable_seed(hero, j) if hero else (i * 137 + j * 31)

            success = False
            for attempt in range(3):
                try:
                    img_bytes, provider = generate_image_bytes(
                        prompt, seed=seed, width=768, height=1344, mood=mood,
                        style_suffix=style_suffix,
                        negative_prompt=scene_negative,
                    )
                    with open(output_path, "wb") as f:
                        f.write(img_bytes)
                    shot_paths.append(output_path)
                    print(f"    [OK] Scene {i+1} shot {j+1}/{len(scene_compositions)} via {provider}")
                    success = True
                    break
                except Exception as e:
                    print(f"    [!] Scene {i+1} shot {j+1} attempt {attempt+1}: {e}")

                wait = (attempt + 1) * 3
                print(f"    Waiting {wait}s...")
                time.sleep(wait)

            if not success:
                _create_placeholder(output_path, i * 3 + j, series=series)
                shot_paths.append(output_path)
                print(f"    [~] Placeholder (outro-asset fallback) for scene {i+1} shot {j+1}")

            # Inter-shot cooldown — give CF's per-IP rate limit time to relax
            # before the next shot's CF call. Bumped from 1s → 5s (2026-05-17)
            # after a sustained burst of CF calls reliably triggered the limit
            # and locked the IP out for the rest of the run. Env-tunable.
            try:
                inter_shot_s = float(os.environ.get("INTER_SHOT_COOLDOWN_S", "5.0"))
            except ValueError:
                inter_shot_s = 5.0
            time.sleep(inter_shot_s)

        # ── Per-scene checkpoint ──────────────────────────────────────
        # Save the just-finished scene to the cache + update partial manifest
        # so the next workflow attempt resumes from here on cancellation.
        if ck is not None:
            cached_paths = []
            for j_idx, temp_path in enumerate(shot_paths):
                ext = os.path.splitext(temp_path)[1] or ".jpg"
                cache_name = f"visuals/scene_{i:02d}_shot_{j_idx:02d}{ext}"
                try:
                    cached_paths.append(ck.save_file(cache_name, temp_path))
                except Exception as _e:
                    print(f"    [warn] Could not checkpoint scene {i+1} shot {j_idx+1}: {_e}")
                    cached_paths.append(temp_path)
            # Replace returned paths with cache paths so downstream uses the cache copy
            shot_paths = cached_paths
            try:
                # Atomic load → update → save (save_json itself is atomic)
                current = ck.load_json(partial_key) if ck.has(partial_key) else {}
                current[str(i)] = cached_paths
                ck.save_json(partial_key, current)
            except Exception as _e:
                print(f"    [warn] Could not update partial manifest for scene {i+1}: {_e}")

        scene_groups.append(shot_paths)
        print(f"    [OK] Scene {i+1}/{len(scenes)} complete — {len(shot_paths)} shots")

    return scene_groups


def _overlay_thumbnail_text(thumb_path: str, overlay_text: str) -> bool:
    """
    Composite a bold Hindi text card onto an existing FLUX-rendered thumbnail.
    Used to add the search-stopping "shock phrase" overlay (e.g. "कर्ण का सच",
    "भीष्म प्रतिज्ञा") that successful Hindi mythology Shorts channels use to
    drive CTR on browse/search/suggested surfaces.

    Added 2026-05-15 after channel analytics showed thumbnails were
    text-free FLUX art — competing channels overlay 2-3 Hindi words for the
    eye-catch. Renders via pipeline.text_renderer (HarfBuzz for Devanagari
    shaping; FLUX can't spell so it's done as a separate raster pass).

    Position: bottom 1/3 of the 1280×720 thumbnail, centered horizontally,
    yellow fill + black outline + soft shadow (the canonical "thumbnail
    text" look). Returns True on success; on any failure the original
    thumbnail is left untouched (defensive — never break the pipeline).
    """
    if not overlay_text or not overlay_text.strip():
        return False
    try:
        from PIL import Image
        from pipeline.text_renderer import render_text_card
        from pipeline.subtitle_generator import FONT_PATH
    except Exception as e:
        print(f"    [!] Thumbnail text overlay deps missing: {e}")
        return False
    if not os.path.exists(thumb_path):
        return False
    try:
        # Auto-size text to fit ~85% of thumbnail width. 1280px wide, target
        # ~1080px text width. For a 4-word Hindi phrase that lands at ~70-90
        # pixels per glyph, font_size ~110-130 works. Start at 130, downscale
        # if too wide.
        text = overlay_text.strip()
        base = Image.open(thumb_path).convert("RGBA")
        W, H = base.size  # 1280, 720
        max_text_w = int(W * 0.88)

        font_size = 130
        for _ in range(5):
            card = render_text_card(
                text, FONT_PATH, font_size=font_size,
                fill=(255, 230, 0, 255),
                outline=(0, 0, 0, 255),
                outline_px=7,
                shadow=(0, 0, 0, 180),
                shadow_offset=(4, 4),
            )
            if card.width <= max_text_w:
                break
            # Too wide; shrink
            font_size = int(font_size * max_text_w / max(card.width, 1))
        if card.width > max_text_w:
            # Last resort: brute-shrink to fit
            ratio = max_text_w / card.width
            card = card.resize(
                (int(card.width * ratio), int(card.height * ratio)),
                Image.LANCZOS,
            )

        # Place at bottom 1/3: y_center = H * 0.72
        x = (W - card.width) // 2
        y = int(H * 0.72) - card.height // 2
        base.paste(card, (x, y), card)
        base.convert("RGB").save(thumb_path, "JPEG", quality=92)
        print(f"    [OK] Thumbnail text overlay: {text!r}")
        return True
    except Exception as e:
        print(f"    [!] Thumbnail text overlay failed: {e}")
        return False


def generate_thumbnail(
    thumbnail_prompt: str,
    output_path: str = "output/thumbnail.jpg",
    series: str = "mahabharata",
    visual_style: str = "",
    overlay_text: str = "",
) -> str:
    """
    Generates a 1280x720 thumbnail (YouTube native size — landscape).

    For WhatIf the thumbnail is the single most clicked-on asset and Pollinations
    output at small sizes tends to be muddy with weak focal hierarchy. We append
    a thumbnail-specific composition rider (centered subject, single high-contrast
    focal point, dark backdrop, no small text) and vary the seed each retry so a
    bad first composition isn't simply repeated four times.

    `overlay_text` (added 2026-05-15): when provided, a bold Hindi text card
    is composited onto the bottom 1/3 of the thumbnail after FLUX renders.
    Used to add the search-stopping shock phrase ("कर्ण का सच", "भीष्म प्रतिज्ञा")
    competing mythology channels use to drive CTR. Empty = no overlay.
    """
    os.makedirs("output", exist_ok=True)
    style_suffix = _resolve_style_suffix(series, visual_style)

    # Thumbnail composition rider — appended to the style suffix only for the
    # landscape thumbnail call, not the scene images. Pushes the model toward
    # phone-feed-readable framing.
    if series == "whatif":
        style_suffix = (
            style_suffix
            + ", thumbnail composition — single centered hero subject filling 60% "
              "of the frame, dramatic single light source, dark moody backdrop "
              "with strong contrast, no small text, no UI elements, no logos, "
              "phone-feed legible at thumbnail size"
        )

    # Vary seeds across attempts so a poor first composition doesn't repeat.
    seeds = [9999, 4242, 8137, 1729]
    for attempt in range(len(seeds)):
        try:
            img_bytes, provider = generate_image_bytes(
                thumbnail_prompt, seed=seeds[attempt], width=1280, height=720,
                style_suffix=style_suffix,
            )
            with open(output_path, "wb") as f:
                f.write(img_bytes)
            print(f"    [OK] Thumbnail generated via {provider} (seed {seeds[attempt]})")
            # Post-render text overlay (Hindi shock phrase via HarfBuzz).
            # Defensive — never crashes the thumbnail step on overlay failure.
            if overlay_text:
                _overlay_thumbnail_text(output_path, overlay_text)
            return output_path
        except Exception as e:
            print(f"    [!] Thumbnail attempt {attempt+1}: {e}")

        wait = (attempt + 1) * 3
        time.sleep(wait)

    print(f"    [ERROR] Thumbnail generation failed after {len(seeds)} attempts")
    return ""


_OUTRO_FALLBACK_BY_SERIES = {
    "mahabharata": "assets/outro/mahabharata.jpg",
    "krishna":     "assets/outro/krishna.jpg",
    "whatif":      "assets/outro/whatif.jpg",
}


def _create_placeholder(output_path: str, index: int, series: str = "mahabharata"):
    """
    Fallback image used when ALL providers (HF + 3 CF accounts + Pollinations)
    fail for a given scene. Reuses the series' hand-picked outro asset so
    the viewer sees ACTUAL imagery instead of a solid color tile.

    2026-05-14 production check (Krishna "Uddhava" video) shipped 5 scenes
    of near-black solid-color tiles when Pollinations 429-stormed during a
    retry — the "video" was effectively just audio over a black screen.
    Reusing the outro asset means worst case = an "outro tableau loop"
    that's visibly mythological even if it's the same image repeated.

    Falls back to the old solid-color tile if the asset is missing on disk.
    """
    import subprocess
    import shutil

    asset = _OUTRO_FALLBACK_BY_SERIES.get(series, "")
    if asset and os.path.exists(asset):
        try:
            shutil.copy2(asset, output_path)
            return
        except Exception:
            pass  # fall through to ffmpeg solid color

    colors = ["#1a0a2e", "#0d1b2a", "#1b2838", "#2d1b69", "#0f0c29"]
    color = colors[index % len(colors)]
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"color={color}:size=768x1344:duration=1",
            "-vframes", "1", output_path,
        ],
        capture_output=True,
    )
