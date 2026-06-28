import asyncio
import requests
import os
import re
import time
import json
import base64
import hashlib
import io
import subprocess
from pathlib import Path
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
    # Phase 13 Royal Glow (2026-06-04) — Mahabharata characters are god-kings
    # and noble dynasties; they should LOOK like the epic they're from. This
    # suffix evolves the prior documentary-realism cues into "god-king majesty"
    # while keeping every carefully-tuned anchor that prevented splotchy skin,
    # dead-doll eyes, and silhouette-dark Karna in earlier iterations.
    # The injured/exile caveat is baked in as a conditional clause so FLUX
    # naturally suppresses the halo when the per-scene image_prompt carries
    # wound/blood/aftermath keywords.
    #
    # Photoreal anchor — kept short for distilled FLUX variants (Pollinations,
    # Cloudflare) which honor long prompts less reliably than HF FLUX-schnell.
    # WALKED BACK from a 2026-05-13 attempt at "physically-based skin with
    # visible pores and fine facial hair" + "ultra-sharp 8K detail throughout"
    # which produced splotchy/scarred face artifacts. The Phase 13 "pristine
    # skin with subtle glow" wording instead leans on lighting cues, not
    # texture, to drive majesty.
    "photoreal cinematic period-film aesthetic, "
    "Mahabharata god-king majesty, "
    "flawless pristine skin with a subtle divine glow "
    "(unless the scene depicts injury, blood, exile, or aftermath — "
    "then maintain gritty/wounded realism over the glow), "
    "natural skin texture, realistic facial features, "
    # Eye-specific anchor — added 2026-05-14 after the Karna-arc local test
    # shipped 7/10 frames with dead-eye / black-void pupils. Distilled FLUX
    # at 4 steps loses eye micro-detail without an explicit eye cue.
    "intense piercing eyes carrying the character's active emotion, "
    "clearly defined iris and pupils, "
    "natural catch-light reflections in the eyes, "
    # Posture + lighting — the literal halo effect.
    "regal posture, high-end epic fantasy film styling, "
    "golden hour rim lighting creating an ethereal halo on hair and shoulders, "
    "85mm portrait, crisp detail on jewelry and fabric, "
    # Face-exposure anchor — added 2026-05-14 after FLUX over-interpreted
    # "oil-lamp" cues and rendered Karna near-black. Canonical Karna is
    # golden-bronze (Surya's son), not silhouette-dark.
    "warm golden-bronze skin tone for Indian characters, "
    "luminous golden glowing complexion, "
    "well-lit faces with key-light on the face, even facial exposure, "
    "ancient India Mahabharat live-action / Baahubali period-film aesthetic, "
    "carved sandstone temple architecture in sharp focus, "
    "balanced natural color grading, neutral whites, true skin tones, "
    "sharp focus on subject, clear facial features, "
    "no global color wash, no orange filter, no magenta or pink cast, "
    "no CGI plastic look, no airbrushed skin, "
    "NO cartoon, NO anime, NO illustration, NO dead doll eyes"
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


# ── Phase 12 per-character palette + lighting (2026-06-03) ──────────────────
# Replaces the hardcoded warm-amber default in script_generator's image_prompt
# template. Each arc gets a distinct visual signature so consecutive uploads
# pass the thumbnail-scroll diversity test — Karna's amber-bronze shadow look
# stops bleeding into Bhishma's cold restraint, Draupadi's firelit fury, etc.
#
# Triggered by: script_generator builds the image_prompt f-string with
# _character_palette_directive(arc_character_devanagari) substituted in.
# Result: every per-scene FLUX prompt carries the character's palette inline.
_CHARACTER_PALETTE = {
    # Phase 19 (2026-06-16) palette rebalance — Heaven's Gate render
    # forensic exposed that "low-key, moral weight, deep shadow, smoke-grey"
    # phrasing pulled FLUX into Eastern Orthodox liturgical / dark-medieval
    # aesthetic. Every entry now ends with explicit "faces clearly lit" so
    # the warm-Indian-skin guard survives FLUX's tendency to render
    # candlelit scenes as silhouettes. Mood preserved (Karna still amber,
    # Bhishma still moonlit, Ashwatthama still fevered) but the dark-fantasy
    # vocabulary is gone.
    "कर्ण": {
        "palette":  "warm sunset amber + bronze, golden divine glow on kavach armor, "
                    "saffron-and-crimson palette, faces clearly lit",
        "lighting": "harsh golden-hour edge-light, embers in foreground, "
                    "faces visible and warm",
    },
    "भीष्म": {
        "palette":  "cool moonlit silver-blue, restrained noble tones, "
                    "white-and-silver silk dhoti, faces clearly lit",
        "lighting": "cold pre-dawn moonlight or candlelit interior, faces "
                    "visible and dignified, no deep shadows",
    },
    "अर्जुन": {
        "palette":  "heroic gold + cobalt, high saturation, golden-bronze skin, "
                    "vital divine charm",
        "lighting": "strong key-light, golden-hour, hero composition with rim-light halo",
    },
    "द्रौपदी": {
        "palette":  "firelit crimson + saffron, regal red-and-gold sari, "
                    "warm Indian skin tone, court grandeur",
        "lighting": "warm torchlight and oil-lamp, contrasted, intense BUT "
                    "faces clearly visible",
    },
    "युधिष्ठिर": {
        # Phase 19 (2026-06-16) — was "solemn candlelit ochre + parchment,
        # contemplative, single oil-lamp source, low-key, moral weight"
        # which FLUX read as Eastern Orthodox liturgical scene. Rebalanced
        # to preserve contemplative tone with explicit Hindu palace anchor.
        "palette":  "warm oil-lamp glow on white-and-gold silk dhoti, "
                    "carved sandstone palace interior, contemplative dignified, "
                    "faces clearly lit with key-light",
        "lighting": "warm oil-lamp + soft fill from skylight, dignified, "
                    "faces visible, NO deep shadows, NO dark fantasy",
    },
    "एकलव्य": {
        "palette":  "forest greens + earth-browns, dappled golden sunlight, "
                    "warm Indian skin tone, untouched by court colors",
        "lighting": "dappled sunlight through banyan canopy, faces clearly lit",
    },
    "अश्वत्थामा": {
        # Phase 19 (2026-06-16) — preserved fever/cursed mood but added
        # explicit "warm Indian skin" anchor so faces don't render dark.
        "palette":  "blood-red + smoke-grey accents on warm bronze skin, fevered, "
                    "cursed but face clearly visible, NOT silhouette",
        "lighting": "flickering firelight with key-light on face, hard shadows "
                    "on armor but face exposed, hellish edge",
    },
}


# ─── Phase 19 (2026-06-16) Wardrobe Context + Hindu Iconography ──────────
# Three layered anchors that combat FLUX's training-data bias toward
# European medieval / dark fantasy when given "Mahabharata" + "warrior":
#   1. _WARDROBE_CONTEXT_PREFIX — clothing + lighting + skin per scene type
#   2. _HINDU_ICONOGRAPHY_ANCHOR — universal Vedic/Pauranic visual lock
#   3. _NEGATIVE_PHASE19_ANTI_BIAS — explicit kills for skull / spike /
#      viking / mosque / Sauron / etc. (appended at bottom of this file)
#
# Failure mode this fixes: 2026-06-16 Heaven's Gate render shipped
# Yudhishthira in skull-emblem breastplate, demonic-spike helmet,
# viking-horn silhouette, Byzantine headband crown, and Christian-Orthodox
# candle staff. Story-critical dog absent from all 6 sampled frames.

# CLOTHING + LIGHTING + SKIN TONE ONLY. No environment / setting / background
# descriptors — those belong to the LLM's image_prompt and would clash with
# the wardrobe prefix in atypical scenes (e.g. river duel = WAR context but
# NOT a dusty battlefield).
_WARDROBE_CONTEXT_PREFIX = {
    "WAR": (
        # Phase 22 (2026-06-25): WAR rewrite — battle damage tokens
        # replace polished-mukut glamour. Phase 19's mukut+kundal+
        # gold-leaf reading was over-curated; forensic showed winners
        # ran ash + torn-dhoti + blood-smear, not mukut + gold-leaf.
        "authentic Vedic-era warrior wearing kavach (engraved chest-plate "
        "with sun or moon motif, NEVER European plate armor, NEVER skull "
        "emblem, NEVER spiked helmet), torn dhoti edges, ash-streaked "
        "face, dust-streaked tilak, blood smear on forearm, frayed "
        "angavastram cape, broken arrow shaft fragment visible, "
        "weathered straps, warm golden-bronze Indian skin tone, "
    ),
    "PALACE": (
        "royal subject in pristine white-and-gold silk dhoti with red "
        "angavastram, classical Indian mukut crown (NEVER European helmet), "
        "tilak on forehead, layered gold necklaces and kundal earrings, "
        "warm oil-lamp glow on warm golden-bronze Indian skin, NO armor, "
        "NO weapons unless ceremonial, "
    ),
    "DIVINE": (
        "subject glowing with ethereal halo of divine charm, wearing "
        "pristine white silk dhoti edged in gold with diaphanous shawl, "
        "celestial mukut, glowing golden-bronze skin, divine golden "
        "light on the face, NO armor, NO darkness, NO shadows, "
    ),
    "FOREST": (
        "subject in simple natural-cotton valkala bark-cloth garments, "
        "hair tied in topknot, rudraksha bead mala around neck, barefoot, "
        "warm dappled-sunlight key-light on face, warm golden-bronze "
        "Indian skin tone, NO crown, NO jewelry except rudraksha, "
    ),
    "JOURNEY": (
        "subject in white-and-gold silk dhoti (NOT armor) with simple "
        "wooden walking staff, companions visible alongside (brothers, "
        "queen Draupadi, loyal stray dog if story-relevant), warm "
        "key-light on faces transitioning to celestial golden glow, "
        "warm golden-bronze Indian skin tone, "
    ),
    "AFTERMATH": (
        # Phase 22 (2026-06-25) — NEW. Required by _check_aftermath_closer
        # for the final broll beat. Forensic showed all 4 winners closed
        # on consequence, not triumph.
        "lone silhouette against fading dusk light, face withheld in "
        "shadow (NOT a portrait, NOT a close-up of the face), helmet "
        "rolled in dust on the ground, abandoned weapon in the "
        "foreground (broken bow / fallen sword / shattered chariot "
        "wheel), prone body partially visible at the edge of frame, "
        "lone diya flickering, low-angle ground-level camera, smoke "
        "and ash hanging in the air, muted desaturated palette, "
    ),
}


def _inject_wardrobe_context(wardrobe_context: str) -> str:
    """Phase 19 (2026-06-16). Return a wardrobe-context-specific positive
    prefix for the LLM-emitted classification. Falls back to a neutral
    "ancient Indian classical" anchor when the context is empty (defensive
    — should never fire if the new validator catches it at script-gen)."""
    ctx = (wardrobe_context or "").strip().upper()
    return _WARDROBE_CONTEXT_PREFIX.get(
        ctx,
        "ancient Indian Mahabharata era, classical Indian wardrobe, "
    )


# Universal anchor prepended to EVERY Mahabharata FLUX prompt.
# Phase 20 (2026-06-17) — photoreal-forward rewrite. Phase 19's
# "Raja Ravi Varma style" + "Amar Chitra Katha" referenced an oil
# painter and a comic-book series — FLUX read them as 2D-style anchors
# and produced cartoonish output. Phase 20 keeps cultural protection via
# the LIVE-ACTION B.R. Chopra Mahabharat 1988 reference + Baahubali (also
# live-action film), explicitly negates 2D in the positive (FLUX weights
# early-token positives heaviest), and front-loads cinematic-camera
# vocabulary that the model has strong training-data signal for.
# Architectural exclusions (NO Islamic domes / NO Gothic / NO European
# castles) live in _NEGATIVE_PHASE19_ANTI_BIAS; 2D-style exclusions live
# in the new _NEGATIVE_PHASE20_ANTI_CARTOON below.
# Phase 22 (2026-06-25) — 3-tier anchor split.
# Phase 23 (2026-06-28) — V2 anchor: STRIP portrait-cinematography tokens
# ("skin pores", "ARRI Alexa 65", "fabric weave" — all instructions to FLUX
# to do facial close-ups) and ADD scene-composition tokens (wide cinematic,
# deep focus, environment readable). Forensic on the first Phase 22 video
# (X-LWlg1DW5s) showed the V1 anchor caused FLUX to render 5/5 portraits.
# The old 500+ view winners (L1ZPCZJLDe0) had multi-character layered
# compositions with readable architecture; their prompts didn't carry
# "skin pores" cinematography baggage.
_HINDU_ICONOGRAPHY_BASE = (
    "Hyper-photorealistic live-action epic film still, 8K resolution, "
    "wide cinematic composition with deep focus across foreground / "
    "mid-ground / background, everything in sharp focus, environment "
    "readable and richly detailed, gritty naturalistic cinematic lighting, "
    "NOT painting, NOT comic book, NOT illustration, NOT 2D art, "
    "Baahubali / B.R. Chopra Mahabharat 1988 live-action reference, "
    "authentic Hindu Vedic civilization, "
)

# Phase 23: scene-descriptor anchors (NOT character-anatomy). The wardrobe
# context determines the SCENE the camera is in — battlefield vs palace
# vs divine realm — not what the character is wearing. Character anatomy
# is handled by _inject_characters_v2 downstream, on a tight ≤45-char
# signature_lock budget.
_HINDU_ICONOGRAPHY_BATTLEFIELD_ANCHOR = _HINDU_ICONOGRAPHY_BASE + (
    "ash and smoke drifting across dust-streaked battlefield, broken "
    "chariot wheels and discarded weapons in mid-ground, distant marching "
    "armies on the horizon, low-angle warm golden-hour light, "
)

_HINDU_ICONOGRAPHY_PALACE_ANCHOR = _HINDU_ICONOGRAPHY_BASE + (
    "ornate Nagara-style palace court with carved stone columns, polished "
    "marble floor with scattered marigold petals, arched windows admitting "
    "morning light, multiple courtiers and elders visible in mid-ground, "
    "warm oil-lamp glow filling the chamber, "
)

_HINDU_ICONOGRAPHY_DIVINE_ANCHOR = _HINDU_ICONOGRAPHY_BASE + (
    "ethereal divine realm with celestial light beams cutting through "
    "swirling mist, towering cosmic architecture, divine radiance filling "
    "the frame, clouds parting to reveal cosmic vistas, "
)

# Phase 23: NEW. AFTERMATH gets its own anchor — the consequence beat
# that closes every Phase-22-compliant render. Wide composition,
# silhouette-friendly framing, abandoned-weapon mid-ground anchor.
_HINDU_ICONOGRAPHY_AFTERMATH_ANCHOR = _HINDU_ICONOGRAPHY_BASE + (
    "lone silhouette against fading dusk sky, abandoned weapons in "
    "foreground (broken bow / fallen sword / shattered chariot wheel), "
    "smoke and ash hanging in the air, low-angle ground-level perspective, "
    "muted desaturated palette with one warm accent (lone diya / "
    "horizon ember), prone body partially visible at the edge of frame, "
)

# Phase 23: NEW. FOREST + JOURNEY share the battlefield-leaning default
# (no ornament). Dedicated scene anchors so the camera sees the forest /
# pilgrimage path, not a character portrait against blurred trees.
_HINDU_ICONOGRAPHY_FOREST_ANCHOR = _HINDU_ICONOGRAPHY_BASE + (
    "dense forest hermitage with banyan canopy and dappled sunlight, "
    "moss-covered stones, simple thatched ashram huts in mid-ground, "
    "wildlife and natural elements visible in deep background, "
)

_HINDU_ICONOGRAPHY_JOURNEY_ANCHOR = _HINDU_ICONOGRAPHY_BASE + (
    "winding mountain pilgrimage path, distant Himalayan peaks on the "
    "horizon, scattered prayer flags and roadside shrines along the way, "
    "companions visible at varying distances along the trail, "
)

# Back-compat alias — default to battlefield (the winners' look).
_HINDU_ICONOGRAPHY_ANCHOR = _HINDU_ICONOGRAPHY_BATTLEFIELD_ANCHOR


def _pick_iconography_anchor(wardrobe_context: str) -> str:
    """Phase 22 (2026-06-25) / Phase 23 (2026-06-28). Route by wardrobe
    context. Each context gets a SCENE descriptor (environment + depth +
    multi-character cues), not character anatomy.
      PALACE     → ornate court + marble + courtiers
      DIVINE     → cosmic realm + light beams + clouds
      AFTERMATH  → silhouette + abandoned weapons + fading dusk
      FOREST     → banyan canopy + ashram huts + dappled sunlight
      JOURNEY    → pilgrimage path + Himalayan peaks
      WAR/default→ ash + smoke + chariot wheels + distant armies
    """
    ctx = (wardrobe_context or "").strip().upper()
    if ctx == "PALACE":
        return _HINDU_ICONOGRAPHY_PALACE_ANCHOR
    if ctx == "DIVINE":
        return _HINDU_ICONOGRAPHY_DIVINE_ANCHOR
    if ctx == "AFTERMATH":
        return _HINDU_ICONOGRAPHY_AFTERMATH_ANCHOR
    if ctx == "FOREST":
        return _HINDU_ICONOGRAPHY_FOREST_ANCHOR
    if ctx == "JOURNEY":
        return _HINDU_ICONOGRAPHY_JOURNEY_ANCHOR
    return _HINDU_ICONOGRAPHY_BATTLEFIELD_ANCHOR


# Phase 23 (2026-06-28) — per-shot-type resolution routing.
# Forensic showed FLUX produced portrait-crop output at 768x1344 even when
# image_prompts described scenes. Wide aspect ratios force FLUX out of
# portrait latent space and into landscape composition. ENVIRONMENT / ACTION
# / PROP shots get 16:9 (1344x768) — these benefit from horizontal sweep.
# REACTION close-ups stay 9:16 (768x1344) — facial emotion is a vertical
# composition. AMBIGUOUS defaults to vertical (safe fallback).
_RESOLUTION_BY_SHOT_TYPE = {
    "ENVIRONMENT": (1344, 768),
    "ACTION":      (1344, 768),
    "PROP":        (1344, 768),
    "REACTION":    (768, 1344),
    "AMBIGUOUS":   (768, 1344),
}


def _resolution_for_shot_type(shot_type: str) -> tuple[int, int]:
    """Phase 23 (2026-06-28). Return (width, height) for a broll shot_type.
    Wide aspect ratios for environment/action/prop, vertical for reaction."""
    return _RESOLUTION_BY_SHOT_TYPE.get(
        (shot_type or "").strip().upper(),
        (768, 1344),
    )


def _character_palette_directive(character_devanagari: str) -> str:
    """Phase 12 (2026-06-03). Return a 2-line image-prompt addendum for the
    given character. Used by script_generator's image_prompt template to
    override the hardcoded warm-amber default. Falls back to a neutral
    cinematic line when the character isn't in the lookup (e.g. forced
    topics outside the 7 main arcs, or krishna direct-address renders)."""
    entry = _CHARACTER_PALETTE.get((character_devanagari or "").strip())
    if not entry:
        return "jewel-toned palette, soft cinematic lighting"
    return f"{entry['palette']}, {entry['lighting']}"

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
    "cross-eyed,bad proportions,"
    # ── Phase 13 Royal Glow companion negatives (2026-06-04) ─────────────
    # The new positive suffix invites "divine glow" failure modes (over-
    # airbrushed, painting-like, doll-faced). These negatives hard-block
    # them without weakening the genuine god-king majesty the positive
    # suffix is asking for. NOT applied to _NEGATIVE_RESTRAINT below
    # because aftermath/grief scenes legitimately benefit from "dull"
    # and "lifeless" textures.
    "painting,oil painting,watercolor,2d illustration,storybook art,"
    "dead doll eyes,dull lifeless skin,sickly skin,jaundiced complexion,"
    "magazine retouching,over-airbrushed,flat lighting,dim flat ambient light"
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

# Phase 19 (2026-06-16) — explicit anti-bias kills. FLUX defaults to
# European medieval / dark fantasy when given generic "warrior" or "epic"
# descriptors. Verified failure modes on 2026-06-16 Heaven's Gate render:
# skull emblem on breastplate (t=23s), demonic-spike helmet (t=18s),
# viking-horn helmet silhouette (t=13s), Byzantine/Mongol headband crowns,
# Christian-Orthodox candle staff (t=0). Also kills Islamic architecture
# (mosque/dome/minaret) which appeared as background in t=4/t=8 — wrong
# civilization for ancient India.
_NEGATIVE_PHASE19_ANTI_BIAS = (
    ",european knight,medieval armor,plate armor,chain mail,gothic armor,"
    "viking horns,viking helmet,crusader,roman armor,greek armor,"
    "european helmet,european crown,gothic crown,byzantine crown,"
    "mongol armor,mongol helmet,ottoman armor,"
    "skull emblem,skull motif,skull on armor,skull iconography,"
    "demonic spikes,dark fantasy,dark souls aesthetic,game of thrones aesthetic,"
    "lord of the rings aesthetic,sauron,nazgul,dementor,horned helmet,"
    "fantasy demon,sinister glow,evil aura,"
    "mosque,minaret,dome architecture,islamic architecture,"
    "gothic cathedral,european castle,medieval keep,"
    "christian crucifix,orthodox priest,liturgical staff,papal staff,"
    "celtic knot,nordic rune,egyptian ankh,star of david,"
    "modern clothing,suit and tie,jeans,t-shirt,"
    "samurai armor,ninja,kimono,chinese armor,japanese armor"
)
_NEGATIVE_DEFAULT   = _NEGATIVE_DEFAULT   + _NEGATIVE_PHASE19_ANTI_BIAS
_NEGATIVE_RESTRAINT = _NEGATIVE_RESTRAINT + _NEGATIVE_PHASE19_ANTI_BIAS

# Phase 20 (2026-06-17) — explicit anti-cartoon / anti-2D / anti-stylized
# kills. Phase 19's _HINDU_ICONOGRAPHY_ANCHOR referenced "Raja Ravi Varma
# style" + "Amar Chitra Katha" (an oil painter + a comic-book series)
# and FLUX read those as 2D-style anchors → cartoonish output.
# Phase 20 rewrites the anchor to be photoreal-forward AND adds this
# negative block to reinforce the demand at the negative-prompt layer.
#
# Note: _NEGATIVE_DEFAULT already had "cartoon,anime,cel shaded,illustration,
# drawing,comic book" + "painting,oil painting,watercolor,2d illustration,
# storybook art" from earlier phases. This block adds the specific
# digital-rendering / stylization terms those lists missed (3d render,
# stylized, plastic skin, vector art, flat shading, anime style,
# disney/pixar style, etc.).
_NEGATIVE_PHASE20_ANTI_CARTOON = (
    ",3d render,plastic skin,smooth plastic skin,stylized,low detail,"
    "vector art,flat shading,toy figurine,clay model,plasticine,"
    "video game cutscene,unreal engine art,digital painting,concept art,"
    "comic panel,manga panel,anime style,cel-shaded render,"
    "cartoon character design,disney style,pixar style,dreamworks style,"
    "stylization,smooth airbrushed face,porcelain doll skin,uncanny valley"
)
_NEGATIVE_DEFAULT   = _NEGATIVE_DEFAULT   + _NEGATIVE_PHASE20_ANTI_CARTOON
_NEGATIVE_RESTRAINT = _NEGATIVE_RESTRAINT + _NEGATIVE_PHASE20_ANTI_CARTOON

# Phase 22 (2026-06-25) — anti-ornament block. Phase 19's iconography
# anchor over-corrected into museum portraiture: mukut + kundal +
# rudraksha ornament tokens out-weighted wound/dust/blood tokens in FLUX
# prompts. The intensity validator passed prompts; pixels reverted to
# glamour. Reinforce at the negative layer.
# NOT applied to _NEGATIVE_RESTRAINT — restraint mode already strips
# glamour and "frontal pose" would fight legitimate face-forward grief
# beats (the restraint cue family targets grief / aftermath / witnessed).
_NEGATIVE_PHASE22_ANTI_ORNAMENT = (
    ",museum portrait,museum quality,calendar art,jewelry catalog,"
    "portrait photography,formal portrait,glamour shot,frontal pose,"
    "ornament-heavy,gold-jewelry-dominant,devotional poster,"
    "Mughal miniature,Tanjore painting,Rajput miniature,"
    "Amar Chitra Katha cover,Raja Ravi Varma painting,"
    "stiff posed subject,studio-clean,airbrushed glamour,"
    "ornate jewelry catalog,crown-focus,jewelry close-up subject"
)
_NEGATIVE_DEFAULT = _NEGATIVE_DEFAULT + _NEGATIVE_PHASE22_ANTI_ORNAMENT

# Phase 23 (2026-06-28) — anti-bokeh block. Applied ONLY to wide-aspect
# shots (ENVIRONMENT / ACTION / PROP). Vertical REACTION shots legitimately
# use shallow DoF for emotional close-ups; don't penalize them. The
# bokeh-wall blur is what made the X-LWlg1DW5s frames look templated —
# every background was muddy out-of-focus sandstone texture instead of
# readable scene context.
_NEGATIVE_PHASE23_ANTI_BOKEH = (
    ",bokeh,shallow depth of field,blurred background,blurred backdrop,"
    "portrait crop,head-and-shoulders shot,macro lens,facial close-up dominant,"
    "out of focus background,bokeh wall,background mush,"
    "subject isolated against blur,blurry environment"
)

# Phase 23.1 (2026-06-28) — anti-Western cultural lock. The Phase 23 force-
# render at 17:02 produced generic Western-fantasy warriors: heavy iron
# plate, leather pauldrons, Mughal/Persian spike-crowns, Viking-style
# dreadlocks. FLUX's training set defaults to European/Norse/medieval
# aesthetics whenever the prompt doesn't aggressively forbid them. Applied
# to _NEGATIVE_DEFAULT (ALWAYS-ON, not conditional like anti-bokeh) — this
# is cultural protection that must fire on every Mahabharata render
# regardless of aspect ratio or shot type. Complements Phase 19's existing
# _NEGATIVE_PHASE19_ANTI_BIAS but adds the specific failure modes the
# Phase 23 anchor rewrite re-opened.
_NEGATIVE_PHASE23_1_ANTI_WESTERN = (
    ",european armor,roman armor,plate mail,iron plate armor,iron breastplate,"
    "leather pauldrons,leather shoulder pads,heavy leather cloak,studded leather,"
    "viking style,viking dreadlocks,medieval knight,spartan helmet,"
    "gladiator,gladiator chestplate,roman centurion,"
    "game of thrones style,western fantasy,norse warrior,germanic warrior,"
    "Persian spike crown,spiked crown,Mughal headdress,Ottoman turban,"
    "chain mail,gambeson,European cloak,Tudor style,Renaissance fair,"
    "Lord of the Rings style,Witcher style,Vikings TV style"
)
_NEGATIVE_DEFAULT = _NEGATIVE_DEFAULT + _NEGATIVE_PHASE23_1_ANTI_WESTERN

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


# ─── Explainer visual_track categories (v4 — documentary realism) ────────
# When a scene has a `visual_track` list (explainer series only), the LLM
# supplies one prompt per shot with a category tag. Each category prepends
# its own composition directive so FLUX gets a consistent framing per type.
#
# Visual-category prefixes — keyed by series, then category. Each series
# has its own visual brand:
#   "explainer"  — documentary realism (v4): grainy, fluorescent-lit, handheld,
#                  "leaked/observed" feel. Anti-cinematic on purpose.
#   "curiosity"  — premium cinematic (Cleo Abram / Lemmino / Kurzgesagt):
#                  volumetric lighting, color grading, motion blur, 35mm film.
#                  OPPOSITE of explainer — beauty-first, awe-evoking.
_CATEGORY_PREFIX_BY_SERIES = {
    "explainer": {
        "human":    "candid documentary photograph, ordinary person, fluorescent or natural light, slightly grainy, handheld feel, imperfect framing, no dramatic lighting, ",
        "system":   "documentary photograph of industrial infrastructure, security camera angle OR handheld investigative shot, fluorescent or sodium-vapor lighting, slightly grainy, no people, no text, no logos, NO cinematic dramatic lighting — feels captured not staged, ",
        "symbolic": "documentary still-life photograph, observed object in a real setting, natural or fluorescent light, slightly imperfect framing, no dramatic isolation lighting, no people, no text, ",
        "ui":       "low-fi screen capture from an investigative documentary OR a photographed-off-monitor shot of a news terminal, slight moire or screen glare, looks filmed off a real screen, no logos, no readable proper-noun text, ",
    },
    "curiosity": {
        # Anti-text guard appears EARLY (front-loaded — FLUX down-weights late tokens)
        # AND at the end. Specifically lists "no fictional brand names" because
        # FLUX-schnell hallucinated a fictional "VIPER" logo on spacecraft in the
        # v2 Short 1 render — generic "no text, no logos" wasn't strong enough.
        "hero":              "premium cinematic photograph (absolutely no readable text, no fictional brand names, no painted logos, no numbers visible), single dramatic subject, volumetric lighting, atmospheric depth, motion blur on action elements, color graded teal-and-orange or cool moonlight, 35mm film aesthetic, no text, no logos, no signs, ",
        "environment":       "vast cinematic landscape (absolutely no readable text, no fictional brand names, no painted logos, no signs visible), scale-emphasizing wide angle, dramatic sky or cosmic backdrop, dust or particles in air, golden-hour or moonlight, NASA / National Geographic photography style, no humans, no text, no logos, ",
        "motion":            "cinematic photograph of motion mid-action (absolutely no readable text, no fictional brand names, no painted logos on any machinery or vehicles) — collapsing or flowing or expanding or disintegrating — slight motion blur, dramatic lighting, dynamic composition, no text, no logos, no signs, ",
        "tension":           "tight cinematic close-up of an object (absolutely no readable text, no fictional brand names, no painted logos, no readable numbers) implying threat or change, shallow depth of field, harsh side-lighting, single light source, intentionally ominous framing, no people, no text, no logos, ",
        "aftermath":         "calm cinematic still (absolutely no readable text, no fictional brand names, no painted logos visible) — after-the-event quietness, soft natural light, single subject in a vast empty space, contemplative composition, no people, no text, no logos, ",
        "human_consequence": "cinematic close-up of a single human in an emotional moment — eyes widening in slow-dawning realization, dilating pupils, a single tear tracing a cheek, parted lips mid-gasp, frozen face mid-thought, an open mouth in silent scream, lashes wet, jaw clenched in dread, brow furrowed in shock — face occupies the frame, shallow depth of field, intense single-source lighting, photoreal skin texture, dramatic emotional weight, NO hands or fingers visible in frame, no text, no logos, no readable numbers, ",
    },
}

# Backwards-compat alias for any callers still expecting flat _CATEGORY_PREFIX
# (the explainer's pre-multi-series shape). New code should use
# _CATEGORY_PREFIX_BY_SERIES[series][category].
_CATEGORY_PREFIX = _CATEGORY_PREFIX_BY_SERIES["explainer"]

# Per-series output resolution. Explainer is portrait (Shorts), curiosity is
# landscape long-form. v2: curiosity also has a Shorts mode (portrait), picked
# via `mode="shorts"` parameter on generate_images.
_RESOLUTION_BY_SERIES = {
    "explainer": (768, 1344),    # portrait 9:16
    "curiosity": (1920, 1080),   # landscape 16:9 long-form (default for curiosity)
}
# v2: per (series, mode) override. mode="shorts" forces portrait Shorts dims.
_RESOLUTION_BY_SERIES_MODE = {
    ("curiosity", "shorts"): (1080, 1920),   # portrait 9:16 Shorts (true Shorts dims)
    ("curiosity", "longform"): (1920, 1080), # landscape 16:9 long-form
}
_DEFAULT_RESOLUTION = (768, 1344)  # backwards-compat default (portrait)


# ── Style anchor (v2.1 — 2026-06-13) ──────────────────────────────────────────
# Per-topic cinematography reference concatenated into every visual prompt
# (curiosity series only). When a topic's strategy.json has a `style_anchor`
# field, that value wins; otherwise this default applies. The anchor sits
# between the category framing prefix and the scene-specific subject prompt:
#     "<category_prefix>, <style_anchor>, <subject>"
# Mahabharata isolation: anchor is ONLY injected when series == "curiosity"
# inside the visual_track fast path. Mahabharata's prompts are untouched.
_DEFAULT_STYLE_ANCHOR = (
    "shot on Arri Alexa LF, anamorphic lens, Roger Deakins natural lighting, "
    "deep teal and amber color grade, 35mm film grain, cinematic shadows, "
    "8k hyper-realistic"
)


def _resolution_for(series: str, mode: str = "longform") -> tuple[int, int]:
    """Pick (width, height) based on series + mode. mode kw added in v2 for
    Shorts vs long-form on the same series."""
    key = (series, mode)
    if key in _RESOLUTION_BY_SERIES_MODE:
        return _RESOLUTION_BY_SERIES_MODE[key]
    return _RESOLUTION_BY_SERIES.get(series, _DEFAULT_RESOLUTION)


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


def _inject_characters(prompt: str, wardrobe_context: str = "") -> str:
    """
    Scans the prompt for known Mahabharata character names and appends
    their visual description so every image is visually consistent.

    Phase 22 (2026-06-25): wardrobe-aware to defeat the multi-character
    merge problem.
    Phase 23 (2026-06-28): TIGHT bracketed format. Frame 1 of X-LWlg1DW5s
    showed that the verbose CHARACTER DETAILS — <125 chars> paragraph
    appended at the END of the prompt was still character-anatomy-heavy
    enough to push FLUX into portrait composition. New format wraps each
    signature_lock in [Name: lock] brackets and keeps total injection
    ≤80 chars per character. In PALACE/DIVINE we still allow the full
    visual fingerprint because court scenes legitimately benefit from
    ornament detail.
    """
    if not _CHARACTERS:
        return prompt
    ctx = (wardrobe_context or "").strip().upper()
    use_full_fingerprint = ctx in ("PALACE", "DIVINE")

    injected = []
    prompt_lower = prompt.lower()
    for name, data in _CHARACTERS.items():
        if name.lower() in prompt_lower:
            if use_full_fingerprint:
                visual = data.get("visual", "")[:250]
                if visual:
                    injected.append(visual)
            else:
                # Phase 23: tight bracketed [Name: lock] format — each
                # entry ~50-70 chars including the brackets, vs the prior
                # 100-150 char per-character paragraph.
                lock = (
                    data.get("signature_lock", "")
                    or data.get("visual", "")[:60]
                )
                if lock:
                    injected.append(f"[{name}: {lock}]")
    if injected:
        if use_full_fingerprint:
            return prompt + ". CHARACTER DETAILS — " + "; ".join(injected)
        # Phase 23: bracket-list format for non-court contexts.
        return prompt + " " + " ".join(injected)
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


def generate_images(scenes_or_script, single_shot: bool = False, series: str = "mahabharata", visual_style: str = "", ck=None, mode: str = "longform", style_anchor: str | None = None) -> list:
    """
    Generates portrait (768x1344) images per scene.

    Phase 18 (2026-06-16) — when called with a script DICT that has a `broll`
    key, iterate over the broll entries instead of the legacy `scenes` array.
    Each broll entry has the same shape as a scene entry for our purposes
    (`image_prompt`, `mood`, optional anchor used as the narration field for
    the hook-visual lookup), so we just shim it into the legacy loop.

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

    # Phase 18 (2026-06-16) — accept (a) a dict with a 'broll' key (Phase 18
    # decoupled path), (b) a dict with a 'scenes' key (full script dict from
    # the new main.py call site), or (c) a raw list of scene dicts (legacy).
    if isinstance(scenes_or_script, dict):
        if "broll" in scenes_or_script:
            scenes = [
                {
                    "image_prompt":     entry.get("image_prompt", ""),
                    "mood":             entry.get("mood", ""),
                    "narration":        entry.get("anchor_phrase", ""),
                    "wardrobe_context": entry.get("wardrobe_context", ""),  # Phase 19
                }
                for entry in scenes_or_script["broll"]
            ]
            print(f"    [phase18-img] iterating {len(scenes)} broll entries (decoupled mode)")
        elif "scenes" in scenes_or_script:
            scenes = scenes_or_script["scenes"]
        else:
            raise ValueError("generate_images: dict missing both 'broll' and 'scenes' keys")
    else:
        scenes = scenes_or_script

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

        # ── VISUAL_TRACK FAST PATH (v2 explainer, v5 curiosity) ──────────
        # When the script supplies a `visual_track` (explainer + curiosity series),
        # the LLM has already authored per-shot prompts with category tags. Use
        # those directly with category-specific framing instead of the generic
        # _SHOT_COMPOSITIONS triplet. The framing prefix is picked from the
        # series-specific dict so explainer gets documentary-realism cues and
        # curiosity gets premium-cinematic cues.
        if (series in ("explainer", "curiosity")
                and isinstance(scene.get("visual_track"), list)
                and scene["visual_track"]):
            track = scene["visual_track"]
            # Per-series category dict + resolution (v2: mode-aware for shorts vs longform)
            series_categories = _CATEGORY_PREFIX_BY_SERIES.get(
                series, _CATEGORY_PREFIX_BY_SERIES["explainer"]
            )
            default_cat_for_series = next(iter(series_categories))  # first key as fallback
            img_w, img_h = _resolution_for(series, mode)
            # v2.1 (2026-06-13) — style anchor injection, curiosity-only.
            # Resolves the per-topic cinematography reference once per scene
            # and concatenates it into every shot's prompt between the
            # category framing and the subject. Mahabharata + explainer skip
            # this entirely (anchor_fragment stays empty string).
            anchor_fragment = ""
            if series == "curiosity":
                _anchor = (style_anchor or _DEFAULT_STYLE_ANCHOR).strip().rstrip(",").strip()
                if _anchor:
                    anchor_fragment = _anchor + ", "
            for shot_idx, shot in enumerate(track):
                if not isinstance(shot, dict):
                    continue
                cat = shot.get("category", default_cat_for_series)
                subject = (shot.get("prompt") or "").strip()
                if not subject:
                    continue
                framing = series_categories.get(cat) or series_categories[default_cat_for_series]
                full_prompt = framing + anchor_fragment + subject
                # Seed: stable per (scene, shot, category) so re-runs reproduce
                seed = i * 211 + shot_idx * 37 + (hash(cat) & 0xFFF)
                output_path = f"{_TEMP_ROOT}/images/scene_{i:02d}_shot_{shot_idx:02d}.jpg"
                success = False
                for attempt in range(3):
                    try:
                        img_bytes, provider = generate_image_bytes(
                            full_prompt, seed=seed, width=img_w, height=img_h,
                            mood="", style_suffix="",  # series LUT applied downstream
                        )
                        with open(output_path, "wb") as f:
                            f.write(img_bytes)
                        shot_paths.append(output_path)
                        print(f"    [OK] Scene {i+1} shot {shot_idx+1}/{len(track)} "
                              f"({cat} {img_w}x{img_h}) via {provider}")
                        success = True
                        break
                    except Exception as e:
                        print(f"    [!] Scene {i+1} shot {shot_idx+1} ({cat}) attempt {attempt+1}: {e}")
                    time.sleep((attempt + 1) * 3)
                if not success:
                    _create_placeholder(output_path, i * len(track) + shot_idx, series=series)
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
            # Phase 22 (2026-06-25): pass wardrobe_context into the
            # injector. In WAR/FOREST/JOURNEY/AFTERMATH it picks the
            # ~120-char signature_lock (defeats multi-subject merge);
            # in PALACE/DIVINE it uses the full 250-char visual.
            scene_wardrobe_ctx = scene.get("wardrobe_context", "") if series == "mahabharata" else ""
            base_prompt = raw_prompt if series == "whatif" else _inject_characters(raw_prompt, scene_wardrobe_ctx)
            # Append the imperfection cue AFTER character injection so it
            # rides on top of the character's existing descriptors instead
            # of getting buried by them. Additive — does not replace any
            # part of the scene prompt. Empty cue = no-op.
            if imperfection_cue:
                base_prompt = f"{base_prompt}. {imperfection_cue}"
            # Phase 19 / Phase 22 / Phase 23 — wardrobe-context-aware +
            # scene-anchored composition. Phase 23 INVERTS the token stack:
            # composition + scene + LLM action FIRST (FLUX weights early
            # tokens heaviest, so the action verb and environment land in
            # the strongest cross-attention slots), cultural anchor SECOND,
            # tight character details LAST. The X-LWlg1DW5s forensic showed
            # the old order (anchor + character anatomy + ... + action verb
            # buried at the end) made FLUX commit to portrait composition
            # before it ever read the verb.
            if series == "mahabharata":
                wardrobe_ctx        = scene_wardrobe_ctx
                shot_type           = (scene.get("shot_type", "") or "AMBIGUOUS").strip().upper()
                iconography_anchor  = _pick_iconography_anchor(wardrobe_ctx)
                # Phase 23: composition first → action verb (in base_prompt)
                # second → tight character bracket third → scene anchor fourth.
                # We drop the legacy wardrobe_prefix entirely — its character-
                # anatomy tokens (ash-streaked face / blood smear / golden-
                # bronze skin) double-loaded the prompt tail with portrait
                # cues. The new signature_lock (≤45 chars) and the per-
                # context iconography_anchor (scene descriptor) carry the
                # same information without crushing the verb + environment.
                prompt = (
                    f"{angle_label}{composition_directive}"
                    + base_prompt          # raw LLM prompt (verb + scene action) + [Name: lock]
                    + " "
                    + iconography_anchor   # scene descriptor (env + depth + multi-character)
                )
            else:
                prompt = f"{angle_label}{composition_directive}{base_prompt}"
            # Stable per-character seed: same hero across scenes → similar
            # face. Falls back to scene-position seed when no known character
            # is mentioned (WhatIf scenes, environment-only shots).
            hero = "" if series == "whatif" else _primary_character(raw_prompt)
            seed = _char_stable_seed(hero, j) if hero else (i * 137 + j * 31)

            # Phase 23 (2026-06-28): dynamic aspect ratio + conditional
            # anti-bokeh negative. Wide shots (ENVIRONMENT/ACTION/PROP)
            # render at 1344x768 (16:9 landscape) — FLUX out of portrait
            # latent space, multi-character compositions possible. Vertical
            # shots (REACTION/AMBIGUOUS) render at 768x1344 (9:16 portrait).
            # Anti-bokeh negative only applies to wide shots — vertical
            # REACTION close-ups legitimately use shallow DoF.
            if series == "mahabharata":
                img_w, img_h = _resolution_for_shot_type(shot_type)
                if img_w > img_h:  # wide → add anti-bokeh
                    scene_negative = scene_negative + _NEGATIVE_PHASE23_ANTI_BOKEH
            else:
                img_w, img_h = 768, 1344  # backwards-compat for non-mahabharata

            success = False
            for attempt in range(3):
                try:
                    img_bytes, provider = generate_image_bytes(
                        prompt, seed=seed, width=img_w, height=img_h, mood=mood,
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
    elif series in ("mahabharata", "krishna"):
        # Phase 11-Plus (2026-05-23) + iter-2 tightenings (2026-05-24) +
        # CF-prompt-cap compression (2026-05-24): same mandates as iter-2
        # but compressed ~65% — the previous verbose rider pushed combined
        # style_suffix past CF FLUX-schnell's 2048-char prompt limit, so
        # every CF account 400'd on every thumbnail and we always fell
        # through to Pollinations. Compressed version preserves all the
        # iter-2 mandates (face dominance, eye dominance, asymmetric framing,
        # uncomfortable proximity, mid-motion, harsh edge lighting, pain
        # artifacts, the forbidden list) but drops redundant emphasis words
        # and explanatory phrasing.
        style_suffix = (
            style_suffix
            + ", phone-feed thumbnail composition: ONE face extreme close-up, "
              "face filling ≥60% of frame, eyes ≥30% of frame, camera at "
              "face level looking at or just past the lens (no profile, no "
              "half-turn, no overhead, eyes open); asymmetric off-center "
              "framing, uncomfortable proximity (forehead/chin/ear may crop "
              "at edge), mid-motion (hair displaced, fabric awkward); harsh "
              "edge lighting from above-left OR below, deep cheek/jaw shadow, "
              "lit half sharp + shadow half near-black; extreme emotion "
              "(rage, anguish, shock, defiance, grief, accusation, open-mouth "
              "anger); visible pain artifacts (tear tracks, sweat, dust, ash, "
              "blood smear, trembling lips, clenched jaw, bared teeth, temple "
              "veins); CAPTURED-NOT-CRAFTED (iter-3 2026-05-29 / Issue #4) — "
              "visible film grain overlay, subtle motion blur on hair or fabric "
              "edge while face stays sharp, slight imperfect focus on one "
              "peripheral element, NOT crystal-clean digital sharpness; "
              "CLARITY FLOOR — at least ONE eye fully visible and "
              "emotionally readable, face occupies ≥40% of frame, contrast "
              "preserves mid-tone gradient (no full silhouette, no double "
              "blowout); FORBID bilateral symmetry, centered subjects, even / "
              "wraparound lighting, soft-focus beauty, peaceful gazes, noble "
              "stoicism, half-smiles, devotional serenity, battlefield wide "
              "shots, multi-figure compositions, both-eyes-obscured crops, "
              "studio portrait quality, magazine-cover composition, AI-render "
              "polish; dark moody background, high contrast, no text, no UI, "
              "no logos"
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


# ── Kaggle primary path (v2.1, 2026-06-13) ───────────────────────────────────
# Async wrapper that tries the Kaggle FLUX + IP-Adapter Master Anchor + LTX
# notebook for curiosity Shorts, falls back to the sync Cloudflare cascade on
# any KaggleClientError.
#
# Mahabharata isolation: this function REFUSES to fire for series != "curiosity"
# and immediately delegates to the existing sync generate_images() path.
# Mahabharata's main.py does not import or call this function — it only calls
# sync generate_images() which is unchanged. Verified safe.

# Module-level lock — serializes EN and HI Kaggle calls inside a single
# process. Without this, the second-language render's `current_run.json`
# write could clobber the first language's config mid-push, and Kaggle would
# render the wrong content for one of the two languages. Held across the
# full lifecycle (write → push → poll → download) so EN's outputs are safely
# on local disk before HI even attempts to write its config.
_KAGGLE_LOCK = asyncio.Lock()


def _stable_master_seed(run_id: str) -> int:
    """Derive a deterministic 31-bit seed from run_id via hashlib (not the
    built-in hash()) so the value is stable across Python processes.
    Required because the seed is sent to Kaggle and must reproduce when
    re-running with the same run_id (debugging, retries)."""
    digest = hashlib.md5(run_id.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % (2**31 - 1)


def _reshape_kaggle_outputs_to_scene_groups(downloaded: list, scenes: list) -> list:
    """Group downloaded Kaggle artifacts by scene/shot index from filename
    pattern `scene_NN_shot_MM.{jpg,mp4}`. Returns list-of-lists matching
    generate_images()'s curiosity-Shorts return shape.

    Motion clips (.mp4) take precedence over stills (.jpg) at the same
    (scene, shot) position — the video assembler's dispatcher routes the
    .mp4 to _make_silent_video_scene_clip while .jpg goes to Ken Burns.
    Filenames are 1-indexed (matches run_flux_phase.py's zfill output);
    we convert to 0-indexed internally to align with scenes[] list."""
    by_pos: dict = {}
    pattern = re.compile(r"scene_(\d+)_shot_(\d+)\.(jpg|jpeg|png|mp4)$", re.IGNORECASE)
    for p in downloaded:
        m = pattern.search(str(p))
        if not m:
            continue
        scene_idx = int(m.group(1)) - 1
        shot_idx = int(m.group(2)) - 1
        ext = m.group(3).lower()
        key = (scene_idx, shot_idx)
        # mp4 wins over jpg at the same (scene, shot) position
        if key not in by_pos or (
            ext == "mp4" and not str(by_pos[key]).lower().endswith(".mp4")
        ):
            by_pos[key] = p

    scene_groups = []
    for i, scene in enumerate(scenes):
        shot_paths = []
        for j in range(len(scene.get("visual_track", []))):
            path = by_pos.get((i, j))
            if path:
                shot_paths.append(str(path))
        scene_groups.append(shot_paths)
    return scene_groups


async def generate_images_with_kaggle_primary(
    scenes: list,
    *,
    series: str = "curiosity",
    mode: str = "shorts",
    style_anchor: str | None = None,
    run_id: str = "",
    ck=None,
    visual_style: str = "",
    single_shot: bool = False,
) -> list:
    """v2.1 Kaggle-primary entry point for curiosity Shorts. Pushes the
    FLUX+IP-Adapter+LTX kernel folder with a per-render `current_run.json`,
    polls for completion, downloads outputs into cache/<run_id>/visuals/.

    Returns the same list-of-lists shape as generate_images() — outer list
    is per-scene, inner list is per-shot paths (.jpg stills and/or .mp4
    motion clips). The video assembler's dispatcher routes each file type.

    On any KaggleClientError (missing kernel-metadata, push failure, poll
    timeout, download error), falls back to the existing sync Cloudflare
    cascade. Network/quota failures degrade gracefully — no IP-Adapter
    consistency and no motion clips, but the pipeline still produces a
    working Short.

    Concurrency: holds _KAGGLE_LOCK for the full push → poll → download
    cycle so EN+HI batches cannot collide on `current_run.json`.

    Mahabharata isolation: if series != "curiosity", returns immediately
    via the sync fallback path. Mahabharata's main.py doesn't import this
    function anyway, so this is double-belt-and-suspenders.
    """
    # --- Hard gate: curiosity only ---
    if series != "curiosity":
        return generate_images(
            scenes, single_shot=single_shot, series=series,
            visual_style=visual_style, ck=ck, mode=mode,
            style_anchor=style_anchor,
        )

    # --- Soft gates: missing config falls through to Cloudflare ---
    kernel_ref = os.environ.get("KAGGLE_KERNEL_REF", "").strip()
    kernel_dir = Path("kaggle_notebooks") / "cinematic-i2v-batch"
    if not kernel_ref:
        print("[kaggle] KAGGLE_KERNEL_REF not set — using Cloudflare cascade")
        return generate_images(
            scenes, single_shot=single_shot, series=series,
            visual_style=visual_style, ck=ck, mode=mode,
            style_anchor=style_anchor,
        )
    if not kernel_dir.exists():
        print(f"[kaggle] kernel folder missing: {kernel_dir} — using Cloudflare cascade")
        return generate_images(
            scenes, single_shot=single_shot, series=series,
            visual_style=visual_style, ck=ck, mode=mode,
            style_anchor=style_anchor,
        )

    # Lazy import — avoids loading kaggle subprocess wrapper for Mahabharata
    from pipeline import kaggle_client

    async with _KAGGLE_LOCK:
        # Build run_config ONCE — the same payload is pushed on each retry
        # (only the master_seed could vary; we keep it stable so each retry
        # would reproduce the same output if it landed on T4).
        requires_motion_list = [
            [i, j]
            for i, scene in enumerate(scenes)
            for j, shot in enumerate(scene.get("visual_track", []) or [])
            if isinstance(shot, dict) and shot.get("requires_motion")
        ]
        master_seed = _stable_master_seed(run_id or "default-run")
        run_config = {
            "scenes": scenes,
            "style_anchor": (style_anchor or _DEFAULT_STYLE_ANCHOR).strip(),
            "requires_motion": requires_motion_list,
            "master_seed": master_seed,
            "run_id": run_id,
        }
        timeout_s = int(os.environ.get("KAGGLE_TIMEOUT_S", "2700"))
        poll_interval_s = int(os.environ.get("KAGGLE_POLL_INTERVAL_S", "60"))
        max_attempts = int(os.environ.get("KAGGLE_P100_RETRIES", "5"))

        # Retry-until-T4 loop. Each P100 fast-fail costs ~60s; a successful
        # FLUX+LTX run takes ~17 min. So 5 P100 retries cost ~5 min worst-
        # case — under 1/3 of one real-run quota. Once T4 lands, FLUX+LTX
        # complete and we return scene_groups directly.
        last_error: str = "<no Kaggle attempt made>"
        for attempt in range(1, max_attempts + 1):
            try:
                print(f"[kaggle] attempt {attempt}/{max_attempts}: pushing kernel "
                      f"{kernel_ref} for run_id={run_id} "
                      f"({len(scenes)} scenes, {len(requires_motion_list)} motion shots, "
                      f"seed={master_seed})")
                version = await kaggle_client.push_kernel_with_run_config(
                    kernel_dir, run_config,
                )
                print(f"[kaggle] polling (interval={poll_interval_s}s, timeout={timeout_s}s)")
                result = await kaggle_client.poll_kernel(
                    kernel_ref,
                    poll_interval_s=poll_interval_s,
                    timeout_s=timeout_s,
                )
                if result["status"] == "complete":
                    target_dir = (
                        Path(f"cache/{run_id}/visuals")
                        if run_id
                        else Path("cache/kaggle_latest/visuals")
                    )
                    target_dir.mkdir(parents=True, exist_ok=True)
                    print(f"[kaggle] downloading outputs to {target_dir}")
                    downloaded = await kaggle_client.download_output(
                        kernel_ref, target_dir,
                        version=version if version > 0 else None,
                    )
                    scene_groups = _reshape_kaggle_outputs_to_scene_groups(downloaded, scenes)
                    n_files = sum(len(g) for g in scene_groups)
                    n_mp4 = sum(
                        1 for g in scene_groups for p in g
                        if p.lower().endswith(".mp4")
                    )
                    print(f"[kaggle] OK on attempt {attempt} — {n_files} files "
                          f"({n_mp4} motion clips, {n_files - n_mp4} stills) "
                          f"across {len(scene_groups)} scenes")
                    return scene_groups

                # status != "complete" — was it our P100 fast-fail?
                was_p100 = await kaggle_client.is_p100_failure(kernel_ref)
                if was_p100 and attempt < max_attempts:
                    print(f"[kaggle] attempt {attempt}/{max_attempts}: P100 allocated "
                          f"(sm_60 incompatible with PyTorch 2.x). Retrying for T4 luck...")
                    last_error = f"attempt {attempt}: P100 fast-fail"
                    continue  # next attempt — Kaggle may give T4 next time
                # Either genuine error (not P100) OR we've exhausted retries
                last_error = (
                    f"attempt {attempt}: kernel status={result['status']}"
                    + (" (P100 fast-fail, retries exhausted)" if was_p100 else " (non-P100 error)")
                )
                print(f"[kaggle] {last_error}")
                break  # don't retry non-P100 failures

            except kaggle_client.KaggleClientError as e:
                last_error = f"attempt {attempt}: {e}"
                # Could be network blip / push timeout / poll timeout —
                # retry once or twice before giving up
                if attempt < max_attempts:
                    print(f"[kaggle] attempt {attempt}/{max_attempts} CLI error "
                          f"({e}) — retrying...")
                    continue
                print(f"[kaggle] CLI error on final attempt: {e}")
                break

        # All Kaggle attempts failed (or hit genuine error) — fall back to
        # the sync Cloudflare cascade. Lock stays held so HI's attempt
        # doesn't race against EN's fallback writes.
        print(f"[kaggle] FAILED after {attempt} attempt(s) — falling back to "
              f"Cloudflare cascade. Last: {last_error}")
        return generate_images(
            scenes, single_shot=single_shot, series=series,
            visual_style=visual_style, ck=ck, mode=mode,
            style_anchor=style_anchor,
        )
