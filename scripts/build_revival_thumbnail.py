"""
Phase 16 D3 — Karna-Kunti revival thumbnail builder (2026-06-09).

Generates the 1080x1920 forced-frame thumbnail that gets baked into the
first 0.8s of the revival MP4 (see pipeline/video_assembler.prepend_thumbnail_frame).

Why this script exists:
    YouTube Shorts disable API-uploaded custom thumbnails, so the only
    way to control the auto-selected thumbnail is to make our chosen
    graphic the literal first frame YouTube ingests. The MP4 prepend
    is handled by main.py's BAKE_THUMBNAIL_PATH env-gated step; this
    script PRODUCES the PNG that path points at.

Pipeline:
    1. Cloudflare FLUX-schnell generates the Kunti close-up portrait
       via the existing image_generator.generate_image_bytes() cascade
       (CF -> HF -> Pollinations, same path as every render's images).
       Negative prompt aggressively excludes text/typography/watermarks
       so the background stays clean for our overlay.
    2. PIL resizes + center-crops the FLUX output to 1080x1920 Shorts
       aspect ratio (FLUX commonly returns 1024x1536-ish portrait).
    3. HarfBuzz renders "कुंती का पाप" via text_renderer.render_text_card
       at thumbnail-scale (massive font + heavy drop shadow + thick
       black stroke) so it stays legible at the ~200x355px Shorts feed
       preview size.
    4. Composites text onto image, anchored upper-third per the user's
       spec (above the YT UI overlay band at the bottom).
    5. Writes assets/revival/karna_kunti_thumbnail.png.

Run:
    python scripts/build_revival_thumbnail.py

Requires CF/HF API tokens in env (.env or shell) — same as every
image-generating render.
"""

import os
import sys
from io import BytesIO

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env exactly the way main.py does (so CF/HF tokens are available)
from dotenv import load_dotenv
load_dotenv()

from PIL import Image

from pipeline.image_generator import (
    generate_image_bytes,
    STYLE_SUFFIX_MORTAL,
    _NEGATIVE_DEFAULT,
)
from pipeline.text_renderer import render_text_card


# ── Configuration ──────────────────────────────────────────────────────────
FLUX_PROMPT = (
    "Extreme macro close up 85mm portrait of ancient Indian Queen Kunti, "
    "a single tear rolling down her cheek, intense sorrow and guilt, "
    "looking directly at the camera. Striking high contrast lighting: "
    "warm fiery orange rim light against a deep cold teal background. "
    "Cinematic, 8k, photorealistic, Royal Glow."
)

# Stronger negative — we will overlay our own text, so the FLUX output
# must have ZERO baked-in typography. The default _NEGATIVE_DEFAULT
# already covers most text artifacts; we extend with extra anti-text
# tokens for paranoia.
NEGATIVE_PROMPT = _NEGATIVE_DEFAULT + (
    ",any text on image,letters anywhere,words,calligraphy,inscription,"
    "tattoo text,banner text,sign text,subtitles,caption box,logo,emblem"
)

TEXT_OVERLAY      = "कुंती का पाप"            # 2 words — "Kunti's Sin"
FONT_PATH         = "assets/fonts/NotoSansDevanagari-Bold.ttf"
OUTPUT_PATH       = "assets/revival/karna_kunti_thumbnail.png"
TARGET_WIDTH      = 1080
TARGET_HEIGHT     = 1920

# FLUX request — generate slightly oversized so we have crop margin.
FLUX_REQ_WIDTH    = 1024
FLUX_REQ_HEIGHT   = 1536
FLUX_SEED         = 42                          # reproducible

# Text-overlay parameters tuned for the 200x355px Shorts feed preview
TEXT_FONT_SIZE    = 220                         # massive
TEXT_FILL         = (255, 220, 70, 255)         # warm amber-yellow
TEXT_OUTLINE      = (0, 0, 0, 255)              # black stroke
TEXT_OUTLINE_PX   = 14                          # very thick stroke
TEXT_SHADOW       = (0, 0, 0, 230)              # heavy drop shadow
TEXT_SHADOW_OFFSET = (8, 10)                    # offset for depth
TEXT_Y_POS_RATIO  = 0.20                        # upper-third anchor (20% from top)
TEXT_MAX_WIDTH_RATIO = 0.88                     # text occupies <=88% of frame width


def _fit_and_crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize `img` so the SHORTER side fits its target, then center-crop
    to the exact target aspect ratio. Preserves face detail and avoids
    letterboxing."""
    src_w, src_h = img.size
    target_aspect = target_w / target_h
    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        # Source wider than target — match heights, crop sides
        new_h = target_h
        new_w = int(src_w * new_h / src_h)
    else:
        # Source taller than target — match widths, crop top/bottom
        new_w = target_w
        new_h = int(src_h * new_w / src_w)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top  = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def main() -> int:
    print(f"[Phase 16 D3] Building revival thumbnail")
    print(f"   prompt:  {FLUX_PROMPT[:80]}...")
    print(f"   text:    '{TEXT_OVERLAY}'")
    print(f"   output:  {OUTPUT_PATH}")
    print()

    # 1. Generate FLUX image via the existing cascade
    print(f"[1/4] FLUX generation ({FLUX_REQ_WIDTH}x{FLUX_REQ_HEIGHT})...")
    img_bytes, provider = generate_image_bytes(
        FLUX_PROMPT,
        seed=FLUX_SEED,
        width=FLUX_REQ_WIDTH,
        height=FLUX_REQ_HEIGHT,
        style_suffix=STYLE_SUFFIX_MORTAL,    # Royal Glow
        negative_prompt=NEGATIVE_PROMPT,
    )
    print(f"      generated via {provider}, {len(img_bytes)/1024:.0f} KB")

    # 2. Resize + center-crop to 1080x1920
    print(f"[2/4] Resize + crop to {TARGET_WIDTH}x{TARGET_HEIGHT}...")
    base = Image.open(BytesIO(img_bytes)).convert("RGB")
    print(f"      FLUX raw size: {base.size}")
    base = _fit_and_crop_center(base, TARGET_WIDTH, TARGET_HEIGHT)
    print(f"      cropped to:    {base.size}")

    # 3. Render Devanagari text card via HarfBuzz
    print(f"[3/4] Rendering text card '{TEXT_OVERLAY}' at {TEXT_FONT_SIZE}px...")
    if not os.path.exists(FONT_PATH):
        raise SystemExit(f"Font missing at {FONT_PATH} — check assets/fonts/")
    text_card = render_text_card(
        text=TEXT_OVERLAY,
        font_path=FONT_PATH,
        font_size=TEXT_FONT_SIZE,
        fill=TEXT_FILL,
        outline=TEXT_OUTLINE,
        outline_px=TEXT_OUTLINE_PX,
        shadow=TEXT_SHADOW,
        shadow_offset=TEXT_SHADOW_OFFSET,
    )
    print(f"      text card size: {text_card.size}")

    # If text card overflows the max-width budget, scale it down
    tw, th = text_card.size
    max_w = int(TARGET_WIDTH * TEXT_MAX_WIDTH_RATIO)
    if tw > max_w:
        scale = max_w / tw
        text_card = text_card.resize(
            (int(tw * scale), int(th * scale)), Image.LANCZOS
        )
        tw, th = text_card.size
        print(f"      scaled to fit width budget: {text_card.size}")

    # 4. Composite + save
    print(f"[4/4] Compositing + saving...")
    base_rgba = base.convert("RGBA")
    paste_x = (TARGET_WIDTH - tw) // 2
    paste_y = int(TARGET_HEIGHT * TEXT_Y_POS_RATIO)
    base_rgba.paste(text_card, (paste_x, paste_y), text_card)
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    # Save as PNG (lossless) so the bake-in pipeline gets clean source
    base_rgba.convert("RGB").save(OUTPUT_PATH, "PNG", optimize=True)
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\n✅ Wrote {OUTPUT_PATH} ({TARGET_WIDTH}x{TARGET_HEIGHT}, {size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
