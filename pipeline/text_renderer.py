"""
Text Renderer — HarfBuzz-shaped subtitle PNGs
=============================================
Renders subtitle cards as transparent RGBA PNGs using uharfbuzz for OpenType
shaping + freetype-py for glyph rasterization.

This replaces FFmpeg's libass/drawtext text rendering, which on the gyan.dev
Windows FFmpeg build silently skips Indic complex shaping despite linking
libharfbuzz — the result is broken Devanagari (e.g. "प्रतिज्ञा" rendered as
"प्रतज्ज्ञा", with i-mātrās in wrong positions and conjuncts not forming).

Doing the shaping in Python ourselves bypasses the broken FFmpeg path entirely
and works identically across platforms / FFmpeg builds. The output PNGs are
then overlaid on the video via FFmpeg's `overlay` filter.

Public API:
    render_text_card(text, font_path, font_size, ...) -> PIL.Image (RGBA)
"""

from __future__ import annotations
from typing import Tuple

import uharfbuzz as hb
import freetype
from PIL import Image, ImageFilter


# Cache hb objects per font path — building Face/Font is mildly expensive
_HB_CACHE: dict = {}
_FT_CACHE: dict = {}


def _hb_font(font_path: str, size_px: int) -> hb.Font:
    key = (font_path, size_px)
    f = _HB_CACHE.get(key)
    if f is not None:
        return f
    blob = hb.Blob.from_file_path(font_path)
    face = hb.Face(blob)
    font = hb.Font(face)
    font.scale = (size_px * 64, size_px * 64)
    _HB_CACHE[key] = font
    return font


def _ft_face(font_path: str, size_px: int) -> freetype.Face:
    key = (font_path, size_px)
    f = _FT_CACHE.get(key)
    if f is not None:
        return f
    face = freetype.Face(font_path)
    face.set_pixel_sizes(0, size_px)
    _FT_CACHE[key] = face
    return face


def _shape(text: str, font_path: str, size_px: int):
    """Returns (glyph_infos, glyph_positions) from HarfBuzz."""
    font = _hb_font(font_path, size_px)
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    return buf.glyph_infos, buf.glyph_positions


def _glyph_mask(text: str, font_path: str, size_px: int) -> Image.Image:
    """
    Rasterize `text` to a grayscale alpha mask using HarfBuzz shaping +
    freetype glyph rendering. Returns an L-mode PIL Image sized just to fit.
    """
    infos, positions = _shape(text, font_path, size_px)
    ft = _ft_face(font_path, size_px)

    # Pass 1 — measure
    glyphs = []
    pen_x = 0.0
    max_above = 0
    max_below = 0
    for info, pos in zip(infos, positions):
        ft.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        bm = ft.glyph.bitmap
        left = ft.glyph.bitmap_left
        top = ft.glyph.bitmap_top
        glyphs.append((
            bytes(bm.buffer), bm.width, bm.rows, left, top,
            pos.x_advance / 64.0, pos.x_offset / 64.0, pos.y_offset / 64.0,
        ))
        if bm.rows:
            max_above = max(max_above, top)
            max_below = max(max_below, bm.rows - top)
        pen_x += pos.x_advance / 64.0

    # Margin to avoid clipping outline pixels at edges
    pad = max(int(size_px * 0.15), 8)
    width  = int(pen_x) + pad * 2 + 4
    height = int(max_above + max_below) + pad * 2

    mask = Image.new("L", (width, height), 0)
    cursor_x = pad
    baseline = max_above + pad

    for buf_bytes, w, h, left, top, x_adv, x_off, y_off in glyphs:
        if w and h:
            gm = Image.frombytes("L", (w, h), buf_bytes)
            mask.paste(
                gm,
                (int(cursor_x + left + x_off), int(baseline - top - y_off)),
                gm,
            )
        cursor_x += x_adv

    return mask


def render_text_card(
    text: str,
    font_path: str,
    font_size: int,
    fill: Tuple[int, int, int, int]    = (255, 230, 0, 255),  # yellow
    outline: Tuple[int, int, int, int] = (0, 0, 0, 255),
    outline_px: int                    = 6,
    shadow: Tuple[int, int, int, int]  = (0, 0, 0, 160),
    shadow_offset: Tuple[int, int]     = (3, 3),
) -> Image.Image:
    """
    Render `text` as an RGBA card with outline + shadow.

    The mask is dilated by `outline_px` pixels (via repeated MaxFilter passes,
    each pass adds 1px of dilation) to form the outline shape. The fill is
    composited on top of the outline, and a softened shadow sits behind both.
    """
    glyph_mask = _glyph_mask(text, font_path, font_size)

    # Outline: dilate mask by outline_px in every direction, fill with outline
    # color. MaxFilter(3) dilates by 1px per pass (3x3 kernel = ±1 px).
    if outline_px > 0:
        outline_mask = glyph_mask
        for _ in range(outline_px):
            outline_mask = outline_mask.filter(ImageFilter.MaxFilter(3))
    else:
        outline_mask = None

    # Compute canvas size that fits the outline + shadow offset
    base_w, base_h = glyph_mask.size
    pad_extra = outline_px + max(abs(shadow_offset[0]), abs(shadow_offset[1]))
    canvas = Image.new(
        "RGBA",
        (base_w + pad_extra * 2, base_h + pad_extra * 2),
        (0, 0, 0, 0),
    )

    # Shadow — softened outline-shape, offset
    if shadow and outline_mask is not None:
        shadow_layer = Image.new("RGBA", outline_mask.size, shadow)
        shadow_softened_mask = outline_mask.filter(ImageFilter.GaussianBlur(2))
        canvas.paste(
            shadow_layer,
            (pad_extra + shadow_offset[0], pad_extra + shadow_offset[1]),
            shadow_softened_mask,
        )

    # Outline
    if outline_mask is not None:
        outline_layer = Image.new("RGBA", outline_mask.size, outline)
        canvas.paste(outline_layer, (pad_extra, pad_extra), outline_mask)

    # Fill
    fill_layer = Image.new("RGBA", glyph_mask.size, fill)
    canvas.paste(fill_layer, (pad_extra, pad_extra), glyph_mask)

    return canvas
