"""
Subtitle Generator
==================
Word-level captions burned into the final video.

Pipeline:
  1. Send the full-narration audio to Groq Whisper (whisper-large-v3) with
     word-level timestamps. Free with the existing GROQ_API_KEY — same quota
     as the script generator (14,400 RPD).
  2. Group words into 1-2 word "cards" so on-screen text is readable on a
     phone in 1 second.
  3. Render each card as an RGBA PNG using HarfBuzz (uharfbuzz) shaping +
     freetype glyph rasterization. This bypasses FFmpeg's libass/drawtext
     text path, which on the gyan.dev Windows build silently skips Indic
     complex shaping (i-mātrās misplaced, conjuncts not forming).
  4. Overlay each PNG onto the video for its time window, with fade in/out.

The whole thing is gated by the BURN_SUBTITLES env var (default "1"). Failures
are silently swallowed — the video is never lost over a subtitle issue.
"""

import os
import glob
import shutil
import subprocess
import requests


# ── Style config ──────────────────────────────────────────────────────────────

# ASS color is &HAABBGGRR (alpha + BGR hex).
# Yellow (FFFF00) = BGR 00FFFF → &H0000FFFF.
# Black outline (000000) → &H00000000.
PRIMARY_COLOR  = "&H0000FFFF"   # bright yellow
OUTLINE_COLOR  = "&H00000000"   # black outline
BACK_COLOR     = "&H80000000"   # 50% black backing (for tough backgrounds)
SECONDARY      = "&H000000FF"   # red karaoke fill (unused, keep for compat)

FONT_NAME      = "Noto Sans Devanagari"  # handles Latin + Devanagari
FONT_SIZE      = 88                       # large for phone readability
OUTLINE_PX     = 6                        # thick outline
SHADOW_PX      = 2

# Bundled font — required because some FFmpeg builds and many systems lack a
# proper Devanagari font with full Indic OpenType tables. Shipping our own
# guarantees consistent rendering across machines.
FONT_PATH      = os.path.join(
    os.path.dirname(__file__), "..", "assets", "fonts", "NotoSansDevanagari-Bold.ttf"
)

# Vertical position from bottom — equivalent to ASS MarginV. 1080x1920 portrait.
# 520 puts the caption band roughly at 70% screen height (lower-third feel).
MARGIN_V       = 520

WORDS_PER_CARD = 2                        # 1-2 short words per card; 1 if word > 6 chars

# Per-card timing
FADE_IN_S      = 0.06
FADE_OUT_S     = 0.08
HOLD_AFTER_S   = 0.06                     # extra hang time after word's end


# ── Groq Whisper ──────────────────────────────────────────────────────────────

def _groq_transcribe_words(audio_path: str, language: str) -> list:
    """
    Send audio to Groq Whisper for word-level timestamps.
    Returns list of {'word': str, 'start': float, 'end': float} or [] on failure.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("    [subs] GROQ_API_KEY missing — skipping subtitles")
        return []

    try:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
            data = {
                "model": "whisper-large-v3",
                "language": language,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            }
            r = requests.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                data=data,
                timeout=90,
            )
        if r.status_code != 200:
            print(f"    [subs] Groq Whisper HTTP {r.status_code}: {r.text[:200]}")
            return []
        body = r.json()
        words = body.get("words") or []
        cleaned = []
        for w in words:
            word = (w.get("word") or "").strip()
            start = w.get("start")
            end   = w.get("end")
            if word and start is not None and end is not None and end > start:
                cleaned.append({"word": word, "start": float(start), "end": float(end)})
        return _correct_brand_transcripts(cleaned)
    except Exception as e:
        print(f"    [subs] Groq Whisper error: {e}")
        return []


# ── Brand transcript corrections ─────────────────────────────────────────────
# Whisper isn't trained on "Vyasa" so it phonetically transcribes the brand as
# "Vyyas" (or "Vyas") and the spoken letters "AI" as "Al" / "AL". Without this
# pass, the burned-in subtitle for the subscribe outro reads "Subscribe to
# Vyyas Al" — embarrassing for a brand that appears in every video.

# Direct word-level fixes (case-insensitive, applied to any matching token).
_DIRECT_FIXES = {
    "vyyas":   "Vyasa",
    "vyaas":   "Vyasa",
    "viyas":   "Vyasa",
    "viyasa":  "Vyasa",
    "vayasa":  "Vyasa",
}


def _correct_brand_transcripts(words: list) -> list:
    """
    Run two passes over Whisper word output:
      1. Per-word direct fixes (Vyyas/Vyas → Vyasa, etc.)
      2. Contextual fix: a token "Al" / "AL" immediately after "Vyasa"
         becomes "AI" (the spoken brand "Vyasa AI", not the proper name).
    Punctuation attached to the token is preserved.
    """
    if not words:
        return words

    import re as _re

    def _fix_token(token: str) -> str:
        # Strip leading/trailing punctuation for matching, restore on output
        m = _re.match(r"^([\W_]*)(.*?)([\W_]*)$", token, flags=_re.UNICODE)
        if not m:
            return token
        lead, core, trail = m.group(1), m.group(2), m.group(3)
        if not core:
            return token
        replacement = _DIRECT_FIXES.get(core.lower())
        if replacement is not None:
            return f"{lead}{replacement}{trail}"
        return token

    # Pass 1: direct per-word fixes
    for w in words:
        w["word"] = _fix_token(w["word"])

    # Pass 2: contextual "Vyasa Al" → "Vyasa AI"
    for i in range(len(words) - 1):
        cur = words[i]["word"].strip(",.!?:;")
        nxt = words[i + 1]
        nxt_core = nxt["word"].strip(",.!?:;")
        if cur == "Vyasa" and nxt_core in ("Al", "AL", "al"):
            # Preserve any trailing punctuation on the next token
            trail = nxt["word"][len(nxt_core):]
            nxt["word"] = "AI" + trail

    return words


# ── Word grouping ─────────────────────────────────────────────────────────────

def _group_into_cards(words: list, max_words: int = WORDS_PER_CARD) -> list:
    """
    Group words into 1-2 word cards. Long words (>6 chars) get their own card
    so on-screen text never overflows a phone line.

    Returns list of {'text': str, 'start': float, 'end': float}.
    """
    cards = []
    i = 0
    while i < len(words):
        w = words[i]
        # Solo card if the word is long, or last word
        if len(w["word"]) > 6 or i == len(words) - 1 or max_words == 1:
            cards.append({"text": w["word"], "start": w["start"], "end": w["end"]})
            i += 1
            continue
        # Otherwise combine with next short word
        nxt = words[i + 1]
        if len(nxt["word"]) > 6:
            cards.append({"text": w["word"], "start": w["start"], "end": w["end"]})
            i += 1
            continue
        cards.append({
            "text":  f"{w['word']} {nxt['word']}",
            "start": w["start"],
            "end":   nxt["end"],
        })
        i += 2
    return cards


# ── PNG card rendering ────────────────────────────────────────────────────────

def _render_card_pngs(cards: list, out_dir: str) -> list:
    """
    Render each card to an RGBA PNG using HarfBuzz shaping. Returns a list of
    dicts: [{path, start, end, width, height}, ...]. Files outside FFmpeg's
    overlay-friendly path or with rendering failures are skipped (caller
    treats an empty list as "no subtitles").
    """
    from pipeline.text_renderer import render_text_card

    # ASS used &H0000FFFF (yellow) as primary fill, &H00000000 black outline,
    # &H80000000 (50% black) as backing. Translating to RGBA tuples:
    fill_rgba    = (255, 230, 0, 255)   # bright phone-readable yellow
    outline_rgba = (0, 0, 0, 255)
    shadow_rgba  = (0, 0, 0, 160)

    os.makedirs(out_dir, exist_ok=True)
    rendered = []
    for i, c in enumerate(cards):
        try:
            img = render_text_card(
                c["text"],
                font_path=FONT_PATH,
                font_size=FONT_SIZE,
                fill=fill_rgba,
                outline=outline_rgba,
                outline_px=OUTLINE_PX,
                shadow=shadow_rgba,
                shadow_offset=(SHADOW_PX, SHADOW_PX),
            )
        except Exception as e:
            print(f"    [subs] card {i} render failed: {e}")
            continue

        path = os.path.join(out_dir, f"card_{i:03d}.png")
        img.save(path, "PNG")
        w, h = img.size
        rendered.append({
            "path":  path,
            "start": float(c["start"]),
            "end":   float(c["end"]) + HOLD_AFTER_S,
            "w":     w,
            "h":     h,
        })
    return rendered


# ── FFmpeg PNG overlay burn ───────────────────────────────────────────────────

# How many PNG overlays to chain into a single FFmpeg invocation. Going too
# wide creates a huge filter graph that's slow to compile; chunking keeps each
# pass fast and isolates failures.
_OVERLAY_CHUNK = 24


def _build_overlay_filter(cards: list, target_h: int) -> str:
    """
    Build an FFmpeg filter_complex that overlays each PNG card on the video,
    centered horizontally, at a fixed vertical position MARGIN_V from the
    bottom. Each overlay is gated to its [start, end] window with fade in/out.
    """
    parts = ["[0:v]format=yuva420p[v0]"]
    prev = "[v0]"

    for idx, c in enumerate(cards, start=1):
        dur = max(c["end"] - c["start"], 0.05)
        fade_out_st = max(dur - FADE_OUT_S, 0.0)
        # Per-card alpha animation: fade in, hold, fade out
        parts.append(
            f"[{idx}:v]format=rgba,"
            f"fade=t=in:st=0:d={FADE_IN_S:.3f}:alpha=1,"
            f"fade=t=out:st={fade_out_st:.3f}:d={FADE_OUT_S:.3f}:alpha=1"
            f"[c{idx}]"
        )
        out_label = f"[v{idx}]" if idx < len(cards) else "[vout]"
        # Vertical position: MARGIN_V from the bottom, top edge of card
        y_expr = f"H-{MARGIN_V}-h"
        parts.append(
            f"{prev}[c{idx}]overlay="
            f"x=(W-w)/2:y={y_expr}"
            f":enable='between(t,{c['start']:.3f},{c['end']:.3f})'"
            f"{out_label}"
        )
        prev = out_label

    return ";".join(parts)


def _burn_chunk(video_path: str, cards: list, output_path: str) -> bool:
    """Run a single FFmpeg pass overlaying up to _OVERLAY_CHUNK cards."""
    inputs = ["-i", video_path]
    for c in cards:
        inputs += ["-loop", "1", "-t", f"{(c['end'] - c['start']) + 0.1:.3f}", "-i", c["path"]]

    filter_complex = _build_overlay_filter(cards, target_h=1920)

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[-800:] if result.stderr else ""
        print(f"    [subs] overlay pass failed:\n    {err}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False
    return True


def _burn_subtitles_overlay(video_path: str, cards: list) -> bool:
    """
    Overlay every card PNG onto the video. Big card lists are processed in
    chunks of _OVERLAY_CHUNK so each FFmpeg filter graph stays manageable.
    """
    if not cards:
        return False

    work_path = video_path
    tmp_paths = []

    chunks = [cards[i:i + _OVERLAY_CHUNK] for i in range(0, len(cards), _OVERLAY_CHUNK)]
    for ci, chunk in enumerate(chunks):
        out_path = video_path.replace(".mp4", f"_subs_pass{ci}.mp4")
        ok = _burn_chunk(work_path, chunk, out_path)
        if not ok:
            for p in tmp_paths:
                if os.path.exists(p):
                    os.remove(p)
            return False
        # The next chunk reads from this pass's output. Keep tmp around so we
        # can clean up only after the whole sequence succeeds.
        tmp_paths.append(out_path)
        work_path = out_path

    # Replace the original video with the final pass, clean intermediates
    final = tmp_paths[-1]
    os.replace(final, video_path)
    for p in tmp_paths[:-1]:
        if os.path.exists(p):
            os.remove(p)
    return True


# ── Public API ────────────────────────────────────────────────────────────────

def apply_subtitles(video_path: str, audio_path: str, language: str = "hi") -> bool:
    """
    End-to-end: transcribe audio → group into cards → write ASS → burn into video.

    Returns True if subtitles were applied, False otherwise. The original video
    file is preserved unchanged on any failure.
    """
    if os.environ.get("BURN_SUBTITLES", "1").strip() not in ("1", "true", "yes"):
        print("    [subs] BURN_SUBTITLES disabled — skipping")
        return False

    if not os.path.exists(video_path) or not os.path.exists(audio_path):
        print("    [subs] missing input file — skipping")
        return False

    print("    Transcribing audio with Groq Whisper for word timestamps...")
    words = _groq_transcribe_words(audio_path, language)
    if not words:
        print("    [subs] no words returned — skipping")
        return False
    print(f"    Got {len(words)} word timings")

    cards = _group_into_cards(words, max_words=WORDS_PER_CARD)
    print(f"    Grouped into {len(cards)} subtitle cards")

    if not os.path.exists(FONT_PATH):
        print(f"    [subs] font missing at {FONT_PATH} — skipping")
        return False

    # Clean any prior PNGs to avoid mixing cards across runs
    cards_dir = "temp/subs/cards"
    if os.path.isdir(cards_dir):
        shutil.rmtree(cards_dir, ignore_errors=True)

    print("    Rendering subtitle cards via HarfBuzz shaping...")
    rendered = _render_card_pngs(cards, cards_dir)
    if not rendered:
        print("    [subs] no cards rendered — skipping")
        return False
    print(f"    Rendered {len(rendered)} card PNGs")

    print("    Overlaying subtitles onto video...")
    if not _burn_subtitles_overlay(video_path, rendered):
        return False

    print("    [OK] Subtitles burned into video")
    return True
