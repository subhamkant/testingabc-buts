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


# ── Title card style (Phase 11 retention refactor, 2026-06-02) ───────────────
# Hardcoded title-card text rendered for the first 2.5s. Larger font than
# word-cards (visible in the 0.5s swipe-decision window), positioned in the
# center-upper third (y=600 from top of 1920) so it doesn't overlap the
# word-subtitle band at MARGIN_V=520 from bottom (≈y=1400 of 1920).
TITLE_CARD_FONT_SIZE     = 96
TITLE_CARD_OUTLINE_PX    = 8       # heavier outline than word cards (6)
TITLE_CARD_FILL_RGBA     = (255, 230, 80, 255)  # warm-amber yellow, matches LUT
TITLE_CARD_OUTLINE_RGBA  = (0, 0, 0, 255)
TITLE_CARD_SHADOW_RGBA   = (0, 0, 0, 200)
TITLE_CARD_SHADOW_OFFSET = (4, 4)
TITLE_CARD_Y_FROM_TOP    = 600   # center-upper third of 1920
TITLE_CARD_START_S       = 0.0
TITLE_CARD_END_S         = 2.5
# Hard cut at t=2.5s is INTENTIONAL — soft fades lull attention. Per
# user direction 2026-06-02: title card vanishing abruptly acts as a
# secondary attention reset just as the first-impression decision
# finishes processing.
TITLE_CARD_FADE_IN_S     = 0.0
TITLE_CARD_FADE_OUT_S    = 0.0


def _render_title_card_png(hook_title: str, out_path: str) -> dict | None:
    """Phase 11 retention refactor 2026-06-02. Render the hook_title as a
    standalone large-font yellow-with-black-stroke PNG for the t=0 overlay.
    Reuses the existing text_renderer pipeline (NotoSansDevanagari-Bold)
    so Hindi + English both render cleanly. Returns a card dict with
    {path, start, end, w, h, y_expr} or None on failure / empty title."""
    if not hook_title or not hook_title.strip():
        return None
    if not os.path.exists(FONT_PATH):
        print(f"    [title-card] font missing at {FONT_PATH} — skipping title card")
        return None
    try:
        from pipeline.text_renderer import render_text_card
        img = render_text_card(
            hook_title.strip(),
            font_path=FONT_PATH,
            font_size=TITLE_CARD_FONT_SIZE,
            fill=TITLE_CARD_FILL_RGBA,
            outline=TITLE_CARD_OUTLINE_RGBA,
            outline_px=TITLE_CARD_OUTLINE_PX,
            shadow=TITLE_CARD_SHADOW_RGBA,
            shadow_offset=TITLE_CARD_SHADOW_OFFSET,
        )
    except Exception as e:
        print(f"    [title-card] render failed: {e}")
        return None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, "PNG")
    w, h = img.size
    return {
        "path":   out_path,
        "start":  TITLE_CARD_START_S,
        "end":    TITLE_CARD_END_S,
        "w":      w,
        "h":      h,
        # Override default lower-third positioning: anchor at fixed Y from top.
        "y_expr": f"{TITLE_CARD_Y_FROM_TOP}",
        # No fade — hard cut. Marker key checked in _build_overlay_filter.
        "_no_fade": True,
    }


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
# Latin variants (English-mode Whisper) plus Devanagari variants (Hindi-mode
# Whisper transcribes the spoken brand as "व्यासा"/"व्यास" since it isn't in
# the model's dictionary).
_DIRECT_FIXES = {
    # Latin — common Whisper mishearings of "Vyasa"
    "vyyas":   "Vyasa",
    "vyaas":   "Vyasa",
    "viyas":   "Vyasa",
    "viyasa":  "Vyasa",
    "vayasa":  "Vyasa",
    "vyas":    "Vyasa",
    "vyassa":  "Vyasa",
    "wyasa":   "Vyasa",
    "wyas":    "Vyasa",
    "byasa":   "Vyasa",
    "byaas":   "Vyasa",
    "biyasa":  "Vyasa",
    "wasa":    "Vyasa",
    "vassa":   "Vyasa",
    "visa":    "Vyasa",   # high-confidence brand mishear in our corpus
    "visor":   "Vyasa",
    "viasa":   "Vyasa",
    # Devanagari — Whisper-Hindi transcribes the spoken brand this way
    "व्यासा":  "Vyasa",
    "व्यास":   "Vyasa",
    "वयासा":   "Vyasa",
    "वायसा":   "Vyasa",
    "वियासा":  "Vyasa",
    "बयासा":   "Vyasa",
}

# Tokens we'd recognize as the spoken letters "AI" (the second half of the
# brand "Vyasa AI"). Whisper sometimes hears it as "Al" (lowercase L), as the
# Devanagari "एआई" / "एआय" / "आई", or drops it entirely.
_AI_TOKENS = {
    "AI", "ai", "Al", "AL", "al", "A.I.", "A.I", "a.i.",
    "एआई", "एआय", "ए.आई", "ए.आय", "आई", "आय", "ऐ.आय",
}

# Concatenated "Vyasai" / "Vyasa.ai" cases — Whisper occasionally emits the
# brand as a single glued token. We split it into two so the burned-in
# subtitle keeps the "Vyasa" → "AI" two-card cadence that matches the other
# correction passes.
_CONCAT_BRAND_RE = None  # initialized lazily inside _correct_brand_transcripts


def _correct_brand_transcripts(words: list) -> list:
    """
    Three passes over Whisper word output:
      1. Per-word direct fixes (Vyyas/व्यासा → Vyasa, etc.)
      2. "Vyasa <Al>" → "Vyasa AI" (mistranscribed second token).
      3. "Vyasa <not-AI>" → insert "AI" word so the burned-in subtitle
         reads "Vyasa AI" even when Whisper dropped the AI token entirely.
    Inserted AI tokens get a tiny synthetic time slice from the preceding
    "Vyasa" word so the overlay still appears in sequence.
    """
    if not words:
        return words

    import re as _re

    # Pre-pass: split concatenated "Vyasai" / "Vyasa.ai" / "vyasaai" tokens
    # into ["Vyasa", "AI"] in place. The Latin variants we accept on the
    # "Vyasa..." prefix mirror the fuzzy keys in _DIRECT_FIXES so a glued
    # mishear like "viyasai" still gets caught here.
    _concat_re = _re.compile(
        r"^(v[iy]+a?s+s?a?|by[ay]?s+a?|w[iy]+a?s+a?|vass+a?)(\s*[\.\-]?\s*)(a[il]\.?i?\.?)$",
        _re.IGNORECASE,
    )
    expanded = []
    for w in words:
        m = _concat_re.match((w.get("word") or "").strip())
        if not m:
            expanded.append(w)
            continue
        start = float(w.get("start", 0.0))
        end   = float(w.get("end",   start + 0.30))
        dur   = max(end - start, 0.10)
        mid   = start + dur * 0.62
        expanded.append({"word": "Vyasa", "start": start, "end": mid})
        expanded.append({"word": "AI",    "start": mid,   "end": end})
    words = expanded

    def _fix_token(token: str) -> str:
        # First try the full token (handles Devanagari with combining vowel
        # signs like व्यासा where Mn/Mc marks would otherwise be misread as
        # trailing punctuation by a generic \W strip).
        full_replacement = _DIRECT_FIXES.get(token.lower())
        if full_replacement is not None:
            return full_replacement
        # Fall back to ASCII-punctuation strip — handles "Vyasa," / "Vyasa."
        # / "Vyasa!" without touching Devanagari combining marks.
        m = _re.match(r"^([\s\.,!?;:'\"\(\)\-]*)(.*?)([\s\.,!?;:'\"\(\)\-]*)$", token)
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
        if cur == "Vyasa" and nxt_core in _AI_TOKENS - {"AI"}:
            trail = nxt["word"][len(nxt_core):]
            nxt["word"] = "AI" + trail

    # Pass 3: insert "AI" after "Vyasa" when the next token isn't already
    # an AI variant. Brand consistency — viewers should ALWAYS see "Vyasa AI"
    # in the burned-in subtitle, never "Vyasa" alone.
    out = []
    i = 0
    while i < len(words):
        out.append(words[i])
        cur_core = words[i]["word"].strip(",.!?:;")
        if cur_core == "Vyasa":
            nxt_core = (
                words[i + 1]["word"].strip(",.!?:;")
                if i + 1 < len(words) else ""
            )
            if nxt_core not in _AI_TOKENS:
                # Synthesize a 0.20s "AI" word right after the Vyasa token,
                # eating into the gap before the next word (or extending past
                # the Vyasa end if no gap). Keeps the overlay sequence intact.
                v_end = float(words[i].get("end", 0.0))
                next_start = (
                    float(words[i + 1].get("start", v_end + 0.30))
                    if i + 1 < len(words) else v_end + 0.30
                )
                ai_start = v_end
                ai_end   = min(v_end + 0.30, max(next_start - 0.02, v_end + 0.05))
                if ai_end > ai_start:
                    out.append({
                        "word":  "AI",
                        "start": ai_start,
                        "end":   ai_end,
                    })
        i += 1

    return out


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

# How many PNG overlays to chain into a single FFmpeg invocation. Each chunk
# re-encodes the FULL video (libx264 fast, crf=20) so total subtitle time
# scales with chunk count — bigger chunks = fewer passes = less total time.
#
# 24 was set defensively for filter-graph compile cost, but FFmpeg handles
# 80+ overlays in one filter_complex without issue. With 80, a typical
# 187-card script runs in 3 passes instead of 8, cutting subtitle overlay
# from ~24 min to ~10 min — critical to fit inside the 29-min GHA cap.
# Output is bit-identical to the 24-chunk version (same overlays, same order).
_OVERLAY_CHUNK = 80


def _build_overlay_filter(cards: list, target_h: int) -> str:
    """
    Build an FFmpeg filter_complex that overlays each PNG card on the video,
    centered horizontally. Default vertical position is MARGIN_V from the
    bottom (word-subtitle lower-third). Each overlay is gated to its
    [start, end] window with fade in/out.

    Per-card overrides (Phase 11 retention refactor 2026-06-02):
      - card["y_expr"] (str): override the default y position. The title
        card uses a fixed Y from top instead of from bottom.
      - card["_no_fade"] (bool): skip the fade-in/out filter — hard cut
        at start/end. Used by the title card for the deliberate t=2.5s
        attention-reset cut.
    """
    parts = ["[0:v]format=yuva420p[v0]"]
    prev = "[v0]"

    for idx, c in enumerate(cards, start=1):
        no_fade = bool(c.get("_no_fade"))
        if no_fade:
            # Title card: no alpha animation, just format conversion
            parts.append(f"[{idx}:v]format=rgba[c{idx}]")
        else:
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
        # Vertical position: per-card override OR default to MARGIN_V from bottom.
        y_expr = c.get("y_expr") or f"H-{MARGIN_V}-h"
        parts.append(
            f"{prev}[c{idx}]overlay="
            f"x=(W-w)/2:y={y_expr}"
            f":enable='between(t,{c['start']:.3f},{c['end']:.3f})'"
            f"{out_label}"
        )
        prev = out_label

    return ";".join(parts)


def _burn_chunk(video_path: str, cards: list, output_path: str,
                hook_anchor_path: str | None = None) -> bool:
    """Run a single FFmpeg pass overlaying up to _OVERLAY_CHUNK cards.

    Phase 11 retention refactor 2026-06-02: when `hook_anchor_path` is
    provided AND the file exists, layer the SFX into the audio via `amix`
    at t=0. The SFX input is appended AFTER all card inputs so it doesn't
    shift any card filter-graph indices. SFX index is computed
    dynamically — `len(inputs) // 2` after appending — so the filter
    graph never references a hardcoded stream index that could shift
    when the title card is absent (this would have crashed the pipeline
    if `hook_anchor_path` was hardcoded to `[2:a]` per the original plan
    draft).
    """
    # Track ffmpeg input INDEX (number of `-i` flags seen so far) — NOT the
    # length of the inputs list, because card inputs use 6 list elements each
    # (`-loop 1 -t X -i path`) while bare `-i path` uses 2. The earlier
    # `len(inputs)//2 - 1` computation crashed when cards >> 1 because it
    # confused the two element counts.
    inputs = ["-i", video_path]
    next_input_idx = 1   # video is input 0, next is index 1
    for c in cards:
        inputs += ["-loop", "1", "-t", f"{(c['end'] - c['start']) + 0.1:.3f}", "-i", c["path"]]
        next_input_idx += 1

    # Dynamic SFX index — exact ffmpeg input position after all card inputs.
    # The os.path.exists() gate ensures a missing SFX file degrades gracefully
    # (render proceeds without the t=0 audio anchor) instead of crashing the
    # cron job.
    sfx_idx = None
    if hook_anchor_path and os.path.exists(hook_anchor_path):
        inputs += ["-i", hook_anchor_path]
        sfx_idx = next_input_idx
        next_input_idx += 1

    filter_complex = _build_overlay_filter(cards, target_h=1920)

    # Audio path: bare-copy unless SFX present, then amix at t=0.
    audio_args = []
    if sfx_idx is not None:
        # Phase 16 D1 (2026-06-09) — TWO changes to make the hook-anchor
        # SFX actually audible at t=0:
        # (1) Pre-boost the SFX +6dB via `volume=2.0` before the mix.
        #     Linear 2.0 = +6dB. Without this, the SFX sits below the
        #     Tier 1.5.e music floor (~0.035 = -29dB) and is inaudible.
        # (2) Add `normalize=0` to amix so it does NOT divide each input
        #     by sum-of-weights. amix's default normalize=1 with weights
        #     '1.0 0.7' divides narration by 1.7 → effective 0.59x =
        #     -4.6dB drop on narration. This silently quietens the
        #     entire video whenever the SFX is present. Disabling
        #     normalize preserves narration at full level AND lets the
        #     boosted SFX punch through.
        filter_complex += (
            f";[{sfx_idx}:a]volume=2.0[sfx_boosted];"
            f"[0:a][sfx_boosted]"
            f"amix=inputs=2:duration=longest:weights='1.0 0.7':normalize=0"
            f"[a_mixed]"
        )
        audio_args = [
            "-map", "[a_mixed]",
            "-c:a", "aac", "-b:a", "192k",
        ]
    else:
        audio_args = [
            "-map", "0:a?",
            "-c:a", "copy",
        ]

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        *audio_args,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
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


def _burn_subtitles_overlay(video_path: str, cards: list,
                             hook_anchor_path: str | None = None) -> bool:
    """
    Overlay every card PNG onto the video. Big card lists are processed in
    chunks of _OVERLAY_CHUNK so each FFmpeg filter graph stays manageable.

    Phase 11 retention refactor 2026-06-02: when `hook_anchor_path` is
    provided, it is applied ONLY to the FIRST chunk's audio (amix at t=0).
    Subsequent chunks copy the already-mixed audio forward. The title
    card (if present) is always card index 0 inside the first chunk.
    """
    if not cards:
        return False

    work_path = video_path
    tmp_paths = []

    chunks = [cards[i:i + _OVERLAY_CHUNK] for i in range(0, len(cards), _OVERLAY_CHUNK)]
    for ci, chunk in enumerate(chunks):
        out_path = video_path.replace(".mp4", f"_subs_pass{ci}.mp4")
        # Only the FIRST chunk gets the SFX amix — subsequent chunks copy
        # whatever audio (already-mixed) is in the input from the prior pass.
        chunk_sfx = hook_anchor_path if ci == 0 else None
        ok = _burn_chunk(work_path, chunk, out_path, hook_anchor_path=chunk_sfx)
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

# Default hook-anchor SFX path. Phase 11 retention refactor 2026-06-02.
# Reuses the existing heartbeat_climax.wav as the t=0 audio jolt (sharp
# attack, < 1s decay). Override via HOOK_ANCHOR_SFX env var to a custom
# wav. The os.path.exists() check inside _burn_chunk ensures pipeline
# degrades gracefully (no SFX) if the file is missing — never crashes.
_DEFAULT_HOOK_ANCHOR_SFX = os.path.join(
    os.path.dirname(__file__), "..", "assets", "heartbeat_climax.wav"
)


def apply_subtitles(video_path: str, audio_path: str, language: str = "hi",
                    hook_title: str = "") -> bool:
    """
    End-to-end: transcribe audio → group into cards → render PNGs →
    overlay onto video. Returns True if subtitles were applied.

    Phase 11 retention refactor 2026-06-02:
      - `hook_title` (optional): when non-empty, render the title-card
        PNG and PREPEND it as the first overlay (card index 0). Visible
        from t=0.0 to t=2.5 with a deliberate HARD cut (no fade). Sits
        at y=600 from top, above the word-subtitle band.
      - Hook-anchor SFX (`HOOK_ANCHOR_SFX` env var or default
        `assets/heartbeat_climax.wav`): when the file exists, layered
        into the audio via amix at t=0. The SFX adds a sharp auditory
        jolt paired with the title card for dual-modal attention.

    The original video file is preserved unchanged on any failure.
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

    # Phase 11 retention refactor: prepend title card if provided.
    if hook_title:
        title_card_path = os.path.join(cards_dir, "title_card.png")
        title_card = _render_title_card_png(hook_title, title_card_path)
        if title_card is not None:
            print(f"    [title-card] '{hook_title}' rendered "
                  f"({title_card['w']}×{title_card['h']}) — overlaid at "
                  f"y={TITLE_CARD_Y_FROM_TOP}, t=0 to {TITLE_CARD_END_S}s")
            rendered = [title_card] + rendered
        else:
            print(f"    [title-card] could not render '{hook_title[:40]}' — proceeding without")

    # Hook-anchor SFX (env-tunable, gracefully degrades if missing).
    hook_anchor_path = os.environ.get(
        "HOOK_ANCHOR_SFX", _DEFAULT_HOOK_ANCHOR_SFX,
    )
    if hook_anchor_path and os.path.exists(hook_anchor_path):
        print(f"    [hook-anchor] SFX layered at t=0 from {os.path.basename(hook_anchor_path)}")
    else:
        # Silent degradation — log once, don't fail.
        print(f"    [hook-anchor] SFX file not found ({hook_anchor_path}) — proceeding without")
        hook_anchor_path = None

    print("    Overlaying subtitles onto video...")
    if not _burn_subtitles_overlay(video_path, rendered,
                                    hook_anchor_path=hook_anchor_path):
        return False

    print("    [OK] Subtitles burned into video")
    return True
