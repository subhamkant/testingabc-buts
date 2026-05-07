"""
Subtitle Generator
==================
Word-level karaoke captions burned into the final video.

Pipeline:
  1. Send the full-narration audio to Groq Whisper (whisper-large-v3) with
     word-level timestamps. Free with the existing GROQ_API_KEY — same quota
     as the script generator (14,400 RPD).
  2. Group words into 1-2 word "cards" so on-screen text is readable on a
     phone in 1 second.
  3. Write an ASS subtitle file with a snappy scale pop-in animation per card.
  4. Burn into video via FFmpeg's `subtitles` filter.

The whole thing is gated by the BURN_SUBTITLES env var (default "1"). Failures
are silently swallowed — the video is never lost over a subtitle issue.
"""

import os
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

# Vertical position from bottom — ASS MarginV. 1080x1920 portrait.
# 600 puts the caption band roughly at 70% screen height (lower-third feel).
MARGIN_V       = 520
MARGIN_LR      = 80                       # side margins to keep words centered

WORDS_PER_CARD = 2                        # 1-2 short words per card; 1 if word > 6 chars

# Pop-in animation timings (ms)
FADE_IN_MS     = 60
SCALE_UP_MS    = 140
SCALE_DOWN_MS  = 220
HOLD_AFTER_MS  = 60                       # extra hang time after word's end


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
        return cleaned
    except Exception as e:
        print(f"    [subs] Groq Whisper error: {e}")
        return []


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


# ── ASS file writer ───────────────────────────────────────────────────────────

def _ass_time(seconds: float) -> str:
    """Format seconds as ASS time string H:MM:SS.cc (centiseconds)."""
    if seconds < 0:
        seconds = 0
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_escape(text: str) -> str:
    """Escape characters that have meaning in ASS dialogue lines."""
    return (text
            .replace("\\", "\\\\")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("\n", "\\N"))


def write_ass_subtitles(cards: list, output_path: str) -> bool:
    """
    Write an ASS file with one Dialogue line per card. Each card has a
    fade-in + scale pop-in animation for snappy, readable phone captions.
    """
    if not cards:
        return False

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes
WrapStyle: 0
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{FONT_NAME},{FONT_SIZE},{PRIMARY_COLOR},{SECONDARY},{OUTLINE_COLOR},{BACK_COLOR},-1,0,0,0,100,100,0,0,1,{OUTLINE_PX},{SHADOW_PX},2,{MARGIN_LR},{MARGIN_LR},{MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for c in cards:
        start_s = c["start"]
        end_s   = c["end"] + HOLD_AFTER_MS / 1000
        start   = _ass_time(start_s)
        end     = _ass_time(end_s)

        # Pop-in: fade(80,60), scale 0%→115% over 140ms, then 115%→100% over 220ms
        anim = (
            f"{{\\fad({FADE_IN_MS},{HOLD_AFTER_MS})"
            f"\\t(0,{SCALE_UP_MS},\\fscx115\\fscy115)"
            f"\\t({SCALE_UP_MS},{SCALE_UP_MS + SCALE_DOWN_MS},\\fscx100\\fscy100)}}"
        )
        text = anim + _ass_escape(c["text"])
        events.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")

    return True


# ── FFmpeg burn ───────────────────────────────────────────────────────────────

def _burn_subtitles(video_path: str, ass_path: str) -> bool:
    """
    Burn the ASS subtitles into the video in-place via FFmpeg.

    libass needs the subtitle file path with forward slashes and (on Windows)
    no drive-letter colon, so we use a relative path from CWD.
    """
    import subprocess

    # FFmpeg subtitles filter wants escaped colons and forward slashes.
    # Easiest workaround: pass a relative path from the working directory.
    rel = os.path.relpath(ass_path).replace("\\", "/")
    # Colons inside the filter argument also need escaping
    safe = rel.replace(":", "\\:").replace("'", "\\'")

    out = video_path.replace(".mp4", "_subs.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles='{safe}'",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        out,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[:500] if result.stderr else ""
        print(f"    [subs] FFmpeg burn failed:\n    {err}")
        if os.path.exists(out):
            os.remove(out)
        return False
    os.replace(out, video_path)
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

    os.makedirs("temp/subs", exist_ok=True)
    ass_path = "temp/subs/captions.ass"
    if not write_ass_subtitles(cards, ass_path):
        print("    [subs] ASS write failed — skipping")
        return False

    print("    Burning subtitles into video...")
    if not _burn_subtitles(video_path, ass_path):
        return False

    print("    [OK] Subtitles burned into video")
    return True
