"""
Wuxia subtitles — pipeline/wuxia_subtitles.py
=============================================
Groq-free Hindi subtitle burn for the wuxia pipeline. Replaces
longform_assembler.apply_longform_subtitles (which depends on Groq Whisper —
observed HTTP 500 outages — and whose fade filter, keyed to each card's input
PTS, makes every card after the first fade to alpha 0 before its enable window).

This module instead:
  * builds cards from the KNOWN per-scene narration + _per_scene_durations
    timing (perfect subtitle<->visual sync, zero transcription dependency),
  * renders them with a DUAL-SCRIPT font (Nirmala UI covers Devanagari + Latin,
    so code-switched "High Heaven Pavilion में" has no tofu boxes),
  * burns with a plain overlay (NO fade) so eof_action=repeat shows each card's
    opaque frame throughout its window.

Font: WUXIA_SUB_FONT env, else assets/fonts/Nirmala.ttc (dual-script), else
NotoSansDevanagari-Bold.ttf (Latin will tofu — Linux/GHA needs a bundled
dual-script font; tracked as a production follow-up).
"""
from __future__ import annotations

import os
import shutil
import subprocess

from pipeline.longform_assembler import _per_scene_durations, LF_WORDS_PER_CARD_MAX
from pipeline.text_renderer import render_text_card
from pipeline.video_assembler import get_audio_duration

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FONT_CANDIDATES = [
    os.environ.get("WUXIA_SUB_FONT", "").strip(),
    os.path.join(_ROOT, "assets", "fonts", "Nirmala.ttc"),
    "/c/Windows/Fonts/Nirmala.ttc",
    r"C:\Windows\Fonts\Nirmala.ttc",
    os.path.join(_ROOT, "assets", "fonts", "NotoSansDevanagari-Bold.ttf"),  # last resort
]
_FONT_SIZE = 46
_MARGIN_BOTTOM = 90


def _font_path() -> str:
    for c in _FONT_CANDIDATES:
        if c and os.path.exists(c):
            return c
    raise RuntimeError("no subtitle font found (set WUXIA_SUB_FONT)")


def _build_cards(scenes: list, char_weights: list, audio_dur: float) -> list:
    nar = [(s.get("narration_hi") or s.get("narration") or "").strip() for s in scenes]
    durs = _per_scene_durations(audio_dur, char_weights, len(scenes))
    cards: list = []
    t = 0.0
    for text, d in zip(nar, durs):
        s_start, s_end = t, t + d
        t = s_end
        words = text.split()
        if not words:
            continue
        chunks = [words[k:k + LF_WORDS_PER_CARD_MAX]
                  for k in range(0, len(words), LF_WORDS_PER_CARD_MAX)]
        tot = sum(len(c) for c in chunks) or 1
        ct = s_start
        for ch in chunks:
            cs = ct
            ce = min(s_end, ct + (len(ch) / tot) * d)
            ct = ce
            cards.append({"text": " ".join(ch), "start": cs, "end": ce})
    return cards


def burn_wuxia_subtitles(video_path: str, audio_path: str, script: dict,
                         char_weights: list) -> bool:
    """Burn scene-synced dual-script Hindi subtitles onto video_path (in place).
    Returns True on success. Non-fatal to the caller (wrap in try/except)."""
    if os.environ.get("BURN_SUBTITLES", "1").strip().lower() not in ("1", "true", "yes"):
        print("    [wuxia-subs] BURN_SUBTITLES disabled — skipping")
        return False
    if not (os.path.exists(video_path) and os.path.exists(audio_path)):
        print("    [wuxia-subs] missing input — skipping")
        return False

    font = _font_path()
    audio_dur = get_audio_duration(audio_path)
    cards = _build_cards(script.get("scenes", []), char_weights, audio_dur)
    if not cards:
        print("    [wuxia-subs] no cards built — skipping")
        return False
    print(f"    [wuxia-subs] {len(cards)} cards, font={os.path.basename(font)}, "
          f"audio={audio_dur:.1f}s", flush=True)

    cdir = "temp/subs/wuxia_cards"
    shutil.rmtree(cdir, ignore_errors=True)
    os.makedirs(cdir, exist_ok=True)
    rendered: list = []
    for i, c in enumerate(cards):
        try:
            img = render_text_card(
                c["text"], font_path=font, font_size=_FONT_SIZE,
                fill=(255, 255, 255, 255), outline=(0, 0, 0, 255), outline_px=3,
                shadow=(0, 0, 0, 180), shadow_offset=(2, 2),
            )
        except Exception as e:
            print(f"    [wuxia-subs] card {i} render failed: {e}")
            continue
        p = os.path.join(cdir, f"card_{i:03d}.png")
        img.save(p, "PNG")
        rendered.append({"path": p, "start": c["start"], "end": c["end"]})
    if not rendered:
        return False

    # ONE ffmpeg pass = card overlays + end fade-to-black (video) + audio fade.
    # This is the episode's FINAL video re-encode, so we fold the end-fade in here
    # (rather than a separate pass in the assembler) to avoid compounding CRF loss.
    # No card fades — eof_action=repeat holds each opaque card during its window.
    inputs = ["-i", video_path]
    for c in rendered:
        inputs += ["-loop", "1", "-t", f"{c['end'] - c['start'] + 0.1:.3f}", "-i", c["path"]]
    parts = ["[0:v]format=yuva420p[v0]"]
    prev = "[v0]"
    for idx, c in enumerate(rendered, start=1):
        out = "[vsub]" if idx == len(rendered) else f"[v{idx}]"
        parts.append(f"[{idx}:v]format=rgba[c{idx}]")
        parts.append(
            f"{prev}[c{idx}]overlay=x=(W-w)/2:y=H-{_MARGIN_BOTTOM}-h:"
            f"enable='between(t,{c['start']:.3f},{c['end']:.3f})':eof_action=repeat{out}"
        )
        prev = out

    fade_s = 1.3
    st = max(audio_dur - fade_s, 0.0)
    parts.append(f"[vsub]fade=t=out:st={st:.3f}:d={fade_s:.3f}[vout]")
    parts.append(f"[0:a]afade=t=out:st={st:.3f}:d={fade_s:.3f}[aout]")
    fc = ";".join(parts)

    tmp = video_path.replace(".mp4", "_subtmp.mp4")
    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", fc,
           "-map", "[vout]", "-map", "[aout]",
           "-c:v", "libx264", "-preset", "slow", "-crf", "16",
           "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", tmp]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace")[-800:] if r.stderr else ""
        print(f"    [wuxia-subs] burn failed:\n    {err}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, video_path)
    print("    [OK] Wuxia subtitles burned in (dual-script, scene-synced) + end fade")
    return True
