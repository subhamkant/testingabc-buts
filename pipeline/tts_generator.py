"""
TTS Generator
=============
Priority cascade (each tried in order, first success wins):
  1. ElevenLabs   (env: ELEVENLABS_API_KEY, ELEVENLABS_VOICE_ID) — most "real person"
  2. Gemini TTS   (gemini-2.5-flash-preview-tts, cinematic Charon voice)
  3. Edge TTS SSML narration style — dramatic pacing, pauses, lower pitch
  4. Edge TTS plain fallback

Two public entry points:
  • generate_full_narration(scenes, language) → (audio_path, char_weights)
        ONE TTS call for the entire script. Voice quality is identical
        throughout the video because only one generation happens. The returned
        char_weights let the video assembler distribute scene clips
        proportionally to narration length.

  • generate_voiceover(scenes, language) → list[audio_path]
        LEGACY per-scene generator. Kept for backward compatibility — main.py
        no longer calls this. Each scene gets its own audio file.

Voice consistency across videos: just keep NARRATOR_VOICE / ELEVENLABS_VOICE_ID
fixed in .env. Both providers are deterministic for the same voice ID.

Gemini TTS returns raw PCM (int16 LE, 24kHz mono) → wrapped in WAV → MP3 via FFmpeg.
"""

import edge_tts
import asyncio
import os
import io
import wave
import subprocess
import requests


# Per-pipeline temp root. Mahabharata leaves the env unset → "temp". The
# explainer driver sets PIPELINE_TEMP_ROOT="temp/fe" BEFORE importing this
# module so its working audio files land under temp/fe/audio/, isolated
# from any concurrent or future mahabharata run.
_TEMP_ROOT = os.environ.get("PIPELINE_TEMP_ROOT", "temp")


# ── Voice config ──────────────────────────────────────────────────────────────

# Edge TTS fallback voices (default)
_EDGE_VOICES = {
    "en": "en-US-ChristopherNeural",
    "hi": "hi-IN-MadhurNeural",
}
_EDGE_FALLBACK = {
    "en": "en-US-GuyNeural",
    "hi": "hi-IN-NeerjaNeural",
}

# v4.1 (2026-06-01): Per-series Edge voice overrides. The curiosity series
# (Five Second World EN + Kant Decodes HI) ships content about Indian economy
# / culture and a US/UK voice destroys credibility. Lock both languages to
# Indian Edge voices: en-IN-PrabhatNeural (deep authoritative Indian English
# male) and hi-IN-MadhurNeural (proven native Hindi voice, same one Mahabharata
# uses). For curiosity, Edge becomes the PRIMARY engine (not just fallback) —
# see cascade reorder in generate_full_narration().
_EDGE_VOICES_BY_SERIES = {
    "curiosity": {
        "en": "en-IN-PrabhatNeural",
        "hi": "hi-IN-MadhurNeural",
    },
}
_EDGE_FALLBACK_BY_SERIES = {
    "curiosity": {
        "en": "en-IN-NeerjaNeural",   # Indian English female fallback
        "hi": "hi-IN-NeerjaNeural",
    },
}


def _edge_voice_for(series: str, language: str, fallback: bool = False) -> str:
    """Resolve the Edge voice for the given series + language. Falls back to
    the global _EDGE_VOICES / _EDGE_FALLBACK if the series isn't registered."""
    series_map = (_EDGE_FALLBACK_BY_SERIES if fallback else _EDGE_VOICES_BY_SERIES).get(series)
    if series_map and language in series_map:
        return series_map[language]
    default_map = _EDGE_FALLBACK if fallback else _EDGE_VOICES
    return default_map.get(language, default_map["en"])

# Gemini voice — read from .env so it can be changed without code edits
# Options: Charon (deep/cinematic), Fenrir (strong), Orus (warm), Puck (energetic)
_GEMINI_VOICE = os.environ.get("NARRATOR_VOICE", "Charon")
_GEMINI_MODEL = "gemini-2.5-flash-preview-tts"

# Series-specific Gemini voice overrides. The Mahabharata/Krishna stack stays
# on the cinematic Charon; WhatIf is science-curiosity content and wants an
# energetic, younger-sounding voice (Puck) to match the register.
# An explicit NARRATOR_VOICE_<SERIES> env var overrides this map at runtime.
_GEMINI_VOICE_BY_SERIES = {
    "whatif":      "Puck",
    "mahabharata": _GEMINI_VOICE,
    "krishna":     _GEMINI_VOICE,
    # Explainer = Anti-Hype Analyzer. Charon (the default) is too cinematic /
    # mythology-coded for a "smart friend talking" register. Orus is warm,
    # male, mature — the most natural-sounding conversational option in the
    # Gemini palette. Override with NARRATOR_VOICE_EXPLAINER if you want Puck
    # (energetic) or Fenrir (strong/commanding).
    "explainer":   "Orus",
}


def _gemini_voice_for(series: str) -> str:
    """Resolve the Gemini voice name for the given series — env override > map > default."""
    env_override = os.environ.get(f"NARRATOR_VOICE_{series.upper()}", "").strip()
    if env_override:
        return env_override
    return _GEMINI_VOICE_BY_SERIES.get(series, _GEMINI_VOICE)

# SSML prosody for storytelling feel
_RATE  = "-12%"   # deliberate, unhurried pace
_PITCH = "-8Hz"   # deeper, more gravitas


# ── Gemini TTS ────────────────────────────────────────────────────────────────

def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000) -> bytes:
    """Wrap raw PCM (int16 LE mono) bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit = 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _gemini_tts_keys() -> list[tuple[str, str]]:
    """Collect all configured Gemini API keys in the order to try them.
    Returns [(label, key), ...]. The TTS quota is per-key per-day (10 req/day
    on the free tier), so rotating through every configured key gives us
    N × 10 daily capacity — critical for bilingual long-form which burns
    2-12 TTS calls per render.
    """
    candidates = [
        ("primary",     "GEMINI_API_KEY"),
        ("fallback",    "GEMINI_API_KEY_FALLBACK"),
    ]
    for n in range(2, 6):
        candidates.append((f"fallback-{n}", f"GEMINI_API_KEY_FALLBACK_{n}"))

    keys: list[tuple[str, str]] = []
    for label, var in candidates:
        v = os.environ.get(var, "").strip()
        if v:
            keys.append((label, v))
    return keys


def _gemini_tts(text: str, output_mp3: str, voice: str = None) -> bool:
    """
    Generates audio via Gemini TTS. Defaults to the global _GEMINI_VOICE
    (Charon) but the caller can pass `voice` to override per series — e.g.
    WhatIf passes "Puck" for an energetic, younger register.

    Rotates through every configured Gemini API key on 429 RESOURCE_EXHAUSTED
    so the per-key daily TTS quota doesn't kill a bilingual long-form render.
    Returns True on first success.
    """
    keys = _gemini_tts_keys()
    if not keys:
        return False

    voice_name = voice or _GEMINI_VOICE

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        print(f"    [Gemini TTS] missing google.genai: {e}")
        return False

    for label, api_key in keys:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name
                            )
                        )
                    ),
                ),
            )

            part = response.candidates[0].content.parts[0]
            audio_data = part.inline_data.data  # raw bytes (PCM or WAV)
            mime_type  = part.inline_data.mime_type or ""

            # Save to a temp WAV, then FFmpeg converts to normalized MP3
            tmp_wav = output_mp3 + ".tmp.wav"

            if "wav" in mime_type:
                with open(tmp_wav, "wb") as f:
                    f.write(audio_data)
            else:
                # Assume PCM int16 LE at 24kHz mono
                wav_bytes = _pcm_to_wav(audio_data, sample_rate=24000)
                with open(tmp_wav, "wb") as f:
                    f.write(wav_bytes)

            # Convert WAV → normalized MP3
            result = subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_wav,
                    "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ar", "48000", "-ab", "128k",
                    output_mp3,
                ],
                capture_output=True,
            )

            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

            if result.returncode == 0 and os.path.exists(output_mp3):
                print(f"    [Gemini TTS/{label}] OK (voice={voice_name})")
                return True

        except Exception as e:
            msg = str(e)
            # On 429 RESOURCE_EXHAUSTED, rotate to the next key
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                print(f"    [Gemini TTS/{label}] quota exhausted — trying next key")
                continue
            # On any other error (network, malformed input, etc.), still try
            # the next key — some errors are transient per-region/per-key
            print(f"    [Gemini TTS/{label}] {msg[:200]}")
            continue

    print(f"    [Gemini TTS] all {len(keys)} key(s) exhausted")
    return False


# ── Edge TTS with SSML ────────────────────────────────────────────────────────

# v4.1: SSML break-duration profiles. The legacy Mahabharata/Krishna pipeline
# wants long dramatic pauses (500/400/600ms). The curiosity pipeline uses
# v4.1 emotional-cue narrations where Pro emits MANY ellipses + (pause) markers
# (the bracket-stripper converts those to "..."); long break times bloat a
# 55s Short to 90s+. Halve the break times for curiosity so the pauses
# REGISTER without inflating the timeline.
_SSML_BREAK_PROFILES = {
    "default":   {"sentence": 500, "period": 400, "ellipsis": 600},
    "curiosity": {"sentence": 250, "period": 200, "ellipsis": 300},
}


def _build_ssml(text: str, voice: str, language: str, series: str = "mahabharata") -> str:
    """
    Wraps narration text in SSML with narration-relaxed style + prosody.
    Adds natural pauses after sentence endings for dramatic effect.

    Break-duration profile is series-aware (v4.1): curiosity uses shorter
    pauses to prevent ellipsis-heavy v4.1 narrations from bloating duration
    via Edge SSML's break engine.
    """
    import re
    xml_lang = "hi-IN" if language == "hi" else "en-US"
    breaks = _SSML_BREAK_PROFILES.get(series, _SSML_BREAK_PROFILES["default"])

    # Escape XML special characters
    text_esc = (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    # Add dramatic pause after sentence endings (series-aware durations)
    text_esc = re.sub(r'([।!?])\s*',
                      rf'\1<break time="{breaks["sentence"]}ms"/>', text_esc)
    text_esc = re.sub(r'\.\s+',
                      rf'.<break time="{breaks["period"]}ms"/>', text_esc)
    text_esc = re.sub(r'\.\.\.',
                      rf'...<break time="{breaks["ellipsis"]}ms"/>', text_esc)

    return (
        f'<speak xmlns="http://www.w3.org/2001/10/synthesis"'
        f' xmlns:mstts="http://www.w3.org/2001/mstts"'
        f' xml:lang="{xml_lang}">'
        f'<voice name="{voice}">'
        f'<mstts:express-as style="narration-relaxed">'
        f'<prosody rate="{_RATE}" pitch="{_PITCH}">'
        f'{text_esc}'
        f'</prosody></mstts:express-as></voice></speak>'
    )


async def _edge_tts(text: str, voice: str, output_path: str, use_ssml: bool = True,
                    series: str = "mahabharata") -> bool:
    """Generate audio via Edge TTS. Returns True on success."""
    try:
        if use_ssml:
            # Detect language from voice name
            lang = "hi" if "hi-IN" in voice else "en"
            ssml_text = _build_ssml(text, voice, lang, series=series)
            communicate = edge_tts.Communicate(ssml_text, voice)
        else:
            communicate = edge_tts.Communicate(text, voice, rate=_RATE, pitch=_PITCH)
        await communicate.save(output_path)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1000
    except Exception as e:
        print(f"    [Edge TTS '{voice}'] {e}")
        return False


def _normalize_audio(path: str) -> None:
    """EBU R128 loudness normalization — -16 LUFS, -1.5 dBTP. Skipped if FFmpeg missing."""
    import shutil
    if not shutil.which("ffmpeg"):
        return
    tmp = path + ".norm.mp3"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", path,
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
         "-ar", "48000", "-ab", "128k", tmp],
        capture_output=True,
    )
    if result.returncode == 0:
        os.replace(tmp, path)


# ── Text sanitizer ────────────────────────────────────────────────────────────

def _strip_emotional_cues(text: str) -> str:
    """v4.1 (2026-06-01): convert Pro's bracketed emotional pause cues to
    TTS-compatible punctuation BEFORE the text hits Gemini Charon (which
    would otherwise read '(pause)' literally as the word 'pause').

    The Pro paste prompt at `pro_prompts/spacex_type2.md` instructs the LLM
    to embed these markers inside narration_en / narration_hi strings as
    delivery cues. This function maps them to ellipses / em-dashes / periods
    that Charon naturally pauses on. Idempotent.

    Mapping:
      (pause)      → ' ... '
      (long pause) → ' ... ... '
      (breath)     → ' ... '
      (beat)       → ' — '
      (stop)       → '. '
      (whisper)    → ''         (vocal cue only; visual reminder for Pro)

    Case-insensitive; allows optional internal whitespace.
    """
    import re
    cue_map = (
        (r'\(\s*long\s*pause\s*\)', ' ... ... '),
        (r'\(\s*pause\s*\)',        ' ... '),
        (r'\(\s*breath\s*\)',       ' ... '),
        (r'\(\s*beat\s*\)',         ' — '),
        (r'\(\s*stop\s*\)',         '. '),
        (r'\(\s*whisper\s*\)',      ''),
    )
    for pattern, replacement in cue_map:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _clean_narration(text: str) -> str:
    """Clean narration text before sending to TTS."""
    import re

    # v4.1: strip bracketed emotional cues FIRST so the converted punctuation
    # survives downstream normalization (ellipses + em-dashes stay intact).
    text = _strip_emotional_cues(text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove hashtags & mentions
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)

    # Strip repeated MS/ML token chains (Gemini TTS bug — bot reads "em-ess
    # em-ess em-ess..." when this artifact appears). Targeted to specific
    # known-bad tokens; v4.1 removed the broader `\b[A-Z]{2,}\b` strip
    # because it was killing intentional ALLCAPS emphasis (RBI, UPI, SIX,
    # ROT, BURN, etc.) that Pro now uses for v3 emotional delivery.
    text = re.sub(r'(एमएस|एमएल|MS|ML)+', '', text)

    # Normalize spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


# ── ElevenLabs TTS ────────────────────────────────────────────────────────────

# Default voice IDs (overridable via ELEVENLABS_VOICE_ID env). These are
# stable across all videos so the same narrator voice is used run-to-run.
_DEFAULT_ELEVENLABS_VOICE = "pNInz6obpgDQGcFmaJgB"  # "Adam" — deep cinematic male


# Per-series ElevenLabs voice_settings. The defaults below produce a stable,
# consistent narrator — right for Mahabharata third-person storytelling but
# deadly for Krishna direct-address, which depends on emotional dynamics
# (soft contemplative passages followed by commanding peaks like "उठो पार्थ!").
# Lowering stability and raising style makes ElevenLabs swing harder.
_VOICE_SETTINGS_BY_SERIES = {
    "krishna": {
        "stability": 0.30,        # lower → more emotional swing per sentence
        "similarity_boost": 0.75,
        "style": 0.55,            # higher → more dramatic delivery / inflection
        "use_speaker_boost": True,
    },
}
_DEFAULT_VOICE_SETTINGS = {
    "stability": 0.55,
    "similarity_boost": 0.80,
    "style": 0.30,
    "use_speaker_boost": True,
}

# ── Krishna per-scene voice settings ─────────────────────────────────────────
# A real human actor varies their delivery scene-by-scene — soft on the
# contemplative truth, loud on the imperative peak, warm on the blessing. A
# single ElevenLabs request can't do that — voice_settings are global per call.
# The fix: send each scene as its own TTS request with its own settings, then
# concatenate the resulting mp3s. This is what produces real LRA in the
# output (vs the flat 4.7 LU we measured before).
#
# Scene index here is the position in the FINAL scenes list including the
# appended subscribe outro: 0=opening, 1=hard-truth, 2=imperative-peak,
# 3=reframe, 4=blessing, 5=outro CTA.
_KRISHNA_PER_SCENE_SETTINGS = {
    0: {  # Opening address — contemplative, intimate
        "stability": 0.40,
        "similarity_boost": 0.75,
        "style": 0.45,
        "use_speaker_boost": True,
    },
    1: {  # Hard truth — grounded, certain
        "stability": 0.40,
        "similarity_boost": 0.75,
        "style": 0.50,
        "use_speaker_boost": True,
    },
    2: {  # Imperative peak — maximum dramatic swing (also gets audio tags)
        "stability": 0.10,
        "similarity_boost": 0.75,
        "style": 0.85,
        "use_speaker_boost": True,
    },
    3: {  # Reframe — contemplative again
        "stability": 0.45,
        "similarity_boost": 0.75,
        "style": 0.45,
        "use_speaker_boost": True,
    },
    4: {  # Blessing / charge — warm command
        "stability": 0.30,
        "similarity_boost": 0.75,
        "style": 0.55,
        "use_speaker_boost": True,
    },
    5: {  # Subscribe outro — warm + intimate, slightly more stable
        "stability": 0.45,
        "similarity_boost": 0.80,
        "style": 0.40,
        "use_speaker_boost": True,
    },
}

# ElevenLabs v3 (alpha) accepts inline emotional tags like [determined] /
# [intense] / [shouting] that nudge the model toward a specific delivery.
# We inject these only on Scene 2 (the imperative peak) to push beyond what
# stability/style alone can produce. v3 may not be enabled on every API key —
# the per-scene TTS path tries v3 first, falls back to eleven_multilingual_v2
# without tags if v3 returns 403/404.
_KRISHNA_PEAK_TAG_PREFIX = "[determined] [intense] "

# Per-scene loudness offsets (in dB) applied at concat time. This is the
# DETERMINISTIC source of macro dynamic range — ElevenLabs voice_settings
# vary expressiveness within a scene but produce nearly-flat overall loudness
# scene-to-scene. Hand-scaling each scene to a target dB level is the only
# reliable way to make the imperative peak feel LOUDER than the contemplative
# opening. ~6 dB total spread across scenes contributes ~6 LU to LRA.
_KRISHNA_PER_SCENE_DB = {
    0: -1.0,  # Opening — slightly softer, intimate
    1:  0.0,  # Hard truth — baseline
    2: +4.0,  # IMPERATIVE PEAK — loudest
    3: -2.0,  # Reframe — quietest, contemplative
    4: +1.0,  # Blessing / charge — warm, slightly elevated
    5: -1.0,  # Outro CTA — intimate again
}


def _resolve_voice_id(series: str) -> str:
    """Pick the right ElevenLabs voice_id for a given series."""
    return (
        os.environ.get(f"ELEVENLABS_VOICE_ID_{series.upper()}", "").strip()
        or os.environ.get("ELEVENLABS_VOICE_ID", "").strip()
        or _DEFAULT_ELEVENLABS_VOICE
    )


def _post_elevenlabs(
    text: str,
    output_mp3: str,
    api_key: str,
    voice_id: str,
    model_id: str,
    voice_settings: dict,
    key_label: str,
    normalize: bool = True,
) -> tuple:
    """
    Single ElevenLabs synthesis call. Returns (ok, status_code).
    Caller decides how to interpret status_code (e.g. fall back to a
    different model on 403/404).

    `normalize=False` skips the per-call EBU R128 normalization. CRITICAL
    for the per-scene Krishna flow — if each scene gets independently
    normalized to -16 LUFS, all the per-scene voice_settings work is wasted
    because the loudness variation between scenes is wiped out. Caller
    runs the normalization once on the concatenated final audio instead.
    """
    try:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        body = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=180)
        if resp.status_code != 200:
            print(f"    [{key_label}/{model_id}] HTTP {resp.status_code}: {resp.text[:200]}")
            return False, resp.status_code
        with open(output_mp3, "wb") as f:
            f.write(resp.content)
        if os.path.exists(output_mp3) and os.path.getsize(output_mp3) > 1000:
            if normalize:
                _normalize_audio(output_mp3)
            return True, 200
        return False, 0
    except Exception as e:
        print(f"    [{key_label}/{model_id}] {e}")
        return False, 0


def _elevenlabs_tts(
    text: str,
    output_mp3: str,
    language: str = "en",
    api_key: str = "",
    key_label: str = "ElevenLabs",
    series: str = "mahabharata",
    voice_settings: dict = None,
    try_v3: bool = False,
    normalize: bool = True,
) -> bool:
    """
    Single-call ElevenLabs TTS used by the standard (non-per-scene) flow and
    by the per-scene Krishna flow's individual scene calls.

    `voice_settings` overrides the per-series default settings — used by the
    Krishna per-scene flow to vary delivery scene-by-scene (contemplative
    Scene 1, dramatic peak Scene 3, warm blessing Scene 5, etc.).

    `try_v3=True` first attempts the eleven_v3 model (alpha — has emotional
    audio-tag support). Falls back to eleven_multilingual_v2 if v3 returns
    a 4xx (typically 403/404 for keys without v3 access).

    `normalize=False` (passed by the per-scene Krishna flow) keeps each
    scene's natural loudness so dramatic-peak scenes come out louder than
    contemplative ones. The default True matches the existing single-call
    behaviour.

    Returns True on success.
    """
    if not api_key:
        return False

    voice_id = _resolve_voice_id(series)
    settings = voice_settings or _VOICE_SETTINGS_BY_SERIES.get(series, _DEFAULT_VOICE_SETTINGS)

    if try_v3:
        ok, code = _post_elevenlabs(
            text, output_mp3, api_key, voice_id, "eleven_v3",
            settings, key_label, normalize,
        )
        if ok:
            return True
        # v3 unavailable / not authorized → fall through to v2 (silent retry)
        if code in (400, 401, 403, 404):
            pass
        elif code == 200:
            return True  # already returned
        else:
            return False  # other error — don't retry on a different model

    ok, _ = _post_elevenlabs(
        text, output_mp3, api_key, voice_id, "eleven_multilingual_v2",
        settings, key_label, normalize,
    )
    return ok


def _elevenlabs_keys() -> list:
    """
    Returns the ordered list of ElevenLabs keys to try, with a label per key.
    Order: primary → fallback → fallback_2 → fallback_3 → fallback_4 → fallback_5.
    Empty entries are filtered out so callers don't need to handle them.

    Multi-key rotation mirrors the Gemini cascade — when one ElevenLabs free-tier
    account gets flagged for "unusual activity" (a real risk that hit us on
    2026-05-31), additional accounts on different emails extend the budget.
    Each ElevenLabs free account has 10k chars/month (~5-6 bilingual Shorts).
    """
    candidates = [
        ("ElevenLabs (primary)",    os.environ.get("ELEVENLABS_API_KEY", "").strip()),
        ("ElevenLabs (fallback)",   os.environ.get("ELEVENLABS_API_KEY_FALLBACK", "").strip()),
    ]
    for n in range(2, 6):
        candidates.append(
            (f"ElevenLabs (fallback-{n})",
             os.environ.get(f"ELEVENLABS_API_KEY_FALLBACK_{n}", "").strip())
        )
    return [(label, key) for label, key in candidates if key]


# ── Per-scene Krishna TTS ────────────────────────────────────────────────────

def _concat_audio_files(
    input_paths: list,
    output_path: str,
    gap_s: float = 0.4,
    db_offsets: list = None,
) -> bool:
    """
    Concatenate per-scene MP3 files into one continuous track with a brief
    silence between scenes. The silence creates a natural breath / pause
    boundary the way a real narrator would breathe between scenes.

    `db_offsets`, if provided, must be a list of dB scaling values (one per
    input). Each scene gets `volume={dB}dB` applied before concat, which is
    the deterministic source of macro loudness dynamics. Without this, the
    output LRA is whatever ElevenLabs naturally produces — typically 2-3 LU
    even with per-scene voice_settings variation.

    Returns True on success.
    """
    import subprocess as _sp
    if not input_paths:
        return False

    n = len(input_paths)
    inputs = []
    for p in input_paths:
        inputs += ["-i", p]
    # Single silent source re-used for inter-scene gaps
    silence_idx = n
    inputs += [
        "-f", "lavfi",
        "-t", f"{gap_s:.2f}",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
    ]

    # Build the filter graph: optionally apply per-scene volume,
    # then concat with silence-glue between scenes.
    pre_parts = []
    concat_inputs = []
    for i in range(n):
        if db_offsets and i < len(db_offsets) and db_offsets[i] != 0.0:
            db = db_offsets[i]
            sign = "" if db >= 0 else "-"
            pre_parts.append(f"[{i}:a]volume={sign}{abs(db):.2f}dB[v{i}]")
            concat_inputs.append(f"[v{i}]")
        else:
            concat_inputs.append(f"[{i}:a]")

    parts = []
    for i in range(n):
        parts.append(concat_inputs[i])
        if i < n - 1:
            parts.append(f"[{silence_idx}:a]")
    concat_part = "".join(parts) + f"concat=n={2 * n - 1}:v=0:a=1[aout]"

    filter_complex = ";".join(pre_parts + [concat_part]) if pre_parts else concat_part

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-c:a", "libmp3lame", "-b:a", "192k",
        output_path,
    ]
    result = _sp.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[-400:] if result.stderr else ""
        print(f"    [concat] failed: {err}")
        return False
    return os.path.exists(output_path)


def _krishna_scene_text(text: str, scene_idx: int) -> str:
    """
    Inject ElevenLabs v3 audio tags for the imperative-peak scene (index 2).
    Other scenes return text unchanged. Tags are interpreted by v3 only —
    if we fall back to multilingual_v2, the tags become inert and the LLM
    pronounces them as text… so we strip them on the v2 retry path.
    """
    if scene_idx == 2 and text:
        return _KRISHNA_PEAK_TAG_PREFIX + text
    return text


_TAG_PATTERN = None
def _strip_audio_tags(text: str) -> str:
    """Remove [tag] markers — used when falling back to a model that
    doesn't speak ElevenLabs v3 audio tags."""
    global _TAG_PATTERN
    if _TAG_PATTERN is None:
        import re
        _TAG_PATTERN = re.compile(r"\[[a-zA-Z _]+\]\s*")
    return _TAG_PATTERN.sub("", text).strip()


async def _generate_per_scene_krishna_tts(
    scenes: list,
    output_path: str,
    language: str = "hi",
) -> bool:
    """
    Generate one TTS request per scene with per-scene voice_settings, then
    concatenate the resulting mp3s into one continuous track. This is what
    produces real LRA in the output — each scene has its own dynamic
    character (contemplative Scene 1, dramatic peak Scene 3, warm Scene 5).

    Tries ElevenLabs v3 (with audio tags on the peak scene) first per scene;
    falls back to eleven_multilingual_v2 with tags stripped if v3 fails.

    Returns True on success. False means the per-scene path itself failed —
    caller can fall back to a single-call generation.
    """
    keys = _elevenlabs_keys()
    if not keys:
        return False
    primary_label, primary_key = keys[0]

    os.makedirs(f"{_TEMP_ROOT}/audio/krishna_scenes", exist_ok=True)
    scene_paths = []

    for i, scene in enumerate(scenes):
        text = _clean_narration(scene["narration"])
        if not text:
            continue

        settings = _KRISHNA_PER_SCENE_SETTINGS.get(
            i,
            _VOICE_SETTINGS_BY_SERIES.get("krishna", _DEFAULT_VOICE_SETTINGS),
        )
        scene_path = f"{_TEMP_ROOT}/audio/krishna_scenes/scene_{i:02d}.mp3"

        # First attempt: v3 with audio tags on the peak scene
        v3_text = _krishna_scene_text(text, i)
        ok = await asyncio.to_thread(
            _elevenlabs_tts,
            v3_text, scene_path, language,
            primary_key, f"{primary_label}/scene{i}",
            "krishna", settings, True, False,  # try_v3=True, normalize=False
        )
        if ok and os.path.exists(scene_path):
            scene_paths.append(scene_path)
            print(f"    [OK] Krishna scene {i+1}/{len(scenes)} via ElevenLabs (settings: stab={settings['stability']:.2f}, style={settings['style']:.2f})")
            continue

        # Retry on the same scene with tags stripped + v2 model only
        clean = _strip_audio_tags(text)
        ok = await asyncio.to_thread(
            _elevenlabs_tts,
            clean, scene_path, language,
            primary_key, f"{primary_label}/scene{i}/v2",
            "krishna", settings, False, False,  # no v3, no normalize
        )
        if ok and os.path.exists(scene_path):
            scene_paths.append(scene_path)
            print(f"    [OK] Krishna scene {i+1}/{len(scenes)} via v2 fallback")
            continue

        # Try the fallback key on this scene before giving up
        if len(keys) > 1:
            fallback_label, fallback_key = keys[1]
            ok = await asyncio.to_thread(
                _elevenlabs_tts,
                clean, scene_path, language,
                fallback_key, f"{fallback_label}/scene{i}/v2",
                "krishna", settings, False,
            )
            if ok and os.path.exists(scene_path):
                scene_paths.append(scene_path)
                print(f"    [OK] Krishna scene {i+1}/{len(scenes)} via fallback key")
                continue

        # All ElevenLabs attempts on this scene failed — abort and let the
        # caller fall back to a single-call generation.
        print(f"    [!] Krishna scene {i+1} failed on all ElevenLabs paths")
        return False

    if len(scene_paths) != len(scenes):
        print(f"    [!] Per-scene TTS produced {len(scene_paths)}/{len(scenes)} scenes")
        return False

    # Concatenate per-scene mp3s into one continuous track, applying per-scene
    # volume offsets so the imperative peak comes through louder than the
    # contemplative passages — the deterministic source of macro dynamics.
    db_offsets = [_KRISHNA_PER_SCENE_DB.get(i, 0.0) for i in range(len(scene_paths))]
    print(f"    Concatenating {len(scene_paths)} scene mp3s with 0.4s gaps + per-scene dB offsets {db_offsets}...")
    return _concat_audio_files(scene_paths, output_path, gap_s=0.4, db_offsets=db_offsets)


# ── Single-pass full narration ────────────────────────────────────────────────

async def generate_full_narration(
    scenes: list,
    language: str = "en",
    series: str = "mahabharata",
) -> tuple:
    """
    Generate ONE continuous audio file from all scene narrations concatenated.

    Returns (audio_path, char_weights):
      audio_path     str            — path to the single MP3 file
      char_weights   list[int]      — per-scene narration char count for the
                                      video assembler to size scene clips
                                      proportionally so visuals stay roughly
                                      in sync with the spoken narrative.

    `series` selects a per-series ElevenLabs voice override
    (ELEVENLABS_VOICE_ID_<SERIES>) so the Krishna direct-address format can
    use a distinct divine voice while the standard Mahabharata flow keeps
    the default narrator.

    Provider cascade order is controlled by the PRIMARY_TTS env var:

      PRIMARY_TTS=gemini  (default, Phase 11 retention refactor 2026-06-02):
        Gemini Charon → ElevenLabs → Edge SSML → Edge plain. Saves ~30s
        per render that was previously wasted on failed ElevenLabs
        attempts (free-tier blocked on this device). Charon is the
        Mahabharata default voice — more dramatic-narrator timbre than
        ElevenLabs Adam for the t=0 attention grab.

      PRIMARY_TTS=elevenlabs  (legacy / rollback):
        ElevenLabs → Gemini → Edge. The original cascade order.

    Krishna per-scene ElevenLabs runs FIRST regardless of PRIMARY_TTS
    when ElevenLabs keys are configured — Krishna's per-scene
    voice_settings tuning is series-specific and shouldn't be swapped.

    Once a provider succeeds, the entire video is consistent voice (no
    half-Gemini / half-Edge seams).
    """
    os.makedirs(f"{_TEMP_ROOT}/audio", exist_ok=True)

    # Concatenate per-scene narrations with a sentence delimiter so the model
    # produces a natural breath / pause between scenes.
    delim = "। " if language == "hi" else ". "

    cleaned_parts = []
    char_weights  = []
    for scene in scenes:
        text = _clean_narration(scene["narration"])
        cleaned_parts.append(text)
        char_weights.append(max(len(text), 1))

    full_text = delim.join(cleaned_parts)
    print(f"    Full narration: {len(full_text)} chars across {len(scenes)} scene(s)")

    output_path = f"{_TEMP_ROOT}/audio/narration_full.mp3"

    # ── 0. Krishna per-scene TTS (only when an ElevenLabs key is set) ──
    # Krishna's per-scene voice_settings tuning is series-specific.
    # Runs FIRST regardless of PRIMARY_TTS so the divine-monologue
    # delivery is preserved.
    if series == "krishna" and _elevenlabs_keys():
        print(f"    Krishna mode — per-scene TTS with per-scene voice settings...")
        ok = await _generate_per_scene_krishna_tts(scenes, output_path, language)
        if ok:
            print(f"    [OK] Krishna per-scene narration -> {output_path}")
            return output_path, char_weights
        print("    [!] Per-scene Krishna TTS failed — falling back to single-call mode")

    # Phase 11 retention refactor 2026-06-02: PRIMARY_TTS env-gated
    # cascade ordering. Default "gemini" — Gemini Charon first because
    # ElevenLabs has been blocked on this device (free-tier abuse flag)
    # and the failed-attempt cascade was eating ~30s per render. Charon
    # is the explicit Mahabharata narrator now.
    primary_tts = os.environ.get("PRIMARY_TTS", "gemini").strip().lower()

    async def _try_elevenlabs() -> bool:
        """Try ElevenLabs single-call across all configured keys."""
        for key_label, key in _elevenlabs_keys():
            print(f"    Trying {key_label} (most realistic)...")
            ok = await asyncio.to_thread(
                _elevenlabs_tts, full_text, output_path, language, key, key_label, series
            )
            if ok:
                print(f"    [OK] Full narration via {key_label} ({series})")
                return True
        return False

    async def _try_gemini() -> bool:
        """Try Gemini TTS with the series-appropriate voice."""
        if not os.environ.get("GEMINI_API_KEY", "").strip():
            return False
        gemini_voice = _gemini_voice_for(series)
        print(f"    Trying Gemini TTS ({gemini_voice} voice for {series})...")
        if await asyncio.to_thread(_gemini_tts, full_text, output_path, gemini_voice):
            print(f"    [OK] Full narration via Gemini {gemini_voice}")
            return True
        return False

    if primary_tts == "gemini":
        # 1. Gemini first (default — Phase 11 retention refactor)
        if await _try_gemini():
            return output_path, char_weights
        # 2. ElevenLabs fallback (if Gemini 5xx / quota)
        if await _try_elevenlabs():
            return output_path, char_weights
    else:
        # Legacy: ElevenLabs first → Gemini fallback
        if await _try_elevenlabs():
            return output_path, char_weights
        if await _try_gemini():
            return output_path, char_weights

    # 3. Edge TTS SSML
    voice_pri = _EDGE_VOICES.get(language, _EDGE_VOICES["en"])
    voice_alt = _EDGE_FALLBACK.get(language, _EDGE_FALLBACK["en"])

    print("    Trying Edge TTS (SSML, primary voice)...")
    if await _edge_tts(full_text, voice_pri, output_path, use_ssml=True, series=series):
        _normalize_audio(output_path)
        print("    [OK] Full narration via Edge SSML")
        return output_path, char_weights

    # 4. Edge TTS fallback voice
    print("    Trying Edge TTS (SSML, fallback voice)...")
    if await _edge_tts(full_text, voice_alt, output_path, use_ssml=True, series=series):
        _normalize_audio(output_path)
        print("    [OK] Full narration via Edge SSML fallback")
        return output_path, char_weights

    # 5. Plain Edge TTS — last resort
    print("    Trying Edge TTS (plain)...")
    if await _edge_tts(full_text, voice_pri, output_path, use_ssml=False, series=series):
        _normalize_audio(output_path)
        print("    [OK] Full narration via Edge plain")
        return output_path, char_weights

    raise RuntimeError("All TTS providers failed for full narration")


# ── Public API ────────────────────────────────────────────────────────────────

async def generate_voiceover(scenes: list, language: str = "en") -> list:
    """
    Converts each scene's narration to a normalized MP3.
    Tries Gemini TTS first (cinematic quality), then SSML Edge TTS, then plain Edge TTS.
    Returns list of audio file paths.
    """
    voice    = _EDGE_VOICES.get(language, _EDGE_VOICES["en"])
    fallback = _EDGE_FALLBACK.get(language, _EDGE_FALLBACK["en"])

    os.makedirs(f"{_TEMP_ROOT}/audio", exist_ok=True)
    audio_files = []

    for i, scene in enumerate(scenes):
        output_path = f"{_TEMP_ROOT}/audio/scene_{i:02d}.mp3"
        text = _clean_narration(scene["narration"])
        success = False

        # 1 — Gemini TTS (best quality) — rate limit: 3 RPM, so pace at 21s apart
        if i > 0:
            await asyncio.sleep(21)
        print(f"    Trying Gemini TTS scene {i+1}...")
        success = await asyncio.to_thread(_gemini_tts, text, output_path)
        if success:
            print(f"    [OK] Audio scene {i+1}/{len(scenes)} — Gemini TTS")

        # 2 — Edge TTS with SSML narration style
        if not success:
            print(f"    Trying Edge TTS (SSML) scene {i+1}...")
            success = await _edge_tts(text, voice, output_path, use_ssml=True)
            if success:
                _normalize_audio(output_path)
                print(f"    [OK] Audio scene {i+1}/{len(scenes)} — Edge SSML")

        # 3 — Edge TTS fallback voice with SSML
        if not success:
            success = await _edge_tts(text, fallback, output_path, use_ssml=True)
            if success:
                _normalize_audio(output_path)
                print(f"    [OK] Audio scene {i+1}/{len(scenes)} — Edge SSML fallback")

        # 4 — Plain Edge TTS (last resort)
        if not success:
            success = await _edge_tts(text, voice, output_path, use_ssml=False)
            if success:
                _normalize_audio(output_path)
                print(f"    [OK] Audio scene {i+1}/{len(scenes)} — Edge plain")

        if success:
            audio_files.append(output_path)
        else:
            print(f"    [ERROR] All TTS methods failed for scene {i+1}")

    return audio_files
