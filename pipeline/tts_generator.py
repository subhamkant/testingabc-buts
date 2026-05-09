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


# ── Voice config ──────────────────────────────────────────────────────────────

# Edge TTS fallback voices
_EDGE_VOICES = {
    "en": "en-US-ChristopherNeural",
    "hi": "hi-IN-MadhurNeural",
}
_EDGE_FALLBACK = {
    "en": "en-US-GuyNeural",
    "hi": "hi-IN-NeerjaNeural",
}

# Gemini voice — read from .env so it can be changed without code edits
# Options: Charon (deep/cinematic), Fenrir (strong), Orus (warm), Puck (energetic)
_GEMINI_VOICE = os.environ.get("NARRATOR_VOICE", "Charon")
_GEMINI_MODEL = "gemini-2.5-flash-preview-tts"

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


def _gemini_tts(text: str, output_mp3: str) -> bool:
    """
    Generates audio via Gemini TTS (Charon voice).
    Returns True on success.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return False

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=_GEMINI_VOICE
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

        return result.returncode == 0 and os.path.exists(output_mp3)

    except Exception as e:
        print(f"    [Gemini TTS] {e}")
        return False


# ── Edge TTS with SSML ────────────────────────────────────────────────────────

def _build_ssml(text: str, voice: str, language: str) -> str:
    """
    Wraps narration text in SSML with narration-relaxed style + prosody.
    Adds natural pauses after sentence endings for dramatic effect.
    """
    import re
    xml_lang = "hi-IN" if language == "hi" else "en-US"

    # Escape XML special characters
    text_esc = (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

    # Add dramatic pause after sentence endings
    text_esc = re.sub(r'([।!?])\s*', r'\1<break time="500ms"/>', text_esc)
    text_esc = re.sub(r'\.\s+', r'.<break time="400ms"/>', text_esc)
    text_esc = re.sub(r'\.\.\.',  r'...<break time="600ms"/>', text_esc)

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


async def _edge_tts(text: str, voice: str, output_path: str, use_ssml: bool = True) -> bool:
    """Generate audio via Edge TTS. Returns True on success."""
    try:
        if use_ssml:
            # Detect language from voice name
            lang = "hi" if "hi-IN" in voice else "en"
            ssml_text = _build_ssml(text, voice, lang)
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

def _clean_narration(text: str) -> str:
    """Clean narration text before sending to TTS."""
    import re

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove hashtags & mentions
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)

    # ❗ NEW: remove English abbreviations (MS, ML, etc.)
    text = re.sub(r'\b[A-Z]{2,}\b', '', text)

    # ❗ NEW: remove repeated weird tokens (common Gemini bug)
    text = re.sub(r'(एमएस|एमएल|MS|ML)+', '', text)

    # Normalize spaces
    text = re.sub(r'\s{2,}', ' ', text)

    return text.strip()


# ── ElevenLabs TTS ────────────────────────────────────────────────────────────

# Default voice IDs (overridable via ELEVENLABS_VOICE_ID env). These are
# stable across all videos so the same narrator voice is used run-to-run.
_DEFAULT_ELEVENLABS_VOICE = "pNInz6obpgDQGcFmaJgB"  # "Adam" — deep cinematic male


def _elevenlabs_tts(
    text: str,
    output_mp3: str,
    language: str = "en",
    api_key: str = "",
    key_label: str = "ElevenLabs",
) -> bool:
    """
    Synthesize via ElevenLabs Multilingual v2 (supports Hindi + English).
    Returns True on success. The MP3 is normalized in-place via FFmpeg
    after download so loudness matches the rest of the pipeline.

    `api_key` is passed explicitly so the caller can run a fallback chain
    (primary key → secondary key) when the primary hits its quota.
    `key_label` shows up in the log so you can tell which key handled the run.
    """
    if not api_key:
        return False

    voice_id = os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or _DEFAULT_ELEVENLABS_VOICE
    model_id = "eleven_multilingual_v2"

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
            "voice_settings": {
                "stability": 0.55,
                "similarity_boost": 0.80,
                "style": 0.30,
                "use_speaker_boost": True,
            },
        }
        resp = requests.post(url, headers=headers, json=body, timeout=180)
        if resp.status_code != 200:
            print(f"    [{key_label}] HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        with open(output_mp3, "wb") as f:
            f.write(resp.content)
        if os.path.exists(output_mp3) and os.path.getsize(output_mp3) > 1000:
            _normalize_audio(output_mp3)
            return True
        return False
    except Exception as e:
        print(f"    [{key_label}] {e}")
        return False


def _elevenlabs_keys() -> list:
    """
    Returns the ordered list of ElevenLabs keys to try, with a label per key.
    Order: primary, then fallback. Empty entries are filtered out so callers
    don't need to handle them.
    """
    candidates = [
        ("ElevenLabs (primary)",  os.environ.get("ELEVENLABS_API_KEY", "").strip()),
        ("ElevenLabs (fallback)", os.environ.get("ELEVENLABS_API_KEY_FALLBACK", "").strip()),
    ]
    return [(label, key) for label, key in candidates if key]


# ── Single-pass full narration ────────────────────────────────────────────────

async def generate_full_narration(scenes: list, language: str = "en") -> tuple:
    """
    Generate ONE continuous audio file from all scene narrations concatenated.

    Returns (audio_path, char_weights):
      audio_path     str            — path to the single MP3 file
      char_weights   list[int]      — per-scene narration char count for the
                                      video assembler to size scene clips
                                      proportionally so visuals stay roughly
                                      in sync with the spoken narrative.

    Provider cascade: ElevenLabs → Gemini Charon → Edge SSML → Edge plain.
    Once a provider succeeds, the entire video is consistent voice (no
    half-Gemini / half-Edge seams).
    """
    os.makedirs("temp/audio", exist_ok=True)

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

    output_path = "temp/audio/narration_full.mp3"

    # 1. ElevenLabs (best, requires ELEVENLABS_API_KEY[_FALLBACK])
    # Walk both keys in order — falls through on quota errors / 401 / network
    # failures, so a single dead key doesn't bump us all the way down to Edge.
    for key_label, key in _elevenlabs_keys():
        print(f"    Trying {key_label} (most realistic)...")
        ok = await asyncio.to_thread(
            _elevenlabs_tts, full_text, output_path, language, key, key_label
        )
        if ok:
            print(f"    [OK] Full narration via {key_label}")
            return output_path, char_weights

    # 2. Gemini Charon
    if os.environ.get("GEMINI_API_KEY", "").strip():
        print("    Trying Gemini TTS (Charon voice)...")
        if await asyncio.to_thread(_gemini_tts, full_text, output_path):
            print("    [OK] Full narration via Gemini Charon")
            return output_path, char_weights

    # 3. Edge TTS SSML
    voice_pri = _EDGE_VOICES.get(language, _EDGE_VOICES["en"])
    voice_alt = _EDGE_FALLBACK.get(language, _EDGE_FALLBACK["en"])

    print("    Trying Edge TTS (SSML, primary voice)...")
    if await _edge_tts(full_text, voice_pri, output_path, use_ssml=True):
        _normalize_audio(output_path)
        print("    [OK] Full narration via Edge SSML")
        return output_path, char_weights

    # 4. Edge TTS fallback voice
    print("    Trying Edge TTS (SSML, fallback voice)...")
    if await _edge_tts(full_text, voice_alt, output_path, use_ssml=True):
        _normalize_audio(output_path)
        print("    [OK] Full narration via Edge SSML fallback")
        return output_path, char_weights

    # 5. Plain Edge TTS — last resort
    print("    Trying Edge TTS (plain)...")
    if await _edge_tts(full_text, voice_pri, output_path, use_ssml=False):
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

    os.makedirs("temp/audio", exist_ok=True)
    audio_files = []

    for i, scene in enumerate(scenes):
        output_path = f"temp/audio/scene_{i:02d}.mp3"
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
