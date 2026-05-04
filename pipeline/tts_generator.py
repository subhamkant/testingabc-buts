"""
TTS Generator
=============
Priority:
  1. Gemini TTS (gemini-2.5-flash-preview-tts) — state-of-the-art, cinematic Charon voice
  2. Edge TTS with SSML narration style — dramatic pacing, pauses, lower pitch
  3. Edge TTS plain fallback

Gemini TTS returns raw PCM (int16 LE, 24kHz mono) → wrapped in WAV → converted to MP3 by FFmpeg.
"""

import edge_tts
import asyncio
import os
import io
import wave
import subprocess


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
