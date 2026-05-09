"""
Mahabharata YouTube Bot — Main Pipeline
========================================
Usage:
    python main.py en           -> English video (full, 4-6 scenes)
    python main.py hi           -> Hindi video
    python main.py en --test    -> 1-scene test run, logs saved to logs/
    python main.py              -> defaults to English
"""

import asyncio
import os
import sys
import shutil
from datetime import datetime

# Force UTF-8 output so emojis print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from pipeline.script_generator import generate_script
from pipeline.tts_generator import generate_full_narration
from pipeline.image_generator import generate_images, generate_thumbnail, update_characters
from pipeline.video_assembler import (
    assemble_video_continuous_audio,
    assemble_from_video_clips_continuous_audio,
)
from pipeline.clip_generator import generate_video_clips
from pipeline.subtitle_generator import apply_subtitles
from pipeline.topic_manager import get_next_topic, log_video
from pipeline.youtube_uploader import upload_to_youtube


# ── Logging (tee to file + console) ──────────────────────────────────────────

class _Tee:
    """Writes to both the original stream and a log file simultaneously."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, text):
        self._stream.write(text)
        self._stream.flush()
        self._log.write(text)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def reconfigure(self, **kwargs):
        if hasattr(self._stream, "reconfigure"):
            self._stream.reconfigure(**kwargs)


def _setup_logging(language: str, test_mode: bool) -> object:
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_test" if test_mode else ""
    log_path = f"logs/pipeline_{language}_{timestamp}{suffix}.log"
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    sys.stdout = _Tee(sys.stdout, log_file)
    print(f"[log] Writing to {log_path}")
    return log_file


# ── Temp cleanup ──────────────────────────────────────────────────────────────

def cleanup_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    for d in ["temp/audio", "temp/images", "temp/clips"]:
        os.makedirs(d, exist_ok=True)
    os.makedirs("output", exist_ok=True)


def _video_output_path(language: str, series: str = "mahabharata") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    series_tag = "" if series == "mahabharata" else f"{series}_"
    return f"output/video_{series_tag}{language}_{timestamp}.mp4"


def _subscribe_outro(series: str, language: str) -> dict:
    """
    Returns a scene dict for the fixed subscribe-CTA outro. Series-aware so the
    Mahabharata and What If videos don't end with mismatched copy/imagery.
    """
    if series == "krishna":
        # First-person Krishna outro — preserves the divine-monologue immersion
        # instead of jarring into a third-person promo voice. Image is a
        # Krishna+Arjuna two-shot consistent with the rest of the speech.
        krishna_image = (
            "Cinematic two-shot of Krishna and Arjuna in a golden chariot at dusk, "
            "Krishna with peacock-feather crown and blue skin gesturing in mudra, "
            "Arjuna in armor listening intently, soft text 'Vyasa AI' subtly glowing "
            "in the lower-third like a divine seal, jewel-toned palette of crimson "
            "gold and lapis, illustrated mythology art, ornate background carvings"
        )
        krishna_video = "Slow push-in on Krishna and Arjuna two-shot, golden light rays, gentle dust motes"
        # Krishna voicing the CTA in first person — no tonal break.
        narration = (
            "ऐसी ही और बातें मैं तुमसे कहूँगा पार्थ। "
            "अगर मेरी वाणी तुम्हारे मन तक पहुँची है, तो Vyasa AI को Subscribe करो।"
        )
        return {
            "narration":    narration,
            "image_prompt": krishna_image,
            "video_prompt": krishna_video,
            "mood":         "divine and intimate",
        }

    if series == "whatif":
        whatif_image = (
            "Vyasa AI logo card — bold cinematic lettering 'Subscribe to Vyasa AI', "
            "starfield background with subtle nebula, deep blue and gold palette, "
            "9:16 portrait composition, modern science-curiosity aesthetic, "
            "clean readable text, no clutter"
        )
        whatif_video = "Slow zoom on Vyasa AI logo card with starfield parallax and golden glow"
        if language == "hi":
            narration = (
                "ऐसी रोचक कहानियों के लिए... Vyasa AI को Subscribe करें — "
                "समय और अंतरिक्ष के पार की कहानियाँ!"
            )
        else:
            narration = (
                "For more thought experiments... Subscribe to Vyasa AI — "
                "stories and curiosities from across time and space."
            )
        return {
            "narration":    narration,
            "image_prompt": whatif_image,
            "video_prompt": whatif_video,
            "mood":         "inviting and curious",
        }

    # Mahabharata (default)
    maha_image = (
        "Epic Mahabharata collage — Krishna, Arjuna, Karna, Draupadi in golden cinematic light, "
        "bold text 'Subscribe to Vyasa AI' glowing in center, lotus and Om symbol, "
        "jewel-toned palette, dramatic portrait composition"
    )
    maha_video = "Cinematic zoom out from Om symbol to full Mahabharata tableau, golden light rays"
    if language == "hi":
        narration = "ऐसी कहानियों के लिए... Vyasa AI को Subscribe करें। Bell Icon ज़रूर दबाएँ!"
    else:
        narration = "For more Mahabharata stories... Subscribe to Vyasa AI. Hit the bell!"
    return {
        "narration":    narration,
        "image_prompt": maha_image,
        "video_prompt": maha_video,
        "mood":         "inspiring and inviting",
    }


def _build_lang_script(dual_script: dict, language: str) -> dict:
    """
    Convert a dual-language WhatIf script into a single-language copy by
    selecting `narration_hi` or `narration` per scene and writing it to the
    common `narration` field that downstream pipeline stages expect.
    """
    out = dict(dual_script)
    out["scenes"] = []
    for scene in dual_script["scenes"]:
        narration = scene.get("narration_hi", "") if language == "hi" else scene.get("narration", "")
        # Build a clean per-scene copy keeping only the fields downstream needs
        out["scenes"].append({
            "narration":    narration,
            "image_prompt": scene.get("image_prompt", ""),
            "video_prompt": scene.get("video_prompt", ""),
            "mood":         scene.get("mood", ""),
        })
    out["language"] = language
    return out


# ── Pipeline ──────────────────────────────────────────────────────────────────

async def run_pipeline(language: str = "en", test_mode: bool = False, test_upload: bool = False):
    lang_name = "Hindi" if language == "hi" else "English"
    mode_tag = " [TEST-UPLOAD - 1 scene]" if test_upload else (" [TEST - 1 scene]" if test_mode else "")

    print(f"\n{'='*55}")
    print(f"  Mahabharata YouTube Bot  |  {lang_name}{mode_tag}")
    print(f"{'='*55}\n")

    cleanup_temp()

    try:
        # ── Step 1: Generate Script ───────────────────────────────
        print("Step 1 — Generating script with Gemini...")
        scheduled_topic = get_next_topic("mahabharata")
        script = generate_script(language, forced_topic=scheduled_topic, series="mahabharata")

        if test_mode or test_upload:
            script["scenes"] = script["scenes"][:1]
            print("    [test] Capped to 1 scene")

        print(f"    Topic       : {script['topic']}")
        print(f"    Title       : {script['title']}")
        print(f"    Content type: {script['content_type']}")

        # Append fixed subscribe outro
        script["scenes"].append(_subscribe_outro("mahabharata", language))
        print(f"    Scenes      : {len(script['scenes'])} (+ subscribe outro)")
        update_characters(script)

        # ── Step 2: Continuous voiceover (single-pass TTS) ────────
        # ONE audio file is generated from the concatenated narration so the
        # voice quality and pacing are identical throughout the video — no
        # seams between scenes, no half-Gemini / half-Edge mismatches.
        print("\nStep 2 — Generating continuous voiceover (single-pass TTS)...")
        audio_path, char_weights = await generate_full_narration(script["scenes"], language)

        if not audio_path or not os.path.exists(audio_path):
            print("No audio generated. Aborting.")
            return

        # ── Steps 3 & 4: AI video clips -> assemble, or fall back to images ──
        output_path = _video_output_path(language)
        video_path = None

        # AI video clips are tried if ANY provider is configured. clip_generator
        # internally cascades fal -> replicate -> HF, so even one configured
        # provider is enough to attempt the AI path.
        _ai_clips_available = any(
            os.environ.get(k, "").strip()
            for k in ("FAL_KEY", "REPLICATE_API_TOKEN", "HF_SPACE")
        )

        if _ai_clips_available:
            try:
                print("\nStep 3 — Generating AI video clips...")
                clip_files = await generate_video_clips(script["scenes"])
                print("\nStep 4 — Assembling video from AI clips with continuous audio...")
                video_path = assemble_from_video_clips_continuous_audio(
                    clip_files, audio_path, script,
                    char_weights=char_weights,
                    output_path=output_path,
                )
                if not os.path.exists(video_path):
                    raise RuntimeError("Assembly produced no output file")
            except Exception as clip_err:
                print(f"\n    AI clips failed: {clip_err}")
                print("    Falling back to static images...")
                video_path = None

        if video_path is None:
            print("\nStep 3 — Generating images via free cascade (HF -> Cloudflare -> Pollinations)...")
            image_files = generate_images(script["scenes"], series="mahabharata")
            if script.get("thumbnail_prompt"):
                generate_thumbnail(script["thumbnail_prompt"], series="mahabharata")
            print("\nStep 4 — Assembling video with continuous audio...")
            video_path = assemble_video_continuous_audio(
                image_files, audio_path, script,
                char_weights=char_weights,
                output_path=output_path,
            )

        # ── Step 4b: Burned-in word-level subtitles ───────────────
        # Run AFTER assembly + polish + music so cinematic effects don't
        # affect subtitle readability and camera shake doesn't jitter them.
        if video_path and os.path.exists(video_path):
            print("\nStep 4b — Burning word-level subtitles via Groq Whisper...")
            try:
                apply_subtitles(video_path, audio_path, language)
            except Exception as sub_err:
                print(f"    Subtitles failed (non-fatal): {sub_err}")

        # ── Step 5: Upload to YouTube ─────────────────────────────
        video_id = None
        if (not test_mode or test_upload) and os.path.exists("client_secrets.json"):
            try:
                print("\nStep 5 — Uploading to YouTube...")
                video_id = upload_to_youtube(
                    video_path, script, language,
                    thumbnail_path="output/thumbnail.jpg",
                    series="mahabharata",
                )
            except Exception as yt_err:
                print(f"    YouTube upload failed: {yt_err}")

        # ── Done ──────────────────────────────────────────────────
        log_video(video_path, script, language)
        print(f"\nPipeline complete!")
        print(f"    Video saved -> {video_path}")
        if video_id:
            print(f"    YouTube    -> https://youtube.com/watch?v={video_id}")
        if os.path.exists("output/thumbnail.jpg"):
            print(f"    Thumbnail  -> output/thumbnail.jpg")
        print()

    finally:
        cleanup_temp()


# ── Krishna direct-address pipeline ──────────────────────────────────────────

async def run_krishna_speech(test_mode: bool = False, test_upload: bool = False):
    """
    Krishna direct-address pipeline: 30-45 second Hindi-only Short where
    Krishna speaks in first person to a named listener (Arjuna ~60% / others
    rotate). Mirrors run_pipeline structure with series="krishna" threaded
    through script generation, TTS (per-series voice override), image
    generation (Krishna+listener two-shot prompts), and the upload step.
    """
    mode_tag = " [TEST-UPLOAD - 1 scene]" if test_upload else (" [TEST - 1 scene]" if test_mode else "")

    print(f"\n{'='*55}")
    print(f"  Krishna Direct Address  |  Hindi{mode_tag}")
    print(f"{'='*55}\n")

    cleanup_temp()

    try:
        # ── Step 1: Generate Krishna speech script ────────────────
        print("Step 1 — Generating Krishna direct-address script...")
        scheduled_topic = get_next_topic("krishna")
        script = generate_script(language="hi", forced_topic=scheduled_topic, series="krishna")

        if test_mode or test_upload:
            script["scenes"] = script["scenes"][:1]
            print("    [test] Capped to 1 scene")

        print(f"    Topic    : {script['topic']}")
        print(f"    Title    : {script['title']}")
        print(f"    Speaker  : {script.get('speaker', 'Krishna')}")
        print(f"    Listener : {script.get('listener', '?')}")

        # Krishna-voiced outro — preserves first-person immersion
        script["scenes"].append(_subscribe_outro("krishna", "hi"))
        print(f"    Scenes   : {len(script['scenes'])} (+ Krishna-voiced outro)")
        update_characters(script)

        # ── Step 2: Continuous voiceover (single-pass TTS) ────────
        # series="krishna" picks ELEVENLABS_VOICE_ID_KRISHNA if set, else
        # falls back to the default narrator voice.
        print("\nStep 2 — Generating continuous voiceover (series=krishna)...")
        audio_path, char_weights = await generate_full_narration(
            script["scenes"], language="hi", series="krishna"
        )

        if not audio_path or not os.path.exists(audio_path):
            print("No audio generated. Aborting.")
            return

        # ── Steps 3 & 4: AI clips → assembly, or static-image fallback ──
        output_path = _video_output_path("hi", series="krishna")
        video_path = None

        _ai_clips_available = any(
            os.environ.get(k, "").strip()
            for k in ("FAL_KEY", "REPLICATE_API_TOKEN", "HF_SPACE")
        )

        if _ai_clips_available:
            try:
                print("\nStep 3 — Generating AI video clips...")
                clip_files = await generate_video_clips(script["scenes"])
                print("\nStep 4 — Assembling video from AI clips with continuous audio...")
                video_path = assemble_from_video_clips_continuous_audio(
                    clip_files, audio_path, script,
                    char_weights=char_weights,
                    output_path=output_path,
                )
                if not os.path.exists(video_path):
                    raise RuntimeError("Assembly produced no output file")
            except Exception as clip_err:
                print(f"\n    AI clips failed: {clip_err}")
                print("    Falling back to static images...")
                video_path = None

        if video_path is None:
            print("\nStep 3 — Generating images (Krishna+listener two-shots)...")
            # series="krishna" reuses the Mahabharata illustrated style suffix
            # (no separate suffix needed) — _inject_characters auto-injects
            # Krishna and the listener's visuals when their names appear.
            image_files = generate_images(script["scenes"], series="krishna")
            if script.get("thumbnail_prompt"):
                generate_thumbnail(script["thumbnail_prompt"], series="krishna")
            print("\nStep 4 — Assembling video with continuous audio...")
            video_path = assemble_video_continuous_audio(
                image_files, audio_path, script,
                char_weights=char_weights,
                output_path=output_path,
            )

        # ── Step 4b: Burned-in word-level subtitles ───────────────
        if video_path and os.path.exists(video_path):
            print("\nStep 4b — Burning word-level subtitles via Groq Whisper...")
            try:
                apply_subtitles(video_path, audio_path, "hi")
            except Exception as sub_err:
                print(f"    Subtitles failed (non-fatal): {sub_err}")

        # ── Step 5: Upload to YouTube ─────────────────────────────
        video_id = None
        if (not test_mode or test_upload) and os.path.exists("client_secrets.json"):
            try:
                print("\nStep 5 — Uploading to YouTube...")
                video_id = upload_to_youtube(
                    video_path, script, "hi",
                    thumbnail_path="output/thumbnail.jpg",
                    series="krishna",
                )
            except Exception as yt_err:
                print(f"    YouTube upload failed: {yt_err}")

        # ── Done ──────────────────────────────────────────────────
        log_video(video_path, script, "hi")
        print(f"\nPipeline complete!")
        print(f"    Video saved -> {video_path}")
        if video_id:
            print(f"    YouTube    -> https://youtube.com/watch?v={video_id}")
        if os.path.exists("output/thumbnail.jpg"):
            print(f"    Thumbnail  -> output/thumbnail.jpg")
        print()

    finally:
        cleanup_temp()


# ── What If — dual-language orchestrator ─────────────────────────────────────

async def run_whatif_dual_language(test_mode: bool = False, test_upload: bool = False):
    """
    What If pipeline: generates ONE script with both English and Hindi
    narration per scene, generates visuals ONCE, then renders + uploads two
    videos (HI and EN) sharing identical visuals. Two YouTube uploads, one
    set of image/clip generations — ~50% cheaper than two separate runs.
    """
    mode_tag = " [TEST-UPLOAD - 1 scene]" if test_upload else (" [TEST - 1 scene]" if test_mode else "")
    print(f"\n{'='*55}")
    print(f"  Vyasa AI What If  |  Hindi + English{mode_tag}")
    print(f"{'='*55}\n")

    cleanup_temp()

    try:
        # ── Step 1: ONE dual-language script ──────────────────────
        print("Step 1 — Generating dual-language What If script...")
        scheduled_topic = get_next_topic("whatif")
        dual_script = generate_script(
            language="dual",
            forced_topic=scheduled_topic,
            series="whatif",
            dual_language=True,
        )

        if test_mode or test_upload:
            dual_script["scenes"] = dual_script["scenes"][:1]
            print("    [test] Capped to 1 scene")

        print(f"    Topic       : {dual_script['topic']}")
        print(f"    Title       : {dual_script['title']}")
        print(f"    Visual style: {dual_script.get('visual_style', 'photoreal-3d')}")

        # Append series-aware bilingual outro
        outro_en = _subscribe_outro("whatif", "en")
        outro_hi = _subscribe_outro("whatif", "hi")
        dual_script["scenes"].append({
            "narration":    outro_en["narration"],
            "narration_hi": outro_hi["narration"],
            "image_prompt": outro_en["image_prompt"],
            "video_prompt": outro_en["video_prompt"],
            "mood":         outro_en["mood"],
        })
        print(f"    Scenes      : {len(dual_script['scenes'])} (+ subscribe outro)")
        # NOTE: skipping update_characters — Mahabharata-specific.

        # ── Step 2: Generate visuals ONCE ─────────────────────────
        # Using the dual script's scenes (each has both narrations) is fine —
        # the visual generators only read image_prompt / video_prompt.
        visual_style = dual_script.get("visual_style", "photoreal-3d")
        clip_files = None
        image_files = None

        _ai_clips_available = any(
            os.environ.get(k, "").strip()
            for k in ("FAL_KEY", "REPLICATE_API_TOKEN", "HF_SPACE")
        )
        if _ai_clips_available:
            try:
                print("\nStep 2 — Generating shared AI video clips...")
                clip_files = await generate_video_clips(dual_script["scenes"])
            except Exception as clip_err:
                print(f"\n    AI clips failed: {clip_err}")
                print("    Falling back to static images...")
                clip_files = None

        if clip_files is None:
            print("\nStep 2 — Generating shared images via free cascade...")
            image_files = generate_images(
                dual_script["scenes"], series="whatif", visual_style=visual_style,
            )
            if dual_script.get("thumbnail_prompt"):
                generate_thumbnail(
                    dual_script["thumbnail_prompt"],
                    series="whatif", visual_style=visual_style,
                )

        # ── Step 3-5: per-language render + upload ────────────────
        video_ids = {}
        for lang in ("hi", "en"):
            lang_name = "Hindi" if lang == "hi" else "English"
            print(f"\n{'─'*55}\n  Rendering {lang_name} version\n{'─'*55}")

            lang_script = _build_lang_script(dual_script, lang)
            output_path = _video_output_path(lang, series="whatif")

            # TTS for this language
            print(f"\nStep 3a [{lang}] — TTS...")
            audio_path, char_weights = await generate_full_narration(
                lang_script["scenes"], lang
            )
            if not audio_path or not os.path.exists(audio_path):
                print(f"    [!] No audio generated for {lang}, skipping")
                continue

            # Assemble using SHARED visuals
            print(f"\nStep 3b [{lang}] — Assembling video...")
            try:
                if clip_files:
                    video_path = assemble_from_video_clips_continuous_audio(
                        clip_files, audio_path, lang_script,
                        char_weights=char_weights,
                        output_path=output_path,
                    )
                else:
                    video_path = assemble_video_continuous_audio(
                        image_files, audio_path, lang_script,
                        char_weights=char_weights,
                        output_path=output_path,
                    )
            except Exception as a_err:
                print(f"    [!] Assembly failed for {lang}: {a_err}")
                continue

            # Subtitles
            if video_path and os.path.exists(video_path):
                print(f"\nStep 3c [{lang}] — Subtitles...")
                try:
                    apply_subtitles(video_path, audio_path, lang)
                except Exception as sub_err:
                    print(f"    Subtitles failed (non-fatal): {sub_err}")

            # Upload
            video_id = None
            if (not test_mode or test_upload) and os.path.exists("client_secrets.json"):
                try:
                    print(f"\nStep 3d [{lang}] — Uploading to YouTube...")
                    video_id = upload_to_youtube(
                        video_path, lang_script, lang,
                        thumbnail_path="output/thumbnail.jpg",
                        series="whatif",
                    )
                    video_ids[lang] = video_id
                except Exception as yt_err:
                    print(f"    YouTube upload failed: {yt_err}")

            log_video(video_path, lang_script, lang)

        # ── Done ──────────────────────────────────────────────────
        print(f"\n{'='*55}\nWhat If Pipeline complete!\n{'='*55}")
        for lang, vid in video_ids.items():
            print(f"    {lang.upper()} -> https://youtube.com/watch?v={vid}")
        print()

    finally:
        cleanup_temp()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    test_mode   = "--test" in args
    test_upload = "--test-upload" in args
    args = [a for a in args if a not in ("--test", "--test-upload")]

    target = args[0] if args else "hi"

    if target == "whatif":
        log_file = _setup_logging("whatif", test_mode or test_upload)
        try:
            asyncio.run(run_whatif_dual_language(test_mode, test_upload))
        finally:
            sys.stdout = sys.stdout._stream
            log_file.close()
    elif target == "krishna":
        log_file = _setup_logging("krishna", test_mode or test_upload)
        try:
            asyncio.run(run_krishna_speech(test_mode, test_upload))
        finally:
            sys.stdout = sys.stdout._stream
            log_file.close()
    elif target in ("en", "hi"):
        log_file = _setup_logging(target, test_mode or test_upload)
        try:
            asyncio.run(run_pipeline(target, test_mode, test_upload))
        finally:
            sys.stdout = sys.stdout._stream
            log_file.close()
    else:
        print("Usage: python main.py [en|hi|whatif|krishna] [--test | --test-upload]")
        sys.exit(1)
