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
from pipeline.tts_generator import generate_voiceover
from pipeline.image_generator import generate_images, generate_thumbnail, update_characters
from pipeline.video_assembler import assemble_video, assemble_from_video_clips
from pipeline.clip_generator import generate_video_clips
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


def _video_output_path(language: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"output/video_{language}_{timestamp}.mp4"


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
        scheduled_topic = get_next_topic()
        script = generate_script(language, forced_topic=scheduled_topic)

        if test_mode or test_upload:
            script["scenes"] = script["scenes"][:1]
            print("    [test] Capped to 1 scene")

        print(f"    Topic       : {script['topic']}")
        print(f"    Title       : {script['title']}")
        print(f"    Content type: {script['content_type']}")

        # Append fixed subscribe outro
        outro = {
            "narration": (
                "ऐसी ही महाभारत की रोमांचक कहानियाँ सुनने के लिए... "
                "अभी Subscribe करें Vyasa AI को। "
                "और Bell Icon ज़रूर दबाएँ... ताकि कोई कहानी छूटे नहीं!"
            ) if language == "hi" else (
                "For more epic tales from the Mahabharata... "
                "Subscribe to Vyasa AI right now. "
                "Hit the bell icon... so you never miss a story!"
            ),
            "image_prompt": (
                "Epic Mahabharata collage — Krishna, Arjuna, Karna, Draupadi in golden cinematic light, "
                "bold text 'Subscribe to Vyasa AI' glowing in center, lotus and Om symbol, "
                "jewel-toned palette, dramatic portrait composition"
            ),
            "video_prompt": "Cinematic zoom out from Om symbol to full Mahabharata tableau, golden light rays",
            "mood": "inspiring and inviting",
        }
        script["scenes"].append(outro)
        print(f"    Scenes      : {len(script['scenes'])} (+ subscribe outro)")
        update_characters(script)

        # ── Step 2: Voiceover ─────────────────────────────────────
        print("\nStep 2 — Generating voiceover with Edge TTS...")
        audio_files = await generate_voiceover(script["scenes"], language)

        if not audio_files:
            print("No audio generated. Aborting.")
            return

        # ── Steps 3 & 4: AI video clips -> assemble, or fall back to images ──
        output_path = _video_output_path(language)
        video_path = None

        _hf_space = os.environ.get("HF_SPACE", "").strip()
        _ai_clips_available = bool(_hf_space)

        if _ai_clips_available:
            try:
                import gradio_client  # noqa: F401
            except ImportError:
                _ai_clips_available = False

        if _ai_clips_available:
            try:
                print("\nStep 3 — Generating AI video clips via HuggingFace Spaces...")
                clip_files = await generate_video_clips(script["scenes"])
                print("\nStep 4 — Assembling video from AI clips...")
                video_path = assemble_from_video_clips(clip_files, audio_files, script,
                                                       output_path=output_path)
                if not os.path.exists(video_path):
                    raise RuntimeError("Assembly produced no output file")
            except Exception as clip_err:
                print(f"\n    AI clips failed: {clip_err}")
                print("    Falling back to static images...")
                video_path = None

        if video_path is None:
            print("\nStep 3 — Generating images with Pollinations.ai...")
            image_files = generate_images(script["scenes"])
            if script.get("thumbnail_prompt"):
                generate_thumbnail(script["thumbnail_prompt"])
            print("\nStep 4 — Assembling video with FFmpeg...")
            video_path = assemble_video(image_files, audio_files, script,
                                        output_path=output_path)

        # ── Step 5: Upload to YouTube ─────────────────────────────
        video_id = None
        if (not test_mode or test_upload) and os.path.exists("client_secrets.json"):
            try:
                print("\nStep 5 — Uploading to YouTube...")
                video_id = upload_to_youtube(
                    video_path, script, language,
                    thumbnail_path="output/thumbnail.jpg",
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


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    test_mode   = "--test" in args
    test_upload = "--test-upload" in args
    args = [a for a in args if a not in ("--test", "--test-upload")]

    lang = args[0] if args else "hi"
    if lang not in ("en", "hi"):
        print("Usage: python main.py [en|hi] [--test | --test-upload]")
        sys.exit(1)

    log_file = _setup_logging(lang, test_mode or test_upload)
    try:
        asyncio.run(run_pipeline(lang, test_mode, test_upload))
    finally:
        sys.stdout = sys.stdout._stream
        log_file.close()
