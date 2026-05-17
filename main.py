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
from datetime import datetime, timezone

# Force UTF-8 output so emojis print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from pipeline.script_generator import generate_script
from pipeline.tts_generator import generate_full_narration
from pipeline.image_generator import (
    generate_images, generate_thumbnail, update_characters,
    reset_provider_tally, get_provider_tally,
)
from pipeline.video_assembler import (
    assemble_video_continuous_audio,
    assemble_from_video_clips_continuous_audio,
)
from pipeline.clip_generator import generate_video_clips
from pipeline.subtitle_generator import apply_subtitles
from pipeline.topic_manager import get_next_topic, log_video
from pipeline.youtube_uploader import upload_to_youtube
from pipeline.checkpoint import CheckpointStore, resolve_run_id


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


_KRISHNA_LISTENER_VOCATIVE = {
    # Map the script's `listener` field to the natural Hindi vocative Krishna
    # would actually use mid-conversation. Default to the bare name when no
    # special vocative exists. Arjuna's "पार्थ" is the canonical case.
    "Arjuna":       "पार्थ",
    "Yudhishthira": "युधिष्ठिर",
    "Karna":        "कर्ण",
    "Bhishma":      "पितामह",
    "Uddhava":      "उद्धव",
}


_OUTRO_ASSETS = {
    "mahabharata": "assets/outro/mahabharata.jpg",
    "krishna":     "assets/outro/krishna.jpg",
    "whatif":      "assets/outro/whatif.jpg",
}


def _subscribe_outro(series: str, language: str, listener: str = "") -> dict:
    """
    Returns a scene dict for the fixed subscribe-CTA outro. Series-aware so the
    Mahabharata and What If videos don't end with mismatched copy/imagery.

    `listener` is used by the Krishna outro to address the right person (e.g.
    "उद्धव" when the speech was to Uddhava, not the hardcoded "पार्थ").

    NEW 2026-05-14: outro scenes now carry a static `image_path` pointing to
    a hand-picked asset in assets/outro/. The image pipeline detects this and
    skips FLUX generation entirely — guarantees the channel name overlay,
    composition, and quality are exactly what was approved (no per-video
    FLUX gambling). `image_prompt` is kept as a fallback in case the asset
    file ever goes missing.
    """
    if series == "krishna":
        # First-person Krishna outro — preserves the divine-monologue immersion
        # instead of jarring into a third-person promo voice. Image is a
        # Krishna + listener two-shot consistent with the rest of the speech.
        listener_name = (listener or "Arjuna").strip()
        vocative = _KRISHNA_LISTENER_VOCATIVE.get(listener_name, listener_name)

        krishna_image = (
            # NO embedded text — FLUX-schnell cannot spell channel/brand names
            # (2026-05-14 production check: rendered "Vyasa AI" as VyssA /
            # Virtasy / Vilysaria / Viysas — all garbled). Channel CTA is
            # delivered via the spoken narration below instead.
            f"Cinematic two-shot of Krishna and {listener_name} at golden hour, "
            "Krishna with peacock-feather crown and blue skin gesturing in mudra, "
            f"{listener_name} listening intently, soft golden glow in the lower-third, "
            "jewel-toned palette of crimson gold and lapis, illustrated mythology art, "
            "ornate background carvings"
        )
        krishna_video = (
            f"Slow push-in on Krishna and {listener_name} two-shot, "
            "golden light rays, gentle dust motes"
        )
        # Krishna voicing the CTA in first person, addressing the actual
        # listener of THIS video — no tonal break, no wrong-name jolt.
        narration = (
            f"ऐसी ही और बातें मैं तुमसे कहूँगा {vocative}। "
            "अगर मेरी वाणी तुम्हारे मन तक पहुँची है, तो Vyasa AI को Subscribe करो।"
        )
        return {
            "narration":    narration,
            "image_path":   _OUTRO_ASSETS["krishna"],
            "image_prompt": krishna_image,
            "video_prompt": krishna_video,
            "mood":         "divine and intimate",
        }

    if series == "whatif":
        whatif_image = (
            # NO embedded text — FLUX-schnell cannot spell channel/brand names.
            # NO people — "portrait composition" pulls FLUX to face renders
            # even without character names (2026-05-14 check shipped 3/3
            # WhatIf candidates as face close-ups). This prompt explicitly
            # bans humans/faces and describes only the cosmic scene.
            "Empty deep-space cosmic scene, swirling nebula clouds with star "
            "clusters, glowing distant planets and ringed gas giant, spiral "
            "galaxy in the distance, atomic orbital patterns and DNA helix "
            "motifs as floating ethereal light, deep blue and gold palette, "
            "modern minimal sci-fi aesthetic, abstract astronomy artwork, "
            "9:16 vertical canvas, no people, no faces, no humans, "
            "no text, no lettering, no logos"
        )
        whatif_video = "Slow zoom into cosmic scene with starfield parallax and golden glow"
        if language == "hi":
            narration = (
                "ऐसी मज़ेदार कहानियों के लिए... Vyasa AI को Subscribe करें — "
                "टाइम और स्पेस के पार की कहानियाँ!"
            )
        else:
            narration = (
                "For more thought experiments... Subscribe to Vyasa AI — "
                "stories and curiosities from across time and space."
            )
        return {
            "narration":    narration,
            "image_path":   _OUTRO_ASSETS["whatif"],
            "image_prompt": whatif_image,
            "video_prompt": whatif_video,
            "mood":         "inviting and curious",
        }

    # Mahabharata (default)
    # NO embedded text — FLUX-schnell cannot spell "Vyasa AI" (production
    # check 2026-05-14 shipped frames with VyssA / Virtasy / Vilysaria /
    # Viysas — all garbled). Channel CTA is delivered via narration; image
    # is a clean visual tableau without typography.
    maha_image = (
        "Epic Mahabharata tableau — Krishna with peacock crown, Arjuna with "
        "Gandiva bow, Karna with golden armor, Draupadi in crimson sari, "
        "arrayed in golden cinematic light around a central glowing Om symbol "
        "above a lotus, jewel-toned palette, dramatic portrait composition, "
        "no text, no lettering, no logos"
    )
    maha_video = "Cinematic zoom out from Om symbol to full Mahabharata tableau, golden light rays"
    if language == "hi":
        narration = "ऐसी कहानियों के लिए... Vyasa AI को Subscribe करें। Bell Icon ज़रूर दबाएँ!"
    else:
        narration = "For more Mahabharata stories... Subscribe to Vyasa AI. Hit the bell!"
    return {
        "narration":    narration,
        "image_path":   _OUTRO_ASSETS["mahabharata"],
        "image_prompt": maha_image,
        "video_prompt": maha_video,
        "mood":         "inspiring and inviting",
    }


def _print_image_summary() -> None:
    """
    Emit the per-run [image-summary] line — visible accounting of which
    provider rendered each scene. Added 2026-05-17 after the #4 Bhishma
    Kurukshetra cron silently fell to Pollinations for Scene 6 (CF quota
    exhausted mid-pipeline) without any error log.
    """
    t = get_provider_tally()
    cf = t.get("cloudflare-flux-schnell", 0)
    hf = t.get("hf-flux-schnell", 0)
    poll = t.get("pollinations-flux-realism", 0)
    total = cf + hf + poll
    if total == 0:
        return
    flag = "  ⚠️ HIGH FALLBACK" if poll > 2 else ""
    print(f"    [image-summary] CF: {cf}/{total}  HF: {hf}/{total}  Pollinations: {poll}/{total}{flag}")


def _build_lang_script(dual_script: dict, language: str) -> dict:
    """
    Convert a dual-language WhatIf script into a single-language copy by
    selecting `narration_hi` or `narration` per scene and writing it to the
    common `narration` field that downstream pipeline stages expect.

    Title handling: the LLM produces a bilingual title "English | Hindi".
    The Hindi upload keeps the bilingual title (Hindi viewers also read the
    English half on the thumbnail). The English upload drops the Hindi half
    so the title reads cleanly for English-only viewers.
    """
    out = dict(dual_script)
    if language == "en":
        title = (out.get("title") or "").strip()
        if "|" in title:
            out["title"] = title.split("|", 1)[0].strip().rstrip("-—,:; ").strip()
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

    # Per-run checkpoint cache. PIPELINE_RUN_ID env var lets the GHA workflow
    # share the same cache between the original run and an auto-retry job
    # (so the retry resumes from the last completed step instead of starting
    # over from scratch).
    ck = CheckpointStore(resolve_run_id("mahabharata", language))
    cached = ck.list_entries()
    if cached:
        print(f"  [resume] Found {len(cached)} cached step(s) in {ck.dir} — completed work will be skipped:")
        for c in cached:
            print(f"    - {c}")
    else:
        print(f"  Checkpoint cache: {ck.dir}")
    print()

    cleanup_temp()

    try:
        # ── Step 1: Generate Script ───────────────────────────────
        if ck.has("script.json"):
            print("Step 1 — [resume] script loaded from checkpoint")
            script = ck.load_json("script.json")
        else:
            print("Step 1 — Generating script with Gemini...")
            scheduled_topic = get_next_topic("mahabharata")
            script = generate_script(language, forced_topic=scheduled_topic, series="mahabharata")

            if test_mode or test_upload:
                script["scenes"] = script["scenes"][:1]
                print("    [test] Capped to 1 scene")

            # Append fixed subscribe outro before caching so the resumed run
            # doesn't re-append (which would produce two outros)
            script["scenes"].append(_subscribe_outro("mahabharata", language))
            update_characters(script)
            ck.save_json("script.json", script)

        print(f"    Topic       : {script.get('topic', 'N/A')}")
        print(f"    Title       : {script.get('title', 'N/A')}")
        print(f"    Content type: {script.get('content_type', 'N/A')}")
        print(f"    Scenes      : {len(script['scenes'])} (incl. subscribe outro)")

        # ── Step 2: Continuous voiceover (single-pass TTS) ────────
        if ck.has("audio.mp3") and ck.has("char_weights.json"):
            print("\nStep 2 — [resume] audio loaded from checkpoint")
            audio_path = ck.path("audio.mp3")
            char_weights = ck.load_json("char_weights.json")
        else:
            print("\nStep 2 — Generating continuous voiceover (single-pass TTS)...")
            audio_path_temp, char_weights = await generate_full_narration(script["scenes"], language)

            if not audio_path_temp or not os.path.exists(audio_path_temp):
                print("No audio generated. Aborting.")
                return

            ck.save_file("audio.mp3", audio_path_temp)
            ck.save_json("char_weights.json", char_weights)
            audio_path = ck.path("audio.mp3")

        # ── Steps 3 & 4: Visuals + Assembly (cached as video_pre_subs.mp4) ──
        output_path = _video_output_path(language)
        video_path = None

        if ck.has("video_pre_subs.mp4"):
            print("\nSteps 3+4 — [resume] assembled video loaded from checkpoint")
            shutil.copy2(ck.path("video_pre_subs.mp4"), output_path)
            video_path = output_path
        else:
            # Try AI clips first, fall back to images
            _ai_clips_available = any(
                os.environ.get(k, "").strip()
                for k in ("FAL_KEY", "REPLICATE_API_TOKEN", "HF_SPACE")
            )

            if _ai_clips_available:
                try:
                    print("\nStep 3 — Generating AI video clips...")
                    clip_files, ref_images = await generate_video_clips(script["scenes"])
                    n_ai = sum(1 for c in clip_files if c)
                    if n_ai == 0:
                        print("    All AI clip providers failed — using full static-image pipeline")
                    else:
                        label = (
                            f"{n_ai} AI clip" if n_ai == len(clip_files)
                            else f"{n_ai} AI / {len(clip_files) - n_ai} Ken Burns"
                        )
                        print(f"\nStep 4 — Assembling video ({label}) with continuous audio...")
                        video_path = assemble_from_video_clips_continuous_audio(
                            clip_files, audio_path, script,
                            char_weights=char_weights,
                            output_path=output_path,
                            fallback_images=ref_images,
                        )
                        if not os.path.exists(video_path):
                            raise RuntimeError("Assembly produced no output file")
                except Exception as clip_err:
                    print(f"\n    AI clips path failed: {clip_err}")
                    print("    Falling back to static images...")
                    video_path = None

            if video_path is None:
                print("\nStep 3 — Generating images via free cascade (HF -> Cloudflare -> Pollinations)...")
                # Pass ck so generate_images can checkpoint after EACH scene
                # completes — survives 29-min cap mid-batch and resumes there.
                reset_provider_tally()
                image_files = generate_images(script["scenes"], series="mahabharata", ck=ck)
                if script.get("thumbnail_prompt"):
                    # Extract Hindi shock-phrase from title for thumbnail overlay.
                    # Title format from prompt: "महाभारत #N: <Hindi half> | <English half>"
                    _ov_text = ""
                    _title = script.get("title", "")
                    if "|" in _title:
                        _hindi_half = _title.split("|")[0].strip()
                        if ":" in _hindi_half:
                            _ov_text = _hindi_half.split(":", 1)[1].strip()
                        else:
                            _ov_text = _hindi_half
                    generate_thumbnail(
                        script["thumbnail_prompt"], series="mahabharata",
                        overlay_text=_ov_text[:25],  # 25-char cap for thumbnail legibility
                    )
                print("\nStep 4 — Assembling video with continuous audio...")
                video_path = assemble_video_continuous_audio(
                    image_files, audio_path, script,
                    char_weights=char_weights,
                    output_path=output_path,
                )

            # Cache the pre-subtitle assembled video so a retry doesn't
            # redo image-gen + Ken Burns + music mix.
            if video_path and os.path.exists(video_path):
                ck.save_file("video_pre_subs.mp4", video_path)

        # ── Step 4b: Burned-in word-level subtitles ───────────────
        # Gated behind BURN_SUBTITLES env var (default OFF per the
        # 2026-05-14 decision — subtitles weren't appearing on every shot
        # consistently AND the FFmpeg overlay pass costs ~4-5 min per
        # video. YouTube's auto-captions cover accessibility while we
        # investigate the timing issue. Re-enable by setting
        # BURN_SUBTITLES=true in GHA secrets or .env.)
        _burn_subs = os.environ.get("BURN_SUBTITLES", "false").lower() in ("1", "true", "yes")
        if ck.has("video.mp4"):
            print("\nStep 4b — [resume] subtitled video loaded from checkpoint")
            shutil.copy2(ck.path("video.mp4"), output_path)
            video_path = output_path
        elif _burn_subs and video_path and os.path.exists(video_path):
            print("\nStep 4b — Burning word-level subtitles via Groq Whisper...")
            try:
                apply_subtitles(video_path, audio_path, language)
            except Exception as sub_err:
                print(f"    Subtitles failed (non-fatal): {sub_err}")
            # Cache the final video regardless of subtitle success/failure
            # so a retry doesn't redo this expensive step.
            if os.path.exists(video_path):
                ck.save_file("video.mp4", video_path)

        # ── Step 5: Upload to YouTube ─────────────────────────────
        # Idempotency: if uploaded.json exists, the upload already finished
        # (possibly in a previous attempt). Skip and reuse the video_id —
        # never double-upload to YouTube.
        video_id = None
        if ck.has("uploaded.json"):
            uploaded = ck.load_json("uploaded.json")
            video_id = uploaded.get("video_id")
            print(f"\nStep 5 — [resume] already uploaded as https://youtube.com/watch?v={video_id}")
        elif (not test_mode or test_upload) and os.path.exists("client_secrets.json"):
            try:
                print("\nStep 5 — Uploading to YouTube...")
                # Checkpoint immediately when YouTube returns a video_id —
                # protects against duplicate re-upload if a downstream step
                # (thumbnail/playlist/comment) fails after the insert.
                def _checkpoint_upload(vid: str):
                    ck.save_json("uploaded.json", {
                        "video_id": vid,
                        "ts":       datetime.now(timezone.utc).isoformat(),
                        "language": language,
                        "series":   "mahabharata",
                    })

                video_id = upload_to_youtube(
                    video_path, script, language,
                    thumbnail_path="output/thumbnail.jpg",
                    series="mahabharata",
                    on_video_id=_checkpoint_upload,
                )
            except Exception as yt_err:
                print(f"    YouTube upload failed: {yt_err}")

        # ── Done ──────────────────────────────────────────────────
        # Fix 2.8 (2026-05-16): only mark topic as "used" when upload
        # actually returned a video_id. Without this guard, a YouTube
        # upload failure (OAuth revoked, quota, etc.) still committed
        # the topic to recent_topics.json — creating "ghost" entries
        # that block the topic from being retried on the next cron.
        # (The 2026-05-16 16:19 UTC cron's invalid_grant failure
        #  created exactly such a ghost for "Bhishma in Kurukshetra war".)
        if video_id and not ck.has("logged.done"):
            log_video(video_path, script, language)
            ck.mark_done("logged.done")
        elif not video_id:
            print("    [skip-log] upload failed/skipped — NOT recording topic "
                  "as used (prevents ghost entries in recent_topics.json)")

        # ── Step 6: Cross-post to Instagram Reels ─────────────────
        # Same mp4, same caption (derived from script_data['description']).
        # NON-FATAL — IG failures never break the YT path or the Fix 2.8
        # state-commit guard above. video_id (YT) stays the source of
        # truth for "topic used".
        ig_media_id = None
        skip_ig = os.environ.get("SKIP_INSTAGRAM", "false").strip().lower() == "true"
        if video_id and skip_ig:
            print(f"    [ig] SKIPPED via SKIP_INSTAGRAM=true (YouTube-only run)")
        elif video_id:
            try:
                from pipeline.instagram_uploader import upload_to_instagram
                ig_media_id = upload_to_instagram(
                    video_path=video_path,
                    script_data=script,
                    youtube_url=f"https://youtube.com/watch?v={video_id}",
                )
            except Exception as ig_err:
                print(f"    Instagram upload failed (non-fatal): {ig_err}")

        _print_image_summary()
        print(f"\nPipeline complete!")
        print(f"    Video saved -> {video_path}")
        if video_id:
            print(f"    YouTube    -> https://youtube.com/watch?v={video_id}")
        if ig_media_id:
            print(f"    Instagram  -> media_id={ig_media_id}")
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

    Resumable: every step is checkpointed under cache/<run_id>/. The
    auto-retry-once GHA job downloads the checkpoint and reinvokes this
    function — completed steps are loaded from cache and skipped.
    """
    mode_tag = " [TEST-UPLOAD - 1 scene]" if test_upload else (" [TEST - 1 scene]" if test_mode else "")

    print(f"\n{'='*55}")
    print(f"  Krishna Direct Address  |  Hindi{mode_tag}")
    print(f"{'='*55}\n")

    ck = CheckpointStore(resolve_run_id("krishna", "hi"))
    cached = ck.list_entries()
    if cached:
        print(f"  [resume] Found {len(cached)} cached step(s) in {ck.dir} — completed work will be skipped:")
        for c in cached:
            print(f"    - {c}")
    else:
        print(f"  Checkpoint cache: {ck.dir}")
    print()

    cleanup_temp()

    try:
        # ── Step 1: Generate Krishna speech script ────────────────
        if ck.has("script.json"):
            print("Step 1 — [resume] script loaded from checkpoint")
            script = ck.load_json("script.json")
        else:
            print("Step 1 — Generating Krishna direct-address script...")
            scheduled_topic = get_next_topic("krishna")
            script = generate_script(language="hi", forced_topic=scheduled_topic, series="krishna")

            if test_mode or test_upload:
                script["scenes"] = script["scenes"][:1]
                print("    [test] Capped to 1 scene")

            # Krishna-voiced outro before caching so a resumed run doesn't
            # re-append (which would produce two outros)
            script["scenes"].append(_subscribe_outro("krishna", "hi", script.get("listener", "")))
            update_characters(script)
            ck.save_json("script.json", script)

        print(f"    Topic    : {script.get('topic', 'N/A')}")
        print(f"    Title    : {script.get('title', 'N/A')}")
        print(f"    Speaker  : {script.get('speaker', 'Krishna')}")
        print(f"    Listener : {script.get('listener', '?')}")
        print(f"    Scenes   : {len(script['scenes'])} (incl. Krishna-voiced outro)")

        # ── Step 2: Continuous voiceover (single-pass TTS) ────────
        if ck.has("audio.mp3") and ck.has("char_weights.json"):
            print("\nStep 2 — [resume] audio loaded from checkpoint")
            audio_path = ck.path("audio.mp3")
            char_weights = ck.load_json("char_weights.json")
        else:
            print("\nStep 2 — Generating continuous voiceover (series=krishna)...")
            audio_path_temp, char_weights = await generate_full_narration(
                script["scenes"], language="hi", series="krishna"
            )

            if not audio_path_temp or not os.path.exists(audio_path_temp):
                print("No audio generated. Aborting.")
                return

            ck.save_file("audio.mp3", audio_path_temp)
            ck.save_json("char_weights.json", char_weights)
            audio_path = ck.path("audio.mp3")

        # ── Steps 3 & 4: Visuals + Assembly (cached as video_pre_subs.mp4) ──
        output_path = _video_output_path("hi", series="krishna")
        video_path = None

        if ck.has("video_pre_subs.mp4"):
            print("\nSteps 3+4 — [resume] assembled video loaded from checkpoint")
            shutil.copy2(ck.path("video_pre_subs.mp4"), output_path)
            video_path = output_path
        else:
            _ai_clips_available = any(
                os.environ.get(k, "").strip()
                for k in ("FAL_KEY", "REPLICATE_API_TOKEN", "HF_SPACE")
            )

            if _ai_clips_available:
                try:
                    print("\nStep 3 — Generating AI video clips...")
                    clip_files, ref_images = await generate_video_clips(script["scenes"])
                    n_ai = sum(1 for c in clip_files if c)
                    if n_ai == 0:
                        print("    All AI clip providers failed — using full static-image pipeline")
                    else:
                        label = (
                            f"{n_ai} AI clip" if n_ai == len(clip_files)
                            else f"{n_ai} AI / {len(clip_files) - n_ai} Ken Burns"
                        )
                        print(f"\nStep 4 — Assembling video ({label}) with continuous audio...")
                        video_path = assemble_from_video_clips_continuous_audio(
                            clip_files, audio_path, script,
                            char_weights=char_weights,
                            output_path=output_path,
                            series="krishna",
                            fallback_images=ref_images,
                        )
                        if not os.path.exists(video_path):
                            raise RuntimeError("Assembly produced no output file")
                except Exception as clip_err:
                    print(f"\n    AI clips path failed: {clip_err}")
                    print("    Falling back to static images...")
                    video_path = None

            if video_path is None:
                print("\nStep 3 — Generating images (Krishna+listener two-shots)...")
                # Per-scene checkpoint via ck — see Mahabharata flow comment above.
                reset_provider_tally()
                image_files = generate_images(script["scenes"], series="krishna", ck=ck)
                if script.get("thumbnail_prompt"):
                    generate_thumbnail(script["thumbnail_prompt"], series="krishna")
                print("\nStep 4 — Assembling video with continuous audio...")
                video_path = assemble_video_continuous_audio(
                    image_files, audio_path, script,
                    char_weights=char_weights,
                    output_path=output_path,
                    series="krishna",
                )

            # Cache the pre-subtitle assembled video
            if video_path and os.path.exists(video_path):
                ck.save_file("video_pre_subs.mp4", video_path)

        # ── Step 4b: Burned-in word-level subtitles ───────────────
        _burn_subs = os.environ.get("BURN_SUBTITLES", "false").lower() in ("1", "true", "yes")
        if ck.has("video.mp4"):
            print("\nStep 4b — [resume] subtitled video loaded from checkpoint")
            shutil.copy2(ck.path("video.mp4"), output_path)
            video_path = output_path
        elif _burn_subs and video_path and os.path.exists(video_path):
            print("\nStep 4b — Burning word-level subtitles via Groq Whisper...")
            try:
                apply_subtitles(video_path, audio_path, "hi")
            except Exception as sub_err:
                print(f"    Subtitles failed (non-fatal): {sub_err}")
            if os.path.exists(video_path):
                ck.save_file("video.mp4", video_path)

        # ── Step 5: Upload to YouTube ─────────────────────────────
        video_id = None
        if ck.has("uploaded.json"):
            uploaded = ck.load_json("uploaded.json")
            video_id = uploaded.get("video_id")
            print(f"\nStep 5 — [resume] already uploaded as https://youtube.com/watch?v={video_id}")
        elif (not test_mode or test_upload) and os.path.exists("client_secrets.json"):
            try:
                print("\nStep 5 — Uploading to YouTube...")
                def _checkpoint_upload(vid: str):
                    ck.save_json("uploaded.json", {
                        "video_id": vid,
                        "ts":       datetime.now(timezone.utc).isoformat(),
                        "language": "hi",
                        "series":   "krishna",
                    })

                video_id = upload_to_youtube(
                    video_path, script, "hi",
                    thumbnail_path="output/thumbnail.jpg",
                    series="krishna",
                    on_video_id=_checkpoint_upload,
                )
            except Exception as yt_err:
                print(f"    YouTube upload failed: {yt_err}")

        # ── Done ──────────────────────────────────────────────────
        # Fix 2.8 (2026-05-16): same upload-success gate as Mahabharata
        # path — see comment at run_pipeline().
        if video_id and not ck.has("logged.done"):
            log_video(video_path, script, "hi")
            ck.mark_done("logged.done")
        elif not video_id:
            print("    [skip-log] upload failed/skipped — NOT recording topic "
                  "as used (prevents ghost entries in recent_topics.json)")

        # ── Step 6: Cross-post to Instagram Reels (non-fatal) ─────
        ig_media_id = None
        skip_ig = os.environ.get("SKIP_INSTAGRAM", "false").strip().lower() == "true"
        if video_id and skip_ig:
            print(f"    [ig] SKIPPED via SKIP_INSTAGRAM=true (YouTube-only run)")
        elif video_id:
            try:
                from pipeline.instagram_uploader import upload_to_instagram
                ig_media_id = upload_to_instagram(
                    video_path=video_path,
                    script_data=script,
                    youtube_url=f"https://youtube.com/watch?v={video_id}",
                )
            except Exception as ig_err:
                print(f"    Instagram upload failed (non-fatal): {ig_err}")

        _print_image_summary()
        print(f"\nPipeline complete!")
        print(f"    Video saved -> {video_path}")
        if video_id:
            print(f"    YouTube    -> https://youtube.com/watch?v={video_id}")
        if ig_media_id:
            print(f"    Instagram  -> media_id={ig_media_id}")
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

    # Backward-compat: this kicks off both phases sequentially in one process.
    # GHA splits into two jobs (whatif-english + whatif-hindi) so each gets
    # its own 90-min budget. Locally, this single-process flow is fine.
    await run_whatif_phase("en", test_mode, test_upload)
    await run_whatif_phase("hi", test_mode, test_upload)


async def run_whatif_phase(language: str, test_mode: bool = False, test_upload: bool = False):
    """
    Run ONE language phase (en or hi) of the WhatIf dual-language pipeline.

    Both phases share the same run_id (whatif_<github.run_id>) so the EN
    phase's cached script + visuals are picked up unchanged by the HI phase
    via the GHA artifact. The EN phase is the "primary" — it generates the
    dual-language script and the shared visuals. The HI phase just consumes
    those + does HI-specific TTS / assembly / subtitles / upload.

    Resume points (per the standard checkpoint pattern):
      - script.json (shared)
      - visuals_manifest.json + visuals/ (shared)
      - thumbnail.jpg (shared)
      - audio_<lang>.mp3 + char_weights_<lang>.json
      - video_pre_subs_<lang>.mp4
      - video_<lang>.mp4
      - uploaded_<lang>.json (video_id captured immediately after insert)
    """
    if language not in ("en", "hi"):
        raise ValueError(f"WhatIf phase language must be 'en' or 'hi', got {language!r}")

    lang_name = "English" if language == "en" else "Hindi"
    mode_tag = " [TEST-UPLOAD - 1 scene]" if test_upload else (" [TEST - 1 scene]" if test_mode else "")

    print(f"\n{'='*55}")
    print(f"  Vyasa AI What If  |  {lang_name} phase{mode_tag}")
    print(f"{'='*55}\n")

    # Run id is language-AGNOSTIC for whatif so EN and HI share one cache dir.
    ck = CheckpointStore(resolve_run_id("whatif"))
    cached = ck.list_entries()
    if cached:
        print(f"  [resume] Found {len(cached)} cached entry(s) in {ck.dir}:")
        for c in cached:
            print(f"    - {c}")
    else:
        print(f"  Checkpoint cache: {ck.dir}")
    print()

    cleanup_temp()

    try:
        # ── Step 1: ONE dual-language script (SHARED across en + hi) ──
        if ck.has("script.json"):
            print("Step 1 — [resume/shared] script loaded from checkpoint")
            dual_script = ck.load_json("script.json")
        elif language == "hi":
            print("[!] HI phase started without a cached script. The EN phase must run first")
            print("    (it produces the dual-language script + visuals that HI consumes).")
            return
        else:
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

            # Append bilingual outro before caching so a resumed run doesn't
            # re-append (which would produce two outros)
            outro_en = _subscribe_outro("whatif", "en")
            outro_hi = _subscribe_outro("whatif", "hi")
            dual_script["scenes"].append({
                "narration":    outro_en["narration"],
                "narration_hi": outro_hi["narration"],
                "image_prompt": outro_en["image_prompt"],
                "video_prompt": outro_en["video_prompt"],
                "mood":         outro_en["mood"],
            })
            ck.save_json("script.json", dual_script)

        print(f"    Topic       : {dual_script.get('topic', 'N/A')}")
        print(f"    Title       : {dual_script.get('title', 'N/A')}")
        print(f"    Visual style: {dual_script.get('visual_style', 'photoreal-3d')}")
        print(f"    Scenes      : {len(dual_script['scenes'])} (incl. subscribe outro)")

        # ── Step 2: Visuals (SHARED across en + hi) ───────────────
        # Manifest records which path was taken (clips vs images) and the
        # paths into the cache. HI phase loads this without re-generating.
        visual_style = dual_script.get("visual_style", "photoreal-3d")
        clip_files = None
        ref_images = None
        image_files = None

        if ck.has("visuals_manifest.json"):
            print("\nStep 2 — [resume/shared] visuals loaded from checkpoint")
            manifest = ck.load_json("visuals_manifest.json")
            kind = manifest.get("kind")
            if kind == "clips":
                clip_files  = manifest.get("clip_files")
                ref_images  = manifest.get("ref_images")
            else:
                image_files = manifest.get("image_files")
        elif language == "hi":
            print("[!] HI phase started without cached visuals. The EN phase must run first.")
            return
        else:
            _ai_clips_available = any(
                os.environ.get(k, "").strip()
                for k in ("FAL_KEY", "REPLICATE_API_TOKEN", "HF_SPACE")
            )
            if _ai_clips_available:
                try:
                    print("\nStep 2 — Generating shared AI video clips...")
                    clips_raw, ref_images_raw = await generate_video_clips(dual_script["scenes"])
                    if any(clips_raw):
                        clip_files = clips_raw
                        ref_images = ref_images_raw
                        n_ai = sum(1 for c in clip_files if c)
                        if n_ai < len(clip_files):
                            print(f"    {n_ai}/{len(clip_files)} AI clips ok — failed scenes "
                                  f"will render as Ken Burns from the I2V reference images")
                    else:
                        print("    All AI clip providers failed — falling back to static images")
                        clip_files = None
                except Exception as clip_err:
                    print(f"\n    AI clips failed: {clip_err}")
                    print("    Falling back to static images...")
                    clip_files = None

            if clip_files is None:
                print("\nStep 2 — Generating shared images via free cascade...")
                # Per-scene checkpoint via ck so a 29-min-cap cancellation
                # mid-batch preserves completed scenes for the retry job.
                # The bulk visuals_manifest.json save below remains as the
                # "all done" marker; partial-resume uses visuals_partial.json.
                reset_provider_tally()
                image_files = generate_images(
                    dual_script["scenes"], series="whatif", visual_style=visual_style,
                    ck=ck,
                )
                if dual_script.get("thumbnail_prompt"):
                    generate_thumbnail(
                        dual_script["thumbnail_prompt"],
                        series="whatif", visual_style=visual_style,
                    )
                    if os.path.exists("output/thumbnail.jpg"):
                        ck.save_file("thumbnail.jpg", "output/thumbnail.jpg")

            # Cache visuals so the HI phase can reuse them
            if clip_files:
                cached_clips = []
                for i, p in enumerate(clip_files):
                    if p and os.path.exists(p):
                        cached_clips.append(ck.save_file(f"visuals/clip_{i:02d}.mp4", p))
                    else:
                        cached_clips.append(None)
                cached_refs = []
                if ref_images:
                    for i, p in enumerate(ref_images):
                        if p and os.path.exists(p):
                            cached_refs.append(ck.save_file(f"visuals/ref_{i:02d}.jpg", p))
                        else:
                            cached_refs.append(None)
                ck.save_json("visuals_manifest.json", {
                    "kind":        "clips",
                    "clip_files":  cached_clips,
                    "ref_images":  cached_refs or None,
                })
                clip_files = cached_clips
                ref_images = cached_refs or None
            elif image_files:
                # generate_images returns list[list[str]] — outer index = scene,
                # inner index = shot (3 shots per scene by default for the rich
                # Ken Burns path). Preserve that nesting through the cache so
                # the assembler still receives the right shape on resume.
                cached_imgs = []
                for scene_idx, scene_shots in enumerate(image_files):
                    if isinstance(scene_shots, str):
                        scene_shots = [scene_shots]
                    scene_cached = []
                    for shot_idx, shot_path in enumerate(scene_shots):
                        if not shot_path or not os.path.exists(shot_path):
                            scene_cached.append(shot_path)
                            continue
                        ext = os.path.splitext(shot_path)[1] or ".jpg"
                        cached_path = ck.save_file(
                            f"visuals/scene_{scene_idx:02d}_shot_{shot_idx:02d}{ext}",
                            shot_path,
                        )
                        scene_cached.append(cached_path)
                    cached_imgs.append(scene_cached)
                ck.save_json("visuals_manifest.json", {
                    "kind":        "images",
                    "image_files": cached_imgs,
                })
                image_files = cached_imgs

        # Restore the thumbnail to its expected location for the upload step
        if ck.has("thumbnail.jpg") and not os.path.exists("output/thumbnail.jpg"):
            os.makedirs("output", exist_ok=True)
            shutil.copy2(ck.path("thumbnail.jpg"), "output/thumbnail.jpg")

        # ── Optional EN-phase early-exit ──────────────────────────
        # When WHATIF_SKIP_EN_UPLOAD is set, the EN phase stops here — script
        # + visuals are already cached (Steps 1-2), which is everything the HI
        # phase needs. We skip EN's TTS / assembly / subtitles / upload entirely,
        # saving ~10-15 min per workflow run while still producing the HI video.
        # Toggle off later by removing the env var from GHA secrets (or setting
        # WHATIF_SKIP_EN_UPLOAD=false).
        if language == "en" and os.environ.get("WHATIF_SKIP_EN_UPLOAD", "").lower() in ("1", "true", "yes"):
            print(f"\n[skip] WHATIF_SKIP_EN_UPLOAD set — EN phase ends after shared visuals.")
            print(f"       HI phase will load script.json + visuals_manifest.json from this cache.")
            print(f"\nEnglish phase complete (shared assets only — no EN upload)!\n")
            return

        # ── Step 3a: Per-language TTS ─────────────────────────────
        lang_script = _build_lang_script(dual_script, language)
        output_path = _video_output_path(language, series="whatif")

        if ck.has(f"audio_{language}.mp3") and ck.has(f"char_weights_{language}.json"):
            print(f"\nStep 3a [{language}] — [resume] audio loaded from checkpoint")
            audio_path = ck.path(f"audio_{language}.mp3")
            char_weights = ck.load_json(f"char_weights_{language}.json")
        else:
            print(f"\nStep 3a [{language}] — TTS...")
            audio_path_temp, char_weights = await generate_full_narration(
                lang_script["scenes"], language
            )
            if not audio_path_temp or not os.path.exists(audio_path_temp):
                print(f"    [!] No audio generated for {language}, aborting phase")
                return
            ck.save_file(f"audio_{language}.mp3", audio_path_temp)
            ck.save_json(f"char_weights_{language}.json", char_weights)
            audio_path = ck.path(f"audio_{language}.mp3")

        # ── Step 3b: Assembly ─────────────────────────────────────
        video_path = None
        if ck.has(f"video_pre_subs_{language}.mp4"):
            print(f"\nStep 3b [{language}] — [resume] assembled video loaded from checkpoint")
            shutil.copy2(ck.path(f"video_pre_subs_{language}.mp4"), output_path)
            video_path = output_path
        else:
            print(f"\nStep 3b [{language}] — Assembling video...")
            try:
                if clip_files:
                    video_path = assemble_from_video_clips_continuous_audio(
                        clip_files, audio_path, lang_script,
                        char_weights=char_weights,
                        output_path=output_path,
                        fallback_images=ref_images,
                    )
                else:
                    video_path = assemble_video_continuous_audio(
                        image_files, audio_path, lang_script,
                        char_weights=char_weights,
                        output_path=output_path,
                    )
            except Exception as a_err:
                print(f"    [!] Assembly failed for {language}: {a_err}")
                return
            if video_path and os.path.exists(video_path):
                ck.save_file(f"video_pre_subs_{language}.mp4", video_path)

        # ── Step 3c: Subtitles ────────────────────────────────────
        _burn_subs = os.environ.get("BURN_SUBTITLES", "false").lower() in ("1", "true", "yes")
        if ck.has(f"video_{language}.mp4"):
            print(f"\nStep 3c [{language}] — [resume] subtitled video loaded from checkpoint")
            shutil.copy2(ck.path(f"video_{language}.mp4"), output_path)
            video_path = output_path
        elif _burn_subs and video_path and os.path.exists(video_path):
            print(f"\nStep 3c [{language}] — Subtitles...")
            try:
                apply_subtitles(video_path, audio_path, language)
            except Exception as sub_err:
                print(f"    Subtitles failed (non-fatal): {sub_err}")
            if os.path.exists(video_path):
                ck.save_file(f"video_{language}.mp4", video_path)

        # ── Step 3d: Upload ───────────────────────────────────────
        video_id = None
        if ck.has(f"uploaded_{language}.json"):
            uploaded = ck.load_json(f"uploaded_{language}.json")
            video_id = uploaded.get("video_id")
            print(f"\nStep 3d [{language}] — [resume] already uploaded as https://youtube.com/watch?v={video_id}")
        elif (not test_mode or test_upload) and os.path.exists("client_secrets.json"):
            try:
                print(f"\nStep 3d [{language}] — Uploading to YouTube...")
                def _checkpoint_upload(vid: str, _lang=language):
                    ck.save_json(f"uploaded_{_lang}.json", {
                        "video_id": vid,
                        "ts":       datetime.now(timezone.utc).isoformat(),
                        "language": _lang,
                        "series":   "whatif",
                    })

                video_id = upload_to_youtube(
                    video_path, lang_script, language,
                    thumbnail_path="output/thumbnail.jpg",
                    series="whatif",
                    on_video_id=_checkpoint_upload,
                )
            except Exception as yt_err:
                print(f"    YouTube upload failed: {yt_err}")

        # ── Done ──────────────────────────────────────────────────
        # Fix 2.8 (2026-05-16): same upload-success gate as Mahabharata
        # path — see comment at run_pipeline().
        if video_id and not ck.has(f"logged_{language}.done"):
            log_video(video_path, lang_script, language)
            ck.mark_done(f"logged_{language}.done")
        elif not video_id:
            print(f"    [skip-log] {language} upload failed/skipped — NOT recording "
                  f"topic as used (prevents ghost entries in recent_topics.json)")

        # ── Cross-post to Instagram Reels (non-fatal) ─────────────
        ig_media_id = None
        skip_ig = os.environ.get("SKIP_INSTAGRAM", "false").strip().lower() == "true"
        if video_id and skip_ig:
            print(f"    [ig] SKIPPED via SKIP_INSTAGRAM=true (YouTube-only run)")
        elif video_id:
            try:
                from pipeline.instagram_uploader import upload_to_instagram
                ig_media_id = upload_to_instagram(
                    video_path=video_path,
                    script_data=lang_script,
                    youtube_url=f"https://youtube.com/watch?v={video_id}",
                )
            except Exception as ig_err:
                print(f"    Instagram upload failed (non-fatal): {ig_err}")

        _print_image_summary()
        print(f"\n{lang_name} phase complete!")
        if video_id:
            print(f"    YouTube   -> https://youtube.com/watch?v={video_id}")
        if ig_media_id:
            print(f"    Instagram -> media_id={ig_media_id}")
        print()

    finally:
        cleanup_temp()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    test_mode   = "--test" in args
    test_upload = "--test-upload" in args
    args = [a for a in args if a not in ("--test", "--test-upload")]

    # Optional --lang=en|hi flag (used by the WhatIf split GHA jobs to run
    # one language at a time). Without --lang, `whatif` runs both phases
    # sequentially in one process for backward-compat / local dev.
    whatif_lang = None
    for a in list(args):
        if a.startswith("--lang="):
            whatif_lang = a.split("=", 1)[1].strip()
            args.remove(a)

    target = args[0] if args else "hi"

    if target == "whatif":
        log_label = f"whatif_{whatif_lang}" if whatif_lang else "whatif"
        log_file = _setup_logging(log_label, test_mode or test_upload)
        try:
            if whatif_lang in ("en", "hi"):
                asyncio.run(run_whatif_phase(whatif_lang, test_mode, test_upload))
            else:
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
        print(
            "Usage: python main.py [en|hi|whatif|krishna] [--lang=en|hi] [--test | --test-upload]\n"
            "  --lang=en | --lang=hi  (whatif only) — runs only the EN or HI phase.\n"
            "                            Without --lang, whatif runs both phases serially."
        )
        sys.exit(1)
