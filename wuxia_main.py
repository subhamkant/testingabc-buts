"""
Wuxia / Martial-Peak — Pipeline Driver (v1, FROM-JSON)
======================================================

Daily 15-min 16:9 donghua-style episodes. Fully isolated from Mahabharata.

Architecture:
  Cloudflare FLUX-schnell -> all stills @ 1920x1080.
  Motion-flagged shots -> Kaggle LTX kernel (skip FLUX, condition on the CF
  still) + in-kernel anime Real-ESRGAN -> 1920x1080 mp4.
  Landscape assembler splices motion clips + Ken-Burns stills + TTS + music.

Checkpoint-resumable (cache/<run_id>/) so the GHA retry chain can resume the
long Kaggle poll without re-pushing (protects the 30h/wk Kaggle quota).

Usage:
    python wuxia_main.py --from-json-dir pro_drafts/wuxia/<slug>/
    python wuxia_main.py --from-json pro_drafts/wuxia/<slug>/longform_en.json
    python wuxia_main.py --from-json-dir ... --test        # 3-scene smoke
    python wuxia_main.py --from-json-dir ... --no-motion    # stills-only (skip Kaggle)

v1 scope: no YouTube upload (finished mp4 is the deliverable, optionally to R2).
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

os.environ["PIPELINE_TEMP_ROOT"] = "temp/wx"
os.environ.setdefault("NARRATOR_VOICE_WUXIA", "Charon")
# Dark/epic bed for donghua action — the generic music pool otherwise defaults to
# a "happy Indian" trailer track, a genre mismatch for wuxia. Overridable via env.
os.environ.setdefault(
    "BACKGROUND_MUSIC_PATH",
    "assets/backgroundmusicforvideos-epic-epic-background-music-334868.mp3",
)

from pipeline.checkpoint import CheckpointStore, resolve_run_id
from pipeline.tts_generator import generate_full_narration
from pipeline.wuxia_images import generate_wuxia_stills
from pipeline.wuxia_motion import run_motion
from pipeline.wuxia_assembler import assemble_wuxia_video


def _setup_logging(test_mode: bool):
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/wuxia_{ts}{'_test' if test_mode else ''}.log"
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")

    class _Tee:
        def __init__(self, s, f):
            self._s, self._f = s, f
        def write(self, t):
            self._s.write(t); self._s.flush(); self._f.write(t); self._f.flush()
        def flush(self):
            self._s.flush(); self._f.flush()
        def reconfigure(self, **k):
            if hasattr(self._s, "reconfigure"):
                self._s.reconfigure(**k)

    sys.stdout = _Tee(sys.stdout, log_file)
    print(f"[log] writing to {log_path}")
    return log_file


def _resolve_from_json(from_json: str | None, from_json_dir: str | None) -> Path:
    if from_json and from_json_dir:
        sys.exit("--from-json and --from-json-dir are mutually exclusive")
    if from_json:
        return Path(from_json.strip())
    if not from_json_dir:
        sys.exit("provide --from-json or --from-json-dir")
    d = Path(from_json_dir)
    if not d.is_dir():
        sys.exit(f"--from-json-dir: not a directory: {d}")
    paths = sorted(d.glob("longform_*.json"))
    if not paths:
        sys.exit(f"--from-json-dir: no longform_*.json in {d}")
    return paths[0]


def _load_wuxia_script(path: Path) -> dict:
    """Load a Pro/Gemini-drafted episode JSON. Requires scenes/title/topic/language.
    Mirrors narration_<lang> into the `narration` key TTS/assembly read."""
    if not path.exists():
        sys.exit(f"--from-json: file does not exist: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"--from-json: {path} invalid JSON: {e}")
    lang = (raw.get("language") or "").strip().lower()
    if lang not in ("en", "hi"):
        sys.exit(f"--from-json: {path} missing 'language' ('en' or 'hi')")
    for req in ("scenes", "title", "topic"):
        if req not in raw:
            sys.exit(f"--from-json: {path} missing {req!r}")
    out = dict(raw)
    out["series"] = "wuxia"
    out["mode"] = "longform"
    nkey = "narration_hi" if lang == "hi" else "narration_en"
    for sc in out.get("scenes", []):
        if nkey in sc and "narration" not in sc:
            sc["narration"] = sc[nkey]
    print(f"[from-json] {lang.upper()} loaded: {path}  scenes={len(out['scenes'])}")
    return out


async def render(script: dict, lang: str, run_id: str, no_motion: bool) -> str:
    ck = CheckpointStore(run_id)
    scenes = script["scenes"]
    style_anchor = script.get("style_anchor")

    # ── Step 1: script (persist for resume) ─────────────────────────────
    if not ck.has("script.json"):
        ck.save_json("script.json", script)

    # ── Step 2: CF stills @ 1920x1080 (resumable per-shot) ──────────────
    print("[stills] Cloudflare FLUX-schnell 1920x1080...")
    still_groups = generate_wuxia_stills(scenes, ck, run_id, style_anchor=style_anchor)

    # ── Step 3: TTS (do before the long motion poll so it's cached) ─────
    if ck.has(f"audio_{lang}.mp3") and ck.has(f"char_weights_{lang}.json"):
        audio_path = ck.path(f"audio_{lang}.mp3")
        char_weights = ck.load_json(f"char_weights_{lang}.json")
        print(f"[tts:{lang}] resumed from checkpoint")
    else:
        print(f"[tts:{lang}] generating narration (ElevenLabs -> Gemini -> Edge)...")
        gen_audio, char_weights = await generate_full_narration(
            scenes, language=lang, series="wuxia"
        )
        ck.save_file(f"audio_{lang}.mp3", gen_audio)
        ck.save_json(f"char_weights_{lang}.json", char_weights)
        audio_path = ck.path(f"audio_{lang}.mp3")

    # ── Step 4: Kaggle LTX motion (resumable; never re-push a running kernel) ─
    if no_motion:
        print("[motion] --no-motion: skipping Kaggle, stills-only")
        final_groups = still_groups
    else:
        print("[motion] Kaggle LTX + anime ESRGAN...")
        final_groups = await run_motion(ck, run_id, scenes, still_groups)

    # ── Step 5: assemble 1920x1080 episode ──────────────────────────────
    out_key = f"episode_{lang}.mp4"
    if ck.has(out_key):
        final_mp4 = ck.path(out_key)
        print(f"[assemble] resumed from checkpoint: {final_mp4}")
    else:
        os.makedirs("output", exist_ok=True)
        output_path = f"output/wuxia_{run_id}_{lang}.mp4"
        assemble_wuxia_video(final_groups, audio_path, script, char_weights, output_path)
        ck.save_file(out_key, output_path)
        final_mp4 = output_path

    # ── Step 5b: burn Hindi subtitles (Groq Whisper word timings) ───────
    if not ck.has(f"subbed_{lang}.done"):
        try:
            from pipeline.longform_assembler import apply_longform_subtitles
            print(f"[subs:{lang}] burning subtitles (Groq Whisper)...")
            if apply_longform_subtitles(final_mp4, audio_path, language=lang):
                ck.mark_done(f"subbed_{lang}.done")
        except Exception as e:
            print(f"[subs] skipped (non-fatal): {e}")

    print(f"[done] episode -> {final_mp4}")

    # ── Step 6: R2 (gated on R2_* env; non-fatal) ───────────────────────
    if os.environ.get("R2_BUCKET") and not ck.has("r2_uploaded.json"):
        try:
            from pipeline.wuxia_r2 import upload_episode
            key = f"wuxia/{run_id}_{lang}.mp4"
            url = upload_episode(final_mp4, key)
            ck.save_json("r2_uploaded.json", {"bucket": os.environ["R2_BUCKET"], "key": key, "url": url})
            print(f"[r2] uploaded -> {url}")
        except Exception as e:
            print(f"[r2] skipped/failed (non-fatal): {e}")

    return final_mp4


def main():
    ap = argparse.ArgumentParser(description="Wuxia daily long-form pipeline (v1)")
    ap.add_argument("--from-json", default=None)
    ap.add_argument("--from-json-dir", default=None)
    ap.add_argument("--lang", default=None, help="override language (default: from JSON)")
    ap.add_argument("--test", action="store_true", help="3-scene smoke")
    ap.add_argument("--no-motion", action="store_true", help="stills-only (skip Kaggle)")
    args = ap.parse_args()

    _setup_logging(args.test)
    path = _resolve_from_json(args.from_json, args.from_json_dir)
    script = _load_wuxia_script(path)
    lang = (args.lang or script.get("language") or "en").strip().lower()

    if args.test:
        script = dict(script)
        script["scenes"] = script["scenes"][:3]
        print(f"[test] trimmed to {len(script['scenes'])} scenes")

    run_id = resolve_run_id("wuxia", lang)
    print(f"[run] run_id={run_id} lang={lang} scenes={len(script['scenes'])}")
    final = asyncio.run(render(script, lang, run_id, args.no_motion))
    print(f"[OK] {final}")


if __name__ == "__main__":
    main()
