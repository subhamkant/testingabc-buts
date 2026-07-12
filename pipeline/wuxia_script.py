"""
Wuxia script generator — pipeline/wuxia_script.py
=================================================

Turns RAW Martial Peak chapter text into a ~15-min Hindi donghua recap script
(the longform_hi.json schema wuxia_main.py renders). Reading the actual source
text (not the model's memory) keeps cultivation realms, character names, and
technique descriptions 100% faithful — no hallucinated lore for fans to roast.

Design (per user):
  source_chapters/chapter_001.txt, chapter_002.txt, ...   (raw web-novel text)
  assets/wuxia_chapter_progress.json  -> {"next_chapter": N, "episode": M}
  Each run pulls the next WUXIA_CHAPTERS_PER_EP chapters, feeds them to Gemini
  in segments (to fit output budgets), and writes
  pro_drafts/wuxia/<slug>/longform_hi.json, then advances the progress pointer.

Reuses pipeline.script_generator._call_llm (Groq->Gemini-Pro->Flash cascade,
6-key rotation). Hindi register = conversational + code-switched (keep English
terms: cultivation, sect, realm, spirit energy, disciple) — NOT Sanskritic.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

from pipeline.script_generator import _call_llm

_ROOT = Path(__file__).resolve().parent.parent
CHAPTERS_DIR = _ROOT / "source_chapters"
PROGRESS_FILE = _ROOT / "assets" / "wuxia_chapter_progress.json"
DRAFTS_DIR = _ROOT / "pro_drafts" / "wuxia"

_STYLE_ANCHOR = (
    "cinematic donghua, Chinese wuxia animation style, dramatic volumetric "
    "lighting, intricate detail, sharp focus, epic 16:9 landscape composition, "
    "moody atmospheric"
)

_TARGET_SCENES = int(os.environ.get("WUXIA_TARGET_SCENES", "110"))   # ~15 min at ~8s/scene
_SCENES_PER_CALL = int(os.environ.get("WUXIA_SCENES_PER_CALL", "24"))  # fits _call_llm budget
_MAX_MOTION = int(os.environ.get("WUXIA_MAX_MOTION", "20"))
_CHAPTERS_PER_EP = int(os.environ.get("WUXIA_CHAPTERS_PER_EP", "3"))


# ── progress + chapter I/O ────────────────────────────────────────────────────

def _load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"next_chapter": 1, "episode": 1}


def _save_progress(p: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(p, indent=2), encoding="utf-8")


def _chapter_path(n: int) -> Path:
    return CHAPTERS_DIR / f"chapter_{n:03d}.txt"


def _next_chapters(start: int, count: int) -> tuple[list[str], list[int]]:
    texts, nums = [], []
    n = start
    while len(texts) < count:
        p = _chapter_path(n)
        if not p.exists():
            break
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
        nums.append(n)
        n += 1
    return texts, nums


# ── segmenting + JSON extraction ──────────────────────────────────────────────

def _split_into_segments(text: str, n: int) -> list[str]:
    """Split by paragraphs into n roughly-equal char-count segments."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return [text]
    total = sum(len(p) for p in paras)
    target = total / n
    segments, cur, cur_len = [], [], 0
    for p in paras:
        cur.append(p)
        cur_len += len(p)
        if cur_len >= target and len(segments) < n - 1:
            segments.append("\n\n".join(cur))
            cur, cur_len = [], 0
    if cur:
        segments.append("\n\n".join(cur))
    return segments


def _extract_json_array(text: str):
    """Tolerantly pull a JSON array out of an LLM response (strips ```json fences,
    finds the outermost [ ... ])."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.MULTILINE).strip()
    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON array found in LLM response")
    return json.loads(t[start:end + 1])


# ── generation ────────────────────────────────────────────────────────────────

def _segment_prompt(segment: str, seg_idx: int, n_segments: int,
                    want: int, prev_last: str | None) -> str:
    cont = ""
    if prev_last:
        cont = (f'\nThe previous part ended with this narration: "{prev_last}". '
                f"Continue the story smoothly from there — do not repeat it.\n")
    return f"""You are the scriptwriter for a Hindi YouTube channel that makes cinematic donghua-style recap videos of the Chinese cultivation web novel "Martial Peak".

Write a faithful, gripping scene-by-scene recap of the SOURCE TEXT below. This is part {seg_idx + 1} of {n_segments} of one episode.

OUTPUT: ONLY a JSON array. Each element is an object with exactly these keys:
  "narration_hi": 1-2 short, punchy sentences in CONVERSATIONAL HINDI written in Devanagari, code-switched with natural English words (cultivation, sect, realm, spirit energy, disciple, elder, technique/skill names, Yuan Qi, etc.). Sound like an excited narrator telling a friend the story — NOT literary or Sanskritic Hindi.
  "prompt": a vivid ENGLISH cinematic visual description of that exact moment for an AI image generator. 16:9 landscape, donghua/wuxia animation style. Describe the character(s), setting, action, and mood exactly as the text describes (techniques, locations, appearances).
  "requires_motion": true ONLY for big action / combat / power-surge / technique-unleashing beats (about 1 in 6 scenes). Dialogue, scenery, and reaction beats are false.

STRICT RULES:
- Stay 100% FAITHFUL to the source text. Do NOT invent events, cultivation realms, character names, or techniques. Use names EXACTLY as written in the text.
- Generate about {want} scenes covering this part, in story order.
- Keep each narration tight (this is a fast-paced recap).
{cont}
SOURCE TEXT (part {seg_idx + 1}/{n_segments}):
\"\"\"
{segment}
\"\"\"

Return ONLY the JSON array. No markdown fences, no commentary."""


def _generate_segment_scenes(segment: str, seg_idx: int, n_segments: int,
                             want: int, prev_last: str | None) -> list[dict]:
    prompt = _segment_prompt(segment, seg_idx, n_segments, want, prev_last)
    resp = _call_llm(prompt, quality="best")
    arr = _extract_json_array(resp)
    scenes = []
    for e in arr:
        nar = (e.get("narration_hi") or "").strip()
        pr = (e.get("prompt") or "").strip()
        if not nar or not pr:
            continue
        scenes.append({
            "narration_hi": nar,
            "prompt": pr,
            "requires_motion": bool(e.get("requires_motion")),
        })
    return scenes


def _finalize_scenes(raw: list[dict], max_motion: int) -> list[dict]:
    """Cap motion beats to ~max_motion (spread evenly) and wrap into the
    visual_track schema with 1-indexed scene_id."""
    motion_idx = [i for i, s in enumerate(raw) if s["requires_motion"]]
    if len(motion_idx) > max_motion:
        step = len(motion_idx) / max_motion
        keep = {motion_idx[int(k * step)] for k in range(max_motion)}
        for i in motion_idx:
            if i not in keep:
                raw[i]["requires_motion"] = False
    scenes = []
    for i, s in enumerate(raw):
        rm = bool(s["requires_motion"])
        scenes.append({
            "scene_id": i + 1,
            "narration_hi": s["narration_hi"],
            "visual_track": [{
                "category": "motion" if rm else "hero",
                "requires_motion": rm,
                "reaction_frame": False,
                "prompt": s["prompt"],
            }],
        })
    return scenes


def generate_next_episode(chapters_per_ep: int = _CHAPTERS_PER_EP,
                          target_scenes: int = _TARGET_SCENES,
                          advance: bool = True) -> tuple[str, dict]:
    """Generate the next episode's script from source_chapters. Returns
    (episode_slug, script_dict) and writes pro_drafts/wuxia/<slug>/longform_hi.json.
    Advances the chapter-progress pointer unless advance=False."""
    prog = _load_progress()
    start = int(prog.get("next_chapter", 1))
    ep_num = int(prog.get("episode", 1))

    texts, nums = _next_chapters(start, chapters_per_ep)
    if not texts:
        raise RuntimeError(
            f"No chapter files found at {CHAPTERS_DIR}/chapter_{start:03d}.txt+ . "
            f"Drop raw web-novel text files there (chapter_001.txt, ...)."
        )

    combined = "\n\n".join(texts)
    n_segments = max(1, math.ceil(target_scenes / _SCENES_PER_CALL))
    segments = _split_into_segments(combined, n_segments)
    per = max(6, target_scenes // len(segments))

    print(f"[wuxia-script] episode {ep_num}: chapters {nums[0]}-{nums[-1]} "
          f"({len(combined)} chars) -> {len(segments)} segment(s), ~{per} scenes each",
          flush=True)

    raw: list[dict] = []
    for i, seg in enumerate(segments):
        prev_last = raw[-1]["narration_hi"] if raw else None
        scenes = _generate_segment_scenes(seg, i, len(segments), per, prev_last)
        print(f"[wuxia-script]   segment {i+1}/{len(segments)}: {len(scenes)} scenes", flush=True)
        raw.extend(scenes)

    if not raw:
        raise RuntimeError("script generation produced no scenes")

    scenes = _finalize_scenes(raw, max_motion=_MAX_MOTION)
    motion_ct = sum(1 for s in scenes if s["visual_track"][0]["requires_motion"])
    print(f"[wuxia-script] episode {ep_num}: {len(scenes)} scenes, {motion_ct} motion", flush=True)

    slug = f"ep{ep_num:02d}_ch{nums[0]:03d}-{nums[-1]:03d}"
    script = {
        "title": f"Martial Peak Episode {ep_num} (Hindi)",
        "topic": f"martial peak episode {ep_num} chapters {nums[0]}-{nums[-1]}",
        "language": "hi",
        "style_anchor": _STYLE_ANCHOR,
        "scenes": scenes,
    }

    out_dir = DRAFTS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "longform_hi.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[wuxia-script] wrote {out_dir / 'longform_hi.json'}", flush=True)

    if advance:
        _save_progress({"next_chapter": nums[-1] + 1, "episode": ep_num + 1})

    return slug, script


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate the next Wuxia episode script from source_chapters/")
    ap.add_argument("--chapters", type=int, default=_CHAPTERS_PER_EP)
    ap.add_argument("--scenes", type=int, default=_TARGET_SCENES)
    ap.add_argument("--no-advance", action="store_true", help="don't advance the chapter pointer")
    args = ap.parse_args()
    slug, _ = generate_next_episode(args.chapters, args.scenes, advance=not args.no_advance)
    print(f"[OK] {slug}")
