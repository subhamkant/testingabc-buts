"""
Checkpoint Store — resumable pipeline state for retry-on-failure.

Each pipeline run is keyed by a `run_id` (set via the PIPELINE_RUN_ID env var
by the workflow, or generated from the current UTC for local invocations).
Every step in main.py writes its output to `cache/<run_id>/<step>.<ext>` via
this module. The next attempt — whether GHA's auto-retry-on-failure or a
manual re-trigger — opens the same cache directory, sees the completed
step's checkpoint, loads it, and skips the work.

Atomicity: every save uses `.tmp` + `os.replace`, which is atomic on the
Linux GHA runner (POSIX rename semantics) and on Windows for single-volume
moves. A process killed mid-write cannot leave a partial checkpoint that
the next attempt would mistake for valid state.

Why this matters:
- Skip ~$0.001 LLM calls + 5-30s on script regeneration
- Skip 10-15 min + provider quota on visuals
- Skip ElevenLabs character quota on TTS
- Skip 25 min FFmpeg work on subtitle overlay
- CRUCIAL: skip duplicate YouTube uploads after a partial-success upload
  (write the video_id checkpoint immediately after videos.insert returns
  so playlist-add or pinned-comment failures never cause a re-upload)

The cache directory itself is gitignored (regenerable transient state). The
GHA workflow uploads the entire `cache/<run_id>/` tree as a workflow artifact
so the auto-retry job can download it before re-invoking main.py.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone


_CACHE_ROOT = "cache"


def default_run_id(series: str, language: str = "") -> str:
    """
    Generate a sensible default run_id when PIPELINE_RUN_ID isn't set
    (mostly: local dev invocations). For scheduled GHA runs, the workflow
    sets PIPELINE_RUN_ID explicitly using the cron timestamp, so this is
    only a fallback.

    WhatIf intentionally omits language so EN and HI sub-runs share one
    cache directory (the HI job picks up EN's script + visuals).
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H")
    if series == "whatif":
        return f"whatif_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    if language:
        return f"{series}_{language}_{stamp}"
    return f"{series}_{stamp}"


def resolve_run_id(series: str, language: str = "") -> str:
    """
    Read PIPELINE_RUN_ID from the env (set by the GHA workflow), or fall
    back to the auto-generated default. Empty/whitespace env values are
    treated as unset.
    """
    env_id = os.environ.get("PIPELINE_RUN_ID", "").strip()
    return env_id or default_run_id(series, language)


class CheckpointStore:
    """
    Per-run cache directory abstraction. Designed for the simple
    "write-then-rename" atomic-save pattern. Files live under:

        cache/<run_id>/
          script.json
          visuals_manifest.json
          visuals/scene_NN_shot_NN.jpg
          thumbnail.jpg
          audio_<lang>.mp3
          char_weights_<lang>.json
          video_pre_subs_<lang>.mp4
          video_<lang>.mp4
          uploaded_<lang>.json

    Methods are deliberately thin — callers use them as primitives in
    "if has X: load X else: do work + save X" patterns.
    """

    def __init__(self, run_id: str, root: str = _CACHE_ROOT) -> None:
        self.run_id = run_id
        self.dir = os.path.join(root, run_id)
        os.makedirs(self.dir, exist_ok=True)

    # ── Core path / existence ─────────────────────────────────────────

    def path(self, name: str) -> str:
        """Absolute path of a cache entry. Caller can pass to FFmpeg etc."""
        return os.path.join(self.dir, name)

    def has(self, name: str) -> bool:
        """
        True iff the entry exists AND is non-empty. Empty files are
        treated as "missing" — defends against partial writes from a
        previous Python crash that didn't reach the rename step.
        """
        p = self.path(name)
        try:
            return os.path.getsize(p) > 0
        except OSError:
            return False

    # ── JSON ──────────────────────────────────────────────────────────

    def load_json(self, name: str):
        with open(self.path(name), encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, name: str, data) -> None:
        """Atomic JSON save — writes to .tmp then renames into place."""
        target = self.path(name)
        tmp = target + ".tmp"
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)

    # ── Files ─────────────────────────────────────────────────────────

    def save_file(self, name: str, src_path: str) -> str:
        """
        Atomically copy src_path into the cache as `name`. Returns the
        cache-relative path. Uses tmp+rename so a kill mid-copy can't
        leave a truncated cache file.
        """
        target = self.path(name)
        os.makedirs(os.path.dirname(target) or self.dir, exist_ok=True)
        tmp = target + ".tmp"
        shutil.copy2(src_path, tmp)
        os.replace(tmp, target)
        return target

    def save_files(self, prefix_dir: str, src_paths: list) -> list:
        """
        Bulk-save a list of files into a subdirectory of the cache.
        Returns a parallel list of cached paths. Used for the visuals
        manifest (image_files / clip_files).
        """
        cached = []
        for src in src_paths:
            name = os.path.join(prefix_dir, os.path.basename(src))
            cached.append(self.save_file(name, src))
        return cached

    # ── Markers (no payload, just "this step finished") ───────────────

    def mark_done(self, name: str) -> None:
        """Atomic empty-but-non-zero marker file. Used as a 'flag' when
        the actual step output is multiple files saved separately."""
        target = self.path(name)
        os.makedirs(os.path.dirname(target) or self.dir, exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(datetime.now(timezone.utc).isoformat())
        os.replace(tmp, target)

    # ── Listing / cleanup ─────────────────────────────────────────────

    def list_entries(self) -> list:
        """All non-tmp entries currently in the cache dir."""
        if not os.path.isdir(self.dir):
            return []
        out = []
        for root, _dirs, files in os.walk(self.dir):
            for f in files:
                if f.endswith(".tmp"):
                    continue
                rel = os.path.relpath(os.path.join(root, f), self.dir)
                out.append(rel.replace(os.sep, "/"))
        return sorted(out)

    def __repr__(self) -> str:
        return f"<CheckpointStore run_id={self.run_id!r} dir={self.dir!r}>"
