"""
Wuxia assembler — pipeline/wuxia_assembler.py
=============================================

16:9 landscape episode assembler that splices BOTH Ken-Burns stills AND LTX
motion mp4 clips. Forks `longform_assembler.assemble_longform_video` (stills-only)
and adds a per-shot dispatcher:

  * shot path endswith .mp4  -> looped/boomeranged motion sub-clip (1920x1080)
  * else (.jpg/.png)         -> Ken Burns sub-clip (existing landscape helper)

Everything else (per-scene duration from char_weights, xfade dissolve, silent
concat, audio mux, sidechain music duck) is reused verbatim from
longform_assembler so the two stay behaviourally identical outside the motion
branch. No shared modules are modified.
"""
from __future__ import annotations

import os
import subprocess

from pipeline.longform_assembler import (
    FPS,
    SHOT_XFADE_S,
    _LF_MOTIONS,
    _apply_landscape_music,
    _make_single_kb_clip,
    _mux_audio_stereo,
    _per_scene_durations,
    get_audio_duration,
)

WIDTH, HEIGHT = 1920, 1080
_SCALE_CROP = (
    f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
    f"crop={WIDTH}:{HEIGHT},format=yuv420p"
)


def _make_boomerang(clip_path: str, out_path: str) -> bool:
    """Forward + reversed concat, so a short LTX clip loops seamlessly (no hard
    jump at the loop seam). Non-fatal — caller falls back to plain loop."""
    cmd = [
        "ffmpeg", "-y", "-i", clip_path,
        "-filter_complex", "[0:v]reverse[r];[0:v][r]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and os.path.exists(out_path)


def _make_motion_sub_clip(clip_path: str, output_path: str, duration: float) -> bool:
    """Loop an LTX motion mp4 to `duration`, scaled/cropped to 1920x1080, silent.
    Uses a boomerang source so loops are seamless; `-stream_loop -1 -t D` both
    loops (if shorter) and trims (if longer)."""
    src = clip_path
    boomerang = output_path.replace(".mp4", "_boom.mp4")
    if _make_boomerang(clip_path, boomerang):
        src = boomerang

    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", src,
        "-t", f"{duration:.3f}",
        "-vf", f"{_SCALE_CROP},fps={FPS}",
        "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if os.path.exists(boomerang):
        os.remove(boomerang)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace")[-600:] if r.stderr else ""
        print(f"    [wuxia-motion-clip] failed:\n    {err}")
        return False
    return os.path.exists(output_path)


def _make_sub_clip(shot_path: str, output_path: str, duration: float, motion_idx: int) -> bool:
    """Dispatch one shot: motion mp4 -> looped clip; still -> Ken Burns."""
    if str(shot_path).lower().endswith(".mp4"):
        return _make_motion_sub_clip(shot_path, output_path, duration)
    return _make_single_kb_clip(
        shot_path, output_path, duration, motion=_LF_MOTIONS[motion_idx % len(_LF_MOTIONS)]
    )


def _make_wuxia_scene_clip(shots: list, output_path: str, total_duration: float) -> bool:
    """Landscape scene clip from N shots (motion or still) dissolved via xfade.
    Mirrors longform_assembler._make_landscape_scene_clip but dispatches per shot."""
    n = len(shots)
    if n == 0:
        return False
    if n == 1:
        return _make_sub_clip(shots[0], output_path, total_duration, 0)

    overlap_per = SHOT_XFADE_S * (n - 1)
    sub_dur = (total_duration + overlap_per) / n
    if sub_dur < 2 * SHOT_XFADE_S + 0.5:
        return _make_sub_clip(shots[0], output_path, total_duration, 0)

    sub_paths: list = []
    for i, shot in enumerate(shots):
        sub_path = output_path.replace(".mp4", f"_sub{i:02d}.mp4")
        if not _make_sub_clip(shot, sub_path, sub_dur, i):
            for p in sub_paths:
                if os.path.exists(p):
                    os.remove(p)
            return False
        sub_paths.append(sub_path)

    inputs: list = []
    for p in sub_paths:
        inputs += ["-i", p]

    filter_parts: list = []
    prev_label = "[0:v]"
    for i in range(1, n):
        offset = i * (sub_dur - SHOT_XFADE_S)
        out_label = f"[v{i}]" if i < n - 1 else "[vout]"
        filter_parts.append(
            f"{prev_label}[{i}:v]xfade=transition=fade:"
            f"duration={SHOT_XFADE_S}:offset={offset:.3f}{out_label}"
        )
        prev_label = out_label

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    for p in sub_paths:
        if os.path.exists(p):
            os.remove(p)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[-600:] if result.stderr else ""
        print(f"    [wuxia-scene-clip] xfade chain failed:\n    {err}")
        return False
    return os.path.exists(output_path)


def _normalize_pix_fmt(path: str) -> None:
    """Re-encode a scene clip in place to limited-range yuv420p + setsar=1 so all
    clips are uniform before concat. Ken-Burns clips come out yuvj420p (full-range
    JPEG source), motion clips yuv420p — the concat demuxer needs them uniform.
    `format=yuv420p` in the filter (not just -pix_fmt) forces the range too."""
    tmp = path.replace(".mp4", "_norm.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", path,
        "-vf", f"format=yuv420p,setsar=1,fps={FPS}",
        "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        tmp,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and os.path.exists(tmp):
        os.replace(tmp, path)
    elif os.path.exists(tmp):
        os.remove(tmp)


def _concat_scenes(silent_paths: list, output_path: str) -> bool:
    """Concat pre-normalized (uniform yuv420p) per-scene clips via the concat
    demuxer with -c copy (fast, lossless — inputs are already identical params)."""
    list_path = "temp/clips/wuxia_concat_list.txt"
    os.makedirs(os.path.dirname(list_path), exist_ok=True)
    with open(list_path, "w", encoding="utf-8") as f:
        for p in silent_paths:
            abs_p = os.path.abspath(p).replace("\\", "/")
            f.write(f"file '{abs_p}'\n")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", "-movflags", "+faststart", output_path,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace")[-600:] if r.stderr else ""
        print(f"    [wuxia-concat] copy-concat failed, retrying with re-encode:\n    {err}")
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(FPS), output_path,
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace")[-600:] if r.stderr else ""
            print(f"    [wuxia-concat] re-encode failed:\n    {err}")
            return False
    return os.path.exists(output_path)


def assemble_wuxia_video(
    image_files: list,
    audio_path: str,
    script: dict,
    char_weights: list = None,
    output_path: str = "output/wuxia_episode.mp4",
) -> str:
    """Build the final 1920x1080 Wuxia episode: per-scene clips (motion or KB)
    -> concat -> mux voiceover -> sidechain music duck. Returns output_path.

    image_files: list[list[str]] — outer=scene, inner=shots; each shot is a
    .mp4 (motion) or .jpg (still)."""
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp/clips", exist_ok=True)

    n = len(image_files)
    audio_duration = get_audio_duration(audio_path)
    durations = _per_scene_durations(audio_duration, char_weights, n)

    print(f"    Audio: {audio_duration:.2f}s across {n} scenes")

    silent_paths: list = []
    for i, (shots, dur) in enumerate(zip(image_files, durations)):
        if isinstance(shots, str):
            shots = [shots]
        silent_path = f"temp/clips/wuxia_silent_{i:02d}.mp4"
        motion_ct = sum(1 for s in shots if str(s).lower().endswith(".mp4"))
        print(f"    Scene {i+1}/{n} ({dur:.2f}s, {len(shots)} shots, {motion_ct} motion)...")
        if not _make_wuxia_scene_clip(shots, silent_path, dur):
            raise RuntimeError(f"Failed to build scene clip {i+1}")
        _normalize_pix_fmt(silent_path)  # uniform yuv420p for a clean concat
        silent_paths.append(silent_path)

    silent_full = "temp/clips/wuxia_silent_full.mp4"
    if not _concat_scenes(silent_paths, silent_full):
        raise RuntimeError("Failed to concat silent scene clips")

    if not _mux_audio_stereo(silent_full, audio_path, output_path):
        raise RuntimeError("Failed to mux audio onto silent video")

    print(f"    [OK] Pre-music episode -> {output_path}")
    _apply_landscape_music(output_path, series="wuxia")
    print(f"    [OK] Final episode -> {output_path}")
    return output_path
