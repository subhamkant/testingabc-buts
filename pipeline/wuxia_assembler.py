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


def _make_motion_sub_clip(clip_path: str, output_path: str, duration: float) -> bool:
    """Fill `duration` with a motion mp4, scaled/cropped to 1920x1080, silent.

    NEW PLATFORM (2026-07-19) 'LIVING FREEZE': the Wan kernel writes 33f@24fps
    but the model's motion intent is ~16fps, so first RETIME 1.5x to native
    speed (kills the 'jarring fast slideshow'). If the scene outlasts the
    retimed clip, the remainder is a slow Ken Burns push on the clip's LAST
    frame — a moving hold, never a dead freeze-frame (the v5 complaint).
    FORWARD ONLY, no boomerang. CRF 17 to avoid compounding softness."""
    retime = float(os.environ.get("WUXIA_MOTION_RETIME", "1.5"))
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", clip_path], capture_output=True, text=True)
        motion_s = float(probe.stdout.strip()) * retime
    except (ValueError, OSError):
        motion_s = 2.0

    base_vf = f"setpts={retime}*PTS,{_SCALE_CROP},setsar=1,fps={FPS}"
    if motion_s >= duration - 0.05:
        cmd = ["ffmpeg", "-y", "-i", clip_path, "-vf", base_vf,
               "-t", f"{duration:.3f}", "-an", "-c:v", "libx264",
               "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p",
               "-r", str(FPS), output_path]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            print("    [wuxia-motion-clip] failed:\n    "
                  + (r.stderr.decode("utf-8", "replace")[-400:] if r.stderr else ""))
            return False
        return os.path.exists(output_path)

    # motion part + living-hold part
    part_a = output_path.replace(".mp4", "_mA.mp4")
    part_b = output_path.replace(".mp4", "_mB.mp4")
    last_png = output_path.replace(".mp4", "_last.png")
    try:
        r = subprocess.run(["ffmpeg", "-y", "-i", clip_path, "-vf", base_vf,
                            "-an", "-c:v", "libx264", "-preset", "slow",
                            "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS),
                            part_a], capture_output=True)
        if r.returncode != 0:
            return False
        subprocess.run(["ffmpeg", "-y", "-sseof", "-0.06", "-i", part_a,
                        "-frames:v", "1", last_png], capture_output=True)
        hold_s = max(0.3, duration - motion_s)
        frames = max(2, int(hold_s * FPS))
        # gentle 5% push on the final pose — reads as a held beat, stays alive
        r2 = subprocess.run(
            ["ffmpeg", "-y", "-loop", "1", "-i", last_png, "-vf",
             (f"scale={int(WIDTH*1.1)}:{int(HEIGHT*1.1)}:flags=lanczos,"
              f"zoompan=z='1+0.05*on/{frames}':x='(iw-iw/zoom)/2':"
              f"y='(ih-ih/zoom)/2':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS}"),
             "-t", f"{hold_s:.3f}", "-an", "-c:v", "libx264", "-preset", "medium",
             "-crf", "17", "-pix_fmt", "yuv420p", part_b], capture_output=True)
        if r2.returncode != 0:
            return False
        lst = output_path.replace(".mp4", "_list.txt")
        with open(lst, "w") as f:
            f.write(f"file '{os.path.abspath(part_a)}'\n"
                    f"file '{os.path.abspath(part_b)}'\n")
        r3 = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                             "-i", lst, "-t", f"{duration:.3f}", "-c", "copy",
                             output_path], capture_output=True)
        return r3.returncode == 0 and os.path.exists(output_path)
    finally:
        for p in (part_a, part_b, last_png,
                  output_path.replace(".mp4", "_list.txt")):
            try:
                os.remove(p)
            except OSError:
                pass


def _make_sub_clip(shot_path: str, output_path: str, duration: float, motion_idx: int) -> bool:
    """Dispatch one shot: motion mp4 -> forward+freeze clip; still -> Ken Burns."""
    if str(shot_path).lower().endswith(".mp4"):
        return _make_motion_sub_clip(shot_path, output_path, duration)
    return _make_single_kb_clip(
        shot_path, output_path, duration, motion=_LF_MOTIONS[motion_idx % len(_LF_MOTIONS)]
    )


def _kb_beat(src_img: str, out: str, dur: float, variant: int) -> bool:
    """One Ken Burns beat with framing that VARIES by `variant` so consecutive
    hard-cut beats off the same still don't look identical: push-in, pan-left
    on a tighter crop, pan-right, tilt-up. Gives the benchmark's cut rhythm
    without needing a second still."""
    frames = max(2, int(dur * FPS))
    z = f"1+0.10*on/{frames}"
    presets = [
        (z, "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"),                 # centre push-in
        ("1.12", f"(iw-iw/zoom)*(0.15+0.20*on/{frames})", "(ih-ih/zoom)/2"),  # pan L->R tight
        ("1.12", f"(iw-iw/zoom)*(0.65-0.20*on/{frames})", "(ih-ih/zoom)/2"),  # pan R->L tight
        (z, "(iw-iw/zoom)/2", f"(ih-ih/zoom)*(0.30+0.15*on/{frames})"),       # push + tilt down
    ]
    zz, xx, yy = presets[variant % len(presets)]
    vf = (f"scale={int(WIDTH*1.18)}:{int(HEIGHT*1.18)}:flags=lanczos,"
          f"zoompan=z='{zz}':x='{xx}':y='{yy}':d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
          f"setsar=1")
    r = subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", src_img, "-vf", vf,
         "-t", f"{dur:.3f}", "-an", "-c:v", "libx264", "-preset", "medium",
         "-crf", "17", "-pix_fmt", "yuv420p", out], capture_output=True)
    return r.returncode == 0 and os.path.exists(out)


def _make_beat_sequence(shot: str, output_path: str, total_duration: float) -> bool:
    """SHORTER-CUTS (2026-07-21): a long single-shot scene becomes several
    ~WUXIA_MAX_CUT_S hard-cut beats instead of 1.4s motion + a long frozen
    hold. Beat 0 = the motion clip (retimed) if this shot is an mp4; the
    remaining beats are varied Ken Burns moves on the clip's last frame (or
    the still), so the eye gets a fresh frame every few seconds."""
    max_cut = float(os.environ.get("WUXIA_MAX_CUT_S", "4.5"))
    if total_duration <= max_cut * 1.35:
        return _make_sub_clip(shot, output_path, total_duration, 0)

    n_beats = max(2, round(total_duration / max_cut))
    beat_dur = total_duration / n_beats
    is_mp4 = str(shot).lower().endswith(".mp4")
    tmp: list = []
    last_png = output_path.replace(".mp4", "_seq_last.png")
    try:
        for b in range(n_beats):
            bp = output_path.replace(".mp4", f"_seq{b:02d}.mp4")
            if b == 0 and is_mp4:
                if not _make_motion_sub_clip(shot, bp, beat_dur):
                    return False
                # grab the motion clip's last frame as the KB source for later beats
                subprocess.run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", bp,
                                "-frames:v", "1", last_png], capture_output=True)
            else:
                src = last_png if (is_mp4 and os.path.exists(last_png)) else shot
                if str(src).lower().endswith(".mp4"):
                    src = shot  # still-only scene
                if not _kb_beat(src, bp, beat_dur, b):
                    return False
            tmp.append(bp)
        lst = output_path.replace(".mp4", "_seq_list.txt")
        with open(lst, "w") as f:
            for p in tmp:
                f.write(f"file '{os.path.abspath(p)}'\n")
        r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                            "-i", lst, "-t", f"{total_duration:.3f}",
                            "-c", "copy", output_path], capture_output=True)
        return r.returncode == 0 and os.path.exists(output_path)
    finally:
        for p in tmp + [last_png, output_path.replace(".mp4", "_seq_list.txt")]:
            try:
                os.remove(p)
            except OSError:
                pass


def _make_wuxia_scene_clip(shots: list, output_path: str, total_duration: float) -> bool:
    """Landscape scene clip from N shots (motion or still) dissolved via xfade.
    Mirrors longform_assembler._make_landscape_scene_clip but dispatches per shot."""
    n = len(shots)
    if n == 0:
        return False
    if n == 1:
        # SHORTER-CUTS: long single-shot scenes are beat-split (hard cuts every
        # ~WUXIA_MAX_CUT_S) unless disabled.
        if os.environ.get("WUXIA_BEAT_CUTS", "1") != "0":
            return _make_beat_sequence(shots[0], output_path, total_duration)
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
        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
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
        "-c:v", "libx264", "-preset", "medium", "-crf", "16",  # near-lossless 2nd pass
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
            "-c:v", "libx264", "-preset", "medium", "-crf", "17",
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
    _apply_landscape_music(output_path, series="wuxia")  # -c:v copy (no re-encode)
    # NOTE: the end fade-to-black is applied by the subtitle pass
    # (pipeline/wuxia_subtitles.py) so it shares that single final re-encode
    # instead of adding another one here.
    if os.environ.get("WUXIA_MASTER_PASS", "1") != "0":
        _master_grade_watermark(output_path)
    print(f"    [OK] Final episode -> {output_path}")
    return output_path


def _master_grade_watermark(path: str) -> None:
    """FINAL MASTER PASS (new platform 2026-07-19): warm filmic grade +
    vignette + bloom + channel watermarks, ONE re-encode, in place.

    Bloom blend order is load-bearing: ffmpeg `blend` treats the FIRST input
    as the base — sharp stream MUST come first ([v][b]); the reversed order
    output ~95% gaussian blur (the v9-v11 'vaseline' bug, user-found).
    Watermarks: 2x top corners @26% + bottom-right @85% (user spec)."""
    wm = os.path.join("assets", "brand", "kd_lockup.png")
    tmp = path.replace(".mp4", "_master.mp4")
    grade = (
        "curves=r='0/0 0.5/0.55 1/1':g='0/0 0.5/0.51 1/1':b='0/0.02 0.5/0.47 1/0.96',"
        "eq=saturation=1.15:contrast=1.05,vignette=PI/4.8,"
        "gblur=sigma=10[bl];[v][bl]blend=all_mode=screen:all_opacity=0.08")
    fc = (
        f"[0:v]unsharp=5:5:0.6:5:5:0.0,split[v][g];[g]{grade}[graded];"
        f"[1:v]scale=-1:38,format=rgba,colorchannelmixer=aa=0.26,split=2[wtl][wtr];"
        f"[1:v]scale=-1:46,format=rgba,colorchannelmixer=aa=0.85[wb];"
        f"[graded][wtl]overlay=26:20[a];[a][wtr]overlay=W-w-26:20[b2];"
        f"[b2][wb]overlay=W-w-30:H-h-24[out]")
    cmd = ["ffmpeg", "-y", "-i", path, "-i", wm, "-filter_complex", fc,
           "-map", "[out]", "-map", "0:a?", "-c:v", "libx264", "-preset",
           "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-c:a", "copy", tmp]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and os.path.exists(tmp):
        os.replace(tmp, path)
        print("    [master] grade + watermarks applied")
    else:
        err = r.stderr.decode("utf-8", "replace")[-400:] if r.stderr else ""
        print(f"    [master] FAILED (episode kept ungraded):\n    {err}")
