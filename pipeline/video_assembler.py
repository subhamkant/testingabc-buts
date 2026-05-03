import subprocess
import json
import os
import random


# ── Constants ─────────────────────────────────────────────────────────────────

_MOTION_TYPES = [
    "zoom_in",
    "zoom_out",
    "pan_left",
    "pan_right",
    "pan_up",
    "pan_down",
    "zoom_in_left",
    "zoom_in_right",
]

_XFADE_TRANSITIONS = ["dissolve", "fade", "wipeleft", "wiperight"]

# Warm cinematic grade
COLOR_GRADE = (
    "curves=r='0/0.04 0.5/0.56 1/1':g='0/0.02 0.5/0.50 1/0.96':b='0/0 0.5/0.44 1/0.90',"
    "eq=contrast=1.08:saturation=1.22:brightness=0.015,"
    "vignette=angle=PI/4"
)

FILM_GRAIN = "noise=alls=14:allf=t+u"

FPS                = 25
XFADE_DURATION     = 0.6
SUB_XFADE_DURATION = 0.25
_MIN_SUB_DURATION  = 2.0   # minimum seconds per sub-clip


# ── Audio helpers ─────────────────────────────────────────────────────────────

def get_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", audio_path],
        capture_output=True, text=True,
    )
    try:
        for stream in json.loads(result.stdout).get("streams", []):
            if stream.get("codec_type") == "audio":
                return float(stream.get("duration", 5.0))
    except Exception:
        pass
    return 5.0


# ── Ken Burns motion expressions (portrait 1080×1920) ─────────────────────────

def _ken_burns_expr(motion: str, frames: int) -> str:
    """
    Returns the FFmpeg filter string for Ken Burns motion at 1080x1920 portrait.
    Pre-scales to 2160x3840 so pans have canvas without hitting black borders.
    """
    base = (
        "scale=2160:3840:force_original_aspect_ratio=increase:flags=lanczos,"
        "crop=2160:3840,"
    )
    out = f"s=1080x1920:fps={FPS}"

    expr_map = {
        "zoom_in": (
            f"zoompan=z='min(1+on/{frames}*0.3,1.3)'"
            f":x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d={frames}:{out}"
        ),
        "zoom_out": (
            f"zoompan=z='max(1.3-on/{frames}*0.3,1)'"
            f":x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d={frames}:{out}"
        ),
        "pan_left": (
            f"zoompan=z='1.15'"
            f":x='(iw-iw/zoom)*(1-on/{frames})':y='(ih-ih/zoom)/2':d={frames}:{out}"
        ),
        "pan_right": (
            f"zoompan=z='1.15'"
            f":x='(iw-iw/zoom)*on/{frames}':y='(ih-ih/zoom)/2':d={frames}:{out}"
        ),
        "pan_up": (
            f"zoompan=z='1.15'"
            f":x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*(1-on/{frames})':d={frames}:{out}"
        ),
        "pan_down": (
            f"zoompan=z='1.15'"
            f":x='(iw-iw/zoom)/2':y='(ih-ih/zoom)*on/{frames}':d={frames}:{out}"
        ),
        "zoom_in_left": (
            f"zoompan=z='min(1+on/{frames}*0.25,1.25)'"
            f":x='0':y='(ih-ih/zoom)/2':d={frames}:{out}"
        ),
        "zoom_in_right": (
            f"zoompan=z='min(1+on/{frames}*0.25,1.25)'"
            f":x='iw-iw/zoom':y='(ih-ih/zoom)/2':d={frames}:{out}"
        ),
    }
    return base + expr_map.get(motion, expr_map["zoom_in"])


# ── Single-image clip renderer ────────────────────────────────────────────────

def _render_image_clip(
    image_path: str,
    output_path: str,
    duration: float,
    motion: str,
    fade_in: bool = True,
    fade_out: bool = True,
    with_audio: str = None,
) -> bool:
    frames = max(int(duration * FPS), 50)

    vf_parts = [
        _ken_burns_expr(motion, frames),
        COLOR_GRADE,
        FILM_GRAIN,
    ]
    if fade_in:
        vf_parts.append("fade=t=in:st=0:d=0.3")
    if fade_out:
        vf_parts.append(f"fade=t=out:st={max(duration - 0.3, 0):.2f}:d=0.3")
    vf_parts.append("setsar=1")

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(FPS), "-i", image_path,
    ]
    if with_audio:
        cmd += ["-i", with_audio]

    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        "-vf", ",".join(vf_parts),
    ]
    cmd += (["-c:a", "aac", "-b:a", "128k", "-shortest"] if with_audio else ["-an"])
    cmd.append(output_path)

    return subprocess.run(cmd, capture_output=True).returncode == 0


def _fallback_image_clip(image_path: str, audio_path: str, output_path: str):
    """Last-resort clip: plain scale to portrait, with audio."""
    subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", image_path,
        "-i", audio_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p", "-shortest",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920,setsar=1",
        output_path,
    ], capture_output=True)


# ── Sub-clip joiner ───────────────────────────────────────────────────────────

def _join_video_clips(clip_paths: list, durations: list, output_path: str, xfade_dur: float) -> bool:
    if len(clip_paths) == 1:
        import shutil
        shutil.copy2(clip_paths[0], output_path)
        return True

    inputs = []
    for cp in clip_paths:
        inputs += ["-i", cp]

    filter_parts = []
    prev_label = "[0:v]"
    offset = 0.0

    for idx in range(1, len(clip_paths)):
        offset += durations[idx - 1] - xfade_dur
        is_last = (idx == len(clip_paths) - 1)
        out_label = "[vout]" if is_last else f"[xf{idx}]"
        filter_parts.append(
            f"{prev_label}[{idx}:v]xfade=transition=dissolve"
            f":duration={xfade_dur}:offset={offset:.3f}{out_label}"
        )
        prev_label = out_label

    result = subprocess.run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-an",
        output_path,
    ], capture_output=True)

    return result.returncode == 0


# ── Scene clip builder ────────────────────────────────────────────────────────

def _make_scene_clip(image_paths, audio_path: str, clip_path: str, duration: float):
    """
    Builds one scene clip from one or more images.

    image_paths: str (single image) or list[str] (multiple shots).
    Each image gets its own sub-clip with a unique Ken Burns motion.
    Sub-clips are joined with dissolve transitions, then TTS audio is muxed in.
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    n_subs = len(image_paths)
    sub_dur = duration / n_subs

    # Guard: sub-clips must be long enough for zoompan to work
    if sub_dur < _MIN_SUB_DURATION:
        n_subs = max(1, int(duration / _MIN_SUB_DURATION))
        image_paths = image_paths[:n_subs]
        sub_dur = duration / n_subs

    motions = random.sample(_MOTION_TYPES, min(n_subs, len(_MOTION_TYPES)))
    base    = clip_path.replace(".mp4", "")

    sub_paths = []
    sub_durs  = []

    for j in range(n_subs):
        img        = image_paths[j % len(image_paths)]
        motion     = motions[j % len(motions)]
        sub_path   = f"{base}_sub{j}.mp4"
        actual_dur = sub_dur + SUB_XFADE_DURATION  # overlap for dissolve

        ok = _render_image_clip(
            img, sub_path, actual_dur, motion,
            fade_in=(j == 0),
            fade_out=(j == n_subs - 1),
        )
        if not ok:
            subprocess.run([
                "ffmpeg", "-y",
                "-loop", "1", "-framerate", str(FPS), "-i", img,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-t", str(actual_dur), "-an",
                "-vf", "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
                       "crop=1080:1920,setsar=1",
                sub_path,
            ], capture_output=True)

        sub_paths.append(sub_path)
        sub_durs.append(actual_dur)

    video_only = f"{base}_vonly.mp4"
    join_ok    = _join_video_clips(sub_paths, sub_durs, video_only, SUB_XFADE_DURATION)

    for sp in sub_paths:
        if os.path.exists(sp):
            os.remove(sp)

    if not join_ok or not os.path.exists(video_only):
        _fallback_image_clip(image_paths[0], audio_path, clip_path)
        return

    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", video_only,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        clip_path,
    ], capture_output=True)

    if os.path.exists(video_only):
        os.remove(video_only)

    if result.returncode != 0:
        _fallback_image_clip(image_paths[0], audio_path, clip_path)


def _make_video_scene_clip(raw_clip_path: str, audio_path: str, output_path: str, duration: float):
    """Processes an AI video clip: loops to TTS duration, applies grade + grain."""
    vf_parts = [
        "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos",
        "crop=1080:1920",
        COLOR_GRADE,
        FILM_GRAIN,
        f"fade=t=in:st=0:d=0.4",
        f"fade=t=out:st={max(duration - 0.4, 0):.2f}:d=0.4",
        "setsar=1",
    ]

    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", raw_clip_path,
        "-i", audio_path,
        "-filter_complex", f"[0:v]{','.join(vf_parts)}[vout]",
        "-map", "[vout]", "-map", "1:a",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p",
        "-t", str(duration + 0.5), "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        subprocess.run([
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", raw_clip_path,
            "-i", audio_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p", "-shortest",
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,crop=1080:1920,setsar=1",
            output_path,
        ], capture_output=True)


# ── Final assembly ────────────────────────────────────────────────────────────

def _write_final_video(clip_paths: list, clip_durations: list, output_path: str):
    """
    Joins all scene clips into the final MP4.
    Video: xfade transitions (rotating styles).
    Audio: concat filter — each scene's audio plays sequentially, never overlapping.
    """
    if len(clip_paths) == 1:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", clip_paths[0],
            "-c", "copy", "-movflags", "+faststart",
            output_path,
        ], capture_output=True)
        if result.returncode != 0:
            print(f"    [ERROR] Export failed:\n{result.stderr.decode()}")
        return

    inputs = []
    for cp in clip_paths:
        inputs += ["-i", cp]

    filter_parts = []
    accumulated  = 0.0
    prev_label   = "[0:v]"

    for idx in range(1, len(clip_paths)):
        accumulated += clip_durations[idx - 1] - XFADE_DURATION
        is_last      = (idx == len(clip_paths) - 1)
        out_label    = "[vout]" if is_last else f"[xf{idx}]"
        transition   = _XFADE_TRANSITIONS[idx % len(_XFADE_TRANSITIONS)]
        filter_parts.append(
            f"{prev_label}[{idx}:v]xfade=transition={transition}"
            f":duration={XFADE_DURATION}:offset={accumulated:.3f}{out_label}"
        )
        prev_label = out_label

    # CRITICAL: concat audio sequentially — amix would play all tracks simultaneously
    audio_inputs = "".join(f"[{i}:a]" for i in range(len(clip_paths)))
    filter_parts.append(
        f"{audio_inputs}concat=n={len(clip_paths)}:v=0:a=1[aout]"
    )

    result = subprocess.run([
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        output_path,
    ], capture_output=True)

    if result.returncode != 0:
        print(f"    [ERROR] Assembly failed:\n{result.stderr.decode()}")


# ── Background music with smart ducking ──────────────────────────────────────

def _pick_music_track() -> str:
    """
    Returns a music track path:
    1. BACKGROUND_MUSIC_PATH in .env if set to a real file
    2. Otherwise picks randomly from assets/ and assets/music/ (CI cache dir)
    Returns empty string if nothing found.
    """
    pinned = os.environ.get("BACKGROUND_MUSIC_PATH", "").strip()
    if pinned and os.path.exists(pinned):
        return pinned

    tracks = []
    for search_dir in ["assets", "assets/music"]:
        if os.path.isdir(search_dir):
            tracks += [
                os.path.join(search_dir, f)
                for f in os.listdir(search_dir)
                if f.lower().endswith(".mp3")
            ]

    real_tracks = [t for t in tracks if "bgmusic" not in os.path.basename(t)]
    pool = real_tracks if real_tracks else tracks
    return random.choice(pool) if pool else ""


def _apply_background_music(output_path: str):
    """
    Mixes a randomly selected background track at ~18% volume with
    sidechain ducking during voiceover. Silently skipped if no tracks found.
    """
    music_path = _pick_music_track()
    if not music_path:
        return

    print(f"    Music: {os.path.basename(music_path)}")

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", output_path],
        capture_output=True, text=True,
    )
    try:
        video_duration = float(json.loads(probe.stdout)["format"]["duration"])
    except Exception:
        video_duration = 9999.0

    music_output = output_path.replace(".mp4", "_music.mp4")

    duck_filter = (
        f"[0:a]asplit=2[voice_mix][voice_sc];"
        f"[1:a]atrim=0:{video_duration:.3f},asetpts=PTS-STARTPTS,"
        f"volume=0.18[music_raw];"
        f"[music_raw][voice_sc]sidechaincompress="
        f"threshold=0.02:ratio=6:attack=150:release=600:makeup=1[music_ducked];"
        f"[voice_mix][music_ducked]amix=inputs=2:normalize=0[aout]"
    )

    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex", duck_filter,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        music_output,
    ], capture_output=True)

    if result.returncode == 0:
        os.replace(music_output, output_path)
        print("    [OK] Background music mixed — smart ducking active")
        return

    flat_filter = (
        f"[1:a]volume=0.08,"
        f"atrim=0:{video_duration:.3f},asetpts=PTS-STARTPTS[music];"
        "[0:a][music]amix=inputs=2:normalize=0[aout]"
    )
    result2 = subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex", flat_filter,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        music_output,
    ], capture_output=True)

    if result2.returncode == 0:
        os.replace(music_output, output_path)
        print("    [OK] Background music mixed at 8%")
    else:
        print("    [!] Music mix failed, continuing without music")


# ── Public assemblers ─────────────────────────────────────────────────────────

def assemble_video(
    image_files: list,
    audio_files: list,
    script_data: dict,
    output_path: str = "output/video.mp4",
) -> str:
    """
    Assembles portrait (1080x1920) MP4 from per-scene images + audio.

    image_files: list[str] or list[list[str]] — one image or multiple shots per scene.
    Each shot gets a unique Ken Burns motion; shots are dissolved together.
    Audio streams are concatenated sequentially (no overlap).
    """
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp/clips", exist_ok=True)

    clip_paths     = []
    clip_durations = []
    pairs          = list(zip(image_files, audio_files))

    for i, (imgs, audio) in enumerate(pairs):
        if isinstance(imgs, str):
            imgs = [imgs]
        duration  = get_audio_duration(audio)
        clip_path = f"temp/clips/clip_{i:02d}.mp4"
        print(f"    Scene {i+1}/{len(pairs)} ({duration:.1f}s, {len(imgs)} shots)...")
        _make_scene_clip(imgs, audio, clip_path, duration)
        clip_paths.append(clip_path)
        clip_durations.append(get_audio_duration(clip_path) or (duration + 0.5))

    _write_final_video(clip_paths, clip_durations, output_path)

    if not os.path.exists(output_path):
        return output_path

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    [OK] Video assembled -> {output_path}  ({size_mb:.1f} MB)")

    _apply_background_music(output_path)

    return output_path


def assemble_from_video_clips(
    clip_files: list,
    audio_files: list,
    script_data: dict,
    output_path: str = "output/video.mp4",
) -> str:
    """
    Assembles portrait MP4 from AI video clips (Kling / Hailuo) + TTS audio.
    """
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp/clips", exist_ok=True)

    processed_paths     = []
    processed_durations = []
    pairs               = list(zip(clip_files, audio_files))

    for i, (clip, audio) in enumerate(pairs):
        duration       = get_audio_duration(audio)
        processed_path = f"temp/clips/processed_{i:02d}.mp4"
        print(f"    Processing clip {i+1}/{len(pairs)} ({duration:.1f}s)...")
        _make_video_scene_clip(clip, audio, processed_path, duration)
        if not os.path.exists(processed_path):
            raise RuntimeError(f"Failed to process clip {i+1} — FFmpeg produced no output")
        processed_paths.append(processed_path)
        processed_durations.append(get_audio_duration(processed_path) or (duration + 0.5))

    _write_final_video(processed_paths, processed_durations, output_path)

    if not os.path.exists(output_path):
        return output_path

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    [OK] Video assembled -> {output_path}  ({size_mb:.1f} MB)")

    _apply_background_music(output_path)

    return output_path
