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

# Cinematic grade — gentle warm bias only. The previous version pulled blues
# and greens down hard (b: 0.5→0.44, g: 0.5→0.50→0.96 ceiling) which, stacked
# on top of FLUX's already-warm jewel-tone palette, produced a heavy magenta/
# pink cast across faces and backgrounds (visible in the "भीम और दुर्योधन"
# upload). Lighter curves keep the warm cinematic feel without the wash.
COLOR_GRADE = (
    "curves=r='0/0.02 0.5/0.53 1/1':g='0/0.01 0.5/0.51 1/0.98':b='0/0 0.5/0.48 1/0.96',"
    "eq=contrast=1.05:saturation=1.10:brightness=0.01,"
    "vignette=angle=PI/4"
)

FILM_GRAIN = "noise=alls=14:allf=t+u"

FPS                = 30
XFADE_DURATION     = 0.5
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


# ── Cinematic polish layer (LUT + light-leak overlay + camera shake) ─────────

def _pick_lut() -> str:
    """Pick a random .cube LUT from assets/luts/. Returns path or empty string."""
    lut_dir = "assets/luts"
    if not os.path.isdir(lut_dir):
        return ""
    cubes = [
        os.path.join(lut_dir, f)
        for f in os.listdir(lut_dir)
        if f.lower().endswith(".cube")
    ]
    return random.choice(cubes) if cubes else ""


def _pick_overlay(kind: str = "lightleaks") -> str:
    """Pick a random overlay MP4 from assets/overlays/<kind>/."""
    d = f"assets/overlays/{kind}"
    if not os.path.isdir(d):
        return ""
    mp4s = [
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.lower().endswith((".mp4", ".mov"))
    ]
    return random.choice(mp4s) if mp4s else ""


def _shake_active_expr(boundaries: list, window: float = 0.30) -> str:
    """
    FFmpeg expression that evaluates to 1 during a shake window around each
    scene boundary, 0 elsewhere. Drives the camera-shake offset amplitude.
    """
    if not boundaries:
        return "0"
    parts = [
        f"between(t,{(t - 0.10):.2f},{(t + window - 0.10):.2f})"
        for t in boundaries
    ]
    return "(" + "+".join(parts) + ")"


def _apply_cinematic_polish(video_path: str, clip_durations: list) -> None:
    """
    Final polish pass: camera shake at scene boundaries → 3D LUT grade →
    light-leak overlay (screen blend) → optional FPS bump. In-place modify.

    Every step degrades gracefully: missing assets / filter errors leave the
    original file untouched and the pipeline continues.

    Light-leak overlay is now gated behind CINEMATIC_LIGHT_LEAK env var
    (default OFF). The 2026-05-13 local test showed the screen-blended
    overlay was causing ghost / double-exposure artifacts at scene
    boundaries — two adjacent shots' frames bleeding through the leak
    texture (visible at the 5s mark of preview/local_test.mp4 — two faces
    overlapping). Set CINEMATIC_LIGHT_LEAK=true to re-enable if you decide
    the polish-layer look is worth the ghost risk.
    """
    lut        = _pick_lut()
    # Light-leak overlay: gated by env var, default OFF to suppress the
    # ghost/double-exposure at scene boundaries.
    leak_enabled = os.environ.get("CINEMATIC_LIGHT_LEAK", "false").lower() in ("1", "true", "yes")
    leak       = _pick_overlay("lightleaks") if leak_enabled else ""
    fps_env    = os.environ.get("CINEMATIC_FPS", "").strip()
    target_fps = int(fps_env) if fps_env.isdigit() else 0

    # Scene-boundary timestamps (where xfades happen in the final timeline)
    boundaries = []
    t = 0.0
    for d in clip_durations[:-1]:
        t += d - XFADE_DURATION
        boundaries.append(t)

    # No early-exit: even with no LUT/leak/shake/fps target, the unsharp +
    # lower-CRF re-encode is worth running on its own (better YouTube
    # transcode tier).

    print(
        f"    Polish: LUT={'yes' if lut else 'no'}  "
        f"leak={'yes' if leak else 'no'}  "
        f"shake={len(boundaries)} boundaries  "
        f"fps={target_fps or 'source'}"
    )

    base_filters = []

    if boundaries:
        active = _shake_active_expr(boundaries)
        base_filters.append(
            f"crop=iw-30:ih-30:"
            f"'15+15*sin(t*45)*{active}':"
            f"'15+15*cos(t*38)*{active}'"
        )
        # Restore full 1080x1920 after the shake-crop so YouTube Shorts
        # gets full-resolution output.
        base_filters.append("scale=1080:1920:flags=lanczos")

    if lut and os.path.exists(lut):
        lut_ff = lut.replace("\\", "/")  # FFmpeg wants forward slashes everywhere
        base_filters.append(f"lut3d=file='{lut_ff}'")

    if target_fps in (30, 50, 60):
        base_filters.append(f"fps={target_fps}")

    # Always end the chain with unsharp — this is the apparent-detail boost
    # that nudges YouTube's transcoder to keep us on a 1080p tier.
    base_filters.append(UNSHARP_FILTER)

    polish_out = video_path.replace(".mp4", "_polish.mp4")

    if leak and os.path.exists(leak):
        v_chain = ",".join(base_filters) if base_filters else "null"
        filter_complex = (
            f"[0:v]{v_chain}[base];"
            f"[1:v]scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop=1080:1920,format=yuv420p[leak];"
            f"[base][leak]blend=all_mode=screen:all_opacity=0.25[vout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-stream_loop", "-1", "-i", leak,
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "0:a",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            "-shortest",
            "-movflags", "+faststart",
            polish_out,
        ]
    else:
        # base_filters always has at least UNSHARP_FILTER appended above, so
        # the chain is never empty — no early return needed here.
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", ",".join(base_filters),
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            polish_out,
        ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0 and os.path.exists(polish_out):
        os.replace(polish_out, video_path)
        print("    [OK] Cinematic polish applied")
    else:
        if os.path.exists(polish_out):
            os.remove(polish_out)
        err = result.stderr.decode("utf-8", errors="replace")[:400] if result.stderr else ""
        print(f"    [!] Polish failed, keeping un-polished video")
        if err:
            print(f"    {err}")


# ── Background music with smart ducking ──────────────────────────────────────

def _pick_music_track(series: str = "mahabharata") -> str:
    """
    Returns a music track path, series-aware:

    1. BACKGROUND_MUSIC_PATH env var wins (manual override for testing).
    2. WhatIf: prefers tracks in `assets/music/whatif/` (curiosity / ambient /
       synthwave register). Filenames containing 'whatif', 'curious', or
       'ambient' anywhere in the pool also count. Falls back to non-mythic
       tracks (i.e. excludes the Mahabharata sad-theme + epic tracks) so a
       science video never ends up under a devotional cue.
    3. Mahabharata / Krishna: original behaviour — 50% weight on the sad-theme
       title track, uniform across the rest.

    Returns empty string if nothing found.
    """
    pinned = os.environ.get("BACKGROUND_MUSIC_PATH", "").strip()
    if pinned and os.path.exists(pinned):
        return pinned

    tracks = []
    search_dirs = ["assets", "assets/music"]
    if series == "whatif":
        # Check the dedicated subdir first; if it has anything use ONLY that
        # so a curated whatif pool wins over any generic mahabharata fallback.
        if os.path.isdir("assets/music/whatif"):
            whatif_dir_tracks = [
                os.path.join("assets/music/whatif", f)
                for f in os.listdir("assets/music/whatif")
                if f.lower().endswith(".mp3")
            ]
            if whatif_dir_tracks:
                return random.choice(whatif_dir_tracks)

    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            tracks += [
                os.path.join(search_dir, f)
                for f in os.listdir(search_dir)
                if f.lower().endswith(".mp3")
            ]

    real_tracks = [t for t in tracks if "bgmusic" not in os.path.basename(t)]
    pool = real_tracks if real_tracks else tracks
    if not pool:
        return ""

    # WhatIf: filter the pool to non-mythic tracks. If the pool name suggests
    # epic / sad / devotional content, exclude it. Whatever remains is a
    # better neutral bed than nothing.
    if series == "whatif":
        _MYTHIC_HINTS = ("sad_theme", "mahabharat", "krishna", "bhakti", "devot", "epic")
        whatif_safe = [
            t for t in pool
            if not any(h in os.path.basename(t).lower() for h in _MYTHIC_HINTS)
        ]
        if whatif_safe:
            return random.choice(whatif_safe)
        # No whatif-safe track in the pool → return empty so we skip music
        # entirely rather than fall back to a mythic cue under science content.
        return ""

    sad_theme = next(
        (t for t in pool if "sad_theme" in os.path.basename(t).lower()),
        None,
    )
    others = [t for t in pool if t != sad_theme]
    # 50% weight on the sad theme. Falls through to uniform-random pick when
    # the theme is missing or when the coin flip lands on the others bucket.
    if sad_theme and (not others or random.random() < 0.5):
        return sad_theme
    return random.choice(others or pool)


# YouTube targets -14 LUFS integrated, -1 dBTP true peak. Hitting that gives
# parity with every other Short in the feed; falling short means YouTube
# attenuates competing videos but plays ours at our (quieter) authored level —
# perceived as "weak" audio, which correlates with higher swipe-away rates.
#
# LRA target varies by content type:
#   - Mahabharata third-person narration: LRA=7. Compressed dynamics keep the
#     voice consistent over BG music — fine for a story narrator.
#   - Krishna direct-address: LRA=14. The whole format depends on emotional
#     dynamics — soft contemplative passages then commanding peaks like
#     "उठो पार्थ!". A reference Sourabh-Jain/Sumedh-Mudgalkar Krishna track
#     measured at 14.5 LU; ours at LRA=7 was 1.8 LU and felt monotone.
_LOUDNORM_LRA_BY_SERIES = {
    "krishna": 14,
}
_DEFAULT_LRA = 7


def _loudnorm_filter(series: str = "mahabharata") -> str:
    lra = _LOUDNORM_LRA_BY_SERIES.get(series, _DEFAULT_LRA)
    return f"loudnorm=I=-14:TP=-1.5:LRA={lra}"


# Krishna direct-address builds macro loudness dynamics at the TTS stage via
# explicit per-scene dB scaling in tts_generator._KRISHNA_PER_SCENE_DB
# (e.g. peak scene +4 dB, contemplative scene -2 dB). dynaudnorm would pump
# the quiet scenes back up to match the loud ones — actively undoing what we
# just engineered. So Krishna runs ONLY loudnorm with a generous LRA=14
# target that preserves the macro variation we built in.
#
# (Earlier iterations tried dynaudnorm here; left out intentionally.)
def _audio_post_chain(series: str) -> str:
    """Per-series audio post-processing chain."""
    parts = [_loudnorm_filter(series), "aresample=48000"]
    return ",".join(parts)


# Kept as the default-series filter for backwards-compatible imports.
LOUDNORM_FILTER = _loudnorm_filter("mahabharata")

# Mild unsharp pass applied during the polish re-encode. AI-generated video
# clips upscaled with lanczos to 1080×1920 lack the high-frequency detail that
# YouTube's transcoder uses to decide whether to serve a 1080p tier. Adding
# unsharp restores apparent detail and helps avoid the "downgraded to 720p"
# fate the user's recent video suffered (served as 712×1280 by YT).
UNSHARP_FILTER = "unsharp=5:5:0.6:5:5:0.0"


def _finalize_audio_no_music(output_path: str, series: str = "mahabharata"):
    """
    When no background music track is found, we still need to bring the voice
    audio up to YouTube's -14 LUFS target and resample to 48 kHz / 192 kbps.
    Without this pass, voice would stay at ~-16 LUFS (the natural TTS level)
    and the audio plays noticeably quieter than competing Shorts.
    For series=krishna, dynaudnorm runs first to expand dynamic range.
    """
    finalized = output_path.replace(".mp4", "_norm.mp4")
    audio_chain = _audio_post_chain(series)
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-c:v", "copy",
        "-af", audio_chain,
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        finalized,
    ], capture_output=True)
    if result.returncode == 0:
        os.replace(finalized, output_path)
        print(f"    [OK] Audio normalized ({audio_chain})")
    else:
        if os.path.exists(finalized):
            os.remove(finalized)
        print("    [!] Audio finalization failed, keeping original audio")


def _apply_background_music(output_path: str, series: str = "mahabharata"):
    """
    Mixes a randomly selected background track at 10% base volume with
    sidechain ducking during voiceover. Light by design — the BG music is
    atmosphere only and should never compete with the narrator's voice. The
    final audio is loudness-normalized to YouTube's -14 LUFS target with a
    series-aware LRA (krishna=14 to preserve emotional dynamics, others=7).
    If no music track is available, voice is still normalized via
    _finalize_audio_no_music.
    """
    music_path = _pick_music_track(series=series)
    if not music_path:
        _finalize_audio_no_music(output_path, series=series)
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
    audio_chain = _audio_post_chain(series)

    # volume=0.085 (was 0.10, originally 0.32) — music is atmosphere only,
    # should never compete with the narrator's voice. Sidechain ducking drops
    # it further during voice. audio_chain adds dynaudnorm (krishna only)
    # before loudnorm + aresample.
    duck_filter = (
        f"[0:a]asplit=2[voice_mix][voice_sc];"
        f"[1:a]atrim=0:{video_duration:.3f},asetpts=PTS-STARTPTS,"
        f"volume=0.085[music_raw];"
        f"[music_raw][voice_sc]sidechaincompress="
        f"threshold=0.02:ratio=6:attack=150:release=600:makeup=1[music_ducked];"
        f"[voice_mix][music_ducked]amix=inputs=2:normalize=0,"
        f"{audio_chain}[aout]"
    )

    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex", duck_filter,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        music_output,
    ], capture_output=True)

    if result.returncode == 0:
        os.replace(music_output, output_path)
        print(f"    [OK] Music mixed + audio normalized ({audio_chain})")
        return

    # Flat fallback (no sidechain ducking) — must stay quieter than the ducked
    # path because there's no auto-attenuation when voice plays.
    flat_filter = (
        f"[1:a]volume=0.06,"
        f"atrim=0:{video_duration:.3f},asetpts=PTS-STARTPTS[music];"
        f"[0:a][music]amix=inputs=2:normalize=0,"
        f"{audio_chain}[aout]"
    )
    result2 = subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex", flat_filter,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        music_output,
    ], capture_output=True)

    if result2.returncode == 0:
        os.replace(music_output, output_path)
        print(f"    [OK] Music mixed (flat fallback) + audio normalized ({audio_chain})")
    else:
        print("    [!] Music mix failed, falling back to voice-only normalization")
        _finalize_audio_no_music(output_path, series=series)


# ── Continuous-audio assemblers ──────────────────────────────────────────────
#
# Architecture: build per-scene SILENT video clips → concat with xfades into
# one silent timeline → mux ONE continuous audio over the whole timeline. This
# gives a single uninterrupted voice track with consistent quality (no scene
# seams from per-scene TTS generations) while visuals still cut to the beat.

def _make_silent_image_scene_clip(image_paths, output_path: str, duration: float):
    """Ken Burns over one or more images, silent video output. Mirrors
    _make_scene_clip but skips the audio muxing step."""
    import shutil as _sh

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    n_subs  = len(image_paths)
    sub_dur = duration / n_subs

    if sub_dur < _MIN_SUB_DURATION:
        n_subs = max(1, int(duration / _MIN_SUB_DURATION))
        image_paths = image_paths[:n_subs]
        sub_dur = duration / n_subs

    motions = random.sample(_MOTION_TYPES, min(n_subs, len(_MOTION_TYPES)))
    base    = output_path.replace(".mp4", "")

    sub_paths = []
    sub_durs  = []

    for j in range(n_subs):
        img        = image_paths[j % len(image_paths)]
        motion     = motions[j % len(motions)]
        sub_path   = f"{base}_sub{j}.mp4"
        actual_dur = sub_dur + SUB_XFADE_DURATION

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

    if n_subs == 1:
        _sh.move(sub_paths[0], output_path)
    else:
        _join_video_clips(sub_paths, sub_durs, output_path, SUB_XFADE_DURATION)
        for sp in sub_paths:
            if os.path.exists(sp):
                os.remove(sp)


def _make_boomerang_source(raw_clip_path: str, output_path: str) -> bool:
    """Build a forward+reverse "boomerang" version of the AI clip. Looping a
    boomerang is visually seamless because the motion at every stitch point
    matches frame-for-frame — the playhead just changes direction."""
    cmd = [
        "ffmpeg", "-y",
        "-i", raw_clip_path,
        "-filter_complex", "[0:v]split=2[fwd][rev_in];[rev_in]reverse[rev];[fwd][rev]concat=n=2:v=1:a=0[vout]",
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and os.path.exists(output_path)


def _make_silent_video_scene_clip(raw_clip_path: str, output_path: str, duration: float):
    """Process AI clip into a silent 1080x1920 scaled cinematic clip of the
    given duration.

    For long scenes (typically 10-15s) where a 3.4s AI clip would loop 3-4x
    with visible jump-cuts, we first build a boomerang (forward+reverse)
    source so each loop transition is frame-matched and seamless. For short
    scenes that fit within one playback of the source, we skip the boomerang
    step to save FFmpeg time."""

    # Probe the source clip duration
    src_dur = get_audio_duration(raw_clip_path) or 3.4

    # Boomerang only when we actually need to loop. If duration <= src_dur,
    # one straight playback covers the scene.
    src_for_loop = raw_clip_path
    boomerang_path = output_path.replace(".mp4", "_boom.mp4")
    used_boomerang = False
    if duration > src_dur * 1.05:
        if _make_boomerang_source(raw_clip_path, boomerang_path):
            src_for_loop = boomerang_path
            used_boomerang = True

    vf_parts = [
        "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos",
        "crop=1080:1920",
        COLOR_GRADE,
        FILM_GRAIN,
        "fade=t=in:st=0:d=0.4",
        f"fade=t=out:st={max(duration - 0.4, 0):.2f}:d=0.4",
        "setsar=1",
    ]
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", src_for_loop,
        "-filter_complex", f"[0:v]{','.join(vf_parts)}[vout]",
        "-map", "[vout]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-t", str(duration), "-an",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # Fallback without color grade / grain
        subprocess.run([
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", src_for_loop,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-t", str(duration), "-an",
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
                   "crop=1080:1920,setsar=1",
            output_path,
        ], capture_output=True)

    if used_boomerang and os.path.exists(boomerang_path):
        os.remove(boomerang_path)


def _build_silent_video_with_xfades(clip_paths: list, durations: list, output_path: str) -> bool:
    """Concatenate silent video clips with rotating xfade transitions. Final
    timeline length = sum(durations) - (n-1)*XFADE_DURATION."""
    if len(clip_paths) == 1:
        import shutil as _sh
        _sh.copy2(clip_paths[0], output_path)
        return True

    inputs = []
    for cp in clip_paths:
        inputs += ["-i", cp]

    filter_parts = []
    accumulated  = 0.0
    prev_label   = "[0:v]"

    for idx in range(1, len(clip_paths)):
        accumulated += durations[idx - 1] - XFADE_DURATION
        is_last      = (idx == len(clip_paths) - 1)
        out_label    = "[vout]" if is_last else f"[xf{idx}]"
        transition   = _XFADE_TRANSITIONS[idx % len(_XFADE_TRANSITIONS)]
        filter_parts.append(
            f"{prev_label}[{idx}:v]xfade=transition={transition}"
            f":duration={XFADE_DURATION}:offset={accumulated:.3f}{out_label}"
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

    if result.returncode != 0:
        print(f"    [ERROR] Silent xfade assembly failed:\n{result.stderr.decode()[:400]}")
        return False
    return True


def _mux_continuous_audio(silent_video: str, audio_path: str, output_path: str) -> bool:
    """Mux a single audio track over a silent video. Audio drives final length
    (-shortest cuts the longer stream)."""
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", silent_video,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v", "-map", "1:a",
        "-shortest",
        "-movflags", "+faststart",
        output_path,
    ], capture_output=True)
    if result.returncode != 0:
        print(f"    [ERROR] Audio mux failed:\n{result.stderr.decode()[:400]}")
        return False
    return True


def _per_scene_durations(audio_duration: float, char_weights: list, n: int) -> list:
    """Per-scene clip durations so the final xfaded timeline = audio_duration.
    Accounts for the (n-1) xfade overlaps. Distributes proportionally to
    char_weights when provided, otherwise evenly."""
    target_sum = audio_duration + (n - 1) * XFADE_DURATION
    if not char_weights or len(char_weights) != n or sum(char_weights) <= 0:
        return [target_sum / max(n, 1)] * n
    total_w = sum(char_weights)
    return [(w / total_w) * target_sum for w in char_weights]


def assemble_video_continuous_audio(
    image_files: list,
    audio_path: str,
    script_data: dict,
    char_weights: list = None,
    output_path: str = "output/video.mp4",
    series: str = "mahabharata",
) -> str:
    """Static-image pipeline with ONE continuous audio track. Per-scene
    Ken Burns clips are sized proportionally to the original narration so
    visuals stay roughly in step with the spoken story.

    `series` controls per-series audio finalization (krishna gets LRA=14
    to preserve emotional dynamic range; others use LRA=7)."""
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp/clips", exist_ok=True)

    n = len(image_files)
    audio_duration = get_audio_duration(audio_path)
    durations = _per_scene_durations(audio_duration, char_weights, n)

    print(f"    Audio duration: {audio_duration:.2f}s  |  scenes: {n}")
    print(f"    Per-scene clip durations: " +
          ", ".join(f"{d:.2f}s" for d in durations))

    silent_paths = []
    for i, (imgs, dur) in enumerate(zip(image_files, durations)):
        if isinstance(imgs, str):
            imgs = [imgs]
        silent_path = f"temp/clips/silent_{i:02d}.mp4"
        print(f"    Scene {i+1}/{n} silent ({dur:.2f}s, {len(imgs)} shot(s))...")
        _make_silent_image_scene_clip(imgs, silent_path, dur)
        silent_paths.append(silent_path)

    silent_full = "temp/clips/silent_full.mp4"
    if not _build_silent_video_with_xfades(silent_paths, durations, silent_full):
        return output_path
    if not _mux_continuous_audio(silent_full, audio_path, output_path):
        return output_path

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    [OK] Continuous-audio video -> {output_path} ({size_mb:.1f} MB)")

    _apply_cinematic_polish(output_path, durations)
    _apply_background_music(output_path, series=series)

    return output_path


def assemble_from_video_clips_continuous_audio(
    clip_files: list,
    audio_path: str,
    script_data: dict,
    char_weights: list = None,
    output_path: str = "output/video.mp4",
    series: str = "mahabharata",
    fallback_images: list = None,
) -> str:
    """AI-clip pipeline with ONE continuous audio track. `series` selects
    per-series audio dynamics (LRA=14 for krishna, LRA=7 default).

    `clip_files[i]` may be None for scenes where every I2V provider failed —
    those scenes render from `fallback_images[i]` via Ken Burns instead, so
    a single failure no longer forces the whole video to static. Pass the
    reference-image list returned by `generate_video_clips` as
    `fallback_images` to enable this."""
    os.makedirs("output", exist_ok=True)
    os.makedirs("temp/clips", exist_ok=True)

    n = len(clip_files)
    audio_duration = get_audio_duration(audio_path)
    durations = _per_scene_durations(audio_duration, char_weights, n)

    n_clips  = sum(1 for c in clip_files if c)
    n_static = n - n_clips
    if n_static:
        print(f"    Audio duration: {audio_duration:.2f}s  |  scenes: {n} "
              f"({n_clips} AI clip, {n_static} Ken Burns fallback)")
    else:
        print(f"    Audio duration: {audio_duration:.2f}s  |  AI clips: {n}")
    print(f"    Per-scene clip durations: " +
          ", ".join(f"{d:.2f}s" for d in durations))

    silent_paths = []
    for i, (clip, dur) in enumerate(zip(clip_files, durations)):
        silent_path = f"temp/clips/silent_clip_{i:02d}.mp4"
        if clip:
            print(f"    AI clip {i+1}/{n} silent ({dur:.2f}s)...")
            _make_silent_video_scene_clip(clip, silent_path, dur)
        else:
            img = fallback_images[i] if fallback_images and i < len(fallback_images) else None
            if not img or not os.path.exists(img):
                raise RuntimeError(
                    f"Scene {i+1} has no AI clip and no usable fallback image — "
                    "cannot assemble. Pass `fallback_images` from generate_video_clips."
                )
            print(f"    Ken Burns {i+1}/{n} silent ({dur:.2f}s) — AI clip missing for this scene")
            _make_silent_image_scene_clip(img, silent_path, dur)
        if not os.path.exists(silent_path):
            raise RuntimeError(f"Silent scene {i+1} missing — FFmpeg produced no output")
        silent_paths.append(silent_path)

    silent_full = "temp/clips/silent_full.mp4"
    if not _build_silent_video_with_xfades(silent_paths, durations, silent_full):
        return output_path
    if not _mux_continuous_audio(silent_full, audio_path, output_path):
        return output_path

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    [OK] Continuous-audio video -> {output_path} ({size_mb:.1f} MB)")

    _apply_cinematic_polish(output_path, durations)
    _apply_background_music(output_path, series=series)

    return output_path


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

    _apply_cinematic_polish(output_path, clip_durations)
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

    _apply_cinematic_polish(output_path, processed_durations)
    _apply_background_music(output_path)

    return output_path
