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


# ── Explainer channel — locked visual identity ───────────────────────────────
# The Mahabharata pipeline keeps its existing cinematic grade + xfades + music.
# Explainer is a different brand: investigative, faster cuts, cooler palette
# with one warm accent, no melodic music. These constants are the visual
# half of the Anti-Hype Analyzer identity (script half lives in
# pipeline/explainer_script.py). Edits are deliberate code commits.

_MOTION_PROFILES = {
    "explainer": {
        "segment_duration_range": (1.5, 3.0),  # per Ken-Burns sub-segment
        "transition":              "hard_cut",  # no xfade — punchier
        "zoom_intensity":          0.6,         # subtler than mahabharata (0.8-1.5)
        "glitch_on_retention_hook": True,
    },
}

# Investigative LUT: cool desaturated midtones, warm reds/oranges preserved
# so a red circle / fire / skin warmth still pops against the blue-gray base.
_EXPLAINER_LUT = (
    "colorchannelmixer="
    "rr=0.95:rg=0.05:rb=0.00:"
    "gr=0.00:gg=0.85:gb=0.15:"
    "br=0.00:bg=0.10:bb=0.95,"
    "curves=preset=increase_contrast,"
    "eq=saturation=0.75:contrast=1.10"
)


# Per-pipeline temp root. Mahabharata leaves env unset → "temp". The explainer
# driver sets PIPELINE_TEMP_ROOT="temp/fe" before importing this module so
# explainer working clips land under temp/fe/clips/ (and don't collide with
# mahabharata's temp/clips/ when both pipelines share a machine).
_TEMP_ROOT = os.environ.get("PIPELINE_TEMP_ROOT", "temp")


# v3: Cold-open duration (paired with pipeline.sound_design.COLD_OPEN_DURATION_S).
# Scene 1 absorbs this 0.8s into its Ken Burns clip so the picture has slow
# drift over the bed-only intro period. sound_design prepends the same 0.8s
# of bed audio before narration starts. If these drift, audio + video desync
# at the climax. Keep them numerically identical.
COLD_OPEN_DURATION_S = 0.8


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

def _ken_burns_expr(motion: str, frames: int, intensity: float = 1.0) -> str:
    """
    Returns the FFmpeg filter string for Ken Burns motion at 1080x1920 portrait.
    Pre-scales to 2160x3840 so pans have canvas without hitting black borders.

    `intensity` scales the zoom delta (defaults to 1.0 = current behavior).
    Per-scene escalation passes intensity 0.8 for the opening hook and ramps
    to 1.4 for the climax so visual motion rises with the music's 4-section
    volume curve. Pan motions are unaffected (they use a fixed zoom = 1.15
    and just shift x/y — scaling those would just amplify the canvas crop).
    """
    base = (
        "scale=2160:3840:force_original_aspect_ratio=increase:flags=lanczos,"
        "crop=2160:3840,"
    )
    out = f"s=1080x1920:fps={FPS}"
    # Cap intensity so we never zoom past 1.6 (any further loses too much
    # canvas and starts looking like a crash-zoom artifact).
    intensity = max(0.5, min(intensity, 1.6))
    zi_delta  = 0.30 * intensity   # base zoom_in delta (was hardcoded 0.3)
    zi_cap    = 1.0 + zi_delta     # final zoom factor
    zo_start  = zi_cap             # zoom_out starts where zoom_in ends
    zir_delta = 0.25 * intensity   # zoom_in_left/right delta
    zir_cap   = 1.0 + zir_delta

    expr_map = {
        "zoom_in": (
            f"zoompan=z='min(1+on/{frames}*{zi_delta:.3f},{zi_cap:.3f})'"
            f":x='(iw-iw/zoom)/2':y='(ih-ih/zoom)/2':d={frames}:{out}"
        ),
        "zoom_out": (
            f"zoompan=z='max({zo_start:.3f}-on/{frames}*{zi_delta:.3f},1)'"
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
            f"zoompan=z='min(1+on/{frames}*{zir_delta:.3f},{zir_cap:.3f})'"
            f":x='0':y='(ih-ih/zoom)/2':d={frames}:{out}"
        ),
        "zoom_in_right": (
            f"zoompan=z='min(1+on/{frames}*{zir_delta:.3f},{zir_cap:.3f})'"
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
    intensity: float = 1.0,
) -> bool:
    frames = max(int(duration * FPS), 50)

    vf_parts = [
        _ken_burns_expr(motion, frames, intensity=intensity),
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


def _apply_cinematic_polish(video_path: str, clip_durations: list, series: str = "mahabharata") -> None:
    """
    Final polish pass: camera shake at scene boundaries → 3D LUT grade →
    light-leak overlay (screen blend) → optional FPS bump. In-place modify.

    `series="explainer"` skips the 3D LUT pick because the explainer pipeline
    has already applied its own cool-blue investigative LUT in
    assemble_explainer_video; layering the warm cinematic LUT on top would
    cancel the brand color. The shake + unsharp + lower-CRF re-encode still
    run (they're brand-agnostic quality wins).

    Every step degrades gracefully: missing assets / filter errors leave the
    original file untouched and the pipeline continues.

    Light-leak overlay is gated behind CINEMATIC_LIGHT_LEAK env var
    (default OFF). The 2026-05-13 local test showed the screen-blended
    overlay was causing ghost / double-exposure artifacts at scene
    boundaries — two adjacent shots' frames bleeding through the leak
    texture (visible at the 5s mark of preview/local_test.mp4 — two faces
    overlapping). Set CINEMATIC_LIGHT_LEAK=true to re-enable if you decide
    the polish-layer look is worth the ghost risk.
    """
    lut        = "" if series == "explainer" else _pick_lut()
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

# ─────────────────────────────────────────────────────────────────────────────
# SFX (sound effects) at scene boundaries — added 2026-05-15
# ─────────────────────────────────────────────────────────────────────────────
# Successful Hindi mythology Shorts layer diegetic SFX at climax moments
# (sword clang on वध, divine bell on Krishna scenes, war drum on Kurukshetra
# transitions). Adds the perceptual polish that distinguishes professional
# mythology Shorts from text-to-speech demos. Gated by ENABLE_SFX env (default
# "true"). Files live in assets/sfx/ — currently ffmpeg-synthesized placeholders;
# replace with real sourced SFX from pixabay/freesound for production quality.

_SFX_DIR = "assets/sfx"

# Mood / narration keyword → SFX file. The mood string is the script-supplied
# "3-6 word English emotional tone phrase" per scene. Keywords match
# substrings (case-insensitive). First match wins; falls back to chime.
_SFX_KEYWORD_MAP = [
    # (keywords, sfx_file, volume)
    (("battle", "war", "fight", "fierce", "rage", "anger", "sword", "vadh", "kill",
      "fury", "duel"),                                       "sword_clang.mp3", 0.45),
    (("divine", "god", "krishna", "miracle", "blessing", "sacred", "holy",
      "celestial"),                                          "divine_bell.mp3", 0.40),
    (("epic", "cosmic", "thunder", "battlefield", "kurukshetra", "earth-shaking",
      "shocking", "doomed"),                                 "war_drum.mp3",    0.45),
    (("vow", "promise", "oath", "declaration", "announcement", "pratigya",
      "decree"),                                             "conch.mp3",       0.40),
]
_SFX_DEFAULT = ("chime.mp3", 0.30)

# Climax SFX boost — added 2026-05-16. The final narrative scene's SFX
# volume is multiplied by this factor so it lands at the peak of the
# music's 4-section curve (which hits vol=0.130 at the climax). Without
# this boost the final SFX gets the same mid-video volume as scene
# transitions — flat energy curve. Tier 1 Fix 1.8 from the plan.
_FINAL_SCENE_BOOST = 1.4


def _pick_sfx_for_mood(mood: str) -> tuple[str, float] | None:
    """Maps a scene's mood phrase to (sfx_filename, volume) or None on miss."""
    if not mood:
        return None
    low = mood.lower()
    for keywords, sfx_file, vol in _SFX_KEYWORD_MAP:
        if any(k in low for k in keywords):
            path = os.path.join(_SFX_DIR, sfx_file)
            if os.path.exists(path):
                return path, vol
    # Fall back to chime
    chime_path = os.path.join(_SFX_DIR, _SFX_DEFAULT[0])
    if os.path.exists(chime_path):
        return chime_path, _SFX_DEFAULT[1]
    return None


def _inject_scene_boundary_sfx(video_path: str, scene_durations: list,
                                scene_moods: list, lead_in_s: float = 0.15) -> bool:
    """
    Re-encode the video's audio track with SFX layered in at each scene
    boundary. Each SFX starts `lead_in_s` before its scene begins so it
    fades into the new scene rather than disrupting the previous.

    Returns True on success; on any ffmpeg failure leaves the video
    unchanged (defensive — never break the assembly).

    Skipped entirely when ENABLE_SFX env var is "false"/"0".
    """
    if os.environ.get("ENABLE_SFX", "true").lower() in ("0", "false", "no"):
        print("    [sfx] ENABLE_SFX=false — skipping scene-boundary SFX")
        return False
    if not os.path.exists(video_path):
        return False
    n = len(scene_durations)
    if n != len(scene_moods) or n == 0:
        return False

    # Compute scene START times (cumulative). Skip scene 0 (don't SFX-bump
    # the opening — let the hook narration land clean).
    starts = []
    t = 0.0
    for d in scene_durations:
        starts.append(t)
        t += d

    # Build per-scene SFX entries with delay+volume.
    #
    # Tier 1.5 (Fix 1.10): SKIP the VALLEY scene entirely so its silence
    # can land — pairs with the music's 0.040 dip and the valley scene's
    # 0.7x motion intensity. The valley creates the contrast that makes
    # the CLIMAX SFX land harder.
    #
    # Layout (n=7: 6 narrative + 1 static outro at the very end):
    #   index 0  = scene 1 (hook)    — skip (existing)
    #   index 1  = scene 2 (setup)   — mood-mapped SFX
    #   index 2  = scene 3 (rehook)  — mood-mapped SFX
    #   index 3  = scene 4 (rise)    — mood-mapped SFX
    #   index 4  = scene 5 (VALLEY)  — SKIP (new, Fix 1.10)         → n-3
    #   index 5  = scene 6 (CLIMAX)  — mood-mapped + ×1.4 boost     → n-2
    #   index 6  = subscribe outro   — silent transition (existing) → n-1
    #
    # For n>=4 valley = n-3, climax = n-2. For n<4 the valley logic
    # gracefully degrades (valley_idx becomes -1 = disabled).
    climax_idx = n - 2 if n >= 2 else n - 1
    valley_idx = n - 3 if n >= 4 else -1   # -1 disables the valley skip
    sfx_inputs = []  # list of (path, start_s, volume)
    skipped_valley = False
    for i in range(1, n):  # skip scene 0
        if i == valley_idx:
            skipped_valley = True
            continue   # let silence land
        pick = _pick_sfx_for_mood(scene_moods[i])
        if pick:
            sfx_path, vol = pick
            if i == climax_idx:
                vol = min(vol * _FINAL_SCENE_BOOST, 1.0)  # clamp to avoid clipping
            sfx_start = max(0.0, starts[i] - lead_in_s)
            sfx_inputs.append((sfx_path, sfx_start, vol))
    if skipped_valley:
        print(f"    [sfx] skipping valley scene {valley_idx + 1} — letting silence land before climax")

    if not sfx_inputs:
        print("    [sfx] no SFX matched any scene mood — skipping")
        return False

    # Build ffmpeg command
    # -i video  -i sfx1 -i sfx2 ... then filter_complex:
    #   [N:a]volume=v,adelay=ms|ms[sfxN]
    #   [0:a][sfx1][sfx2]...amix=...[out]
    cmd = ["ffmpeg", "-y", "-i", video_path]
    for sfx_path, _, _ in sfx_inputs:
        cmd += ["-i", sfx_path]

    fc_parts = []
    sfx_labels = []
    for idx, (_, start_s, vol) in enumerate(sfx_inputs):
        delay_ms = int(start_s * 1000)
        # input index is idx+1 (because video audio is [0:a])
        label = f"sfx{idx}"
        fc_parts.append(
            f"[{idx+1}:a]volume={vol},adelay={delay_ms}|{delay_ms},apad[{label}]"
        )
        sfx_labels.append(f"[{label}]")
    # Mix video audio with all delayed SFX. duration=first to keep original length.
    mix = "[0:a]" + "".join(sfx_labels) + f"amix=inputs={1+len(sfx_inputs)}:duration=first:dropout_transition=0[out]"
    fc_parts.append(mix)

    out_path = video_path + ".sfx.mp4"
    cmd += [
        "-filter_complex", ";".join(fc_parts),
        "-map", "0:v",
        "-map", "[out]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        out_path,
    ]

    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace")[-400:] if r.stderr else ""
        print(f"    [!] SFX mix failed (non-fatal): {err}")
        return False

    try:
        os.replace(out_path, video_path)
        print(f"    [OK] Layered {len(sfx_inputs)} SFX into audio at scene boundaries (climax boosted x{_FINAL_SCENE_BOOST})")
        return True
    except Exception as e:
        print(f"    [!] SFX swap failed: {e}")
        return False


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

    # Copyrighted-music blocklist: tracks empirically confirmed to trigger
    # YouTube Content ID claims. Defense-in-depth alongside the physical
    # quarantine in `assets/quarantine/` (picker doesn't scan that dir).
    #
    # Tunetank-wide ban (2026-05-17): #4 Bhishma Kurukshetra was blocked
    # for matching "Tunetank Inc. - People Of Mughai" — Tunetank licenses
    # the same compositions through multiple distributors, so EVERY
    # tunetank-* file is unsafe regardless of their "royalty-free" claim.
    _BANNED_MUSIC = (
        # B.R. Chopra Mahabharat OST (Sony/Doordarshan-owned), hit #2
        "mahabharat_sad_theme.mp3",       # "Ek Maa Ki Santane - Ye Kaisi..."
        # Generic-named, no Pixabay attribution
        "bgmusic.mp3",
        # Tunetank family — banned as a class after #4 block
        "tunetank-indian-hindi-song-music-349213.mp3",   # "People Of Mughai" (#4 block)
        "tunetank-indian-hindi-song-music-349033.mp3",
        "tunetank-india-indian-hingi-music-348347.mp3",
        "tunetank-epic-indian-hindi-song-music-347195.mp3",
    )

    # Reserved-asset list: files that live in `assets/` for other purposes
    # (ambient bed for chunking mode, future SFX overlays) but must NEVER
    # be selected as a main music track. Excluded from the picker pool in
    # addition to the Content-ID ban list.
    _NON_MUSIC_RESERVED = (
        "dark_ambient.mp3",   # synthesized drone — ambient bed only, not a melodic music track
    )
    real_tracks = [
        t for t in tracks
        if os.path.basename(t).lower() not in _BANNED_MUSIC
        and os.path.basename(t).lower() not in _NON_MUSIC_RESERVED
    ]
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


# Path to the continuous ambient bed that fills gaps between music chunks
# when MUSIC_CHUNKING=true. Looped to cover the full video duration.
#
# Origin: SYNTHESIZED procedurally via ffmpeg lavfi (brown noise + 55Hz +
# 82Hz sine layers + aecho reverb) by tools/synthesize_ambient_bed.py.
# Procedural origin = guaranteed no Content ID match. The recipe in that
# tool is fully reproducible — anyone can regenerate the exact bed.
_AMBIENT_BED_PATH   = "assets/dark_ambient.mp3"
_AMBIENT_BED_ORIGIN = "SYNTHESIZED (ffmpeg lavfi, see tools/synthesize_ambient_bed.py)"


def _build_chunked_music_track(
    source_mp3: str, output_mp3: str, total_dur: float, seed: int,
) -> list[tuple[float, float, float]] | None:
    """
    Build a pre-chunked music track from `source_mp3` for `total_dur` seconds.

    Each chunk:
      - 2.5-4.5 sec random length (seeded)
      - random non-linear offset into source (chunk N's source position is
        independent of chunk N-1's — breaks melody continuity for Content ID)
      - 50ms fade-in + 50ms fade-out per chunk (smooths edges, no clicks)
    Gaps between chunks: 400-700 ms of silence (anullsrc). Filled in the
    final mix by the ambient bed layer (so the output never feels like
    abrupt silence).

    Returns the chunk schedule on success (list of (chunk_dur, src_start, gap_after)
    tuples — useful for claim-risk scoring), or None if ffmpeg fails. On failure
    the caller falls back to passing `source_mp3` directly (no chunking).
    """
    if not os.path.exists(source_mp3):
        return None
    # Probe source duration
    p = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", source_mp3],
        capture_output=True, text=True,
    )
    try:
        source_dur = float(json.loads(p.stdout)["format"]["duration"])
    except Exception:
        return None
    if source_dur < 5.0:
        return None  # source too short to chunk meaningfully

    rng = random.Random(seed)
    # Build schedule of (source_start, source_end) for each chunk; gaps go
    # between in the concat step.
    schedule: list[tuple[float, float, float]] = []  # (chunk_dur, source_start, gap_after)
    t_out = 0.0
    while t_out < total_dur - 0.5:  # leave 0.5s safety margin
        chunk_dur = rng.uniform(2.5, 4.5)
        chunk_dur = min(chunk_dur, total_dur - t_out)
        if chunk_dur < 1.5:
            break  # tail too short, stop chunking
        # Random non-linear offset — anywhere in source where chunk fits
        max_offset = max(0.1, source_dur - chunk_dur - 0.1)
        src_start = rng.uniform(0.0, max_offset)
        gap = rng.uniform(0.4, 0.7)
        schedule.append((chunk_dur, src_start, gap))
        t_out += chunk_dur + gap

    if not schedule:
        return None

    # Build filtergraph: extract each chunk via atrim, fade edges, concat
    # with anullsrc silence gaps in between.
    filter_parts = []
    concat_inputs = []
    for i, (cdur, sstart, _gap) in enumerate(schedule):
        fade_out_st = max(0.0, cdur - 0.05)
        filter_parts.append(
            f"[0:a]atrim=start={sstart:.3f}:end={sstart + cdur:.3f},"
            f"asetpts=PTS-STARTPTS,"
            f"afade=t=in:d=0.05,afade=t=out:st={fade_out_st:.3f}:d=0.05"
            f"[c{i}]"
        )
        concat_inputs.append(f"[c{i}]")
        # Silence gap (except after the last chunk)
        if i < len(schedule) - 1:
            _, _, gap = schedule[i]
            filter_parts.append(
                f"anullsrc=r=44100:cl=stereo:d={gap:.3f}[g{i}]"
            )
            concat_inputs.append(f"[g{i}]")

    n_inputs = len(concat_inputs)
    filter_parts.append(
        f"{''.join(concat_inputs)}concat=n={n_inputs}:v=0:a=1[out]"
    )
    filtergraph = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        "-i", source_mp3,
        "-filter_complex", filtergraph,
        "-map", "[out]",
        "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "44100",
        output_mp3,
    ]
    r = subprocess.run(cmd, capture_output=True)
    ok = r.returncode == 0 and os.path.exists(output_mp3)
    if ok:
        n_chunks = len(schedule)
        total_music_s = sum(c[0] for c in schedule)
        longest_chunk = max(c[0] for c in schedule)
        print(f"    [chunking] {n_chunks} chunks, total music {total_music_s:.1f}s "
              f"(longest={longest_chunk:.2f}s), non-linear offsets from source")
        return schedule
    err = r.stderr.decode("utf-8", errors="replace")[-200:] if r.stderr else ""
    print(f"    [chunking] WARN: chunk build failed — falling back to continuous. {err}")
    return None


def _print_audio_risk(
    chunk_schedule: list[tuple[float, float, float]] | None,
    video_duration: float,
    ambient_bed_path: str,
    ambient_origin: str = "UNVERIFIED",
) -> None:
    """
    Emit a per-render [audio-risk] log block summarizing Content-ID risk
    exposure. Tier A heuristic — Tier B adds voice_coverage + pitch_variance
    fields once those features are implemented.

    Verdict heuristic:
      LOW    : longest_continuous_chunk < 5.0
      MEDIUM : longest_continuous_chunk 5.0-7.0 OR ambient_bed UNVERIFIED
      HIGH   : longest_continuous_chunk > 7.0

    If STRICT_AUDIO_RISK=true and verdict=HIGH, raise to fail the run.
    Default (false) is print-only so existing crons keep shipping while
    the heuristic gets calibrated.
    """
    if not chunk_schedule:
        # Continuous music (chunking disabled or fell back) — flag as HIGH
        # because the entire track plays continuously = Content-ID jackpot.
        longest = video_duration
        n_chunks = 1
        total_music = video_duration
    else:
        longest = max(c[0] for c in chunk_schedule)
        n_chunks = len(chunk_schedule)
        total_music = sum(c[0] for c in chunk_schedule)

    music_exposure_pct = (total_music / video_duration * 100.0) if video_duration > 0 else 0.0

    bed_label = (
        f"{os.path.basename(ambient_bed_path)} (origin: {ambient_origin})"
        if ambient_bed_path else "NONE"
    )

    # Verdict heuristic. Continuous-music mode (chunk_schedule is None) is
    # treated as LOW assuming the actual Content-ID defenses are active:
    # Tunetank ban + <60s cap + narration dominance via sidechain ducking.
    # That stack is what blocks claims in practice — chunking adds tiny
    # marginal protection at the cost of cinematic continuity (user
    # feedback 2026-05-17 after the chunked-#4 sounded fragmented).
    bed_active = bool(ambient_bed_path)
    if chunk_schedule is None:
        verdict = "LOW (continuous mode — relies on ban + <60s + ducking)"
    elif longest > 7.0:
        verdict = "HIGH"
    elif longest > 5.0:
        verdict = "MEDIUM"
    elif bed_active and ambient_origin == "UNVERIFIED":
        verdict = "MEDIUM"
    else:
        verdict = "LOW"

    print(f"    [audio-risk]")
    print(f"      music_exposure_s            = {total_music:.1f} / {video_duration:.1f}  ({music_exposure_pct:.1f}%)")
    print(f"      longest_continuous_chunk_s  = {longest:.2f}")
    print(f"      num_chunks                  = {n_chunks}")
    print(f"      ambient_bed                 = {bed_label}")
    print(f"      verdict                     = {verdict}")

    if verdict == "HIGH" and os.environ.get("STRICT_AUDIO_RISK", "false").strip().lower() == "true":
        raise RuntimeError(
            f"audio-risk verdict=HIGH (longest_chunk={longest:.2f}s); "
            f"STRICT_AUDIO_RISK=true blocks upload. Re-render with shorter chunks "
            f"or set STRICT_AUDIO_RISK=false to override."
        )


_LIGHT_LEAK_PATH = "assets/overlays/lightleaks/synth_warm_sweep.mp4"
_LENS_FLARE_PATH = "assets/overlays/lensflares/synth_flare.mp4"


def _apply_phase3_overlays(output_path: str) -> None:
    """
    Phase 3 (2026-05-17) — subtle cinematic motion overlays:
      • Light leak  (assets/overlays/lightleaks/synth_warm_sweep.mp4)
        Looped, screen blend, alpha 0.22. Warm atmospheric haze.
      • Lens flare  (assets/overlays/lensflares/synth_flare.mp4)
        Looped, screen blend, alpha 0.28. "In the air" rather than
        "edited in" — conservative intensity per user round-5 feedback.
      • Synthetic particle layer
        Sparse twinkling brightness specks above luma=200, blend=addition,
        alpha 0.03. Adds the "alive" feeling of micro-motion without
        crossing into cheap-template territory.

    All three intensities calibrated per round-5/6 plan refinements:
    "subconsciously felt, not consciously noticed".

    Disable via env: CINEMATIC_OVERLAYS=false
    Override alphas: PHASE3_LEAK_ALPHA / PHASE3_FLARE_ALPHA /
                     PHASE3_PARTICLE_ALPHA

    No-op if the overlay assets are missing or if disabled by env.
    """
    # ⚠ 2026-05-18 EMERGENCY DISABLE: the Phase 3 overlay pass produced a
    # magenta cast across every frame of the #5 Shikhandi render. Two bugs
    # stacked:
    #   (1) `blend=all_mode=screen:all_opacity=1` operates in the input's
    #       color space, which is YUV420P here. Screen blend is defined for
    #       RGB — applied to U/V chroma channels it shifts hue wildly
    #       (orange + yellow overlays → magenta result).
    #   (2) The intended layer-transparency (`colorchannelmixer=aa=0.22`) is
    #       overridden by `all_opacity=1`. The overlays effectively blend at
    #       100% strength, not the 22-28% I documented.
    # Plus the source overlays themselves are solid-color fills (not the
    # textured leak/flare assets the names suggest) — even a correctly-
    # coded blend would tint every frame heavily.
    # Default flipped from "true" → "false" until the color-space handling
    # is rewritten (likely: convert base + overlays to gbrp before blending,
    # use blend's per-input alpha or replace with proper overlay filter).
    if os.environ.get("CINEMATIC_OVERLAYS", "false").strip().lower() != "true":
        return
    if not (os.path.exists(_LIGHT_LEAK_PATH) and os.path.exists(_LENS_FLARE_PATH)):
        print("    [phase3] overlay assets missing — skipping (light leak / lens flare not found)")
        return
    if not os.path.exists(output_path):
        return

    dur = get_audio_duration(output_path)
    if not dur or dur < 1.0:
        return

    try:
        leak_alpha = float(os.environ.get("PHASE3_LEAK_ALPHA", "0.22"))
        flare_alpha = float(os.environ.get("PHASE3_FLARE_ALPHA", "0.28"))
        particle_alpha = float(os.environ.get("PHASE3_PARTICLE_ALPHA", "0.03"))
    except ValueError:
        leak_alpha, flare_alpha, particle_alpha = 0.22, 0.28, 0.03

    # Synthetic particle source — sparse brightness specks via lavfi.
    # noise=alls=30 generates per-pixel noise; geq thresholds it so only
    # the brightest pixels survive (sparse twinkles, not uniform grain).
    particle_lavfi = (
        f"color=c=black:s=1080x1920:d={dur:.3f}:r=30,"
        f"noise=alls=30:allf=t,"
        f"format=yuv420p,"
        f"geq=lum='if(gt(lum(X,Y),230),255,0)':cb=128:cr=128"
    )

    # Filter graph:
    #   [0:v] = original video    [1:v] = light leak (looped)
    #   [2:v] = lens flare (looped)  [3:v] = synthesized particles
    filter_graph = (
        f"[1:v]scale=1080:1920,trim=duration={dur:.3f},"
        f"format=yuva420p,colorchannelmixer=aa={leak_alpha:.3f}[leak];"
        f"[2:v]scale=1080:1920,trim=duration={dur:.3f},"
        f"format=yuva420p,colorchannelmixer=aa={flare_alpha:.3f}[flare];"
        f"[3:v]scale=1080:1920,trim=duration={dur:.3f},"
        f"format=yuva420p,colorchannelmixer=aa={particle_alpha:.3f}[particles];"
        f"[0:v][leak]blend=all_mode=screen:all_opacity=1[v1];"
        f"[v1][flare]blend=all_mode=screen:all_opacity=1[v2];"
        f"[v2][particles]blend=all_mode=addition:all_opacity=1[vout]"
    )

    overlaid = output_path.replace(".mp4", "_overlay.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", output_path,
        "-stream_loop", "-1", "-i", _LIGHT_LEAK_PATH,
        "-stream_loop", "-1", "-i", _LENS_FLARE_PATH,
        "-f", "lavfi", "-i", particle_lavfi,
        "-filter_complex", filter_graph,
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "copy",
        "-movflags", "+faststart",
        overlaid,
    ]
    print(f"    [phase3] applying overlays — leak α={leak_alpha} flare α={flare_alpha} particles α={particle_alpha}")
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and os.path.exists(overlaid):
        os.replace(overlaid, output_path)
        print(f"    [phase3] [OK] subtle cinematic overlays applied")
    else:
        err = r.stderr.decode("utf-8", errors="replace")[-300:] if r.stderr else ""
        print(f"    [phase3] WARN: overlay pass failed, keeping pre-overlay video — {err}")
        if os.path.exists(overlaid):
            os.remove(overlaid)


# Phase 2/3 Part F.3 (2026-05-19): atempo ceilings for the early-section
# safety net. Per user direction: emotional Hindi narration with restrained
# pauses + breath-heavy cadence starts sounding subtly rushed above 1.05.
# 1.07 is the hard ceiling — beyond that we refuse the silent-fix and fall
# back to end-chop with a loud warning. Most overflow events should land
# between 1.00 and 1.05 (transparent fix).
_SAFE_EARLY_ATEMPO_PREFERRED = 1.05
_SAFE_EARLY_ATEMPO_CEIL      = 1.07


def _enforce_max_duration(output_path: str, durations: list | None = None) -> None:
    """
    Hard-cap the final mp4 at MAX_DURATION_S seconds (default 58.5). If
    the file is already under the cap, no-op.

    YouTube applies tighter Content ID restrictions on Shorts >60s — the
    2026-05-17 #4 Bhishma Kurukshetra video (70.4s) was blocked partly
    because its duration triggered the over-60s restriction tier. Capping
    at 58.5s keeps every upload in the safer <60s bucket.

    Phase 2/3 Part F.3 (2026-05-19): when `durations` (the per-scene visual
    clip durations from `_per_scene_durations()`) is provided, this function
    PROTECTS THE TAIL on overflow. Instead of blind end-chop (which kills
    aftermath + outro residue), it applies atempo + setpts to the EARLY
    SECTION only — final scene + outro stay at exactly 1.0x. Three tiers:

      atempo factor ≤ 1.05  → transparent fix, no warning
      1.05 < factor ≤ 1.07  → soft NOTE log, proceed (still imperceptible
                              to most listeners but flag for cross-arc
                              monitoring; tighten F.1 word target if it
                              keeps firing)
      factor > 1.07         → WARN + fall through to end-chop (tail will
                              be chopped; this means F.1 is too generous
                              and the script-side word target needs to
                              come down)

    WhatIf longform pipeline can opt out by setting MAX_DURATION_S=999 in
    its workflow env. Default 58.5 applies to Mahabharata + Krishna + WhatIf
    Shorts.
    """
    try:
        max_s = float(os.environ.get("MAX_DURATION_S", "58.5"))
    except ValueError:
        max_s = 58.5
    if max_s >= 999:
        return  # opt-out for longform
    if not os.path.exists(output_path):
        return
    dur = get_audio_duration(output_path)
    if not dur or dur <= max_s + 0.05:  # 50ms safety margin to avoid pointless re-encodes
        return

    overshoot = dur - max_s

    # ── Fast path: legacy callers without per-scene durations ─────────
    # Explainer/WhatIf longform + legacy code paths that don't track scene
    # boundaries fall back to blind end-trim. This branch should never
    # fire for Mahabharata renders post-F.2 — all four mahabharata call
    # sites pass durations.
    if not durations or len(durations) < 3:
        return _legacy_end_chop(output_path, dur, max_s,
                                reason="durations not provided")

    # Protected tail = sum of last 2 visual clip durations (final scene + outro)
    protected_tail = sum(durations[-2:])
    early_section_end = max(0.0, dur - protected_tail)

    # Required atempo factor on EARLY section only.
    target_early_dur = early_section_end - overshoot
    if target_early_dur <= 0.5:
        # Overshoot is so large the early section can't absorb it — fall
        # through to end-chop with explicit reason. F.1 word target needs
        # to come down significantly.
        return _legacy_end_chop(output_path, dur, max_s,
                                reason=f"overshoot {overshoot:.2f}s exceeds early-section "
                                       f"budget {early_section_end:.2f}s — tighten F.1")

    factor = early_section_end / target_early_dur

    if factor > _SAFE_EARLY_ATEMPO_CEIL:
        return _legacy_end_chop(output_path, dur, max_s,
                                reason=f"early-section atempo {factor:.4f}x exceeds "
                                       f"{_SAFE_EARLY_ATEMPO_CEIL:.2f} HARD ceiling — "
                                       f"tighten F.1 word target. Residue WILL be chopped.")

    if factor > _SAFE_EARLY_ATEMPO_PREFERRED:
        print(f"    [auto-cap] NOTE: early-section atempo {factor:.4f}x exceeds 1.05 "
              f"preferred (Hindi pacing comfort zone). Proceeding because under 1.07 "
              f"hard ceiling. Watch for audible compression on next renders.")

    # ── Per-segment compress: speed up EARLY section, leave tail untouched ──
    print(f"    [auto-cap] per-segment compress: early {early_section_end:.2f}s @ "
          f"{factor:.4f}x → {target_early_dur:.2f}s; protected tail "
          f"{protected_tail:.2f}s untouched (overshoot {overshoot:.2f}s absorbed)")
    sped = output_path.replace(".mp4", "_sped.mp4")
    filter_graph = (
        f"[0:v]trim=0:{early_section_end:.6f},setpts=PTS-STARTPTS,"
        f"setpts=PTS/{factor:.6f}[v1];"
        f"[0:v]trim=start={early_section_end:.6f},setpts=PTS-STARTPTS[v2];"
        f"[0:a]atrim=0:{early_section_end:.6f},asetpts=PTS-STARTPTS,"
        f"atempo={factor:.6f}[a1];"
        f"[0:a]atrim=start={early_section_end:.6f},asetpts=PTS-STARTPTS[a2];"
        f"[v1][v2]concat=n=2:v=1:a=0[v];"
        f"[a1][a2]concat=n=2:v=0:a=1[a]"
    )
    cmd = [
        "ffmpeg", "-y", "-i", output_path,
        "-filter_complex", filter_graph,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        sped,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and os.path.exists(sped):
        os.replace(sped, output_path)
        new_dur = get_audio_duration(output_path)
        print(f"    [auto-cap] new duration: {new_dur:.2f}s (target {max_s:.2f}s, "
              f"tail preserved at 1.0x)")
    else:
        err = r.stderr.decode("utf-8", errors="replace")[-300:] if r.stderr else ""
        print(f"    [auto-cap] WARN: per-segment compress failed — falling back to end-chop. {err}")
        if os.path.exists(sped):
            os.remove(sped)
        _legacy_end_chop(output_path, dur, max_s, reason="ffmpeg filter_complex failed")


def _legacy_end_chop(output_path: str, dur: float, max_s: float, reason: str = "") -> None:
    """Blind end-trim fallback used by legacy callers + F.3 safety-net failure
    cases. This is the OLD pre-F.3 behavior: re-encode with -t to force exact
    duration. Loses any content past max_s (including aftermath + outro
    residue on overflow days). Loud warning when invoked so cross-arc
    monitoring can spot it."""
    if reason:
        print(f"    [auto-cap] mp4 is {dur:.2f}s, end-chopping to {max_s:.2f}s — {reason}")
    else:
        print(f"    [auto-cap] mp4 is {dur:.2f}s, end-chopping to {max_s:.2f}s")
    trimmed = output_path.replace(".mp4", "_capped.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", output_path,
        "-t", f"{max_s:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        trimmed,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode == 0 and os.path.exists(trimmed):
        os.replace(trimmed, output_path)
        new_dur = get_audio_duration(output_path)
        print(f"    [auto-cap] new duration: {new_dur:.2f}s")
    else:
        err = r.stderr.decode("utf-8", errors="replace")[-300:] if r.stderr else ""
        print(f"    [auto-cap] WARN: trim failed, keeping original at {dur:.2f}s — {err}")
        if os.path.exists(trimmed):
            os.remove(trimmed)


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

    # MUSIC_CHUNKING: pre-build a chunked music track + layer an ambient bed
    # underneath.
    #
    # Default: OFF (continuous music). The user-reviewed #4 Bhishma render
    # 2026-05-17 with 15 chunks at 2.5-4.5s sounded fragmented — the
    # emotional arc broke at every chunk boundary and the ambient bed was
    # too exposed in gaps. Cinematic mythological content depends on
    # tonal continuity that chunking actively destroys.
    #
    # The actual Content-ID defenses are now:
    #   1. Tunetank-class ban + safe-only picker pool (root cause #4 block)
    #   2. 58.5s auto-cap (the >60s threshold is what triggers stricter scans)
    #   3. Sidechain ducking + narration dominance (existing 5-section curve)
    # Those three protect us. Chunking adds tiny marginal protection at the
    # cost of cinematic quality. Set MUSIC_CHUNKING=true to opt in for
    # experimentation; production defaults to continuous music.
    chunking_enabled = os.environ.get("MUSIC_CHUNKING", "false").strip().lower() == "true"
    ambient_bed_path = _AMBIENT_BED_PATH if os.path.exists(_AMBIENT_BED_PATH) else ""

    chunked_music_path = music_path  # default: use the source as-is
    chunk_schedule: list[tuple[float, float, float]] | None = None
    if chunking_enabled:
        # Seed chunking off the output_path basename — deterministic per-video,
        # but different across re-renders / different scenes.
        seed_int = abs(hash(os.path.basename(output_path))) % (2**31)
        candidate = "temp/_chunked_music.mp3"
        os.makedirs("temp", exist_ok=True)
        chunk_schedule = _build_chunked_music_track(music_path, candidate, video_duration, seed_int)
        if chunk_schedule:
            chunked_music_path = candidate
            if ambient_bed_path:
                print(f"    Ambient bed: {os.path.basename(ambient_bed_path)} (flat 0.020)")
            else:
                print(f"    [chunking] WARN: no ambient bed at {_AMBIENT_BED_PATH} — chunk gaps will be silence")
        # If chunk-build failed, chunked_music_path stays = music_path (graceful fallback)

    music_output = output_path.replace(".mp4", "_music.mp4")
    audio_chain = _audio_post_chain(series)

    # 5-section emotional volume curve with VALLEY DIP (Tier 1.5, Fix 1.10
    # + 1.11.b, 2026-05-16). User watched the first 1.5 render and confirmed
    # music still ~10% too loud — applied an additional across-the-curve
    # reduction of ~10% so narration carries the emotion and music sits
    # under it. Music's job is atmosphere; narration is the protagonist.
    #
    #   Window                  Volume    Tier 1     Tier 1.5    Tier 1.5.b
    #   0-3s (mystery)          0.050     was 0.060  was 0.055   now 0.050
    #   3-8s (tension)          0.067     was 0.080  was 0.075   now 0.067
    #   8 → valley_start (emot) 0.085     was 0.100  was 0.095   now 0.085
    #   VALLEY window           0.040     —          was 0.040   now 0.040 (unchanged dip)
    #   post-valley (climax)    0.098     was 0.130  was 0.110   now 0.098
    #
    # Sidechain still tightened: attack 80ms, release 450ms, ratio 7.
    valley_end_t   = max(8.0, video_duration - 7.0)   # ~7s of climax tail
    valley_start_t = max(5.0, valley_end_t - 5.0)     # 5s valley window
    # Valley floor: 0.040 → 0.048 (Tier 2 Fix 2.2.b, 2026-05-16). User
    # predicted pure 0.040 dip would feel "too empty" for mobile listeners
    # in noisy environments (bus, street). 0.048 preserves the perceptible
    # dip vs the 0.085 emotion-section ceiling but stays audible enough
    # that the scene doesn't read as broken silence.
    music_volume_expr = (
        f"if(lt(t,3),0.050,"
        f"if(lt(t,8),0.067,"
        f"if(lt(t,{valley_start_t:.2f}),0.085,"
        f"if(lt(t,{valley_end_t:.2f}),0.048,"
        f"0.098))))"
    )
    # Build the filter graph + ffmpeg input list. If chunking is active AND
    # the ambient bed exists, layer the bed as a 3rd input ([2:a]) at flat
    # 0.020 volume so chunk gaps don't feel like dead silence. Otherwise
    # use the 2-input continuous-music graph (Tier 1.5 behavior).
    use_ambient_layer = chunking_enabled and ambient_bed_path and chunked_music_path != music_path
    if use_ambient_layer:
        duck_filter = (
            f"[0:a]asplit=2[voice_mix][voice_sc];"
            f"[1:a]atrim=0:{video_duration:.3f},asetpts=PTS-STARTPTS,"
            f"volume='{music_volume_expr}':eval=frame[music_raw];"
            f"[music_raw][voice_sc]sidechaincompress="
            f"threshold=0.02:ratio=7:attack=80:release=450:makeup=1[music_ducked];"
            f"[2:a]atrim=0:{video_duration:.3f},asetpts=PTS-STARTPTS,"
            f"volume=0.020[ambient_flat];"
            f"[voice_mix][music_ducked][ambient_flat]amix=inputs=3:normalize=0,"
            f"{audio_chain}[aout]"
        )
        ffmpeg_inputs = [
            "-i", output_path,
            "-stream_loop", "-1", "-i", chunked_music_path,
            "-stream_loop", "-1", "-i", ambient_bed_path,
        ]
    else:
        duck_filter = (
            f"[0:a]asplit=2[voice_mix][voice_sc];"
            f"[1:a]atrim=0:{video_duration:.3f},asetpts=PTS-STARTPTS,"
            f"volume='{music_volume_expr}':eval=frame[music_raw];"
            f"[music_raw][voice_sc]sidechaincompress="
            f"threshold=0.02:ratio=7:attack=80:release=450:makeup=1[music_ducked];"
            f"[voice_mix][music_ducked]amix=inputs=2:normalize=0,"
            f"{audio_chain}[aout]"
        )
        ffmpeg_inputs = [
            "-i", output_path,
            "-stream_loop", "-1", "-i", chunked_music_path,
        ]

    result = subprocess.run([
        "ffmpeg", "-y",
        *ffmpeg_inputs,
        "-filter_complex", duck_filter,
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        music_output,
    ], capture_output=True)

    if result.returncode == 0:
        os.replace(music_output, output_path)
        print(f"    [OK] Music mixed (5-section curve w/ valley dip: "
              f"0.050→0.067→0.085→[VALLEY {valley_start_t:.1f}-{valley_end_t:.1f}s @ 0.048]→0.098, "
              f"sidechain a80/r450/ratio7) + audio normalized ({audio_chain})")
        # In continuous mode the ambient bed isn't mixed in, so report it as
        # not active rather than showing a misleading file path.
        risk_bed_path   = ambient_bed_path if use_ambient_layer else ""
        risk_bed_origin = _AMBIENT_BED_ORIGIN if use_ambient_layer else "n/a (continuous mode)"
        _print_audio_risk(chunk_schedule, video_duration, risk_bed_path, risk_bed_origin)
        return

    # Flat fallback (no sidechain ducking) — must stay quieter than the ducked
    # path because there's no auto-attenuation when voice plays. Still uses
    # chunked music + ambient bed when available.
    if use_ambient_layer:
        flat_filter = (
            f"[1:a]volume=0.06,"
            f"atrim=0:{video_duration:.3f},asetpts=PTS-STARTPTS[music];"
            f"[2:a]atrim=0:{video_duration:.3f},asetpts=PTS-STARTPTS,"
            f"volume=0.020[ambient_flat];"
            f"[0:a][music][ambient_flat]amix=inputs=3:normalize=0,"
            f"{audio_chain}[aout]"
        )
    else:
        flat_filter = (
            f"[1:a]volume=0.06,"
            f"atrim=0:{video_duration:.3f},asetpts=PTS-STARTPTS[music];"
            f"[0:a][music]amix=inputs=2:normalize=0,"
            f"{audio_chain}[aout]"
        )
    result2 = subprocess.run([
        "ffmpeg", "-y",
        *ffmpeg_inputs,
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
        # In continuous mode the ambient bed isn't mixed in, so report it as
        # not active rather than showing a misleading file path.
        risk_bed_path   = ambient_bed_path if use_ambient_layer else ""
        risk_bed_origin = _AMBIENT_BED_ORIGIN if use_ambient_layer else "n/a (continuous mode)"
        _print_audio_risk(chunk_schedule, video_duration, risk_bed_path, risk_bed_origin)
    else:
        print("    [!] Music mix failed, falling back to voice-only normalization")
        _finalize_audio_no_music(output_path, series=series)


# ── Continuous-audio assemblers ──────────────────────────────────────────────
#
# Architecture: build per-scene SILENT video clips → concat with xfades into
# one silent timeline → mux ONE continuous audio over the whole timeline. This
# gives a single uninterrupted voice track with consistent quality (no scene
# seams from per-scene TTS generations) while visuals still cut to the beat.

def _make_freeze_clip(image_path: str, duration: float, output_path: str) -> bool:
    """Static (no Ken Burns) freeze frame at 1080x1920. Used for the Phase 24
    cold-open prepend: a 1.0s flash of the climax scene's first image,
    crossfading into scene 0's Ken Burns motion. Pure ffmpeg fallback —
    matches the encoding spec of `_render_image_clip`'s fallback path so
    `_build_silent_video_with_xfades` can concat the result cleanly."""
    result = subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", str(FPS), "-i", image_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-t", str(duration), "-an",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=increase:flags=lanczos,"
               "crop=1080:1920,setsar=1",
        output_path,
    ], capture_output=True)
    return result.returncode == 0


def _make_silent_image_scene_clip(image_paths, output_path: str, duration: float, intensity: float = 1.0):
    """Ken Burns over one or more images, silent video output. Mirrors
    _make_scene_clip but skips the audio muxing step.

    `intensity` (added 2026-05-16) scales the per-scene Ken Burns zoom delta.
    Caller passes 0.8 for opening scenes, ramping to ~1.4 for the climax so
    the visual motion rises with the music's 4-section emotional curve.
    """
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
            intensity=intensity,
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


# Phase 2/3 Part F.2 (2026-05-19): minimum visual-clip durations for the
# protected tail (final narrative scene + subscribe outro). These are
# env-tunable per render. The aftermath validator (Part A) + outro
# restraint (Part C) only land emotionally if the visual clips hold long
# enough for the residue to settle. Without minimums, char_weight
# proportional distribution can shrink either to ~3s on dense-narration
# days, killing the emotional landing.
_FINAL_SCENE_MIN_S = float(os.environ.get("FINAL_SCENE_MIN_S", "7.5"))
_OUTRO_MIN_S       = float(os.environ.get("OUTRO_MIN_S", "3.5"))


def _per_scene_durations(audio_duration: float, char_weights: list, n: int,
                         protect_tail: bool = True) -> list:
    """Per-scene clip durations so the final xfaded timeline = audio_duration.
    Accounts for the (n-1) xfade overlaps. Distributes proportionally to
    char_weights when provided, otherwise evenly.

    Phase 2/3 Part F.2 (2026-05-19): when `protect_tail=True` (default), the
    final two scenes (scene 6 narrative + subscribe outro) are guaranteed
    to land AT LEAST `_FINAL_SCENE_MIN_S` / `_OUTRO_MIN_S` seconds. If the
    proportional split would shrink them below those floors, time is
    "stolen" from the earlier scenes proportionally — the donor scenes
    contract, the tail expands. Audio is unchanged (visual-clip retiming
    only); char_weights still drives the baseline. The protected tail is
    what gives Parts A-E (aftermath imagery + outro restraint) the
    emotional-settling time they were architected to need.

    Set `protect_tail=False` for non-mahabharata callers (explainer
    longform, WhatIf science) that don't follow the aftermath architecture.
    """
    target_sum = audio_duration + (n - 1) * XFADE_DURATION
    if not char_weights or len(char_weights) != n or sum(char_weights) <= 0:
        return [target_sum / max(n, 1)] * n
    total_w = sum(char_weights)
    durs = [(w / total_w) * target_sum for w in char_weights]

    if not protect_tail or n < 3:
        return durs

    # Compute deficit: how much each protected-tail slot is short of its floor.
    final_deficit = max(0.0, _FINAL_SCENE_MIN_S - durs[-2])
    outro_deficit = max(0.0, _OUTRO_MIN_S       - durs[-1])
    deficit = final_deficit + outro_deficit
    if deficit <= 0:
        return durs

    # Steal from earlier scenes proportionally. Require donor scenes to
    # retain at least 1.0s of total after donation — fall through to
    # baseline split otherwise (rare; only on very short audio).
    donor_total = sum(durs[:-2])
    if donor_total <= deficit + 1.0:
        print(f"    [per-scene] WARN: tail-protect needs {deficit:.2f}s but "
              f"early scenes only have {donor_total:.2f}s — keeping baseline "
              f"split (tail will be short).")
        return durs

    scale = (donor_total - deficit) / donor_total
    for i in range(n - 2):
        durs[i] *= scale
    durs[-2] = max(durs[-2], _FINAL_SCENE_MIN_S)
    durs[-1] = max(durs[-1], _OUTRO_MIN_S)
    print(f"    [per-scene] protected tail: stole {deficit:.2f}s from early "
          f"scenes (final={durs[-2]:.1f}s, outro={durs[-1]:.1f}s)")
    return durs


# ── Explainer-only assembler ─────────────────────────────────────────────────
# Lives alongside (not inside) assemble_video_continuous_audio. Same building
# blocks reused (_make_silent_image_scene_clip, _mux_continuous_audio), but the
# orchestration is different: hard cuts instead of xfades, lower Ken Burns
# intensity, explainer LUT applied at the end, and NO mahabharata-specific
# steps (no background music, no cinematic polish, no light-leak overlays,
# no scene-boundary SFX). The audio track passed in is expected to be already
# mixed by pipeline.sound_design so the assembler doesn't touch audio post.

def _per_scene_durations_explainer(audio_duration: float, char_weights: list, n: int) -> list:
    """Hard-cut version of _per_scene_durations: timeline = sum(durations) =
    audio_duration exactly (no xfade overlap to compensate for)."""
    if not char_weights or len(char_weights) != n or sum(char_weights) <= 0:
        return [audio_duration / max(n, 1)] * n
    total_w = sum(char_weights)
    return [(w / total_w) * audio_duration for w in char_weights]


def _concat_hard_cuts(clip_paths: list, output_path: str) -> bool:
    """Concat clips with zero crossfade via FFmpeg concat demuxer. All input
    clips must share codec/resolution/fps (they do — _make_silent_image_scene_clip
    standardizes to 1080×1920 H.264 30fps)."""
    if len(clip_paths) == 1:
        import shutil as _sh
        _sh.copy2(clip_paths[0], output_path)
        return True

    list_file = output_path + ".concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for cp in clip_paths:
            cp_abs = os.path.abspath(cp).replace("\\", "/")
            f.write(f"file '{cp_abs}'\n")

    result = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path,
    ], capture_output=True)

    if os.path.exists(list_file):
        os.remove(list_file)

    if result.returncode != 0:
        # `-c copy` fails when codec params drift even slightly — fall back to re-encode
        print(f"    [explainer] hard-cut concat copy failed, re-encoding...")
        list_file = output_path + ".concat.txt"
        with open(list_file, "w", encoding="utf-8") as f:
            for cp in clip_paths:
                cp_abs = os.path.abspath(cp).replace("\\", "/")
                f.write(f"file '{cp_abs}'\n")
        result = subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-an",
            output_path,
        ], capture_output=True)
        if os.path.exists(list_file):
            os.remove(list_file)

    if result.returncode != 0:
        print(f"    [ERROR] explainer hard-cut concat failed:\n{result.stderr.decode()[:400]}")
        return False
    return True


def _apply_explainer_lut_inplace(video_path: str) -> bool:
    """Re-encode video with the explainer LUT applied (cool blue-gray base,
    warm accent preserved). Overwrites in place."""
    tmp = video_path + ".lut.mp4"
    result = subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", _EXPLAINER_LUT,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        tmp,
    ], capture_output=True)
    if result.returncode != 0:
        print(f"    [ERROR] explainer LUT failed:\n{result.stderr.decode()[:400]}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, video_path)
    return True


def assemble_explainer_video(
    image_files: list,
    audio_path: str,
    script_data: dict,
    char_weights: list = None,
    output_path: str = "output/video_explainer.mp4",
) -> str:
    """
    Explainer pipeline: hard cuts + low-zoom Ken Burns + cool-LUT grade.
    Audio (passed in) is expected to be PRE-mixed by pipeline.sound_design with
    the ambient bed + retention-hook hits. This function does not touch audio
    beyond muxing it onto the picture.
    """
    os.makedirs("output", exist_ok=True)
    os.makedirs(f"{_TEMP_ROOT}/clips", exist_ok=True)

    n = len(image_files)
    audio_duration = get_audio_duration(audio_path)

    # v3: the audio file from sound_design.apply_explainer_audio_bed includes
    # a 0.8s bed-only cold open BEFORE narration starts. Per-scene clip
    # durations distribute the NARRATION portion proportionally to char_weights,
    # then scene 1 absorbs the cold-open seconds so its Ken Burns clip plays
    # over both the silent intro and the start of the hook narration. Without
    # this, scene 1 would be sized for narration-only and scene 2's clip would
    # start playing during the cold-open silence.
    narration_duration = max(audio_duration - COLD_OPEN_DURATION_S, 0.1)
    durations = _per_scene_durations_explainer(narration_duration, char_weights, n)
    if n > 0:
        durations[0] += COLD_OPEN_DURATION_S

    profile = _MOTION_PROFILES["explainer"]
    zoom_intensity = profile["zoom_intensity"]

    print(f"    [explainer] audio={audio_duration:.2f}s (incl. {COLD_OPEN_DURATION_S}s cold open)  "
          f"scenes={n}  zoom_intensity={zoom_intensity}")
    print(f"    [explainer] per-scene durations: " +
          ", ".join(f"{d:.2f}s" for d in durations))

    silent_paths = []
    for i, (imgs, dur) in enumerate(zip(image_files, durations)):
        if isinstance(imgs, str):
            imgs = [imgs]
        silent_path = f"{_TEMP_ROOT}/clips/explainer_silent_{i:02d}.mp4"
        print(f"    [explainer] scene {i+1}/{n} silent ({dur:.2f}s, {len(imgs)} shot(s))...")
        _make_silent_image_scene_clip(imgs, silent_path, dur, intensity=zoom_intensity)
        silent_paths.append(silent_path)

    silent_full = f"{_TEMP_ROOT}/clips/explainer_silent_full.mp4"
    if not _concat_hard_cuts(silent_paths, silent_full):
        return output_path
    if not _mux_continuous_audio(silent_full, audio_path, output_path):
        return output_path

    # Color identity — the final brand-defining pass.
    print(f"    [explainer] applying investigative LUT (cool blue-gray + warm accent)")
    _apply_explainer_lut_inplace(output_path)

    # Phase 1 — Cinematic Foundation. Same shake + unsharp + lower-CRF pass
    # the Mahabharata pipeline runs, with the warm LUT skipped (series flag)
    # so the explainer's cool blue stays the dominant grade.
    print(f"    [explainer] Phase 1 — Cinematic Foundation (shake + unsharp + grade preserve)")
    _apply_cinematic_polish(output_path, durations, series="explainer")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    [OK] Explainer video -> {output_path} ({size_mb:.1f} MB)")

    # Deliberately skipped:
    #   _apply_background_music   — explainer audio is pre-mixed by sound_design
    #   _apply_phase3_overlays    — globally disabled (magenta-cast bug)
    #   _inject_scene_boundary_sfx — sword clangs / bells are mahabharata-only

    return output_path


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

    # Per-scene Ken Burns intensity (Tier 1.5, Fix 1.10): linear ramp from
    # 0.8x → 1.4x like Tier 1, but with TWO overrides on the narrative
    # scenes right before the outro:
    #   • Valley scene (index n-3):  forced to 0.7x (slow drift — matches
    #     the audio dip in the music's 5-section curve)
    #   • Climax scene (index n-2):  forced to 1.5x (boost from ramp's 1.4x
    #     — compensates for the valley energy drop right before)
    #   • Outro scene  (index n-1):  static asset (no Ken Burns), so any
    #     intensity passed in is moot for that scene.
    # The valley→climax intensity gap (0.7→1.5) is what makes the climax
    # FEEL more aggressive than mid-video transitions.
    def _scene_intensity(i: int) -> float:
        if n <= 1:
            return 1.0
        if n >= 4 and i == n - 3:
            return 0.7   # valley scene
        if n >= 3 and i == n - 2:
            return 1.5   # climax scene (the narrative climax, not the outro)
        return 0.8 + 0.6 * (i / (n - 1))

    # Phase 24 (2026-05-21): cold-open prepend — flash the climax scene's
    # first frame for ~0.5s of pure freeze + 0.5s xfade into scene 0's hook.
    # Decides Shorts swipe rate: most viewers see one frame before deciding to
    # stay. Climax frame is far more attention-grabbing than the typical
    # establishing-shot scene 1 image. Audio plays continuously; scene 0's
    # visual is shortened by (COLD_OPEN_S - XFADE_DURATION) so the total
    # video timeline still matches audio_duration exactly. The trim is
    # VISUAL ONLY — narration audio for scene 1 is unaffected, which is why
    # we preserve `original_durations` for scene-boundary SFX timing below.
    COLD_OPEN_S      = 1.0
    SCENE_0_TRIM     = COLD_OPEN_S - XFADE_DURATION   # 0.5s
    original_durations = list(durations)  # audio-timing reference for SFX
    apply_cold_open  = n >= 3 and durations[0] > SCENE_0_TRIM + 0.5
    if apply_cold_open:
        durations[0] -= SCENE_0_TRIM
        print(f"    [Phase 24] Scene 1 visual trimmed -{SCENE_0_TRIM:.2f}s to absorb cold-open prepend")

    silent_paths = []
    for i, (imgs, dur) in enumerate(zip(image_files, durations)):
        if isinstance(imgs, str):
            imgs = [imgs]
        silent_path = f"temp/clips/silent_{i:02d}.mp4"
        intensity = _scene_intensity(i)
        tag = ""
        if n >= 4 and i == n - 3:
            tag = " [VALLEY]"
        elif n >= 3 and i == n - 2:
            tag = " [CLIMAX]"
        print(f"    Scene {i+1}/{n} silent ({dur:.2f}s, {len(imgs)} shot(s), KB intensity={intensity:.2f}){tag}...")
        _make_silent_image_scene_clip(imgs, silent_path, dur, intensity=intensity)
        silent_paths.append(silent_path)

    if n >= 4:
        print(f"    [OK] Ken Burns intensity: ramp 0.80→1.30 across scenes 1-{n-3}, "
              f"VALLEY scene {n-2} @ 0.70x, CLIMAX scene {n-1} @ 1.50x, outro scene {n} static")

    if apply_cold_open:
        climax_idx = n - 2
        climax_imgs = image_files[climax_idx]
        climax_first = climax_imgs[0] if isinstance(climax_imgs, list) else climax_imgs
        cold_open_path = "temp/clips/silent_cold_open.mp4"
        if _make_freeze_clip(climax_first, COLD_OPEN_S, cold_open_path):
            silent_paths = [cold_open_path] + silent_paths
            durations    = [COLD_OPEN_S] + list(durations)
            print(f"    [Phase 24] Cold-open prepend: {COLD_OPEN_S:.1f}s static of climax (scene {climax_idx+1}) image")
        else:
            # Roll back the scene-0 trim if the freeze clip failed to generate
            durations[0] += SCENE_0_TRIM
            print(f"    [Phase 24] Cold-open prepend FAILED; falling back to original scene-1 cold open")

    silent_full = "temp/clips/silent_full.mp4"
    if not _build_silent_video_with_xfades(silent_paths, durations, silent_full):
        return output_path
    if not _mux_continuous_audio(silent_full, audio_path, output_path):
        return output_path

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    [OK] Continuous-audio video -> {output_path} ({size_mb:.1f} MB)")

    # Scene-boundary SFX (mood-mapped): sword clang on battle, divine bell on
    # Krishna, war drum on Kurukshetra, conch on vows, chime default. Layered
    # into narration audio BEFORE music mix so SFX gets sidechain ducking too.
    # Phase 24 (2026-05-21): uses `original_durations` (pre-cold-open-trim) so
    # SFX/shake/duration-cap all align to narrative scene boundaries, not to
    # the cold-open visual prepend.
    scene_moods = [s.get("mood", "") for s in (script_data.get("scenes") or [])]
    if len(scene_moods) == n:
        _inject_scene_boundary_sfx(output_path, original_durations, scene_moods)

    _apply_cinematic_polish(output_path, original_durations)
    _apply_background_music(output_path, series=series)
    _apply_phase3_overlays(output_path)
    _enforce_max_duration(output_path, original_durations)

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

    # Per-scene Ken Burns intensity (Tier 1.5, Fix 1.10) — only applies to
    # fallback Ken Burns scenes; AI clips have their own motion. Same
    # ramp+overrides as the image-only assembler:
    #   index n-3 = valley = 0.7x
    #   index n-2 = climax = 1.5x
    #   index n-1 = outro (static, intensity moot)
    def _scene_intensity(i: int) -> float:
        if n <= 1:
            return 1.0
        if n >= 4 and i == n - 3:
            return 0.7
        if n >= 3 and i == n - 2:
            return 1.5
        return 0.8 + 0.6 * (i / (n - 1))

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
            intensity = _scene_intensity(i)
            print(f"    Ken Burns {i+1}/{n} silent ({dur:.2f}s, intensity={intensity:.2f}) — AI clip missing for this scene")
            _make_silent_image_scene_clip(img, silent_path, dur, intensity=intensity)
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

    # Scene-boundary SFX (mood-mapped): sword clang on battle, divine bell on
    # Krishna, war drum on Kurukshetra, conch on vows, chime default. Layered
    # into narration audio BEFORE music mix so SFX gets sidechain ducking too.
    scene_moods = [s.get("mood", "") for s in (script_data.get("scenes") or [])]
    if len(scene_moods) == n:
        _inject_scene_boundary_sfx(output_path, durations, scene_moods)

    _apply_cinematic_polish(output_path, durations)
    _apply_background_music(output_path, series=series)
    _apply_phase3_overlays(output_path)
    _enforce_max_duration(output_path, durations)

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
    _apply_phase3_overlays(output_path)
    _enforce_max_duration(output_path, clip_durations)

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
    _apply_phase3_overlays(output_path)
    _enforce_max_duration(output_path, processed_durations)

    return output_path
