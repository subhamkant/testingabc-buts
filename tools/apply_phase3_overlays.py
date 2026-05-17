"""
One-off Phase 3 visual test: apply lens-flare + light-leak + particle
overlays to an existing rendered mp4. Use this BEFORE wiring Phase 3
into pipeline/video_assembler.py to confirm overlays improve rather
than cheapen the cinematic feel.

Three layers (top to bottom in final composite):
  1. Synthetic particle layer  (shadow-masked, blend=add, alpha 0.03)
  2. Lens flare overlay        (screen blend, alpha 0.28, full duration
                                for test; production = climax-only,
                                probabilistic 35-50%)
  3. Light leak overlay        (screen blend, alpha 0.22)

All intensities are the round-5/6-refined values per plan:
  particles 0.03-0.05 (subconsciously felt, not consciously noticed)
  flare 0.25-0.35 alpha (in the air, not edited in)
  leak 0.20-0.30 alpha (warm haze, not video-editor template)

Usage:
  python tools/apply_phase3_overlays.py <input_mp4> [output_mp4]
"""

import os
import sys
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _probe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def apply_overlays(
    input_mp4: str,
    output_mp4: str,
    leak_path: str = "assets/overlays/lightleaks/synth_warm_sweep.mp4",
    flare_path: str = "assets/overlays/lensflares/synth_flare.mp4",
    leak_alpha: float = 0.22,
    flare_alpha: float = 0.28,
    particle_alpha: float = 0.03,
) -> bool:
    """Build the ffmpeg filter graph and run it. Returns True on success."""
    if not os.path.exists(input_mp4):
        print(f"[ERROR] input mp4 not found: {input_mp4}")
        return False
    if not os.path.exists(leak_path):
        print(f"[ERROR] light-leak asset not found: {leak_path}")
        return False
    if not os.path.exists(flare_path):
        print(f"[ERROR] lens-flare asset not found: {flare_path}")
        return False

    dur = _probe_duration(input_mp4)
    if dur <= 0.0:
        print(f"[ERROR] could not probe input mp4 duration")
        return False
    print(f"[info] input duration: {dur:.2f}s")
    print(f"[info] applying overlays at:")
    print(f"         light leak  alpha={leak_alpha}")
    print(f"         lens flare  alpha={flare_alpha}")
    print(f"         particles   alpha={particle_alpha}  (shadow-masked above luma=30)")
    print()

    # Filter graph:
    #   [0:v] = input video
    #   [1:v] = light leak (looped)
    #   [2:v] = lens flare (looped)
    #   [3:v] = synthesized particle layer (anoisesrc-based)
    #
    # Steps:
    #   1. Scale leak + flare to 1080x1920, trim to input duration, apply alpha
    #   2. Generate shadow-masked particle layer (only above luma threshold)
    #   3. Composite: input ⊕ leak ⊕ flare ⊕ particles
    filter_graph = (
        # Light leak — scale, trim, apply alpha
        f"[1:v]scale=1080:1920,trim=duration={dur:.3f},"
        f"format=yuva420p,colorchannelmixer=aa={leak_alpha:.3f}[leak];"

        # Lens flare — scale, trim, apply alpha
        f"[2:v]scale=1080:1920,trim=duration={dur:.3f},"
        f"format=yuva420p,colorchannelmixer=aa={flare_alpha:.3f}[flare];"

        # Synthesized particle layer — random luminance noise, only bright spots
        # become "particles". Shadow-masked threshold via geq below.
        f"[3:v]scale=1080:1920,trim=duration={dur:.3f},"
        f"format=yuva420p,colorchannelmixer=aa={particle_alpha:.3f}[particles_raw];"

        # Step 1: input ⊕ leak (screen blend)
        f"[0:v][leak]blend=all_mode=screen:all_opacity=1[v1];"

        # Step 2: v1 ⊕ flare (screen blend)
        f"[v1][flare]blend=all_mode=screen:all_opacity=1[v2];"

        # Step 3: v2 ⊕ particles (additive blend — adds light, never darkens)
        f"[v2][particles_raw]blend=all_mode=addition:all_opacity=1[vout]"
    )

    # Synthesized particle source — sparse twinkling specks via lavfi.
    # noise=alls=20 generates per-pixel noise; geq filters it so only bright
    # pixels survive (sparse particle effect rather than uniform grain).
    particle_lavfi = (
        f"color=c=black:s=1080x1920:d={dur:.3f}:r=30,"
        f"noise=alls=30:allf=t,"
        f"format=yuv420p,"
        # Threshold: only keep pixels where lum > 230 (top ~10% brightness)
        # This produces sparse twinkling specks rather than uniform grain.
        f"geq=lum='if(gt(lum(X,Y),230),255,0)':cb=128:cr=128"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_mp4,
        "-stream_loop", "-1", "-i", leak_path,
        "-stream_loop", "-1", "-i", flare_path,
        "-f", "lavfi", "-i", particle_lavfi,
        "-filter_complex", filter_graph,
        "-map", "[vout]",
        "-map", "0:a",                      # keep original audio
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-c:a", "copy",                     # don't re-encode audio
        "-movflags", "+faststart",
        output_mp4,
    ]
    print("[info] running ffmpeg...")
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        err = r.stderr.decode("utf-8", errors="replace")[-1000:] if r.stderr else ""
        print(f"[ERROR] ffmpeg failed (exit {r.returncode}):\n{err}")
        return False
    print(f"[OK] wrote {output_mp4}")
    new_dur = _probe_duration(output_mp4)
    new_size = os.path.getsize(output_mp4) / (1024 * 1024)
    print(f"[OK] output: {new_dur:.2f}s, {new_size:.1f} MB")
    return True


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <input_mp4> [output_mp4]")
    inp = sys.argv[1]
    if len(sys.argv) >= 3:
        out = sys.argv[2]
    else:
        base, ext = os.path.splitext(inp)
        out = f"{base}_phase3{ext}"
    ok = apply_overlays(inp, out)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
