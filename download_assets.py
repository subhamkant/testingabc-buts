"""
Download free cinematic assets — LUTs (.cube) + light-leak / lens-flare MP4 overlays.
======================================================================================
These are used by pipeline/video_assembler.py's polish layer to add filmic color
grading, light-leak transitions, and lens-flare accents.

The polish layer works WITHOUT these files (it falls back to native FFmpeg filters),
but real LUTs and overlays produce a noticeably more cinematic look.

Run once:
    python download_assets.py

Layout:
    assets/luts/*.cube                  — 3D LUT files
    assets/overlays/lightleaks/*.mp4    — short light-leak loops (~3-6s)
    assets/overlays/lensflares/*.mp4    — short lens-flare loops (~2-3s)

All URLs below point to CC0 / royalty-free / free-for-commercial-use sources.
URLs occasionally rot — the script keeps going on individual failures.
"""

import os
import requests

LUT_DIR        = "assets/luts"
LIGHTLEAK_DIR  = "assets/overlays/lightleaks"
LENSFLARE_DIR  = "assets/overlays/lensflares"


# .cube LUT files (small text files, very stable URLs).
# We also synthesize a teal-orange .cube ourselves below so the script always
# produces at least one working LUT even if every URL rots.
LUT_URLS = [
    # CC0 cinematic LUT collections from public GitHub repos
    ("https://raw.githubusercontent.com/aras-p/smol-cube/main/test/luts/Arri_LogC2Video_709.cube", "logc_to_rec709.cube"),
]

# Royalty-free overlays. URLs from Pixabay/Mixkit need User-Agent header.
LIGHTLEAK_URLS = [
    # Pixabay free videos — direct CDN links sometimes work; fallback to synth if not.
    ("https://cdn.pixabay.com/video/2020/05/29/40456-425554927_tiny.mp4",         "lightleak_warm.mp4"),
    ("https://cdn.pixabay.com/video/2023/06/26/167991-840701620_tiny.mp4",        "lightleak_orange.mp4"),
]

LENSFLARE_URLS = [
    ("https://cdn.pixabay.com/video/2017/12/20/13497-247711337_tiny.mp4",         "lensflare_anamorphic.mp4"),
]


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    )
}


def _download(url: str, dest: str, min_bytes: int = 4_000) -> bool:
    try:
        print(f"  -> {os.path.basename(dest)} ({url[:70]}...)")
        r = requests.get(url, headers=_HEADERS, timeout=60, stream=True)
        if r.status_code != 200:
            print(f"     HTTP {r.status_code}")
            return False
        content = r.content
        if len(content) < min_bytes:
            print(f"     too small ({len(content)} bytes)")
            return False
        with open(dest, "wb") as f:
            f.write(content)
        size_kb = len(content) // 1024
        print(f"     OK ({size_kb} KB)")
        return True
    except Exception as e:
        print(f"     error: {e}")
        return False


def _write_synthetic_teal_orange_lut(path: str) -> None:
    """
    Writes a 17-point 3D LUT that approximates a Teal & Orange filmic look.
    Shadows pushed toward teal (cyan-blue), highlights pushed toward orange.
    This is our guaranteed-always-available LUT.
    """
    size = 17
    lines = [
        "# Synthetic Teal & Orange LUT — Mahabharata Bot",
        "TITLE \"Teal Orange Synth\"",
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]

    def teal_orange(c: float, channel: str) -> float:
        # Shadows -> teal: lift B and a touch of G in low values, crush R in shadows.
        # Highlights -> orange: boost R and G in highs, attenuate B.
        # c is [0,1].
        if channel == "r":
            # crush shadows, boost highlights
            return max(0.0, min(1.0, (c ** 1.10) * 1.05 + 0.02 * (c - 0.5) * 2))
        if channel == "g":
            return max(0.0, min(1.0, (c ** 1.00) * 1.02))
        # blue
        # boost shadows, slightly cut highlights
        shadow_lift = 0.04 * (1.0 - c)
        highlight_cut = 1.0 - 0.10 * (c ** 2)
        return max(0.0, min(1.0, c * highlight_cut + shadow_lift))

    for b in range(size):
        for g in range(size):
            for r in range(size):
                rn = r / (size - 1)
                gn = g / (size - 1)
                bn = b / (size - 1)
                rr = teal_orange(rn, "r")
                gg = teal_orange(gn, "g")
                bb = teal_orange(bn, "b")
                lines.append(f"{rr:.6f} {gg:.6f} {bb:.6f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_synthetic_warm_filmic_lut(path: str) -> None:
    """A second LUT — warm filmic (less aggressive than teal/orange)."""
    size = 17
    lines = [
        "# Synthetic Warm Filmic LUT — Mahabharata Bot",
        "TITLE \"Warm Filmic Synth\"",
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]

    def warm(c: float, channel: str) -> float:
        if channel == "r":
            return max(0.0, min(1.0, c * 1.05 + 0.02))
        if channel == "g":
            return max(0.0, min(1.0, c * 1.00 + 0.005))
        return max(0.0, min(1.0, c * 0.92))

    for b in range(size):
        for g in range(size):
            for r in range(size):
                rn = r / (size - 1)
                gn = g / (size - 1)
                bn = b / (size - 1)
                lines.append(
                    f"{warm(rn,'r'):.6f} {warm(gn,'g'):.6f} {warm(bn,'b'):.6f}"
                )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _synthesize_overlay(path: str, kind: str = "lightleak") -> bool:
    """
    Generates a 4-second overlay MP4 using two lavfi color sources + fade.
    The result is a black canvas with a soft warm flash fading in and out;
    looped via -stream_loop, this gives the polish layer a periodic warm
    pulse to screen-blend over scenes.

    These synth overlays are last-resort fallbacks — real CC0 downloads from
    Pixabay / Mixkit look much better.
    """
    import subprocess

    if kind == "lightleak":
        flash_color = "#ffaa55"     # warm amber
        flash_height = "1920"
    else:
        flash_color = "#fff0c0"     # white-yellow flare
        flash_height = "1920"

    filter_complex = (
        f"[1:v]fade=t=in:st=0:d=1.2,fade=t=out:st=2.2:d=1.6[flash];"
        f"[0:v][flash]overlay=0:0:format=auto,format=yuv420p[vout]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=#000000:size=1080x1920:duration=4:rate=30",
        "-f", "lavfi", "-i", f"color=c={flash_color}:size=1080x{flash_height}:duration=4:rate=30",
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-t", "4",
        "-c:v", "libx264", "-preset", "fast", "-crf", "26",
        "-pix_fmt", "yuv420p",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[:300] if result.stderr else ""
        print(f"     synth FFmpeg error: {err}")
        return False
    return os.path.exists(path) and os.path.getsize(path) > 5_000


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(LUT_DIR, exist_ok=True)
    os.makedirs(LIGHTLEAK_DIR, exist_ok=True)
    os.makedirs(LENSFLARE_DIR, exist_ok=True)

    print("=" * 60)
    print("  Cinematic Asset Downloader")
    print("=" * 60)

    # ── LUTs ────────────────────────────────────────────────────────────────
    print("\n[1/3] LUTs (.cube files)")
    for url, name in LUT_URLS:
        dest = os.path.join(LUT_DIR, name)
        if os.path.exists(dest):
            print(f"  - {name} (already present)")
            continue
        _download(url, dest, min_bytes=500)

    # Always generate the synthetic LUTs — guarantees the polish layer has
    # at least 2 working LUTs even if every URL above rotted.
    synth_to = os.path.join(LUT_DIR, "synth_teal_orange.cube")
    if not os.path.exists(synth_to):
        print("  - synth_teal_orange.cube (synthesizing...)")
        _write_synthetic_teal_orange_lut(synth_to)
    synth_warm = os.path.join(LUT_DIR, "synth_warm_filmic.cube")
    if not os.path.exists(synth_warm):
        print("  - synth_warm_filmic.cube (synthesizing...)")
        _write_synthetic_warm_filmic_lut(synth_warm)

    # ── Light leaks ─────────────────────────────────────────────────────────
    print("\n[2/3] Light-leak overlays")
    any_leak = False
    for url, name in LIGHTLEAK_URLS:
        dest = os.path.join(LIGHTLEAK_DIR, name)
        if os.path.exists(dest):
            print(f"  - {name} (already present)")
            any_leak = True
            continue
        if _download(url, dest, min_bytes=20_000):
            any_leak = True

    if not any_leak:
        synth = os.path.join(LIGHTLEAK_DIR, "synth_warm_sweep.mp4")
        if not os.path.exists(synth):
            print("  - synth_warm_sweep.mp4 (synthesizing via FFmpeg...)")
            if _synthesize_overlay(synth, "lightleak"):
                print("    [OK] synth light-leak created")
            else:
                print("    [!] synth failed — polish layer will skip light leaks")

    # ── Lens flares ─────────────────────────────────────────────────────────
    print("\n[3/3] Lens-flare overlays")
    any_flare = False
    for url, name in LENSFLARE_URLS:
        dest = os.path.join(LENSFLARE_DIR, name)
        if os.path.exists(dest):
            print(f"  - {name} (already present)")
            any_flare = True
            continue
        if _download(url, dest, min_bytes=20_000):
            any_flare = True

    if not any_flare:
        synth = os.path.join(LENSFLARE_DIR, "synth_flare.mp4")
        if not os.path.exists(synth):
            print("  - synth_flare.mp4 (synthesizing via FFmpeg...)")
            if _synthesize_overlay(synth, "lensflare"):
                print("    [OK] synth lens-flare created")
            else:
                print("    [!] synth failed — polish layer will skip lens flares")

    # ── Summary ─────────────────────────────────────────────────────────────
    luts    = sorted(os.listdir(LUT_DIR))         if os.path.isdir(LUT_DIR)        else []
    leaks   = sorted(os.listdir(LIGHTLEAK_DIR))   if os.path.isdir(LIGHTLEAK_DIR)  else []
    flares  = sorted(os.listdir(LENSFLARE_DIR))   if os.path.isdir(LENSFLARE_DIR)  else []

    print("\n" + "=" * 60)
    print(f"LUTs       : {len(luts)}")
    for n in luts:   print(f"   - {n}")
    print(f"Light leaks: {len(leaks)}")
    for n in leaks:  print(f"   - {n}")
    print(f"Lens flares: {len(flares)}")
    for n in flares: print(f"   - {n}")
    print("=" * 60)


if __name__ == "__main__":
    main()
