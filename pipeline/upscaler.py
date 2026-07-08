"""Real-ESRGAN frame upscaling (2026-07-05, user request: "more detailed
images").

Runs the ncnn-vulkan Real-ESRGAN binary on every generated scene image
BEFORE video assembly: 896x1600 -> 1792x3200 with synthesized texture detail
(hair strands, skin, fabric weave, metal engraving), which the assembler then
samples down to 1080x1920 — a supersampled, visibly crisper final video.

Model: realesr-animevideov3-x2. Benchmarked 2026-07-05 on the render machine
(Intel UHD 620 vulkan): 10.6s/frame vs 4m25s for realesrgan-x4plus, with
near-identical quality once downscaled into the video. Despite the "anime"
name it is a general video-enhance model and looks excellent on the
photoreal Mahabharata frames (side-by-side eye-crop verified).

Design rules:
  - GRACEFUL: any failure (missing binary, no vulkan device, timeout) logs
    and returns the ORIGINAL paths — a render is never lost to the upscaler.
  - Env-gated: UPSCALE_FRAMES=false disables entirely (default ON).
  - Idempotent: frames taller than _ALREADY_UPSCALED_MIN_H are skipped, so
    checkpoint-resumed runs don't double-upscale.
  - First-frame probe: if frame 1 exceeds the per-frame timeout the rest are
    skipped (protects slow software-vulkan environments like GHA llvmpipe).
"""
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

_MODEL = "realesr-animevideov3"
_SCALE = 2
_ALREADY_UPSCALED_MIN_H = 2000     # skip frames already >= this height
_PER_FRAME_TIMEOUT_S = int(os.environ.get("UPSCALE_FRAME_TIMEOUT_S", "150"))

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "upscaler")
_RELEASE = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.2.5.0/realesrgan-ncnn-vulkan-20220424-{plat}.zip")


def _binary_path() -> str | None:
    """Locate (or on Linux, download) the platform binary. None on failure."""
    if platform.system() == "Windows":
        exe = os.path.join(_BASE_DIR, "ncnn-win", "realesrgan-ncnn-vulkan.exe")
        return exe if os.path.exists(exe) else None
    # Linux (GHA runner): download once per runner into the repo tree.
    d = os.path.join(_BASE_DIR, "ncnn-linux")
    exe = os.path.join(d, "realesrgan-ncnn-vulkan")
    if os.path.exists(exe):
        return exe
    try:
        os.makedirs(d, exist_ok=True)
        zpath = os.path.join(d, "rr.zip")
        url = _RELEASE.format(plat="ubuntu")
        print(f"    [upscale] downloading binary ({url.rsplit('/',1)[-1]})...")
        urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(d)
        os.remove(zpath)
        os.chmod(exe, 0o755)
        return exe if os.path.exists(exe) else None
    except Exception as e:
        print(f"    [upscale] binary setup failed ({str(e)[:100]}) — skipping")
        return None


def _frame_height(path: str) -> int:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size[1]
    except Exception:
        return 0


def _flatten(items):
    """Yield string paths from arbitrarily nested lists/tuples. The
    Mahabharata Phase-23 multi-shot format returns [[shot, shot, shot], ...]
    per scene (caught 2026-07-06: TypeError crashed the first proof render at
    the upscale hook); WhatIf returns a flat list. Handle both."""
    for it in items or []:
        if isinstance(it, (list, tuple)):
            yield from _flatten(it)
        elif isinstance(it, (str, os.PathLike)) and it:
            yield str(it)


def upscale_images(image_paths: list, label: str = "") -> list:
    """Upscale each image 2x IN PLACE (original file overwritten with the
    upscaled JPEG). Accepts flat or nested path lists; returns the input
    structure unchanged (files are modified in place) for drop-in use:
        image_files = upscale_images(image_files)
    """
    if os.environ.get("UPSCALE_FRAMES", "true").strip().lower() in ("false", "0", "no"):
        return image_paths
    todo = [p for p in _flatten(image_paths)
            if os.path.exists(p)
            and _frame_height(p) < _ALREADY_UPSCALED_MIN_H]
    if not todo:
        return image_paths
    exe = _binary_path()
    if not exe:
        print("    [upscale] no binary available — using original frames")
        return image_paths

    from PIL import Image
    t_all = time.time()
    done = 0
    for i, p in enumerate(todo):
        out_png = p + ".up.png"
        t0 = time.time()
        try:
            r = subprocess.run(
                # abspath both sides (2026-07-08 fix): the pipeline passes
                # RELATIVE temp/ paths and we run with cwd=binary-dir so the
                # model folder resolves — the binary then couldn't find the
                # input, silently wrote nothing, and exited rc=0. This was
                # the entire "GHA probe fails" mystery (llvmpipe was fine).
                [exe, "-i", os.path.abspath(p), "-o", os.path.abspath(out_png),
                 "-n", f"{_MODEL}-x{_SCALE}", "-s", str(_SCALE)],
                capture_output=True, timeout=_PER_FRAME_TIMEOUT_S,
                cwd=os.path.dirname(exe),
            )
            ok = (r.returncode == 0 and os.path.exists(out_png)
                  and os.path.getsize(out_png) > 0)
            if not ok and i == 0:
                # Surface WHY the probe failed (GHA has failed the
                # probe on every run; stderr was swallowed until now).
                _err = (r.stderr or b'').decode(errors='replace')[-400:]
                print(f"    [upscale] probe rc={r.returncode} "
                      f"stderr: {_err.strip()[:300]}", flush=True)
        except subprocess.TimeoutExpired:
            ok = False
            print(f"    [upscale] frame {i+1}/{len(todo)} exceeded "
                  f"{_PER_FRAME_TIMEOUT_S}s — skipping remaining frames "
                  f"(slow/software vulkan)")
        except Exception as e:
            ok = False
            print(f"    [upscale] frame {i+1} failed: {str(e)[:100]}")

        if not ok:
            try:
                os.path.exists(out_png) and os.remove(out_png)
            except OSError:
                pass
            if i == 0:
                # First-frame probe failed → environment can't do this fast
                # enough / at all. Don't burn time on the rest.
                print("    [upscale] first-frame probe failed — originals kept")
                return image_paths
            continue

        # Re-encode PNG -> JPEG over the original path (keeps every
        # downstream path/checkpoint reference valid).
        try:
            with Image.open(out_png) as im:
                im.convert("RGB").save(p, "JPEG", quality=92)
            os.remove(out_png)
            done += 1
            if i == 0:
                print(f"    [upscale] {_MODEL}-x{_SCALE}: frame 1 in "
                      f"{time.time()-t0:.1f}s — continuing batch")
        except Exception as e:
            print(f"    [upscale] re-encode failed: {str(e)[:100]}")
            try:
                os.remove(out_png)
            except OSError:
                pass

    if done:
        print(f"    [upscale] {done}/{len(todo)} frames supersampled 2x "
              f"in {time.time()-t_all:.0f}s{' (' + label + ')' if label else ''}")
    return image_paths
