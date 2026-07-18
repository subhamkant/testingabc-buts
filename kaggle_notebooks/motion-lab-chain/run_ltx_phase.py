"""VEO-LITE CHAIN — 5-10s Wan2.2-5B fight sequences on a free T4.

4-stage pipeline in ONE kernel run (per the approved blueprint):
  1. Wan2.2-TI2V-5B fp16 (low_cpu_mem_usage, offload, tiling) @ 832x480.
  2. AUTOREGRESSIVE CHAIN: N chunks x 49f; each chunk conditions on the previous
     chunk's LAST frame; seed varies +1 per chunk; first frame of chunks 2+ is
     dropped at stitch (it duplicates the previous last frame).
  3. TEMPORAL INTERPOLATION: ffmpeg minterpolate 24 -> 48 fps (optical-flow MCI)
     at 480p (cheap on CPU) — kills micro-stutter.
  4. SPATIAL UPSCALE: anime Real-ESRGAN x4 per frame (spandrel) -> 1920x1080,
     CRF 17 slow yuv420p final.
VRAM hygiene: Wan pipe is deleted + cache emptied BEFORE ESRGAN loads.
Named run_ltx_phase.py so push_wuxia_kernel embeds it unchanged.
"""
import base64
import gc
import json
import os
import signal
import subprocess
import sys
import time
from io import BytesIO

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# HF download hardening: two runs stalled mid-download for 87-100 min. The
# rust downloader (hf_transfer) + a per-request timeout make fetches fast
# and fail-fast (huggingface_hub auto-resumes on retry).
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers", "ftfy", "hf_transfer"], check=False)

import numpy as np
import torch
from PIL import Image

_ESR_URLS = {
    "anime": ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
              "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"),
    "photo": ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
              "v0.1.0/RealESRGAN_x4plus.pth"),
}
_ESR_PATH = "/kaggle/working/esrgan.pth"


def _write_mp4(frames_uint8, out_path, fps=24):
    import imageio
    imageio.mimwrite(
        out_path, list(frames_uint8), fps=fps, codec="libx264",
        macro_block_size=None,
        output_params=["-crf", "17", "-preset", "slow",
                       "-pix_fmt", "yuv420p", "-profile:v", "high"],
    )
    h, w = frames_uint8[0].shape[0], frames_uint8[0].shape[1]
    print(f"SAVED {out_path} ({len(frames_uint8)}f @ {w}x{h})", flush=True)


def _load_esrgan(kind="photo"):
    import urllib.request
    if not os.path.exists(_ESR_PATH):
        print(f"Downloading Real-ESRGAN weights ({kind})...", flush=True)
        urllib.request.urlretrieve(_ESR_URLS.get(kind, _ESR_URLS["photo"]), _ESR_PATH)
    from spandrel import ModelLoader
    return ModelLoader().load_from_file(_ESR_PATH).cuda().eval()


def _esrgan_frames(esr, frames_np, out_w, out_h):
    import torch.nn.functional as Fnn
    outs = []
    for i in range(frames_np.shape[0]):
        t = torch.from_numpy(frames_np[i]).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        with torch.no_grad():
            o = esr(t)
            o = Fnn.interpolate(o, size=(out_h, out_w), mode="bicubic", align_corners=False)
        outs.append((o.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        del t, o
        torch.cuda.empty_cache()
        if i % 50 == 0:
            print(f"  esrgan frame {i}/{frames_np.shape[0]}", flush=True)
    return outs


def _sharpness(img):
    """Gradient-variance sharpness (no scipy needed)."""
    g = img.mean(axis=2)
    gy, gx = np.gradient(g)
    return float((gx * gx + gy * gy).var())


def main():
    with open("current_run.json") as f:
        cfg = json.load(f)
    hero = cfg.get("hero_motion", [])
    if not hero:
        print("No hero_motion entries. Exiting.", flush=True)
        return
    entry = hero[0]  # chain mode: ONE sequence per kernel run

    W = int(cfg.get("wan_width", 832))
    H = int(cfg.get("wan_height", 480))
    NF = int(cfg.get("chunk_frames", 49))
    CHUNKS = int(cfg.get("chain_chunks", 3))
    STEPS = int(cfg.get("wan_steps", 35))
    G = float(cfg.get("wan_guidance", 5.0))
    seed0 = int(cfg.get("master_seed", 104))
    clip_timeout = int(cfg.get("clip_timeout_s", 1200))  # per-chunk watchdog
    out_w = int(cfg.get("out_w", 1920))
    out_h = int(cfg.get("out_h", 1080))
    target_fps = int(cfg.get("interp_fps", 48))

    _has_alarm = hasattr(signal, "SIGALRM")
    if _has_alarm:
        def _on_alarm(signum, frame):
            raise TimeoutError(f"stage exceeded {clip_timeout}s watchdog")
        signal.signal(signal.SIGALRM, _on_alarm)

    # ── Stage 1: load Wan ────────────────────────────────────────────────
    print("Loading Wan2.2-TI2V-5B (fp16, low_cpu_mem, offload, tiling)...", flush=True)
    t0 = time.time()
    from diffusers import WanImageToVideoPipeline
    pipe = WanImageToVideoPipeline.from_pretrained(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception as e:
        print(f"vae tiling unavailable: {e}", flush=True)
    print(f"Wan loaded in {time.time()-t0:.0f}s", flush=True)

    # ── Stage 2: autoregressive chunk-and-chain ─────────────────────────
    cond = (Image.open(BytesIO(base64.b64decode(entry["image_b64"])))
            .convert("RGB").resize((W, H), Image.BICUBIC))
    # Per-chunk story beats: same-prompt chaining makes later chunks go static
    # (model believes the action already happened). chunk_prompts progresses
    # the choreography chunk by chunk.
    chunk_prompts = cfg.get("chunk_prompts") or []
    prompt = entry.get("prompt", "")
    neg = entry.get("negative",
                    "blurry, low quality, deformed, melting, warping, extra limbs, mutated")
    all_frames = []
    ok_chunks = 0
    for c in range(CHUNKS):
        try:
            if _has_alarm:
                signal.alarm(clip_timeout)
            gen = torch.Generator("cuda").manual_seed(seed0 + c)
            c_prompt = chunk_prompts[c] if c < len(chunk_prompts) else prompt
            print(f"[chunk {c+1}/{CHUNKS}] {W}x{H} {NF}f {STEPS}steps g{G} seed={seed0+c}", flush=True)
            t0 = time.time()
            out = pipe(image=cond, prompt=c_prompt, negative_prompt=neg,
                       height=H, width=W, num_frames=NF,
                       num_inference_steps=STEPS, guidance_scale=G,
                       generator=gen).frames[0]
            print(f"  gen {time.time()-t0:.0f}s", flush=True)
            if isinstance(out[0], Image.Image):
                arr = np.stack([np.asarray(f) for f in out]).astype("uint8")
            else:
                arr = (np.asarray(out) * 255).clip(0, 255).astype("uint8")
            _write_mp4(arr, f"/kaggle/working/chunk_{c+1:02d}.mp4")
            # chain: condition the next chunk on the SHARPEST of the last 8
            # frames (seeding from a motion-blurred frame propagates blur).
            tail = arr[-8:]
            sharp_i = int(np.argmax([_sharpness(f) for f in tail]))
            cond = Image.fromarray(tail[sharp_i])
            print(f"  next-chunk seed frame: tail[{sharp_i}] "
                  f"(sharpness={_sharpness(tail[sharp_i]):.0f})", flush=True)
            # drop duplicate first frame on chunks 2+
            all_frames.append(arr if c == 0 else arr[1:])
            ok_chunks += 1
            del out
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            import traceback
            print(f"CHUNK FAILED [{c}]:\n" + traceback.format_exc(), flush=True)
            break
        finally:
            if _has_alarm:
                signal.alarm(0)
    if not all_frames:
        print("NO CHUNKS SUCCEEDED", flush=True)
        return
    full = np.concatenate(all_frames, axis=0)
    print(f"chained {ok_chunks} chunk(s) -> {full.shape[0]} frames "
          f"({full.shape[0]/24:.2f}s @24fps)", flush=True)
    _write_mp4(full, "/kaggle/working/chain_raw24.mp4")

    # free Wan BEFORE post-processing
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # ── Stage 3: temporal interpolation 24 -> target_fps (CPU, 480p) ────
    try:
        if _has_alarm:
            signal.alarm(3000)
        t0 = time.time()
        r = subprocess.run([
            "ffmpeg", "-y", "-i", "/kaggle/working/chain_raw24.mp4",
            "-vf", f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
            "-c:v", "libx264", "-crf", "15", "-preset", "fast", "-pix_fmt", "yuv420p",
            "/kaggle/working/chain_interp.mp4"], capture_output=True)
        print(f"interpolation {'OK' if r.returncode == 0 else 'FAILED'} "
              f"({time.time()-t0:.0f}s)", flush=True)
        interp_src = ("/kaggle/working/chain_interp.mp4" if r.returncode == 0
                      else "/kaggle/working/chain_raw24.mp4")
        out_fps = target_fps if r.returncode == 0 else 24
    finally:
        if _has_alarm:
            signal.alarm(0)

    # ── Stage 4: ESRGAN x4 -> 1080p final ────────────────────────────────
    try:
        if _has_alarm:
            signal.alarm(3600)
        import imageio
        frames = np.stack([f for f in imageio.mimread(interp_src, memtest=False)])
        print(f"upscaling {frames.shape[0]} frames to {out_w}x{out_h}...", flush=True)
        esr = _load_esrgan(cfg.get("esr_model", "photo"))
        up = _esrgan_frames(esr, frames, out_w, out_h)
        _write_mp4(up, "/kaggle/working/scene_01_shot_01.mp4", fps=out_fps)
        del esr, frames, up
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        import traceback
        print("UPSCALE FAILED (raw kept):\n" + traceback.format_exc(), flush=True)
    finally:
        if _has_alarm:
            signal.alarm(0)
    print("CHAIN PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        print("RUN_CHAIN FAILED (non-fatal):\n" + traceback.format_exc(), flush=True)
    sys.exit(0)
