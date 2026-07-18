"""MOTION LAB — Wan 2.2 TI2V-5B image-to-video on a free T4 (heavy hero-shot test).

Tests whether a 5B model can do ARTICULATED fight motion (the thing LTX-2B melts
on) within free-Kaggle constraints: fp16 + model CPU offload + VAE tiling at
832x480. One clip per entry; per-clip SIGALRM watchdog; always exit 0.
Named run_ltx_phase.py so pipeline/wuxia_kaggle.push_wuxia_kernel embeds it as-is.
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

# extra deps beyond the notebook's base pip line
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers", "ftfy", "hf_transfer"], check=False)

import numpy as np
import torch
from PIL import Image


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


def main():
    with open("current_run.json") as f:
        cfg = json.load(f)
    hero = cfg.get("hero_motion", [])
    if not hero:
        print("No hero_motion entries. Exiting.", flush=True)
        return

    W = int(cfg.get("wan_width", 832))
    H = int(cfg.get("wan_height", 480))
    NF = int(cfg.get("wan_frames", 49))
    STEPS = int(cfg.get("wan_steps", 35))
    G = float(cfg.get("wan_guidance", 5.0))
    seed0 = int(cfg.get("master_seed", 42))
    clip_timeout = int(cfg.get("clip_timeout_s", 5400))

    _has_alarm = hasattr(signal, "SIGALRM")
    if _has_alarm:
        def _on_alarm(signum, frame):
            raise TimeoutError(f"clip exceeded {clip_timeout}s watchdog")
        signal.signal(signal.SIGALRM, _on_alarm)

    print("Loading Wan2.2-TI2V-5B (fp16, cpu-offload, vae-tiling)...", flush=True)
    t0 = time.time()
    from diffusers import WanImageToVideoPipeline
    _dt = torch.bfloat16 if cfg.get("wan_dtype", "fp16") == "bf16" else torch.float16
    print(f"dtype={_dt}", flush=True)
    pipe = WanImageToVideoPipeline.from_pretrained(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers", torch_dtype=_dt,
        low_cpu_mem_usage=True,  # meta-device load: no 2x host-RAM spike (SIGKILL fix)
    )
    if cfg.get("wan_vae_fp32"):
        pipe.vae = pipe.vae.to(torch.float32)  # fp32 decode kills fp16 color drift
        print("VAE upcast to fp32", flush=True)
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception as e:
        print(f"vae tiling unavailable: {e}", flush=True)
    print(f"Wan loaded in {time.time()-t0:.0f}s", flush=True)

    failures = 0
    for entry in hero:
        idx = int(entry["idx"])
        scene_idx = int(entry.get("scene_idx", idx))
        shot_idx = int(entry.get("shot_idx", 0))
        out_path = f"/kaggle/working/scene_{scene_idx+1:02d}_shot_{shot_idx+1:02d}.mp4"
        try:
            if _has_alarm:
                signal.alarm(clip_timeout)
            img = (Image.open(BytesIO(base64.b64decode(entry["image_b64"])))
                   .convert("RGB").resize((W, H), Image.BICUBIC))
            gen = torch.Generator("cuda").manual_seed(seed0 + idx)
            print(f"[{idx}] WAN render {W}x{H} {NF}f {STEPS}steps g{G}", flush=True)
            t0 = time.time()
            out = pipe(
                image=img,
                prompt=entry.get("prompt", ""),
                negative_prompt=entry.get("negative", "blurry, low quality, deformed, melting, warping, extra limbs, mutated"),
                height=H, width=W,
                num_frames=NF,
                num_inference_steps=STEPS,
                guidance_scale=G,
                generator=gen,
            ).frames[0]
            print(f"  gen {time.time()-t0:.0f}s", flush=True)
            arr = (np.asarray(out) * 255).clip(0, 255).astype("uint8") \
                if not isinstance(out[0], Image.Image) \
                else np.stack([np.asarray(f) for f in out]).astype("uint8")
            _write_mp4(arr, out_path)
            del out, arr
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            import traceback
            print(f"CLIP FAILED [{idx}]:\n" + traceback.format_exc(), flush=True)
            failures += 1
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            if _has_alarm:
                signal.alarm(0)
    if failures:
        print(f"WAN_PHASE_PARTIAL: {failures} clip(s) failed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        print("RUN_WAN_PHASE FAILED (non-fatal):\n" + traceback.format_exc(), flush=True)
    sys.exit(0)
