"""MOTION LAB — CogVideoX 1.5-5B I2V on a free T4 via SEQUENTIAL cpu offload.

Our earlier CogVideoX OOM used model offload; sequential offload runs in ~5GB
VRAM (very slow — hero shots only). Tests articulated fight motion vs LTX melt.
Named run_ltx_phase.py so push_wuxia_kernel embeds it as-is.
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "transformers", "ftfy", "sentencepiece"], check=False)

import numpy as np
import torch
from PIL import Image


def _write_mp4(frames_uint8, out_path, fps=16):
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

    W = int(cfg.get("cog_width", 720))
    H = int(cfg.get("cog_height", 480))
    NF = int(cfg.get("cog_frames", 49))
    STEPS = int(cfg.get("cog_steps", 30))
    G = float(cfg.get("cog_guidance", 6.0))
    seed0 = int(cfg.get("master_seed", 42))
    clip_timeout = int(cfg.get("clip_timeout_s", 5400))

    _has_alarm = hasattr(signal, "SIGALRM")
    if _has_alarm:
        def _on_alarm(signum, frame):
            raise TimeoutError(f"clip exceeded {clip_timeout}s watchdog")
        signal.signal(signal.SIGALRM, _on_alarm)

    print("Loading CogVideoX1.5-5B-I2V (fp16, SEQUENTIAL offload, tiling)...", flush=True)
    t0 = time.time()
    from diffusers import CogVideoXImageToVideoPipeline
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX1.5-5B-I2V", torch_dtype=torch.float16
    )
    pipe.enable_sequential_cpu_offload()
    try:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    except Exception as e:
        print(f"vae tiling/slicing unavailable: {e}", flush=True)
    print(f"Cog loaded in {time.time()-t0:.0f}s", flush=True)

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
            print(f"[{idx}] COG render {W}x{H} {NF}f {STEPS}steps g{G}", flush=True)
            t0 = time.time()
            out = pipe(
                image=img,
                prompt=entry.get("prompt", ""),
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
        print(f"COG_PHASE_PARTIAL: {failures} clip(s) failed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        print("RUN_COG_PHASE FAILED (non-fatal):\n" + traceback.format_exc(), flush=True)
    sys.exit(0)
