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


_ESR_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.1.0/RealESRGAN_x4plus.pth")
_ESR_PATH = "/kaggle/working/esrgan.pth"
_ESR_MODEL = None


def _esrgan_up(frames_np, out_w, out_h):
    """2x-ish detail restore on the T4 before the mp4 ever leaves Kaggle.
    x4plus then bicubic down to (out_w, out_h). Lazy-loads spandrel + weights."""
    global _ESR_MODEL
    import torch.nn.functional as Fnn
    if _ESR_MODEL is None:
        try:
            from spandrel import ModelLoader
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "spandrel"], check=True)
            from spandrel import ModelLoader
        if not os.path.exists(_ESR_PATH):
            import urllib.request
            print("Downloading Real-ESRGAN weights...", flush=True)
            urllib.request.urlretrieve(_ESR_URL, _ESR_PATH)
        _ESR_MODEL = ModelLoader().load_from_file(_ESR_PATH).cuda().eval()
    outs = []
    for i in range(frames_np.shape[0]):
        t = (torch.from_numpy(frames_np[i]).permute(2, 0, 1).unsqueeze(0)
             .float().cuda() / 255.0)
        with torch.no_grad():
            o = _ESR_MODEL(t)
            o = Fnn.interpolate(o, size=(out_h, out_w), mode="bicubic",
                                align_corners=False)
        outs.append((o.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                     * 255).astype("uint8"))
        del t, o
        torch.cuda.empty_cache()
        if i % 16 == 0:
            print(f"  esrgan {i}/{frames_np.shape[0]}", flush=True)
    return np.stack(outs)


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

    # esrgan_only: finishing pass — no Wan load. Clips arrive as b64 in the
    # run config, get ESRGAN-upscaled 2x, and land in /kaggle/working.
    if cfg.get("esrgan_only"):
        import imageio
        for item in cfg.get("esrgan_clips", []):
            name = item["name"]
            raw = base64.b64decode(item["mp4_b64"])
            src = f"/kaggle/working/_in_{name}"
            with open(src, "wb") as fh:
                fh.write(raw)
            frames = np.stack([np.asarray(f) for f in
                               imageio.mimread(src, memtest=False)])
            h, w = frames.shape[1], frames.shape[2]
            print(f"{name}: {frames.shape[0]}f {w}x{h} -> {w*2}x{h*2}",
                  flush=True)
            t0 = time.time()
            up = _esrgan_up(frames, w * 2, h * 2)
            fps = float(item.get("fps", 16))
            _write_mp4(up, f"/kaggle/working/{name}", fps=fps)
            print(f"  done {time.time()-t0:.0f}s", flush=True)
            os.remove(src)
            del frames, up
            gc.collect()
            torch.cuda.empty_cache()
        print("ESRGAN_ONLY_COMPLETE", flush=True)
        return

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

    # flow_shift = the motion-amplitude knob (UniPC scheduler). Default config
    # value is conservative; higher shift => stronger scene motion. Per-entry
    # override lets one run sweep it.
    from diffusers import UniPCMultistepScheduler
    _base_sched_cfg = dict(pipe.scheduler.config)

    failures = 0
    for entry in hero:
        idx = int(entry["idx"])
        scene_idx = int(entry.get("scene_idx", idx))
        shot_idx = int(entry.get("shot_idx", 0))
        out_path = f"/kaggle/working/scene_{scene_idx+1:02d}_shot_{shot_idx+1:02d}.mp4"
        try:
            if _has_alarm:
                signal.alarm(clip_timeout)
            e_w = int(entry.get("width", W))
            e_h = int(entry.get("height", H))
            e_nf = int(entry.get("num_frames", NF))
            e_shift = entry.get("flow_shift", cfg.get("flow_shift"))
            if e_shift is not None:
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    {**_base_sched_cfg, "flow_shift": float(e_shift)})
            img = (Image.open(BytesIO(base64.b64decode(entry["image_b64"])))
                   .convert("RGB").resize((e_w, e_h), Image.BICUBIC))
            gen = torch.Generator("cuda").manual_seed(seed0 + idx)
            print(f"[{idx}] WAN render {e_w}x{e_h} {e_nf}f {STEPS}steps g{G} "
                  f"shift={e_shift}", flush=True)
            t0 = time.time()
            out = pipe(
                image=img,
                prompt=entry.get("prompt", ""),
                negative_prompt=entry.get("negative", "blurry, low quality, deformed, melting, warping, extra limbs, mutated"),
                height=e_h, width=e_w,
                num_frames=e_nf,
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

    # ESRGAN as a SEPARATE PHASE after the Wan pipe is released — running it
    # with the pipe resident SIGKILLed the kernel at -9 (host RAM, v20).
    if cfg.get("esrgan"):
        print("Releasing Wan pipeline before ESRGAN phase...", flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        import glob as _glob
        import imageio
        for mp4 in sorted(_glob.glob("/kaggle/working/scene_*.mp4")):
            try:
                if _has_alarm:
                    signal.alarm(int(cfg.get("clip_timeout_s", 1500)))
                frames = np.stack([np.asarray(f) for f in
                                   imageio.mimread(mp4, memtest=False)])
                h, w = frames.shape[1], frames.shape[2]
                t1 = time.time()
                up = _esrgan_up(frames, w * 2, h * 2)
                _write_mp4(up, mp4)
                print(f"  esrgan {mp4} {time.time()-t1:.0f}s -> {w*2}x{h*2}",
                      flush=True)
                del frames, up
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                import traceback
                print(f"ESRGAN FAILED {mp4} (raw kept):\n"
                      + traceback.format_exc(), flush=True)
            finally:
                if _has_alarm:
                    signal.alarm(0)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        print("RUN_WAN_PHASE FAILED (non-fatal):\n" + traceback.format_exc(), flush=True)
    sys.exit(0)
