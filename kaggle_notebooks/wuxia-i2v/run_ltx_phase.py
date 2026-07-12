"""
Wuxia landscape LTX-Video I2V + anime Real-ESRGAN — run_ltx_phase.py
====================================================================

Forked from kaggle_notebooks/cinematic-i2v-batch/run_ltx_phase.py, specialised
for the Wuxia long-form pipeline:

  * LANDSCAPE (16:9-ish) render — LTX conditions/renders at ltx_width x ltx_height
    (default 1152x640) instead of the curiosity 512x768 portrait.
  * SKIP-FLUX motion-only: every clip is conditioned on a pre-made still shipped
    in the run_config as `hero_motion[].image_b64` (a Cloudflare FLUX-schnell
    still). No FLUX weights load here at all.
  * In-kernel anime super-resolution: each 65-frame LTX clip is upscaled with
    RealESRGAN_x4plus_anime_6B (via `spandrel`) 4x per-frame, then downscaled to
    out_w x out_h (default 1920x1080). Per-frame `torch.cuda.empty_cache()` +
    interleaved (render one clip -> upscale -> write -> free) keeps peak RAM/VRAM
    low so the T4 doesn't SIGKILL (the failure the base kernel's _write_mp4
    comment documents).
  * Output naming: scene_{scene_idx+1:02d}_shot_{shot_idx+1:02d}.mp4 (1-indexed)
    so pipeline/image_generator._reshape_kaggle_outputs_to_scene_groups parses it
    unchanged.
  * mp4 writer forces pix_fmt=yuv420p + profile High (imageio's default emits
    "High 4:4:4 Predictive" which Windows/YouTube players can't decode).

Motion is an ENHANCEMENT layer: always exit 0 so a per-clip failure never fails
the kernel — missing mp4s are Ken-Burns-fallback'd by the local assembler.
"""
import base64
import gc
import json
import os
import sys
import time
from io import BytesIO

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ESR_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
)
_ESR_PATH = "/kaggle/working/anime_esrgan.pth"

_DEFAULT_NEG = (
    "blurry, low quality, low resolution, soft focus, distorted face, "
    "deformed hands, extra fingers, mutated limbs, jpeg artifacts, "
    "oversaturated, washed out, foggy haze, melting, morphing, warping, "
    "flicker, duplicate, watermark, text"
)


def _load_ltx():
    """LTX-Video 2B sized for a T4: fp16 + model CPU offload + VAE tiling."""
    from diffusers import LTXImageToVideoPipeline

    pipe = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video", torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception as e:
        print(f"VAE tiling unavailable ({e}) — continuing", flush=True)
    return pipe


def _load_esrgan():
    """RealESRGAN_x4plus_anime_6B via spandrel (avoids basicsr import hell)."""
    import urllib.request

    if not os.path.exists(_ESR_PATH):
        print("Downloading anime Real-ESRGAN weights...", flush=True)
        urllib.request.urlretrieve(_ESR_URL, _ESR_PATH)
    from spandrel import ModelLoader

    model = ModelLoader().load_from_file(_ESR_PATH).cuda().eval()
    return model


def _write_mp4(frames_uint8, out_path, fps=24):
    """Write frames as H.264 yuv420p/High — universally playable."""
    import imageio

    imageio.mimwrite(
        out_path,
        list(frames_uint8),
        fps=fps,
        codec="libx264",
        quality=9,
        macro_block_size=None,
        output_params=["-pix_fmt", "yuv420p", "-profile:v", "high"],
    )
    h, w = frames_uint8[0].shape[0], frames_uint8[0].shape[1]
    print(f"SAVED {out_path} ({len(frames_uint8)}f @ {w}x{h})", flush=True)


def _esrgan_upscale(esr, frames_np, out_w, out_h):
    """4x per-frame then bicubic-downscale to (out_w,out_h). Frees VRAM per frame."""
    import torch.nn.functional as Fnn

    outs = []
    for i in range(frames_np.shape[0]):
        t = torch.from_numpy(frames_np[i]).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
        with torch.no_grad():
            o = esr(t)  # 4x
            o = Fnn.interpolate(o, size=(out_h, out_w), mode="bicubic", align_corners=False)
        outs.append((o.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        del t, o
        torch.cuda.empty_cache()
    return outs


def main():
    with open("current_run.json") as f:
        cfg = json.load(f)

    hero = cfg.get("hero_motion", [])
    if not hero:
        print("No hero_motion entries. Exiting.", flush=True)
        return

    W = int(cfg.get("ltx_width", 1152))
    H = int(cfg.get("ltx_height", 640))
    NF = int(cfg.get("ltx_num_frames", 65))       # must be 8*k+1
    STEPS = int(cfg.get("ltx_num_steps", 30))
    G = float(cfg.get("ltx_guidance", 3.0))
    NEG = cfg.get("ltx_negative", _DEFAULT_NEG)
    master_seed = int(cfg.get("master_seed", 42))
    do_esrgan = bool(cfg.get("esrgan", True))
    out_w = int(cfg.get("out_w", 1920))
    out_h = int(cfg.get("out_h", 1080))

    print("Loading LTX-Video (2B, fp16, cpu-offload, vae-tiling)...", flush=True)
    t0 = time.time()
    pipe = _load_ltx()
    print(f"LTX loaded in {time.time()-t0:.0f}s", flush=True)

    esr = None
    if do_esrgan:
        try:
            esr = _load_esrgan()
            print(f"ESRGAN loaded (scale={getattr(esr, 'scale', '?')})", flush=True)
        except Exception:
            import traceback
            print("ESRGAN load failed — will emit resized-raw:\n" + traceback.format_exc(), flush=True)
            esr = None

    failures = 0
    for entry in hero:
        idx = int(entry["idx"])
        scene_idx = int(entry.get("scene_idx", idx))
        shot_idx = int(entry.get("shot_idx", 0))
        out_path = f"/kaggle/working/scene_{scene_idx+1:02d}_shot_{shot_idx+1:02d}.mp4"
        try:
            if not entry.get("image_b64"):
                print(f"[{idx}] missing image_b64 — skip", flush=True)
                failures += 1
                continue
            cond = (
                Image.open(BytesIO(base64.b64decode(entry["image_b64"])))
                .convert("RGB")
                .resize((W, H), Image.BICUBIC)
            )
            gen = torch.Generator("cuda").manual_seed(master_seed + idx)
            print(f"[{idx}] render {W}x{H} {NF}f {STEPS}steps g{G} -> {out_path}", flush=True)
            t0 = time.time()
            vid = pipe(
                prompt=entry.get("prompt", ""),
                negative_prompt=NEG,
                image=cond,
                width=W,
                height=H,
                num_frames=NF,
                num_inference_steps=STEPS,
                guidance_scale=G,
                generator=gen,
                output_type="np",
            ).frames[0]
            print(f"  gen {time.time()-t0:.0f}s", flush=True)
            arr = (np.asarray(vid) * 255).clip(0, 255).astype("uint8")
            del vid
            gc.collect()
            torch.cuda.empty_cache()

            if esr is not None:
                t0 = time.time()
                frames = _esrgan_upscale(esr, arr, out_w, out_h)
                print(f"  esrgan {time.time()-t0:.0f}s", flush=True)
            else:
                frames = [
                    np.asarray(Image.fromarray(a).resize((out_w, out_h), Image.BICUBIC))
                    for a in arr
                ]
            _write_mp4(frames, out_path)
            del arr, frames
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            import traceback
            print(f"CLIP FAILED [{idx}]:\n" + traceback.format_exc(), flush=True)
            failures += 1
            gc.collect()
            torch.cuda.empty_cache()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    if failures:
        print(f"LTX_PHASE_PARTIAL: {failures} clip(s) failed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        print("RUN_LTX_PHASE FAILED (non-fatal, stills preserved locally):\n"
              + traceback.format_exc(), flush=True)
    sys.exit(0)
