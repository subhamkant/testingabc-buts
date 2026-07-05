import json
import os
import sys
import gc
import numpy as np
import torch
from PIL import Image

# Same fragmentation guard as the FLUX phase (T4 14.6GiB headroom).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _load_pipe():
    """LTX I2V pipeline sized for a T4.

    2026-07-05 fixes (both were June-run killers):
      - LTXImageToVideoPipeline, NOT LTXPipeline. LTXPipeline is the
        text-to-video class; passing `image=` to it is an error, and the
        June run OOM'd at load before that could even surface.
      - fp16 + model CPU offload + VAE tiling. The June load did a bare
        `pipe.to("cuda")` — full weights + decode activations on a 15GB
        T4 → 'CUDA out of memory. Tried to allocate 32.00 MiB'.
    """
    from diffusers import LTXImageToVideoPipeline
    pipe = LTXImageToVideoPipeline.from_pretrained(
        "Lightricks/LTX-Video", torch_dtype=torch.float16)
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception as e:
        print(f"VAE tiling unavailable ({e}) — continuing without")
    return pipe


def _write_mp4(video_frames, out_path: str):
    """Write the LTX clip at NATIVE resolution (512x768), no in-kernel upscale.

    2026-07-05 fix: the previous version upscaled all 97 frames to 1920x1080
    as float tensors and stacked them (~2.4GB) while the LTX model sat in
    system RAM from CPU-offload — the Linux OOM killer SIGKILL'd the process
    (exit -9) right after a clip generated fine in 141s. Upscaling belongs in
    the LOCAL video_assembler (abundant RAM), not the T4 kernel. Here we just
    emit a valid native-res mp4 with minimal memory (one ~115MB uint8 array).
    """
    arr = video_frames
    if isinstance(arr, list):
        arr = np.stack([np.asarray(f) for f in arr])
    arr = np.asarray(arr)                                # (T,H,W,C)
    if arr.dtype != np.uint8:
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    # Encode with imageio-ffmpeg (present in the Kaggle image). The prior
    # torchvision.io.write_video needs PyAV, which Kaggle does NOT ship —
    # it ImportError'd at write time after a clip generated fine in 144s
    # (2026-07-05). imageio is the reliable backend; torchvision is fallback.
    fps = 24
    try:
        import imageio
        imageio.mimwrite(out_path, list(arr), fps=fps, codec="libx264",
                         quality=8, macro_block_size=None)
        backend = "imageio"
    except Exception as e_io:
        print(f"imageio write failed ({str(e_io)[:80]}); trying torchvision...",
              flush=True)
        from torchvision.io import write_video
        write_video(out_path, torch.from_numpy(arr), fps=fps)
        backend = "torchvision"
    print(f"Saved {out_path} via {backend} "
          f"({arr.shape[0]} frames @ {arr.shape[2]}x{arr.shape[1]})", flush=True)


def main():
    with open("current_run.json") as f:
        cfg = json.load(f)

    hero_motion = cfg.get("hero_motion", [])
    requires_motion = cfg.get("requires_motion", [])
    if not hero_motion and not requires_motion:
        print("No motion clips requested. Exiting.")
        return

    master_seed = cfg.get("master_seed", 42)
    num_frames = int(cfg.get("ltx_num_frames", 121))    # 121 = ~5s @ 24fps
    num_steps = int(cfg.get("ltx_num_steps", 40))

    print("Loading LTX-Video (I2V, fp16, cpu-offload, vae-tiling)...",
          flush=True)
    pipe = _load_pipe()

    def _render(init_image, motion_prompt: str, seed: int, out_path: str):
        # init_image: PIL.Image (already RGB). LTX native conditioning is
        # 512x768 portrait; upscale/crop back to 1080x1920 after generation.
        cond_image = init_image.resize((512, 768), Image.BICUBIC)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        import time as _time
        _t0 = _time.time()
        print(f"Rendering {out_path} ({num_frames}f, {num_steps} steps)...",
              flush=True)
        video_frames = pipe(
            prompt=motion_prompt,
            image=cond_image,
            width=512,
            height=768,
            num_frames=num_frames,
            num_inference_steps=num_steps,
            generator=generator,
            output_type="np",
        ).frames[0]
        print(f"  clip done in {_time.time()-_t0:.0f}s", flush=True)
        _write_mp4(video_frames, out_path)

    failures = 0

    # ── hero_mode motion: animate the hero still for each entry ─────────
    # Conditioning image source, in priority order:
    #   1. entry['image_b64'] — a pre-made still shipped in the config
    #      (skip_flux motion-only runs: CF-schnell still or approved anchor)
    #   2. /kaggle/working/hero_<idx>.jpg — written by the FLUX phase
    import base64 as _b64
    from io import BytesIO as _BytesIO
    for entry in hero_motion:
        idx = int(entry["idx"])
        out_path = f"/kaggle/working/hero_{idx:02d}.mp4"
        cond_src = None
        if entry.get("image_b64"):
            cond_src = Image.open(
                _BytesIO(_b64.b64decode(entry["image_b64"]))).convert("RGB")
            print(f"hero_{idx:02d}: conditioning on provided image "
                  f"{cond_src.size}", flush=True)
        else:
            in_path = f"/kaggle/working/hero_{idx:02d}.jpg"
            if not os.path.exists(in_path):
                print(f"Missing {in_path} and no image_b64. Skipping.",
                      flush=True)
                failures += 1
                continue
            cond_src = Image.open(in_path).convert("RGB")
        try:
            _render(cond_src, entry.get("prompt", "subtle cinematic motion"),
                    master_seed + idx, out_path)
        except Exception:
            import traceback
            print(f"CLIP FAILED hero_{idx:02d}:\n" + traceback.format_exc(),
                  flush=True)
            failures += 1
        gc.collect()
        torch.cuda.empty_cache()

    # ── legacy curiosity path: scene_XX_shot_YY.jpg tuples ──────────────
    for (scene_idx, shot_idx) in requires_motion:
        in_path = (f"/kaggle/working/scene_{scene_idx+1:02d}"
                   f"_shot_{shot_idx+1:02d}.jpg")
        if not os.path.exists(in_path):
            print(f"Missing {in_path}. Skipping motion.", flush=True)
            failures += 1
            continue
        shot = cfg["scenes"][scene_idx]["visual_track"][shot_idx]
        try:
            _render(Image.open(in_path).convert("RGB"), shot.get("prompt", ""),
                    master_seed + (scene_idx * 10) + shot_idx,
                    f"/kaggle/working/scene_{scene_idx+1:02d}"
                    f"_shot_{shot_idx+1:02d}.mp4")
        except Exception:
            import traceback
            print("CLIP FAILED:\n" + traceback.format_exc(), flush=True)
            failures += 1
        gc.collect()
        torch.cuda.empty_cache()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    if failures:
        print(f"LTX_PHASE_PARTIAL: {failures} clip(s) failed", flush=True)


if __name__ == "__main__":
    # Motion is an ENHANCEMENT layer: the stills from the FLUX phase are
    # the core deliverable and must survive an LTX failure. A non-zero
    # exit here would fail CELL 3 -> kernel status=error -> the client
    # discards the ENTIRE output including good stills. So: log loudly,
    # always exit 0. Missing mp4s are visible in CELL 4's listing and the
    # video assembler Ken-Burns-falls-back per missing clip.
    try:
        main()
    except BaseException:
        import traceback
        print("RUN_LTX_PHASE FAILED (non-fatal, stills preserved):\n"
              + traceback.format_exc(), flush=True)
    sys.exit(0)
