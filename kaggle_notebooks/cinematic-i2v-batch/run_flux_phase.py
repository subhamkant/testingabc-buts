import json
import os
import sys
import torch
import gc
from diffusers import FluxPipeline


def _resolve_hf_token(cfg: dict | None = None) -> str | None:
    """Find HF_TOKEN from (a) Kaggle Secrets (UserSecretsClient — works inside
    Kaggle kernels), (b) cfg['hf_token'] passed via current_run.json, or
    (c) os.environ['HF_TOKEN'] (local debug runs). Returns None if none.

    black-forest-labs/FLUX.1-schnell is a GATED HuggingFace repo as of
    2026-06-28 — anonymous download returns 401. The kernel must authenticate
    before calling FluxPipeline.from_pretrained().

    (b) exists because CLI kernel pushes silently RESET the notebook's
    Add-ons -> Secrets attachment (diagnosed 2026-07-04: three straight
    'empty-log' kernel errors were this fast-fail). The kernel is private,
    the token is read-scoped, and Secrets remain the preferred source —
    the cfg fallback only engages when the attachment has been dropped."""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token
    except Exception:
        # Either not running in Kaggle env OR the secret isn't attached
        pass
    if cfg and cfg.get("hf_token"):
        return cfg["hf_token"]
    return os.environ.get("HF_TOKEN")


def main():
    with open('current_run.json') as f:
        cfg = json.load(f)

    style_anchor = cfg.get("style_anchor", "")
    master_seed = cfg.get("master_seed", 42)
    scenes = cfg.get("scenes", [])

    # Authenticate with HuggingFace BEFORE loading the gated FLUX-schnell repo.
    # Without this, from_pretrained() returns 401 Unauthorized and the whole
    # phase silently crashes (the !python invocation in the notebook cell
    # exits non-zero but Kaggle reports the cell as COMPLETE — a separate
    # bug now fixed in the notebook's subprocess wrapper).
    hf_token = _resolve_hf_token(cfg)
    if not hf_token:
        print("FATAL: no HF_TOKEN found. black-forest-labs/FLUX.1-schnell is "
              "a gated repo and the kernel cannot download it anonymously. "
              "Add HF_TOKEN as a Kaggle Secret (Add-ons -> Secrets) and "
              "attach it to this kernel.")
        sys.exit(1)
    from huggingface_hub import login as hf_login
    hf_login(token=hf_token, add_to_git_credential=False)
    print(f"HuggingFace authenticated (HF_TOKEN length={len(hf_token)})")

    # ── Memory-fit load (2026-07-04) ─────────────────────────────────────
    # FLUX-schnell's 12B transformer is ~24GB in fp16 — it fits NEITHER a
    # T4's 16GB VRAM nor Kaggle's ~30GB system RAM alongside T5-XXL. The
    # first-ever run to get past the HF 401 died at 'Loading checkpoint
    # shards 1/3' (RAM kill). NF4-quantize the transformer (~6.5GB) and
    # 8-bit the T5 (~5GB): total fits a T4 with headroom. Compute dtype is
    # float16 — T4 (sm_75) has no native bfloat16.
    # Unconditional UPGRADE — the Kaggle image ships an old bitsandbytes
    # that imports fine but fails diffusers' quantizer validation
    # ("requires the latest version", caught 2026-07-04). ~20s.
    import subprocess as _sp
    print("Upgrading bitsandbytes...")
    _sp.run([sys.executable, "-m", "pip", "install", "-q", "-U",
             "bitsandbytes"], check=True)

    from diffusers import FluxTransformer2DModel
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from transformers import T5EncoderModel
    from transformers import BitsAndBytesConfig as TransformersBnb

    MODEL = "black-forest-labs/FLUX.1-schnell"
    print("Loading FLUX-schnell transformer (NF4)...")
    transformer = FluxTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer",
        quantization_config=DiffusersBnb(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16),
        torch_dtype=torch.float16)
    print("Loading T5-XXL text encoder (8-bit)...")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL, subfolder="text_encoder_2",
        quantization_config=TransformersBnb(load_in_8bit=True))
    print("Assembling pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL, transformer=transformer, text_encoder_2=text_encoder_2,
        torch_dtype=torch.float16)

    # Load XLabs IP-Adapter weights from the attached Kaggle Dataset.
    # The dataset `subhamkant11/xlabs-flux-ip-adapter` is referenced via
    # kernel-metadata.json's `dataset_sources` and mounts read-only under
    # /kaggle/input/. NOTE (2026-07-04): this call has never executed in
    # production (every prior run died at the HF 401) — if the checkpoint
    # lacks a bundled CLIP image encoder, retry with the standard one.
    try:
        pipe.load_ip_adapter(
            "/kaggle/input/xlabs-flux-ip-adapter",
            subfolder="",
            weight_name="ip_adapter.safetensors"
        )
    except Exception as _ip_err:
        print(f"IP-Adapter load failed ({_ip_err}); retrying with explicit "
              f"CLIP image encoder...")
        pipe.load_ip_adapter(
            "/kaggle/input/xlabs-flux-ip-adapter",
            subfolder="",
            weight_name="ip_adapter.safetensors",
            image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
        )
    # Quantized modules pin themselves to the GPU; offload the rest on
    # demand instead of a blanket .to('cuda').
    pipe.enable_model_cpu_offload()

    # Sprint 2.2 (2026-07-04) — hero_mode: lock the video's hero frames to
    # the channel's APPROVED master anchor (assets/character_anchors/) via
    # IP-Adapter. The anchor image rides in current_run.json as base64 —
    # no Kaggle Dataset round-trip (30s-3min indexing delay) and no
    # dependency on repo visibility. Each scene entry is self-contained:
    # {idx, prompt, w, h, seed}. Outputs: /kaggle/working/hero_<idx>.jpg.
    if cfg.get("hero_mode", False):
        import base64
        from io import BytesIO
        from PIL import Image
        anchor_pil = Image.open(
            BytesIO(base64.b64decode(cfg["anchor_b64"]))).convert("RGB")
        ip_scale = float(cfg.get("ip_scale", 0.6))
        pipe.set_ip_adapter_scale(ip_scale)
        print(f"hero_mode: {len(cfg['scenes'])} frames, ip_scale={ip_scale}, "
              f"anchor={anchor_pil.size}")
        for sc in cfg["scenes"]:
            generator = torch.Generator(device="cuda").manual_seed(int(sc["seed"]))
            print(f"Generating hero frame idx={sc['idx']}...")
            image = pipe(
                prompt=sc["prompt"],
                height=int(sc.get("h", 1344)),
                width=int(sc.get("w", 768)),
                ip_adapter_image=anchor_pil,
                guidance_scale=0.0,
                num_inference_steps=8,
                generator=generator
            ).images[0]
            image.save(f"/kaggle/working/hero_{int(sc['idx']):02d}.jpg")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        return

    # Sprint 2.1 (2026-07-03) — anchor_mode: generate INDEPENDENT master-
    # anchor portraits, one per character. The normal flow below locks every
    # shot after the first to the first face via IP-Adapter 0.6 — correct
    # for one video's hero, wrong for a 5-character anchor batch (all five
    # would inherit the first character's face). anchor_mode keeps the
    # adapter at 0.0 for every shot and names outputs <name>_anchor.jpg.
    if cfg.get("anchor_mode", False):
        pipe.set_ip_adapter_scale(0.0)
        for i, scene in enumerate(scenes):
            shot = scene["visual_track"][0]
            name = scene.get("name", f"anchor_{str(i).zfill(2)}")
            seed = int(scene.get("seed", master_seed + i))
            prompt = f"{style_anchor}, {shot.get('prompt', '')}"
            generator = torch.Generator(device="cuda").manual_seed(seed)
            print(f"Generating anchor {name} (seed={seed})...")
            image = pipe(
                prompt=prompt,
                height=1344,
                width=768,
                guidance_scale=0.0,
                num_inference_steps=8,
                generator=generator
            ).images[0]
            image.save(f"/kaggle/working/{name}_anchor.jpg")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        return

    master_anchor_pil = None

    for i, scene in enumerate(scenes):
        for j, shot in enumerate(scene.get("visual_track", [])):
            prompt = f"{shot.get('category', '')}, {style_anchor}, {shot.get('prompt', '')}"
            shot_seed = master_seed + (i * 10) + j
            generator = torch.Generator(device="cuda").manual_seed(shot_seed)

            print(f"Generating Scene {i+1} Shot {j+1}...")

            if master_anchor_pil is None:
                # FIRST SHOT: The Master Anchor (Weight 0.0)
                pipe.set_ip_adapter_scale(0.0)
                image = pipe(
                    prompt=prompt,
                    height=1920,
                    width=1080,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    generator=generator
                ).images[0]
                master_anchor_pil = image
                # Re-engage IP-Adapter for all subsequent shots
                pipe.set_ip_adapter_scale(0.6)
            else:
                # SUBSEQUENT SHOTS: Anchored to Master
                image = pipe(
                    prompt=prompt,
                    height=1920,
                    width=1080,
                    ip_adapter_image=master_anchor_pil,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    generator=generator
                ).images[0]

            out_path = f"/kaggle/working/scene_{str(i+1).zfill(2)}_shot_{str(j+1).zfill(2)}.jpg"
            image.save(out_path)

    # Defensive cleanup before process exits
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
