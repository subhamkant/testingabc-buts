import json
import os
import sys
import torch
import gc
from diffusers import FluxPipeline


def _resolve_hf_token() -> str | None:
    """Find HF_TOKEN from (a) Kaggle Secrets (UserSecretsClient — works inside
    Kaggle kernels) or (b) os.environ['HF_TOKEN'] (works in local debug runs).
    Returns None if neither is available.

    black-forest-labs/FLUX.1-schnell is a GATED HuggingFace repo as of
    2026-06-28 — anonymous download returns 401. The kernel must authenticate
    before calling FluxPipeline.from_pretrained()."""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("HF_TOKEN")
        if token:
            return token
    except Exception:
        # Either not running in Kaggle env OR the secret isn't attached
        pass
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
    hf_token = _resolve_hf_token()
    if not hf_token:
        print("FATAL: no HF_TOKEN found. black-forest-labs/FLUX.1-schnell is "
              "a gated repo and the kernel cannot download it anonymously. "
              "Add HF_TOKEN as a Kaggle Secret (Add-ons -> Secrets) and "
              "attach it to this kernel.")
        sys.exit(1)
    from huggingface_hub import login as hf_login
    hf_login(token=hf_token, add_to_git_credential=False)
    print(f"HuggingFace authenticated (HF_TOKEN length={len(hf_token)})")

    print("Loading FLUX-schnell + IP-Adapter...")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)

    # Load XLabs IP-Adapter weights from the attached Kaggle Dataset.
    # The dataset `subhamkant11/xlabs-flux-ip-adapter` is referenced via
    # kernel-metadata.json's `dataset_sources` and mounts read-only under
    # /kaggle/input/. This replaces the previous runtime `huggingface-cli
    # download` call which timed out at the Papermill 12h ceiling on
    # 2026-06-14 — auth-gated downloads hang indefinitely in non-interactive
    # notebooks.
    pipe.load_ip_adapter(
        "/kaggle/input/xlabs-flux-ip-adapter",
        subfolder="",
        weight_name="ip_adapter.safetensors"
    )
    pipe.to("cuda")

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
