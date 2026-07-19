import json
import os
import subprocess
import sys

# bitsandbytes MUST be upgraded BEFORE diffusers is imported — diffusers
# caches the bitsandbytes version in its import_utils at import time, so
# an in-main() upgrade is invisible to its NF4 validator (caught
# 2026-07-04: 'requires the latest version' persisted across an upgrade
# that ran after `from diffusers import ...`).
print("Upgrading bitsandbytes (pre-import)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "bitsandbytes"], check=True)

# Reduce fragmentation headroom loss on the 14.6GiB T4 (OOM'd at 80MiB
# with 76MiB reserved-unallocated, 2026-07-04).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import os as _os
_os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # encoder is small; plain dl + timeout
_os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
# Loader materializes checkpoint tensors on-GPU pre-quantization; expandable
# segments stop that transient from fragmenting the 14.5GB T4 pool.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _mem(tag):
    import torch as _t
    _f, _tot = _t.cuda.mem_get_info()
    print(f"[mem] {tag}: live={_t.cuda.memory_allocated()/1e9:.2f}GB "
          f"reserved={_t.cuda.memory_reserved()/1e9:.2f}GB "
          f"free_phys={_f/1e9:.2f}GB", flush=True)
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

    # skip_flux (2026-07-05): motion-only runs feed LTX a pre-made
    # conditioning image (e.g. a CF-schnell still or an approved anchor)
    # via hero_motion[].image_b64, so no FLUX model needs to load at all.
    # Early-exit BEFORE any weight download to keep the phase fast + light.
    if cfg.get("skip_flux", False):
        print("FLUX phase SKIPPED (skip_flux=true) — LTX will use provided "
              "conditioning image(s).")
        return

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
    # (bitsandbytes upgraded at module top, pre-diffusers-import.)
    from diffusers import FluxTransformer2DModel
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from transformers import T5EncoderModel
    from transformers import BitsAndBytesConfig as TransformersBnb

    # flux_model knob: schnell (8-step, fast) | dev (25-step, benchmark detail).
    # dev is GATED on HF — token account must have accepted the license.
    MODEL = ("black-forest-labs/FLUX.1-dev"
             if cfg.get("flux_model") == "dev"
             else "black-forest-labs/FLUX.1-schnell")
    print(f"FLUX model: {MODEL}", flush=True)
    # DTYPE (2026-07-05 fix): FLUX's native dtype is bfloat16 — the text
    # encoders emit bf16 hidden states. The June code loaded the transformer
    # as float16 (a T4 tensor-core optimization), which crashed the FIRST
    # real generation with 'mat1 and mat2 must have the same dtype, but got
    # BFloat16 and Half' (v45, 2026-07-05). bf16 everywhere matches FLUX
    # reference and eliminates the mismatch. T4 (sm_75) runs bf16 on CUDA
    # cores (no tensor-core accel) — a bit slower than fp16 but correct, and
    # for an overnight batch the speed cost is irrelevant.
    DT = torch.bfloat16
    # ORDER MATTERS (2026-07-04, third OOM iteration): transformers' new
    # core_model_loading materializes checkpoint tensors on the GPU BEFORE
    # bnb quantizes them — T5-XXL's transient spike (~9GB) needs an EMPTY
    # GPU. So: spiky T5 first, well-behaved transformer second.
    print("Loading T5-XXL text encoder (NF4 bf16, empty GPU)...")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        MODEL, subfolder="text_encoder_2",
        quantization_config=TransformersBnb(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DT))
    gc.collect()
    torch.cuda.empty_cache()
    _mem("after T5 load")
    # Transformer precision by GPU count: Kaggle allocates T4 x2 — 31GB
    # combined. Shard the transformer in full bf16 across both (~12GB each
    # via accelerate device_map). NF4 transformer remains the single-GPU
    # fallback — functional but slower; production timeouts catch it.
    n_gpu = torch.cuda.device_count()
    if cfg.get('force_nf4'):
        n_gpu = 1  # wuxia face-lock: NF4 path leaves headroom for the CLIP image encoder
    if n_gpu >= 2:
        print(f"Loading FLUX-schnell transformer (bf16, sharded across "
              f"{n_gpu} GPUs)...")
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL, subfolder="transformer",
            torch_dtype=DT, device_map="auto")
    else:
        print("Loading FLUX-schnell transformer (NF4 bf16, single GPU)...")
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL, subfolder="transformer",
            quantization_config=DiffusersBnb(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=DT),
            torch_dtype=DT)
    gc.collect()
    torch.cuda.empty_cache()  # transformer load has its own materialization transient
    _mem("after transformer load")
    print("Assembling pipeline...")
    pipe = FluxPipeline.from_pretrained(
        MODEL, transformer=transformer, text_encoder_2=text_encoder_2,
        torch_dtype=DT)
    gc.collect()
    torch.cuda.empty_cache()
    _mem("after pipe assembly")

    # IP-Adapter face-lock is OPTIONAL (2026-07-05 fix). It is gated by
    # cfg['use_ip_adapter'] (default True to preserve production intent) and
    # ANY load failure degrades gracefully to seed+prompt consistency —
    # NEVER the hanging fallback that stalled v43 for 3 hours.
    #
    # Root cause of the v43 hang: the XLabs FLUX IP-Adapter checkpoint has
    # no bundled CLIP image encoder, so the first load_ip_adapter() failed;
    # the old retry passed image_encoder_pretrained_model_name_or_path=
    # "openai/clip-vit-large-patch14", which triggered an un-timeout'd HF
    # download that hung indefinitely. That retry is DELETED. Face-lock is
    # an enhancement, not a hard dependency — the character-lock prompt +
    # per-character stable seed already give strong cross-video consistency.
    # hero_mode VRAM choreography (v65 OOM'd inside encode_prompt): with T5
    # NF4 (~5GB) + transformer NF4 + IPA weights + CLIP image encoder all
    # resident, the T4 hits 56MB free before the first forward pass. So:
    # encode prompts NOW (only T5+transformer loaded), evict both text
    # encoders, and only THEN load the IP-Adapter + its image encoder.
    _hero_embeds = None
    if cfg.get("hero_mode", False):
        pipe.text_encoder.to("cuda")
        _mem("before hero pre-encode")
        _hero_embeds = []
        # no_grad is ESSENTIAL: pipe.__call__ wraps encoding itself, but a
        # direct encode_prompt() builds autograd graphs — v67 showed one T5
        # forward retaining 2.2GB of activations and OOMing the T4.
        with torch.no_grad():
            for _sc in cfg["scenes"]:
                _pe, _ppe, _tid = pipe.encode_prompt(
                    prompt=_sc["prompt"], prompt_2=_sc["prompt"],
                    device="cuda", max_sequence_length=512)
                _hero_embeds.append((_pe, _ppe))
        pipe.text_encoder_2 = None
        pipe.text_encoder = None
        del text_encoder_2
        gc.collect()
        torch.cuda.empty_cache()
        _mem("after text-encoder eviction")
        print("hero prompts pre-encoded, text encoders evicted", flush=True)

    ip_ready = False
    if cfg.get("use_ip_adapter", True):
        try:
            pipe.load_ip_adapter(
                "/kaggle/input/xlabs-flux-ip-adapter",
                subfolder="",
                weight_name="ip_adapter.safetensors",
                image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14",
            )
            ip_ready = True
            print("IP-Adapter loaded OK (face-lock active)")
        except Exception as _ip_err:
            print(f"IP-Adapter load failed ({str(_ip_err)[:140]}); continuing "
                  f"WITHOUT face-lock (seed+prompt consistency only)", flush=True)
    else:
        print("IP-Adapter disabled by config (use_ip_adapter=false)")

    # Quantized modules pin themselves to the GPU and CANNOT be moved
    # (bnb 4-bit forbids .to()/offload hooks). Move ONLY the small
    # unquantized parts explicitly: VAE ~0.2GB, CLIP ~0.3GB, and (if the
    # adapter loaded) its image encoder ~0.6GB. Total ~10.5GB on a 15.6GB T4.
    pipe.vae.to("cuda")
    if pipe.text_encoder is not None:  # evicted in hero_mode
        pipe.text_encoder.to("cuda")
    if ip_ready and getattr(pipe, "image_encoder", None) is not None:
        pipe.image_encoder.to("cuda")
    print(f"Pipeline placed (ip_ready={ip_ready}): quantized transformer+T5 "
          f"pinned, vae/clip moved to cuda", flush=True)
    _mem("after IPA + placement")

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
        anchor_pil = None
        _anchor_cache = {}

        def _scene_anchor(sc):
            # Per-scene anchor so ONE push face-locks MANY characters.
            # Anchors ride ONCE in cfg["anchors"] = {key: b64} and scenes
            # reference them by anchor_key (payload stays small — a b64
            # per scene blew past Kaggle's push size limit, 400 error).
            key = sc.get("anchor_key")
            if key and key in cfg.get("anchors", {}):
                if key not in _anchor_cache:
                    _anchor_cache[key] = Image.open(BytesIO(
                        base64.b64decode(cfg["anchors"][key]))).convert("RGB")
                return _anchor_cache[key]
            b64 = sc.get("anchor_b64") or cfg.get("anchor_b64")
            if not b64:
                return None
            ck = b64[:64]
            if ck not in _anchor_cache:
                _anchor_cache[ck] = Image.open(
                    BytesIO(base64.b64decode(b64))).convert("RGB")
            return _anchor_cache[ck]

        if ip_ready:
            anchor_pil = _scene_anchor(cfg["scenes"][0])
            ip_scale = float(cfg.get("ip_scale", 0.6))
            pipe.set_ip_adapter_scale(ip_scale)
            print(f"hero_mode: {len(cfg['scenes'])} frames, FACE-LOCK ON "
                  f"ip_scale={ip_scale}, per-scene anchors", flush=True)
        else:
            print(f"hero_mode: {len(cfg['scenes'])} frames, FACE-LOCK OFF "
                  f"(seed+prompt only)", flush=True)
        import time as _time
        n_steps = int(cfg.get("num_steps", 8))
        for _j, sc in enumerate(cfg["scenes"]):
            generator = torch.Generator(device="cuda").manual_seed(int(sc["seed"]))
            print(f"Generating hero frame idx={sc['idx']} "
                  f"({sc.get('w')}x{sc.get('h')}, {n_steps} steps)...",
                  flush=True)
            _t0 = _time.time()

            def _step_cb(p, i, t, kw):
                print(f"  step {i+1}/{n_steps} at {_time.time()-_t0:.1f}s",
                      flush=True)
                return kw

            gen_kwargs = dict(
                prompt_embeds=_hero_embeds[_j][0],
                pooled_prompt_embeds=_hero_embeds[_j][1],
                height=int(sc.get("h", 1344)),
                width=int(sc.get("w", 768)),
                guidance_scale=float(cfg.get('guidance_scale', 0.0)),
                num_inference_steps=n_steps,
                generator=generator,
                callback_on_step_end=_step_cb,
            )
            _a = _scene_anchor(sc) if ip_ready else None
            if _a is not None:
                gen_kwargs["ip_adapter_image"] = _a
            image = pipe(**gen_kwargs).images[0]
            print(f"  frame done in {_time.time()-_t0:.1f}s", flush=True)
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
                guidance_scale=float(cfg.get('guidance_scale', 0.0)),
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
                    guidance_scale=float(cfg.get('guidance_scale', 0.0)),
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
                    guidance_scale=float(cfg.get('guidance_scale', 0.0)),
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
    # Print any failure to STDOUT — Kaggle's log stream truncates stderr
    # under heavy progress-bar noise, which made smoke v8's death after
    # 'Generating hero frame idx=0' completely silent.
    try:
        main()
    except BaseException:
        import traceback
        print("RUN_FLUX_PHASE FAILED:\n" + traceback.format_exc(),
              flush=True)
        sys.exit(1)
