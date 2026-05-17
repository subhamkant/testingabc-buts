"""
Diagnostic: probe all configured Cloudflare Workers AI accounts with a
REAL FLUX-schnell call (same params the pipeline uses) and save the
returned image to disk. Lets us see whether CF is genuinely degraded
or just intermittently slow.

Saves outputs to: temp/cf_test/cf_acct_<label>.png
"""

import os
import sys
import time
import base64

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from dotenv import load_dotenv
load_dotenv()

# Match the pipeline's exact params for a fair test
PROMPT = (
    "Bhishma the warrior, full body, standing on a battlefield at dawn, "
    "saffron-bronze dhoti, brown leather chest plate with silver sun-emblem, "
    "full silver-white beard, weathered noble face, cinematic golden-hour "
    "lighting, vertical 9:16 composition"
)
NEGATIVE = (
    "blurry, low quality, deformed, mutated, extra limbs, watermark, text, "
    "signature, ugly, distorted, low resolution, washed out, magenta cast, "
    "orange filter, plastic skin"
)
STEPS = 8       # pipeline default
TIMEOUT = 60    # pipeline default
OUT_DIR = "temp/cf_test"


def collect_accounts():
    out = []
    pid = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    ptk = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    if pid and ptk:
        out.append(("cf-primary", pid, ptk))
    for n in range(2, 6):
        aid = os.environ.get(f"CLOUDFLARE_ACCOUNT_ID_{n}", "").strip()
        atk = os.environ.get(f"CLOUDFLARE_API_TOKEN_{n}", "").strip()
        if aid and atk:
            out.append((f"cf-acct-{n}", aid, atk))
    return out


def probe(label, account_id, token):
    url = (
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
        f"/ai/run/@cf/black-forest-labs/flux-1-schnell"
    )
    headers = {"Authorization": f"Bearer {token}"}
    body = {"prompt": PROMPT, "steps": STEPS, "seed": 42, "negative_prompt": NEGATIVE}

    t0 = time.time()
    try:
        r = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
    except Exception as e:
        elapsed = time.time() - t0
        return False, f"EXCEPTION after {elapsed:.1f}s: {type(e).__name__}: {str(e)[:120]}"

    elapsed = time.time() - t0
    if r.status_code != 200:
        return False, f"HTTP {r.status_code} after {elapsed:.1f}s — body: {r.text[:200]}"

    try:
        data = r.json()
    except Exception as e:
        return False, f"bad JSON after {elapsed:.1f}s: {e}"

    if not data.get("success"):
        return False, f"success=false after {elapsed:.1f}s: {data.get('errors')}"

    img_b64 = data.get("result", {}).get("image", "")
    if not img_b64:
        return False, f"no result.image after {elapsed:.1f}s"

    try:
        raw = base64.b64decode(img_b64)
    except Exception as e:
        return False, f"bad base64 after {elapsed:.1f}s: {e}"

    if len(raw) < 1000:
        return False, f"image too small ({len(raw)} bytes) after {elapsed:.1f}s"

    out_path = os.path.join(OUT_DIR, f"{label}.png")
    with open(out_path, "wb") as f:
        f.write(raw)
    kb = len(raw) / 1024
    return True, f"OK in {elapsed:.1f}s — saved {out_path} ({kb:.0f} KB)"


def main():
    accounts = collect_accounts()
    if not accounts:
        sys.exit("[ERROR] no CF accounts configured")
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Probing {len(accounts)} CF account(s) with steps={STEPS} timeout={TIMEOUT}s")
    print(f"Prompt: {PROMPT[:80]}...")
    print()

    results = []
    for label, aid, tok in accounts:
        ok, msg = probe(label, aid, tok)
        flag = "[OK]" if ok else "[FAIL]"
        print(f"  {flag} {label}: {msg}")
        results.append((label, ok, msg))

    print()
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"Summary: {n_ok}/{len(accounts)} accounts returned a valid image.")
    print(f"Inspect saved images in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
