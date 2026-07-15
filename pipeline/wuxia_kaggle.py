"""
Wuxia Kaggle client — pipeline/wuxia_kaggle.py
==============================================

Thin builder+pusher for the LANDSCAPE Wuxia LTX kernel
(subhamkant11/wuxia-i2v). Reuses pipeline.kaggle_client verbatim for the
generic push-CLI / poll / download / P100-detect machinery; the ONLY thing that
differs is the notebook we build.

Why NOT reuse kaggle_client.push_kernel_with_run_config:
  It hardcodes a 4-cell FLUX+LTX notebook, `pip install ... av` (no spandrel for
  Real-ESRGAN), and a FATAL IP-Adapter dataset check that aborts the kernel even
  when skip_flux=true. The Wuxia kernel is LTX-only + ESRGAN, no FLUX, no
  IP-Adapter — so it needs its own 3-cell notebook builder.

Isolation: this module is NEW and imported only by pipeline/wuxia_motion.py.
Zero blast radius on Mahabharata / curiosity.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from pipeline.kaggle_client import (  # reuse verbatim
    KaggleClientError,
    _PUSH_TIMEOUT_S,
    _kaggle_env,
    _make_code_cell,
    _parse_version_from_push_stdout,
    _run_kaggle_cli,
    download_output,
    is_p100_failure,
    poll_kernel,
)

__all__ = [
    "push_wuxia_kernel",
    "poll_kernel",
    "download_output",
    "is_p100_failure",
    "KaggleClientError",
    "poll_kernel_as",
    "download_output_as",
]


# ── Multi-account support (Phase 3: 3 accounts x 2 kernels = 6 GPU slots) ──────
# The kaggle CLI resolves creds from KAGGLE_USERNAME/KAGGLE_KEY env vars FIRST
# (they beat KAGGLE_CONFIG_DIR's kaggle.json). Each CLI call is its own
# subprocess, so overlaying creds into the CHILD env — never os.environ — lets
# concurrent per-account calls run without racing each other. These are forks of
# kaggle_client's internals (that module is shared with Mahabharata: no edits).

def _acct_env_overlay(creds_json: str | None) -> dict[str, str]:
    """{KAGGLE_USERNAME, KAGGLE_KEY} for a kaggle.json-style file; {} = ambient
    default account (repo-root kaggle.json via KAGGLE_CONFIG_DIR)."""
    if not creds_json:
        return {}
    d = json.loads(Path(creds_json).read_text(encoding="utf-8"))
    return {"KAGGLE_USERNAME": d["username"], "KAGGLE_KEY": d["key"]}


def _run_cli_as(creds_json: str | None, args: list[str], *, timeout_s: int
                ) -> subprocess.CompletedProcess:
    """Account-aware fork of kaggle_client._run_kaggle_cli (same semantics)."""
    env = _kaggle_env()
    env.update(_acct_env_overlay(creds_json))
    cmd = ["kaggle"] + args
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise KaggleClientError(f"kaggle CLI timed out after {timeout_s}s: {' '.join(cmd)}") from e
    except FileNotFoundError as e:
        raise KaggleClientError("kaggle CLI not on PATH. `pip install kaggle>=1.6.0`.") from e
    if result.returncode != 0:
        raise KaggleClientError(
            f"kaggle CLI returned exit {result.returncode}: {' '.join(cmd)}\n"
            f"stderr: {result.stderr.strip()[:600]}"
        )
    return result


async def poll_kernel_as(kernel_ref: str, creds_json: str | None, *,
                         poll_interval_s: int = 60, timeout_s: int = 7200) -> dict:
    """Account-aware fork of kaggle_client.poll_kernel: prints on status change,
    tolerates up to 5 consecutive transient CLI errors, never re-pushes."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    last_status, polls, errs = "unknown", 0, 0
    while True:
        polls += 1
        try:
            result = await asyncio.to_thread(
                _run_cli_as, creds_json, ["kernels", "status", kernel_ref], timeout_s=30)
            errs = 0
        except KaggleClientError as e:
            errs += 1
            print(f"    [kaggle] poll #{polls}: CLI error ({str(e)[:120]}) "
                  f"— consecutive #{errs}/5, retrying poll", flush=True)
            if errs >= 5:
                raise
            if asyncio.get_event_loop().time() + poll_interval_s > deadline:
                raise KaggleClientError(
                    f"kernel {kernel_ref} poll exhausted timeout during CLI-error recovery") from e
            await asyncio.sleep(poll_interval_s)
            continue
        m = re.search(r'status\s+"?(?:KernelWorkerStatus\.)?([a-zA-Z_]+)"?', result.stdout)
        status = m.group(1).lower() if m else "unknown"
        if status != last_status:
            print(f"    [kaggle] poll #{polls}: status={status} ({kernel_ref})", flush=True)
            last_status = status
        if status in ("complete", "error", "cancelled", "failed"):
            return {"status": status, "message": result.stdout.strip()}
        if asyncio.get_event_loop().time() + poll_interval_s > deadline:
            raise KaggleClientError(
                f"kernel {kernel_ref} did not complete within {timeout_s}s "
                f"(last status: {status}).")
        await asyncio.sleep(poll_interval_s)


async def download_output_as(kernel_ref: str, target_dir: Path,
                             creds_json: str | None) -> list[Path]:
    """Account-aware fork of kaggle_client.download_output (same crash-tolerant
    semantics: non-zero CLI exit is OK iff target files actually landed)."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", kernel_ref, "-p", str(target_dir), "--force"]
    try:
        await asyncio.to_thread(_run_cli_as, creds_json, args, timeout_s=600)
    except KaggleClientError as e:
        landed = [p for p in target_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4")]
        if not landed:
            raise
        print(f"    [kaggle] download CLI non-zero ({str(e)[:120]}) but "
              f"{len(landed)} file(s) landed — treating as success", flush=True)
    return sorted(p for p in target_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4"))


def _build_wuxia_notebook(run_config: dict[str, Any], ltx_script: str) -> dict[str, Any]:
    """3-cell notebook: setup + P100 fast-fail + write artifacts / run LTX+ESRGAN / verify.

    The P100 fast-fail print MUST keep the literal "compute capability" + "< 7.0"
    markers so pipeline.kaggle_client.is_p100_failure() still matches on the logs.
    """
    config_b64 = base64.b64encode(
        json.dumps(run_config, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    ltx_b64 = base64.b64encode(ltx_script.encode("utf-8")).decode("ascii")

    cell1 = "\n".join([
        "# CELL 1: setup + GPU sanity + write inlined artifacts to /kaggle/working/",
        "# AUTO-GENERATED by pipeline/wuxia_kaggle.py — do NOT hand-edit.",
        "!pip install -q --upgrade diffusers accelerate imageio-ffmpeg spandrel",
        "",
        "import torch, os, sys, json, base64",
        "",
        "# --- P100 fast-fail: PyTorch 2.x dropped sm_60 support ---",
        "if not torch.cuda.is_available():",
        "    print('FATAL: no GPU allocated. Re-run kernel.')",
        "    sys.exit(1)",
        "_major, _minor = torch.cuda.get_device_capability()",
        "_gpu_name = torch.cuda.get_device_name(0)",
        "print(f'GPU: {_gpu_name}, compute={_major}.{_minor}')",
        "if (_major, _minor) < (7, 0):",
        "    print(f'FATAL: GPU compute capability {_major}.{_minor} < 7.0. PyTorch 2.x dropped sm_60 (P100/Pascal). Re-run kernel \\u2014 Kaggle may give a T4 next time.')",
        "    sys.exit(1)",
        "",
        "# --- Decode + write inlined artifacts ---",
        f"_CONFIG_B64 = '{config_b64}'",
        f"_LTX_B64 = '{ltx_b64}'",
        "with open('current_run.json', 'w', encoding='utf-8') as _f:",
        "    _f.write(base64.b64decode(_CONFIG_B64).decode('utf-8'))",
        "with open('run_ltx_phase.py', 'wb') as _f:",
        "    _f.write(base64.b64decode(_LTX_B64))",
        "_cfg = json.loads(base64.b64decode(_CONFIG_B64).decode('utf-8'))",
        "print(f'Setup OK. run_id={_cfg.get(\"run_id\")}, motion_clips={len(_cfg.get(\"hero_motion\", []))}')",
    ])

    cell2 = "\n".join([
        "# CELL 2: LTX + anime Real-ESRGAN (subprocess for clean VRAM lifecycle)",
        "import subprocess, sys",
        "_to = int(_cfg.get('ltx_timeout_s', 3600))",
        "try:",
        "    _r = subprocess.run([sys.executable, 'run_ltx_phase.py'], timeout=_to)",
        "except subprocess.TimeoutExpired:",
        "    print(f'run_ltx_phase.py exceeded {_to}s hard timeout \\u2014 aborting (anti-hang)')",
        "    _r = None",
        "if _r is not None and _r.returncode != 0:",
        "    raise RuntimeError(f'run_ltx_phase.py exited with code {_r.returncode}')",
    ])

    cell3 = "\n".join([
        "# CELL 3: output verification",
        "!ls -la /kaggle/working/*.mp4 2>/dev/null || echo 'no mp4 outputs'",
        "print('Kernel processing complete.')",
    ])

    return {
        "cells": [
            _make_code_cell(cell1),
            _make_code_cell(cell2),
            _make_code_cell(cell3),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


async def push_wuxia_kernel(kernel_dir: Path, run_config: dict[str, Any],
                            creds_json: str | None = None) -> int:
    """Build a self-contained notebook.ipynb (run_config + run_ltx_phase.py inlined
    as base64) into kernel_dir, then `kaggle kernels push -p kernel_dir`.

    creds_json selects the Kaggle ACCOUNT to push as (kaggle.json-style file);
    None = ambient default account. The kernel-metadata.json `id` inside
    kernel_dir must belong to that account.

    Returns the new kernel version number (0 if the CLI didn't print one).
    Raises KaggleClientError on push failure/timeout.

    Concurrency: caller MUST serialize pushes to the SAME kernel_dir (writing
    notebook.ipynb is not atomic with the push). Different kernel_dirs are safe
    concurrently. See pipeline/wuxia_motion.py's asyncio.Lock.
    """
    kernel_dir = Path(kernel_dir)
    if not (kernel_dir / "kernel-metadata.json").exists():
        raise KaggleClientError(f"kernel_dir missing kernel-metadata.json: {kernel_dir}")
    ltx_path = kernel_dir / "run_ltx_phase.py"
    if not ltx_path.exists():
        raise KaggleClientError(f"missing required phase script: {ltx_path}")

    ltx_script = ltx_path.read_text(encoding="utf-8")
    notebook = _build_wuxia_notebook(run_config, ltx_script)
    (kernel_dir / "notebook.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    result = await asyncio.to_thread(
        _run_cli_as,
        creds_json,
        ["kernels", "push", "-p", str(kernel_dir)],
        timeout_s=_PUSH_TIMEOUT_S,
    )
    return _parse_version_from_push_stdout(result.stdout)
