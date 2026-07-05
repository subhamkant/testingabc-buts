"""
Kaggle Kernels client — pipeline/kaggle_client.py
=================================================

Thin async wrapper around the `kaggle` CLI that gives the curiosity Shorts
pipeline a way to:

  1. Build a self-contained Kaggle notebook with the per-render `run_config`
     + the two phase scripts (`run_flux_phase.py`, `run_ltx_phase.py`)
     all INLINED into CELL 1 as base64 strings, then push that notebook
  2. Poll `kaggle kernels status` until completion / error / timeout
  3. Download all kernel outputs via `kaggle kernels output`

Why inlined notebook (post-2026-06-15 architecture)
---------------------------------------------------
Caught the hard way: `kaggle kernels push` uploads ONLY the file specified
by `kernel-metadata.json`'s `code_file` field. Auxiliary files in the same
folder (other .py scripts, .json configs) are silently dropped. The earlier
architecture had `current_run.json`/`run_flux_phase.py`/`run_ltx_phase.py`
sitting in the kernel folder expecting them to ride along on the push —
they never reached Kaggle, and CELL 2 errored with `FATAL: current_run.json
missing`. See feedback_kaggle_gpu_pipeline_gotchas.md items #9 and #10.

Fix: `push_kernel_with_run_config` now READS the static phase scripts from
the kernel folder + base64-encodes them + builds a fresh `notebook.ipynb`
with everything inlined into CELL 1. CELL 1 base64-decodes the artifacts
back to /kaggle/working/ at runtime so `!python run_flux_phase.py` /
`!python run_ltx_phase.py` invocations (which require subprocess VRAM
isolation between FLUX and LTX) can find them on disk.

CELL 1 also does TWO sanity checks before writing artifacts:
  - GPU compute capability >= 7.0 (PyTorch 2.x dropped sm_60 support;
    Kaggle Free Tier randomly allocates P100 which we must fast-fail on)
  - IP-Adapter Kaggle Dataset is mounted at /kaggle/input/

Why subprocess wrapping (not the `kaggle` Python SDK)
-----------------------------------------------------
The `kaggle` PyPI package's Python API is sync-only and has a long history
of subtle threading bugs around its `KaggleApi` singleton. Wrapping the
proven CLI via `asyncio.to_thread(subprocess.run, ...)` gives us:
  - Clean async semantics for the pipeline's `await` chain
  - Explicit subprocess timeout via `subprocess.run(..., timeout=N)`
  - Subprocess `.kill()` on hang (vs. asyncio.wait_for which only cancels
    the task — the subprocess can outlive the task without explicit kill)

Concurrency contract
--------------------
This module exports stateless functions only. The CALLER is responsible
for serializing access — typically by wrapping the entire push → poll →
download lifecycle in an `asyncio.Lock` (see image_generator.py:_KAGGLE_LOCK)
so EN+HI batches don't write their inlined notebooks back-to-back where
the second push would overwrite the first's queued run.

Mahabharata isolation
---------------------
This file is NEW — Mahabharata's main.py does not import from it.
Zero blast radius on the Mahabharata pipeline.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


class KaggleClientError(RuntimeError):
    """Raised when the Kaggle CLI fails, hangs, or returns an unexpected
    status. The caller's cascade fallback should catch this and degrade
    to the existing Cloudflare FLUX path."""


# ── Internals ────────────────────────────────────────────────────────────────

_PUSH_TIMEOUT_S = 180
"""Hard timeout for `kaggle kernels push`. The push itself is just a HTTP
multipart upload of ~100KB of code/config, so 3 min is generous."""

_STATUS_TIMEOUT_S = 30
"""Hard timeout per `kaggle kernels status` call. Status is a lightweight
GET; if it takes more than 30s the API is degraded."""

_DOWNLOAD_TIMEOUT_S = 300
"""Hard timeout for `kaggle kernels output`. Downloads all files under
/kaggle/working/; for our use case ~50 MB of MP4s + JPGs."""


def _kaggle_env() -> dict[str, str]:
    """Build the environment for kaggle subprocess calls. Points
    KAGGLE_CONFIG_DIR at the repo root so the CLI finds kaggle.json there.
    Also forces PYTHONIOENCODING=utf-8 so the CLI (which is a Python script)
    doesn't crash with 'charmap' codec errors on Windows when it tries to
    print Unicode from kernel logs/status outputs."""
    env = os.environ.copy()
    # Repo root is where this module's parent's parent lives:
    # pipeline/kaggle_client.py → repo_root/pipeline/ → repo_root/
    repo_root = Path(__file__).resolve().parent.parent
    env["KAGGLE_CONFIG_DIR"] = str(repo_root)
    env["PYTHONIOENCODING"] = "utf-8"
    # PYTHONUTF8=1 enables full UTF-8 mode in the child — critical on Windows
    # because the kaggle library's `kernels_output` writes the run log to a
    # file with the default codec (cp1252), which UnicodeEncodeError-crashes
    # on the progress-bar box chars in LTX/FLUX logs (verified 2026-07-05,
    # kaggle_api_extended.py:5171 `out.write(log)`). PYTHONIOENCODING alone
    # does NOT cover file opens; PYTHONUTF8 does.
    env["PYTHONUTF8"] = "1"
    return env


def _run_kaggle_cli(args: list[str], *, timeout_s: int) -> subprocess.CompletedProcess:
    """Synchronous subprocess wrapper. Always raises CalledProcessError on
    non-zero exit, KaggleClientError on timeout. Caller wraps in
    asyncio.to_thread().

    encoding="utf-8" + errors="replace" force the PARENT process to decode
    captured stdout/stderr as UTF-8. Without this, Windows defaults to
    cp1252 which crashes on kernel-log fetches that contain ANSI escapes
    or non-ASCII characters (verified failure 2026-06-16: UnicodeDecodeError
    in subprocess._readerthread → result.stdout=None → TypeError in caller).
    PYTHONIOENCODING=utf-8 in _kaggle_env() handles the child side; this
    flag handles the parent."""
    cmd = ["kaggle"] + args
    try:
        result = subprocess.run(
            cmd,
            env=_kaggle_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise KaggleClientError(
            f"kaggle CLI timed out after {timeout_s}s: {' '.join(cmd)}"
        ) from e
    except FileNotFoundError as e:
        raise KaggleClientError(
            "kaggle CLI not on PATH. Install via `pip install kaggle>=1.6.0`."
        ) from e

    if result.returncode != 0:
        # Surface the stderr — Kaggle CLI prints useful diagnostics there
        # (e.g. "401 unauthorized — bad credentials in kaggle.json").
        raise KaggleClientError(
            f"kaggle CLI returned exit {result.returncode}: {' '.join(cmd)}\n"
            f"stderr: {result.stderr.strip()[:600]}"
        )
    return result


# ── Public API ───────────────────────────────────────────────────────────────


def _make_code_cell(source_text: str) -> dict[str, Any]:
    """Build a Jupyter code cell from a multi-line string. Jupyter expects
    `source` as a list of strings where every line except (optionally) the
    last ends with '\\n' — `splitlines(keepends=True)` produces exactly
    that shape."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_text.splitlines(keepends=True),
    }


def _build_inlined_notebook(
    run_config: dict[str, Any],
    flux_script: str,
    ltx_script: str,
    hf_token: str | None = None,
) -> dict[str, Any]:
    """Build a self-contained Kaggle notebook with run config + both phase
    scripts inlined into CELL 1 as base64 strings. The cell base64-decodes
    them at runtime and writes to /kaggle/working/ so the subsequent
    `!python run_flux_phase.py` / `!python run_ltx_phase.py` invocations
    (kept as subprocesses for VRAM isolation between FLUX and LTX) can
    find them on disk.

    CELL 1 also runs two sanity checks BEFORE writing artifacts:
      - `torch.cuda.get_device_capability() >= (7, 0)` — PyTorch 2.x
        dropped sm_60 (P100/Pascal). Kaggle Free Tier randomly allocates
        P100; we fast-fail in ~30s so the local Cloudflare fallback
        fires immediately instead of burning quota.
      - `/kaggle/input/xlabs-flux-ip-adapter/ip_adapter.safetensors`
        exists — confirms the attached Kaggle Dataset is mounted.

    Returns a dict that serializes cleanly via `json.dumps` to a valid
    Jupyter .ipynb file.
    """
    # base64 encoding sidesteps every escape-character / unicode issue
    # the inline-string approach would otherwise hit
    config_b64 = base64.b64encode(
        json.dumps(run_config, ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    flux_b64 = base64.b64encode(flux_script.encode("utf-8")).decode("ascii")
    ltx_b64 = base64.b64encode(ltx_script.encode("utf-8")).decode("ascii")
    # Optional inlined HF_TOKEN — bypasses the Kaggle Secrets UI attachment step.
    # The token is base64-encoded into the notebook source. Kernel is private to
    # the user's Kaggle account, so the only person who can read this token is
    # the kernel owner. Trade-off vs Kaggle Secrets: token rotates require a new
    # render (re-push picks up new .env value); Secrets are persistent.
    hf_token_b64 = (
        base64.b64encode(hf_token.encode("utf-8")).decode("ascii")
        if hf_token else ""
    )

    cell1 = "\n".join([
        "# CELL 1: Setup + GPU sanity + write inlined artifacts to /kaggle/working/",
        "# AUTO-GENERATED by pipeline/kaggle_client.py — do NOT hand-edit; the next",
        "# pipeline render overwrites this file with a freshly-built notebook.",
        "# imageio-ffmpeg: LTX mp4 encoder (Kaggle lacks PyAV that torchvision.io",
        "# needs). av: belt-and-suspenders fallback for torchvision.io.write_video.",
        "!pip install -q --upgrade diffusers accelerate imageio-ffmpeg av",
        "",
        "import torch, os, sys, json, base64",
        "",
        "# --- P100 fast-fail: PyTorch 2.x dropped sm_60 support ---",
        "if not torch.cuda.is_available():",
        "    print('FATAL: no GPU allocated. Re-run kernel.')",
        "    sys.exit(1)",
        "_major, _minor = torch.cuda.get_device_capability()",
        "_gpu_name = torch.cuda.get_device_name(0)",
        "_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9",
        "print(f'GPU: {_gpu_name}, compute={_major}.{_minor}, VRAM={_vram_gb:.1f}GB')",
        "if (_major, _minor) < (7, 0):",
        "    print(f'FATAL: GPU compute capability {_major}.{_minor} < 7.0. PyTorch 2.x dropped sm_60 (P100/Pascal). Re-run kernel \\u2014 Kaggle may give a T4 next time.')",
        "    sys.exit(1)",
        "",
        "# --- IP-Adapter Kaggle Dataset mount check ---",
        "_ip_path = '/kaggle/input/xlabs-flux-ip-adapter/ip_adapter.safetensors'",
        "if not os.path.exists(_ip_path):",
        "    print(f'FATAL: IP-Adapter weights missing at {_ip_path}. Dataset subhamkant11/xlabs-flux-ip-adapter must be attached via kernel-metadata.json dataset_sources.')",
        "    sys.exit(1)",
        "print(f'IP-Adapter ready: {os.path.getsize(_ip_path)/1e6:.1f} MB')",
        "",
        "# --- HF_TOKEN (inlined from local .env) for gated FLUX-schnell download ---",
        f"_HF_B64 = '{hf_token_b64}'",
        "if _HF_B64:",
        "    os.environ['HF_TOKEN'] = base64.b64decode(_HF_B64).decode('utf-8')",
        "    print(f'HF_TOKEN inlined into env (length={len(os.environ[\"HF_TOKEN\"])})')",
        "else:",
        "    print('NOTE: no HF_TOKEN inlined — will rely on Kaggle Secrets / public mirror')",
        "",
        "# --- Decode + write inlined artifacts to /kaggle/working/ ---",
        f"_CONFIG_B64 = '{config_b64}'",
        f"_FLUX_B64 = '{flux_b64}'",
        f"_LTX_B64 = '{ltx_b64}'",
        "_RUN_CONFIG = json.loads(base64.b64decode(_CONFIG_B64).decode('utf-8'))",
        "with open('current_run.json', 'w', encoding='utf-8') as _f:",
        "    json.dump(_RUN_CONFIG, _f, ensure_ascii=False, indent=2)",
        "with open('run_flux_phase.py', 'wb') as _f:",
        "    _f.write(base64.b64decode(_FLUX_B64))",
        "with open('run_ltx_phase.py', 'wb') as _f:",
        "    _f.write(base64.b64decode(_LTX_B64))",
        "print(f'Setup OK. run_id={_RUN_CONFIG[\"run_id\"]}, scenes={len(_RUN_CONFIG[\"scenes\"])}, motion_shots={len(_RUN_CONFIG[\"requires_motion\"])}')",
    ])

    cell2 = "\n".join([
        "# CELL 2: FLUX-schnell + IP-Adapter Master Anchor pass",
        "# Subprocess invocation so VRAM is fully reclaimed before the LTX phase",
        "# (PyTorch's CUDA allocator fragments aggressively across model swaps).",
        "# Note: !python's exit code is NOT propagated to the cell — Kaggle would",
        "# report the cell as COMPLETE even if the subprocess crashed. Use",
        "# subprocess.run(check=True) instead so any FLUX failure (e.g. HF 401",
        "# on gated repo) raises and the kernel status reflects reality.",
        "# ANTI-HANG (2026-07-05): hard timeout. v43 hung 3h on an un-timeout'd",
        "# IP-Adapter CLIP download. A TimeoutExpired here fails the cell -> the",
        "# kernel ends with status=error in bounded time instead of burning the",
        "# whole weekly GPU quota. Ceiling read from run_config (default 1800s).",
        "import subprocess, sys",
        "_flux_to = int(_RUN_CONFIG.get('flux_timeout_s', 1800))",
        "try:",
        "    _r = subprocess.run([sys.executable, 'run_flux_phase.py'], timeout=_flux_to)",
        "except subprocess.TimeoutExpired:",
        "    raise RuntimeError(f'run_flux_phase.py exceeded {_flux_to}s hard timeout — aborting (anti-hang guard)')",
        "if _r.returncode != 0:",
        "    raise RuntimeError(f'run_flux_phase.py exited with code {_r.returncode}')",
    ])

    cell3 = "\n".join([
        "# CELL 3: LTX-Video native-portrait I2V pass (pristine VRAM after FLUX exits)",
        "# ANTI-HANG: same hard-timeout guard as CELL 2. LTX exits 0 even on",
        "# per-clip failure (stills are preserved), so a non-zero code here is a",
        "# real crash; a timeout means a clip stalled and we abort in bounded time.",
        "import subprocess, sys",
        "_ltx_to = int(_RUN_CONFIG.get('ltx_timeout_s', 1800))",
        "try:",
        "    _r = subprocess.run([sys.executable, 'run_ltx_phase.py'], timeout=_ltx_to)",
        "except subprocess.TimeoutExpired:",
        "    print(f'run_ltx_phase.py exceeded {_ltx_to}s — motion aborted, stills preserved')",
        "    _r = None",
        "if _r is not None and _r.returncode != 0:",
        "    raise RuntimeError(f'run_ltx_phase.py exited with code {_r.returncode}')",
    ])

    cell4 = "\n".join([
        "# CELL 4: Output verification",
        "!ls -la /kaggle/working/*.jpg /kaggle/working/*.mp4 2>/dev/null || echo 'no jpg/mp4 outputs'",
        "print('Kernel processing complete.')",
    ])

    return {
        "cells": [
            _make_code_cell(cell1),
            _make_code_cell(cell2),
            _make_code_cell(cell3),
            _make_code_cell(cell4),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


async def push_kernel_with_run_config(
    kernel_dir: Path,
    run_config: dict[str, Any],
) -> int:
    """Build a self-contained `notebook.ipynb` with `run_config` and the
    two phase scripts (`run_flux_phase.py`, `run_ltx_phase.py`) inlined
    into CELL 1 as base64 strings, write it into `kernel_dir`, then run
    `kaggle kernels push -p kernel_dir`.

    Why inlined: `kaggle kernels push` uploads ONLY the file specified by
    `kernel-metadata.json`'s `code_file` field — auxiliary files in the
    same folder are silently dropped. The 2026-06-15 FIFA test pulled a
    fresh copy of the kernel from Kaggle and confirmed Kaggle had only 2
    files (notebook.ipynb + kernel-metadata.json), missing the 3
    auxiliary files we expected to ride along. The inlined-notebook
    architecture sidesteps this entirely.

    Returns the new kernel version number (parsed from the CLI's stdout).

    Raises KaggleClientError on subprocess failure, timeout, or unparseable
    output. The caller is expected to fall back to the Cloudflare cascade.

    Concurrency: the caller MUST serialize calls across languages —
    writing `notebook.ipynb` is not atomic with the push, so overlapping
    calls could clobber the file mid-push, and the push itself is an
    HTTP request whose new kernel version queues + replaces any prior
    queued version for the same slug.
    """
    if not kernel_dir.exists() or not kernel_dir.is_dir():
        raise KaggleClientError(f"kernel_dir does not exist: {kernel_dir}")
    if not (kernel_dir / "kernel-metadata.json").exists():
        raise KaggleClientError(
            f"kernel_dir missing kernel-metadata.json: {kernel_dir}"
        )

    # Read the two source-of-truth phase scripts from disk. They stay on
    # local disk (NOT uploaded by `kaggle kernels push`) and get inlined
    # into the generated notebook as base64.
    flux_path = kernel_dir / "run_flux_phase.py"
    ltx_path = kernel_dir / "run_ltx_phase.py"
    if not flux_path.exists():
        raise KaggleClientError(f"missing required phase script: {flux_path}")
    if not ltx_path.exists():
        raise KaggleClientError(f"missing required phase script: {ltx_path}")

    flux_script = flux_path.read_text(encoding="utf-8")
    ltx_script = ltx_path.read_text(encoding="utf-8")

    # Pull HF_TOKEN from local environment (typically loaded from .env by the
    # driver via dotenv) so the kernel can authenticate against gated
    # HuggingFace repos like black-forest-labs/FLUX.1-schnell. If absent, the
    # kernel will fail loudly in CELL 2 with our explicit FATAL message.
    hf_token = (os.environ.get("HF_TOKEN") or "").strip() or None

    # Build the self-contained notebook with everything inlined into CELL 1
    notebook = _build_inlined_notebook(
        run_config, flux_script, ltx_script,
        hf_token=hf_token,
    )

    # Overwrite notebook.ipynb in the kernel folder. This file is the
    # ONLY artifact `kaggle kernels push` actually uploads. Source scripts
    # + this generated notebook coexist in the folder; the CLI ignores
    # everything except the file named by kernel-metadata.json:code_file.
    notebook_path = kernel_dir / "notebook.ipynb"
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    # Push (uploads + queues a fresh kernel run with new version number)
    result = await asyncio.to_thread(
        _run_kaggle_cli,
        ["kernels", "push", "-p", str(kernel_dir)],
        timeout_s=_PUSH_TIMEOUT_S,
    )

    # Stdout shape (Kaggle CLI):
    #   "Kernel version 7 successfully pushed.  Please check progress at https://..."
    # OR
    #   "Your kernel was successfully pushed to Kaggle. Kernel URL: ...kernel/..."
    version = _parse_version_from_push_stdout(result.stdout)
    return version


def _parse_version_from_push_stdout(stdout: str) -> int:
    """Extract the kernel version number from `kaggle kernels push` output.
    Returns 0 if the version isn't explicit (some CLI versions don't print
    it). Caller can still poll/download by kernel_ref alone — the
    downstream `kaggle kernels output` uses the LATEST version implicitly
    when version isn't passed."""
    m = re.search(r"[Kk]ernel\s+version\s+(\d+)", stdout)
    if m:
        return int(m.group(1))
    return 0  # implicit-latest sentinel


async def is_p100_failure(kernel_ref: str) -> bool:
    """After a kernel run ends with `status=error`, check whether the
    failure was specifically the CELL 1 P100 fast-fail check
    (`compute capability < 7.0`). Returns True iff so — caller can then
    retry the kernel push in hope of T4 allocation on the next attempt.

    Why this matters: Kaggle Free Tier randomly allocates T4 OR P100 on
    `enable_gpu: true`. T4 (sm_75) works with our PyTorch install; P100
    (sm_60) doesn't. We can't control the allocation but we CAN cheaply
    retry: each P100 fast-fail costs ~60s of GPU quota, vs ~17 min for a
    real FLUX+LTX run, so 5 retries cost < 1 of a real run worth of quota.

    Returns False on:
      - genuine errors (CELL 2/3/4 failed for some other reason)
      - log-fetch failures (be conservative — don't retry blindly)
    """
    try:
        result = await asyncio.to_thread(
            _run_kaggle_cli,
            ["kernels", "logs", kernel_ref],
            timeout_s=30,
        )
    except KaggleClientError:
        return False  # can't fetch logs → conservative: don't retry
    if not result.stdout:
        return False  # CLI returned empty stdout → conservative
    # Try parsing as JSON (Kaggle CLI 2.x returns a JSON array of stream
    # entries). If parse fails, fall through to a raw-text grep on the
    # whole stdout as a defensive backup.
    try:
        log = json.loads(result.stdout)
        all_text = "\n".join(
            e.get("data", "") for e in log if isinstance(e, dict)
        )
    except (json.JSONDecodeError, TypeError):
        all_text = result.stdout
    # The literal "compute capability" + "< 7.0" pair comes from CELL 1's
    # FATAL print and is specific enough not to false-positive on unrelated
    # errors. Both markers must be present to confirm a P100 fast-fail.
    return ("compute capability" in all_text) and ("< 7.0" in all_text)


async def poll_kernel(
    kernel_ref: str,
    *,
    poll_interval_s: int = 60,
    timeout_s: int = 5400,
) -> dict[str, Any]:
    """Loop on `kaggle kernels status <kernel_ref>` until the latest run
    terminates. Default timeout 90 min — first-run kernels need ~25-35 min
    of model downloads (FLUX-schnell ~7 GB + LTX-Video ~28 GB) BEFORE any
    inference. After the first successful kernel run, Kaggle caches model
    weights per-kernel-slug so subsequent runs only need ~17-20 min of
    inference. 90 min ceiling covers the cold-cache worst case + Kaggle
    free-tier queue wait (5-10 min).

    Returns: dict with keys `{"status": <complete|error|cancelled>,
                              "message": <CLI message string>}`.

    Raises KaggleClientError on timeout or unparseable status output.
    """
    deadline = asyncio.get_event_loop().time() + timeout_s
    last_status = "unknown"
    poll_count = 0
    consecutive_cli_errors = 0
    _MAX_CONSECUTIVE_CLI_ERRORS = 5  # tolerate ~5 min of network/CLI blips before giving up

    while True:
        poll_count += 1
        try:
            result = await asyncio.to_thread(
                _run_kaggle_cli,
                ["kernels", "status", kernel_ref],
                timeout_s=_STATUS_TIMEOUT_S,
            )
            consecutive_cli_errors = 0  # success — reset error counter
        except KaggleClientError as e:
            # 2026-06-29 fix: transient CLI errors (Connection aborted /
            # RemoteDisconnected / JSON parse failures on Kaggle's API
            # response) used to bubble up to the caller's retry loop, which
            # pushed a NEW kernel version that CANCELLED the still-running
            # one. We threw away ~45 min of real T4 work chasing phantom
            # client-side errors. Now: count consecutive CLI errors, sleep
            # the normal poll interval, and retry the status check on the
            # SAME kernel. Only bubble up if many consecutive errors happen
            # AND the deadline is approaching.
            consecutive_cli_errors += 1
            print(f"    [kaggle] poll #{poll_count}: CLI error ({str(e)[:120]}) "
                  f"— consecutive #{consecutive_cli_errors}/{_MAX_CONSECUTIVE_CLI_ERRORS}, "
                  f"retrying poll (kernel may still be running)")
            if consecutive_cli_errors >= _MAX_CONSECUTIVE_CLI_ERRORS:
                raise KaggleClientError(
                    f"poll for {kernel_ref} hit {_MAX_CONSECUTIVE_CLI_ERRORS} consecutive "
                    f"CLI errors. Network / Kaggle API may be down. Last error: {e}"
                ) from e
            # Sleep then retry the status call WITHOUT pushing a new kernel
            if asyncio.get_event_loop().time() + poll_interval_s > deadline:
                raise KaggleClientError(
                    f"kernel {kernel_ref} poll exhausted timeout during CLI-error "
                    f"recovery (last error: {e})"
                ) from e
            await asyncio.sleep(poll_interval_s)
            continue
        # Status output shape (varies by CLI version):
        #   CLI 2.1.x: "<kernel_ref> has status \"KernelWorkerStatus.RUNNING\""
        #              "<kernel_ref> has status \"KernelWorkerStatus.COMPLETE\""
        #   older:     "<kernel_ref> has status \"running\""
        #              "<kernel_ref> has status \"complete\""
        # Match the trailing state token (after optional "KernelWorkerStatus." prefix),
        # lowercase it so the terminal-state check below works uniformly.
        m = re.search(
            r'status\s+"?(?:KernelWorkerStatus\.)?([a-zA-Z_]+)"?',
            result.stdout,
        )
        status = m.group(1).lower() if m else "unknown"

        if status != last_status:
            print(f"    [kaggle] poll #{poll_count}: status={status} ({kernel_ref})")
            last_status = status

        if status in ("complete", "error", "cancelled", "failed"):
            return {"status": status, "message": result.stdout.strip()}

        # Timeout check before sleeping (avoids one final wasted sleep)
        if asyncio.get_event_loop().time() + poll_interval_s > deadline:
            raise KaggleClientError(
                f"kernel {kernel_ref} did not complete within {timeout_s}s "
                f"(last status: {status}). Caller should fall back to "
                f"Cloudflare cascade."
            )

        await asyncio.sleep(poll_interval_s)


async def download_output(
    kernel_ref: str,
    target_dir: Path,
    *,
    version: int | None = None,  # legacy param — IGNORED in CLI 2.2+
) -> list[Path]:
    """Run `kaggle kernels output <kernel_ref> [-v <version>] -p <target_dir>`.
    Downloads everything from /kaggle/working/ of the latest (or specified)
    kernel run. Returns the list of downloaded file paths.

    For the curiosity_shorts pipeline, target_dir is typically
    `cache/<run_id>/visuals/` so files match the existing checkpoint
    naming convention `scene_NN_shot_MM.{jpg,mp4}`.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle CLI 2.2+ removed `-v <version>` from `kernels output` — it now
    # always downloads the latest version's outputs. We keep the `version`
    # kwarg in our signature for back-compat but ignore it. Pass --force so
    # the CLI doesn't skip files that already exist locally (idempotent
    # re-downloads when the pipeline reruns the same run_id).
    args = ["kernels", "output", kernel_ref, "-p", str(target_dir), "--force"]

    # The CLI's print() of progress lines crashes with 'charmap' codec on
    # Windows when output contains Unicode (ANSI progress bars / non-ASCII
    # filenames). The crash happens AFTER the actual file downloads succeed,
    # so we tolerate non-zero exit codes IFF files actually landed in
    # target_dir. _run_kaggle_cli raises on non-zero — wrap to inspect.
    try:
        await asyncio.to_thread(
            _run_kaggle_cli,
            args,
            timeout_s=_DOWNLOAD_TIMEOUT_S,
        )
    except KaggleClientError as e:
        # Tolerate the crash IFF files actually landed
        downloaded_now = [p for p in target_dir.iterdir()
                          if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4")]
        if not downloaded_now:
            raise  # genuine failure — no files came down
        print(f"    [kaggle] download CLI exited non-zero ({str(e)[:120]}) but "
              f"{len(downloaded_now)} target file(s) landed — treating as success")

    # Enumerate everything that landed in target_dir
    downloaded = sorted(
        p for p in target_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".mp4")
    )
    return downloaded


# ── CLI: smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick smoke test — just verifies the kaggle CLI is reachable and
    creds are valid by listing the user's kernels.

        python -m pipeline.kaggle_client

    On success: prints "kaggle CLI OK" + first line of kernels list.
    On failure: prints the KaggleClientError so the user can diagnose
    (typically missing kaggle.json or wrong KAGGLE_CONFIG_DIR).
    """
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    async def _smoke():
        try:
            result = await asyncio.to_thread(
                _run_kaggle_cli,
                ["kernels", "list", "-m", "--page-size", "1"],
                timeout_s=30,
            )
            print("kaggle CLI OK")
            for line in result.stdout.splitlines()[:3]:
                print(f"  {line}")
        except KaggleClientError as e:
            print(f"kaggle CLI ERROR: {e}")
            sys.exit(1)

    asyncio.run(_smoke())
