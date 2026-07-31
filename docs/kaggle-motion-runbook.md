# Kaggle Motion Runbook (Phase 30–31)

Free image-to-video (I2V) **motion clips** for Mahabharata Shorts, generated on
Kaggle's free T4 GPUs (LTX-Video, 512×768 portrait, 65 frames). Motion is an
**enhancement layer**: any clip that fails falls back to a Ken-Burns pan of the
still, so the daily video always ships.

## What it is

- Still frames are rendered by the normal free-FLUX cascade; selected scenes are
  then **animated** on Kaggle.
- Kaggle is **async + slow** (~10–15 min warm, ~10–25 min cold per kernel), so it
  is decoupled from the 29-min GHA job cap by being **resumable**: a job killed
  mid-poll resumes by polling the already-pushed kernels — it never re-pushes a
  running kernel, so no GPU quota is wasted. This rides the existing single-run
  retry chain (primary + 3 retry siblings); **no cron/workflow changes** are
  needed for the wait.
- **6-slot pool**: 3 Kaggle accounts × 2 kernels animate all scenes in parallel.

Core code: `pipeline/maha_kaggle_motion.py` (`generate_motion_clips_pool`,
`_build_maha_pool`, `_kernel_task`) — mirrors the proven `pipeline/wuxia_motion.py`
pattern and reuses `pipeline/wuxia_kaggle.py` (account-aware push/poll/download).

## The pool (kernels + accounts)

| Account | Cred file (repo root, gitignored) | Kernel slugs | Local folders |
|---|---|---|---|
| subhamkant11 | `kaggle.json` (ambient) | `maha-i2v`, `maha-i2v-2` | `kaggle_notebooks/maha-i2v{,-2}` |
| subhamkant9 | `sk9_kaggle.json` | `maha-i2v`, `maha-i2v-2` | `kaggle_notebooks/maha-i2v-sk9{,-2}` |
| vyasaai | `vyasa_ai_kaggle.json` | `maha-i2v`, `maha-i2v-2` | `kaggle_notebooks/maha-i2v-vyasa{,-2}` |

The pool auto-degrades: an account whose cred file is absent is skipped, so with
only `kaggle.json` present it runs a 2-kernel pool; with all three it runs 6.

`notebook.ipynb` in each folder is a **build artifact** (rebuilt on every push,
gitignored) — only `kernel-metadata.json` + `run_ltx_phase.py` are committed.

## Enable knobs (env)

| Env | Default | Meaning |
|---|---|---|
| `ENABLE_AI_CLIPS` | `false` | Master switch for any AI motion. Must be `true`. |
| `MAHA_MOTION_KAGGLE` | `false` | Route motion to the Kaggle pool (vs the fal/HF cascade). |
| `AI_CLIP_SCENES` | *(unset → all scenes)* | `all`, or a csv like `0,4,8` to limit which scenes get motion. |
| `MAHA_KERNEL_POOL` | `main,sk9,vyasa` | Which accounts to use. |
| `MAHA_MOTION_MAX_ATTEMPTS` | `4` | Bounded re-push per kernel on terminal failure (never on a running one). |
| `FACE_SELECT` | `true` | Set `false` for LOCAL Windows runs (OpenCV face model can native-crash). |
| `MAHA_LTX_FRAMES` / `MAHA_LTX_STEPS` / `MAHA_LTX_GUIDANCE` | `65` / `40` / `2.5` | LTX tuning (the proven winner). |

## Run it LOCALLY (recommended first)

```bash
ENABLE_AI_CLIPS=true MAHA_MOTION_KAGGLE=true FACE_SELECT=false LOCAL_ONLY=true \
  python main.py hi
```

- `LOCAL_ONLY=true` skips YouTube + Instagram upload (renders `output/video_hi_*.mp4`).
- Requires the cred files (`kaggle.json` etc.) present at the repo root and the
  Kaggle CLI authenticated.
- **Resume behavior**: kill it mid-poll, re-run with the same `PIPELINE_RUN_ID`
  (or the same date/hour so the cache dir matches) → it logs
  `RESUME poll … (no re-push)` and continues. This is exactly what the GHA retry
  chain does.

## Run the GHA private test

GitHub → **Actions → "Mahabharata YouTube Bot" → Run workflow**:

- `target: hi`
- `enable_motion: true`
- `yt_privacy: private`

This writes the Kaggle cred file(s) from secrets, sets `ENABLE_AI_CLIPS` +
`MAHA_MOTION_KAGGLE`, renders with motion, and uploads **privately** for review.
The cron (no inputs → `enable_motion` false) is unaffected and keeps shipping
static Ken-Burns video.

### Required GitHub secrets

| Secret | Needed for | How to make it |
|---|---|---|
| `KAGGLE_JSON_B64` | main account (2 kernels) — **required** | `base64 -w0 kaggle.json` |
| `SK9_KAGGLE_JSON_B64` | +subhamkant9 (→ 4 kernels) — optional | `base64 -w0 sk9_kaggle.json` |
| `VYASA_KAGGLE_JSON_B64` | +vyasaai (→ full 6 kernels) — optional | `base64 -w0 vyasa_ai_kaggle.json` |

Paste each base64 string into a new repository secret. With only
`KAGGLE_JSON_B64` the test runs 2-kernel (all scenes batched across 2 slots,
slower); add the other two for the full 6-slot pool.

## One-time kernel registration (already done)

A fresh Kaggle slug **cannot be created with the full (~MB) payload** — Kaggle
returns `400 Bad Request`. Register with a **minimal notebook first**, then real
pushes update it. All 6 slugs are already registered. If you ever add a new
account/kernel, push a trivial `notebook.ipynb` once before the first real run.
(Also: conditioning stills are downscaled to 512×768 before base64 so the pushed
notebook stays small — a full-res seed re-triggers the 400.)

## Failsafe / cost

- Bounded to `MAHA_MOTION_MAX_ATTEMPTS` (4) re-pushes per kernel, **only on
  terminal failure** — never on a still-running kernel. After that the kernel's
  scenes fall back to Ken Burns.
- **Honest limit**: an already-*running* Kaggle kernel can't be cancelled via the
  API. The guard prevents *re-*spend on retries, not an in-flight run's GPU-seconds.
- To hard-disable motion, leave `enable_motion` off (GHA) or unset
  `MAHA_MOTION_KAGGLE` (local) — the pipeline reverts to static Ken Burns.

## Troubleshooting

- **"no Kaggle kernels available — all Ken Burns"** → no cred files present (GHA:
  the `*_B64` secret(s) aren't set; local: the `*.json` files are missing).
- **Push `400 Bad Request`** → the notebook payload is too large (seed not
  downscaled) or a brand-new slug needs the minimal-notebook registration first.
- **All clips missing but kernels "complete"** → the kernel ran but LTX failed the
  render (it always exits 0 so stills survive); check the per-kernel `*.log` in the
  download dir. Occasional single-clip failure is expected → Ken Burns.
- **Local native crash / silent exit 1 during stills** → OpenCV face model on
  Windows; set `FACE_SELECT=false`.
