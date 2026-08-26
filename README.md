

# Whisper-UI

Speech-to-text system using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(large-v3, int8_float16 on GPU) with speaker diarization via
[pyannote-audio](https://github.com/pyannote/pyannote-audio),
a [FastAPI](https://fastapi.tiangolo.com/) + [htmx](https://htmx.org/) + [Alpine.js](https://alpinejs.dev/) web interface,
and Docker deployment (NVIDIA GPU / CPU / AMD ROCm).

> **Note:** The UI is in Traditional Chinese (繁體中文).

## Features

- Upload audio/video files for transcription
- Transcribe directly from a URL: YouTube videos and playlists, X (Twitter)
  posts, and Google Drive files are downloaded by the worker
- Batch upload with automatic filtering of unsupported files
- Speaker diarization with pyannote speaker-diarization-3.1 (optional)
- Optional LLM text correction via Ollama (a Gemma model, per-job toggle)
- Real-time progress tracking via Redis
- Export to SRT, VTT, TXT, JSON, DOCX
- Batch download of results as ZIP
- Docker Compose deployment with NVIDIA GPU, CPU, and AMD ROCm profiles

**Supported formats:** `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.wma`, `.aac`, `.opus`, `.mp4`, `.webm`, `.mkv`

## Architecture

```text
+------------------+     +------------------+     +------------------+
|     FastAPI      |     |      Redis       |     |     Worker       |
|  Frontend (htmx  |<--->|   (queue+state)  |<--->|  (RQ + Pipeline) |
|  + Alpine.js)    |     |                  |     |  (GPU or CPU)    |
+------------------+     +------------------+     +------------------+
        |                                                  |
        +------ Shared Volume: app-data (uploads/outputs/db) ------+
```

## Quick Start

### Prerequisites

- Docker and Docker Compose

**For GPU deployment (recommended for speed):**

- NVIDIA GPU with 8GB+ VRAM
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

**For AMD GPU deployment (ROCm):**

- AMD GPU with ROCm support (tested on Radeon 8060S / gfx1151, ROCm 7.2)
- ROCm on the host: the worker image bundles the ROCm 7.2 runtime, but the
  host must provide the `amdgpu` kernel driver and expose `/dev/kfd` + `/dev/dri`

**Optional (for speaker diarization):**

- HuggingFace token ([get one here](https://huggingface.co/settings/tokens))
- Accept the model agreements:
  - <https://huggingface.co/pyannote/speaker-diarization-3.1>
  - <https://huggingface.co/pyannote/segmentation-3.0>

### GPU Deployment

```bash
cp .env.example .env
# Edit .env: set HF_TOKEN if you need speaker diarization

docker compose --profile gpu up -d
```

### CPU Deployment

```bash
cp .env.example .env
# Edit .env: uncomment and set WHISPER_MODEL=base
# (smaller models like 'base' or 'small' are recommended for CPU)

docker compose --profile cpu up -d
```

### AMD GPU Deployment (ROCm)

Transcription runs on whisper.cpp (HIP); alignment and speaker diarization run
on PyTorch-ROCm. CTranslate2 / faster-whisper has no ROCm backend, so the
`whisperx` transcription path cannot drive an AMD GPU — the `rocm` profile uses
whisper.cpp instead and selects it automatically (`TRANSCRIBE_BACKEND=whispercpp`).

```bash
cp .env.example .env
# Set HF_TOKEN if you need speaker diarization.
# Match the device-passthrough groups to your host: getent group video render
#   VIDEO_GID / RENDER_GID  (defaults 44 / 993 suit a Strix Halo box)
# For a non-gfx1151 GPU, build whisper.cpp for it: AMDGPU_TARGETS=gfx1100

docker compose --profile rocm up -d
```

The first run compiles whisper.cpp for the target GPU arch and downloads the
GGML model (`ggml-large-v3.bin`), so allow extra time. On every `DEVICE=rocm`
worker the MIOpen backend is disabled (native HIP kernels are used instead) to
avoid a known `miopenStatusUnknownError` in pyannote's segmentation model,
first observed on gfx1151.

The whisper.cpp backend ships with hallucination guards enabled: Silero VAD
pre-segmentation skips non-speech (silence / music that otherwise produces
looping subtitle-style hallucinations) and cross-window text conditioning is
disabled so a bad window cannot contaminate the rest of the transcript.
Settings specific to this backend:

| Variable                 | Default                  | Description                                                              |
| ------------------------ | ------------------------ | ------------------------------------------------------------------------ |
| `WHISPERCPP_BINARY`      | `whisper-cli`            | whisper.cpp CLI binary name or path inside the worker image              |
| `WHISPERCPP_THREADS`     | `0`                      | CPU threads for whisper-cli (`0` = let it decide)                        |
| `WHISPERCPP_VAD`         | `true`                   | Silero VAD pre-segmentation (the main anti-hallucination guard)          |
| `WHISPERCPP_VAD_MODEL`   | `ggml-silero-v5.1.2.bin` | VAD model file; auto-downloaded and cached on the model-cache volume     |
| `WHISPERCPP_MAX_CONTEXT` | `0`                      | Text-context tokens carried across windows (`0` = off, `-1` = unlimited) |

Open <http://localhost:8080> in your browser.

> **Production note:** the bundled Redis starts without authentication
> when `REDIS_PASSWORD` is unset — compose always passes
> `--requirepass "${REDIS_PASSWORD:-}"`, and an empty password means no auth.
> For any deployment reachable beyond the local Docker network — even on a
> trusted LAN — set `REDIS_PASSWORD` in `.env` before bringing the stack up.

### Pre-built Images

Pre-built Docker images are published to GHCR on each release.
`docker compose up` pulls them automatically; if unavailable, it falls back to a local build.

| Image                                      | Description           |
| ------------------------------------------ | --------------------- |
| `ghcr.io/fdff87554/whisper-ui-frontend`    | FastAPI web interface |
| `ghcr.io/fdff87554/whisper-ui-worker`      | GPU worker (CUDA)     |
| `ghcr.io/fdff87554/whisper-ui-worker-cpu`  | CPU worker            |
| `ghcr.io/fdff87554/whisper-ui-worker-rocm` | AMD GPU worker (ROCm) |

> The ROCm worker image (~38 GB) is published only on **release tags**, not on
> every `main` push. Until a release exists, `docker compose --profile rocm up`
> falls back to building it locally.

**Pin a specific version** by setting `WHISPER_UI_VERSION` in your `.env` file:

```bash
WHISPER_UI_VERSION=2.13.0
```

**Build locally** instead of pulling (optional):

```bash
docker compose --profile gpu build
```

## Configuration

All settings are configured via environment variables (`.env` file):

| Variable                      | Default                             | Description                                                                                                                                                                                        |
| ----------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WHISPER_MODEL`               | `large-v3`                          | Whisper model variant (see model list below)                                                                                                                                                       |
| `COMPUTE_TYPE`                | `int8_float16` (GPU) / `int8` (CPU) | CTranslate2 compute type                                                                                                                                                                           |
| `DEVICE`                      | `auto`                              | Inference device; compose profiles set `cuda` (GPU) / `rocm` (AMD) / `cpu` (CPU)                                                                                                                   |
| `TRANSCRIBE_BACKEND`          | `whisperx`                          | `whisperx` (CTranslate2, CUDA/CPU) or `whispercpp` (whisper.cpp HIP; set by the rocm profile)                                                                                                      |
| `BATCH_SIZE`                  | `4`                                 | Transcription batch size (whisperx backend only)                                                                                                                                                   |
| `LANGUAGE`                    | `zh`                                | Default language code                                                                                                                                                                              |
| `HF_TOKEN`                    | (empty)                             | HuggingFace token for speaker diarization                                                                                                                                                          |
| `DIARIZATION_DEFAULT_ENABLED` | `true`                              | Initial state of the upload-form diarization toggle (only when `HF_TOKEN` is set). Set `false` so new uploads default to no diarization — the slowest stage — with per-job opt-in still available. |
| `MAX_UPLOAD_SIZE`             | `2147483648`                        | Per-file upload cap in bytes (2 GB); also caps Google Drive downloads                                                                                                                              |
| `YOUTUBE_MAX_DURATION`        | `14400`                             | Longest accepted YouTube video in seconds (4 h); live streams are always rejected                                                                                                                  |
| `TWITTER_MAX_DURATION`        | `14400`                             | Longest accepted X (Twitter) video in seconds                                                                                                                                                      |
| `TWITTER_COOKIES_FILE`        | (empty)                             | Container path to a cookies.txt for login-walled X posts (see `.env.example`)                                                                                                                      |
| `PIP_INDEX_URL`               | (empty)                             | Custom PyPI mirror for Docker builds                                                                                                                                                               |
| `WHISPER_UI_VERSION`          | `latest`                            | Docker image version tag to pull                                                                                                                                                                   |

**Whisper models:** `tiny`, `tiny.en`, `base`, `base.en`, `small`, `small.en`, `medium`, `medium.en`, `large-v1`, `large-v2`, `large-v3`, `large-v3-turbo`

> **Tip:** For CPU deployment, use smaller models (`base`, `small`) for reasonable
> processing times. GPU deployment with `large-v3` and `int8_float16` gives the
> best accuracy-to-speed ratio.

### Multi-user authentication

The web tier ships with a lightweight session-cookie authentication
layer so multiple people can share one deployment without seeing each
other's transcripts. There is no built-in default account; the **first
visitor** to a freshly-started instance is bounced to a one-shot
`/register?bootstrap=1` page that creates the system's first admin.
Every subsequent visit goes through `/login` or self-service `/register`.

**Required env vars:**

| Variable                       | Default | Description                                                                                                                                                                                                                                                       |
| ------------------------------ | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SESSION_SECRET`               | (empty) | Cookie signing key. Generate once with `openssl rand -hex 32` and keep it stable. Empty value falls back to a per-process random secret (dev only).                                                                                                               |
| `SESSION_HTTPS_ONLY`           | `false` | Set to `true` for any production deployment so the session cookie carries the `Secure` flag (otherwise it is sent in clear text on the proxy↔client hop).                                                                                                         |
| `ALLOW_REGISTRATION`           | `true`  | When `false`, self-service `/register` is closed after the bootstrap admin exists, so only an admin can create accounts. The first admin is always creatable.                                                                                                     |
| `MAX_LOGIN_ATTEMPTS`           | `5`     | After this many failed logins per username, the next attempt is blocked for `LOGIN_LOCKOUT_SECONDS` regardless of password correctness.                                                                                                                           |
| `MAX_LOGIN_ATTEMPTS_PER_IP`    | `20`    | Separate, higher per-IP threshold. The default is comfortable for a small office sharing one NAT egress IP; raise it for larger NATs, or enable `TRUST_PROXY_HEADERS` so each user is rate-limited by their real address.                                         |
| `LOGIN_LOCKOUT_SECONDS`        | `900`   | Window length for both per-user and per-IP counters.                                                                                                                                                                                                              |
| `MAX_REGISTER_ATTEMPTS_PER_IP` | `10`    | Per-IP cap on open (non-bootstrap) registrations within `LOGIN_LOCKOUT_SECONDS`, bounding scripted account creation and username-enumeration probing. The first-run bootstrap admin is never throttled.                                                           |
| `TRUST_PROXY_HEADERS`          | `false` | When `true`, the app uses the left-most `X-Forwarded-For` entry as the client IP (for rate-limit bucketing) and accepts `X-Forwarded-Host` as a valid CSRF host. **Only enable behind a controlled reverse proxy that resets these headers** — see warning below. |

**Roles:** every account is either a regular user or an admin. Admins
gain access to `/admin/users` (create / deactivate / promote / reset
password) and `/admin/jobs` (every user's transcripts, including legacy
rows from pre-auth deployments). All other capabilities — upload,
transcript view, export — are identical for both roles.

**Operational notes:**

- Admin resets a forgotten password directly (no SMTP required). Pass
  the new password to the user through any out-of-band channel.
- An admin can also reset their own password through the same UI; the
  action is logged at WARNING level and invalidates their current
  session, so they must log in again with the new password.
- A locked-out account auto-clears after the rate-limit window expires.
  Unblock immediately with `redis-cli DEL auth:rl:user:<username>` or
  `auth:rl:ip:<address>`.
- The middleware enforces CSRF by comparing `Origin` (falling back to
  `Referer`) against `request.url.netloc`. The reverse proxy must
  preserve the original `Host` header (nginx: `proxy_set_header Host
$host;`); Traefik does this by default.
- **Reverse-proxy hardening**: if `TRUST_PROXY_HEADERS=true`, the proxy
  **must** strip any client-supplied `X-Forwarded-For` and
  `X-Forwarded-Host` and set them itself — otherwise a hostile client
  can spoof its IP and the host name to defeat both rate-limit and
  CSRF protections. nginx example: `proxy_set_header X-Forwarded-For
$remote_addr;` (overrides any client-sent header).
- Existing jobs from pre-auth deployments are stored with `owner_id IS NULL`
  and remain visible only on the admin `/admin/jobs` view.

### Worker topology and queues (advanced)

Each upload is dispatched as an RQ **DAG of sub-jobs** (one per pipeline
stage) rather than a single monolithic task. Sub-jobs are routed to
resource-class queues so that, in a scaled topology (below), a long-running
IO or network stage need not block a GPU worker from picking up the next job:

| Queue         | Stages                            |
| ------------- | --------------------------------- |
| `whisper:gpu` | `transcribe_align`, `diarize`     |
| `whisper:io`  | `download`, `preprocess`          |
| `whisper:cpu` | `assign_speakers`, `postprocess`  |
| `whisper:llm` | `llm_correction` (optional, slow) |

**Single-container (default).** `docker compose --profile gpu up -d` keeps
the existing behaviour: `worker-gpu` listens to every queue so one
container drains the full pipeline end-to-end. You do not need to touch
any queue variables for this layout.

**Scaled topology.** To stop the GPU worker from picking up IO/LLM work,
add the `io` profile and narrow the GPU worker's queue set:

```bash
# .env
WORKER_GPU_QUEUES="whisper:gpu default"
WORKER_IO_QUEUES="whisper:io whisper:cpu whisper:llm default"

docker compose --profile gpu --profile io up -d
```

`worker-io` is a lightweight CPU container that drains `whisper:io`
(download / preprocess), `whisper:cpu`, and `whisper:llm` in parallel with
`worker-gpu` running transcribe_align / diarize on the GPU. Two jobs enqueued
back to back overlap: job B can be downloading while job A is on the GPU.
(Keeping `whisper:llm` here means the io worker also runs LLM correction; to
isolate a slow LLM onto its own worker, see the AMD example below.)

The optional, slow LLM correction has its own `whisper:llm` queue, so a
dedicated `worker-llm` (the `llm-worker` profile) can run it without blocking
the fast io/cpu finalisation path. Every worker drains `whisper:llm` by
default, so by default a slow LLM still shares a worker with io/cpu; the
isolation is opt-in — run `worker-llm` and drop `whisper:llm` from the other
workers' `WORKER_*_QUEUES` (see the AMD example below).

**AMD / ROCm single-GPU box.** The ROCm worker has a single GPU, so the same
split applies, plus a dedicated LLM worker so a slow Ollama model never blocks
the io/cpu path:

```bash
# .env
WORKER_ROCM_QUEUES="whisper:gpu default"
WORKER_IO_QUEUES="whisper:io whisper:cpu default"
WORKER_LLM_QUEUES="whisper:llm default"

docker compose --profile rocm --profile io --profile llm-worker up -d --scale worker-io=2
```

`worker-io` (×2) and `worker-llm` use the CPU image (none of these stages need
the GPU). `worker-io` drains download / preprocess / assign / postprocess;
`worker-llm` drains the optional LLM correction (HTTP calls to Ollama); the
single `worker-rocm` stays dedicated to transcribe_align / diarize. Do **not**
scale `worker-rocm` past one on a single card — two whisper.cpp / pyannote
processes would contend for the same VRAM and compute queue. Ollama itself
still shares the GPU with the worker; if that contention hurts transcription,
point Ollama at CPU or use a smaller model.

**Multi-GPU hosts.** When the DAG fans out transcribe_align and diarize
as sibling branches they will automatically run in parallel once you
provision more than one GPU worker. Example with two cards:

```bash
# Launch two GPU workers, each pinned to one card.
docker compose --profile gpu up -d --scale worker-gpu=2
# Or run named services and set WORKER_GPU_DEVICE_ID per container.
```

**Upgrading from v1.x to v2.0 (BREAKING).** The legacy single-task
`process_transcription` entry point has been removed in v2.0. Any RQ
sub-jobs enqueued under it from a v1.x worker will fail import when a
v2.0 worker picks them up. To upgrade:

1. Stop accepting new uploads (e.g. take the frontend offline) or wait
   for the dashboard to show no active jobs.
2. Let the existing workers drain whatever is in flight.
3. Pull the v2.0 images and `docker compose --profile gpu up -d`.

If a queue is non-empty when the worker is upgraded, drop the queues
manually before restart: `redis-cli -n 0 FLUSHDB` against the
`REDIS_URL` Redis only clears RQ state (uploads, the SQLite DB, and
saved transcripts are untouched).

### GPU worker resource lifecycle (advanced)

GPU workers (`cuda` / `rocm`) run RQ's non-forking `SimpleWorker`: every job
executes in one long-lived process, because a CUDA/HIP context cannot be
re-initialised in a forked child. CPU workers keep RQ's default forking
`Worker`, where each job runs in a work-horse child that exits on completion, so
the OS reclaims its memory automatically.

The trade-off is that a `SimpleWorker` never exits on its own. Each pipeline
stage releases its model after running (`del` + `gc.collect()` +
`torch.cuda.empty_cache()`), which returns the model weights to the driver's
free pool — but `empty_cache()` cannot destroy the process's CUDA/HIP context or
the mapped GPU runtime libraries (cuBLAS / MIOpen / rocBLAS, etc.). Those stay
resident, so a GPU worker holds a baseline of VRAM and host RSS between jobs
until the process restarts.

`WORKER_MAX_IDLE_TIME` closes that gap: the worker self-exits after a spell with
no work, and `restart: unless-stopped` respawns a fresh process, handing the
context and RSS back to the OS. It triggers only while idle, so a running job is
never interrupted and back-to-back jobs stay warm.

| Variable               | Default | Description                                                                                                                                                                                                                                                                                                                                             |
| ---------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WORKER_MAX_IDLE_TIME` | `300`   | Seconds a GPU worker stays alive with no job before exiting so its `restart` policy respawns a fresh process and frees VRAM + RSS. `0` disables (always-resident). Must be a non-negative integer; an invalid value is ignored with a warning. Defaulted on only for `worker-gpu` / `worker-rocm`; CPU/io/llm workers fork per job and release on exit. |

> A low value (e.g. `60`) releases resources quickly after each session at the
> cost of reloading models on the next job; a high value keeps the worker warm
> for longer. Within a single pipeline run the GPU sub-jobs (transcribe/align
> and diarize) run back-to-back with no idle gap, so the default `300` never
> restarts mid-run. On AMD/ROCm only diarization and alignment run in-process
> (transcription shells out to whisper.cpp, which already frees on exit), so the
> resident baseline is smaller than on CUDA.

### Queue / Timeout tuning (advanced)

The RQ job timeout is derived from the probed audio duration
(`duration * JOB_TIMEOUT_AUDIO_MULTIPLIER`) and clamped into
`[JOB_TIMEOUT_FLOOR, JOB_TIMEOUT_MAX]`. All values below have sensible
defaults for mixed short/long audio on an 8 GB VRAM GPU; override them
only if you routinely process audio longer than 2 hours or need tighter
short-audio SLAs.

| Variable                       | Default | Description                                                                                                                           |
| ------------------------------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `JOB_TIMEOUT_DEFAULT`          | `7200`  | Fallback `job_timeout` (seconds) when audio duration is unknown (e.g., YouTube URL before download).                                  |
| `JOB_TIMEOUT_FLOOR`            | `1800`  | Lower bound on the dynamic timeout so short audio still has room for model loading.                                                   |
| `JOB_TIMEOUT_AUDIO_MULTIPLIER` | `3.0`   | `estimated_timeout = duration * multiplier`. `3.0` covers large-v3 + alignment + diarization with headroom.                           |
| `JOB_TIMEOUT_MAX`              | `28800` | Absolute upper cap (8h). Also the basis for stale-job recovery.                                                                       |
| `STALE_JOB_BUFFER`             | `1800`  | Extra margin (30min) added to `JOB_TIMEOUT_MAX` before the stale-job reaper marks a processing job as dead.                           |
| `REDIS_PROCESSING_EXPIRY`      | `30600` | TTL (seconds) for Redis progress keys. See invariant below.                                                                           |
| `DIARIZE_HEARTBEAT_INTERVAL`   | `30`    | How often the diarize stage re-posts progress so the UI and stale-job reaper can tell the task is still alive. Set to `0` to disable. |

> **Invariants** (validated at startup — the service will refuse to start
> if any of these is violated):
>
> - `0 < JOB_TIMEOUT_FLOOR ≤ JOB_TIMEOUT_DEFAULT ≤ JOB_TIMEOUT_MAX`
> - `JOB_TIMEOUT_AUDIO_MULTIPLIER > 0`
> - `STALE_JOB_BUFFER ≥ 0`
> - `DIARIZE_HEARTBEAT_INTERVAL ≥ 0`
> - `REDIS_PROCESSING_EXPIRY ≥ JOB_TIMEOUT_MAX + STALE_JOB_BUFFER`
>
> When raising `JOB_TIMEOUT_MAX`, you **must** also raise
> `REDIS_PROCESSING_EXPIRY` so the Redis progress key outlives the longest
> possible job window. Otherwise the service will fail to start with a
> `ValidationError` pointing at the violated invariant.

### Optional LLM text correction

Whisper-UI can optionally post-process each transcription through a small
LLM running on [Ollama](https://ollama.com) to fix obvious typos,
homophones and punctuation errors — without rewriting wording or touching
timestamps / speaker labels. The feature is:

- **Per-job** — users tick a checkbox on the upload form. Uninterested
  users see no change.
- **Optional at deployment time** — the bundled Ollama server is a
  separate compose profile. Leaving `OLLAMA_BASE_URL` empty disables the
  feature globally and greys out the UI toggle.
- **Fail-safe** — any network, parsing or validation failure falls back
  to the original segment text. LLM correction can never turn a
  successful transcription into a failed job.

**Option A — bundled Ollama (easiest):**

```bash
# .env
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=gemma4:e4b

docker compose --profile gpu --profile llm up -d
```

The `ollama-pull` init sidecar will wait until Ollama is healthy and then
pull `OLLAMA_MODEL` on first start (~9.6 GB download for `gemma4:e4b`).
Check its exit status with `docker compose ps ollama-pull`; the model is
cached in the `ollama-data` volume so restarts are instant.

**Option B — external Ollama server (no profile needed):**

```bash
# .env
OLLAMA_BASE_URL=http://192.168.1.20:11434
OLLAMA_MODEL=gemma4:e4b

docker compose --profile gpu up -d
# Make sure the model is pulled on the external server yourself:
#   ollama pull gemma4:e4b
```

**Dual-GPU hosts:** the `gpu` profile pins the Whisper worker to
`WORKER_GPU_DEVICE_ID` (default 0); adding the `llm` profile also pins
the bundled Ollama container to `OLLAMA_GPU_DEVICE_ID` (default 1).
Override either variable in `.env` if your topology differs.

> **Operational breaking change (multi-GPU hosts only):** this release
> switches `worker-gpu`'s GPU reservation from `count: 1` (runtime picks
> any available GPU) to `device_ids: ["${WORKER_GPU_DEVICE_ID:-0}"]`
> (pinned to GPU 0 by default). This is required so the `llm` profile can
> reliably split Whisper and Ollama onto different devices. Impact:
>
> - **Single-GPU hosts:** no change.
> - **Multi-GPU hosts using only the `gpu` profile (no `llm`):** the
>   worker now always uses GPU 0 unless you set `WORKER_GPU_DEVICE_ID`
>   explicitly. If you previously relied on the NVIDIA runtime's
>   automatic selection (e.g. to route workloads away from a GPU already
>   held by another container), set `WORKER_GPU_DEVICE_ID` in your `.env`
>   to restore the intended placement.

**Model size vs VRAM:** the default `gemma4:e4b` is ~9.6 GB and wants a 10 GB+
GPU to stay resident. On an 8 GB or shared GPU (e.g. running Whisper on the same
card), use `gemma4:e2b` (~7.2 GB), point `OLLAMA_BASE_URL` at an external Ollama
server, or split Whisper and Ollama onto separate GPUs as above.

> **VRAM retention vs reload latency:** `OLLAMA_KEEP_ALIVE` (default `30m`) sets
> how long Ollama keeps the model resident in VRAM after the last correction
> request. On a GPU shared with Whisper, a long keep-alive holds several GB
> hostage between jobs — lower it (e.g. `5m`, or `0` to unload immediately) to
> free VRAM for transcription; raise it only on a dedicated LLM GPU where reload
> latency matters more than footprint.

**Tuning (all optional):**

| Variable                 | Default      | Description                                                                                                                                                      |
| ------------------------ | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OLLAMA_BASE_URL`        | (empty)      | Empty disables the feature globally. Set to reach a bundled or external Ollama server.                                                                           |
| `OLLAMA_MODEL`           | `gemma4:e4b` | Any Ollama-compatible chat model. Bigger = better accuracy, more VRAM (see note).                                                                                |
| `OLLAMA_KEEP_ALIVE`      | `30m`        | How long Ollama keeps the model loaded in VRAM between requests.                                                                                                 |
| `OLLAMA_REQUEST_TIMEOUT` | `120`        | Per-request timeout in seconds.                                                                                                                                  |
| `LLM_CHUNK_SIZE`         | `8`          | Segments corrected per Ollama request. Larger reduces HTTP overhead.                                                                                             |
| `LLM_CHUNK_CONTEXT`      | `2`          | Neighbor segments attached as read-only context for disambiguation.                                                                                              |
| `LLM_TEMPERATURE`        | `0.1`        | Sampling temperature. Low values keep corrections deterministic.                                                                                                 |
| `OLLAMA_THINK`           | `false`      | Let a thinking-capable model reason before answering. Off is faster and gives cleaner JSON for correction; set `true` only for a reasoning model proven to help. |

### Optional upload retention

Long-running deployments accumulate per-job upload directories under
`data/uploads/`. Set `UPLOAD_RETENTION_DAYS` to have the web app
hourly reclaim the upload directory of any **COMPLETED** job whose
last update is older than the threshold. FAILED jobs are intentionally
preserved so the retry button keeps working — retry reuses the
original upload path and would otherwise fail at the preprocess step.

The DB row and the saved transcript (`data/outputs/<id>/result.json`)
are always kept, so viewer and export routes remain functional. The
"Download Media" button for URL jobs hides itself once the source
media is reclaimed.

```bash
# .env
UPLOAD_RETENTION_DAYS=30

docker compose --profile gpu up -d
```

| Variable                | Default | Description                                                                                                          |
| ----------------------- | ------- | -------------------------------------------------------------------------------------------------------------------- |
| `UPLOAD_RETENTION_DAYS` | `0`     | `0` disables the sweep (legacy behaviour). `>0` reclaims COMPLETED job upload dirs older than that many days hourly. |

### Logging and request tracing

The frontend and worker both initialise stdlib logging from
[`whisper_ui.core.logging_setup`][logging_setup]. Every log line renders
through one formatter so an operator can pipe `docker compose logs` into
`grep` without parsing two different layouts:

```text
2026-05-22T03:14:01+0000 INFO whisper_ui.web.access [req=a3f12bc4 user=alice] method=POST path=/upload status=204 duration_ms=1342 ip=192.168.50.200
```

[logging_setup]: ./src/whisper_ui/core/logging_setup.py

**Request correlation.** `RequestIdMiddleware` reads `X-Request-ID` from
the inbound request (8-64 hex chars; anything else is regenerated) or
generates a fresh 8-char id, then publishes it on contextvars. Every
log line emitted during the request — middleware, route handler,
filestore, exception handler — carries the same `req=<id>` tag. The
response echoes the id back as `X-Request-ID` so an upstream nginx or
browser devtools can join on the same key. When investigating "where
did this upload go?", grep one id and read the full trace.

**Authenticated user tag.** `AuthMiddleware` overlays the resolved
username into the `user=` tag once the session is decoded; pre-auth
and worker-side lines render `user=-`.

**Access log replacement.** The frontend Dockerfile passes
`--no-access-log` to uvicorn; the structured access log emitted from
`RequestIdMiddleware` covers method / path / status / duration_ms / ip
plus the `req=` / `user=` tags. The stock uvicorn access log lacks all
three of those, which is why it is silenced.

**Tunables.**

| Variable    | Default | Description                                                                                                                                                                                                        |
| ----------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `LOG_LEVEL` | `INFO`  | One of `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`; invalid values fall back.                                                                                                                                  |
| `LOG_JSON`  | `false` | Truthy (`1`/`true`/`yes`/`on`) emits one JSON object per line (ts/level/logger/request_id/user_id/message + structured fields like `stage`/`job_id`/`elapsed_ms`) for Loki/`jq`. Default is the text format above. |

The worker container starts via `python -m whisper_ui.worker`, which
calls `setup_logging()` before delegating to RQ so the dictConfig (and
the RQ-noise suppression) applies inside the worker process.

**Log rotation.** Every long-running service in `compose.yml` pins
`logging.driver=json-file` with `max-size: 20m` and `max-file: 5`
(100 MB per container); the two one-shot init sidecars (`volume-init`,
`ollama-pull`) are exempt — they emit a handful of lines and exit. Larger deployments should plug a centralised
driver (Loki, Splunk, Datadog) via a compose override; the file-driver
ceiling is sized for small-office deployments only.

### Observability: metrics & monitoring (minimal)

The frontend exposes Prometheus metrics at **`/metrics`** — unauthenticated,
same posture as `/health`, so keep it internal. They are computed at scrape
time from RQ/Redis and the SQLite `jobs` table, with no persistent counters:

| Metric                                         | Type  | Labels   | Source             |
| ---------------------------------------------- | ----- | -------- | ------------------ |
| `whisper_jobs_total`                           | gauge | `status` | SQLite `jobs`      |
| `whisper_queue_depth`                          | gauge | `queue`  | RQ queue length    |
| `whisper_failed_jobs` / `whisper_started_jobs` | gauge | `queue`  | RQ registries      |
| `whisper_rq_workers`                           | gauge | —        | RQ worker registry |

A Redis blip degrades the scrape to the SQLite-only metrics rather than failing
it. Per-stage latency is intentionally **not** a metric here (workers have no
HTTP server); it ships as `elapsed_ms` in the structured JSON logs
(`LOG_JSON=true`) and a histogram is a follow-up.

Bring up a self-hosted stack with the `monitoring` profile:

```bash
docker compose --profile rocm --profile monitoring up -d
# Prometheus UI: http://127.0.0.1:9090/targets  (whisper-frontend / redis / gpu should be UP)
```

It adds **Prometheus** (config in `monitoring/prometheus.yml`), a **Redis
exporter**, and an **AMD GPU exporter** (`rocm/device-metrics-exporter`,
reusing the ROCm `/dev/kfd`+`/dev/dri` passthrough). Caveats: that GPU exporter
is officially validated on Instinct (MI) GPUs — on a gfx1151 APU some fields may
be empty; if so, fall back to `node_exporter` + an `amd-smi metric --json`
textfile collector. Grafana dashboards are a deliberate follow-up — Prometheus
alone is queryable in the meantime.

## Local Development

```bash
# Install mise (tool manager); also pulls uv at the pinned version
mise install

# Install Python dependencies from uv.lock for a reproducible env
uv sync --extra dev --frozen

# Run tests
uv run pytest

# Run linting
uv run ruff format . && uv run ruff check .

# Start FastAPI dev server (requires Redis running)
uv run uvicorn whisper_ui.web.app:app --reload --reload-dir=src
```

> `uv.lock` is committed and is the source of truth for dependency
> versions in CI and reproducible local installs. After editing
> `pyproject.toml`, run `uv lock` to refresh it and commit both files
> together.

## Tech Stack

| Component           | Technology                                                      |
| ------------------- | --------------------------------------------------------------- |
| STT Engine          | faster-whisper large-v3 via WhisperX; whisper.cpp (HIP) on ROCm |
| Speaker Diarization | pyannote-audio (optional, requires HF token)                    |
| Task Queue          | RQ + Redis                                                      |
| Frontend            | FastAPI + htmx + Alpine.js                                      |
| Storage             | SQLite + local filesystem                                       |
| Containerization    | Docker Compose (NVIDIA GPU / CPU / AMD ROCm)                    |

## Project Structure

```text
src/whisper_ui/
  core/               # Config, models, exceptions
  pipeline/           # STT processing stages
  worker/             # RQ task definitions
  storage/            # SQLite + file I/O
  export/             # SRT/VTT/TXT/JSON/DOCX exporters
  ui/                 # Shared labels
  web/                # FastAPI application
    app.py            # Application entry point
    deps.py           # Dependency injection (DB, Redis, templates)
    routes/           # Route handlers (upload, jobs, viewer)
    templates/        # Jinja2 HTML templates
    static/           # CSS and static assets
```

**Layer dependencies** (each layer may only import from layers above it):

```text
core <- pipeline <- worker <- web
core <- storage  <- worker <- web
core <- export              <- web
ui               <- worker <- web
```

`ui` holds shared UI strings (`ui/labels.py`); the worker imports them too
(e.g. `worker/pipeline_callbacks.py` uses the per-stage failure labels), so
`ui` sits above `worker`, not only `web`.

## Troubleshooting

### Speaker diarization not working

- Ensure `HF_TOKEN` is set in your `.env` file
- Accept **both** model agreements on HuggingFace:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- Diarization is optional; transcription works without it

### CPU run logs a compute type warning

- `int8_float16` and `float16` need a GPU; on CPU the worker automatically
  downgrades them to `int8` and logs a WARNING — the job still runs
- Set `COMPUTE_TYPE=int8` explicitly to silence the warning

### Redis connection error

- Ensure Redis is running: `docker compose ps`
- For local development, start Redis: `docker run -d -p 6379:6379 redis:7-alpine`

### Docker build is slow

- Set `PIP_INDEX_URL` in `.env` to a regional PyPI mirror

## License

[MIT](LICENSE)
