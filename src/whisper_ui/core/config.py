from __future__ import annotations

import functools
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from whisper_ui.core.languages import DEFAULT_WHISPER_MODEL

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    # extra="ignore" because the same .env feeds both Settings (read by the
    # FastAPI process) and the worker entrypoint shell, which reads its own
    # WORKER_GPU_QUEUES / WORKER_IO_QUEUES variables that are not Settings
    # fields. Forbidding extras would refuse to start the web tier whenever
    # an operator follows the README's "scaled topology" tuning.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    # Socket timeouts for the web/worker Redis clients built by
    # ``core.redis_client.create_redis``. Without these, a Redis host that
    # accepts the connection but then goes silent (kernel freeze, firewall
    # drop, network partition) blocks the caller's recv indefinitely — the web
    # event loop or a worker would hang instead of raising RedisError and
    # taking the existing graceful-degradation path. Seconds; 0 disables the
    # bound (legacy blocking behaviour). These do not apply to the RQ worker
    # loop's own connection (see core/redis_client.py).
    redis_socket_timeout: int = 10
    redis_socket_connect_timeout: int = 5
    # PING an idle pooled connection after this many seconds so a half-open
    # connection surfaces as an error before the next command blocks on it.
    redis_health_check_interval: int = 30

    # Storage
    database_path: Path = Field(default=_PROJECT_ROOT / "data" / "db" / "whisper_ui.db")
    upload_dir: Path = Field(default=_PROJECT_ROOT / "data" / "uploads")
    output_dir: Path = Field(default=_PROJECT_ROOT / "data" / "outputs")

    # Whisper
    whisper_model: str = DEFAULT_WHISPER_MODEL
    compute_type: str = "int8_float16"
    device: str = "auto"
    batch_size: int = 4

    # Transcription backend:
    #   "whisperx"  -> faster-whisper / CTranslate2 (CUDA + CPU; the default).
    #   "whispercpp" -> whisper.cpp HIP CLI, used by the rocm worker because
    #                   CTranslate2 has no ROCm backend. align/diarize still
    #                   run on torch-ROCm regardless of this setting.
    transcribe_backend: str = "whisperx"
    # whisper.cpp CLI settings (only consulted when transcribe_backend ==
    # "whispercpp"). The GGML model is fetched lazily as ggml-<whisper_model>.bin.
    whispercpp_binary: str = "whisper-cli"
    whispercpp_threads: int = 0  # 0 -> whisper.cpp's own default
    # VAD pre-segmentation for the whisper.cpp CLI. Without it, long
    # silence/music stretches make the decoder hallucinate and the loop
    # propagates through cross-window text conditioning (observed in
    # production as entire transcripts collapsing to one repeated line).
    # The whisperx backend gets the same protection from its built-in VAD.
    whispercpp_vad: bool = True
    # GGML Silero VAD model file, resolved like the main GGML model: a
    # pre-baked copy under the model dir wins, else fetched from the
    # ggml-org/whisper-vad HF repo and cached under HF_HOME. Must be loadable
    # by the pinned whisper.cpp (WHISPER_CPP_REF in Dockerfile.worker.rocm);
    # validated against the current pin, so revalidate when bumping either.
    whispercpp_vad_model: str = "ggml-silero-v5.1.2.bin"
    # Maximum text-context tokens carried across 30 s decode windows
    # (whisper-cli ``-mc``). 0 disables conditioning entirely — parity with
    # the whisperx batched pipeline and the guard that stops a hallucination
    # loop from spreading. -1 keeps whisper.cpp's own default.
    whispercpp_max_context: int = 0

    # Logging. Exposed as Settings fields (not just process env) so a value in
    # .env is honoured — setup_logging reads these. log_level is validated /
    # normalised in logging_setup._resolve_level (an unknown value falls back to
    # INFO), so no validator is needed here.
    log_level: str = "INFO"
    log_json: bool = False

    # Language
    language: str = "zh"

    # HuggingFace
    hf_token: str = ""

    # Diarization
    # Initial state of the upload form's "enable speaker diarization" toggle.
    # Diarization (pyannote) is the slowest pipeline stage — clustering runs on
    # CPU and is unbounded when num_speakers is unset — so deployments that
    # rarely need speaker labels can default it off and let users opt in per
    # job. Only takes effect when diarization_available (an HF token is set);
    # see the diarization_default_for_form property.
    diarization_default_enabled: bool = True

    # Authentication
    # Signing key for session cookies. MUST be set to a stable random value in
    # production (e.g. ``openssl rand -hex 32``); leaving it empty causes
    # :func:`whisper_ui.web.app.create_app` to generate an ephemeral secret at
    # startup, which invalidates every session whenever the process restarts.
    session_secret: str = ""
    # Rate-limit window for login failures. Five attempts in fifteen minutes
    # matches OWASP guidance for "stop credential stuffing without locking
    # legitimate users out for the rest of the day".
    max_login_attempts: int = 5
    login_lockout_seconds: int = 900  # 15 minutes
    # Separate (higher) threshold for per-IP failures. The per-user counter
    # stops credential stuffing against one account; the per-IP counter
    # stops mass enumeration from one source. The defaults assume a small
    # office sharing one NAT egress IP — 20 failures / 15 minutes is high
    # enough that legitimate users do not accidentally lock the office out,
    # but low enough that a scripted attacker is throttled quickly. Larger
    # NATs should either raise this or enable TRUST_PROXY_HEADERS so the
    # per-IP key uses each user's real address.
    max_login_attempts_per_ip: int = 20
    # Per-IP cap on *open* (non-bootstrap) registration attempts within the
    # ``login_lockout_seconds`` window. Bounds scripted account creation and
    # the username-taken enumeration oracle; the first-run bootstrap admin is
    # never throttled so an instance can always be initialised.
    max_register_attempts_per_ip: int = 10
    # When true, the session cookie is only sent over HTTPS. Default False so
    # the bundled compose profiles work over plain HTTP; production deployments
    # behind a TLS-terminating proxy should set ``SESSION_HTTPS_ONLY=true``.
    session_https_only: bool = False
    # Trust X-Forwarded-For (for rate-limit client IP) and X-Forwarded-Host
    # (for CSRF host comparison) when computing per-request context. ONLY
    # enable when a controlled reverse proxy is in front of the app and the
    # proxy resets these headers — otherwise a hostile client can spoof
    # them to evade rate limits and CSRF.
    trust_proxy_headers: bool = False
    # Number of trusted reverse proxies in front of the app. When
    # trust_proxy_headers is on, the client IP is read as the Nth entry from the
    # RIGHT of X-Forwarded-For — the rightmost entries are appended by our own
    # trusted proxies, so the (N)th-from-right is the real client while anything
    # further left is client-controlled and must NOT be trusted for rate-limit
    # bucketing. Default 1 = a single reverse proxy directly in front. Taking
    # the left-most entry (the old behaviour) let a client spoof X-Forwarded-For
    # to a fresh value per request and evade the per-IP limit entirely.
    trusted_proxy_count: int = 1
    # Allow open self-service registration once the first admin exists. The
    # initial bootstrap account is always allowed (an admin must be created to
    # manage the instance); when this is False every later /register attempt
    # is refused so accounts can only be provisioned by an admin. Default True
    # preserves the original open-signup behaviour.
    allow_registration: bool = True

    # Optional bearer token for the /metrics endpoint. Empty (default) keeps
    # /metrics open (it exposes only counts/depths, no PII) — an operator
    # exposing the box publicly should either set this or block /metrics at the
    # reverse proxy. When set, a scrape must send ``Authorization: Bearer <token>``.
    metrics_token: str = ""

    # Upload
    max_upload_size: int = 2 * 1024 * 1024 * 1024  # 2 GB
    # Optional retention: when > 0, the web app's background loop reclaims
    # the upload directory of any COMPLETED job whose updated_at is older
    # than this many days. FAILED jobs are preserved so the retry button
    # keeps working (retry reuses the original upload path). The DB row
    # and the saved transcript (result.json) are kept so the viewer
    # keeps working. Default 0 means never auto-reclaim — legacy behaviour.
    upload_retention_days: int = 0

    # YouTube
    youtube_max_duration: int = 14400  # seconds (4 hours)
    # Backstop on the bytes a single yt-dlp download may write. The duration
    # cap above is the primary bound; this catches the cases it cannot see --
    # media whose reported duration is missing or wrong, and formats that are
    # far denser than the resolution ceiling suggests. Sized to clear any
    # legitimate source at the duration cap: four hours of 1080p H.264 from
    # YouTube runs around 3-4.5 Mbit/s, i.e. well under 8 GiB. 0 disables it.
    # Enforced from yt-dlp's progress events plus a final check on the file
    # on disk. It stops the transfer part-way whenever the chosen downloader
    # reports progress; when an external downloader (ffmpeg) handles the
    # transfer the only report comes at the end, so an oversized file lands
    # and is then removed. See _guard_download_size.
    youtube_max_download_size: int = 8 * 1024 * 1024 * 1024  # 8 GiB

    # Twitter / X
    # Reject X posts whose attached video is longer than this many seconds.
    twitter_max_duration: int = 14400  # seconds (4 hours), mirrors YouTube
    # Optional Netscape-format cookies file (exported from a logged-in browser)
    # mounted into the io worker. When set AND the file exists, it is passed to
    # yt-dlp as cookiefile so login-walled / age-restricted X posts can be
    # fetched. Unset or missing file => anonymous attempt for public posts.
    twitter_cookies_file: str | None = None

    # Queue / job timeouts (seconds)
    # Fallback when audio duration is unknown (e.g., YouTube URL before download).
    job_timeout_default: int = 7200  # 2h
    # Lower bound applied to dynamically calculated timeouts so short audio still
    # gets enough headroom for model loading and overhead.
    job_timeout_floor: int = 1800  # 30min
    # Dynamic timeout = audio_duration_seconds * multiplier, then clamped into
    # [job_timeout_floor, job_timeout_max]. 3x gives comfortable headroom on GPU
    # for large-v3 + alignment + diarization across the pipeline.
    job_timeout_audio_multiplier: float = 3.0
    # Hard upper cap to prevent runaway jobs; also the basis for stale recovery.
    job_timeout_max: int = 28800  # 8h
    # Extra buffer added on top of job_timeout_max when reclaiming stale jobs
    # whose worker died without updating the DB.
    stale_job_buffer: int = 1800  # 30min
    # Redis TTL for the per-job progress HSET while a job is running. Must
    # exceed the longest possible job timeout so the UI does not lose
    # progress state mid-run. See core/constants.py for the relationship
    # to PIPELINE_STATE_TTL_SECONDS and worker/progress._DEFAULT_PROCESSING_TTL.
    redis_processing_expiry: int = 30600  # job_timeout_max + stale_job_buffer
    # How often DiarizeStage's background heartbeat refreshes progress so
    # stale-job-recovery and the UI can see the task is still alive.
    diarize_heartbeat_interval: int = 30

    # -- Optional LLM text correction (Ollama) --
    # Empty string disables the feature entirely — even jobs that opt in via
    # the UI checkbox will be silently skipped. Set to a reachable Ollama URL
    # to enable: "http://ollama:11434" for the bundled `llm` compose profile,
    # or any external Ollama server.
    ollama_base_url: str = ""
    ollama_model: str = "gemma4:e4b"
    # keep_alive format follows Ollama's duration string (e.g. "30m", "1h",
    # "-1" = forever). Longer values amortize model load cost across chunks.
    ollama_keep_alive: str = "30m"
    ollama_request_timeout: int = 120
    llm_chunk_size: int = 8
    llm_chunk_context: int = 2
    llm_temperature: float = 0.1
    # Whether to let a thinking-capable Ollama model emit chain-of-thought
    # before its answer. Default false: for JSON transcript correction,
    # thinking is markedly slower and (measured on gemma-class models)
    # degrades the corrected output. Set true only if a reasoning model is
    # shown to benefit.
    ollama_think: bool = False

    @field_validator("transcribe_backend")
    @classmethod
    def _validate_transcribe_backend(cls, v: str) -> str:
        """Normalize to lowercase and reject unknown backends (fail fast).

        Returning the lowercased value keeps downstream ``== "whispercpp"``
        comparisons robust against case in the .env (e.g. ``WHISPERCPP``).
        """
        v_lower = v.lower()
        allowed = {"whisperx", "whispercpp"}
        if v_lower not in allowed:
            raise ValueError(f"transcribe_backend must be one of {sorted(allowed)}, got {v!r}")
        return v_lower

    @field_validator("whispercpp_max_context")
    @classmethod
    def _validate_whispercpp_max_context(cls, v: int) -> int:
        """whisper-cli accepts -1 (its own default) or a non-negative count."""
        if v < -1:
            raise ValueError(f"whispercpp_max_context must be >= -1, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_whispercpp_vad_model(self) -> Settings:
        """VAD without a model file cannot work; fail at startup, not mid-job."""
        if self.whispercpp_vad and not self.whispercpp_vad_model:
            raise ValueError("whispercpp_vad_model must be set when whispercpp_vad is enabled")
        return self

    @field_validator("ollama_base_url", mode="before")
    @classmethod
    def _normalize_ollama_base_url(cls, v: object) -> object:
        """Strip trailing ``/`` and ``/api`` so that ``POST /api/chat`` built
        via ``httpx.Client(base_url=...)`` never ends up as ``/api/api/chat``.

        Users sometimes copy the URL from third-party Ollama docs that show
        the value as ``http://host:11434/api``. httpx concatenates (rather
        than replaces) the base when the request path also starts with
        ``/api``, which silently duplicates the prefix and makes every
        ``/api/chat`` call 404 — the whole LLM stage then fails over to its
        no-op fallback and the job completes as if LLM correction had never
        been enabled. Normalize defensively here so the footgun cannot land.
        """
        if not isinstance(v, str):
            return v
        return v.rstrip("/").removesuffix("/api")

    @field_validator("ollama_base_url", mode="after")
    @classmethod
    def _validate_ollama_base_url(cls, v: str) -> str:
        """Reject malformed Ollama URLs at startup so misconfigurations fail fast.

        Empty string is allowed (disables LLM correction entirely). Non-empty
        values must parse as an ``http`` / ``https`` URL with a non-empty
        host — this catches typos like ``http://ollama:bad`` that would
        otherwise raise ``httpx.InvalidURL`` deep inside
        ``HttpxOllamaClient.__init__`` and break an opted-in job before the
        per-chunk fallback has a chance to run. Fail-fast is strictly better
        than letting every job silently skip with no visible error.

        Imports httpx lazily so the dependency only needs to be installed
        when LLM correction is actually configured (the ``worker-llm``
        extras group). Workers running without the LLM extras can still
        boot Settings as long as OLLAMA_BASE_URL stays unset.
        """
        if not v:
            return v
        try:
            import httpx
        except ImportError as exc:
            raise ValueError(
                "OLLAMA_BASE_URL is set but httpx is not installed. "
                "Install the worker-llm extras (pip install '.[worker,worker-llm]') "
                "or unset OLLAMA_BASE_URL to disable LLM correction."
            ) from exc
        try:
            parsed = httpx.URL(v)
        except httpx.InvalidURL as exc:
            raise ValueError(f"OLLAMA_BASE_URL is not a valid URL: {exc}") from exc
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"OLLAMA_BASE_URL must use http or https scheme, got {parsed.scheme!r}")
        if not parsed.host:
            raise ValueError(f"OLLAMA_BASE_URL must include a host, got {v!r}")
        return v

    @property
    def diarization_available(self) -> bool:
        """Whether speaker diarization can run on this deployment.

        Diarization needs a HuggingFace token to fetch the pyannote model.
        Routes clamp ``enable_diarization`` against this so a job is never
        persisted (or a diarize sub-job enqueued) for a feature the stage
        would only skip — see ``DiarizeStage.execute``.
        """
        return bool(self.hf_token)

    @property
    def diarization_default_for_form(self) -> bool:
        """Initial value of the upload form's diarization toggle.

        The operator default (``diarization_default_enabled``) clamped by
        capability: never pre-check the box when diarization can't run at all,
        so the rendered default always matches what a submit would persist.
        """
        return self.diarization_default_enabled and self.diarization_available

    @property
    def llm_correction_available(self) -> bool:
        """Whether the LLM correction stage can run on this deployment.

        Single source of truth for the "Ollama configured" check, consumed
        by both the route-level flag clamp and ``is_llm_active``.
        """
        return bool(self.ollama_base_url)

    @property
    def stale_job_timeout(self) -> int:
        """Threshold (seconds) after which a PROCESSING job is considered stale.

        Derived from ``job_timeout_max`` plus ``stale_job_buffer`` so it always
        stays consistent when the cap is tuned.
        """
        return self.job_timeout_max + self.stale_job_buffer

    @model_validator(mode="after")
    def _validate_timeout_invariants(self) -> Settings:
        """Fail-fast at startup if queue-timeout settings are inconsistent.

        Without these checks operators can set values that silently produce
        counter-intuitive behavior — e.g. ``JOB_TIMEOUT_MAX=60`` with the
        default ``JOB_TIMEOUT_FLOOR=1800`` would clamp every computed timeout
        back up to 1800 rather than the intended 60. The constraints below
        mirror what ``calculate_job_timeout`` already assumes and what
        ``stale_job_timeout`` depends on.
        """
        if self.job_timeout_floor <= 0:
            raise ValueError(f"job_timeout_floor must be > 0, got {self.job_timeout_floor}")
        if self.job_timeout_default < self.job_timeout_floor:
            raise ValueError(
                f"job_timeout_default ({self.job_timeout_default}) must be >= "
                f"job_timeout_floor ({self.job_timeout_floor})"
            )
        if self.job_timeout_max < self.job_timeout_default:
            raise ValueError(
                f"job_timeout_max ({self.job_timeout_max}) must be >= job_timeout_default ({self.job_timeout_default})"
            )
        if self.job_timeout_audio_multiplier <= 0:
            raise ValueError(f"job_timeout_audio_multiplier must be > 0, got {self.job_timeout_audio_multiplier}")
        if self.stale_job_buffer < 0:
            raise ValueError(f"stale_job_buffer must be >= 0, got {self.stale_job_buffer}")
        if self.redis_socket_timeout < 0:
            raise ValueError(f"redis_socket_timeout must be >= 0, got {self.redis_socket_timeout}")
        if self.redis_socket_connect_timeout < 0:
            raise ValueError(f"redis_socket_connect_timeout must be >= 0, got {self.redis_socket_connect_timeout}")
        if self.redis_health_check_interval < 0:
            raise ValueError(f"redis_health_check_interval must be >= 0, got {self.redis_health_check_interval}")
        if self.trusted_proxy_count < 1:
            raise ValueError(f"trusted_proxy_count must be >= 1, got {self.trusted_proxy_count}")
        if self.diarize_heartbeat_interval < 0:
            raise ValueError(f"diarize_heartbeat_interval must be >= 0, got {self.diarize_heartbeat_interval}")
        # Redis progress keys must outlive the longest possible job run,
        # otherwise the UI loses progress state mid-pipeline on long jobs.
        if self.redis_processing_expiry < self.stale_job_timeout:
            raise ValueError(
                f"redis_processing_expiry ({self.redis_processing_expiry}) must be >= "
                f"stale_job_timeout ({self.stale_job_timeout}) = "
                f"job_timeout_max + stale_job_buffer"
            )
        if self.llm_chunk_size <= 0:
            raise ValueError(f"llm_chunk_size must be > 0, got {self.llm_chunk_size}")
        if self.llm_chunk_context < 0:
            raise ValueError(f"llm_chunk_context must be >= 0, got {self.llm_chunk_context}")
        if not 0.0 <= self.llm_temperature <= 2.0:
            raise ValueError(f"llm_temperature must be within [0.0, 2.0], got {self.llm_temperature}")
        if self.ollama_request_timeout <= 0:
            raise ValueError(f"ollama_request_timeout must be > 0, got {self.ollama_request_timeout}")
        return self


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    from whisper_ui.core.device import detect_device, validate_compute_type

    settings = Settings()
    resolved_device = detect_device(settings.device)
    resolved_compute = validate_compute_type(resolved_device, settings.compute_type)
    return settings.model_copy(update={"device": resolved_device, "compute_type": resolved_compute})
