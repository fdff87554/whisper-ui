"""Per-stage RQ task entrypoints for the parallel pipeline DAG.

Each ``run_*`` function is a single RQ job. The dispatcher assembles these
into a DAG per pipeline run, passing the parent job id only — the actual
pipeline context lives in Redis and is read/written through
:class:`PipelineContextStore` so that fan-out branches can run in different
worker processes.

The ``transcribe`` and ``align`` stages are merged into
``run_transcribe_align`` because they share an in-memory audio buffer that is
expensive to serialize across processes and always run back-to-back on the
same GPU worker anyway. Diarize stays separate so it can fan out to a second
GPU worker when multiple cards are available.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from rq.timeouts import BaseTimeoutException

from whisper_ui.core.device import torch_device_for
from whisper_ui.core.exceptions import PipelineError
from whisper_ui.core.models import JobStatus
from whisper_ui.core.url_validation import is_twitter_url
from whisper_ui.pipeline.align import AlignStage
from whisper_ui.pipeline.assign_speakers import AssignSpeakersStage
from whisper_ui.pipeline.diarize import DiarizeStage
from whisper_ui.pipeline.download import DownloadStage
from whisper_ui.pipeline.postprocess import PostprocessStage
from whisper_ui.pipeline.preprocess import PreprocessStage
from whisper_ui.pipeline.progress_bands import (
    StageWeights,
    build_stage_weights,
)
from whisper_ui.pipeline.transcribe import TranscribeStage
from whisper_ui.pipeline.whispercpp_transcribe import WhisperCppTranscribeStage
from whisper_ui.worker.context_store import PipelineContextStore
from whisper_ui.worker.runtime import (
    WorkerRuntime,
    build_worker_runtime,
    is_llm_active,
    make_throttled_progress_reporter,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from whisper_ui.core.models import Job
    from whisper_ui.pipeline.base import PipelineStage, ProgressCallback

logger = logging.getLogger(__name__)


def pick_stage_weights(job: Job, runtime: WorkerRuntime) -> StageWeights:
    """Build the stage weight bands matching this job's pipeline shape."""
    return build_stage_weights(
        has_download=bool(job.source_url),
        has_llm=is_llm_active(job, runtime.settings),
        has_diarization=job.enable_diarization,
    )


def _load_job(runtime: WorkerRuntime, parent_job_id: str) -> Job:
    job = runtime.db.get_job(parent_job_id)
    if job is None:
        raise PipelineError(f"Job {parent_job_id} not found while running stage task")
    return job


def _build_transcribe_stage(job: Job, runtime: WorkerRuntime, torch_device: str) -> PipelineStage:
    """Select the transcription engine for this worker.

    AMD/ROCm workers set ``transcribe_backend=whispercpp`` because CTranslate2
    (whisperx / faster-whisper) has no ROCm backend; every other worker keeps
    the whisperx path. Both backends emit the same ``transcription_result`` /
    ``whisperx_audio`` context keys, so align/diarize stay backend-agnostic.
    """
    if runtime.settings.transcribe_backend == "whispercpp":
        return WhisperCppTranscribeStage(
            model_name=job.model_name,
            binary=runtime.settings.whispercpp_binary,
            threads=runtime.settings.whispercpp_threads,
            device=runtime.settings.device,
            vad=runtime.settings.whispercpp_vad,
            vad_model=runtime.settings.whispercpp_vad_model,
            max_context=runtime.settings.whispercpp_max_context,
        )
    return TranscribeStage(
        model_name=job.model_name,
        compute_type=runtime.settings.compute_type,
        device=torch_device,
    )


def _mark_processing_if_queued(runtime: WorkerRuntime, job: Job) -> None:
    """Flip the parent job from QUEUED to PROCESSING on first stage entry.

    Multiple sub-jobs can be the "first" one to actually run (e.g.
    transcribe_align and diarize start in parallel after preprocess), so
    every stage task idempotently promotes the job when it observes
    QUEUED. The SQLite write path serialises concurrent writers via WAL
    mode + busy_timeout, so two parallel branches flipping simultaneously
    converge on the same PROCESSING state without corrupting the row.

    This guards two downstream behaviours that depend on the
    QUEUED → PROCESSING transition:

    - Progress/status accuracy: a job that has actually started work should
      read as PROCESSING, not QUEUED. (The stale reaper now also recovers
      genuinely-dead QUEUED jobs via ``list_stale_active_job_ids`` +
      ``is_pipeline_dead``, so a crashed DAG no longer strands a parent in
      QUEUED forever — but that recovery only kicks in after the stale
      timeout, whereas this flip is immediate.)
    - Dashboard polling speed and status badges branch on PROCESSING.
    """
    if job.status == JobStatus.QUEUED:
        job.status = JobStatus.PROCESSING
        runtime.db.update_job(job)


def _banded_progress(
    throttled: Callable[[float, str], None],
    band: tuple[float, float],
) -> ProgressCallback:
    """Wrap a throttled reporter so stages can emit local [0, 1] progress.

    Each stage's local progress is linearly mapped into its global band using
    the same formula as the single-process test orchestrator in
    ``tests/helpers/orchestrator.py``, so a stage reports identically whether
    driven by the production RQ DAG or that test helper.
    """
    start, end = band
    span = end - start

    def on_progress(local: float, message: str) -> None:
        global_progress = start + local * span
        throttled(global_progress, message)

    return on_progress


def _execute_stage(
    stage: PipelineStage,
    context: dict[str, Any],
    on_progress: ProgressCallback,
    *,
    stage_name: str,
) -> dict[str, Any]:
    """Run a stage and convert non-timeout failures into ``PipelineError``.

    Emits the same error shape ``finalize_failure`` already knows how to
    classify — the contract shared with the single-process test orchestrator
    in ``tests/helpers/orchestrator.py``.
    """
    try:
        updated = stage.execute(context, on_progress=on_progress)
    except BaseTimeoutException:
        raise
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(f"Stage '{stage_name}' failed: {e}") from e
    finally:
        stage.cleanup()
    return updated


def _current_generation() -> int | None:
    """Return the generation id the currently-running RQ job was enqueued
    under, or None outside an RQ worker context (e.g. unit tests that
    invoke stage tasks directly).

    Stage tasks use this to tell the context store which attempt they
    belong to, so a stale write from a previous retry attempt can be
    rejected even after ``send_stop_job_command`` has fired.
    """
    try:
        from rq import get_current_job

        current = get_current_job()
        if current is None or not current.meta:
            return None
        gen = current.meta.get("generation")
        return int(gen) if gen is not None else None
    except Exception:
        logger.debug("rq.get_current_job() unavailable while reading generation", exc_info=True)
        return None


def _current_job_timeout() -> int | None:
    """Return the RQ ``job_timeout`` configured at enqueue time, or None.

    Returns None outside an RQ worker context (unit tests that invoke
    stage tasks directly) and on any unexpected error so the log line
    just renders '-' rather than crashing the stage.
    """
    try:
        from rq import get_current_job

        current = get_current_job()
        if current is None or not current.timeout:
            return None
        return int(current.timeout)
    except Exception:
        logger.debug("rq.get_current_job() unavailable while reading timeout", exc_info=True)
        return None


def _log_stage_start(stage_name: str, parent_job_id: str) -> int:
    """Emit the INFO stage-start line and return ``start_ns`` for the finish line.

    Shared between :func:`_run_single_stage` and :func:`run_transcribe_align`
    (which has its own driver but still wants identical observability
    coverage) so the two paths cannot drift on log format or which
    context fields are included.
    """
    timeout_seconds = _current_job_timeout()
    generation = _current_generation()
    logger.info(
        "Stage %s starting for job %s (generation=%s timeout=%ss)",
        stage_name,
        parent_job_id,
        generation if generation is not None else "-",
        timeout_seconds if timeout_seconds is not None else "-",
        extra={
            "event": "stage_start",
            "stage": stage_name,
            "job_id": parent_job_id,
            "generation": generation,
            "timeout_s": timeout_seconds,
        },
    )
    return time.perf_counter_ns()


def _log_stage_finish(stage_name: str, parent_job_id: str, start_ns: int) -> None:
    """Emit the INFO stage-finish line with ``elapsed_ms`` taken from ``start_ns``.

    Called from the ``finally`` of both stage drivers so even a timeout
    still produces the elapsed counter — operators can tell whether the
    stage exited because it finished or because the configured timeout
    fired by comparing elapsed_ms to the timeout from the start log.
    """
    elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000
    logger.info(
        "Stage %s finished for job %s (elapsed_ms=%d)",
        stage_name,
        parent_job_id,
        elapsed_ms,
        extra={
            "event": "stage_finish",
            "stage": stage_name,
            "job_id": parent_job_id,
            "elapsed_ms": elapsed_ms,
        },
    )


def _persist_outputs(
    ctx_store: PipelineContextStore,
    updated: dict[str, Any],
    output_keys: tuple[str, ...],
    *,
    stage_name: str,
) -> None:
    """Write only the declared outputs back to the context store.

    Declaring outputs explicitly (instead of diffing the whole dict) keeps
    fan-in safe: two concurrent branches writing to disjoint keys never stomp
    each other, and large intermediates that should not leave the producing
    process (e.g. ``whisperx_audio``) are automatically excluded.

    When the current RQ job meta carries a ``generation``, the write goes
    through the context store's generation-gated path so a retry that has
    already incremented the generation counter causes this stale stage's
    output to be silently dropped. If the generation cannot be determined
    (e.g. running outside a worker in unit tests), fall back to the
    unconditional update so tests keep working.
    """
    updates = {key: updated[key] for key in output_keys if key in updated}
    generation = _current_generation()
    if generation is None:
        ctx_store.update(updates)
        return
    committed = ctx_store.update_if_generation_matches(updates, generation)
    if not committed:
        logger.warning(
            "Stage %s output dropped: parent job was retried under a new generation",
            stage_name,
        )


def _run_single_stage(
    parent_job_id: str,
    *,
    stage_name: str,
    build_stage: Callable[[Job, WorkerRuntime], PipelineStage],
    output_keys: tuple[str, ...],
    pre_context_update: Callable[[Job, WorkerRuntime, dict[str, Any]], None] | None = None,
    post_persist: Callable[[Job, WorkerRuntime, dict[str, Any]], None] | None = None,
) -> str:
    """Shared driver used by every ``run_*`` entry below.

    ``pre_context_update`` lets stages that need to seed context fields from
    the live ``Job`` record (e.g. download preparing ``download_dir``) do so
    without duplicating the runtime setup. ``post_persist`` runs after the
    stage outputs are persisted, with the updated context, for stages that
    need to act on a freshly-computed value (e.g. preprocess resizing
    downstream sub-job timeouts once the audio duration is known). It is
    best-effort: it runs only after the stage's real output is already
    durable, so its failure is logged but never re-raised — a side effect
    must not turn a successful, persisted stage into a failed RQ job.
    """
    with build_worker_runtime(parent_job_id, generation=_current_generation()) as runtime:
        job = _load_job(runtime, parent_job_id)
        _mark_processing_if_queued(runtime, job)
        ctx_store = PipelineContextStore(runtime.redis, parent_job_id)
        context = ctx_store.load()
        # Make parent_job_id visible to stages that probe / log on behalf
        # of this job. The key never reaches the persisted output_keys
        # set, so adding it here is in-memory only and does not pollute
        # the Redis context hash.
        context["parent_job_id"] = parent_job_id

        if pre_context_update is not None:
            pre_context_update(job, runtime, context)

        stage = build_stage(job, runtime)
        throttled = make_throttled_progress_reporter(runtime.reporter, runtime.db, job)
        weights = pick_stage_weights(job, runtime)
        band = weights.get(stage_name, (0.0, 1.0))
        on_progress = _banded_progress(throttled, band)

        start_ns = _log_stage_start(stage_name, parent_job_id)
        try:
            updated = _execute_stage(stage, context.copy(), on_progress, stage_name=stage_name)
            _persist_outputs(ctx_store, updated, output_keys, stage_name=stage_name)
            if post_persist is not None:
                try:
                    post_persist(job, runtime, updated)
                except Exception:
                    # The stage output is already persisted; a post-persist
                    # side effect must never fail the stage. Log and continue.
                    logger.warning(
                        "post_persist hook failed for stage %s (job %s); stage output already persisted",
                        stage_name,
                        parent_job_id,
                        exc_info=True,
                    )
        finally:
            _log_stage_finish(stage_name, parent_job_id, start_ns)
        return f"{stage_name}:{parent_job_id}"


def _seed_download_context(job: Job, runtime: WorkerRuntime, context: dict[str, Any]) -> None:
    """Ensure download-specific context keys exist when the pipeline starts
    from a URL source.
    """
    context.setdefault("source_url", job.source_url or "")
    download_dir = str(runtime.filestore.prepare_upload_dir(job.id))
    context["download_dir"] = download_dir
    context["input_path"] = context.get("input_path", "")
    # Persist the seeded keys so DownloadStage running in this same task
    # sees them after it reloads via its own context argument.
    ctx_store = PipelineContextStore(runtime.redis, job.id)
    ctx_store.update(
        {
            "source_url": context["source_url"],
            "download_dir": context["download_dir"],
            "input_path": context["input_path"],
        }
    )


def run_download(parent_job_id: str) -> str:
    return _run_single_stage(
        parent_job_id,
        stage_name="download",
        build_stage=lambda job, runtime: DownloadStage(
            max_duration=(
                runtime.settings.twitter_max_duration
                if is_twitter_url(job.source_url or "")
                else runtime.settings.youtube_max_duration
            ),
            max_file_size=runtime.settings.max_upload_size,
            max_download_size=runtime.settings.youtube_max_download_size,
            twitter_cookies_file=runtime.settings.twitter_cookies_file,
        ),
        output_keys=("input_path", "video_title"),
        pre_context_update=_seed_download_context,
    )


def run_preprocess(parent_job_id: str) -> str:
    def _seed_file_input(job: Job, runtime: WorkerRuntime, context: dict[str, Any]) -> None:
        # For direct-file uploads the input_path is on the Job record and
        # may not yet be in the Redis context (this is the first stage).
        if not context.get("input_path") and job.filepath:
            context["input_path"] = job.filepath
            PipelineContextStore(runtime.redis, job.id).update({"input_path": job.filepath})

    def _resize_downstream_timeouts(job: Job, runtime: WorkerRuntime, context: dict[str, Any]) -> None:
        # URL jobs were enqueued before their media existed, so every sub-job
        # got job_timeout_default. Now that preprocess knows the real audio
        # duration, give the still-deferred GPU stages a duration-scaled
        # death-penalty (what a file upload already gets at enqueue time).
        # Lazy import: pipeline_dispatcher imports this module at load time.
        from whisper_ui.worker.pipeline_dispatcher import adjust_subjob_timeouts

        generation = _current_generation()
        if generation is None:
            return
        adjust_subjob_timeouts(runtime.redis, parent_job_id, generation, context.get("duration"), runtime.settings)

    return _run_single_stage(
        parent_job_id,
        stage_name="preprocess",
        build_stage=lambda job, runtime: PreprocessStage(),
        output_keys=("audio_path", "duration"),
        pre_context_update=_seed_file_input,
        post_persist=_resize_downstream_timeouts,
    )


def run_transcribe_align(parent_job_id: str) -> str:
    """Run transcribe and align back-to-back in a single worker process.

    They share ``whisperx_audio`` (a large numpy buffer) through the local
    context dict, which avoids pickling it through Redis between two separate
    RQ tasks. The global progress bar still advances through the ``transcribe``
    and ``align`` bands independently.
    """
    with build_worker_runtime(parent_job_id, generation=_current_generation()) as runtime:
        job = _load_job(runtime, parent_job_id)
        _mark_processing_if_queued(runtime, job)
        ctx_store = PipelineContextStore(runtime.redis, parent_job_id)
        context = ctx_store.load()

        throttled = make_throttled_progress_reporter(runtime.reporter, runtime.db, job)
        weights = pick_stage_weights(job, runtime)

        # On ROCm the logical device label is "rocm"; PyTorch/whisperx address
        # the AMD GPU through the "cuda" namespace, so translate before use.
        torch_device = torch_device_for(runtime.settings.device)
        transcribe = _build_transcribe_stage(job, runtime, torch_device)
        align = AlignStage(device=torch_device)

        transcribe_progress = _banded_progress(throttled, weights.get("transcribe", (0.0, 1.0)))
        align_progress = _banded_progress(throttled, weights.get("align", (0.0, 1.0)))

        # Wrap transcribe + align in a single "transcribe_align" log span:
        # they share a numpy buffer and always run back-to-back on the
        # same task, so a single start/finish pair matches what an
        # operator cares about (did the GPU stage make progress?). The
        # internal throttled progress reporter still differentiates
        # transcribe vs align via the banded progress percentages.
        start_ns = _log_stage_start("transcribe_align", parent_job_id)
        try:
            after_transcribe = _execute_stage(transcribe, context.copy(), transcribe_progress, stage_name="transcribe")
            after_align = _execute_stage(align, after_transcribe, align_progress, stage_name="align")

            _persist_outputs(
                ctx_store,
                after_align,
                output_keys=("transcription_result", "aligned_result"),
                stage_name="transcribe_align",
            )
        finally:
            _log_stage_finish("transcribe_align", parent_job_id, start_ns)

        return f"transcribe_align:{parent_job_id}"


def run_diarize(parent_job_id: str) -> str:
    return _run_single_stage(
        parent_job_id,
        stage_name="diarize",
        build_stage=lambda job, runtime: DiarizeStage(
            hf_token=runtime.settings.hf_token,
            device=torch_device_for(runtime.settings.device),
            enabled=job.enable_diarization,
            heartbeat_interval=runtime.settings.diarize_heartbeat_interval,
        ),
        output_keys=("diarize_result",),
    )


def run_assign_speakers(parent_job_id: str) -> str:
    return _run_single_stage(
        parent_job_id,
        stage_name="assign_speakers",
        build_stage=lambda job, runtime: AssignSpeakersStage(),
        output_keys=("final_result",),
    )


def run_postprocess(parent_job_id: str) -> str:
    return _run_single_stage(
        parent_job_id,
        stage_name="postprocess",
        build_stage=lambda job, runtime: PostprocessStage(
            convert_to_traditional=job.convert_to_traditional,
        ),
        output_keys=("transcript_result", "quality_warning"),
    )


def run_llm_correction(parent_job_id: str) -> str:
    def _build(job: Job, runtime: WorkerRuntime) -> PipelineStage:
        # Lazy import so workers built without the worker-llm extras (httpx)
        # can still boot and process non-LLM jobs.
        from whisper_ui.pipeline.llm_correction import LLMCorrectionStage

        settings = runtime.settings
        return LLMCorrectionStage(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            keep_alive=settings.ollama_keep_alive,
            chunk_size=settings.llm_chunk_size,
            chunk_context=settings.llm_chunk_context,
            temperature=settings.llm_temperature,
            request_timeout=float(settings.ollama_request_timeout),
            think=settings.ollama_think,
        )

    return _run_single_stage(
        parent_job_id,
        stage_name="llm_correction",
        build_stage=_build,
        output_keys=("transcript_result",),
    )


__all__ = [
    "pick_stage_weights",
    "run_assign_speakers",
    "run_diarize",
    "run_download",
    "run_llm_correction",
    "run_postprocess",
    "run_preprocess",
    "run_transcribe_align",
]
