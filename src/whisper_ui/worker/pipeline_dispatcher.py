"""Dispatcher that assembles a per-pipeline DAG of RQ sub-jobs.

``enqueue_pipeline`` seeds the shared context in Redis and fans out one
RQ sub-job per logical stage (or stage group) instead of running every
stage inside one monolithic worker task. The sub-jobs are wired
together via ``depends_on`` so that:

* download (if any) runs before preprocess
* transcribe_align and diarize start in parallel after preprocess
* assign_speakers fans them back in
* postprocess (and the optional llm_correction) finish the chain

The final job carries an ``on_success`` callback that saves the transcript
result, marks the parent job COMPLETED, and clears the context store. Every
sub-job also carries an ``on_failure`` callback that marks the parent FAILED,
cancels any dependent jobs that have not yet run, and cleans up the
intermediate 16 kHz WAV.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rq import Callback, Queue

from whisper_ui.core.constants import (
    PIPELINE_STATE_TTL_SECONDS,
    WORKER_QUEUE_CPU,
    WORKER_QUEUE_GPU,
    WORKER_QUEUE_IO,
    WORKER_QUEUE_LLM,
)
from whisper_ui.core.messages import LLM_CORRECTION_SKIPPED, PIPELINE_COMPLETE, RESULT_PERSIST_FAILED
from whisper_ui.core.models import JobStatus
from whisper_ui.worker.context_store import PipelineContextStore
from whisper_ui.worker.pipeline_callbacks import (
    cancel_remaining_subjobs,
    extract_meta_generation,
    format_failure_message,
    is_stale_callback,
    mark_failed,
)
from whisper_ui.worker.runtime import (
    build_worker_runtime,
    cleanup_preprocessed_audio,
    is_llm_active,
)
from whisper_ui.worker.stage_tasks import (
    run_assign_speakers,
    run_diarize,
    run_download,
    run_llm_correction,
    run_postprocess,
    run_preprocess,
    run_transcribe_align,
)
from whisper_ui.worker.timeout import calculate_job_timeout

# Stage function → queue name. IO stages (download, preprocess) go to a queue
# serviced by workers that do not hold a CUDA context; GPU stages go to the GPU
# queue; lightweight CPU finalisation stages go to the CPU queue; the optional,
# slow LLM correction goes to its own queue so it can be isolated onto a
# dedicated worker, keeping a slow LLM off the io/cpu finalisation path. See
# whisper_ui.core.constants for the rationale.
_STAGE_QUEUES = {
    run_download.__name__: WORKER_QUEUE_IO,
    run_preprocess.__name__: WORKER_QUEUE_IO,
    run_llm_correction.__name__: WORKER_QUEUE_LLM,
    run_transcribe_align.__name__: WORKER_QUEUE_GPU,
    run_diarize.__name__: WORKER_QUEUE_GPU,
    run_assign_speakers.__name__: WORKER_QUEUE_CPU,
    run_postprocess.__name__: WORKER_QUEUE_CPU,
}

# Stages whose runtime scales with audio duration. Their death-penalty is
# resized once preprocess knows the real duration (see adjust_subjob_timeouts).
# The other stages have ~fixed cost (download already done, preprocess running,
# assign/postprocess are quick), so they keep their enqueue-time timeout.
_DURATION_SCALED_STAGES = frozenset({run_transcribe_align.__name__, run_diarize.__name__})

if TYPE_CHECKING:
    from redis import Redis
    from rq.job import Job as RQJob

    from whisper_ui.core.config import Settings
    from whisper_ui.core.models import Job
    from whisper_ui.storage.database import JobDatabase
    from whisper_ui.storage.filestore import FileStore

logger = logging.getLogger(__name__)


# Redis keys for tracking sub-jobs and the generation counter. The
# subjobs set is scoped per-generation so a stale callback from a
# superseded attempt cannot accidentally enumerate the new attempt's
# sub-jobs. An earlier design stored all sub-jobs under a single
# parent-scoped key and cleared it on retry, which made a new attempt's
# ids visible to the previous attempt's late finalize callback.
def _subjobs_key(parent_job_id: str, generation: int) -> str:
    return f"whisper:pipeline:{parent_job_id}:subjobs:{generation}"


def _generation_key(parent_job_id: str) -> str:
    return f"whisper:pipeline:{parent_job_id}:generation"


def _bump_generation(redis: Redis, parent_job_id: str) -> int:
    """Atomically advance the generation counter for ``parent_job_id``.

    Returns the new generation. Every enqueue_pipeline call (including
    retries) goes through this, so sibling sub-jobs from a previous attempt
    that somehow kept running will see a newer generation when they try
    to commit their output and silently drop the write.
    """
    new_gen = redis.incr(_generation_key(parent_job_id))
    redis.expire(_generation_key(parent_job_id), PIPELINE_STATE_TTL_SECONDS)
    return int(new_gen)


def _current_generation(redis: Redis, parent_job_id: str) -> int | None:
    """Return the generation counter for ``parent_job_id``, or None when
    no attempt has ever been enqueued (or the counter has expired).
    """
    raw = redis.get(_generation_key(parent_job_id))
    if raw is None:
        return None
    return int(raw)


def _record_subjob(redis: Redis, parent_job_id: str, generation: int, sub_job_id: str) -> None:
    key = _subjobs_key(parent_job_id, generation)
    redis.sadd(key, sub_job_id)
    redis.expire(key, PIPELINE_STATE_TTL_SECONDS)


def _load_subjob_ids(redis: Redis, parent_job_id: str, generation: int) -> list[str]:
    raw = redis.smembers(_subjobs_key(parent_job_id, generation))
    return [item.decode() if isinstance(item, bytes) else item for item in raw]


def _clear_subjob_set(redis: Redis, parent_job_id: str, generation: int) -> None:
    redis.delete(_subjobs_key(parent_job_id, generation))


def _refresh_pipeline_state_ttl(redis: Redis, parent_job_id: str) -> None:
    """Re-arm the TTL on a live pipeline's generation and subjobs keys.

    A spared stale candidate has already been PROCESSING longer than the
    stale threshold; behind a deep enough backlog its state keys could
    outlive PIPELINE_STATE_TTL_SECONDS and expire, after which the next
    liveness probe would misread the still-queued pipeline as dead and reap
    it. Refreshing on every spare keeps the keys alive exactly as long as
    the job is.
    """
    generation = _current_generation(redis, parent_job_id)
    if generation is None:
        return
    redis.expire(_generation_key(parent_job_id), PIPELINE_STATE_TTL_SECONDS)
    redis.expire(_subjobs_key(parent_job_id, generation), PIPELINE_STATE_TTL_SECONDS)


def adjust_subjob_timeouts(
    redis: Redis,
    parent_job_id: str,
    generation: int,
    duration_seconds: float | None,
    settings: Settings,
) -> None:
    """Resize the death-penalty of not-yet-started, duration-scaled sub-jobs.

    URL jobs are enqueued before their media is downloaded, so
    :func:`enqueue_pipeline` sizes every sub-job's ``job_timeout`` from a
    ``None`` duration and they all fall back to ``job_timeout_default``. Once
    preprocess has probed the real duration, this re-applies
    :func:`calculate_job_timeout` to the still-deferred GPU stages
    (transcribe/align, diarize), giving a URL job the same duration-scaled
    death-penalty a file upload already gets at enqueue time. No-op when the
    duration is unknown, so a failed probe simply keeps the default.

    Safe against the retry race: those stages ``depends_on`` preprocess, so
    while this runs inside the preprocess task they are still DEFERRED and
    cannot have started. Scoped to ``generation`` so a stale attempt's
    sub-jobs are never touched. Best-effort: a resize failure is logged, never
    raised, so it cannot fail a preprocess that already produced its output.
    """
    if not duration_seconds or duration_seconds <= 0:
        return
    # The whole body is wrapped so this stays truly best-effort: a resize
    # failure (e.g. the SMEMBERS in _load_subjob_ids or a save() hitting a
    # Redis blip) must never propagate out of the preprocess post_persist hook
    # and fail a preprocess that already produced its output. The inner
    # per-subjob handler lets one bad sub-job be skipped without aborting the
    # rest; the outer handler covers id loading and timeout calculation.
    try:
        from rq.exceptions import NoSuchJobError
        from rq.job import Job as RQJob
        from rq.job import JobStatus as RQJobStatus

        new_timeout = calculate_job_timeout(duration_seconds, settings)
        resizable = {RQJobStatus.DEFERRED, RQJobStatus.QUEUED, RQJobStatus.SCHEDULED}
        resized = 0
        for sub_id in _load_subjob_ids(redis, parent_job_id, generation):
            try:
                sub = RQJob.fetch(sub_id, connection=redis)
                if sub.meta.get("stage") not in _DURATION_SCALED_STAGES:
                    continue
                if sub.get_status(refresh=True) not in resizable:
                    continue
                if sub.timeout != new_timeout:
                    sub.timeout = new_timeout
                    sub.save()
                    resized += 1
            except NoSuchJobError:
                continue
            except Exception:
                logger.warning("Could not resize timeout for sub-job %s of %s", sub_id, parent_job_id, exc_info=True)
        if resized:
            logger.info(
                "Resized %d duration-scaled sub-job timeout(s) for job %s to %ss (duration=%.0fs gen=%d)",
                resized,
                parent_job_id,
                new_timeout,
                duration_seconds,
                generation,
            )
    except Exception:
        logger.warning(
            "Could not resize sub-job timeouts for %s; keeping enqueue-time defaults",
            parent_job_id,
            exc_info=True,
        )


def enqueue_pipeline(
    job: Job,
    *,
    redis: Redis,
    settings: Settings,
    filestore: FileStore,
) -> str:
    """Build and enqueue the RQ DAG for ``job``.

    The DAG shape depends on three flags that come from the Job record and
    the deployment settings:

    * ``job.source_url`` → prepend a download sub-job
    * ``job.enable_diarization`` → add a diarize branch parallel to transcribe
    * ``job.llm_correction_enabled`` (AND ``settings.ollama_base_url``) → add
      an llm_correction sub-job after postprocess

    Returns the id of the last sub-job in the chain, so callers can attach
    monitoring to the tail of the pipeline if they want to.
    """
    queues = {
        name: Queue(name=name, connection=redis)
        for name in (WORKER_QUEUE_IO, WORKER_QUEUE_GPU, WORKER_QUEUE_CPU, WORKER_QUEUE_LLM)
    }

    initial_context: dict = {
        "language": job.language,
        "batch_size": settings.batch_size,
        "num_speakers": job.num_speakers,
    }
    if job.source_url:
        initial_context["source_url"] = job.source_url
        initial_context["download_dir"] = str(filestore.prepare_upload_dir(job.id))
        initial_context["input_path"] = ""
    else:
        initial_context["input_path"] = job.filepath or ""

    # Bump the generation counter *before* initializing the context so any
    # stale writer from the previous attempt sees the new value and drops
    # its write the next time it calls update_if_generation_matches.
    generation = _bump_generation(redis, job.id)

    # Own the progress-hash lifecycle here so the seed is self-contained: a
    # retry must not leave the previous attempt's ``error``/``result_path``
    # fields behind, and callers should not have to remember to clear the
    # hash first. Delete then re-seed with the new generation so a stale
    # writer sees the fresh generation=N via both the central counter
    # (checked in Lua KEYS[2]) AND the hash field, and the UI gets a clean
    # ``progress=0, status=queued`` read right after retry.
    progress_key = f"job:{job.id}"
    redis.delete(progress_key)
    redis.hset(
        progress_key,
        mapping={
            "progress": "0",
            "status": "queued",
            "generation": str(generation),
        },
    )
    redis.expire(progress_key, settings.redis_processing_expiry)

    ctx_store = PipelineContextStore(redis, job.id)
    ctx_store.initialize(initial_context)
    # The subjobs set is now per-generation, so we do NOT clear previous
    # attempts' sets here. An attempt 1 sub-job set will be cleaned up by
    # its own finalize callback (or expire naturally via PIPELINE_STATE_TTL_SECONDS),
    # and attempt 2's callback only ever looks at its own generation's set.
    # This is what keeps a stale attempt 1 callback from cancelling attempt
    # 2's live sub-jobs.

    timeout = calculate_job_timeout(job.duration, settings)
    success_cb = Callback("whisper_ui.worker.pipeline_dispatcher.finalize_success")
    failure_cb = Callback("whisper_ui.worker.pipeline_dispatcher.finalize_failure")
    # Sub-job meta carries both the parent_job_id (for callbacks that need
    # to route to the right Job row) and the generation (so stage tasks
    # can gate their writes against stale retries).
    meta = {"parent_job_id": job.id, "generation": generation}

    llm_active = is_llm_active(job, settings)

    enqueued: list[RQJob] = []

    def _enqueue(func, *, depends_on=None, is_final: bool) -> RQJob:
        queue_name = _STAGE_QUEUES[func.__name__]
        kwargs = {
            "job_timeout": timeout,
            # Tag each sub-job with its stage name so finalize_failure can
            # recognise the optional llm_correction tail and complete with the
            # already-produced transcript instead of failing the whole job.
            "meta": {**meta, "stage": func.__name__},
            "on_failure": failure_cb,
        }
        if depends_on is not None:
            kwargs["depends_on"] = depends_on
        if is_final:
            kwargs["on_success"] = success_cb
        sub = queues[queue_name].enqueue(func, job.id, **kwargs)
        enqueued.append(sub)
        _record_subjob(redis, job.id, generation, sub.id)
        return sub

    # Build the DAG. Note the "is_final" flag is only true on the very last
    # sub-job of the chosen branch, so finalize_success runs exactly once.
    if job.source_url:
        download_job = _enqueue(run_download, is_final=False)
        preprocess_job = _enqueue(run_preprocess, depends_on=download_job, is_final=False)
    else:
        preprocess_job = _enqueue(run_preprocess, is_final=False)

    transcribe_job = _enqueue(
        run_transcribe_align,
        depends_on=preprocess_job,
        is_final=False,
    )

    if job.enable_diarization:
        diarize_job = _enqueue(run_diarize, depends_on=preprocess_job, is_final=False)
        assign_deps: list[RQJob] = [transcribe_job, diarize_job]
    else:
        assign_deps = [transcribe_job]

    assign_job = _enqueue(run_assign_speakers, depends_on=assign_deps, is_final=False)
    postprocess_final = not llm_active
    postprocess_job = _enqueue(
        run_postprocess,
        depends_on=assign_job,
        is_final=postprocess_final,
    )

    if llm_active:
        llm_job = _enqueue(run_llm_correction, depends_on=postprocess_job, is_final=True)
        tail_id = llm_job.id
    else:
        tail_id = postprocess_job.id

    stage_summary = ",".join(s.description.split("(")[0].rsplit(".", 1)[-1] for s in enqueued)
    logger.info(
        "Enqueued pipeline DAG for job %s (generation=%d sub_jobs=%d stages=[%s] "
        "model=%s language=%s diarize=%s llm=%s timeout=%ss)",
        job.id,
        generation,
        len(enqueued),
        stage_summary,
        job.model_name,
        job.language,
        job.enable_diarization,
        llm_active,
        timeout,
    )
    return tail_id


def _apply_filename_from_video_title(job: Job, context: dict) -> None:
    """If the pipeline downloaded from a URL source, surface the title as the
    user-facing filename so the UI shows a human-readable name instead
    of the auto-generated "_" placeholder used at enqueue time.
    """
    if job.source_url and context.get("video_title"):
        job.filename = context["video_title"]


def _persist_completion(
    runtime,
    job: Job,
    context: dict,
    ctx_store: PipelineContextStore,
    transcript_result,
    meta_generation: int | None,
    *,
    progress_message: str = PIPELINE_COMPLETE,
) -> None:
    """Save ``transcript_result``, mark ``job`` COMPLETED, then clean up.

    Shared by ``finalize_success`` (the normal pipeline tail) and
    ``finalize_failure``'s best-effort branch for an optional stage
    (``llm_correction``) that failed after the transcript was already
    produced. Keeping the persist + cleanup in one place means the success
    path and the optional-stage-salvage path cannot drift on what they save
    or clear. The ``finally`` mirrors ``finalize_success``'s original
    cleanup: delete the preprocessed WAV, drop the Redis context hash, and
    clear the per-generation sub-job tracking set.
    """
    reporter = runtime.reporter
    try:
        try:
            _apply_filename_from_video_title(job, context)
            result_path = runtime.filestore.save_result(job.id, transcript_result)

            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.progress_message = progress_message
            job.result_path = str(result_path)
            job.duration = transcript_result.duration
            # Unconditional: a retry that completes cleanly must clear any
            # stale warning persisted by a previous degenerate attempt.
            job.quality_warning = context.get("quality_warning")
            runtime.db.update_job(job)
        except Exception:
            # Persisting the finished transcript failed (e.g. disk full writing
            # the result file, or a DB error). Never leave the parent stuck in
            # PROCESSING: mark it FAILED so the UI and the stale reaper reflect
            # reality. We only reach here while the durable DB status is still
            # pre-COMPLETED, so this can never demote an already-COMPLETED job.
            # If the DB write itself is what failed, mark_failed re-raises and
            # the liveness-based stale reaper is the backstop — nothing at this
            # layer can persist a status while the DB is unavailable.
            # The raw exception stays in the log above and out of job.error:
            # an OSError from this path carries the internal result-file path,
            # and job.error is rendered to the user in _job_card.html. This is
            # the policy format_failure_message already states for the RQ
            # failure path; this site was the one exception to it.
            logger.exception("Failed to persist completion for job %s; marking FAILED", job.id)
            mark_failed(job, runtime.db, reporter, RESULT_PERSIST_FAILED)
            return
        # The DB durably says COMPLETED now; the Redis terminal write is
        # best-effort. Swallow its errors (log only) so a Redis hiccup can
        # neither raise out of the callback nor demote a job the DB already
        # completed.
        try:
            reporter.complete(str(result_path))
        except Exception:
            logger.exception("reporter.complete failed for job %s (already COMPLETED in DB)", job.id)
        logger.info("Job %s completed successfully via DAG pipeline", job.id)
    finally:
        # Best-effort housekeeping: by now the DB already records the terminal
        # status (COMPLETED here, or FAILED via mark_failed above) and the
        # Redis keys have a TTL backstop. A cleanup failure must neither
        # surface as an RQ callback error on the success path nor mask a
        # re-raised mark_failed exception on the failure path, so swallow it.
        try:
            cleanup_preprocessed_audio(context)
            ctx_store.delete()
            if meta_generation is not None:
                _clear_subjob_set(runtime.redis, job.id, meta_generation)
        except Exception:
            logger.warning("Best-effort cleanup failed for job %s", job.id, exc_info=True)


def _is_llm_correction_subjob(rq_job) -> bool:
    """True when ``rq_job`` is the optional llm_correction tail.

    Prefers the explicit ``meta["stage"]`` tag set at enqueue, but falls back
    to the RQ ``func_name`` so sub-jobs enqueued *before* this change — which
    survive a redeploy/host reboot via Redis AOF and are then picked up by a
    new worker — are still recognised and salvaged rather than failed.
    ``func_name`` is always present on a real sub-job; a bare MagicMock used by
    some unit tests has no string ``func_name`` and correctly resolves to
    non-llm.
    """
    name = run_llm_correction.__name__
    meta = getattr(rq_job, "meta", None)
    if meta and meta.get("stage") == name:
        return True
    func_name = getattr(rq_job, "func_name", None)
    return isinstance(func_name, str) and func_name.rsplit(".", 1)[-1] == name


def finalize_success(rq_job, connection, _result) -> None:
    """RQ ``on_success`` callback for the final sub-job.

    Converts the accumulated context into a persisted transcript file,
    updates the parent Job row to COMPLETED, and clears the Redis context
    hash + sub-job tracking set for this attempt's generation.

    Callback staleness: if the parent's generation counter has moved on
    since this sub-job was enqueued (e.g. the user retried mid-pipeline),
    the callback short-circuits without touching any state. Without this
    guard, a stale attempt-1 success callback could mark an in-progress
    attempt 2 as COMPLETED with attempt 1's transcript file.
    """
    parent_job_id = rq_job.meta.get("parent_job_id") if rq_job.meta else None
    if not parent_job_id:
        logger.error("finalize_success invoked without parent_job_id in meta")
        return

    meta_generation = extract_meta_generation(rq_job)

    # Pass meta_generation into build_worker_runtime so runtime.reporter's
    # terminal writes (complete / fail) are gated by the Lua scripts even
    # if the Python short-circuit below is somehow bypassed — defense in
    # depth.
    with build_worker_runtime(parent_job_id, generation=meta_generation) as runtime:
        if is_stale_callback(_current_generation(runtime.redis, parent_job_id), meta_generation):
            logger.warning(
                "finalize_success dropped for job %s: stale callback from generation %s "
                "(current generation has moved on)",
                parent_job_id,
                meta_generation,
            )
            return

        job = runtime.db.get_job(parent_job_id)
        if job is None:
            logger.error("finalize_success could not find parent job %s", parent_job_id)
            return

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            # Never resurrect a terminal job. COMPLETED guards a duplicate
            # finalize; FAILED guards a late/zombie success callback arriving
            # after the stale reaper (or a fail callback) already failed the
            # job — without this, a callback whose meta generation is absent
            # (ungated) could flip a FAILED row back to COMPLETED.
            logger.debug(
                "finalize_success: job %s already terminal (%s); skipping finalize to avoid resurrecting it",
                parent_job_id,
                job.status.value,
            )
            return

        ctx_store = PipelineContextStore(runtime.redis, parent_job_id)
        context = ctx_store.load()
        reporter = runtime.reporter

        transcript_result = context.get("transcript_result")
        if transcript_result is None:
            logger.error("finalize_success for job %s: no transcript_result in context", parent_job_id)
            mark_failed(job, runtime.db, reporter, "transcript_result missing")
            ctx_store.delete()
            if meta_generation is not None:
                _clear_subjob_set(runtime.redis, parent_job_id, meta_generation)
            return

        _persist_completion(runtime, job, context, ctx_store, transcript_result, meta_generation)


def finalize_failure(rq_job, connection, _exc_type, exc_value, _traceback) -> None:
    """RQ ``on_failure`` callback attached to every sub-job.

    Called once per failing sub-job. The first invocation (for the
    current generation) marks the parent FAILED and cancels the other
    sub-jobs in the same generation; subsequent invocations from the
    same generation are no-ops because the parent is already terminal.

    Callback staleness: when the parent's generation counter has moved
    on since this sub-job was enqueued (e.g. the user retried mid-run),
    the callback short-circuits without cancelling or marking anything.
    Without this guard an attempt 1 failure could mark an in-flight
    attempt 2 as FAILED and cancel all of attempt 2's sub-jobs — the
    bug the reproduction test below covers.
    """
    parent_job_id = rq_job.meta.get("parent_job_id") if rq_job.meta else None
    if not parent_job_id:
        logger.error("finalize_failure invoked without parent_job_id in meta")
        return

    meta_generation = extract_meta_generation(rq_job)
    error_msg = format_failure_message(_exc_type, exc_value)
    # Log the raw exception class AND the raw detail (str(exc_value)) here,
    # server-side only. error_msg is the generic message the UI shows; the raw
    # detail — which can carry ffmpeg / whisper-cli stderr and internal paths —
    # stays in the operator log so debugging is unaffected while the user sees
    # only the sanitised per-stage message.
    logger.error(
        "Pipeline failure for job %s (sub_job=%s generation=%s exception=%s detail=%r user_message=%r)",
        parent_job_id,
        rq_job.id,
        meta_generation if meta_generation is not None else "-",
        _exc_type.__name__ if _exc_type is not None else "?",
        str(exc_value) if exc_value is not None else "-",
        error_msg,
    )

    # Pass meta_generation into build_worker_runtime so runtime.reporter is
    # gated by the Lua fail script even if the Python short-circuit below
    # is bypassed — same defense-in-depth as finalize_success.
    with build_worker_runtime(parent_job_id, generation=meta_generation) as runtime:
        if is_stale_callback(_current_generation(runtime.redis, parent_job_id), meta_generation):
            logger.warning(
                "finalize_failure dropped for job %s: stale callback from generation %s "
                "(current generation has moved on)",
                parent_job_id,
                meta_generation,
            )
            return

        job = runtime.db.get_job(parent_job_id)
        if job is None:
            logger.error("finalize_failure could not find parent job %s", parent_job_id)
            return

        if job.status == JobStatus.FAILED:
            logger.debug(
                "finalize_failure: job %s already FAILED, skipping duplicate cleanup",
                parent_job_id,
            )
            return

        ctx_store = PipelineContextStore(runtime.redis, parent_job_id)
        context = ctx_store.load()
        reporter = runtime.reporter

        # Optional stages must never discard a finished transcript. If the
        # failing sub-job is the optional llm_correction tail and the
        # transcript is already in context (postprocess ran before it), the
        # job is effectively done — complete it with the un-corrected
        # transcript instead of marking it FAILED. This covers every failure
        # route for that sub-job uniformly: an in-task exception, the RQ
        # death-penalty, and the AbandonedJobError raised when a worker/host
        # restart (e.g. a scheduled reboot) kills the long LLM stage mid-run.
        transcript_result = context.get("transcript_result")
        if _is_llm_correction_subjob(rq_job) and transcript_result is not None:
            logger.warning(
                "Optional stage %s failed for job %s (%s); completing with the "
                "un-corrected transcript instead of failing the job",
                run_llm_correction.__name__,
                parent_job_id,
                _exc_type.__name__ if _exc_type is not None else "?",
            )
            _persist_completion(
                runtime,
                job,
                context,
                ctx_store,
                transcript_result,
                meta_generation,
                progress_message=LLM_CORRECTION_SKIPPED,
            )
            return

        try:
            if meta_generation is not None:
                cancel_remaining_subjobs(
                    runtime.redis,
                    _load_subjob_ids(runtime.redis, parent_job_id, meta_generation),
                    exclude=rq_job.id,
                )
            mark_failed(job, runtime.db, reporter, error_msg)
            # FAILED is terminal for this attempt: bump the generation so a
            # zombie sibling that survives the best-effort cancel above (a
            # SimpleWorker stage cannot be stopped mid-run) has its gated
            # writes rejected instead of resurrecting the context/progress
            # keys deleted below as orphans.
            _bump_generation(runtime.redis, parent_job_id)
        finally:
            cleanup_preprocessed_audio(context)
            ctx_store.delete()
            if meta_generation is not None:
                _clear_subjob_set(runtime.redis, parent_job_id, meta_generation)


def is_pipeline_dead(redis: Redis, parent_job_id: str) -> bool:
    """Return True when a QUEUED/PROCESSING parent has no live RQ work left.

    "Live" means at least one current-generation sub-job is still queued,
    deferred, scheduled, or started. A parent whose sub-jobs are all finished,
    failed, cancelled, or gone has nothing progressing it and is a genuine
    stale/dead pipeline. A job merely waiting in a backed-up queue is ALIVE and
    must not be reaped — that is the whole point of moving the stale reaper from
    wall-age to liveness, so a slow single-worker box stops mass-failing a
    healthy batch whose tail simply has not been reached yet.

    A started sub-job whose worker has died is still reported STARTED until RQ's
    own maintenance moves it to the failed registry (bounded by its
    death-penalty ``job_timeout``); treating STARTED as live therefore only ever
    errs toward sparing a job, never toward failing a healthy one.
    """
    from rq.exceptions import NoSuchJobError
    from rq.job import Job as RQJob
    from rq.job import JobStatus as RQJobStatus

    generation = _current_generation(redis, parent_job_id)
    if generation is None:
        return True
    sub_ids = _load_subjob_ids(redis, parent_job_id, generation)
    if not sub_ids:
        return True
    live = {RQJobStatus.QUEUED, RQJobStatus.STARTED, RQJobStatus.DEFERRED, RQJobStatus.SCHEDULED}
    for sub_id in sub_ids:
        try:
            status = RQJob.fetch(sub_id, connection=redis).get_status(refresh=True)
        except NoSuchJobError:
            # The sub-job hash is genuinely gone → that branch is dead; keep
            # checking the rest. Only NoSuchJobError is swallowed: a transient
            # redis.exceptions.RedisError must NOT be treated as "dead" (that
            # would mark a live, queued job FAILED on a Redis blip) — it
            # propagates so the stale checker's outer handler logs it and skips
            # this round, leaving every candidate untouched.
            continue
        if status in live:
            return False
    return True


def recover_stale_pipeline_jobs(
    db: JobDatabase,
    redis: Redis,
    timeout_seconds: int,
    error_message: str,
) -> int:
    """Fail only genuinely-dead stale jobs; spare jobs still waiting in RQ.

    Replaces the blind wall-age reaper. A parent QUEUED/PROCESSING for longer
    than ``timeout_seconds`` is failed only if :func:`is_pipeline_dead` confirms
    none of its current-generation sub-jobs is still alive in RQ. This stops a
    single-/slow-worker box from mass-failing a healthy batch whose tail had
    simply not been reached yet (the production incident: 31/38 jobs reaped
    while their transcribe sub-jobs were still queued behind one worker).

    QUEUED is a candidate too, which closes the state-machine dead-end where a
    job stranded in QUEUED (its worker crashed, or RQ data was lost, before any
    stage promoted it to PROCESSING) could never be recovered, retried, or
    deleted. is_pipeline_dead returns True for such a job (no live sub-jobs), so
    it is failed and becomes retryable; a job merely waiting in a backed-up
    queue keeps live sub-jobs and is spared.

    For each job it does fail, the generation counter is bumped so any zombie
    sub-job that later resurfaces drops its writes through the existing
    generation gate. Spared jobs get their pipeline state keys' TTL re-armed
    so a backlog deeper than PIPELINE_STATE_TTL_SECONDS cannot expire them
    into a false "dead" verdict on a later round.
    """
    candidates = db.list_stale_active_job_ids(timeout_seconds)
    if not candidates:
        return 0
    dead: list[str] = []
    for job_id in candidates:
        if is_pipeline_dead(redis, job_id):
            dead.append(job_id)
        else:
            _refresh_pipeline_state_ttl(redis, job_id)
    spared = len(candidates) - len(dead)
    if spared:
        logger.info(
            "stale check: %d candidate(s) older than %ds, %d still live (spared), %d dead",
            len(candidates),
            timeout_seconds,
            spared,
            len(dead),
        )
    if not dead:
        return 0
    for job_id in dead:
        _bump_generation(redis, job_id)
    return db.recover_stale_jobs(timeout_seconds, error_message, only_ids=dead)


__all__ = [
    "adjust_subjob_timeouts",
    "enqueue_pipeline",
    "finalize_failure",
    "finalize_success",
    "is_pipeline_dead",
    "recover_stale_pipeline_jobs",
]
