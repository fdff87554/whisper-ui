from __future__ import annotations

from unittest.mock import MagicMock, patch

import fakeredis
import pytest
from rq.job import Job as RQJob

from whisper_ui.core.constants import (
    WORKER_QUEUE_CPU,
    WORKER_QUEUE_GPU,
    WORKER_QUEUE_IO,
    WORKER_QUEUE_LLM,
)
from whisper_ui.core.models import Job, JobStatus
from whisper_ui.worker.context_store import PipelineContextStore
from whisper_ui.worker.pipeline_dispatcher import (
    _current_generation,
    _load_subjob_ids,
    adjust_subjob_timeouts,
    enqueue_pipeline,
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

STAGE_TASK_NAMES = {
    run_download.__name__,
    run_preprocess.__name__,
    run_transcribe_align.__name__,
    run_diarize.__name__,
    run_assign_speakers.__name__,
    run_postprocess.__name__,
    run_llm_correction.__name__,
}


def _build_settings(*, ollama: str = "http://ollama.internal:11434") -> MagicMock:
    settings = MagicMock()
    settings.batch_size = 16
    settings.ollama_base_url = ollama
    # Mirror the real Settings.llm_correction_available property so is_llm_active
    # gates on a concrete bool rather than a truthy MagicMock attribute.
    settings.llm_correction_available = bool(ollama)
    settings.job_timeout_default = 3600
    settings.job_timeout_floor = 300
    settings.job_timeout_max = 14_400
    settings.job_timeout_audio_multiplier = 2.0
    settings.redis_processing_expiry = 7200
    return settings


def _build_filestore(tmp_path) -> MagicMock:
    fs = MagicMock()
    fs.prepare_upload_path.return_value = tmp_path / "subdir" / "file.mp3"
    return fs


def _load_subjobs(redis, parent_id, generation: int | None = None) -> list[RQJob]:
    """Fetch every sub-job enqueued under ``parent_id``. When ``generation``
    is None, auto-resolve to the current generation counter so test call
    sites that predate the per-generation set layout keep working.
    """
    if generation is None:
        generation = _current_generation(redis, parent_id)
        assert generation is not None, f"no generation counter found for {parent_id}"
    return [RQJob.fetch(sub_id, connection=redis) for sub_id in _load_subjob_ids(redis, parent_id, generation)]


def _stage_func(sub: RQJob) -> str:
    return sub.func_name.rsplit(".", 1)[-1]


def _by_stage(subs: list[RQJob]) -> dict[str, RQJob]:
    return {_stage_func(s): s for s in subs}


def test_file_upload_dag_includes_preprocess_through_postprocess(tmp_path):
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-file",
        filename="meeting.mp3",
        filepath=str(tmp_path / "meeting.mp3"),
        status=JobStatus.QUEUED,
        duration=120.0,
        enable_diarization=True,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    subs = _by_stage(_load_subjobs(redis, job.id))

    assert set(subs.keys()) == {
        "run_preprocess",
        "run_transcribe_align",
        "run_diarize",
        "run_assign_speakers",
        "run_postprocess",
    }


def test_url_upload_dag_prepends_download(tmp_path):
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-url",
        filename="video",
        source_url="https://www.youtube.com/watch?v=abc",
        status=JobStatus.QUEUED,
        enable_diarization=True,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    subs = _by_stage(_load_subjobs(redis, job.id))

    assert "run_download" in subs
    assert subs["run_preprocess"]._dependency_ids == [subs["run_download"].id]


def test_url_job_subjobs_get_default_timeout_at_enqueue(tmp_path):
    """A URL job has no duration when enqueued (media not downloaded yet), so
    every sub-job falls back to job_timeout_default — the gap that
    adjust_subjob_timeouts later closes."""
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-url-timeout",
        filename="video",
        source_url="https://www.youtube.com/watch?v=abc",
        status=JobStatus.QUEUED,
        enable_diarization=True,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    subs = _by_stage(_load_subjobs(redis, job.id))

    assert subs["run_transcribe_align"].timeout == settings.job_timeout_default
    assert subs["run_diarize"].timeout == settings.job_timeout_default


def test_adjust_subjob_timeouts_rescales_gpu_stages_once_duration_known(tmp_path):
    """After preprocess probes the duration, the deferred GPU stages get a
    duration-scaled death-penalty; fixed-cost stages keep the default."""
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-url-resize",
        filename="video",
        source_url="https://www.youtube.com/watch?v=abc",
        status=JobStatus.QUEUED,
        enable_diarization=True,
    )
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    generation = _current_generation(redis, job.id)

    # duration 2000s * multiplier 2.0 = 4000s, within [floor, max] and distinct
    # from the 3600s default baked at enqueue.
    adjust_subjob_timeouts(redis, job.id, generation, 2000.0, settings)

    subs = _by_stage(_load_subjobs(redis, job.id))
    assert subs["run_transcribe_align"].timeout == 4000
    assert subs["run_diarize"].timeout == 4000
    # Non-duration-scaled stages keep the enqueue-time default.
    assert subs["run_download"].timeout == settings.job_timeout_default
    assert subs["run_preprocess"].timeout == settings.job_timeout_default
    assert subs["run_assign_speakers"].timeout == settings.job_timeout_default


def test_adjust_subjob_timeouts_is_noop_without_duration(tmp_path):
    """An unknown/zero duration must leave the enqueue-time timeouts intact."""
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-url-noop",
        filename="video",
        source_url="https://www.youtube.com/watch?v=abc",
        status=JobStatus.QUEUED,
    )
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    generation = _current_generation(redis, job.id)

    adjust_subjob_timeouts(redis, job.id, generation, None, settings)
    adjust_subjob_timeouts(redis, job.id, generation, 0.0, settings)

    subs = _by_stage(_load_subjobs(redis, job.id))
    assert subs["run_transcribe_align"].timeout == settings.job_timeout_default


def test_adjust_subjob_timeouts_swallows_subjob_load_failure(tmp_path, monkeypatch):
    """A Redis failure while loading the sub-job set must not propagate: the
    preprocess hook that calls this runs after the stage already persisted its
    output, so a resize failure must never fail that preprocess job."""
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(id="job-load-fail", filename="video", source_url="https://youtu.be/abc", status=JobStatus.QUEUED)
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    generation = _current_generation(redis, job.id)

    def _boom(*args, **kwargs):
        raise RuntimeError("redis smembers failed")

    monkeypatch.setattr("whisper_ui.worker.pipeline_dispatcher._load_subjob_ids", _boom)

    # Must not raise (the reviewer's regression).
    adjust_subjob_timeouts(redis, job.id, generation, 2000.0, settings)


def test_adjust_subjob_timeouts_swallows_timeout_calc_failure(tmp_path, monkeypatch):
    """A failure computing the new timeout is also best-effort, not fatal."""
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(id="job-calc-fail", filename="video", source_url="https://youtu.be/abc", status=JobStatus.QUEUED)
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    generation = _current_generation(redis, job.id)

    def _boom(*args, **kwargs):
        raise RuntimeError("timeout calc failed")

    monkeypatch.setattr("whisper_ui.worker.pipeline_dispatcher.calculate_job_timeout", _boom)

    adjust_subjob_timeouts(redis, job.id, generation, 2000.0, settings)

    # Sub-jobs keep their enqueue-time default since the resize was abandoned.
    subs = _by_stage(_load_subjobs(redis, job.id))
    assert subs["run_transcribe_align"].timeout == settings.job_timeout_default


def test_llm_enabled_dag_appends_llm_correction(tmp_path):
    redis = fakeredis.FakeRedis()
    settings = _build_settings()
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-llm",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        llm_correction_enabled=True,
        enable_diarization=True,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    subs = _by_stage(_load_subjobs(redis, job.id))

    assert "run_llm_correction" in subs
    assert subs["run_llm_correction"]._dependency_ids == [subs["run_postprocess"].id]
    # on_success must attach to the *final* job only
    assert subs["run_llm_correction"]._success_callback_name is not None
    assert subs["run_postprocess"]._success_callback_name is None


def test_llm_opt_in_without_ollama_url_is_ignored(tmp_path):
    """The deployment-level kill switch (empty ollama_base_url) must drop
    the llm_correction sub-job even if the user opted in, otherwise the
    dispatcher would allocate a sub-job that silently skips itself."""
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-llm-off",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        llm_correction_enabled=True,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    subs = _by_stage(_load_subjobs(redis, job.id))

    assert "run_llm_correction" not in subs
    assert subs["run_postprocess"]._success_callback_name is not None


def test_diarize_disabled_removes_diarize_sub_job(tmp_path):
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-nodiar",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        enable_diarization=False,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    subs = _by_stage(_load_subjobs(redis, job.id))

    assert "run_diarize" not in subs
    assert subs["run_assign_speakers"]._dependency_ids == [subs["run_transcribe_align"].id]


def test_assign_speakers_fans_in_transcribe_and_diarize(tmp_path):
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-fanin",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        enable_diarization=True,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    subs = _by_stage(_load_subjobs(redis, job.id))
    deps = set(subs["run_assign_speakers"]._dependency_ids)

    assert deps == {subs["run_transcribe_align"].id, subs["run_diarize"].id}


def test_all_subjobs_carry_failure_callback_and_parent_meta(tmp_path):
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-meta",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        enable_diarization=True,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)

    for sub in _load_subjobs(redis, job.id):
        assert sub._failure_callback_name is not None
        assert sub.meta.get("parent_job_id") == job.id
        assert _stage_func(sub) in STAGE_TASK_NAMES


def test_enqueue_pipeline_seeds_initial_context(tmp_path):
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-ctx",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        language="zh",
        num_speakers=3,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    ctx = PipelineContextStore(redis, job.id).load()

    assert ctx["language"] == "zh"
    assert ctx["num_speakers"] == 3
    assert ctx["batch_size"] == settings.batch_size
    assert ctx["input_path"] == str(tmp_path / "m.mp3")
    assert "source_url" not in ctx


def test_enqueue_pipeline_clears_stale_progress_hash_on_retry(tmp_path):
    """Re-enqueuing a job must not leave the previous attempt's error/result
    fields in the progress hash; the seed owns the hash lifecycle so callers
    do not have to delete it first."""
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-retry",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
    )
    # Simulate a failed first attempt that left error/result fields behind.
    redis.hset(f"job:{job.id}", mapping={"status": "failed", "error": "boom", "result_path": "/old"})

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)

    stored = {
        k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
        for k, v in redis.hgetall(f"job:{job.id}").items()
    }
    assert stored["status"] == "queued"
    assert stored["progress"] == "0"
    assert "error" not in stored
    assert "result_path" not in stored


def test_enqueue_pipeline_logs_dag_summary_with_stage_metadata(tmp_path, caplog):
    """Operators must be able to read one line and know which stages were
    enqueued, the model + language settings, and the configured timeout —
    the prior 'with N sub-jobs' summary forced them to cross-reference
    the Job row separately.
    """
    import logging as _logging

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-enq-log",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        language="zh",
        model_name="large-v3",
        enable_diarization=True,
        duration=600.0,
    )

    with caplog.at_level(_logging.INFO, logger="whisper_ui.worker.pipeline_dispatcher"):
        enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)

    summary = next(r.getMessage() for r in caplog.records if "Enqueued pipeline DAG" in r.getMessage())
    assert "model=large-v3" in summary
    assert "language=zh" in summary
    assert "diarize=True" in summary
    assert "stages=[" in summary
    assert "timeout=" in summary


def test_subjobs_are_routed_to_resource_specific_queues(tmp_path):
    """Each stage must land on the queue matching the resource it consumes.

    Without this partitioning a long-running IO or LLM stage on the generic
    queue would keep a GPU worker blocked from picking up the next job,
    which is the whole reason for the DAG refactor. The assertion guards
    against accidental regressions of the _STAGE_QUEUES map.
    """
    redis = fakeredis.FakeRedis()
    settings = _build_settings()
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-queues",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        source_url="https://www.youtube.com/watch?v=abc",
        enable_diarization=True,
        llm_correction_enabled=True,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    subs = _by_stage(_load_subjobs(redis, job.id))

    expected = {
        "run_download": WORKER_QUEUE_IO,
        "run_preprocess": WORKER_QUEUE_IO,
        "run_llm_correction": WORKER_QUEUE_LLM,
        "run_transcribe_align": WORKER_QUEUE_GPU,
        "run_diarize": WORKER_QUEUE_GPU,
        "run_assign_speakers": WORKER_QUEUE_CPU,
        "run_postprocess": WORKER_QUEUE_CPU,
    }
    for stage, queue_name in expected.items():
        assert subs[stage].origin == queue_name, f"{stage} should be on {queue_name}, got {subs[stage].origin}"


def test_url_upload_seeds_download_context(tmp_path):
    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-ctx-url",
        source_url="https://www.youtube.com/watch?v=abc",
        status=JobStatus.QUEUED,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    ctx = PipelineContextStore(redis, job.id).load()

    assert ctx["source_url"] == job.source_url
    assert ctx["input_path"] == ""
    assert "download_dir" in ctx


def test_finalize_success_marks_job_completed(monkeypatch, tmp_path):
    from whisper_ui.core.models import TranscriptResult
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(
        id="job-success",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
        # Simulate a retry of a previously degenerate attempt: the clean run
        # must clear this, and starting non-None keeps the assertion below
        # from passing vacuously off the dataclass default.
        quality_warning="舊的品質警告",
    )

    # Seed the context store as if postprocess just wrote its result.
    transcript = TranscriptResult(language="zh", duration=42.0)
    preprocessed = tmp_path / "preprocessed.wav"
    preprocessed.write_bytes(b"fake")
    PipelineContextStore(redis, job.id).initialize({"transcript_result": transcript, "audio_path": str(preprocessed)})

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.filestore.save_result.return_value = tmp_path / "result.json"
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    from whisper_ui.worker.progress import RedisProgressReporter

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        # Replicate build_worker_runtime's contract: the bundled reporter
        # is generation-aware so terminal writes hit Redis through the Lua
        # gating script. Tests still observe real Redis state.
        runtime.reporter = RedisProgressReporter(redis, job_id, processing_ttl=7200, generation=generation)
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    fake_rq_job = MagicMock()
    fake_rq_job.meta = {"parent_job_id": job.id}

    pd.finalize_success(fake_rq_job, redis, None)

    assert job.status == JobStatus.COMPLETED
    assert job.progress == 1.0
    assert job.result_path == str(tmp_path / "result.json")
    assert job.duration == pytest.approx(42.0)
    # Observable terminal state in Redis — finalize_success now builds its
    # own generation-aware reporter instead of reusing runtime.reporter.
    stored = {
        k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
        for k, v in redis.hgetall(f"job:{job.id}").items()
    }
    assert stored["status"] == "completed"
    assert stored["result_path"] == str(tmp_path / "result.json")
    assert not preprocessed.exists(), "preprocessed WAV should be cleaned up"
    assert PipelineContextStore(redis, job.id).load() == {}
    # Clean run: any warning left over from a previous degenerate attempt is
    # cleared by the unconditional assignment.
    assert job.quality_warning is None


def test_finalize_success_persists_quality_warning(monkeypatch, tmp_path):
    from whisper_ui.core.models import TranscriptResult
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(
        id="job-warned",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
    )

    transcript = TranscriptResult(language="zh", duration=42.0)
    PipelineContextStore(redis, job.id).initialize({"transcript_result": transcript, "quality_warning": "轉錄結果異常"})

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.filestore.save_result.return_value = tmp_path / "result.json"
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    from whisper_ui.worker.progress import RedisProgressReporter

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        runtime.reporter = RedisProgressReporter(redis, job_id, processing_ttl=7200, generation=generation)
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    fake_rq_job = MagicMock()
    fake_rq_job.meta = {"parent_job_id": job.id}

    pd.finalize_success(fake_rq_job, redis, None)

    assert job.status == JobStatus.COMPLETED
    assert job.quality_warning == "轉錄結果異常"


def test_finalize_success_completes_even_if_cleanup_raises(monkeypatch, tmp_path):
    """Cleanup in the completion finally is best-effort: a Redis failure
    dropping the context hash must not surface as a callback error after the
    job is already COMPLETED in the DB."""
    from whisper_ui.core.models import TranscriptResult
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(id="job-cleanup-fail", filename="m.mp3", filepath=str(tmp_path / "m.mp3"), status=JobStatus.PROCESSING)
    transcript = TranscriptResult(language="zh", duration=10.0)
    PipelineContextStore(redis, job.id).initialize({"transcript_result": transcript})

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.filestore.save_result.return_value = tmp_path / "result.json"
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    from whisper_ui.worker.progress import RedisProgressReporter

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        runtime.reporter = RedisProgressReporter(redis, job_id, processing_ttl=7200, generation=generation)
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)
    # Make the context-store cleanup blow up inside the finally.
    monkeypatch.setattr(
        PipelineContextStore, "delete", lambda self: (_ for _ in ()).throw(RuntimeError("redis delete failed"))
    )

    fake_rq_job = MagicMock()
    fake_rq_job.meta = {"parent_job_id": job.id}

    # Must not raise despite the cleanup failure.
    pd.finalize_success(fake_rq_job, redis, None)

    assert job.status == JobStatus.COMPLETED
    assert job.result_path == str(tmp_path / "result.json")


def test_finalize_success_skips_already_completed_job(monkeypatch, tmp_path):
    """A second finalize_success for an already-COMPLETED job is a no-op, so
    it never re-saves a result or touches terminal state (mirrors the
    already-FAILED guard in finalize_failure)."""
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(
        id="job-already-done",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.COMPLETED,
    )

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        runtime.reporter = MagicMock()
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    fake_rq_job = MagicMock()
    fake_rq_job.meta = {"parent_job_id": job.id}

    pd.finalize_success(fake_rq_job, redis, None)

    runtime.filestore.save_result.assert_not_called()
    runtime.db.update_job.assert_not_called()


def test_finalize_success_does_not_resurrect_failed_job(monkeypatch, tmp_path):
    """A late/zombie success callback for a job the stale reaper already FAILED
    must not flip it back to COMPLETED, even when its meta generation is absent
    (the ungated path the Python stale-callback guard cannot catch)."""
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(
        id="job-already-failed",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.FAILED,
    )

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        runtime.reporter = MagicMock()
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    fake_rq_job = MagicMock()
    fake_rq_job.meta = {"parent_job_id": job.id}  # no generation -> ungated path

    pd.finalize_success(fake_rq_job, redis, None)

    runtime.filestore.save_result.assert_not_called()
    runtime.db.update_job.assert_not_called()
    assert job.status == JobStatus.FAILED


def test_finalize_failure_marks_job_failed_and_cancels_siblings(monkeypatch, tmp_path):
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-fail",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
        enable_diarization=True,
    )
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)

    # Pick one sub-job to "fail". The rest should get cancelled.
    subs = _load_subjobs(redis, job.id)
    failing = next(s for s in subs if _stage_func(s) == "run_transcribe_align")
    others = [s for s in subs if s.id != failing.id]

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    from whisper_ui.worker.progress import RedisProgressReporter

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        runtime.reporter = RedisProgressReporter(redis, job_id, processing_ttl=7200, generation=generation)
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    # The sub-job meta already carries parent_job_id + generation from the
    # enqueue_pipeline call above; leave it alone so the callback sees the
    # same generation that the dispatcher wrote, otherwise the generation
    # check short-circuits the test against its own setup.
    pd.finalize_failure(failing, redis, RuntimeError, RuntimeError("kaboom"), None)

    from whisper_ui.ui import labels as ui_labels

    assert job.status == JobStatus.FAILED
    # User-facing error is the generic message; the raw "kaboom" detail only
    # goes to the server log, never job.error.
    assert job.error == ui_labels.JOBS_STAGE_FAILED_GENERIC
    assert "kaboom" not in (job.error or "")
    # Observable terminal state — finalize_failure now builds its own
    # generation-aware reporter instead of reusing runtime.reporter.
    stored = {
        k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
        for k, v in redis.hgetall(f"job:{job.id}").items()
    }
    assert stored["status"] == "failed"
    assert "kaboom" not in stored["error"]

    for other in others:
        refreshed = RQJob.fetch(other.id, connection=redis)
        assert refreshed.is_canceled, f"sub-job {_stage_func(other)} should be cancelled"
    # context store should be wiped
    assert PipelineContextStore(redis, job.id).load() == {}


def test_finalize_failure_bumps_generation_to_fence_zombie_writes(monkeypatch, tmp_path):
    """A sibling that survives the best-effort cancel (SimpleWorker cannot
    stop a running job) must have its gated writes rejected once the parent
    is FAILED, not resurrect the deleted context hash as an orphan key."""
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-fence",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
        enable_diarization=True,
    )
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    generation_at_enqueue = int(redis.get(f"whisper:pipeline:{job.id}:generation"))

    subs = _load_subjobs(redis, job.id)
    failing = next(s for s in subs if _stage_func(s) == "run_transcribe_align")

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    from whisper_ui.worker.progress import RedisProgressReporter

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        runtime.reporter = RedisProgressReporter(redis, job_id, processing_ttl=7200, generation=generation)
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    pd.finalize_failure(failing, redis, RuntimeError, RuntimeError("kaboom"), None)

    assert int(redis.get(f"whisper:pipeline:{job.id}:generation")) == generation_at_enqueue + 1
    # A zombie sibling still holding the failed attempt's generation must be
    # rejected by the gated write instead of recreating the context hash.
    ctx = PipelineContextStore(redis, job.id)
    assert ctx.update_if_generation_matches({"diarization_result": "zombie"}, generation_at_enqueue) is False
    assert ctx.load() == {}


def test_finalize_failure_logs_exception_class_separately_from_user_label(monkeypatch, tmp_path, caplog):
    """The user-facing error message is localised; the logged exception
    class is not. Operators counting timeouts vs preprocess errors vs
    OOMs need the raw class name in the log without having to parse the
    Chinese label.
    """
    import logging as _logging

    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-fail-log",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
    )
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    failing = next(s for s in _load_subjobs(redis, job.id) if _stage_func(s) == "run_preprocess")

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    from whisper_ui.worker.progress import RedisProgressReporter

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        runtime.reporter = RedisProgressReporter(redis, job_id, processing_ttl=7200, generation=generation)
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    with caplog.at_level(_logging.ERROR, logger="whisper_ui.worker.pipeline_dispatcher"):
        pd.finalize_failure(failing, redis, ValueError, ValueError("bad input"), None)

    failure = next(r.getMessage() for r in caplog.records if "Pipeline failure for job" in r.getMessage())
    assert "exception=ValueError" in failure
    assert job.id in failure


def test_finalize_failure_uses_chinese_timeout_label_for_rq_timeout(monkeypatch, tmp_path):
    """When an RQ death-penalty timeout kills a DAG sub-job, the parent
    job's ``error`` field must render the same Chinese JOBS_TIMEOUT_ERROR
    label the legacy monolithic worker used. Without this, users see the
    raw "Task exceeded maximum timeout value (N seconds)" English message
    which is both ugly and an i18n regression.
    """
    from rq.timeouts import JobTimeoutException

    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-timeout",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
    )
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    failing = next(s for s in _load_subjobs(redis, job.id) if _stage_func(s) == "run_transcribe_align")

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.reporter = MagicMock()

    from contextlib import contextmanager

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    # Meta is already set by enqueue_pipeline with parent_job_id + generation;
    # do not strip it or the generation short-circuit will hide this test's
    # own assertion setup.
    # Simulate an RQ death-penalty outside the timing-out worker: current
    # job lookup returns None, so extract_rq_timeout_seconds must parse
    # the formatted exception message.
    with patch("rq.get_current_job", return_value=None):
        pd.finalize_failure(
            failing,
            redis,
            JobTimeoutException,
            JobTimeoutException("Task exceeded maximum timeout value (3600 seconds)"),
            None,
        )

    assert job.status == JobStatus.FAILED
    assert job.error is not None
    assert "任務總執行時間超出上限" in job.error
    assert "3600" in job.error
    # The raw English message must not leak through.
    assert "Task exceeded" not in job.error


def test_enqueue_pipeline_increments_generation_counter(tmp_path):
    """Each enqueue_pipeline call (original submit and every retry) must
    bump the per-parent generation counter so stale writers from previous
    attempts can detect their attempt has been superseded.
    """
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-gen",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    gen_after_first = int(redis.get(pd._generation_key(job.id)))

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    gen_after_second = int(redis.get(pd._generation_key(job.id)))

    assert gen_after_first == 1
    assert gen_after_second == 2


def test_subjobs_carry_current_generation_in_meta(tmp_path):
    """Every sub-job enqueued for a given attempt must carry the
    generation matching the counter at enqueue time, so its write-back
    path can reject stale late writes atomically.
    """
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-gen-meta",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        enable_diarization=True,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    subs = _load_subjobs(redis, job.id)
    expected_gen = int(redis.get(pd._generation_key(job.id)))

    assert subs, "enqueue_pipeline should have produced at least one sub-job"
    for sub in subs:
        assert sub.meta.get("generation") == expected_gen

    # After a retry, the old sub-job metadata must be distinct from the new
    # ones. The old subs still sit in the tracking set because finalize_*
    # has not yet cleared it, but any late writer reading a fresh meta
    # will see the incremented generation.
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    new_subs = _load_subjobs(redis, job.id)
    new_gen = int(redis.get(pd._generation_key(job.id)))
    assert new_gen == expected_gen + 1
    # Every sub-job currently tracked should carry either the old or new
    # generation (the set may still contain the previous attempt's ids).
    for sub in new_subs:
        assert sub.meta.get("generation") in {expected_gen, new_gen}


def test_cancel_remaining_subjobs_sends_stop_command_to_running_siblings(monkeypatch, tmp_path):
    """Regression for PR #39 R3 Layer 1: a failed transcribe_align must
    fire send_stop_job_command for every sibling sub-job (so diarize
    running on a second GPU worker is actually stopped), not just
    sub.cancel() which leaves running jobs untouched. The test mocks
    send_stop_job_command on the dispatcher module and asserts it was
    invoked for each non-excluded sub-job.
    """
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-stop",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
        enable_diarization=True,
    )
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)

    gen = _current_generation(redis, job.id)
    sub_ids = set(pd._load_subjob_ids(redis, job.id, gen))
    assert len(sub_ids) >= 3  # preprocess + transcribe_align + diarize at minimum

    failing = next(s for s in _load_subjobs(redis, job.id) if _stage_func(s) == "run_transcribe_align")
    expected_targets = sub_ids - {failing.id}

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.reporter = MagicMock()

    from contextlib import contextmanager

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    from whisper_ui.worker import pipeline_callbacks as pc

    with patch.object(pc, "send_stop_job_command") as mock_stop:
        # Leave failing.meta alone — enqueue_pipeline already set the
        # parent_job_id + generation that the callback needs to pass the
        # staleness check for this test's own attempt.
        pd.finalize_failure(failing, redis, RuntimeError, RuntimeError("kaboom"), None)

    called_sub_ids = {call.args[1] for call in mock_stop.call_args_list}
    assert called_sub_ids == expected_targets, (
        f"send_stop_job_command should target every sibling except the failing one. "
        f"expected {expected_targets}, got {called_sub_ids}"
    )
    # The failing sub-job itself must not receive a stop command (RQ would
    # reject it anyway, but asserting no-op keeps the boundary clean).
    assert failing.id not in called_sub_ids


def test_stale_finalize_failure_short_circuits_after_retry(monkeypatch, tmp_path):
    """Regression for PR #39 Round 2 R2-1. An attempt-1 sub-job that failed
    and fires its ``finalize_failure`` callback AFTER the user retried the
    job must NOT touch any attempt-2 state: the parent Job row stays at
    PROCESSING and none of attempt 2's live sub-jobs get cancelled.

    Before the fix, ``_clear_subjob_set`` at attempt 2 enqueue time
    replaced the parent-scoped tracking set with attempt 2's ids, so the
    stale attempt-1 callback would read attempt 2's sub-jobs out of it
    and cancel all of them while also flipping the Job row to FAILED.
    """
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-stale-fail",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        enable_diarization=True,
    )

    # Attempt 1 — bumps generation to 1, creates its own subjobs set.
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    attempt1_subs = _load_subjobs(redis, job.id, generation=1)
    attempt1_failing = next(s for s in attempt1_subs if _stage_func(s) == "run_transcribe_align")
    assert attempt1_failing.meta.get("generation") == 1

    # Attempt 2 — simulates the user retrying. Dispatcher bumps generation
    # to 2 and allocates a fresh subjobs set under gen=2. Attempt 2's Job
    # row transitions back to PROCESSING as the first stage would have done.
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    job.status = JobStatus.PROCESSING
    attempt2_subs = _load_subjobs(redis, job.id, generation=2)
    attempt2_sub_ids = {s.id for s in attempt2_subs}

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.reporter = MagicMock()

    from contextlib import contextmanager

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    # The stale attempt-1 callback fires AFTER attempt 2 is already running.
    pd.finalize_failure(attempt1_failing, redis, RuntimeError, RuntimeError("attempt1-boom"), None)

    # Parent Job row must still be in attempt 2's state.
    assert job.status == JobStatus.PROCESSING, (
        f"stale attempt-1 finalize_failure must not flip the job to FAILED; got {job.status}"
    )
    runtime.reporter.fail.assert_not_called()
    runtime.db.update_job.assert_not_called()

    # None of attempt 2's sub-jobs may be cancelled by the stale callback.
    for sub_id in attempt2_sub_ids:
        refreshed = RQJob.fetch(sub_id, connection=redis)
        assert not refreshed.is_canceled, f"attempt-2 sub-job {sub_id} was cancelled by a stale attempt-1 callback"


def test_stale_finalize_success_short_circuits_after_retry(monkeypatch, tmp_path):
    """Symmetric regression for R2-1: a stale attempt-1 success callback
    must not flip attempt 2's Job row to COMPLETED or persist a stale
    transcript file.
    """
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-stale-success",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    attempt1_tail = next(s for s in _load_subjobs(redis, job.id, generation=1) if _stage_func(s) == "run_postprocess")
    assert attempt1_tail.meta.get("generation") == 1

    # Retry bumps the generation.
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.reporter = MagicMock()

    from contextlib import contextmanager

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    pd.finalize_success(attempt1_tail, redis, None)

    # Parent must not be marked COMPLETED by the stale callback.
    assert job.status != JobStatus.COMPLETED, (
        f"stale attempt-1 finalize_success must not flip the job to COMPLETED; got {job.status}"
    )
    runtime.reporter.complete.assert_not_called()
    runtime.filestore.save_result.assert_not_called()


def test_subjobs_set_is_scoped_per_generation(tmp_path):
    """Core invariant for Round 2 R2-1 defense-in-depth: each attempt's
    sub-jobs live under their own ``whisper:pipeline:{parent}:subjobs:{gen}``
    key, so a callback looking up sub-jobs under its own generation never
    accidentally sees another attempt's ids.
    """
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-per-gen-set",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.QUEUED,
        enable_diarization=True,
    )

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    gen1_ids = set(pd._load_subjob_ids(redis, job.id, 1))

    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    gen2_ids = set(pd._load_subjob_ids(redis, job.id, 2))

    assert gen1_ids, "attempt 1 should have recorded sub-jobs under its own generation"
    assert gen2_ids, "attempt 2 should have recorded sub-jobs under its own generation"
    assert gen1_ids.isdisjoint(gen2_ids), "attempt 1 and attempt 2 sub-job sets must not overlap"


def test_finalize_failure_generic_exception_uses_sanitised_message(monkeypatch, tmp_path):
    """Non-timeout, unmapped exceptions surface the generic per-stage message,
    never the raw str(exc_value) (which can carry stderr / internal paths)."""
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-generic-fail",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
    )
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    failing = next(s for s in _load_subjobs(redis, job.id) if _stage_func(s) == "run_transcribe_align")

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.reporter = MagicMock()

    from contextlib import contextmanager

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    # Preserve enqueue_pipeline's meta (parent_job_id + generation) so the
    # callback's staleness check passes for this attempt.
    pd.finalize_failure(failing, redis, RuntimeError, RuntimeError("whisper model oom"), None)

    from whisper_ui.ui import labels as ui_labels

    assert job.status == JobStatus.FAILED
    # Raw detail is sanitised out of the user-facing error.
    assert job.error == ui_labels.JOBS_STAGE_FAILED_GENERIC
    assert "whisper model oom" not in (job.error or "")
    # Must not be silently rewritten to the timeout label either.
    assert "任務總執行時間" not in (job.error or "")


def _setup_llm_failure_case(monkeypatch, tmp_path, *, seed_transcript: bool):
    """Build a pipeline with an active llm_correction tail and return the
    handles a finalize_failure-on-LLM test needs.

    Mirrors the production shape: postprocess has already written
    ``transcript_result`` into the context (when ``seed_transcript``) by the
    time the optional llm_correction sub-job fails.
    """
    from whisper_ui.core.models import TranscriptResult
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings()  # ollama set -> llm_correction active
    filestore = _build_filestore(tmp_path)
    job = Job(
        id="job-llm-besteffort",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
        llm_correction_enabled=True,
    )
    enqueue_pipeline(job, redis=redis, settings=settings, filestore=filestore)
    llm_sub = _by_stage(_load_subjobs(redis, job.id))["run_llm_correction"]

    preprocessed = tmp_path / "preprocessed.wav"
    preprocessed.write_bytes(b"fake")
    if seed_transcript:
        PipelineContextStore(redis, job.id).update(
            {"transcript_result": TranscriptResult(language="zh", duration=42.0), "audio_path": str(preprocessed)}
        )

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.filestore.save_result.return_value = tmp_path / "result.json"
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    from whisper_ui.worker.progress import RedisProgressReporter

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        runtime.reporter = RedisProgressReporter(redis, job_id, processing_ttl=7200, generation=generation)
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)
    return pd, redis, job, llm_sub, preprocessed


def test_finalize_failure_on_llm_correction_completes_with_uncorrected_transcript(monkeypatch, tmp_path):
    """An optional llm_correction sub-job that is abandoned mid-run (the
    production incident: a scheduled host reboot killed the long LLM stage)
    must NOT discard the finished transcript — the job completes with the
    un-corrected text instead of being marked FAILED.
    """
    from whisper_ui.core.messages import LLM_CORRECTION_SKIPPED

    pd, redis, job, llm_sub, preprocessed = _setup_llm_failure_case(monkeypatch, tmp_path, seed_transcript=True)

    # AbandonedJobError-style failure: the worker died, RQ fires on_failure.
    pd.finalize_failure(llm_sub, redis, RuntimeError, RuntimeError("Moved to FailedJobRegistry"), None)

    assert job.status == JobStatus.COMPLETED, "optional LLM failure must not fail the whole job"
    assert job.result_path == str(tmp_path / "result.json")
    assert job.progress == 1.0
    assert job.progress_message == LLM_CORRECTION_SKIPPED
    stored = {
        k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
        for k, v in redis.hgetall(f"job:{job.id}").items()
    }
    assert stored["status"] == "completed"
    assert not preprocessed.exists(), "preprocessed WAV should still be cleaned up"
    assert PipelineContextStore(redis, job.id).load() == {}


def test_finalize_failure_on_llm_correction_timeout_completes_not_fails(monkeypatch, tmp_path):
    """Even an RQ death-penalty (BaseTimeoutException) on the optional LLM
    tail must salvage the transcript rather than surface the timeout error —
    LLM correction is strictly best-effort.
    """
    from rq.timeouts import JobTimeoutException

    pd, redis, job, llm_sub, _ = _setup_llm_failure_case(monkeypatch, tmp_path, seed_transcript=True)

    pd.finalize_failure(llm_sub, redis, JobTimeoutException, JobTimeoutException("timed out"), None)

    assert job.status == JobStatus.COMPLETED
    assert job.result_path == str(tmp_path / "result.json")
    assert "任務總執行時間" not in (job.error or "")


def test_finalize_failure_on_llm_correction_without_transcript_still_fails(monkeypatch, tmp_path):
    """Defensive fallback: if there is genuinely no transcript to salvage
    (postprocess never produced one), an LLM failure still fails the job
    instead of completing with nothing.
    """
    pd, redis, job, llm_sub, _ = _setup_llm_failure_case(monkeypatch, tmp_path, seed_transcript=False)

    pd.finalize_failure(llm_sub, redis, RuntimeError, RuntimeError("boom"), None)

    from whisper_ui.ui import labels as ui_labels

    assert job.status == JobStatus.FAILED
    # Sanitised generic message, not the raw "boom".
    assert job.error == ui_labels.JOBS_STAGE_FAILED_GENERIC


def _build_completion_runtime(redis, job, tmp_path, *, save_side_effect=None):
    """Build a MagicMock WorkerRuntime + a real generation-aware reporter for
    the persist-failure tests. ``save_side_effect`` (when set) makes
    ``filestore.save_result`` raise that exception."""
    from whisper_ui.worker.progress import RedisProgressReporter

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    if save_side_effect is not None:
        runtime.filestore.save_result.side_effect = save_side_effect
    else:
        runtime.filestore.save_result.return_value = tmp_path / "result.json"
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        runtime.reporter = RedisProgressReporter(redis, job_id, processing_ttl=7200, generation=generation)
        yield runtime

    return runtime, _fake_builder


def test_persist_completion_save_failure_on_salvage_marks_failed_not_stuck(monkeypatch, tmp_path):
    """PR #94 review #1: if salvaging the un-corrected transcript fails to
    persist (disk full / DB error), the parent must be marked FAILED — never
    left stuck in PROCESSING with its context already deleted."""
    from whisper_ui.core.messages import RESULT_PERSIST_FAILED
    from whisper_ui.core.models import TranscriptResult
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(
        id="job-salvage-savefail",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
        llm_correction_enabled=True,
    )
    enqueue_pipeline(job, redis=redis, settings=_build_settings(), filestore=_build_filestore(tmp_path))
    llm_sub = _by_stage(_load_subjobs(redis, job.id))["run_llm_correction"]
    PipelineContextStore(redis, job.id).update({"transcript_result": TranscriptResult(language="zh", duration=42.0)})

    _, builder = _build_completion_runtime(redis, job, tmp_path, save_side_effect=OSError("disk full"))
    monkeypatch.setattr(pd, "build_worker_runtime", builder)

    pd.finalize_failure(llm_sub, redis, RuntimeError, RuntimeError("llm crashed"), None)

    assert job.status == JobStatus.FAILED, "save failure during salvage must not leave the job PROCESSING"
    assert RESULT_PERSIST_FAILED in (job.error or "")
    # The finally still runs, so context is cleaned up regardless.
    assert PipelineContextStore(redis, job.id).load() == {}


def test_persist_failure_keeps_the_raw_exception_out_of_the_user_error(monkeypatch, tmp_path):
    """job.error is rendered to the user (_job_card.html); an OSError from this
    path carries the internal result-file path.

    format_failure_message already keeps raw str(exc_value) off the RQ failure
    path for exactly this reason; this site used to interpolate it anyway. The
    operator still gets the full detail from the logger.exception above.
    """
    from whisper_ui.core.messages import RESULT_PERSIST_FAILED
    from whisper_ui.core.models import TranscriptResult
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(
        id="job-persist-no-leak",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
        llm_correction_enabled=True,
    )
    enqueue_pipeline(job, redis=redis, settings=_build_settings(), filestore=_build_filestore(tmp_path))
    llm_sub = _by_stage(_load_subjobs(redis, job.id))["run_llm_correction"]
    PipelineContextStore(redis, job.id).update({"transcript_result": TranscriptResult(language="zh", duration=42.0)})

    leak = OSError("[Errno 28] No space left on device: '/srv/whisper/outputs/secret-job/result.json'")
    _, builder = _build_completion_runtime(redis, job, tmp_path, save_side_effect=leak)
    monkeypatch.setattr(pd, "build_worker_runtime", builder)

    pd.finalize_failure(llm_sub, redis, RuntimeError, RuntimeError("llm crashed"), None)

    assert job.status == JobStatus.FAILED
    assert job.error == RESULT_PERSIST_FAILED
    assert "/srv/whisper/outputs" not in (job.error or "")
    assert "Errno 28" not in (job.error or "")
    assert "/srv/whisper/outputs" not in (job.progress_message or "")


def test_finalize_success_save_failure_marks_failed(monkeypatch, tmp_path):
    """The same guard protects the normal completion path: a save_result
    failure in finalize_success marks FAILED instead of leaving PROCESSING
    (pre-existing latent bug, now fixed via the shared helper)."""
    from whisper_ui.core.messages import RESULT_PERSIST_FAILED
    from whisper_ui.core.models import TranscriptResult
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(id="job-succ-savefail", filename="m.mp3", filepath=str(tmp_path / "m.mp3"), status=JobStatus.PROCESSING)
    PipelineContextStore(redis, job.id).initialize(
        {"transcript_result": TranscriptResult(language="zh", duration=10.0)}
    )

    _, builder = _build_completion_runtime(redis, job, tmp_path, save_side_effect=OSError("disk full"))
    monkeypatch.setattr(pd, "build_worker_runtime", builder)

    rq_job = MagicMock()
    rq_job.meta = {"parent_job_id": job.id}
    pd.finalize_success(rq_job, redis, None)

    assert job.status == JobStatus.FAILED
    assert RESULT_PERSIST_FAILED in (job.error or "")


def test_finalize_success_db_update_failure_marks_failed(monkeypatch, tmp_path):
    """A DB error while writing the COMPLETED row also marks FAILED rather than
    leaving the job stuck (save_result succeeds, the first update_job raises)."""
    from whisper_ui.core.models import TranscriptResult
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(id="job-succ-dbfail", filename="m.mp3", filepath=str(tmp_path / "m.mp3"), status=JobStatus.PROCESSING)
    PipelineContextStore(redis, job.id).initialize(
        {"transcript_result": TranscriptResult(language="zh", duration=10.0)}
    )

    runtime, builder = _build_completion_runtime(redis, job, tmp_path)
    # First update_job (COMPLETED) raises; mark_failed's update_job then succeeds.
    runtime.db.update_job.side_effect = [RuntimeError("db locked"), None]
    monkeypatch.setattr(pd, "build_worker_runtime", builder)

    rq_job = MagicMock()
    rq_job.meta = {"parent_job_id": job.id}
    pd.finalize_success(rq_job, redis, None)

    assert job.status == JobStatus.FAILED


def test_finalize_success_reporter_failure_keeps_completed(monkeypatch, tmp_path):
    """A best-effort Redis terminal-write failure (reporter.complete) AFTER the
    DB durably recorded COMPLETED must neither raise out of the callback nor
    demote the finished job to FAILED."""
    from whisper_ui.core.models import TranscriptResult
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(
        id="job-succ-reporterfail", filename="m.mp3", filepath=str(tmp_path / "m.mp3"), status=JobStatus.PROCESSING
    )
    PipelineContextStore(redis, job.id).initialize(
        {"transcript_result": TranscriptResult(language="zh", duration=10.0)}
    )

    runtime = MagicMock()
    runtime.redis = redis
    runtime.db.get_job.return_value = job
    runtime.filestore.save_result.return_value = tmp_path / "result.json"
    runtime.settings.redis_processing_expiry = 7200

    from contextlib import contextmanager

    @contextmanager
    def _fake_builder(job_id, *, generation=None):
        reporter = MagicMock()
        reporter.complete.side_effect = RuntimeError("redis down")
        runtime.reporter = reporter
        yield runtime

    monkeypatch.setattr(pd, "build_worker_runtime", _fake_builder)

    rq_job = MagicMock()
    rq_job.meta = {"parent_job_id": job.id}
    pd.finalize_success(rq_job, redis, None)  # must not raise

    assert job.status == JobStatus.COMPLETED
    runtime.db.update_job.assert_called_once()


def test_is_llm_correction_subjob_identifies_via_func_name_without_meta(tmp_path):
    """PR #94 review #2: the salvage path must recognise the llm_correction
    tail even for a sub-job enqueued before meta["stage"] existed (in-flight
    across a rolling deploy / reboot), via the stable RQ func_name."""
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(
        id="job-funcname",
        filename="m.mp3",
        filepath=str(tmp_path / "m.mp3"),
        status=JobStatus.PROCESSING,
        llm_correction_enabled=True,
    )
    enqueue_pipeline(job, redis=redis, settings=_build_settings(), filestore=_build_filestore(tmp_path))
    subs = _by_stage(_load_subjobs(redis, job.id))
    llm_sub = subs["run_llm_correction"]
    transcribe_sub = subs["run_transcribe_align"]

    # Simulate pre-change sub-jobs by stripping the new meta["stage"] tag.
    llm_sub.meta.pop("stage", None)
    transcribe_sub.meta.pop("stage", None)

    assert pd._is_llm_correction_subjob(llm_sub) is True
    assert pd._is_llm_correction_subjob(transcribe_sub) is False
    # A bare mock (no string func_name) is never mis-identified as llm.
    assert pd._is_llm_correction_subjob(MagicMock()) is False


def test_is_pipeline_dead_false_when_a_subjob_is_still_queued(tmp_path):
    """A job whose pipeline still has a queued/deferred sub-job is ALIVE — the
    liveness gate must not let the stale reaper fail it. This is the 211 fix:
    jobs merely waiting behind a slow worker were being mass-failed.
    """
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(id="job-live", filename="m.mp3", filepath=str(tmp_path / "m.mp3"), status=JobStatus.PROCESSING)
    enqueue_pipeline(job, redis=redis, settings=_build_settings(ollama=""), filestore=_build_filestore(tmp_path))

    # Fresh out of enqueue: preprocess is QUEUED and the rest DEFERRED.
    assert pd.is_pipeline_dead(redis, job.id) is False


def test_is_pipeline_dead_true_when_all_subjobs_terminal(tmp_path):
    """When every sub-job is cancelled/finished (nothing queued/started), the
    pipeline is genuinely dead and eligible for stale recovery."""
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(id="job-dead", filename="m.mp3", filepath=str(tmp_path / "m.mp3"), status=JobStatus.PROCESSING)
    enqueue_pipeline(job, redis=redis, settings=_build_settings(ollama=""), filestore=_build_filestore(tmp_path))

    for sub in _load_subjobs(redis, job.id):
        sub.cancel()

    assert pd.is_pipeline_dead(redis, job.id) is True


def test_is_pipeline_dead_true_when_generation_missing(tmp_path):
    """No generation counter (never enqueued / expired) → nothing to keep the
    pipeline alive → treated as dead."""
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    assert pd.is_pipeline_dead(redis, "never-enqueued") is True


def test_recover_stale_pipeline_jobs_spares_live_and_fails_dead(tmp_path):
    """End-to-end: a dead pipeline (all sub-jobs cancelled) is failed, while an
    equally-old job still waiting in the queue is spared."""
    from whisper_ui.storage.database import JobDatabase
    from whisper_ui.ui.labels import JOBS_STALE_ERROR
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    db = JobDatabase(tmp_path / "stale.db")
    try:
        dead = Job(id="dead", filename="d.mp3", filepath=str(tmp_path / "d.mp3"), status=JobStatus.PROCESSING)
        live = Job(id="live", filename="l.mp3", filepath=str(tmp_path / "l.mp3"), status=JobStatus.PROCESSING)
        db.insert_job(dead)
        db.insert_job(live)
        enqueue_pipeline(dead, redis=redis, settings=settings, filestore=filestore)
        enqueue_pipeline(live, redis=redis, settings=settings, filestore=filestore)

        # dead pipeline: cancel every sub-job. live pipeline: leave it waiting.
        for sub in _load_subjobs(redis, dead.id):
            sub.cancel()

        # Backdate both jobs so they clear the stale-recovery threshold.
        db._conn.execute(
            "UPDATE jobs SET updated_at = '2000-01-01T00:00:00+00:00' WHERE status = ?",
            (JobStatus.PROCESSING.value,),
        )
        db._conn.commit()

        recovered = pd.recover_stale_pipeline_jobs(db, redis, timeout_seconds=60, error_message=JOBS_STALE_ERROR)

        assert recovered == 1
        assert db.get_job(dead.id).status == JobStatus.FAILED
        assert db.get_job(dead.id).error == JOBS_STALE_ERROR
        assert db.get_job(live.id).status == JobStatus.PROCESSING, "a job still waiting in queue must be spared"
    finally:
        db.close()


def test_recover_stale_pipeline_jobs_recovers_stranded_queued_job(tmp_path):
    """A parent stranded in QUEUED whose pipeline is dead (no live sub-jobs) is
    recovered, closing the dead-end where such a job could never be retried or
    deleted. A QUEUED parent whose sub-jobs are still live is spared."""
    from whisper_ui.storage.database import JobDatabase
    from whisper_ui.ui.labels import JOBS_STALE_ERROR
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    db = JobDatabase(tmp_path / "stale.db")
    try:
        dead = Job(id="qdead", filename="d.mp3", filepath=str(tmp_path / "d.mp3"), status=JobStatus.QUEUED)
        live = Job(id="qlive", filename="l.mp3", filepath=str(tmp_path / "l.mp3"), status=JobStatus.QUEUED)
        db.insert_job(dead)
        db.insert_job(live)
        enqueue_pipeline(dead, redis=redis, settings=settings, filestore=filestore)
        enqueue_pipeline(live, redis=redis, settings=settings, filestore=filestore)

        # dead: cancel every sub-job so the pipeline has no live work; the parent
        # DB row is still QUEUED (it never got promoted to PROCESSING).
        for sub in _load_subjobs(redis, dead.id):
            sub.cancel()

        db._conn.execute(
            "UPDATE jobs SET updated_at = '2000-01-01T00:00:00+00:00' WHERE status = ?",
            (JobStatus.QUEUED.value,),
        )
        db._conn.commit()

        recovered = pd.recover_stale_pipeline_jobs(db, redis, timeout_seconds=60, error_message=JOBS_STALE_ERROR)

        assert recovered == 1
        assert db.get_job(dead.id).status == JobStatus.FAILED
        assert db.get_job(live.id).status == JobStatus.QUEUED, "a QUEUED job with live sub-jobs must be spared"
    finally:
        db.close()


def test_recover_stale_pipeline_jobs_refreshes_ttl_of_spared_candidates(tmp_path):
    """A spared candidate's generation/subjobs keys get their TTL re-armed so a
    backlog deeper than PIPELINE_STATE_TTL_SECONDS cannot expire them and flip
    a later liveness probe to a false 'dead' verdict."""
    from whisper_ui.storage.database import JobDatabase
    from whisper_ui.ui.labels import JOBS_STALE_ERROR
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    db = JobDatabase(tmp_path / "stale.db")
    try:
        job = Job(id="spared", filename="m.mp3", filepath=str(tmp_path / "m.mp3"), status=JobStatus.PROCESSING)
        db.insert_job(job)
        enqueue_pipeline(job, redis=redis, settings=_build_settings(ollama=""), filestore=_build_filestore(tmp_path))
        db._conn.execute(
            "UPDATE jobs SET updated_at = '2000-01-01T00:00:00+00:00' WHERE status = ?",
            (JobStatus.PROCESSING.value,),
        )
        db._conn.commit()

        gen_key = pd._generation_key(job.id)
        sub_key = pd._subjobs_key(job.id, int(redis.get(gen_key)))
        # Simulate keys nearing expiry after a long wait in a deep backlog.
        redis.expire(gen_key, 60)
        redis.expire(sub_key, 60)

        recovered = pd.recover_stale_pipeline_jobs(db, redis, timeout_seconds=60, error_message=JOBS_STALE_ERROR)

        assert recovered == 0
        assert db.get_job(job.id).status == JobStatus.PROCESSING
        assert redis.ttl(gen_key) > 60, "spared candidate's generation key TTL must be re-armed"
        assert redis.ttl(sub_key) > 60, "spared candidate's subjobs key TTL must be re-armed"
    finally:
        db.close()


def test_recover_stale_pipeline_jobs_propagates_redis_error_without_reaping(tmp_path, monkeypatch):
    """PR #95 review #1: a transient RedisError during the liveness probe must
    PROPAGATE (so the stale checker logs it and skips the round), not be
    swallowed as 'dead' and used to reap a live job."""
    from redis.exceptions import RedisError

    from whisper_ui.storage.database import JobDatabase
    from whisper_ui.ui.labels import JOBS_STALE_ERROR
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    db = JobDatabase(tmp_path / "stale.db")
    try:
        job = Job(id="live-candidate", filename="m.mp3", filepath=str(tmp_path / "m.mp3"), status=JobStatus.PROCESSING)
        db.insert_job(job)
        enqueue_pipeline(job, redis=redis, settings=_build_settings(ollama=""), filestore=_build_filestore(tmp_path))
        db._conn.execute(
            "UPDATE jobs SET updated_at = '2000-01-01T00:00:00+00:00' WHERE status = ?",
            (JobStatus.PROCESSING.value,),
        )
        db._conn.commit()

        def _raise_redis(*args, **kwargs):
            raise RedisError("connection reset")

        monkeypatch.setattr("rq.job.Job.fetch", _raise_redis)

        with pytest.raises(RedisError):
            pd.recover_stale_pipeline_jobs(db, redis, timeout_seconds=60, error_message=JOBS_STALE_ERROR)

        # The candidate must be left untouched for a later round, not reaped.
        assert db.get_job(job.id).status == JobStatus.PROCESSING
    finally:
        db.close()


def test_is_pipeline_dead_treats_missing_subjob_as_dead_branch(tmp_path):
    """A genuinely-deleted sub-job (RQJob.fetch raises NoSuchJobError) is a dead
    branch; when every sub-job is gone the pipeline is dead. Contrast with the
    propagation test: only NoSuchJobError is swallowed, never a RedisError."""
    from whisper_ui.worker import pipeline_dispatcher as pd

    redis = fakeredis.FakeRedis()
    job = Job(id="job-gone", filename="m.mp3", filepath=str(tmp_path / "m.mp3"), status=JobStatus.PROCESSING)
    enqueue_pipeline(job, redis=redis, settings=_build_settings(ollama=""), filestore=_build_filestore(tmp_path))

    # Remove every RQ job hash so RQJob.fetch raises NoSuchJobError (while the
    # sub-job id set still lists them).
    for sub in _load_subjobs(redis, job.id):
        sub.delete()

    assert pd.is_pipeline_dead(redis, job.id) is True


def test_run_stale_recovery_offload_reaps_dead_via_short_lived_db(tmp_path):
    """PR #95 review (Gemini): the stale check now runs in a thread via a
    module-level helper that opens its OWN short-lived JobDatabase (instead of
    blocking the event loop on the shared connection). It still reaps
    genuinely-dead jobs, equivalent to calling recover_stale_pipeline_jobs."""
    from whisper_ui.storage.database import JobDatabase
    from whisper_ui.ui.labels import JOBS_STALE_ERROR
    from whisper_ui.web.app import _run_stale_recovery

    redis = fakeredis.FakeRedis()
    settings = _build_settings(ollama="")
    filestore = _build_filestore(tmp_path)
    db_path = tmp_path / "stale.db"
    seed = JobDatabase(db_path)
    try:
        dead = Job(id="dead", filename="d.mp3", filepath=str(tmp_path / "d.mp3"), status=JobStatus.PROCESSING)
        seed.insert_job(dead)
        enqueue_pipeline(dead, redis=redis, settings=settings, filestore=filestore)
        for sub in _load_subjobs(redis, dead.id):
            sub.cancel()
        seed._conn.execute(
            "UPDATE jobs SET updated_at = '2000-01-01T00:00:00+00:00' WHERE status = ?",
            (JobStatus.PROCESSING.value,),
        )
        seed._conn.commit()
    finally:
        seed.close()

    recovered = _run_stale_recovery(db_path, redis, 60, JOBS_STALE_ERROR)

    assert recovered == 1
    verify = JobDatabase(db_path)
    try:
        assert verify.get_job(dead.id).status == JobStatus.FAILED
    finally:
        verify.close()
