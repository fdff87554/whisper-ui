"""Unit tests for pipeline stages (transcribe, align, diarize, assign_speakers)."""

from __future__ import annotations

import logging
from itertools import pairwise
from unittest.mock import MagicMock, patch

import pytest

from whisper_ui.core.exceptions import AlignmentError, DiarizationError, TranscriptionError
from whisper_ui.core.messages import (
    ALIGN_DONE,
    ALIGN_LOADING,
    ALIGN_SKIPPED,
    ASSIGN_DONE,
    ASSIGN_FAILED,
    ASSIGN_SKIPPED,
    DIARIZE_DONE,
    DIARIZE_SKIPPED,
    DIARIZE_SKIPPED_DISABLED,
    TRANSCRIBE_DONE,
    TRANSCRIBE_LOADING,
    TRANSCRIBE_RUNNING,
)
from whisper_ui.pipeline.align import AlignStage
from whisper_ui.pipeline.assign_speakers import AssignSpeakersStage
from whisper_ui.pipeline.diarize import (
    _HEARTBEAT_PROGRESS_CAP,
    _HEARTBEAT_PROGRESS_START,
    _HEARTBEAT_TAU_FALLBACK_SEC,
    _HEARTBEAT_TAU_MAX_SEC,
    _HEARTBEAT_TAU_MIN_SEC,
    DiarizeStage,
    _compute_heartbeat_tau,
    _heartbeat_progress,
)
from whisper_ui.pipeline.transcribe import TranscribeStage


class TestTranscribeStage:
    def test_execute_sets_context_keys(self):
        stage = TranscribeStage(model_name="base", compute_type="int8", device="cpu")
        mock_whisperx = MagicMock()
        mock_whisperx.load_model.return_value = MagicMock(transcribe=MagicMock(return_value={"segments": []}))
        mock_whisperx.load_audio.return_value = "audio_array"

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            context = {"audio_path": "/tmp/test.wav", "language": "zh", "batch_size": 4}
            result = stage.execute(context)

        assert "transcription_result" in result
        assert "whisperx_audio" in result

    def test_auto_language_passes_none_to_whisperx(self):
        """language=auto must reach whisperx as None so it runs detection."""
        stage = TranscribeStage(model_name="base", compute_type="int8", device="cpu")
        mock_whisperx = MagicMock()
        mock_model = MagicMock(transcribe=MagicMock(return_value={"segments": [], "language": "en"}))
        mock_whisperx.load_model.return_value = mock_model
        mock_whisperx.load_audio.return_value = "audio_array"

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            stage.execute({"audio_path": "/tmp/test.wav", "language": "auto", "batch_size": 4})

        assert mock_whisperx.load_model.call_args.kwargs["language"] is None
        assert mock_model.transcribe.call_args.kwargs["language"] is None

    def test_progress_callback_called(self):
        stage = TranscribeStage(device="cpu")
        mock_whisperx = MagicMock()
        mock_whisperx.load_model.return_value = MagicMock(transcribe=MagicMock(return_value={"segments": []}))
        mock_whisperx.load_audio.return_value = "audio"
        progress_calls = []

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            stage.execute(
                {"audio_path": "/tmp/test.wav"},
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )

        assert progress_calls[0] == (0.0, TRANSCRIBE_LOADING)
        assert progress_calls[-1] == (1.0, TRANSCRIBE_DONE)

    def test_passes_progress_callback_when_whisperx_supports_it(self):
        """Modern whisperx (>=3.4) exposes progress_callback. Wire it through
        so the bar moves continuously instead of jumping from 10% to 100%.
        """
        stage = TranscribeStage(device="cpu")
        captured: dict = {}

        def fake_transcribe(audio, *, batch_size, language, progress_callback=None):
            captured["progress_callback"] = progress_callback
            if progress_callback is not None:
                progress_callback(0.0)
                progress_callback(50.0)
                progress_callback(100.0)
            return {"segments": []}

        fake_model = MagicMock()
        fake_model.transcribe = fake_transcribe
        mock_whisperx = MagicMock()
        mock_whisperx.load_model.return_value = fake_model
        mock_whisperx.load_audio.return_value = "audio"

        progress_calls: list[tuple[float, str]] = []
        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            stage.execute(
                {"audio_path": "/tmp/test.wav"},
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )

        assert callable(captured["progress_callback"])

        # Stage band is 0.1 → 0.95; whisperx 0/50/100 should map there.
        running_calls = [(p, m) for p, m in progress_calls if m == TRANSCRIBE_RUNNING]
        # First entry comes from the manual on_progress(0.1, TRANSCRIBE_RUNNING)
        # before model.transcribe is invoked; the next three are from the
        # forwarded whisperx percent values.
        assert running_calls[0] == (pytest.approx(0.1), TRANSCRIBE_RUNNING)
        assert running_calls[1] == (pytest.approx(0.1), TRANSCRIBE_RUNNING)
        assert running_calls[2] == (pytest.approx(0.525), TRANSCRIBE_RUNNING)
        assert running_calls[3] == (pytest.approx(0.95), TRANSCRIBE_RUNNING)
        # And the final settle to 1.0 still fires after transcribe returns.
        assert progress_calls[-1] == (1.0, TRANSCRIBE_DONE)

    def test_skips_progress_callback_on_legacy_whisperx_signature(self):
        """If a future / older whisperx drops progress_callback, fall back
        cleanly to the coarse 3-point progress instead of crashing the worker
        with TypeError: unexpected keyword argument.
        """
        stage = TranscribeStage(device="cpu")

        def legacy_transcribe(audio, *, batch_size, language):
            return {"segments": []}

        fake_model = MagicMock()
        fake_model.transcribe = legacy_transcribe
        mock_whisperx = MagicMock()
        mock_whisperx.load_model.return_value = fake_model
        mock_whisperx.load_audio.return_value = "audio"

        progress_calls: list[tuple[float, str]] = []
        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            stage.execute(
                {"audio_path": "/tmp/test.wav"},
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )

        # Only the 3 manual progress points; nothing from whisperx percent.
        assert progress_calls == [
            (0.0, TRANSCRIBE_LOADING),
            (0.1, TRANSCRIBE_RUNNING),
            (1.0, TRANSCRIBE_DONE),
        ]

    def test_progress_callback_clamps_out_of_range_percent(self):
        """whisperx has previously emitted slightly-over-100 percent values
        in edge cases; the adapter must clamp instead of overflowing past
        the stage band end and confusing the orchestrator weight maths.
        """
        stage = TranscribeStage(device="cpu")

        def fake_transcribe(audio, *, batch_size, language, progress_callback=None):
            progress_callback(-5.0)
            progress_callback(150.0)
            return {"segments": []}

        fake_model = MagicMock()
        fake_model.transcribe = fake_transcribe
        mock_whisperx = MagicMock()
        mock_whisperx.load_model.return_value = fake_model
        mock_whisperx.load_audio.return_value = "audio"

        progress_calls: list[tuple[float, str]] = []
        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            stage.execute(
                {"audio_path": "/tmp/test.wav"},
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )

        running = [p for p, m in progress_calls if m == TRANSCRIBE_RUNNING]
        # -5 clamps to 0.1, 150 clamps to 0.95
        assert pytest.approx(0.1) == running[1]
        assert pytest.approx(0.95) == running[2]

    def test_cleanup_guard_no_model(self):
        stage = TranscribeStage(device="cpu")
        assert stage._model is None
        with patch("whisper_ui.pipeline.transcribe.gc.collect") as mock_gc:
            stage.cleanup()
        mock_gc.assert_not_called()

    def test_cleanup_releases_model(self):
        stage = TranscribeStage(device="cpu")
        stage._model = MagicMock()
        with (
            patch("whisper_ui.pipeline.transcribe.gc.collect") as mock_gc,
            patch("whisper_ui.pipeline.transcribe.release_gpu_memory"),
        ):
            stage.cleanup()
        assert stage._model is None
        mock_gc.assert_called_once()

    def test_import_error_raises_transcription_error(self):
        stage = TranscribeStage(device="cpu")
        with (
            patch.dict("sys.modules", {"whisperx": None}),
            pytest.raises(TranscriptionError, match="not installed"),
        ):
            stage.execute({"audio_path": "/tmp/test.wav"})


class TestAlignStage:
    def test_execute_sets_aligned_result(self):
        stage = AlignStage(device="cpu")
        mock_whisperx = MagicMock()
        mock_whisperx.load_align_model.return_value = ("model", "metadata")
        mock_whisperx.align.return_value = {"segments": [{"text": "hi"}]}

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            context = {
                "transcription_result": {"segments": [], "language": "zh"},
                "whisperx_audio": "audio",
            }
            result = stage.execute(context)

        assert "aligned_result" in result

    def test_progress_callback_called(self):
        stage = AlignStage(device="cpu")
        mock_whisperx = MagicMock()
        mock_whisperx.load_align_model.return_value = ("model", "metadata")
        mock_whisperx.align.return_value = {"segments": []}
        progress_calls = []

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            stage.execute(
                {"transcription_result": {"segments": []}, "whisperx_audio": "audio"},
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )

        assert progress_calls[0] == (0.0, ALIGN_LOADING)
        assert progress_calls[-1] == (1.0, ALIGN_DONE)

    def test_cleanup_guard_no_resources(self):
        stage = AlignStage(device="cpu")
        with patch("whisper_ui.pipeline.align.gc.collect") as mock_gc:
            stage.cleanup()
        mock_gc.assert_not_called()

    def test_cleanup_releases_both(self):
        stage = AlignStage(device="cpu")
        stage._model = MagicMock()
        stage._metadata = MagicMock()
        with (
            patch("whisper_ui.pipeline.align.gc.collect") as mock_gc,
            patch("whisper_ui.pipeline.align.release_gpu_memory"),
        ):
            stage.cleanup()
        assert stage._model is None
        assert stage._metadata is None
        mock_gc.assert_called_once()

    def test_import_error_raises_alignment_error(self):
        stage = AlignStage(device="cpu")
        with (
            patch.dict("sys.modules", {"whisperx": None}),
            pytest.raises(AlignmentError, match="not installed"),
        ):
            stage.execute({"transcription_result": {"segments": []}, "whisperx_audio": "audio"})

    def test_missing_context_keys_fail_fast_not_degrade(self):
        """A broken context contract (missing transcription_result /
        whisperx_audio) is an upstream bug and must raise, not be disguised
        as an alignment failure that degrades to the unaligned result."""
        stage = AlignStage(device="cpu")
        mock_whisperx = MagicMock()

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            with pytest.raises(KeyError):
                stage.execute({"whisperx_audio": "audio"})
            with pytest.raises(KeyError):
                stage.execute({"transcription_result": {"segments": [], "language": "zh"}})

    def test_alignment_failure_falls_back_to_transcription_result(self):
        """An align failure must keep speaker assignment possible: the
        unaligned transcription becomes the aligned_result so assign_speakers
        can still attach segment-level speakers instead of skipping entirely
        (which would silently discard diarization too)."""
        stage = AlignStage(device="cpu")
        mock_whisperx = MagicMock()
        mock_whisperx.load_align_model.side_effect = ValueError("model not found")
        transcription = {"segments": [{"text": "hi"}], "language": "ja"}

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            context = {
                "transcription_result": transcription,
                "whisperx_audio": "audio",
                "language": "ja",
            }
            result = stage.execute(context)

        assert result["aligned_result"] is transcription

    def test_alignment_failure_reports_skipped_progress(self):
        stage = AlignStage(device="cpu")
        mock_whisperx = MagicMock()
        mock_whisperx.load_align_model.side_effect = ValueError("model not found")
        progress_calls = []

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            stage.execute(
                {
                    "transcription_result": {"segments": [], "language": "ja"},
                    "whisperx_audio": "audio",
                    "language": "ja",
                },
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )

        assert progress_calls[-1] == (1.0, ALIGN_SKIPPED)

    def test_alignment_failure_logs_warning(self):
        stage = AlignStage(device="cpu")
        mock_whisperx = MagicMock()
        mock_whisperx.load_align_model.side_effect = ValueError("model not found")

        with (
            patch.dict("sys.modules", {"whisperx": mock_whisperx}),
            patch("whisper_ui.pipeline.align.logger") as mock_logger,
        ):
            stage.execute(
                {
                    "transcription_result": {"segments": [], "language": "ja"},
                    "whisperx_audio": "audio",
                    "language": "ja",
                },
            )

        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "ja" in str(call_args)
        assert "model not found" in str(call_args)

    def test_align_execution_failure_falls_back_to_transcription_result(self):
        stage = AlignStage(device="cpu")
        mock_whisperx = MagicMock()
        mock_whisperx.load_align_model.return_value = ("model", "metadata")
        mock_whisperx.align.side_effect = RuntimeError("alignment crashed")
        transcription = {"segments": [{"text": "hello"}], "language": "ja"}
        progress_calls = []

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            context = {
                "transcription_result": transcription,
                "whisperx_audio": "audio",
                "language": "ja",
            }
            result = stage.execute(context, on_progress=lambda p, m: progress_calls.append((p, m)))

        assert result["aligned_result"] is transcription
        assert progress_calls[-1] == (1.0, ALIGN_SKIPPED)
        assert stage._model is not None
        assert stage._metadata is not None


class TestDiarizeStage:
    def test_disabled_skips(self):
        stage = DiarizeStage(enabled=False, device="cpu")
        progress_calls = []
        result = stage.execute({}, on_progress=lambda p, m: progress_calls.append((p, m)))
        assert result.get("diarize_result") is None
        assert progress_calls[-1] == (1.0, DIARIZE_SKIPPED_DISABLED)

    def test_no_token_skips(self):
        stage = DiarizeStage(hf_token="", device="cpu")
        progress_calls = []
        result = stage.execute({}, on_progress=lambda p, m: progress_calls.append((p, m)))
        assert result.get("diarize_result") is None
        assert progress_calls[-1] == (1.0, DIARIZE_SKIPPED)

    def test_execute_success(self):
        stage = DiarizeStage(hf_token="test-token", device="cpu")
        mock_pipeline_cls = MagicMock()
        mock_pipeline_instance = MagicMock(return_value="diarize_segments")
        mock_pipeline_cls.return_value = mock_pipeline_instance

        mock_diarize_module = MagicMock()
        mock_diarize_module.DiarizationPipeline = mock_pipeline_cls

        with patch.dict("sys.modules", {"whisperx.diarize": mock_diarize_module, "whisperx": MagicMock()}):
            context = {"audio_path": "/tmp/test.wav"}
            progress_calls = []
            result = stage.execute(context, on_progress=lambda p, m: progress_calls.append((p, m)))

        assert result["diarize_result"] == "diarize_segments"
        assert progress_calls[-1] == (1.0, DIARIZE_DONE)

    def test_auth_error_message(self):
        stage = DiarizeStage(hf_token="bad-token", device="cpu")
        mock_pipeline_cls = MagicMock(side_effect=Exception("401 Unauthorized"))
        mock_diarize_module = MagicMock()
        mock_diarize_module.DiarizationPipeline = mock_pipeline_cls

        with (
            patch.dict("sys.modules", {"whisperx.diarize": mock_diarize_module, "whisperx": MagicMock()}),
            pytest.raises(DiarizationError, match="authorization error"),
        ):
            stage.execute({"audio_path": "/tmp/test.wav"})

    def test_cleanup_guard_no_pipeline(self):
        stage = DiarizeStage(device="cpu")
        with patch("whisper_ui.pipeline.diarize.gc.collect") as mock_gc:
            stage.cleanup()
        mock_gc.assert_not_called()

    def test_cleanup_releases_pipeline(self):
        stage = DiarizeStage(device="cpu")
        stage._pipeline = MagicMock()
        with (
            patch("whisper_ui.pipeline.diarize.gc.collect") as mock_gc,
            patch("whisper_ui.pipeline.diarize.release_gpu_memory"),
        ):
            stage.cleanup()
        assert stage._pipeline is None
        mock_gc.assert_called_once()

    def test_rq_timeout_is_not_wrapped_as_diarization_error(self):
        from rq.timeouts import JobTimeoutException

        stage = DiarizeStage(hf_token="test-token", device="cpu")
        mock_pipeline_cls = MagicMock()
        mock_pipeline_instance = MagicMock(
            side_effect=JobTimeoutException("Task exceeded maximum timeout value (3600 seconds)")
        )
        mock_pipeline_cls.return_value = mock_pipeline_instance

        mock_diarize_module = MagicMock()
        mock_diarize_module.DiarizationPipeline = mock_pipeline_cls

        with (
            patch.dict("sys.modules", {"whisperx.diarize": mock_diarize_module, "whisperx": MagicMock()}),
            pytest.raises(JobTimeoutException, match="Task exceeded"),
        ):
            stage.execute({"audio_path": "/tmp/test.wav"})

    def test_heartbeat_progress_is_monotonic_and_bounded(self):
        """Elapsed time must map to a non-decreasing curve that never
        reaches 1.0 — DIARIZE_DONE (which reports 1.0) is the only thing
        that should complete the stage.
        """
        samples = [_heartbeat_progress(t, tau=60.0) for t in (0, 5, 30, 60, 120, 600, 3600)]
        assert samples[0] == pytest.approx(_HEARTBEAT_PROGRESS_START)
        # Early samples strictly grow; far-out samples may saturate at the
        # cap due to float precision, so allow equality there.
        for earlier, later in pairwise(samples[:5]):
            assert later > earlier
        assert samples[-1] <= _HEARTBEAT_PROGRESS_CAP
        assert samples[-1] < 1.0

    def test_heartbeat_progress_handles_degenerate_tau(self):
        """A zero-or-negative tau must not NaN/explode the bar; it should
        pin to the band start so the worker still emits a clean value."""
        assert _heartbeat_progress(10.0, tau=0.0) == pytest.approx(_HEARTBEAT_PROGRESS_START)
        assert _heartbeat_progress(-5.0, tau=60.0) == pytest.approx(_HEARTBEAT_PROGRESS_START)

    def test_compute_tau_falls_back_when_duration_unknown(self):
        assert _compute_heartbeat_tau(None) == pytest.approx(_HEARTBEAT_TAU_FALLBACK_SEC)
        assert _compute_heartbeat_tau(0.0) == pytest.approx(_HEARTBEAT_TAU_FALLBACK_SEC)

    def test_compute_tau_clamps_to_bounds(self):
        # 10-second clip: raw 2.5 → clamps up to min.
        assert _compute_heartbeat_tau(10.0) == pytest.approx(_HEARTBEAT_TAU_MIN_SEC)
        # 10-hour clip: raw 9000 → clamps down to max.
        assert _compute_heartbeat_tau(36000.0) == pytest.approx(_HEARTBEAT_TAU_MAX_SEC)

    def test_compute_tau_scales_within_band(self):
        # 20-minute clip, ratio 0.25 → 300 s, inside the [30, 600] band.
        assert _compute_heartbeat_tau(1200.0) == pytest.approx(300.0)

    def test_heartbeat_progress_varies_across_ticks(self):
        """Regression: the previous implementation pegged the bar at 0.2
        for the whole diarize stage. Confirm consecutive heartbeat ticks
        produce distinct progress values so users see visible motion.
        """
        import time as time_module

        stage = DiarizeStage(hf_token="test-token", device="cpu", heartbeat_interval=1)

        def _slow_diarize(**_kwargs):
            time_module.sleep(2.5)
            return "diarize_segments"

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value = MagicMock(side_effect=_slow_diarize)
        mock_diarize_module = MagicMock()
        mock_diarize_module.DiarizationPipeline = mock_pipeline_cls

        progress_calls: list[tuple[float, str]] = []
        with patch.dict("sys.modules", {"whisperx.diarize": mock_diarize_module, "whisperx": MagicMock()}):
            stage.execute(
                {"audio_path": "/tmp/test.wav", "duration": 600.0},
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )

        heartbeat_progress = [p for p, m in progress_calls if "已執行" in m]
        assert len(heartbeat_progress) >= 1
        # Progress is strictly greater than the stage-start value (0.2), and
        # if there are multiple ticks they are monotonically non-decreasing.
        assert all(p > _HEARTBEAT_PROGRESS_START for p in heartbeat_progress)
        assert all(p < _HEARTBEAT_PROGRESS_CAP for p in heartbeat_progress)
        for earlier, later in pairwise(heartbeat_progress):
            assert later >= earlier
        # Folded in from the removed twin sleep test: the stage still ends
        # with the genuine DIARIZE_DONE completion call.
        assert progress_calls[-1] == (1.0, DIARIZE_DONE)

    def test_heartbeat_disabled_when_interval_zero(self):
        stage = DiarizeStage(hf_token="test-token", device="cpu", heartbeat_interval=0)
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value = MagicMock(return_value="diarize_segments")
        mock_diarize_module = MagicMock()
        mock_diarize_module.DiarizationPipeline = mock_pipeline_cls

        progress_calls: list[tuple[float, str]] = []
        with patch.dict("sys.modules", {"whisperx.diarize": mock_diarize_module, "whisperx": MagicMock()}):
            stage.execute(
                {"audio_path": "/tmp/test.wav"},
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )
        assert not any("已執行" in m for _, m in progress_calls)

    def test_heartbeat_thread_cleaned_up_when_pipeline_raises(self):
        """The heartbeat is wrapped in a contextmanager whose finally
        block sets stop_event and joins the daemon. If the pipeline
        raises mid-execution the thread must still be reaped, otherwise
        future jobs would accumulate idle daemons. This pins the
        contract by ensuring no ``diarize-heartbeat`` thread survives
        the failed call.
        """
        import threading as threading_mod

        stage = DiarizeStage(hf_token="t", device="cpu", heartbeat_interval=1)
        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value = MagicMock(side_effect=RuntimeError("kaboom"))
        mock_diarize_module = MagicMock()
        mock_diarize_module.DiarizationPipeline = mock_pipeline_cls

        names_before = {t.name for t in threading_mod.enumerate()}

        with (
            patch.dict("sys.modules", {"whisperx.diarize": mock_diarize_module, "whisperx": MagicMock()}),
            pytest.raises(DiarizationError, match="kaboom"),
        ):
            stage.execute(
                {"audio_path": "/tmp/test.wav"},
                on_progress=lambda p, m: None,
            )

        new_threads = {t.name for t in threading_mod.enumerate()} - names_before
        assert "diarize-heartbeat" not in new_threads


class TestAssignSpeakersStage:
    def test_no_diarize_skips(self):
        stage = AssignSpeakersStage()
        progress_calls = []
        result = stage.execute(
            {"diarize_result": None, "aligned_result": {"segments": []}},
            on_progress=lambda p, m: progress_calls.append((p, m)),
        )
        assert "final_result" not in result
        assert progress_calls[-1] == (1.0, ASSIGN_SKIPPED)

    def test_success(self):
        stage = AssignSpeakersStage()
        mock_whisperx = MagicMock()
        mock_whisperx.assign_word_speakers.return_value = {"segments": [{"speaker": "S1"}]}
        progress_calls = []

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            result = stage.execute(
                {"diarize_result": "diarize_data", "aligned_result": {"segments": []}},
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )

        assert "final_result" in result
        assert progress_calls[-1] == (1.0, ASSIGN_DONE)

    def test_exception_falls_back(self):
        stage = AssignSpeakersStage()
        mock_whisperx = MagicMock()
        mock_whisperx.assign_word_speakers.side_effect = RuntimeError("fail")
        aligned = {"segments": [{"text": "hello"}]}
        progress_calls = []

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}):
            result = stage.execute(
                {"diarize_result": "data", "aligned_result": aligned},
                on_progress=lambda p, m: progress_calls.append((p, m)),
            )

        assert result["final_result"] is aligned
        assert progress_calls[-1] == (1.0, ASSIGN_FAILED)

    def test_a_failure_message_cannot_forge_a_second_log_line(self, caplog):
        """The fallback logs the exception text, so a control character in it
        would put an attacker-shaped line into the operator log.

        U+2028 is the interesting one: it is Zl, not a C-category character,
        so a control-character filter misses it while ``str.splitlines()``
        still breaks on it. ``%r`` escapes all of them.
        """
        stage = AssignSpeakersStage()
        mock_whisperx = MagicMock()
        mock_whisperx.assign_word_speakers.side_effect = RuntimeError("boom\u2028ERROR forged line\r\n\x1b[31mred")

        with patch.dict("sys.modules", {"whisperx": mock_whisperx}), caplog.at_level(logging.WARNING):
            stage.execute({"diarize_result": "data", "aligned_result": {"segments": []}})

        records = [r for r in caplog.records if "Speaker assignment failed" in r.getMessage()]
        assert len(records) == 1
        message = records[0].getMessage()
        assert len(message.splitlines()) == 1, message
        assert "\u2028" not in message
        assert "\x1b" not in message
