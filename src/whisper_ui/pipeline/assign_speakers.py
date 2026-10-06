from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from rq.timeouts import BaseTimeoutException

from whisper_ui.core.messages import ASSIGN_DONE, ASSIGN_FAILED, ASSIGN_RUNNING, ASSIGN_SKIPPED

if TYPE_CHECKING:
    from whisper_ui.pipeline.base import ProgressCallback

logger = logging.getLogger(__name__)


class AssignSpeakersStage:
    @property
    def name(self) -> str:
        return "assign_speakers"

    def execute(self, context: dict[str, Any], on_progress: ProgressCallback | None = None) -> dict[str, Any]:
        if on_progress:
            on_progress(0.0, ASSIGN_RUNNING)

        diarize_result = context.get("diarize_result")
        aligned_result = context.get("aligned_result")

        if diarize_result is None or aligned_result is None:
            if on_progress:
                on_progress(1.0, ASSIGN_SKIPPED)
            return context

        try:
            import whisperx

            result = whisperx.assign_word_speakers(diarize_result, aligned_result)

            if on_progress:
                on_progress(1.0, ASSIGN_DONE)

            context["final_result"] = result
            return context

        except BaseTimeoutException:
            # RQ's death penalty must never be swallowed by the "degrade to
            # unassigned" fallback — otherwise the job would keep running
            # past its deadline. Propagate unchanged.
            raise
        except Exception as e:
            # %r, not %s: repr escapes control characters, so an exception
            # message carrying CR, U+2028 or an ANSI escape cannot forge a
            # second log line. Same reason finalize_failure logs detail=%r.
            logger.warning("Speaker assignment failed: %r. Using aligned result without speakers.", e)
            context["final_result"] = aligned_result
            if on_progress:
                on_progress(1.0, ASSIGN_FAILED)
            return context

    def cleanup(self) -> None:
        pass
