from __future__ import annotations

import logging

import pytest

from whisper_ui.core.ytdlp_logging import YtDlpLogger, neutralise


class TestNeutralise:
    @pytest.mark.parametrize(
        "raw",
        [
            "title\nduration: 9999",
            "title\rduration: 9999",
            "title\x1b[2Kduration: 9999",
            "title‮duration: 9999",
            "title\x08\x08duration: 9999",
        ],
    )
    def test_control_characters_cannot_forge_a_second_line(self, raw: str) -> None:
        cleaned = neutralise(raw)
        assert not any(ch in cleaned for ch in "\n\r\x1b\x08")
        assert "‮" not in cleaned

    def test_printable_text_survives_unchanged(self) -> None:
        assert neutralise("Lecture 3 — 第三講 (2026)") == "Lecture 3 — 第三講 (2026)"

    def test_overlong_text_is_truncated(self) -> None:
        assert neutralise("x" * 1000, max_length=10) == "x" * 10 + "..."


class TestYtDlpLogger:
    def test_debug_prefixed_messages_stay_at_debug_level(self, caplog) -> None:
        target = logging.getLogger("test.ytdlp.debug")
        with caplog.at_level(logging.DEBUG, logger=target.name):
            YtDlpLogger(target).debug("[debug] probing formats")
        assert caplog.records[0].levelno == logging.DEBUG

    def test_unprefixed_debug_is_informational(self, caplog) -> None:
        # yt-dlp routes its ordinary progress output through debug() without
        # the prefix, so treating all of it as DEBUG would hide it.
        target = logging.getLogger("test.ytdlp.info")
        with caplog.at_level(logging.DEBUG, logger=target.name):
            YtDlpLogger(target).debug("Downloading webpage")
        assert caplog.records[0].levelno == logging.INFO

    @pytest.mark.parametrize(
        ("method", "level"),
        [("info", logging.INFO), ("warning", logging.WARNING), ("error", logging.ERROR)],
    )
    def test_levels_map_through(self, caplog, method: str, level: int) -> None:
        target = logging.getLogger(f"test.ytdlp.{method}")
        with caplog.at_level(logging.DEBUG, logger=target.name):
            getattr(YtDlpLogger(target), method)("something happened")
        assert caplog.records[0].levelno == level

    def test_remote_text_is_neutralised_before_it_reaches_the_log(self, caplog) -> None:
        target = logging.getLogger("test.ytdlp.neutralise")
        with caplog.at_level(logging.ERROR, logger=target.name):
            YtDlpLogger(target).error("ERROR: [youtube] evil\ntitle: forged")
        assert "\n" not in caplog.records[0].getMessage()
