from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from whisper_ui.core.constants import YT_DLP_TRANSIENT_MARKERS
from whisper_ui.core.ytdlp_logging import YtDlpLogger
from whisper_ui.web.playlist import (
    PlaylistEmptyError,
    PlaylistFetchError,
    PlaylistTooLargeError,
    PlaylistUnavailableError,
    expand_playlist,
)

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf"


def _entry(video_id: str, title: str = "Some Video") -> dict[str, Any]:
    return {"id": video_id, "title": title}


def _make_mock_module(info: dict[str, Any] | None = None, error: Exception | None = None) -> MagicMock:
    """Build a fake yt_dlp module whose YoutubeDL yields ``info`` or raises ``error``."""
    mock_ydl = MagicMock()
    if error is not None:
        mock_ydl.extract_info.side_effect = error
    else:
        mock_ydl.extract_info.return_value = info
    mock_ydl.__enter__ = lambda self: self
    mock_ydl.__exit__ = MagicMock(return_value=False)
    module = MagicMock()
    module.YoutubeDL.return_value = mock_ydl
    return module


class TestExpandPlaylistSuccess:
    def test_returns_title_and_canonical_watch_urls(self):
        info = {
            "title": "Team Meetings 2026Q2",
            "playlist_count": 2,
            "entries": [_entry("dQw4w9WgXcQ"), _entry("0VH1Lim8gL8")],
        }
        with patch.dict("sys.modules", {"yt_dlp": _make_mock_module(info)}):
            result = expand_playlist(PLAYLIST_URL, limit=50)

        assert result.title == "Team Meetings 2026Q2"
        assert result.video_urls == [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/watch?v=0VH1Lim8gL8",
        ]
        assert result.unavailable_count == 0

    def test_private_and_deleted_entries_skipped_and_counted(self):
        info = {
            "title": "Mixed",
            "entries": [
                _entry("dQw4w9WgXcQ"),
                _entry("aaaaaaaaaaa", title="[Private video]"),
                _entry("bbbbbbbbbbb", title="[Deleted video]"),
            ],
        }
        with patch.dict("sys.modules", {"yt_dlp": _make_mock_module(info)}):
            result = expand_playlist(PLAYLIST_URL, limit=50)

        assert result.video_urls == ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
        assert result.unavailable_count == 2

    def test_entry_with_invalid_video_id_skipped_and_counted(self):
        info = {"title": "T", "entries": [_entry("dQw4w9WgXcQ"), _entry("bad"), {"title": "no id"}]}
        with patch.dict("sys.modules", {"yt_dlp": _make_mock_module(info)}):
            result = expand_playlist(PLAYLIST_URL, limit=50)

        assert result.video_urls == ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
        assert result.unavailable_count == 2

    def test_missing_title_defaults_to_empty_string(self):
        info = {"entries": [_entry("dQw4w9WgXcQ")]}
        with patch.dict("sys.modules", {"yt_dlp": _make_mock_module(info)}):
            result = expand_playlist(PLAYLIST_URL, limit=50)

        assert result.title == ""

    def test_passes_flat_extraction_options_to_yt_dlp(self):
        module = _make_mock_module({"title": "T", "entries": [_entry("dQw4w9WgXcQ")]})
        with patch.dict("sys.modules", {"yt_dlp": module}):
            expand_playlist(PLAYLIST_URL, limit=50)

        opts = module.YoutubeDL.call_args.args[0]
        assert opts["extract_flat"] == "in_playlist"
        assert opts["allowed_extractors"] == ["youtube:tab"]
        assert opts["playlist_items"] == "1:51"
        assert opts["socket_timeout"] > 0
        # quiet/no_warnings leave YoutubeDL.trouble() writing to stderr, so a
        # logger is the only thing keeping extraction failures inside the
        # logging framework.
        assert isinstance(opts["logger"], YtDlpLogger)


class TestExpandPlaylistTransientFailures:
    @pytest.mark.parametrize(
        "message",
        [
            "HTTP Error 503: Service Unavailable",
            "Service temporarily unavailable",
            "HTTP Error 429: Too Many Requests",
        ],
    )
    def test_server_side_blips_are_not_reported_as_a_missing_playlist(self, message: str):
        # Every one of these contains a substring in _UNAVAILABLE_MARKERS, so
        # without the transient check first the user is told their playlist is
        # gone and goes off to recreate a playlist that is fine.
        module = _make_mock_module(error=Exception(message))
        with patch.dict("sys.modules", {"yt_dlp": module}), pytest.raises(PlaylistFetchError) as exc:
            expand_playlist(PLAYLIST_URL, limit=50)

        assert not isinstance(exc.value, PlaylistUnavailableError)

    @pytest.mark.parametrize(
        "message",
        [
            "This playlist is private",
            "The playlist does not exist",
            "Video unavailable",
            "This playlist has been removed",
        ],
    )
    def test_permanent_failures_still_report_the_playlist_as_inaccessible(self, message: str):
        module = _make_mock_module(error=Exception(message))
        with patch.dict("sys.modules", {"yt_dlp": module}), pytest.raises(PlaylistUnavailableError):
            expand_playlist(PLAYLIST_URL, limit=50)

    def test_transient_markers_are_shared_with_the_download_path(self):
        # The two paths disagreeing about what "unavailable" means is the bug
        # this classification order exists to close.
        from whisper_ui.pipeline.download import YT_DLP_TRANSIENT_MARKERS as download_markers

        assert download_markers is YT_DLP_TRANSIENT_MARKERS


class TestExpandPlaylistTooLarge:
    def test_over_limit_raises_with_reported_count(self):
        info = {"title": "Big", "playlist_count": 250, "entries": [_entry(f"vid{i:08d}") for i in range(3)]}
        with (
            patch.dict("sys.modules", {"yt_dlp": _make_mock_module(info)}),
            pytest.raises(PlaylistTooLargeError) as exc_info,
        ):
            expand_playlist(PLAYLIST_URL, limit=2)

        assert exc_info.value.count == 250
        assert exc_info.value.limit == 2

    def test_over_limit_without_playlist_count_reports_none(self):
        info = {"title": "Big", "entries": [_entry(f"vid{i:08d}") for i in range(3)]}
        with (
            patch.dict("sys.modules", {"yt_dlp": _make_mock_module(info)}),
            pytest.raises(PlaylistTooLargeError) as exc_info,
        ):
            expand_playlist(PLAYLIST_URL, limit=2)

        assert exc_info.value.count is None


class TestExpandPlaylistEmpty:
    @pytest.mark.parametrize(
        "info",
        [
            None,
            {"title": "T", "entries": []},
            {"title": "T"},
            {"title": "T", "entries": [_entry("aaaaaaaaaaa", title="[Private video]")]},
        ],
    )
    def test_no_playable_videos_raises_empty(self, info):
        with (
            patch.dict("sys.modules", {"yt_dlp": _make_mock_module(info)}),
            pytest.raises(PlaylistEmptyError),
        ):
            expand_playlist(PLAYLIST_URL, limit=50)


class TestExpandPlaylistErrors:
    @pytest.mark.parametrize(
        "message",
        [
            "ERROR: [youtube:tab] This playlist is private",
            "ERROR: [youtube:tab] The playlist does not exist",
            "ERROR: [youtube:tab] This playlist is unavailable",
        ],
    )
    def test_inaccessible_playlist_raises_unavailable(self, message):
        with (
            patch.dict("sys.modules", {"yt_dlp": _make_mock_module(error=Exception(message))}),
            pytest.raises(PlaylistUnavailableError),
        ):
            expand_playlist(PLAYLIST_URL, limit=50)

    def test_network_error_raises_fetch_error(self):
        with (
            patch.dict("sys.modules", {"yt_dlp": _make_mock_module(error=Exception("urlopen error timed out"))}),
            pytest.raises(PlaylistFetchError),
        ):
            expand_playlist(PLAYLIST_URL, limit=50)

    def test_missing_yt_dlp_raises_fetch_error(self):
        with (
            patch.dict("sys.modules", {"yt_dlp": None}),
            pytest.raises(PlaylistFetchError, match="not installed"),
        ):
            expand_playlist(PLAYLIST_URL, limit=50)


class TestExpandPlaylistLogging:
    """Expansion failures must leave a server-side trail: the route turns
    them into generic UI messages, so the log is the only place a missing
    dependency or blocked egress is diagnosable."""

    def test_fetch_error_logs_error_with_traceback(self, caplog):
        with (
            caplog.at_level(logging.ERROR, logger="whisper_ui.web.playlist"),
            patch.dict("sys.modules", {"yt_dlp": _make_mock_module(error=Exception("urlopen error timed out"))}),
            pytest.raises(PlaylistFetchError),
        ):
            expand_playlist(PLAYLIST_URL, limit=50)

        record = next(r for r in caplog.records if "Failed to fetch playlist metadata" in r.getMessage())
        assert record.exc_info is not None

    def test_missing_yt_dlp_logs_error(self, caplog):
        with (
            caplog.at_level(logging.ERROR, logger="whisper_ui.web.playlist"),
            patch.dict("sys.modules", {"yt_dlp": None}),
            pytest.raises(PlaylistFetchError),
        ):
            expand_playlist(PLAYLIST_URL, limit=50)

        assert any("yt-dlp is not installed" in r.getMessage() for r in caplog.records)

    def test_inaccessible_playlist_logs_warning_without_error(self, caplog):
        error = Exception("ERROR: [youtube:tab] This playlist is private")
        with (
            caplog.at_level(logging.WARNING, logger="whisper_ui.web.playlist"),
            patch.dict("sys.modules", {"yt_dlp": _make_mock_module(error=error)}),
            pytest.raises(PlaylistUnavailableError),
        ):
            expand_playlist(PLAYLIST_URL, limit=50)

        warning = next(r for r in caplog.records if "Playlist not accessible" in r.getMessage())
        assert warning.levelno == logging.WARNING
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)
