"""YouTube playlist flat-extraction for submit-time expansion (metadata only, no media download)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from whisper_ui.core.constants import YT_DLP_SOCKET_TIMEOUT, YT_DLP_TRANSIENT_MARKERS
from whisper_ui.core.messages import DOWNLOAD_SOURCE_TRANSIENT
from whisper_ui.core.url_validation import YouTubeURLError, validate_youtube_url
from whisper_ui.core.ytdlp_logging import YtDlpLogger, neutralise

logger = logging.getLogger(__name__)

# Flat entries surface inaccessible videos only through these placeholder
# titles; there is no availability field in flat mode. Skipping them here
# avoids creating jobs that are guaranteed to fail at the download stage.
_UNAVAILABLE_TITLES = frozenset({"[private video]", "[deleted video]"})

# Lowercase substring markers that classify an extraction failure as the
# playlist itself being inaccessible -- permanently, from the user's side.
# These are only consulted after YT_DLP_TRANSIENT_MARKERS has been ruled out:
# the phrases yt-dlp uses for a throttled or briefly-down source ("HTTP Error
# 503: Service Unavailable", "Service temporarily unavailable") also contain
# "unavailable", and telling someone their playlist is gone when the server
# merely hiccuped sends them to recreate a playlist that is fine.
_UNAVAILABLE_MARKERS = ("private", "does not exist", "unavailable", "removed")


class PlaylistExpansionError(Exception):
    """Base class for failures while expanding a playlist into videos."""


class PlaylistTooLargeError(PlaylistExpansionError):
    """Playlist holds more videos than one submission may create.

    ``count`` is the total reported by YouTube, or None when the flat
    extraction did not include it.
    """

    def __init__(self, count: int | None, limit: int) -> None:
        self.count = count
        self.limit = limit
        super().__init__(f"Playlist has {count or 'more than ' + str(limit)} videos; limit is {limit}.")


class PlaylistEmptyError(PlaylistExpansionError):
    """Playlist exists but contains no playable videos."""


class PlaylistUnavailableError(PlaylistExpansionError):
    """Playlist is private, deleted, or otherwise inaccessible."""


class PlaylistFetchError(PlaylistExpansionError):
    """Playlist metadata could not be fetched (network/server error)."""


@dataclass(frozen=True)
class PlaylistInfo:
    title: str
    video_urls: list[str]
    unavailable_count: int


def expand_playlist(playlist_url: str, *, limit: int) -> PlaylistInfo:
    """Resolve a canonical playlist URL into its videos' canonical watch URLs.

    Runs a metadata-only flat extraction (no media download, typically one
    HTTP round trip). Private/deleted entries are skipped and reported via
    ``unavailable_count``. Raises a PlaylistExpansionError subclass when the
    playlist cannot be expanded; callers map those onto UI error messages.
    """
    try:
        import yt_dlp  # Lazy: only deployments with the frontend extra ship yt-dlp; tests inject a fake module.
    except ImportError as err:
        logger.exception("yt-dlp is not installed; playlist expansion requires the frontend extra")
        raise PlaylistFetchError("yt-dlp is not installed.") from err

    ydl_opts: dict[str, Any] = {
        "extract_flat": "in_playlist",
        # Defense in depth: the URL is already canonicalised by
        # validate_youtube_playlist_url, but pinning the extractor stops
        # yt-dlp from ever falling back to the generic extractor and
        # fetching an arbitrary (e.g. internal) host.
        "allowed_extractors": ["youtube:tab"],
        # Fetch one entry past the limit so an over-limit playlist is
        # detectable without enumerating it in full.
        "playlist_items": f"1:{limit + 1}",
        "socket_timeout": YT_DLP_SOCKET_TIMEOUT,
        "quiet": True,
        "no_warnings": True,
        # quiet/no_warnings do not cover YoutubeDL.trouble(), which writes
        # straight to stderr; see whisper_ui.core.ytdlp_logging.
        "logger": YtDlpLogger(logger),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(playlist_url, download=False)
    except Exception as e:
        msg = str(e).lower()
        # yt-dlp quotes the playlist or video title in its errors, so every
        # branch below handles remote-controlled text. It reaches the log and,
        # for the two that interpolate it, the message the submitter sees.
        detail = neutralise(str(e))
        if any(m in msg for m in YT_DLP_TRANSIENT_MARKERS):
            # Order matters: this has to win over _UNAVAILABLE_MARKERS below.
            # No traceback -- a throttled or briefly-down source is routine.
            logger.warning("Playlist source responded transiently: %s (url=%s)", detail, playlist_url)
            raise PlaylistFetchError(DOWNLOAD_SOURCE_TRANSIENT) from e
        if any(m in msg for m in _UNAVAILABLE_MARKERS):
            # One line, no traceback: an inaccessible playlist is routine user
            # data, but the original message is kept visible so a blocked
            # egress that happens to match a marker can still be diagnosed.
            logger.warning("Playlist not accessible: %s (url=%s)", detail, playlist_url)
            raise PlaylistUnavailableError(f"Playlist is not accessible: {detail}") from e
        logger.exception("Failed to fetch playlist metadata for %s", playlist_url)
        raise PlaylistFetchError(f"Failed to fetch playlist metadata: {detail}") from e

    if info is None:
        raise PlaylistEmptyError("Playlist metadata extraction returned nothing.")

    entries = [entry for entry in (info.get("entries") or []) if entry]
    if len(entries) > limit:
        raise PlaylistTooLargeError(info.get("playlist_count"), limit)

    video_urls: list[str] = []
    unavailable_count = 0
    for entry in entries:
        if (entry.get("title") or "").lower() in _UNAVAILABLE_TITLES:
            unavailable_count += 1
            continue
        try:
            video_urls.append(validate_youtube_url(f"https://www.youtube.com/watch?v={entry.get('id') or ''}"))
        except YouTubeURLError:
            logger.warning("Skipping playlist entry with invalid video id: %r", entry.get("id"))
            unavailable_count += 1

    if not video_urls:
        raise PlaylistEmptyError("Playlist contains no playable videos.")

    return PlaylistInfo(
        title=info.get("title") or "",
        video_urls=video_urls,
        unavailable_count=unavailable_count,
    )
