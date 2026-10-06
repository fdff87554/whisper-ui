from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from rq.timeouts import BaseTimeoutException

from whisper_ui.core.constants import (
    YT_DLP_FORMAT_SORT,
    YT_DLP_SOCKET_TIMEOUT,
    YT_DLP_VIDEO_FORMAT,
)
from whisper_ui.core.exceptions import DownloadError
from whisper_ui.core.messages import (
    DOWNLOAD_DONE,
    DOWNLOAD_EXTRACTING_INFO,
    DOWNLOAD_GDRIVE_IN_PROGRESS,
    DOWNLOAD_IN_PROGRESS,
    DOWNLOAD_SOURCE_TRANSIENT,
    DOWNLOAD_TWITTER_RESTRICTED,
)
from whisper_ui.core.url_validation import (
    extract_gdrive_file_id,
    is_google_drive_url,
    is_twitter_url,
    is_valid_gdrive_file_id,
)

if TYPE_CHECKING:
    from whisper_ui.pipeline.base import ProgressCallback

logger = logging.getLogger(__name__)

# Substrings (lowercase) that mark an X download failure the user can act on:
# login walls, age/NSFW gating, protected accounts, or media that needs a
# different (unsupported) extractor such as Broadcasts / Spaces. The
# "no suitable extractor" / "broadcast" markers fire when the ["twitter"] pin
# blocks yt-dlp from re-extracting into a twitter:broadcast / twitter:spaces
# sub-extractor. The markers are heuristic and not version-specific; this was
# last verified against yt-dlp 2026.03.17 on a real Broadcast tweet.
# Deliberately NOT included: bare "unavailable" (matches the transient
# "HTTP Error 503: Service Unavailable") and HTTP 5xx/429 — those are retryable
# and must keep the generic "Failed to download" path, not the cookies hint.
_TWITTER_RESTRICTED_MARKERS = (
    "log in",
    "login",
    "authenticate",
    "nsfw",
    "not authorized",
    "private",
    "unsupported url",
    "no suitable extractor",
    "broadcast",
)

# Substrings (lowercase) that mark a *transient* download failure worth
# retrying with a fresh yt-dlp client. The headline case is X throttling its
# anonymous guest-token endpoint ("Bad guest token"): yt-dlp fetches a new
# token on every attempt but does not retry the rejection itself, so a clean
# retry clears the blip. HTTP 429 and 5xx are likewise server-side and
# retryable. The HTTP codes are matched in their "http error NNN" form so a
# tweet/video id that merely contains "503" cannot trip a false positive.
# (Markers are heuristic, not version-specific; this behaviour was observed on
# yt-dlp 2026.03.17 on the 129 production host.)
_RETRYABLE_MARKERS = (
    "guest token",
    "http error 429",
    "http error 500",
    "http error 502",
    "http error 503",
    "http error 504",
    "service unavailable",
    "temporarily unavailable",
)

# A transient failure gets this many total extraction attempts; the backoff is
# multiplied by the attempt number (2s, then 4s) so X's per-IP guest-token rate
# limit has a moment to clear without risking the stage's job timeout.
_MAX_DOWNLOAD_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 2


def _as_size(size_bytes: int) -> str:
    """Render a byte count for a user-facing message.

    Picks the unit from the magnitude. A fixed "MB" rendered the default
    8 GiB cap as "8192 MB" and anything under half a megabyte as "0 MB",
    which told a reader with a small cap that nothing was allowed at all.
    """
    for unit, scale in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if size_bytes >= scale:
            return f"{size_bytes / scale:.1f} {unit}"
    return f"{size_bytes} bytes"


class DownloadStage:
    def __init__(
        self,
        *,
        max_duration: int = 14400,
        max_file_size: int = 0,
        max_download_size: int = 0,
        twitter_cookies_file: str | None = None,
    ) -> None:
        # max_file_size guards the Google Drive path, which has no duration
        # metadata to enforce the cap on; 0 disables the check. The dispatcher
        # passes the same limit as direct file uploads (max_upload_size).
        self._max_duration = max_duration
        self._max_file_size = max_file_size
        # Separate, looser bound for the yt-dlp path: there the duration cap is
        # the primary limit and this only has to stop a runaway, so it must sit
        # above any legitimate source at that duration rather than at the
        # upload limit. 0 disables the check.
        self._max_download_size = max_download_size
        self._twitter_cookies_file = twitter_cookies_file

    @property
    def name(self) -> str:
        return "download"

    def execute(self, context: dict[str, Any], on_progress: ProgressCallback | None = None) -> dict[str, Any]:
        source_url = context.get("source_url")
        if not source_url:
            return context

        if is_google_drive_url(source_url):
            return self._download_google_drive(source_url, context, on_progress)
        if is_twitter_url(source_url):
            return self._download_via_ytdlp(
                source_url,
                context,
                on_progress,
                allowed_extractors=["twitter"],
                cookiefile=self._cookiefile_if_present(),
            )
        return self._download_via_ytdlp(source_url, context, on_progress, allowed_extractors=["youtube"])

    def _cookiefile_if_present(self) -> str | None:
        """Return the configured Twitter cookies file only when it exists.

        Unset, or a path that does not point at a real file, degrades to None so
        the download falls back to an anonymous attempt instead of crashing.
        """
        path = self._twitter_cookies_file
        if path and Path(path).is_file():
            return path
        return None

    def _download_google_drive(
        self,
        source_url: str,
        context: dict[str, Any],
        on_progress: ProgressCallback | None,
    ) -> dict[str, Any]:
        try:
            import gdown
        except ImportError as err:
            raise DownloadError("gdown is not installed.") from err

        download_dir = Path(context["download_dir"])
        download_dir.mkdir(parents=True, exist_ok=True)

        if on_progress:
            on_progress(0.0, DOWNLOAD_EXTRACTING_INFO)

        parsed = urlparse(source_url)
        file_id = extract_gdrive_file_id(parsed.path, parse_qs(parsed.query))
        # Re-validate at this boundary rather than trusting that source_url is
        # always the canonical, regex-checked URL the submit route produced —
        # the same defense-in-depth posture as the yt-dlp allowed_extractors pin.
        if not file_id or not is_valid_gdrive_file_id(file_id):
            raise DownloadError("Could not extract Google Drive file ID from URL.")

        gdrive_url = f"https://drive.google.com/uc?id={file_id}"
        output_path = str(download_dir) + "/"

        try:
            if on_progress:
                on_progress(0.1, DOWNLOAD_GDRIVE_IN_PROGRESS)

            # gdown 6 dropped the ``fuzzy`` flag (it always extracts the file ID
            # from any Drive URL now) and raises gdown.exceptions.DownloadError
            # on failure instead of returning None. The None guard stays as
            # defence; a raised failure falls through to the generic handler
            # below, whose message already echoes gdown's "share with Anyone" hint.
            result = gdown.download(url=gdrive_url, output=output_path, quiet=True)
            if result is None:
                raise DownloadError(
                    "Failed to download from Google Drive. Make sure the file is shared as 'Anyone with the link'."
                )
        except DownloadError:
            raise
        except BaseTimeoutException:
            raise
        except Exception as e:
            raise DownloadError(f"Failed to download from Google Drive: {e}") from e

        downloaded = Path(result)
        if not downloaded.is_file() or downloaded.stat().st_size == 0:
            raise DownloadError("Download completed but the file was empty or not found.")

        if self._max_file_size and downloaded.stat().st_size > self._max_file_size:
            size_mb = downloaded.stat().st_size / (1024 * 1024)
            limit_mb = self._max_file_size / (1024 * 1024)
            downloaded.unlink(missing_ok=True)
            raise DownloadError(f"Downloaded file ({size_mb:.1f} MB) exceeds the maximum allowed ({limit_mb:.1f} MB).")

        from whisper_ui.pipeline.preprocess import SUPPORTED_EXTENSIONS

        if downloaded.suffix.lower() not in SUPPORTED_EXTENSIONS:
            downloaded.unlink(missing_ok=True)
            raise DownloadError(f"Downloaded file '{downloaded.name}' is not a supported audio or video format.")

        if on_progress:
            on_progress(1.0, DOWNLOAD_DONE)

        context["input_path"] = str(downloaded)
        context["video_title"] = downloaded.stem
        return context

    def _download_via_ytdlp(
        self,
        source_url: str,
        context: dict[str, Any],
        on_progress: ProgressCallback | None,
        *,
        allowed_extractors: list[str],
        cookiefile: str | None = None,
    ) -> dict[str, Any]:
        try:
            import yt_dlp
        except ImportError as err:
            raise DownloadError("yt-dlp is not installed.") from err

        download_dir = Path(context["download_dir"])
        download_dir.mkdir(parents=True, exist_ok=True)

        # A reaped-then-retried job reuses the same download dir, so a killed
        # attempt can leave ``video.mp4.part`` or unmerged DASH fragments
        # behind. Clear them so the glob fallback below cannot pick one up.
        for stale in download_dir.glob("video.*"):
            if stale.is_file():
                stale.unlink()

        if on_progress:
            on_progress(0.0, DOWNLOAD_EXTRACTING_INFO)

        # Size of every file this download has finished, keyed by the path
        # yt-dlp reports. A video+audio merge finishes two files and both have
        # to count, but a *retry* re-reports the half that already succeeded
        # (yt-dlp reuses the complete file and still fires `finished` for it).
        # Keying by name makes the second report overwrite rather than add; a
        # running total would reject the retry for bytes it already counted.
        completed: dict[str, int] = {}

        def progress_hook(d: dict[str, Any]) -> None:
            name = d.get("filename") or ""
            if d["status"] == "downloading":
                # Everything finished *except* this file: its live byte count
                # supersedes any total recorded for it on an earlier attempt.
                #
                # This is the only bound that survives a source which
                # under-reports or omits its size. The pre-download check
                # cannot see those, and yt-dlp's own max_filesize silently
                # skips the download instead of failing, leaving the caller
                # with "no video file was found".
                others = sum(size for other, size in completed.items() if other != name)
                self._guard_download_size(others + d.get("downloaded_bytes", 0))
            elif d["status"] == "finished":
                completed[name] = d.get("total_bytes") or d.get("downloaded_bytes") or 0
                # External downloaders never emit `downloading` at all --
                # yt-dlp's ExternalFD reports a single `finished` event -- so
                # without this the cap is never consulted for them.
                self._guard_download_size(sum(completed.values()))

            if not on_progress:
                return
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
                if total > 0:
                    p = d["downloaded_bytes"] / total
                    on_progress(min(p, 0.99), DOWNLOAD_IN_PROGRESS)
            elif d["status"] == "finished":
                on_progress(1.0, DOWNLOAD_DONE)

        ydl_opts: dict[str, Any] = {
            "format": YT_DLP_VIDEO_FORMAT,
            "outtmpl": str(download_dir / "video.%(ext)s"),
            "merge_output_format": "mp4",
            "format_sort": list(YT_DLP_FORMAT_SORT),
            "noplaylist": True,
            # Defense in depth: the URL is already whitelisted and canonicalised
            # by the validate_*_url helper, but pinning the extractor stops
            # yt-dlp from ever falling back to the generic extractor and
            # fetching an arbitrary (e.g. internal) host.
            "allowed_extractors": allowed_extractors,
            "socket_timeout": YT_DLP_SOCKET_TIMEOUT,
            "progress_hooks": [progress_hook],
            "quiet": True,
            "no_warnings": True,
        }
        # Operator-supplied login cookies (X login-walled / age-restricted posts).
        # Only set when the file actually exists, so an unset/missing path stays
        # an anonymous attempt rather than failing the download.
        if cookiefile:
            ydl_opts["cookiefile"] = cookiefile

        try:
            info = self._extract_with_retries(yt_dlp, source_url, ydl_opts, allowed_extractors=allowed_extractors)
        except BaseException:
            # An aborted download leaves "video.fNNN.mp4.part" behind. The
            # stale sweep above only runs before the next attempt, so without
            # this the bytes sit on disk until the job is deleted -- which is
            # exactly the disk the byte cap exists to protect.
            for partial in download_dir.glob("video.*"):
                if partial.is_file():
                    partial.unlink(missing_ok=True)
            raise

        downloaded = self._resolve_downloaded_path(info, download_dir)
        # Last line: the progress events describe what a downloader *says* it
        # fetched, and a merge writes a file neither event covers. This is the
        # byte count that actually occupies the disk, so it is the one the cap
        # has to agree with. Removing the file keeps an over-cap download from
        # leaving its bytes behind, the same way a failed attempt does.
        landed_bytes = downloaded.stat().st_size
        if self._max_download_size and landed_bytes > self._max_download_size:
            downloaded.unlink(missing_ok=True)
        self._guard_download_size(landed_bytes)

        context["input_path"] = str(downloaded)
        context["video_title"] = info.get("title", "")
        return context

    @staticmethod
    def _resolve_downloaded_path(info: dict[str, Any], download_dir: Path) -> Path:
        """Return the file yt-dlp actually produced.

        ``requested_downloads[0]["filepath"]`` is the post-merge final path
        yt-dlp reports for the download pass. The glob fallback covers info
        dicts that lack it (e.g. older extractor results); it sorts for
        determinism and skips ``.part`` files since ``Path.glob`` order is
        filesystem-dependent and ``video.*`` also matches partial downloads.
        """
        requested = info.get("requested_downloads") or [{}]
        reported = requested[0].get("filepath")
        if reported and Path(reported).is_file():
            return Path(reported)

        candidates = sorted(p for p in download_dir.glob("video.*") if p.is_file() and p.suffix != ".part")
        if not candidates:
            raise DownloadError("Download completed but no video file was found.")
        return candidates[0]

    def _guard_download_size(self, total_bytes: int) -> None:
        """Reject a download once it passes the byte cap.

        Raising ``DownloadError`` rather than a private exception matters:
        ``_extract_with_retries`` re-raises it untouched, so an oversized
        source fails immediately instead of being retried three times.

        **When this stops a download part-way, and when it does not.** The
        guard runs from the progress hook, so it can only interrupt a
        downloader that reports progress. yt-dlp picks one per protocol
        (``downloader/__init__.py``): ``https`` and ``http_dash_segments``
        report continuously, and ``m3u8_native`` reports per fragment, so
        on those the cap stops the transfer mid-flight.

        An external downloader does not report progress. ``FFmpegFD``
        inherits ``ExternalFD`` and emits a single ``finished`` event when
        the child process exits, and two routes reach it. The ``m3u8``
        protocol maps to it directly. ``m3u8_native`` also hands it the
        *whole* transfer whenever ``HlsFD`` cannot handle the manifest
        itself (``downloader/hls.py``, ``can_download`` -> ``FFmpegFD``):
        non-AES-128 ``#EXT-X-KEY``, or AES-128 with ffmpeg present and
        pycryptodomex absent, which is exactly this project's worker image.
        On those routes an oversized file is written to disk in full, then
        rejected and removed -- the bytes do not survive, but they do briefly
        occupy the disk. Nothing here selects an HLS format deliberately; the
        format string prefers https mp4/m4a, so it would take a source that
        offers nothing else.

        Bounding the transient usage would need a watchdog polling the
        partial file and killing the child, which is not worth the
        concurrency for a path this project does not take on purpose.
        """
        if self._max_download_size and total_bytes > self._max_download_size:
            raise DownloadError(f"Media is larger than the maximum allowed ({_as_size(self._max_download_size)}).")

    @staticmethod
    def _declared_size(info: dict[str, Any]) -> int:
        """Bytes the probe says the selected formats add up to, 0 when unknown.

        A merge reports its halves in ``requested_formats``; a single
        progressive stream reports its own size on the info dict.
        """
        formats = info.get("requested_formats") or [info]
        return sum((f.get("filesize") or f.get("filesize_approx") or 0) for f in formats)

    def _extract_with_retries(
        self,
        yt_dlp_mod: Any,
        source_url: str,
        ydl_opts: dict[str, Any],
        *,
        allowed_extractors: list[str],
    ) -> dict[str, Any]:
        """Extract+download via yt-dlp, retrying only *transient* failures.

        Each attempt uses a fresh ``YoutubeDL`` instance so a rejected guest
        token (or other server-side blip) is re-fetched cleanly. Login walls,
        age/NSFW gating and over-length media are not transient and fail fast;
        RQ timeouts propagate untouched so the worker still classifies them as
        timeouts rather than download failures.
        """
        for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
            try:
                return self._extract_once(yt_dlp_mod, source_url, ydl_opts)
            except DownloadError:
                raise
            except BaseTimeoutException:
                raise
            except Exception as e:
                msg = str(e).lower()
                if "twitter" in allowed_extractors and any(m in msg for m in _TWITTER_RESTRICTED_MARKERS):
                    raise DownloadError(DOWNLOAD_TWITTER_RESTRICTED) from e
                if not any(m in msg for m in _RETRYABLE_MARKERS):
                    raise DownloadError(f"Failed to download video: {e}") from e
                if attempt >= _MAX_DOWNLOAD_ATTEMPTS:
                    raise DownloadError(DOWNLOAD_SOURCE_TRANSIENT) from e
                logger.warning(
                    "Download attempt %d/%d failed transiently (%s); retrying",
                    attempt,
                    _MAX_DOWNLOAD_ATTEMPTS,
                    e,
                )
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
        # Defensive: every loop iteration returns or raises above.
        raise DownloadError(DOWNLOAD_SOURCE_TRANSIENT)

    def _extract_once(self, yt_dlp_mod: Any, source_url: str, ydl_opts: dict[str, Any]) -> dict[str, Any]:
        """Run one yt-dlp pass: probe metadata, enforce the duration cap, then download."""
        with yt_dlp_mod.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_url, download=False)
            if info is None:
                raise DownloadError("Failed to extract video information.")

            # Live/upcoming streams report duration=None, which would slip
            # past the cap below as 0 and download until the job timeout
            # kills the worker. yt-dlp auto-fills live_status from is_live,
            # but check both so a sparse extractor result still trips this.
            live_status = info.get("live_status")
            if info.get("is_live") or live_status in ("is_live", "is_upcoming"):
                raise DownloadError("Live or upcoming streams are not supported. Retry after the stream has ended.")

            duration = info.get("duration") or 0
            if duration > self._max_duration:
                # :g keeps whole hours bare (4h) and fractional ones exact
                # (1.5h) so a custom non-hour-aligned cap is not understated.
                raise DownloadError(
                    f"Video duration ({duration}s) exceeds the maximum allowed ({self._max_duration / 3600:g}h)."
                )

            # Cheapest place to stop an oversized source: the probe already
            # ran, so a declared size over the cap costs no bytes at all.
            self._guard_download_size(self._declared_size(info))

            info = ydl.extract_info(source_url, download=True)
            if info is None:
                raise DownloadError("Download returned no information.")
            return info

    def cleanup(self) -> None:
        pass
