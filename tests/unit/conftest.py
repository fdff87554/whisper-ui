"""The yt-dlp conformance adapter for this repository.

``tests/unit/test_ytdlp_conformance.py`` is a verbatim copy shared with
Youtube-Downloader and voice-forge, and its canonical region is hashed
and compared across all three. It imports no repository code: this
adapter is the single seam, and everything project-specific lives
here. See ``docs/ytdlp-conformance/`` in Youtube-Downloader, the
canonical home, for the spec and the copy instructions.
"""

from __future__ import annotations

import tempfile
from contextlib import suppress
from functools import partial
from typing import Any

import pytest
import yt_dlp

# tests/ is a package here, so the qualified spelling is the one that
# imports under pytest's default prepend mode.
from tests.unit.test_ytdlp_conformance import DOWNLOAD, EXTRACT, CallSite
from whisper_ui.core.config import Settings

# The entry points the production code calls, never the option dicts
# underneath them: a dict rebuilt here cannot notice one entry point
# being changed on its own.
from whisper_ui.core.url_validation import (
    validate_youtube_url,
)
from whisper_ui.core.ytdlp_logging import neutralise
from whisper_ui.pipeline.download import DownloadStage
from whisper_ui.web.playlist import PlaylistExpansionError, expand_playlist

_YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
_TWITTER_URL = "https://x.com/jack/status/20"
_PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLFgquLnL59alCl_2TQvOiD5Vgm1hCaGSI"
_PLAYLIST_LIMIT = 50


class _StopAfterOptions(BaseException):
    """Raised once there is nothing left to observe, to stop the call.

    Derived from BaseException rather than Exception so that neither
    ``_extract_with_retries``'s retry predicates nor the broad handlers
    in ``expand_playlist`` can swallow it and turn an observation into a
    second attempt or a mapped error.
    """


# What the probe pass has to return for a download to get as far as
# asking for bytes: not None, not live, inside the duration cap, and
# with no declared size to reject. Short of this the download site is
# never seen as one -- the first version of this adapter raised on the
# probe and reported all three sites as extraction.
_PROBE_RESULT = {"id": "probe", "title": "probe", "duration": 1, "formats": []}


class _RecordingYoutubeDL:
    """Stands in for ``yt_dlp.YoutubeDL`` and records how it was built.

    Every call site builds its options inline, so the only honest way
    to read them is to let the entry point run and watch what it hands
    yt-dlp. Rebuilding the dict here would make the adapter a second
    copy of production, and a copy cannot notice production changing.

    ``extract_info`` records the ``download`` flag and then raises,
    because the real work after it -- a landed file to resolve, a size
    to verify -- is not what is being observed and cannot be faked
    without inventing a download.

    This is the only mock in the adapter and it replaces the call that
    would reach the network, which is the case CLAUDE.md allows one
    for.
    """

    recorded: list[dict[str, Any]] = []

    def __init__(self, options: dict[str, Any]) -> None:
        self._record: dict[str, Any] = {"options": dict(options), "downloads": []}
        type(self).recorded.append(self._record)

    def __enter__(self) -> _RecordingYoutubeDL:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def extract_info(self, url: str, **kwargs: Any) -> dict[str, Any]:
        """Serve the probe pass, record every pass, stop at the download.

        ``DownloadStage`` calls this twice on one instance -- once with
        ``download=False`` to read duration and declared size, then
        again with ``download=True`` -- so answering the first call is
        what makes the second one visible at all.
        """
        download = kwargs.get("download")
        self._record["downloads"].append(download)
        if download:
            raise _StopAfterOptions(url)
        return dict(_PROBE_RESULT)


def _record(call: Any) -> dict[str, Any]:
    """Run ``call`` with the stand-in installed and return what it built.

    Two endings are expected and no others. A download reaches
    ``download=True`` and is stopped there. An extraction completes the
    probe pass and then fails on the empty result the stand-in gave it
    -- ``PlaylistExpansionError`` for the playlist -- which is the entry
    point working correctly on data that describes no video. Anything
    else propagates, because it means the call no longer reaches yt-dlp
    the way this adapter assumes.
    """
    real = yt_dlp.YoutubeDL
    _RecordingYoutubeDL.recorded = []
    try:
        yt_dlp.YoutubeDL = _RecordingYoutubeDL  # type: ignore[misc]
        with suppress(_StopAfterOptions, PlaylistExpansionError):
            call()
    finally:
        yt_dlp.YoutubeDL = real  # type: ignore[misc]

    records = _RecordingYoutubeDL.recorded
    if len(records) != 1:
        raise RuntimeError(
            f"expected one YoutubeDL construction, saw {len(records)}; the "
            "entry point no longer drives yt-dlp the way this adapter assumes"
        )
    record = records[0]
    if not record["downloads"]:
        raise RuntimeError(
            "the entry point built a YoutubeDL and never called extract_info, so nothing about its purpose can be read"
        )
    return record


def _download_sites() -> list[CallSite]:
    """The two download sites, read off ``DownloadStage.execute``.

    Built from the real ``Settings`` the way ``worker.stage_tasks``
    builds it, so ``max_bytes`` in the profile is checked against the
    limit production actually runs with rather than a literal repeated
    here.

    Both URLs go through the same public ``execute``; which extractor
    pin they get is the dispatcher's decision, and reaching them
    separately is how a pin applied to one source and not the other
    would show up.
    """
    settings = Settings()
    stage = DownloadStage(
        max_duration=settings.youtube_max_duration,
        max_file_size=settings.max_upload_size,
        max_download_size=settings.youtube_max_download_size,
        twitter_cookies_file=settings.twitter_cookies_file,
    )
    sites = []
    for name, url in (("execute (youtube)", _YOUTUBE_URL), ("execute (x)", _TWITTER_URL)):
        with tempfile.TemporaryDirectory() as download_dir:
            context = {"source_url": url, "download_dir": download_dir}
            record = _record(partial(stage.execute, context))
        sites.append(
            CallSite(
                name=f"DownloadStage.{name}",
                purpose=DOWNLOAD if any(record["downloads"]) else EXTRACT,
                options=record["options"],
            )
        )
    return sites


def _playlist_site() -> CallSite:
    record = _record(partial(expand_playlist, _PLAYLIST_URL, limit=_PLAYLIST_LIMIT))
    return CallSite(
        name="expand_playlist",
        purpose=DOWNLOAD if any(record["downloads"]) else EXTRACT,
        options=record["options"],
    )


class YtDlpSubject:
    """What the conformance test needs to know about this repository."""

    def call_sites(self) -> list[CallSite]:
        """Every place this repository hands work to yt-dlp.

        Three, all in-process: two downloads (YouTube and X, which the
        dispatcher pins to different extractors) and one flat playlist
        extraction. The Google Drive path in ``DownloadStage.execute``
        does not use yt-dlp at all, so it is not a call site here.

        All three are observed rather than declared -- their options
        come from watching the real entry point hand them to yt-dlp.
        """
        return [*_download_sites(), _playlist_site()]

    def canonicalize(self, url: str) -> str | None:
        """The rebuilt URL, or ``None`` when this repository refuses it.

        ``None`` must mean refused: ``validate_youtube_url`` raises, and
        the test reads ``None`` as the declared rejection and an
        exception as a bug.
        """
        try:
            return validate_youtube_url(url)
        except Exception:
            return None

    def neutralize(self, text: str) -> str:
        """Remote-controlled text, made safe to display or log."""
        return neutralise(text)


@pytest.fixture(scope="session")
def ytdlp_subject() -> YtDlpSubject:
    return YtDlpSubject()


# The conformance gaps this repository has not closed, by node id.
# Marked from here rather than in the test file because that file is
# byte-identical across the three projects and its canonical region is
# hashed: an xfail written into it would be drift. Every entry cites
# the issue that closes it, and
# docs/ytdlp-conformance/check-conformance-drift.sh fails when more
# items are marked than xfail_budget allows, so the list can only
# shrink.
#
# Two of the five are findings against the shared test rather than
# against this repository, and say so. Recording them as gaps is the
# least-wrong holding position -- counted and greppable beats hidden --
# but the thing that closes them is a spec revision, not a change here.
_CONFORMANCE_MODULE = "tests/unit/test_ytdlp_conformance.py"

CONFORMANCE_XFAILS = {
    f"{_CONFORMANCE_MODULE}::TestTheExtractorPinHolds::test_every_call_site_passes_the_declared_pin": (
        "The shared test expects one pin per repository; this one narrows "
        "the extractor per source (youtube / youtube:tab / twitter), which "
        "is stricter, not looser. The check needs widening, not the code; "
        "see fdff87554/youtube-downloader#146"
    ),
    f"{_CONFORMANCE_MODULE}::TestTheByteCeilingIsDeclaredAndReachable"
    "::test_every_downloading_site_rejects_unmeterable_formats": (
        "C7.selectable: YT_DLP_VIDEO_FORMAT constrains ext and height but "
        "not protocol, so an HLS rendition is selectable and is preferred "
        "when it carries the higher bitrate; see #187"
    ),
    f"{_CONFORMANCE_MODULE}::TestTheByteCeilingIsDeclaredAndReachable"
    "::test_no_downloading_selector_falls_back_to_an_unmeterable_format": (
        "C7.selectable: every fallback branch of YT_DLP_VIDEO_FORMAT is unconstrained on protocol; see #187"
    ),
    f"{_CONFORMANCE_MODULE}::TestTheTimeBoundIsExplicit"
    "::test_every_in_process_downloading_site_sets_retries_explicitly": (
        "C8.retries: retries and fragment_retries are unset, which is 0 on "
        "the Python API path rather than the CLI's 10; see #188"
    ),
    f"{_CONFORMANCE_MODULE}::TestYtDlpOutputIsCaptured::test_no_call_site_silences_warnings": (
        "no_warnings is set at both call sites, but so is a logger, and "
        "yt-dlp reads no_warnings only when logger is None "
        "(YoutubeDL.py:1145) -- so it silences nothing here. The check is "
        "over-broad; see fdff87554/youtube-downloader#145"
    ),
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Mark the known conformance gaps.

    Marking only. Whether an entry still names a real test is checked
    in ``test_conformance_registry.py``, which resolves the names
    against the module rather than against whatever this run happened
    to collect -- a check that reads the collection cannot tell a
    renamed test from a narrowed selection.
    """
    for item in items:
        reason = CONFORMANCE_XFAILS.get(item.nodeid)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True))
