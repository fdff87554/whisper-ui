"""The error-string heuristics have to decay loudly, not silently."""

from __future__ import annotations

import yt_dlp

from whisper_ui.core.constants import (
    YT_DLP_MARKERS_VERIFIED_AGAINST,
    YT_DLP_TRANSIENT_MARKERS,
)


class TestMarkersAreVersionBound:
    def test_markers_are_bound_to_the_installed_yt_dlp(self) -> None:
        # Failing here is the intended outcome of a yt-dlp bump: re-read the
        # release's error strings, adjust the marker lists if they moved, then
        # move YT_DLP_MARKERS_VERIFIED_AGAINST forward in the same commit.
        assert yt_dlp.version.__version__ == YT_DLP_MARKERS_VERIFIED_AGAINST


class TestTransientMarkersMatchRealPhrasings:
    def test_http_codes_are_matched_in_their_full_form(self) -> None:
        # A bare "503" would also match a video id containing those digits.
        assert all(marker.startswith("http error ") for marker in YT_DLP_TRANSIENT_MARKERS if marker[-3:].isdigit())

    def test_the_phrases_yt_dlp_actually_emits_are_covered(self) -> None:
        # "Service Unavailable" is the stdlib reason phrase for 503; the other
        # two are literals in yt-dlp's own extractors.
        for phrase in (
            "HTTP Error 503: Service Unavailable",
            "Service temporarily unavailable",
            "HTTP Error 429: Too Many Requests",
        ):
            assert any(m in phrase.lower() for m in YT_DLP_TRANSIENT_MARKERS), phrase
