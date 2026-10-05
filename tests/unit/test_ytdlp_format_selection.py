"""What the format string actually selects, run through yt-dlp's own selector.

Asserting on the string alone cannot catch the two mistakes that matter here:
a filter whose syntax yt-dlp rejects, and a fallback that quietly admits the
resolutions the ceiling exists to exclude. These drive the real selector over
synthetic format lists, so no network is involved.
"""

from __future__ import annotations

from typing import Any

import yt_dlp

from whisper_ui.core.constants import YT_DLP_FORMAT_SORT, YT_DLP_MAX_HEIGHT, YT_DLP_VIDEO_FORMAT


def _select(formats: list[dict[str, Any]]) -> list[str]:
    ydl = yt_dlp.YoutubeDL(
        {
            "quiet": True,
            "no_warnings": True,
            "format": YT_DLP_VIDEO_FORMAT,
            "format_sort": list(YT_DLP_FORMAT_SORT),
            "simulate": True,
        }
    )
    chosen = ydl.build_format_selector(YT_DLP_VIDEO_FORMAT)({"formats": formats, "incomplete_formats": False})
    return [f["format_id"] for f in chosen]


def _video(format_id: str, height: int | None, *, ext: str = "mp4", vcodec: str = "avc1") -> dict[str, Any]:
    fmt: dict[str, Any] = {
        "format_id": format_id,
        "ext": ext,
        "vcodec": vcodec,
        "acodec": "aac",
        "url": "https://example.invalid/media",
    }
    if height is not None:
        fmt["height"] = height
    return fmt


class TestTheFormatStringIsValid:
    def test_yt_dlp_accepts_the_selector(self) -> None:
        # "[height<=1080?]" -- the question mark after the value rather than
        # after the operator -- raises SyntaxError at construction time, and
        # no string assertion would notice.
        assert _select([_video("only", 720)]) == ["only"]


class TestTheCeilingHolds:
    def test_a_known_height_over_the_ceiling_is_rejected(self) -> None:
        # Selecting nothing is what yt-dlp then reports as "Requested format
        # is not available"; the point is that 2160p is not silently taken.
        assert _select([_video("uhd", 2160)]) == []

    def test_the_highest_allowed_rendition_wins(self) -> None:
        assert _select([_video("sd", 480), _video("hd", YT_DLP_MAX_HEIGHT), _video("uhd", 2160)]) == ["hd"]


class TestUnknownHeightsStaySelectable:
    def test_a_format_with_no_height_is_still_chosen(self) -> None:
        # Some X posts report no height. The last resort exists for them, and
        # relaxing it must not also admit known-oversized formats.
        assert _select([_video("noheight", None)]) == ["noheight"]

    def test_unknown_height_is_preferred_over_an_oversized_known_one(self) -> None:
        assert _select([_video("noheight", None), _video("uhd", 2160)]) == ["noheight"]
