from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from whisper_ui.core.constants import YT_DLP_MAX_HEIGHT
from whisper_ui.core.exceptions import DownloadError
from whisper_ui.pipeline.download import _MAX_DOWNLOAD_ATTEMPTS, DownloadStage


class TestDownloadStageNoOp:
    def test_no_source_url_passes_through(self):
        stage = DownloadStage()
        context: dict[str, Any] = {"input_path": "/some/file.mp3"}
        result = stage.execute(context)
        assert result["input_path"] == "/some/file.mp3"

    def test_empty_source_url_passes_through(self):
        stage = DownloadStage()
        context: dict[str, Any] = {"source_url": "", "input_path": "/some/file.mp3"}
        result = stage.execute(context)
        assert result["input_path"] == "/some/file.mp3"


class TestDownloadStageWithMock:
    @pytest.fixture
    def download_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "downloads"
        d.mkdir()
        return d

    @pytest.fixture
    def context(self, download_dir: Path) -> dict[str, Any]:
        return {
            "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "download_dir": str(download_dir),
            "input_path": "",
        }

    def _make_mock_ydl(
        self,
        download_dir: Path,
        duration: int = 120,
        title: str = "Test Video",
        extra_info: dict[str, Any] | None = None,
    ):
        """Create a mock YoutubeDL that simulates a successful download."""
        mock_ydl_instance = MagicMock()

        def extract_info(url, download=True):
            info = {"duration": duration, "title": title, **(extra_info or {})}
            if download:
                # Simulate file creation
                (Path(download_dir) / "video.mp4").write_bytes(b"fake video")
            return info

        mock_ydl_instance.extract_info = extract_info
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        return mock_ydl_instance

    def test_successful_download(self, context, download_dir):
        mock_ydl = self._make_mock_ydl(download_dir)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage()
            result = stage.execute(context)

        assert result["input_path"] == str(download_dir / "video.mp4")
        assert result["video_title"] == "Test Video"

    def test_progress_callback_called(self, context, download_dir):
        mock_ydl = self._make_mock_ydl(download_dir)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl

        progress_log: list[tuple[float, str]] = []

        def on_progress(p: float, msg: str) -> None:
            progress_log.append((p, msg))

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage()
            stage.execute(context, on_progress=on_progress)

        assert len(progress_log) > 0
        assert progress_log[0][1] == "正在取得影片資訊..."

    def test_duration_exceeds_limit(self, context, download_dir):
        mock_ydl = self._make_mock_ydl(download_dir, duration=50000)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage(max_duration=3600)
            with pytest.raises(DownloadError, match="exceeds the maximum"):
                stage.execute(context)

    def test_live_stream_rejected_before_download_pass(self, context, download_dir):
        # A live stream reports duration=None, which must not bypass the
        # duration cap as 0; the rejection has to happen on the metadata
        # probe, before any download pass starts.
        mock_ydl_instance = MagicMock()

        def extract_info(url, download=True):
            assert not download, "live stream must be rejected before the download pass"
            return {"duration": None, "title": "Live", "is_live": True, "live_status": "is_live"}

        mock_ydl_instance.extract_info = extract_info
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match=r"[Ll]ive"):
                stage.execute(context)

    def test_upcoming_stream_rejected(self, context, download_dir):
        mock_ydl = self._make_mock_ydl(download_dir, extra_info={"live_status": "is_upcoming"})
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match=r"[Ll]ive"):
                stage.execute(context)

    def test_finished_live_vod_downloads_normally(self, context, download_dir):
        mock_ydl = self._make_mock_ydl(download_dir, extra_info={"live_status": "was_live", "is_live": False})
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage()
            result = stage.execute(context)

        assert result["input_path"] == str(download_dir / "video.mp4")

    def test_input_path_prefers_ytdlp_reported_filepath(self, context, download_dir):
        # When two media files coexist, the path yt-dlp reports wins over the
        # filesystem-order glob fallback.
        reported = download_dir / "video.webm"

        def write_files(url, download=True):
            if download:
                (download_dir / "video.mp4").write_bytes(b"other file")
                reported.write_bytes(b"reported file")
            return {
                "duration": 120,
                "title": "Test Video",
                "requested_downloads": [{"filepath": str(reported)}],
            }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info = write_files
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage()
            result = stage.execute(context)

        assert result["input_path"] == str(reported)

    def test_stale_artifacts_cleared_before_download(self, context, download_dir):
        # A reaped attempt can leave partial files behind; they must not be
        # picked up as input_path by the retry.
        (download_dir / "video.mp4.part").write_bytes(b"stale partial")
        (download_dir / "video.f614.mp4").write_bytes(b"stale fragment")

        mock_ydl = self._make_mock_ydl(download_dir)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage()
            result = stage.execute(context)

        assert result["input_path"] == str(download_dir / "video.mp4")
        assert not (download_dir / "video.mp4.part").exists()
        assert not (download_dir / "video.f614.mp4").exists()

    def test_glob_fallback_skips_part_files_and_sorts(self, context, download_dir):
        # Without requested_downloads, the fallback must never resolve to a
        # .part file regardless of filesystem ordering.
        def write_files(url, download=True):
            if download:
                (download_dir / "video.mp4.part").write_bytes(b"partial")
                (download_dir / "video.mp4").write_bytes(b"final")
            return {"duration": 120, "title": "Test Video"}

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info = write_files
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage()
            result = stage.execute(context)

        assert result["input_path"] == str(download_dir / "video.mp4")

    def test_extract_info_returns_none(self, context, download_dir):
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = None
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match="Failed to extract"):
                stage.execute(context)

    def test_network_error_raises_download_error(self, context, download_dir):
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = Exception("Network unreachable")
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match="Failed to download"):
                stage.execute(context)

    def test_youtube_error_with_marker_word_stays_generic(self, context, download_dir):
        # The restricted-marker classification must never run on the youtube
        # path: a youtube "Private video" error keeps the generic message.
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = Exception("ERROR: Private video. Sign in to view.")
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with (
            patch.dict("sys.modules", {"yt_dlp": mock_module}),
            pytest.raises(DownloadError, match="Failed to download"),
        ):
            DownloadStage().execute(context)

    def test_restricts_yt_dlp_to_youtube_extractor(self, context, download_dir):
        mock_ydl = self._make_mock_ydl(download_dir)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            DownloadStage().execute(context)

        ydl_opts = mock_module.YoutubeDL.call_args.args[0]
        assert ydl_opts["allowed_extractors"] == ["youtube"]

    def test_video_format_caps_every_bounded_fallback(self, context, download_dir):
        # A ceiling on only the first selector lets a video with no capped
        # adaptive rendition fall through to full resolution, which is the
        # behaviour this cap exists to stop.
        mock_ydl = self._make_mock_ydl(download_dir)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            DownloadStage().execute(context)

        selectors = mock_module.YoutubeDL.call_args.args[0]["format"].split("/")
        assert selectors[-1] == "best", "the last resort must stay unbounded"
        assert all(f"height<={YT_DLP_MAX_HEIGHT}" in s for s in selectors[:-1])

    def test_format_sort_prefers_h264_over_the_yt_dlp_default(self, context, download_dir):
        # yt-dlp's default sort ranks AV1 first and YouTube serves AV1 inside
        # mp4, so without this the viewer's <video> element gets a codec many
        # devices cannot decode in hardware.
        mock_ydl = self._make_mock_ydl(download_dir)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            DownloadStage().execute(context)

        assert mock_module.YoutubeDL.call_args.args[0]["format_sort"][0] == "vcodec:h264"

    def test_codec_preference_does_not_become_a_filter(self, context, download_dir):
        # A [vcodec^=avc1] filter would reject AV1-only videos outright. The
        # preference has to live in format_sort so those still download.
        mock_ydl = self._make_mock_ydl(download_dir)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            DownloadStage().execute(context)

        assert "vcodec" not in mock_module.YoutubeDL.call_args.args[0]["format"]

    def test_cleanup_is_noop(self):
        stage = DownloadStage()
        stage.cleanup()  # Should not raise


class TestDownloadStageImportError:
    def test_missing_yt_dlp_raises(self):
        context: dict[str, Any] = {
            "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "download_dir": "/tmp/test",
            "input_path": "",
        }
        with patch.dict("sys.modules", {"yt_dlp": None}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match="yt-dlp is not installed"):
                stage.execute(context)


class TestGoogleDriveDownload:
    @pytest.fixture
    def download_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "downloads"
        d.mkdir()
        return d

    @pytest.fixture
    def context(self, download_dir: Path) -> dict[str, Any]:
        return {
            "source_url": "https://drive.google.com/uc?export=download&id=1AbCdEfGhIjKlMnOpQrStUvWxYz",
            "download_dir": str(download_dir),
            "input_path": "",
        }

    def test_successful_gdrive_download(self, context, download_dir):
        resolved_file = str(download_dir / "meeting_recording.m4a")

        def mock_download(url, output, quiet=True):
            out_path = Path(output)
            if out_path.is_dir():
                out_path = out_path / "meeting_recording.m4a"
            out_path.write_bytes(b"fake audio content")
            return str(out_path)

        mock_gdown = MagicMock()
        mock_gdown.download = mock_download

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage()
            result = stage.execute(context)

        assert result["input_path"] == resolved_file
        assert result["video_title"] == "meeting_recording"

    def test_gdrive_progress_callback(self, context, download_dir):
        def mock_download(url, output, quiet=True):
            out_path = Path(output) / "audio.mp3"
            out_path.write_bytes(b"fake audio content")
            return str(out_path)

        mock_gdown = MagicMock()
        mock_gdown.download = mock_download

        progress_log: list[tuple[float, str]] = []

        def on_progress(p: float, msg: str) -> None:
            progress_log.append((p, msg))

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage()
            stage.execute(context, on_progress=on_progress)

        assert len(progress_log) >= 2
        assert progress_log[0][1] == "正在取得影片資訊..."
        assert progress_log[-1][1] == "音訊下載完成。"

    def test_gdrive_download_returns_none(self, context, download_dir):
        mock_gdown = MagicMock()
        mock_gdown.download.return_value = None

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match="Failed to download from Google Drive"):
                stage.execute(context)

    def test_gdrive_missing_gdown_raises(self):
        context: dict[str, Any] = {
            "source_url": "https://drive.google.com/uc?export=download&id=abc123xyz",
            "download_dir": "/tmp/test",
            "input_path": "",
        }
        with patch.dict("sys.modules", {"gdown": None}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match="gdown is not installed"):
                stage.execute(context)

    def test_gdrive_invalid_file_id(self, download_dir):
        context: dict[str, Any] = {
            "source_url": "https://drive.google.com/drive/my-drive",
            "download_dir": str(download_dir),
            "input_path": "",
        }
        mock_gdown = MagicMock()

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match="Could not extract Google Drive file ID"):
                stage.execute(context)

    def test_gdrive_network_error(self, context, download_dir):
        mock_gdown = MagicMock()
        mock_gdown.download.side_effect = Exception("Connection refused")

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match="Failed to download from Google Drive"):
                stage.execute(context)

    def test_gdrive_empty_file_raises(self, context, download_dir):
        def mock_download(url, output, quiet=True):
            out_path = Path(output) / "audio.mp3"
            out_path.write_bytes(b"")
            return str(out_path)

        mock_gdown = MagicMock()
        mock_gdown.download = mock_download

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match="empty or not found"):
                stage.execute(context)

    def test_gdrive_file_exceeding_size_cap_raises_and_deletes(self, context, download_dir):
        # Drive files carry no duration metadata, so the byte cap is the only
        # guard against an oversized file filling the disk.
        def mock_download(url, output, quiet=True):
            out_path = Path(output) / "audio.mp3"
            out_path.write_bytes(b"x" * 100)
            return str(out_path)

        mock_gdown = MagicMock()
        mock_gdown.download = mock_download

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage(max_file_size=10)
            with pytest.raises(DownloadError, match="exceeds the maximum allowed"):
                stage.execute(context)
        assert not (download_dir / "audio.mp3").exists()

    def test_gdrive_size_cap_message_uses_fractional_megabytes(self, context, download_dir):
        # Integer division understated both numbers (a sub-MB file read
        # "0 MB"); the message must carry one decimal place.
        def mock_download(url, output, quiet=True):
            out_path = Path(output) / "audio.mp3"
            out_path.write_bytes(b"x" * int(1.5 * 1024 * 1024))
            return str(out_path)

        mock_gdown = MagicMock()
        mock_gdown.download = mock_download

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage(max_file_size=1024 * 1024)
            with pytest.raises(DownloadError, match=r"1\.5 MB.*1\.0 MB"):
                stage.execute(context)

    def test_gdrive_file_within_size_cap_succeeds(self, context, download_dir):
        def mock_download(url, output, quiet=True):
            out_path = Path(output) / "audio.mp3"
            out_path.write_bytes(b"x" * 100)
            return str(out_path)

        mock_gdown = MagicMock()
        mock_gdown.download = mock_download

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage(max_file_size=1024)
            result = stage.execute(context)

        assert result["input_path"] == str(download_dir / "audio.mp3")

    def test_gdrive_size_cap_disabled_by_default(self, context, download_dir):
        def mock_download(url, output, quiet=True):
            out_path = Path(output) / "audio.mp3"
            out_path.write_bytes(b"x" * 100)
            return str(out_path)

        mock_gdown = MagicMock()
        mock_gdown.download = mock_download

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage()
            result = stage.execute(context)

        assert result["input_path"] == str(download_dir / "audio.mp3")

    def test_gdrive_unsupported_extension_raises(self, context, download_dir):
        def mock_download(url, output, quiet=True):
            out_path = Path(output) / "document.txt"
            out_path.write_bytes(b"some text content")
            return str(out_path)

        mock_gdown = MagicMock()
        mock_gdown.download = mock_download

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            stage = DownloadStage()
            with pytest.raises(DownloadError, match="is not a supported audio or video format"):
                stage.execute(context)
        assert not (download_dir / "document.txt").exists()


class TestTwitterDownload:
    @pytest.fixture
    def download_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "downloads"
        d.mkdir()
        return d

    @pytest.fixture
    def context(self, download_dir: Path) -> dict[str, Any]:
        return {
            "source_url": "https://x.com/i/status/2052048266687332852",
            "download_dir": str(download_dir),
            "input_path": "",
        }

    def _make_mock_ydl(self, download_dir: Path, duration: int = 120, title: str = "X Post"):
        mock_ydl_instance = MagicMock()

        def extract_info(url, download=True):
            info = {"duration": duration, "title": title}
            if download:
                (Path(download_dir) / "video.mp4").write_bytes(b"fake video")
            return info

        mock_ydl_instance.extract_info = extract_info
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        return mock_ydl_instance

    def test_successful_twitter_download(self, context, download_dir):
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = self._make_mock_ydl(download_dir)

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            result = DownloadStage().execute(context)

        assert result["input_path"] == str(download_dir / "video.mp4")
        assert result["video_title"] == "X Post"

    def test_restricts_yt_dlp_to_twitter_extractor(self, context, download_dir):
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = self._make_mock_ydl(download_dir)

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            DownloadStage().execute(context)

        ydl_opts = mock_module.YoutubeDL.call_args.args[0]
        assert ydl_opts["allowed_extractors"] == ["twitter"]

    def test_cookiefile_passed_when_file_exists(self, context, download_dir, tmp_path):
        cookies = tmp_path / "cookies.txt"
        cookies.write_text("# Netscape HTTP Cookie File\n")
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = self._make_mock_ydl(download_dir)

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            DownloadStage(twitter_cookies_file=str(cookies)).execute(context)

        ydl_opts = mock_module.YoutubeDL.call_args.args[0]
        assert ydl_opts["cookiefile"] == str(cookies)

    def test_cookiefile_omitted_when_unset(self, context, download_dir):
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = self._make_mock_ydl(download_dir)

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            DownloadStage().execute(context)

        ydl_opts = mock_module.YoutubeDL.call_args.args[0]
        assert "cookiefile" not in ydl_opts

    def test_cookiefile_omitted_when_file_missing(self, context, download_dir):
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = self._make_mock_ydl(download_dir)

        with patch.dict("sys.modules", {"yt_dlp": mock_module}):
            DownloadStage(twitter_cookies_file="/nonexistent/cookies.txt").execute(context)

        ydl_opts = mock_module.YoutubeDL.call_args.args[0]
        assert "cookiefile" not in ydl_opts

    def test_restricted_post_raises_actionable_error(self, context, download_dir):
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = Exception(
            "Sorry, you are not authorized to view this tweet. Log in."
        )
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with patch.dict("sys.modules", {"yt_dlp": mock_module}), pytest.raises(DownloadError, match="無法下載此貼文"):
            DownloadStage().execute(context)

    def test_broadcast_extractor_block_raises_actionable_error(self, context, download_dir):
        # A /status/ tweet whose video is an X Broadcast: the ["twitter"] pin
        # blocks the twitter:broadcast re-extraction. yt-dlp 2026.03.17 reports
        # this as "No suitable extractor (TwitterBroadcast) found".
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = Exception(
            "ERROR: No suitable extractor (TwitterBroadcast) found for URL https://twitter.com/i/broadcasts/1abc"
        )
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with patch.dict("sys.modules", {"yt_dlp": mock_module}), pytest.raises(DownloadError, match="無法下載此貼文"):
            DownloadStage().execute(context)

    def test_transient_guest_token_error_retries_then_succeeds(self, context, download_dir):
        # X intermittently rejects anonymous guest tokens. The first attempt
        # fails with "Bad guest token"; a fresh-client retry fetches a new token
        # and succeeds, so the user never sees the error.
        calls = {"n": 0}

        def extract_info(url, download=True):
            calls["n"] += 1
            if calls["n"] == 1:
                raise Exception(
                    "ERROR: [twitter] 123: Error(s) while querying API: Bad guest token; please report this issue"
                )
            info = {"duration": 120, "title": "X Post"}
            if download:
                (Path(download_dir) / "video.mp4").write_bytes(b"fake video")
            return info

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info = extract_info
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with (
            patch.dict("sys.modules", {"yt_dlp": mock_module}),
            patch("whisper_ui.pipeline.download.time.sleep"),
        ):
            result = DownloadStage().execute(context)

        assert result["input_path"] == str(download_dir / "video.mp4")
        # First attempt failed at the metadata probe, second attempt built a
        # fresh client and completed: two YoutubeDL instances in total.
        assert mock_module.YoutubeDL.call_count == 2

    def test_transient_error_exhausts_retries_then_raises_transient_message(self, context, download_dir):
        # A retryable HTTP 503 must NOT be reported as "restricted" (which would
        # wrongly tell the user to export cookies). It is retried with a fresh
        # client up to the cap, then surfaces the actionable transient hint.
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = Exception(
            "ERROR: Unable to download API page: HTTP Error 503: Service Unavailable"
        )
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with (
            patch.dict("sys.modules", {"yt_dlp": mock_module}),
            patch("whisper_ui.pipeline.download.time.sleep"),
            pytest.raises(DownloadError, match="來源暫時無法回應"),
        ):
            DownloadStage().execute(context)

        assert mock_module.YoutubeDL.call_count == _MAX_DOWNLOAD_ATTEMPTS

    def test_restricted_error_fails_fast_without_retry(self, context, download_dir):
        # A login wall is permanent: it must raise on the first attempt, never
        # burning retry budget on a request that cannot succeed.
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = Exception(
            "Sorry, you are not authorized to view this tweet. Log in."
        )
        mock_ydl_instance.__enter__ = lambda self: self
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = mock_ydl_instance

        with (
            patch.dict("sys.modules", {"yt_dlp": mock_module}),
            patch("whisper_ui.pipeline.download.time.sleep") as mock_sleep,
            pytest.raises(DownloadError, match="無法下載此貼文"),
        ):
            DownloadStage().execute(context)

        assert mock_module.YoutubeDL.call_count == 1
        mock_sleep.assert_not_called()

    def test_duration_exceeds_limit(self, context, download_dir):
        mock_module = MagicMock()
        mock_module.YoutubeDL.return_value = self._make_mock_ydl(download_dir, duration=50000)

        with (
            patch.dict("sys.modules", {"yt_dlp": mock_module}),
            pytest.raises(DownloadError, match="exceeds the maximum"),
        ):
            DownloadStage(max_duration=3600).execute(context)
