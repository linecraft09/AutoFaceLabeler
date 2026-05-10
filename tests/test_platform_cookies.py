import importlib.util
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if importlib.util.find_spec("yt_dlp") is None:
    yt_dlp = types.ModuleType("yt_dlp")
    yt_dlp_utils = types.ModuleType("yt_dlp.utils")

    class DownloadError(Exception):
        pass

    class ExtractorError(Exception):
        pass

    class YoutubeDL:
        pass

    yt_dlp.YoutubeDL = YoutubeDL
    yt_dlp_utils.DownloadError = DownloadError
    yt_dlp_utils.ExtractorError = ExtractorError
    yt_dlp.utils = yt_dlp_utils
    sys.modules["yt_dlp"] = yt_dlp
    sys.modules["yt_dlp.utils"] = yt_dlp_utils

from aflutils.platform_cookies import detect_platform_from_url, resolve_platform_cookies
import downloaders.base_downloader as base_downloader
from downloaders.base_downloader import DefaultDownloader
import searchers.ytdlp_search_api as ytdlp_search_api
from searchers.ytdlp_search_api import YtDlpSearchApi


PLATFORM_COOKIE_CONFIG = {
    "platforms": {
        "youtube": {"cookies": "config/youtube_cookies.txt"},
        "bilibili": {"cookies": "config/bilibili_cookies.txt"},
    }
}


def test_resolve_platform_specific_cookies():
    assert resolve_platform_cookies(PLATFORM_COOKIE_CONFIG, "youtube") == "config/youtube_cookies.txt"
    assert resolve_platform_cookies(PLATFORM_COOKIE_CONFIG, "bilibili") == "config/bilibili_cookies.txt"
    assert resolve_platform_cookies(PLATFORM_COOKIE_CONFIG, "unknown") is None


def test_resolve_legacy_top_level_cookies_fallback():
    config = {"cookies": "config/legacy_cookies.txt"}
    assert resolve_platform_cookies(config, "youtube") == "config/legacy_cookies.txt"
    assert resolve_platform_cookies(config, None) == "config/legacy_cookies.txt"


def test_detect_platform_from_url():
    assert detect_platform_from_url("https://www.youtube.com/watch?v=abc") == "youtube"
    assert detect_platform_from_url("https://youtu.be/abc") == "youtube"
    assert detect_platform_from_url("https://www.bilibili.com/video/BV123") == "bilibili"
    assert detect_platform_from_url("https://b23.tv/abc") == "bilibili"
    assert detect_platform_from_url("https://example.com/video") is None


def test_default_downloader_applies_cookie_by_url(monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("HTTP_PROXY", raising=False)

    downloader = DefaultDownloader(config_dict=PLATFORM_COOKIE_CONFIG)

    youtube_options = downloader.get_options_for_url("https://www.youtube.com/watch?v=abc")
    bilibili_options = downloader.get_options_for_url("https://www.bilibili.com/video/BV123")
    unknown_options = downloader.get_options_for_url("https://example.com/video")

    assert youtube_options["cookiefile"] == "config/youtube_cookies.txt"
    assert bilibili_options["cookiefile"] == "config/bilibili_cookies.txt"
    assert "cookiefile" not in unknown_options


def test_search_api_keeps_cookiefile_out_of_search_options():
    youtube_search = YtDlpSearchApi(platform="youtube", search_config=PLATFORM_COOKIE_CONFIG)
    bilibili_search = YtDlpSearchApi(platform="bilibili", search_config=PLATFORM_COOKIE_CONFIG)

    assert youtube_search._cookies == "config/youtube_cookies.txt"
    assert bilibili_search._cookies == "config/bilibili_cookies.txt"
    assert "cookiefile" not in youtube_search.ydl_opts_fast
    assert "cookiefile" not in youtube_search.ydl_opts_detail
    assert "cookiefile" not in bilibili_search.ydl_opts_fast
    assert "cookiefile" not in bilibili_search.ydl_opts_detail


def test_downloader_handles_ignored_ytdlp_error_returning_none(monkeypatch):
    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download):
            return None

    monkeypatch.setattr(base_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    downloader = DefaultDownloader(config_dict={"ignoreerrors": True})

    assert downloader.download(["https://www.bilibili.com/video/BV123"]) == {
        "https://www.bilibili.com/video/BV123": None
    }


def test_search_fast_search_retries_anti_bot_with_backoff(monkeypatch):
    calls = {"count": 0}
    sleeps = []

    class FakeDownloadError(Exception):
        pass

    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, query, download):
            calls["count"] += 1
            if calls["count"] == 1:
                raise FakeDownloadError("HTTP Error 412: Precondition Failed")
            return {"entries": [{"id": "BV123", "url": "https://www.bilibili.com/video/BV123"}]}

    monkeypatch.setattr(ytdlp_search_api, "DownloadError", FakeDownloadError)
    monkeypatch.setattr(ytdlp_search_api.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(ytdlp_search_api.time, "sleep", lambda seconds: sleeps.append(seconds))

    searcher = YtDlpSearchApi(
        platform="bilibili",
        search_config={
            "fast_search": {
                "max_attempts": 2,
                "backoff": {
                    "initial_seconds": 2,
                    "max_seconds": 8,
                    "multiplier": 2,
                    "jitter_seconds": 0,
                },
            },
        },
    )

    entries = searcher._extract_fast_search("bilisearch1:test", "test")

    assert [entry["id"] for entry in entries] == ["BV123"]
    assert calls["count"] == 2
    assert sleeps == [2]


def test_search_fast_search_falls_back_to_cookies_on_anti_bot(monkeypatch):
    seen_cookiefiles = []

    class FakeDownloadError(Exception):
        pass

    class FakeYoutubeDL:
        def __init__(self, options):
            seen_cookiefiles.append(options.get("cookiefile"))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, query, download):
            if seen_cookiefiles[-1] is None:
                raise FakeDownloadError("HTTP Error 412: Precondition Failed")
            return {"entries": [{"id": "BV123", "url": "https://www.bilibili.com/video/BV123"}]}

    monkeypatch.setattr(ytdlp_search_api, "DownloadError", FakeDownloadError)
    monkeypatch.setattr(ytdlp_search_api.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    searcher = YtDlpSearchApi(
        platform="bilibili",
        search_config={
            **PLATFORM_COOKIE_CONFIG,
            "cookie_fallback_on_anti_bot": True,
            "fast_search": {"max_attempts": 2},
        },
    )

    entries = searcher._extract_fast_search("bilisearch1:test", "test")

    assert [entry["id"] for entry in entries] == ["BV123"]
    assert seen_cookiefiles == [None, "config/bilibili_cookies.txt"]


def test_search_detail_fetch_falls_back_to_cookies_on_anti_bot(monkeypatch):
    seen_cookiefiles = []

    class FakeDownloadError(Exception):
        pass

    class FakeYoutubeDL:
        def __init__(self, options):
            seen_cookiefiles.append(options.get("cookiefile"))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download):
            if seen_cookiefiles[-1] is None:
                raise FakeDownloadError("Sign in to confirm you're not a bot")
            return {"id": "abc", "title": "ok"}

    monkeypatch.setattr(ytdlp_search_api, "DownloadError", FakeDownloadError)
    monkeypatch.setattr(ytdlp_search_api.yt_dlp, "YoutubeDL", FakeYoutubeDL)

    searcher = YtDlpSearchApi(
        platform="youtube",
        search_config={
            **PLATFORM_COOKIE_CONFIG,
            "cookie_fallback_on_anti_bot": True,
            "detail_fetch": {"max_attempts": 2},
        },
    )

    assert searcher._get_video_details("https://www.youtube.com/watch?v=abc") == {"id": "abc", "title": "ok"}
    assert seen_cookiefiles == [None, "config/youtube_cookies.txt"]


def test_downloader_retries_anti_bot_and_strips_internal_options(monkeypatch, tmp_path):
    output = tmp_path / "video.mkv"
    output.write_text("video")
    calls = {"count": 0}
    seen_options = []
    sleeps = []

    class FakeYoutubeDL:
        def __init__(self, options):
            seen_options.append(options)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_info(self, url, download):
            calls["count"] += 1
            if calls["count"] == 1:
                raise Exception("Sign in to confirm you're not a bot")
            return {"requested_downloads": [{"filepath": str(output)}]}

    monkeypatch.setattr(base_downloader.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(base_downloader.time, "sleep", lambda seconds: sleeps.append(seconds))

    downloader = DefaultDownloader(
        config_dict={
            "max_attempts": 2,
            "backoff": {
                "initial_seconds": 3,
                "max_seconds": 9,
                "multiplier": 2,
                "jitter_seconds": 0,
            },
        }
    )

    assert downloader.download(["https://www.youtube.com/watch?v=abc"]) == {
        "https://www.youtube.com/watch?v=abc": str(output)
    }
    assert calls["count"] == 2
    assert sleeps == [3]
    assert all("max_attempts" not in options for options in seen_options)
    assert all("backoff" not in options for options in seen_options)
