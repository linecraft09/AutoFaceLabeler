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
from downloaders.base_downloader import DefaultDownloader
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


def test_search_api_applies_platform_specific_cookiefile():
    youtube_search = YtDlpSearchApi(platform="youtube", search_config=PLATFORM_COOKIE_CONFIG)
    bilibili_search = YtDlpSearchApi(platform="bilibili", search_config=PLATFORM_COOKIE_CONFIG)

    assert youtube_search.ydl_opts_fast["cookiefile"] == "config/youtube_cookies.txt"
    assert youtube_search.ydl_opts_detail["cookiefile"] == "config/youtube_cookies.txt"
    assert bilibili_search.ydl_opts_fast["cookiefile"] == "config/bilibili_cookies.txt"
    assert bilibili_search.ydl_opts_detail["cookiefile"] == "config/bilibili_cookies.txt"
