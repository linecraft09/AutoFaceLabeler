"""Helpers for selecting yt-dlp cookie files by video platform."""

from typing import Any, Dict, Optional
from urllib.parse import urlparse


def detect_platform_from_url(url: str) -> Optional[str]:
    """Infer the supported platform from a video URL."""
    host = urlparse(url).netloc.lower()
    if "youtube.com" in host or "youtu.be" in host:
        return "youtube"
    if "bilibili.com" in host or "b23.tv" in host:
        return "bilibili"
    return None


def resolve_platform_cookies(config: Dict[str, Any], platform: Optional[str]) -> Optional[str]:
    """
    Resolve a cookie file for a platform-aware config.

    Preferred shape:
      platforms:
        youtube:
          cookies: config/youtube_cookies.txt

    Legacy fallback:
      cookies: config/cookies.txt
    """
    if platform:
        platforms = config.get("platforms")
        if isinstance(platforms, dict):
            platform_config = platforms.get(platform.lower())
            if isinstance(platform_config, dict) and platform_config.get("cookies"):
                return platform_config["cookies"]

    return config.get("cookies") or None
