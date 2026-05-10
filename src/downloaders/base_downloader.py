"""
Base collector for yt-dlp based video downloading.

This module provides an abstract base class that handles:
- Loading yt-dlp configuration from YAML file or dict
- Loading URL lists from file or list
- Downloading videos using yt-dlp with configurable options
- Error handling and logging

Subclasses must implement:
- get_options_for_url(url): Return a yt-dlp options dict for the given URL.
- on_download_complete(url, info): Process the downloaded video info.
"""

import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import yaml
import yt_dlp

from aflutils.logger import get_logger
from aflutils.platform_cookies import detect_platform_from_url, resolve_platform_cookies

logger = get_logger(__name__)


def _retry_policy(config: Dict[str, Any]) -> Dict[str, float]:
    backoff_cfg = config.get("backoff", {}) if isinstance(config.get("backoff"), dict) else {}
    return {
        "max_attempts": max(1, int(config.get("max_attempts", config.get("download_attempts", 1)))),
        "initial_seconds": float(config.get("backoff_initial_seconds", backoff_cfg.get("initial_seconds", 1))),
        "max_seconds": float(config.get("backoff_max_seconds", backoff_cfg.get("max_seconds", 30))),
        "multiplier": max(1.0, float(config.get("backoff_multiplier", backoff_cfg.get("multiplier", 2)))),
        "jitter_seconds": max(0.0, float(config.get("backoff_jitter_seconds", backoff_cfg.get("jitter_seconds", 0)))),
    }


def _backoff_delay(policy: Dict[str, float], attempt_index: int) -> float:
    delay = policy["initial_seconds"] * (policy["multiplier"] ** attempt_index)
    delay = min(policy["max_seconds"], delay)
    if policy["jitter_seconds"]:
        delay += random.uniform(0, policy["jitter_seconds"])
    return delay


def _strip_internal_retry_options(options: Dict[str, Any]) -> Dict[str, Any]:
    ydl_options = options.copy()
    for key in (
            "max_attempts",
            "download_attempts",
            "backoff",
            "backoff_initial_seconds",
            "backoff_max_seconds",
            "backoff_multiplier",
            "backoff_jitter_seconds",
    ):
        ydl_options.pop(key, None)
    return ydl_options


class BaseDownloader(ABC):
    """Abstract base class for video downloader using yt-dlp."""

    def __init__(
            self,
            config_path: Optional[str] = None,
            config_dict: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the downloader.

        Args:
            config_path: Path to YAML configuration file.
            config_dict: Configuration dictionary (overrides config_path if both given).
        """
        self.config_path = config_path
        self.config_dict = config_dict

        self.logger = logger or self._create_default_logger()
        self.config: Dict[str, Any] = {}
        self.urls: List[str] = []

        self.load_config()

    def _create_default_logger(self) -> logging.Logger:
        """Create a default console logger."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_config(self) -> None:
        """
        Load yt-dlp configuration from YAML file or dict.
        """
        if self.config_dict is not None:
            self.config = self.config_dict.copy()
            self.logger.info("Configuration loaded from provided dictionary.")
        elif self.config_path is not None:
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
                self.logger.info(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load config file {self.config_path}: {e}")
                raise
        else:
            self.logger.warning("No configuration provided. Using empty config.")
            self.config = {}

        # If config doesn't specify a proxy, try environment variables
        if "proxy" not in self.config or not self.config.get("proxy"):
            proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
            if proxy:
                self.config["proxy"] = proxy
                self.logger.info(f"Proxy loaded from environment: {proxy}")

        # Map legacy top-level 'cookies' config key to yt-dlp's 'cookiefile' option.
        # Platform-specific cookies are applied per URL in DefaultDownloader.
        if self.config.get("cookies"):
            self.config["cookiefile"] = self.config["cookies"]
            self.logger.info(f"Cookies loaded from: {self.config['cookies']}")

        # Add remote_components for yt-dlp remote extraction (JS challenge etc.)
        self.config.setdefault('remote_components', ['ejs:github'])

        # Convert list back to tuple for 'cookiesfrombrowser' if present
        if "cookiesfrombrowser" in self.config and isinstance(
                self.config["cookiesfrombrowser"], list
        ):
            self.config["cookiesfrombrowser"] = tuple(self.config["cookiesfrombrowser"])

    def on_download_error(self, url: str, error: Exception) -> None:
        """
        Handle download errors. Can be overridden by subclasses.

        Args:
            url: The URL that caused the error.
            error: The exception raised.
        """
        self.logger.error(f"Failed to download {url}: {error}")

    @abstractmethod
    def get_options_for_url(self, url: str) -> Dict[str, Any]:
        """
        Return yt-dlp options for a specific URL.
        """
        raise NotImplementedError

    @abstractmethod
    def on_download_complete(self, url: str, info: Dict[str, Any]) -> None:
        """
        Hook called after a successful yt-dlp extraction.
        """
        raise NotImplementedError

    def _download_single(self, url: str) -> Optional[str]:
        """
        Download a single video using yt-dlp.

        Args:
            url: The video URL.

        Returns:
            The local file path if download succeeded, None otherwise.
        """
        # Merge base config with URL‑specific options
        options = self.config.copy()
        options.update(self.get_options_for_url(url) or {})

        policy = _retry_policy(options)
        max_attempts = int(policy["max_attempts"])
        ydl_options = _strip_internal_retry_options(options)

        for attempt in range(max_attempts):
            try:
                with yt_dlp.YoutubeDL(ydl_options) as ydl:
                    self.logger.info(f"Starting download: {url}")
                    info = ydl.extract_info(url, download=True)
                    if not info:
                        self.logger.warning(
                            "Download produced no metadata for %s; yt-dlp likely ignored an error "
                            "(ignoreerrors=%r)",
                            url,
                            ydl_options.get("ignoreerrors"),
                        )
                        return None

                    # Determine the downloaded file path(s)
                    file_path = None
                    if "entries" in info:
                        # Playlist case – return the path of the first successfully downloaded video
                        for entry in info["entries"]:
                            if entry:
                                # Get file path for the first entry
                                file_path = self._get_downloaded_file_path(ydl, entry)
                                if file_path:
                                    break
                    else:
                        file_path = self._get_downloaded_file_path(ydl, info)

                    self.on_download_complete(url, info)

                    if file_path and os.path.exists(file_path):
                        self.logger.info(f"Download completed: {file_path}")
                        return file_path
                    else:
                        self.logger.warning(f"Download succeeded but file not found: {url}")
                        return None

            except Exception as e:
                kind = self._classify_download_error(e)
                if kind in {"rate_limit", "network", "anti_bot"} and attempt < max_attempts - 1:
                    delay = _backoff_delay(policy, attempt)
                    self.logger.warning(
                        "Retryable download failure for %s (kind=%s, attempt %s/%s), "
                        "backing off %.1fs: %s",
                        url,
                        kind,
                        attempt + 1,
                        max_attempts,
                        delay,
                        e,
                    )
                    time.sleep(delay)
                    continue
                self.logger.debug("Download traceback for %s", url, exc_info=True)
                self.on_download_error(url, e)
                return None
        return None

    @staticmethod
    def _classify_download_error(exc: Exception) -> str:
        msg = str(exc).lower()
        if any(k in msg for k in (
                "sign in to confirm", "not a bot", "precondition failed", "http error 412",
                "captcha", "verify you are human"
        )):
            return "anti_bot"
        if any(k in msg for k in ("429", "too many requests", "rate limit", "ratelimit")):
            return "rate_limit"
        if any(k in msg for k in (
                "timed out", "timeout", "connection", "network", "name resolution", "temporary failure",
                "connection reset", "proxy error", "unreachable", "ssl", "eof"
        )):
            return "network"
        return "other"

    def _get_downloaded_file_path(self, ydl: yt_dlp.YoutubeDL, info: dict) -> Optional[str]:
        """
        Helper to extract the actual downloaded file path from yt-dlp info dict.
        """
        # Method 1: use 'requested_downloads' (most reliable for yt-dlp >= 2023)
        if 'requested_downloads' in info and info['requested_downloads']:
            return info['requested_downloads'][0].get('filepath')

        # Method 2: use prepare_filename and check existence
        filename = ydl.prepare_filename(info)
        if os.path.exists(filename):
            return filename

        # Method 3: try with common extensions (if extension was omitted)
        base, _ = os.path.splitext(filename)
        for ext in ['.mp4', '.mkv', '.webm', '.flv']:
            candidate = base + ext
            if os.path.exists(candidate):
                return candidate

        return None

    def download(self, urls: List[str]) -> Dict[str, Optional[str]]:
        """
        Start the collection process: download all URLs.

        Returns:
            A dictionary mapping each URL to a boolean success flag.
        """
        results = {}
        for url in urls:
            filepath = self._download_single(url)
            results[url] = filepath
        self.logger.info(
            f"Collection finished. Success: {sum([1 if r else 0 for r in results.values()])} / {len(results)}"
        )
        return results


class DefaultDownloader(BaseDownloader):
    """Default downloader with no per-URL option overrides."""

    def get_options_for_url(self, url: str) -> Dict[str, Any]:
        platform = detect_platform_from_url(url)
        cookies = resolve_platform_cookies(self.config, platform)
        if cookies:
            return {"cookiefile": cookies}
        return {}

    def on_download_complete(self, url: str, info: Dict[str, Any]) -> None:
        self.logger.debug(f"Download metadata processed for URL: {url}")
