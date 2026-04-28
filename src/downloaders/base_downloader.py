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
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import yaml
import yt_dlp

from aflutils.logger import get_logger

logger = get_logger(__name__)


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

        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                self.logger.info(f"Starting download: {url}")
                info = ydl.extract_info(url, download=True)

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
            self.on_download_error(url, e)
            return None

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
        return {}

    def on_download_complete(self, url: str, info: Dict[str, Any]) -> None:
        self.logger.debug(f"Download metadata processed for URL: {url}")
