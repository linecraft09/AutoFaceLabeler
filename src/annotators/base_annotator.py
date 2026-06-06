from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAnnotator(ABC):
    """Abstract base class for all Milestone 2 annotators."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cpu")
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model resources."""
        raise NotImplementedError

    @abstractmethod
    def annotate(self, video_path: str, **kwargs: Any) -> Dict[str, Any]:
        """Run annotation for a single video."""
        raise NotImplementedError

    @property
    @abstractmethod
    def label_name(self) -> str:
        """Stable label identifier for this annotator."""
        raise NotImplementedError

    def unload(self) -> None:
        self._loaded = False
