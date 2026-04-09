# src/searcher/base.py
from abc import ABC, abstractmethod
from typing import List

from core.models.video_meta import VideoMeta


class SearchApi(ABC):
    """搜索API基类"""

    @abstractmethod
    def search(self, query: str, max_results: int = 50) -> List[VideoMeta]:
        """
        根据查询词搜索视频，返回 VideoMeta 列表
        """
        pass

    @abstractmethod
    def get_platform(self) -> str:
        """返回平台标识: 'youtube' 或 'bilibili'"""
        pass
