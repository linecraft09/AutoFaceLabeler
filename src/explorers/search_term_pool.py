# src/explorers/search_term_pool.py
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import numpy as np

from aflutils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchTermStats:
    """搜索词的统计信息"""
    v1_pass_rate: float = 0.0  # 最近一批预筛选通过率
    v2_pass_rate: float = 0.0  # 最终合格率
    total_tried: int = 0  # 累计搜索返回总数
    total_downloaded: int = 0  # 累计下载数
    total_qualified: int = 0  # 累计合格数
    failure_reasons: Dict[str, int] = field(default_factory=dict)  # 失败原因计数


@dataclass
class SearchTerm:
    """单个搜索词实体"""
    text: str
    category: str  # "interview", "educational", "vlog"...
    weight: float = 1.0  # 采样权重
    stats: SearchTermStats = field(default_factory=SearchTermStats)
    # 可选元信息
    original_text: Optional[str] = None  # 如果是 LLM 生成的，记录原始词
    generation_round: int = 0  # 第几轮优化生成


class SearchTermPool:
    """
    管理所有搜索词，支持分层采样、权重更新、统计跟踪。
    """

    def __init__(self, initial_terms: List[Dict[str, Any]], target_distribution: Dict[str, float],
                 min_weight: float = 0.1, weight_decay_factor: float = 0.9):
        """
        :param initial_terms: 初始搜索词列表，每个元素包含 {text, platform, category, weight?}
        :param target_distribution: 各分类的目标占比，如 {"interview":0.25, "educational":0.25}
        :param min_weight: 最小权重，低于此值视为可丢弃或不再采样
        :param weight_decay_factor: 未使用的词每次反馈后权重衰减因子
        """
        self.target_dist = target_distribution
        self.min_weight = min_weight
        self.weight_decay_factor = weight_decay_factor
        self._lock = threading.Lock()

        self.terms: List[SearchTerm] = []
        for t in initial_terms:
            term = SearchTerm(
                text=t['text'],
                category=t.get('category', 'unknown'),
                weight=t.get('weight', 1.0)
            )
            self.terms.append(term)

        self._normalize_weights()
        logger.info(f"Initialized pool with {len(self.terms)} terms")

    def _normalize_weights(self):
        """按类别分别归一化权重，保持分层比例"""
        # 按类别分组
        by_category = {}
        for term in self.terms:
            by_category.setdefault(term.category, []).append(term)
        for cat, terms in by_category.items():
            total = sum(t.weight for t in terms)
            if total > 0:
                for t in terms:
                    t.weight /= total

    def update_weights(self, term_text: str, delta: float):
        """调整某个词的权重（增量）"""
        with self._lock:
            term = self._find_term(term_text)
            if term:
                term.weight += delta
                if term.weight < self.min_weight:
                    term.weight = self.min_weight
                # 重新归一化该类别的权重
                self._normalize_weights()

    def set_weight(self, term_text: str, new_weight: float):
        with self._lock:
            term = self._find_term(term_text)
            if term:
                term.weight = max(self.min_weight, new_weight)
                self._normalize_weights()

    def _find_term(self, text: str) -> Optional[SearchTerm]:
        for t in self.terms:
            if t.text == text:
                return t
        return None

    def update_stats(self, term_text: str, v1_pass_rate: float = None,
                     v2_pass_rate: float = None, failure_reasons: Dict[str, int] = None,
                     total_tried: int = None, total_downloaded: int = None,
                     total_qualified: int = None):
        """更新搜索词的统计信息（增量或覆盖）"""
        with self._lock:
            term = self._find_term(term_text)
            if not term:
                logger.warning(f"Term '{term_text}' not found in pool")
                return

            if v1_pass_rate is not None:
                term.stats.v1_pass_rate = v1_pass_rate
            if v2_pass_rate is not None:
                term.stats.v2_pass_rate = v2_pass_rate
            if total_tried is not None:
                term.stats.total_tried += total_tried
            if total_downloaded is not None:
                term.stats.total_downloaded += total_downloaded
            if total_qualified is not None:
                term.stats.total_qualified += total_qualified
            if failure_reasons:
                for k, v in failure_reasons.items():
                    term.stats.failure_reasons[k] = term.stats.failure_reasons.get(k, 0) + v

    def _sample_category_unlocked(self, batch_size: int, category: str) -> List[SearchTerm]:
        terms_in_cat = [t for t in self.terms if t.category == category]
        if not terms_in_cat:
            return []
        weights = [t.weight for t in terms_in_cat]
        total_weight = sum(weights)
        if total_weight == 0:
            return []
        probs = [w / total_weight for w in weights]
        indices = np.random.choice(len(terms_in_cat), size=batch_size, p=probs, replace=True)
        return [terms_in_cat[i] for i in indices]

    def sample(self, batch_size: int, category: Optional[str] = None) -> List[SearchTerm]:
        """
        按分层采样返回一批搜索词。
        :param batch_size: 本次采样总数
        :param category: 若指定，只从该类别采样；否则按目标分布从各层采样
        :return: SearchTerm 列表（可重复）
        """
        with self._lock:
            if category:
                # 单一类别采样
                return self._sample_category_unlocked(batch_size, category)

            # 分层采样：按目标分布决定每个类别采多少
            samples = []
            # 确保所有类别都在 target_dist 中，否则使用均匀分布
            categories = set(t.category for t in self.terms)
            if not categories:
                return []
            target = self.target_dist.copy()
            for cat in categories:
                if cat not in target:
                    target[cat] = 1.0 / len(categories)
            # 归一化
            total_target = sum(target.values())
            for cat in target:
                target[cat] /= total_target

            # 计算每个类别应该采样的数量
            counts = {cat: int(round(batch_size * target[cat])) for cat in categories}
            # 调整浮点误差
            diff = batch_size - sum(counts.values())
            if diff > 0:
                # 把多余的分配给最大目标比例的类别
                max_cat = max(target.items(), key=lambda x: x[1])[0]
                counts[max_cat] += diff
            elif diff < 0:
                # 减少最小目标比例的类别
                min_cat = min(target.items(), key=lambda x: x[1])[0]
                counts[min_cat] += diff  # diff 负数

            for cat, count in counts.items():
                if count <= 0:
                    continue
                cat_samples = self._sample_category_unlocked(count, cat)
                samples.extend(cat_samples)
            return samples

    def add_term(self, text: str, *args, **kwargs):
        """添加新的搜索词（如 LLM 生成的）"""
        # Backward-compatible signatures:
        # 1) add_term(text, category, weight=..., original_text=..., generation_round=...)
        # 2) add_term(text, platform, category, weight=..., original_text=..., generation_round=...)
        if len(args) >= 2:
            category = args[1]
            weight = kwargs.pop("weight", 1.0)
        elif len(args) == 1:
            category = args[0]
            weight = kwargs.pop("weight", 1.0)
        else:
            category = kwargs.pop("category")
            weight = kwargs.pop("weight", 1.0)

        original_text = kwargs.pop("original_text", None)
        generation_round = kwargs.pop("generation_round", 0)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {', '.join(kwargs.keys())}")

        with self._lock:
            if self._find_term(text):
                logger.warning(f"Term '{text}' already exists, skipping")
                return
            term = SearchTerm(
                text=text,
                category=category,
                weight=weight,
                original_text=original_text,
                generation_round=generation_round
            )
            self.terms.append(term)
            self._normalize_weights()
            logger.info(f"Added new term: {text} (category={category})")

    def get_underperforming_terms(self, v2_pass_threshold: float = 0.05,
                                  v1_pass_threshold: float = 0.1) -> List[SearchTerm]:
        """
        获取表现不佳的搜索词（用于 LLM 优化）。
        :param v2_pass_threshold: 最终合格率低于此值视为差
        :param v1_pass_threshold: 预筛选通过率低于此值视为差
        """
        with self._lock:
            result = []
            for term in self.terms:
                if term.stats.total_tried < 10:  # 样本不足，暂不判断
                    continue
                if term.stats.v2_pass_rate < v2_pass_threshold or \
                        (term.stats.v1_pass_rate < v1_pass_threshold and term.stats.total_tried > 20):
                    result.append(term)
            return result

    def get_category_counts(self) -> Dict[str, int]:
        """返回每个类别当前累计合格数量"""
        with self._lock:
            counts = {}
            for term in self.terms:
                counts[term.category] = counts.get(term.category, 0) + term.stats.total_qualified
            return counts

    def to_dict(self) -> dict:
        """序列化，用于保存状态"""
        with self._lock:
            return {
                "terms": [
                    {
                        "text": t.text,
                        "category": t.category,
                        "weight": t.weight,
                        "stats": {
                            "v1_pass_rate": t.stats.v1_pass_rate,
                            "v2_pass_rate": t.stats.v2_pass_rate,
                            "total_tried": t.stats.total_tried,
                            "total_downloaded": t.stats.total_downloaded,
                            "total_qualified": t.stats.total_qualified,
                            "failure_reasons": t.stats.failure_reasons
                        },
                        "original_text": t.original_text,
                        "generation_round": t.generation_round
                    }
                    for t in self.terms
                ],
                "target_dist": self.target_dist,
                "min_weight": self.min_weight,
                "weight_decay_factor": self.weight_decay_factor
            }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchTermPool":
        pool = cls(
            initial_terms=[],
            target_distribution=data["target_dist"],
            min_weight=data["min_weight"],
            weight_decay_factor=data["weight_decay_factor"]
        )
        pool.terms = []
        for t_data in data["terms"]:
            stats = SearchTermStats(
                v1_pass_rate=t_data["stats"]["v1_pass_rate"],
                v2_pass_rate=t_data["stats"]["v2_pass_rate"],
                total_tried=t_data["stats"]["total_tried"],
                total_downloaded=t_data["stats"]["total_downloaded"],
                total_qualified=t_data["stats"]["total_qualified"],
                failure_reasons=t_data["stats"]["failure_reasons"]
            )
            term = SearchTerm(
                text=t_data["text"],
                category=t_data["category"],
                weight=t_data["weight"],
                stats=stats,
                original_text=t_data.get("original_text"),
                generation_round=t_data.get("generation_round", 0)
            )
            pool.terms.append(term)
        pool._normalize_weights()
        return pool
