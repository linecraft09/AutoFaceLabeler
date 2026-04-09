# src/explorer/adaptive_scheduler.py
import os
from typing import List, Dict, Any

from aflutils.logger import get_logger
from .llm_optimizer import LLMOptimizer
from .search_term_pool import SearchTermPool, SearchTerm

logger = get_logger(__name__)


class AdaptiveScheduler:
    """
    Explorer (E) 的核心逻辑：生成搜索批次、处理反馈、触发优化。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        :param config: 包含 explorer 段的完整配置字典
        """
        explorer_cfg = config
        self.pool = SearchTermPool(
            initial_terms=explorer_cfg.get('initial_search_terms', []),
            target_distribution=explorer_cfg.get('sampling', {}).get('target_distribution', {}),
            min_weight=explorer_cfg.get('sampling', {}).get('min_weight', 0.1),
            weight_decay_factor=explorer_cfg.get('sampling', {}).get('weight_decay_factor', 0.9)
        )
        # LLM 优化器
        llm_cfg = explorer_cfg.get('llm', {})
        if llm_cfg.get('enabled', False):
            self.llm = LLMOptimizer(
                model=llm_cfg.get('model', 'gpt-4o-mini'),
                api_key_env=llm_cfg.get('api_key_env', 'OPENAI_API_KEY'),
                base_url=llm_cfg.get('base_url'),
                temperature=llm_cfg.get('temperature', 0.7)
            )
        else:
            self.llm = None
        self.v2_optimization_threshold = llm_cfg.get('optimization_trigger_v2_pass_rate', 0.05)
        self.v1_optimization_threshold = llm_cfg.get('optimization_trigger_v1_pass_rate', 0.1)

        # 全局目标合格数量
        self.target_qualified = config.get('orchestrator', {}).get('target_qualified', 200)
        self.json_file = config.get('json_file', "D:/WorkDir/AutoFaceLabeler/data/explorer.json")
        self.load_state(self.json_file)

    def generate_batch(self, batch_size: int = 20) -> List[SearchTerm]:
        """
        生成下一批搜索词。
        :return: SearchTerm 列表
        """
        # 分层采样
        batch = self.pool.sample(batch_size)
        logger.info(f"Generated batch of {len(batch)} search terms")
        return batch

    def receive_feedback(self, feedback: Dict[str, Any]):
        """
        接收 V1 或 V2 的反馈并更新池子。
        :param feedback: 包含 'search_term', 'v1_pass_rate' 或 'v2_pass_rate', 'fail_reasons' 等
        """
        term_text = feedback.get('search_term')
        if not term_text:
            logger.warning("Feedback missing search_term")
            return

        # 更新统计
        self.pool.update_stats(
            term_text,
            v1_pass_rate=feedback.get('v1_pass_rate'),
            v2_pass_rate=feedback.get('v2_pass_rate'),
            failure_reasons=feedback.get('fail_reasons'),
            total_tried=feedback.get('total_received'),
            total_downloaded=feedback.get('total_downloaded'),
            total_qualified=feedback.get('total_qualified')
        )

        # 简单反馈：根据通过率调整权重（可选）
        if 'v1_pass_rate' in feedback:
            rate = feedback['v1_pass_rate']
            # 通过率越高权重增加，越低减少
            delta = (rate - 0.3) * 0.5  # 以0.3为基准
            self.pool.update_weights(term_text, delta)
        elif 'v2_pass_rate' in feedback:
            rate = feedback['v2_pass_rate']
            delta = (rate - 0.05) * 1.0  # 以0.05为基准
            self.pool.update_weights(term_text, delta)

        logger.info(f"Updated stats for '{term_text}'")

    def adapt_strategy(self):
        """
        全局自适应策略：检查不足的类别、表现不佳的搜索词，触发 LLM 优化。
        应定期调用（例如每 N 个 batch 后）。
        """
        # 1. 检查各类别是否达到目标数量
        category_counts = self.pool.get_category_counts()
        for cat, target_ratio in self.pool.target_dist.items():
            target_count = int(self.target_qualified * target_ratio)
            current = category_counts.get(cat, 0)
            if current < target_count and self.llm:
                logger.info(f"Category {cat} is under target ({current}/{target_count}), generating new terms")
                new_terms = self.llm.generate_new_terms_for_category(
                    category=cat,
                    target_qualities=["single person", "front face", "clear audio"],
                    num_terms=3
                )
                for term in new_terms:
                    self.pool.add_term(
                        text=term,
                        category=cat,
                        weight=0.5,  # 初始权重为中等
                        original_text=None,
                        generation_round=1
                    )

        # 2. 处理表现不佳的搜索词
        if self.llm:
            poor_terms = self.pool.get_underperforming_terms(
                v2_pass_threshold=self.v2_optimization_threshold,
                v1_pass_threshold=self.v1_optimization_threshold
            )
            for term in poor_terms:
                # 避免过于频繁优化同一个词（例如每5轮一次）
                if term.generation_round > 3:
                    continue
                logger.info(f"Optimizing poor term: {term.text} (v2_pass={term.stats.v2_pass_rate})")
                variants = self.llm.generate_variants(
                    original_term=term.text,
                    failure_reasons=term.stats.failure_reasons,
                    category=term.category,
                    num_variants=3
                )
                for var in variants:
                    self.pool.add_term(
                        text=var,
                        category=term.category,
                        weight=term.weight * 0.6,  # 新词继承部分权重
                        original_text=term.text,
                        generation_round=term.generation_round + 1
                    )
                # 降低原词权重，避免继续大量使用
                self.pool.set_weight(term.text, term.weight * 0.3)

    def should_stop(self) -> bool:
        """检查是否已达到目标数量"""
        total_qualified = sum(self.pool.get_category_counts().values())
        return total_qualified >= self.target_qualified

    def get_status(self) -> Dict[str, Any]:
        """返回当前状态，用于监控"""
        return {
            "total_qualified": sum(self.pool.get_category_counts().values()),
            "category_counts": self.pool.get_category_counts(),
            "total_terms": len(self.pool.terms),
            "target_distribution": self.pool.target_dist,
        }

    def save_state(self):
        """保存池子状态到 JSON"""
        import json
        with open(self.json_file, 'w') as f:
            json.dump(self.pool.to_dict(), f, indent=2)
        logger.info(f"Saved explorer state to {self.json_file}")

    def load_state(self, filepath: str):
        """从 JSON 恢复状态"""
        import json
        if not filepath:
            logger.warning("Explorer state filepath is empty, using fresh pool state")
            return
        if not os.path.exists(filepath):
            logger.warning(f"Explorer state file not found: {filepath}, using fresh pool state")
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.pool = SearchTermPool.from_dict(data)
            logger.info(f"Loaded explorer state from {filepath}")
        except (json.JSONDecodeError, OSError, ValueError, KeyError, TypeError) as e:
            logger.warning(
                f"Failed to load explorer state from {filepath}, using fresh pool state: {e}"
            )
