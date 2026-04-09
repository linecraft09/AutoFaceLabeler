#!/usr/bin/env python3
"""
测试 Explorer (E) 模块：SearchTermPool, AdaptiveScheduler, 模拟 V1/V2 反馈。
"""

import sys
import os
import tempfile
import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# 确保能够导入 src 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.explorers.search_term_pool import SearchTermPool, SearchTerm, SearchTermStats
from src.explorers.adaptive_scheduler import AdaptiveScheduler
from src.explorers.llm_optimizer import LLMOptimizer


class TestSearchTermPool(unittest.TestCase):
    """测试搜索词池管理"""

    def setUp(self):
        self.initial_terms = [
            {"text": "tutorial", "platform": "youtube", "category": "educational", "weight": 1.0},
            {"text": "lecture", "platform": "youtube", "category": "educational", "weight": 1.0},
            {"text": "vlog", "platform": "youtube", "category": "vlog", "weight": 1.0},
            {"text": "人物专访", "platform": "bilibili", "category": "interview", "weight": 1.0},
            {"text": "播客", "platform": "bilibili", "category": "interview", "weight": 1.0}
        ]
        self.target_dist = {"educational": 0.5, "vlog": 0.3, "interview": 0.2}
        self.pool = SearchTermPool(
            initial_terms=self.initial_terms,
            target_distribution=self.target_dist,
            min_weight=0.1,
            weight_decay_factor=0.9
        )

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(len(self.pool.terms), 5)
        # 检查权重归一化（初始权重和为 1 每个类别）
        edu_terms = [t for t in self.pool.terms if t.category == "educational"]
        self.assertAlmostEqual(sum(t.weight for t in edu_terms), 1.0)

    def test_sample_distribution(self):
        """测试分层采样大致符合目标分布"""
        batch_size = 100
        batch = self.pool.sample(batch_size)
        # 统计各类别数量
        counts = {}
        for term in batch:
            counts[term.category] = counts.get(term.category, 0) + 1
        # 验证比例大致接近目标（允许浮动）
        for cat, target_ratio in self.target_dist.items():
            actual_ratio = counts.get(cat, 0) / batch_size
            self.assertAlmostEqual(actual_ratio, target_ratio, delta=0.1)

    def test_update_stats(self):
        """测试更新统计信息"""
        term_text = "tutorial"
        self.pool.update_stats(term_text, v1_pass_rate=0.3, total_tried=50, total_downloaded=20)
        term = self.pool._find_term(term_text)
        self.assertEqual(term.stats.v1_pass_rate, 0.3)
        self.assertEqual(term.stats.total_tried, 50)
        self.assertEqual(term.stats.total_downloaded, 20)

        # 再次更新（累计）
        self.pool.update_stats(term_text, total_tried=30, total_qualified=5)
        self.assertEqual(term.stats.total_tried, 80)
        self.assertEqual(term.stats.total_qualified, 5)

    def test_update_weights(self):
        """测试权重调整"""
        term = self.pool._find_term("tutorial")
        old_weight = term.weight
        self.pool.update_weights("tutorial", delta=0.2)
        self.assertGreater(term.weight, old_weight)
        # 验证归一化后该类别的权重和为 1
        edu_terms = [t for t in self.pool.terms if t.category == "educational"]
        self.assertAlmostEqual(sum(t.weight for t in edu_terms), 1.0)

    def test_get_underperforming_terms(self):
        """测试获取表现不佳的搜索词"""
        # 设置一些低通过率的数据
        self.pool.update_stats("tutorial", v1_pass_rate=1, v2_pass_rate=0.02, total_tried=100)
        self.pool.update_stats("vlog", v1_pass_rate=1, v2_pass_rate=0.2, total_tried=100)
        poor = self.pool.get_underperforming_terms(v2_pass_threshold=0.05)
        poor_texts = [t.text for t in poor]
        self.assertIn("tutorial", poor_texts)
        self.assertNotIn("vlog", poor_texts)

    def test_add_term(self):
        """测试添加新搜索词"""
        self.pool.add_term("new term", "youtube", "educational", weight=0.5)
        new_term = self.pool._find_term("new term")
        self.assertIsNotNone(new_term)
        self.assertEqual(new_term.category, "educational")
        # 验证权重归一化
        edu_terms = [t for t in self.pool.terms if t.category == "educational"]
        self.assertAlmostEqual(sum(t.weight for t in edu_terms), 1.0)

    def test_serialization(self):
        """测试序列化与反序列化"""
        # 修改一些状态
        self.pool.update_stats("tutorial", v1_pass_rate=0.4, total_tried=100, total_qualified=10)
        data = self.pool.to_dict()
        new_pool = SearchTermPool.from_dict(data)
        self.assertEqual(len(new_pool.terms), len(self.pool.terms))
        new_term = new_pool._find_term("tutorial")
        self.assertEqual(new_term.stats.v1_pass_rate, 0.4)
        self.assertEqual(new_term.stats.total_qualified, 10)


class TestAdaptiveScheduler(unittest.TestCase):
    """测试 AdaptiveScheduler（主控逻辑）"""

    def setUp(self):
        # 配置（禁用 LLM）
        self.config = {
            "explorer": {
                "initial_search_terms": [
                    {"text": "tutorial", "platform": "youtube", "category": "educational", "weight": 1.0},
                    {"text": "vlog", "platform": "youtube", "category": "vlog", "weight": 1.0},
                    {"text": "人物专访", "platform": "bilibili", "category": "interview", "weight": 1.0},
                ],
                "sampling": {
                    "target_distribution": {"educational": 0.5, "vlog": 0.3, "interview": 0.2},
                    "min_weight": 0.1,
                    "weight_decay_factor": 0.9
                },
                "llm": {
                    "enabled": False   # 禁用 LLM，避免真实调用
                }
            },
            "orchestrator": {
                "target_qualified": 200
            }
        }
        self.scheduler = AdaptiveScheduler(self.config)

    def test_generate_batch(self):
        """测试生成批次"""
        batch = self.scheduler.generate_batch(batch_size=10)
        self.assertEqual(len(batch), 10)
        # 检查每个元素都是 SearchTerm 实例
        for term in batch:
            self.assertIsInstance(term, SearchTerm)

    def test_receive_feedback_v1(self):
        """测试接收 V1 反馈并更新统计"""
        feedback = {
            "search_term": "tutorial",
            "v1_pass_rate": 0.3,
            "total_received": 50,
            "fail_reasons": {"duration": 10, "resolution": 5}
        }
        self.scheduler.receive_feedback(feedback)
        term = self.scheduler.pool._find_term("tutorial")
        self.assertEqual(term.stats.v1_pass_rate, 0.3)
        self.assertEqual(term.stats.total_tried, 50)
        self.assertEqual(term.stats.failure_reasons.get("duration"), 10)

    def test_receive_feedback_v2(self):
        """测试接收 V2 反馈"""
        feedback = {
            "search_term": "tutorial",
            "v2_pass_rate": 0.1,
            "total_qualified": 5
        }
        self.scheduler.receive_feedback(feedback)
        term = self.scheduler.pool._find_term("tutorial")
        self.assertEqual(term.stats.v2_pass_rate, 0.1)
        self.assertEqual(term.stats.total_qualified, 5)

    @patch('src.explorers.adaptive_scheduler.LLMOptimizer')
    def test_adapt_strategy_with_llm(self, mock_llm_class):
        """测试自适应策略（LLM 启用）模拟生成新词"""
        # 重新创建 scheduler 并启用 LLM
        self.config["explorer"]["llm"]["enabled"] = True
        scheduler = AdaptiveScheduler(self.config)
        # Mock LLM 实例
        mock_llm = Mock()
        mock_llm.generate_variants.return_value = ["new tutorial", "solo tutorial"]
        mock_llm.generate_new_terms_for_category.return_value = ["fresh term"]
        scheduler.llm = mock_llm

        # 设置一个表现不佳的词
        scheduler.pool.update_stats("tutorial", v2_pass_rate=0.02, total_tried=100)
        # 模拟某个类别不足（目标 educational 需要 100 个，当前 0）
        scheduler.target_qualified = 200
        # 类别不足的判断会调用 generate_new_terms_for_category
        scheduler.adapt_strategy()

        # 验证 LLM 被调用了（至少一次）
        self.assertTrue(mock_llm.generate_variants.called or mock_llm.generate_new_terms_for_category.called)
        # 检查新词是否被添加
        new_term = scheduler.pool._find_term("new tutorial")
        self.assertIsNotNone(new_term)
        # 原词权重应降低
        old_term = scheduler.pool._find_term("tutorial")
        self.assertLess(old_term.weight, 1.0)

    def test_should_stop(self):
        """测试停止条件"""
        # 初始未达到目标
        self.assertFalse(self.scheduler.should_stop())
        # 模拟已收集 200 个合格视频
        self.scheduler.pool.update_stats("tutorial", total_qualified=200)
        # 注意 get_category_counts 会统计所有词的 total_qualified
        # 需要把其他词的合格数也加上，简单粗暴：直接修改池子内部
        for term in self.scheduler.pool.terms:
            term.stats.total_qualified = 200
        self.assertTrue(self.scheduler.should_stop())

    def test_get_status(self):
        """测试状态获取"""
        self.scheduler.pool.update_stats("tutorial", total_qualified=10)
        status = self.scheduler.get_status()
        self.assertIn("total_qualified", status)
        self.assertIn("category_counts", status)
        self.assertEqual(status["total_qualified"], 10)

    def test_save_load_state(self):
        """测试保存与加载状态"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        try:
            # 修改一些数据
            self.scheduler.pool.update_stats("tutorial", total_qualified=15, total_tried=100)
            self.scheduler.save_state(temp_path)
            # 创建新的 scheduler 并加载
            new_scheduler = AdaptiveScheduler(self.config)
            new_scheduler.load_state(temp_path)
            # 验证数据恢复
            term = new_scheduler.pool._find_term("tutorial")
            self.assertEqual(term.stats.total_qualified, 15)
            self.assertEqual(term.stats.total_tried, 100)
        finally:
            os.unlink(temp_path)


class TestLLMOptimizerMock(unittest.TestCase):
    """测试 LLM 优化器（模拟 API 调用）"""

    @patch('openai.OpenAI')
    def test_generate_variants(self, mock_openai):
        """测试生成变体（模拟响应）"""
        # 配置 mock client
        mock_client = Mock()
        mock_completion = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "new term1\nnew term2\nnew term3"
        mock_completion.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # 设置环境变量
        os.environ["OPENAI_API_KEY"] = "fake-key"
        optimizer = LLMOptimizer(model="gpt-4o-mini")
        variants = optimizer.generate_variants(
            original_term="tutorial",
            failure_reasons={"head_pose": 10},
            platform="youtube",
            category="educational",
            num_variants=3
        )
        self.assertEqual(len(variants), 3)
        self.assertEqual(variants[0], "new term1")
        # 清理环境变量
        del os.environ["OPENAI_API_KEY"]


if __name__ == "__main__":
    unittest.main()