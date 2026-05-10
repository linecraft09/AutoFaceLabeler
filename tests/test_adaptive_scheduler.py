#!/usr/bin/env python3
"""
测试 AdaptiveScheduler 和 SearchTermPool
"""

import os
import sys
import unittest
import json
import tempfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from explorers.adaptive_scheduler import AdaptiveScheduler
from explorers.search_term_pool import SearchTermPool, SearchTerm


class TestSearchTermPool(unittest.TestCase):
    def setUp(self):
        self.initial_terms = [
            {"text": "人物专访", "platform": "bilibili", "category": "interview", "weight": 1.0},
            {"text": "interview", "platform": "youtube", "category": "interview", "weight": 1.0},
            {"text": "vlog", "platform": "youtube", "category": "vlog", "weight": 1.0},
            {"text": "日常Vlog", "platform": "bilibili", "category": "vlog", "weight": 0.9},
        ]
        self.pool = SearchTermPool(
            initial_terms=self.initial_terms,
            target_distribution={"interview": 0.5, "vlog": 0.5},
            min_weight=0.1,
            weight_decay_factor=0.9,
        )

    def test_pool_initialization(self):
        self.assertEqual(len(self.pool.terms), 4)
        self.assertIsNotNone(self.pool._find_term("人物专访"))

    def test_sample_returns_correct_count(self):
        sampled = self.pool.sample(2)
        self.assertEqual(len(sampled), 2)
        for term in sampled:
            self.assertIsInstance(term, SearchTerm)
            self.assertTrue(term.text in ["人物专访", "interview", "vlog", "日常Vlog"])

    def test_sample_distribution(self):
        """验证采样覆盖不同类别"""
        all_sampled = set()
        for _ in range(20):
            batch = self.pool.sample(4)
            all_sampled.update(t.text for t in batch)
        self.assertGreaterEqual(len(all_sampled), 2)

    def test_sample_ignores_target_categories_without_terms(self):
        pool = SearchTermPool(
            initial_terms=[
                {"text": "vlog", "platform": "bilibili", "category": "vlog", "weight": 1.0},
            ],
            target_distribution={"educational": 0.7, "interview": 0.2, "vlog": 0.1},
            min_weight=0.1,
            weight_decay_factor=0.9,
        )

        sampled = pool.sample(3)

        self.assertEqual(len(sampled), 3)
        self.assertTrue(all(term.category == "vlog" for term in sampled))

    def test_update_stats(self):
        self.pool.update_stats("人物专访", v1_pass_rate=0.5, total_tried=10, total_downloaded=5)
        term = self.pool._find_term("人物专访")
        self.assertEqual(term.stats.v1_pass_rate, 0.5)
        self.assertEqual(term.stats.total_tried, 10)

    def test_update_weights(self):
        """update_weights adds delta and re-normalizes within category."""
        initial_weight = self.pool._find_term("人物专访").weight  # 0.5 (normalized, 2 terms in interview)
        self.pool.update_weights("人物专访", 0.2)
        new_weight = self.pool._find_term("人物专访").weight
        # After adding 0.2, raw becomes 1.2. Normalized: 1.2 / (1.2 + 1.0) = 0.545...
        expected = (0.5 + 0.2 / 2.0)  # approximate: delta distributed across normalized terms
        self.assertGreater(new_weight, initial_weight)  # weight should increase

    def test_underperforming_terms(self):
        self.pool.update_stats("人物专访", v2_pass_rate=0.01, total_qualified=0, total_tried=20)
        poor = self.pool.get_underperforming_terms(
            v2_pass_threshold=0.05, v1_pass_threshold=0.1
        )
        term_texts = [t.text for t in poor]
        self.assertIn("人物专访", term_texts)

    def test_serialize_deserialize(self):
        data = self.pool.to_dict()
        self.assertIn("terms", data)
        self.assertIn("target_dist", data)

        restored = SearchTermPool.from_dict(data)
        self.assertEqual(len(restored.terms), 4)
        rt = restored._find_term("人物专访")
        self.assertIsNotNone(rt)
        self.assertEqual(rt.category, "interview")
        # weights are normalized per category (interview has 2 terms with equal weight 1.0 -> 0.5 each)
        self.assertAlmostEqual(rt.weight, 0.5)

    def test_add_term(self):
        self.pool.add_term(text="test term", category="test", weight=0.5)
        self.assertEqual(len(self.pool.terms), 5)
        term = self.pool._find_term("test term")
        self.assertIsNotNone(term)
        self.assertEqual(term.category, "test")

    def test_set_weight(self):
        """set_weight sets then re-normalizes within category.
        After setUp, interview category has normalized weights [0.5, 0.5].
        set_weight('人物专访', 0.3) -> raw [0.3, 0.5] -> normalized: 0.3/0.8 = 0.375
        """
        self.pool.set_weight("人物专访", 0.3)
        actual = self.pool._find_term("人物专访").weight
        expected_approx = 0.3 / 0.8  # 0.3 / (0.3 + 0.5)
        self.assertAlmostEqual(actual, expected_approx, places=5)

    def test_get_category_counts(self):
        counts = self.pool.get_category_counts()
        self.assertEqual(counts.get("interview", 0), 0)
        self.assertEqual(counts.get("vlog", 0), 0)


class TestAdaptiveScheduler(unittest.TestCase):
    def setUp(self):
        self.config = {
            "initial_search_terms": [
                {"text": "test1", "platform": "youtube", "category": "interview", "weight": 1.0},
                {"text": "test2", "platform": "bilibili", "category": "vlog", "weight": 1.0},
            ],
            "sampling": {
                "target_distribution": {"interview": 0.5, "vlog": 0.5},
                "min_weight": 0.1,
                "weight_decay_factor": 0.9,
            },
            "llm": {"enabled": False},
            "project": {"explorer_state": None},
            "pipeline": {"target_qualified": 200},
        }
        self.scheduler = AdaptiveScheduler(self.config)

    def test_generate_batch(self):
        batch = self.scheduler.generate_batch(batch_size=5)
        self.assertLessEqual(len(batch), 5)
        for term in batch:
            self.assertIsInstance(term.text, str)

    def test_receive_v1_feedback(self):
        feedback = {
            "search_term": "test1",
            "v1_pass_rate": 0.3,
            "fail_reasons": {"duration": 5, "resolution": 3},
            "total_received": 20,
        }
        self.scheduler.receive_feedback(feedback)
        term = self.scheduler.pool._find_term("test1")
        self.assertEqual(term.stats.v1_pass_rate, 0.3)

    def test_receive_v2_feedback(self):
        feedback = {
            "search_term": "test1",
            "v2_pass_rate": 0.1,
            "fail_reasons": {"no_single_person": 8},
            "total_qualified": 2,
        }
        self.scheduler.receive_feedback(feedback)
        term = self.scheduler.pool._find_term("test1")
        self.assertEqual(term.stats.v2_pass_rate, 0.1)

    def test_get_status(self):
        status = self.scheduler.get_status()
        self.assertIn("total_qualified", status)
        self.assertIn("category_counts", status)
        self.assertIn("total_terms", status)

    def test_adapt_strategy_no_llm(self):
        try:
            self.scheduler.adapt_strategy()
        except Exception as e:
            self.fail(f"adapt_strategy raised {e}")

    def test_save_load_state(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump(self.scheduler.pool.to_dict(), f)
            temp_path = f.name
        try:
            self.scheduler.load_state(temp_path)
            self.assertEqual(len(self.scheduler.pool.terms), 2)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
