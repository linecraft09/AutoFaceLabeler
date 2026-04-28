#!/usr/bin/env python3
"""
测试 PreFilter 模块：
1. filter 方法在多条件下返回正确结果
2. 每个过滤条件可独立生效
"""

import os
import sys
import unittest

# 让 Python 能直接导入 src 下的模块（core/validators/aflutils）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from core.models.video_meta import VideoMeta
from validators.pre_filter import PreFilter


def make_video(
    video_id: str,
    duration_seconds: int = 300,
    resolution: str = "1080p",
    title: str = "Great cooking tips",
    channel: str = "good_channel",
    view_count: int = 50_000,
) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        url=f"https://example.com/watch/{video_id}",
        platform="youtube",
        title=title,
        duration_seconds=duration_seconds,
        resolution=resolution,
        channel=channel,
        publish_date="2026-01-01",
        view_count=view_count,
        tags=["cooking", "tutorial"],
        search_term="cooking",
        extra={},
    )


class TestPreFilter(unittest.TestCase):
    def test_filter_returns_expected_result_with_mixed_conditions(self):
        config = {
            "min_duration": 60,
            "max_duration": 600,
            "min_resolution": 720,
            "title_blacklist": [r"shorts", r"trailer"],
            "channel_blacklist": [r"spam_channel"],
            "min_views": 1000,
        }
        pre_filter = PreFilter(config)

        videos = [
            make_video("pass_1"),
            make_video("pass_2", resolution="720p", view_count=1000),
            make_video("fail_duration", duration_seconds=30),
            make_video("fail_resolution", resolution="480p"),
            make_video("fail_title", title="Best shorts of the day"),
            make_video("fail_channel", channel="my_spam_channel_123"),
            make_video("fail_views", view_count=999),
        ]

        passed, feedback = pre_filter.filter(videos, "cooking")
        passed_ids = [v.video_id for v in passed]

        self.assertEqual(passed_ids, ["pass_1", "pass_2"])
        self.assertEqual(feedback["search_term"], "cooking")
        self.assertEqual(feedback["platform"], "youtube")
        self.assertEqual(feedback["total_received"], 7)
        self.assertEqual(feedback["pass_count"], 2)
        self.assertAlmostEqual(feedback["v1_pass_rate"], 2 / 7)
        self.assertEqual(feedback["candidate_urls"], [videos[0].url, videos[1].url])
        self.assertEqual(
            feedback["fail_reasons"],
            {
                "duration": 1,
                "resolution": 1,
                "title_blacklist": 1,
                "channel_blacklist": 1,
                "views": 1,
                "other": 0,
            },
        )

    def test_duration_filter_independent(self):
        pre_filter = PreFilter(
            {
                "min_duration": 60,
                "max_duration": 600,
                "min_resolution": 1,
                "min_views": 0,
            }
        )
        videos = [make_video("ok"), make_video("bad", duration_seconds=30)]

        passed, feedback = pre_filter.filter(videos, "duration_test")

        self.assertEqual([v.video_id for v in passed], ["ok"])
        self.assertEqual(feedback["fail_reasons"]["duration"], 1)
        self.assertEqual(feedback["fail_reasons"]["resolution"], 0)
        self.assertEqual(feedback["fail_reasons"]["views"], 0)

    def test_resolution_filter_independent(self):
        pre_filter = PreFilter(
            {
                "min_duration": 1,
                "max_duration": 10_000,
                "min_resolution": 720,
                "min_views": 0,
            }
        )
        videos = [make_video("ok", resolution="1080p"), make_video("bad", resolution="480p")]

        passed, feedback = pre_filter.filter(videos, "resolution_test")

        self.assertEqual([v.video_id for v in passed], ["ok"])
        self.assertEqual(feedback["fail_reasons"]["resolution"], 1)
        self.assertEqual(feedback["fail_reasons"]["duration"], 0)
        self.assertEqual(feedback["fail_reasons"]["views"], 0)

    def test_title_blacklist_filter_independent(self):
        pre_filter = PreFilter(
            {
                "min_duration": 1,
                "max_duration": 10_000,
                "min_resolution": 1,
                "title_blacklist": [r"forbidden"],
                "min_views": 0,
            }
        )
        videos = [make_video("ok", title="normal title"), make_video("bad", title="Forbidden content")]

        passed, feedback = pre_filter.filter(videos, "title_test")

        self.assertEqual([v.video_id for v in passed], ["ok"])
        self.assertEqual(feedback["fail_reasons"]["title_blacklist"], 1)
        self.assertEqual(feedback["fail_reasons"]["duration"], 0)
        self.assertEqual(feedback["fail_reasons"]["resolution"], 0)
        self.assertEqual(feedback["fail_reasons"]["views"], 0)

    def test_channel_blacklist_filter_independent(self):
        pre_filter = PreFilter(
            {
                "min_duration": 1,
                "max_duration": 10_000,
                "min_resolution": 1,
                "channel_blacklist": [r"blocked_channel"],
                "min_views": 0,
            }
        )
        videos = [
            make_video("ok", channel="safe_channel"),
            make_video("bad", channel="my_blocked_channel_x"),
        ]

        passed, feedback = pre_filter.filter(videos, "channel_test")

        self.assertEqual([v.video_id for v in passed], ["ok"])
        self.assertEqual(feedback["fail_reasons"]["channel_blacklist"], 1)
        self.assertEqual(feedback["fail_reasons"]["duration"], 0)
        self.assertEqual(feedback["fail_reasons"]["resolution"], 0)
        self.assertEqual(feedback["fail_reasons"]["views"], 0)

    def test_views_filter_independent(self):
        pre_filter = PreFilter(
            {
                "min_duration": 1,
                "max_duration": 10_000,
                "min_resolution": 1,
                "min_views": 1000,
            }
        )
        videos = [make_video("ok", view_count=1000), make_video("bad", view_count=999)]

        passed, feedback = pre_filter.filter(videos, "views_test")

        self.assertEqual([v.video_id for v in passed], ["ok"])
        self.assertEqual(feedback["fail_reasons"]["views"], 1)
        self.assertEqual(feedback["fail_reasons"]["duration"], 0)
        self.assertEqual(feedback["fail_reasons"]["resolution"], 0)


if __name__ == "__main__":
    unittest.main()
