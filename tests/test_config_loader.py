#!/usr/bin/env python3
"""
测试 ConfigLoader 核心能力：
- 加载 YAML 配置
- get / get_required
- as_dict
- 环境变量替换
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# 确保能够导入 src 模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.aflutils.config_loader import ConfigLoader


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_dir.name) / "config.yaml"
        self.config_path.write_text(
            """
app:
  name: "AutoFaceLabeler"
server:
  host: "127.0.0.1"
  port: "${TEST_PORT:8080}"
database:
  user: "${DB_USER:test_user}"
  password: "${DB_PASS}"
nested:
  value: "prefix-${TEST_PORT:8080}-suffix"
list_values:
  - "${DB_USER:test_user}"
  - "static"
""".strip(),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_and_env_replace(self):
        with patch.dict(os.environ, {"TEST_PORT": "9001", "DB_USER": "alice"}, clear=False):
            loader = ConfigLoader(self.config_path)
            self.assertEqual(loader.get("server.port"), "9001")
            self.assertEqual(loader.get("database.user"), "alice")
            self.assertEqual(loader.get("nested.value"), "prefix-9001-suffix")
            self.assertEqual(loader.get("list_values"), ["alice", "static"])
            self.assertEqual(loader.get("database.password"), "${DB_PASS}")

    def test_get(self):
        loader = ConfigLoader(self.config_path, load_env=False)
        self.assertEqual(loader.get("app.name"), "AutoFaceLabeler")
        self.assertEqual(loader.get("not.exists", "fallback"), "fallback")
        self.assertIsNone(loader.get("not.exists"))

    def test_get_required(self):
        loader = ConfigLoader(self.config_path, load_env=False)
        self.assertEqual(loader.get_required("server.host"), "127.0.0.1")
        with self.assertRaises(KeyError):
            loader.get_required("server.not_exists")

    def test_as_dict(self):
        loader = ConfigLoader(self.config_path, load_env=False)
        data = loader.as_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["app"]["name"], "AutoFaceLabeler")
        data["new_key"] = "new_value"
        self.assertIsNone(loader.get("new_key"))

    def test_load_without_env_replacement(self):
        with patch.dict(os.environ, {"TEST_PORT": "9001", "DB_USER": "alice"}, clear=False):
            loader = ConfigLoader(self.config_path, load_env=False)
            self.assertEqual(loader.get("server.port"), "${TEST_PORT:8080}")
            self.assertEqual(loader.get("database.user"), "${DB_USER:test_user}")


if __name__ == "__main__":
    unittest.main()
