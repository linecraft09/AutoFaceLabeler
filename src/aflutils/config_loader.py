"""
配置加载工具类
支持：
- 从 YAML 文件加载配置
- 环境变量替换 ${VAR_NAME} 或 ${VAR_NAME:default_value}
- 嵌套属性访问（如 config.database.host）
- 获取配置时支持默认值
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Union

import yaml
from aflutils.logger import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """配置加载器，支持环境变量替换和嵌套属性访问"""

    # 环境变量替换模式：${VAR_NAME} 或 ${VAR_NAME:default}
    _ENV_VAR_PATTERN = re.compile(r'\$\{([^:}]+)(?::([^}]*))?\}')

    def __init__(self, config_path: Union[str, Path], load_env: bool = True):
        """
        初始化配置加载器
        :param config_path: YAML 配置文件路径
        :param load_env: 是否进行环境变量替换，默认为 True
        """
        self.config_path = Path(config_path)
        self._raw_config: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
        self.load(load_env)

    def load(self, load_env: bool = True) -> 'ConfigLoader':
        """加载并解析配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f) or {}

        self._config = self._raw_config.copy()
        self._validate_schema(self._config)
        if load_env:
            self._config = self._replace_env_vars(self._config)
            self._validate_schema(self._config)
        return self

    def _validate_schema(self, config: Dict[str, Any]) -> None:
        """轻量级配置校验：检查关键路径类型与未知键。"""
        expected_schema: Dict[str, Any] = {
            "orchestrator": {
                "target_qualified": int,
            },
            "validator2": {
                "coarse_filter": {
                    "model_path": str,
                },
                "fine_filter": {
                    "face_db_path": str,
                },
                "qualified_dir": str,
            },
        }

        def _type_name(t: Any) -> str:
            return getattr(t, "__name__", str(t))

        def _validate_required(node: Any, schema_node: Any, prefix: str = "") -> None:
            if not isinstance(schema_node, dict):
                return
            if not isinstance(node, dict):
                logger.warning(
                    f"Config key '{prefix or '<root>'}' should be a mapping, got {type(node).__name__}"
                )
                return

            for key, expected in schema_node.items():
                key_path = f"{prefix}.{key}" if prefix else key
                if key not in node:
                    logger.warning(f"Missing required config key: {key_path}")
                    continue
                value = node[key]
                if isinstance(expected, dict):
                    _validate_required(value, expected, key_path)
                elif not isinstance(value, expected):
                    logger.warning(
                        f"Invalid type for '{key_path}': expected {_type_name(expected)}, "
                        f"got {type(value).__name__}"
                    )

        def _warn_unexpected(node: Any, schema_node: Any, prefix: str = "") -> None:
            if not isinstance(schema_node, dict) or not isinstance(node, dict):
                return
            expected_keys = set(schema_node.keys())
            for key, value in node.items():
                key_path = f"{prefix}.{key}" if prefix else key
                if key not in expected_keys:
                    logger.warning(f"Unexpected config key: {key_path}")
                    continue
                _warn_unexpected(value, schema_node[key], key_path)

        _validate_required(config, expected_schema)
        for scoped_key in ("orchestrator", "validator2"):
            if scoped_key in config and scoped_key in expected_schema:
                _warn_unexpected(config[scoped_key], expected_schema[scoped_key], scoped_key)

    def _replace_env_vars(self, obj: Any) -> Any:
        """递归替换对象中的环境变量"""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._expand_env_vars(obj)
        else:
            return obj

    def _expand_env_vars(self, value: str) -> str:
        """替换字符串中的环境变量"""

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default_value is not None:
                return default_value
            # 没有默认值且环境变量不存在，保留原占位符（或抛出异常，根据需求选择）
            return match.group(0)

        return self._ENV_VAR_PATTERN.sub(replacer, value)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        通过点分隔的路径获取配置值
        :param key_path: 如 'database.host' 或 'server.port'
        :param default: 如果路径不存在返回的默认值
        :return: 配置值
        """
        keys = key_path.split('.')
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_required(self, key_path: str) -> Any:
        """获取必需的配置，若不存在则抛出 KeyError"""
        value = self.get(key_path)
        if value is None:
            raise KeyError(f"必需的配置项缺失: {key_path}")
        return value

    def as_dict(self) -> Dict[str, Any]:
        """返回整个配置字典（已进行环境变量替换）"""
        return self._config.copy()

    def raw_dict(self) -> Dict[str, Any]:
        """返回未进行环境变量替换的原始配置字典"""
        return self._raw_config.copy()

    def reload(self) -> 'ConfigLoader':
        """重新加载配置文件（例如文件变更后）"""
        return self.load()

    def __getitem__(self, key: str) -> Any:
        """支持 config['database.host'] 语法"""
        return self.get(key)

    def __contains__(self, key_path: str) -> bool:
        """支持 'database.host' in config 语法"""
        return self.get(key_path) is not None


# 使用示例
if __name__ == "__main__":
    # 假设存在 config.yaml 文件，内容如下：
    """
    project:
      name: "FaceVideoPipeline"
      version: "1.0"

    server:
      host: "0.0.0.0"
      port: ${PORT:8000}

    youtube:
      api_key: ${YOUTUBE_API_KEY}
      max_results: 50

    validator1:
      min_duration: 30
      title_blacklist:
        - "reaction"
        - "multi"
    """

    # 设置环境变量（示例）
    os.environ["PORT"] = "9000"
    # os.environ["YOUTUBE_API_KEY"] = "your-actual-api-key"  # 实际使用时设置

    # 加载配置
    config = ConfigLoader("config.yaml")
    print("项目名称:", config.get("project.name"))
    print("服务器端口:", config.get("server.port"))  # 输出 9000
    print("YouTube API Key:", config.get("youtube.api_key"))  # 若环境变量未设置，输出 None
    print("标题黑名单:", config.get("validator1.title_blacklist"))

    # 使用 get_required 获取必需项
    try:
        api_key = config.get_required("youtube.api_key")
        print("API Key (必需):", api_key)
    except KeyError as e:
        print(e)

    # 支持字典式访问
    print("项目版本:", config["project.version"])

    # 检查配置是否存在
    if "validator1.min_duration" in config:
        print("min_duration =", config["validator1.min_duration"])
