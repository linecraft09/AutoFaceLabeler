# src/explorers/llm_optimizer.py
import os
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from aflutils.logger import get_logger

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / "api_key.env")


class LLMOptimizer:
    """调用大语言模型优化搜索词"""

    def __init__(self, model: str = "gpt-4o-mini", api_key_env: str = "OPENAI_API_KEY",
                 base_url: Optional[str] = None, temperature: float = 0.7):
        """
        :param model: OpenAI 模型名称
        :param api_key_env: 环境变量名
        :param base_url: 可选的自定义 API 地址（如代理）
        :param temperature: 生成温度
        """
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"Environment variable {api_key_env} not set")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def generate_variants(self, original_term: str, failure_reasons: Dict[str, int],
                          category: str, num_variants: int = 3) -> List[str]:
        """
        根据失败原因生成变体搜索词。
        :param original_term: 原始搜索词
        :param failure_reasons: 失败原因计数（来自 V1/V2）
        :param category: 内容类别
        :param num_variants: 生成数量
        :return: 新的搜索词列表
        """
        # 构建失败原因描述
        fail_desc = ", ".join([f"{k}: {v}" for k, v in failure_reasons.items() if v > 0])
        if not fail_desc:
            fail_desc = "unknown reasons"

        prompt = f"""你是一个视频数据采集助手。当前搜索词 "{original_term}" 用于搜索视频平台中的 {category} 类别视频。
在筛选过程中，主要失败原因是: {fail_desc}。

请生成 {num_variants} 个新的搜索词，要求能够检索到“单人、正面、清晰、有音频”的视频。
输出仅列出搜索词，每行一个，不要有额外解释。

示例输出格式：
solo tutorial front face
one-person how-to video
single presenter tutorial
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个搜索词优化专家，只输出搜索词列表，每行一个。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )
            content = response.choices[0].message.content.strip()
            variants = [line.strip() for line in content.split('\n') if line.strip()]
            # 去重并限制数量
            variants = list(dict.fromkeys(variants))[:num_variants]
            logger.info(f"Generated {len(variants)} variants for '{original_term}': {variants}")
            return variants
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return []

    def generate_new_terms_for_category(self, category: str, target_qualities: List[str] = None,
                                        num_terms: int = 3) -> List[str]:
        """
        为一个类别生成全新的搜索词（当该类别数量不足时）。
        :param category: 类别名称
        :param target_qualities: 期望的视频质量描述，如 ["single person", "front face"]
        :param num_terms: 生成数量
        """
        quality_desc = ", ".join(target_qualities) if target_qualities else "单人正面"
        prompt = f"""请为视频平台生成 {num_terms} 个搜索词，用于查找 {category} 类别的视频。
这些视频应满足以下质量要求: {quality_desc}。
输出仅列出搜索词，每行一个。
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个搜索词生成专家，只输出搜索词列表。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
            )
            content = response.choices[0].message.content.strip()
            terms = [line.strip() for line in content.split('\n') if line.strip()]
            terms = list(dict.fromkeys(terms))[:num_terms]
            logger.info(f"Generated new terms for category {category}: {terms}")
            return terms
        except Exception as e:
            logger.error(f"LLM generation for category failed: {e}")
            return []
