"""Prompts of the curriculum dictionary (Section 5.2 and Appendix B)."""

from __future__ import annotations

import json

from .kb import TAXONOMY

CATEGORY_LIST = "、".join(f"{cid}（{zh}）" for cid, (zh, _) in TAXONOMY.items())

# ---------------------------------------------------------- dictionary: stage A
CONCEPT_EXTRACTION_SYSTEM = "你是中国高中地理课程专家，熟悉人教版、湘教版、中图版高中地理教材。"

CONCEPT_EXTRACTION_USER = """下面是《普通高中地理课程标准》中的一条内容要求。
请列出学生掌握这条要求所需的核心地理概念与术语，包括：上位概念、主要类型、
关键要素与过程、典型实例（如典型区域、地貌、气候类型、河流、产业等）。

要求：
1. 共 20~40 个，每个是简短的名词或名词短语（2~10 个字），不要句子；
2. 使用教材中的规范术语；
3. 只输出 JSON：{{"concepts": ["...", "..."]}}

模块：{module}
内容要求：{text}"""

# ---------------------------------------------------------- dictionary: stage B
ENTRY_SYSTEM = "你是中国高中地理知识工程专家，负责把地理术语整理成结构化知识条目，内容须与高中地理教材的表述一致。"

ENTRY_USER = """为下列每个地理术语生成一个结构化知识条目。字段说明：
- term：术语本身
- category：从以下类别中选一个类别代号：{categories}
- aliases：严格同义词或常见别称（可为空；不得填写上位概念、下位概念或相关概念）
- is_a：上位概念（1~3 个）
- subtypes：下位概念、主要类型或典型实例（0~6 个）
- features：本质特征或典型性质（2~6 个）
- causes：形成原因、影响因素或作用机制（0~5 个）
- effects：产生的影响、结果或意义（0~5 个）
- locations：典型分布区域或地点（0~5 个）

要求：每个值是 2~10 个字的短语，使用教材常用术语，不要句子；不确定的字段留空列表。
只输出 JSON：{{"entries": [{{"term": "...", "category": "...", "aliases": [], "is_a": [], "subtypes": [], "features": [], "causes": [], "effects": [], "locations": []}}]}}

术语：{terms}"""


def concept_extraction_messages(module: str, text: str) -> list[dict]:
    return [
        {"role": "system", "content": CONCEPT_EXTRACTION_SYSTEM},
        {"role": "user", "content": CONCEPT_EXTRACTION_USER.format(module=module, text=text)},
    ]


def entry_messages(terms: list[str]) -> list[dict]:
    return [
        {"role": "system", "content": ENTRY_SYSTEM},
        {
            "role": "user",
            "content": ENTRY_USER.format(categories=CATEGORY_LIST, terms=json.dumps(terms, ensure_ascii=False)),
        },
    ]
