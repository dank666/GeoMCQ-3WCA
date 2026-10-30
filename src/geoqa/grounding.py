"""Concept linking: map question text to dictionary concepts and out-of-vocabulary candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import jieba
import jieba.analyse
import jieba.posseg as pseg

from .kb import KnowledgeBase, norm

jieba.setLogLevel(60)

# Question-template words that are never geographic concepts (OOV filter g(w), condition iii).
TEMPLATE_WORDS = set(
    """下列 选项 说法 叙述 表述 正确 错误 主要 原因 影响 因素 作用 特点 特征 可能 属于 不属于 符合 反映 体现
    说明 表明 推测 判断 据此 完成 小题 下面 其中 以下 关于 有关 方面 方式 措施 目的 意义 结果 现象 问题
    变化 情况 条件 程度 状况 地方 当地 该地 该区 此地 时间 时期 期间 一般 通常 明显 显著 较大 较小 最大 最小
    增加 减少 提高 降低 增强 减弱 形成 导致 造成 使得 由于 因为 所以 因此 结合 根据 分析 材料 图示 如图
    我国 中国 世界 全球 地区 区域 城市 国家""".split()
)

# Generic abstract words that are not geographic concepts in their own right.
GENERIC_WORDS = set(
    """发展 差异 空间 管理 社会 结构 数据 规划 生活 能力 指导 服务 要素 观测 功能 质量 过程 建设 生产 研究
    中心 水平 活动 国际 信息 历史 效率 物质 优化 产生 范围 利用 综合 辅助 系统 方向 工程 属性 压力 风险 基础
    距离 决策 驱动 运动 规模 数量 目标 联系 合作 精度 产品 规范 关系 性质 价值 流动 制度 统一 人为 尺度 阶段
    制约 指标 损失 类型 网络 调查 效益 科学 规律 分类 传统 威胁 传播 强度 移动 大量 合理 调节 环节 核心 含量
    基本 两者 生长 面积 方法 手段 技术 模式 机制 原理 概念 内容 形式 部分 整体 主体 对象 来源 途径 措施 政策
    意识 观念 理念 思想 文化 时间 速度 状态 环境 资源 问题 作用 影响 地区 区域 当地 地方 世界""".split()
)

NEGATION = re.compile(r"不正确|不属于|不是|错误的是|不能|不可能|不符合|不合理|不包括|不会|不宜|不利于|无关|最不")

_OOV_POS = {"n", "nr", "ns", "nt", "nz", "vn", "nl", "ng"}


@dataclass
class Grounding:
    concepts: list[str] = field(default_factory=list)  # canonical dictionary terms, in order of first mention
    oov: list[str] = field(default_factory=list)  # out-of-vocabulary candidates U(q)


class Linker:
    """Longest-match concept linker over the dictionary's surface forms."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.max_len = max((len(s) for s in kb.surfaces()), default=1)
        self.surfaces = set(kb.surfaces())

    def link(self, text: str) -> tuple[list[str], list[tuple[int, int]]]:
        text = norm_text(text)
        concepts, spans, i = [], [], 0
        while i < len(text):
            for L in range(min(self.max_len, len(text) - i), 1, -1):
                piece = text[i : i + L]
                if piece in self.surfaces:
                    concepts.append(self.kb.canon(piece))
                    spans.append((i, i + L))
                    i += L
                    break
            else:
                i += 1
        return list(dict.fromkeys(concepts)), spans

    def oov_candidates(self, text: str, spans: list[tuple[int, int]]) -> list[str]:
        text = norm_text(text)
        covered = set()
        for a, b in spans:
            covered.update(range(a, b))
        out, pos = [], 0
        for w in pseg.cut(text):
            word, flag = w.word, w.flag
            start, pos = pos, pos + len(word)
            if flag not in _OOV_POS or not (2 <= len(word) <= 10):
                continue
            if not re.fullmatch(r"[一-鿿A-Za-z]+", word):  # condition (i)
                continue
            if word in TEMPLATE_WORDS or word in GENERIC_WORDS or any(k in covered for k in range(start, pos)):
                continue
            if norm(word) in self.surfaces:
                continue
            out.append(norm(word))
        return list(dict.fromkeys(out))

    def ground(self, text: str) -> Grounding:
        concepts, spans = self.link(text)
        return Grounding(concepts=concepts, oov=self.oov_candidates(text, spans))


def norm_text(text: str) -> str:
    text = re.sub(r"\s+", "", str(text or ""))
    return text.replace("（", "(").replace("）", ")")


def is_negative_stem(stem: str) -> bool:
    """Negatively phrased items ask for the option that does NOT hold; only the question sentence counts."""
    question = re.split(r"[。！？；\n]", norm_text(stem).rstrip("。！？；"))[-1]
    return bool(NEGATION.search(question[-30:]))


NOUNS = ("n", "nr", "ns", "nt", "nz", "vn", "nl", "ng")


def textrank(text: str) -> dict[str, float]:
    """TextRank salience of the nouns of a text (damping 0.85, co-occurrence window 5)."""
    if not text.strip():
        return {}
    return dict(jieba.analyse.textrank(norm_text(text), topK=None, withWeight=True, allowPOS=NOUNS))
