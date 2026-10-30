"""Dynamic Concept Retrieval (Section 5.4).

For each question q, DCR detects the out-of-vocabulary terms U(q) among the salient
nouns of the stem and the options, asks an LLM for a dictionary entry of each term
given the question context (term, stem, options and representative dictionary
entries; never the answer), keeps the entries whose self-reported confidence is at
least tau, and returns them as the question-local extension of D_q^+ = D ∪ {e_w}.
The base dictionary is never modified.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .grounding import GENERIC_WORDS, TEMPLATE_WORDS, Linker, norm_text, textrank
from .kb import TAXONOMY, Entry, KnowledgeBase, norm
from .llm import LLMClient, LLMConfig

CATEGORIES = "；".join(f"{cid}（{zh}）" for cid, (zh, _) in TAXONOMY.items())

SYSTEM = "你是高中地理知识工程专家，负责为地理术语编写结构化词典条目。"

USER = """请为下面这道地理选择题中出现的术语“{term}”编写一个词典条目，供符号推理系统使用。

题干：{stem}
选项：
{options}

词典中已有条目的示例：
{examples}

字段要求：
- category：从以下类别中选择一个代号：{categories}
- parent_level：该术语的上位概念、形成机制、影响因素或所属主题，2~10 字的短语，用分号分隔
- child_level：该术语的具体表现、典型实例、产生的结果或相关概念，2~10 字的短语，用分号分隔
- confidence：你对这个条目准确性的把握，0 到 1 之间的数

只输出 JSON：{{"category": "...", "parent_level": "...", "child_level": "...", "confidence": 0.0}}"""


def candidate_ok(w: str) -> bool:
    """g(w): letters only, 2-10 characters, not a question-template or generic word."""
    return (
        bool(re.fullmatch(r"[一-鿿A-Za-z]+", w))
        and 2 <= len(w) <= 10
        and w not in TEMPLATE_WORDS
        and w not in GENERIC_WORDS
    )


def oov_terms(q, kb: KnowledgeBase, max_terms: int) -> list[str]:
    """U(q): salient nouns of the stem and the options (TextRank above the mean) missing from D."""
    text = "\n".join([q.passage or "", q.stem, *q.options.values()])
    tr = textrank(text)
    mean = sum(tr.values()) / len(tr) if tr else 0.0
    out = []
    for w, s in sorted(tr.items(), key=lambda x: (-x[1], x[0])):
        w = norm(w)
        if s > mean and candidate_ok(w) and w not in kb.surface:
            out.append(w)
    return list(dict.fromkeys(out))[:max_terms]


def render_entry(e: Entry) -> str:
    return json.dumps(
        {"term": e.term, "category": e.category, "parent_level": "；".join(e.is_a + e.causes[:2]),
         "child_level": "；".join(e.subtypes[:3] + e.features[:3])},
        ensure_ascii=False,
    )


def examples(q, kb: KnowledgeBase, linker: Linker, k: int = 2) -> str:
    """Representative entries: the first dictionary concepts mentioned in the question."""
    linked, _ = linker.link(f"{q.stem}\n" + "\n".join(q.options.values()))
    picked = [kb.get(c) for c in linked if kb.get(c) is not None][:k]
    if len(picked) < k:
        picked += [kb.entries[t] for t in ("季风", "三角洲") if t in kb.entries][: k - len(picked)]
    return "\n".join(render_entry(e) for e in picked)


def split_phrases(value) -> list[str]:
    if isinstance(value, list):
        value = "；".join(map(str, value))
    return [p for p in (norm(x) for x in re.split(r"[；;、，,]", str(value or ""))) if 2 <= len(p) <= 12]


class DCR:
    def __init__(self, client: LLMClient, tau: float = 0.5, max_terms: int = 8):
        self.client = client
        self.tau = tau
        self.max_terms = max_terms

    def messages(self, term: str, q, kb: KnowledgeBase, linker: Linker) -> list[dict]:
        options = "\n".join(f"{k}. {v}" for k, v in sorted(q.options.items()))
        stem = norm_text(f"{q.passage}\n{q.stem}" if q.passage else q.stem)
        user = USER.format(term=term, stem=stem, options=options, examples=examples(q, kb, linker),
                           categories=CATEGORIES)
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def retrieve(self, term: str, q, kb: KnowledgeBase, linker: Linker) -> tuple[Entry | None, float | None]:
        msgs = self.messages(term, q, kb, linker)
        key = f"{term}||{hashlib.md5(msgs[1]['content'].encode()).hexdigest()[:12]}"
        data, _ = self.client.chat_json(msgs, tag=f"DCR:{key}")
        if not isinstance(data, dict):
            return None, None
        try:
            conf = float(data.get("confidence"))
        except (TypeError, ValueError):
            return None, None
        cat = str(data.get("category", "")).strip().split("（")[0]
        entry = Entry(term=term, category=cat if cat in TAXONOMY else "OTHER",
                      is_a=split_phrases(data.get("parent_level"))[:6],
                      features=split_phrases(data.get("child_level"))[:8], origin=["dcr"])
        return entry, conf

    def augment(self, q, kb: KnowledgeBase, linker: Linker) -> list[Entry]:
        """The entries e_w (w ∈ U(q), confidence ≥ tau) of the question-local dictionary."""
        out = []
        for term in oov_terms(q, kb, self.max_terms):
            e, conf = self.retrieve(term, q, kb, linker)
            if e is not None and conf >= self.tau and (e.is_a or e.features):
                out.append(e)
        return out

    def prefetch(self, questions, kb: KnowledgeBase, workers: int = 8) -> None:
        """Warm the cache in parallel; answers are read back from the cache in question order."""
        linker = Linker(kb)
        jobs = [(t, q) for q in questions for t in oov_terms(q, kb, self.max_terms)]
        self.client.map(lambda job: self.retrieve(job[0], job[1], kb, linker), jobs, workers)

    def describe(self) -> dict:
        return {**self.client.config.public(), "tau": self.tau, "max_terms": self.max_terms,
                "usage": self.client.usage.as_dict()}


def make_dcr(model: str, root: Path, tau: float = 0.5, temperature: float = 0.0, seed: int | None = None) -> DCR:
    if model.startswith("deepseek") and "/" not in model:  # DeepSeek platform
        cfg = LLMConfig.from_env("KB_LLM", model=model, extra_body={"thinking": {"type": "disabled"}})
    else:  # OpenAI-compatible endpoint (SiliconFlow)
        extra = {"enable_thinking": False} if model.startswith(("Qwen/Qwen3", "deepseek-ai/DeepSeek-V4")) else None
        cfg = LLMConfig.from_env("BASELINE", model=model, extra_body=extra)
    cfg.temperature, cfg.seed, cfg.max_tokens, cfg.json_mode = temperature, seed, 512, True
    return DCR(LLMClient(cfg, root / "kb" / "cache" / "dcr", max_retries=40), tau=tau)
