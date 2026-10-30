"""Three-way concept reasoning over an LLM-elicited, dictionary-grounded three-valued context.

For a question q the method builds the incomplete formal context

    K_q = (O_q, M_q, R+, R-)

  O_q  decision units: the options, or the statements of a statement-combination item
  M_q  conditions that the correct answer must satisfy, elicited from the stem alone
  R+   (o, m): o satisfies m          R-  (o, m): o violates m
  pairs in neither relation are undetermined.

The LLM judges every unit against every condition in one call, given the question
and the dictionary knowledge D_q^+ of the concepts the question mentions; it is
never asked for the answer. Units fall into three regions (certain / possible
semantics of incomplete information):

  POS  units satisfying every condition
  NEG  units violating at least one condition
  BND  units with no violation and at least one undetermined condition

The answer is the extent of a three-way concept, so composite answers need no
enumeration of option combinations. Cases the context leaves undecided are
deferred to a comparison among the remaining candidates (sequential three-way
decision).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from .fca import FormalContext
from .grounding import Linker, is_negative_stem
from .kb import KnowledgeBase
from .llm import LLMClient
from .prompts import direct_messages
from .prompts_tw import condition_messages, judge_messages

VALUE = {"是": "+", "否": "-"}
STATEMENT_MARK = re.compile(r"[①②③④⑤⑥]")
# A condition that already states the negation of the question ("不属于…", "是错误的说法").
NEGATED_CONDITION = re.compile(r"不正确|不属于|不是|错误|不能|不可能|不符合|不合理|不包括|不会|不宜|无关|并非")


@dataclass
class Params:
    knowledge: bool = True  # ground the judgments in dictionary knowledge
    max_knowledge: int = 6  # dictionary entries shown per question
    min_multi: int = 2  # a multiple-choice answer has at least two options
    defer: bool = True  # resolve undecided single answers in a second stage
    three_valued: bool = True  # False: undetermined judgments count as satisfied (two-valued context)
    max_conditions: int = 6


@dataclass
class Context:
    conditions: list[str]
    polarity: str
    judgments: dict[str, list[str]]  # unit -> "+", "-" or "?" per condition

    def formal(self) -> FormalContext:
        units = list(self.judgments)
        pos = [(u, m) for u in units for m, v in zip(self.conditions, self.judgments[u]) if v == "+"]
        neg = [(u, m) for u in units for m, v in zip(self.conditions, self.judgments[u]) if v == "-"]
        return FormalContext(units, self.conditions, pos, neg)


@dataclass
class Result:
    uid: str
    answer: list[str]
    gold: list[str]
    correct: bool
    jaccard: float
    stage: str  # "context" (decided by the three-way context) or "deferred"
    regions: dict[str, str]
    context: dict
    explanation: dict = field(default_factory=dict)
    dcr_terms: list[str] = field(default_factory=list)
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def tiebreak(uid: str, label: str) -> int:
    """Deterministic, position-independent tie-breaking."""
    return int(hashlib.md5(f"{uid}:{label}".encode()).hexdigest()[:8], 16)


def stem_of(q) -> str:
    """The stem without the statement list of a statement-combination item."""
    return STATEMENT_MARK.split(q.stem)[0] if q.is_statement_combo else q.stem


def knowledge_text(kb: KnowledgeBase, linker: Linker, q, limit: int) -> str:
    texts = [q.passage or "", stem_of(q)] + list((q.statements or q.options).values())
    concepts = []
    for t in texts:
        concepts += linker.link(t)[0]
    lines = []
    for c in dict.fromkeys(concepts):
        e = kb.get(c)
        if e is None:
            continue
        parts = [f"{name}：{'、'.join(vals[:5])}" for name, vals in
                 (("上位", e.is_a), ("特征", e.features), ("成因", e.causes), ("影响", e.effects), ("分布", e.locations))
                 if vals]
        if parts:
            lines.append(f"- {e.term}：" + "；".join(parts))
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def parse_letters(text: str, allowed: list[str]) -> list[str]:
    """Option letters of a short answer such as "AC", "A, C" or "A\nC" (letters inside words are ignored)."""
    runs = re.findall(r"(?<![A-Za-z])([A-H]{1,8})(?![A-Za-z])", text or "")
    return sorted({ch for run in runs for ch in run if ch in allowed})


class Solver:
    def __init__(self, client: LLMClient, text_client: LLMClient, kb: KnowledgeBase, params: Params | None = None,
                 dcr=None):
        self.client = client  # JSON mode: conditions and judgments
        self.text_client = text_client  # plain text: second-stage comparison
        self.kb = kb
        self.linker = Linker(kb)
        self.params = params or Params()
        self.dcr = dcr
        self.usage = [0, 0, 0]  # calls, prompt tokens, completion tokens of the current question

    def _record(self, rec: dict) -> None:
        u = rec.get("usage") or {}
        self.usage[0] += 1
        self.usage[1] += u.get("prompt_tokens", 0)
        self.usage[2] += u.get("completion_tokens", 0)

    def dictionary(self, q) -> tuple[KnowledgeBase, Linker, list[str]]:
        """D_q^+: the dictionary with the question-local DCR entries."""
        if self.dcr is None:
            return self.kb, self.linker, []
        entries = self.dcr.augment(q, self.kb, self.linker)
        if not entries:
            return self.kb, self.linker, []
        kb = self.kb.with_entries(entries)
        return kb, Linker(kb), [e.term for e in entries]

    def elicit(self, q, kb: KnowledgeBase, linker: Linker) -> Context:
        stem = stem_of(q)
        data, rec = self.client.chat_json(condition_messages(stem, q.passage), tag=f"COND:{q.uid}")
        self._record(rec)
        conds = [str(c).strip() for c in (data or {}).get("conditions", []) if str(c).strip()]
        conds = list(dict.fromkeys(conds))[: self.params.max_conditions] or ["是本题的正确答案"]
        polarity = (data or {}).get("polarity", "positive")
        if polarity not in ("positive", "negative"):
            polarity = "positive"
        if any(NEGATED_CONDITION.search(c) for c in conds):  # the negation is already part of a condition
            polarity = "positive"
        elif is_negative_stem(stem):  # the lexical cue overrides an inconsistent label
            polarity = "negative"
        units = dict(q.statements) if q.is_statement_combo else dict(q.options)
        know = knowledge_text(kb, linker, q, self.params.max_knowledge) if self.params.knowledge else ""
        msgs = judge_messages(stem, q.passage, units, conds, know, statements=q.is_statement_combo)
        data, rec = self.client.chat_json(msgs, tag=f"JUDGE:{q.uid}")
        self._record(rec)
        judgments = {}
        for u in units:
            raw = (data or {}).get(u) or []
            vals = [VALUE.get(str(v).strip(), "?") for v in raw][: len(conds)]
            judgments[u] = vals + ["?"] * (len(conds) - len(vals))
        if not self.params.three_valued:
            judgments = {u: ["-" if v == "-" else "+" for v in vals] for u, vals in judgments.items()}
        return Context(conds, polarity, judgments)

    @staticmethod
    def regions(ctx: Context) -> dict[str, str]:
        return {u: "NEG" if "-" in v else "POS" if all(x == "+" for x in v) else "BND" for u, v in ctx.judgments.items()}

    def _defer(self, q, candidates: list[str], multiple: bool = False) -> list[str]:
        """Second stage: a direct comparison among the remaining candidates."""
        opts = {k: q.options[k] for k in candidates}
        rec = self.text_client.chat(direct_messages(q.stem, q.passage, opts, multiple=multiple), tag=f"DEFER:{q.uid}")
        self._record(rec)
        ans = parse_letters(rec.get("content", ""), candidates)
        ans = ans if multiple else ans[:1]
        return ans or [min(candidates, key=lambda c: tiebreak(q.uid, c))]

    def decide(self, q, ctx: Context) -> tuple[list[str], str, dict]:
        reg = self.regions(ctx)
        target, opposite = ("NEG", "POS") if ctx.polarity == "negative" else ("POS", "NEG")
        sign = 1 if target == "POS" else -1
        bal = {u: sign * (v.count("+") - v.count("-")) for u, v in ctx.judgments.items()}

        if q.is_statement_combo:
            val = {u: 1 if reg[u] == target else -1 if reg[u] == opposite else 0 for u in reg}

            def agree(option):
                chosen = set(q.option_statements[option])
                return sum(val[s] if s in chosen else -val[s] for s in val)

            best = max(agree(o) for o in q.options)
            cands = sorted(o for o in q.options if agree(o) == best)
            if len(cands) == 1 or not self.params.defer:
                return [min(cands, key=lambda c: tiebreak(q.uid, c))], "context", reg
            return self._defer(q, cands), "deferred", reg

        if q.qtype == "multiple":
            k = self.params.min_multi
            ans = [u for u in reg if reg[u] == target]
            stage = "context"
            if len(ans) < k:
                open_ = [u for u in reg if reg[u] != opposite]
                if len(open_) == k:
                    ans = open_
                elif self.params.defer:
                    cands = open_ if len(open_) > k else list(reg)
                    ans = sorted(set(ans) | set(self._defer(q, cands, multiple=True)))
                    stage = "deferred"
            for u in sorted(reg, key=lambda u: (-bal[u], reg[u] == opposite, tiebreak(q.uid, u))):
                if len(ans) >= k:
                    break
                if u not in ans:
                    ans.append(u)
            return sorted(ans), stage, reg

        for region in (target, "BND"):
            cands = [u for u in reg if reg[u] == region]
            if len(cands) == 1:
                return cands, "context", reg
            if cands:
                break
        else:
            cands = list(reg)
        top = max(bal[u] for u in cands)
        cands = sorted(u for u in cands if bal[u] == top)
        if len(cands) == 1 or not self.params.defer:
            return [min(cands, key=lambda c: tiebreak(q.uid, c))], "context", reg
        return self._defer(q, cands), "deferred", reg

    @staticmethod
    def explain(ctx: Context, answer: list[str]) -> dict:
        """The three-way concept generated by the answer: (X, (X*, X^-*))."""
        K = ctx.formal()
        X = K.obj_mask([a for a in answer if a in ctx.judgments])
        return {"extent": sorted(K.objs(X)), "jointly_satisfied": sorted(K.attrs(K.up(X))),
                "jointly_violated": sorted(K.attrs(K.neg_up(X)))}

    def solve(self, q) -> Result:
        self.usage = [0, 0, 0]
        kb, linker, added = self.dictionary(q)
        ctx = self.elicit(q, kb, linker)
        answer, stage, reg = self.decide(q, ctx)
        gold, answer = sorted(q.answer), sorted(answer)
        union = set(answer) | set(gold)
        return Result(
            uid=q.uid, answer=answer, gold=gold, correct=answer == gold,
            jaccard=len(set(answer) & set(gold)) / len(union) if union else 0.0,
            stage=stage, regions=reg,
            context={"conditions": ctx.conditions, "polarity": ctx.polarity, "judgments": ctx.judgments},
            explanation=self.explain(ctx, answer) if not q.is_statement_combo else {},
            dcr_terms=added, calls=self.usage[0], prompt_tokens=self.usage[1], completion_tokens=self.usage[2],
        )
