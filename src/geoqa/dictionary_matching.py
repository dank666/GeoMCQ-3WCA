"""Dictionary matching: a rule-based pipeline in the tradition of FCA-based question answering, used as a baseline.

The concepts of the stem are linked to the options through the curriculum
dictionary; objects are the stem concepts, conditional attributes their
dictionary features, and decision attributes the composite answers (all non-empty
option combinations). Rules between three-way merged concepts are scored and
filtered, and the answer is the composite attribute
with the largest rule vote. No LLM is called at inference.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from itertools import combinations

from .fca import FormalContext, MergedConcept
from .grounding import Linker, is_negative_stem, textrank
from .kb import KnowledgeBase, norm

@dataclass
class Params:
    theta: float = 0.25  # minimum support(g, o) for object g to support option o
    max_objects: int = 16
    base_threshold: float = 0.3  # Stage 1: minimum rule_score of a candidate rule
    premise_match: float = 0.6  # Stage 2: w_o threshold of the premise matching set P
    pct_conf: float = 0.8  # percentile thresholds of the adaptive filter
    pct_supp: float = 0.8
    pct_strength: float = 0.9
    relax: float = 0.7  # relaxation factor when fewer than min_rules rules pass
    min_rules: int = 3
    min_multi: int = 2  # a multiple-choice answer has at least this many options


@dataclass
class Rule:
    premise: frozenset  # A: conditional attributes
    conclusion: frozenset  # C: decision attributes (composite answers)
    objects: frozenset  # X ∩ Z
    incl: float
    sim: float
    rule_score: float
    conf: float = 0.0
    supp: float = 0.0
    lift: float = 0.0
    strength: float = 0.0


@dataclass
class Structure:
    objects: list[str]  # G
    keywords: list[str]  # K(q)
    units: dict[str, str]  # option (or statement) label -> text
    unit_concepts: dict[str, list[str]]
    negative: bool


@dataclass
class Answer:
    answer: list[str]
    votes: dict[str, float]
    rules: list[Rule] = field(default_factory=list)
    relaxed: bool = False
    fallback: bool = False
    n_candidates: int = 0
    contexts: tuple = ()


# ------------------------------------------------------------ question structuring


def keywords(text: str, linker: Linker) -> list[str]:
    """K(q): TextRank terms above the mean score together with the dictionary matches."""
    tr = textrank(text)
    mean = sum(tr.values()) / len(tr) if tr else 0.0
    salient = [w for w, s in sorted(tr.items(), key=lambda x: -x[1]) if s > mean]
    linked, _ = linker.link(text)
    return list(dict.fromkeys(salient + linked))


def structure(q, kb: KnowledgeBase, linker: Linker, params: Params) -> Structure:
    stem = q.stem if not q.is_statement_combo else re.split(r"[①②③④⑤⑥]", q.stem)[0]
    text = f"{q.passage}\n{stem}" if q.passage else stem
    tr = textrank(text)
    linked, _ = linker.link(text)
    # Objects G: dictionary concepts of the stem, most salient first.
    order = {c: i for i, c in enumerate(linked)}
    sal = {c: max([tr.get(s, 0.0) for s, t in kb.surface.items() if t == c and s in tr] or [0.0]) for c in linked}
    objects = sorted(linked, key=lambda c: (-sal[c], order[c]))[: params.max_objects]
    units = dict(q.statements) if q.is_statement_combo else dict(q.options)
    unit_concepts = {}
    for label, t in units.items():
        cs, _ = linker.link(t)
        unit_concepts[label] = cs or [norm(t)]
    return Structure(objects, keywords(text, linker), units, unit_concepts, is_negative_stem(stem))


# ------------------------------------------------- support of an option by a concept


def _phrase_match(node: str, phrase: str) -> bool:
    return node == phrase or (len(node) >= 2 and node in phrase) or (len(phrase) >= 2 and phrase in node)


def two_hop(kb: KnowledgeBase, g: str) -> set[str]:
    out = set()
    for m in kb.neighbors(g):
        if m in kb.entries:
            out |= kb.neighbors(m)
    return out


def support(kb: KnowledgeBase, g: str, option_concepts: list[str], hop2: float = 0.5) -> float:
    """Strength in [0, 1] with which evidence concept g supports an option.

    1.0 for a direct edge between g and an option concept, `hop2` for a path of
    length two, and otherwise the IDF-weighted overlap of their neighbourhoods.
    Option phrases that are not dictionary terms are matched against knowledge
    nodes by substring containment.
    """
    Ng = kb.closed_neighbors(g)
    N2g = None
    best = 0.0
    for c in option_concepts:
        if c in kb.entries:
            if c == g or c in Ng or g in kb.neighbors(c):
                return 1.0
            N2g = two_hop(kb, g) if N2g is None else N2g
            if c in N2g or g in two_hop(kb, c) or any(m in kb.entries for m in kb.neighbors(c) & Ng):
                best = max(best, hop2)
            Nc = kb.closed_neighbors(c)
            shared = (Ng & Nc) - {g, c}
            if shared:
                num = sum(kb.idf(x) for x in shared)
                den = min(sum(kb.idf(x) for x in Ng), sum(kb.idf(x) for x in Nc))
                best = max(best, hop2 * num / den if den else 0.0)
        else:
            if any(_phrase_match(x, c) for x in Ng):
                return 1.0
            N2g = two_hop(kb, g) if N2g is None else N2g
            if any(_phrase_match(x, c) for x in N2g):
                best = max(best, hop2)
    return best


# ---------------------------------------------------------------------- contexts


def expansion(kb: KnowledgeBase, m: str) -> set[str]:
    """Γ(m): m with its synonyms, hyponyms, hypernyms, category, components and spatial relations."""
    out = {m}
    e = kb.get(m)
    if e is not None:
        out |= {kb.canon(x) for x in e.aliases + e.subtypes + e.is_a + e.features + e.locations}
        out.add(f"category:{e.category}")
    return out


def concepts_of(kb: KnowledgeBase, g: str) -> set[str]:
    """Ĉ(g): the normalized concept g with its synonyms, hypernyms and category."""
    out = {kb.canon(g)}
    e = kb.get(g)
    if e is not None:
        out |= {kb.canon(x) for x in e.aliases + e.is_a}
        out.add(f"category:{e.category}")
    return out


def conditional_context(kb: KnowledgeBase, G: list[str], unit_nodes: set[str]) -> FormalContext:
    """(G, M, I) with Match_M(g, m) = 1 iff Ĉ(g) ∩ Γ(m) ≠ ∅."""
    cand = set()
    for g in G:
        e = kb.get(g)
        if e is not None:
            cand |= {kb.canon(x) for x in e.is_a + e.features + e.locations}
            cand.add(f"category:{e.category}")
    C = {g: concepts_of(kb, g) for g in G}
    gamma = {m: expansion(kb, m) for m in cand}
    inc = [(g, m) for g in G for m in cand if C[g] & gamma[m]]
    count = {}
    for _, m in inc:
        count[m] = count.get(m, 0) + 1
    # Attributes shared by at least two objects or bridging to an option.
    M = sorted(m for m in cand if count.get(m, 0) >= 2 or (m in unit_nodes and count.get(m, 0) >= 1))
    keep = set(M)
    return FormalContext(G, M, [(g, m) for g, m in inc if m in keep])


def label(n: frozenset) -> str:
    return "+".join(sorted(n))


def decision_attributes(q, units: dict[str, str]) -> dict[str, frozenset]:
    """N: composite answers. Single-choice: the options; multiple-choice: all 2^n - 1 combinations;
    statement-combination items: the combinations offered by the options."""
    if q.is_statement_combo:
        return {o: frozenset(s) for o, s in q.option_statements.items()}
    if q.qtype == "single":
        return {o: frozenset([o]) for o in units}
    labels = sorted(units)
    out = {}
    for k in range(1, len(labels) + 1):
        for c in combinations(labels, k):
            out[label(frozenset(c))] = frozenset(c)
    return out


def supported_units(kb: KnowledgeBase, g: str, s: Structure, params: Params) -> tuple[frozenset, dict]:
    """d(g): the option combination whose every option the knowledge of g supports."""
    sup = {o: support(kb, g, cs) for o, cs in s.unit_concepts.items()}
    return frozenset(o for o, v in sup.items() if v >= params.theta), sup


def decision_context(q, kb, s: Structure, params: Params):
    """(G, N, J) with Match_N(g, n) = 1 iff d(g) = n (the options of q supported by g are exactly n)."""
    N = decision_attributes(q, s.units)
    d, sup = {}, {}
    for g in s.objects:
        d[g], sup[g] = supported_units(kb, g, s, params)
    if q.qtype == "single" and not q.is_statement_combo:
        J = [(g, o) for g in s.objects for o in N if N[o] <= d[g]]
    else:
        J = [(g, n) for g in s.objects for n, units in N.items() if d[g] and d[g] == units]
    return FormalContext(s.objects, list(N), J), N, d, sup


# ------------------------------------------------------------------------- rules


def jaccard(a: frozenset, b: frozenset) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def concept_strength(c: MergedConcept) -> float:
    X, Y, A, B = c.pos_extent, c.neg_extent, c.pos_intent, c.neg_intent
    coverage = len(X) / (len(X) + len(Y) + 1)
    specificity = 1 / (len(A) + len(B) + 1)
    compactness = (bool(X) + bool(Y) + bool(A) + bool(B)) / 4
    return coverage * specificity * compactness


def percentile(values: list[float], p: float) -> float:
    v = sorted(values)
    if not v:
        return 0.0
    k = p * (len(v) - 1)
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def extract_rules(cond: FormalContext, dec: FormalContext, params: Params) -> tuple[list[Rule], bool, int]:
    """Stage 1 (candidate generation) and Stage 2 (statistics and adaptive selection)."""
    cmcs = [c for c in cond.merged_concepts() if c.pos_extent and c.pos_intent]
    dmcs = [d for d in dec.merged_concepts() if d.pos_extent and d.pos_intent]
    best: dict[tuple, Rule] = {}
    for c in cmcs:
        X, Y, A, B = c.pos_extent, c.neg_extent, c.pos_intent, c.neg_intent
        sc = concept_strength(c)
        for d in dmcs:
            Z, W, C, D = d.pos_extent, d.neg_extent, d.pos_intent, d.neg_intent
            if not X & Z:
                continue
            incl = 0.5 * (len(X & Z) / max(len(X), 1) + len(Y & W) / max(len(Y), 1))
            sim = 0.3 * jaccard(X, Z) + 0.3 * jaccard(Y, W) + 0.2 * jaccard(A, C) + 0.2 * jaccard(B, D)
            score = 0.4 * incl + 0.3 * sim + 0.15 * sc + 0.15 * concept_strength(d)
            if score <= params.base_threshold:
                continue
            r = Rule(A, C, X & Z, incl, sim, score)
            key = (r.objects, C)  # semantic redundancy: one rule per supporting objects and conclusion
            if key not in best or r.rule_score > best[key].rule_score or (
                r.rule_score == best[key].rule_score and len(r.premise) < len(best[key].premise)
            ):
                best[key] = r
    rules = list(best.values())
    U = cond.objects
    for r in rules:
        w = {o: len(r.premise & frozenset(a for a in cond.attributes if cond.has(o, a))) / len(r.premise) for o in U}
        P = {o for o in U if w[o] >= params.premise_match}
        Cobj = {o for o in U if all(dec.has(o, n) for n in r.conclusion)}
        I = P & Cobj
        r.conf = 0.5 * (len(I) / len(P) + sum(w[o] for o in I) / len(P)) if P else 0.0
        r.supp = len(I) / len(U) if U else 0.0
        pC = len(Cobj) / len(U) if U else 0.0
        r.lift = r.conf / pC if pC else 0.0
        r.strength = 0.4 * r.rule_score + 0.3 * r.conf + 0.2 * r.supp + 0.1 * min(r.lift, 2.0)
    if not rules:
        return [], False, 0
    tc = percentile([r.conf for r in rules], params.pct_conf)
    ts = percentile([r.supp for r in rules], params.pct_supp)
    tr = percentile([r.strength for r in rules], params.pct_strength)

    def select(f):
        return [r for r in rules if r.conf >= f * tc and r.supp >= f * ts and r.strength >= f * tr]

    R1 = select(1.0)
    relaxed = len(R1) < params.min_rules
    final = select(params.relax) if relaxed else R1
    final.sort(key=lambda r: (-r.strength, sorted(r.conclusion), sorted(r.premise)))
    return final, relaxed, len(rules)


# --------------------------------------------------------------------- answer


def tiebreak(uid: str, key: str) -> int:
    """Deterministic, position-independent tie-breaking."""
    return int(hashlib.md5(f"{uid}:{key}".encode()).hexdigest()[:8], 16)


def answer(q, kb: KnowledgeBase, linker: Linker, params: Params) -> Answer:
    s = structure(q, kb, linker, params)
    unit_nodes = set()
    for cs in s.unit_concepts.values():
        for c in cs:
            unit_nodes |= kb.closed_neighbors(c)
    cond = conditional_context(kb, s.objects, unit_nodes)
    dec, N, d, sup = decision_context(q, kb, s, params)
    rules, relaxed, n_cand = extract_rules(cond, dec, params) if s.objects else ([], False, 0)

    votes = {n: 0.0 for n in N}
    for r in rules:
        for n in r.conclusion:
            votes[n] += r.strength
    fallback = not any(votes.values())
    if fallback:  # no rule: vote by the objects' support of the decision attributes
        for g in s.objects:
            for n in N:
                if dec.has(g, n):
                    votes[n] += 1.0
    allowed = list(N)
    if q.qtype == "multiple" and not q.is_statement_combo:
        allowed = [n for n in N if len(N[n]) >= params.min_multi] or allowed
    sign = -1.0 if (s.negative and (q.qtype == "single" or q.is_statement_combo)) else 1.0
    best = max(allowed, key=lambda n: (sign * votes[n], tiebreak(q.uid, n)))
    if q.is_statement_combo:
        pred = [best]
    else:
        pred = sorted(N[best])
    return Answer(pred, votes, rules, relaxed, fallback, n_cand, (cond, dec))
