"""Curriculum-grounded geography dictionary and its concept graph.

An entry describes one geographic term with typed relations. Relation values are
short phrases; when a phrase matches the surface form of another entry it is
linked to that entry, so the dictionary doubles as a concept graph whose nodes
are canonical terms or (unlinked) normalized phrases.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Fixed category system aligned with the modules of the curriculum standard.
TAXONOMY = {
    "EARTH": ("宇宙与地球运动", "Earth and universe"),
    "GEOMORPH": ("地质与地貌", "Geology and landforms"),
    "ATMOS": ("大气与气候", "Atmosphere and climate"),
    "WATER": ("水体与海洋", "Hydrology and oceans"),
    "SOILVEG": ("土壤与植被", "Soil and vegetation"),
    "ENVSYS": ("自然环境的整体性与差异性", "Integrity and zonation of the environment"),
    "HAZARD": ("自然灾害", "Natural hazards"),
    "POPCITY": ("人口与城乡", "Population and settlements"),
    "INDUSTRY": ("产业区位与交通", "Industry, agriculture and transport"),
    "REGION": ("区域发展", "Regional development"),
    "RESENV": ("资源、环境与可持续发展", "Resources, environment and sustainability"),
    "PLACE": ("地名与区域", "Places and regions"),
    "TECH": ("地理信息技术与方法", "Geospatial technology and methods"),
    "OTHER": ("其他", "Other"),
}
CATEGORY_BY_NAME = {zh: cid for cid, (zh, _) in TAXONOMY.items()}

RELATIONS = ("is_a", "subtypes", "features", "causes", "effects", "locations")

MAX_PHRASE_LEN = 12

_PUNCT = re.compile(r"[\s\"'“”‘’《》「」\[\]【】]")


def norm(text: str) -> str:
    text = _PUNCT.sub("", str(text or ""))
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\(.*?\)", "", text)
    return text.strip("，。；、,.;:：的")


@dataclass
class Entry:
    term: str
    category: str = "OTHER"
    aliases: list[str] = field(default_factory=list)
    is_a: list[str] = field(default_factory=list)
    subtypes: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    origin: list[str] = field(default_factory=list)  # curriculum item ids, "expansion", "dcr"

    @classmethod
    def from_llm(cls, raw: dict, origin: list[str]) -> "Entry | None":
        """Validate and normalize one LLM-produced entry; None if unusable."""
        term = norm(raw.get("term", ""))
        if not (2 <= len(term) <= MAX_PHRASE_LEN):
            return None
        cat = str(raw.get("category", "")).strip()
        cat = cat if cat in TAXONOMY else CATEGORY_BY_NAME.get(cat, "OTHER")

        def phrases(key: str, limit: int) -> list[str]:
            vals = raw.get(key) or []
            if isinstance(vals, str):
                vals = re.split(r"[、；;，,]", vals)
            out = []
            for v in vals:
                p = norm(v)
                if 2 <= len(p) <= MAX_PHRASE_LEN and p != term and p not in out:
                    out.append(p)
            return out[:limit]

        return cls(
            term=term,
            category=cat,
            aliases=phrases("aliases", 5),
            is_a=phrases("is_a", 3),
            subtypes=phrases("subtypes", 8),
            features=phrases("features", 8),
            causes=phrases("causes", 6),
            effects=phrases("effects", 6),
            locations=phrases("locations", 6),
            origin=list(origin),
        )

    def relation_values(self, relations=RELATIONS) -> list[tuple[str, str]]:
        return [(r, v) for r in relations for v in getattr(self, r)]


class KnowledgeBase:
    def __init__(self, entries: list[Entry]):
        self.entries: dict[str, Entry] = {}
        for e in entries:
            self.add(e)
        self._index()

    # ----------------------------------------------------------------- build
    def add(self, e: Entry) -> None:
        if e.term in self.entries:
            old = self.entries[e.term]
            for r in ("aliases",) + RELATIONS:
                merged = getattr(old, r) + [v for v in getattr(e, r) if v not in getattr(old, r)]
                setattr(old, r, merged)
            old.origin = sorted(set(old.origin) | set(e.origin))
        else:
            self.entries[e.term] = e

    def _index(self) -> None:
        self.surface: dict[str, str] = {}
        for term in self.entries:
            self.surface[term] = term
        for term, e in self.entries.items():
            for a in e.aliases:
                # An alias never overrides a canonical term of its own.
                self.surface.setdefault(a, term)
        # Node degree over the concept graph, used to down-weight hub nodes.
        deg = Counter()
        for term, e in self.entries.items():
            for _, v in e.relation_values():
                deg[self.canon(v)] += 1
        self.degree = deg
        self.n_entries = max(len(self.entries), 1)

    def canon(self, phrase: str) -> str:
        p = norm(phrase)
        return self.surface.get(p, p)

    def idf(self, node: str) -> float:
        return math.log((1 + self.n_entries) / (1 + self.degree.get(node, 0))) + 1.0

    # ----------------------------------------------------------------- query
    def get(self, phrase: str) -> Entry | None:
        return self.entries.get(self.canon(phrase))

    def contained_terms(self, phrase: str) -> set[str]:
        """Dictionary terms occurring inside a relation phrase ("淋溶作用强" -> {"淋溶作用"})."""
        p = norm(phrase)
        cache = self.__dict__.setdefault("_contained", {})
        if p not in cache:
            found = set()
            for i in range(len(p)):
                for j in range(i + 2, len(p) + 1):
                    if j - i < len(p) and p[i:j] in self.surface:
                        found.add(self.surface[p[i:j]])
            cache[p] = found
        return cache[p]

    def neighbors(self, term: str, relations=RELATIONS) -> set[str]:
        """Nodes adjacent to a term: canonical relation values plus the terms they contain."""
        e = self.get(term)
        if e is None:
            return set()
        out = set()
        for _, v in e.relation_values(relations):
            out.add(self.canon(v))
            out |= self.contained_terms(v)
        out.discard(e.term)
        return out

    def closed_neighbors(self, term: str, relations=RELATIONS) -> set[str]:
        return {self.canon(term)} | self.neighbors(term, relations)

    def surfaces(self) -> list[str]:
        return list(self.surface)

    def category_counts(self) -> Counter:
        return Counter(e.category for e in self.entries.values())

    # -------------------------------------------------------------------- io
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = (json.dumps(asdict(e), ensure_ascii=False) for e in sorted(self.entries.values(), key=lambda e: e.term))
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeBase":
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls([Entry(**json.loads(ln)) for ln in lines if ln.strip()])

    def with_entries(self, extra: list[Entry]) -> "KnowledgeBase":
        """Question-local dictionary D_q^+ = D ∪ extra as a read-only overlay on D."""
        return OverlayKB(self, extra) if extra else self


class OverlayKB(KnowledgeBase):
    """D_q^+: new entries layered over a base dictionary that is never modified."""

    def __init__(self, base: KnowledgeBase, extra: list[Entry]):
        self.base = base
        self.extra = {}
        for e in extra:
            if e.term not in base.entries and e.term not in self.extra:
                self.extra[e.term] = e
        self.entries = _ChainMap(self.extra, base.entries)
        self.surface = _ChainMap({}, base.surface)
        for t in self.extra:
            self.surface.first.setdefault(t, t)
        for t, e in self.extra.items():
            for a in e.aliases:
                if a not in self.surface:
                    self.surface.first[a] = t
        deg = Counter()
        for e in self.extra.values():
            for _, v in e.relation_values():
                deg[self.canon(v)] += 1
        self.degree = _CounterSum(base.degree, deg)
        self.n_entries = base.n_entries + len(self.extra)


class _ChainMap(dict):
    """Minimal read-mostly chain of two dicts (writes go to the first)."""

    def __init__(self, first: dict, second: dict):
        super().__init__()
        self.first, self.second = first, second

    def __contains__(self, k):
        return k in self.first or k in self.second

    def __getitem__(self, k):
        return self.first[k] if k in self.first else self.second[k]

    def get(self, k, default=None):
        return self.first.get(k, self.second.get(k, default))

    def __iter__(self):
        yield from self.first
        yield from (k for k in self.second if k not in self.first)

    def __len__(self):
        return len(self.first) + sum(1 for k in self.second if k not in self.first)

    def keys(self):
        return list(iter(self))

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]


class _CounterSum:
    def __init__(self, a: Counter, b: Counter):
        self.a, self.b = a, b

    def get(self, k, default=0):
        return self.a.get(k, 0) + self.b.get(k, 0) or default

    def __getitem__(self, k):
        return self.get(k, 0)
