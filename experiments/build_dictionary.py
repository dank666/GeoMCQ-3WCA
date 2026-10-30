"""Build the curriculum-grounded geography dictionary.

Stage A  concept extraction: each curriculum content requirement -> core concepts
Stage B  entry generation:   each concept -> typed relations (batched)
Stage C  one expansion round: concept-like relation values that are not yet
         entries (subtypes, is_a, and values used by >= 2 entries) -> entries
Stage D  validation and statistics

Only kb/seeds/curriculum_requirements.jsonl is read. No evaluation question,
option or answer is ever given to the builder.

Usage: python experiments/build_dictionary.py [--max-expansion 3000] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import re  # noqa: E402

import jieba  # noqa: E402
import jieba.posseg as pseg  # noqa: E402

from geoqa.grounding import GENERIC_WORDS, TEMPLATE_WORDS  # noqa: E402
from geoqa.kb import Entry, KnowledgeBase, norm  # noqa: E402
from geoqa.llm import LLMClient, LLMConfig  # noqa: E402
from geoqa.prompts_kb import concept_extraction_messages, entry_messages  # noqa: E402

SEEDS = ROOT / "kb" / "seeds" / "curriculum_requirements.jsonl"
OUT = ROOT / "kb" / "geo_dictionary.jsonl"
STATS = ROOT / "kb" / "geo_dictionary_stats.json"
BATCH = 8

# Methodological words that the concept extractor sometimes returns.
META_WORDS = {"示意图", "图表", "资料", "实例", "案例", "地图", "图像", "视频", "野外观察", "专题地图", "地理工具"}


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i : i + n]


def stage_a(client, requirements, workers):
    def run(req):
        data, _ = client.chat_json(concept_extraction_messages(req["module"], req["text"]), tag=f"A:{req['item_id']}")
        concepts = (data or {}).get("concepts", [])
        return req["item_id"], [c for c in (norm(x) for x in concepts) if 2 <= len(c) <= 10 and c not in META_WORDS]

    origin = {}
    for item_id, concepts in client.map(run, requirements, workers):
        for c in concepts:
            origin.setdefault(c, []).append(item_id)
    return origin


def stage_b(client, origin: dict[str, list[str]], workers, label):
    terms = sorted(origin)

    def run(batch):
        data, _ = client.chat_json(entry_messages(batch), tag=f"B:{label}")
        out = []
        for raw in (data or {}).get("entries", []):
            e = Entry.from_llm(raw, origin.get(norm(raw.get("term", "")), [label]))
            if e is not None:
                out.append(e)
        return out

    entries = [e for chunk in client.map(run, list(batched(terms, BATCH)), workers) for e in chunk]
    missing = set(terms) - {e.term for e in entries}
    return entries, missing


def expansion_candidates(kb: KnowledgeBase, limit: int) -> list[str]:
    counts = Counter()
    for e in kb.entries.values():
        for rel, v in e.relation_values():
            node = kb.canon(v)
            if node in kb.entries or not (2 <= len(node) <= 8):
                continue
            counts[node] += 3 if rel in ("subtypes", "is_a") else 1
    ranked = [t for t, c in counts.most_common() if c >= 2]
    return ranked[:limit]


def atomic_candidates(kb: KnowledgeBase, min_freq: int) -> list[str]:
    """Short nouns occurring inside relation phrases that are not entries yet (Stage C2).

    E.g. "河流含沙量大" yields "含沙量". Mined from the dictionary itself.
    """
    jieba.setLogLevel(60)
    for s in kb.surface:
        jieba.add_word(s, freq=2000)
    counts = Counter()
    for e in kb.entries.values():
        for _, v in e.relation_values():
            for w in pseg.cut(v):
                word = w.word
                if (
                    w.flag in {"n", "vn", "nz", "ns", "nr", "nt", "ng", "nl"}
                    and 2 <= len(word) <= 5
                    and re.fullmatch(r"[\u4e00-\u9fff]+", word)
                    and word not in kb.surface
                    and word not in TEMPLATE_WORDS
                    and word not in GENERIC_WORDS
                ):
                    counts[word] += 1
    return sorted(t for t, c in counts.items() if c >= min_freq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-expansion", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--stage-a-only", action="store_true")
    ap.add_argument("--atomic-min-freq", type=int, default=4, help="Stage C2 threshold; 0 disables C2")
    args = ap.parse_args()

    # DeepSeek-V4.1-Flash with thinking disabled: no reasoning tokens, deterministic decoding.
    config = LLMConfig.from_env(
        "KB_LLM", model=args.model, temperature=0.0, max_tokens=4096, extra_body={"thinking": {"type": "disabled"}}
    )
    client = LLMClient(config, ROOT / "kb" / "cache" / "llm")
    requirements = [json.loads(ln) for ln in SEEDS.read_text(encoding="utf-8").splitlines() if ln.strip()]

    origin = stage_a(client, requirements, args.workers)
    print(f"[A] {len(requirements)} requirements -> {len(origin)} unique concepts")
    print(f"    usage so far: {client.usage.as_dict()}")
    if args.stage_a_only:
        return

    entries, missing = stage_b(client, origin, args.workers, "curriculum")
    kb = KnowledgeBase(entries)
    print(f"[B] {len(kb.entries)} entries ({len(missing)} concepts without a valid entry)")

    cands = expansion_candidates(kb, args.max_expansion)
    exp_entries, _ = stage_b(client, {c: ["expansion"] for c in cands}, args.workers, "expansion")
    for e in exp_entries:
        kb.add(e)
    kb._index()
    print(f"[C] expansion: {len(cands)} candidates -> {len(exp_entries)} entries; total {len(kb.entries)}")

    if args.atomic_min_freq > 0:
        atoms = atomic_candidates(kb, args.atomic_min_freq)
        atom_entries, _ = stage_b(client, {a: ["atomic"] for a in atoms}, args.workers, "atomic")
        for e in atom_entries:
            kb.add(e)
        kb._index()
        print(f"[C2] atomic vocabulary: {len(atoms)} candidates -> {len(atom_entries)} entries; total {len(kb.entries)}")

    kb.save(OUT)
    n_rel = Counter()
    linked = total = 0
    for e in kb.entries.values():
        for rel, v in e.relation_values():
            n_rel[rel] += 1
            total += 1
            linked += kb.canon(v) in kb.entries
    stats = {
        "model_config": config.public(),
        "served_models": sorted({json.loads(p.read_text())["model"] for p in (ROOT / "kb" / "cache" / "llm").rglob("*.json")}),
        "requirements": len(requirements),
        "stage_a_concepts": len(origin),
        "entries": len(kb.entries),
        "entries_from_curriculum": sum(1 for e in kb.entries.values() if any(o not in ("expansion", "atomic") for o in e.origin)),
        "entries_from_expansion": sum(1 for e in kb.entries.values() if e.origin == ["expansion"]),
        "entries_from_atomic": sum(1 for e in kb.entries.values() if e.origin == ["atomic"]),
        "aliases": sum(len(e.aliases) for e in kb.entries.values()),
        "relations": dict(n_rel),
        "relation_values_linked_to_entries": round(linked / max(total, 1), 4),
        "categories": dict(kb.category_counts()),
        "llm_usage": client.usage.as_dict(),
    }
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
