"""Build the processed datasets and the GeoMCQ-1200 dev/test split.

Outputs (data/processed/):
  geomcq1200.jsonl      all 1200 GeoMCQ items in the unified schema
  gaokao_geo.jsonl      AGIEval gaokao-geography + non-duplicate GAOKAO-Bench items
  splits.json           GeoMCQ dev/test uids (stratified by domain x type)
  data_report.json      counts used in the paper's dataset table
"""

from __future__ import annotations

import difflib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoqa.data import (  # noqa: E402
    load_agieval,
    load_gaokao_bench,
    load_geomcq,
    normalize,
    permute_options,
    write_jsonl,
)

SEED = 2026
DEV_PER_CELL = 20  # per (domain, type) cell -> 240 dev / 960 test


def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _options_key(q) -> str:
    return normalize("".join(q.options[k] for k in sorted(q.options)))


def find_duplicates(gaokao, agieval) -> dict[str, str]:
    """Map GAOKAO-Bench uid -> AGIEval uid for items present in both."""
    dup = {}
    for q in gaokao:
        best, best_score = None, 0.0
        for p in agieval:
            so = _sim(_options_key(q), _options_key(p))
            if so < 0.85:
                continue
            stem_tail = _sim(normalize(q.stem)[-30:], normalize(p.stem)[-30:])
            head = _sim(normalize(q.full_text)[:60], normalize(p.full_text)[:60])
            score = min(so, max(stem_tail, head))
            if score > best_score:
                best, best_score = p, score
        if best is not None and best_score > 0.6:
            if best.answer != q.answer:
                raise ValueError(f"answer mismatch between duplicates {q.uid} / {best.uid}")
            dup[q.uid] = best.uid
    return dup


def stratified_split(questions):
    cells = defaultdict(list)
    for q in questions:
        cells[(q.domain, q.qtype)].append(q.uid)
    rng = random.Random(SEED)
    dev, test = [], []
    for key in sorted(cells):
        uids = sorted(cells[key])
        rng.shuffle(uids)
        dev += sorted(uids[:DEV_PER_CELL])
        test += sorted(uids[DEV_PER_CELL:])
    return dev, test


def main() -> None:
    raw = ROOT / "data" / "raw"
    out = ROOT / "data" / "processed"

    geomcq = load_geomcq(raw / "geomcq_public")
    agieval = load_agieval(raw / "public" / "agieval_gaokao_geography.jsonl")
    gaokao, failures = load_gaokao_bench(raw / "public" / "gaokao_bench_2010_2022_geography_mcqs.json")
    if failures:
        raise SystemExit(f"unparsed GAOKAO-Bench groups: {failures}")

    dup = find_duplicates(gaokao, agieval)
    public = agieval + [q for q in gaokao if q.uid not in dup]

    # Evaluation uses the position-balanced version; the original order is kept for reference.
    write_jsonl(geomcq, out / "geomcq1200_original_order.jsonl")
    position_prior = {
        "single": dict(Counter(q.answer[0] for q in geomcq if q.qtype == "single")),
        "multiple": dict(Counter("".join(q.answer) for q in geomcq if q.qtype == "multiple")),
    }
    geomcq = [permute_options(q) for q in geomcq]
    write_jsonl(geomcq, out / "geomcq1200.jsonl")
    write_jsonl(public, out / "gaokao_geo.jsonl")

    dev, test = stratified_split(geomcq)
    (out / "splits.json").write_text(
        json.dumps({"seed": SEED, "geomcq_dev": dev, "geomcq_test": test}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )

    def summary(qs):
        return {
            "n": len(qs),
            "by_type": dict(Counter(q.qtype for q in qs)),
            "by_domain": dict(Counter(q.domain for q in qs if q.domain)),
            "answer_size": dict(Counter(len(q.answer) for q in qs)),
            "n_options": dict(Counter(len(q.options) for q in qs)),
            "statement_combo": sum(q.is_statement_combo for q in qs),
            "figure_dependent": sum(q.figure_dependent for q in qs),
            "with_passage": sum(bool(q.passage) for q in qs),
        }

    by_uid = {q.uid: q for q in geomcq}
    report = {
        "geomcq1200": summary(geomcq),
        "geomcq_answer_position_original": position_prior,
        "geomcq_answer_position_balanced": {
            "single": dict(Counter(q.answer[0] for q in geomcq if q.qtype == "single")),
            "multiple": dict(Counter("".join(q.answer) for q in geomcq if q.qtype == "multiple")),
        },
        "geomcq_dev": summary([by_uid[u] for u in dev]),
        "geomcq_test": summary([by_uid[u] for u in test]),
        "gaokao_geo": summary(public),
        "gaokao_geo_sources": {
            "agieval": len(agieval),
            "gaokao_bench_total": len(gaokao),
            "gaokao_bench_duplicates_of_agieval": len(dup),
            "gaokao_bench_kept": len(gaokao) - len(dup),
        },
        "duplicates": dup,
    }
    (out / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "duplicates"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
