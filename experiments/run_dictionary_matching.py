"""Dictionary matching baseline: a rule-based pipeline that calls no LLM at inference (Table 2).

  python experiments/run_dictionary_matching.py --split test
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path

from common import DICTIONARY, ROOT, load_split
from geoqa.dictionary_matching import Params, answer
from geoqa.grounding import Linker
from geoqa.kb import KnowledgeBase


def parse_overrides(items) -> dict:
    types = {f.name: f.type for f in fields(Params)}
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        out[k] = int(v) if types[k] in (int, "int") else float(v)
    return out


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        keys = ["all", f"type:{r['qtype']}"]
        if r["domain"]:
            keys += [f"domain:{r['domain']}", f"{r['qtype']}:{r['domain']}"]
        if r["statement_combo"]:
            keys.append("format:statement_combo")
        for k in keys:
            by[k][0] += r["correct"]
            by[k][1] += 1
            by[k][2] += r["jaccard"]
    out = {k: {"acc": round(100 * c / n, 2), "jaccard": round(100 * j / n, 2), "n": n} for k, (c, n, j) in sorted(by.items())}
    n = len(rows)
    out["_diagnostics"] = {
        "fallback_rate": round(sum(r["fallback"] for r in rows) / n, 4),
        "relaxed_rate": round(sum(r["relaxed"] for r in rows) / n, 4),
        "mean_rules": round(sum(r["n_rules"] for r in rows) / n, 2),
        "mean_objects": round(sum(len(r["objects"]) for r in rows) / n, 2),
    }
    return out


def run(questions, kb, params) -> list[dict]:
    rows = []
    linker = Linker(kb)
    for q in questions:
        a = answer(q, kb, linker, params)
        gold, pred = sorted(q.answer), sorted(a.answer)
        union = set(gold) | set(pred)
        rows.append({
            "uid": q.uid, "qtype": q.qtype, "domain": q.domain, "statement_combo": q.is_statement_combo,
            "gold": gold, "pred": pred, "correct": gold == pred,
            "jaccard": len(set(gold) & set(pred)) / len(union) if union else 0.0,
            "votes": {k: round(v, 4) for k, v in a.votes.items()},
            "objects": a.contexts[0].objects, "n_attributes": len(a.contexts[0].attributes),
            "n_candidates": a.n_candidates, "n_rules": len(a.rules), "relaxed": a.relaxed, "fallback": a.fallback,
            "rules": [{"premise": sorted(r.premise), "conclusion": sorted(r.conclusion), "objects": sorted(r.objects),
                       "conf": round(r.conf, 3), "supp": round(r.supp, 3), "lift": round(r.lift, 3),
                       "strength": round(r.strength, 4)} for r in a.rules[:5]],
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "gaokao"], default="dev")
    ap.add_argument("--set", nargs="*", help="parameter overrides, e.g. theta=0.3")
    args = ap.parse_args()

    kb = KnowledgeBase.load(DICTIONARY)
    params = Params(**parse_overrides(args.set))
    rows = run(load_split(args.split), kb, params)
    out = ROOT / "results" / "dictionary_matching" / args.split
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    summary = {"split": args.split, "params": asdict(params), "dictionary_entries": len(kb.entries),
               "results": summarize(rows)}
    Path(f"{out}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    res = summary["results"]
    for k in sorted(res):
        if not k.startswith("_"):
            print(f"{k:24s} acc={res[k]['acc']:6.2f}  jac={res[k]['jaccard']:6.2f}  n={res[k]['n']}")
    print(res["_diagnostics"])


if __name__ == "__main__":
    main()
