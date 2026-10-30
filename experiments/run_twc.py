"""Run the three-way concept method on a split.

  python experiments/run_twc.py --split test --model THUDM/GLM-4-9B-0414 --dcr THUDM/GLM-4-9B-0414 --tag full
  python experiments/run_twc.py --split test --model THUDM/GLM-4-9B-0414 --set knowledge=0 --tag noknow

Writes results/twc/<split>_<model>_<tag>.jsonl (one line per question) and a summary.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path

from common import DICTIONARY, ROOT, client, load_split
from geoqa.dcr import make_dcr
from geoqa.kb import KnowledgeBase
from geoqa.twc import Params, Solver


def parse_overrides(items) -> dict:
    types = {f.name: f.type for f in fields(Params)}
    out = {}
    for item in items or []:
        k, v = item.split("=", 1)
        t = types[k]
        out[k] = bool(int(v)) if t in (bool, "bool") else int(v) if t in (int, "int") else float(v)
    return out


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(lambda: [0, 0, 0.0])
    for r in rows:
        keys = ["all", f"type:{r['qtype']}"] + ([f"domain:{r['domain']}"] if r["domain"] else [])
        if r["statement_combo"]:
            keys.append("format:statement_combo")
        for k in keys:
            by[k][0] += r["correct"]
            by[k][1] += 1
            by[k][2] += r["jaccard"]
    out = {k: {"acc": round(100 * c / n, 2), "jaccard": round(100 * j / n, 2), "n": n} for k, (c, n, j) in sorted(by.items())}
    n = len(rows)
    ctx = [r for r in rows if r["stage"] == "context"]
    out["_diagnostics"] = {
        "deferred_rate": round(1 - len(ctx) / n, 4),
        "acc_context": round(100 * sum(r["correct"] for r in ctx) / max(len(ctx), 1), 2),
        "mean_calls": round(sum(r["calls"] for r in rows) / n, 3),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / n, 1),
        "mean_completion_tokens": round(sum(r["completion_tokens"] for r in rows) / n, 1),
        "mean_conditions": round(sum(len(r["context"]["conditions"]) for r in rows) / n, 2),
        "undetermined_rate": round(sum(v.count("?") for r in rows for v in r["context"]["judgments"].values())
                                   / max(sum(len(v) for r in rows for v in r["context"]["judgments"].values()), 1), 4),
        "mean_dcr_terms": round(sum(len(r["dcr_terms"]) for r in rows) / n, 3),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "gaokao"], default="dev")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dcr", default=None, help="model of dynamic concept retrieval; omit to disable")
    ap.add_argument("--tau", type=float, default=0.5, help="confidence threshold of DCR")
    ap.add_argument("--set", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    kb = KnowledgeBase.load(DICTIONARY)
    params = Params(**parse_overrides(args.set))
    jc, tc = client(args.model, True, 1024), client(args.model, False, 16)
    dcr = make_dcr(args.dcr, ROOT, tau=args.tau) if args.dcr else None
    questions = load_split(args.split)
    if dcr is not None:
        dcr.prefetch(questions, kb, args.workers)
    local = threading.local()

    def solve(q):
        if not hasattr(local, "solver"):
            local.solver = Solver(jc, tc, kb, params, dcr)
        return local.solver.solve(q)

    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(solve, questions))
    rows = []
    for q, r in zip(questions, results):
        row = asdict(r)
        row.update(qtype=q.qtype, domain=q.domain, statement_combo=q.is_statement_combo)
        rows.append(row)
    name = args.model.split("/")[-1] + (f"_{args.tag}" if args.tag else "")
    out = ROOT / "results" / "twc" / f"{args.split}_{name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    summary = {"split": args.split, "model": args.model, "params": asdict(params),
               "dcr": None if dcr is None else dcr.describe(), "results": summarize(rows)}
    Path(f"{out}.summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    res = summary["results"]
    for k in sorted(res):
        if not k.startswith("_") and not k.startswith("domain"):
            print(f"{k:24s} acc={res[k]['acc']:6.2f}  jac={res[k]['jaccard']:6.2f}  n={res[k]['n']}")
    print(res["_diagnostics"])


if __name__ == "__main__":
    main()
