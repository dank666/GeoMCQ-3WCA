"""Direct answering with the prompt of Appendix B.

  python experiments/run_direct.py --split test --model THUDM/GLM-4-9B-0414
  python experiments/run_direct.py --split test --model THUDM/GLM-4-9B-0414 --at-least-two

--at-least-two tells the model that a multiple-choice item has at least two correct
options, the same type constraint the three-way decision uses.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, client, load_split
from geoqa.prompts import direct_messages
from geoqa.twc import parse_letters

AT_LEAST_TWO = ("1. One or more options may be correct.", "1. At least two options are correct.")


def messages(q, at_least_two: bool) -> list[dict]:
    msgs = direct_messages(q.stem, q.passage, q.options, multiple=q.qtype == "multiple")
    if at_least_two and q.qtype == "multiple":
        msgs[1]["content"] = msgs[1]["content"].replace(*AT_LEAST_TWO)
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "gaokao"], default="dev")
    ap.add_argument("--model", required=True)
    ap.add_argument("--at-least-two", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()

    c = client(args.model, False, 16)
    questions = load_split(args.split)
    if args.at_least_two:
        questions = [q for q in questions if q.qtype == "multiple"]

    def ask(q):
        rec = c.chat(messages(q, args.at_least_two), tag=f"DIRECT:{q.uid}")
        pred = parse_letters(rec.get("content", ""), sorted(q.options))
        if q.qtype == "single":
            pred = pred[:1]
        u = rec.get("usage") or {}
        return {"uid": q.uid, "qtype": q.qtype, "domain": q.domain, "gold": sorted(q.answer), "pred": pred,
                "correct": pred == sorted(q.answer), "raw": rec.get("content", ""),
                "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0)}

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(ask, questions))
    name = args.model.split("/")[-1] + ("_k2" if args.at_least_two else "")
    out = ROOT / "results" / "direct" / f"{args.split}_{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    for t in ("single", "multiple"):
        R = [r for r in rows if r["qtype"] == t]
        if R:
            print(f"{t:8s} acc={100 * sum(r['correct'] for r in R) / len(R):6.2f}  n={len(R)}")


if __name__ == "__main__":
    main()
