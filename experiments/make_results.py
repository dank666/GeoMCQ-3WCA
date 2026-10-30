"""Collect the test results into results/summary.json and the LaTeX tables of Section 6 and Appendix C.

The figures are drawn by plot_figures.py.

  python experiments/make_results.py
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

from common import ROOT, load_split

MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("Qwen3-8B", "Qwen3-8B"), ("GLM-4-9B-0414", "GLM-4-9B"),
          ("Qwen3.5-122B-A10B", "Qwen3.5-122B")]
DOMAINS = ["CL", "HY", "TG", "SV", "HEG", "GPP"]
TABLES = ROOT / "results" / "tables"


def read(path: Path) -> dict | None:
    if not path.exists():
        return None
    return {r["uid"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    """Exact two-sided McNemar test: (only a correct, only b correct, p)."""
    n01 = sum(x and not y for x, y in zip(a, b))
    n10 = sum(y and not x for x, y in zip(a, b))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(p, 1.0)


def acc(rows, uids) -> float:
    return 100 * sum(rows[u]["correct"] for u in uids) / len(uids) if uids else float("nan")


def jac(rows, uids) -> float:
    def j(r):
        g, p = set(r["gold"]), set(r.get("pred") or r.get("answer") or [])
        return len(g & p) / len(g | p) if g | p else 0.0
    return 100 * sum(j(rows[u]) for u in uids) / len(uids) if uids else float("nan")


def main() -> None:
    summary = {}
    for split in ("test", "gaokao"):
        qs = load_split(split)
        groups = {"all": [q.uid for q in qs], "single": [q.uid for q in qs if q.qtype == "single"],
                  "multiple": [q.uid for q in qs if q.qtype == "multiple"]}
        if split == "gaokao":
            groups["statement_combo"] = [q.uid for q in qs if q.is_statement_combo]
            groups["passage"] = [q.uid for q in qs if q.passage]
            groups["figure"] = [q.uid for q in qs if q.figure_dependent]
            groups["text_only"] = [q.uid for q in qs if not q.figure_dependent]
        else:
            for d in DOMAINS:
                groups[f"domain:{d}"] = [q.uid for q in qs if q.domain == d]
        for name, label in MODELS:
            ours = read(ROOT / "results" / "twc" / f"{split}_{name}_full.jsonl")
            direct = read(ROOT / "results" / "direct" / f"{split}_{name}.jsonl")
            k2 = read(ROOT / "results" / "direct" / f"{split}_{name}_k2.jsonl")
            if ours is None:
                continue
            out = {}
            for g, uids in groups.items():
                row = {"n": len(uids), "ours": acc(ours, uids)}
                if direct is not None:
                    row["direct"] = acc(direct, uids)
                    row["mcnemar_vs_direct"] = mcnemar([ours[u]["correct"] for u in uids],
                                                       [direct[u]["correct"] for u in uids])
                if g == "multiple":
                    row["ours_jaccard"] = jac(ours, uids)
                    if direct is not None:
                        row["direct_jaccard"] = jac(direct, uids)
                    if k2 is not None:
                        row["direct_k2"] = acc(k2, uids)
                        row["direct_k2_jaccard"] = jac(k2, uids)
                        row["mcnemar_vs_k2"] = mcnemar([ours[u]["correct"] for u in uids],
                                                       [k2[u]["correct"] for u in uids])
                ctx = [u for u in uids if ours[u]["stage"] == "context"]
                dfr = [u for u in uids if ours[u]["stage"] == "deferred"]
                row.update(deferred=100 * len(dfr) / len(uids) if uids else 0.0, acc_context=acc(ours, ctx),
                           acc_deferred=acc(ours, dfr))
                if direct is not None:
                    row.update(direct_on_context=acc(direct, ctx), direct_on_deferred=acc(direct, dfr))
                out[g] = row
            n = len(groups["all"])
            out["cost"] = {
                "calls": sum(ours[u]["calls"] for u in groups["all"]) / n,
                "prompt_tokens": sum(ours[u]["prompt_tokens"] for u in groups["all"]) / n,
                "completion_tokens": sum(ours[u]["completion_tokens"] for u in groups["all"]) / n,
                "dcr_terms": sum(len(ours[u]["dcr_terms"]) for u in groups["all"]) / n,
                "direct_prompt_tokens": (sum(direct[u]["prompt_tokens"] for u in groups["all"]) / n) if direct else None,
            }
            summary.setdefault(split, {})[label] = out
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for split, models in summary.items():
        for m, out in models.items():
            a = out["all"]
            print(f"{split:7s} {m:13s} ours {a['ours']:5.1f} direct {a.get('direct', float('nan')):5.1f} "
                  f"deferred {a['deferred']:4.1f}%  " + "  ".join(
                      f"{g} {out[g]['ours']:5.1f}/{out[g].get('direct', float('nan')):5.1f}"
                      for g in ("single", "multiple", "statement_combo") if g in out and out[g]["n"]))




# ------------------------------------------------------------------ LaTeX tables

def fmt(x, digits=1):
    return "--" if x is None or x != x else f"{x:.{digits}f}"


def tables() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    dict_only = {s: read(ROOT / "results" / "dictionary_matching" / f"{s}.jsonl") for s in ("test", "gaokao")}
    T, G = summary.get("test", {}), summary.get("gaokao", {})

    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"& & \multicolumn{3}{c}{GeoMCQ-1200 test} & GaoKao-Geo \\ \cmidrule(lr){3-5}\cmidrule(l){6-6}",
             r"Model & Method & Single & Multiple & Jaccard & Single \\", r"\midrule"]
    t, g = dict_only["test"], dict_only["gaokao"]
    if t and g:
        su = [u for u in t if t[u]["qtype"] == "single"]
        mu = [u for u in t if t[u]["qtype"] == "multiple"]
        lines.append(f"-- & Dictionary matching & {fmt(acc(t, su))} & {fmt(acc(t, mu))} & {fmt(jac(t, mu))} & "
                     f"{fmt(acc(g, list(g)))} \\\\")
        lines.append(r"\midrule")
    for _, label in MODELS:
        if label not in T:
            continue
        s, m = T[label]["single"], T[label]["multiple"]
        gk = G.get(label, {}).get("all", {})
        lines.append(f"\\multirow{{3}}{{*}}{{{label}}} & Direct & {fmt(s.get('direct'))} & "
                     f"{fmt(m.get('direct'))} & {fmt(m.get('direct_jaccard'))} & {fmt(gk.get('direct'))} \\\\")
        lines.append(f" & Direct, $\\geq$2 correct & -- & {fmt(m.get('direct_k2'))} & {fmt(m.get('direct_k2_jaccard'))} & -- \\\\")
        lines.append(f" & Proposed & {fmt(s['ours'])} & {fmt(m['ours'])} & {fmt(m['ours_jaccard'])} & "
                     f"{fmt(gk.get('ours'))} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (TABLES / "tab_main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ablation_tables()


ABLATIONS = [("full", "Full framework"), ("nodcr", "without DCR"), ("noknow", "without the dictionary"),
             ("twovalued", "two-valued context"), ("nodefer", "without deferral"),
             ("min1", "without the type constraint")]
DCR_VARIANTS = [("nodcr", "none", "--"), ("tau0", "GLM-4-9B", "0"), ("full", "GLM-4-9B", "0.5"),
                ("tau08", "GLM-4-9B", "0.8"), ("tau09", "GLM-4-9B", "0.9"), ("dcr122b", "Qwen3.5-122B", "0.5")]
ABL_MODEL = "GLM-4-9B-0414"


def run_rows(split, tag):
    return read(ROOT / "results" / "twc" / f"{split}_{ABL_MODEL}_{tag}.jsonl")


def ablation_tables() -> None:
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"& \multicolumn{2}{c}{GeoMCQ-1200} & GaoKao-Geo & Deferred \\ \cmidrule(lr){2-3}\cmidrule(lr){4-4}",
             r"Variant & Single & Multiple & & (\%) \\", r"\midrule"]
    for tag, name in ABLATIONS:
        t, g = run_rows("test", tag), run_rows("gaokao", tag)
        if t is None:
            continue
        su = [u for u in t if t[u]["qtype"] == "single"]
        mu = [u for u in t if t[u]["qtype"] == "multiple"]
        dfr = 100 * sum(r["stage"] == "deferred" for r in t.values()) / len(t)
        lines.append(f"{name} & {fmt(acc(t, su))} & {fmt(acc(t, mu))} & {fmt(acc(g, list(g)) if g else None)} & "
                     f"{fmt(dfr)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             r"& & \multicolumn{2}{c}{Entries per question} & \multicolumn{2}{c}{Accuracy (\%)} \\"
             r" \cmidrule(lr){3-4}\cmidrule(l){5-6}",
             r"DCR model & $\tau$ & GeoMCQ & GaoKao & GeoMCQ & GaoKao \\", r"\midrule"]
    for tag, model, tau in DCR_VARIANTS:
        t, g = run_rows("test", tag), run_rows("gaokao", tag)
        if t is None or g is None:
            continue
        e = lambda rows: sum(len(r["dcr_terms"]) for r in rows.values()) / len(rows)  # noqa: E731
        lines.append(f"{model} & {tau} & {fmt(e(t), 2)} & {fmt(e(g), 2)} & {fmt(acc(t, list(t)))} & "
                     f"{fmt(acc(g, list(g)))} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "tab_dcr.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
    tables()
