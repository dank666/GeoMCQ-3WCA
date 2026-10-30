"""Unified question schema and loaders for GeoMCQ-1200, AGIEval and GAOKAO-Bench."""

from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

CIRCLED = "①②③④⑤⑥⑦⑧"
FIGURE_CUES = re.compile(
    r"如图|读图|下图|上图|图中|图示|示意图|据图|图例|右图|左图|图[甲乙丙丁]|图\s*\d|"
    r"下表|表中|据表|上表|如表|表[一二三]|表\s*\d|统计图|等高线图|等值线图"
)

GEOMCQ_DOMAINS = {
    "Climatology": "CL",
    "Geographical Processes and Principles": "GPP",
    "Human and Economic Geography": "HEG",
    "Hydrology": "HY",
    "Soil and Vegetation": "SV",
    "Topography and Geomorphology": "TG",
}


@dataclass
class Question:
    uid: str
    dataset: str  # geomcq1200 | agieval | gaokao_bench
    qtype: str  # single | multiple
    stem: str
    options: dict[str, str]
    answer: list[str]
    domain: str | None = None
    passage: str | None = None
    # Statement-combination items: statements keyed by circled numeral and
    # the statement subset each option denotes.
    statements: dict[str, str] = field(default_factory=dict)
    option_statements: dict[str, list[str]] = field(default_factory=dict)
    figure_dependent: bool = False
    source: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_statement_combo(self) -> bool:
        return bool(self.option_statements)

    @property
    def full_text(self) -> str:
        return f"{self.passage}\n{self.stem}" if self.passage else self.stem

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(**d)


def normalize(text: str) -> str:
    text = text.replace("．", ".").replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    return text


def _mentions_figure(text: str) -> bool:
    return bool(FIGURE_CUES.search(text))


def parse_statements(stem: str, options: dict[str, str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Detect ①②③④-style statement-combination items.

    Returns (statements, option_statements); both empty when the item is not a
    statement-combination item.
    """
    option_sets = {}
    for label, text in options.items():
        marks = [c for c in text if c in CIRCLED]
        residue = re.sub(rf"[{CIRCLED}\s、,，和与]", "", text)
        if not marks or residue:
            return {}, {}
        option_sets[label] = marks
    parts = re.split(rf"([{CIRCLED}])", stem)
    statements = {}
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1].strip(" \n\t；;。.,，")
        if body:
            statements[parts[i]] = body
    used = {m for marks in option_sets.values() for m in marks}
    if not used or not used <= set(statements):
        return {}, {}
    return statements, option_sets


# --------------------------------------------------------------------- GeoMCQ


def load_geomcq(root: Path) -> list[Question]:
    questions = []
    for qtype in ("single", "multiple"):
        for full_name, abbr in GEOMCQ_DOMAINS.items():
            path = root / qtype / f"{full_name}.json"
            items = json.loads(path.read_text(encoding="utf-8"))["geography_questions"]
            for item in items:
                options = {k: v.strip() for k, v in item["options"].items()}
                stem = item["question"].strip()
                statements, option_statements = parse_statements(stem, options)
                questions.append(
                    Question(
                        uid=f"geomcq-{qtype}-{abbr}-{int(item['id']):03d}",
                        dataset="geomcq1200",
                        qtype=qtype,
                        stem=stem,
                        options=options,
                        answer=sorted(str(a).strip() for a in item["correct_answer"]),
                        domain=abbr,
                        statements=statements,
                        option_statements=option_statements,
                        figure_dependent=_mentions_figure(stem),
                    )
                )
    return questions


# -------------------------------------------------------------------- AGIEval


def load_agieval(path: Path) -> list[Question]:
    questions = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        item = json.loads(line)
        options = {}
        for opt in item["options"]:
            m = re.match(r"\s*\(([A-D])\)\s*(.*)", opt, re.S) or re.match(r"\s*([A-D])\s*[.．]\s*(.*)", opt, re.S)
            options[m.group(1)] = m.group(2).strip()
        stem = item["question"].strip()
        passage = (item.get("passage") or "").strip() or None
        statements, option_statements = parse_statements(stem, options)
        questions.append(
            Question(
                uid=f"agieval-{i:03d}",
                dataset="agieval",
                qtype="single",
                stem=stem,
                options=options,
                answer=[item["label"].strip()],
                passage=passage,
                statements=statements,
                option_statements=option_statements,
                figure_dependent=_mentions_figure((passage or "") + stem),
                source=item.get("other", {}).get("source"),
            )
        )
    return questions


# --------------------------------------------------------------- GAOKAO-Bench

# Sub-question markers such as "9．" or "1．（4 分）". OCR sometimes glues the
# marker to its neighbours ("提高效 益3. 无人机", "1．2010～2020 年"), so the
# pattern is permissive; consecutive numbering and option parsing reject
# false matches such as decimals in tables.
_SUBQ = re.compile(r"(?<!\d)(\d{1,2})\s*[．.]\s*(?:[（(]\s*\d+\s*分\s*[）)])?")
_CJK_GAP = re.compile(r"(?<=[一-鿿，。、；：（）])\s+(?=[一-鿿，。、；：（）])")

# Documented corrections of OCR errors in the raw GAOKAO-Bench file.
GAOKAO_OCR_FIXES = {
    27: [("C.科研实力 B.配套产业", "C.科研实力 D.配套产业")],
}
_OPT = re.compile(r"([A-D])\s*[．.]")


def _split_options(segment: str) -> tuple[str, dict[str, str]] | None:
    """Split a sub-question segment into stem and A-D options."""
    positions = []
    start = 0
    for label in "ABCD":
        for m in _OPT.finditer(segment, start):
            if m.group(1) == label:
                positions.append((label, m.start(), m.end()))
                start = m.end()
                break
        else:
            return None
    stem = segment[: positions[0][1]].strip()
    options = {}
    for idx, (label, _, end) in enumerate(positions):
        stop = positions[idx + 1][1] if idx + 1 < len(positions) else len(segment)
        options[label] = re.sub(r"\s+", " ", segment[end:stop]).strip(" ．.\n")
    return stem, options


def _parse_gaokao_group(text: str, n_answers: int) -> tuple[str, list[tuple[str, dict[str, str]]]] | None:
    marks = list(_SUBQ.finditer(text))
    # Take the first run of consecutively numbered markers whose length equals
    # the number of answers and whose segments all parse into A-D options.
    for first in range(len(marks)):
        run = [marks[first]]
        for m in marks[first + 1 :]:
            if int(m.group(1)) == int(run[-1].group(1)) + 1:
                run.append(m)
            if len(run) == n_answers:
                break
        if len(run) != n_answers:
            continue
        segments = []
        for j, m in enumerate(run):
            stop = run[j + 1].start() if j + 1 < len(run) else len(text)
            parsed = _split_options(text[m.end() : stop])
            if parsed is None:
                break
            segments.append(parsed)
        else:
            passage = re.sub(r"\s+", " ", text[: run[0].start()]).strip()
            return passage, segments
    return None


def load_gaokao_bench(path: Path) -> tuple[list[Question], list[dict]]:
    items = json.loads(path.read_text(encoding="utf-8"))["example"]
    questions, failures = [], []
    for item in items:
        text = item["question"]
        for old, new in GAOKAO_OCR_FIXES.get(int(item["index"]), []):
            text = text.replace(old, new)
        parsed = _parse_gaokao_group(text, len(item["answer"]))
        if parsed is None:
            failures.append({"index": item["index"], "year": item["year"]})
            continue
        passage, subs = parsed
        passage = _CJK_GAP.sub("", passage)
        for k, ((stem, options), ans) in enumerate(zip(subs, item["answer"])):
            stem = _CJK_GAP.sub("", re.sub(r"\s+", " ", stem).strip())
            options = {label: _CJK_GAP.sub("", text) for label, text in options.items()}
            statements, option_statements = parse_statements(stem, options)
            questions.append(
                Question(
                    uid=f"gaokao-{int(item['index']):02d}-{k + 1}",
                    dataset="gaokao_bench",
                    qtype="single",
                    stem=stem,
                    options=options,
                    answer=[ans.strip()],
                    passage=passage or None,
                    statements=statements,
                    option_statements=option_statements,
                    figure_dependent=_mentions_figure(passage + stem),
                    source=f"{item['year']}{item['category']}",
                )
            )
    return questions, failures


# ------------------------------------------------------- position balancing


def permute_options(q: Question, salt: str = "geomcq-perm-2026") -> Question:
    """Deterministically shuffle option order (answer keys remapped).

    GeoMCQ-1200 has a strong answer-position imbalance (most single-choice keys
    are A, most multiple-choice keys are AB); evaluating on permuted options
    prevents any method from exploiting position priors.
    """
    import hashlib

    labels = sorted(q.options)
    seed = int(hashlib.sha256(f"{salt}:{q.uid}".encode()).hexdigest()[:16], 16)
    order = labels[:]
    random.Random(seed).shuffle(order)  # order[i] = original label placed at position labels[i]
    new_of_old = {old: new for new, old in zip(labels, order)}
    options = {new: q.options[old] for new, old in zip(labels, order)}
    answer = sorted(new_of_old[a] for a in q.answer)
    meta = dict(q.meta, original_label_of={new: old for new, old in zip(labels, order)})
    return Question(**{**asdict(q), "options": options, "answer": answer, "meta": meta})


# ----------------------------------------------------------------------- I/O


def write_jsonl(questions: list[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(q.to_json() for q in questions) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[Question]:
    return [Question.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------- international exams

def split_statements(stem: str) -> str:
    """Stem text without the ①–④ statement list of a statement-combination item."""
    m = re.search(rf"[{CIRCLED}]", stem)
    return stem[: m.start()] if m else stem
