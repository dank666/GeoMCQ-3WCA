"""Extract the numbered content requirements of every module of the curriculum standard.

Source: 普通高中地理课程标准（2017 年版）, kb/sources/curriculum_standard_geography_2017.pdf.
Output: kb/seeds/curriculum_requirements.jsonl, one requirement per line.
These requirements are the only seed material for dictionary construction; no
evaluation item is ever shown to the dictionary builder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "kb" / "sources" / "curriculum_standard_geography_2017.pdf"
OUT = ROOT / "kb" / "seeds" / "curriculum_requirements.jsonl"

MODULES = [
    ("M1", "地理1", "必修"),
    ("M2", "地理2", "必修"),
    ("S1", "自然地理基础", "选择性必修"),
    ("S2", "区域发展", "选择性必修"),
    ("S3", "资源、环境与国家安全", "选择性必修"),
    ("E1", "天文学基础", "选修"),
    ("E2", "海洋地理", "选修"),
    ("E3", "自然灾害与防治", "选修"),
    ("E4", "环境保护", "选修"),
    ("E5", "旅游地理", "选修"),
    ("E6", "城乡规划", "选修"),
    ("E7", "政治地理", "选修"),
    ("E8", "地理信息技术应用", "选修"),
    ("E9", "地理野外实习", "选修"),
]

# Running headers and page numbers inserted by the PDF layout.
_NOISE = re.compile(r"^(│.*│|\d+|普通高中地理课程标准（2017 年版）)$")


def clean_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not _NOISE.match(ln.strip())]


def main() -> None:
    reader = PdfReader(str(PDF))
    text = "\n".join(page.extract_text() for page in reader.pages)
    body = text[text.index("四、课程内容\n（一）必修课程") :]
    body = body[: body.index("五、学业质量")]

    # Split the course-content chapter at module headings.
    heads = []
    for mid, name, level in MODULES:
        if mid.startswith("M"):
            pattern = rf"\n{name}\n本模块"
        else:
            pattern = rf"\n(?:选择性必修|选修)\s*\d\s*{re.escape(name)}\n"
        m = re.search(pattern, body)
        if not m:
            raise SystemExit(f"module heading not found: {name}")
        heads.append((m.start(), mid, name, level))
    heads.sort()

    records = []
    for k, (start, mid, name, level) in enumerate(heads):
        stop = heads[k + 1][0] if k + 1 < len(heads) else len(body)
        section = body[start:stop]
        req = section[section.index("【内容要求】") + len("【内容要求】") :]
        req = req[: req.index("【教学提示】")] if "【教学提示】" in req else req
        lines = clean_lines(req)
        items, current = [], None
        for ln in lines:
            m = re.match(r"^(\d+\.\d+)\s*(.*)", ln)
            if m:
                current = [m.group(1), m.group(2)]
                items.append(current)
            elif current is not None:
                current[1] += ln
        for item_id, item_text in items:
            records.append(
                {
                    "module_id": mid,
                    "module": name,
                    "level": level,
                    "item_id": f"{mid}-{item_id}",
                    "text": re.sub(r"\s+", "", item_text),
                }
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    by_module = {}
    for r in records:
        by_module[r["module"]] = by_module.get(r["module"], 0) + 1
    print(f"{len(records)} requirements", by_module)


if __name__ == "__main__":
    main()
