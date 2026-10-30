"""Prompts of direct answering (Appendix B), used by the baseline and by the second-stage comparison."""

from __future__ import annotations

DIRECT_SYSTEM = "You are a geography expert."

DIRECT_SINGLE = """Please answer the following geography question.

{passage}Question:
{stem}

Options:
{options}

Instructions:
1. Select the single best answer.
2. Return only one option letter.
3. Do not provide explanations or additional text.

Answer:"""

DIRECT_MULTIPLE = """Please answer the following geography question.

{passage}Question:
{stem}

Options:
{options}

Instructions:
1. One or more options may be correct.
2. Return only the corresponding option letters in alphabetical order.
3. Do not provide explanations or additional text.

Examples:
A
AC
ABC

Answer:"""


def _passage_block(passage: str | None, zh: bool = True) -> str:
    if not passage:
        return ""
    return f"材料：{passage}\n" if zh else f"Material:\n{passage}\n\n"


def direct_messages(stem: str, passage: str | None, options: dict[str, str], multiple: bool) -> list[dict]:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    tmpl = DIRECT_MULTIPLE if multiple else DIRECT_SINGLE
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": tmpl.format(passage=_passage_block(passage, zh=False), stem=stem, options=opts)},
    ]
