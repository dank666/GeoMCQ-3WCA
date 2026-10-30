"""Prompts of the method: condition elicitation and the joint three-valued judgment."""

from __future__ import annotations

CONDITION_SYSTEM = "你是高中地理教师，擅长分析选择题的题意。"

CONDITION_USER = """下面是一道地理选择题的题干（不含选项）。请找出题干对“正确答案”提出的判断条件。

要求：
1. 条件是每一个正确选项都必须满足的要求，不是某个正确选项的具体内容；用一个条件判断所有选项；
2. 第一个条件复述题目的问法，例如：
   - “下列关于X的说法正确的有”→“关于X的说法符合地理事实”
   - “X和Y的共同特征有”→“是X和Y的共同特征”
   - “X与Y的差异主要体现在”→“是X与Y之间确实存在的差异”
   - “造成该现象的主要原因是”→“是造成该现象的主要原因”
3. 其余条件是题干中对答案的额外限定（如时间、地点、数量、程度），每个 4~20 个字；没有就不写；共 1~4 个；
4. 条件一律用肯定形式表述；若题目问的是“不正确/不属于/不是/错误”的选项，把 polarity 设为 "negative"，否则为 "positive"。

只输出 JSON：{{"conditions": ["...", "..."], "polarity": "positive"}}

{passage}题干：{stem}"""

JUDGE_SYSTEM = "你是严谨的高中地理教师，只依据地理知识逐条作出判断。"

JUDGE_USER = """{passage}题干：{stem}

待判断的{unit_name}：
{units}
{knowledge}
请对每个{unit_name}逐条判断它是否满足下列条件：
{conditions}

每项判断只能是“是”（满足）、“否”（不满足）或“不确定”。
只输出 JSON，键为{unit_name}编号，值为与条件顺序一致的 {k} 个判断，例如：{example}"""


def passage_block(passage: str | None) -> str:
    return f"材料：{passage}\n" if passage else ""


def condition_messages(stem: str, passage: str | None) -> list[dict]:
    return [
        {"role": "system", "content": CONDITION_SYSTEM},
        {"role": "user", "content": CONDITION_USER.format(passage=passage_block(passage), stem=stem)},
    ]


def judge_messages(stem: str, passage: str | None, units: dict[str, str], conditions: list[str],
                   knowledge: str = "", statements: bool = False) -> list[dict]:
    unit_name = "说法" if statements else "选项"
    labels = sorted(units)
    example = "{" + ", ".join(f'"{u}": [' + ", ".join(['"是"'] * len(conditions)) + "]" for u in labels[:2]) + "}"
    know = f"\n参考知识（来自高中地理课程词典，仅供参考）：\n{knowledge}\n" if knowledge else ""
    user = JUDGE_USER.format(
        passage=passage_block(passage), stem=stem, unit_name=unit_name,
        units="\n".join(f"{u}. {units[u]}" for u in labels), knowledge=know,
        conditions="\n".join(f"{i + 1}. {c}" for i, c in enumerate(conditions)), k=len(conditions), example=example,
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
