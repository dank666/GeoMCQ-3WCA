"""Paths, data splits and the LLM client shared by the experiment scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geoqa.data import Question, read_jsonl  # noqa: E402
from geoqa.llm import LLMClient, LLMConfig  # noqa: E402

DICTIONARY = ROOT / "kb" / "geo_dictionary.jsonl"


def load_split(name: str) -> list[Question]:
    processed = ROOT / "data" / "processed"
    if name == "gaokao":
        return read_jsonl(processed / "gaokao_geo.jsonl")
    splits = json.loads((processed / "splits.json").read_text(encoding="utf-8"))
    keep = set(splits[f"geomcq_{name}"])
    return [q for q in read_jsonl(processed / "geomcq1200.jsonl") if q.uid in keep]


HYBRID = ("Qwen/Qwen3",)  # hybrid reasoning models: thinking is disabled


def client(model: str, json_mode: bool, max_tokens: int) -> LLMClient:
    if model.startswith("deepseek") and "/" not in model:
        cfg = LLMConfig.from_env("KB_LLM", model=model, extra_body={"thinking": {"type": "disabled"}})
    else:
        cfg = LLMConfig.from_env("BASELINE", model=model,
                                 extra_body={"enable_thinking": False} if model.startswith(HYBRID) else None)
    cfg.temperature, cfg.max_tokens, cfg.json_mode = 0.0, max_tokens, json_mode
    return LLMClient(cfg, ROOT / "results" / "cache" / "twc", max_retries=40)
