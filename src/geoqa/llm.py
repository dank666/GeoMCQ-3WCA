"""OpenAI-compatible chat client with a content-addressed disk cache.

Every request is keyed by (base_url, model, messages, decoding parameters); a
cached response is returned verbatim, which makes reruns deterministic and lets
the paper report exact call counts and token usage. The model name reported by
the server and the system fingerprint are stored with each response.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


# Endpoints used in the paper; the base URL is part of the cache key.
DEFAULT_BASE_URLS = {"BASELINE": "https://api.siliconflow.cn/v1", "KB_LLM": "https://api.deepseek.com"}


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    json_mode: bool = True
    seed: int | None = None
    extra_body: dict | None = None  # provider-specific, e.g. {"enable_thinking": False} for Qwen3

    @classmethod
    def from_env(cls, prefix: str, **overrides) -> "LLMConfig":
        cfg = cls(
            base_url=os.environ.get(f"{prefix}_BASE_URL") or DEFAULT_BASE_URLS.get(prefix, ""),
            api_key=os.environ.get(f"{prefix}_API_KEY", ""),
            model=os.environ.get(f"{prefix}_MODEL", ""),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def public(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "api_key"}


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, record: dict, cached: bool) -> None:
        with self.lock:
            self.calls += 1
            self.cached += cached
            u = record.get("usage") or {}
            self.prompt_tokens += u.get("prompt_tokens", 0)
            self.completion_tokens += u.get("completion_tokens", 0)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "lock"}


class LLMClient:
    def __init__(self, config: LLMConfig, cache_dir: Path, max_retries: int = 8):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.usage = Usage()
        self._client = None

    def _openai(self):
        if self._client is None:
            from openai import OpenAI

            if not self.config.api_key:
                raise RuntimeError("missing API key; fill in .env (see .env.example)")
            self._client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key, timeout=120)
        return self._client

    def _key(self, messages: list[dict]) -> str:
        c = self.config
        payload = json.dumps(
            [c.base_url, c.model, messages, c.temperature, c.top_p, c.max_tokens, c.json_mode, c.seed, c.extra_body],
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def chat(self, messages: list[dict], tag: str = "") -> dict:
        """Return {"content", "model", "fingerprint", "usage", ...}; cached on disk."""
        key = self._key(messages)
        path = self._path(key)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            self.usage.add(record, cached=True)
            return record

        c = self.config
        kwargs = dict(
            model=c.model,
            messages=messages,
            temperature=c.temperature,
            top_p=c.top_p,
            max_tokens=c.max_tokens,
        )
        if c.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if c.seed is not None:
            kwargs["seed"] = c.seed
        if c.extra_body:
            kwargs["extra_body"] = c.extra_body

        if not c.api_key:
            raise RuntimeError(f"no cached response and no API key for {c.model}; fill in .env (see .env.example)")
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self._openai().chat.completions.create(**kwargs)
                break
            except Exception:  # network / rate limit / server errors
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 60.0)  # rate limits (HTTP 429) can last about a minute
        record = {
            "tag": tag,
            "request": {"messages": messages, **c.public()},
            "content": resp.choices[0].message.content,
            "model": resp.model,
            "fingerprint": getattr(resp, "system_fingerprint", None),
            "usage": resp.usage.model_dump() if resp.usage else {},
            "created": resp.created,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")  # unique per writer
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        self.usage.add(record, cached=False)
        return record

    def chat_json(self, messages: list[dict], tag: str = "") -> tuple[dict | None, dict]:
        record = self.chat(messages, tag)
        return parse_json(record["content"]), record

    def map(self, fn, items, workers: int = 8):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(fn, items))


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None
