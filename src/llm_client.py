# -*- coding: utf-8 -*-
"""
Minimal multi-key chat client for the OpenAI-compatible endpoints used in this project
(Cerebras, Groq, NVIDIA). Keys are read from .env, used round-robin, retired on 401/402/403
(no credit) and backed off on 429. Thread-safe; token usage is accounted per provider.

    from llm_client import LLM
    qwen = LLM("cerebras", "qwen-3.8-27b")
    text = qwen.chat([{"role": "user", "content": "..."}], json_mode=True, max_tokens=800)
"""
import os
import re
import json
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"

ENDPOINTS = {
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    # NVIDIA build.nvidia.com: free developer tier, 40 requests/min per account, no daily token cap (gpt-oss-20b, etc.)
    "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
}
KEY_VARS = {"cerebras": ["CEREBRAS_API_KEYS"], "groq": ["GROQ_API_KEYS", "GROQ_API_KEY"], "nvidia": ["NVIDIA_API_KEYS", "NVIDIA_API_KEY"]}
# USD per million tokens (input, output); Groq free tier is 0
PRICES = {"qwen-3.8-27b": (0.99, 1.49), "gpt-oss-120b": (0.35, 0.75)}
RPM = {"nvidia": 38.0}


def load_keys(provider: str) -> List[str]:
    if not ENV_FILE.exists():
        return []
    s = ENV_FILE.read_text(encoding="utf-8")
    out: List[str] = []
    for var in KEY_VARS[provider]:
        m = re.search(r"^" + var + r"=([^\r\n]*)", s, re.M)
        if m:
            out += [k.strip() for k in m.group(1).split(",") if k.strip()]
    seen, uniq = set(), []
    for k in out:
        if k not in seen:
            seen.add(k); uniq.append(k)
    return uniq


class LLM:
    def __init__(self, provider: str, model: str, keys: Optional[List[str]] = None, timeout: int = 90, verbose: bool = True,
                 rpm: Optional[float] = None):
        self.provider, self.model, self.timeout, self.verbose = provider, model, timeout, verbose
        self.url = ENDPOINTS[provider]
        self.keys = keys if keys is not None else load_keys(provider)
        if not self.keys:
            raise RuntimeError(f"no API keys for {provider} in .env")
        self.dead: set = set()
        self.lock = threading.Lock()
        self.rr = 0
        # client-side throttle (requests per minute, whole account): NVIDIA's free tier is a hard 40 RPM
        self.rpm = rpm if rpm is not None else RPM.get(provider)
        self.rate_wait = 2.0
        self._next_slot = 0.0
        self.in_tokens = 0
        self.out_tokens = 0
        self.calls = 0
        self.failures = 0

    # ------------------------------------------------------------------ keys
    def live(self) -> List[str]:
        with self.lock:
            return [k for k in self.keys if k not in self.dead]

    def _next_key(self) -> Optional[str]:
        with self.lock:
            live = [k for k in self.keys if k not in self.dead]
            if not live:
                return None
            self.rr += 1
            return live[self.rr % len(live)]

    def _retire(self, key: str, why: str):
        with self.lock:
            if key in self.dead:
                return
            self.dead.add(key)
            left = len([k for k in self.keys if k not in self.dead])
        if self.verbose:
            print(f"\n[{self.provider}] cheie retrasă ...{key[-6:]} ({why}); rămase {left}", flush=True)

    # ------------------------------------------------------------------ calls
    def _throttle(self):
        if not self.rpm:
            return
        with self.lock:
            now = time.time()
            slot = max(now, self._next_slot)
            self._next_slot = slot + 60.0 / self.rpm
        if slot > now:
            time.sleep(slot - now)

    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False, temperature: float = 0.0,
             max_tokens: int = 512, reasoning: Optional[str] = None, retries: int = 8) -> Optional[str]:
        """Returns the assistant text (or None after `retries` failures / no live key)."""
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # both Cerebras models and gpt-oss on Groq are reasoning models: without this they may spend the budget on hidden reasoning
        if reasoning is None:
            reasoning = "none" if "qwen" in self.model.lower() else "low"
        if self.provider == "nvidia":
            if "nemotron" in self.model.lower():
                payload["chat_template_kwargs"] = {"enable_thinking": False}   # Nemotron 3.x: no hidden reasoning, answer directly
        elif self.provider == "cerebras" or "gpt-oss" in self.model.lower():
            payload["reasoning_effort"] = reasoning
        attempt, rate_hits = 0, 0
        while attempt < retries and rate_hits < 60:
            key = self._next_key()
            if key is None:
                return None
            self._throttle()
            try:
                r = requests.post(self.url, json=payload, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout=self.timeout)
            except Exception:
                attempt += 1
                time.sleep(2 + attempt)
                continue
            if r.status_code == 200:
                data = r.json()
                u = data.get("usage", {}) or {}
                with self.lock:
                    self.calls += 1
                    self.in_tokens += int(u.get("prompt_tokens", 0) or 0)
                    self.out_tokens += int(u.get("completion_tokens", 0) or 0)
                msg = data["choices"][0]["message"]
                return msg.get("content") or ""
            body = r.text[:300].lower()
            if r.status_code == 429:
                # rate limit (Cerebras labels it "param: quota" - that is NOT a credit problem): back off, next key.
                # Without a retry-after header (NVIDIA) wait briefly and do not burn the retry budget.
                wait = r.headers.get("retry-after")
                rate_hits += 1
                try:
                    wait = float(wait)
                    attempt += 1
                except Exception:
                    wait = self.rate_wait if self.rpm else 3.0 + 3 * attempt
                time.sleep(min(60.0, wait))
                continue
            attempt += 1
            if r.status_code in (401, 402, 403) or (r.status_code == 400 and ("credit" in body or "insufficient" in body)):
                self._retire(key, f"HTTP {r.status_code}")
            elif r.status_code == 400 and "json" in body:
                # the model could not satisfy json_mode: retry without it once
                payload.pop("response_format", None)
                time.sleep(1)
            else:
                time.sleep(2 + attempt)
        with self.lock:
            self.failures += 1
        return None

    def chat_json(self, messages: List[Dict[str, str]], **kw) -> Optional[Dict[str, Any]]:
        txt = self.chat(messages, json_mode=True, **kw)
        if txt is None:
            return None
        try:
            return json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt, re.S)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None

    def cost(self) -> float:
        pi, po = PRICES.get(self.model.split("/")[-1], (0.0, 0.0))
        return self.in_tokens / 1e6 * pi + self.out_tokens / 1e6 * po

    def stats(self) -> Dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "calls": self.calls, "failures": self.failures,
                "in_tokens": self.in_tokens, "out_tokens": self.out_tokens, "cost_usd": round(self.cost(), 3), "live_keys": len(self.live())}


LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_letter(obj_or_text: Any) -> Optional[str]:
    """Extracts a single A-D answer from a JSON object ({"letter": "B"} / {"answer": "B"}) or free text."""
    if obj_or_text is None:
        return None
    if isinstance(obj_or_text, dict):
        for k in ("letter", "answer", "choice", "option", "raspuns", "răspuns"):
            v = obj_or_text.get(k)
            if isinstance(v, str):
                m = LETTER_RE.search(v.strip().upper()[:3])
                if m:
                    return m.group(1)
        obj_or_text = json.dumps(obj_or_text, ensure_ascii=False)
    m = LETTER_RE.search(str(obj_or_text).strip().upper()[:12])
    return m.group(1) if m else None


def run_parallel(fn, items: List[Any], workers: int, desc: str = "") -> List[Any]:
    """Maps fn over items with a thread pool, preserving order; prints a one-line progress bar."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    out: List[Any] = [None] * len(items)
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                out[i] = f.result()
            except Exception as e:
                out[i] = None
                print(f"\n[{desc}] eroare la item {i}: {e}", flush=True)
            done += 1
            if done % 10 == 0 or done == len(items):
                el = time.time() - t0
                print(f"\r[{desc}] {done}/{len(items)} ({el:.0f}s)", end="", flush=True)
    print(flush=True)
    return out
