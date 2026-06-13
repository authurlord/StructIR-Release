"""35B vLLM access — one retried async entry point (parity with a standard async LLM client:
enable_thinking=False, T=0 deterministic eval, bounded retries)."""
from __future__ import annotations
import asyncio
from . import config


_client = None


def client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        if config.LLM_PROVIDER != "online_api":
            config.unset_proxies()       # localhost vLLM must bypass proxies
        _client = AsyncOpenAI(base_url=config.LLM_BASE_URL,
                              api_key=config.LLM_API_KEY or "EMPTY",
                              timeout=300)
    return _client


USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
         "cached_tokens": 0, "errors": 0}


def _track(r):
    try:
        USAGE["calls"] += 1
        USAGE["prompt_tokens"] += r.usage.prompt_tokens or 0
        USAGE["completion_tokens"] += r.usage.completion_tokens or 0
        det = getattr(r.usage, "prompt_tokens_details", None)
        if det and getattr(det, "cached_tokens", None):
            USAGE["cached_tokens"] += det.cached_tokens
        # also honor an online frontier model's prompt_cache_hit_tokens field
        hit = getattr(r.usage, "prompt_cache_hit_tokens", None)
        if hit:
            USAGE["cached_tokens"] += hit
    except Exception:
        pass


def dump_usage(path=None):
    """Billing-grade token totals for this process (call at phase end)."""
    import json
    if path:
        json.dump(USAGE, open(path, "w"), indent=2)
    return dict(USAGE)


async def call_llm(prompt: str, max_tokens=512, stop=None, retries=4) -> str:
    """Single-turn call. Returns text, or '__ERR__<e>' after exhausting retries
    (callers MUST check — silent EM=0 scoring of API failures was a catalogued
    Phase-7 bug). Token usage is accumulated in USAGE / dump_usage()."""
    for attempt in range(retries):
        try:
            r = await client().chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, stop=stop, **config.LLM_KWARGS)
            _track(r)
            return r.choices[0].message.content or ""
        except Exception as e:
            if attempt == retries - 1:
                USAGE["errors"] += 1
                return f"__ERR__{e}"
            await asyncio.sleep(2 * (attempt + 1))


async def health_check() -> str:
    """Verify the endpoint serves the expected model id BEFORE any run
    (user directive: do not let the port drift to another port/other servers)."""
    import httpx
    headers = {}
    if config.LLM_PROVIDER == "online_api":
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    else:
        config.unset_proxies()
    async with httpx.AsyncClient(timeout=15, trust_env=False) as c:
        r = await c.get(config.LLM_BASE_URL.rstrip("/") + "/models",
                        headers=headers)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", [])]
    if config.LLM_MODEL not in ids:
        raise RuntimeError(f"endpoint {config.LLM_BASE_URL} serves {ids}, "
                           f"expected {config.LLM_MODEL}")
    return f"OK {config.LLM_BASE_URL} -> {config.LLM_MODEL}"
