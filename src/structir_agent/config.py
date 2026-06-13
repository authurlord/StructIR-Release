"""Configuration — all paths/endpoints env-overridable; defaults = a local cached-asset layout.

The agent + QA reader run on an OpenAI-compatible LLM endpoint (set
STRUCTIR_LLM_BASE / STRUCTIR_LLM_MODEL). Embedding + reranker run on a local GPU."""
from __future__ import annotations
import os
from pathlib import Path

# ---- LLM endpoint ----
# default = a local OpenAI-compatible 35B endpoint. Set STRUCTIR_LLM_PROVIDER=online_api
# to run the agent+reader on the online an online frontier model API (OpenAI-compatible) —
# the agent-paper configuration (frontier online model, independent of the
# offline 35B used by the a separate QA system journal experiments).
LLM_PROVIDER = os.environ.get("STRUCTIR_LLM_PROVIDER", "vllm")
if LLM_PROVIDER == "online_api":
    LLM_BASE_URL = os.environ.get("STRUCTIR_LLM_BASE", "https://api.your-llm.example/v1")
    LLM_MODEL = os.environ.get("STRUCTIR_LLM_MODEL", "<online-model>")
    LLM_API_KEY = os.environ.get("STRUCTIR_LLM_KEY", "")
    # an online frontier model is a hybrid reasoning model with thinking ON by default; disable
    # for protocol parity with the local model (non-thinking, T=0) and cost
    # (probe 2026-06-10: 'PONG' = 23 completion tokens w/ thinking vs 2 without)
    LLM_KWARGS = dict(temperature=0.0,
                      extra_body={"thinking": {"type": "disabled"}})
else:
    LLM_BASE_URL = os.environ.get("STRUCTIR_LLM_BASE", "http://localhost:8000/v1")
    LLM_MODEL = os.environ.get("STRUCTIR_LLM_MODEL", "a local 35B model")
    LLM_API_KEY = "EMPTY"
    # non-thinking deterministic recipe
    LLM_KWARGS = dict(
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
LLM_CONCURRENCY = int(os.environ.get("STRUCTIR_LLM_CONC", "8"))

# ---- cached-asset layout ----
B = Path(os.environ.get("STRUCTIR_B", "./assets"))  # cached corpus/emb/reranker root
DATA = B / "data"
EMB = B / "emb"
QRELS = B / "qrels"
UNIVRANK = B / "rerank_results_universal"
RERANKER_CKPT = str(B / "rerankers_universal/model")
BGE_M3 = os.environ.get("STRUCTIR_BGE_M3", "BAAI/bge-m3")
DEPS = B / "structir_skill_deps"          # hybridqa_metrics.py + hybridqa_qa_cot.txt
OUT_ROOT = Path(os.environ.get("STRUCTIR_OUT", str(B / "results_lake_agent")))

# ---- the mixed lake ----
LAKE_SOURCES = ["ottqa", "nqtables", "aitqa", "multihiertt"]
QA_SOURCES = ["ottqa", "nqtables"]        # have wired gold answers
N_AGENT_QUERIES = int(os.environ.get("STRUCTIR_NQ", "150"))   # per source
TOP_K = 5
MAX_TOOL_CALLS = 2

# committed full-eval references (STRUCTIR_MAIN.md) for gates
COMMITTED_R5 = {"ottqa": 94.53, "nqtables": 87.63, "aitqa": 43.95, "multihiertt": 71.42}


def unset_proxies():
    for v in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "ALL_PROXY", "all_proxy", "SOCKS_PROXY", "socks_proxy"]:
        os.environ.pop(v, None)
    os.environ["no_proxy"] = "localhost,127.0.0.1"
    os.environ["NO_PROXY"] = "localhost,127.0.0.1"
