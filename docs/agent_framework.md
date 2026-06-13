# structir_agent — open-domain table QA agent framework

A minimal, the host agent-convention agent framework that consumes the StructIR skill for
**open-domain table QA over a heterogeneous table data lake**. Built 2026-06-09/10
as the RQ4 "live agent" experiment harness; reusable as a standalone framework.

```
                       ┌──────────────────────────────────────────────┐
 question ──▶ ReAct    │ tool: structir_retrieve_tables(query,source,k)│
 (the local 35B-35B,         │   = TableLakeSkill.as_tool()                 │
  the LLM endpoint)          │   lake catalog in system prompt              │
   │  decides source + │   per-source frozen plans (fit-time)        │
   │  verbatim query   └──────────────────────────────────────────────┘
   ▼
 final top-5 evidence ──▶ fixed per-dataset 35B reader ──▶ answer (EM/F1)
```

## Design decisions (each traces to a catalogued failure of the 2026-06-06 run)

| decision | rationale |
|---|---|
| VERBATIM-FIRST prompting (rewrite only on clearly-irrelevant results) | free reformulation collapsed OTT-QA Hit@5 94.5→70.0 |
| agent's decision = SOURCE ROUTING, not query rewriting | routing is where an LLM adds value in a multi-source lake |
| corpus dicts carry real `headers`/`cells` via `load_structured_corpus` | id/title/text-only made every corpus profile as `docs` |
| answers come from a separate fixed reader over final evidence, never from in-loop freeform | 600-char observations made the agent "assume" answers |
| reader API failures stay `__ERR__`, excluded from EM=0 scoring | Phase-7 catalogued silent-zero bug |
| `health_check()` asserts endpoint serves `a local 35B model` before any run | port-drift directive (the configured port, never another port) |

## Modules

| file | role |
|---|---|
| `config.py` | endpoints/paths (env-overridable), GPU policy notes |
| `llm.py` | retried async 35B calls (enable_thinking=False, T=0) + health check |
| `lake_setup.py` | build the 4-source lake from committed caches + flat baselines (BM25/dense/cascade over the merged lake) |
| `agent_loop.py` | hardened ReAct (one Action regex, one corrective nudge, max 2 calls) |
| `reader.py` | per-dataset reader protocols (nqtables top-5 tables; ottqa cell-link passages + CoT), standard QA scoring |
| `runner.py` | phases: gates → retrieval → agent → qa → report; per-condition jsonl with resume |

## Conditions

no-LLM rows: `bm25_flat`, `dense_flat`, `cascade_flat` (always-rerank),
`structir_auto` (skill probe-routes), `structir_family` / `structir_source` (oracle
routing bounds). Agent rows: `agent_dense` vs `agent_structir` — same agent, same
prompts, only the tool differs. QA: fixed reader over each condition's final top-5.

## Run (the GPU host, conda `deepspeed`; reader = the LLM endpoint)

```bash
export PYTHONPATH=<asset-root>
python3 -m structir_agent.runner --phase gates      # T0-plan + retrieval gates
python3 -m structir_agent.runner --phase retrieval  # no-LLM condition rows
python3 -m structir_agent.runner --phase agent      # ReAct episodes (35B)
python3 -m structir_agent.runner --phase qa         # fixed readers (35B)
python3 -m structir_agent.runner --phase report
```

Outputs under `$STRUCTIR_OUT` (default `<asset-root>/results_lake_agent/`): per-condition
`*.jsonl` (resumable) + `*.summary.json` + `RAW_SUMMARY.md`.
