# StructIR Skill — source-adaptive table retrieval for data agents

A drop-in **retrieval skill / toolkit plugin** an LLM data agent can call to find relevant
tables in a heterogeneous table data lake. The agent never picks a retrieval strategy —
`fit()` profiles each corpus and **freezes a per-corpus retrieval plan** (registered
signals, SA-RRF fusion weights, serialization mode, output stage), then `retrieve()`
executes that plan deterministically and `as_tool()` exposes it to the agent.

**Explicitly disjoint from a separate QA system**: no QA reader, no sufficiency verifier, no router, no
budget operators. This is *retrieval* (R@K), exposed as an agent tool.

Agent-skill packaging (rdblearn-style): see `../SKILLS_INSTALL.md` and
`../skills/use-structir/SKILL.md`.

## Interface — single corpus

```python
from structir_skill import TableRetrievalSkill, SARRFPolicy, load_structured_corpus

tables = load_structured_corpus("corpus_linearized.jsonl", "corpus_structure.jsonl")
skill = TableRetrievalSkill(
    dense_model="BAAI/bge-m3",
    reranker_ckpt="models/structir_reranker",       # offline-trained (ships with toolkit)
    policy=SARRFPolicy(mode="heuristic"),
)

# OFFLINE/fit: profile → FROZEN PLAN → indexes
skill.fit(corpus=tables, queries=sample_queries,
          dense_cache=...,            # optional: precomputed corpus embeddings
          calibration=...)            # optional: labeled split → 2-fold cross-fit weights
print(skill.card())
# {'family': 'financial',
#  'frozen_plan': {'signals': ['bm25','dense'], 'serialization': 'struct',
#                  'fusion_bias': 'bm25', 'output_stage': 'sa_rrf'},
#  'sa_rrf_weights': {'bm25': 0.91, 'dense': 0.09}, 'policy_mode': 'cross-fit', ...}

# ONLINE: frozen plan (SA-RRF fusion → plan-gated reranker) → top_k
hits = skill.retrieve("which airline had the highest 2017 passenger revenue?", top_k=10)

tool = skill.as_tool()   # → {"type":"function","function":{...}} for an LLM agent
```

## Interface — multi-source data lake

```python
from structir_skill import TableLakeSkill
lake = TableLakeSkill(reranker_ckpt=...)
lake.add_source("wiki_tables", wiki_corpus, queries=qs1)   # → wikipedia plan (rerank/std)
lake.add_source("fin_reports", fin_corpus, queries=qs2)    # → financial plan (sa_rrf/struct)
out = lake.retrieve(q, source="auto")    # probe-routes; or a family / source name
tool = lake.as_tool()                    # ONE tool: source enum + lake catalog
```

Multi-source merging only happens on comparable score scales (shared cross-encoder for
rerank-stage sources; same-κ L1-normalized RRF mass for sa_rrf-stage sources) — mixed
stages refuse to merge and route instead.

## The frozen plan (T0 mapping; thresholds are global constants, never dataset names)

```
corpus ──fit()──▶ φ(D) = (ρ_link, ρ_header, ρ_idf, ρ_len)
   ρ_link  ≥ 0.5 → register Graph candidate signal (deployed only with a ckpt)
   ρ_header≥ 0.4 → struct (header-path) serialization for the reranker
   family → fusion bias (financial/docs BM25-heavy, wikipedia/statistical dense-heavy)
   ρ_len   ≥ 0.6 → output stage Rerank; else STOP at SA-RRF
                    (512-token cross-encoder truncates long tables — committed evidence:
                     MultiHierTT R@5 71.4→55.6 when force-reranked)
```

Live `fit()` on the real structural corpora reproduces the committed T0 plans on all 8
IBM TableIR datasets (`scripts/lake/gate_profile_t0.py`).

## The offline-trained / fitted core

| Artifact | What | Where it comes from |
|---|---|---|
| **Cross-encoder reranker** | bge-reranker-v2-m3 fine-tuned on nqtables-train (universal, zero-shot elsewhere); structure-aware (header-path) input on financial corpora | the training scripts in `structir_gnn/` |
| **Cell-entity GNN** | profile-registered candidate on link-rich corpora; deployed only with a ckpt (committed finding: zero-shot null) | StructIR §3.3 |

## Graceful degradation
If `sentence-transformers` / `torch` / a checkpoint is absent, the skill runs **BM25-only**
(dense/GNN/reranker become no-ops) so it is always callable in a minimal environment.

## Files
`profile.py` (φ(D) + frozen-plan thresholds) · `policy.py` (SA-RRF weight policy) ·
`signals.py` (BM25/dense/GNN + RRF fusion) · `skill.py` (TableRetrievalSkill:
fit/retrieve/as_tool + cross-fit calibration) · `lake.py` (TableLakeSkill: multi-source
onboarding, probe routing, comparable-scale merging) · `corpus.py` (structural loaders).
Examples: `example_smoke.py` (single corpus, CPU-only) · `../examples/lake_demo.py`
(mixed lake, CPU-only). Agent framework on top: `../structir_agent/` (see
`../docs/agent_framework.md`).
