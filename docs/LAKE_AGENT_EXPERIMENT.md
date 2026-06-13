# Mixed-Lake Agent Experiment — design (2026-06-09)

**Goal (RQ4 upgrade).** The 2026-06-06 live-skill run proved *callability* only — it
could not test profile-driven source adaptation (corpus dicts lacked structural
fields) and its free query reformulation degraded retrieval. This experiment
redesigns the live test so the skill's *adaptation* is exercised end-to-end by a
genuine agent on a heterogeneous lake.

## The lake

One merged table data lake, 4 sources / 2 families, ~190K tables:

| source | family | #tables | frozen plan (live fit) | cross-fit α (bm25:dense) |
|---|---|---:|---|---|
| ottqa | wikipedia | 8,891 | rerank / std | 0.10 : 0.90 |
| nqtables | wikipedia | 169,898 | rerank / std | 0.08 : 0.92 |
| aitqa | financial | 1,937 | sa_rrf / struct | 0.25 : 0.75 |
| multihiertt | financial | 9,902 | sa_rrf / struct | 0.91 : 0.09 |

Lake ids are `source::table_id`. Note the two financial sources need *opposite*
fusion biases (multihiertt BM25-heavy, aitqa dense-leaning) — a family-level
heuristic cannot reproduce both (gate evidence: heuristic aitqa M@5 29.75 vs
committed 43.95). This is precisely the source-adaptive claim: adaptation must be
per-corpus, which the calibration cross-fit in `fit()` provides.

## Conditions (same queries, same reader; only retrieval differs)

| condition | retrieval | routing |
|---|---|---|
| bm25_flat | single BM25 index over 190K | none (flat) |
| dense_flat | bge-m3 over 190K (one space) | none (flat) |
| cascade_flat | dense top-100 → trained reranker ALWAYS | none (flat) |
| structir_auto | per-source frozen plans | skill probe-vote (dense top-20 majority) |
| structir_family | per-source frozen plans | oracle family |
| structir_source | per-source frozen plans | oracle source (upper bound) |
| agent_dense | dense_flat as the tool | agent (no source arg) |
| agent_structir | TableLakeSkill.as_tool() | **agent decides source** (catalog in prompt) |

Agent scaffold identical across agent rows (verbatim-first ReAct, max 2 calls,
hardened parsing); the ONLY difference is the tool. n=150/source (deterministic
sorted sample, same qids as the 2026-06-06 run where applicable).

## Metrics

- Retrieval: Hit@5, R@5 (= mean |gold∩top5|/|gold|; M@5 formula identical), top-1
  acc, routing accuracy, tool calls/query, reformulation rate.
- QA (ottqa + nqtables, the sources with wired gold answers): EM/F1, standard QA
  scoring; fixed per-dataset readers (nqtables: top-5 tables · 1100 chars;
  ottqa: top-1 table + cell-link passages + CoT prompt).
- OTT-QA QA penalizes wrong-source retrieval *structurally*: the passage stage
  needs the ottqa-native cell links, so a near-duplicate table from nqtables
  cannot substitute — routing accuracy gates QA. (Wiki near-duplicates across
  ottqa/nqtables also mean gold-id retrieval metrics UNDERCOUNT all conditions
  equally; QA is the meaningful end metric.)

## Gates (run before any condition; all must pass)

1. **Profile gate** (`gate_profile_t0.py`): live `fit()` on real structural corpora
   reproduces the committed T0 frozen plan (family, output stage, serialization)
   per source. Status 2026-06-09: **4/4 PASS** (and 8/8 in the local all-corpus
   check, statcan/watsonx included).
2. **Retrieval gate** (runner `--phase gates`): per-source live `retrieve()` R@5
   vs committed full-eval values (94.53 / 87.63 / 43.95 / 71.42). The cross-fit
   calibration reproduces the committed fold weights exactly
   (ottqa (0.1,0.8)/(0.1,1.0) · nqtables (0.1,1.0)/(0,0.1) ·
   aitqa (0.1,0.6)/(0.4,0.9) · multihiertt (1.0,0.1)×2).

## Hypotheses

- H-A (skill, no agent): structir_auto beats the best fixed flat baseline on the
  financial slice by a large margin and is ≥ on wiki — one-size-fits-all fails in
  a heterogeneous lake even with the same trained components (cascade_flat).
- H-B (agent value): agent_structir ≈ structir_family/auto and ≫ agent_dense —
  same agent, the skill is the difference; family routing is learnable by an LLM.
- H-C (decomposition): structir_source − agent_structir quantifies the routing
  cost; structir_source − committed per-dataset values quantifies the lake
  (near-duplicate) cost.

## Infra

the GPU host (deepspeed env, CUDA 0) runs the lake + reranker + runner; the LLM endpoint
`a local 35B model` (the reader GPUs) serves both the agent LLM and the QA readers
(enable_thinking=false, T=0; endpoint health-checked before every LLM phase).
Outputs: `<asset-root>/results_lake_agent/` → mirrored to `results/lake_agent/` in the repo.
