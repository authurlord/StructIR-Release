---
name: use-structir
description: Source-adaptive table retrieval skill for LLM data agents (StructIR). Use when an agent must retrieve relevant tables from one or more heterogeneous table corpora (data lakes) — it profiles each corpus at fit time, freezes a per-source retrieval plan (signals, SA-RRF fusion weights, serialization, rerank-or-not), and exposes retrieve()/as_tool() for online calls.
---

# Use StructIR

When asked to retrieve tables, build a table-retrieval tool for an agent, or run
table-retrieval / open-domain table-QA experiments with StructIR, follow:

1. Read the docs in `use-structir/docs/` (this repo) — `structir_README.md` first, then
   `examples/` (`single_corpus_smoke.py` = single corpus CPU-only;
   `lake_demo.py` = multi-source lake + routing).
2. Install: `pip install -e <repo>` (deps:
   `rank_bm25 numpy`; optional GPU extras: `sentence-transformers torch`).
3. Core API (single corpus):

   ```python
   from structir_skill import TableRetrievalSkill, SARRFPolicy
   skill = TableRetrievalSkill(policy=SARRFPolicy(mode="heuristic"),
                               reranker_ckpt=...)   # optional trained reranker
   skill.fit(corpus=tables, queries=sample_queries) # tables: [{"id","text","title","headers","cells"}]
   skill.card()                                     # profile + FROZEN PLAN (audit this!)
   hits = skill.retrieve("...", top_k=5)
   tool = skill.as_tool()                           # LLM function spec
   ```

4. Multi-source data lake (per-source fit + routing):

   ```python
   from structir_skill import TableLakeSkill
   lake = TableLakeSkill(reranker_ckpt=...)
   lake.add_source("wiki_tables", wiki_corpus, queries=qs1)      # → wikipedia plan
   lake.add_source("fin_reports", fin_corpus, queries=qs2)       # → financial plan
   out = lake.retrieve(question, source="auto")     # probe-routes, or pass a family/source
   tool = lake.as_tool()                            # one tool, source enum + catalog
   ```

5. Corpus dicts MUST carry real structural fields (`headers`, `cells`, linearized
   `text`) — passing only `id/title/text` makes every corpus profile as `docs`
   and silently disables source adaptation (catalogued failure, 2026-06-06).
   Loader: `structir_skill.load_structured_corpus(linearized_jsonl, structure_jsonl)`.
6. Write the code for the user's request, run it, and debug if errors raise.
   Verify `skill.card()['frozen_plan']` matches expectations BEFORE large runs
   (wikipedia→rerank/std, financial→sa_rrf/struct).
7. For the full agent framework (ReAct loop + 35B reader + mixed-lake eval) read
   `docs/agent_framework.md` and use `structir_agent.runner`.
