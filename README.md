# StructIR 🧭 — anonymous code release

Anonymous code release accompanying the paper *"StructIR: A Lightweight
Toolkit for Adaptive and Agentic Text2Table IR"*. StructIR packages
table retrieval over **heterogeneous table data lakes** as a
self-configuring skill an LLM data agent can call: it reads each
corpus's observable structure at `fit` time, **freezes a retrieval
plan** (registered signals, fusion weights, serialization mode, output
stage), executes it deterministically at `retrieve` time, and exposes
one stable `as_tool()` schema plus an auditable configuration card.

This repository contains the **main-method** code (the toolkit and the
agent framework), the **rdblearn-style skill packaging** for coding
agents, runnable CPU-only examples, and the headline result summaries
from the paper's mixed-lake benchmark.

## 📑 Table of Contents
- [Introduction](#-introduction)
- [Repository layout](#-repository-layout)
- [Installation](#-installation)
- [Usage](#-usage)
- [Core API Reference](#-core-api-reference)
- [Coding-agent skill](#-coding-agent-skill)
- [Reproducing the paper numbers](#-reproducing-the-paper-numbers)
- [Models & Datasets](#-models--datasets)
- [Build & usability](#-build--usability)
- [License](#-license)

## 🎯 Introduction

In a heterogeneous lake the best retrieval configuration *reverses*
across sources (dense wins on Wikipedia-style tables, exact matching on
financial filings; a reranking stage that lifts compact corpora
truncates long hierarchical ones), and the agent holding a fixed tool
cannot fix this at query time. StructIR moves the configuration offline
into `fit`, leaving the agent only what it can do reliably.

**Core components**
- **`structir_skill`** — the toolkit. `TableRetrievalSkill`
  (`fit`/`retrieve`/`as_tool`/`card`) for one corpus; `TableLakeSkill`
  for a lake of corpora with a fit-time self-diagnosed routing policy
  (vote-leak statistic → route-or-merge) and comparable-scale merging.
- **`structir_agent`** — a small agent framework that calls the skill
  inside a verbatim-first ReAct loop with a stage-aware session-evidence
  union, plus fixed per-dataset readers and a sharded experiment runner.

## 📁 Repository layout

```
StructIR-Release/
├── README.md                ← you are here
├── requirements.txt
├── pyproject.toml           ← pip install -e .  (src layout)
├── LICENSE                  ← MIT
├── SKILLS_INSTALL.md        ← install as a coding-agent skill
├── skill/
│   └── SKILL.md             ← the skill card (frontmatter + procedure)
├── src/
│   ├── structir_skill/      ← profile · policy · signals · skill · lake · corpus
│   ├── structir_gnn/        ← graph slot: build_graph · train_eval_gnn (HGTConv+InfoNCE)
│   └── structir_agent/      ← config · llm · lake_setup · agent_loop · reader · runner
├── scripts/lake/            ← profile-gate + shard-merge utilities
├── examples/
│   ├── single_corpus_smoke.py   ← one corpus, CPU-only, no downloads
│   └── lake_demo.py             ← a mixed lake, CPU-only, shows per-source frozen plans
├── docs/
│   ├── agent_framework.md
│   └── LAKE_AGENT_EXPERIMENT.md
└── results/
    ├── LAKE_AGENT_RESULTS.md     ← full mixed-lake results + the fix log
    └── full_qwen_summaries/      ← every headline number's source summary.json
```

## ⚙️ Installation

```bash
pip install -e .
# minimal (BM25-only graceful mode) needs only: rank_bm25 numpy
# dense leg + cross-encoder reranker: pip install sentence-transformers torch
# agent framework (online LLM via OpenAI-compatible API): pip install openai httpx
```

## 🚀 Usage

No GPU or model download is required for the two examples; the skill
degrades gracefully to BM25-only when the dense stack is absent.

```bash
python examples/single_corpus_smoke.py   # fit -> card -> retrieve -> as_tool
python examples/lake_demo.py             # a mixed lake: per-source frozen plans + routing
```

### Single corpus

```python
from structir_skill import TableRetrievalSkill, SARRFPolicy

skill = TableRetrievalSkill(policy=SARRFPolicy(mode="heuristic"),
                            reranker_ckpt=CKPT)        # optional trained reranker
skill.fit(corpus, queries=qs, calibration=cal)        # profile -> FROZEN PLAN -> indexes
print(skill.card())                                   # audit every frozen decision
hits = skill.retrieve("which airline had the highest 2017 revenue?", top_k=5)
tool = skill.as_tool()                                # OpenAI/Anthropic function spec
```

### A lake of corpora

```python
from structir_skill import TableLakeSkill

lake = TableLakeSkill(reranker_ckpt=CKPT)
for name, corpus, qs in sources:
    lake.add_source(name, corpus, queries=qs)         # onboarding = one call per source
out  = lake.retrieve(question, source="auto")         # probe-routes; or a family / source name
tool = lake.as_tool()                                 # one schema, source enum + lake catalog
```

## 📚 Core API Reference

| Call | What it does |
|---|---|
| `TableRetrievalSkill.fit(corpus, queries=None, calibration=None, dense_cache=None)` | Profile φ(corpus), freeze the plan (signals R_φ, weights α_φ, serialization m_φ, output stage o_φ), build indexes. With `calibration` (labeled split) the fusion weights are cross-fit; otherwise a label-free profile heuristic is used. |
| `.retrieve(q, top_k=10)` → `[{id, score, stage, table}]` | Execute the frozen plan: SA-RRF fusion → plan-gated reranker → top-k. |
| `.card()` | Every frozen decision with the statistic that triggered it. |
| `.as_tool()` | Function spec for an LLM agent. |
| `TableLakeSkill.add_source(name, corpus, queries=None, calibration=None)` | Onboard one source (per-source `fit`); returns its card. |
| `.retrieve(q, source="auto", top_k=5)` → `{hits, routing}` | `source` ∈ {member, family, `"auto"`}. `auto` probe-routes by the fit-time vote-leak self-diagnostic, then routes or merges on a comparable score scale. |
| `.as_tool()` | One agent schema with a `source` enum and the lake catalog. |

Corpus dicts must carry real structural fields (`headers`, `cells`, the
linearized `text`); `structir_skill.load_structured_corpus(linearized,
structure)` builds them from the two parallel JSONL files. Passing only
`id/title/text` makes every corpus profile as `docs` and disables
source adaptation.

## 🤖 Coding-agent skill

StructIR ships as a coding-agent skill (Claude Code / Codex / OpenCode)
in the spirit of [rdblearn](https://github.com/HKUSHXLab/rdblearn). See
`SKILLS_INSTALL.md` for the folder layout and `skill/SKILL.md` for the
skill card. After installing, the agent reads the card, installs the
toolkit, writes code against the API above, and runs it.

## 📈 Reproducing the paper numbers

`results/full_qwen_summaries/` holds one `summary.json` per condition
of the full mixed-lake benchmark (all eligible queries); every R@5 /
top-1 / EM in the paper traces to a file there, and
`results/LAKE_AGENT_RESULTS.md` is the readable table plus the
experiment-design and fix log. The agent/reader stages call any
OpenAI-compatible LLM endpoint (`STRUCTIR_LLM_BASE`,
`STRUCTIR_LLM_MODEL`); the retrieval and configuration study are
LLM-free. Full per-query trajectories are large and omitted from this
release; the summaries are sufficient to check every reported number.

## 🔗 Models & Datasets

All models and datasets are public and used off the shelf; see
[`docs/MODEL_CARDS.md`](docs/MODEL_CARDS.md) for links. Key components:
frozen dense encoder [bge-m3](https://huggingface.co/BAAI/bge-m3),
cross-encoder reranker
[bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3),
baselines [Granite-R2](https://huggingface.co/ibm-granite/granite-embedding-english-r2) /
[arctic-embed-v2](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0) /
[SPLADE](https://huggingface.co/naver/splade-cocondenser-ensembledistil),
the agent/reader LLM (a local 35B MoE such as
the [Qwen3 MoE family](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f), or any OpenAI-compatible online model), and the public [IBM Table+Text IR collection](https://huggingface.co/collections/ibm-research/table-text-ir-evaluation)
([OTT-QA](https://huggingface.co/datasets/ibm-research/OTTQASmallRetrieval), [NQ-Tables](https://huggingface.co/datasets/ibm-research/NQTablesRetrieval), [OpenWikiTables](https://huggingface.co/datasets/ibm-research/OpenWikiTablesRetrieval), [FeTaQA](https://huggingface.co/datasets/ibm-research/FeTaQARetrieval), [MultiHierTT](https://huggingface.co/datasets/ibm-research/MultiHierttRetrieval), [AIT-QA](https://huggingface.co/datasets/ibm-research/AITQARetrieval), [StatCan](https://huggingface.co/datasets/ibm-research/StatCanDialogueRetrieval), [WatsonxDocs](https://huggingface.co/datasets/ibm-research/WatsonxDocsQARetrieval)).

## 🧪 Build & usability

The toolkit installs with `pip install -e .` and both examples run
CPU-only with no model download (the dense stack degrades to BM25-only
when absent). Verified: `import structir_skill` (v0.2.0),
`single_corpus_smoke.py`, and `lake_demo.py` (4 sources profiled into
their per-source frozen plans) all pass.

## 📜 License

MIT (see `LICENSE`). Released anonymously for double-blind review.
