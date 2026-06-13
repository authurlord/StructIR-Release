# StructIR as a Data-Agent Skill — design note (RRF findings → toolkit)

How the empirical findings become a *trained, agent-callable retrieval skill* — and why this
framing is disjoint from a separate QA system.

## 1. The finding that justifies a *skill* (not just a method)

The retrieval experiments produced one actionable fact: **no single signal or fixed fusion wins
across a heterogeneous table lake** — the right signal mix is a function of the corpus's
*observable* structure.

| source family | observable profile φ | what wins | (RRF evidence) |
|---|---|---|---|
| Wikipedia (ρ_link≈1) | entity-linked | dense / GNN ≫ BM25 | learned α learns dense+gnn (bm25→0 on nqtables) |
| financial (ρ_header 0.58–0.71) | multi-level headers | BM25 > dense; struct-reranker | learned α: bm25-heavy → multihiertt 71.42 > BM25 69.05 |
| statistical (ρ_lex low) | dimensional | dense (both weak) | — |
| docs | free-form | BM25 ≥ dense | — |

So the deployable artifact is exactly: **a policy that reads φ(D) and allocates the fusion
weights + picks the reranker variant** — i.e. "allocate weights by demand". That is a *skill*.

## 2. Offline-trained core (ships with the toolkit)

| artifact | training | role at agent time |
|---|---|---|
| **SA-RRF weight policy** `SARRFPolicy` | heuristic (source→signal map) **or** learned α=softmax(W·φ̃), W fit offline on train-split datasets | maps a NEW corpus's profile → α with **zero labels** |
| **cross-encoder reranker** | bge-reranker-v2-m3 fine-tuned (+ header-path structure-aware variant for financial) | reranks SA-RRF top-K; structure-aware auto-selected when family=financial |
| **cell-entity GNN** | HGTConv refinement on hyperlink corpora | dense-embedding refinement; auto-skipped when ρ_link=0 |

The SA-RRF result validates the policy: **scheme-A (heuristic) beats equal-weight RRF on 6/8**
(source-adaptive > one-size-fits-all = the paper's core claim); **scheme-B (learned, 2-fold,
deployable) auto-discovers the right adaptation** (bm25=0 on nqtables, bm25-heavy on multihiertt
→ 71.42 > BM25). The oracle bound shows headroom remains (multihiertt 71.42, aitqa 45.12). [Full
numbers: `structir_skill/policy.py` + `SA_RRF_FINAL.md`.]

## 3. The skill lifecycle (maps to a data agent)

```
OFFLINE (ships):  train reranker + GNN; fit SA-RRF policy W      → frozen artifacts
AGENT.fit(corpus): profile φ(D) → register signals → policy(φ)=α → build indexes
AGENT.retrieve(q): SA-RRF(α) fuse → reranker → top_k             ← one tool call
AGENT.as_tool():   OpenAI/Anthropic function spec                ← register with the agent
```

`as_tool()` already emits a function spec; the corpus profile + chosen weights are embedded in
the tool description so the agent (and a human) can see *why* this corpus is configured the way
it is — useful for transparency in a data-management setting.

## 4. Does it need a ReAct agent? (design stance — see the README integration example for a concrete ADK integration)

`retrieve()` is a single deterministic call, so the **minimum** integration is just registering
`as_tool()` — no ReAct loop required. A ReAct loop adds value only for behaviours *around* the
skill that need reasoning:
- **query reformulation** when the top-K looks low-confidence,
- **multi-corpus routing** — a real data lake is many sub-corpora; the agent fits/selects the
  right StructIR index per sub-corpus (each auto-profiled), and the ReAct trace *shows the skill
  adapting weights per source* — this is the demo that makes the "toolkit for agents" framing
  credible in the paper,
- **multi-hop** table joining / follow-up retrieval.

Recommendation: ship the `as_tool()` integration as the core; add **one ReAct trace over a
heterogeneous (multi-source) corpus** as the paper's agent-framing figure — it demonstrates the
source-adaptive policy doing its job inside an agent loop, without overclaiming a full agent system.

## 5. Boundary vs a separate QA system
This skill is retrieval-only (R@K): no QA reader, sufficiency verifier, router, or budget
operators. a separate QA system consumes retrieved evidence for QA reasoning; StructIR *produces* the ranked
tables an agent (or a separate QA system) would consume. Clean separation of concerns.
