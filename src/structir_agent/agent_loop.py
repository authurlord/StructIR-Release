"""ReAct agent loop over a retrieval tool.

Redesigned after the 2026-06-06 live-skill diagnostic, whose trajectory analysis
showed two failure modes this loop explicitly addresses:

  1. Unrestricted query reformulation DEGRADED retrieval (OTT-QA Hit@5 70.0 vs the
     committed 94.53; 144/150 episodes ended on a rewritten query). → The prompt now
     instructs VERBATIM-FIRST: pass the question unchanged on the first call;
     reformulate only if the first results are clearly irrelevant.
  2. The agent's real, value-adding decision in a multi-source lake is *routing*
     (which source family to search), not rewriting. → The tool takes a `source`
     argument and the system prompt carries the lake catalog (per-source fit() cards).

Format-following hardening: a single Action regex; on a malformed step the agent
gets one corrective nudge; the final answer is produced by a SEPARATE fixed reader
(reader.py) over the final evidence, so QA quality never depends on in-loop
freeform text (the third catalogued failure: 600-char observations made the agent
'assume' answers)."""
from __future__ import annotations
import re
from . import config
from .llm import call_llm

ACTION_RE = re.compile(
    r"Action:\s*retrieve_tables\(\s*query\s*=\s*\"(.*?)\"\s*"
    r"(?:,\s*source\s*=\s*\"([\w\-]+)\")?\s*(?:,\s*k\s*=\s*(\d+))?\s*\)",
    re.DOTALL)
FINISH_RE = re.compile(r"\bFINISH\b")

SYSTEM_LAKE = """You are a retrieval agent. A user question must be answered from a \
heterogeneous TABLE DATA LAKE with multiple sources. You have ONE tool:

  retrieve_tables(query="...", source="...", k=5)
    -> returns the top-k most relevant tables (source, id, title, snippet).

Lake catalog (built automatically by the retrieval skill at fit time):
{catalog}

Valid `source` values: {source_values} or "auto" (the skill probe-routes by itself).

Protocol (follow EXACTLY):
1. Think about which source family the question belongs to (e.g. financial-report \
questions name airlines/quarters/income statement items; encyclopedic questions \
name people/places/events/works).
2. First call: pass the user's question VERBATIM as `query` (do NOT rewrite it - \
the question's exact entities are the strongest retrieval signal), and your chosen \
`source`.
3. Inspect the Observation. If the tables clearly match the question, output FINISH.
4. Only if the results are clearly irrelevant, you may make ONE more call: either a \
different `source`, or a minimally reformulated `query` that KEEPS all entity names.
5. Then output FINISH.

Respond each turn in this format and nothing else:
Thought: <one short sentence>
Action: retrieve_tables(query="...", source="...", k=5)
OR
Thought: <one short sentence>
FINISH
"""

SYSTEM_FLAT = """You are a retrieval agent. A user question must be answered from a \
large heterogeneous TABLE collection. You have ONE tool:

  retrieve_tables(query="...", k=5)
    -> returns the top-k most relevant tables (id, title, snippet).

Protocol (follow EXACTLY):
1. First call: pass the user's question VERBATIM as `query` (do NOT rewrite it - \
the question's exact entities are the strongest retrieval signal).
2. Inspect the Observation. If the tables clearly match the question, output FINISH.
3. Only if the results are clearly irrelevant, you may make ONE more call with a \
minimally reformulated `query` that KEEPS all entity names.
4. Then output FINISH.

Respond each turn in this format and nothing else:
Thought: <one short sentence>
Action: retrieve_tables(query="...", k=5)
OR
Thought: <one short sentence>
FINISH
"""


def render_observation(hits) -> str:
    blocks = []
    for rank, h in enumerate(hits, start=1):
        t = h.get("table") or {}
        title = (t.get("title") or h.get("local_id") or h["id"])[:160]
        snippet = (t.get("text") or "")[:600]
        src = h.get("source", "")
        blocks.append(f"[Table {rank}] source={src} id={h['id']}\n"
                      f"title: {title}\n{snippet}")
    return "\n\n".join(blocks) if blocks else "(no tables found)"


async def react_episode(question: str, tool_fn, system_prompt: str,
                        max_calls: int = config.MAX_TOOL_CALLS) -> dict:
    """Run one episode. `tool_fn(query, source, k)` → (hits, meta). Returns the
    trajectory with every call's hits kept (final call drives the QA stage)."""
    transcript = system_prompt + f"\nQuestion: {question}\n"
    steps, calls = [], []
    nudged = False
    for _turn in range(max_calls + 3):
        gen = await call_llm(transcript, max_tokens=300,
                             stop=["Observation:", "\nObservation"])
        if gen.startswith("__ERR__"):
            steps.append({"type": "error", "detail": gen[:300]})
            break
        thought = ""
        tm = re.search(r"Thought:\s*(.+?)(?:\nAction:|\nFINISH|$)", gen, re.DOTALL)
        if tm:
            thought = tm.group(1).strip()[:500]
        act = ACTION_RE.search(gen)
        if act and len(calls) < max_calls:
            query = act.group(1).strip()
            source = (act.group(2) or "auto").strip()
            k = max(1, min(int(act.group(3) or config.TOP_K), 10))
            hits, meta = await tool_fn(query, source, k)
            calls.append({"query": query, "source": source, "k": k,
                          "verbatim": query.strip() == question.strip(),
                          "hits": [h["id"] for h in hits], "meta": meta})
            obs = render_observation(hits)
            steps.append({"type": "tool_call", "thought": thought,
                          "query": query, "source": source})
            transcript += (f"\nThought: {thought}\nAction: retrieve_tables("
                           f"query=\"{query}\", source=\"{source}\", k={k})\n"
                           f"Observation:\n{obs}\n")
            calls[-1]["_hits_full"] = hits
            continue
        if FINISH_RE.search(gen) or (act and len(calls) >= max_calls):
            steps.append({"type": "finish", "thought": thought})
            break
        # malformed step → one corrective nudge, then stop
        steps.append({"type": "malformed", "raw": gen[:300]})
        if nudged:
            break
        nudged = True
        transcript += ("\n(Follow the format exactly: either an Action line or "
                       "FINISH.)\n")
    final = calls[-1] if calls else None
    # SESSION EVIDENCE: STAGE-AWARE union of all calls' hits, top-5.
    # Why union at all (measured, agent_dense v1): committing to the LAST call
    # lets a worse reformulation destroy the verbatim first call (hit@5 48.9
    # reformulated-final vs 84.6 verbatim-final); the union makes extra calls
    # monotonically helpful.
    # Why stage-aware (two measured failures, 2026-06-10): (a) raw scores are
    # NOT comparable when the `source` argument changes the output stage
    # across calls — an online frontier model's cross-family retries polluted financial
    # evidence (RRF mass) with wiki reranker logits; (b) pure RANK union
    # throws away genuine cross-call comparability when all calls share one
    # scorer — qwen's member-level wiki hedges tied sibling near-duplicates
    # at every rank and ottqa QA fell 50.7→40.7. Rule: union on the finest
    # comparable scale — by SCORE when every call ends in the same stage
    # (one scorer ⇒ one scale), by per-call rank only across mixed stages.
    best = {}
    stages = {h.get("stage") for c in calls for h in c.get("_hits_full", [])}
    same_scale = len(stages) <= 1
    for c in calls:
        for rank, h in enumerate(c.get("_hits_full", []), start=1):
            key = h["score"] if same_scale else 1.0 / rank
            cur = best.get(h["id"])
            if cur is None or key > cur[0]:
                best[h["id"]] = (key, h)
    session_hits = [h for _, h in sorted(best.values(), key=lambda x: -x[0])[:5]]
    return {"steps": steps, "calls": [
                {k: v for k, v in c.items() if k != "_hits_full"} for c in calls],
            "n_tool_calls": len(calls),
            "final_hits": session_hits,
            "last_call_hits": [h["id"] for h in (final or {}).get("_hits_full", [])],
            "final_query": (final or {}).get("query"),
            "final_source": (final or {}).get("source"),
            "reformulated_final": (not final["verbatim"]) if final else None}
