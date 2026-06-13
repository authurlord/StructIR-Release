"""Fixed 35B readers — one protocol per QA dataset, IDENTICAL across all retrieval
conditions (the controlled variable is retrieval, never the reader).

  nqtables : top-5 retrieved tables, 1100 chars each, short-answer prompt
             (verbatim the committed agent_tool_reader.py protocol; EM/F1 there:
             StructIR 51.83/64.33 on the full set)
  ottqa    : top-1 table + its cell-link passages (BM25 top-20 within links) +
             the a separate QA system hybridqa_qa_cot prompt (verbatim the committed
             agent_skill_runner.py protocol; oracle-table ceiling EM ~74)

Scoring = standard QA (hybridqa_metrics.exact_match + token_precision_recall_f1).
Reader API failures are kept as __ERR__ and excluded from silent zero-scoring."""
from __future__ import annotations
import json, re, sys
from . import config
from .llm import call_llm

from . import qa_metrics as _hm

exact_match = _hm.exact_match
token_precision_recall_f1 = _hm.token_precision_recall_f1


def em_any(pred, golds):
    return 1.0 if any(exact_match(pred, g) for g in golds) else 0.0


def f1_best(pred, golds):
    return max(token_precision_recall_f1(pred, g)[2] for g in golds) if golds else 0.0


# --------------------------- nqtables protocol ---------------------------
NQ_PROMPT = (
    "You are an agent answering a question. A retrieval tool returned the following "
    "candidate tables (most relevant first). Answer the question using ONLY these tables.\n"
    "Respond with the short answer only, no explanation.\n\n"
    "Retrieved tables:\n{evidence}\n\nQuestion: {q}\nAnswer:"
)
NQ_PER_TABLE_CHARS = 1100


def _extract_short(text):
    s = (text or "").strip()
    for line in s.splitlines():
        line = line.strip()
        if line:
            line = re.sub(r"^(answer|the answer is)\s*[:\-]?\s*", "", line, flags=re.I)
            return line.strip().strip(".").strip()
    return s


async def answer_nqtables(question, hits, tables_by_lid):
    blocks = []
    for rank, lid in enumerate(hits[:config.TOP_K], start=1):
        txt = (tables_by_lid(lid) or "").strip()
        blocks.append(f"[Table {rank}] (id={lid})\n{txt[:NQ_PER_TABLE_CHARS]}")
    prompt = NQ_PROMPT.format(evidence="\n\n".join(blocks), q=question.strip())
    raw = await call_llm(prompt, max_tokens=128)
    if raw.startswith("__ERR__"):
        return raw, None
    return _extract_short(raw), raw[:500]


# ---------------------------- ottqa protocol -----------------------------
_OTTQA_TABLES = None
_OTTQA_POOL = None
_COT_HEADER = None


def _load_ottqa_assets():
    global _OTTQA_TABLES, _OTTQA_POOL, _COT_HEADER
    if _OTTQA_TABLES is None:
        _OTTQA_TABLES = json.load(open(config.DATA / "ottqa/traindev_tables.json"))
        pool = {}
        with open(config.DATA / "ottqa/passages.jsonl") as f:
            for line in f:
                if line.strip():
                    rr = json.loads(line)
                    pid = (rr.get("pid") or "").lower()
                    if pid:
                        pool[pid] = (rr.get("summary") or "").strip()
        _OTTQA_POOL = pool
        _COT_HEADER = (
            "Answer the question using the Selected Table and the Linked "
            "Passages. Think step by step, then give the final short answer on "
            "a line starting with 'Answer:'.")
    return _OTTQA_TABLES, _OTTQA_POOL


def _bm25_rank(query, candidates):
    if not candidates:
        return []
    from rank_bm25 import BM25Okapi
    tok = lambda s: re.findall(r"\w+", s.lower())
    docs = [tok(a + " " + t) for a, t in candidates]
    if not any(docs):
        return list(range(len(candidates)))
    sc = BM25Okapi(docs).get_scores(tok(query))
    return sorted(range(len(candidates)), key=lambda i: -sc[i])


def _render_table(t, max_rows=30, cell_chars=200):
    title = (t.get("title") or "")[:300]
    section = (t.get("section_title") or "")[:200]
    cap = (f"# {title}{' — ' + section if section else ''}" if title else "").strip()
    h_names = [h[0] if isinstance(h, list) else str(h) for h in t.get("header", [])]
    lines = ([cap] if cap else []) + ["col : row_id | " + " | ".join(h_names)]
    for r_idx, row in enumerate(t.get("data", [])[:max_rows]):
        cells = [str(r_idx)] + [str(c[0] if isinstance(c, list) and c else c)[:cell_chars]
                                for c in row]
        lines.append(f"row {r_idx} : " + " | ".join(cells))
    return "\n".join(lines)


async def answer_ottqa(question, top1_local_id, k_pas=20, passage_chars=2500):
    tables, pool = _load_ottqa_assets()
    t = tables.get(top1_local_id, {})
    cand, seen = [], set()
    for row in t.get("data", []):
        for cell in row:
            if isinstance(cell, list) and len(cell) >= 2:
                for u in (cell[1] or []):
                    pid = u.lower()
                    if pid in seen or pid not in pool:
                        continue
                    seen.add(pid)
                    cand.append((pid, str(cell[0])[:60], pool[pid]))
    order = _bm25_rank(question, [(c[1], c[2][:passage_chars]) for c in cand])[:k_pas]
    chosen = [cand[i] for i in order]
    passages = "\n".join(
        f"[Passage {i+1}] pid={c[0][:50]} cell={c[1]}\n{c[2][:passage_chars]}"
        for i, c in enumerate(chosen)) or "(none)"
    prompt = _COT_HEADER + (
        f"\n\nNow answer:\nQuestion: {question}\nSelected Table:\n{_render_table(t)}\n"
        "SQL Result:\n(no SQL run; rely on Selected Table and Linked Passages directly)\n"
        f"Linked Passages:\n{passages}\n")
    raw = await call_llm(prompt, max_tokens=2048)
    if raw.startswith("__ERR__"):
        return raw, None
    try:
        ans = _hm.parse_answer(raw)
        if ans:
            return ans, raw[:1200]
    except Exception:
        pass
    return _extract_short(raw), raw[:1200]
