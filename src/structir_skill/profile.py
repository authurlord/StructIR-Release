"""Corpus profiling — compute the observable structural descriptor φ(D) that drives
source-adaptive retrieval. NO labels / qrels are used; everything here is read off the
corpus itself, so it is available at agent fit() time on any new table corpus.

The four statistics match the committed T0 definitions (analysis/profile_plan_notes.md):
  rho_link   : entity/hyperlink coverage — fraction of tables with >=1 linked cell
               (wiki markup, /wiki/ URL, or proper-noun entity span; the committed
               `hyperlink_density` definition from scripts/profile_corpora.py)
  rho_header : multi-level-header fraction — header depth >=2 (structured headers OR
               markdown grouped financial headers detected inside `text`)
  rho_idf    : lexical specificity / BM25-effectiveness proxy — mean fraction of query
               terms present verbatim in the best-matching corpus surface (q_contain).
               Falls back to a corpus-internal type-token estimate without queries.
  rho_len    : reranker-window feasibility — fraction of table serializations whose
               length fits the cross-encoder window (words*1.3 <= 512 token proxy)

Thresholds → frozen plan (global constants, NOT per-dataset branching):
  rho_link  >= 0.5  → register the Graph candidate signal
  rho_header>= 0.4  → struct (header-path) serialization for the reranker
  rho_idf   >= 0.6  → BM25-heavy fusion bias (else Dense-heavy)
  rho_len   >= 0.6  → output stage Rerank (else stop at SA-RRF — long tables truncate)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import re, math, random

_TOK = re.compile(r"\w+")
def _tok(s: str): return _TOK.findall((s or "").lower())

# ---- entity-link surface forms (committed hyperlink_density definition) ----
_WIKI_RE = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]*)?\]\]")
_URL_RE = re.compile(r"/wiki/([A-Za-z0-9_%().,'\-]+)")
_PROPER_RE = re.compile(r"\b([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){1,4})\b")  # >=2 cap words
_NUM_RE = re.compile(r"^[\$£€]?\s*[-+(]?[\d,]+(?:\.\d+)?\s*%?\)?$")
_DATE_RE = re.compile(
    r"\b(19|20)\d{2}\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE)
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_SEP_ROW_RE = re.compile(r"^\s*\|?[\s:|-]*-{2,}[\s:|-]*\|?\s*$")

# plan thresholds (global constants)
TH_LINK, TH_HEADER, TH_IDF, TH_LEN = 0.5, 0.4, 0.6, 0.6
RERANK_TOKEN_WINDOW = 512
TOKEN_PER_WORD = 1.3


@dataclass
class CorpusProfile:
    """Observable structural descriptor of a table corpus (no labels)."""
    n_corpus: int
    rho_link: float      # entity/hyperlink coverage
    rho_header: float    # multi-level-header fraction (financial/hierarchical signal)
    rho_idf: float       # query-term containment = BM25 effectiveness proxy
    rho_len: float       # fraction of tables fitting the reranker window
    family: str = "unknown"   # wikipedia | financial | statistical | docs | unknown

    # backward-compat alias (older policy code used rho_lex)
    @property
    def rho_lex(self) -> float:
        return self.rho_idf

    def as_dict(self): return asdict(self)

    def classify(self) -> str:
        """Coarse source family from the profile — the agent-facing label.
        NOTE: the live rho_idf is a label-free containment proxy that runs
        ~+0.1 above the committed gold-pair diagnostic, so the statistical
        cutoff is 0.35 here (vs 0.25 on the gold-pair scale)."""
        if self.rho_link >= TH_LINK:
            return "wikipedia"
        if self.rho_header >= TH_HEADER:
            return "financial"
        if self.rho_idf < 0.35:
            return "statistical"
        return "docs"

    # ---- the frozen plan (T0 mapping; thresholds are global constants) ----
    def plan(self) -> dict:
        signals = ["bm25", "dense"]
        if self.rho_link >= TH_LINK:
            signals.append("graph")          # candidate; deployed only with a ckpt
        # fusion bias follows the family map the deployed SARRFPolicy uses
        # (financial/docs = exact-term queries → BM25-heavy; wikipedia/statistical
        # = paraphrastic → dense-heavy). rho_idf stays a reported diagnostic: its
        # label-free proxy preserves the financial>wiki ordering but is too noisy
        # on very large corpora for a hard global threshold.
        fam = self.classify()
        return {
            "signals": signals,
            "serialization": "struct" if self.rho_header >= TH_HEADER else "std",
            "fusion_bias": "bm25" if fam in ("financial", "docs") else "dense",
            "output_stage": "rerank" if self.rho_len >= TH_LEN else "sa_rrf",
        }


def profile_corpus(tables: list[dict], queries: list[str] | None = None,
                   sample: int = 8000) -> CorpusProfile:
    """Profile a table corpus.

    tables: list of {"title","text"(linearized),"headers","cells","links"(optional),
                     "n_header_levels"(optional int)} — structure dicts. `text` (the
            linearized serialization the retrieval pipeline operates on) drives
            rho_len and the markdown header detection for corpora whose tables live
            inside free text (financial 10-K reports).
    queries: optional sample of queries to estimate rho_idf (query containment); if
             absent rho_idf falls back to a corpus-internal redundancy estimate.
    """
    n = len(tables)
    if n == 0:
        return CorpusProfile(0, 0.0, 0.0, 0.0, 0.0, "unknown")
    rng = random.Random(0)
    samp = tables if n <= sample else [tables[i] for i in rng.sample(range(n), sample)]

    linked = mlvl = fit_len = 0
    for t in samp:
        # rho_link: explicit links field OR entity surface forms in title+cells
        if t.get("links") or _has_entity_link(t):
            linked += 1
        # rho_header: explicit level field, structured headers, or markdown headers
        if _header_levels(t) >= 2:
            mlvl += 1
        # rho_len: serialization fits the cross-encoder window
        w = len((_table_blob(t) or "").split())
        if w * TOKEN_PER_WORD <= RERANK_TOKEN_WINDOW:
            fit_len += 1

    rho_link = linked / len(samp)
    rho_header = mlvl / len(samp)
    rho_len = fit_len / len(samp)
    rho_idf = _estimate_rho_idf(samp, queries, rng)

    p = CorpusProfile(n, round(rho_link, 3), round(rho_header, 3),
                      round(rho_idf, 3), round(rho_len, 3))
    p.family = p.classify()
    return p


def _table_blob(t: dict) -> str:
    """The serialization rho_len measures — prefer the linearized `text` the
    retrieval pipeline actually feeds the cross-encoder."""
    if t.get("text"):
        return str(t["text"])
    return _struct_text(t)


def _struct_text(t: dict) -> str:
    return " ".join([str(t.get("title", ""))] +
                    [str(x) for x in (t.get("headers") or [])] +
                    [str(x) for x in (t.get("cells") or [])][:200])


def _has_entity_link(t: dict) -> bool:
    """Committed hyperlink_density semantics (scripts/profile_corpora.py): wiki
    markup, /wiki/ URL, or a proper-noun (>=2 capitalized words) entity span — in
    title + CELLS only. Free-running `text` is deliberately excluded from the
    proper-noun channel: financial-report prose is full of company names, which
    would fake cell-level entity links (markup links in text still count)."""
    cells = t.get("cells") or []
    cell_blob = " ".join([str(t.get("title", ""))] + [str(c) for c in cells[:200]])[:4000]
    if _WIKI_RE.search(cell_blob) or _URL_RE.search(cell_blob):
        return True
    if cells and _PROPER_RE.search(cell_blob):
        return True
    txt = str(t.get("text", ""))[:4000]
    return bool(_WIKI_RE.search(txt) or _URL_RE.search(txt))


def _header_levels(t: dict) -> int:
    if t.get("n_header_levels"):
        return int(t["n_header_levels"])
    h = t.get("headers")
    if isinstance(h, list) and h:
        if isinstance(h[0], (list, tuple)):
            return len(h)              # list-of-rows headers
        flat = [str(x).strip() for x in h]
        blank = sum(1 for x in flat if not x)
        cnt = {}
        for x in flat:
            if x:
                cnt[x] = cnt.get(x, 0) + 1
        repeated = any(v > 1 for v in cnt.values())
        if blank > 0 or repeated:      # spanning/grouped header indicators
            return 2
        return 1
    # markdown-table corpora (financial reports): detect grouped headers in text
    txt = t.get("text") or ""
    if txt:
        return _markdown_header_levels(txt)
    return 1


def _md_row_cells(ln):
    return [c.strip() for c in ln.strip().strip("|").split("|")]


def _markdown_header_levels(text: str) -> int:
    """Markdown grouped-header detection (ported from the committed
    scripts/profile_corpora.py::detect_markdown_table): explicit pre-separator
    header rows, multiple separator rows (several stacked tables / grouped
    sections in one document), plus the financial spanning-group pattern (a
    header row with trailing blanks followed by a date/numeric-only sub-header
    row). Committed rule: multilevel iff header_levels+grouped > 1 OR n_sep > 1."""
    lines = text.split("\n")
    pipe_lines = [ln for ln in lines if _PIPE_ROW_RE.match(ln)]
    if len(pipe_lines) < 2:
        return 1
    n_sep = sum(1 for ln in pipe_lines if _SEP_ROW_RE.match(ln))
    if n_sep > 1:
        return 2
    header_levels = 0
    for i, ln in enumerate(pipe_lines):
        if _SEP_ROW_RE.match(ln):
            header_levels = i
            break
    grouped = 0
    non_sep = [ln for ln in pipe_lines if not _SEP_ROW_RE.match(ln)]
    for i in range(len(non_sep) - 1):
        cur = _md_row_cells(non_sep[i])
        nxt = _md_row_cells(non_sep[i + 1])
        if len(cur) < 3:
            continue
        trailing_blank = sum(1 for c in reversed(cur) if c == "")
        has_label = any(c and not _NUM_RE.match(c) and not _DATE_RE.search(c)
                        for c in cur)
        nonempty = [c for c in nxt if c]
        if not nonempty:
            continue
        periodish = sum(1 for c in nonempty
                        if _DATE_RE.search(c) or _NUM_RE.match(c))
        if (trailing_blank >= 1 and has_label
                and periodish / len(nonempty) >= 0.6 and len(nonempty) >= 2):
            grouped = 1
            break
    return max(header_levels, 1) + grouped


def _estimate_rho_idf(samp, queries, rng) -> float:
    """Query-term containment (q_contain): mean over a query sample of the max
    fraction of query tokens present verbatim in a corpus table's token set.
    High ⇒ exact-term queries ⇒ BM25 effective. Without queries, fall back to a
    type-token-ratio proxy for keyword distinctiveness."""
    if queries:
        qs = queries if len(queries) <= 300 else rng.sample(list(queries), 300)
        sub = samp if len(samp) <= 2000 else [samp[i] for i in rng.sample(range(len(samp)), 2000)]
        toksets = [set(_tok(_table_blob(t))) for t in sub]
        scores = []
        for q in qs:
            qt = set(_tok(q))
            if not qt:
                continue
            best = max((len(qt & ts) / len(qt)) for ts in toksets) if toksets else 0.0
            scores.append(best)
        return sum(scores) / len(scores) if scores else 0.0
    ttr = []
    for t in samp[:500]:
        tk = _tok(_table_blob(t))
        if tk:
            ttr.append(len(set(tk)) / len(tk))
    return sum(ttr) / len(ttr) if ttr else 0.3
