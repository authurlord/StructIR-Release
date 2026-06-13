"""Hyperlink graph signal.

Builds an entity -> tables map from the corpus and scores tables by how many of
the query's entity tokens point (via that map) into the table.

IBM's ``corpus_structure`` view stores ``cells`` as plain strings (no embedded
``<a href>``), so we extract *entity surface forms* heuristically:

* explicit wiki-style URLs / ``[[Entity]]`` markup if present (future-proofing),
* otherwise capitalized multi-word spans inside cells and ``meta_data``
  (proper-noun-ish entities), plus the table ``title``.

Each such entity becomes a node linking to every table that mentions it. A
query is mapped to its own capitalized spans; the score for a candidate table
is the IDF-weighted count of shared entities (rare bridge entities count more).

Only meaningful for the hyperlink dataset family. For match datasets the entity
map is sparse/empty and ``search`` returns nothing.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

# [[Entity]] or [[Entity|surface]] wiki markup
_WIKI_RE = re.compile(r"\[\[([^\]\|]+)(?:\|[^\]]*)?\]\]")
# /wiki/Entity_Name URL fragments
_URL_RE = re.compile(r"/wiki/([A-Za-z0-9_%().,'\-]+)")
# Capitalized multi-word span (proper-noun-ish), 1-5 tokens
_PROPER_RE = re.compile(r"\b([A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,4})\b")


def _entity_norm(e: str) -> str:
    e = e.replace("_", " ").strip()
    e = re.sub(r"\s+", " ", e)
    return e.lower()


def extract_entities(text: str):
    if not text:
        return set()
    ents = set()
    for m in _WIKI_RE.findall(text):
        ents.add(_entity_norm(m))
    for m in _URL_RE.findall(text):
        ents.add(_entity_norm(m))
    # proper-noun spans (length>=2 chars, not pure number)
    for m in _PROPER_RE.findall(text):
        n = _entity_norm(m)
        if len(n) >= 2 and not n.replace(" ", "").isdigit():
            ents.add(n)
    return ents


def _table_entities(doc: dict):
    ents = set()
    if doc.get("title"):
        ents.add(_entity_norm(str(doc["title"])))
    # join cells into one string for span detection (cell-internal capitals)
    cells = doc.get("cells") or []
    if cells:
        ents |= extract_entities(" . ".join(str(c) for c in cells))
    ents |= extract_entities(str(doc.get("meta_data", "")))
    ents.discard("")
    return ents


class HyperlinkGraph:
    def __init__(self):
        self.entity_to_docs = defaultdict(set)
        self.idf = {}
        self.N = 0

    def build(self, corpus: dict):
        for did, doc in corpus.items():
            for ent in _table_entities(doc):
                self.entity_to_docs[ent].add(did)
        self.N = max(1, len(corpus))
        for ent, docs in self.entity_to_docs.items():
            df = len(docs)
            self.idf[ent] = math.log(1.0 + self.N / df)
        return self

    def search(self, query: str, top_k=100):
        q_ents = extract_entities(query)
        scores = defaultdict(float)
        for ent in q_ents:
            docs = self.entity_to_docs.get(ent)
            if not docs:
                continue
            w = self.idf.get(ent, 0.0)
            for did in docs:
                scores[did] += w
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def build_graph(corpus: dict) -> HyperlinkGraph:
    return HyperlinkGraph().build(corpus)


def search_all(graph: HyperlinkGraph, queries: dict, top_k=100) -> dict:
    return {qid: graph.search(q, top_k) for qid, q in queries.items()}
