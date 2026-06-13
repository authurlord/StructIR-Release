"""Corpus adapters — turn on-disk corpus files into the table dicts the skill expects.

The skill's fit() wants, per table:
    {"id", "text" (linearized serialization the committed pipeline retrieves over),
     "title", "headers", "cells"}           — structural fields drive the profile.

The committed IBM TableIR layout ships two parallel files per dataset:
    corpus_linearized.jsonl : {"_id", "text"}            (what BM25/dense indexed)
    corpus_structure.jsonl  : {"_id", "title", "text", "headers", "cells", ...}

`load_structured_corpus` merges them by id. The earlier live-skill run loaded ONLY
the linearized file, so the profiler saw no links/headers and misclassified both
Wikipedia corpora as `docs` (codex review 2026-06-06). This loader is the fix.
"""
from __future__ import annotations
import json


def _norm(d) -> str:
    """Project-wide doc-id normalization: space → underscore (CRITICAL before qrels)."""
    return str(d).replace(" ", "_")


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_structured_corpus(linearized_path: str, structure_path: str | None = None,
                           max_tables: int | None = None) -> list[dict]:
    """Merge corpus_linearized.jsonl (+ optional corpus_structure.jsonl) into skill
    table dicts. `text` always comes from the linearized file (committed retrieval
    serialization); title/headers/cells come from the structure file when present."""
    struct = {}
    if structure_path:
        for r in iter_jsonl(structure_path):
            tid = _norm(r.get("_id"))
            struct[tid] = {
                "title": r.get("title") or "",
                "headers": r.get("headers") or [],
                "cells": r.get("cells") or [],
            }
    tables = []
    for r in iter_jsonl(linearized_path):
        tid = _norm(r.get("_id"))
        txt = r.get("text", "") or ""
        s = struct.get(tid, {})
        title = s.get("title") or (txt.split("  ")[0].strip()[:200] if txt else tid)
        tables.append({"id": tid, "text": txt, "title": title,
                       "headers": s.get("headers", []), "cells": s.get("cells", [])})
        if max_tables and len(tables) >= max_tables:
            break
    return tables
