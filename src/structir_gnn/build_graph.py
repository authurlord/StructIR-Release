#!/usr/bin/env python3
"""Stage 2.5 — build a table<->entity bipartite graph for a StructIR Wikipedia
dataset (ottqa / nqtables / openwikitables / fetaqa).

IBM's corpus_structure stores cells as plain strings (no <a href>), so the
"hyperlink" / cell-link structure is recovered as *bridge entities*: proper-noun
surface forms shared across tables (identical logic to src/hyperlink_signal.py,
which defines the hyperlink_density profile). Two tables sharing a rare entity
are graph-neighbours; this is the cell-link table<->passage analogue used by
a table<->entity bipartite construction, with "entity" playing the
"passage" role.

Node types:
  table  : index aligned to emb/<ds>/corpus_ids.json  (bge-m3 features added at build time)
  entity : pruned bridge entities (df in [df_min, df_max])

Edges (bidirectional):
  table -- has_entity  --> entity   (edge weight = mention count)
  entity -- in_table   --> table

We DO NOT store node features here (kept compact; features = bge-m3 corpus.npy
added at train time). We store entity df for diagnostics.

Output (compact, ~MB): results/graphs/<ds>_graph.pt with:
  { table_ids: [...], entity_list: [...],
    t2e_src, t2e_dst (int64 arrays), t2e_w (float32),
    entity_df: {ent: df}, profile: {...} }

Run locally (CPU). corpus_structure is large for nqtables (~840MB) -> stream it.
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hyperlink_signal import _table_entities  # noqa: E402

DATA = REPO / "data" / "ibm_tableir"
EMB_IDS = REPO / "results" / "_corpus_ids"  # optional local copy of corpus_ids


def load_corpus_ids(ds: str) -> list[str]:
    """Table id order MUST match emb/<ds>/corpus.npy rows. corpus_structure.jsonl
    on disk is the exact same source the embedder read, in the same order."""
    ids = []
    with (DATA / ds / "corpus_structure.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(json.loads(line)["_id"])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--df-min", type=int, default=2,
                    help="drop singleton entities (no bridge)")
    ap.add_argument("--df-max-frac", type=float, default=0.01,
                    help="drop entities appearing in > this fraction of tables "
                         "(too generic, no discriminative bridge signal)")
    ap.add_argument("--max-cells", type=int, default=400,
                    help="cap cells scanned per table for entity extraction")
    ap.add_argument("--max-table-degree", type=int, default=0,
                    help="if >0, keep only the N rarest (lowest-df) entities per "
                         "table -> bounds total edges for big graphs (nqtables)")
    ap.add_argument("--out-dir", default=str(REPO / "results" / "graphs"))
    args = ap.parse_args()

    ds = args.ds
    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{ds}] streaming corpus_structure ...", flush=True)
    table_ids = []
    # table -> {entity: count}
    table_ent_counts: list[dict] = []
    ent_df = defaultdict(int)
    n = 0
    with (DATA / ds / "corpus_structure.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            table_ids.append(r["_id"])
            doc = {
                "title": r.get("title", "") or "",
                "meta_data": r.get("meta_data", "") or "",
                "cells": (r.get("cells") or [])[: args.max_cells],
            }
            ents = _table_entities(doc)
            counts = {}
            for e in ents:
                counts[e] = counts.get(e, 0) + 1
                ent_df[e] += 1
            table_ent_counts.append(counts)
            n += 1
            if n % 20000 == 0:
                print(f"  ... {n} tables, {len(ent_df)} raw entities "
                      f"{time.time()-t0:.0f}s", flush=True)
    n_tables = len(table_ids)
    print(f"[{ds}] {n_tables} tables, {len(ent_df)} raw entities", flush=True)

    df_max = max(args.df_min, int(args.df_max_frac * n_tables))
    keep = {e for e, df in ent_df.items() if args.df_min <= df <= df_max}
    print(f"[{ds}] keep entities with {args.df_min}<=df<={df_max}: "
          f"{len(keep)} / {len(ent_df)}", flush=True)

    entity_list = sorted(keep)
    ent_to_idx = {e: i for i, e in enumerate(entity_list)}

    src, dst, w = [], [], []
    n_tables_with_edge = 0
    for ti, counts in enumerate(table_ent_counts):
        items = [(e, c) for e, c in counts.items() if e in ent_to_idx]
        if args.max_table_degree > 0 and len(items) > args.max_table_degree:
            # keep the rarest (lowest df) entities -> strongest bridges
            items.sort(key=lambda ec: ent_df[ec[0]])
            items = items[: args.max_table_degree]
        had = False
        for e, c in items:
            src.append(ti)
            dst.append(ent_to_idx[e])
            w.append(float(c))
            had = True
        if had:
            n_tables_with_edge += 1

    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    w = np.asarray(w, dtype=np.float32)

    avg_ent_deg = len(src) / max(1, len(entity_list))
    avg_tab_deg = len(src) / max(1, n_tables)
    profile = {
        "dataset": ds,
        "n_tables": n_tables,
        "n_entities_raw": len(ent_df),
        "n_entities_kept": len(entity_list),
        "n_edges": int(len(src)),
        "df_min": args.df_min,
        "df_max": df_max,
        "frac_tables_with_edge": round(n_tables_with_edge / max(1, n_tables), 4),
        "avg_entity_degree": round(avg_ent_deg, 3),
        "avg_table_degree": round(avg_tab_deg, 3),
    }
    print(f"[{ds}] graph: {profile}", flush=True)

    out = out_dir / f"{ds}_graph.pt"
    torch.save({
        "table_ids": table_ids,
        "entity_list": entity_list,
        "t2e_src": torch.from_numpy(src),
        "t2e_dst": torch.from_numpy(dst),
        "t2e_w": torch.from_numpy(w),
        "profile": profile,
    }, out)
    print(f"[{ds}] wrote {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
