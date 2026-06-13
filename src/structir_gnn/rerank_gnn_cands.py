#!/usr/bin/env python3
"""Stage 2.5 (d) — rerank the GNN-refined candidate set .

Mirrors scripts/rerank_eval.py exactly, but the candidate pool fed to the
trained CrossEncoder is the GNN top-100 (gnn/<ds>_gnn_dev_top100.json) instead
of the dense top-100. This answers: does GNN as a first-stage leg improve the
candidate set the reranker sees? We also recompute the dense->rerank baseline on
the SAME dev qids for an apples-to-apples delta.

Outputs: gnn/<ds>_rerank.json with
  dense_rerank  (dense top100 -> rerank, dev subset)
  gnn_rerank    (gnn  top100 -> rerank, dev subset)
"""
from __future__ import annotations
import os
import json, os, sys, time

BASE = os.environ.get("STRUCTIR_B", "./assets")
EMB = f"{BASE}/emb"
DATA = f"{BASE}/data"
QRELS = f"{BASE}/qrels"
RERANKERS = f"{BASE}/rerankers"
GNN = f"{BASE}/gnn"
OTTQA_RERANKER = os.environ.get("STRUCTIR_RERANKER", "BAAI/bge-reranker-v2-m3")
KS = [1, 5, 10, 20]
MAX_LEN = 512
SCORE_BATCH = 256
MATCH = {"multihiertt", "aitqa"}


def _norm(d):
    return str(d).replace(" ", "_")


def _corpus(ds):
    out = {}
    with open(f"{DATA}/{ds}/corpus_linearized.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[_norm(r["_id"])] = r.get("text", "") or ""
    return out


def _queries(ds):
    return json.load(open(f"{DATA}/{ds}/queries.json"))


def _qrels(ds):
    raw = json.load(open(f"{QRELS}/{ds}_qrels.json"))
    return {q: {_norm(d): s for d, s in dd.items()} for q, dd in raw.items()}


def recall_at_k(ranked, gold, ks):
    g = set(gold); ng = len(g) or 1
    return {k: len(g & set(ranked[:k])) / ng for k in ks}


def evaluate(ranked, qrels, ks=KS):
    agg = {k: 0.0 for k in ks}; n = 0
    for q, gold in qrels.items():
        if q not in ranked:
            continue
        n += 1
        r = recall_at_k(ranked[q], gold, ks)
        for k in ks:
            agg[k] += r[k]
    out = {"n_queries": n}
    for k in ks:
        out[f"R@{k}"] = round(100 * agg[k] / max(1, n), 2)
    return out


def run(ds):
    mpath = OTTQA_RERANKER if ds == "ottqa" else f"{RERANKERS}/{ds}"
    corpus = _corpus(ds)
    queries = _queries(ds)
    qrels = _qrels(ds)
    dense = {q: [(_norm(d), s) for d, s in lst]
             for q, lst in json.load(
                 open(f"{GNN}/{ds}_dense_dev_top100.json")).items()}
    gnn = {q: [(_norm(d), s) for d, s in lst]
           for q, lst in json.load(
               open(f"{GNN}/{ds}_gnn_dev_top100.json")).items()}
    eval_qids = [q for q in qrels if q in dense and q in gnn]
    qrels_sub = {q: qrels[q] for q in eval_qids}

    from sentence_transformers.cross_encoder import CrossEncoder
    model = CrossEncoder(mpath, max_length=MAX_LEN, device="cuda")

    def rerank(cand_lists):
        out = {}
        for q in eval_qids:
            cand = [d for d, _ in cand_lists[q] if d in corpus]
            if not cand:
                out[q] = [d for d, _ in cand_lists[q]]
                continue
            qt = queries.get(q, "") or ""
            scores = model.predict([[qt, corpus[d]] for d in cand],
                                   batch_size=SCORE_BATCH,
                                   show_progress_bar=False, convert_to_numpy=True)
            order = sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)
            out[q] = [cand[i] for i in order]
        return out

    t0 = time.time()
    dense_rr = evaluate(rerank(dense), qrels_sub)
    gnn_rr = evaluate(rerank(gnn), qrels_sub)
    dt = time.time() - t0

    # first-stage candidate recall (does GNN feed a better pool?)
    dense_pool = evaluate({q: [d for d, _ in dense[q]] for q in eval_qids}, qrels_sub)
    gnn_pool = evaluate({q: [d for d, _ in gnn[q]] for q in eval_qids}, qrels_sub)

    res = {
        "dataset": ds, "model_path": mpath, "n_queries": len(eval_qids),
        "seconds": round(dt, 1),
        "dense_pool": dense_pool, "gnn_pool": gnn_pool,
        "dense_rerank": dense_rr, "gnn_rerank": gnn_rr,
        "delta_rerank_R@1": round(gnn_rr["R@1"] - dense_rr["R@1"], 2),
        "delta_rerank_R@5": round(gnn_rr["R@5"] - dense_rr["R@5"], 2),
        "delta_rerank_R@10": round(gnn_rr["R@10"] - dense_rr["R@10"], 2),
    }
    json.dump(res, open(f"{GNN}/{ds}_rerank.json", "w"), indent=2)
    print(f"[{ds}] dense_rr R@5={dense_rr['R@5']} -> gnn_rr R@5={gnn_rr['R@5']} "
          f"(d={res['delta_rerank_R@5']:+.2f}) {dt:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "ALL_PROXY", "all_proxy"):
        os.environ.pop(k, None)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(sys.argv[1])
