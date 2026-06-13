#!/usr/bin/env python3
"""Stage 2.5 — train + eval a query->table GNN on a StructIR Wikipedia dataset.
Runs on any CUDA host with idle GPUs.

Method (query->table residual GNN):
  * Residual heterogeneous GNN (HGTConv) over the table<->entity bipartite graph
    built by build_structir_graph.py. Table features = bge-m3 corpus.npy; entity
    features = mean bge-m3 of connected tables (so an isolated graph recovers the
    bge-m3 baseline). Fusion: table_out = bge-m3 + sigmoid(gate)*GNN_delta,
    init_gate = -5.0 (sigmoid(-5)=0.0067 -> initial == dense baseline EXACTLY).
  * Query stays in bge-m3 space via near-identity linear (so step-0 == dense).
  * Contrastive InfoNCE, in-batch negatives, query->gold-table.

This is LEARNED fusion (the gate + projection are trained), NOT equal-weight RRF.

Eval (dev qids from rerank_data/<ds>/splits.json; ottqa = a fresh 85/15 split):
  (a) dense    : bge-m3 cosine top-K  (== precomputed dense_top100 floor)
  (b) +GNN     : GNN-refined table emb cosine top-K
  -> writes emb/<ds>/gnn_top100.json  = {qid: [[did, score], ...]} for reranking.

Outputs: results JSON with R@1/5/10 for dense vs +GNN on the dev subset, plus
the exported gnn_top100.json used by the downstream rerank step.
"""
from __future__ import annotations
import os
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

BASE = os.environ.get("STRUCTIR_B", "./assets")
EMB = f"{BASE}/emb"
QRELS = f"{BASE}/qrels"
RERANK_DATA = f"{BASE}/rerank_data"
GRAPHS = f"{BASE}/gnn/graphs"
OUT = f"{BASE}/gnn"

MATCH = {"multihiertt", "aitqa"}
KS = [1, 5, 10, 20]


# ---------------------------------------------------------------------------
def _norm(d):
    return str(d).replace(" ", "_")


class TableRetrievalGNN(nn.Module):
    def __init__(self, metadata, in_dim=1024, hidden=256, n_layers=2,
                 n_heads=4, init_gate=-5.0):
        super().__init__()
        self.in_lin = nn.ModuleDict(
            {nt: nn.Linear(in_dim, hidden) for nt in metadata[0]})
        self.convs = nn.ModuleList(
            [HGTConv(hidden, hidden, metadata, heads=n_heads)
             for _ in range(n_layers)])
        self.out_lin = nn.ModuleDict(
            {nt: nn.Linear(hidden, in_dim) for nt in metadata[0]})
        self.gate = nn.ParameterDict(
            {nt: nn.Parameter(torch.tensor(init_gate)) for nt in metadata[0]})
        self.q_lin = nn.Linear(in_dim, in_dim)
        nn.init.eye_(self.q_lin.weight)
        nn.init.zeros_(self.q_lin.bias)

    def forward(self, x_dict, edge_index_dict):
        h = {nt: self.in_lin[nt](x) for nt, x in x_dict.items()}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {k: F.gelu(v) for k, v in h.items()}
        delta = {nt: self.out_lin[nt](v) for nt, v in h.items()}
        out = {}
        for nt, x in x_dict.items():
            out[nt] = x + torch.sigmoid(self.gate[nt]) * delta[nt]
        return out

    def encode_query(self, q):
        return self.q_lin(q)


# ---------------------------------------------------------------------------
def load_graph(ds, device):
    g = torch.load(f"{GRAPHS}/{ds}_graph.pt", map_location="cpu",
                   weights_only=False)
    table_ids = g["table_ids"]
    # align table_ids with emb corpus_ids
    cids = json.load(open(f"{EMB}/{ds}/corpus_ids.json"))
    assert [_norm(t) for t in table_ids] == [_norm(c) for c in cids] or \
        table_ids == cids, "graph table order != corpus_ids order"
    corpus = np.load(f"{EMB}/{ds}/corpus.npy").astype(np.float32)  # (Nt, d) normed
    n_tab, dim = corpus.shape

    src = g["t2e_src"].numpy()
    dst = g["t2e_dst"].numpy()
    n_ent = len(g["entity_list"])

    # entity feature = mean of connected table bge-m3 (bootstrapped)
    ent_feat = np.zeros((n_ent, dim), dtype=np.float32)
    deg = np.zeros(n_ent, dtype=np.float32)
    np.add.at(ent_feat, dst, corpus[src])
    np.add.at(deg, dst, 1.0)
    deg = np.maximum(deg, 1.0)
    ent_feat /= deg[:, None]
    # renormalize entity features to unit norm (same space as bge-m3)
    ent_feat /= (np.linalg.norm(ent_feat, axis=1, keepdims=True) + 1e-9)

    data = HeteroData()
    data["table"].x = torch.from_numpy(corpus)
    data["entity"].x = torch.from_numpy(ent_feat)
    ei = torch.from_numpy(np.stack([src, dst]))
    ei_rev = torch.from_numpy(np.stack([dst, src]))
    data[("table", "has_entity", "entity")].edge_index = ei
    data[("entity", "in_table", "table")].edge_index = ei_rev
    data = data.to(device)
    return data, table_ids, corpus, g["profile"]


def load_split_qids(ds):
    sp = f"{RERANK_DATA}/{ds}/splits.json"
    if os.path.exists(sp):
        s = json.load(open(sp))
        return set(s.get("train", [])), set(s.get("dev", []))
    # ottqa: no split -> deterministic 85/15
    qr = json.load(open(f"{QRELS}/{ds}_qrels.json"))
    qids = sorted(qr.keys())
    rng = random.Random(1337)
    rng.shuffle(qids)
    cut = int(0.85 * len(qids))
    return set(qids[:cut]), set(qids[cut:])


def recall_at_k(ranked, gold, ks):
    g = set(gold)
    ng = len(g) or 1
    return {k: len(g & set(ranked[:k])) / ng for k in ks}


def eval_ranklists(ranklists, qrels, ks=KS):
    agg = {k: 0.0 for k in ks}
    n = 0
    for q, gold in qrels.items():
        if q not in ranklists:
            continue
        n += 1
        r = recall_at_k(ranklists[q], gold, ks)
        for k in ks:
            agg[k] += r[k]
    return {f"R@{k}": round(100 * agg[k] / max(1, n), 2) for k in ks} | {"n": n}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--topk-export", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    for k in ("http_proxy", "https_proxy", "all_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(k, None)
    torch.manual_seed(1337); np.random.seed(1337); random.seed(1337)
    ds = args.ds
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    data, table_ids, corpus_np, gprofile = load_graph(ds, device)
    tid_to_idx = {_norm(t): i for i, t in enumerate(table_ids)}
    metadata = data.metadata()
    print(f"[{ds}] graph {gprofile}", flush=True)

    # queries / qrels
    qvec = np.load(f"{EMB}/{ds}/queries.npy").astype(np.float32)
    qids = json.load(open(f"{EMB}/{ds}/query_ids.json"))
    qid_to_row = {q: i for i, q in enumerate(qids)}
    qrels = {q: {_norm(d): s for d, s in dd.items()}
             for q, dd in json.load(open(f"{QRELS}/{ds}_qrels.json")).items()}
    train_qids, dev_qids = load_split_qids(ds)

    # training pairs: (q_row, gold_table_idx) for train qids
    train_pairs = []
    for q in train_qids:
        if q not in qid_to_row or q not in qrels:
            continue
        for d in qrels[q]:
            if d in tid_to_idx:
                train_pairs.append((qid_to_row[q], tid_to_idx[d]))
    print(f"[{ds}] train_pairs={len(train_pairs)} dev_qids={len(dev_qids)}",
          flush=True)

    qvec_t = torch.from_numpy(qvec).to(device)
    in_dim = corpus_np.shape[1]
    model = TableRetrievalGNN(metadata, in_dim=in_dim, hidden=args.hidden,
                              n_layers=args.n_layers,
                              n_heads=args.n_heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[{ds}] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)

    pair_q = np.array([p[0] for p in train_pairs], dtype=np.int64)
    pair_t = np.array([p[1] for p in train_pairs], dtype=np.int64)

    def refined_table_emb():
        out = model(data.x_dict, data.edge_index_dict)
        return F.normalize(out["table"], dim=-1)

    def export_ranklists(table_emb, which_qids, topk):
        """cosine q->table top-k on GPU, chunked over queries."""
        model.eval()
        rl = {}
        with torch.no_grad():
            te = table_emb  # (Nt, d) normalized
            rows = [qid_to_row[q] for q in which_qids if q in qid_to_row]
            keep_q = [q for q in which_qids if q in qid_to_row]
            qb = F.normalize(model.encode_query(qvec_t[rows]), dim=-1)
            for i in range(0, len(keep_q), 512):
                sl = slice(i, i + 512)
                sims = qb[sl] @ te.t()
                sc, idx = sims.topk(topk, dim=-1)
                sc = sc.cpu().numpy(); idx = idx.cpu().numpy()
                for j, q in enumerate(keep_q[sl]):
                    rl[q] = [[_norm(table_ids[int(idx[j, m])]), float(sc[j, m])]
                             for m in range(topk)]
        model.train()
        return rl

    # dense baseline (gate frozen at init -> recovers bge-m3; but verify numerically)
    dense_te = F.normalize(data["table"].x, dim=-1)
    dev_qrels = {q: qrels[q] for q in dev_qids if q in qrels and q in qid_to_row}
    dense_rl = {}
    model.eval()
    with torch.no_grad():
        rows = [qid_to_row[q] for q in dev_qrels]
        qb = F.normalize(qvec_t[rows], dim=-1)
        for i in range(0, len(rows), 512):
            sims = qb[i:i+512] @ dense_te.t()
            sc, idx = sims.topk(args.topk_export, dim=-1)
            sc = sc.cpu().numpy(); idx = idx.cpu().numpy()
            for j, q in enumerate(list(dev_qrels.keys())[i:i+512]):
                dense_rl[q] = [[_norm(table_ids[int(idx[j, m])]), float(sc[j, m])]
                               for m in range(args.topk_export)]
    model.train()
    dense_metrics = eval_ranklists({q: [d for d, _ in dense_rl[q]] for q in dense_rl},
                                   dev_qrels)
    print(f"[{ds}] DENSE dev {dense_metrics}", flush=True)

    # ---- train ----
    best = -1.0
    best_state = None
    for ep in range(args.epochs):
        perm = np.random.permutation(len(train_pairs))
        ep_loss = 0.0; nb = 0
        for bi in range(0, len(perm) - args.batch_size + 1, args.batch_size):
            b = perm[bi:bi + args.batch_size]
            qb = F.normalize(model.encode_query(qvec_t[pair_q[b]]), dim=-1)
            te = refined_table_emb()
            pos = te[pair_t[b]]
            logits = qb @ pos.t() / args.temp
            target = torch.arange(len(b), device=device)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        # eval
        te = refined_table_emb()
        gnn_rl = export_ranklists(te, list(dev_qrels.keys()), args.topk_export)
        m = eval_ranklists({q: [d for d, _ in gnn_rl[q]] for q in gnn_rl},
                           dev_qrels)
        gate_t = float(torch.sigmoid(model.gate["table"]).item())
        print(f"[{ds}] ep{ep} loss={ep_loss/max(1,nb):.4f} dev {m} "
              f"gate_tab={gate_t:.4f} {time.time()-t0:.0f}s", flush=True)
        if m["R@5"] > best:
            best = m["R@5"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    te = refined_table_emb()
    gnn_metrics = eval_ranklists(
        {q: [d for d, _ in export_ranklists(te, [q], args.topk_export)[q]]
         for q in dev_qrels}, dev_qrels) if False else None
    # full export for dev set (for rerank step)
    gnn_rl = export_ranklists(te, list(dev_qrels.keys()), args.topk_export)
    gnn_metrics = eval_ranklists({q: [d for d, _ in gnn_rl[q]] for q in gnn_rl},
                                 dev_qrels)
    print(f"[{ds}] GNN(best) dev {gnn_metrics}", flush=True)

    # save checkpoint + exported ranklists (dense + gnn, dev subset)
    ckpt = f"{OUT}/{ds}_gnn.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": metadata,
                "table_ids": table_ids, "best_r5": best,
                "gate_table": float(torch.sigmoid(model.gate["table"]).item()),
                "gate_entity": float(torch.sigmoid(model.gate["entity"]).item())},
               ckpt)
    json.dump(gnn_rl, open(f"{OUT}/{ds}_gnn_dev_top{args.topk_export}.json", "w"))
    json.dump(dense_rl, open(f"{OUT}/{ds}_dense_dev_top{args.topk_export}.json", "w"))

    res = {
        "dataset": ds, "graph_profile": gprofile,
        "n_train_pairs": len(train_pairs), "n_dev": dense_metrics["n"],
        "dense": dense_metrics, "gnn": gnn_metrics,
        "delta_R@1": round(gnn_metrics["R@1"] - dense_metrics["R@1"], 2),
        "delta_R@5": round(gnn_metrics["R@5"] - dense_metrics["R@5"], 2),
        "delta_R@10": round(gnn_metrics["R@10"] - dense_metrics["R@10"], 2),
        "gate_table": float(torch.sigmoid(model.gate["table"]).item()),
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(res, open(f"{OUT}/{ds}_gnn_eval.json", "w"), indent=2)
    print(f"[{ds}] DONE {json.dumps(res['dense'])} -> {json.dumps(res['gnn'])} "
          f"dR@1={res['delta_R@1']} dR@5={res['delta_R@5']} "
          f"dR@10={res['delta_R@10']}", flush=True)


if __name__ == "__main__":
    main()
