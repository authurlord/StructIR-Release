#!/usr/bin/env python3
"""UNIVERSAL GNN: train ONE TableRetrievalGNN on the nqtables graph + nqtables
TRAIN-split queries (the only wiki dataset with a real HF train split), then
apply the SAME weights to the FULL eval set of all 4 Wikipedia datasets
(ottqa/nqtables/openwikitables/fetaqa). For the 3 datasets it never trained on
this is pure zero-shot; for nqtables it's leakage-free (train split != dev split).

Architecture mirrors gnn_scripts/train_eval_structir_gnn.py:
  residual HGTConv table<->entity bipartite, table_out = bge-m3 + sigmoid(gate)*delta,
  init_gate=-5, query stays in bge-m3 space via near-identity q_lin.

Attribution (codex hardening lesson): we train TWO models on the SAME nqtables
train data:
  gnn_full   : full graph message passing + gate + q_lin
  qproj_only : graph DISABLED (table emb == raw bge-m3), only q_lin trains
At eval on each dataset's FULL eval we report dense / gnn_full / qproj_only and
the genuine GRAPH contribution = gnn_full - qproj_only.

RUN ON the GPU host env a torch2.4+PyG env (torch2.4+PyG). IDLE GPU only.
Outputs: gnn_universal/<ds>_full_eval.json ; gnn_universal/universal_summary.json
"""
import os, sys, json, time, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

BASE = os.environ.get("STRUCTIR_B", "./assets")
EMB = f"{BASE}/emb"; QRELS = f"{BASE}/qrels"; GRAPHS = f"{BASE}/gnn/graphs"
TRAIN = f"{BASE}/train_data"; OUT = f"{BASE}/gnn_universal"
KS = [1, 5, 10, 20]
TRAIN_DS = "nqtables"
EVAL_DS = ["ottqa", "nqtables", "openwikitables", "fetaqa"]

def _norm(d): return str(d).replace(" ", "_")

HIDDEN = int(os.environ.get("GNN_HIDDEN", "128"))
N_HEADS = int(os.environ.get("GNN_HEADS", "2"))

class GNN(nn.Module):
    def __init__(self, metadata, in_dim=1024, hidden=HIDDEN, n_layers=2, n_heads=N_HEADS,
                 init_gate=-5.0, graph_off=False):
        super().__init__()
        self.graph_off = graph_off
        if not graph_off:
            self.in_lin = nn.ModuleDict({nt: nn.Linear(in_dim, hidden) for nt in metadata[0]})
            self.convs = nn.ModuleList([HGTConv(hidden, hidden, metadata, heads=n_heads) for _ in range(n_layers)])
            self.out_lin = nn.ModuleDict({nt: nn.Linear(hidden, in_dim) for nt in metadata[0]})
            self.gate = nn.ParameterDict({nt: nn.Parameter(torch.tensor(init_gate)) for nt in metadata[0]})
        self.q_lin = nn.Linear(in_dim, in_dim)
        nn.init.eye_(self.q_lin.weight); nn.init.zeros_(self.q_lin.bias)
    def forward(self, x_dict, edge_index_dict):
        if self.graph_off: return x_dict
        h = {nt: self.in_lin[nt](x) for nt, x in x_dict.items()}
        for conv in self.convs:
            h = conv(h, edge_index_dict); h = {k: F.gelu(v) for k, v in h.items()}
        delta = {nt: self.out_lin[nt](v) for nt, v in h.items()}
        return {nt: x + torch.sigmoid(self.gate[nt]) * delta[nt] for nt, x in x_dict.items()}
    def encode_query(self, q): return self.q_lin(q)

def load_graph(ds, device):
    g = torch.load(f"{GRAPHS}/{ds}_graph.pt", map_location="cpu", weights_only=False)
    table_ids = g["table_ids"]
    cids = json.load(open(f"{EMB}/{ds}/corpus_ids.json"))
    assert [_norm(t) for t in table_ids] == [_norm(c) for c in cids] or table_ids == cids, "table order mismatch"
    corpus = np.load(f"{EMB}/{ds}/corpus.npy").astype(np.float32)
    dim = corpus.shape[1]
    src = g["t2e_src"].numpy(); dst = g["t2e_dst"].numpy(); n_ent = len(g["entity_list"])
    ent = np.zeros((n_ent, dim), np.float32); deg = np.zeros(n_ent, np.float32)
    np.add.at(ent, dst, corpus[src]); np.add.at(deg, dst, 1.0)
    deg = np.maximum(deg, 1.0); ent /= deg[:, None]
    ent /= (np.linalg.norm(ent, axis=1, keepdims=True) + 1e-9)
    data = HeteroData()
    data["table"].x = torch.from_numpy(corpus); data["entity"].x = torch.from_numpy(ent)
    data[("table","has_entity","entity")].edge_index = torch.from_numpy(np.stack([src, dst]))
    data[("entity","in_table","table")].edge_index = torch.from_numpy(np.stack([dst, src]))
    return data.to(device), table_ids, corpus, g["profile"]

def recall(ranked, gold, ks):
    g = set(gold); ng = len(g) or 1
    return {k: len(g & set(ranked[:k]))/ng for k in ks}

def agg(rl, qrels, ks=KS):
    a = {k: 0.0 for k in ks}; n = 0
    for q, gold in qrels.items():
        if q not in rl: continue
        n += 1; r = recall(rl[q], gold, ks)
        for k in ks: a[k] += r[k]
    return {f"R@{k}": round(100*a[k]/max(1,n), 2) for k in ks} | {"n": n}

def export(model, data, qvec_t, qid_to_row, table_ids, which, topk=20):
    model.eval(); rl = {}
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        te = F.normalize(out["table"], dim=-1)
        keep = [q for q in which if q in qid_to_row]
        rows = [qid_to_row[q] for q in keep]
        qb = F.normalize(model.encode_query(qvec_t[rows]), dim=-1)
        for i in range(0, len(keep), 512):
            sims = qb[i:i+512] @ te.t()
            _, idx = sims.topk(topk, dim=-1); idx = idx.cpu().numpy()
            for j, q in enumerate(keep[i:i+512]):
                rl[q] = [_norm(table_ids[int(idx[j,m])]) for m in range(topk)]
    model.train(); return rl

def train_model(graph_off, data, qvec_t, qid_to_row, tid_to_idx, qrels, train_qids,
                in_dim, device, epochs=8, batch=256, lr=1e-3, temp=0.05, seed=1337):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = GNN(data.metadata(), in_dim=in_dim, graph_off=graph_off).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pairs = []
    for q in train_qids:
        if q not in qid_to_row or q not in qrels: continue
        for d in qrels[q]:
            if d in tid_to_idx: pairs.append((qid_to_row[q], tid_to_idx[d]))
    pq = np.array([p[0] for p in pairs]); pt = np.array([p[1] for p in pairs])
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(epochs):
        perm = np.random.permutation(len(pairs))
        for bi in range(0, len(perm)-batch+1, batch):
            b = perm[bi:bi+batch]
            opt.zero_grad()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                qb = F.normalize(model.encode_query(qvec_t[pq[b]]), dim=-1)
                out = model(data.x_dict, data.edge_index_dict)
                te = F.normalize(out["table"], dim=-1)
                logits = qb @ te[pt[b]].t() / temp
                loss = F.cross_entropy(logits, torch.arange(len(b), device=device))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
    torch.cuda.empty_cache()
    return model, len(pairs)

def main():
    for k in ("http_proxy","https_proxy","all_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY"):
        os.environ.pop(k, None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True); t0 = time.time()

    # ---- train on nqtables graph + nqtables TRAIN queries ----
    tdata, ttable_ids, tcorpus, tprof = load_graph(TRAIN_DS, device)
    t_tid_to_idx = {_norm(t): i for i, t in enumerate(ttable_ids)}
    in_dim = tcorpus.shape[1]
    tqvec = np.load(f"{TRAIN}/{TRAIN_DS}/train_queries.npy").astype(np.float32)
    tqids = json.load(open(f"{TRAIN}/{TRAIN_DS}/train_query_ids.json"))
    t_qid_to_row = {q: i for i, q in enumerate(tqids)}
    tqrels = {q: {_norm(d): s for d, s in dd.items()}
              for q, dd in json.load(open(f"{TRAIN}/{TRAIN_DS}/train_qrels.json")).items()}
    tqvec_t = torch.from_numpy(tqvec).to(device)
    print(f"[train] {TRAIN_DS} graph={tprof.get('n_edges')} edges "
          f"train_q={len(tqids)} {tprof}", flush=True)

    m_full, np_full = train_model(False, tdata, tqvec_t, t_qid_to_row, t_tid_to_idx,
                                  tqrels, list(tqrels.keys()), in_dim, device)
    m_qproj, np_q = train_model(True, tdata, tqvec_t, t_qid_to_row, t_tid_to_idx,
                                tqrels, list(tqrels.keys()), in_dim, device)
    gate_t = float(torch.sigmoid(m_full.gate["table"]).item())
    print(f"[train] DONE pairs={np_full} gate_table={gate_t:.4f} {time.time()-t0:.0f}s", flush=True)
    # save universal weights
    torch.save({"state_dict": m_full.state_dict(), "gate_table": gate_t,
                "trained_on": TRAIN_DS, "n_pairs": np_full},
               f"{OUT}/universal_gnn_full.pt")
    torch.save({"state_dict": m_qproj.state_dict()}, f"{OUT}/universal_gnn_qproj.pt")

    # ---- apply to each dataset's FULL eval ----
    summary = {"trained_on": TRAIN_DS, "gate_table": gate_t, "per_dataset": {}}
    for ds in EVAL_DS:
        data, table_ids, corpus, prof = load_graph(ds, device)
        tid_to_idx = {_norm(t): i for i, t in enumerate(table_ids)}
        qvec = np.load(f"{EMB}/{ds}/queries.npy").astype(np.float32)
        qids = json.load(open(f"{EMB}/{ds}/query_ids.json"))
        qid_to_row = {q: i for i, q in enumerate(qids)}
        qrels = {q: {_norm(d): s for d, s in dd.items()}
                 for q, dd in json.load(open(f"{QRELS}/{ds}_qrels.json")).items()}
        qvec_t = torch.from_numpy(qvec).to(device)
        eval_qids = [q for q in qrels if q in qid_to_row and any(d in tid_to_idx for d in qrels[q])]
        qrels_eval = {q: qrels[q] for q in eval_qids}
        # dense
        dense_te = F.normalize(data["table"].x, dim=-1)
        d_rl = {}
        with torch.no_grad():
            rows = [qid_to_row[q] for q in eval_qids]
            qb = F.normalize(qvec_t[rows], dim=-1)
            for i in range(0, len(eval_qids), 512):
                sims = qb[i:i+512] @ dense_te.t()
                _, idx = sims.topk(20, dim=-1); idx = idx.cpu().numpy()
                for j, q in enumerate(eval_qids[i:i+512]):
                    d_rl[q] = [_norm(table_ids[int(idx[j,m])]) for m in range(20)]
        dense_m = agg(d_rl, qrels_eval)
        f_rl = export(m_full, data, qvec_t, qid_to_row, table_ids, eval_qids)
        q_rl = export(m_qproj, data, qvec_t, qid_to_row, table_ids, eval_qids)
        full_m = agg(f_rl, qrels_eval); qproj_m = agg(q_rl, qrels_eval)
        rec = {"dataset": ds, "n_eval": dense_m["n"], "graph_profile": prof,
               "zero_shot": ds != TRAIN_DS,
               "dense": dense_m, "gnn_full": full_m, "qproj_only": qproj_m,
               "graph_genuine_R@5": round(full_m["R@5"]-qproj_m["R@5"], 2),
               "graph_genuine_R@1": round(full_m["R@1"]-qproj_m["R@1"], 2),
               "gnn_minus_dense_R@5": round(full_m["R@5"]-dense_m["R@5"], 2)}
        summary["per_dataset"][ds] = rec
        del data, dense_te, qvec_t; torch.cuda.empty_cache()
        json.dump(rec, open(f"{OUT}/{ds}_full_eval.json", "w"), indent=2)
        print(f"[{ds}] {'ZEROSHOT' if rec['zero_shot'] else 'IN-DOMAIN'} n={dense_m['n']} "
              f"dense R@5={dense_m['R@5']} qproj R@5={qproj_m['R@5']} gnn R@5={full_m['R@5']} "
              f"graph_genuine={rec['graph_genuine_R@5']:+.2f}", flush=True)
    summary["seconds"] = round(time.time()-t0, 1)
    json.dump(summary, open(f"{OUT}/universal_summary.json", "w"), indent=2)
    print("DONE_UNIVERSAL_GNN", flush=True)

if __name__ == "__main__":
    main()
