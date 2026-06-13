"""First-stage retrieval signals + RRF fusion. Each signal exposes the same interface:
    .index(tables)            build the index over a corpus
    .search(query, k) -> [(table_id, score), ...]
Graceful: dense/gnn degrade to no-op if their backends aren't installed, so the toolkit
still runs (BM25-only) in a minimal environment."""
from __future__ import annotations
import re
from .profile import _tok

KAPPA = 60  # RRF damping


class BM25Signal:
    name = "bm25"
    def __init__(self): self.bm25 = None; self.ids = []
    def index(self, tables: list[dict]):
        from rank_bm25 import BM25Okapi
        self.ids = [t["id"] for t in tables]
        corpus = [_tok(_serialize(t)) for t in tables]
        self.bm25 = BM25Okapi(corpus)
        return self
    def search(self, query: str, k=100):
        import numpy as np
        sc = self.bm25.get_scores(_tok(query))
        top = np.argsort(-sc)[:k]
        return [(self.ids[i], float(sc[i])) for i in top]


class DenseSignal:
    name = "dense"
    def __init__(self, model_path="BAAI/bge-m3", device="cpu"):
        self.model_path, self.device = model_path, device
        self.model = None; self.ids = []; self.emb = None
    def _load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_path, device=self.device)
    def index(self, tables):
        self._load()
        self.ids = [t["id"] for t in tables]
        self.emb = self.model.encode([_serialize(t) for t in tables],
                                     normalize_embeddings=True, convert_to_numpy=True)
        return self
    def search(self, query, k=100):
        import numpy as np
        self._load()
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sc = self.emb @ q
        top = np.argsort(-sc)[:k]
        return [(self.ids[i], float(sc[i])) for i in top]


class CachedDenseSignal:
    """Dense leg that LOADS precomputed corpus embeddings (bge-m3) instead of
    re-encoding a large corpus at fit() time. This is the only adaptation needed to
    run the skill on big corpora (e.g. 170K NQTables) without paying the re-encode cost.

    The retrieval math is identical to DenseSignal: cosine over L2-normalized vectors.
    Query encoding is REAL at agent time — the live bge-m3 model encodes whatever query
    string the agent passes (it may have reformulated the question), so the ReAct loop is
    genuine. A precomputed query-embedding cache (keyed by the original query id) can be
    supplied as a fast path for the *unmodified* original query; any other query string
    triggers a real bge-m3 encode. This keeps the dense leg faithful to the committed
    `dense_top100.json` for the original query while still exercising the encoder for
    reformulations.
    """
    name = "dense"

    def __init__(self, corpus_emb, corpus_ids, model_path="BAAI/bge-m3", device="cpu",
                 query_emb_by_id=None):
        import numpy as np
        self.emb = np.asarray(corpus_emb, dtype=np.float32)
        # ensure L2-normalized (cached bge-m3 corpus.npy is already normalized, but be safe)
        norms = np.linalg.norm(self.emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.emb = self.emb / norms
        self.ids = list(corpus_ids)
        self.model_path, self.device = model_path, device
        self.model = None
        self.query_emb_by_id = query_emb_by_id or {}  # {qid: np.ndarray(d,)}

    def index(self, tables):
        # corpus already indexed via the supplied embeddings; nothing to build.
        return self

    def _load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_path, device=self.device)

    def encode_query(self, query, qid=None):
        import numpy as np
        if qid is not None and qid in self.query_emb_by_id:
            v = np.asarray(self.query_emb_by_id[qid], dtype=np.float32)
            n = np.linalg.norm(v) or 1.0
            return v / n
        self._load()
        return self.model.encode([query], normalize_embeddings=True,
                                 convert_to_numpy=True)[0].astype(np.float32)

    def search(self, query, k=100, qid=None, q_emb=None):
        import numpy as np
        if q_emb is not None:
            q = np.asarray(q_emb, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
        else:
            q = self.encode_query(query, qid=qid)
        sc = self.emb @ q
        top = np.argsort(-sc)[:k]
        return [(self.ids[i], float(sc[i])) for i in top]


class GNNSignal:
    """Cell-entity graph signal. Only registers when the corpus exposes hyperlinks
    (ρ_link>0). Consumes a GNN-refined table embedding
    (table_out = dense + sigmoid(gate)·GNN_delta) exported by the training code in
    structir_gnn/ (HGTConv + gated residual + InfoNCE). If no checkpoint, no-op."""
    name = "gnn"
    def __init__(self, ckpt=None, dense: "DenseSignal | None" = None):
        self.ckpt, self.dense, self.ids, self.emb = ckpt, dense, [], None
    def index(self, tables):
        if not self.ckpt or self.dense is None or self.dense.emb is None:
            return self  # no-op; fusion will skip gnn
        import numpy as np, torch
        self.ids = list(self.dense.ids)
        delta = _gnn_delta(self.ckpt, self.dense.emb, tables)  # (N,d) refinement
        refined = self.dense.emb + delta
        refined /= (np.linalg.norm(refined, axis=1, keepdims=True) + 1e-9)
        self.emb = refined
        return self
    def search(self, query, k=100):
        if self.emb is None: return []
        import numpy as np
        q = self.dense.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sc = self.emb @ q
        top = np.argsort(-sc)[:k]
        return [(self.ids[i], float(sc[i])) for i in top]


def sa_rrf_fuse(rankings: dict[str, list], weights: dict[str, float], k=100) -> list:
    """Source-adaptive weighted RRF over per-signal ranked lists.
    rankings: {signal_name: [(table_id, score), ...]}; weights: {signal_name: α}."""
    fused = {}
    for name, ranked in rankings.items():
        w = weights.get(name, 0.0)
        if w == 0: continue
        for rank, (tid, _) in enumerate(ranked):
            fused[tid] = fused.get(tid, 0.0) + w / (KAPPA + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]


def _serialize(t: dict) -> str:
    """Flat linearization. If the table carries a precomputed linearized `text`
    (the corpus serialization the committed BM25/dense pipeline indexed), use it
    verbatim and UNTRUNCATED so the BM25 leg reproduces the committed floors
    exactly (financial reports average 3-7K chars — truncating at 4K cost
    multihiertt/aitqa ~3-4 M@5 in the 2026-06-09 gate). The structure-aware
    variant lives in the reranker (header-path)."""
    if t.get("text"):
        return str(t["text"])
    parts = [str(t.get("title", ""))]
    h = t.get("headers")
    if h: parts.append(" | ".join(str(x) for x in (h if not isinstance(h[0], (list, tuple)) else h[-1]))) if h else None
    cells = t.get("cells") or []
    parts.append(" ".join(str(c) for c in cells[:300]))
    return " ".join(p for p in parts if p)[:4000]


def _gnn_delta(ckpt, dense_emb, tables):
    """Load the HGTConv refinement (trained by structir_gnn/train_eval_gnn.py).
    Falls back to zeros if torch_geometric / checkpoint unavailable (signal becomes no-op)."""
    import numpy as np
    try:
        import torch
        sd = torch.load(ckpt, map_location="cpu")
        # the shipped checkpoint stores precomputed deltas keyed by table id when the
        # full graph can't be rebuilt at agent-time; prefer that for portability.
        if isinstance(sd, dict) and "delta_by_id" in sd:
            d = sd["delta_by_id"]; dim = dense_emb.shape[1]
            return np.array([d.get(t["id"], [0.0]*dim) for t in tables], dtype=np.float32)
    except Exception:
        pass
    return np.zeros_like(dense_emb)
