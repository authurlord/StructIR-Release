"""Build the mixed table lake + the fixed flat-baseline indexes from cached corpus/embedding/reranker assets.

Lake = 4 sources × (structural corpus, cached bge-m3 embeddings, queries):
    ottqa (8.9K wiki) + nqtables (170K wiki) + aitqa (1.9K fin) + multihiertt (9.9K fin)

Everything reuses committed caches — NO re-encoding. The dense flat baseline is the
row-concatenation of the per-source corpus embeddings (same encoder ⇒ one space)."""
from __future__ import annotations
import json
import numpy as np
from . import config
from structir_skill import TableLakeSkill, SARRFPolicy, load_structured_corpus
from structir_skill.corpus import _norm


def load_queries(ds):
    return json.load(open(config.DATA / ds / "queries.json"))


def load_qrels(ds):
    raw = json.load(open(config.QRELS / f"{ds}_qrels.json"))
    return {q: {_norm(d): float(s) for d, s in dd.items()} for q, dd in raw.items()}


def load_dense_cache(ds):
    corpus_emb = np.load(config.EMB / ds / "corpus.npy").astype(np.float32)
    corpus_ids = [_norm(x) for x in json.load(open(config.EMB / ds / "corpus_ids.json"))]
    qe = np.load(config.EMB / ds / "queries.npy").astype(np.float32)
    qids = list(json.load(open(config.EMB / ds / "query_ids.json")))
    return {"corpus_emb": corpus_emb, "corpus_ids": corpus_ids,
            "query_emb_by_id": {q: qe[i] for i, q in enumerate(qids)}}


def load_corpus(ds):
    lin = config.DATA / ds / "corpus_linearized.jsonl"
    st = config.DATA / ds / "corpus_structure.jsonl"
    return load_structured_corpus(str(lin), str(st) if st.exists() else None)


def _load_leg_rank(path):
    """Committed leg caches: {qid: [docid,...]} or {qid: [[docid,score],...]}."""
    o = json.load(open(path))
    return {q: [_norm(x[0] if isinstance(x, (list, tuple)) else x) for x in l]
            for q, l in o.items()}


def load_calibration(ds, queries, qrels):
    """Calibration split = the dataset's labeled queries + the committed leg
    top-100 caches (pure compute caches of the skill's own legs)."""
    legs = {}
    bm = config.B / "bm25_cache" / f"bm25_top100_{ds}.json"
    dn = config.EMB / ds / "dense_top100.json"
    if bm.exists():
        legs["bm25"] = _load_leg_rank(bm)
    if dn.exists():
        legs["dense"] = _load_leg_rank(dn)
    return {"queries": queries, "qrels": qrels, "leg_top100": legs}


def build_lake(sources=None, device="cuda", verbose=True):
    """Fit the lake (per-source profile → frozen plan → indexes). Returns
    (lake, {source: {queries, qrels, dense_cache}})."""
    sources = sources or config.LAKE_SOURCES
    lake = TableLakeSkill(dense_model=config.BGE_M3,
                          reranker_ckpt=config.RERANKER_CKPT,
                          policy=SARRFPolicy(mode="heuristic"),
                          device=device, reranker_fp16=True)
    assets = {}
    for ds in sources:
        corpus = load_corpus(ds)
        queries = load_queries(ds)
        qrels = load_qrels(ds)
        cache = load_dense_cache(ds)
        calib = load_calibration(ds, queries, qrels)
        card = lake.add_source(ds, corpus,
                               queries=list(queries.values())[:300],
                               dense_cache=cache, calibration=calib)
        assets[ds] = {"queries": queries, "qrels": qrels, "dense_cache": cache}
        if verbose:
            p = card["profile"]
            print(f"[lake] {ds}: n={p['n_corpus']} family={p['family']} "
                  f"plan={card['frozen_plan']['output_stage']}/"
                  f"{card['frozen_plan']['serialization']} "
                  f"weights={card['sa_rrf_weights']} "
                  f"mode={card['policy_mode']}", flush=True)
    return lake, assets


# --------------------- fixed flat baselines over the lake ---------------------
class DenseFlat:
    """One-size-fits-all dense retriever over the merged lake (bge-m3)."""
    def __init__(self, lake_assets, lake):
        embs, ids = [], []
        for ds, a in lake_assets.items():
            embs.append(a["dense_cache"]["corpus_emb"])
            ids.extend(f"{ds}::{t}" for t in a["dense_cache"]["corpus_ids"])
        self.emb = np.vstack(embs).astype(np.float32)
        n = np.linalg.norm(self.emb, axis=1, keepdims=True); n[n == 0] = 1
        self.emb /= n
        self.ids = ids
        self._assets = lake_assets
        self._lake = lake

    def q_emb(self, ds, qid, query):
        v = self._assets[ds]["dense_cache"]["query_emb_by_id"].get(qid)
        if v is None:                      # reformulated query → live encode
            sk = self._lake.sources[ds].signals["dense"]
            v = sk.encode_query(query)
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def search(self, ds, qid, query, k=100):
        q = self.q_emb(ds, qid, query)
        sc = self.emb @ q
        top = np.argpartition(-sc, k)[:k]
        top = top[np.argsort(-sc[top])]
        return [(self.ids[i], float(sc[i])) for i in top]


class BM25Flat:
    """One-size-fits-all BM25 over the merged lake (single index, lake-level IDF)."""
    def __init__(self, lake):
        from rank_bm25 import BM25Okapi
        import re
        self._tok = lambda s: re.findall(r"\w+", (s or "").lower())
        self.ids, corpus = [], []
        for ds, sk in lake.sources.items():
            for tid, t in sk.tables_by_id.items():
                self.ids.append(f"{ds}::{tid}")
                corpus.append(self._tok(t.get("text", "")))
        self.bm25 = BM25Okapi(corpus)

    def search(self, query, k=100):
        sc = self.bm25.get_scores(self._tok(query))
        top = np.argpartition(-sc, k)[:k]
        top = top[np.argsort(-sc[top])]
        return [(self.ids[i], float(sc[i])) for i in top]


class CascadeFlat:
    """Fixed cascade: dense-flat top-100 → the trained cross-encoder, ALWAYS
    (the 'always-rerank' configuration — no source adaptivity)."""
    def __init__(self, dense_flat: DenseFlat, lake):
        self.dense = dense_flat
        self.lake = lake

    def search(self, ds, qid, query, k=100, rerank_k=100):
        cand = self.dense.search(ds, qid, query, k=rerank_k)
        ids = [c[0] for c in cand]
        texts = []
        for lid in ids:
            src, tid = lid.split("::", 1)
            t = self.lake.sources[src].tables_by_id.get(tid, {})
            texts.append(str(t.get("text", ""))[:2500])
        self.lake._ensure_shared_reranker()
        rr = self.lake._shared_reranker
        scores = rr.predict([[query, tx] for tx in texts], batch_size=512,
                            convert_to_numpy=True, show_progress_bar=False)
        order = np.argsort(-scores)[:k]
        return [(ids[i], float(scores[i])) for i in order]
