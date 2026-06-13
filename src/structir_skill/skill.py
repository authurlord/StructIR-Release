"""StructIR TableRetrievalSkill — the agent-facing toolkit.

Lifecycle (the paper's "offline fit, online retrieve"):
    fit(corpus)  → profile φ(D) → FROZEN PLAN: registered signals, SA-RRF weights,
                   serialization mode, output stage (rerank vs sa_rrf) → build indexes
    retrieve(q)  → execute the frozen plan (deterministic per query)
    as_tool()    → function spec exposing the configured skill to an LLM agent

The frozen plan implements the T0 mapping (analysis/profile_plan_notes.md):
    rho_link  >= 0.5 → Graph candidate signal registered (deployed only with a ckpt)
    rho_header>= 0.4 → struct (header-path) serialization for the reranker
    rho_idf   → fusion bias via the SARRFPolicy (BM25-heavy vs Dense-heavy)
    rho_len   >= 0.6 → output stage Rerank; else stop at SA-RRF (the committed
                       when-NOT-to-rerank behavior: the 512-token cross-encoder
                       truncates long financial tables, MultiHierTT 71.4→55.6)

Design goal: a data agent calls .as_tool() and never has to know *which* retrieval
strategy a corpus needs — the offline-trained policy allocates the weights from the
observable corpus profile. Disjoint from a separate QA system (no QA reader / sufficiency / router here).
"""
from __future__ import annotations
import json, os
from .profile import profile_corpus, CorpusProfile
from .policy import SARRFPolicy
from .signals import BM25Signal, DenseSignal, CachedDenseSignal, GNNSignal, sa_rrf_fuse, _serialize


class TableRetrievalSkill:
    def __init__(self,
                 dense_model="BAAI/bge-m3",
                 reranker_model="BAAI/bge-reranker-v2-m3",
                 reranker_ckpt: str | None = None,      # offline-trained StructIR reranker
                 gnn_ckpt: str | None = None,           # offline-trained cell-entity GNN
                 policy: SARRFPolicy | None = None,     # offline-trained SA-RRF weight policy
                 device="cpu", use_reranker="auto",
                 reranker_max_length=512, reranker_doc_chars=2500, reranker_fp16=False,
                 shared_reranker=None):
        self.dense_model, self.reranker_model = dense_model, reranker_model
        self.reranker_ckpt, self.gnn_ckpt = reranker_ckpt, gnn_ckpt
        self.policy = policy or SARRFPolicy(mode="heuristic")
        self.device = device
        # use_reranker: "auto" → frozen plan decides (rho_len threshold);
        #               True/False → explicit override (logged in card)
        self.use_reranker = use_reranker
        self.reranker_max_length = reranker_max_length
        self.reranker_doc_chars = reranker_doc_chars
        self.reranker_fp16 = reranker_fp16
        self.profile: CorpusProfile | None = None
        self.plan: dict | None = None
        self.weights: dict | None = None
        self.signals: dict = {}
        self.tables_by_id: dict = {}
        self._fold_w: dict = {}
        self.policy_mode = (policy or SARRFPolicy()).mode
        self._reranker = shared_reranker   # may be shared across lake sources

    # ----------------------------- offline / fit -----------------------------
    def fit(self, corpus: list[dict], queries: list[str] | None = None,
            corpus_format="structure",
            dense_cache: dict | None = None,
            calibration: dict | None = None):
        """Profile the corpus, freeze the retrieval plan, register the signals it
        supports, build indexes, and configure SA-RRF weights from the profile.

        `corpus`: list of table dicts each with an "id" plus structure
        (title/text/headers/cells/links/n_header_levels). `text` should be the
        linearized serialization the committed pipeline retrieves over.

        `dense_cache` (optional): pass precomputed corpus embeddings to AVOID
        re-encoding a large corpus. Shape: {"corpus_emb": np.ndarray(N,d),
        "corpus_ids": [id,...], "query_emb_by_id": {qid: vec}? }. When given, the dense
        leg becomes a CachedDenseSignal (loads embeddings; encodes queries live at agent
        time). Everything else — profile, plan, SA-RRF, reranker — runs exactly the same.

        `calibration` (optional): a calibration split with relevance labels —
        {"queries": {qid: text}, "qrels": {qid: {doc_id: rel}},
         "leg_top100": {signal: {qid: [doc_id,...]}}?}. When given, the SA-RRF
        weights are LEARNED by the committed leakage-free 2-fold cross-fit
        (alternating folds over sorted qids, {0,0.1..1.0}^2 grid, each fold
        scored with the other fold's weights — leakage-free cross-fit protocol).
        Per-qid fold weights are stored for leakage-free evaluation of the
        calibration queries themselves; UNSEEN queries get the fold-mean.
        Without calibration the label-free profile heuristic is used.
        """
        assert corpus and "id" in corpus[0], "each table needs an 'id'"
        self.tables_by_id = {t["id"]: t for t in corpus}
        self.profile = profile_corpus(corpus, queries)
        self.plan = self.profile.plan()

        # register signals available on this corpus
        self.signals = {"bm25": BM25Signal().index(corpus)}
        try:
            if dense_cache is not None:
                dense = CachedDenseSignal(
                    dense_cache["corpus_emb"], dense_cache["corpus_ids"],
                    model_path=self.dense_model, device=self.device,
                    query_emb_by_id=dense_cache.get("query_emb_by_id"),
                ).index(corpus)
            else:
                dense = DenseSignal(self.dense_model, self.device).index(corpus)
            self.signals["dense"] = dense
            # Graph: profile-registered candidate; deployed only when a ckpt exists
            if "graph" in self.plan["signals"] and self.gnn_ckpt:
                self.signals["gnn"] = GNNSignal(self.gnn_ckpt, dense).index(corpus)
        except Exception as e:                                   # minimal env → BM25 only
            self._warn(f"dense/gnn unavailable ({e}); running BM25-only")

        self.weights = self.policy.weights(self.profile, set(self.signals))
        self._fold_w = {}
        self.policy_mode = self.policy.mode
        if calibration:
            self._cross_fit(calibration)
        return self

    # ---------------- calibration: leakage-free 2-fold cross-fit ----------------
    def _cross_fit(self, calibration: dict, kappa=60, grid=None):
        """Leakage-free 2-fold cross-fit protocol: alternating folds over sorted qids;
        per-fold best weights on a {0,0.1..1.0}^2 grid; each calibration query is
        scored with the OTHER fold's weights (stored in self._fold_w). The frozen
        per-corpus weights (self.weights) = L1-normalized fold mean."""
        grid = grid or [x / 10 for x in range(11)]
        qrels = {q: {d for d, s in dd.items() if float(s) > 0}
                 for q, dd in calibration["qrels"].items()}
        queries = calibration["queries"]
        provided = calibration.get("leg_top100") or {}
        sigs = [s for s in ("bm25", "dense") if s in self.signals]
        legs = {}
        for name in sigs:
            if name in provided:
                legs[name] = provided[name]
            else:
                sig = self.signals[name]
                legs[name] = {}
                for qid, qtext in queries.items():
                    if qid not in qrels:
                        continue
                    if name == "dense" and isinstance(sig, CachedDenseSignal):
                        ranked = sig.search(qtext, 100, qid=qid)
                    else:
                        ranked = sig.search(qtext, 100)
                    legs[name][qid] = [d for d, _ in ranked]
        qids = sorted(q for q in qrels
                      if all(q in legs[s] for s in sigs) and qrels[q])
        if len(qids) < 10:
            self._warn(f"calibration too small ({len(qids)}); keeping heuristic")
            return
        folds = [set(qids[::2]), set(qids[1::2])]

        def rrf_rank(w, qid):
            sc = {}
            for nm in sigs:
                if w.get(nm, 0) == 0:
                    continue
                for r, d in enumerate(legs[nm][qid]):
                    sc[d] = sc.get(d, 0.0) + w[nm] / (kappa + r + 1)
            return [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])]

        def fold_score(w, fold):
            a = n = 0
            for qid in fold:
                g = qrels[qid]
                a += len(g & set(rrf_rank(w, qid)[:5])) / len(g)
                n += 1
            return a / n if n else 0.0

        import itertools
        combos = [dict(zip(sigs, w)) for w in itertools.product(grid, repeat=len(sigs))
                  if sum(w) > 0]
        best = []
        for fold in folds:
            best.append(max(combos, key=lambda w: fold_score(w, fold)))
        # per-qid weights = the OTHER fold's fit (leakage-free for eval queries)
        for i, fold in enumerate(folds):
            w = best[1 - i]
            z = sum(w.values()) or 1.0
            wn = {s: round(v / z, 4) for s, v in w.items()}
            for qid in fold:
                self._fold_w[qid] = wn
        mean = {s: (best[0].get(s, 0) + best[1].get(s, 0)) / 2 for s in sigs}
        z = sum(mean.values()) or 1.0
        self.weights = {s: round(v / z, 4) for s, v in mean.items()}
        self.policy_mode = "cross-fit"
        self._calib_fold_weights = best
        self._warn(f"cross-fit weights: folds={best} -> frozen mean={self.weights}")

    def _rerank_enabled(self) -> bool:
        if self.use_reranker == "auto":
            return bool(self.plan and self.plan["output_stage"] == "rerank")
        return bool(self.use_reranker)

    # ------------------------------- online ---------------------------------
    def retrieve(self, query: str, top_k=10, first_stage_k=100, rerank_k=100,
                 qid: str | None = None, q_emb=None) -> list[dict]:
        """Execute the frozen plan: SA-RRF first-stage fusion → (plan-gated) reranker
        → top_k. Returns list of {id, score, stage, table} dicts.

        `qid` / `q_emb`: optional fast path for the *unmodified* original query —
        the cached dense leg uses the precomputed embedding instead of re-encoding.
        Any reformulated query string (both omitted) is encoded live."""
        if not self.signals:
            raise RuntimeError("call fit(corpus) first")
        rankings = {}
        for n, s in self.signals.items():
            if n == "dense" and isinstance(s, CachedDenseSignal):
                rankings[n] = s.search(query, first_stage_k, qid=qid, q_emb=q_emb)
            else:
                rankings[n] = s.search(query, first_stage_k)
        # calibration queries use their leakage-free fold weights; unseen → frozen mean
        w = self._fold_w.get(qid, self.weights) if qid else self.weights
        fused = sa_rrf_fuse(rankings, w, k=max(first_stage_k, rerank_k))
        stage = "sa_rrf"
        if self._rerank_enabled():
            fused = self._rerank(query, fused[:rerank_k]); stage = "reranker"
        out = []
        for tid, sc in fused[:top_k]:
            out.append({"id": tid, "score": round(float(sc), 5), "stage": stage,
                        "table": self.tables_by_id.get(tid)})
        return out

    def _load_reranker(self):
        if self._reranker is None:
            from sentence_transformers.cross_encoder import CrossEncoder
            src = self.reranker_ckpt or self.reranker_model
            self._reranker = CrossEncoder(src, device=self.device,
                                          max_length=self.reranker_max_length)
            if self.reranker_fp16:
                try:
                    self._reranker.model.half()  # fp16 inference (V100), matches committed eval
                except Exception as e:
                    self._warn(f"fp16 cast skipped: {e}")
        return self._reranker

    def rerank_pairs(self, query, tids):
        """Score (query, table) pairs with the trained cross-encoder. Exposed so a
        lake can merge candidates from several rerank-stage sources on one scale."""
        rr = self._load_reranker()
        struct = bool(self.plan and self.plan["serialization"] == "struct")
        pairs = [[query, self._table_text(t, structure_aware=struct)] for t in tids]
        return rr.predict(pairs, batch_size=512, convert_to_numpy=True,
                          show_progress_bar=False)

    def _rerank(self, query, cand):
        if not cand: return cand
        try:
            scores = self.rerank_pairs(query, [tid for tid, _ in cand])
        except Exception as e:
            self._warn(f"reranker unavailable ({e}); returning SA-RRF order"); return cand
        order = sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)
        return [(cand[i][0], float(scores[i])) for i in order]

    def _table_text(self, tid, structure_aware=False):
        t = self.tables_by_id.get(tid, {})
        if structure_aware:                          # header-path serialization (financial)
            return _header_path_serialize(t)[: self.reranker_doc_chars]
        # If the table dict carries a precomputed linearized "text" (the corpus
        # serialization the dense/reranker pipeline was committed on), use it verbatim
        # so retrieve() reproduces the committed reranker input exactly.
        if t.get("text"):
            return str(t["text"])[: self.reranker_doc_chars]
        return _serialize(t)[: self.reranker_doc_chars]

    # ------------------------------ agent tool -------------------------------
    def as_tool(self) -> dict:
        """Return an OpenAI/Anthropic-style function spec so an LLM agent can call this skill."""
        fam = self.profile.family if self.profile else "unknown"
        return {
            "type": "function",
            "function": {
                "name": "structir_retrieve_tables",
                "description": (
                    "Retrieve the most relevant tables from a heterogeneous table corpus for a "
                    f"natural-language query. This corpus was profiled as source family '{fam}'; "
                    "the skill auto-configured source-adaptive retrieval weights "
                    f"{self.weights} and output stage "
                    f"'{(self.plan or {}).get('output_stage','?')}'. Returns ranked tables "
                    "with ids and scores. Pass the user's question VERBATIM as the query "
                    "unless the previous result was clearly irrelevant."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "the information need"},
                        "top_k": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            "_structir_meta": {"profile": self.profile.as_dict() if self.profile else None,
                               "plan": self.plan,
                               "weights": self.weights, "signals": list(self.signals)},
        }

    def card(self) -> dict:
        """Human-readable summary of what fit() configured (for logging / agent transparency)."""
        return {"profile": self.profile.as_dict() if self.profile else None,
                "family": self.profile.family if self.profile else None,
                "frozen_plan": self.plan,
                "signals_registered": list(self.signals),
                "sa_rrf_weights": self.weights,
                "policy_mode": self.policy_mode,
                "calibrated": bool(self._fold_w),
                "reranker_enabled": self._rerank_enabled() if self.plan else None,
                "reranker_override": (None if self.use_reranker == "auto"
                                      else bool(self.use_reranker))}

    def _warn(self, m): print(f"[structir] WARN: {m}")


def _header_path_serialize(t: dict) -> str:
    """Structure-aware serialization for hierarchical/financial tables (the genuine-gain
    variant from StructIR §5.3): TopHeader > SubHeader : cell."""
    parts = [f"TABLE: {t.get('title','')}"]
    h = t.get("headers") or []
    if h and isinstance(h[0], (list, tuple)):           # multi-level header
        for col in zip(*h):
            parts.append("HEADER_PATH: " + " > ".join(str(x) for x in col))
    else:
        parts.append("HEADERS: " + " | ".join(str(x) for x in h))
    for row in (t.get("rows") or [])[:50]:
        parts.append("ROW: " + " | ".join(str(c) for c in row))
    if not t.get("rows"):
        body = " ".join(str(c) for c in (t.get("cells") or [])[:200])
        if not body and t.get("text"):
            body = str(t["text"])[:3000]
        parts.append(body)
    return "\n".join(parts)[:4000]
