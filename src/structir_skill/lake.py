"""TableLakeSkill — StructIR over a heterogeneous table data lake.

A *lake* is a set of named sources. Each source is onboarded with `add_source` →
its own `TableRetrievalSkill.fit()` (profile → frozen plan → indexes). The lake then
answers `retrieve(query, source=...)`:

    source = "<name>"     execute that source's frozen plan
    source = "<family>"   ("wikipedia"/"financial"/...) — run every member source's
                          plan, merge candidates on a comparable scale
    source = "auto"       probe-route: dense scores over the WHOLE lake (single
                          encoder ⇒ comparable), majority-vote the source among the
                          top-PROBE_K, then execute that source's plan

Merging across sources is only done on score scales that are actually comparable:
  - rerank-stage sources share the SAME cross-encoder → merge by reranker score.
  - sa_rrf-stage sources output weighted-RRF mass → merge by RRF score (same κ,
    L1-normalized weights). Mixed-stage merges are refused (route instead) — an
    honest boundary, not a silent hack.

This module contains NO QA reader / router / budget logic (disjoint from a separate QA system);
it is retrieval only, exposed as one agent tool with a `source` argument.
"""
from __future__ import annotations
import numpy as np
from .skill import TableRetrievalSkill
from .policy import SARRFPolicy
from .signals import CachedDenseSignal

PROBE_K = 20


class TableLakeSkill:
    def __init__(self, dense_model="BAAI/bge-m3", reranker_ckpt=None,
                 gnn_ckpt=None, policy: SARRFPolicy | None = None, device="cpu",
                 reranker_fp16=False):
        self.dense_model = dense_model
        self.reranker_ckpt = reranker_ckpt
        self.gnn_ckpt = gnn_ckpt
        self.policy = policy or SARRFPolicy(mode="heuristic")
        self.device = device
        self.reranker_fp16 = reranker_fp16
        self.sources: dict[str, TableRetrievalSkill] = {}
        self._shared_reranker = None       # one cross-encoder shared by all sources
        self._flat_emb = None              # (N_lake, d) for probe routing
        self._flat_src = None              # source name per row
        self._probe_encoder = None

    # ----------------------------- onboarding -------------------------------
    def add_source(self, name: str, corpus: list[dict],
                   queries: list[str] | None = None,
                   dense_cache: dict | None = None,
                   calibration: dict | None = None) -> dict:
        """Onboard one source: fit a per-source skill (profile → frozen plan;
        SA-RRF weights cross-fit when a calibration split is given).
        `queries` also feeds the lake-level routing-confusion statistic.
        Returns the source card."""
        skill = TableRetrievalSkill(
            dense_model=self.dense_model, reranker_ckpt=self.reranker_ckpt,
            gnn_ckpt=self.gnn_ckpt,
            policy=self.policy, device=self.device, use_reranker="auto",
            reranker_fp16=self.reranker_fp16, shared_reranker=self._shared_reranker)
        skill.fit(corpus, queries=queries, dense_cache=dense_cache,
                  calibration=calibration)
        self._source_queries = getattr(self, "_source_queries", {})
        self._source_queries[name] = list(queries or [])[:100]
        # share one loaded cross-encoder across sources (GPU memory)
        if skill._reranker is None and self._shared_reranker is not None:
            skill._reranker = self._shared_reranker
        self.sources[name] = skill
        self._flat_emb = None              # invalidate probe index
        return skill.card()

    def _ensure_shared_reranker(self):
        if self._shared_reranker is None:
            any_skill = next(iter(self.sources.values()))
            self._shared_reranker = any_skill._load_reranker()
            for s in self.sources.values():
                s._reranker = self._shared_reranker

    def _auto_routable(self) -> bool:
        """auto routing needs a shared dense space (CachedDenseSignal) on every
        source; without it (BM25-only lake) the probe cannot run."""
        return all(isinstance(sk.signals.get("dense"), CachedDenseSignal)
                   for sk in self.sources.values())

    # ------------------------------ routing ---------------------------------
    CONFUSION_TH = 0.30          # vote-leak rate above which a family merges

    def _build_probe(self):
        embs, srcs = [], []
        for name, sk in self.sources.items():
            d = sk.signals.get("dense")
            if not isinstance(d, CachedDenseSignal):
                raise RuntimeError(f"probe routing needs a cached dense leg on '{name}'")
            embs.append(d.emb)
            srcs.extend([name] * len(d.ids))
        self._flat_emb = np.vstack(embs).astype(np.float32)
        self._flat_src = np.array(srcs)
        self._probe_encoder = next(
            s.signals["dense"] for s in self.sources.values()
            if isinstance(s.signals.get("dense"), CachedDenseSignal))
        self._measure_route_confusion()

    def _measure_route_confusion(self):
        """Fit-time, label-free ROUTING-CONFUSION statistic per multi-member
        family: probe each member's own sample queries and measure the fraction
        of top-PROBE_K votes leaking to sibling sources. High leak ⇒ the probe
        cannot identify the member for this workload (e.g. two Wikipedia corpora
        whose topical neighborhoods interleave — the bigger corpus swallows the
        vote) ⇒ route() must MERGE the family. Low leak ⇒ content-disjoint
        members ⇒ member-level voting is reliable.

        (v3 used an embedding near-duplicate rate, which measured ~0 across
        differently-serialized Wikipedia corpora and missed the failure entirely
        — confusion is the operative quantity, so measure it directly.)"""
        self._family_confusion = {}
        fams = self.families()
        for fam, members in fams.items():
            if len(members) < 2:
                continue
            leaks = []
            for m in members:
                qs = (getattr(self, "_source_queries", {}) or {}).get(m, [])[:100]
                if not qs:
                    continue
                enc = self.sources[m].signals["dense"]
                leak_total = vote_total = 0
                for q in qs:
                    v = enc.encode_query(q)
                    sc = self._flat_emb @ v
                    kk = min(PROBE_K, len(sc) - 1) if len(sc) > 1 else len(sc)
                    top = (np.argpartition(-sc, kk)[:kk] if len(sc) > kk
                           else np.arange(len(sc)))
                    src = self._flat_src[top]
                    leak_total += int((src != m).sum() - (~np.isin(src, members)).sum())
                    vote_total += PROBE_K - int((~np.isin(src, members)).sum())
                if vote_total:
                    leaks.append(leak_total / vote_total)
            self._family_confusion[fam] = round(max(leaks), 3) if leaks else 0.0

    def route(self, query: str, q_emb=None) -> dict:
        """Probe-route: dense top-PROBE_K over the whole lake, majority vote at the
        FAMILY level. Single shared encoder ⇒ scores comparable across sources.

        Why family, not source: same-family corpora can hold near-duplicate content
        (e.g. two Wikipedia-table corpora), making the query's origin source
        unidentifiable from content — measured 2026-06-10: source-level voting sent
        80% of ottqa queries to the 19×-larger nqtables corpus (R@5 19.3). Voting
        decides the FAMILY (which the profile genuinely distinguishes); within a
        multi-member family the skill merges candidates on the shared comparable
        scale and lets the reranker/RRF mass disambiguate. Single-member families
        still route straight to the source. `q_emb` = precomputed verbatim-query
        embedding (fast path)."""
        if self._flat_emb is None:
            self._build_probe()
        if q_emb is not None:
            q = np.asarray(q_emb, dtype=np.float32)
            q = q / (np.linalg.norm(q) or 1.0)
        else:
            q = self._probe_encoder.encode_query(query, qid=None)
        sc = self._flat_emb @ q
        k = min(PROBE_K, len(sc) - 1) if len(sc) > 1 else len(sc)
        top = np.argpartition(-sc, k)[:k] if len(sc) > k else np.arange(len(sc))
        top = top[np.argsort(-sc[top])]
        votes = {}
        for i in top:
            s = str(self._flat_src[i])
            votes[s] = votes.get(s, 0) + 1
        fams = self.families()
        fam_of = {m: f for f, ms in fams.items() for m in ms}
        fam_votes = {}
        for s, v in votes.items():
            fam_votes[fam_of[s]] = fam_votes.get(fam_of[s], 0) + v
        win_fam = max(fam_votes, key=fam_votes.get)
        members = fams[win_fam]
        confusion = getattr(self, "_family_confusion", {}).get(win_fam, 0.0)
        if len(members) == 1:
            routed = members[0]
        elif confusion >= self.CONFUSION_TH:
            routed = win_fam                 # route-ambiguous family → merge
        else:                                # probe identifies the member reliably
            routed = max(members, key=lambda m: votes.get(m, 0))
        return {"source": routed, "votes": votes, "family_votes": fam_votes,
                "family_confusion": confusion}

    # ------------------------------ families --------------------------------
    def families(self) -> dict[str, list[str]]:
        fam = {}
        for name, sk in self.sources.items():
            fam.setdefault(sk.profile.family, []).append(name)
        return fam

    # ------------------------------ retrieval -------------------------------
    def retrieve(self, query: str, source: str = "auto", top_k: int = 5,
                 first_stage_k: int = 100, rerank_k: int = 100,
                 qid: str | None = None, q_emb=None) -> dict:
        """Retrieve from the lake. Returns {"hits": [...], "routing": {...}}.
        Hit ids are 'source::table_id' so lake-level ids never collide."""
        routing = {"requested": source}
        members = None
        if source in self.sources:
            members = [source]
        else:
            fams = self.families()
            if source in fams:
                members = fams[source]
            else:                                   # auto
                if not self._auto_routable():
                    # no comparable dense space across sources (e.g. BM25-only
                    # lake): fall back to family-merge when one family, else
                    # search all sources and merge on the shared scale.
                    fams_all = self.families()
                    routed = (next(iter(fams_all)) if len(fams_all) == 1
                              else "__all__")
                    members = (list(self.sources) if routed == "__all__"
                               else fams_all[routed])
                    routing.update({"source": routed, "fallback": "no_probe"})
                else:
                    r = self.route(query, q_emb=q_emb)
                    routing.update(r)
                    routed = r["source"]
                    members = (fams[routed] if routed in fams else [routed])
        routing["sources_searched"] = members

        if len(members) == 1:
            sk = self.sources[members[0]]
            hits = sk.retrieve(query, top_k=top_k, first_stage_k=first_stage_k,
                               rerank_k=rerank_k, qid=qid, q_emb=q_emb)
            out = [dict(h, source=members[0], id=f"{members[0]}::{h['id']}",
                        local_id=h["id"]) for h in hits]
            return {"hits": out, "routing": routing}

        # ---- multi-source family merge ----
        stages = {m: (self.sources[m].plan or {}).get("output_stage") for m in members}
        if len(set(stages.values())) != 1:
            raise RuntimeError(f"mixed output stages across {stages}: route to one "
                               "source instead of merging incomparable scales")
        stage = next(iter(set(stages.values())))
        merged = []
        if stage == "rerank":
            self._ensure_shared_reranker()
            # per-source SA-RRF candidates, then ONE shared cross-encoder scale
            for m in members:
                sk = self.sources[m]
                rankings = {}
                for n, s in sk.signals.items():
                    if n == "dense" and isinstance(s, CachedDenseSignal):
                        rankings[n] = s.search(query, first_stage_k, qid=qid,
                                               q_emb=q_emb)
                    else:
                        rankings[n] = s.search(query, first_stage_k)
                from .signals import sa_rrf_fuse
                fused = sa_rrf_fuse(rankings, sk.weights, k=rerank_k)
                cand = [tid for tid, _ in fused[:rerank_k]]
                if cand:
                    scores = sk.rerank_pairs(query, cand)
                    merged.extend((m, tid, float(sc)) for tid, sc in zip(cand, scores))
        else:
            # sa_rrf-stage merge: per-source weighted-RRF *masses* are NOT
            # comparable when the sources' weight vectors differ (measured
            # 2026-06-10: mass-merge dragged aitqa M@5 43.9→29.0 against
            # multihiertt's BM25-heavy masses). Merge in RANK space instead —
            # rank-RRF over each source's own fused list is comparable by
            # construction (same formula, same κ) — and weight each member list
            # by the probe-vote posterior for THIS query (plain rank-RRF ties
            # every same-rank pair, and a stable sort then always prefers the
            # first-enumerated source: measured multihiertt top1 = 0.0).
            from .signals import KAPPA
            pv = self.route(query, q_emb=q_emb)["votes"]
            tot = sum(pv.get(m, 0) for m in members) + len(members)
            for m in members:
                w_m = (pv.get(m, 0) + 1.0) / tot          # Laplace-smoothed share
                sk = self.sources[m]
                hits = sk.retrieve(query, top_k=rerank_k, first_stage_k=first_stage_k,
                                   rerank_k=rerank_k, qid=qid, q_emb=q_emb)
                merged.extend((m, h["id"], w_m / (KAPPA + r))
                              for r, h in enumerate(hits, start=1))
        merged.sort(key=lambda x: -x[2])
        out = []
        for m, tid, sc in merged[:top_k]:
            out.append({"id": f"{m}::{tid}", "local_id": tid, "source": m,
                        "score": round(sc, 5), "stage": stage,
                        "table": self.sources[m].tables_by_id.get(tid)})
        return {"hits": out, "routing": routing}

    # ------------------------------ agent tool ------------------------------
    def cards(self) -> dict:
        return {name: sk.card() for name, sk in self.sources.items()}

    def catalog_text(self) -> str:
        """Compact human/LLM-readable lake catalog (for the agent system prompt)."""
        lines = []
        for name, sk in self.sources.items():
            p, plan = sk.profile, sk.plan
            lines.append(
                f"- source '{name}': family={p.family}, {p.n_corpus} tables, "
                f"rho_link={p.rho_link}, rho_header={p.rho_header}, "
                f"plan={plan['output_stage']}/{plan['serialization']}, "
                f"weights={sk.weights}")
        return "\n".join(lines)

    def as_tool(self) -> dict:
        fams = sorted(self.families())
        names = sorted(self.sources)
        return {
            "type": "function",
            "function": {
                "name": "structir_retrieve_tables",
                "description": (
                    "Retrieve the most relevant tables from a heterogeneous table DATA "
                    "LAKE containing multiple sources. Lake catalog:\n"
                    + self.catalog_text() +
                    "\nChoose `source` by the question's domain: a source family "
                    f"({fams}), a specific source ({names}), or 'auto' to let the "
                    "skill probe-route. Pass the user's question VERBATIM as the "
                    "query unless the previous result was clearly irrelevant."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "the information need"},
                        "source": {"type": "string",
                                   "enum": ["auto"] + fams + names, "default": "auto"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            "_structir_meta": {"cards": self.cards()},
        }
