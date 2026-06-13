"""Source-Adaptive RRF weight policy — the offline-trained core that turns a corpus
profile φ(D) into per-signal fusion weights α_c. This is the "allocate weights by demand"
component: the agent never picks a retrieval strategy; the policy does, from the profile.

Two modes (both shipped):
  - "heuristic": α from the profile→signal map (StructIR §3.5), label-free.
  - "learned":   α = W·φ̃ per signal, W fit offline on datasets with train splits.

NOTE on normalization: weighted RRF sums w_c/(κ+rank). What matters is the RELATIVE
ratio between signal weights (e.g. the committed financial fit is bm25:dense = 1.0:0.1).
We therefore L1-normalize the raw weights — a softmax here would squash a 10:1 ratio
to ~1.5:1 and silently degrade BM25-heavy corpora toward equal-RRF (a bug the earlier
version had; fixed 2026-06-09)."""
from __future__ import annotations
from dataclasses import dataclass
import json, math
from .profile import CorpusProfile

SIGNALS = ("bm25", "dense", "gnn")


def _feat(p: CorpusProfile) -> list[float]:
    """Normalized profile features φ̃ used by the learned policy."""
    return [
        1.0,                                   # bias
        p.rho_link,                            # entity/hyperlink coverage
        p.rho_header,                          # hierarchy
        p.rho_idf,                             # BM25 effectiveness proxy
        min(1.0, math.log10(max(p.n_corpus, 1)) / 6.0),  # log corpus size ~[0,1]
    ]


@dataclass
class SARRFPolicy:
    mode: str = "heuristic"
    # learned weight matrix W[signal] = vector over _feat dims (used when mode=="learned")
    W: dict | None = None

    # ---- heuristic table (profile-family → signal ratios, label-free) ----
    # Ratios follow the committed 2-fold SA-RRF fits (sa_rrf_CLEAN_bm25dense.json):
    # financial fit = bm25:dense 1.0:0.1 (BM25-heavy); wikipedia fit = 0.1:0.9
    # (dense-heavy); statistical = 0.1:1.0 (BM25 near-useless); docs = mildly
    # BM25-leaning (watsonx fit 0.1:0.15 with n=30 — too small, keep mild).
    _HEUR = {
        "wikipedia":   {"bm25": 0.10, "dense": 0.90, "gnn": 0.0},
        "financial":   {"bm25": 1.00, "dense": 0.10, "gnn": 0.0},
        "statistical": {"bm25": 0.10, "dense": 1.00, "gnn": 0.0},
        "docs":        {"bm25": 0.55, "dense": 0.45, "gnn": 0.0},
        "unknown":     {"bm25": 0.50, "dense": 0.50, "gnn": 0.0},
    }

    def weights(self, profile: CorpusProfile, available: set[str]) -> dict:
        """Return α_c for the signals that are *available* on this corpus.
        GNN is auto-dropped when ρ_link==0 (no graph) even if requested."""
        if profile.rho_link <= 0.0:
            available = {s for s in available if s != "gnn"}
        if self.mode == "learned" and self.W:
            raw = {s: max(0.0, sum(w * x for w, x in zip(self.W[s], _feat(profile))))
                   for s in available if s in self.W}
        else:
            base = self._HEUR.get(profile.family, self._HEUR["unknown"])
            raw = {s: base.get(s, 0.0) for s in available}
        # L1-normalize (preserves the fitted ratios; RRF ranking is scale-invariant)
        z = sum(raw.values()) or 1.0
        return {s: round(v / z, 4) for s, v in raw.items()}

    # ---------------- offline training of the learned policy ----------------
    @staticmethod
    def fit_policy(samples: list[tuple[CorpusProfile, dict]], lr=0.5, steps=400):
        """Fit W so W·φ̃ ≈ the optimal per-signal weights observed offline.
        samples: list of (profile, target_alpha_dict) from train-split datasets.
        Tiny ridge-regression-style fit per signal (few datasets → keep it simple/robust)."""
        dim = len(_feat(samples[0][0])) if samples else 5
        W = {s: [0.0] * dim for s in SIGNALS}
        for _ in range(steps):
            for prof, tgt in samples:
                x = _feat(prof)
                for s in SIGNALS:
                    pred = sum(w * xi for w, xi in zip(W[s], x))
                    err = tgt.get(s, 0.0) - pred
                    W[s] = [w + lr * err * xi for w, xi in zip(W[s], x)]
        return SARRFPolicy(mode="learned", W={s: [round(w, 5) for w in v] for s, v in W.items()})

    def save(self, path):
        json.dump({"mode": self.mode, "W": self.W}, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path):
        d = json.load(open(path)); return cls(mode=d["mode"], W=d.get("W"))
