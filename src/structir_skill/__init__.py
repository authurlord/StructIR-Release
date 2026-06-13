"""StructIR — source-adaptive table retrieval skill for LLM data agents.

    from structir_skill import TableRetrievalSkill, SARRFPolicy
    skill = TableRetrievalSkill(reranker_ckpt="...", gnn_ckpt="...")
    skill.fit(corpus=tables, queries=sample_queries)   # profiles corpus, sets SA-RRF weights
    hits = skill.retrieve("which airline had the highest 2017 revenue?", top_k=10)
    tool = skill.as_tool()                              # → LLM function spec for an agent

Independent of a separate QA system: no QA reader, sufficiency verifier, router, or budget operators.
The contribution is the offline-trained SA-RRF weight policy + reranker that the skill
auto-configures from the observable corpus profile φ(D).
"""
from .skill import TableRetrievalSkill
from .policy import SARRFPolicy
from .profile import CorpusProfile, profile_corpus
from .signals import CachedDenseSignal
from .lake import TableLakeSkill
from .corpus import load_structured_corpus

__all__ = ["TableRetrievalSkill", "SARRFPolicy", "CorpusProfile", "profile_corpus",
           "CachedDenseSignal", "TableLakeSkill", "load_structured_corpus"]
__version__ = "0.2.0"
