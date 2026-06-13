import sys, os
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
"""Runnable BM25-only smoke test (no GPU / no model downloads needed).
Verifies: package imports, corpus profiling, SA-RRF weight assignment, retrieve(), as_tool()."""
from structir_skill import TableRetrievalSkill, SARRFPolicy, profile_corpus

# tiny heterogeneous-ish corpus
TABLES = [
    {"id": "t1", "title": "United Airlines 2017 Revenue",
     "headers": [["Financials", "Financials"], ["Metric", "2017"]],
     "n_header_levels": 2, "cells": ["Passenger revenue", "37,000", "Cargo", "1,100"]},
    {"id": "t2", "title": "Delta 2017 Revenue",
     "headers": [["Financials", "Financials"], ["Metric", "2017"]],
     "n_header_levels": 2, "cells": ["Passenger revenue", "39,450", "Cargo", "744"]},
    {"id": "t3", "title": "List of countries by population",
     "headers": ["Country", "Population"], "cells": ["China", "1,400,000,000", "India", "1,390,000,000"]},
]
QUERIES = ["passenger revenue 2017 airline", "most populous country"]

if __name__ == "__main__":
    print("profile:", profile_corpus(TABLES, QUERIES).as_dict())

    # heuristic policy, BM25-only env (dense/reranker degrade to no-op gracefully)
    skill = TableRetrievalSkill(policy=SARRFPolicy(mode="heuristic"),
                                use_reranker=False, device="cpu")
    skill.fit(corpus=TABLES, queries=QUERIES)
    print("card:", skill.card())

    for q in QUERIES:
        hits = skill.retrieve(q, top_k=2)
        print(f"\nQ: {q}")
        for h in hits:
            print(f"  {h['id']}  score={h['score']}  ({h['table']['title']})")

    print("\ntool spec name:", skill.as_tool()["function"]["name"])
    print("OK")
