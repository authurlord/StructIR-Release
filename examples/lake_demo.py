"""Runnable multi-source LAKE demo (CPU-only, BM25-only legs, no downloads).

Builds a tiny heterogeneous lake — a Wikipedia-style source (entity-rich cells)
and a financial-report source (markdown nested-header tables in text) — then shows
the full skill lifecycle:

    add_source → per-source profile + FROZEN PLAN (different per source!)
    retrieve(source=...) / families / catalog → as_tool()

Run:  python examples/lake_demo.py
"""
import sys, os
# CPU-only demo: never hit the network/HF hub — the dense leg fails fast and the
# skill degrades gracefully to BM25-only (its documented minimal-env behavior).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from structir_skill import TableLakeSkill

WIKI = [
    {"id": "w1", "title": "Dancing with the Stars (American season 5)",
     "headers": ["Celebrity", "Professional", "Result"],
     "cells": ["Marie Osmond", "Jonathan Roberts", "Third place",
               "Helio Castroneves", "Julianne Hough", "Winner"],
     "text": "Dancing with the Stars (American season 5)  Celebrity Professional "
             "Result  Marie Osmond Jonathan Roberts Third place Helio Castroneves "
             "Julianne Hough Winner"},
    {"id": "w2", "title": "List of songs recorded by Marie Osmond",
     "headers": ["Song", "Year", "Album"],
     "cells": ["Paper Roses", "1973", "Paper Roses",
               "Meet Me in Montana", "1985", "There's No Stopping Your Heart"],
     "text": "List of songs recorded by Marie Osmond  Song Year Album  Paper Roses "
             "1973 Paper Roses Meet Me in Montana 1985 There's No Stopping Your Heart"},
]

FIN = [
    {"id": "f1", "title": "", "headers": [], "cells": [],
     "text": "Annual Report of United Airlines for year 2018\n\n"
             "| ($ in millions) | Three Months Ended March 31 |  |\n"
             "|  | 2018 | 2017 |\n|---|---|---|\n"
             "| Net income | 145 | 99 |\n| Passenger revenue | 7,883 | 7,180 |\n"},
    {"id": "f2", "title": "", "headers": [], "cells": [],
     "text": "Annual Report of Delta Air Lines for year 2018\n\n"
             "| ($ in millions) | Year Ended December 31 |  |\n"
             "|  | 2018 | 2017 |\n|---|---|---|\n"
             "| Net income | 3,935 | 3,205 |\n| Cargo revenue | 865 | 744 |\n"},
]

if __name__ == "__main__":
    lake = TableLakeSkill()                       # no GPU models → BM25-only legs
    c1 = lake.add_source("wiki_tables", WIKI,
                         queries=["best known song by Marie Osmond",
                                  "who won season 5"])
    c2 = lake.add_source("fin_reports", FIN,
                         queries=["2018 net income in the March quarter for United"])

    print("\n=== per-source cards (note the DIFFERENT frozen plans) ===")
    for name, card in [("wiki_tables", c1), ("fin_reports", c2)]:
        p, plan = card["profile"], card["frozen_plan"]
        print(f"  {name}: family={card['family']} "
              f"plan={plan['output_stage']}/{plan['serialization']} "
              f"weights={card['sa_rrf_weights']}")

    print("\n=== catalog (what the agent's system prompt sees) ===")
    print(lake.catalog_text())

    print("\n=== retrieve with explicit source ===")
    out = lake.retrieve("2018 net income in the March quarter for United Airlines",
                        source="financial", top_k=1)
    h = out["hits"][0]
    print(f"  Q(financial) -> {h['id']}  (routing={out['routing']})")

    out = lake.retrieve("best known song recorded by Marie Osmond",
                        source="wiki_tables", top_k=1)
    print(f"  Q(wiki)      -> {out['hits'][0]['id']}")

    print("\n=== as_tool spec ===")
    spec = lake.as_tool()
    print("  name:", spec["function"]["name"])
    print("  source enum:", spec["function"]["parameters"]["properties"]["source"]["enum"])
    print("\nOK")
