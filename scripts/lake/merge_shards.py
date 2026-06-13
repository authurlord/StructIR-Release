#!/usr/bin/env python3
"""Merge sharded runner outputs into the base dir and recompute summaries.

Usage: merge_shards.py BASE_DIR SHARD_GLOB           e.g. shard%d / ashard%d
"""
import glob, json, os, sys
from collections import defaultdict

BASE = sys.argv[1]
PAT = sys.argv[2]                 # e.g. "shard*" or "ashard*"
DS = ["ottqa", "nqtables", "aitqa", "multihiertt"]


def summarize(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    s = {"n": n,
         "Hit@5": round(100 * sum(r["hit@5"] for r in rows) / n, 2),
         "R@5": round(100 * sum(r["r@5"] for r in rows) / n, 2),
         "top1_acc": round(100 * sum(r["top1"] for r in rows) / n, 2)}
    rr = [r for r in rows if r.get("routing_correct") is not None]
    if rr:
        s["routing_acc"] = round(100 * sum(r["routing_correct"] for r in rr) / len(rr), 2)
    if any("n_tool_calls" in r for r in rows):
        s["avg_tool_calls"] = round(sum(r.get("n_tool_calls", 0) for r in rows) / n, 3)
        ref = [r for r in rows if r.get("reformulated_final") is not None]
        if ref:
            s["reformulated_final_rate"] = round(
                sum(bool(r["reformulated_final"]) for r in ref) / len(ref), 3)
    return s


names = set()
for d in glob.glob(os.path.join(BASE, PAT)):
    for f in glob.glob(os.path.join(d, "*.jsonl")):
        if os.path.basename(f).startswith(("retrieval_", "agent_")):
            names.add(os.path.basename(f))

for name in sorted(names):
    rows = {}
    for d in sorted(glob.glob(os.path.join(BASE, PAT))):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                rows[(r["dataset"], r["qid"])] = r
    out = os.path.join(BASE, name)
    with open(out, "w") as w:
        for r in rows.values():
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = {ds: summarize([r for r in rows.values() if r["dataset"] == ds]) for ds in DS}
    per["micro"] = summarize(list(rows.values()))
    json.dump(per, open(out.replace(".jsonl", ".summary.json"), "w"), indent=2)
    print(f"{name}: n={len(rows)} micro={per['micro']}")
