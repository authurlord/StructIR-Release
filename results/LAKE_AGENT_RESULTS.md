
---

# 2026-06-11 FULL-SCALE qwen run (all eligible queries; supersedes the 150-sample as headline)

n: ottqa 1690 / nqtables 1067 / aitqa 515 / multihiertt 929 (total 4201).
Summaries: `full_qwen/`; per-query jsonl on the GPU host `<asset-root>/results_lake_agent_full/`.

## Retrieval (R@5/top1)

| condition | ottqa | nqtables | aitqa | multihiertt | micro R@5 |
|---|---|---|---|---|---:|
| bm25_flat | 71.2/53.3 | 30.5/17.2 | 33.2/19.6 | 69.4/49.0 | 55.8 |
| dense_flat | 80.1/62.0 | 76.2/44.9 | 39.4/35.3 | 53.0/37.6 | 68.1 |
| cascade_flat | 86.3/70.7 | 87.2/61.1 | 40.0/35.3 | 55.0/35.0 | 73.9 |
| agent_dense | 78.6/60.3 | 76.7/44.8 | 40.6/35.3 | 51.6/35.7 | 67.5 |
| agent_structir | 79.2/65.4 | 87.0/60.2 | 39.0/32.2 | 64.6/41.8 | 73.0 (route 72.2) |
| **structir_auto** | **87.3/70.8** | 87.2/60.3 | 42.6/35.5 | 68.4/47.3 | **77.6 (route 99.0)** |
| oracle family | 87.3/70.8 | 87.3/60.4 | 43.8/36.1 | 70.6/48.8 | 78.3 |
| oracle source | 93.3/80.1 | 87.6/60.8 | 43.9/36.1 | **71.4**/49.5 | 81.0 |

oracle-source per-source = committed single-corpus values (multihiertt 71.4 exact).

## QA (EM, full)

| condition | ottqa | nqtables |
|---|---:|---:|
| agent_dense | 43.7 | 52.1 |
| dense_flat | 46.2 | 51.3 |
| agent_structir | 47.6 | 51.4 |
| cascade_flat | 50.8 | 52.1 |
| **structir_auto** | **52.1** | 51.8 |
| oracle source | 60.0 | 51.9 |

Zero reader errors. an online frontier model 150-sample retained as frontier check (above).
Ops note: first full attempt OOM'd 5/6 shards on one GPU (6x ~5GB models on
32GB); fixed by distributing shards over GPUs 0-3; resume preserved all rows.
