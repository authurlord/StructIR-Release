"""Mixed-lake experiment runner.

Conditions (same merged lake, same queries, same reader — only retrieval differs):

  no-LLM retrieval rows:
    bm25_flat        one-size-fits-all BM25 over the merged 190K-table lake
    dense_flat       one-size-fits-all bge-m3 dense over the merged lake
    cascade_flat     dense_flat top-100 → trained cross-encoder, ALWAYS (no adaptivity)
    structir_auto    TableLakeSkill, source="auto"  (skill probe-routes internally)
    structir_family  oracle FAMILY routing (gold source's family)
    structir_source  oracle SOURCE routing (gold dataset)        ← upper bound

  agent rows (genuine ReAct, 35B decides source + query, verbatim-first):
    agent_dense      same agent loop, tool = dense_flat (no skill)
    agent_structir   same agent loop, tool = TableLakeSkill.as_tool()

QA stage (ottqa + nqtables): fixed per-dataset reader over each condition's final
top-5 → EM/F1 (standard QA). Per-query jsonl per condition with resume.

Usage:
    python3 -m structir_agent.runner --phase gates
    python3 -m structir_agent.runner --phase retrieval
    python3 -m structir_agent.runner --phase agent
    python3 -m structir_agent.runner --phase qa
    python3 -m structir_agent.runner --phase report
"""
from __future__ import annotations
import argparse, asyncio, json, time
from pathlib import Path

import numpy as np

from . import config
from .llm import health_check
from .lake_setup import build_lake, DenseFlat, BM25Flat, CascadeFlat, load_queries, load_qrels
from .agent_loop import react_episode, SYSTEM_LAKE, SYSTEM_FLAT
from . import reader as reader_mod

RETRIEVAL_CONDITIONS = ["bm25_flat", "dense_flat", "cascade_flat",
                        "structir_auto", "structir_family", "structir_source"]
AGENT_CONDITIONS = ["agent_dense", "agent_structir"]
QA_CONDITIONS = ["dense_flat", "cascade_flat", "structir_auto",
                 "structir_source", "agent_dense", "agent_structir"]


# ------------------------------ eligibility ------------------------------
def gold_answers(ds):
    if ds == "nqtables":
        return json.load(open(config.DATA / "nqtables/gold_answers.json"))
    if ds == "ottqa":
        rows = [json.loads(l) for l in open(config.DATA / "ottqa/ottqa_dev_strict1690.jsonl")
                if l.strip()]
        return {r["question_id"]: [r["answer_text"]] for r in rows}
    return {}


def eligible_qids(ds, queries, qrels):
    """Deterministic eligible list. QA sources replicate the 2026-06-06 sample
    (sorted ∩ committed-rank ∩ qrels ∩ golds) for cross-run comparability."""
    if ds in config.QA_SOURCES:
        committed = json.load(open(config.UNIVRANK / f"{ds}_reranked_rank.json"))
        golds = gold_answers(ds)
        elig = sorted(q for q in queries if q in committed and q in qrels and q in golds)
    else:
        elig = sorted(q for q in queries if q in qrels)
    return elig


def lake_gold(ds, qrels, qid):
    return {f"{ds}::{d}" for d, s in qrels[qid].items() if s > 0}


def retrieval_metrics(top_ids, gset, k=config.TOP_K):
    inter = gset & set(top_ids[:k])
    hit = 1.0 if inter else 0.0
    r_at_k = len(inter) / len(gset) if gset else 0.0
    top1 = bool(top_ids and top_ids[0] in gset)
    return hit, r_at_k, top1


# ------------------------------ persistence ------------------------------
def jsonl_resume(path: Path):
    done = {}
    if path.exists():
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                done[(r["dataset"], r["qid"])] = r
    return done


def summarize(rows, extra=None):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    s = {
        "n": n,
        "Hit@5": round(100 * sum(r["hit@5"] for r in rows) / n, 2),
        "R@5": round(100 * sum(r["r@5"] for r in rows) / n, 2),
        "top1_acc": round(100 * sum(r["top1"] for r in rows) / n, 2),
    }
    if any("routing_correct" in r for r in rows):
        rr = [r for r in rows if r.get("routing_correct") is not None]
        if rr:
            s["routing_acc"] = round(100 * sum(r["routing_correct"] for r in rr) / len(rr), 2)
    if any("n_tool_calls" in r for r in rows):
        s["avg_tool_calls"] = round(sum(r.get("n_tool_calls", 0) for r in rows) / n, 3)
        ref = [r for r in rows if r.get("reformulated_final") is not None]
        if ref:
            s["reformulated_final_rate"] = round(
                sum(bool(r["reformulated_final"]) for r in ref) / len(ref), 3)
    if extra:
        s.update(extra)
    return s


# ------------------------------ phases ------------------------------
def phase_gates(ctx):
    """GATE 2 — per-source live retrieval must reproduce committed full-eval R@5
    (sampled n=200/source for wall-time; financial sources are cheap, run full)."""
    lake, assets = ctx["lake"], ctx["assets"]
    out = {}
    for ds in ctx.get("gate_sources") or config.LAKE_SOURCES:
        qrels = assets[ds]["qrels"]
        queries = assets[ds]["queries"]
        elig = eligible_qids(ds, queries, qrels)
        sample = elig if ds in ("aitqa", "multihiertt") else elig[:200]
        hits5 = []
        t0 = time.time()
        for qid in sample:
            q = queries[qid]
            emb = assets[ds]["dense_cache"]["query_emb_by_id"].get(qid)
            r = lake.retrieve(q, source=ds, top_k=5, qid=qid, q_emb=emb)
            gset = lake_gold(ds, qrels, qid)
            _, r5, _ = retrieval_metrics([h["id"] for h in r["hits"]], gset)
            hits5.append(r5)
        live = round(100 * sum(hits5) / len(hits5), 2)
        ref = config.COMMITTED_R5[ds]
        out[ds] = {"n": len(sample), "live_R@5": live, "committed_full_R@5": ref,
                   "wall_sec": round(time.time() - t0, 1)}
        print(f"[gate2:{ds}] live R@5={live} (committed full-eval {ref}, "
              f"n={len(sample)})", flush=True)
    json.dump(out, open(ctx["out_dir"] / "gate_retrieval.json", "w"), indent=2)
    return out


async def phase_retrieval(ctx):
    lake, assets = ctx["lake"], ctx["assets"]
    dense_flat: DenseFlat = ctx["dense_flat"]
    bm25_flat: BM25Flat = ctx["bm25_flat"]
    cascade: CascadeFlat = ctx["cascade_flat"]
    out_dir = ctx["out_dir"]

    fam_of = {ds: lake.sources[ds].profile.family for ds in lake.sources}

    for cond in RETRIEVAL_CONDITIONS:
        path = out_dir / f"retrieval_{cond}.jsonl"
        done = jsonl_resume(path)
        fout = open(path, "a")
        all_rows = list(done.values())
        for ds in config.LAKE_SOURCES:
            queries, qrels = assets[ds]["queries"], assets[ds]["qrels"]
            elig = eligible_qids(ds, queries, qrels)
            sample = elig[:config.N_AGENT_QUERIES]
            if ctx.get("shard"):
                i, n = ctx["shard"]
                sample = sample[i::n]
            t0 = time.time()
            for qid in sample:
                if (ds, qid) in done:
                    continue
                q = queries[qid]
                emb = assets[ds]["dense_cache"]["query_emb_by_id"].get(qid)
                routing = None
                if cond == "bm25_flat":
                    top = [i for i, _ in bm25_flat.search(q, k=20)]
                elif cond == "dense_flat":
                    top = [i for i, _ in dense_flat.search(ds, qid, q, k=20)]
                elif cond == "cascade_flat":
                    top = [i for i, _ in cascade.search(ds, qid, q, k=20)]
                else:
                    src = {"structir_auto": "auto",
                           "structir_family": fam_of[ds],
                           "structir_source": ds}[cond]
                    r = lake.retrieve(q, source=src, top_k=20, qid=qid, q_emb=emb)
                    top = [h["id"] for h in r["hits"]]
                    routing = r["routing"]
                gset = lake_gold(ds, qrels, qid)
                hit, r5, top1 = retrieval_metrics(top, gset)
                row = {"dataset": ds, "qid": qid, "condition": cond,
                       "top5": top[:5], "hit@5": hit, "r@5": round(r5, 4),
                       "top1": top1, "routing": routing}
                if routing and "source" in routing:
                    routed = routing["source"]
                    # family-level routing counts as correct for member sources
                    row["routing_correct"] = (routed == ds or routed == fam_of[ds])
                fout.write(json.dumps(row) + "\n")
                fout.flush()
                all_rows.append(row)
            print(f"[retrieval:{cond}:{ds}] done n={len(sample)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        fout.close()
        per_ds = {}
        for ds in config.LAKE_SOURCES:
            per_ds[ds] = summarize([r for r in all_rows if r["dataset"] == ds])
        per_ds["micro"] = summarize(all_rows)
        json.dump(per_ds, open(out_dir / f"retrieval_{cond}.summary.json", "w"),
                  indent=2)
        print(f"[retrieval:{cond}] micro: {per_ds['micro']}", flush=True)


async def phase_agent(ctx):
    lake, assets = ctx["lake"], ctx["assets"]
    dense_flat: DenseFlat = ctx["dense_flat"]
    out_dir = ctx["out_dir"]
    fams = sorted(lake.families())
    catalog = lake.catalog_text()
    sys_lake = SYSTEM_LAKE.format(
        catalog=catalog,
        source_values=", ".join(f'"{f}"' for f in fams + sorted(lake.sources)))

    retrieve_lock = asyncio.Lock()       # GPU retrieval is fast; serialize it

    for cond in AGENT_CONDITIONS:
        path = out_dir / f"agent_{cond}.jsonl"
        done = jsonl_resume(path)
        fout = open(path, "a")
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(config.LLM_CONCURRENCY)
        all_rows = list(done.values())

        async def run_one(ds, qid, q, gset, fam):
            emb = assets[ds]["dense_cache"]["query_emb_by_id"].get(qid)

            if cond == "agent_structir":
                async def tool_fn(query, source, k):
                    verbatim = query.strip() == q.strip()
                    async with retrieve_lock:
                        r = await asyncio.to_thread(
                            lake.retrieve, query, source=source, top_k=k,
                            qid=qid if verbatim else None,
                            q_emb=emb if verbatim else None)
                    return r["hits"], r["routing"]
                system = sys_lake
            else:
                async def tool_fn(query, source, k):
                    verbatim = query.strip() == q.strip()
                    async with retrieve_lock:
                        res = await asyncio.to_thread(
                            dense_flat.search, ds,
                            qid if verbatim else "__live__", query, 20)
                    hits = []
                    for lid, sc in res[:k]:
                        src, tid = lid.split("::", 1)
                        hits.append({"id": lid, "local_id": tid, "source": src,
                                     "score": round(sc, 5), "stage": "dense",
                                     "table": lake.sources[src].tables_by_id.get(tid)})
                    return hits, None
                system = SYSTEM_FLAT

            async with sem:
                ep = await react_episode(q, tool_fn, system)
            top = [h["id"] for h in ep["final_hits"]]
            hit, r5, top1 = retrieval_metrics(top, gset)
            routed = ep.get("final_source")
            routing_correct = None
            if cond == "agent_structir" and routed:
                routing_correct = (routed == ds or routed == fam or
                                   (routed == "auto" and top and
                                    top[0].split("::")[0] == ds))
            lc = ep.get("last_call_hits") or []
            lc_hit, lc_r5, lc_top1 = retrieval_metrics(lc, gset)
            row = {"dataset": ds, "qid": qid, "condition": cond,
                   "question": q,
                   "top5": top[:5], "hit@5": hit, "r@5": round(r5, 4), "top1": top1,
                   "lastcall_hit@5": lc_hit, "lastcall_top1": lc_top1,
                   "n_tool_calls": ep["n_tool_calls"],
                   "final_source": routed,
                   "reformulated_final": ep["reformulated_final"],
                   "routing_correct": routing_correct,
                   "calls": ep["calls"], "steps": ep["steps"]}
            async with lock:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                all_rows.append(row)
            return row

        jobs = []
        for ds in config.LAKE_SOURCES:
            queries, qrels = assets[ds]["queries"], assets[ds]["qrels"]
            elig = eligible_qids(ds, queries, qrels)[:config.N_AGENT_QUERIES]
            if ctx.get("shard"):
                i, n = ctx["shard"]
                elig = elig[i::n]
            fam = lake.sources[ds].profile.family
            for qid in elig:
                if (ds, qid) in done:
                    continue
                jobs.append(run_one(ds, qid, queries[qid],
                                    lake_gold(ds, qrels, qid), fam))
        print(f"[agent:{cond}] {len(jobs)} episodes to run "
              f"({len(done)} resumed)", flush=True)
        t0 = time.time()
        CHUNK = 64
        for i in range(0, len(jobs), CHUNK):
            await asyncio.gather(*jobs[i:i + CHUNK])
            print(f"[agent:{cond}] {min(i+CHUNK, len(jobs))}/{len(jobs)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        fout.close()
        per_ds = {ds: summarize([r for r in all_rows if r["dataset"] == ds])
                  for ds in config.LAKE_SOURCES}
        per_ds["micro"] = summarize(all_rows)
        json.dump(per_ds, open(out_dir / f"agent_{cond}.summary.json", "w"), indent=2)
        print(f"[agent:{cond}] micro: {per_ds['micro']}", flush=True)


async def phase_qa(ctx):
    """Fixed reader over each condition's final top-5 (QA sources only)."""
    lake = ctx["lake"]
    assets = ctx["assets"]
    out_dir = ctx["out_dir"]

    def table_text(lid):
        src, tid = lid.split("::", 1)
        t = lake.sources[src].tables_by_id.get(tid, {})
        return t.get("text", "")

    for cond in QA_CONDITIONS:
        src_file = out_dir / (f"agent_{cond}.jsonl" if cond.startswith("agent_")
                              else f"retrieval_{cond}.jsonl")
        if not src_file.exists():
            print(f"[qa:{cond}] SKIP (no {src_file.name})", flush=True)
            continue
        rows = jsonl_resume(src_file)
        path = out_dir / f"qa_{cond}.jsonl"
        done = jsonl_resume(path)
        fout = open(path, "a")
        lock = asyncio.Lock()
        sem = asyncio.Semaphore(config.LLM_CONCURRENCY)
        out_rows = list(done.values())

        async def qa_one(ds, qid, rrow, golds, queries):
            q = queries[qid]
            top5 = rrow.get("top5") or []
            async with sem:
                if ds == "ottqa":
                    # passage stage needs the ottqa-native table (cell links)
                    top1_local = None
                    for lid in top5[:1]:
                        src, tid = lid.split("::", 1)
                        top1_local = tid if src == "ottqa" else None
                    if top1_local:
                        pred, raw = await reader_mod.answer_ottqa(q, top1_local)
                    else:
                        pred, raw = "", "(top-1 not from ottqa source: no cell-link passages)"
                else:
                    pred, raw = await reader_mod.answer_nqtables(q, top5, table_text)
            err = isinstance(pred, str) and pred.startswith("__ERR__")
            em = 0.0 if err else reader_mod.em_any(pred, golds[qid])
            f1 = 0.0 if err else reader_mod.f1_best(pred, golds[qid])
            row = {"dataset": ds, "qid": qid, "condition": cond,
                   "answer": pred, "gold": golds[qid], "EM": em, "F1": round(f1, 4),
                   "reader_error": bool(err),
                   "hit@5": rrow.get("hit@5"), "top1": rrow.get("top1")}
            async with lock:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                out_rows.append(row)

        jobs = []
        for ds in config.QA_SOURCES:
            golds = gold_answers(ds)
            queries = assets[ds]["queries"]
            for (dds, qid), rrow in rows.items():
                if dds != ds or (ds, qid) in done or qid not in golds:
                    continue
                jobs.append(qa_one(ds, qid, rrow, golds, queries))
        print(f"[qa:{cond}] {len(jobs)} reader calls ({len(done)} resumed)", flush=True)
        t0 = time.time()
        CHUNK = 96
        for i in range(0, len(jobs), CHUNK):
            await asyncio.gather(*jobs[i:i + CHUNK])
            print(f"[qa:{cond}] {min(i+CHUNK, len(jobs))}/{len(jobs)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        fout.close()
        summ = {}
        for ds in config.QA_SOURCES:
            sub = [r for r in out_rows if r["dataset"] == ds]
            n = len(sub)
            if n:
                summ[ds] = {"n": n,
                            "EM": round(100 * sum(r["EM"] for r in sub) / n, 2),
                            "F1": round(100 * sum(r["F1"] for r in sub) / n, 2),
                            "reader_errors": sum(r["reader_error"] for r in sub)}
        json.dump(summ, open(out_dir / f"qa_{cond}.summary.json", "w"), indent=2)
        print(f"[qa:{cond}] {summ}", flush=True)


def phase_report(ctx):
    out_dir = ctx["out_dir"]
    lines = ["# Mixed-lake results (raw collection)\n"]
    for kind in ["retrieval", "agent", "qa"]:
        lines.append(f"\n## {kind}\n")
        for f in sorted(out_dir.glob(f"{kind}_*.summary.json")):
            lines.append(f"### {f.stem}\n```json\n"
                         + json.dumps(json.load(open(f)), indent=2) + "\n```\n")
    (out_dir / "RAW_SUMMARY.md").write_text("\n".join(lines))
    print(f"wrote {out_dir / 'RAW_SUMMARY.md'}")


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["gates", "retrieval", "agent", "qa", "report", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gate-sources", nargs="*", default=None,
                    help="restrict --phase gates to these sources")
    ap.add_argument("--shard", nargs=2, type=int, default=None,
                    metavar=("I", "N"),
                    help="process only eligible qids [I::N] (parallel full-"
                         "scale runs; merge shard out-dirs afterwards)")
    args = ap.parse_args()

    config.unset_proxies()
    out_dir = config.OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in ("agent", "qa", "all"):
        print(await health_check(), flush=True)

    print("[setup] building lake ...", flush=True)
    t0 = time.time()
    lake, assets = build_lake(device=args.device)
    print(f"[setup] lake fit in {time.time()-t0:.0f}s", flush=True)

    ctx = {"lake": lake, "assets": assets, "out_dir": out_dir,
           "gate_sources": args.gate_sources,
           "shard": tuple(args.shard) if args.shard else None}
    need_flat = args.phase in ("retrieval", "agent", "qa", "all")
    if need_flat:
        print("[setup] dense flat ...", flush=True)
        ctx["dense_flat"] = DenseFlat(assets, lake)
        print("[setup] cascade flat ...", flush=True)
        ctx["cascade_flat"] = CascadeFlat(ctx["dense_flat"], lake)
        if args.phase in ("retrieval", "all"):
            print("[setup] BM25 flat (one-time index over the merged lake) ...",
                  flush=True)
            t0 = time.time()
            ctx["bm25_flat"] = BM25Flat(lake)
            print(f"[setup] BM25 flat in {time.time()-t0:.0f}s", flush=True)

    if args.phase in ("gates", "all"):
        phase_gates(ctx)
    if args.phase in ("retrieval", "all"):
        await phase_retrieval(ctx)
    if args.phase in ("agent", "all"):
        await phase_agent(ctx)
    if args.phase in ("qa", "all"):
        await phase_qa(ctx)
    if args.phase in ("report", "all"):
        phase_report(ctx)

    from .llm import dump_usage
    u = dump_usage(out_dir / f"llm_usage_{args.phase}.json")
    if u["calls"]:
        print(f"[usage:{args.phase}] calls={u['calls']} "
              f"in={u['prompt_tokens']/1e6:.2f}M out={u['completion_tokens']/1e6:.3f}M "
              f"cached={u['cached_tokens']/1e6:.2f}M errors={u['errors']}", flush=True)


if __name__ == "__main__":
    asyncio.run(amain())
