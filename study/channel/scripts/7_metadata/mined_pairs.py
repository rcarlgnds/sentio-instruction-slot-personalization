import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])

AI = ROOT / "system"
Q2 = ROOT / "study/channel"
RAW = Q2 / "results/raw"
TAB = Q2 / "results/tables"
PX = RAW / "pools_paired_expanded.jsonl"
CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"
PUBLISHED_EVAL = AI / "data/paired_set/eval_paired.json"
GATE_PASS_FILE = RAW / "dits_gate_lolos.json"
ARMS = ["S1", "S2", "B"]

os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
sys.path.insert(0, str(AI))
os.chdir(AI)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ndcg_at_k(order, gold, k=5):
    rels = [1.0 if c in gold else 0.0 for c in order[:k]]
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def compute_metrics(result_rows):
    pw, top1, nd = [], [], []
    for ids, gold, other in result_rows:
        pos = {c: i for i, c in enumerate(ids)}
        g = min((pos[c] for c in gold if c in pos), default=None)
        o = min((pos[c] for c in other if c in pos), default=None)
        if g is not None and o is not None:
            pw.append(1.0 if g < o else 0.0)
        nd.append(ndcg_at_k(ids, gold))
        top1.append(1.0 if ids and ids[0] in gold else 0.0)
    return {"pairwise": statistics.fmean(pw) if pw else None,
            "n_komparabel": len(pw),
            "top1": statistics.fmean(top1), "ndcg": statistics.fmean(nd),
            "n_baris": len(result_rows)}


def mode_gate():
    import copy
    from app.benchmark.eval import rerank as rr
    from app.benchmark.eval.configs import CONFIG_BY_KEY
    from app.benchmark.eval.harness import EvalHarness
    from app.ingestion.embedder import HybridEmbedder
    from app.retrieval import cps
    from app.retrieval.prerequisite_graph import PrerequisiteGraph
    from app.routing.semantic_router import SemanticRouter
    from app.storage.llm_client import get_llm_client
    from app.storage.postgres_client import PostgresAdapter
    from app.storage.qdrant_client import QdrantAdapter

    rows = [json.loads(l) for l in (AI / "data/paired_set/paired_queries.jsonl").open()
            if l.strip()]
    print(f"GATE: {len(rows)} core track rows, eval_paired.py path verbatim")
    embedder = HybridEmbedder(); embedder.load()
    router = SemanticRouter(encoder=embedder); router.load()
    pg = PostgresAdapter()
    cps.load_department_relations(pg)
    graph = PrerequisiteGraph.load(pg)
    harness = EvalHarness(embedder=embedder, qdrant=QdrantAdapter(),
                          ollama=get_llm_client(), prerequisite_graph=graph,
                          router=router)
    result = {a: [] for a in ARMS}
    for i, q in enumerate(rows):
        pf = harness.prefetch(q, time.perf_counter)
        if not pf["results"]:
            continue
        gold = set(q["gold_relevant_chunks"])
        other = set(q["counterpart_chunks"])
        for a in ARMS:
            pool = [copy.copy(r) for r in pf["results"]]
            ranked = rr.rerank(pool, pf["student"], CONFIG_BY_KEY[a],
                               q["query_text"], prerequisite_graph=graph)
            ids = [r.payload.get("chunk_id", "") for r in ranked]
            result[a].append((ids, gold, other))
        if (i + 1) % 26 == 0:
            print(f"  {i+1}/{len(rows)}")

    published = json.loads(PUBLISHED_EVAL.read_text())
    verdict, detail = True, {}
    for a in ARMS:
        m = compute_metrics(result[a])
        t = published[a]
        match = (round(m["pairwise"], 10) == round(t["pairwise"], 10)
                 and round(m["top1"], 10) == round(t["top1"], 10)
                 and round(m["ndcg"], 10) == round(t["ndcg"], 10))
        detail[a] = {"dihitung": m, "terbit": t, "identik": match}
        print(f"  {a}: pairwise {m['pairwise']:.6f} vs published "
              f"{t['pairwise']:.6f} -> {'IDENTICAL' if match else 'DIFFER'}")
        verdict &= match
    if not verdict:
        print("GATE FAIL: the metrics are not identical, so the mined broad track must not be scored.")
        raise SystemExit(2)
    GATE_PASS_FILE.write_text(json.dumps({
        "waktu": datetime.now(timezone.utc).isoformat(),
        "vonis": "identical, 3 arms x 3 metrics", "detail": detail}, indent=2))
    print(f"GATE PASS -> {GATE_PASS_FILE}")


def mode_xpair(px=PX, name="dits_xpair", raw_out=RAW):
    if not GATE_PASS_FILE.exists():
        raise SystemExit("ABORT: replication gate has not passed "
                         "(run --gate first).")
    import numpy as np
    from app.benchmark.eval import rerank as rr
    from app.benchmark.eval.configs import CONFIG_BY_KEY
    from app.config import settings
    from app.domain.models import RetrievalResult, StudentContext
    from app.ingestion.embedder import HybridEmbedder
    from app.retrieval import cps
    from app.retrieval.normalization import CalibrationBounds, normalize_sims
    from app.storage.postgres_client import PostgresAdapter

    bounds = CalibrationBounds.load(settings.calibration_path)
    print(f"production bounds: [{bounds.absolute_min}, {bounds.absolute_max}]")

    pg = PostgresAdapter()
    rel_uuid = pg.fetch_department_relations()
    with pg._conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name FROM departments")
            id2code = {str(r["id"]): r["name"] for r in cur.fetchall()}
    code_relations = {}
    for (a, b), v in rel_uuid.items():
        ka, kb = id2code.get(a), id2code.get(b)
        if ka and kb:
            code_relations[tuple(sorted([ka, kb]))] = v
    cps._DEPT_RELATIONS.clear()
    cps._DEPT_RELATIONS.update(code_relations)
    cps._DEPT_PG_ADAPTER = None
    print(f"relatedness: {len(code_relations)} code pairs, "
          f"default {settings.cps_dept_match_default}")

    meta = {}
    for l in CORPUS.open():
        r = json.loads(l)
        meta[r["_id"]] = (r["metadata"].get("department"),
                          r["metadata"].get("semester"))

    ids = json.load(open(RAW / "emb_korpus_ids.json"))
    emb = np.load(RAW / "emb_korpus.npy")
    idx = {c: i for i, c in enumerate(ids)}
    norms = np.linalg.norm(emb, axis=1)
    print(f"embedding cache: {emb.shape}, mean norm {norms.mean():.4f}")

    rows = [json.loads(l) for l in px.open() if l.strip()]
    per_row = {a: [] for a in ARMS}
    unique_q = sorted({r["query"] for r in rows})
    embedder = HybridEmbedder(); embedder.load()
    print(f"embed {len(unique_q)} unique queries (CPU) ...")
    q_vec = {}
    for i in range(0, len(unique_q), 32):
        batch = unique_q[i:i + 32]
        out = embedder.embed_text_dense(batch)
        for t, v in zip(batch, out):
            q_vec[t] = np.asarray(v, dtype=np.float32)
        if (i // 32) % 5 == 0:
            print(f"  {min(i+32,len(unique_q))}/{len(unique_q)}")

    result = {a: [] for a in ARMS}
    per_dept = {a: defaultdict(list) for a in ARMS}
    n_no_emb = 0
    for r in rows:
        qv = q_vec[r["query"]]
        qn = qv / (np.linalg.norm(qv) or 1.0)
        pool = []
        for c in r["cands"]:
            j = idx.get(c["id"])
            if j is None:
                n_no_emb += 1
                continue
            cv = emb[j]
            simc = float(np.dot(qn, cv / (norms[j] or 1.0)))
            dept, sem = meta.get(c["id"], (None, None))
            pool.append(RetrievalResult(
                chunk_id=c["id"], sim_score=simc,
                payload={"chunk_id": c["id"], "department_id": dept or "",
                         "semester": sem}))
        prof = r["profile"]
        student = StudentContext(
            user_id="xpair", department_id=prof.get("department")
            or prof.get("department_name"),
            current_semester=int(prof.get("current_semester") or 1),
            enrolled_courses=[], role="student")
        normalize_sims(pool, bounds)
        gold, other = set(r["gold"] or []), set(r["counterpart"] or [])
        for a in ARMS:
            import copy
            p2 = [copy.copy(x) for x in pool]
            ranked = rr.rerank(p2, student, CONFIG_BY_KEY[a], r["query"])
            ids_r = [x.payload["chunk_id"] for x in ranked]
            result[a].append((ids_r, gold, other))
            per_row[a].append({"query_id": r["query_id"], "gold": r["gold"],
                                 "counterpart": r["counterpart"], "ranking": ids_r})
            pos = {c: i for i, c in enumerate(ids_r)}
            g = min((pos[c] for c in gold if c in pos), default=None)
            o = min((pos[c] for c in other if c in pos), default=None)
            if g is not None and o is not None:
                per_dept[a][student.department_id].append(1.0 if g < o else 0.0)

    out = {"dibuat": datetime.now(timezone.utc).isoformat(),
           "_apa_ini": "DITS and its ablations on one channel track (the file name says which), scored with the "
                       "production module after the replication gate passed (dits_gate_lolos.json).",
           "gerbang": json.loads(GATE_PASS_FILE.read_text())["vonis"],
           "bounds": [bounds.absolute_min, bounds.absolute_max],
           "flag_produksi": {
               "cps_prox_semester_mode": settings.cps_prox_semester_mode,
               "cps_prox_asym_cross_dept": settings.cps_prox_asym_cross_dept,
               "cps_zero_sim_not_candidate": settings.cps_zero_sim_not_candidate,
               "cps_dept_match_default": settings.cps_dept_match_default},
           "catatan_metode": [
               "Sim = BGE-M3 dense dot product from the expansion cache (the same files that built the "
               "track), normalized with the production normalize_sims",
               "production relatedness remapped from UUID to program code, values unchanged",
               "Struct inert: profiles without enrolled_courses, as on the original track", f"candidates without an embedding skipped: {n_no_emb}"],
           "sumber": {"pools": sha(px)},
           "arms": {}}
    for a in ARMS:
        m = compute_metrics(result[a])
        m["menang_eksak"] = (f"{round(m['pairwise']*m['n_komparabel'])}/"
                             f"{m['n_komparabel']}" if m["pairwise"] is not None
                             else None)
        m["per_departemen"] = {d: {"pairwise": round(statistics.fmean(v), 4),
                                   "n": len(v)}
                               for d, v in sorted(per_dept[a].items())}
        out["arms"][a] = m
        print(f"{a}: pairwise={m['pairwise']:.4f} ({m['menang_eksak']}) "
              f"top1={m['top1']:.4f} ndcg={m['ndcg']:.4f}")

    TAB.mkdir(parents=True, exist_ok=True)
    p = TAB / f"{name}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    if name != "dits_xpair":
        raw_out.mkdir(parents=True, exist_ok=True)
        for a in ARMS:
            with (raw_out / f"{name}_baris_{a}.jsonl").open("w") as f:
                for b in per_row[a]:
                    f.write(json.dumps(b) + "\n")
    (raw_out / f"{name}.manifest.json").write_text(json.dumps({
        "run_id": "dits_xpair_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/7_metadata/mined_pairs.py",
        "output": {"path": f"results/tables/{name}.json", "sha256": sha(p)},
        "sumber": {"pools": sha(px), "emb_ids": sha(RAW / "emb_korpus_ids.json")},
    }, indent=2))
    print(f"-> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--xpair", action="store_true")
    ap.add_argument("--pools", default=None,
                    help="score another frozen broad pool, such as the stratified resample; default is the "
                         "original track")
    ap.add_argument("--nama", default="dits_xpair",
                    help="output table name; a non-default name also writes per-row rankings")
    ap.add_argument("--raw-out", default=None, help="folder for the manifest and per-row files")
    a = ap.parse_args()
    if a.gate:
        mode_gate()
    elif a.xpair:
        mode_xpair(Path(a.pools) if a.pools else PX, a.nama,
                   Path(a.raw_out) if a.raw_out else RAW)
    else:
        raise SystemExit("choose --gate or --xpair")
