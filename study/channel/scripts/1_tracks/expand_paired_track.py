import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
AI = ROOT / "system"
CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"
QUERIES = ROOT / "system/data/rads_sample/queries.jsonl"
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"
CACHE = Q2 / "raw/emb_korpus.npy"
CACHE_ID = Q2 / "raw/emb_korpus_ids.json"
POOL_SIZE = 71


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ambang", type=float, default=0.66,
                    help="lower similarity bound, set from the original pairs")
    ap.add_argument("--ambang-atas", type=float, default=0.89,
                    help="upper similarity bound. The original pairs reach at most 0.8858, and candidates above "
                         "it are often copies of the same text")
    ap.add_argument("--jaccard-maks", type=float, default=0.36,
                    help="maximum token overlap. The original pairs reach at most 0.355")
    ap.add_argument("--maks-pasangan", type=int, default=600)
    ap.add_argument("--min-free-mib", type=int, default=3500)
    args = ap.parse_args()

    free = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True).split()[0])
    print(f"free VRAM {free} MiB")
    if free < args.min_free_mib:
        raise SystemExit("ABORT VRAM: wait for the other GPU job to finish")

    lock = json.loads((SHARED / "config_lock.json").read_text())
    split = json.loads((SHARED / "split_manifest.json").read_text())["assignment"]

    ids, text, meta = [], [], {}
    for l in CORPUS.open():
        r = json.loads(l)
        m = r["metadata"]
        t = r.get("text") or ""
        if len(t) < 200:
            continue
        ids.append(r["_id"])
        text.append(t)
        meta[r["_id"]] = m
    print(f"eligible chunks: {len(ids)}")

    sys.path.insert(0, str(AI))
    os.chdir(AI)
    import numpy as np

    if CACHE.exists() and CACHE_ID.exists() and \
            json.loads(CACHE_ID.read_text()) == ids:
        print("using corpus embeddings from cache")
        X = np.load(CACHE)
    else:
        from app.ingestion.embedder import HybridEmbedder
        emb = HybridEmbedder()
        emb.load()
        vecs = []
        B = 64
        for i in range(0, len(text), B):
            vecs.extend(emb.embed_text_dense(text[i:i + B]))
            if (i // B) % 100 == 0:
                print(f"  embed {i}/{len(text)}", flush=True)
        X = np.asarray(vecs, dtype=np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        np.save(CACHE, X)
        CACHE_ID.write_text(json.dumps(ids))
        print(f"embeddings saved to cache: {CACHE}")

    pos = {c: i for i, c in enumerate(ids)}
    per_program = defaultdict(list)
    for c in ids:
        per_program[meta[c].get("department")].append(pos[c])
    program_idx = {d: np.asarray(v) for d, v in per_program.items()}

    query = []
    for l in QUERIES.open():
        q = json.loads(l)
        if split.get(q["query_id"]) not in ("train", "dev"):
            continue
        a = (q.get("gold_anchor") or [{}])[0]
        cid = a.get("canonical_id")
        prof = q.get("student_profile") or {}
        student_dept = prof.get("department") or prof.get("department_name")
        if not cid or cid not in pos or not student_dept:
            continue
        if meta[cid].get("department") != student_dept:
            continue
        query.append((q, cid, student_dept))
    print(f"candidate queries (gold in own program): {len(query)}")

    rows, stats = [], defaultdict(int)
    spread = []
    n_pair = 0
    for q, cid, dept in query:
        if n_pair >= args.maks_pasangan:
            break
        v = X[pos[cid]]
        matches = []
        for d2, idxs in program_idx.items():
            if d2 == dept or d2 is None:
                continue
            sims = X[idxs] @ v
            local_order = np.argsort(-sims)
            for jj in local_order[:20]:
                s2 = float(sims[int(jj)])
                if s2 > args.ambang_atas:
                    continue
                if s2 < args.ambang:
                    break
                cid2 = ids[int(idxs[int(jj)])]
                ta = set(text[pos[cid]].lower().split())
                tb = set(text[pos[cid2]].lower().split())
                jac = len(ta & tb) / max(len(ta | tb), 1)
                if jac > args.jaccard_maks:
                    continue
                matches.append((d2, cid2, s2))
                break
        spread.append(max([p[2] for p in matches], default=0.0))
        if not matches:
            stats["tanpa_padanan"] += 1
            continue
        matches.sort(key=lambda x: -x[2])
        used = matches[:2]
        stats["triple" if len(used) == 2 else "pasangan"] += 1
        n_pair += 1

        all_sims = X @ v
        order = np.argsort(-all_sims)
        pool, seen = [], set()
        required = {cid} | {p[1] for p in used}
        for j in order:
            c = ids[int(j)]
            if c in seen:
                continue
            seen.add(c)
            pool.append(c)
            if len(pool) >= POOL_SIZE - len(required):
                break
        for w in required:
            if w not in seen:
                pool.append(w)
                seen.add(w)
        text_by_id = {c: text[pos[c]] for c in pool}
        cands = [{"id": c, "text": text_by_id[c][:4000]} for c in pool]

        members = [(dept, cid)] + [(d2, c2) for d2, c2, _ in used]
        pid = f"xpair-{n_pair}"
        for d_self, c_self in members:
            counterpart = [c for d, c in members if c != c_self]
            rows.append({
                "query_id": f"{pid}-{d_self.lower()}",
                "pair_id": pid,
                "query": q["query_text"],
                "gold": [c_self],
                "counterpart": counterpart,
                "profile": {"department_name": d_self,
                            "department": d_self,
                            "current_semester": meta[c_self].get("semester")},
                "cands": cands,
                "provenance": {
                    "kueri_asal": q["query_id"],
                    "class_label": q.get("class_label"),
                    "buku": meta[c_self].get("book_slug"),
                    "sim_padanan": [round(s, 4) for _, _, s in used],
                },
            })

    import statistics as st
    spread_nonzero = [s for s in spread if s > 0]
    out = Q2 / "raw/pools_paired_expanded.jsonl"
    with out.open("w") as f:
        for b in rows:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")

    threshold_result = {}
    for t in (0.80, 0.85, 0.88, 0.90, 0.92):
        threshold_result[str(t)] = sum(1 for s in spread if s >= t)

    meta_out = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("Paired track extended by tying it to the main benchmark queries: the original gold "
                     "becomes the answer for the first profile, and a near twin in a book of another program "
                     "becomes the answer for the second profile (and the third, if there is one)."),
        "populasi_sumber": "traindev 2,493; test_holdout was not touched",
        "jendela_kemiripan": [args.ambang, args.ambang_atas],
        "jaccard_teks_maks": args.jaccard_maks,
        "kalibrasi": ("bounds derived from the 400 original candidate pairs of the paper: sim 0.6594 to 0.8858 "
                      "(median 0.7211), text Jaccard 0.000 to 0.355 (median 0.127), none above 0.5."),
        "acuan_ambang": ("the original pairs of the paper have sim 0.8858 in the first example; the bound was "
                         "chosen a little below it, and the results at several bounds are reported below"),
        "yield_per_ambang": threshold_result,
        "n_kueri_kandidat": len(query),
        "n_pasangan_dibangun": n_pair,
        "n_baris": len(rows),
        "rincian": dict(stats),
        "sim_padanan": {
            "n_dengan_padanan": len(spread_nonzero),
            "mean": round(st.fmean(spread_nonzero), 4) if spread_nonzero else None,
            "median": round(st.median(spread_nonzero), 4) if spread_nonzero else None,
        },
        "kontrol_kualitas": [
            "the counterpart must come from a book of another program",
            "queries whose gold already crosses programs are left out",
            "the pool is the same for the profiles of a pair",
            "no LLM call and no newly invented gold",
        ],
        "batasan_wajib_ditulis": (
            "the counterparts are set by embedding similarity, not by human judgment; that two "
            "chunks give equal answers is a measured assumption, and a sample must be audited before "
            "the numbers are used"),
        "output": {"path": "results/raw/pools_paired_expanded.jsonl",
                   "sha256": sha(out)},
    }
    (Q2 / "raw/pools_paired_expanded.manifest.json").write_text(
        json.dumps(meta_out, indent=2, ensure_ascii=False))

    audit = []
    for b in rows[:30:3]:
        g = next(c["text"][:220] for c in b["cands"] if c["id"] == b["gold"][0])
        o = next(c["text"][:220] for c in b["cands"] if c["id"] == b["counterpart"][0])
        audit.append({"query_id": b["query_id"], "kueri": b["query"][:180],
                      "prodi": b["profile"]["department_name"],
                      "sim": b["provenance"]["sim_padanan"],
                      "gold": g, "counterpart": o})
    (Q2 / "raw/paired_expanded_audit10.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False))

    print(f"\nyield per threshold: {threshold_result}")
    print(f"pairs built {n_pair}, rows {len(rows)}, "
          f"breakdown {dict(stats)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
