import os
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"
QUERIES = ROOT / "system/data/rads_sample/queries.jsonl"
SHARED = ROOT / "study/_shared"
RAW = ROOT / "study/channel/results/raw"
RAW_R1 = ROOT / "study/channel/results/raw_r1"
TABLE = ROOT / "study/channel/results/tables"
CACHE = RAW / "emb_korpus.npy"
CACHE_ID = RAW / "emb_korpus_ids.json"
OLD_POOL = RAW / "pools_paired_expanded.jsonl"
NEW_POOL = RAW_R1 / "pools_xpair_r1.jsonl"
PREREG = ROOT / "protocol/revision_round1.md"
POOL_SIZE = 71
SEED = 20261012
N_GROUPS = 600
THRESHOLD, UPPER_THRESHOLD, JACCARD_MAX = 0.66, 0.89, 0.36


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_corpus():
    ids, text, meta = [], [], {}
    for rows in CORPUS.open():
        r = json.loads(rows)
        t = r.get("text") or ""
        if len(t) < 200:
            continue
        ids.append(r["_id"])
        text.append(t)
        meta[r["_id"]] = r["metadata"]
    return ids, text, meta


def load_embeddings(ids):
    if not (CACHE.exists() and CACHE_ID.exists()):
        raise SystemExit(f"ABORT: embedding cache missing ({CACHE}); this script never embeds.")
    if json.loads(CACHE_ID.read_text()) != ids:
        raise SystemExit("ABORT: embedding cache ids do not match the corpus chunk list.")
    return np.load(CACHE)


def eligible_queries(split, pos, meta):
    out = []
    for i, rows in enumerate(QUERIES.open()):
        q = json.loads(rows)
        if split.get(q["query_id"]) not in ("train", "dev"):
            continue
        cid = ((q.get("gold_anchor") or [{}])[0]).get("canonical_id")
        prof = q.get("student_profile") or {}
        dept = prof.get("department") or prof.get("department_name")
        if not cid or cid not in pos or not dept:
            continue
        if meta[cid].get("department") != dept:
            continue
        out.append({"q": q, "cid": cid, "dept": dept, "indeks_berkas": i})
    return out


def find_partners(cid, dept, X, pos, ids, text, program_idx):
    v = X[pos[cid]]
    ta = set(text[pos[cid]].lower().split())
    matches = []
    for d2, idxs in program_idx.items():
        if d2 == dept or d2 is None:
            continue
        sims = X[idxs] @ v
        for jj in np.argsort(-sims)[:20]:
            s2 = float(sims[int(jj)])
            if s2 > UPPER_THRESHOLD:
                continue
            if s2 < THRESHOLD:
                break
            cid2 = ids[int(idxs[int(jj)])]
            tb = set(text[pos[cid2]].lower().split())
            if len(ta & tb) / max(len(ta | tb), 1) > JACCARD_MAX:
                continue
            matches.append((d2, cid2, s2))
            break
    return matches


def build_pool(cid, used, X, pos, ids):
    order = np.argsort(-(X @ X[pos[cid]]))
    required = [cid] + [p[1] for p in used]
    pool, seen = [], set()
    for j in order:
        c = ids[int(j)]
        if c in seen:
            continue
        seen.add(c)
        pool.append(c)
        if len(pool) >= POOL_SIZE - len(set(required)):
            break
    for w in required:
        if w not in seen:
            pool.append(w)
            seen.add(w)
    return pool


def group_rows(pid, candidates, used, pool, text, pos, meta):
    q, cid, dept = candidates["q"], candidates["cid"], candidates["dept"]
    cands = [{"id": c, "text": text[pos[c]][:4000]} for c in pool]
    members = [(dept, cid)] + [(d2, c2) for d2, c2, _ in used]
    out = []
    for d_self, c_self in members:
        out.append({
            "query_id": f"{pid}-{d_self.lower()}",
            "pair_id": pid,
            "query": q["query_text"],
            "gold": [c_self],
            "counterpart": [c for _, c in members if c != c_self],
            "profile": {"department_name": d_self, "department": d_self,
                        "current_semester": meta[c_self].get("semester")},
            "cands": cands,
            "provenance": {
                "kueri_asal": q["query_id"],
                "class_label": q.get("class_label"),
                "buku": meta[c_self].get("book_slug"),
                "sim_padanan": [round(s, 4) for _, _, s in used],
            },
        })
    return out


def proportional_allocation(sizes: dict, total: int) -> dict:
    n = sum(sizes.values())
    quota = {k: total * v / n for k, v in sizes.items()}
    alloc = {k: int(np.floor(v)) for k, v in quota.items()}
    rest = total - sum(alloc.values())
    for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    for k in alloc:
        alloc[k] = min(alloc[k], sizes[k])
    return alloc


def prepare_population():
    split = json.loads((SHARED / "split_manifest.json").read_text())["assignment"]
    ids, text, meta = load_corpus()
    X = load_embeddings(ids)
    pos = {c: i for i, c in enumerate(ids)}
    per_program = defaultdict(list)
    for c in ids:
        per_program[meta[c].get("department")].append(pos[c])
    program_idx = {d: np.asarray(v) for d, v in per_program.items()}
    candidates = eligible_queries(split, pos, meta)
    return split, ids, text, meta, X, pos, program_idx, candidates


def mode_original():
    _, ids, text, meta, X, pos, program_idx, candidates = prepare_population()
    rows, n_read = [], 0
    n_pair = 0
    for k in candidates:
        if n_pair >= N_GROUPS:
            break
        n_read += 1
        pad = find_partners(k["cid"], k["dept"], X, pos, ids, text, program_idx)
        if not pad:
            continue
        pad.sort(key=lambda x: -x[2])
        used = pad[:2]
        n_pair += 1
        pool = build_pool(k["cid"], used, X, pos, ids)
        rows.extend(group_rows(f"xpair-{n_pair}", k, used, pool, text, pos, meta))
    frozen = [json.loads(b) for b in OLD_POOL.open() if b.strip()]
    exact_same = sum(1 for a, b in zip(rows, frozen) if a == b)
    order_only = 0
    differs = []
    for a, b in zip(rows, frozen):
        if a == b:
            continue
        ia = [c["id"] for c in a["cands"]]
        ib = [c["id"] for c in b["cands"]]
        without = {k: v for k, v in a.items() if k != "cands"} == {k: v for k, v in b.items() if k != "cands"}
        if without and sorted(ia) == sorted(ib) and ia[:len(ia) - 2] == ib[:len(ib) - 2]:
            order_only += 1
        else:
            differs.append(a["query_id"])
    result = {
        "n_baris_ulang": len(rows), "n_baris_beku": len(frozen),
        "n_kueri_layak": len(candidates), "n_kueri_dibaca": n_read, "n_grup": n_pair,
        "baris_identik": exact_same,
        "baris_beda_hanya_urutan_gold_tambahan": order_only,
        "baris_beda_lain": len(differs), "contoh_beda_lain": differs[:10],
        "lolos": len(rows) == len(frozen) and not differs,
    }
    print(json.dumps(result, indent=1))
    return result


def mode_random(dry_run: bool = False):
    if not PREREG.exists() or f"seed {SEED}" not in PREREG.read_text():
        raise SystemExit(f"ABORT: {PREREG.name} must exist and state 'seed {SEED}' before "
                         "the sample is drawn.")
    split, ids, text, meta, X, pos, program_idx, candidates = prepare_population()
    old = original_mode_summary(candidates, X, pos, ids, text, program_idx)
    population = []
    for k in candidates:
        pad = find_partners(k["cid"], k["dept"], X, pos, ids, text, program_idx)
        pad.sort(key=lambda x: -x[2])
        population.append({**k, "padanan": pad[:2], "n_padanan_semua": len(pad),
                         "gate": (k["q"].get("qc") or {}).get("protocol")})
    usable = [p for p in population if p["padanan"]]
    strata = defaultdict(list)
    for p in usable:
        strata[(p["gate"], p["dept"])].append(p)
    sizes = {k: len(v) for k, v in strata.items()}
    alloc = proportional_allocation(sizes, N_GROUPS)
    rng = np.random.default_rng(SEED)
    selected = []
    for key in sorted(strata):
        members = strata[key]
        if alloc[key]:
            idx = rng.choice(len(members), size=alloc[key], replace=False)
            selected.extend(members[i] for i in sorted(int(x) for x in idx))
    selected.sort(key=lambda p: p["indeks_berkas"])
    if any(split.get(p["q"]["query_id"]) not in ("train", "dev") for p in selected):
        raise SystemExit("ABORT: a sampled query is outside train and dev.")
    if dry_run:
        n_rows = sum(1 + len(p["padanan"]) for p in selected)
        print(json.dumps({
            "dry_run": True, "eligible": len(population), "groupable": len(usable),
            "groupable_by_gate": {g: sum(1 for p in usable if p["gate"] == g)
                                  for g in sorted({p["gate"] for p in usable})},
            "allocation_by_stratum": {f"{g}|{d}": v for (g, d), v in sorted(alloc.items())},
            "sampled_groups": len(selected), "sampled_rows": n_rows,
            "sampled_by_gate": {g: sum(1 for p in selected if p["gate"] == g)
                                for g in sorted({p["gate"] for p in selected})},
            "original_miner_last_file_index": old["indeks_terakhir"],
        }, indent=1))
        return
    RAW_R1.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, p in enumerate(selected, start=1):
        pool = build_pool(p["cid"], p["padanan"], X, pos, ids)
        rows.extend(group_rows(f"xr1-{n}", p, p["padanan"], pool, text, pos, meta))
    valid = {q for q, s in split.items() if s in ("train", "dev")}
    leak = sorted({b["provenance"]["kueri_asal"] for b in rows} - valid)
    if leak:
        raise SystemExit(f"ABORT: {len(leak)} pool rows come from queries outside train and dev.")
    tmp = NEW_POOL.with_name(NEW_POOL.name + ".tmp")
    with tmp.open("w") as f:
        for b in rows:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    old_origin = {json.loads(b)["provenance"]["kueri_asal"] for b in OLD_POOL.open() if b.strip()}
    selected_q = [p["q"]["query_id"] for p in selected]
    pop_table = TABLE / "r1_population_broad.json"
    pop_table.write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(),
        "what": "one record per eligible training or development query of the broad-track "
                "population, with its gate version, program, partner count and sampling outcome",
        "records": [{"query_id": p["q"]["query_id"], "file_index": p["indeks_berkas"],
                     "gate": p["gate"], "program": p["dept"],
                     "n_partners_found": p["n_padanan_semua"],
                     "groupable": bool(p["padanan"]),
                     "in_original_sample": p["q"]["query_id"] in old_origin,
                     "read_by_original_miner": p["indeks_berkas"] <= old["indeks_terakhir"],
                     "sampled_r1": p["q"]["query_id"] in set(selected_q)} for p in population],
    }, indent=1))

    def compute(entries, key):
        c = defaultdict(int)
        for p in entries:
            c[key(p)] += 1
        return dict(sorted(c.items()))
    manifest = {
        "run_id": "pools_xpair_r1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/9_revision/revision_broad_sample.py",
        "command": "revision_broad_sample.py --mode acak",
        "config_version_lock": json.loads((SHARED / "config_lock.json").read_text())["config_version"],
        "purpose": "broad-track sample representing both quality-gate versions",
        "rules": {"similarity_window": [THRESHOLD, UPPER_THRESHOLD], "jaccard_max": JACCARD_MAX,
                  "top_k_per_program": 20, "max_partners": 2, "pool_size": POOL_SIZE,
                  "pool_centre": "gold chunk embedding", "corpus_min_chars": 200},
        "sampling": {"design": "stratified random sampling without replacement",
                     "strata": "gate version x asker program",
                     "allocation": "proportional to groupable queries, largest remainder",
                     "n_groups": N_GROUPS, "seed": SEED, "rng": "numpy.random.default_rng (PCG64)",
                     "numpy_version": np.__version__},
        "population": {"eligible_queries": len(population), "groupable_queries": len(usable),
                       "eligible_by_gate": compute(population, lambda p: p["gate"]),
                       "groupable_by_gate": compute(usable, lambda p: p["gate"]),
                       "groupable_by_stratum": {f"{g}|{d}": v for (g, d), v in sorted(sizes.items())},
                       "allocation_by_stratum": {f"{g}|{d}": v for (g, d), v in sorted(alloc.items())}},
        "sample": {"n_groups": len(selected), "n_rows": len(rows),
                   "pairs": sum(1 for p in selected if len(p["padanan"]) == 1),
                   "triples": sum(1 for p in selected if len(p["padanan"]) == 2),
                   "by_gate": compute(selected, lambda p: p["gate"]),
                   "by_program": compute(selected, lambda p: p["dept"]),
                   "overlap_with_original_sample": sum(1 for q in selected_q if q in old_origin),
                   "not_read_by_original_miner": sum(1 for p in selected
                                                     if p["indeks_berkas"] > old["indeks_terakhir"]),
                   "query_ids": selected_q},
        "original_miner_replay": old,
        "preregistration": {"path": "protocol/revision_round1.md",
                            "sha256_at_draw": sha(PREREG)},
        "inputs": {"corpus": sha(CORPUS), "queries": sha(QUERIES), "embedding_cache": sha(CACHE),
                   "split_manifest": sha(SHARED / "split_manifest.json")},
        "holdout_check": {"source_queries": len({b["provenance"]["kueri_asal"] for b in rows}),
                          "outside_train_dev": 0},
        "output": {"path": "results/raw_r1/pools_xpair_r1.jsonl", "sha256": sha(tmp),
                   "population_table": {"path": "results/tables/r1_population_broad.json",
                                        "sha256": sha(pop_table)}},
        "label_source": "deterministic (embedding similarity and a seeded draw, no judge)",
    }
    (RAW_R1 / "pools_xpair_r1.manifest.json").write_text(json.dumps(manifest, indent=1))
    tmp.replace(NEW_POOL)
    print(json.dumps({k: manifest[k] for k in ("population", "sample")}, indent=1)[:4000])
    print(f"written: {NEW_POOL}")


def original_mode_summary(candidates, X, pos, ids, text, program_idx):
    n_pair, last = 0, -1
    for k in candidates:
        if n_pair >= N_GROUPS:
            break
        last = k["indeks_berkas"]
        if find_partners(k["cid"], k["dept"], X, pos, ids, text, program_idx):
            n_pair += 1
    return {"n_groups": n_pair, "indeks_terakhir": last}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["lama", "acak"], required=True)
    ap.add_argument("--kering", action="store_true",
                    help="with --mode acak: compute the population and the draw, print a "
                         "summary, write nothing")
    args = ap.parse_args()
    if args.mode == "lama":
        result = mode_original()
        raise SystemExit(0 if result["lolos"] else 1)
    if args.kering:
        mode_random(dry_run=True)
        return
    if NEW_POOL.exists():
        raise SystemExit(f"ABORT: {NEW_POOL}"
                         f" exists. The sample is drawn only once. To draw it again, delete the "
                         f"file and record why.")
    mode_random()


if __name__ == "__main__":
    main()
