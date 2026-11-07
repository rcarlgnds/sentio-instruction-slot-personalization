import os
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
POOL_PAIRED = ROOT / "study/retrieval/results/raw/pools_paired.jsonl"
BROAD_POOL = ROOT / "study/channel/results/raw/pools_paired_expanded.jsonl"
TOPICS = ROOT / "system/data/paired_set/topic_pairs.jsonl"
PAIRED_QUERIES = ROOT / "system/data/paired_set/paired_queries.jsonl"
LOCK = ROOT / "study/_shared/config_lock.json"
QDRANT = ROOT / "system/app/storage/qdrant_client.py"
MINER = ROOT / "study/channel/scripts/1_tracks/expand_paired_track.py"
OUT = ROOT / "study/channel/results/tables/pool_statistics.json"
MANIFEST = ROOT / "study/channel/results/raw/pool_statistics.manifest.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.startswith("["):
        return [s.strip(" '\"") for s in x.strip("[]").split(",") if s.strip(" '\"")]
    return [x] if x else []


def summarize(value):
    return {"min": min(value), "median": statistics.median(value), "max": max(value)}


def read(p: Path):
    return [json.loads(ln) for ln in p.open() if ln.strip()]


def main():
    lock = json.loads(LOCK.read_text())
    flags = lock["static_fingerprint"]["flags"]
    pool_size = flags["retrieval_pool_size"]
    rrf_k = flags["qdrant_rrf_k"]
    code = QDRANT.read_text()
    search_limit = sorted({int(x) for x in re.findall(r"limit=(\d+)", code)})
    miner = MINER.read_text()
    broad_pool_cfg = re.search(r"POOL_SIZE\s*=\s*(\d+)", miner)

    paired = read(POOL_PAIRED)
    per_pair = defaultdict(list)
    for r in paired:
        per_pair[r["pair_id"]].append(r)
    total = [len({c["id"] for c in r["cands"]}) for r in paired]
    max_text = max(len(c.get("text") or "") for r in paired for c in r["cands"])
    complete, dropped, identical = [], [], 0
    for pid, rows in sorted(per_pair.items()):
        ids = [tuple(c["id"] for c in r["cands"]) for r in rows]
        if len(rows) == 2 and ids[0] == ids[1]:
            identical += 1
        within = all(set(as_list(r["gold"])) & {c["id"] for c in r["cands"]} for r in rows)
        (complete if within and len(rows) == 2 else dropped).append(pid)

    broad = read(BROAD_POOL)
    broad_count = [len({c["id"] for c in r["cands"]}) for r in broad]
    max_text_broad = max(len(c.get("text") or "") for r in broad for c in r["cands"])
    group = defaultdict(list)
    for r in broad:
        group[r["pair_id"]].append(r)
    gold_present = sum(
        1 for r in broad
        if set(as_list(r["gold"])) | set(as_list(r["counterpart"])) <= {c["id"] for c in r["cands"]})

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "_apa_ini": ("What the candidate pools of the two channel tracks hold: candidates per row, whether "
                     "both golds are in, whether the pool of a pair is identical, and the text-length limit."),
        "track_berpasangan": {
            "cara_bangun": ("retrieval stage of the frozen system: BGE-M3 dense and sparse search each return the "
                            "limit below, RRF merges them on the client side, and the result is cut to "
                            "retrieval_pool_size; child chunks only, no program filter"),
            "batas_pencarian_per_cabang_di_kode": search_limit,
            "rrf_k_config_lock": rrf_k,
            "retrieval_pool_size_config_lock": pool_size,
            "pasangan_topik_kandidat": sum(1 for _ in TOPICS.open() if _.strip()),
            "baris_kueri": sum(1 for _ in PAIRED_QUERIES.open() if _.strip()),
            "baris_pool": len(paired),
            "pasangan": len(per_pair),
            "kandidat_unik_per_baris": summarize(total),
            "pasangan_pool_identik_kedua_baris": identical,
            "pasangan_kedua_gold_di_pool": len(complete),
            "baris_komparabel": 2 * len(complete),
            "pasangan_gugur_karena_gold_di_luar_pool": dropped,
            "teks_kandidat_terpanjang_karakter": max_text,
        },
        "track_luas": {
            "cara_bangun": ("mining script: the nearest neighbours of the gold chunk by dense cosine, then every "
                            "gold of the group is forced in, so all golds are present by construction"),
            "pool_size_di_skrip_penambang": int(broad_pool_cfg.group(1)) if broad_pool_cfg else None,
            "baris": len(broad),
            "grup": len(group),
            "ukuran_grup": dict(sorted(Counter(len(v) for v in group.values()).items())),
            "kandidat_unik_per_baris": summarize(broad_count),
            "baris_dengan_semua_gold_grup_di_pool": gold_present,
            "teks_kandidat_terpanjang_karakter": max_text_broad,
        },
        "definisi_resolved": ("a row is resolved when the asker's gold ranks above every gold of the other profiles in "
                              "its group; a profile-blind scorer gives one ranking for the whole group, so exactly one "
                              "row per group can win"),
        "label_source": "deterministic (counts over the frozen pool files, the configuration, and the code)",
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "pool_statistics_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/1_tracks/pool_statistics.py",
        "input": {
            "pools_paired": {"path": "study/retrieval/results/raw/pools_paired.jsonl", "sha256": sha(POOL_PAIRED)},
            "pools_paired_expanded": {"path": "study/channel/results/raw/pools_paired_expanded.jsonl",
                                      "sha256": sha(BROAD_POOL)},
        },
        "output": {"path": "results/tables/pool_statistics.json", "sha256": sha(OUT)},
        "label_source": "deterministic",
    }, indent=1))

    t = output["track_berpasangan"]
    print(f"paired: {t['baris_pool']} rows, {t['pasangan']} pairs, candidates per row {t['kandidat_unik_per_baris']}, "
          f"identical pools {t['pasangan_pool_identik_kedua_baris']}, both golds in pool {t['pasangan_kedua_gold_di_pool']}, "
          f"dropped {t['pasangan_gugur_karena_gold_di_luar_pool']}, longest text {t['teks_kandidat_terpanjang_karakter']}")
    u = output["track_luas"]
    print(f"broad: {u['baris']} rows, {u['grup']} groups, sizes {u['ukuran_grup']}, candidates per row "
          f"{u['kandidat_unik_per_baris']}, all golds present {u['baris_dengan_semua_gold_grup_di_pool']}, "
          f"longest text {u['teks_kandidat_terpanjang_karakter']}")
    print(f"search limit per branch {t['batas_pencarian_per_cabang_di_kode']}, rrf k {t['rrf_k_config_lock']}, "
          f"pool size {t['retrieval_pool_size_config_lock']}, topic pairs {t['pasangan_topik_kandidat']}")


if __name__ == "__main__":
    main()
