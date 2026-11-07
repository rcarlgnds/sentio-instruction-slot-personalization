import os
import fcntl
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import mcnemar as MC  # noqa: E402
import metrics as M  # noqa: E402
from score_arms import (ROOT, Q2, SHARED, TASK_DEFAULT, profile_instruction,  # noqa: E402
                          load_global_profiles, sha)
from score_architectures import PREREG, write_status  # noqa: E402
from model_pins import revision_for  # noqa: E402

MODEL = "zeroentropy/zerank-2"
POOL = ROOT / "study/retrieval/results/raw/pools_paired.jsonl"
FROZEN = {"default": Q2 / "raw/scores_zerank2_zerank-2_paired_default.jsonl",
        "profile": Q2 / "raw/scores_zerank2_zerank-2_paired_profile.jsonl"}
OUT = Q2 / "tables/zerank2_equivalence_full.json"
MAN = Q2 / "raw/zerank2_equivalence_full.manifest.json"
MAX_LEN = 1536


def main():
    if OUT.exists():
        print(f"SKIP (idempotent): {OUT.name} exists")
        return
    import torch
    import sentence_transformers
    import transformers
    from sentence_transformers import CrossEncoder
    from architecture_scorers import wait_for_free_gpu, vram_free_mib

    M.replication_gate(verbose=False)
    rows = [json.loads(l) for l in POOL.open() if l.strip()]
    M.seal_gate([r["query_id"] for r in rows])
    global_profile = load_global_profiles()

    key = open(Q2 / "raw/.gpu.lock", "w")
    fcntl.flock(key, fcntl.LOCK_EX)
    wait_for_free_gpu()
    write_status(nama="zerank2_equivalence_full", keadaan="memuat", n_selesai=0, n_total=2)
    t0 = time.perf_counter()
    B = CrossEncoder(MODEL, revision=revision_for(MODEL), model_kwargs={"torch_dtype": torch.bfloat16}, max_length=MAX_LEN,
                     device="cuda")
    result, fpath = {}, {}
    for cond in ("default", "profile"):
        out_path = Q2 / f"raw/scores_zerank2official_zerank-2_paired_{cond}_equivfull.jsonl"
        with out_path.open("w") as f:
            for i, r in enumerate(rows):
                if cond == "profile":
                    p = r.get("profile") or global_profile.get(r["query_id"], {})
                    query = f"{profile_instruction(p)}\n\nQuery: {r['query']}"
                else:
                    query = f"{TASK_DEFAULT}\n\nQuery: {r['query']}"
                doc = [(c["text"] or "")[:4000] for c in r["cands"]]
                t1 = time.perf_counter()
                score = [float(x) for x in B.predict([(query, d) for d in doc], batch_size=2)]
                order = sorted(zip((c["id"] for c in r["cands"]), score), key=lambda x: (-x[1], x[0]))
                f.write(json.dumps({"query_id": r["query_id"], "gold": r["gold"],
                                    "counterpart": r.get("counterpart"),
                                    "ranking": [c for c, _ in order],
                                    "scores": [round(s, 6) for _, s in order],
                                    "latency_s": round(time.perf_counter() - t1, 4)}) + "\n")
                if (i + 1) % 20 == 0:
                    write_status(nama=f"zerank2_equivalence_full {cond}", keadaan="berjalan",
                                 n_selesai=i + 1, n_total=len(rows),
                                 elapsed_s=round(time.perf_counter() - t0, 1),
                                 vram_free_mib=vram_free_mib())
        fpath[cond] = out_path
        m = M.paired_metrics(M.read_ranking(out_path))
        item_b = MC.per_item_results(out_path)
        item_a = MC.per_item_results(FROZEN[cond])
        shared = sorted(set(item_a) & set(item_b))
        wins_b = sum(1 for v in item_b.values() if v)
        wins_a = sum(1 for v in item_a.values() if v)
        agree = sum(1 for q in shared if item_a[q] == item_b[q])
        result[cond] = {
            "menang_jalur_resmi": f"{wins_b}/{len(item_b)}",
            "menang_beku": f"{wins_a}/{len(item_a)}",
            "hitungan_identik": wins_a == wins_b,
            "n_baris_komparabel": len(shared),
            "per_baris_setuju": agree,
            "per_baris_beda": [q for q in shared if item_a[q] != item_b[q]],
            "metrik_produksi_jalur_resmi": m,
        }
        print(f"  {cond}: official {wins_b}/{len(item_b)} vs frozen {wins_a}/{len(item_a)}; "
              f"per-row agreement {agree}/{len(shared)}", flush=True)
    del B
    torch.cuda.empty_cache()

    ia = MC.per_item_results(fpath["default"])
    ib = MC.per_item_results(fpath["profile"])
    shared = sorted(set(ia) & set(ib))
    n01 = sum(1 for q in shared if not ia[q] and ib[q])
    n10 = sum(1 for q in shared if ia[q] and not ib[q])
    lock = json.loads((SHARED / "config_lock.json").read_text())
    table = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "praregistrasi": PREREG + " (full-equivalence addendum)",
        "_apa_ini": ("The vendor's official CrossEncoder.predict() path over the whole paired pool in two "
                     "conditions, compared with the frozen zerank-2 cells at 48/96 and 47/96."),
        "model": MODEL, "n_baris": len(rows), "max_length": MAX_LEN, "dtype": "bfloat16",
        "versi": {"sentence_transformers": sentence_transformers.__version__,
                  "transformers": transformers.__version__, "torch": torch.__version__},
        "per_kondisi": result,
        "mcnemar_default_vs_profile_jalur_resmi": {
            "n_berpasangan": len(shared), "diskordan_default_kalah_profile_menang": n01,
            "diskordan_default_menang_profile_kalah": n10,
            "p_mcnemar_exact": round(MC.mcnemar_exact(n01, n10), 8)},
        "kriteria_lolos": "win counts identical to the frozen cells in both conditions",
        "lolos": all(v["hitungan_identik"] for v in result.values()),
        "durasi_s": round(time.perf_counter() - t0, 1),
        "label_source": "deterministic (model scores)",
    }
    OUT.write_text(json.dumps(table, indent=2, ensure_ascii=False))
    MAN.write_text(json.dumps({
        "run_id": "zerank2_equivalence_full_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/2_scoring/zerank2_equivalence_full.py",
        "praregistrasi": table["praregistrasi"], "config_version": lock["config_version"],
        "sumber": {"pools": sha(POOL), **{f"beku_{k}": sha(v) for k, v in FROZEN.items()}},
        "durasi_s": table["durasi_s"],
        "output": {"path": "results/tables/zerank2_equivalence_full.json", "sha256": sha(OUT),
                   "skor": {k: {"path": f"results/raw/{v.name}", "sha256": sha(v)} for k, v in fpath.items()}},
    }, indent=2, ensure_ascii=False))
    write_status(nama="zerank2_equivalence_full", keadaan="selesai", n_selesai=2, n_total=2,
                 elapsed_s=table["durasi_s"])
    print(f"PASS={table['lolos']}  written: {OUT}")


if __name__ == "__main__":
    main()
