import os
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from score_arms import ROOT, Q2, SHARED, TASK_DEFAULT, Zerank2Scorer, sha  # noqa: E402
from score_architectures import PREREG, write_status  # noqa: E402
from model_pins import revision_for  # noqa: E402

MODEL = "zeroentropy/zerank-2"
POOL = ROOT / "study/retrieval/results/raw/pools_paired.jsonl"
FROZEN = Q2 / "raw/scores_zerank2_zerank-2_paired_default.jsonl"
OUT = Q2 / "tables/zerank2_equivalence.json"
MAN = Q2 / "raw/zerank2_equivalence.manifest.json"
N_PAIRS = 100
MAX_LEN = 1536
TOL = 0.05


def pairs():
    output = []
    for l in POOL.open():
        r = json.loads(l)
        query = f"{TASK_DEFAULT}\n\nQuery: {r['query']}"
        for c in r["cands"]:
            output.append({"query_id": r["query_id"], "cand_id": c["id"],
                           "kueri": query, "dok": (c["text"] or "")[:4000]})
            if len(output) == N_PAIRS:
                return output
    return output


def count_inversions(a: list, b: list) -> int:
    n = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if (a[i] - a[j]) * (b[i] - b[j]) < 0:
                n += 1
    return n


def main():
    if OUT.exists():
        print(f"SKIP (idempotent): {OUT.name} exists")
        return
    import fcntl
    import torch
    import numpy as np
    from scipy.stats import spearmanr
    import sentence_transformers
    import transformers

    P = pairs()
    print(f"{len(P)} pairs from {len({p['query_id'] for p in P})} queries", flush=True)
    key = open(Q2 / "raw/.gpu.lock", "w")
    fcntl.flock(key, fcntl.LOCK_EX)
    from architecture_scorers import wait_for_free_gpu
    pending = wait_for_free_gpu()
    if pending:
        print(f"waited {pending}s for the previous process to release the GPU", flush=True)
    write_status(nama="zerank2_equivalence", keadaan="berjalan", n_selesai=0, n_total=2)

    t0 = time.perf_counter()
    A = Zerank2Scorer(MODEL, "bfloat16")
    score_a = [None] * len(P)
    for qid in dict.fromkeys(p["query_id"] for p in P):
        idx = [i for i, p in enumerate(P) if p["query_id"] == qid]
        s = A.score(P[idx[0]]["kueri"], [P[i]["dok"] for i in idx], 2, MAX_LEN)
        for i, v in zip(idx, s):
            score_a[i] = float(v)
    duration_a = time.perf_counter() - t0
    del A
    torch.cuda.empty_cache()
    write_status(nama="zerank2_equivalence", keadaan="berjalan", n_selesai=1, n_total=2)

    from sentence_transformers import CrossEncoder
    t1 = time.perf_counter()
    B = CrossEncoder(MODEL, revision=revision_for(MODEL), model_kwargs={"torch_dtype": torch.bfloat16}, max_length=MAX_LEN,
                     device="cuda")
    score_b = [float(x) for x in B.predict([(p["kueri"], p["dok"]) for p in P], batch_size=2)]
    duration_b = time.perf_counter() - t1
    del B
    torch.cuda.empty_cache()

    frozen = {}
    for l in FROZEN.open():
        r = json.loads(l)
        for cid, s in zip(r["ranking"], r["scores"]):
            frozen[(r["query_id"], cid)] = s
    frozen_scores = [frozen.get((p["query_id"], p["cand_id"])) for p in P]
    n_frozen = sum(1 for s in frozen_scores if s is not None)
    frozen_diff = [abs(a - b) for a, b in zip(score_a, frozen_scores) if b is not None]

    diff = [abs(a - b) for a, b in zip(score_a, score_b)]
    rho = spearmanr(score_a, score_b).correlation
    per_q = {}
    for qid in dict.fromkeys(p["query_id"] for p in P):
        idx = [i for i, p in enumerate(P) if p["query_id"] == qid]
        per_q[qid] = {"n_kandidat": len(idx),
                      "inversi": count_inversions([score_a[i] for i in idx], [score_b[i] for i in idx])}
    total_inv = sum(v["inversi"] for v in per_q.values())
    passed = total_inv == 0 and max(diff) <= TOL
    lock = json.loads((SHARED / "config_lock.json").read_text())
    result = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "praregistrasi": PREREG,
        "_apa_ini": ("Equivalence test of the project's Zerank2Scorer against the vendor's official "
                     "CrossEncoder.predict() on 100 identical pairs."),
        "model": MODEL, "n_pasangan": len(P), "n_kueri": len(per_q),
        "jalur_a": {"skorer": "score_arms.Zerank2Scorer", "dtype": "bfloat16",
                    "max_len": MAX_LEN, "durasi_s": round(duration_a, 1)},
        "jalur_b": {"skorer": "sentence_transformers.CrossEncoder.predict",
                    "sentence_transformers": sentence_transformers.__version__,
                    "transformers": transformers.__version__, "torch": torch.__version__,
                    "dtype": "bfloat16", "max_length": MAX_LEN, "durasi_s": round(duration_b, 1)},
        "spearman_rho": round(float(rho), 6),
        "inversi_total": total_inv,
        "inversi_per_kueri": per_q,
        "selisih_absolut": {"maks": round(max(diff), 6), "median": round(float(np.median(diff)), 6),
                            "mean": round(float(np.mean(diff)), 6),
                            "n_di_atas_toleransi": sum(1 for d in diff if d > TOL),
                            "toleransi": TOL},
        "reproduksi_berkas_beku": {"n_pasangan_ditemukan": n_frozen,
                                   "selisih_maks_jalur_a_vs_beku": round(max(frozen_diff), 6) if frozen_diff else None},
        "kriteria_lolos": "zero inversions and a maximum absolute difference of at most 0.05",
        "lolos": passed,
        "skor": [{"query_id": p["query_id"], "cand_id": p["cand_id"], "a": round(a, 6),
                  "b": round(b, 6), "beku": s}
                 for p, a, b, s in zip(P, score_a, score_b, frozen_scores)],
        "label_source": "deterministic (model scores)",
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    MAN.write_text(json.dumps({
        "run_id": "zerank2_equivalence_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/2_scoring/zerank2_equivalence.py",
        "praregistrasi": PREREG, "config_version": lock["config_version"],
        "sumber": {"pools": sha(POOL), "berkas_beku": sha(FROZEN)},
        "durasi_s": round(duration_a + duration_b, 1),
        "output": {"path": "results/tables/zerank2_equivalence.json", "sha256": sha(OUT)},
    }, indent=2, ensure_ascii=False))
    write_status(nama="zerank2_equivalence", keadaan="selesai", n_selesai=2, n_total=2)
    print(f"rho {rho:.6f}  inversions {total_inv}  max|diff| {max(diff):.4f}  "
          f"frozen max|diff| {max(frozen_diff) if frozen_diff else None}  PASS={passed}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
