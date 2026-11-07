import os
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
T = ROOT / "study/channel/results/tables"
RAW = ROOT / "study/channel/results/raw"
TABLE = T / "followir_calibration.json"
MANIFEST = RAW / "followir_calibration.manifest.json"

REF = "FollowIR-7B"
COLLECTION = ["Core17", "News21", "Robust04"]
MODEL = {
    "Qwen3-Reranker-0.6B": "qwen3rr_Qwen3-Reranker-0.6B",
    "Qwen3-Reranker-4B": "qwen3rr_Qwen3-Reranker-4B",
    "zerank-2": "zerank2official_zerank-2",
    "FollowIR-7B": "followir_FollowIR-7B",
    "ctxl-6B": "ctxl_ctxl-rerank-v2-instruct-multilingual-6b",
    "bge-reranker-v2-gemma": "gemma_bge-reranker-v2-gemma",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(collection: str, suffix: str) -> tuple:
    p = T / f"followir_{collection}_{suffix}.json"
    d = json.loads(p.read_text())
    value = next((d[k] for k in ("p_mrr", "pmrr", "delta_mrr") if k in d), None)
    if value is None:
        value = next(v for k, v in d.items()
                     if "mrr" in k.lower() and isinstance(v, (int, float)))
    return value, {
        "berkas": p.name,
        "sha256": sha(p),
        "n_kueri": d.get("n_kueri"),
        "kandidat_per_kueri": d.get("kandidat_per_kueri"),
        "dataset": d.get("dataset"),
        "implementasi_metrik": d.get("implementasi_metrik"),
        "dtype": d.get("dtype"),
    }


def main():
    missing = [f"{k} {m}" for k in COLLECTION for m in MODEL
              if not (T / f"followir_{k}_{MODEL[m]}.json").exists()]
    if missing:
        raise SystemExit("ABORT: cells missing, table not written:\n  " + "\n  ".join(missing))

    board, setting = {}, {}
    for collection in COLLECTION:
        ref, meta = read(collection, MODEL[REF])
        setting[collection] = meta
        rows = {}
        for name, suffix in MODEL.items():
            value, _ = read(collection, suffix)
            rows[name] = {
                "p_mrr": round(value, 6),
                "positif": value > 0,
                "pangsa_dari_acuan": round(value / ref, 4) if ref else None,
            }
        board[collection] = {"acuan_p_mrr": round(ref, 6), "model": rows}

    n_cells = len(COLLECTION) * len(MODEL)
    positive = sum(1 for k in COLLECTION for m in MODEL if board[k]["model"][m]["positif"])
    exceeds = {k: [m for m in MODEL if m != REF
                    and board[k]["model"][m]["p_mrr"] > board[k]["acuan_p_mrr"]]
                for k in COLLECTION}
    candidates = {setting[k]["kandidat_per_kueri"] for k in COLLECTION}
    metric = {setting[k]["implementasi_metrik"] for k in COLLECTION}

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "_apa_ini": ("Internal scale for the FollowIR p-MRR values. The reference is FollowIR-7B, the model "
                     "the authors of that benchmark trained to follow search instructions, measured through "
                     "the same harness as the five other models."),
        "yang_tidak_dilakukan": ("Our numbers are not compared with the tables of the FollowIR paper. Those come from "
                                 "other models over another first stage, and the paper declines cross-setting comparisons "
                                 "elsewhere for the same reason."),
        "acuan": REF,
        "n_sel": n_cells,
        "n_sel_positif": positive,
        "model_yang_melebihi_acuan": exceeds,
        "setelan_seragam": {
            "kandidat_per_kueri": sorted(candidates),
            "implementasi_metrik": sorted(metric),
            "seragam": len(candidates) == 1 and len(metric) == 1,
        },
        "setelan_per_koleksi": setting,
        "papan": board,
        "label_source": "deterministic (p-MRR metric, no rater)",
    }
    TABLE.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "followir_calibration_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/5_why_slot_fails/followir_calibration.py",
        "output": {"path": "results/tables/followir_calibration.json", "sha256": sha(TABLE)},
        "label_source": "deterministic",
    }, indent=1))

    width = max(len(m) for m in MODEL)
    print(f"  reference {REF}, {positive} of {n_cells} cells positive, "
          f"uniform settings {output['setelan_seragam']['seragam']}\n")
    print(f"  {'model':{width}s} " + "  ".join(f"{k:>18s}" for k in COLLECTION))
    print("  " + "-" * (width + 62))
    for name in MODEL:
        cell = []
        for k in COLLECTION:
            b = board[k]["model"][name]
            mark = "*" if name != REF and b["p_mrr"] > board[k]["acuan_p_mrr"] else " "
            cell.append(f"{b['p_mrr']:>+8.4f} {b['pangsa_dari_acuan']:>6.0%}{mark}")
        print(f"  {name:{width}s} " + "  ".join(cell))
    print("\n  an asterisk marks a value above the reference on that collection")
    for k in COLLECTION:
        if exceeds[k]:
            print(f"    {k}: {', '.join(exceeds[k])}")
    print("  written:", TABLE)


if __name__ == "__main__":
    main()
