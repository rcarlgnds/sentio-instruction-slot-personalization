import os
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
QUERIES = ROOT / "system/data/rads_sample/queries.jsonl"
SPLIT = ROOT / "study/_shared/split_manifest.json"
OUT = ROOT / "study/channel/results/tables/benchmark_provenance.json"
MANIFEST = ROOT / "study/channel/results/raw/benchmark_provenance.manifest.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    q = [json.loads(l) for l in QUERIES.open() if l.strip()]
    split = json.loads(SPLIT.read_text())["assignment"]
    gen = [x.get("generation") or {} for x in q]
    timestamp = sorted(datetime.fromisoformat(g["generated_at"]).astimezone(timezone.utc).isoformat()
                   for g in gen if g.get("generated_at"))
    per_proto = Counter(x["qc"].get("protocol") for x in q)
    top_cps_v1 = [x["qc"].get("top_cps") for x in q if x["qc"].get("protocol") == "v1-cps-gate"
                  and x["qc"].get("top_cps") is not None]
    top_cps_v2 = [x["qc"].get("top_cps") for x in q if x["qc"].get("protocol") == "v2-multiarm"
                  and x["qc"].get("top_cps") is not None]

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "_apa_ini": ("Origin of the 2,776 accepted benchmark queries: model, backend, template version, "
                     "generation period, QC gate protocol, annotation status, and split."),
        "n_kueri_diterima": len(q),
        "model": dict(Counter(g.get("model") for g in gen)),
        "backend": dict(Counter(g.get("backend") for g in gen)),
        "versi_templat": dict(Counter(g.get("template_version") for g in gen)),
        "generasi_paling_awal_utc": timestamp[0] if timestamp else None,
        "generasi_paling_akhir_utc": timestamp[-1] if timestamp else None,
        "protokol_gerbang_qc": dict(per_proto),
        "top_cps_minimum_per_protokol": {
            "v1-cps-gate": round(min(top_cps_v1), 4) if top_cps_v1 else None,
            "v2-multiarm": round(min(top_cps_v2), 4) if top_cps_v2 else None,
        },
        "status_anotasi": dict(Counter(x.get("annotation_status") for x in q)),
        "split": dict(Counter(split.get(x["query_id"]) for x in q)),
        "kelas": dict(Counter(x.get("class_label") for x in q)),
        "kode_di_paper1_code": {
            "templat_prompt": "system/app/benchmark/rads_synthesis/templates.py",
            "gerbang_qc": "system/app/benchmark/rads_synthesis/qc.py",
            "pipeline": "system/app/benchmark/rads_synthesis/pipeline.py",
        },
        "label_source": "deterministic (generation and QC metadata stored with each query)",
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "benchmark_provenance_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/1_tracks/benchmark_provenance.py",
        "input": {"path": "system/data/rads_sample/queries.jsonl", "sha256": sha(QUERIES)},
        "output": {"path": "results/tables/benchmark_provenance.json", "sha256": sha(OUT)},
        "label_source": "deterministic",
    }, indent=1))
    print(json.dumps({k: output[k] for k in ("n_kueri_diterima", "model", "backend", "versi_templat",
                                            "generasi_paling_awal_utc", "generasi_paling_akhir_utc",
                                            "protokol_gerbang_qc", "top_cps_minimum_per_protokol",
                                            "status_anotasi", "split")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
