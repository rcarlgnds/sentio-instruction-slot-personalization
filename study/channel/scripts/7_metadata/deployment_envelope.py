import hashlib
import json
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
Q2 = ROOT / "study/channel/results"
N1 = ROOT / "study/retrieval/results"
SHARED = ROOT / "study/_shared"

CONDITION = "paired_profile_docmeta"
MODELS = {
    "Qwen3-Reranker-0.6B": f"scores_qwen3rr_Qwen3-Reranker-0.6B_{CONDITION}",
    "zerank-2": f"scores_zerank2_zerank-2_{CONDITION}",
    "Qwen3-Reranker-4B": f"scores_qwen3rr_Qwen3-Reranker-4B_{CONDITION}",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def find(stem, suffix):
    for base in (Q2 / "raw", N1 / "raw"):
        p = base / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def latency_mean(path):
    if path is None:
        return None
    vals = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            v = json.loads(line).get("latency_s")
        except json.JSONDecodeError:
            continue
        if isinstance(v, (int, float)):
            vals.append(v)
    return round(statistics.mean(vals), 3) if vals else None


def main():
    lock = json.loads((SHARED / "config_lock.json").read_text())

    rows, missing = [], []
    for name, stem in MODELS.items():
        man = find(stem, ".manifest.json")
        if man is None:
            missing.append(stem)
            continue
        m = json.loads(man.read_text())
        rows.append({
            "model": name,
            "kondisi": CONDITION,
            "vram_puncak_mib": m.get("vram_peak_mib"),
            "latensi_detik_mean": latency_mean(find(stem, ".jsonl")),
            "sumber_manifest": str(man.relative_to(ROOT)),
            "sha256_manifest": sha(man),
        })
    if missing:
        print("ABORT: scorer manifests incomplete, table not written.")
        for s in missing:
            print(f"  missing: {s}.manifest.json")
        raise SystemExit(2)

    summary = json.loads((N1 / "raw/summary.json").read_text())
    card_total = summary.get("device_vram_mb", {}).get("sesudah", {}).get("total")
    config_path = ROOT / "system/app/config.py"
    m = re.search(r"^\s*vllm_vram_bytes:\s*int\s*=\s*(\d+)", config_path.read_text(), re.M)
    resident = round(int(m.group(1)) / 2**20) if m else None

    headroom = (card_total - resident) if (card_total and resident) else None

    out = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("Deployment envelope: peak VRAM and mean latency per "
                     "reranker under one shared condition, against the free "
                     "VRAM left beside the resident stack."),
        "kondisi": CONDITION,
        "per_model": rows,
        "headroom": {
            "kartu_total_mib": card_total,
            "generator_reservation_mib": resident,
            "generator_reservation_source": {
                "path": str(config_path.relative_to(ROOT)),
                "field": "vllm_vram_bytes",
                "sha256": sha(config_path),
            },
            "sisa_mib": headroom,
            "catatan": ("remaining = the total card minus the reservation of the generator. The embedder loads "
                        "lazily on the first query and is not included, so this remaining amount is an upper "
                        "bound on the room for a reranker, not free memory."),
        },
        "muat": {r["model"]: (None if headroom is None
                              else r["vram_puncak_mib"] <= headroom)
                 for r in rows},
        "label_source": "deterministic (read from the frozen run manifests)",
    }

    dest = Q2 / "tables/deployment_envelope.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    (Q2 / "raw/deployment_envelope.manifest.json").write_text(json.dumps({
        "run_id": "q2_envelope_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/7_metadata/deployment_envelope.py",
        "config_version": lock["config_version"],
        "label_source": "deterministic",
        "sumber": {**{r["model"]: r["sha256_manifest"] for r in rows},
                   "config_py": sha(config_path), "summary_json": sha(N1 / "raw/summary.json")},
        "output": {"path": "results/tables/deployment_envelope.json",
                   "sha256": sha(dest), "tingkat_reproduksi": "1_bit_identical"},
    }, indent=2, ensure_ascii=False))

    print(f"{'model':24s} {'VRAM MiB':>10s} {'s/query':>9s}")
    for r in rows:
        lat = "n/a" if r["latensi_detik_mean"] is None else f"{r['latensi_detik_mean']:.3f}"
        print(f"{r['model']:24s} {r['vram_puncak_mib']:10.1f} {lat:>9s}")
    if headroom is None:
        print("\nHeadroom not derived: the generator reservation was not found.")
        print("The card total is", card_total, "MiB.")
    else:
        print(f"\nCard {card_total} MiB, generator reservation {resident} MiB, "
              f"free {headroom} MiB")
    print(f"\nwritten: {dest}")


if __name__ == "__main__":
    main()
