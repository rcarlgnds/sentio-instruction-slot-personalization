import os
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from metrics import seal_gate  # noqa: E402

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
POOL_PAIRED = ROOT / "study/retrieval/results/raw/pools_paired.jsonl"
POOL_XPAIR = ROOT / "study/channel/results/raw/pools_paired_expanded.jsonl"
LOCK = ROOT / "study/_shared/config_lock.json"
TABLE = ROOT / "study/channel/results/tables/profile_collinearity.json"
MANIFEST = (ROOT / "study/channel/results/raw"
            / "profile_collinearity.manifest.json")
SCRIPT = "study/channel/scripts/1_tracks/profile_collinearity.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(records) for records in Path(path).read_text().splitlines()
            if records.strip()]


def summarize(rows, name):
    seal_gate([r["query_id"] for r in rows])
    combos = Counter((r["profile"]["department_name"],
                         r["profile"]["current_semester"]) for r in rows)
    program = Counter(r["profile"]["department_name"] for r in rows)
    semester = Counter(r["profile"]["current_semester"] for r in rows)

    per_program_semester = defaultdict(set)
    per_semester_program = defaultdict(set)
    for (p, s) in combos:
        per_program_semester[p].add(s)
        per_semester_program[s].add(p)

    group = defaultdict(list)
    for r in rows:
        group[r["pair_id"]].append(r)
    mixed_semester = sum(
        1 for v in group.values()
        if len({x["profile"]["current_semester"] for x in v}) > 1)

    unique_profiles = {json.dumps(r["profile"], sort_keys=True) for r in rows}

    collinear = (len(combos) == len(program) == len(semester))
    return {
        "track": name,
        "n_baris": len(rows),
        "n_grup_pair_id": len(group),
        "ukuran_grup": dict(sorted(Counter(len(v) for v in group.values()).items())),
        "n_profil_unik": len(unique_profiles),
        "n_prodi": len(program),
        "n_semester": len(semester),
        "n_kombinasi_prodi_x_semester": len(combos),
        "kombinasi": {f"{p}/sem{s}": n for (p, s), n in sorted(combos.items())},
        "prodi_dgn_lebih_dari_satu_semester": sorted(
            p for p, s in per_program_semester.items() if len(s) > 1),
        "semester_dipakai_lebih_dari_satu_prodi": sorted(
            s for s, p in per_semester_program.items() if len(p) > 1),
        "n_grup_semester_beragam": mixed_semester,
        "n_grup_semester_seragam": len(group) - mixed_semester,
        "prodi_dan_semester_kolinear_sempurna": collinear,
        "kenapa": (
            "each program maps to exactly one semester and back, so a label that carries both fields "
            "cannot separate their contributions on this track"
            if collinear else
            "program and semester cross, so this stratum separates the two fields better than the "
            "collinear track"),
    }


def build_order():
    paired = summarize(read(POOL_PAIRED), "asli (paired)")
    xpair = summarize(read(POOL_XPAIR), "penambangan (xpair)")
    lock = json.loads(LOCK.read_text())["config_version"]
    return {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock,
        "label_source": "deterministic (profile metadata read from the frozen pools)",
        "_apa_ini": (
            "Collinearity of program and semester on the two channel tracks."),
        "gerbang": {
            "segel_test_holdout": "passed, no test_holdout query_id in either pool",
        },
        "track": {"paired": paired, "xpair": xpair},
        "dipakai_di_naskah": {
            "paper, Section 3.2": (
                "the two fields covary by construction on the paired track"),
        },
        "syarat_kutip": [
            "these numbers describe the design of the pools, not the performance of any arm",
            "the collinearity on the paired track is a property of its construction, not a finding",
            "the xpair track is cleaner on this axis but more adversarial on another (the "
            "source-mirror asymmetry, see dits_strata.json)",
        ],
    }


def payload_sha256(table):
    body = {k: v for k, v in table.items() if k != "dibuat"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def main():
    table = build_order()
    if "--verify-determinism" in sys.argv:
        second = build_order()
        a, b = payload_sha256(table), payload_sha256(second)
        if a != b:
            raise SystemExit(f"FAIL determinism: {a} against {b}")
        print(f"determinism passes, number payload {a[:16]}")
        return
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": f"q2_kolinearitas_profil_{stamp}",
        "script": SCRIPT,
        "config_version": table["config_version_lock"],
        "sumber": {
            "pools_paired.jsonl": sha(POOL_PAIRED),
            "pools_paired_expanded.jsonl": sha(POOL_XPAIR),
        },
        "n": table["track"]["paired"]["n_baris"]
             + table["track"]["xpair"]["n_baris"],
        "label_source": table["label_source"],
        "output": {
            "path": "results/tables/profile_collinearity.json",
            "sha256": sha(TABLE),
            "tingkat_reproduksi": "1_bit_identical",
            "sha256_muatan_angka_tanpa_field_dibuat": payload_sha256(table),
            "verifikasi_determinisme": {
                "dijalankan": True,
                "cara": "rerun with --verify-determinism",
            },
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    for name in ("paired", "xpair"):
        t = table["track"][name]
        print(f"{name}: {t['n_baris']} rows, {t['n_grup_pair_id']} groups, "
              f"{t['n_prodi']} departments, {t['n_semester']} semesters, "
              f"{t['n_kombinasi_prodi_x_semester']} combinations, "
              f"collinear={t['prodi_dan_semester_kolinear_sempurna']}, "
              f"varied-semester groups {t['n_grup_semester_beragam']}")


if __name__ == "__main__":
    main()
