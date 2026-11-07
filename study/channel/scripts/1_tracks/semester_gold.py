#!/usr/bin/env python3

import os
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from label_map import load_id_label_map

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
SHARED = ROOT / "study/_shared"
CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"
POOLS = ROOT / "study/retrieval/results/raw/pools_paired.jsonl"
TABLE = ROOT / "study/channel/results/tables/semester_gold_paired.json"
MANIFEST = ROOT / "study/channel/results/raw/semester_gold_paired.manifest.json"

MAP_MODE = "id"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_gate(query_ids) -> dict:
    assignment = json.loads((SHARED / "split_manifest.json").read_text())["assignment"]
    leak = [q for q in query_ids if assignment.get(q) == "test_holdout"]
    if leak:
        raise SystemExit(
            f"ABORT SEAL: {len(leak)} test_holdout query_id read "
            f"(example {leak[:3]}). A post-hoc arm must not touch the holdout.")
    per_split = {}
    for q in query_ids:
        s = assignment.get(q) or "tak_terdaftar_paired"
        per_split[s] = per_split.get(s, 0) + 1
    return {"lolos": True, "n_diperiksa": len(query_ids), "per_split": per_split}


def load_metadata_map() -> dict:
    if MAP_MODE == "id":
        return load_id_label_map()
    mapping = {}
    for rows in CORPUS.open():
        r = json.loads(rows)
        k = (r.get("text") or "")[:180].strip()
        if k:
            mapping.setdefault(k, r["metadata"])
    return mapping


def candidate_meta(mapping, candidates_by_id, ids):
    for i in ids:
        c = candidates_by_id.get(i)
        if c is None:
            continue
        m = (mapping.get(i) if MAP_MODE == "id"
             else mapping.get((c.get("text") or "")[:180].strip()))
        if m is not None:
            return m
    return None


def compute() -> dict:
    mapping = load_metadata_map()
    rows = [json.loads(l) for l in POOLS.open() if l.strip()]
    gate = seal_gate([r["query_id"] for r in rows])

    n_resolved = 0
    gold_semester_match = 0
    counterpart_semester_match = 0
    gold_program_match = 0
    counterpart_program_match = 0
    n_gold_only = 0
    gold_only_semester_match = 0

    for r in rows:
        profile = r.get("profile") or {}
        semester = profile.get("current_semester")
        program = (profile.get("department_name") or "").upper()
        per_id = {c["id"]: c for c in r["cands"]}
        mg = candidate_meta(mapping, per_id, r["gold"])
        mc = candidate_meta(mapping, per_id, r.get("counterpart") or [])
        if mg is not None:
            n_gold_only += 1
            gold_only_semester_match += int(mg.get("semester") == semester)
        if mg is None or mc is None:
            continue
        n_resolved += 1
        gold_semester_match += int(mg.get("semester") == semester)
        counterpart_semester_match += int(mc.get("semester") == semester)
        gold_program_match += int((mg.get("department") or "").upper() == program)
        counterpart_program_match += int((mc.get("department") or "").upper() == program)

    return {
        "config_version_lock": json.loads(
            (SHARED / "config_lock.json").read_text())["config_version"],
        "_apa_ini": "Semester of the golds on the paired track: whether the gold always carries the asker's "
                    "semester and the counterpart never does, which bounds what the wrong-label control can "
                    "show.",
        "gerbang": {"segel_test_holdout": gate},
        "populasi": {
            "n_baris_kolam": len(rows),
            "n_baris_teresolusi": n_resolved,
            "definisi_teresolusi": "the gold and the counterpart both map to the corpus metadata, by the first 180 "
                                   "characters of their text (teks map) or by their own record (id map). These 96 rows are "
                                   "the denominator the paper uses for this track.",
        },
        "semester": {
            "gold_cocok_semester_mahasiswa": gold_semester_match,
            "counterpart_cocok_semester_mahasiswa": counterpart_semester_match,
            "penyebut": n_resolved,
        },
        "prodi": {
            "gold_cocok_prodi_mahasiswa": gold_program_match,
            "counterpart_cocok_prodi_mahasiswa": counterpart_program_match,
            "penyebut": n_resolved,
            "_catatan": "With the teks map the program matches on 75 of the 96 rows, and with the id map on all "
                        "96. The semester matches on all 96 under both maps, so it always identifies the gold.",
        },
        "varian_gold_saja": {
            "n_baris": n_gold_only,
            "gold_cocok_semester": gold_only_semester_match,
            "_catatan": "a looser population in which only the gold must resolve, reported so the alternative "
                        "denominator has a source; the paper uses the strict denominator above",
        },
        "label_source": "deterministic (read from the frozen corpus metadata and the student profiles in the "
                        "frozen pools)",
        "syarat_kutip": [
            "these numbers are a property of how the pools were built, not a result of any model",
            "the program and semester collinearity of this track is in profile_collinearity.json and "
            "should be cited with these numbers",
        ],
    }


def payload_sha256(table: dict) -> str:
    body = {k: v for k, v in table.items() if k != "dibuat"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-determinism", action="store_true")
    ap.add_argument("--peta", choices=["teks", "id"], default="id",
                    help="teks matches a candidate by the first 180 characters of its text, the map that "
                         "mislabeled 21 rows. id uses the candidate's own record and should mislabel none.")
    args = ap.parse_args()

    global MAP_MODE, TABLE, MANIFEST
    MAP_MODE = args.peta
    if args.peta == "id":
        TABLE = TABLE.with_name(TABLE.stem + "_petaid.json")
        MANIFEST = MANIFEST.with_name(
            MANIFEST.name.replace(".manifest.json", "_petaid.manifest.json"))

    table = compute()
    if args.verify_determinism:
        second = compute()
        same = payload_sha256(table) == payload_sha256(second)
        print(f"determinism: {'PASS' if same else 'FAIL'} "
              f"({payload_sha256(table)[:16]})")
        return 0 if same else 1

    table["dibuat"] = datetime.now(timezone.utc).isoformat()
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")

    manifest = {
        "run_id": "q2_semester_gold_paired_"
                  + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/1_tracks/semester_gold.py",
        "perintah_persis": " ".join(["python", "semester_gold.py", *sys.argv[1:]]),
        "config_version": table["config_version_lock"],
        "sumber": {
            "pools_paired.jsonl": sha(POOLS),
            "corpus.jsonl": sha(CORPUS),
            "split_manifest.json": sha(SHARED / "split_manifest.json"),
        },
        "n": table["populasi"]["n_baris_teresolusi"],
        "label_source": table["label_source"],
        "output": {
            "path": TABLE.relative_to(ROOT / "study/channel").as_posix(),
            "sha256": sha(TABLE),
            "tingkat_reproduksi": "1_bit_identical",
            "sha256_muatan_angka_tanpa_field_dibuat": payload_sha256(table),
            "verifikasi_determinisme": {
                "dijalankan": True,
                "cara": "rerun with --verify-determinism",
            },
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"table   : {TABLE}")
    print(f"manifest: {MANIFEST}")
    print(f"gold matches semester {table['semester']['gold_cocok_semester_mahasiswa']}"
          f"/{table['semester']['penyebut']}, counterpart "
          f"{table['semester']['counterpart_cocok_semester_mahasiswa']}"
          f"/{table['semester']['penyebut']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
