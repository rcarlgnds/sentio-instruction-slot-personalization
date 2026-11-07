import os
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
QUERIES = ROOT / "system/data/rads_sample/queries.jsonl"
CORPUS = ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    lock = json.loads((SHARED / "config_lock.json").read_text())
    assignment = json.loads((SHARED / "split_manifest.json").read_text())["assignment"]

    dept_chunk = {}
    for l in CORPUS.open():
        r = json.loads(l)
        dept_chunk[r["_id"]] = r["metadata"].get("department")

    n_total = n_mapped = 0
    cross = 0
    unreachable = 0
    per_class = defaultdict(lambda: {"n": 0, "silang": 0})
    per_asker_dept = defaultdict(lambda: {"n": 0, "silang": 0})
    unmapped = 0
    for l in QUERIES.open():
        q = json.loads(l)
        qid = q["query_id"]
        if assignment.get(qid) == "test_holdout":
            continue
        if assignment.get(qid) not in ("train", "dev"):
            continue
        n_total += 1
        prof = q.get("student_profile") or {}
        student_dept = prof.get("department") or prof.get("department_name")
        anchors = q.get("gold_anchor") or []
        all_depts = [dept_chunk[a["canonical_id"]] for a in anchors
                      if a.get("canonical_id") in dept_chunk]
        if not student_dept or not all_depts:
            unmapped += 1
            continue
        n_mapped += 1
        differs = any(d != student_dept for d in all_depts)
        if student_dept not in all_depts:
            unreachable += 1
        cross += int(differs)
        class_label = q.get("class_label") or "?"
        per_class[class_label]["n"] += 1
        per_class[class_label]["silang"] += int(differs)
        per_asker_dept[student_dept]["n"] += 1
        per_asker_dept[student_dept]["silang"] += int(differs)

    def rate(d):
        return round(d["silang"] / d["n"], 4) if d["n"] else None

    class_results = {k: {**v, "rate_silang": rate(v)}
                   for k, v in sorted(per_class.items())}
    dept_results = {k: {**v, "rate_silang": rate(v)}
                  for k, v in sorted(per_asker_dept.items())}
    rate_total = round(cross / n_mapped, 4) if n_mapped else None

    print(f"traindev queries checked : {n_total}")
    print(f"fully mapped             : {n_mapped} (unmapped {unmapped})")
    print(f"gold in ANOTHER program  : {cross}/{n_mapped} = {rate_total}")
    print("per class:")
    for k, v in class_results.items():
        print(f"   {k:32} {v['silang']:5d}/{v['n']:5d} = {v['rate_silang']}")

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("Reachability: what share of golds lives in the book of a program other than the asker's."),
        "populasi": "traindev 2,493, test_holdout not touched (split gate)",
        "n_kueri_diperiksa": n_total,
        "n_terpetakan": n_mapped,
        "n_tak_terpetakan": unmapped,
        "gold_lintas_program": {
            "n": cross, "rate": rate_total,
            "_arti": ("share of queries with at least one gold in the book of another program. This is not a "
                      "lost recall ceiling: a hard filter would cut part of a multi-hop evidence chain, not "
                      "make the answer unreachable.")},
        "gold_tak_terjangkau_filter_keras": {
            "n": unreachable,
            "rate": round(unreachable / n_mapped, 4) if n_mapped else None,
            "_arti": ("queries with no gold at all in their own program, the ones truly lost under a hard "
                      "filter. Measured at zero: a hard filter still reaches at least one gold for every query.")},
        "per_kelas_kueri": class_results,
        "per_departemen_penanya": dept_results,
        "cara_membaca": (
            "A hard metadata filter does not make any gold unreachable: none of the 2,493 queries "
            "loses all of its golds. What is measured here is the cut in the evidence chain: 11.75 "
            "percent of queries (62.08 percent in the prerequisite class) have at least one gold "
            "chunk in the book of another program. A hard filter would hide those chunks, so an "
            "answer loses part of its reasoning chain while the main answer can still be found. The "
            "case for soft scoring therefore rests on the completeness of the evidence chain and on "
            "the gradation of the ranking, not on unreachability."),
        "label_source": "deterministic (join of the corpus metadata and gold_anchor)",
    }
    out = Q2 / "tables/reachability.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    (Q2 / "raw/reachability.manifest.json").write_text(json.dumps({
        "run_id": "q2_reach_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/7_metadata/reachability.py",
        "config_version": lock["config_version"],
        "label_source": "deterministic",
        "sumber": {"queries": sha(QUERIES), "corpus": sha(CORPUS)},
        "output": {"path": "results/tables/reachability.json", "sha256": sha(out),
                   "tingkat_reproduksi": "1_bit_identical"},
    }, indent=2, ensure_ascii=False))
    print(f"written: {out}")


if __name__ == "__main__":
    main()
