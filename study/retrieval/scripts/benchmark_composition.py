import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[3])
N1 = ROOT / "study/retrieval/results"
SHARED = ROOT / "study/_shared"

SPLITS = ("dev", "traindev", "test_holdout")
CLASS_PREFIX = {
    "adve": "adversarial_routing",
    "mult": "multi_disciplinary_ambiguity",
    "temp": "temporal_scope_drift",
    "prer": "prerequisite_chain_depth",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_qrels(path):
    gold = Counter()
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        if int(parts[3]) > 0:
            gold[parts[0]] += 1
        else:
            gold.setdefault(parts[0], 0)
    return gold


def main():
    lock = json.loads((SHARED / "config_lock.json").read_text())
    split = json.loads((SHARED / "split_manifest.json").read_text())
    assignment = split["assignment"]

    per_split, qrels_sha = {}, {}
    all_ids = {}
    for name in SPLITS:
        path = N1 / f"tables/qrels_{name}.txt"
        gold = read_qrels(path)
        qrels_sha[name] = sha(path)
        counts = Counter(gold.values())
        per_split[name] = {
            "queries": len(gold),
            "gold_passages": sum(gold.values()),
            "gold_per_query": round(sum(gold.values()) / len(gold), 4),
            "queries_by_gold_count": {str(k): v for k, v in sorted(counts.items())},
        }
        if name != "dev":
            for q in gold:
                all_ids[q] = name

    expected = {"dev": {"dev"}, "traindev": {"train", "dev"},
                "test_holdout": {"test_holdout"}}
    for name in SPLITS:
        gold = read_qrels(N1 / f"tables/qrels_{name}.txt")
        wrong = [q for q in gold if assignment.get(q) not in expected[name]]
        if wrong:
            raise SystemExit(f"qrels_{name}: {len(wrong)} queries are not on "
                             f"that split in split_manifest.json, e.g. {wrong[:3]}")

    classes = Counter(CLASS_PREFIX.get(q.split("-")[0], "unknown") for q in all_ids)
    if "unknown" in classes:
        raise SystemExit("query ids with an unrecognized class prefix")
    manifest_counts = Counter(assignment.values())

    out = {
        "created": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "what_this_is": (
            "Descriptive counts of the benchmark read from the frozen qrels "
            "files and the split manifest: split sizes, gold passages per "
            "query, and the class composition recovered from the class "
            "prefix each query identifier carries."),
        "benchmark": {
            "queries": len(all_ids),
            "by_class": dict(sorted(classes.items())),
            "class_design_target": 750,
            "classes_below_target": [k for k, v in classes.items() if v < 750],
        },
        "splits": {
            "from_split_manifest": dict(sorted(manifest_counts.items())),
            "train_plus_dev": manifest_counts["train"] + manifest_counts["dev"],
            "consistent_with_qrels": (
                per_split["traindev"]["queries"]
                == manifest_counts["train"] + manifest_counts["dev"]
                and per_split["dev"]["queries"] == manifest_counts["dev"]
                and per_split["test_holdout"]["queries"] == manifest_counts["test_holdout"]),
        },
        "per_split": per_split,
        "label_source": "deterministic (counts over frozen qrels)",
    }

    dest = N1 / "tables/benchmark_composition.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    (N1 / "raw/benchmark_composition.manifest.json").write_text(json.dumps({
        "run_id": "benchmark_composition_"
                  + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/retrieval/scripts/benchmark_composition.py",
        "config_version": lock["config_version"],
        "label_source": "deterministic",
        "sumber": {f"qrels_{k}": v for k, v in qrels_sha.items()}
                  | {"split_manifest": sha(SHARED / "split_manifest.json")},
        "output": {"path": "results/tables/benchmark_composition.json",
                   "sha256": sha(dest),
                   "tingkat_reproduksi": "1_bit_identical except the timestamp"},
    }, indent=1, ensure_ascii=False) + "\n")

    print(f"benchmark {len(all_ids)} queries: "
          + ", ".join(f"{k} {v}" for k, v in sorted(classes.items())))
    for name, s in per_split.items():
        print(f"  {name:13s} {s['queries']:5d} queries, "
              f"{s['gold_per_query']:.4f} gold per query")
    print(f"splits consistent with manifest: {out['splits']['consistent_with_qrels']}")
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
