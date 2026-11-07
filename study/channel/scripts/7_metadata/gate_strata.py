import os
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import channel_cells as K  # noqa: E402

OUT = K.TABLE / "gate_strata.json"
MANIFEST = K.RAW / "gate_strata.manifest.json"
POOL = K.RAW / "pools_paired_expanded.jsonl"
QUERIES = K.ROOT / "system/data/rads_sample/queries.jsonl"
CORPUS = K.ROOT / "system/data/benchmark_release/corpus/corpus.jsonl"


def as_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str) and x.startswith("["):
        return [s.strip(" '\"") for s in x.strip("[]").split(",") if s.strip(" '\"")]
    return [x] if x else []


def main():
    protocol, position = {}, {}
    for i, ln in enumerate(QUERIES.open()):
        q = json.loads(ln)
        protocol[q["query_id"]] = q["qc"].get("protocol")
        position[q["query_id"]] = i

    dept = {}
    for ln in CORPUS.open():
        r = json.loads(ln)
        dept[r["_id"]] = r["metadata"].get("department")

    strata_rows, strata_groups = {}, defaultdict(set)
    filter_wins = filter_n = 0
    no_origin = 0
    used_origins = set()
    for ln in POOL.open():
        r = json.loads(ln)
        origin = (r.get("provenance") or {}).get("kueri_asal")
        if origin in position:
            used_origins.add(origin)
        s = protocol.get(origin)
        if s is None:
            no_origin += 1
            s = "tidak_terlacak"
        strata_rows[r["query_id"]] = s
        strata_groups[s].add(r["pair_id"])

        prof = r.get("profile")
        prof = prof if isinstance(prof, dict) else json.loads(prof.replace("'", '"'))
        owner = prof.get("department") or prof.get("department_name")
        gold = as_list(r["gold"])
        counterpart = as_list(r["counterpart"])
        remaining_gold = [g for g in gold if dept.get(g) == owner]
        remaining_counterparts = [c for c in counterpart if dept.get(c) == owner]
        filter_n += 1
        filter_wins += 1 if remaining_gold and not remaining_counterparts else 0

    result = {}
    for s in sorted(set(strata_rows.values())):
        rows = {q for q, v in strata_rows.items() if v == s}
        block = {"grup": len(strata_groups[s]), "baris": len(rows),
                "kunci_struktural": len(strata_groups[s]), "sel": {}}
        for cell in K.PAPER["xpair"]:
            wins = K.manuscript_gate("xpair", cell)
            block["sel"][cell] = {m: sum(w for q, w in wins[m].items() if q in rows)
                                for m in K.MODEL}
        result[s] = block

    for cell in K.PAPER["xpair"]:
        for i, m in enumerate(K.MODEL):
            total = sum(result[s]["sel"][cell][m] for s in result)
            if total != K.PAPER["xpair"][cell][i]:
                raise SystemExit(f"STOP: strata {cell}/{m} sum to {total}, not "
                                 f"{K.PAPER['xpair'][cell][i]}")
    span = {}
    for s in sorted(set(protocol.values())):
        pos_s = [position[q] for q, v in protocol.items() if v == s]
        span[s] = [min(pos_s), max(pos_s)]
    file_order = {
        "posisi_baris_per_protokol_min_maks": span,
        "posisi_kueri_asal_terakhir_yang_dipakai": max(position[q] for q in used_origins),
        "kueri_asal_unik": len(used_origins),
        "cara_baca": ("the miner reads queries.jsonl from the top and stops at the limit of 600 pairs; if the "
                      "position of the last source query is below the first v2 position, the whole broad track "
                      "inherits the v1 gate because of the file order, not by choice"),
    }

    if filter_wins != filter_n:
        print(f"NOTE: the hard filter resolves {filter_wins} of {filter_n}, not all")

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "_apa_ini": ("Channel results of the broad track split by the version of the QC gate that accepted "
                     "their source query, plus a check of the hard-filter rows in the same pool."),
        "gerbang_qc": {
            "v1-cps-gate": ("a query is accepted only when its top candidate reaches CPS 0.55 under the frozen DITS "
                            "weights, so its acceptance depends on the reported scorer"),
            "v2-multiarm": "the CPS condition is dropped",
            "benchmark_penuh": "1,412 queries through v1, 1,364 through v2, 2,776 in total",
        },
        "strata": result,
        "baris_tanpa_asal_terlacak": no_origin,
        "urutan_berkas_kueri": file_order,
        "hard_filter_track_luas": {
            "menang": filter_wins, "n": filter_n,
            "aturan": ("drop the candidates whose payload department is not the asker's, then check whether the "
                       "asker's gold is the only gold of the group left"),
            "catatan": ("this result follows from the mining rule and is not a finding: the gold is always in "
                        "the book of the asker's program and the rival gold always in the book of another program"),
        },
        "gerbang": "the count per stratum matches the paper's count for every cell",
        "zerank2": "official vendor path",
        "label_source": "deterministic (join of the row provenance to the qc.protocol of the benchmark query)",
    }
    OUT.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "gate_strata_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/7_metadata/gate_strata.py",
        "output": {"path": "results/tables/gate_strata.json", "sha256": K.sha(OUT)},
        "label_source": "deterministic",
    }, indent=1))

    for s, b in result.items():
        print(f"\n{s}: {b['grup']} groups, {b['baris']} rows, lock {b['kunci_struktural']}")
        for cell, per in b["sel"].items():
            print(f"  {cell:<6}" + "".join(f"{per[m]:>7}" for m in K.MODEL))
    print(f"\nhard filter, broad track: {filter_wins}/{filter_n}; rows without traced source: {no_origin}")
    print(f"file positions per gate: {span}; last source query used at position "
          f"{file_order['posisi_kueri_asal_terakhir_yang_dipakai']}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
