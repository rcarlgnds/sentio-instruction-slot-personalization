"""Table of the rebuilt broad track (r2), made with the code of revision_analysis.py.

It imports revision_analysis.py unchanged and points it at the r2 cells, so the contrasts, the Holm
correction over the six models, and the family-wise limits match the other tracks. The parts it shares
with extended_paired_analysis.py are in track_tables.py. Writes results/tables/r2_broad.json.
"""

import json
import sys
from functools import partial
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import revision_analysis as A  # noqa: E402
import track_tables as T  # noqa: E402

RAW_R2 = A.Q2 / "results/raw_r2"
POOL_R2 = RAW_R2 / "pools/pools_xpair_r2_ids.jsonl"
MANIFEST_POOL = RAW_R2 / "pools_xpair_r2.manifest.json"
POPULATION = A.TAB / "r2_population_broad.json"
DITS = RAW_R2 / "r2_dits_xr2_baris_B.jsonl"
DITS_MANIFEST = RAW_R2 / "r2_dits_xr2.manifest.json"
OUT = A.TAB / "r2_broad.json"
FAMILIES = T.FAMILIES


def point_to_queue():
    A.PLAN = RAW_R2 / "_rencana_r2.json"
    A.STATUS_R1 = RAW_R2 / "_status_r1.json"
    A.TAG["r2"] = "xr2"


def cell_source(model: str, cell: str) -> Path:
    return RAW_R2 / A.cell_file_name(model, cell, "r2")


subset_family = partial(T.subset_family, cell_source)
main_family = partial(T.main_family, cell_source)
gemma_appended = partial(T.gemma_appended, cell_source)
blind_count = partial(T.blind_count, cell_source)
dits_r2 = partial(T.dits_rows, DITS, DITS_MANIFEST, "r2")


def main():
    point_to_queue()
    for p in (POOL_R2, MANIFEST_POOL, POPULATION, A.PLAN, A.STATUS_R1):
        if not p.exists():
            raise SystemExit(f"ABORT: input missing, table not written: {p}")
    load = A.CellLoader()
    pop = {r["query_id"]: r for r in json.loads(POPULATION.read_text())["records"]}
    grp = A.group_sources(POOL_R2)
    gate = {g: pop[v["source_query"]]["gate"] for g, v in grp.items()}
    new = {g for g, v in grp.items() if not (pop[v["source_query"]]["in_original_sample"]
                                              or pop[v["source_query"]]["in_r1_sample"])}
    book = A.book_clusters(POOL_R2)
    id_pool = {json.loads(b)["query_id"] for b in POOL_R2.open(encoding="utf-8") if b.strip()}
    n_rows = len(id_pool)

    section = T.TablePart()
    result, per_gate, new_groups, appended = {}, {}, {}, {}
    for fam, fam_pairs in FAMILIES.items():
        result[fam] = section(f"contrasts.{fam}", main_family, load, fam_pairs, book)
        per_gate[fam] = {g_name: section(f"by_gate.{fam}.{g_name}", subset_family, load, fam_pairs,
                                        {g for g, v in gate.items() if v == g_name})
                         for g_name in sorted(set(gate.values()))}
        new_groups[fam] = section(f"groups_not_in_earlier_samples.{fam}", subset_family, load,
                                fam_pairs, new)
    for fam in ("slot_vs_none", "both_vs_slot"):
        appended[fam] = section(f"bge_gemma_appended_placement_secondary.{fam}", gemma_appended, load,
                             *FAMILIES[fam])
    blind = {m: section(f"blind_invariant.{m}", blind_count, load, m, len(grp)) for m in A.MODEL}
    dits = section("metadata_references.DITS", dits_r2, load, id_pool)

    m_pool = json.loads(MANIFEST_POOL.read_text())
    content = {
        "what": f"r2 broad sample of {len(grp)}"
                f" groups whose partner golds the judge rated full answers, as "
                f"pre-registered; the blind baseline resolves one row per group",
        "n_groups": len(grp), "n_rows": n_rows,
        "not_computed": section.skipped,
        "sample_design": {"sampling": m_pool["sampling"], "population": m_pool["population"],
                          "judge": {k: m_pool["judge"][k] for k in ("model", "prompt_version",
                                                                   "verdicts_used",
                                                                   "unreadable_replies_used")}},
        "blind_invariant": blind,
        "bge_gemma": "documented placement (slot_vendor, both_vendor, wrong_vendor) in every family; "
                     "its appended slot and both cells are secondary (pre-registration 11.1 a)",
        "contrasts": result, "by_gate": per_gate, "groups_not_in_earlier_samples": new_groups,
        "bge_gemma_appended_placement_secondary": appended,
        "earlier_broad_samples_unchecked_partners": {
            "original_sample": T.bound_summary("r1_cluster_broad.json"),
            "r1_resample": T.bound_summary("r1_random_broad.json")},
        "metadata_references": {**dits, "hard_filter_resolved_by_construction": n_rows,
                                "hard_filter_note": "not scored: the filter keeps only the asker's "
                                                    "program, so every other gold leaves the pool "
                                                    "by construction"},
    }
    inputs = dict(load.used)
    for p in (POOL_R2, MANIFEST_POOL, POPULATION):
        inputs[str(p.relative_to(A.ROOT))] = A.sha(p)
    T.write_table(out=OUT, raw=RAW_R2, name="r2_broad",
                  script="study/channel/scripts/10_rebuilt_and_extended/rebuilt_broad_analysis.py",
                  section_name="luas_r2", status="pre-registered",
                  label_source=("deterministic (gold matching); the golds were chosen by the judge of the rebuilt broad "
                                "track"),
                  inputs=inputs, content=content, section=section, result=result,
                  n_groups=len(grp), n_rows=n_rows)


if __name__ == "__main__":
    main()
