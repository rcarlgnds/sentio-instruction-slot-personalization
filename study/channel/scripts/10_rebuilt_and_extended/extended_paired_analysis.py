"""Table of the extended paired track (p2), made with the code of revision_analysis.py.

Like rebuilt_broad_analysis.py, it imports that module unchanged and points it at the p2 cells, and the
parts the two scripts share are in track_tables.py. Writes results/tables/p2_paired.json.
"""

import json
import sys
from functools import partial
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import extended_paired_common as U  # noqa: E402
import revision_analysis as A  # noqa: E402
import track_tables as T  # noqa: E402

U.POOL = U.RAW_P2 / "pools/pools_p2_ids.jsonl"
U.POPULATION = U.TABLE / "p2_population_public.json"
DITS = U.RAW_P2 / "p2_dits_xp2_baris_B.jsonl"
DITS_MANIFEST = U.RAW_P2 / "p2_dits_xp2.manifest.json"
OUT = A.TAB / "p2_paired.json"
FAMILIES = T.FAMILIES


def point_to_queue():
    A.PLAN = U.RAW_P2 / "_rencana_p2.json"
    A.STATUS_R1 = U.RAW_P2 / "_status_r1.json"
    A.TAG["p2"] = "xp2"


def cell_source(model: str, cell: str) -> Path:
    return U.RAW_P2 / A.cell_file_name(model, cell, "p2")


subset_family = partial(T.subset_family, cell_source)
main_family = partial(T.main_family, cell_source)
gemma_appended = partial(T.gemma_appended, cell_source)
blind_count = partial(T.blind_count, cell_source)
dits_p2 = partial(T.dits_rows, DITS, DITS_MANIFEST, "p2")


def main():
    point_to_queue()
    for p in (U.POOL, U.POOL_MANIFEST, U.POPULATION, A.PLAN, A.STATUS_R1):
        if not p.exists():
            raise SystemExit(f"ABORT: input missing, table not written: {p}")
    load = A.CellLoader()
    grp = A.group_sources(U.POOL)
    program_pairs = {}
    for b in U.POOL.open(encoding="utf-8"):
        if b.strip():
            r = json.loads(b)
            program_pairs[r["pair_id"]] = r["provenance"]["program_pair"]
    book = A.book_clusters(U.POOL)
    id_pool = {json.loads(b)["query_id"] for b in U.POOL.open(encoding="utf-8") if b.strip()}
    strict = set()
    for b in U.POOL.open(encoding="utf-8"):
        if b.strip():
            r = json.loads(b)
            if r["provenance"]["judge"].get("strict"):
                strict.add(r["pair_id"])

    section = T.TablePart()
    result, per_program, appended = {}, {}, {}
    for fam, fam_pairs in FAMILIES.items():
        result[fam] = section(f"contrasts.{fam}", main_family, load, fam_pairs, book)
        per_program[fam] = {k: section(f"by_program_pair.{fam}.{k}", subset_family, load, fam_pairs,
                                    {g for g, v in program_pairs.items() if v == k})
                          for k in sorted(set(program_pairs.values()))}
    strict_subset = {fam: section(f"strict_subset.{fam}", subset_family, load, fam_pairs, strict)
                    for fam, fam_pairs in FAMILIES.items()}
    for fam in ("slot_vs_none", "both_vs_slot"):
        appended[fam] = section(f"bge_gemma_appended_placement_secondary.{fam}", gemma_appended, load,
                             *FAMILIES[fam])
    blind = {m: section(f"blind_invariant.{m}", blind_count, load, m, len(grp)) for m in A.MODEL}
    dits = section("metadata_references.DITS", dits_p2, load, id_pool)
    m_pool = json.loads(U.POOL_MANIFEST.read_text())
    content = {
        "what": f"track p2: {len(grp)}"
                f" groups whose question was written for two passages of books of two "
                f"programs, screened, and both passages judged full answers, as "
                f"pre-registered; the blind baseline resolves one row per group",
        "n_groups": len(grp), "n_rows": len(id_pool),
        "not_computed": section.skipped,
        "sample_design": {k: m_pool[k] for k in ("rules", "stopping", "attrition_by_program_pair",
                                                 "groups_by_program_pair", "groups_by_class",
                                                 "strict_subset", "sample")},
        "blind_invariant": blind,
        "bge_gemma": "documented placement (slot_vendor, both_vendor, wrong_vendor) in every family; "
                     "its appended slot and both cells are secondary (pre-registration 11.1 a)",
        "contrasts": result, "by_program_pair": per_program,
        "strict_subset": {"what": "groups whose two passages the judge rated full answers under both wordings, reported "
                                       "beside every result as pre-registered",
                               "n_groups": len(strict), "families": strict_subset},
        "bge_gemma_appended_placement_secondary": appended,
        "paired_52_questions_reference": T.bound_summary("r1_cluster_paired.json"),
        "metadata_references": dits,
    }
    inputs = dict(load.used)
    for p in (U.POOL, U.POOL_MANIFEST, U.POPULATION):
        inputs[str(p.relative_to(A.ROOT))] = A.sha(p)
    T.write_table(out=OUT, raw=U.RAW_P2, name="p2_paired",
                  script="study/channel/scripts/10_rebuilt_and_extended/extended_paired_analysis.py",
                  section_name="paired_p2", status="pre-registered",
                  label_source=("deterministic (gold matching); the questions were written and screened by LLM agents "
                                "and the passages checked by the judge of the rebuilt broad track"),
                  inputs=inputs, content=content, section=section, result=result,
                  n_groups=len(grp), n_rows=len(id_pool))


if __name__ == "__main__":
    main()
