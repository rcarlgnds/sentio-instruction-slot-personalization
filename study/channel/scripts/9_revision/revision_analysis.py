"""Tables of the follow-up analyses. Each --bagian value builds one table.

For example, klaster runs group-level tests on the broad sample, gemma tests bge-gemma with the prompt
where its model card puts it, and sensitivitas shows how the slot limits widen when some golds are
wrong. Six-model families use Holm, and a section writes nothing if any of its cells is missing. The
queue files mark every cell as passed. Their paths are redacted, so the status check here matches no
score file. rebuilt_broad_analysis.py, extended_paired_analysis.py, and human_ratings_analysis.py
reuse this code.
"""

import os
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from revision_statistics import SEED, group_id, contrast_family, contrast  # noqa: E402
from inference_ladder import wins_per_row  # noqa: E402

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
Q2 = ROOT / "study/channel"
RAW = Q2 / "results/raw"
RAW_R1 = Q2 / "results/raw_r1"
G0 = RAW_R1 / "_g0"
TAB = Q2 / "results/tables"
RAW_N1 = ROOT / "study/retrieval/results/raw"
PREREG = ROOT / "protocol/revision_round1.md"
PREREG_SHA256 = "3bfe129e05b9ddebee6d9597834205253d6adceba65227f6440bf98b88e55119"
POOL_PAIRED = RAW_R1 / "pools/pools_paired_ids.jsonl"
BROAD_POOL = RAW_R1 / "pools/pools_paired_expanded_ids.jsonl"
POOL_R1 = RAW_R1 / "pools/pools_xpair_r1_ids.jsonl"
POPULATION = TAB / "r1_population_broad.json"
PLAN = RAW_R1 / "_rencana_r1.json"
STATUS_R1 = RAW_R1 / "_status_r1.json"
QUERIES = ROOT / "system/data/rads_sample/queries.jsonl"

MODEL = ["Qwen3-0.6B", "Qwen3-4B", "zerank-2", "FollowIR-7B", "ctxl-6B", "bge-gemma"]
ARM = {"Qwen3-0.6B": ("qwen3rr", "Qwen3-Reranker-0.6B"),
       "Qwen3-4B": ("qwen3rr", "Qwen3-Reranker-4B"),
       "zerank-2": ("zerank2official", "zerank-2")}
ARCH = {"FollowIR-7B": ("followir", "FollowIR-7B"),
        "ctxl-6B": ("ctxl", "ctxl-rerank-v2-instruct-multilingual-6b"),
        "bge-gemma": ("gemma", "bge-reranker-v2-gemma")}
ARM_CELLS = {"none": "default", "slot": "profile", "both": "profile_docmeta",
           "wrong": "profile_docmetasalah",
           "slot_program": "profile_program", "slot_semester": "profile_semester",
           "slot_semtetap1": "profile_tetapsem1", "slot_semtetap2": "profile_tetapsem2",
           "slot_prodtetap_csci": "profile_tetapprodiCSCI",
           "slot_prodtetap_stat": "profile_tetapprodiSTAT",
           "both_semsalah": "profile_docmetasemsalah", "both_duasalah": "profile_docmetaduasalah",
           "label_acak": "profile_docmetaacak", "label_seragam": "profile_docmetaseragam"}
TAG = {"paired": "paired", "luas": "xpair", "r1": "xr1"}
THREE_CONTRASTS = {"slot_vs_none": ("none", "slot"), "both_vs_slot": ("slot", "both"),
            "wrong_vs_both": ("both", "wrong")}
ORTHOGONAL_CELLS = ["slot_program", "slot_semester", "slot_semtetap1", "slot_semtetap2",
                 "slot_prodtetap_csci", "slot_prodtetap_stat"]


class MissingInput(Exception):
    pass


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cell_file_name(model: str, cell: str, route: str) -> str:
    if model in ARM:
        fam, name = ARM[model]
        return f"scores_{fam}_{name}_{TAG[route]}_{ARM_CELLS[cell]}.jsonl"
    fam, name = ARCH[model]
    return f"scores_{fam}_{name}_{TAG[route]}_{cell}.jsonl"


def frozen_cell(model: str, cell: str, route: str) -> Path:
    if route == "paired" and model in ("Qwen3-0.6B", "Qwen3-4B") and cell in ("none", "slot"):
        sizes = "0.6b" if model == "Qwen3-0.6B" else "4b"
        return RAW_N1 / f"qwen_scores_{sizes}_paired_{'default' if cell == 'none' else 'profile'}.jsonl"
    return RAW / cell_file_name(model, cell, route)


def new_cell(model: str, cell: str, route: str) -> Path:
    return RAW_R1 / cell_file_name(model, cell, route)


def paired_reference(model: str, cell: str) -> Path:
    if model in ("Qwen3-0.6B", "Qwen3-4B") and cell in ("none", "slot"):
        return G0 / cell_file_name(model, cell, "paired")
    return frozen_cell(model, cell, "paired")


def queue_status() -> dict:
    if not PLAN.exists():
        return {}
    status = json.loads(STATUS_R1.read_text()).get("sel", {}) if STATUS_R1.exists() else {}
    return {str(Path(c["out_dir"]) / c["keluaran"]): status.get(c["id"], {}).get("status", "belum")
            for c in json.loads(PLAN.read_text())["cells"] if c["skrip"] != "followir_architectures.py"}


def book_clusters(pool: Path) -> dict:
    gold, members = defaultdict(set), defaultdict(list)
    for b in pool.open():
        if b.strip():
            r = json.loads(b)
            gold[r["pair_id"]].update(x.rsplit("-", 1)[0] for x in r["gold"])
            members[r["pair_id"]].append(r["query_id"])
    return {q: "|".join(sorted(gold[g])) for g, qs in members.items() for q in qs}


class CellLoader:
    def __init__(self):
        self.used = {}
        self.cache = {}
        self.cell_status = queue_status()

    def __call__(self, p: Path) -> dict:
        if not p.exists():
            raise MissingInput(str(p))
        st = self.cell_status.get(str(p))
        if st is not None and st != "lolos":
            raise MissingInput(f"{p} belongs to a queue cell whose status is {st}, not lolos")
        if p not in self.cache:
            self.cache[p] = wins_per_row(p)
            self.used[str(p.relative_to(ROOT))] = sha(p)
        return self.cache[p]


def write_table(name: str, content: dict, load: CellLoader, extra=(), exploratory=False):
    inputs = dict(load.used)
    for p in extra:
        inputs[str(Path(p).relative_to(ROOT))] = sha(Path(p))
    output = {"created": datetime.now(timezone.utc).isoformat(),
              "script": "study/channel/scripts/9_revision/revision_analysis.py",
              "section": name,
              "preregistration": {"path": str(PREREG.relative_to(ROOT)),
                                  "sha256": sha(PREREG) if PREREG.exists() else PREREG_SHA256},
              "analysis_status": ("exploratory: built from data that existed before the "
                                  "pre-registration was committed (section 11.1 h)"
                                  if exploratory else "pre-registered"),
              "seed": SEED, "inputs": inputs,
              "label_source": "deterministic (gold matching, no judge)", **content}
    TAB.mkdir(parents=True, exist_ok=True)
    RAW_R1.mkdir(parents=True, exist_ok=True)
    p = TAB / f"r1_{name}.json"
    p.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    (RAW_R1 / f"r1_{name}.manifest.json").write_text(json.dumps({
        "run_id": f"r1_{name}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/9_revision/revision_analysis.py",
        "command": "revision_analysis.py " + " ".join(sys.argv[1:]),
        "output": {"path": f"results/tables/r1_{name}.json", "sha256": sha(p)},
        "inputs": inputs, "seed": SEED}, indent=1))
    print(f"written: {p}")


def contrast_family_table(load, source, pairs, subset=None, clusters=None, replacement=None):
    res = {}
    for m in MODEL:
        cell_swap = (replacement or {}).get(m, {})
        sa, sb = cell_swap.get(pairs[0], pairs[0]), cell_swap.get(pairs[1], pairs[1])
        A, B = load(source(m, sa)), load(source(m, sb))
        if subset is not None:
            A = {k: v for k, v in A.items() if group_id(k) in subset}
            B = {k: v for k, v in B.items() if group_id(k) in subset}
        res[m] = contrast(A, B, m=len(MODEL), clusters=clusters)
        if (sa, sb) != tuple(pairs):
            res[m]["cells"] = [sa, sb]
    contrast_family(res)
    return res


def summarize(res: dict) -> dict:
    return {"decisive_test": "row-level exact McNemar, Holm over the six models (section 11.1 c)",
            "largest_gain_pp_max_over_models": max(r["largest_gain_pp"] for r in res.values()),
            "family_wise_largest_gain_pp_max_over_models": max(
                r["family_wise"]["largest_gain_pp"] for r in res.values()),
            "significant_after_holm_mcnemar": [m for m, r in res.items()
                                               if r["p_holm_mcnemar_exact"] < 0.05],
            "significant_after_holm_group_test": [m for m, r in res.items()
                                                  if r["p_holm_signflip_group"] < 0.05]}


def section_cluster():
    load = CellLoader()
    result = {}
    for fam, fam_pairs in THREE_CONTRASTS.items():
        res = contrast_family_table(load, lambda m, s: frozen_cell(m, s, "luas"), fam_pairs,
                               clusters=book_clusters(BROAD_POOL))
        result[fam] = {"models": res, "summary": summarize(res)}
    write_table("cluster_broad", {"what": "original broad sample, 1,621 rows in 600 groups",
                           "contrasts": result}, load, extra=[BROAD_POOL], exploratory=True)


def section_paired():
    load = CellLoader()
    result = {}
    for fam, fam_pairs in THREE_CONTRASTS.items():
        res = contrast_family_table(load, lambda m, s: frozen_cell(m, s, "paired"), fam_pairs)
        result[fam] = {"models": res, "summary": summarize(res)}
    write_table("cluster_paired", {"what": "paired track, 96 comparable rows in 48 pairs",
                             "contrasts": result}, load, exploratory=True)


def uniform_semester_groups():
    sem = defaultdict(set)
    for records in BROAD_POOL.open():
        r = json.loads(records)
        sem[r["pair_id"]].add(r["profile"]["current_semester"])
    return {g for g, s in sem.items() if len(s) == 1}


def section_uniform_semester():
    load = CellLoader()
    subset = uniform_semester_groups()
    result = {}
    for fam, fam_pairs in THREE_CONTRASTS.items():
        res = contrast_family_table(load, lambda m, s: frozen_cell(m, s, "luas"), fam_pairs, subset=subset,
                               clusters=book_clusters(BROAD_POOL))
        result[fam] = {"models": res, "summary": summarize(res)}
    n_rows = next(iter(result["slot_vs_none"]["models"].values()))["n_rows"]
    write_table("uniform_semester", {
        "what": "original broad groups whose rows all share one semester, so only the "
                "program separates the profiles",
        "n_groups": len(subset), "n_rows": n_rows, "contrasts": result}, load, extra=[BROAD_POOL],
        exploratory=True)


def group_sources(pool: Path) -> dict:
    out = {}
    for records in pool.open():
        r = json.loads(records)
        if r["pair_id"] not in out:
            out[r["pair_id"]] = {"source_query": r["provenance"]["kueri_asal"],
                                 "program": r["profile"]["department_name"]}
    return out


def section_selection():
    if not POPULATION.exists():
        raise MissingInput(str(POPULATION))
    load = CellLoader()
    pop = json.loads(POPULATION.read_text())["records"]
    composition = defaultdict(lambda: defaultdict(int))
    for r in pop:
        composition[r["gate"]]["eligible"] += 1
        composition[r["gate"]]["groupable"] += int(r["groupable"])
        composition[r["gate"]]["in_original_sample"] += int(r["in_original_sample"])
        composition[r["gate"]]["sampled_r1"] += int(r["sampled_r1"])
    top_cps = {}
    for records in QUERIES.open():
        q = json.loads(records)
        top_cps[q["query_id"]] = (q.get("qc") or {}).get("top_cps")
    grp = group_sources(BROAD_POOL)
    value = np.array([top_cps[v["source_query"]] for v in grp.values()], dtype=float)
    q1, q2 = np.quantile(value, [1 / 3, 2 / 3])
    strata = defaultdict(set)
    for g, v in grp.items():
        t = top_cps[v["source_query"]]
        strata[f"gate1_score_tertile_{1 if t <= q1 else 2 if t <= q2 else 3}"].add(g)
        strata[f"program_{v['program']}"].add(g)
    result = {}
    for name, subset in sorted(strata.items()):
        res = contrast_family_table(load, lambda m, s: frozen_cell(m, s, "luas"), ("none", "slot"),
                               subset=subset, clusters=book_clusters(BROAD_POOL))
        result[name] = {"n_groups": len(subset), "models": res, "summary": summarize(res)}
    write_table("selection_broad", {
        "what": "original broad sample: composition against the population, and the slot "
                "contrast per gate-1 score tertile and per source program",
        "population_by_gate": {k: dict(v) for k, v in sorted(composition.items())},
        "gate1_score_tertile_cuts": [round(float(q1), 4), round(float(q2), 4)],
        "strata": result}, load, extra=[POPULATION, BROAD_POOL, QUERIES])


G0_CELLS = [
    ("Qwen3-0.6B", "both", RAW, 1.5e-6), ("Qwen3-4B", "both", RAW, 1.5e-6),
    ("zerank-2", "both", RAW, 0.30), ("FollowIR-7B", "both", RAW, 0.30),
    ("ctxl-6B", "both", RAW, 0.30), ("bge-gemma", "both", RAW, 0.30),
    ("bge-gemma", "slot_vendor", RAW, 0.30),
    ("Qwen3-0.6B", "none", None, 1.5e-6), ("Qwen3-0.6B", "slot", None, 1.5e-6),
    ("Qwen3-4B", "none", None, 1.5e-6), ("Qwen3-4B", "slot", None, 1.5e-6),
]


def compare_scores(p_new: Path, p_ref: Path, tolerance: float) -> dict:
    a = {json.loads(b)["query_id"]: json.loads(b) for b in p_ref.open() if b.strip()}
    b = {json.loads(x)["query_id"]: json.loads(x) for x in p_new.open() if x.strip()}
    wa, wb = wins_per_row(p_ref), wins_per_row(p_new)
    flip = sorted(k for k in set(wa) & set(wb) if wa[k] != wb[k])
    limit = 0.0
    for q in set(a) & set(b):
        sb = dict(zip(b[q]["ranking"], b[q]["scores"]))
        for c, s in zip(a[q]["ranking"], a[q]["scores"]):
            limit = max(limit, abs(s - sb.get(c, float("inf"))))
    return {"same_query_set": set(a) == set(b), "rows": len(b), "flips": len(flip),
            "flip_rows": flip[:20], "max_abs_score_diff": limit, "tolerance": tolerance,
            "resolved_reference": sum(wa.values()), "resolved_rerun": sum(wb.values()),
            "passed": set(a) == set(b) and not flip and limit <= tolerance}


def section_gate():
    result, used = {}, []
    for model, cell, folder, tol in G0_CELLS:
        p_new = G0 / cell_file_name(model, cell, "paired")
        p_ref = (folder / cell_file_name(model, cell, "paired")) if folder else frozen_cell(model, cell, "paired")
        for p in (p_new, p_ref):
            if not p.exists():
                raise MissingInput(str(p))
        result[f"{model}|{cell}"] = compare_scores(p_new, p_ref, tol)
        used += [p_new, p_ref]
    load = CellLoader()
    anchor = {k for k in result if k.startswith("Qwen3") and k.endswith(("|none", "|slot"))}
    write_table("gate_g0", {
        "what": "G0: the seven original paired cells scored again after the GPU driver and the scoring "
                "scripts changed. A cell passes with the same queries, zero flips, and its largest score "
                "difference within tolerance. The four Qwen3 none and slot cells are the references of "
                "the new Qwen3 paired cells. Their comparison with the Qwen3 score files in "
                "study/retrieval/results/raw/ (the t2 files) is reported, not used as a gate.",
        "gate_cells_all_passed": all(h["passed"] for k, h in result.items() if k not in anchor),
        "anchor_cells_identical_to_t2_files": all(result[k]["passed"] for k in anchor),
        "cells": result}, load, extra=used)


def section_gemma():
    load = CellLoader()
    m = "bge-gemma"
    result = {}
    for route, path_name in (("paired", "paired"), ("luas", "broad_original")):
        def cell_for(s, j=route):
            if s == "none" or (s == "slot_vendor" and j == "paired"):
                return frozen_cell(m, s, j)
            return new_cell(m, s, j)
        tests = {"slot_vendor_vs_none": contrast(load(cell_for("none")), load(cell_for("slot_vendor")), m=3),
               "both_vendor_vs_slot_vendor": contrast(load(cell_for("slot_vendor")),
                                                     load(cell_for("both_vendor")), m=3),
               "wrong_vendor_vs_both_vendor": contrast(load(cell_for("both_vendor")),
                                                      load(cell_for("wrong_vendor")), m=3)}
        contrast_family(tests)
        ref = {"slot_vs_none_appended_prompt": contrast(load(frozen_cell(m, "none", route)),
                                                         load(frozen_cell(m, "slot", route))),
                 "both_vs_slot_appended_prompt": contrast(load(frozen_cell(m, "slot", route)),
                                                         load(frozen_cell(m, "both", route)))}
        result[path_name] = {"documented_placement": tests, "appended_prompt_reference": ref}
    write_table("gemma_vendor", {"what": "bge-gemma with the profile inside a README-form prompt "
                                   "that ends with the documented Yes/No request",
                           "tracks": result}, load)


def section_orthogonal():
    load = CellLoader()
    result = {}
    for cell in ORTHOGONAL_CELLS:
        vs_blind, vs_slot = {}, {}
        for m in MODEL:
            S_ = load(new_cell(m, cell, "paired"))
            vs_blind[m] = contrast(load(paired_reference(m, "none")), S_, m=len(MODEL))
            vs_slot[m] = contrast(load(paired_reference(m, "slot")), S_, m=len(MODEL))
        contrast_family(vs_blind)
        contrast_family(vs_slot)
        result[cell] = {"vs_blind": vs_blind, "vs_full_profile_slot": vs_slot,
                      "summary_vs_blind": summarize(vs_blind)}
    write_table("slot_orthogonal", {"what": "paired track, one profile field in the slot or one field "
                                     "held fixed; references are each model's blind and full-"
                                     "profile slot cells",
                             "bge_gemma_rows": "appended placement of the frozen paper cells, "
                                               "secondary (pre-registration 11.1 a)",
                             "cells": result}, load)


def coincident_label_rows():
    from score_arms import wrong_dept
    from label_map import candidate_meta, load_id_label_map
    from revision_labels import wrong_semester, observed_semesters
    mapping = load_id_label_map()
    all_depts = sorted({m["department"] for m in mapping.values() if m and m.get("department")})
    all_sems = observed_semesters(mapping)
    rows = [json.loads(b) for b in POOL_PAIRED.open() if b.strip()]
    profile = {r["query_id"]: (r["profile"]["department_name"], r["profile"]["current_semester"])
              for r in rows}
    others = {}
    for r in rows:
        partner = [x for x in rows if x["pair_id"] == r["pair_id"] and x["query_id"] != r["query_id"]]
        others[r["query_id"]] = profile[partner[0]["query_id"]] if partner else (None, None)
    hit = {"wrong": set(), "both_semsalah": set(), "both_duasalah": set()}
    for r in rows:
        text = {c["id"]: c["text"] or "" for c in r["cands"]}
        meta = {c["id"]: candidate_meta(mapping, c) for c in r["cands"]}
        asker, other = profile[r["query_id"]], others[r["query_id"]]
        counterpart = r["counterpart"] or []
        counterpart = [counterpart] if isinstance(counterpart, str) else counterpart
        for cid, target in [(g, other) for g in r["gold"]] + [(c, asker) for c in counterpart]:
            if cid not in meta or meta[cid] is None:
                continue
            d_wrong = wrong_dept(meta[cid]["department"], text[cid], all_depts)
            s_wrong = wrong_semester(meta[cid]["semester"], text[cid], all_sems)
            key = f"{r['query_id']}|"
            if d_wrong == target[0]:
                hit["wrong"].add(key)
                hit["both_duasalah"].add(key)
            if s_wrong == target[1]:
                hit["both_semsalah"].add(key)
                hit["both_duasalah"].add(key)
    return {k: sorted(v) for k, v in hit.items()}


def section_factorial():
    load = CellLoader()
    def cell_for(m, s):
        return frozen_cell(m, s, "paired") if s in ("both", "wrong") else (
            paired_reference(m, s) if s == "slot" else new_cell(m, s, "paired"))
    design = {
        "program_correct_vs_wrong_at_correct_semester": ("wrong", "both"),
        "program_correct_vs_wrong_at_wrong_semester": ("both_duasalah", "both_semsalah"),
        "semester_correct_vs_wrong_at_correct_program": ("both_semsalah", "both"),
        "semester_correct_vs_wrong_at_wrong_program": ("both_duasalah", "wrong"),
        "random_label_vs_slot": ("slot", "label_acak"),
        "uniform_label_vs_slot": ("slot", "label_seragam"),
    }
    hit = coincident_label_rows()
    result, table = {}, {}
    for name, (sa, sb) in design.items():
        res, clean_res = {}, {}
        for m in MODEL:
            A, B = load(cell_for(m, sa)), load(cell_for(m, sb))
            res[m] = contrast(A, B, m=len(MODEL))
            drop = set(hit.get(sa, [])) | set(hit.get(sb, []))
            if drop:
                clean_res[m] = contrast({k: v for k, v in A.items() if k not in drop},
                                        {k: v for k, v in B.items() if k not in drop},
                                        m=len(MODEL))
        contrast_family(res)
        if clean_res:
            contrast_family(clean_res)
        result[name] = {"cell_A": sa, "cell_B": sb, "models": res, "summary": summarize(res),
                       "without_coincident_rows": clean_res or None}
    for m in MODEL:
        table[m] = {s: sum(load(cell_for(m, s)).values()) for s in
                    ("slot", "both", "wrong", "both_semsalah", "both_duasalah",
                     "label_acak", "label_seragam")}
    write_table("label_factorial", {
        "what": "paired track, full profile in the slot; labels with correct or wrong program "
                "and semester, a random label and a uniform label",
        "reading_rule": "primary analysis on all comparable rows; without_coincident_rows is "
                        "the pre-specified sensitivity; a claim that label content matters "
                        "needs the primary contrast significant and the sensitivity contrast "
                        "in the same direction (pre-registration 11.1 b)",
        "bge_gemma_rows": "appended placement, secondary (pre-registration 11.1 a)",
        "resolved_rows_out_of_96": table,
        "coincident_rows": {k: {"n": len(v), "rows": v} for k, v in hit.items()},
        "contrasts": result}, load, extra=[POOL_PAIRED])


def section_random_broad():
    for p in (POOL_R1, POPULATION, RAW_R1 / "r1_dits_xr1_baris_B.jsonl"):
        if not p.exists():
            raise MissingInput(str(p))
    load = CellLoader()
    pop = {r["query_id"]: r for r in json.loads(POPULATION.read_text())["records"]}
    grp = group_sources(POOL_R1)
    gate = {g: pop[v["source_query"]]["gate"] for g, v in grp.items()}
    new_only = {g for g, v in grp.items() if not pop[v["source_query"]]["read_by_original_miner"]}
    def source(m, s):
        return new_cell(m, s, "r1")
    vendor = {"bge-gemma": {"slot": "slot_vendor", "both": "both_vendor"}}
    book = book_clusters(POOL_R1)
    result, per_gate, unread, gemma_appended = {}, {}, {}, {}
    for fam, fam_pairs in (("slot_vs_none", ("none", "slot")), ("both_vs_slot", ("slot", "both"))):
        res = contrast_family_table(load, source, fam_pairs, clusters=book, replacement=vendor)
        result[fam] = {"models": res, "summary": summarize(res)}
        per_gate[fam] = {}
        for g_name in sorted(set(gate.values())):
            subset = {g for g, v in gate.items() if v == g_name}
            r = contrast_family_table(load, source, fam_pairs, subset=subset, replacement=vendor)
            per_gate[fam][g_name] = {"n_groups": len(subset), "models": r, "summary": summarize(r)}
        r = contrast_family_table(load, source, fam_pairs, subset=new_only, replacement=vendor)
        unread[fam] = {"n_groups": len(new_only), "models": r, "summary": summarize(r)}
        gemma_appended[fam] = contrast(load(source("bge-gemma", fam_pairs[0])), load(source("bge-gemma", fam_pairs[1])))
    blind = {m: {"resolved": sum(load(source(m, "none")).values()), "n_groups": len(grp)}
             for m in MODEL}
    dits = load(RAW_R1 / "r1_dits_xr1_baris_B.jsonl")
    n_rows = len(load(source(MODEL[0], "none")))
    write_table("random_broad", {
        "what": "stratified random broad sample (gate version x program, seed 20261012), "
                "600 groups; blind baseline is one row per group",
        "n_groups": len(grp), "n_rows": n_rows,
        "blind_invariant": blind,
        "bge_gemma": "documented placement (slot_vendor, both_vendor) in every family; its "
                     "appended-placement cells are reported as secondary (pre-registration 11.1 a)",
        "contrasts": result, "by_gate": per_gate,
        "groups_never_read_by_original_miner": unread,
        "bge_gemma_appended_placement_secondary": gemma_appended,
        "metadata_references": {"DITS_resolved": sum(dits.values()), "DITS_rows": len(dits),
                                "hard_filter_resolved_by_construction": n_rows,
                                "hard_filter_note": "not scored: the filter keeps only the "
                                                    "asker's program, so every other gold "
                                                    "leaves the pool by construction"}},
        load, extra=[POOL_R1, POPULATION])


def section_sensitivity():
    load = CellLoader()
    result = {}
    for route, name in (("paired", "paired"), ("luas", "broad_original")):
        res = contrast_family_table(load, lambda m, s, j=route: frozen_cell(m, s, j), ("none", "slot"))
        result[name] = {}
        for m, r in res.items():
            u = r["largest_gain_pp"]
            result[name][m] = {"largest_gain_pp_observed": u, "grid": {
                f"{e:.2f}": {"model_no_signal": round(u / (1 - e), 3),
                             "model_reversed": round(u / (1 - 2 * e), 3),
                             "worst_case": round((u + 100 * e) / (1 - e), 3)}
                for e in (0.05, 0.10, 0.20)}}
    write_table("gold_sensitivity", {
        "what": "slot bounds when a share e of rows is touched by an invalid gold. Two models: invalid "
                "golds carry no signal, d / (1 - e), or reverse it, d / (1 - 2e). Worst case, any effect "
                "on the invalid rows: (U + e) / (1 - e). e is the row share of pre-registration 11.1 g "
                "and 11.2, which human_ratings.json takes from the lecturers' ratings of each track", "tracks": result}, load,
        exploratory=True)


SECTIONS = {"klaster": section_cluster, "paired": section_paired, "seragam": section_uniform_semester,
          "seleksi": section_selection, "gerbang": section_gate, "gemma": section_gemma,
          "ortogonal": section_orthogonal, "faktorial": section_factorial,
          "luas_acak": section_random_broad, "sensitivitas": section_sensitivity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bagian", choices=list(SECTIONS) + ["semua"], required=True)
    args = ap.parse_args()
    if args.bagian != "semua":
        try:
            SECTIONS[args.bagian]()
        except MissingInput as e:
            raise SystemExit(f"ABORT: input missing, table not written: {e}")
        return
    skipped = {}
    for name, f in SECTIONS.items():
        try:
            f()
        except MissingInput as e:
            skipped[name] = str(e)
    for name, reason in skipped.items():
        print(f"skipped {name}: missing {reason}")


if __name__ == "__main__":
    main()
