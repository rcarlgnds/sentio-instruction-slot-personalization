"""Results of the lecturers' check of the gold passages, from the coded rating table.

Lecturers who are not authors rated each gold as a full, partial, or no answer to its question. This
script reads results/human_ratings/human_ratings.csv, which holds the settled answers. It reports whether
every settled answer of the non-statistics items is the median of its three ratings. It also reports the
invalid golds, the agreement (Cohen's kappa and Gwet's AC1), and the slot limits rescaled for gold
errors. Writes results/tables/human_ratings.json, the source of Table 6.
"""

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import revision_analysis as A  # noqa: E402
from revision_statistics import SEED, contrast, contrast_family, group_id  # noqa: E402

RATINGS = A.Q2 / "results/human_ratings/human_ratings.csv"
RAW_R2 = A.Q2 / "results/raw_r2"
RAW_P2 = A.Q2 / "results/raw_p2"
OUT = A.TAB / "human_ratings.json"
PREREG = A.ROOT / "result/paper_q2/PRAREGISTRASI_R1_IJIES.md"
POOL = {"paired": A.POOL_PAIRED, "r2": RAW_R2 / "pools/pools_xpair_r2_ids.jsonl",
        "p2": RAW_P2 / "pools/pools_p2_ids.jsonl", "broad_original": A.BROAD_POOL,
        "broad_r1_resample": A.POOL_R1}
TRACKS = ("paired", "r2", "audit", "p2")
POPULATION = {"paired": "paired", "r2": "r2", "p2": "p2", "audit": "broad_original"}
ANSWERS = ("penuh", "sebagian", "tidak")
CLEAR = ("ya", "tidak")
RATERS = ("S1", "S2", "N1", "N2", "N3")
RULES = ("consensus", "all_raters")
N_BOOT = 2000
VENDOR = {"bge-gemma": {"slot": "slot_vendor", "both": "both_vendor", "wrong": "wrong_vendor"}}
FAMILIES = {"slot_vs_none": ("none", "slot"), "both_vs_slot": ("slot", "both"), "wrong_vs_both": ("both", "wrong")}
BUSINESS = ("acct", "mgmt", "mktg", "bana")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def valid(answer: str) -> bool:
    return answer in ("penuh", "sebagian")


def kappa(a: list, b: list, classes) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(k) / n) * (b.count(k) / n) for k in classes)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def ac1(a: list, b: list, classes) -> float:
    n, q = len(a), len(classes)
    po = sum(x == y for x, y in zip(a, b)) / n
    pi = {k: (a.count(k) + b.count(k)) / (2 * n) for k in classes}
    pe = sum(v * (1 - v) for v in pi.values()) / (q - 1)
    return None if pe >= 1 else (po - pe) / (1 - pe)


def agree(a: list, b: list, classes=None) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def bootstrap(per_group: dict, fn, classes) -> dict:
    groups = sorted(per_group)
    a = [x for g in groups for x, _ in per_group[g]]
    b = [y for g in groups for _, y in per_group[g]]
    point = fn(a, b, classes)
    rng = np.random.default_rng(SEED)
    draws, undefined = [], 0
    for _ in range(N_BOOT):
        pick = [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        aa = [x for g in pick for x, _ in per_group[g]]
        bb = [y for g in pick for _, y in per_group[g]]
        v = fn(aa, bb, classes)
        if v is None:
            undefined += 1
        else:
            draws.append(v)
    interval = [round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)] \
        if draws else None
    return {"estimate": None if point is None else round(point, 4), "interval_95": interval,
            "resamples_undefined": undefined}


def clopper_pearson(x: int, n: int) -> dict:
    lo = 0.0 if x == 0 else float(beta.ppf(0.025, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(0.975, x + 1, n - x))
    upper = 1.0 if x == n else float(beta.ppf(0.95, x + 1, n - x))
    return {"count": x, "n": n, "share": round(x / n, 4) if n else None, "exact_95": [round(lo, 4), round(hi, 4)],
            "one_sided_95_upper": round(upper, 4)}


def rows_per_group(pool: Path) -> list:
    size = defaultdict(int)
    for b in pool.open():
        if b.strip():
            size[group_id(json.loads(b)["query_id"])] += 1
    return [k for k in size.values() for _ in range(k)]


def e_rows(e_gold: float, k_per_row: list) -> float:
    return round(float(np.mean([1 - (1 - e_gold) ** k for k in k_per_row])), 4)


def validity_summary(items: list, ok: dict, k_rows: list) -> dict:
    x = sum(not ok[it["key"]] for it in items)
    out = {"invalid_golds": clopper_pearson(x, len(items))}
    roles = sorted({it["role"] for it in items})
    if len(roles) > 1:
        out["by_role"] = {r: clopper_pearson(sum(not ok[it["key"]] for it in items if it["role"] == r),
                                             sum(it["role"] == r for it in items)) for r in roles}
    groups = defaultdict(list)
    for it in items:
        groups[it["group"]].append(ok[it["key"]])
    out["groups_all_valid"] = clopper_pearson(sum(all(v) for v in groups.values()), len(groups))
    out["e_rows"] = {"from_share": e_rows(x / len(items), k_rows),
                     "from_one_sided_upper": e_rows(out["invalid_golds"]["one_sided_95_upper"], k_rows)}
    return out


def clarity(items: list) -> dict:
    question = defaultdict(lambda: [True, True])
    for it in items:
        question[it["group"]][0] &= it["r1"][1] == "ya"
        question[it["group"]][1] &= it["r2"][1] == "ya"
    a = ["ya" if v[0] else "tidak" for v in question.values()]
    b = ["ya" if v[1] else "tidak" for v in question.values()]
    k = kappa(a, b, CLEAR)
    return {"questions": len(question), "unclear_rater_1": a.count("tidak"), "unclear_rater_2": b.count("tidak"),
            "unclear_both": sum(x == y == "tidak" for x, y in zip(a, b)),
            "agreement": round(agree(a, b), 4), "kappa": None if k is None else round(k, 4)}


def rater_agreement(items: list) -> dict:
    three, two = defaultdict(list), defaultdict(list)
    for it in items:
        three[it["group"]].append((it["r1"][0], it["r2"][0]))
        two[it["group"]].append(("sah" if valid(it["r1"][0]) else "tidak", "sah" if valid(it["r2"][0]) else "tidak"))
    return {"three_classes": {n: bootstrap(three, f, ANSWERS) for n, f in
                              (("raw_agreement", agree), ("cohen_kappa", kappa), ("gwet_ac1", ac1))},
            "valid_vs_not": {n: bootstrap(two, f, ("sah", "tidak")) for n, f in
                             (("raw_agreement", agree), ("cohen_kappa", kappa), ("gwet_ac1", ac1))}}


def judge_vs_raters(items: list, ok: dict) -> dict:
    full = [it for it in items if it["judge"]["full_both"]]
    j = [it["judge"]["wording_a"] for it in items]
    d1, d2 = [it["r1"][0] for it in items], [it["r2"][0] for it in items]
    table = defaultdict(int)
    for it in items:
        table[f"judge_{'valid' if valid(it['judge']['wording_a']) else 'invalid'}|raters_"
              f"{'valid' if ok[it['key']] else 'invalid'}"] += 1
    by_class = {}
    for c in ANSWERS:
        sub = [it for it in items if it["judge"]["wording_a"] == c]
        if sub:
            by_class[c] = clopper_pearson(sum(ok[it["key"]] for it in sub), len(sub))
    clear = defaultdict(int)
    for it in items:
        clear[f"judge_{it['judge']['clear']}|rater_1_{it['r1'][1]}|rater_2_{it['r2'][1]}"] += 1

    def r4(x):
        return None if x is None else round(x, 4)
    return {"judge_full_answers_rated_valid": clopper_pearson(sum(ok[it["key"]] for it in full), len(full)) if full
            else None,
            "judge_vs_raters_2x2": dict(table), "rated_valid_by_judge_class_wording_a": by_class,
            "three_class_kappa": {"judge_rater_1": r4(kappa(j, d1, ANSWERS)),
                                  "judge_rater_2": r4(kappa(j, d2, ANSWERS)),
                                  "rater_1_rater_2": r4(kappa(d1, d2, ANSWERS))},
            "clarity_items": dict(clear)}


def slot_bounds() -> dict:
    def models(name):
        return {m: r["largest_gain_pp"] for m, r in
                json.loads((A.TAB / name).read_text())["contrasts"]["slot_vs_none"]["models"].items()}
    paired = models("r1_cluster_paired.json")
    gv = json.loads((A.TAB / "r1_gemma_vendor.json").read_text())
    paired["bge-gemma"] = gv["tracks"]["paired"]["documented_placement"]["slot_vendor_vs_none"]["largest_gain_pp"]
    return {"paired": paired, "r2": models("r2_broad.json"), "p2": models("p2_paired.json"),
            "broad_original": models("r1_cluster_broad.json"), "broad_r1_resample": models("r1_random_broad.json")}


def rescale(bounds: dict, e: float) -> dict:
    out = {}
    for m, u in bounds.items():
        out[m] = {"bound_pp": u, "no_signal": round(u / (1 - e), 3),
                  "reversed": round(u / (1 - 2 * e), 3) if e < 0.5 else None,
                  "worst_case": round((u + 100 * e) / (1 - e), 3)}
    return out


def point_to_track(track: str):
    if track == "paired":
        A.PLAN, A.STATUS_R1 = A.RAW_R1 / "_rencana_r1.json", A.RAW_R1 / "_status_r1.json"

        def source(m, s):
            if m == "bge-gemma" and s in ("both_vendor", "wrong_vendor"):
                return A.new_cell(m, s, "paired")
            return A.frozen_cell(m, s, "paired")
    elif track == "r2":
        A.PLAN, A.STATUS_R1 = RAW_R2 / "_rencana_r2.json", RAW_R2 / "_status_r1.json"
        A.TAG["r2"] = "xr2"

        def source(m, s):
            return RAW_R2 / A.cell_file_name(m, s, "r2")
    else:
        A.PLAN, A.STATUS_R1 = RAW_P2 / "_rencana_p2.json", RAW_P2 / "_status_r1.json"
        A.TAG["p2"] = "xp2"

        def source(m, s):
            return RAW_P2 / A.cell_file_name(m, s, "p2")
    return A.CellLoader(), source


def family_on_rows(load, source, pair, keep) -> dict:
    res = {}
    for m in A.MODEL:
        swap = VENDOR.get(m, {})
        sa, sb = swap.get(pair[0], pair[0]), swap.get(pair[1], pair[1])
        X = {k: v for k, v in load(source(m, sa)).items() if keep(k.split("|", 1)[0])}
        Y = {k: v for k, v in load(source(m, sb)).items() if keep(k.split("|", 1)[0])}
        common = set(X) & set(Y)
        if len({group_id(k.split("|", 1)[0]) for k in common}) < 2:
            return {"n_rows": len(common), "note": "fewer than two groups, not computed"}
        res[m] = contrast(X, Y, m=len(A.MODEL))
    contrast_family(res)
    return {"models": res, "summary": A.summarize(res)}


def recompute(track: str, items: list, ok: dict, loaders: list) -> dict:
    load, source = point_to_track(track)
    loaders.append(load)
    rated = {it["group"] for it in items}
    rows_valid = {it["row"] for it in items if ok[it["key"]]}
    groups_valid = {g for g in rated if all(ok[it["key"]] for it in items if it["group"] == g)}
    subsets = {"rated_groups": lambda q: group_id(q) in rated,
               "rows_own_gold_valid": lambda q: q in rows_valid,
               "groups_all_golds_valid": lambda q: group_id(q) in groups_valid}
    return {"n_groups_rated": len(rated), "n_rows_own_gold_valid": len(rows_valid),
            "n_groups_all_golds_valid": len(groups_valid),
            "subsets": {name: {f: family_on_rows(load, source, pair, keep) for f, pair in FAMILIES.items()}
                        for name, keep in subsets.items()}}


def pair_of(row: dict) -> tuple:
    a, b = ("S1", "S2") if row["rater_group"] == "statistics" else ("N1", "N2")
    return (row[f"{a}_answer"], row[f"{a}_clear"]), (row[f"{b}_answer"], row[f"{b}_clear"])


def read_items(rows: list) -> list:
    items = []
    for row in rows:
        r1, r2 = pair_of(row)
        judge = None
        if row["judge_wording_a"]:
            judge = {"clear": row["judge_clear"], "wording_a": row["judge_wording_a"], "wording_b": row["judge_wording_b"],
                     "full_both": row["judge_full_both"] == "1"}
        items.append({"key": (row["track"], row["item_id"]), "track": row["track"], "group": row["group"],
                      "row": row["query_id"], "role": row["gold_role"], "judge": judge, "book": row["book"],
                      "r1": r1, "r2": r2})
    return items


def rule_labels(rows: list) -> dict:
    labels = {r: {} for r in RULES}
    for row in rows:
        key = (row["track"], row["item_id"])
        settled = row["settled_answer"] or pair_of(row)[0][0]
        labels["consensus"][key] = valid(settled)
        if labels["consensus"][key] != (row["valid_settled"] == "1"):
            raise SystemExit(f"ABORT: item {row['item_id']}: valid_settled does not follow from its answers")
        answers = [row[f"{k}_answer"] for k in RATERS if row[f"{k}_answer"]]
        labels["all_raters"][key] = "tidak" not in answers
    return labels


def track_results(items: list, labels: dict, loaders: list) -> dict:
    bounds = slot_bounds()
    k_rows = {t: rows_per_group(POOL[t]) for t in POOL}
    out = {}
    for track in TRACKS:
        sub = [it for it in items if it["track"] == track]
        bag = {"n_items": len(sub), "n_groups": len({it["group"] for it in sub}),
               "answer_counts": {f"rater_{i}": {a: sum(it[f"r{i}"][0] == a for it in sub) for a in ANSWERS} for i in (1, 2)},
               "agreement": rater_agreement(sub), "question_clarity": clarity(sub),
               "validity": {r: validity_summary(sub, labels[r], k_rows[POPULATION[track]]) for r in RULES}}
        if sub[0]["judge"] is not None:
            bag["judge_vs_raters"] = {r: judge_vs_raters(sub, labels[r]) for r in RULES}
        e = {r: bag["validity"][r]["e_rows"]["from_one_sided_upper"] for r in RULES}
        if track == "audit":
            bag["sensitivity_bounds"] = {pop: {r: rescale(bounds[pop], e[r]) for r in RULES}
                                         for pop in ("broad_original", "broad_r1_resample")}
            bag["sensitivity_bounds_note"] = ("e of the audited groups, from the original sample; the R1 "
                                              "resample used the same partner rule")
        else:
            bag["sensitivity_bounds"] = {r: rescale(bounds[track], e[r]) for r in RULES}
            bag["recomputed_contrasts"] = {r: recompute(track, sub, labels[r], loaders) for r in RULES}
        out[track] = bag
    return out


def third_rater(rows: list) -> dict:
    non = [r for r in rows if r["rater_group"] == "non_statistics"]
    out = {"items": len(non),
           "answer_counts": {k: {a: sum(r[f"{k}_answer"] == a for r in non) for a in ANSWERS} for k in ("N1", "N2", "N3")},
           "validity_agreement": {}, "by_track": {}}
    for first, other in (("N3", "N1"), ("N3", "N2"), ("N1", "N2")):
        per_group = defaultdict(list)
        for r in non:
            per_group[f"{r['track']}|{r['group']}"].append(
                ("sah" if valid(r[f"{first}_answer"]) else "tidak", "sah" if valid(r[f"{other}_answer"]) else "tidak"))
        out["validity_agreement"][f"{first}_vs_{other}"] = {
            n: bootstrap(per_group, f, ("sah", "tidak")) for n, f in
            (("raw_agreement", agree), ("cohen_kappa", kappa), ("gwet_ac1", ac1))}
        out["validity_agreement"][f"{first}_vs_{other}"]["agreeing_items"] = sum(
            valid(r[f"{first}_answer"]) == valid(r[f"{other}_answer"]) for r in non)
    for track in sorted({r["track"] for r in non}):
        sub = [r for r in non if r["track"] == track]
        unclear = {k: {r["group"] for r in sub if r[f"{k}_clear"] == "tidak"} for k in ("N1", "N2", "N3")}
        out["by_track"][track] = {
            "items": len(sub),
            "rejected_by_N3": sum(r["N3_answer"] == "tidak" for r in sub),
            "rejected_by_N3_alone": sum(r["N3_answer"] == "tidak" and r["N1_answer"] != "tidak"
                                        and r["N2_answer"] != "tidak" for r in sub),
            "items_N3_unclear": sum(r["N3_clear"] == "tidak" for r in sub),
            "questions_unclear": {k: len(v) for k, v in unclear.items()},
            "questions_unclear_N3_alone": len(unclear["N3"] - unclear["N1"] - unclear["N2"])}
    return out


def settlement(rows: list) -> dict:
    order = {"tidak": 0, "sebagian": 1, "penuh": 2}
    non = [r for r in rows if r["rater_group"] == "non_statistics"]
    differ = [r for r in non if r["N1_answer"] != r["N2_answer"]]
    median = all(r["settled_answer"] == sorted((r["N1_answer"], r["N2_answer"], r["N3_answer"]), key=order.get)[1]
                 for r in differ)
    agreed = all(r["settled_answer"] == r["N1_answer"] for r in non if r["N1_answer"] == r["N2_answer"])
    st = [r for r in rows if r["rater_group"] == "statistics"]
    return {"settled_by": dict(sorted(Counter(r["settled_by"] for r in rows).items())),
            "non_statistics_items_n1_n2_differ": len(differ), "settled_by_median_of_three": median and agreed,
            "statistics_items_answers_differ": sum(r["S1_answer"] != r["S2_answer"] for r in st),
            "statistics_items_validity_differs": sum(valid(r["S1_answer"]) != valid(r["S2_answer"]) for r in st)}


def rated_books(rows: list) -> dict:
    non = [r for r in rows if r["rater_group"] == "non_statistics"]
    prefix = Counter(r["book"].split("-", 1)[0] for r in non)
    business = sum(prefix[p] for p in BUSINESS)
    return {"non_statistics_items": len(non), "by_book_prefix": dict(sorted(prefix.items())),
            "business_prefixes": list(BUSINESS), "business_items": business,
            "business_share": round(business / len(non), 4)}


def statistics_raters(rows: list) -> dict:
    st = [r for r in rows if r["rater_group"] == "statistics"]
    mined = [r for r in st if r["gold_role"] == "mined_partner_gold"]
    judge_no = [r for r in mined if r["judge_wording_a"] == "tidak"]
    return {"items": len(st), "rated_not_at_all": {k: sum(r[f"{k}_answer"] == "tidak" for r in st) for k in ("S1", "S2")},
            "answers_differ": sum(r["S1_answer"] != r["S2_answer"] for r in st),
            "validity_differs": sum(valid(r["S1_answer"]) != valid(r["S2_answer"]) for r in st),
            "mined_partners": len(mined), "mined_partners_judge_not_at_all": len(judge_no),
            "of_those_valid_for_both_raters": sum(valid(r["S1_answer"]) and valid(r["S2_answer"]) for r in judge_no)}


def kit(rows: list) -> dict:
    pools = {"paired": POOL["paired"], "r2": POOL["r2"], "p2": POOL["p2"], "audit": POOL["broad_original"]}
    out = {}
    for track, path in pools.items():
        gold = defaultdict(set)
        for b in path.open(encoding="utf-8"):
            if b.strip():
                r = json.loads(b)
                g = r.get("gold") or []
                gold[group_id(r["query_id"])].update((r["query_id"], c) for c in (g if isinstance(g, list) else [g]))
        sub = [r for r in rows if r["track"] == track]
        rated = {r["group"] for r in sub}
        pairs = {(r["query_id"], r["chunk_id"]) for r in sub}
        out[track] = {"items": len(sub), "groups": len(rated), "pool_groups": len(gold),
                      "every_gold_of_the_rated_groups": pairs == set().union(*(gold[g] for g in rated)),
                      "every_group_of_the_pool": rated == set(gold)}
    return out


def main():
    for p in (RATINGS, *POOL.values()):
        if not p.exists():
            raise SystemExit(f"ABORT: input missing, table not written: {p}")
    rows = list(csv.DictReader(RATINGS.read_text(encoding="utf-8").splitlines()))
    items = read_items(rows)
    labels = rule_labels(rows)
    loaders = []
    tracks = track_results(items, labels, loaders)
    inputs = {str(p.relative_to(A.ROOT)): sha(p) for p in (RATINGS, *POOL.values())}
    for name in ("r1_cluster_paired.json", "r1_gemma_vendor.json", "r2_broad.json", "p2_paired.json",
                 "r1_cluster_broad.json", "r1_random_broad.json"):
        inputs[str((A.TAB / name).relative_to(A.ROOT))] = sha(A.TAB / name)
    for load in loaders:
        inputs.update(load.used)
    out = {"created": datetime.now(timezone.utc).isoformat(),
           "script": "study/channel/scripts/11_human_ratings/human_ratings_analysis.py",
           "what": "the lecturers' ratings of the gold passages, per track, under the settled labels and under the "
                   "rule that every rater must accept a gold",
           "preregistration": {"path": str(PREREG.relative_to(A.ROOT))},
           "analysis_status": {"consensus": "pre-registered, "
                                            "except the split of the kit by field and the median rule of the third "
                                            "rater, which were set later",
                               "all_raters": "not pre-registered, a sensitivity check added after the ratings were read"},
           "label_source": "lecturers who are not authors, blind to the student's program and semester, the book, "
                           "and every model output",
           "validity_rules": list(RULES), "primary_rule": "consensus",
           "seed": SEED, "bootstrap_resamples": N_BOOT, "inputs": inputs,
           "kit": kit(rows), "settlement": settlement(rows), "tracks": tracks, "third_rater": third_rater(rows),
           "rated_books": rated_books(rows), "statistics_raters": statistics_raters(rows)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"written: {OUT}")
    for track, bag in tracks.items():
        v = bag["validity"]["consensus"]["invalid_golds"]
        print(f"{track:7} items {bag['n_items']:3}  invalid golds {v['count']}/{v['n']}  upper {v['one_sided_95_upper']}")


if __name__ == "__main__":
    main()
