"""Exact McNemar tests for the paired contrasts of the Qwen3 models.

Two setups scored on the same 96 rows are compared on the rows where they disagree, with an exact
two-sided binomial test and Holm within each family. A profile-blind setup sits at 48 of 96 by
construction, so it is never tested against one half. Writes results/tables/mcnemar_paired.json.
"""

import os
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import sys  # noqa: E402
for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import metrics as M
from exact_tests import holm as holm_list, mcnemar_exact

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW_N1 = ROOT / "study/retrieval/results/raw"
Q2 = ROOT / "study/channel/results"
SHARED = ROOT / "study/_shared"

ARM = {
    "C3_default": RAW_N1 / "qwen_scores_0.6b_paired_default.jsonl",
    "C3_profil": RAW_N1 / "qwen_scores_0.6b_paired_profile.jsonl",
    "C3_docmeta": Q2 / "raw/scores_qwen3rr_Qwen3-Reranker-0.6B_paired_default_docmeta.jsonl",
    "C3_docmeta_profil": Q2 / "raw/scores_qwen3rr_Qwen3-Reranker-0.6B_paired_profile_docmeta.jsonl",
    "C4_default": RAW_N1 / "qwen_scores_4b_paired_default.jsonl",
    "C4_profil": RAW_N1 / "qwen_scores_4b_paired_profile.jsonl",
    "C4_docmeta": Q2 / "raw/scores_qwen3rr_Qwen3-Reranker-4B_paired_default_docmeta.jsonl",
    "C4_docmeta_profil": Q2 / "raw/scores_qwen3rr_Qwen3-Reranker-4B_paired_profile_docmeta.jsonl",
}
N_PAIRED = 96


def fixed_arm() -> dict:
    paired = json.loads((RAW_N1 / "eval_paired.json").read_text())
    qwen = json.loads((ROOT / "study/retrieval/results/tables/"
                       "qwen_rerankers_c3_c4.json").read_text())
    return {
        "B_DITS": (round(paired["B"]["pairwise"] * N_PAIRED), N_PAIRED),
        "D_cascade": (round(paired["D"]["pairwise"] * N_PAIRED), N_PAIRED),
        "filter_metadata": (round(qwen["paired"]["filter_metadata"]["pairwise"]
                                  * N_PAIRED), N_PAIRED),
    }

CONTRASTS = [
    ("C3_default", "C3_profil"),
    ("C3_profil", "C3_docmeta_profil"),
    ("C3_docmeta", "C3_docmeta_profil"),
    ("C4_default", "C4_profil"),
    ("C4_profil", "C4_docmeta_profil"),
    ("C4_docmeta", "C4_docmeta_profil"),
    ("C3_docmeta_profil", "C4_docmeta_profil"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def per_item_results(path: Path) -> dict:
    output = {}
    for r in M.read_ranking(path):
        pos = {c: i for i, c in enumerate(r["ranking"])}
        g = min((pos[c] for c in (r["gold"] or []) if c in pos), default=None)
        o = min((pos[c] for c in (r.get("counterpart") or []) if c in pos),
                default=None)
        if g is not None and o is not None:
            output[r["query_id"]] = g < o
    return output


def wilson(k: int, n: int, z=1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - m), 4), round(min(1.0, c + m), 4)]


def holm(p: dict) -> dict:
    keys = [k for k, _ in sorted(p.items(), key=lambda kv: kv[1])]
    return {k: round(adj, 6) for k, adj in zip(keys, holm_list([p[k] for k in keys]))}


def main():
    lock = json.loads((SHARED / "config_lock.json").read_text())
    missing = [name for name, p in ARM.items() if not p.exists()]
    if missing:
        print("ABORT: per-item scores incomplete, table not written.")
        print("Arms whose file is missing:")
        for name in missing:
            print(f"  {name}: {ARM[name]}")
        print("\nThe per-item score files ship in study/retrieval/results/raw/, so restore them")
        print("from a clean copy of the repository. The saved table in results/tables has the result.")
        raise SystemExit(2)
    per_item, summary = {}, {}
    for name, p in ARM.items():
        if not p.exists():
            print(f"  (skip, not there yet) {name}")
            continue
        d = per_item_results(p)
        per_item[name] = d
        wins = sum(1 for v in d.values() if v)
        n = len(d)
        locked = ("docmeta_profil" not in name and "profil" not in name) or \
                   (name.endswith("_docmeta"))
        summary[name] = {
            "menang": wins, "n": n, "eksak": f"{wins}/{n}",
            "rate": round(wins / n, 6),
            "wilson95_marginal": wilson(wins, n),
            "terkunci_konstruksi": bool(locked and wins * 2 == n),
        }
        mark = " [locked by construction]" if summary[name]["terkunci_konstruksi"] else ""
        print(f"  {name:22} {wins:3d}/{n}{mark}")

    tests, praw = {}, {}
    for a, b in CONTRASTS:
        if a not in per_item or b not in per_item:
            continue
        shared = sorted(set(per_item[a]) & set(per_item[b]))
        n01 = sum(1 for q in shared if not per_item[a][q] and per_item[b][q])
        n10 = sum(1 for q in shared if per_item[a][q] and not per_item[b][q])
        p = mcnemar_exact(n01, n10)
        key = f"{a} vs {b}"
        praw[key] = p
        tests[key] = {
            "n_berpasangan": len(shared),
            "diskordan_a_kalah_b_menang": n01,
            "diskordan_a_menang_b_kalah": n10,
            "selisih_bersih": n01 - n10,
            "p_mcnemar_exact": round(p, 8),
        }
    p_holm = holm(praw)
    for k in tests:
        tests[k]["p_holm_keluarga_posthoc"] = p_holm[k]
        tests[k]["signifikan_0.05"] = p_holm[k] < 0.05
        print(f"  {k:42} discordant {tests[k]['diskordan_a_kalah_b_menang']}"
              f" against {tests[k]['diskordan_a_menang_b_kalah']}"
              f"  p_exact {tests[k]['p_mcnemar_exact']:.4g}"
              f"  p_holm {p_holm[k]:.4g}")

    dits_note = None
    FIXED = fixed_arm()
    if "C4_docmeta_profil" in summary:
        k4 = summary["C4_docmeta_profil"]["menang"]
        b_dits = FIXED["B_DITS"][0]
        delta = b_dits - k4
        span = []
        for c in range(0, 96 - b_dits + 1):
            span.append(mcnemar_exact(c + delta, c))
        dits_note = {
            "kontras": f"B_DITS {b_dits}/96 lawan C4_docmeta_profil {k4}/96",
            "status": ("the per-item DITS output was not stored, so its discordant table cannot be computed "
                       "directly"),
            "batas_analitik_p_exact": [round(min(span), 6), round(max(span), 6)],
            "penalaran": (f"the net discordant difference is fixed at {delta}"
                          f"; the number of DITS losses is at most {96 - b_dits}"
                          f", so the two-sided exact p lies in that range for every possible configuration"),
        }
        print(f"\n  B_DITS {b_dits}/96 vs C4+docmeta {k4}/96: exact p within "
              f"[{min(span):.4g}; {max(span):.4g}] (analytic bound)")

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock["config_version"],
        "_apa_ini": ("Exact per-item McNemar tests for every paired contrast. Overlapping marginal intervals "
                     "do not answer whether two cells differ. p_mcnemar_exact is stored with eight decimals "
                     "and the Holm values with six, so a 0.0 is below what those decimals can show."),
        "aturan_pelaporan": {
            "hitungan_eksak": ("measurement grid 1/96 = 0.0104; report a/96, not four decimals"),
            "null_degenerate": ("the profile-blind arm is fixed at 48/96 algebraically, because the two rows of a pair "
                                "get identical rankings; do not test against the 0.5 null"),
            "keluarga_holm": "a separate post hoc family, m = the number of contrasts in this table",
        },
        "per_arm": summary,
        "arm_tetap_terbit": {k: f"{v[0]}/{v[1]}" for k, v in FIXED.items()},
        "kontras_mcnemar": tests,
        "kontras_terhadap_dits": dits_note,
        "label_source": "deterministic (retesting the frozen score files)",
    }
    out = Q2 / "tables/mcnemar_paired.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    (Q2 / "raw/mcnemar_paired.manifest.json").write_text(json.dumps({
        "run_id": "q2_mcnemar_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/4_slot_vs_baseline/mcnemar.py",
        "config_version": lock["config_version"],
        "label_source": "deterministic",
        "sumber": {k: sha(v) for k, v in ARM.items() if v.exists()},
        "output": {"path": "results/tables/mcnemar_paired.json", "sha256": sha(out),
                   "tingkat_reproduksi": "1_bit_identical"},
    }, indent=2, ensure_ascii=False))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
