import os
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import mcnemar as MC  # noqa: E402
import metrics as M  # noqa: E402
from aggregate_channel_board import run_family, contrast, board_cell, content_fingerprint  # noqa: E402
from equivalence_bounds import analysis  # noqa: E402

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
Q2 = ROOT / "study/channel/results"
RAW = Q2 / "raw"
TAB = Q2 / "tables"
SHARED = ROOT / "study/_shared"
PREREG = "PRAREGISTRASI_LINTAS_ARSITEKTUR.md"
NAMES = "channel_board_paired_architectures"

MODEL = {
    "gemma": "bge-reranker-v2-gemma",
    "ctxl": "ctxl-rerank-v2-instruct-multilingual-6b",
    "followir": "FollowIR-7B",
}
LINEAGE = {"gemma": "Gemma (Google)", "ctxl": "Mistral", "followir": "Mistral, tuned to follow retrieval instructions"}
CORE_CELLS = ["none", "slot", "text", "both", "wrong"]
FROZEN_BOARD = TAB / "channel_board_paired.json"
FROZEN_ZERANK_DEFAULT = RAW / "scores_zerank2_zerank-2_paired_default.jsonl"
FROZEN_FOLLOWIR = {
    "Qwen3-Reranker-0.6B": TAB / "followir_Core17_qwen3rr_Qwen3-Reranker-0.6B.json",
    "Qwen3-Reranker-4B": TAB / "followir_Core17_qwen3rr_Qwen3-Reranker-4B.json",
    "zerank-2": TAB / "followir_Core17_zerank2_zerank-2.json",
}
N1 = ROOT / "study/retrieval/results/raw"
NEW_FAMILY = ("gemma", "ctxl", "followir")


def tally_gpu() -> dict:
    pattern = [f"scores_{f}_*_paired_*.manifest.json" for f in NEW_FAMILY]
    pattern += ["scores_zerank2_zerank-2_paired_bare.manifest.json", "zerank2_equivalence*.manifest.json"]
    pattern += [f"followir_Core17_{f}_*.manifest.json" for f in NEW_FAMILY]
    fpath = [b for pat in pattern for b in RAW.glob(pat) if "_smoke" not in b.name]
    if (N1 / "profile_shuffle.manifest.json").exists():
        fpath.append(N1 / "profile_shuffle.manifest.json")
    step, per_family = [], {}
    for b in sorted(set(fpath)):
        try:
            d = json.loads(b.read_text())
        except (OSError, ValueError):
            continue
        dur = d.get("durasi_s") or d.get("durasi_total_s") or 0.0
        peak = d.get("vram_peak_mib") or 0.0
        family = next((f for f in NEW_FAMILY if f"_{f}_" in b.name), "lain")
        step.append({"manifest": b.name, "durasi_s": dur, "vram_peak_mib": peak})
        k = per_family.setdefault(family, {"n": 0, "durasi_s": 0.0, "vram_peak_mib": 0.0})
        k["n"] += 1
        k["durasi_s"] += dur
        k["vram_peak_mib"] = max(k["vram_peak_mib"], peak)
    return {"n_manifest": len(step), "total_s": round(sum(l["durasi_s"] for l in step), 1),
            "per_keluarga": per_family, "langkah": step}


def cell_path(fam: str, cell: str) -> Path:
    return RAW / f"scores_{fam}_{MODEL[fam]}_paired_{cell}.jsonl"


def build_order(allow_missing: bool) -> dict:
    lock = json.loads((SHARED / "config_lock.json").read_text())["config_version"]
    print("GATE 1: metric replication")
    evidence = M.replication_gate()

    missing = [str(cell_path(f, c).name) for f in MODEL for c in CORE_CELLS if not cell_path(f, c).exists()]
    missing += [str(cell_path(f, c).name) for f in MODEL for c in ("both_program", "both_semester") if not cell_path(f, c).exists()]
    missing += [n for n in ("scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_none_generic.jsonl",
                           "scores_zerank2_zerank-2_paired_bare.jsonl") if not (RAW / n).exists()]
    missing += [f"followir_Core17_{f}_{MODEL[f]}.json" for f in MODEL
               if not (TAB / f"followir_Core17_{f}_{MODEL[f]}.json").exists()]
    if missing and not allow_missing:
        print("ABORT: pre-registered cells missing, the table is not written:")
        for h in missing:
            print("  ", h)
        raise SystemExit(2)

    print("\nGATES 2-4: per cell")
    board, item = {}, {}
    for fam, mdl in MODEL.items():
        board[mdl], item[mdl] = {}, {}
        for c in CORE_CELLS:
            p = cell_path(fam, c)
            if not p.exists():
                continue
            board[mdl][c] = board_cell(p)
            item[mdl][c] = MC.per_item_results(p)
            print(f"  {mdl:42} {c:6} {board[mdl][c]['eksak']:>7}  pairwise {board[mdl][c]['pairwise']}")

    secondary = {}
    p = RAW / "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_none_generic.jsonl"
    if p.exists():
        secondary["ctxl_none_generic"] = board_cell(p)
        item[MODEL["ctxl"]]["none_generic"] = MC.per_item_results(p)
        print(f"  {'ctxl none_generic':42} {secondary['ctxl_none_generic']['eksak']:>7}")
    p = RAW / "scores_zerank2_zerank-2_paired_bare.jsonl"
    if p.exists():
        secondary["zerank2_bare"] = board_cell(p)
        item["zerank-2"] = {"bare": MC.per_item_results(p),
                            "default_beku": MC.per_item_results(FROZEN_ZERANK_DEFAULT)}
        print(f"  {'zerank-2 bare':42} {secondary['zerank2_bare']['eksak']:>7}")

    print("\nMcNEMAR family E: none vs slot, bge-gemma, ctxl-6B, and FollowIR-7B")
    e_list = [(f"{mdl} none vs slot", (mdl, "none"), (mdl, "slot"))
                for mdl in MODEL.values() if {"none", "slot"} <= set(item.get(mdl, {}))]
    family_e = run_family(item, e_list, "E") if e_list else None

    print("\nMcNEMAR other contrasts, exact p")
    other = {}
    for mdl in MODEL.values():
        for a, b in (("slot", "both"), ("text", "both"), ("wrong", "both")):
            if {a, b} <= set(item.get(mdl, {})):
                u = contrast(item, (mdl, a), (mdl, b))
                u.pop("_p_raw", None)
                other[f"{mdl} {a} vs {b}"] = u
                print(f"  {mdl} {a} vs {b}: discordant {u['diskordan_a_kalah_b_menang']} vs "
                      f"{u['diskordan_a_menang_b_kalah']}  p_exact {u['p_terbaca']}")
    if "none_generic" in item.get(MODEL["ctxl"], {}):
        u = contrast(item, (MODEL["ctxl"], "none"), (MODEL["ctxl"], "none_generic"))
        u.pop("_p_raw", None)
        other["ctxl none vs none_generic"] = u
    if "zerank-2" in item:
        u = contrast(item, ("zerank-2", "default_beku"), ("zerank-2", "bare"))
        u.pop("_p_raw", None)
        other["zerank-2 default(frozen) vs bare"] = u

    print("\nMcNEMAR family G: one-field labels")
    for fam, mdl in MODEL.items():
        for c in ("both_program", "both_semester"):
            p = cell_path(fam, c)
            if p.exists():
                board[mdl][c] = board_cell(p)
                item[mdl][c] = MC.per_item_results(p)
                print(f"  {mdl:42} {c:14} {board[mdl][c]['eksak']:>7}")
    g_list = []
    for mdl in MODEL.values():
        k = set(item.get(mdl, {}))
        if {"both_program", "both"} <= k:
            g_list.append((f"{mdl} program-only vs both", (mdl, "both_program"), (mdl, "both")))
        if {"both_semester", "both"} <= k:
            g_list.append((f"{mdl} semester-only vs both", (mdl, "both_semester"), (mdl, "both")))
        if {"both_program", "both_semester"} <= k:
            g_list.append((f"{mdl} program-only vs semester-only", (mdl, "both_program"), (mdl, "both_semester")))
    family_g = run_family(item, g_list, "G") if g_list else None

    bound = {}
    if family_e:
        for mdl in MODEL.values():
            k = f"{mdl} none vs slot"
            if k in family_e["kontras"]:
                v = family_e["kontras"][k]
                bound[mdl] = analysis(mdl, v["diskordan_a_kalah_b_menang"],
                                      v["diskordan_a_menang_b_kalah"], v["n_berpasangan"])

    core17 = {}
    for fam, mdl in MODEL.items():
        p = TAB / f"followir_Core17_{fam}_{mdl}.json"
        if p.exists():
            d = json.loads(p.read_text())
            core17[mdl] = {"p_MRR": d["p_MRR"], "ndcg5_og": d["ndcg5_og"],
                           "ndcg5_changed": d["ndcg5_changed"], "status": "new, cross-architecture"}
    for mdl, p in FROZEN_FOLLOWIR.items():
        if p.exists():
            d = json.loads(p.read_text())
            core17[mdl] = {"p_MRR": d["p_MRR"], "ndcg5_og": d["ndcg5_og"],
                           "ndcg5_changed": d["ndcg5_changed"], "status": "frozen, Qwen3 lineage"}

    frozen = json.loads(FROZEN_BOARD.read_text()) if FROZEN_BOARD.exists() else {}
    frozen_board = {m: {k: v["eksak"] for k, v in col.items()} for m, col in frozen.get("papan", {}).items()}

    summary = {}
    for fam, mdl in MODEL.items():
        col = board.get(mdl, {})
        summary[mdl] = {
            "keturunan": LINEAGE[fam],
            "none": col.get("none", {}).get("eksak"), "slot": col.get("slot", {}).get("eksak"),
            "text": col.get("text", {}).get("eksak"), "both": col.get("both", {}).get("eksak"),
            "wrong": col.get("wrong", {}).get("eksak"),
            "slot_lawan_none_p_exact": (family_e["kontras"].get(f"{mdl} none vs slot", {}).get("p_terbaca")
                                        if family_e else None),
            "slot_lawan_none_p_holm": (family_e["kontras"].get(f"{mdl} none vs slot", {}).get("p_holm_keluarga")
                                       if family_e else None),
            "batas_efek_slot_poin_persen": bound.get(mdl, {}).get("batas_atas_poin_persen"),
            "core17_p_MRR": core17.get(mdl, {}).get("p_MRR"),
        }

    table = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": lock,
        "praregistrasi": PREREG,
        "label_source": "deterministic (re-aggregation of the frozen score files)",
        "_apa_ini": ("Channel board of the paired track for bge-gemma, ctxl-6B, and FollowIR-7B (Gemma, "
                     "Mistral, and an instruction-tuned Mistral), with the pre-registered cells and "
                     "contrasts, next to the frozen board of the three Qwen3-based models."),
        "gerbang": {"replikasi_metrik_q2_metrik": evidence,
                    "segel_test_holdout": "passed, no test_holdout query_id in any source",
                    "integritas_sha256": "every source jsonl matches its own manifest",
                    "konsistensi_silang": "per-item win counts identical to the production pairwise rate"},
        "track": "original (paired), 104 rows, 96 comparable",
        "n_populasi": 96,
        "kunci_buta_struktural": {"nilai": 48, "kenapa": "identical rankings for the two rows of a pair; do not test against 0.5"},
        "ringkas": summary,
        "papan": board,
        "sel_sekunder": secondary,
        "mcnemar_keluarga_E_none_lawan_slot": family_e,
        "mcnemar_kontras_lain_p_exact": other,
        "mcnemar_keluarga_G_label_satu_field": family_g,
        "batas_ekuivalensi_slot": bound,
        "core17_kontrol_positif": core17,
        "papan_beku_keturunan_qwen3": frozen_board,
        "waktu_gpu": {"_apa_ini": ("GPU seconds of the cross-architecture package, summed from the durasi_s field of each "
                                   "manifest; smoke runs and the frozen manifests are not counted"),
                      **tally_gpu()},
        "sel_hilang": missing or None,
        "syarat_kutip": [
            "the none cells use the vendor default input of each family: ctxl none has no "
            "instruction, gemma none uses the default prompt, followir none is the plain query; ctxl "
            "none_generic is reported alongside",
            "the wrong column is a control of the label's content (wrong program, right semester), not of its presence",
            "p_holm of family E covers only the three none-against-slot contrasts of the new models; the published family is unchanged",
            "the slot-effect bound is an interval, not a test against a margin chosen by the authors",
        ],
    }
    source_shas = {f"{m}/{k}": v["integritas"]["sha256"] for m in board for k, v in board[m].items()}
    return {"tabel": table, "sumber_sha": source_shas, "lock": lock}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-determinism", action="store_true")
    ap.add_argument("--allow-missing", action="store_true",
                    help="print what exists, and write no table when cells are missing")
    a = ap.parse_args()
    h = build_order(a.allow_missing)
    if h["tabel"]["sel_hilang"]:
        print("\ncells missing, nothing written")
        return
    det = None
    if a.verify_determinism:
        h2 = build_order(a.allow_missing)
        s1, s2 = content_fingerprint(h["tabel"]), content_fingerprint(h2["tabel"])
        print(f"\ndeterminism: {s1[:16]} vs {s2[:16]} {'IDENTICAL' if s1 == s2 else 'DIFFERENT'}")
        if s1 != s2:
            raise SystemExit("ABORT: table content changed between two builds")
        det = {"dijalankan": True, "lolos": True, "n_ulangan": 2}
    out = TAB / f"{NAMES}.json"
    out.write_text(json.dumps(h["tabel"], indent=2, ensure_ascii=False))
    (RAW / f"{NAMES}.manifest.json").write_text(json.dumps({
        "run_id": f"q2_arch_agg_{NAMES}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/4_slot_vs_baseline/aggregate_architecture_board.py",
        "praregistrasi": PREREG, "config_version": h["lock"],
        "sumber": h["sumber_sha"], "n": 96,
        "label_source": "deterministic (re-aggregation of the frozen score files)",
        "output": {"path": f"results/tables/{NAMES}.json", "sha256": MC.sha(out),
                   "tingkat_reproduksi": "1_bit_identical",
                   "sha256_muatan_angka_tanpa_field_dibuat": content_fingerprint(h["tabel"]),
                   "verifikasi_determinisme": det},
    }, indent=2, ensure_ascii=False))
    print(f"\nwritten: {out}")
    print(json.dumps(h["tabel"]["ringkas"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
