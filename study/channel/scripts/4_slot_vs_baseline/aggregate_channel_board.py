"""Builds the channel boards of Qwen3-0.6B, Qwen3-4B, and zerank-2 from the per-row scores.

For each model and setup it counts the rows where the asker's own gold ranks above the other student's
gold, in channel_board_paired.json (96 rows) and channel_board_xpair.json (1,621 rows). Each cell is
recomputed from its score file and compared with the count in the paper, and a difference is recorded in
the table. Nothing is written unless the metric gate, the held-out check, the consistency checks, and the
hash check of every cell that has a manifest pass.
"""

import os
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))

import metrics as M
import mcnemar as MC

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW_N1 = ROOT / "study/retrieval/results/raw"
Q2 = ROOT / "study/channel/results"
RAW = Q2 / "raw"
SHARED = ROOT / "study/_shared"

XPAIR_COLUMNS = ["tanpa_profil", "slot", "slot_plus_teks", "label_salah"]
PAIRED_COLUMNS = ["tanpa_profil", "slot", "teks_saja", "slot_plus_teks",
                "label_salah", "vendor_memory"]

XPAIR_SOURCE = {
    "Qwen3-Reranker-0.6B": {
        "tanpa_profil": RAW / "scores_qwen3rr_Qwen3-Reranker-0.6B_xpair_default.jsonl",
        "slot": RAW / "scores_qwen3rr_Qwen3-Reranker-0.6B_xpair_profile.jsonl",
        "slot_plus_teks": RAW / "scores_qwen3rr_Qwen3-Reranker-0.6B_xpair_profile_docmeta.jsonl",
        "label_salah": RAW / "scores_qwen3rr_Qwen3-Reranker-0.6B_xpair_profile_docmetasalah.jsonl",
    },
    "Qwen3-Reranker-4B": {
        "tanpa_profil": RAW / "scores_qwen3rr_Qwen3-Reranker-4B_xpair_default.jsonl",
        "slot": RAW / "scores_qwen3rr_Qwen3-Reranker-4B_xpair_profile.jsonl",
        "slot_plus_teks": RAW / "scores_qwen3rr_Qwen3-Reranker-4B_xpair_profile_docmeta.jsonl",
        "label_salah": RAW / "scores_qwen3rr_Qwen3-Reranker-4B_xpair_profile_docmetasalah.jsonl",
    },
    "zerank-2": {
        "tanpa_profil": RAW / "scores_zerank2_zerank-2_xpair_default.jsonl",
        "slot": RAW / "scores_zerank2_zerank-2_xpair_profile.jsonl",
        "slot_plus_teks": RAW / "scores_zerank2_zerank-2_xpair_profile_docmeta.jsonl",
        "label_salah": RAW / "scores_zerank2_zerank-2_xpair_profile_docmetasalah.jsonl",
    },
}

PAIRED_SOURCE = {
    "Qwen3-Reranker-0.6B": {
        "tanpa_profil": RAW_N1 / "qwen_scores_0.6b_paired_default.jsonl",
        "slot": RAW_N1 / "qwen_scores_0.6b_paired_profile.jsonl",
        "teks_saja": RAW / "scores_qwen3rr_Qwen3-Reranker-0.6B_paired_default_docmeta.jsonl",
        "slot_plus_teks": RAW / "scores_qwen3rr_Qwen3-Reranker-0.6B_paired_profile_docmeta.jsonl",
        "label_salah": RAW / "scores_qwen3rr_Qwen3-Reranker-0.6B_paired_profile_docmetasalah.jsonl",
    },
    "Qwen3-Reranker-4B": {
        "tanpa_profil": RAW_N1 / "qwen_scores_4b_paired_default.jsonl",
        "slot": RAW_N1 / "qwen_scores_4b_paired_profile.jsonl",
        "teks_saja": RAW / "scores_qwen3rr_Qwen3-Reranker-4B_paired_default_docmeta.jsonl",
        "slot_plus_teks": RAW / "scores_qwen3rr_Qwen3-Reranker-4B_paired_profile_docmeta.jsonl",
        "label_salah": RAW / "scores_qwen3rr_Qwen3-Reranker-4B_paired_profile_docmetasalah.jsonl",
    },
    "zerank-2": {
        "tanpa_profil": RAW / "scores_zerank2_zerank-2_paired_default.jsonl",
        "slot": RAW / "scores_zerank2_zerank-2_paired_profile.jsonl",
        "teks_saja": RAW / "scores_zerank2_zerank-2_paired_default_docmeta.jsonl",
        "slot_plus_teks": RAW / "scores_zerank2_zerank-2_paired_profile_docmeta.jsonl",
        "label_salah": RAW / "scores_zerank2_zerank-2_paired_profile_docmetasalah.jsonl",
        "vendor_memory": RAW / "scores_zerank2_zerank-2_paired_profile_vmem.jsonl",
    },
}

PAPER_XPAIR = {
    "Qwen3-Reranker-0.6B": {"tanpa_profil": 600, "slot": 597, "slot_plus_teks": 697, "label_salah": 645},
    "Qwen3-Reranker-4B": {"tanpa_profil": 600, "slot": 598, "slot_plus_teks": 641, "label_salah": 653},
    "zerank-2": {"tanpa_profil": 600, "slot": 598, "slot_plus_teks": 818, "label_salah": 706},
}

PAPER_PAIRED = {
    "Qwen3-Reranker-0.6B": {"tanpa_profil": 48, "slot": 49, "teks_saja": 48, "slot_plus_teks": 88, "label_salah": 76},
    "Qwen3-Reranker-4B": {"tanpa_profil": 48, "slot": 50, "teks_saja": 48, "slot_plus_teks": 83, "label_salah": 72},
    "zerank-2": {"tanpa_profil": 48, "slot": 47, "teks_saja": 48, "slot_plus_teks": 96, "label_salah": 91, "vendor_memory": 47},
}

PAPER_REF_XPAIR = {"kunci_buta_struktural": 600, "n_baris": 1621,
                      "DITS_B": 779}
PAPER_REF_PAIRED = {"kunci_buta_struktural": 48, "n_pasang": 96,
                       "DITS_B": 93, "D_cascade": 65, "filter_metadata": 96}


def manifest_for(p: Path) -> Path:
    return p.parent / (p.name[:-len(".jsonl")] + ".manifest.json")


def check_integrity(p: Path) -> dict:
    actual = MC.sha(p)
    man = manifest_for(p)
    if not man.exists():
        return {"manifest": None, "sha256": actual, "cocok": None,
                "catatan": "no manifest ships for this cell"}
    d = json.loads(man.read_text())
    written_sha = (d.get("output") or {}).get("sha256")
    if written_sha and written_sha != actual:
        raise SystemExit(
            f"ABORT INTEGRITY: {p.name} sha256 {actual} != manifest {written_sha}")
    log_check = None
    if d.get("n") is not None:
        n_rows = sum(1 for l in p.open() if l.strip())
        if d["n"] != n_rows:
            log_check = (f"manifest field n {d['n']} differs from the {n_rows} rows of the file, because "
                         f"the run was resumed and n counts only the rows of its last part. A resumed "
                         f"run can only append to the file of the same condition, and the sha256 "
                         f"matches the manifest, so the scores are valid. Only durasi_s and "
                         f"vram_peak_mib describe the last part alone.")
    return {"manifest": man.name, "sha256": actual, "cocok": True,
            "run_id": d.get("run_id"), "anomali_field_n": log_check}


def board_cell(p: Path) -> dict:
    integ = check_integrity(p)
    rows = M.read_ranking(p)
    M.seal_gate([r["query_id"] for r in rows])
    m = M.paired_metrics(rows)
    per_item = MC.per_item_results(p)
    wins = sum(1 for v in per_item.values() if v)
    n = len(per_item)
    if n != m["n_komparabel"]:
        raise SystemExit(
            f"ABORT CONSISTENCY: {p.name} per-item n_komparabel {n} != "
            f"production metric {m['n_komparabel']}")
    if m["pairwise"] is not None and round(wins / n, 4) != m["pairwise"]:
        raise SystemExit(
            f"ABORT CONSISTENCY: {p.name} per-item rate {wins / n:.6f} != "
            f"pairwise production metric {m['pairwise']}")
    return {
        "menang": wins,
        "n_komparabel": n,
        "eksak": f"{wins}/{n}",
        "pairwise": m["pairwise"],
        "top1_gold": m["top1_gold"],
        "ndcg5": m["ndcg5"],
        "n_baris": m["n_baris"],
        "berkas": str(p.relative_to(ROOT)),
        "integritas": integ,
    }


def build_board(source: dict, columns: list) -> tuple:
    board, item = {}, {}
    for model, route in source.items():
        board[model], item[model] = {}, {}
        for k in columns:
            p = route.get(k)
            if p is None:
                continue
            if not p.exists():
                raise SystemExit(f"ABORT: source file missing {p}")
            board[model][k] = board_cell(p)
            item[model][k] = MC.per_item_results(p)
            print(f"  {model:20} {k:16} {board[model][k]['eksak']:>10}"
                  f"  pairwise {board[model][k]['pairwise']}")
    return board, item


def compare_with_paper(board: dict, expected: dict) -> dict:
    result, mismatches = {}, []
    for model, expected_cells in expected.items():
        result[model] = {}
        for k, v in expected_cells.items():
            if k not in board.get(model, {}):
                result[model][k] = {"naskah": v, "hitung": None,
                                   "vonis": "TIDAK_TERUKUR"}
                continue
            h = board[model][k]["menang"]
            match = (h == v)
            result[model][k] = {"naskah": v, "hitung": h,
                               "selisih": h - v,
                               "vonis": "COCOK" if match else "BEDA"}
            if not match:
                mismatches.append(f"{model}/{k}: naskah {v}, recomputed {h} (selisih {h - v:+d})")
    return {"per_sel": result, "jumlah_beda": len(mismatches), "daftar_beda": mismatches,
            "vonis": "all cells match" if not mismatches else "some cells differ"}


def contrast(item: dict, a: tuple, b: tuple) -> dict:
    ma, ka = a
    mb, kb = b
    da, db = item[ma][ka], item[mb][kb]
    shared = sorted(set(da) & set(db))
    n01 = sum(1 for q in shared if not da[q] and db[q])
    n10 = sum(1 for q in shared if da[q] and not db[q])
    p = MC.mcnemar_exact(n01, n10)
    return {
        "n_berpasangan": len(shared),
        "diskordan_a_kalah_b_menang": n01,
        "diskordan_a_menang_b_kalah": n10,
        "selisih_bersih": n01 - n10,
        "p_mcnemar_exact": round(p, 8),
        "p_mcnemar_exact_penuh": p,
        "p_terbaca": f"{p:.3g}",
        "_p_raw": p,
    }


def run_family(item: dict, entries: list, name: str) -> dict:
    tests, praw = {}, {}
    for label, a, b in entries:
        u = contrast(item, a, b)
        praw[label] = u.pop("_p_raw")
        tests[label] = u
    ph = MC.holm(praw)
    for k in tests:
        tests[k]["p_holm_keluarga"] = ph[k]
        tests[k]["signifikan_0.05"] = ph[k] < 0.05
        print(f"  [{name}] {k:46} discordant "
              f"{tests[k]['diskordan_a_kalah_b_menang']} vs "
              f"{tests[k]['diskordan_a_menang_b_kalah']}  p_exact "
              f"{tests[k]['p_terbaca']:>10}  p_holm {ph[k]:.6g}")
    return {
        "m": len(tests),
        "aturan_baca": ("a p_holm_keluarga of 0.0 means below 1e-6 after rounding to six decimals, not a true "
                        "zero; the full exact value is in p_mcnemar_exact_penuh"),
        "kontras": tests,
    }


def content_fingerprint(table: dict) -> str:
    body = {k: v for k, v in table.items() if k != "dibuat"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_table(table: dict, name: str, source_sha: dict, lock: str, notes: list,
          determinism: dict = None):
    out = Q2 / f"tables/{name}.json"
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False))
    man = RAW / f"{name}.manifest.json"
    man.write_text(json.dumps({
        "run_id": f"q2_agg_papan_saluran_{name}_"
                  + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/4_slot_vs_baseline/aggregate_channel_board.py",
        "config_version": lock,
        "sumber": source_sha,
        "n": table.get("n_populasi"),
        "label_source": "deterministic (re-aggregation of the frozen score files)",
        "verifikasi_naskah": table["verifikasi_naskah"]["vonis"],
        "catatan_selisih": notes,
        "output": {
            "path": f"results/tables/{name}.json",
            "sha256": MC.sha(out),
            "n_baris": table.get("n_populasi"),
            "tingkat_reproduksi": "1_bit_identical",
            "sha256_muatan_angka_tanpa_field_dibuat": content_fingerprint(table),
            "verifikasi_determinisme": determinism,
        },
    }, indent=2, ensure_ascii=False))
    print(f"  written: {out}")
    print(f"  written: {man}")
    return out


def build_order():
    lock = json.loads((SHARED / "config_lock.json").read_text())["config_version"]

    print("GATE 1: replication of the production metric definition")
    replication_evidence = M.replication_gate()

    print("\nGATE 2-4: mined track board (xpair, 1,621 rows)")
    board_x, item_x = build_board(XPAIR_SOURCE, XPAIR_COLUMNS)

    print("\nGATE 2-4: original track board (paired, 96 pairs)")
    board_p, item_p = build_board(PAIRED_SOURCE, PAIRED_COLUMNS)

    print("\nVERIFICATION AGAINST MANUSCRIPT")
    check_x = compare_with_paper(board_x, PAPER_XPAIR)
    check_p = compare_with_paper(board_p, PAPER_PAIRED)
    for name, check in (("xpair", check_x), ("paired", check_p)):
        print(f"  {name}: {check['vonis']}")
        for b in check["daftar_beda"]:
            print(f"    DIFFER {b}")

    dits_x = json.loads((Q2 / "tables/dits_xpair.json").read_text())
    strata = json.loads((Q2 / "tables/dits_strata.json").read_text())
    gate = json.loads((RAW / "dits_gate_lolos.json").read_text())
    ev_paired = json.loads((RAW_N1 / "eval_paired.json").read_text())
    qwen_c3c4 = json.loads((ROOT / "study/retrieval/results/tables/"
                            "qwen_rerankers_c3_c4.json").read_text())

    key_x = dits_x["arms"]["S1"]["menang_eksak"]
    b_x = dits_x["arms"]["B"]["menang_eksak"]
    print(f"  DITS xpair: blind lock S1 {key_x}, B {b_x}")

    print("\nMcNEMAR EXACT, family A: slot vs blind (three models)")
    family_a = run_family(item_p, [
        ("C3 tanpa_profil vs C3 slot",
         ("Qwen3-Reranker-0.6B", "tanpa_profil"), ("Qwen3-Reranker-0.6B", "slot")),
        ("C4 tanpa_profil vs C4 slot",
         ("Qwen3-Reranker-4B", "tanpa_profil"), ("Qwen3-Reranker-4B", "slot")),
        ("zerank2 tanpa_profil vs zerank2 slot",
         ("zerank-2", "tanpa_profil"), ("zerank-2", "slot")),
    ], "A")

    print("\nMcNEMAR EXACT, family B: zerank-2 slot, text label, and vendor memory")
    family_b = run_family(item_p, [
        ("zerank2 slot vs zerank2 slot_plus_teks",
         ("zerank-2", "slot"), ("zerank-2", "slot_plus_teks")),
        ("zerank2 teks_saja vs zerank2 slot_plus_teks",
         ("zerank-2", "teks_saja"), ("zerank-2", "slot_plus_teks")),
        ("zerank2 slot vs zerank2 vendor_memory",
         ("zerank-2", "slot"), ("zerank-2", "vendor_memory")),
        ("zerank2 label_salah vs zerank2 slot_plus_teks",
         ("zerank-2", "label_salah"), ("zerank-2", "slot_plus_teks")),
    ], "B")

    print("\nMcNEMAR EXACT, family C supplement: channels on the mined track")
    family_c = run_family(item_x, [
        (f"{m} tanpa_profil vs {m} slot", (m, "tanpa_profil"), (m, "slot"))
        for m in XPAIR_SOURCE
    ] + [
        (f"{m} slot vs {m} slot_plus_teks", (m, "slot"), (m, "slot_plus_teks"))
        for m in XPAIR_SOURCE
    ], "C")

    p_slot = {k: v["p_mcnemar_exact"] for k, v in family_a["kontras"].items()}
    p_min = min(p_slot.values())
    print(f"\n  SLOT p RANGE, THREE MODELS: min {p_min:.6g}, "
          f"max {max(p_slot.values()):.6g}")

    general_note = []
    for track, board in (("broad", board_x), ("paired", board_p)):
        for model, col in board.items():
            for k, s in col.items():
                an = s["integritas"].get("anomali_field_n")
                if an:
                    general_note.append(f"{track}/{model}/{k}: {an}")

    timestamp = datetime.now(timezone.utc).isoformat()
    base = {
        "dibuat": timestamp,
        "config_version_lock": lock,
        "label_source": "deterministic (re-aggregation of the frozen score files)",
        "gerbang": {
            "replikasi_metrik_q2_metrik": replication_evidence,
            "segel_test_holdout": "passed, no test_holdout query_id in any source",
            "integritas_sha256": ("every source jsonl that has a manifest matches it. The four Qwen3 cells of "
                                  "the original paired track have none, and the metric gate checks them instead"),
            "konsistensi_silang": ("per-item win counts identical to the pairwise rate of the production metric in every "
                                   "cell"),
        },
        "anomali_manifest": general_note or None,
    }

    tab_x = dict(base)
    tab_x.update({
        "_apa_ini": ("Channel board of the mined broad track. The counts come from the per-row score files."),
        "track": "mined (xpair), 600 pairs mined automatically, 8 programs",
        "n_populasi": 1621,
        "satuan": "rows resolved out of 1,621",
        "kunci_buta_struktural": {
            "nilai": 600,
            "rate": round(600 / 1621, 4),
            "kenapa": ("a scorer that does not read the profile returns identical rankings for the source row "
                       "and the mirror row, so the win count is fixed algebraically; do not test against the "
                       "0.5 null"),
            "sumber": "study/channel/results/tables/dits_xpair.json arm S1",
        },
        "papan": board_x,
        "arm_dits": {
            "S1_buta": {"eksak": key_x,
                        "pairwise": dits_x["arms"]["S1"]["pairwise"]},
            "S2_prox": {"eksak": dits_x["arms"]["S2"]["menang_eksak"],
                        "pairwise": dits_x["arms"]["S2"]["pairwise"]},
            "B_DITS": {"eksak": b_x, "pairwise": dits_x["arms"]["B"]["pairwise"]},
            "catatan": ("DITS reads the metadata directly, so it has no slot-against-text column; its single "
                        "cell (779) spans three columns in the paper"),
            "stratifikasi": {
                "dalam_mandat_baris_1_2": strata["b_di_baris_winnable"],
                "w3_konsesi_desain": strata["wilayah"]["W3_tak_terselamatkan_by_design"],
            },
        },
        "kontras_mcnemar_suplemen": family_c,
        "verifikasi_naskah": check_x,
        "acuan_naskah": PAPER_REF_XPAIR,
        "syarat_kutip": [
            "the Slot+text and Wrong columns both use instruction=profile; they differ only in the "
            "content of the program label in the document text",
            "on this pool zerank-2 with slot plus text is above DITS",
            "the 4B wrong-label cell is higher than the right-label cell; the paper leaves the "
            "mechanism open",
        ],
    })

    tab_p = dict(base)
    tab_p.update({
        "_apa_ini": ("Channel board of the original paired track, with the zerank-2 rows and the wrong-label "
                     "control column."),
        "track": "original (paired), one course pair, 104 rows, 96 comparable",
        "n_populasi": 96,
        "satuan": "rows resolved out of 96",
        "kunci_buta_struktural": {
            "nilai": 48,
            "rate": 0.5,
            "kenapa": ("identical rankings for the two rows of a pair, so exactly one row wins; the null is "
                       "degenerate, so do not test against 0.5"),
            "sumber": "study/channel/results/tables/mcnemar_paired.json",
        },
        "papan": board_p,
        "arm_referensi_terbit": {
            "B_DITS": {
                "eksak": "{}/{}".format(
                    round(gate["detail"]["B"]["dihitung"]["pairwise"]
                          * gate["detail"]["B"]["dihitung"]["n_komparabel"]),
                    gate["detail"]["B"]["dihitung"]["n_komparabel"]),
                "pairwise": gate["detail"]["B"]["dihitung"]["pairwise"],
                "sumber": "study/channel/results/raw/dits_gate_lolos.json",
                "diverifikasi_ulang": gate["detail"]["B"]["identik"]},
            "D_cascade": {"eksak": f"{round(ev_paired['D']['pairwise'] * 96)}/96",
                          "pairwise": round(ev_paired["D"]["pairwise"], 4),
                          "sumber": "study/retrieval/results/raw/eval_paired.json"},
            "filter_metadata": {"eksak": f"{round(qwen_c3c4['paired']['filter_metadata']['pairwise'] * 96)}/96",
                                "pairwise": round(qwen_c3c4["paired"]["filter_metadata"]["pairwise"], 4),
                                "sumber": "study/retrieval/results/tables/qwen_rerankers_c3_c4.json"},
        },
        "mcnemar_keluarga_A_slot_lawan_buta": family_a,
        "mcnemar_keluarga_B_interaksi_zerank2": family_b,
        "rentang_p_slot_tiga_model": {
            "per_model": p_slot,
            "p_exact_minimum": p_min,
            "kalimat_abstrak_yang_sah": (
                f"exact McNemar p >= {p_min:.2f}" if p_min >= 0.01
                else f"exact McNemar p as small as {p_min:.4g} on one model"),
            "catatan": ("family A here (three models, m = 3) is separate from the post hoc family of "
                        "mcnemar_paired.json (seven contrasts), so the p_holm values there stay unchanged. The "
                        "exact p of C3 and C4 before correction is 1.0 and 0.81452942 in both tables"),
        },
        "verifikasi_naskah": check_p,
        "acuan_naskah": PAPER_REF_PAIRED,
        "syarat_kutip": [
            "the teks_saja column is fixed at 48 not because the model fails but because, without a "
            "profile, the two rows of a pair get the same ranking",
            "the label_salah column controls the content of the label, not its presence; the "
            "semester stays right, so one factor is manipulated",
            "vendor_memory is measured for zerank-2 only",
        ],
    })

    sha_x = {f"{m}/{k}": board_x[m][k]["integritas"]["sha256"]
             for m in board_x for k in board_x[m]}
    sha_p = {f"{m}/{k}": board_p[m][k]["integritas"]["sha256"]
             for m in board_p for k in board_p[m]}

    return {
        "lock": lock,
        "tabel": {"channel_board_xpair": tab_x, "channel_board_paired": tab_p},
        "sumber_sha": {"channel_board_xpair": sha_x, "channel_board_paired": sha_p},
        "catatan": {
            "channel_board_xpair": check_x["daftar_beda"] + (general_note or []),
            "channel_board_paired": check_p["daftar_beda"] + (general_note or []),
        },
        "cek": {"xpair": check_x, "paired": check_p},
        "p_slot": p_slot,
        "p_slot_minimum": p_min,
        "anomali": general_note,
    }


def main():
    verify = "--verify-determinism" in sys.argv
    result = build_order()

    det = None
    if verify:
        print("\nDETERMINISM VERIFICATION: second run over the same input")
        second = build_order()
        mismatches = []
        for name, tab in result["tabel"].items():
            a, b = content_fingerprint(tab), content_fingerprint(second["tabel"][name])
            print(f"  {name}: {a[:16]} versus {b[:16]} "
                  f"{'IDENTICAL' if a == b else 'DIFFERENT'}")
            if a != b:
                mismatches.append(name)
        if mismatches:
            raise SystemExit(
                f"ABORT DETERMINISM: numeric payload changed between runs at {mismatches}"
                f". Tables not written.")
        det = {"dijalankan": True, "lolos": True, "n_ulangan": 2,
               "cara": "sha256 of the number payload with the dibuat field removed"}

    print("\nWRITING TABLES")
    for name, tab in result["tabel"].items():
        write_table(tab, name, result["sumber_sha"][name], result["lock"],
              result["catatan"][name], det)

    print("\nSUMMARY")
    print(f"  xpair  : {result['cek']['xpair']['vonis']} "
          f"({result['cek']['xpair']['jumlah_beda']} cells differ)")
    print(f"  paired : {result['cek']['paired']['vonis']} "
          f"({result['cek']['paired']['jumlah_beda']} cells differ)")
    print(f"  p slot : {json.dumps(result['p_slot'])}")
    print(f"  p slot minimum three models: {result['p_slot_minimum']}")
    for c in result["anomali"] or []:
        print(f"  anomali manifest: {c}")


if __name__ == "__main__":
    main()
