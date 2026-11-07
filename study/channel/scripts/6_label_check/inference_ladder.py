import os
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
import metrics as M  # noqa: E402
from exact_tests import holm, mcnemar_exact as exact_mcnemar  # noqa: E402

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW = ROOT / "study/channel/results/raw"
TABLE = ROOT / "study/channel/results/tables/marker_ladder.json"
MANIFEST = RAW / "marker_ladder.manifest.json"
BLIND_KEY = 48
PREREG = "PRAREGISTRASI_TANGGA_JARAK_INFERENSI.md"

MODEL = [
    ("0.6B", "scores_qwen3rr_Qwen3-Reranker-0.6B_paired_profile_docmeta_fieldprogram.jsonl",
     "scores_qwen3rr_Qwen3-Reranker-0.6B_tangga{g}_profile_docmeta.jsonl"),
    ("4B", "scores_qwen3rr_Qwen3-Reranker-4B_paired_profile_docmeta_fieldprogram.jsonl",
     "scores_qwen3rr_Qwen3-Reranker-4B_tangga{g}_profile_docmeta.jsonl"),
    ("zerank-2", "scores_zerank2official_zerank-2_paired_profile_docmeta_fieldprogram.jsonl",
     "scores_zerank2official_zerank-2_tangga{g}_profile_docmeta.jsonl"),
    ("FollowIR-7B", "scores_followir_FollowIR-7B_paired_both_program.jsonl",
     "scores_followir_FollowIR-7B_paired_label_{g}.jsonl"),
    ("ctxl-6B", "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_both_program.jsonl",
     "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_label_{g}.jsonl"),
    ("gemma", "scores_gemma_bge-reranker-v2-gemma_paired_both_program.jsonl",
     "scores_gemma_bge-reranker-v2-gemma_paired_label_{g}.jsonl"),
]
STYLE = [("nama", "b. program name"), ("alami", "c. natural header"), ("opak", "d. opaque code")]
BLIND = {
    "0.6B": "scores_qwen3rr_Qwen3-Reranker-0.6B_paired_default_docmeta.jsonl",
    "4B": "scores_qwen3rr_Qwen3-Reranker-4B_paired_default_docmeta.jsonl",
    "zerank-2": "scores_zerank2official_zerank-2_paired_default_docmeta.jsonl",
    "FollowIR-7B": "scores_followir_FollowIR-7B_paired_text.jsonl",
    "ctxl-6B": "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_text.jsonl",
    "gemma": "scores_gemma_bge-reranker-v2-gemma_paired_text.jsonl",
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()
def is_finished(p: Path) -> bool:
    return p.exists() and p.with_name(p.stem + ".manifest.json").exists()


def wins_per_row(p: Path) -> dict:
    output = {}
    for r in M.read_ranking(p):
        gold = set(r.get("gold") or [])
        cp = r.get("counterpart")
        if cp is None:
            continue
        cp = set(cp) if isinstance(cp, list) else {cp}
        rank = r["ranking"]
        pos_g = next((i for i, c in enumerate(rank) if c in gold), None)
        pos_c = next((i for i, c in enumerate(rank) if c in cp), None)
        if pos_g is None or pos_c is None:
            continue
        key = f"{r['query_id']}|{r.get('profil') or r.get('profile') or ''}"
        output[key] = 1 if pos_g < pos_c else 0
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--izinkan-hilang", action="store_true",
                    help="write the table even when some cells have not run yet")
    args = ap.parse_args()

    board, missing = {}, []
    for name, file_a, pattern in MODEL:
        pa = RAW / file_a
        if not is_finished(pa):
            missing.append(f"{name} rung a ({file_a})")
            continue
        A = wins_per_row(pa)
        p_blind = RAW / BLIND[name]
        W = wins_per_row(p_blind)
        block = {"a_kode_literal": {"menang": sum(A.values()), "n": len(A),
                                   "berkas": file_a, "sha256": sha(pa)},
                "buta_berlabel": {"menang": sum(W.values()), "n": len(W),
                                  "berkas": BLIND[name], "sha256": sha(p_blind)}}
        for g, title in STYLE:
            pb = RAW / pattern.format(g=g)
            if not is_finished(pb):
                missing.append(f"{name} {title}")
                block[g] = {"status": "missing"}
                continue
            B = wins_per_row(pb)
            shared = sorted(set(A) & set(B))
            gains = sum(1 for q in shared if B[q] == 1 and A[q] == 0)
            drops = sum(1 for q in shared if B[q] == 0 and A[q] == 1)
            shared_w = sorted(set(W) & set(B))
            gains_w = sum(1 for q in shared_w if B[q] == 1 and W[q] == 0)
            drops_w = sum(1 for q in shared_w if B[q] == 0 and W[q] == 1)
            block[g] = {
                "judul": title,
                "vs_buta_diskordan_naik": gains_w,
                "vs_buta_diskordan_turun": drops_w,
                "vs_buta_p_mcnemar_eksak": round(exact_mcnemar(gains_w, drops_w), 6),
                "menang": sum(B[q] for q in shared),
                "n_komparabel": len(shared),
                "vs_a_diskordan_naik": gains,
                "vs_a_diskordan_turun": drops,
                "vs_a_selisih_bersih": gains - drops,
                "vs_a_p_mcnemar_eksak": round(exact_mcnemar(gains, drops), 6),
                "jarak_ke_kunci_buta": sum(B[q] for q in shared) - BLIND_KEY,
                "berkas": pattern.format(g=g),
                "sha256": sha(pb),
            }
        board[name] = block

    if missing and not args.izinkan_hilang:
        raise SystemExit("ABORT: cells missing, table not written, so a complete table is not replaced by one "
                         "with gaps:\n  " + "\n  ".join(missing))

    for g, _ in STYLE:
        for key, tag in (("vs_a_p_mcnemar_eksak", "vs_a"), ("vs_buta_p_mcnemar_eksak", "vs_buta")):
            valid = [(nm, board[nm][g]) for nm in board
                   if g in board[nm] and key in board[nm][g]]
            if len(valid) == len(board) and valid:
                for (nm, block), ph in zip(valid, holm([b[key] for _, b in valid])):
                    block[f"{tag}_p_holm_keluarga_enam"] = round(ph, 6)
                    block[f"{tag}_signifikan_0.05"] = ph < 0.05

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "praregistrasi": PREREG,
        "_apa_ini": ("Inference-distance ladder on the document channel, paired track of 96 rows, six models. "
                     "Every rung uses the program label only, because the single-field control shows the "
                     "semester alone nearly matches the full label. Its p values are stored with six "
                     "decimals, so a p of 0.0 is below 0.0000005."),
        "kunci_buta_struktural": {
            "nilai": BLIND_KEY,
            "kenapa": "a scorer that does not read the profile returns identical rankings for the two rows of "
                      "a pair, so exactly one row wins"},
        "anak_tangga": {
            "a": "[Source: <DEPT> degree program textbook], the same code as the instruction",
            "b": "[Source: <program name> degree program textbook], no shared token",
            "c": "Program studi <program name>., without the Source frame, exploratory",
            "d": "[Source: PRG-<nn> degree program textbook], opaque code, negative control"},
        "uji": ("Exact paired McNemar, two-sided. Against rung a, to measure its collapse, and against "
                "the text-labeled cell without a profile, to test whether the cell can still be told "
                "apart from a blind scorer. Both are Holm-corrected within the family of six models. The "
                "paper does not test against 0.5, only against the key, and the blind cell is the key by "
                "construction."),
        "label_source": "deterministic (gold matching, no rater)",
        "sel_belum_ada": missing,
        "papan": board,
    }
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "marker_ladder_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/6_label_check/inference_ladder.py",
        "output": {"path": "results/tables/marker_ladder.json", "sha256": sha(TABLE)},
        "praregistrasi": PREREG,
        "label_source": "deterministic",
    }, indent=1))

    for nm, block in board.items():
        a = block["a_kode_literal"]["menang"]
        truncate = []
        for g, _ in STYLE:
            b = block.get(g, {})
            truncate.append(f"{g} {b.get('menang', '--'):>3}" if "menang" in b else f"{g}  --")
        print(f"  {nm:12s} a {a:>3} | " + " | ".join(truncate))
    if missing:
        print("MISSING:", len(missing), "cells")
    print("written:", TABLE)


if __name__ == "__main__":
    main()
