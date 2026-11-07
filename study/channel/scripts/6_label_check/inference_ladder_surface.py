import os
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

for _stage in sorted(p for p in Path(__file__).resolve().parents[1].iterdir() if p.is_dir()):
    sys.path.insert(0, str(_stage))
from inference_ladder import BLIND, MODEL, holm, exact_mcnemar, wins_per_row  # noqa: E402

ROOT = Path(os.environ.get("SENTIO_ROOT")
            or Path(__file__).resolve().parents[4])
RAW = ROOT / "study/channel/results/raw"
TABLE = ROOT / "study/channel/results/tables/surface_ladder.json"
MANIFEST = RAW / "surface_ladder.manifest.json"
BLIND_KEY = 48
PREREG = "PRAREGISTRASI_TANGGA_ADENDUM5_PERMUKAAN.md"

PATTERNS = {
    "0.6B": ("arm", "scores_qwen3rr_Qwen3-Reranker-0.6B_{tag}_{inst}_docmeta.jsonl"),
    "4B": ("arm", "scores_qwen3rr_Qwen3-Reranker-4B_{tag}_{inst}_docmeta.jsonl"),
    "zerank-2": ("arm", "scores_zerank2official_zerank-2_{tag}_{inst}_docmeta.jsonl"),
    "FollowIR-7B": ("arch", "scores_followir_FollowIR-7B_paired_{cell}.jsonl"),
    "ctxl-6B": ("arch", "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_{cell}.jsonl"),
    "gemma": ("arch", "scores_gemma_bge-reranker-v2-gemma_paired_{cell}.jsonl"),
}
CHILD = [
    ("p1_nama_en", "label_en", "tanggaP1", "profile_nama_en", "p1. English program-name label"),
    ("p2_nama_id", "label_id_en", "tanggaP2", "profile_nama_en", "p2. Indonesian program-name label"),
    ("p3_kode", "label_kode_en", "tanggaP3", "profile_nama_en", "p3. program code label"),
    ("s2_semester_kata", "sem_kata", "tanggaS2", "profile", "s2. semester as a word"),
    ("s3_semester_opak", "sem_opak", "tanggaS3", "profile", "s3. opaque term code"),
]
ANCHORS = {
    "a_kode_instruksi_kode": {m[0]: m[1] for m in MODEL},
    "s1_semester_saja": {
        "0.6B": "scores_qwen3rr_Qwen3-Reranker-0.6B_paired_profile_docmeta_fieldsemester.jsonl",
        "4B": "scores_qwen3rr_Qwen3-Reranker-4B_paired_profile_docmeta_fieldsemester.jsonl",
        "zerank-2": "scores_zerank2official_zerank-2_paired_profile_docmeta_fieldsemester.jsonl",
        "FollowIR-7B": "scores_followir_FollowIR-7B_paired_both_semester.jsonl",
        "ctxl-6B": "scores_ctxl_ctxl-rerank-v2-instruct-multilingual-6b_paired_both_semester.jsonl",
        "gemma": "scores_gemma_bge-reranker-v2-gemma_paired_both_semester.jsonl",
    },
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()
def is_finished(p: Path) -> bool:
    return p.exists() and p.with_name(p.stem + ".manifest.json").exists()


def cell_path(model_name: str, cell: str, tag: str, inst: str) -> Path:
    kind, pattern = PATTERNS[model_name]
    return RAW / (pattern.format(tag=tag, inst=inst) if kind == "arm"
                  else pattern.format(cell=cell))


def compare(A: dict, B: dict) -> dict:
    rows = sorted(set(A) & set(B))
    gains = sum(1 for q in rows if B[q] == 1 and A[q] == 0)
    drops = sum(1 for q in rows if B[q] == 0 and A[q] == 1)
    return {"n": len(rows), "diskordan_naik": gains, "diskordan_turun": drops,
            "p_mcnemar_eksak": round(exact_mcnemar(gains, drops), 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--izinkan-hilang", action="store_true",
                    help="write the table even when some cells have not run yet")
    args = ap.parse_args()

    board, missing = {}, []
    for name, _file_a, _pattern in MODEL:
        block = {}
        W = wins_per_row(RAW / BLIND[name])
        block["buta_berlabel"] = {"menang": sum(W.values()), "berkas": BLIND[name]}
        for key, mapping in ANCHORS.items():
            pj = RAW / mapping[name]
            if not is_finished(pj):
                missing.append(f"{name} jangkar {key}")
                continue
            J = wins_per_row(pj)
            block[key] = {"menang": sum(J.values()), "n": len(J),
                           "berkas": mapping[name], "sha256": sha(pj), "beku": True}
        for key, cell, tag, inst, title in CHILD:
            pb = cell_path(name, cell, tag, inst)
            if not is_finished(pb):
                missing.append(f"{name} {title}")
                block[key] = {"status": "missing"}
                continue
            B = wins_per_row(pb)
            cell_id = {"judul": title, "menang": sum(B.values()), "n": len(B),
                   "jarak_ke_kunci_buta": sum(B.values()) - BLIND_KEY,
                   "berkas": pb.name, "sha256": sha(pb),
                   "vs_buta": compare(W, B)}
            parent = "a_kode_instruksi_kode" if key.startswith("p") else "s1_semester_saja"
            if parent in block:
                A = wins_per_row(RAW / ANCHORS[parent][name])
                cell_id[f"vs_{parent}"] = compare(A, B)
            block[key] = cell_id
        board[name] = block

    if missing and not args.izinkan_hilang:
        raise SystemExit("ABORT: cells missing, table not written, so a complete table is not replaced by one "
                         "with gaps:\n  " + "\n  ".join(missing))

    for key, _c, _t, _i, _j in CHILD:
        for field in ("vs_buta", "vs_a_kode_instruksi_kode", "vs_s1_semester_saja"):
            valid = [(nm, board[nm][key]) for nm in board
                   if field in (board[nm].get(key) or {})]
            if len(valid) == len(board) and valid:
                ps = [b[field]["p_mcnemar_eksak"] for _, b in valid]
                for (nm, block), ph in zip(valid, holm(ps)):
                    block[field]["p_holm_keluarga_enam"] = round(ph, 6)
                    block[field]["signifikan_0.05"] = ph < 0.05

    output = {
        "dibuat": datetime.now(timezone.utc).isoformat(),
        "config_version_lock": "0a7b2e4de856",
        "praregistrasi": PREREG,
        "_apa_ini": ("Surface-form ladder. It separates the wording of the program label from whether the "
                     "label and the instruction share a language, and adds a ladder on the semester field."),
        "kontras_penentu": ("p1 against p2. Both use an English instruction that names the program; they differ only "
                            "in the language of the label."),
        "kunci_buta_struktural": BLIND_KEY,
        "uji": ("Exact paired McNemar against the text-labeled cell without a profile, which sits at the "
                "key by construction, and against the anchor of each ladder. Holm within the family of "
                "six models. No test against 0.5."),
        "label_source": "deterministic (gold matching, no rater)",
        "sel_belum_ada": missing,
        "papan": board,
    }
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(json.dumps(output, indent=1, ensure_ascii=False))
    MANIFEST.write_text(json.dumps({
        "run_id": "surface_ladder_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "script": "study/channel/scripts/6_label_check/inference_ladder_surface.py",
        "output": {"path": "results/tables/surface_ladder.json", "sha256": sha(TABLE)},
        "praregistrasi": PREREG,
        "label_source": "deterministic",
    }, indent=1))

    print(f"{'model':13s} {'buta':>5} {'a':>4} {'p1':>4} {'p2':>4} {'p3':>4} | "
          f"{'s1':>4} {'s2':>4} {'s3':>4}")
    print("-" * 62)
    for nm, block in board.items():
        def v(k):
            return block.get(k, {}).get("menang", "--")
        print(f"{nm:13s} {v('buta_berlabel'):>5} {v('a_kode_instruksi_kode'):>4} "
              f"{v('p1_nama_en'):>4} {v('p2_nama_id'):>4} {v('p3_kode'):>4} | "
              f"{v('s1_semester_saja'):>4} {v('s2_semester_kata'):>4} {v('s3_semester_opak'):>4}")
    if missing:
        print("MISSING:", len(missing), "sel")
    print("written:", TABLE)


if __name__ == "__main__":
    main()
